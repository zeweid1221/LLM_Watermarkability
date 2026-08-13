"""Teacher-forced watermarkability analysis on one shared text trajectory.

The script generates one unwatermarked continuation per held-out prompt and
reuses each base-model next-token distribution to score every requested
watermark rule. Text, context, model, and sampling remain fixed across rules;
only the target-set request changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from reproducibility import validate_tokenizer_identity
from analyze_teacher_forced_watermarkability import minimum_intervention
from run_experiment import load_manifest, sample_logits, stable_int
from targets import (
    DYNAMIC_RULE_IDS,
    RULE_IDS,
    Partition,
    TargetRegistry,
    kgw_row_seed,
    make_rule_schedule,
)


DEFAULT_RULES = (
    "ecc",
    "binary",
    "cw_abcd",
    "unigram",
    "atomic_kgw",
    "atomic_kgw25",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--partition-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--reference-details", type=Path)
    parser.add_argument("--reference-summary", type=Path)
    parser.add_argument("--rules", default=",".join(DEFAULT_RULES))
    parser.add_argument("--test-count", type=int, default=256)
    parser.add_argument("--generated-tokens", type=int, default=96)
    parser.add_argument("--max-prompt-tokens", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--target-mass", type=float, default=0.99)
    parser.add_argument("--fixed-bias", type=float, default=20.0)
    parser.add_argument("--train-key-count", type=int, default=1)
    parser.add_argument("--eval-key-count", type=int, default=1)
    parser.add_argument("--kgw-seed-count", type=int, default=1)
    parser.add_argument("--kgw-seed-offset", type=int, default=3779)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--require-reference-match", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_rules(value: str) -> list[str]:
    rules = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(rules) - set(RULE_IDS))
    if unknown:
        raise ValueError(f"Unknown rules: {unknown}")
    if not rules or len(rules) != len(set(rules)):
        raise ValueError("--rules must be a nonempty list without duplicates")
    if "entropy_kgw" in rules:
        raise ValueError(
            "Entropy-KGW has inactive positions and is excluded from this "
            "matched-request analysis."
        )
    return rules


def fixed_bias_statistics(
    native_mass: np.ndarray, bias: float
) -> tuple[np.ndarray, np.ndarray]:
    mass = np.asarray(native_mass, dtype=np.float64)
    numerical_tolerance = 1e-5
    if np.any(~np.isfinite(mass)) or np.any(
        (mass < -numerical_tolerance) | (mass > 1.0 + numerical_tolerance)
    ):
        raise ValueError("native target masses must be finite probabilities")
    finfo = np.finfo(np.float64)
    mass = np.clip(mass, finfo.tiny, 1.0 - finfo.eps)
    exp_bias = math.exp(float(bias))
    normalizer = 1.0 - mass + exp_bias * mass
    controlled_mass = exp_bias * mass / normalizer
    kl = float(bias) * controlled_mass - np.log(normalizer)
    return controlled_mass, kl


def load_reference_texts(path: Path, count: int) -> list[str]:
    details = pd.read_csv(path)
    required = {"mode", "sequence_index", "generated_text", "target_rule"}
    missing = required.difference(details.columns)
    if missing:
        raise ValueError(f"Reference details missing columns: {sorted(missing)}")
    clean = details.loc[details["mode"].eq("unwatermarked")].copy()
    if clean.empty:
        raise ValueError("Reference details contain no unwatermarked rows")
    per_sequence = clean.groupby("sequence_index")["generated_text"].nunique(
        dropna=False
    )
    if not bool((per_sequence == 1).all()):
        raise ValueError("Reference unwatermarked text differs across rules")
    canonical = (
        clean.sort_values(["sequence_index", "target_rule"])
        .drop_duplicates("sequence_index")
        .sort_values("sequence_index")
    )
    canonical = canonical.loc[canonical["sequence_index"] < count]
    expected_indices = np.arange(count)
    if not np.array_equal(canonical["sequence_index"].to_numpy(), expected_indices):
        raise ValueError("Reference sequence indices are incomplete or reordered")
    return canonical["generated_text"].fillna("").astype(str).tolist()


def reference_ppl(
    path: Path, rules: Sequence[str]
) -> dict[str, tuple[float, float]]:
    summary = pd.read_csv(path)
    output: dict[str, tuple[float, float]] = {}
    for rule in rules:
        selected = summary.loc[summary["target_rule"].eq(rule)]
        clean = selected.loc[selected["mode"].eq("unwatermarked")]
        fixed = selected.loc[selected["mode"].eq("fixed_delta20")]
        if len(clean) != 1 or len(fixed) != 1:
            raise ValueError(f"Reference summary lacks unique rows for {rule}")
        output[rule] = (
            float(clean.iloc[0]["base_ppl_from_corpus_nll"]),
            float(fixed.iloc[0]["base_ppl_from_corpus_nll"]),
        )
    return output


def partition_hashes(root: Path) -> dict[str, str]:
    names = (
        "leaf_ids.npy",
        "semantic_group_ids.npy",
        "boundary_ids.npy",
        "banned_ids.npy",
        "hierarchy_config.json",
        "watermark_compositions.json",
    )
    return {name: sha256_file(root / name) for name in names}


@torch.inference_mode()
def analyze(args: argparse.Namespace) -> None:
    started = time.time()
    rules = parse_rules(args.rules)
    if not 0.0 < args.target_mass < 1.0:
        raise ValueError("--target-mass must lie strictly between zero and one")
    if args.generated_tokens <= 0 or args.test_count <= 0:
        raise ValueError("token and sequence counts must be positive")

    root = Path(__file__).resolve().parent
    manifest_path = (root / args.manifest).resolve()
    partition_path = (root / args.partition_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = (
        (root / args.reference_details).resolve() if args.reference_details else None
    )
    summary_path = (
        (root / args.reference_summary).resolve() if args.reference_summary else None
    )
    if args.require_reference_match and details_path is None:
        raise ValueError("--require-reference-match needs --reference-details")

    splits = load_manifest(manifest_path)
    if len(splits["test"]) < args.test_count:
        raise ValueError("Manifest has too few held-out prompts")
    prompts = [str(row["prompt"]) for row in splits["test"][: args.test_count]]
    prompt_hashes = np.asarray(
        [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts]
    )

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    partition = Partition.load(partition_path)
    model_vocab_size = int(model.get_output_embeddings().weight.shape[0])
    if partition.vocab_size != model_vocab_size:
        raise ValueError(
            f"Partition vocab {partition.vocab_size} != model vocab {model_vocab_size}"
        )
    validate_tokenizer_identity(
        partition.metadata, tokenizer, context=f"Partition {partition_path}"
    )
    partition_model_name = partition.metadata.get("model_name")
    if partition_model_name is not None and str(partition_model_name) != str(
        args.model_name
    ):
        raise ValueError(
            f"Partition model_name={partition_model_name!r} does not match "
            f"--model-name={args.model_name!r}."
        )
    if len(tokenizer) < partition.vocab_size:
        partition.leaf_ids[len(tokenizer) :] = -1
        padded = np.arange(len(tokenizer), partition.vocab_size, dtype=np.int64)
        partition.banned_ids = np.unique(
            np.concatenate([partition.banned_ids, padded])
        ).astype(np.int64)

    train_keys = [1000 + index for index in range(args.train_key_count)]
    eval_keys = [900000 + index for index in range(args.eval_key_count)]
    registry = TargetRegistry(
        partition, train_keys + eval_keys, model.get_input_embeddings().weight
    )
    schedules: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for rule in rules:
        local_keys, codes = make_rule_schedule(
            rule,
            len(prompts),
            args.generated_tokens,
            seed=stable_int(args.seed, "held-out-eval", rule),
            key_count=args.eval_key_count,
        )
        schedules[rule] = (local_keys + args.train_key_count, codes)

    shape = (len(rules), len(prompts), args.generated_tokens)
    native_mass = np.empty(shape, dtype=np.float32)
    required_bias = np.empty(shape, dtype=np.float32)
    watermarkability = np.empty(shape, dtype=np.float32)
    fixed_controlled_mass = np.empty(shape, dtype=np.float32)
    fixed_kl = np.empty(shape, dtype=np.float32)
    target_fraction = np.empty(shape, dtype=np.float32)
    observed_membership = np.empty(shape, dtype=bool)
    dynamic_seeds = np.full(shape, -1, dtype=np.int64)
    generated = np.empty((len(prompts), args.generated_tokens), dtype=np.int32)

    generator = torch.Generator(device=device).manual_seed(
        stable_int(args.seed, "eval")
    )
    for start in range(0, len(prompts), args.batch_size):
        stop = min(len(prompts), start + args.batch_size)
        encoded = tokenizer(
            prompts[start:stop],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_tokens,
        )
        current_ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        past = None
        for position in range(args.generated_tokens):
            outputs = model(
                input_ids=current_ids,
                attention_mask=attention,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            logits = outputs.logits[:, -1, :].float()
            probabilities = torch.softmax(logits, dim=-1)
            step_masks: list[torch.Tensor] = []
            for rule_index, rule in enumerate(rules):
                keys, codes = schedules[rule]
                batch_keys = torch.from_numpy(keys[start:stop].astype(np.int64))
                batch_codes = torch.from_numpy(
                    codes[start:stop, position].astype(np.int64)
                )
                rule_id = RULE_IDS[rule]
                seeds = np.full(stop - start, -1, dtype=np.int64)
                if rule_id in DYNAMIC_RULE_IDS:
                    for row in range(stop - start):
                        key_index = int(batch_keys[row])
                        seeds[row] = kgw_row_seed(
                            generated[start + row, :position].tolist(),
                            key=registry.keys[key_index],
                            seed_count=args.kgw_seed_count,
                            seed_offset=args.kgw_seed_offset,
                        )
                dynamic_seeds[rule_index, start:stop, position] = seeds
                mask = registry.mixed_masks(
                    batch_keys,
                    batch_codes,
                    torch.full_like(batch_keys, rule_id),
                    torch.from_numpy(seeds),
                    device=device,
                )
                step_masks.append(mask)
                mass = (probabilities * mask).sum(dim=-1).cpu().numpy()
                bias, frontier = minimum_intervention(mass, args.target_mass)
                fixed_mass, fixed_cost = fixed_bias_statistics(mass, args.fixed_bias)
                native_mass[rule_index, start:stop, position] = mass
                required_bias[rule_index, start:stop, position] = bias
                watermarkability[rule_index, start:stop, position] = frontier
                fixed_controlled_mass[rule_index, start:stop, position] = fixed_mass
                fixed_kl[rule_index, start:stop, position] = fixed_cost
                target_fraction[rule_index, start:stop, position] = (
                    mask.float().mean(dim=-1).cpu().numpy()
                )

            sampleable_logits = logits
            if len(tokenizer) < logits.shape[-1]:
                sampleable_logits = logits.clone()
                sampleable_logits[:, len(tokenizer) :] = -torch.inf
            next_tokens = sample_logits(
                sampleable_logits, args.temperature, generator, top_k=args.top_k
            )
            generated[start:stop, position] = (
                next_tokens.cpu().numpy().astype(np.int32)
            )
            for rule_index, mask in enumerate(step_masks):
                observed_membership[rule_index, start:stop, position] = (
                    mask.gather(1, next_tokens.unsqueeze(1))
                    .squeeze(1)
                    .cpu()
                    .numpy()
                )
            if position + 1 < args.generated_tokens:
                current_ids = next_tokens.unsqueeze(1)
                attention = torch.cat(
                    [
                        attention,
                        torch.ones(
                            (stop - start, 1),
                            dtype=attention.dtype,
                            device=device,
                        ),
                    ],
                    dim=1,
                )
        print(f"[cross-rule] {stop}/{len(prompts)}", flush=True)

    generated_text = [
        tokenizer.decode(row.tolist(), skip_special_tokens=True) for row in generated
    ]
    reference_match: bool | None = None
    if details_path is not None:
        expected = load_reference_texts(details_path, len(prompts))
        reference_match = generated_text == expected
        if args.require_reference_match and not reference_match:
            mismatches = sum(a != b for a, b in zip(generated_text, expected))
            raise RuntimeError(
                f"Shared trajectories differ from formal evaluation in {mismatches} rows"
            )

    ppl = reference_ppl(summary_path, rules) if summary_path is not None else {}
    summary_rows: list[dict[str, object]] = []
    sequence_rows: list[dict[str, object]] = []
    for rule_index, rule in enumerate(rules):
        clean_ppl, biased_ppl = ppl.get(rule, (math.nan, math.nan))
        log_ppl_increase = (
            math.log(biased_ppl) - math.log(clean_ppl)
            if clean_ppl > 0.0 and biased_ppl > 0.0
            else math.nan
        )
        summary_rows.append(
            {
                "model": args.model_name,
                "target_rule": rule,
                "sequences": len(prompts),
                "positions_per_sequence": args.generated_tokens,
                "mean_native_target_mass": float(native_mass[rule_index].mean()),
                "mean_required_bias_at_tau": float(required_bias[rule_index].mean()),
                "empirical_watermarkability_at_tau": float(
                    watermarkability[rule_index].mean()
                ),
                "mean_fixed_bias_kl": float(fixed_kl[rule_index].mean()),
                "mean_fixed_controlled_mass": float(
                    fixed_controlled_mass[rule_index].mean()
                ),
                "mean_target_vocab_fraction": float(
                    target_fraction[rule_index].mean()
                ),
                "unwatermarked_membership_rate": float(
                    observed_membership[rule_index].mean()
                ),
                "reference_unwatermarked_ppl": clean_ppl,
                "reference_fixed_bias_ppl": biased_ppl,
                "reference_log_ppl_increase": log_ppl_increase,
            }
        )
        for sequence_index in range(len(prompts)):
            sequence_rows.append(
                {
                    "model": args.model_name,
                    "target_rule": rule,
                    "sequence_index": sequence_index,
                    "mean_native_target_mass": float(
                        native_mass[rule_index, sequence_index].mean()
                    ),
                    "mean_required_bias_at_tau": float(
                        required_bias[rule_index, sequence_index].mean()
                    ),
                    "empirical_watermarkability_at_tau": float(
                        watermarkability[rule_index, sequence_index].mean()
                    ),
                    "mean_fixed_bias_kl": float(
                        fixed_kl[rule_index, sequence_index].mean()
                    ),
                    "unwatermarked_membership_rate": float(
                        observed_membership[rule_index, sequence_index].mean()
                    ),
                }
            )

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "cross_rule_summary.csv", index=False
    )
    pd.DataFrame(sequence_rows).to_csv(
        output_dir / "cross_rule_sequence_summary.csv", index=False
    )
    pd.DataFrame(
        {
            "sequence_index": np.arange(len(prompts)),
            "prompt_sha256": prompt_hashes,
            "generated_text": generated_text,
        }
    ).to_csv(output_dir / "shared_trajectories.csv", index=False)
    np.savez_compressed(
        output_dir / "cross_rule_steps.npz",
        rule_names=np.asarray(rules),
        prompt_sha256=prompt_hashes,
        generated_token_ids=generated,
        target_codes=np.stack([schedules[rule][1] for rule in rules]),
        key_indices=np.stack([schedules[rule][0] for rule in rules]),
        dynamic_seeds=dynamic_seeds,
        native_target_mass=native_mass,
        required_bias_at_tau=required_bias,
        watermarkability_at_tau=watermarkability,
        fixed_controlled_mass=fixed_controlled_mass,
        fixed_bias_kl=fixed_kl,
        target_vocab_fraction=target_fraction,
        observed_membership=observed_membership,
    )

    metadata = {
        "protocol_version": "cross_rule_shared_trajectory_v1",
        "model_name": args.model_name,
        "rules": rules,
        "test_count": args.test_count,
        "generated_tokens": args.generated_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "target_mass": args.target_mass,
        "fixed_bias": args.fixed_bias,
        "seed": args.seed,
        "sampling_seed": stable_int(args.seed, "eval"),
        "manifest_sha256": sha256_file(manifest_path),
        "partition_sha256": partition_hashes(partition_path),
        "source_sha256": {
            name: sha256_file(root / name)
            for name in (
                "analyze_cross_rule_watermarkability.py",
                "analyze_teacher_forced_watermarkability.py",
                "run_experiment.py",
                "targets.py",
            )
        },
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "tokenizer_commit_hash": tokenizer.init_kwargs.get("_commit_hash"),
        "generated_token_ids_sha256": hashlib.sha256(
            generated.tobytes()
        ).hexdigest(),
        "reference_details_sha256": (
            sha256_file(details_path) if details_path else None
        ),
        "reference_summary_sha256": (
            sha256_file(summary_path) if summary_path else None
        ),
        "reference_text_exact_match": reference_match,
        "runtime_seconds": time.time() - started,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(json.dumps(metadata, indent=2))


def main() -> None:
    analyze(parse_args())


if __name__ == "__main__":
    main()
