from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from modeling import TargetConditionedLoRA
from run_experiment import generate_mode, load_adapter, load_manifest, write_csv
from targets import Partition, TargetRegistry, make_fixed_ecc_schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe prompt/key in-sample generalization of a trained adapter."
    )
    parser.add_argument("--manifest", default="data/mixed_prompts_5756.jsonl")
    parser.add_argument("--partition-dir", default="assets/opt_hierarchy")
    parser.add_argument("--adapter-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="facebook/opt-125m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--generated-tokens", type=int, default=96)
    parser.add_argument("--max-prompt-tokens", type=int, default=160)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--train-key-count", type=int, default=16)
    parser.add_argument("--eval-key-count", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--target-adherence", type=float, default=0.99)
    parser.add_argument("--condition-summary-mode", default="registry_mean")
    parser.add_argument("--condition-top-k", type=int, default=128)
    parser.add_argument("--kgw-seed-count", type=int, default=1)
    parser.add_argument("--kgw-seed-offset", type=int, default=3779)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    manifest_path = (root / args.manifest).resolve()
    partition_path = (root / args.partition_dir).resolve()
    checkpoint_path = Path(args.adapter_checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (root / checkpoint_path).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    splits = load_manifest(manifest_path)
    if len(splits["train"]) < args.sample_count:
        raise ValueError("The manifest does not contain enough training prompts.")
    if len(splits["test"]) < args.sample_count:
        raise ValueError("The manifest does not contain enough test prompts.")
    prompts = {
        "train_prompt": [
            str(row["prompt"]) for row in splits["train"][: args.sample_count]
        ],
        "heldout_prompt": [
            str(row["prompt"]) for row in splits["test"][: args.sample_count]
        ],
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        dtype=dtype,
    ).to(device)
    model = TargetConditionedLoRA(
        base_model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    ).to(device)
    load_adapter(model, checkpoint_path)

    partition = Partition.load(partition_path)
    model_vocab_size = int(base_model.get_output_embeddings().weight.shape[0])
    if partition.vocab_size != model_vocab_size:
        raise ValueError(
            f"Partition vocab {partition.vocab_size} != model vocab {model_vocab_size}."
        )
    if len(tokenizer) < partition.vocab_size:
        partition.leaf_ids[len(tokenizer) :] = -1
        padded_ids = np.arange(len(tokenizer), partition.vocab_size, dtype=np.int64)
        partition.banned_ids = np.unique(
            np.concatenate([partition.banned_ids, padded_ids])
        ).astype(np.int64)

    train_keys = [1000 + index for index in range(args.train_key_count)]
    eval_keys = [900000 + index for index in range(args.eval_key_count)]
    registry = TargetRegistry(
        partition,
        train_keys + eval_keys,
        base_model.get_input_embeddings().weight,
    )

    # These are exactly the key/code schedules assigned to the first training
    # rows. Reusing the codes across all four cells isolates prompt and key
    # generalization instead of changing the watermark target sequence.
    train_local_keys, target_codes = make_fixed_ecc_schedule(
        args.sample_count,
        args.generated_tokens,
        seed=args.seed,
        key_count=args.train_key_count,
    )
    key_groups = {
        "train_key": train_local_keys,
        "heldout_key": train_local_keys + args.train_key_count,
    }

    details: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    modes = ("closed_form", "lora_only", "lora_residual")
    for prompt_group, group_prompts in prompts.items():
        for key_group, key_indices in key_groups.items():
            condition = f"{prompt_group}__{key_group}"
            condition_summaries: list[dict[str, object]] = []
            for mode in modes:
                mode_rows, mode_summary = generate_mode(
                    model,
                    tokenizer,
                    group_prompts,
                    registry,
                    key_indices,
                    target_codes,
                    mode=mode,
                    rule_name="ecc",
                    args=args,
                    device=device,
                )
                for row, prompt in zip(mode_rows, group_prompts):
                    row["probe_condition"] = condition
                    row["prompt_group"] = prompt_group
                    row["key_group"] = key_group
                    row["prompt"] = prompt
                    row["source"] = "in_sample_probe"
                mode_summary.update(
                    {
                        "probe_condition": condition,
                        "prompt_group": prompt_group,
                        "key_group": key_group,
                    }
                )
                details.extend(mode_rows)
                condition_summaries.append(mode_summary)

            closed_ppl = float(
                next(
                    row["base_ppl_from_corpus_nll"]
                    for row in condition_summaries
                    if row["mode"] == "closed_form"
                )
            )
            for row in condition_summaries:
                ppl = float(row["base_ppl_from_corpus_nll"])
                row["ppl_reduction_vs_closed_pct"] = (
                    100.0 * (closed_ppl - ppl) / closed_ppl
                    if row["mode"] == "lora_residual"
                    else None
                )
            summaries.extend(condition_summaries)

    write_csv(output_dir / "insample_probe_details.csv", details)
    write_csv(output_dir / "insample_probe_summary.csv", summaries)
    (output_dir / "insample_probe_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    config = {
        **vars(args),
        "adapter_checkpoint_resolved": str(checkpoint_path),
        "device_resolved": str(device),
        "probe_design": {
            "train_prompt_train_key": "prompt and exact key/code schedule seen in training",
            "train_prompt_heldout_key": "seen prompt, same code schedule, unseen key",
            "heldout_prompt_train_key": "unseen prompt, seen key and code schedule",
            "heldout_prompt_heldout_key": "unseen prompt and unseen key",
        },
    }
    (output_dir / "insample_probe_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
