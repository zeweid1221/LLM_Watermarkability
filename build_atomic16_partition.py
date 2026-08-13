from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from watermarkability.partition_optimizer import (  # noqa: E402
    HierarchicalPartitionOptimizer,
    PartitionOptimizationConfig,
    evaluate_leaf_masses,
)
from reproducibility import tokenizer_identity  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an isolated 16-atom, probability-balanced partition."
    )
    parser.add_argument("--model-name", default="facebook/opt-125m")
    parser.add_argument("--manifest", default="data/lfqa_prompts_5756.jsonl")
    parser.add_argument(
        "--source-partition-dir",
        default=None,
        help=(
            "Optional legacy partition whose banned-token list is retained. "
            "The atomic partition can be built without another repository."
        ),
    )
    parser.add_argument("--output-dir", default="assets/atomic16/opt-125m")
    parser.add_argument("--calibration-count", type=int, default=5000)
    parser.add_argument("--validation-count", type=int, default=500)
    parser.add_argument("--max-prompt-tokens", type=int, default=160)
    parser.add_argument("--forward-batch-size", type=int, default=16)
    parser.add_argument("--partition-steps", type=int, default=160)
    parser.add_argument("--context-batch-size", type=int, default=256)
    parser.add_argument("--rounding-context-count", type=int, default=256)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--semantic-lsh-bits", type=int, default=8)
    parser.add_argument("--mass-strata-bins", type=int, default=256)
    parser.add_argument("--partition-learning-rate", type=float, default=0.08)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    local = Path(__file__).resolve().parent / path
    return local.resolve()


def load_manifest_prompts(path: Path, split: str, count: int) -> list[str]:
    prompts: list[str] = []
    sources: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("split")) != split:
                continue
            sources.add(str(row.get("source")))
            prompts.append(str(row["prompt"]).strip())
            if len(prompts) == count:
                break
    if len(prompts) != count:
        raise ValueError(f"Need {count} {split} prompts, found {len(prompts)} in {path}.")
    if sources != {"lfqa"}:
        raise ValueError(f"Atomic16 calibration must be LFQA-only, found sources={sorted(sources)}.")
    return prompts


@torch.inference_mode()
def collect_last_logits(
    model,
    tokenizer,
    prompts: list[str],
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        outputs = model(**encoded, use_cache=False, return_dict=True)
        positions = torch.arange(
            encoded["attention_mask"].shape[1], device=device
        ).unsqueeze(0)
        last_attended = positions.masked_fill(
            encoded["attention_mask"] == 0, -1
        ).max(dim=1).values
        if bool((last_attended < 0).any()):
            raise ValueError("Tokenizer produced an empty calibration prompt.")
        row_ids = torch.arange(len(batch), device=device)
        logits = (
            outputs.logits[row_ids, last_attended]
            .float()
            .cpu()
            .numpy()
            .astype(np.float16)
        )
        rows.append(logits)
        del outputs
    return np.concatenate(rows, axis=0)


def logits_to_probabilities(logits: np.ndarray, batch_size: int) -> np.ndarray:
    probabilities = np.empty(logits.shape, dtype=np.float32)
    for start in range(0, logits.shape[0], batch_size):
        stop = min(logits.shape[0], start + batch_size)
        values = torch.from_numpy(logits[start:stop].astype(np.float32, copy=False))
        probabilities[start:stop] = torch.softmax(values, dim=-1).numpy()
    return probabilities


def derive_banned_ids(
    tokenizer,
    embedding_vocab_size: int,
    source_banned_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = (
        np.asarray(source_banned_ids, dtype=np.int64)
        if source_banned_ids is not None
        else np.asarray([], dtype=np.int64)
    )
    special = np.asarray(
        [
            int(token_id)
            for token_id in tokenizer.all_special_ids
            if 0 <= int(token_id) < int(embedding_vocab_size)
        ],
        dtype=np.int64,
    )
    padded = np.arange(
        min(len(tokenizer), int(embedding_vocab_size)),
        int(embedding_vocab_size),
        dtype=np.int64,
    )
    return np.unique(np.concatenate([source, special, padded])), special, padded


def atom_masses(
    probabilities: np.ndarray,
    leaf_ids: np.ndarray,
    *,
    num_atoms: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    usable_ids = np.flatnonzero(leaf_ids >= 0)
    assignment = np.zeros((usable_ids.size, num_atoms), dtype=np.float32)
    assignment[np.arange(usable_ids.size), leaf_ids[usable_ids]] = 1.0
    assignment_tensor = torch.as_tensor(assignment, device=device)
    result = np.empty((probabilities.shape[0], num_atoms), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, probabilities.shape[0], batch_size):
            stop = min(probabilities.shape[0], start + batch_size)
            probs = torch.as_tensor(probabilities[start:stop, usable_ids], device=device)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
            result[start:stop] = (probs @ assignment_tensor).cpu().numpy()
    return result


def subset_score(masses: np.ndarray, atoms: Iterable[int], target: float) -> float:
    selected = masses[:, list(atoms)].sum(axis=1)
    error = np.abs(selected - target)
    return float(np.square(error).mean() + 0.10 * np.quantile(error, 0.95))


def best_subset(
    masses: np.ndarray,
    candidates: Iterable[int],
    size: int,
    target: float,
    *,
    skip_complements: bool = False,
) -> tuple[int, ...]:
    candidate_tuple = tuple(sorted(int(x) for x in candidates))
    best: tuple[int, ...] | None = None
    best_score = float("inf")
    for atoms in itertools.combinations(candidate_tuple, size):
        if skip_complements and atoms[0] != candidate_tuple[0]:
            continue
        score = subset_score(masses, atoms, target)
        if score < best_score:
            best = atoms
            best_score = score
    if best is None:
        raise RuntimeError("No atomic subset was selected.")
    return best


def choose_compositions(masses: np.ndarray) -> dict[str, object]:
    atoms = tuple(range(16))
    binary_0 = best_subset(masses, atoms, 8, 0.5, skip_complements=True)
    binary_1 = tuple(sorted(set(atoms) - set(binary_0)))

    boundary = best_subset(masses, atoms, 2, 2.0 / 16.0)
    payload_atoms = tuple(sorted(set(atoms) - set(boundary)))
    payload_0 = best_subset(masses, payload_atoms, 7, 7.0 / 16.0, skip_complements=True)
    payload_1 = tuple(sorted(set(payload_atoms) - set(payload_0)))

    remaining = set(atoms)
    cw_groups: list[tuple[int, ...]] = []
    for _ in range(3):
        group = best_subset(masses, sorted(remaining), 4, 0.25)
        cw_groups.append(group)
        remaining.difference_update(group)
    cw_groups.append(tuple(sorted(remaining)))

    return {
        "num_atoms": 16,
        "binary": {"bucket_0": list(binary_0), "bucket_1": list(binary_1)},
        "ecc": {
            "payload_0": list(payload_0),
            "payload_1": list(payload_1),
            "boundary": list(boundary),
            "composition": "7:7:2",
        },
        "cw_abcd": {
            label: list(group) for label, group in zip(("A", "B", "C", "D"), cw_groups)
        },
        "atomic_kgw": {
            "gamma_0.25": "prefix/key-dependent 4-of-16",
            "gamma_0.50": "prefix/key-dependent 8-of-16",
        },
    }


def composition_report(masses: np.ndarray, compositions: dict[str, object]) -> dict[str, object]:
    rows: dict[str, object] = {}
    groups = {
        "binary_0": compositions["binary"]["bucket_0"],
        "binary_1": compositions["binary"]["bucket_1"],
        "ecc_payload_0": compositions["ecc"]["payload_0"],
        "ecc_payload_1": compositions["ecc"]["payload_1"],
        "ecc_boundary": compositions["ecc"]["boundary"],
        **{f"cw_{label}": atoms for label, atoms in compositions["cw_abcd"].items()},
    }
    for name, atoms in groups.items():
        values = masses[:, atoms].sum(axis=1)
        rows[name] = {
            "atoms": atoms,
            "mean_mass": float(values.mean()),
            "std_mass": float(values.std()),
            "p05_mass": float(np.quantile(values, 0.05)),
            "p95_mass": float(np.quantile(values, 0.95)),
        }
    return rows


def main() -> None:
    args = parse_args()
    started = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    manifest_path = resolve_path(args.manifest)
    source_partition_path = (
        resolve_path(args.source_partition_dir) if args.source_partition_dir else None
    )
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_prompts = load_manifest_prompts(
        manifest_path, "train", args.calibration_count
    )
    validation_prompts = load_manifest_prompts(
        manifest_path, "validation", args.validation_count
    )
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    embedding_matrix = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    calibration_logits = collect_last_logits(
        model,
        tokenizer,
        calibration_prompts,
        batch_size=args.forward_batch_size,
        max_length=args.max_prompt_tokens,
        device=device,
    )
    validation_logits = collect_last_logits(
        model,
        tokenizer,
        validation_prompts,
        batch_size=args.forward_batch_size,
        max_length=args.max_prompt_tokens,
        device=device,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    calibration_probabilities = logits_to_probabilities(
        calibration_logits, args.validation_batch_size
    )
    validation_probabilities = logits_to_probabilities(
        validation_logits, args.validation_batch_size
    )
    del calibration_logits, validation_logits

    old_boundary_ids = np.asarray([], dtype=np.int64)
    source_banned_ids = np.asarray([], dtype=np.int64)
    if source_partition_path is not None:
        boundary_path = source_partition_path / "boundary_ids.npy"
        if not boundary_path.exists():
            boundary_path = source_partition_path / "bucket2_ids.npy"
        if boundary_path.exists():
            old_boundary_ids = np.load(
                boundary_path, allow_pickle=False
            ).astype(np.int64)
        banned_path = source_partition_path / "banned_ids.npy"
        if banned_path.exists():
            source_banned_ids = np.load(
                banned_path, allow_pickle=False
            ).astype(np.int64)

    banned_ids, tokenizer_special_ids, padded_output_ids = derive_banned_ids(
        tokenizer,
        int(embedding_matrix.shape[0]),
        source_banned_ids,
    )
    config = PartitionOptimizationConfig(
        num_leaves=16,
        frontiers=(2, 4, 8, 16),
        semantic_lsh_bits=args.semantic_lsh_bits,
        mass_strata_bins=args.mass_strata_bins,
        steps=args.partition_steps,
        learning_rate=args.partition_learning_rate,
        context_batch_size=args.context_batch_size,
        rounding_context_count=args.rounding_context_count,
        seed=args.seed,
    )
    optimizer = HierarchicalPartitionOptimizer(config)
    partition, history = optimizer.optimize(
        calibration_probabilities,
        embedding_matrix,
        boundary_ids=(),
        banned_ids=banned_ids,
        device=str(device),
        metadata={
            "partition_version": "model_specific_atomic16_v1",
            "model_name": args.model_name,
            "manifest": str(manifest_path),
            "calibration_split": "train",
            "validation_split": "validation",
            "old_boundary_tokens_reintegrated": len(old_boundary_ids),
            "source_partition_dir": (
                str(source_partition_path) if source_partition_path is not None else None
            ),
            "special_tokens_banned": int(tokenizer_special_ids.size),
            "padded_output_rows_banned": int(padded_output_ids.size),
            **tokenizer_identity(tokenizer),
        },
    )
    calibration_metrics = evaluate_leaf_masses(
        calibration_probabilities,
        partition.leaf_ids,
        config.frontiers,
        config.num_leaves,
        batch_size=args.validation_batch_size,
    )
    validation_metrics = evaluate_leaf_masses(
        validation_probabilities,
        partition.leaf_ids,
        config.frontiers,
        config.num_leaves,
        batch_size=args.validation_batch_size,
    )
    calibration_atom_masses = atom_masses(
        calibration_probabilities,
        partition.leaf_ids,
        num_atoms=16,
        batch_size=args.validation_batch_size,
        device=device,
    )
    validation_atom_masses = atom_masses(
        validation_probabilities,
        partition.leaf_ids,
        num_atoms=16,
        batch_size=args.validation_batch_size,
        device=device,
    )
    compositions = choose_compositions(calibration_atom_masses)
    compositions["selection_split"] = "train"
    compositions["validation"] = composition_report(validation_atom_masses, compositions)
    partition.boundary_ids = np.flatnonzero(
        np.isin(
            partition.leaf_ids,
            np.asarray(compositions["ecc"]["boundary"], dtype=np.int16),
        )
    ).astype(np.int64)
    partition.save(output_dir)
    pd.DataFrame(history).to_csv(output_dir / "partition_optimization_history.csv", index=False)
    (output_dir / "watermark_compositions.json").write_text(
        json.dumps(compositions, indent=2), encoding="utf-8"
    )

    counts = np.bincount(partition.leaf_ids[partition.leaf_ids >= 0], minlength=16)
    atom_rows = []
    for atom in range(16):
        atom_rows.append(
            {
                "atom": atom,
                "token_count": int(counts[atom]),
                "calibration_mean_mass": float(calibration_atom_masses[:, atom].mean()),
                "calibration_std_mass": float(calibration_atom_masses[:, atom].std()),
                "validation_mean_mass": float(validation_atom_masses[:, atom].mean()),
                "validation_std_mass": float(validation_atom_masses[:, atom].std()),
            }
        )
    pd.DataFrame(atom_rows).to_csv(output_dir / "atom_statistics.csv", index=False)
    report = {
        "model_name": args.model_name,
        "manifest": str(manifest_path),
        "calibration_contexts": len(calibration_prompts),
        "validation_contexts": len(validation_prompts),
        "num_atoms": 16,
        "old_boundary_tokens_reintegrated": len(old_boundary_ids),
        "banned_tokens": len(partition.banned_ids),
        "calibration_metrics": calibration_metrics,
        "validation_metrics": validation_metrics,
        "runtime_seconds": time.time() - started,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    (output_dir / "atomic16_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
