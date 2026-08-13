from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from modeling import (
    TargetConditionedLoRA,
    apply_closed_form_residual,
    target_geometry_features,
    topk_contrast_summaries,
)
from reproducibility import (
    array_sha256,
    canonical_json,
    encode_prompt,
    files_sha256,
    sha256_file,
    text_sequence_sha256,
    tokenizer_identity,
    validate_tokenizer_identity,
)
from targets import (
    RULE_ECC,
    RULE_KGW,
    RULE_ATOMIC_KGW,
    RULE_ATOMIC_KGW25,
    RULE_ENTROPY_KGW,
    RULE_UNIGRAM,
    COMPOSITION_CARDINALITY_BY_RULE,
    DYNAMIC_RULE_IDS,
    RULE_IDS,
    RULE_NAMES,
    Partition,
    TargetRegistry,
    kgw_row_seed,
    make_fixed_ecc_schedule,
    make_composition_curriculum,
    make_rule_schedule,
    make_rule_assignments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the target-conditioned watermarkability adapter."
    )
    parser.add_argument("--manifest", default="data/mixed_prompts_5756.jsonl")
    parser.add_argument("--partition-dir", default="assets/opt_hierarchy")
    parser.add_argument("--output-dir", default="outputs/opt125m_lora_5k_seed20260810")
    parser.add_argument("--model-name", default="facebook/opt-125m")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-count", type=int, default=5000)
    parser.add_argument("--validation-count", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=256)
    parser.add_argument("--generated-tokens", type=int, default=96)
    parser.add_argument("--max-prompt-tokens", type=int, default=160)
    parser.add_argument("--trajectory-batch-size", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--residual-loss-weight", type=float, default=0.02)
    parser.add_argument("--target-mass-loss-weight", type=float, default=0.0)
    parser.add_argument("--kl-loss-weight", type=float, default=0.02)
    parser.add_argument("--target-adherence", type=float, default=0.99)
    parser.add_argument("--kgw-train-fraction", type=float, default=0.0)
    parser.add_argument(
        "--train-rules",
        default="ecc",
        help="Comma-separated rules; each receives the full train/validation set.",
    )
    parser.add_argument(
        "--training-protocol",
        choices=("composition_pool", "schedule_pool"),
        default="composition_pool",
        help="Train on generic atom unions or reproduce the legacy schedule-pooled protocol.",
    )
    parser.add_argument("--composition-cardinalities", default="2,4,7,8")
    parser.add_argument("--composition-repeats", type=int, default=4)
    parser.add_argument("--eval-rules", default="ecc")
    parser.add_argument(
        "--condition-summary-mode",
        choices=("registry_mean", "topk_contrast"),
        default="registry_mean",
    )
    parser.add_argument("--condition-top-k", type=int, default=128)
    parser.add_argument(
        "--condition-feature-mode",
        choices=("none", "geometry"),
        default="none",
        help="Optionally append rule-agnostic target geometry features.",
    )
    parser.add_argument(
        "--objective-aggregation",
        choices=("token_micro", "rule_macro"),
        default="token_micro",
        help="Aggregate active-token losses globally or equally across rules.",
    )
    parser.add_argument(
        "--ecc-boundary-loss-weight",
        type=float,
        default=1.0,
        help="Relative loss weight for ECC boundary positions (target code 3).",
    )
    parser.add_argument("--kgw-seed-count", type=int, default=1)
    parser.add_argument("--kgw-seed-offset", type=int, default=3779)
    parser.add_argument(
        "--entropy-gate-threshold",
        type=float,
        default=2.0,
        help="Enable entropy-gated KGW only when base next-token entropy reaches this value (nats).",
    )
    parser.add_argument("--training-targets-per-sequence", type=int, default=0)
    parser.add_argument("--train-key-count", type=int, default=16)
    parser.add_argument("--eval-key-count", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--adapter-checkpoint")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--trajectory-cache")
    return parser.parse_args()


def stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def load_manifest(path: Path) -> dict[str, list[dict[str, object]]]:
    splits = {"train": [], "validation": [], "test": []}
    seen_prompts: dict[str, tuple[str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "split" not in row or "prompt" not in row:
                raise ValueError(
                    f"Manifest row {line_number} must contain split and prompt."
                )
            split = str(row["split"])
            if split not in splits:
                raise ValueError(
                    f"Manifest row {line_number} has unknown split {split!r}."
                )
            prompt = str(row["prompt"]).strip()
            if not prompt:
                raise ValueError(f"Manifest row {line_number} has an empty prompt.")
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if prompt_hash in seen_prompts:
                previous_split, previous_line = seen_prompts[prompt_hash]
                raise ValueError(
                    "Duplicate prompt across manifest rows "
                    f"{previous_line} ({previous_split}) and {line_number} ({split})."
                )
            seen_prompts[prompt_hash] = (split, line_number)
            row["prompt"] = prompt
            splits[split].append(row)
    return splits


def trajectory_cache_metadata(
    *,
    prompts: Sequence[str],
    key_indices: np.ndarray,
    target_codes: np.ndarray,
    rule_ids: np.ndarray,
    preset_mask_indices: np.ndarray | None,
    manifest_path: Path,
    partition_path: Path,
    tokenizer,
    args: argparse.Namespace,
) -> dict[str, object]:
    partition_files = [
        "leaf_ids.npy",
        "semantic_group_ids.npy",
        "boundary_ids.npy",
        "banned_ids.npy",
        "hierarchy_config.json",
        "watermark_compositions.json",
    ]
    partition_files = [
        name for name in partition_files if (partition_path / name).is_file()
    ]
    return {
        "format_version": "trajectory-cache-v2-special-token-prompts",
        "model_name": str(args.model_name),
        "tokenizer": tokenizer_identity(tokenizer),
        "manifest_sha256": sha256_file(manifest_path),
        "partition_sha256": files_sha256(partition_path, partition_files),
        "prompts_sha256": text_sequence_sha256(prompts),
        "key_indices_sha256": array_sha256(key_indices),
        "target_codes_sha256": array_sha256(target_codes),
        "rule_ids_sha256": array_sha256(rule_ids),
        "preset_mask_indices_sha256": (
            array_sha256(preset_mask_indices)
            if preset_mask_indices is not None
            else None
        ),
        "generated_tokens": int(args.generated_tokens),
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "temperature": float(args.temperature),
        "kgw_seed_count": int(args.kgw_seed_count),
        "kgw_seed_offset": int(args.kgw_seed_offset),
        "entropy_gate_threshold": float(args.entropy_gate_threshold),
        "seed": int(args.seed),
        "training_protocol": str(args.training_protocol),
    }


def validate_trajectory_cache_metadata(
    cache: np.lib.npyio.NpzFile, expected: dict[str, object]
) -> None:
    if "metadata_json" not in cache.files:
        raise ValueError(
            "Trajectory cache has no reproducibility metadata. Rebuild it without "
            "--skip-prepare."
        )
    actual = json.loads(str(cache["metadata_json"].item()))
    if canonical_json(actual) != canonical_json(expected):
        changed = sorted(
            key
            for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise ValueError(
            "Trajectory cache does not match this run configuration; changed "
            f"fields: {changed}. Rebuild it without --skip-prepare."
        )


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_rule_names(value: str, *, option: str) -> list[str]:
    names = [item.strip().lower() for item in str(value).split(",") if item.strip()]
    unknown = sorted(set(names) - set(RULE_IDS))
    if unknown:
        raise ValueError(f"Unknown {option} values: {unknown}")
    if not names:
        raise ValueError(f"{option} must contain at least one rule.")
    if len(names) != len(set(names)):
        raise ValueError(f"{option} contains duplicate rules: {names}")
    return names


def parse_cardinalities(value: str) -> list[int]:
    cardinalities = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    unknown = sorted(set(cardinalities) - set(COMPOSITION_CARDINALITY_BY_RULE.values()))
    if unknown or not cardinalities:
        raise ValueError(
            "--composition-cardinalities must be a nonempty subset of 2,4,7,8; "
            f"received {cardinalities}."
        )
    if len(cardinalities) != len(set(cardinalities)):
        raise ValueError("--composition-cardinalities contains duplicates.")
    return cardinalities


def composition_coverage_report(
    rule_ids: np.ndarray,
    combination_indices: np.ndarray,
    *,
    num_atoms: int,
) -> list[dict[str, int | float]]:
    rows: list[dict[str, int | float]] = []
    for rule_id, cardinality in sorted(
        COMPOSITION_CARDINALITY_BY_RULE.items(), key=lambda item: item[1]
    ):
        selected = np.asarray(rule_ids) == int(rule_id)
        if not bool(selected.any()):
            continue
        observed = int(np.unique(np.asarray(combination_indices)[selected]).size)
        possible = int(math.comb(int(num_atoms), int(cardinality)))
        rows.append(
            {
                "cardinality": int(cardinality),
                "sequences": int(selected.sum()),
                "positions": int(selected.sum() * combination_indices.shape[1]),
                "observed_unions": observed,
                "possible_unions": possible,
                "coverage": observed / possible,
            }
        )
    return rows


def expand_rule_targets(
    prompts: Sequence[str],
    rule_names: Sequence[str],
    *,
    generated_tokens: int,
    seed: int,
    key_count: int,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Give every rule the complete prompt set rather than splitting examples."""
    expanded_prompts: list[str] = []
    key_parts: list[np.ndarray] = []
    code_parts: list[np.ndarray] = []
    rule_parts: list[np.ndarray] = []
    for rule_name in rule_names:
        keys, codes = make_rule_schedule(
            rule_name,
            len(prompts),
            generated_tokens,
            seed=stable_int(seed, "rule-schedule", rule_name),
            key_count=key_count,
        )
        expanded_prompts.extend(str(prompt) for prompt in prompts)
        key_parts.append(keys)
        code_parts.append(codes)
        rule_parts.append(
            np.full(len(prompts), RULE_IDS[rule_name], dtype=np.uint8)
        )
    return (
        expanded_prompts,
        np.concatenate(key_parts, axis=0),
        np.concatenate(code_parts, axis=0),
        np.concatenate(rule_parts, axis=0),
    )


def sample_logits(
    logits: torch.Tensor,
    temperature: float,
    generator: torch.Generator,
    *,
    top_k: int = 0,
) -> torch.Tensor:
    scores = logits.float() / float(temperature)
    if 0 < int(top_k) < scores.shape[-1]:
        cutoff = torch.topk(scores, k=int(top_k), dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < cutoff, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(-1)


def next_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


@torch.inference_mode()
def prepare_trajectories(
    model: TargetConditionedLoRA,
    tokenizer,
    prompts: Sequence[str],
    registry: TargetRegistry,
    key_indices: np.ndarray,
    target_codes: np.ndarray,
    rule_ids: np.ndarray,
    preset_mask_indices: np.ndarray | None,
    *,
    max_prompt_tokens: int,
    batch_size: int,
    temperature: float,
    kgw_seed_count: int,
    kgw_seed_offset: int,
    entropy_gate_threshold: float,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    generated_tokens = int(target_codes.shape[1])
    generated = np.empty((len(prompts), generated_tokens), dtype=np.int32)
    kgw_seeds = (
        np.asarray(preset_mask_indices, dtype=np.int64).copy()
        if preset_mask_indices is not None
        else np.full((len(prompts), generated_tokens), -1, dtype=np.int64)
    )
    if kgw_seeds.shape != target_codes.shape:
        raise ValueError("Preset mask indices must match target_codes shape.")
    model.eval()
    tokenizer.padding_side = "left"
    generator = torch.Generator(device=device).manual_seed(int(seed))
    for start in range(0, len(prompts), batch_size):
        stop = min(len(prompts), start + batch_size)
        encoded = tokenizer(
            list(prompts[start:stop]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_adapters=False,
            use_cache=True,
            return_dict=True,
        )
        batch_keys = torch.from_numpy(key_indices[start:stop].astype(np.int64))
        batch_rules = torch.from_numpy(rule_ids[start:stop].astype(np.int64))
        for position in range(generated_tokens):
            step_logits = outputs.logits[:, -1, :].float()
            step_codes = target_codes[start:stop, position].astype(np.int64).copy()
            entropy_rows = batch_rules.numpy() == RULE_ENTROPY_KGW
            if bool(entropy_rows.any()):
                entropies = next_token_entropy(step_logits).cpu().numpy()
                inactive = entropy_rows & (entropies < float(entropy_gate_threshold))
                step_codes[inactive] = -1
                target_codes[start:stop, position] = step_codes
            codes = torch.from_numpy(step_codes)
            position_seeds = kgw_seeds[start:stop, position].copy()
            for local_row in range(stop - start):
                if int(batch_rules[local_row]) not in DYNAMIC_RULE_IDS or int(codes[local_row]) < 0:
                    continue
                key_index = int(batch_keys[local_row])
                position_seeds[local_row] = kgw_row_seed(
                    generated[start + local_row, :position].tolist(),
                    key=registry.keys[key_index],
                    seed_count=kgw_seed_count,
                    seed_offset=kgw_seed_offset,
                )
            kgw_seeds[start:stop, position] = position_seeds
            active_rows = codes >= 0
            target_mask = torch.from_numpy(
                np.repeat(registry.kgw_eligible_mask[None, :], stop - start, axis=0)
            ).to(device)
            active_device = active_rows.to(device)
            if bool(active_rows.any()):
                target_mask[active_device] = registry.mixed_masks(
                    batch_keys[active_rows],
                    codes[active_rows],
                    batch_rules[active_rows],
                    torch.from_numpy(position_seeds)[active_rows],
                    device=device,
                )
            hard_logits = step_logits.masked_fill(~target_mask, -torch.inf)
            next_tokens = sample_logits(hard_logits, temperature, generator)
            generated[start:stop, position] = next_tokens.cpu().numpy().astype(np.int32)
            if position + 1 < generated_tokens:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((stop - start, 1), device=device, dtype=attention_mask.dtype)],
                    dim=1,
                )
                outputs = model(
                    input_ids=next_tokens.unsqueeze(1),
                    attention_mask=attention_mask,
                    use_adapters=False,
                    past_key_values=outputs.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
        print(f"[trajectory] {stop}/{len(prompts)}", flush=True)
    return generated, kgw_seeds


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        prompts: Sequence[str],
        generated_ids: np.ndarray,
        key_indices: np.ndarray,
        target_codes: np.ndarray,
        rule_ids: np.ndarray,
        kgw_seeds: np.ndarray,
        tokenizer,
        max_prompt_tokens: int,
    ):
        self.prompt_ids = [
            encode_prompt(tokenizer, prompt, max_length=max_prompt_tokens)
            for prompt in prompts
        ]
        self.generated_ids = generated_ids
        self.key_indices = key_indices
        self.target_codes = target_codes
        self.rule_ids = rule_ids
        self.kgw_seeds = kgw_seeds

    def __len__(self) -> int:
        return len(self.prompt_ids)

    def __getitem__(self, index: int):
        prompt = self.prompt_ids[index]
        return (
            prompt,
            self.generated_ids[index].astype(np.int64),
            int(self.key_indices[index]),
            self.target_codes[index].astype(np.int64),
            int(self.rule_ids[index]),
            self.kgw_seeds[index].astype(np.int64),
            int(index),
        )


def collate_trajectories(rows, pad_token_id: int):
    lengths = [len(prompt) + len(generated) - 1 for prompt, generated, *_ in rows]
    max_length = max(lengths)
    batch_size = len(rows)
    input_ids = torch.full((batch_size, max_length), int(pad_token_id), dtype=torch.long)
    attention = torch.zeros((batch_size, max_length), dtype=torch.long)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
    target_codes = torch.full((batch_size, max_length), -1, dtype=torch.long)
    key_indices = torch.full((batch_size, max_length), -1, dtype=torch.long)
    rule_ids = torch.full((batch_size, max_length), -1, dtype=torch.long)
    kgw_seeds = torch.full((batch_size, max_length), -1, dtype=torch.long)
    sequence_indices = torch.empty(batch_size, dtype=torch.long)
    for row_index, (
        prompt,
        generated,
        key_index,
        codes,
        rule_id,
        row_kgw_seeds,
        sequence_index,
    ) in enumerate(rows):
        sequence = list(prompt) + generated[:-1].tolist()
        length = len(sequence)
        input_ids[row_index, :length] = torch.tensor(sequence, dtype=torch.long)
        attention[row_index, :length] = 1
        first_prediction = len(prompt) - 1
        last_prediction = first_prediction + len(generated)
        labels[row_index, first_prediction:last_prediction] = torch.from_numpy(generated)
        target_codes[row_index, first_prediction:last_prediction] = torch.from_numpy(codes)
        key_indices[row_index, first_prediction:last_prediction] = int(key_index)
        rule_ids[row_index, first_prediction:last_prediction] = int(rule_id)
        kgw_seeds[row_index, first_prediction:last_prediction] = torch.from_numpy(row_kgw_seeds)
        sequence_indices[row_index] = int(sequence_index)
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "labels": labels,
        "target_codes": target_codes,
        "key_indices": key_indices,
        "rule_ids": rule_ids,
        "kgw_seeds": kgw_seeds,
        "sequence_indices": sequence_indices,
    }


def select_active_positions(
    target_codes: torch.Tensor,
    sequence_indices: torch.Tensor,
    *,
    max_per_sequence: int,
    seed: int,
) -> torch.Tensor:
    active = target_codes >= 0
    limit = int(max_per_sequence)
    if limit <= 0:
        return active
    selected = torch.zeros_like(active)
    for row in range(active.shape[0]):
        positions = torch.nonzero(active[row], as_tuple=False).flatten().cpu().numpy()
        if positions.size > limit:
            rng = np.random.default_rng(
                stable_int("target-position-sample", seed, int(sequence_indices[row]))
            )
            positions = np.sort(rng.choice(positions, size=limit, replace=False))
        selected[row, torch.from_numpy(positions).to(selected.device)] = True
    return selected


def build_condition_tensor(
    registry: TargetRegistry,
    target_codes: torch.Tensor,
    key_indices: torch.Tensor,
    rule_ids: torch.Tensor,
    kgw_seeds: torch.Tensor,
    active: torch.Tensor,
    base_logits: torch.Tensor,
    embedding_weight: torch.Tensor,
    *,
    summary_mode: str,
    condition_top_k: int,
    condition_feature_mode: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    summaries = torch.zeros(
        (*target_codes.shape, registry.summaries.shape[-1]), dtype=torch.float32, device=device
    )
    active_keys = key_indices[active].long()
    active_codes = target_codes[active].long()
    active_rules = rule_ids[active].long()
    active_kgw_seeds = kgw_seeds[active].long()
    masks = registry.mixed_masks(
        active_keys,
        active_codes,
        active_rules,
        active_kgw_seeds,
        device=device,
    )
    if summary_mode == "registry_mean":
        if bool((active_rules != RULE_ECC).any()):
            raise ValueError(
                "Non-ECC target rules require --condition-summary-mode topk_contrast."
            )
        descriptor_ids = active_keys * 4 + active_codes
        active_summaries = registry.summaries.index_select(0, descriptor_ids.cpu()).to(device)
    elif summary_mode == "topk_contrast":
        active_summaries = topk_contrast_summaries(
            base_logits,
            masks,
            embedding_weight,
            top_k=condition_top_k,
        )
    else:
        raise ValueError(f"Unknown condition summary mode: {summary_mode}")
    summaries[active] = active_summaries
    features = None
    if condition_feature_mode == "geometry":
        active_features = target_geometry_features(
            base_logits,
            masks,
        )
        features = torch.zeros(
            (*target_codes.shape, active_features.shape[-1]),
            dtype=torch.float32,
            device=device,
        )
        features[active] = active_features
    elif condition_feature_mode != "none":
        raise ValueError(
            f"Unknown condition feature mode: {condition_feature_mode}"
        )
    return active, summaries, masks, features


def aggregate_active_loss(
    values: torch.Tensor,
    rule_ids: torch.Tensor,
    target_codes: torch.Tensor,
    *,
    aggregation: str,
    ecc_boundary_weight: float,
) -> torch.Tensor:
    """Aggregate per-position losses while preserving legacy defaults."""
    if values.ndim != 1 or rule_ids.shape != values.shape or target_codes.shape != values.shape:
        raise ValueError("Active loss values and descriptors must be one-dimensional.")
    weights = torch.ones_like(values)
    boundary = (rule_ids == RULE_ECC) & (target_codes == 3)
    weights[boundary] = float(ecc_boundary_weight)
    if bool((weights <= 0).any()):
        raise ValueError("Loss weights must be positive.")
    if aggregation == "token_micro":
        return (values * weights).sum() / weights.sum()
    if aggregation == "rule_macro":
        per_rule = []
        for rule_id in torch.unique(rule_ids).tolist():
            selected = rule_ids == int(rule_id)
            per_rule.append(
                (values[selected] * weights[selected]).sum() / weights[selected].sum()
            )
        return torch.stack(per_rule).mean()
    raise ValueError(f"Unknown objective aggregation: {aggregation}")


def batch_objective(
    model: TargetConditionedLoRA,
    registry: TargetRegistry,
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    input_ids = batch["input_ids"].to(device)
    attention = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    target_codes = batch["target_codes"].to(device)
    key_indices = batch["key_indices"].to(device)
    rule_ids = batch["rule_ids"].to(device)
    kgw_seeds = batch["kgw_seeds"].to(device)
    active = select_active_positions(
        target_codes,
        batch["sequence_indices"],
        max_per_sequence=args.training_targets_per_sequence,
        seed=args.seed,
    )
    with torch.no_grad():
        base_outputs = model(
            input_ids=input_ids,
            attention_mask=attention,
            use_adapters=False,
            use_cache=False,
            return_dict=True,
        )
        base_logits = base_outputs.logits[active].float()
    active, summaries, masks, features = build_condition_tensor(
        registry,
        target_codes,
        key_indices,
        rule_ids,
        kgw_seeds,
        active,
        base_logits,
        model.base_model.get_input_embeddings().weight,
        summary_mode=args.condition_summary_mode,
        condition_top_k=args.condition_top_k,
        condition_feature_mode=getattr(args, "condition_feature_mode", "none"),
        device=device,
    )
    adapted_outputs = model(
        input_ids=input_ids,
        attention_mask=attention,
        target_summaries=summaries,
        target_features=features,
        use_adapters=True,
        use_cache=False,
        return_dict=True,
    )
    adapted_logits = adapted_outputs.logits[active].float()
    active_labels = labels[active]
    residual = apply_closed_form_residual(
        adapted_logits,
        masks,
        target_adherence=args.target_adherence,
    )
    per_token_nll = F.cross_entropy(
        residual.controlled_logits, active_labels, reduction="none"
    )
    adapted_log_probs = F.log_softmax(adapted_logits, dim=-1)
    adapted_probs = adapted_log_probs.exp()
    base_log_probs = F.log_softmax(base_logits, dim=-1)
    per_token_kl = (adapted_probs * (adapted_log_probs - base_log_probs)).sum(dim=-1)
    per_token_residual = residual.residual_bias.square()
    per_token_target_mass = -residual.native_mass.log()
    active_rules = rule_ids[active].long()
    active_codes = target_codes[active].long()
    aggregate = lambda values: aggregate_active_loss(
        values,
        active_rules,
        active_codes,
        aggregation=getattr(args, "objective_aggregation", "token_micro"),
        ecc_boundary_weight=getattr(args, "ecc_boundary_loss_weight", 1.0),
    )
    nll = aggregate(per_token_nll)
    conditional_kl = aggregate(per_token_kl)
    residual_loss = aggregate(per_token_residual)
    target_mass_loss = aggregate(per_token_target_mass)
    loss = (
        nll
        + args.residual_loss_weight * residual_loss
        + args.target_mass_loss_weight * target_mass_loss
        + args.kl_loss_weight * conditional_kl
    )
    return loss, {
        "loss": float(loss.detach()),
        "controlled_nll": float(nll.detach()),
        "residual_loss": float(residual_loss.detach()),
        "target_mass_loss": float(target_mass_loss.detach()),
        "conditional_kl": float(conditional_kl.detach()),
        "native_target_mass": float(residual.native_mass.mean().detach()),
        "mean_residual_bias": float(residual.residual_bias.mean().detach()),
        "metric_weight": float(per_token_nll.numel()),
    }


def weighted_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot aggregate an empty metric list.")
    weights = np.asarray([row["metric_weight"] for row in rows], dtype=np.float64)
    if not bool(np.isfinite(weights).all()) or float(weights.sum()) <= 0.0:
        raise ValueError("Metric weights must be finite and positive in aggregate.")
    return {
        name: float(
            np.average(
                np.asarray([row[name] for row in rows], dtype=np.float64),
                weights=weights,
            )
        )
        for name in rows[0]
        if name != "metric_weight"
    }


@torch.no_grad()
def validate(
    model: TargetConditionedLoRA,
    registry: TargetRegistry,
    loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    for batch in loader:
        _, metrics = batch_objective(model, registry, batch, args, device)
        rows.append(metrics)
    return weighted_metrics(rows)


def train_adapter(
    model: TargetConditionedLoRA,
    registry: TargetRegistry,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> list[dict[str, object]]:
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, object]] = []
    best_loss = math.inf
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.base_model.eval()
        model.conditioner.train()
        for layer in model.lora_layers:
            layer.dropout.train()
        running: list[dict[str, float]] = []
        for step, batch in enumerate(train_loader, start=1):
            loss, metrics = batch_objective(model, registry, batch, args, device)
            (loss / args.gradient_accumulation).backward()
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running.append(metrics)
            if step % 100 == 0:
                print(
                    f"[train] epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={np.mean([row['loss'] for row in running[-100:]]):.4f}",
                    flush=True,
                )
        validation = validate(model, registry, validation_loader, args, device)
        row: dict[str, object] = {"epoch": epoch}
        row.update(
            {
                f"train_{name}": value
                for name, value in weighted_metrics(running).items()
            }
        )
        row.update({f"validation_{name}": value for name, value in validation.items()})
        history.append(row)
        print(json.dumps(row, indent=2), flush=True)
        torch.save(model.adapter_state_dict(), output_dir / f"adapter_epoch{epoch}.pt")
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save(model.adapter_state_dict(), output_dir / "adapter_best.pt")
    write_csv(output_dir / "training_history.csv", history)
    return history


def load_adapter(model: TargetConditionedLoRA, path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = set(model.adapter_state_dict())
    supplied = set(state)
    missing_adapter = sorted(expected - supplied)
    unexpected_adapter = sorted(supplied - expected)
    if missing_adapter or unexpected_adapter:
        raise RuntimeError(
            "Adapter checkpoint keys do not match this architecture. "
            f"Missing={missing_adapter}; unexpected={unexpected_adapter}."
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected adapter keys: {unexpected}")
    if not any("base_model" in name for name in missing):
        raise RuntimeError("Adapter load did not report frozen base-model keys as missing.")


@torch.inference_mode()
def score_base_nll(
    model: TargetConditionedLoRA,
    prompt_ids: Sequence[Sequence[int]],
    generated: np.ndarray,
    *,
    pad_token_id: int,
    device: torch.device,
) -> np.ndarray:
    lengths = [len(prompt) + generated.shape[1] for prompt in prompt_ids]
    width = max(lengths)
    input_ids = torch.full((len(prompt_ids), width), int(pad_token_id), dtype=torch.long, device=device)
    attention = torch.zeros((len(prompt_ids), width), dtype=torch.long, device=device)
    starts: list[int] = []
    for row, prompt in enumerate(prompt_ids):
        sequence = list(prompt) + generated[row].tolist()
        input_ids[row, : len(sequence)] = torch.tensor(sequence, device=device)
        attention[row, : len(sequence)] = 1
        starts.append(len(prompt) - 1)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention,
        use_adapters=False,
        use_cache=False,
        return_dict=True,
    )
    log_probs = F.log_softmax(outputs.logits.float(), dim=-1)
    nll = np.zeros((len(prompt_ids), generated.shape[1]), dtype=np.float32)
    for row, start in enumerate(starts):
        positions = torch.arange(start, start + generated.shape[1], device=device)
        tokens = torch.from_numpy(generated[row].astype(np.int64)).to(device)
        nll[row] = (-log_probs[row, positions, tokens]).cpu().numpy()
    return nll


@torch.inference_mode()
def generate_mode(
    model: TargetConditionedLoRA,
    tokenizer,
    prompts: Sequence[str],
    registry: TargetRegistry,
    key_indices: np.ndarray,
    target_codes: np.ndarray,
    *,
    mode: str,
    rule_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if rule_name not in RULE_IDS:
        raise ValueError(f"Unknown evaluation rule: {rule_name}")
    rule_id = RULE_IDS[rule_name]
    if (
        rule_id != RULE_ECC
        and mode in {"lora_only", "lora_residual"}
        and args.condition_summary_mode != "topk_contrast"
    ):
        raise ValueError(
            "Non-ECC adapter evaluation requires --condition-summary-mode topk_contrast."
        )
    model.eval()
    tokenizer.padding_side = "left"
    all_rows: list[dict[str, object]] = []
    # Reuse the same random-number stream for paired operating-point comparisons.
    generator = torch.Generator(device=device).manual_seed(stable_int(args.seed, "eval"))
    for start in range(0, len(prompts), args.eval_batch_size):
        stop = min(len(prompts), start + args.eval_batch_size)
        batch_prompts = list(prompts[start:stop])
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_tokens,
        )
        input_ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        prompt_token_lists = [
            encode_prompt(tokenizer, prompt, max_length=args.max_prompt_tokens)
            for prompt in batch_prompts
        ]
        batch_keys = torch.from_numpy(key_indices[start:stop].astype(np.int64))
        generated = np.empty((stop - start, args.generated_tokens), dtype=np.int32)
        residual_values = np.zeros((stop - start, args.generated_tokens), dtype=np.float32)
        token_adherence_values = np.full(
            (stop - start, args.generated_tokens), np.nan, dtype=np.float32
        )
        past = None
        base_past = None
        current_ids = input_ids
        current_attention = attention
        for position in range(args.generated_tokens):
            codes = torch.from_numpy(target_codes[start:stop, position].astype(np.int64))
            batch_rules = torch.full_like(batch_keys, int(rule_id))
            use_adapter = mode in {"lora_only", "lora_residual"}
            needs_base_pass = (
                rule_id == RULE_ENTROPY_KGW
                or (use_adapter and args.condition_summary_mode == "topk_contrast")
            )
            base_outputs = None
            if needs_base_pass:
                base_outputs = model(
                    input_ids=current_ids,
                    attention_mask=current_attention,
                    use_adapters=False,
                    past_key_values=base_past,
                    use_cache=True,
                    return_dict=True,
                )
                base_past = base_outputs.past_key_values
            active_rows = torch.ones(stop - start, dtype=torch.bool)
            if rule_id == RULE_ENTROPY_KGW:
                active_rows = (
                    next_token_entropy(base_outputs.logits[:, -1, :])
                    >= float(args.entropy_gate_threshold)
                ).cpu()
            position_seeds = np.full(stop - start, -1, dtype=np.int64)
            if rule_id in DYNAMIC_RULE_IDS:
                for local_row in range(stop - start):
                    if not bool(active_rows[local_row]):
                        continue
                    key_index = int(batch_keys[local_row])
                    position_seeds[local_row] = kgw_row_seed(
                        generated[local_row, :position].tolist(),
                        key=registry.keys[key_index],
                        seed_count=args.kgw_seed_count,
                        seed_offset=args.kgw_seed_offset,
                    )
            target_mask = torch.from_numpy(
                np.repeat(registry.kgw_eligible_mask[None, :], stop - start, axis=0)
            ).to(device)
            active_device = active_rows.to(device)
            if bool(active_rows.any()):
                target_mask[active_device] = registry.mixed_masks(
                    batch_keys[active_rows],
                    codes[active_rows],
                    batch_rules[active_rows],
                    torch.from_numpy(position_seeds)[active_rows],
                    device=device,
                )
            if use_adapter and args.condition_summary_mode == "topk_contrast":
                target_summary = torch.zeros(
                    (stop - start, model.base_model.config.hidden_size),
                    dtype=torch.float32,
                    device=device,
                )
                if bool(active_rows.any()):
                    target_summary[active_device] = topk_contrast_summaries(
                        base_outputs.logits[active_device, -1, :].float(),
                        target_mask[active_device],
                        model.base_model.get_input_embeddings().weight,
                        top_k=args.condition_top_k,
                    )
                target_feature_tensor = None
                if getattr(args, "condition_feature_mode", "none") == "geometry":
                    target_features = torch.zeros(
                        (stop - start, 3), dtype=torch.float32, device=device
                    )
                    if bool(active_rows.any()):
                        target_features[active_device] = target_geometry_features(
                            base_outputs.logits[active_device, -1, :].float(),
                            target_mask[active_device],
                        )
                    target_feature_tensor = torch.zeros(
                        (*current_ids.shape, 3),
                        dtype=torch.float32,
                        device=device,
                    )
                    target_feature_tensor[:, -1, :] = target_features
                summaries = torch.zeros(
                    (*current_ids.shape, target_summary.shape[-1]),
                    dtype=torch.float32,
                    device=device,
                )
                summaries[:, -1, :] = target_summary
                outputs = model(
                    input_ids=current_ids,
                    attention_mask=current_attention,
                    target_summaries=summaries,
                    target_features=target_feature_tensor,
                    use_adapters=True,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                if use_adapter:
                    descriptor_ids = batch_keys * 4 + codes
                    target_summary = registry.summaries.index_select(
                        0, descriptor_ids
                    ).to(device)
                    summaries = torch.zeros(
                        (*current_ids.shape, target_summary.shape[-1]),
                        dtype=torch.float32,
                        device=device,
                    )
                    summaries[:, -1, :] = target_summary
                else:
                    summaries = None
                if base_outputs is not None and not use_adapter:
                    outputs = base_outputs
                else:
                    outputs = model(
                        input_ids=current_ids,
                        attention_mask=current_attention,
                        target_summaries=summaries,
                        use_adapters=use_adapter,
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
            logits = outputs.logits[:, -1, :].float()
            if rule_id == RULE_ENTROPY_KGW and use_adapter and bool((~active_rows).any()):
                logits[~active_device] = base_outputs.logits[~active_device, -1, :].float()
            if mode == "fixed_delta20":
                controlled_logits = logits + (
                    20.0
                    * target_mask.to(logits.dtype)
                    * active_device.unsqueeze(1)
                )
                residual_values[active_rows.numpy(), position] = 20.0
            elif mode == "lora_only":
                controlled_logits = logits
            elif mode in {"closed_form", "lora_residual"}:
                residual = apply_closed_form_residual(
                    logits, target_mask, target_adherence=args.target_adherence
                )
                controlled_logits = torch.where(
                    active_device.unsqueeze(1), residual.controlled_logits, logits
                )
                residual_values[active_rows.numpy(), position] = (
                    residual.residual_bias[active_device].cpu().numpy()
                )
            elif mode == "unwatermarked":
                controlled_logits = logits
            else:
                raise ValueError(f"Unknown generation mode: {mode}")
            if len(tokenizer) < controlled_logits.shape[-1]:
                controlled_logits[:, len(tokenizer) :] = -torch.inf
            next_tokens = sample_logits(
                controlled_logits,
                args.temperature,
                generator,
                top_k=args.top_k,
            )
            generated[:, position] = next_tokens.cpu().numpy().astype(np.int32)
            adherence = (
                target_mask.gather(1, next_tokens.unsqueeze(1))
                .squeeze(1)
                .cpu()
                .numpy()
            )
            token_adherence_values[active_rows.numpy(), position] = adherence[
                active_rows.numpy()
            ]
            past = outputs.past_key_values
            current_ids = next_tokens.unsqueeze(1)
            current_attention = torch.cat(
                [current_attention, torch.ones((stop - start, 1), device=device, dtype=current_attention.dtype)],
                dim=1,
            )
        base_nll = score_base_nll(
            model,
            prompt_token_lists,
            generated,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
        )
        for row in range(stop - start):
            key_index = int(key_indices[start + row])
            active_token_mask = np.isfinite(token_adherence_values[row])
            token_ok = token_adherence_values[row, active_token_mask].astype(bool).tolist()
            active_rate = float(np.mean(active_token_mask))
            if rule_name == "ecc":
                block_ok = [
                    bool(np.all(token_adherence_values[row, index : index + 8] == 1.0))
                    for index in range(0, args.generated_tokens, 8)
                ]
                exact_block_rate: float | None = float(np.mean(block_ok))
                kgw_z_score: float | None = None
            elif rule_name == "binary":
                block_ok = [
                    bool(np.all(token_adherence_values[row, index : index + 2] == 1.0))
                    for index in range(0, args.generated_tokens, 2)
                ]
                exact_block_rate = float(np.mean(block_ok))
                kgw_z_score = None
            elif rule_name == "cw_abcd":
                block_ok = [
                    bool(np.all(token_adherence_values[row, index : index + 8] == 1.0))
                    for index in range(0, args.generated_tokens, 8)
                ]
                exact_block_rate = float(np.mean(block_ok))
                kgw_z_score = None
            elif rule_name == "unigram":
                exact_block_rate = None
                kgw_z_score = None
            else:
                exact_block_rate = None
                green_count = int(np.sum(token_ok))
                token_count = len(token_ok)
                gamma = 0.25 if rule_name == "atomic_kgw25" else 0.5
                kgw_z_score = float(
                    (green_count - gamma * token_count)
                    / math.sqrt(gamma * (1.0 - gamma) * token_count)
                ) if token_count else None
            active_residuals = residual_values[row, active_token_mask]
            all_rows.append(
                {
                    "mode": mode,
                    "target_rule": rule_name,
                    "sequence_index": start + row,
                    "source": "held_out_mixed",
                    "key_index": key_index,
                    "generated_text": tokenizer.decode(generated[row].tolist(), skip_special_tokens=True),
                    "token_adherence": float(np.mean(token_ok)) if token_ok else None,
                    "active_token_rate": active_rate,
                    "active_tokens": int(active_token_mask.sum()),
                    "exact_block_rate": exact_block_rate,
                    "kgw_z_score": kgw_z_score,
                    "mean_base_nll": float(base_nll[row].mean()),
                    "base_ppl": float(math.exp(min(20.0, float(base_nll[row].mean())))),
                    "mean_residual_bias": (
                        float(active_residuals.mean()) if active_residuals.size else 0.0
                    ),
                }
            )
        print(f"[eval:{rule_name}:{mode}] {stop}/{len(prompts)}", flush=True)
    exact_rates = [
        float(row["exact_block_rate"])
        for row in all_rows
        if row["exact_block_rate"] is not None
    ]
    kgw_scores = [
        float(row["kgw_z_score"])
        for row in all_rows
        if row["kgw_z_score"] is not None
    ]
    summary = {
        "mode": mode,
        "target_rule": rule_name,
        "sequences": len(all_rows),
        "token_adherence": float(
            np.mean(
                [row["token_adherence"] for row in all_rows if row["token_adherence"] is not None]
            )
        ),
        "active_token_rate": float(np.mean([row["active_token_rate"] for row in all_rows])),
        "exact_block_rate": float(np.mean(exact_rates)) if exact_rates else None,
        "mean_kgw_z_score": float(np.mean(kgw_scores)) if kgw_scores else None,
        "mean_base_nll": float(np.mean([row["mean_base_nll"] for row in all_rows])),
        "base_ppl_from_corpus_nll": float(
            math.exp(min(20.0, np.mean([row["mean_base_nll"] for row in all_rows])))
        ),
        "mean_residual_bias": float(np.mean([row["mean_residual_bias"] for row in all_rows])),
    }
    return all_rows, summary


def main() -> None:
    args = parse_args()
    train_rules = parse_rule_names(args.train_rules, option="--train-rules")
    eval_rules = parse_rule_names(args.eval_rules, option="--eval-rules")
    composition_cardinalities = parse_cardinalities(args.composition_cardinalities)
    if (
        args.training_protocol == "schedule_pool"
        and args.kgw_train_fraction > 0
        and train_rules != ["ecc"]
    ):
        raise ValueError(
            "Legacy --kgw-train-fraction cannot be combined with explicit mixed --train-rules."
        )
    if (
        args.training_protocol == "composition_pool"
        or args.kgw_train_fraction > 0
        or any(rule != "ecc" for rule in train_rules + eval_rules)
    ) and args.condition_summary_mode != "topk_contrast":
        raise ValueError(
            "Mixed/static-dynamic rule experiments require --condition-summary-mode topk_contrast."
        )
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    root = Path(__file__).resolve().parent
    manifest_path = (root / args.manifest).resolve()
    partition_path = (root / args.partition_dir).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    splits = load_manifest(manifest_path)
    requested = {"train": args.train_count, "validation": args.validation_count, "test": args.test_count}
    for split, count in requested.items():
        if len(splits[split]) < count:
            raise ValueError(f"Manifest contains {len(splits[split])} {split} rows; need {count}.")
        splits[split] = splits[split][:count]
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        torch_dtype=dtype,
    ).to(device)
    model = TargetConditionedLoRA(
        base_model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        condition_feature_dim=(
            3 if args.condition_feature_mode == "geometry" else 0
        ),
    ).to(device)
    partition = Partition.load(partition_path)
    model_vocab_size = int(base_model.get_output_embeddings().weight.shape[0])
    if partition.vocab_size != model_vocab_size:
        raise ValueError(
            f"Partition vocab {partition.vocab_size} != model output vocab {model_vocab_size}."
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
    # OPT pads its embedding/output matrix beyond the tokenizer vocabulary.
    # Those rows must never become payload or boundary candidates.
    if len(tokenizer) < partition.vocab_size:
        partition.leaf_ids[len(tokenizer) :] = -1
        padded_ids = np.arange(len(tokenizer), partition.vocab_size, dtype=np.int64)
        partition.banned_ids = np.unique(
            np.concatenate([partition.banned_ids, padded_ids])
        ).astype(np.int64)
    train_keys = [1000 + index for index in range(args.train_key_count)]
    eval_keys = [900000 + index for index in range(args.eval_key_count)]
    all_keys = train_keys + eval_keys
    registry = TargetRegistry(partition, all_keys, base_model.get_input_embeddings().weight)
    uses_ecc = "ecc" in eval_rules or (
        args.training_protocol != "composition_pool" and "ecc" in train_rules
    )
    if uses_ecc and not partition.boundary_ids.size:
        raise ValueError("ECC requires a nonempty boundary-token set.")

    base_train_prompts = [str(row["prompt"]) for row in splits["train"]]
    base_validation_prompts = [str(row["prompt"]) for row in splits["validation"]]
    train_preset_mask_indices = None
    validation_preset_mask_indices = None
    if args.training_protocol == "composition_pool":
        if args.kgw_train_fraction > 0:
            raise ValueError(
                "--kgw-train-fraction is incompatible with --training-protocol composition_pool."
            )
        (
            train_prompts,
            train_key_indices,
            train_target_codes,
            train_rule_ids,
            train_preset_mask_indices,
        ) = make_composition_curriculum(
            base_train_prompts,
            repeats=args.composition_repeats,
            generated_tokens=args.generated_tokens,
            cardinalities=composition_cardinalities,
            num_atoms=partition.num_leaves,
        )
        (
            validation_prompts,
            validation_key_indices,
            validation_target_codes,
            validation_rule_ids,
            validation_preset_mask_indices,
        ) = make_composition_curriculum(
            base_validation_prompts,
            repeats=args.composition_repeats,
            generated_tokens=args.generated_tokens,
            cardinalities=composition_cardinalities,
            num_atoms=partition.num_leaves,
        )
        coverage = {
            "train": composition_coverage_report(
                train_rule_ids,
                train_preset_mask_indices,
                num_atoms=partition.num_leaves,
            ),
            "validation": composition_coverage_report(
                validation_rule_ids,
                validation_preset_mask_indices,
                num_atoms=partition.num_leaves,
            ),
        }
        (output_dir / "composition_coverage.json").write_text(
            json.dumps(coverage, indent=2), encoding="utf-8"
        )
    elif args.kgw_train_fraction > 0:
        train_prompts = base_train_prompts
        validation_prompts = base_validation_prompts
        train_key_indices, train_target_codes = make_fixed_ecc_schedule(
            len(train_prompts),
            args.generated_tokens,
            seed=stable_int(args.seed, "legacy-train"),
            key_count=args.train_key_count,
        )
        validation_key_indices, validation_target_codes = make_fixed_ecc_schedule(
            len(validation_prompts),
            args.generated_tokens,
            seed=stable_int(args.seed, "legacy-validation"),
            key_count=args.train_key_count,
        )
        train_rule_ids = make_rule_assignments(
            len(train_prompts), args.kgw_train_fraction, seed=args.seed
        )
        validation_rule_ids = make_rule_assignments(
            len(validation_prompts),
            args.kgw_train_fraction,
            seed=stable_int(args.seed, "legacy-validation-rules"),
        )
    else:
        (
            train_prompts,
            train_key_indices,
            train_target_codes,
            train_rule_ids,
        ) = expand_rule_targets(
            base_train_prompts,
            train_rules,
            generated_tokens=args.generated_tokens,
            seed=stable_int(args.seed, "train"),
            key_count=args.train_key_count,
        )
        (
            validation_prompts,
            validation_key_indices,
            validation_target_codes,
            validation_rule_ids,
        ) = expand_rule_targets(
            base_validation_prompts,
            train_rules,
            generated_tokens=args.generated_tokens,
            seed=stable_int(args.seed, "validation"),
            key_count=args.train_key_count,
        )

    trajectory_path = (
        (root / args.trajectory_cache).resolve()
        if args.trajectory_cache
        else output_dir / "trajectory_cache.npz"
    )
    history: list[dict[str, object]] = []
    cache_metadata: dict[str, object] | None = None
    if not args.eval_only:
        prepare_prompts = train_prompts + validation_prompts
        key_indices = np.concatenate(
            [train_key_indices, validation_key_indices], axis=0
        )
        target_codes = np.concatenate(
            [train_target_codes, validation_target_codes], axis=0
        )
        rule_ids = np.concatenate(
            [train_rule_ids, validation_rule_ids], axis=0
        )
        preset_mask_indices = (
            np.concatenate(
                [train_preset_mask_indices, validation_preset_mask_indices], axis=0
            )
            if train_preset_mask_indices is not None
            and validation_preset_mask_indices is not None
            else None
        )
        cache_metadata = trajectory_cache_metadata(
            prompts=prepare_prompts,
            key_indices=key_indices,
            target_codes=target_codes,
            rule_ids=rule_ids,
            preset_mask_indices=preset_mask_indices,
            manifest_path=manifest_path,
            partition_path=partition_path,
            tokenizer=tokenizer,
            args=args,
        )
    if not args.eval_only and not args.skip_prepare:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        generated_ids, kgw_seeds = prepare_trajectories(
            model,
            tokenizer,
            prepare_prompts,
            registry,
            key_indices,
            target_codes,
            rule_ids,
            preset_mask_indices,
            max_prompt_tokens=args.max_prompt_tokens,
            batch_size=args.trajectory_batch_size,
            temperature=args.temperature,
            kgw_seed_count=args.kgw_seed_count,
            kgw_seed_offset=args.kgw_seed_offset,
            entropy_gate_threshold=args.entropy_gate_threshold,
            seed=args.seed,
            device=device,
        )
        np.savez_compressed(
            trajectory_path,
            generated_ids=generated_ids,
            key_indices=key_indices,
            target_codes=target_codes,
            rule_ids=rule_ids,
            kgw_seeds=kgw_seeds,
            metadata_json=np.asarray(canonical_json(cache_metadata)),
        )
    if args.prepare_only:
        return
    if args.eval_only:
        if not args.adapter_checkpoint:
            raise ValueError("--eval-only requires --adapter-checkpoint.")
        checkpoint_path = Path(args.adapter_checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = (root / checkpoint_path).resolve()
        load_adapter(model, checkpoint_path)
    else:
        if not trajectory_path.exists():
            raise FileNotFoundError(f"Missing trajectory cache: {trajectory_path}")
        cache = np.load(trajectory_path, allow_pickle=False)
        if cache_metadata is None:
            raise RuntimeError("Internal error: missing expected cache metadata.")
        validate_trajectory_cache_metadata(cache, cache_metadata)
        if args.kgw_train_fraction > 0 and "rule_ids" not in cache.files:
            raise ValueError(
                "The selected trajectory cache predates dynamic targets. "
                "Rebuild it without --skip-prepare."
            )
        cached_rule_ids = (
            cache["rule_ids"]
            if "rule_ids" in cache.files
            else np.full(cache["key_indices"].shape, RULE_ECC, dtype=np.uint8)
        )
        cached_kgw_seeds = (
            cache["kgw_seeds"]
            if "kgw_seeds" in cache.files
            else np.full(cache["target_codes"].shape, -1, dtype=np.int64)
        )
        train_stop = len(train_prompts)
        validation_stop = train_stop + len(validation_prompts)
        if int(cache["generated_ids"].shape[0]) != validation_stop:
            raise ValueError(
                "Trajectory cache row count does not match the expanded training rules: "
                f"{cache['generated_ids'].shape[0]} != {validation_stop}."
            )
        train_dataset = TrajectoryDataset(
            train_prompts,
            cache["generated_ids"][:train_stop],
            cache["key_indices"][:train_stop],
            cache["target_codes"][:train_stop],
            cached_rule_ids[:train_stop],
            cached_kgw_seeds[:train_stop],
            tokenizer,
            args.max_prompt_tokens,
        )
        validation_dataset = TrajectoryDataset(
            validation_prompts,
            cache["generated_ids"][train_stop:validation_stop],
            cache["key_indices"][train_stop:validation_stop],
            cache["target_codes"][train_stop:validation_stop],
            cached_rule_ids[train_stop:validation_stop],
            cached_kgw_seeds[train_stop:validation_stop],
            tokenizer,
            args.max_prompt_tokens,
        )
        collate = lambda rows: collate_trajectories(rows, tokenizer.pad_token_id)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=0,
            generator=torch.Generator().manual_seed(args.seed),
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        history = train_adapter(model, registry, train_loader, validation_loader, args, output_dir, device)
        load_adapter(model, output_dir / "adapter_best.pt")

    test_prompts = [str(row["prompt"]) for row in splits["test"]]
    details: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for rule_name in eval_rules:
        eval_local_keys, eval_codes = make_rule_schedule(
            rule_name,
            len(test_prompts),
            args.generated_tokens,
            seed=stable_int(args.seed, "held-out-eval", rule_name),
            key_count=args.eval_key_count,
        )
        eval_registry_keys = eval_local_keys + args.train_key_count
        for mode in (
            "unwatermarked",
            "fixed_delta20",
            "closed_form",
            "lora_only",
            "lora_residual",
        ):
            mode_rows, mode_summary = generate_mode(
                model,
                tokenizer,
                test_prompts,
                registry,
                eval_registry_keys,
                eval_codes,
                mode=mode,
                rule_name=rule_name,
                args=args,
                device=device,
            )
            details.extend(mode_rows)
            summaries.append(mode_summary)
    write_csv(output_dir / "evaluation_details.csv", details)
    write_csv(output_dir / "evaluation_summary.csv", summaries)
    runtime = time.time() - started
    config = {
        **vars(args),
        "resolved_train_rules": (
            [] if args.training_protocol == "composition_pool" else train_rules
        ),
        "training_protocol": args.training_protocol,
        "composition_cardinalities": composition_cardinalities,
        "composition_repeats": args.composition_repeats,
        "resolved_eval_rules": eval_rules,
        "effective_train_sequences": len(train_prompts),
        "effective_validation_sequences": len(validation_prompts),
        "held_out_sequences_per_rule": len(test_prompts),
        "manifest_sha256": sha256_file(manifest_path),
        "tokenization_contract": "add_special_tokens=True for generation, training, and PPL",
        "tokenizer_identity": tokenizer_identity(tokenizer),
        "partition_sha256": files_sha256(
            partition_path,
            [
                "leaf_ids.npy",
                "semantic_group_ids.npy",
                "boundary_ids.npy",
                "banned_ids.npy",
                "hierarchy_config.json",
                "watermark_compositions.json",
            ],
        ),
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.trainable_parameters())),
        "lora_layers": len(model.lora_layers),
        "runtime_seconds": runtime,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "RESULTS.json").write_text(
        json.dumps({"training_history": history, "evaluation": summaries}, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
