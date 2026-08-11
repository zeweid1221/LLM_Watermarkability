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
    topk_contrast_summaries,
)
from targets import (
    RULE_ECC,
    RULE_KGW,
    RULE_ATOMIC_KGW,
    RULE_ATOMIC_KGW25,
    RULE_ENTROPY_KGW,
    RULE_UNIGRAM,
    DYNAMIC_RULE_IDS,
    RULE_IDS,
    RULE_NAMES,
    Partition,
    TargetRegistry,
    kgw_row_seed,
    make_fixed_ecc_schedule,
    make_rule_schedule,
    make_rule_assignments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the isolated OPT-125M conditional-LoRA diagnostic.")
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
    parser.add_argument("--eval-rules", default="ecc")
    parser.add_argument(
        "--condition-summary-mode",
        choices=("registry_mean", "topk_contrast"),
        default="registry_mean",
    )
    parser.add_argument("--condition-top-k", type=int, default=128)
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
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            split = str(row["split"])
            if split in splits:
                splits[split].append(row)
    return splits


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
    kgw_seeds = np.full((len(prompts), generated_tokens), -1, dtype=np.int64)
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
            position_seeds = np.full(stop - start, -1, dtype=np.int64)
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
            tokenizer.encode(prompt, add_special_tokens=False, truncation=True, max_length=max_prompt_tokens)
            for prompt in prompts
        ]
        self.generated_ids = generated_ids
        self.key_indices = key_indices
        self.target_codes = target_codes
        self.rule_ids = rule_ids
        self.kgw_seeds = kgw_seeds
        self.eos_token_id = int(tokenizer.eos_token_id)

    def __len__(self) -> int:
        return len(self.prompt_ids)

    def __getitem__(self, index: int):
        prompt = self.prompt_ids[index] or [self.eos_token_id]
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
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    return active, summaries, masks


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
    active, summaries, masks = build_condition_tensor(
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
        device=device,
    )
    adapted_outputs = model(
        input_ids=input_ids,
        attention_mask=attention,
        target_summaries=summaries,
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
    nll = F.cross_entropy(residual.controlled_logits, active_labels)
    adapted_log_probs = F.log_softmax(adapted_logits, dim=-1)
    adapted_probs = adapted_log_probs.exp()
    base_log_probs = F.log_softmax(base_logits, dim=-1)
    conditional_kl = (adapted_probs * (adapted_log_probs - base_log_probs)).sum(dim=-1).mean()
    residual_loss = residual.residual_bias.square().mean()
    target_mass_loss = -residual.native_mass.log().mean()
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
    return {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}


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
        row.update({f"train_{name}": float(np.mean([item[name] for item in running])) for name in running[0]})
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
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [name for name in unexpected if "base_model" not in name]
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
            tokenizer.encode(prompt, add_special_tokens=True, truncation=True, max_length=args.max_prompt_tokens)
            or [tokenizer.eos_token_id]
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
    if args.kgw_train_fraction > 0 and train_rules != ["ecc"]:
        raise ValueError(
            "Legacy --kgw-train-fraction cannot be combined with explicit mixed --train-rules."
        )
    if (
        args.kgw_train_fraction > 0
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
    ).to(device)
    partition = Partition.load(partition_path)
    model_vocab_size = int(base_model.get_output_embeddings().weight.shape[0])
    if partition.vocab_size != model_vocab_size:
        raise ValueError(
            f"Partition vocab {partition.vocab_size} != model output vocab {model_vocab_size}."
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

    base_train_prompts = [str(row["prompt"]) for row in splits["train"]]
    base_validation_prompts = [str(row["prompt"]) for row in splits["validation"]]
    if args.kgw_train_fraction > 0:
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
    if not args.eval_only and not args.skip_prepare:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
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
        generated_ids, kgw_seeds = prepare_trajectories(
            model,
            tokenizer,
            prepare_prompts,
            registry,
            key_indices,
            target_codes,
            rule_ids,
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
        "resolved_train_rules": train_rules,
        "resolved_eval_rules": eval_rules,
        "effective_train_sequences": len(train_prompts),
        "effective_validation_sequences": len(validation_prompts),
        "held_out_sequences_per_rule": len(test_prompts),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
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
