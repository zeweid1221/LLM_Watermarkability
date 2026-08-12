from __future__ import annotations

import argparse
import gc
import json
import math
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from modeling import (
    TargetConditionedLoRA,
    apply_closed_form_residual,
    topk_contrast_summaries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Qwen3 preparation, LoRA training, and evaluation."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--batch-sizes", default="2,4,8")
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--prompt-length", type=int, default=160)
    parser.add_argument("--generated-tokens", type=int, default=96)
    parser.add_argument("--condition-top-k", type=int, default=128)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--target-adherence", type=float, default=0.99)
    parser.add_argument("--timed-training-steps", type=int, default=2)
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def gib(value: int) -> float:
    return float(value) / float(2**30)


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def make_inputs(
    batch_size: int,
    sequence_length: int,
    usable_vocab: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(
        0, usable_vocab, (batch_size, sequence_length), device=device
    )
    attention = torch.ones_like(input_ids)
    active = torch.zeros(
        (batch_size, sequence_length), dtype=torch.bool, device=device
    )
    return input_ids, attention, active


def objective_step(
    model: TargetConditionedLoRA,
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    prompt_length: int,
    generated_tokens: int,
    usable_vocab: int,
    vocab_size: int,
    allowed: torch.Tensor,
    condition_top_k: int,
    target_adherence: float,
    device: torch.device,
    backward: bool,
) -> float:
    sequence_length = prompt_length + generated_tokens - 1
    input_ids, attention, active = make_inputs(
        batch_size, sequence_length, usable_vocab, device
    )
    active[:, prompt_length - 1 :] = True
    masks = allowed.unsqueeze(0).expand(int(active.sum()), -1)

    with torch.no_grad():
        base_outputs = model(
            input_ids=input_ids,
            attention_mask=attention,
            use_adapters=False,
            use_cache=False,
            return_dict=True,
        )
        base_logits = base_outputs.logits[active].float()
        summaries_active = topk_contrast_summaries(
            base_logits,
            masks,
            model.base_model.get_input_embeddings().weight,
            top_k=condition_top_k,
        )

    summaries = torch.zeros(
        (batch_size, sequence_length, summaries_active.shape[-1]),
        dtype=torch.float32,
        device=device,
    )
    summaries[active] = summaries_active
    adapted_outputs = model(
        input_ids=input_ids,
        attention_mask=attention,
        target_summaries=summaries,
        use_adapters=True,
        use_cache=False,
        return_dict=True,
    )
    adapted_logits = adapted_outputs.logits[active].float()
    residual = apply_closed_form_residual(
        adapted_logits, masks, target_adherence=target_adherence
    )
    labels = residual.controlled_logits.detach().argmax(dim=-1)
    nll = F.cross_entropy(residual.controlled_logits, labels)
    adapted_log_probs = F.log_softmax(adapted_logits, dim=-1)
    adapted_probs = adapted_log_probs.exp()
    base_log_probs = F.log_softmax(base_logits, dim=-1)
    conditional_kl = (
        adapted_probs * (adapted_log_probs - base_log_probs)
    ).sum(dim=-1).mean()
    loss = nll + 0.02 * residual.residual_bias.square().mean() + 0.02 * conditional_kl

    if backward:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        optimizer.step()
    return float(loss.detach())


def benchmark_objective(
    model: TargetConditionedLoRA,
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int,
    args: argparse.Namespace,
    usable_vocab: int,
    vocab_size: int,
    allowed: torch.Tensor,
    device: torch.device,
) -> dict[str, float | int | str]:
    model.train()
    model.base_model.eval()
    objective_step(
        model,
        optimizer,
        batch_size=batch_size,
        prompt_length=args.prompt_length,
        generated_tokens=args.generated_tokens,
        usable_vocab=usable_vocab,
        vocab_size=vocab_size,
        allowed=allowed,
        condition_top_k=args.condition_top_k,
        target_adherence=args.target_adherence,
        device=device,
        backward=True,
    )
    synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    losses = []
    for _ in range(args.timed_training_steps):
        losses.append(
            objective_step(
                model,
                optimizer,
                batch_size=batch_size,
                prompt_length=args.prompt_length,
                generated_tokens=args.generated_tokens,
                usable_vocab=usable_vocab,
                vocab_size=vocab_size,
                allowed=allowed,
                condition_top_k=args.condition_top_k,
                target_adherence=args.target_adherence,
                device=device,
                backward=True,
            )
        )
    synchronize(device)
    train_seconds = (time.perf_counter() - started) / args.timed_training_steps
    peak_reserved = gib(torch.cuda.max_memory_reserved(device))

    model.eval()
    synchronize(device)
    started = time.perf_counter()
    validation_loss = objective_step(
        model,
        optimizer,
        batch_size=batch_size,
        prompt_length=args.prompt_length,
        generated_tokens=args.generated_tokens,
        usable_vocab=usable_vocab,
        vocab_size=vocab_size,
        allowed=allowed,
        condition_top_k=args.condition_top_k,
        target_adherence=args.target_adherence,
        device=device,
        backward=False,
    )
    synchronize(device)
    validation_seconds = time.perf_counter() - started
    return {
        "status": "passed",
        "batch_size": batch_size,
        "training_seconds_per_batch": train_seconds,
        "training_seconds_per_sequence": train_seconds / batch_size,
        "validation_seconds_per_batch": validation_seconds,
        "validation_loss": validation_loss,
        "mean_training_loss": sum(losses) / len(losses),
        "peak_reserved_gib": peak_reserved,
    }


@torch.inference_mode()
def benchmark_generation(
    model: TargetConditionedLoRA,
    *,
    batch_size: int,
    prompt_length: int,
    generated_tokens: int,
    usable_vocab: int,
    vocab_size: int,
    allowed: torch.Tensor,
    condition_top_k: int,
    target_adherence: float,
    device: torch.device,
    use_adapter: bool,
) -> tuple[float, torch.Tensor, float]:
    model.eval()
    prompt_ids = torch.randint(
        0, usable_vocab, (batch_size, prompt_length), device=device
    )
    attention = torch.ones_like(prompt_ids)
    masks = allowed.unsqueeze(0).expand(batch_size, vocab_size)
    generated = []
    current_ids = prompt_ids
    current_attention = attention
    past = None
    base_past = None
    torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = time.perf_counter()
    for _ in range(generated_tokens):
        if use_adapter:
            base_outputs = model(
                input_ids=current_ids,
                attention_mask=current_attention,
                use_adapters=False,
                past_key_values=base_past,
                use_cache=True,
                return_dict=True,
            )
            base_past = base_outputs.past_key_values
            target_summary = topk_contrast_summaries(
                base_outputs.logits[:, -1, :].float(),
                masks,
                model.base_model.get_input_embeddings().weight,
                top_k=condition_top_k,
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
            controlled = apply_closed_form_residual(
                outputs.logits[:, -1, :].float(),
                masks,
                target_adherence=target_adherence,
            ).controlled_logits
        else:
            outputs = model(
                input_ids=current_ids,
                attention_mask=current_attention,
                use_adapters=False,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            controlled = outputs.logits[:, -1, :].float().masked_fill(
                ~masks, -torch.inf
            )
        next_tokens = controlled.argmax(dim=-1)
        generated.append(next_tokens)
        past = outputs.past_key_values
        current_ids = next_tokens.unsqueeze(1)
        current_attention = torch.cat(
            [
                current_attention,
                torch.ones(
                    (batch_size, 1),
                    dtype=current_attention.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
    synchronize(device)
    seconds = time.perf_counter() - started
    peak_reserved = gib(torch.cuda.max_memory_reserved(device))
    return seconds, torch.stack(generated, dim=1), peak_reserved


@torch.inference_mode()
def benchmark_scoring(
    model: TargetConditionedLoRA,
    prompt_ids: torch.Tensor,
    generated: torch.Tensor,
    device: torch.device,
) -> float:
    sequence = torch.cat([prompt_ids, generated], dim=1)
    attention = torch.ones_like(sequence)
    synchronize(device)
    started = time.perf_counter()
    model(
        input_ids=sequence,
        attention_mask=attention,
        use_adapters=False,
        use_cache=False,
        return_dict=True,
    )
    synchronize(device)
    return time.perf_counter() - started


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda")
    torch.manual_seed(20260811)
    torch.cuda.manual_seed_all(20260811)

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
        dtype=torch.float16,
    ).to(device)
    model = TargetConditionedLoRA(
        base_model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=0.05,
    ).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=2e-4)
    synchronize(device)
    load_seconds = time.perf_counter() - load_started

    vocab_size = int(base_model.get_output_embeddings().weight.shape[0])
    usable_vocab = min(len(tokenizer), vocab_size)
    allowed = torch.arange(vocab_size, device=device).remainder(2).eq(0)
    batch_results = []
    for batch_size in [int(value) for value in args.batch_sizes.split(",")]:
        try:
            batch_results.append(
                benchmark_objective(
                    model,
                    optimizer,
                    batch_size=batch_size,
                    args=args,
                    usable_vocab=usable_vocab,
                    vocab_size=vocab_size,
                    allowed=allowed,
                    device=device,
                )
            )
        except torch.cuda.OutOfMemoryError as error:
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            batch_results.append(
                {"status": "oom", "batch_size": batch_size, "error": str(error)}
            )

    passed = [row for row in batch_results if row["status"] == "passed"]
    if not passed:
        raise RuntimeError("Every requested training batch size ran out of memory.")
    best = max(passed, key=lambda row: int(row["batch_size"]))

    generation_batch = args.generation_batch_size
    base_seconds, base_generated, base_peak = benchmark_generation(
        model,
        batch_size=generation_batch,
        prompt_length=args.prompt_length,
        generated_tokens=args.generated_tokens,
        usable_vocab=usable_vocab,
        vocab_size=vocab_size,
        allowed=allowed,
        condition_top_k=args.condition_top_k,
        target_adherence=args.target_adherence,
        device=device,
        use_adapter=False,
    )
    gc.collect()
    torch.cuda.empty_cache()
    lora_seconds, lora_generated, lora_peak = benchmark_generation(
        model,
        batch_size=generation_batch,
        prompt_length=args.prompt_length,
        generated_tokens=args.generated_tokens,
        usable_vocab=usable_vocab,
        vocab_size=vocab_size,
        allowed=allowed,
        condition_top_k=args.condition_top_k,
        target_adherence=args.target_adherence,
        device=device,
        use_adapter=True,
    )
    prompt_for_score = torch.randint(
        0,
        usable_vocab,
        (generation_batch, args.prompt_length),
        device=device,
    )
    score_seconds = benchmark_scoring(
        model, prompt_for_score, lora_generated, device
    )

    train_batch = int(best["batch_size"])
    train_batches = math.ceil(35000 / train_batch) * 6
    validation_batches = math.ceil(3500 / train_batch) * 6
    preparation_batches = math.ceil(38500 / generation_batch)
    evaluation_batches_per_mode = math.ceil(256 / generation_batch)
    preparation_total = preparation_batches * base_seconds
    training_total = train_batches * float(best["training_seconds_per_batch"])
    validation_total = validation_batches * float(best["validation_seconds_per_batch"])
    evaluation_total = 7 * evaluation_batches_per_mode * (
        3 * base_seconds + 2 * lora_seconds + 5 * score_seconds
    )
    compute_total = (
        preparation_total + training_total + validation_total + evaluation_total
    )

    report = {
        "status": "passed",
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_gib": gib(torch.cuda.get_device_properties(device).total_memory),
        "model_load_seconds": load_seconds,
        "model_name": args.model_name,
        "vocab_size": vocab_size,
        "lora_layers": len(model.lora_layers),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.trainable_parameters()
        ),
        "training_batch_benchmarks": batch_results,
        "generation_benchmark": {
            "batch_size": generation_batch,
            "tokens_per_sequence": args.generated_tokens,
            "base_seconds_per_batch": base_seconds,
            "lora_residual_seconds_per_batch": lora_seconds,
            "base_peak_reserved_gib": base_peak,
            "lora_peak_reserved_gib": lora_peak,
            "base_scoring_seconds_per_batch": score_seconds,
        },
        "full_protocol": {
            "training_sequences": 35000,
            "validation_sequences": 3500,
            "epochs": 6,
            "prepared_trajectories": 38500,
            "evaluation_rules": 7,
            "evaluation_modes": 5,
            "held_out_per_rule": 256,
        },
        "estimate": {
            "selected_training_batch_size": train_batch,
            "preparation_hours": preparation_total / 3600.0,
            "training_hours": training_total / 3600.0,
            "validation_hours": validation_total / 3600.0,
            "evaluation_hours": evaluation_total / 3600.0,
            "compute_total_hours": compute_total / 3600.0,
            "total_with_10pct_overhead_hours": compute_total * 1.10 / 3600.0,
            "atomic_partition_build_excluded": True,
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
