from __future__ import annotations

import argparse
import json
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
        description="Exercise one Qwen3 target-conditioned LoRA training step."
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--sequence-length", type=int, default=255)
    parser.add_argument("--target-positions", type=int, default=96)
    parser.add_argument("--condition-top-k", type=int, default=128)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=32.0)
    parser.add_argument("--target-adherence", type=float, default=0.99)
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def gib(value: int) -> float:
    return float(value) / float(2**30)


def main() -> None:
    args = parse_args()
    if args.target_positions <= 0 or args.target_positions > args.sequence_length:
        raise ValueError("target-positions must be within the sequence length.")
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a CUDA GPU.")

    device = torch.device("cuda")
    torch.manual_seed(20260811)
    torch.cuda.manual_seed_all(20260811)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, local_files_only=args.local_files_only
    )
    started = time.perf_counter()
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

    vocab_size = int(base_model.get_output_embeddings().weight.shape[0])
    usable_vocab = min(int(len(tokenizer)), vocab_size)
    input_ids = torch.randint(
        low=0,
        high=usable_vocab,
        size=(1, args.sequence_length),
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    active = torch.zeros(
        (1, args.sequence_length), dtype=torch.bool, device=device
    )
    active[:, -args.target_positions :] = True

    # A balanced synthetic target set exercises the same exact-mask memory path
    # as the real watermark rules without requiring a prebuilt Qwen partition.
    allowed = torch.arange(vocab_size, device=device).remainder(2).eq(0)
    target_masks = allowed.unsqueeze(0).expand(args.target_positions, -1)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    model.base_model.eval()

    with torch.no_grad():
        base_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_adapters=False,
            use_cache=False,
            return_dict=True,
        )
        base_logits = base_outputs.logits[active].float()
        summaries_active = topk_contrast_summaries(
            base_logits,
            target_masks,
            model.base_model.get_input_embeddings().weight,
            top_k=args.condition_top_k,
        )

    summaries = torch.zeros(
        (1, args.sequence_length, summaries_active.shape[-1]),
        dtype=torch.float32,
        device=device,
    )
    summaries[active] = summaries_active
    adapted_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        target_summaries=summaries,
        use_adapters=True,
        use_cache=False,
        return_dict=True,
    )
    adapted_logits = adapted_outputs.logits[active].float()
    residual = apply_closed_form_residual(
        adapted_logits,
        target_masks,
        target_adherence=args.target_adherence,
    )

    labels = residual.controlled_logits.detach().argmax(dim=-1)
    nll = F.cross_entropy(residual.controlled_logits, labels)
    adapted_log_probs = F.log_softmax(adapted_logits, dim=-1)
    adapted_probs = adapted_log_probs.exp()
    base_log_probs = F.log_softmax(base_logits, dim=-1)
    conditional_kl = (
        adapted_probs * (adapted_log_probs - base_log_probs)
    ).sum(dim=-1).mean()
    residual_loss = residual.residual_bias.square().mean()
    loss = nll + 0.02 * residual_loss + 0.02 * conditional_kl

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize(device)

    report = {
        "status": "passed",
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_gib": gib(torch.cuda.get_device_properties(device).total_memory),
        "peak_allocated_gib": gib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_gib": gib(torch.cuda.max_memory_reserved(device)),
        "sequence_length": args.sequence_length,
        "target_positions": args.target_positions,
        "vocab_size": vocab_size,
        "lora_rank": args.lora_rank,
        "lora_layers": len(model.lora_layers),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.trainable_parameters()
        ),
        "loss": float(loss.detach()),
        "native_target_mass": float(residual.native_mass.mean().detach()),
        "mean_residual_bias": float(residual.residual_bias.mean().detach()),
        "runtime_seconds": time.perf_counter() - started,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
