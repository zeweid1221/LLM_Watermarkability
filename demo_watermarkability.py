from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from modeling import (
    TargetConditionedLoRA,
    apply_closed_form_residual,
    topk_contrast_summaries,
)
from reproducibility import encode_prompt, validate_tokenizer_identity
from targets import Partition, RULE_IDS, TargetRegistry, make_rule_schedule, stable_seed


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "opt125m_composition_pool" / "adapter_best.pt"
DEFAULT_METADATA = ROOT / "checkpoints" / "opt125m_composition_pool" / "metadata.json"
DEFAULT_PARTITION = ROOT / "assets" / "atomic16" / "opt-125m-lfqa5000-v1"
SUPPORTED_RULES = ("binary", "unigram", "cw_abcd", "ecc")


@dataclass
class DemoResult:
    mode: str
    text: str
    mean_native_mass: float
    mean_controlled_mass: float
    mean_residual_bias: float
    token_adherence: float
    base_ppl: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the base model, the analytic watermark controller, and the "
            "released target-conditioned adapter on one prompt."
        )
    )
    parser.add_argument(
        "--prompt",
        default="Photosynthesis is the process by which plants",
    )
    parser.add_argument("--rule", choices=SUPPORTED_RULES, default="binary")
    parser.add_argument(
        "--tau",
        type=float,
        default=0.99,
        help="Requested target-set probability mass (paper setting: 0.99).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-prompt-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use an already cached Hugging Face model without network access.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--check-artifacts",
        action="store_true",
        help="Verify the released checkpoint and partition without loading the model.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--partition-dir", type=Path, default=DEFAULT_PARTITION)
    args = parser.parse_args(argv)
    validate_options(args)
    return args


def validate_options(args: argparse.Namespace) -> None:
    if not 0.0 < float(args.tau) < 1.0:
        raise ValueError("--tau must be strictly between 0 and 1.")
    if int(args.max_new_tokens) <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if int(args.max_prompt_tokens) <= 0:
        raise ValueError("--max-prompt-tokens must be positive.")
    if float(args.temperature) <= 0.0:
        raise ValueError("--temperature must be positive.")
    if int(args.top_k) < 0:
        raise ValueError("--top-k cannot be negative.")
    if args.rule == "ecc" and int(args.max_new_tokens) % 8:
        raise ValueError("The ECC schedule requires --max-new-tokens divisible by 8.")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_verify_metadata(
    metadata_path: Path,
    checkpoint_path: Path,
    partition_dir: Path,
) -> dict[str, object]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_checkpoint = str(metadata["checkpoint_sha256"]).lower()
    actual_checkpoint = sha256_file(checkpoint_path)
    if actual_checkpoint != expected_checkpoint:
        raise RuntimeError(
            f"Checkpoint hash mismatch: expected {expected_checkpoint}, got {actual_checkpoint}."
        )
    for name, expected in dict(metadata["partition_sha256"]).items():
        path = partition_dir / name
        actual = sha256_file(path)
        if actual != str(expected).lower():
            raise RuntimeError(
                f"Partition hash mismatch for {name}: expected {expected}, got {actual}."
            )
    return metadata


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_adapter(model: TargetConditionedLoRA, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected = set(model.adapter_state_dict())
    supplied = set(state)
    if expected != supplied:
        raise RuntimeError(
            "Adapter checkpoint keys do not match the constructed model. "
            f"Missing={sorted(expected - supplied)}; unexpected={sorted(supplied - expected)}."
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or not any("base_model" in name for name in missing):
        raise RuntimeError("The adapter checkpoint did not load as an adapter-only state dict.")


def sample_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    generator: torch.Generator,
) -> torch.Tensor:
    scores = logits.float() / float(temperature)
    if 0 < int(top_k) < scores.shape[-1]:
        cutoff = torch.topk(scores, k=int(top_k), dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < cutoff, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.multinomial(
        probabilities, num_samples=1, generator=generator
    ).squeeze(-1)


@torch.inference_mode()
def base_perplexity(
    model: TargetConditionedLoRA,
    prompt_ids: list[int],
    generated_ids: list[int],
    device: torch.device,
) -> float:
    sequence = torch.tensor([prompt_ids + generated_ids], dtype=torch.long, device=device)
    outputs = model(
        input_ids=sequence,
        attention_mask=torch.ones_like(sequence),
        use_adapters=False,
        use_cache=False,
        return_dict=True,
    )
    start = len(prompt_ids) - 1
    positions = torch.arange(start, start + len(generated_ids), device=device)
    tokens = torch.tensor(generated_ids, dtype=torch.long, device=device)
    nll = -F.log_softmax(outputs.logits[0, positions].float(), dim=-1)[
        torch.arange(len(generated_ids), device=device), tokens
    ]
    return math.exp(min(20.0, float(nll.mean())))


@torch.inference_mode()
def generate_mode(
    *,
    mode: str,
    model: TargetConditionedLoRA,
    tokenizer,
    registry: TargetRegistry,
    prompt: str,
    rule: str,
    target_codes: np.ndarray,
    tau: float,
    max_prompt_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
    condition_top_k: int,
    device: torch.device,
) -> DemoResult:
    prompt_ids = encode_prompt(tokenizer, prompt, max_length=max_prompt_tokens)
    current_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(current_ids)
    generated: list[int] = []
    native_masses: list[float] = []
    controlled_masses: list[float] = []
    residual_biases: list[float] = []
    adherences: list[float] = []
    adapted_past = None
    base_past = None
    generator = torch.Generator(device=device).manual_seed(
        stable_seed("watermarkability-demo", seed)
    )
    rule_id = RULE_IDS[rule]

    for position, code in enumerate(target_codes.tolist()):
        code_tensor = torch.tensor([int(code)], dtype=torch.long)
        target_mask = registry.mixed_masks(
            torch.tensor([0], dtype=torch.long),
            code_tensor,
            torch.tensor([rule_id], dtype=torch.long),
            torch.tensor([-1], dtype=torch.long),
            device=device,
        )

        if mode == "adapter_controller":
            base_outputs = model(
                input_ids=current_ids,
                attention_mask=attention,
                use_adapters=False,
                past_key_values=base_past,
                use_cache=True,
                return_dict=True,
            )
            base_past = base_outputs.past_key_values
            summary = topk_contrast_summaries(
                base_outputs.logits[:, -1, :].float(),
                target_mask,
                model.base_model.get_input_embeddings().weight,
                top_k=condition_top_k,
            )
            summaries = torch.zeros(
                (*current_ids.shape, summary.shape[-1]),
                dtype=torch.float32,
                device=device,
            )
            summaries[:, -1, :] = summary
            outputs = model(
                input_ids=current_ids,
                attention_mask=attention,
                target_summaries=summaries,
                use_adapters=True,
                past_key_values=adapted_past,
                use_cache=True,
                return_dict=True,
            )
            adapted_past = outputs.past_key_values
        else:
            outputs = model(
                input_ids=current_ids,
                attention_mask=attention,
                use_adapters=False,
                past_key_values=base_past,
                use_cache=True,
                return_dict=True,
            )
            base_past = outputs.past_key_values

        logits = outputs.logits[:, -1, :].float()
        native_mass = (
            torch.softmax(logits, dim=-1) * target_mask.to(logits.dtype)
        ).sum(dim=-1)
        native_masses.append(float(native_mass.item()))

        if mode == "base":
            controlled_logits = logits
            residual_biases.append(0.0)
        else:
            residual = apply_closed_form_residual(
                logits, target_mask, target_adherence=tau
            )
            controlled_logits = residual.controlled_logits
            residual_biases.append(float(residual.residual_bias.item()))

        if len(tokenizer) < controlled_logits.shape[-1]:
            controlled_logits[:, len(tokenizer) :] = -torch.inf
        controlled_mass = (
            torch.softmax(controlled_logits, dim=-1)
            * target_mask.to(controlled_logits.dtype)
        ).sum(dim=-1)
        controlled_masses.append(float(controlled_mass.item()))
        next_token = sample_token(
            controlled_logits,
            temperature=temperature,
            top_k=top_k,
            generator=generator,
        )
        adherences.append(float(target_mask[0, int(next_token.item())].item()))
        generated.append(int(next_token.item()))
        current_ids = next_token.unsqueeze(1)
        attention = torch.cat(
            [
                attention,
                torch.ones((1, 1), dtype=attention.dtype, device=device),
            ],
            dim=1,
        )

    labels = {
        "base": "Base model",
        "controller": "Controller only",
        "adapter_controller": "Adapter + controller",
    }
    return DemoResult(
        mode=labels[mode],
        text=tokenizer.decode(generated, skip_special_tokens=True),
        mean_native_mass=float(np.mean(native_masses)),
        mean_controlled_mass=float(np.mean(controlled_masses)),
        mean_residual_bias=float(np.mean(residual_biases)),
        token_adherence=float(np.mean(adherences)),
        base_ppl=base_perplexity(model, prompt_ids, generated, device),
    )


def print_results(args: argparse.Namespace, results: list[DemoResult]) -> None:
    print(f"\nPrompt: {args.prompt}")
    print(f"Rule: {args.rule} | tau: {args.tau:.4f} | tokens: {args.max_new_tokens}")
    print()
    header = (
        f"{'Mode':<22} {'Native mass':>12} {'Final mass':>11} "
        f"{'Bias':>9} {'Adherence':>11} {'Base PPL':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.mode:<22} {result.mean_native_mass:>12.4f} "
            f"{result.mean_controlled_mass:>11.4f} "
            f"{result.mean_residual_bias:>9.4f} "
            f"{result.token_adherence:>11.4f} {result.base_ppl:>10.3f}"
        )
    for result in results:
        print(f"\n[{result.mode}]\n{result.text.strip()}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    metadata = load_and_verify_metadata(
        args.metadata.resolve(), args.checkpoint.resolve(), args.partition_dir.resolve()
    )
    if args.check_artifacts:
        print("Released checkpoint and partition hashes are valid.")
        return

    device = choose_device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model_name = str(metadata["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=args.local_files_only,
        torch_dtype=dtype,
    ).to(device)
    model = TargetConditionedLoRA(
        base_model,
        rank=int(metadata["lora_rank"]),
        alpha=float(metadata["lora_alpha"]),
        dropout=float(metadata["lora_dropout"]),
    ).to(device)
    load_adapter(model, args.checkpoint.resolve())
    model.eval()

    partition = Partition.load(args.partition_dir.resolve())
    if partition.vocab_size != int(base_model.get_output_embeddings().weight.shape[0]):
        raise RuntimeError("The released partition does not match the model vocabulary.")
    validate_tokenizer_identity(
        partition.metadata, tokenizer, context=f"Partition {args.partition_dir}"
    )
    if len(tokenizer) < partition.vocab_size:
        partition.leaf_ids[len(tokenizer) :] = -1
        padded_ids = np.arange(len(tokenizer), partition.vocab_size, dtype=np.int64)
        partition.banned_ids = np.unique(
            np.concatenate([partition.banned_ids, padded_ids])
        ).astype(np.int64)
    registry = TargetRegistry(
        partition, [900000], base_model.get_input_embeddings().weight
    )
    _, schedules = make_rule_schedule(
        args.rule,
        1,
        args.max_new_tokens,
        seed=stable_seed("watermarkability-demo-schedule", args.seed, args.rule),
        key_count=1,
    )

    results = [
        generate_mode(
            mode=mode,
            model=model,
            tokenizer=tokenizer,
            registry=registry,
            prompt=args.prompt,
            rule=args.rule,
            target_codes=schedules[0],
            tau=args.tau,
            max_prompt_tokens=args.max_prompt_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
            condition_top_k=int(metadata["condition_top_k"]),
            device=device,
        )
        for mode in ("base", "controller", "adapter_controller")
    ]
    print_results(args, results)
    if args.output_json:
        payload = {
            "prompt": args.prompt,
            "rule": args.rule,
            "tau": args.tau,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "results": [result.__dict__ for result in results],
        }
        args.output_json.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
