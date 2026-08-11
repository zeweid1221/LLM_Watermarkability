from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Frozen linear layer with a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scale = float(alpha) / float(max(1, rank))
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.enabled = True
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        if self.enabled:
            residual = self.lora_b(self.dropout(self.lora_a(inputs.float())))
            output = output + (self.scale * residual).to(output.dtype)
        return output


def inject_lora(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_suffixes: Sequence[str] = ("q_proj", "v_proj"),
) -> list[LoRALinear]:
    installed: list[LoRALinear] = []
    for child_name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child_name.endswith(tuple(target_suffixes)):
            replacement = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
            setattr(module, child_name, replacement)
            installed.append(replacement)
        else:
            installed.extend(
                inject_lora(
                    child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    target_suffixes=target_suffixes,
                )
            )
    return installed


class ConditionProjector(nn.Module):
    """Inject an exact target-set summary into the current prediction state."""

    def __init__(self, hidden_size: int, rank: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, summaries: torch.Tensor) -> torch.Tensor:
        return self.up(torch.nn.functional.silu(self.down(self.norm(summaries.float()))))


class TargetConditionedLoRA(nn.Module):
    """A frozen causal LM with conditional input adaptation and internal LoRA."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        target_suffixes: Sequence[str] = ("q_proj", "v_proj"),
    ):
        super().__init__()
        self.base_model = base_model
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        hidden_size = int(base_model.get_input_embeddings().embedding_dim)
        self.conditioner = ConditionProjector(hidden_size, rank)
        self.lora_layers = inject_lora(
            self.base_model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_suffixes=target_suffixes,
        )
        if not self.lora_layers:
            raise RuntimeError(f"No LoRA target modules matched {list(target_suffixes)}.")

    @contextmanager
    def adapters_enabled(self, enabled: bool) -> Iterator[None]:
        previous = [layer.enabled for layer in self.lora_layers]
        for layer in self.lora_layers:
            layer.enabled = bool(enabled)
        try:
            yield
        finally:
            for layer, value in zip(self.lora_layers, previous):
                layer.enabled = value

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        target_summaries: torch.Tensor | None = None,
        use_adapters: bool = True,
        **kwargs,
    ):
        embeddings = self.base_model.get_input_embeddings()(input_ids)
        if use_adapters and target_summaries is not None:
            embeddings = embeddings + self.conditioner(target_summaries).to(embeddings.dtype)
        with self.adapters_enabled(use_adapters):
            return self.base_model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                **kwargs,
            )

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if "lora_" in name or name.startswith("conditioner.")
        }


@dataclass
class ResidualStats:
    native_mass: torch.Tensor
    residual_bias: torch.Tensor
    controlled_logits: torch.Tensor


def topk_contrast_summaries(
    base_logits: torch.Tensor,
    target_mask: torch.Tensor,
    embedding_weight: torch.Tensor,
    *,
    top_k: int,
) -> torch.Tensor:
    """Compress the relevant portion of an exact target set into hidden space.

    The summary contrasts allowed and disallowed tokens among the base model's
    most plausible candidates. Unlike a fixed descriptor lookup, it can encode
    a new context-dependent green list at every generation step.
    """
    if base_logits.ndim != 2 or target_mask.ndim != 2:
        raise ValueError("Expected [B,V] logits and target masks.")
    if base_logits.shape != target_mask.shape:
        raise ValueError("base_logits and target_mask shapes must match.")
    k = min(max(2, int(top_k)), int(base_logits.shape[-1]))
    top_values, top_ids = torch.topk(base_logits.float(), k=k, dim=-1)
    weights = torch.softmax(top_values, dim=-1)
    membership = target_mask.gather(1, top_ids)
    signs = membership.to(weights.dtype).mul(2.0).sub(1.0)
    token_embeddings = embedding_weight.index_select(0, top_ids.reshape(-1)).reshape(
        *top_ids.shape, -1
    )
    token_embeddings = torch.nn.functional.normalize(
        token_embeddings.float(), dim=-1
    )
    summaries = (weights.mul(signs).unsqueeze(-1) * token_embeddings).sum(dim=1)
    return torch.nn.functional.normalize(summaries, dim=-1)


def apply_closed_form_residual(
    logits: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    target_adherence: float,
    max_bias: float = 50.0,
) -> ResidualStats:
    probabilities = torch.softmax(logits.float(), dim=-1)
    mask = target_mask.to(probabilities.dtype)
    native_mass = (probabilities * mask).sum(dim=-1).clamp(1e-7, 1.0 - 1e-7)
    target = torch.as_tensor(
        float(target_adherence), device=logits.device, dtype=torch.float32
    ).clamp(1e-7, 1.0 - 1e-7)
    residual = (torch.logit(target) - torch.logit(native_mass)).clamp(0.0, float(max_bias))
    controlled = logits.float() + residual.unsqueeze(-1) * mask
    return ResidualStats(native_mass, residual, controlled)
