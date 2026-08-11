from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .partition import HierarchicalPartition, stable_seed


@dataclass
class PartitionOptimizationConfig:
    num_leaves: int = 8
    frontiers: Tuple[int, ...] = (2, 4, 8)
    semantic_lsh_bits: int = 6
    mass_strata_bins: int = 128
    steps: int = 120
    learning_rate: float = 0.08
    temperature_start: float = 1.5
    temperature_end: float = 0.15
    mass_weight: float = 1.0
    count_weight: float = 0.25
    semantic_weight: float = 0.10
    entropy_weight: float = 0.02
    context_batch_size: int = 0
    rounding_context_count: int = 0
    seed: int = 20260810

    def validate(self) -> None:
        if self.num_leaves <= 0 or self.num_leaves & (self.num_leaves - 1):
            raise ValueError("num_leaves must be a positive power of two.")
        for frontier in self.frontiers:
            if frontier <= 0 or self.num_leaves % frontier != 0:
                raise ValueError("Every frontier must divide num_leaves.")
        if self.steps <= 0:
            raise ValueError("steps must be positive.")
        if self.mass_strata_bins <= 0:
            raise ValueError("mass_strata_bins must be positive.")
        if self.context_batch_size < 0:
            raise ValueError("context_batch_size must be non-negative.")
        if self.rounding_context_count < 0:
            raise ValueError("rounding_context_count must be non-negative.")


def semantic_lsh_groups(
    embedding_matrix: np.ndarray,
    usable_ids: np.ndarray,
    bits: int,
    seed: int,
) -> np.ndarray:
    if bits <= 0:
        return np.zeros(usable_ids.shape[0], dtype=np.int32)
    emb = np.asarray(embedding_matrix[usable_ids], dtype=np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    rng = np.random.default_rng(stable_seed("semantic-lsh", seed))
    projection = rng.standard_normal((emb.shape[1], bits), dtype=np.float32)
    signs = (emb @ projection >= 0).astype(np.int32)
    powers = (1 << np.arange(bits, dtype=np.int32)).reshape(1, -1)
    return (signs * powers).sum(axis=1).astype(np.int32)


def _balanced_initial_leaves(group_ids: np.ndarray, num_leaves: int, seed: int) -> np.ndarray:
    assignments = np.empty(group_ids.shape[0], dtype=np.int16)
    for group_id in np.unique(group_ids).tolist():
        indices = np.flatnonzero(group_ids == group_id)
        rng = np.random.default_rng(stable_seed("partition-init", seed, int(group_id)))
        rng.shuffle(indices)
        offset = int(rng.integers(0, num_leaves))
        assignments[indices] = (np.arange(indices.size, dtype=np.int64) + offset) % num_leaves
    return assignments


def _balanced_round(scores: np.ndarray) -> np.ndarray:
    token_count, num_leaves = scores.shape
    base = token_count // num_leaves
    capacities = np.full(num_leaves, base, dtype=np.int64)
    capacities[: token_count % num_leaves] += 1
    sorted_leaves = np.argsort(-scores, axis=1)
    top = scores[np.arange(token_count), sorted_leaves[:, 0]]
    second = scores[np.arange(token_count), sorted_leaves[:, 1]] if num_leaves > 1 else top
    order = np.argsort(-(top - second))
    assignments = np.full(token_count, -1, dtype=np.int16)
    for token_index in order.tolist():
        for leaf in sorted_leaves[token_index].tolist():
            if capacities[leaf] > 0:
                assignments[token_index] = int(leaf)
                capacities[leaf] -= 1
                break
    if np.any(assignments < 0) or capacities.sum() != 0:
        raise RuntimeError("Balanced rounding failed to assign every token.")
    return assignments


def _mass_aware_balanced_round(
    scores: np.ndarray,
    probabilities: np.ndarray,
    group_ids: np.ndarray,
    frontiers: Sequence[int],
) -> np.ndarray:
    """Round soft leaves while preserving probability mass across hierarchy levels."""
    token_count, num_leaves = scores.shape
    base = token_count // num_leaves
    capacities = np.full(num_leaves, base, dtype=np.int64)
    capacities[: token_count % num_leaves] += 1
    assignments = np.full(token_count, -1, dtype=np.int16)
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    loads = {
        int(frontier): np.zeros((int(frontier), probs.shape[0]), dtype=np.float64)
        for frontier in frontiers
    }
    group_count = int(group_ids.max()) + 1 if group_ids.size else 0
    group_sizes = np.bincount(group_ids, minlength=group_count).astype(np.float64)
    semantic_counts = np.zeros((group_count, num_leaves), dtype=np.int32)
    # A token with concentrated probability in any context can dominate a hard bucket.
    importance = np.square(probs).sum(axis=0)
    order = np.argsort(-importance, kind="stable")

    for token_index in order.tolist():
        token_probs = probs[:, token_index]
        group = int(group_ids[token_index])
        best_leaf = -1
        best_cost = float("inf")
        for leaf in np.flatnonzero(capacities > 0).tolist():
            cost = 0.0
            for frontier in frontiers:
                frontier = int(frontier)
                bucket = int(leaf) // (num_leaves // frontier)
                before = loads[frontier][bucket] - 1.0 / frontier
                cost += float(np.mean(2.0 * before * token_probs + np.square(token_probs)))
            if group_sizes[group] > 0:
                before_semantic = semantic_counts[group, leaf] / group_sizes[group] - 1.0 / num_leaves
                after_semantic = (semantic_counts[group, leaf] + 1) / group_sizes[group] - 1.0 / num_leaves
                cost += 0.01 * float(after_semantic**2 - before_semantic**2)
            cost -= 1e-4 * float(scores[token_index, leaf])
            if cost < best_cost:
                best_cost = cost
                best_leaf = int(leaf)
        if best_leaf < 0:
            raise RuntimeError("Mass-aware rounding exhausted all leaf capacities.")
        assignments[token_index] = best_leaf
        capacities[best_leaf] -= 1
        semantic_counts[group, best_leaf] += 1
        for frontier in frontiers:
            frontier = int(frontier)
            bucket = best_leaf // (num_leaves // frontier)
            loads[frontier][bucket] += token_probs

    if np.any(assignments < 0) or capacities.sum() != 0:
        raise RuntimeError("Mass-aware rounding failed to assign every token.")
    return assignments


def evaluate_leaf_masses(
    probabilities: np.ndarray,
    leaf_ids: np.ndarray,
    frontiers: Sequence[int],
    num_leaves: int,
    *,
    batch_size: int = 0,
) -> Dict[str, float]:
    usable_ids = np.flatnonzero(leaf_ids >= 0)
    context_count = int(probabilities.shape[0])
    effective_batch_size = context_count if batch_size <= 0 else min(batch_size, context_count)
    errors: Dict[int, List[np.ndarray]] = {int(frontier): [] for frontier in frontiers}
    minimum_mass = {int(frontier): float("inf") for frontier in frontiers}
    bucket_ids_by_frontier = {
        int(frontier): leaf_ids[usable_ids] // (num_leaves // int(frontier))
        for frontier in frontiers
    }
    for start in range(0, context_count, effective_batch_size):
        stop = min(context_count, start + effective_batch_size)
        probs = np.asarray(probabilities[start:stop, usable_ids], dtype=np.float64)
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
        for frontier in frontiers:
            frontier = int(frontier)
            bucket_ids = bucket_ids_by_frontier[frontier]
            masses = np.stack(
                [probs[:, bucket_ids == bucket].sum(axis=1) for bucket in range(frontier)],
                axis=1,
            )
            errors[frontier].append(np.abs(masses - 1.0 / frontier))
            minimum_mass[frontier] = min(minimum_mass[frontier], float(masses.min()))
    metrics: Dict[str, float] = {}
    for frontier in frontiers:
        frontier = int(frontier)
        abs_error = np.concatenate(errors[frontier], axis=0)
        metrics[f"k{frontier}_mean_abs_mass_error"] = float(abs_error.mean())
        metrics[f"k{frontier}_p95_abs_mass_error"] = float(np.quantile(abs_error, 0.95))
        metrics[f"k{frontier}_min_mass"] = minimum_mass[frontier]
    return metrics


class HierarchicalPartitionOptimizer:
    def __init__(self, config: PartitionOptimizationConfig):
        config.validate()
        self.config = config

    def optimize(
        self,
        probabilities: np.ndarray,
        embedding_matrix: np.ndarray,
        *,
        boundary_ids: Sequence[int] = (),
        banned_ids: Sequence[int] = (),
        device: str = "cpu",
        metadata: Dict[str, object] | None = None,
    ) -> Tuple[HierarchicalPartition, List[Dict[str, float]]]:
        probabilities = np.asarray(probabilities, dtype=np.float32)
        if probabilities.ndim != 2:
            raise ValueError("probabilities must have shape [contexts, vocab].")
        vocab_size = probabilities.shape[1]
        if embedding_matrix.shape[0] != vocab_size:
            raise ValueError("embedding_matrix and probabilities disagree on vocab size.")
        excluded = set(int(x) for x in boundary_ids) | set(int(x) for x in banned_ids)
        usable_ids = np.asarray([x for x in range(vocab_size) if x not in excluded], dtype=np.int64)
        semantic_group_ids = semantic_lsh_groups(
            embedding_matrix,
            usable_ids,
            self.config.semantic_lsh_bits,
            self.config.seed,
        )
        mean_native_mass = probabilities.mean(axis=0)[usable_ids]
        mass_order = np.argsort(mean_native_mass, kind="stable")
        mass_bins = np.empty(usable_ids.size, dtype=np.int32)
        mass_bins[mass_order] = np.minimum(
            self.config.mass_strata_bins - 1,
            np.arange(usable_ids.size, dtype=np.int64)
            * self.config.mass_strata_bins
            // max(1, usable_ids.size),
        )
        group_ids = (
            semantic_group_ids.astype(np.int64) * self.config.mass_strata_bins
            + mass_bins.astype(np.int64)
        ).astype(np.int32)
        unique_groups, compact_groups = np.unique(group_ids, return_inverse=True)
        del unique_groups
        initial = _balanced_initial_leaves(compact_groups, self.config.num_leaves, self.config.seed)

        torch_device = torch.device(device)
        group_tensor = torch.as_tensor(compact_groups, dtype=torch.long, device=torch_device)
        group_count = int(compact_groups.max()) + 1 if compact_groups.size else 0
        group_sizes = torch.bincount(group_tensor, minlength=group_count).float().clamp_min(1.0)

        logits = torch.full(
            (usable_ids.size, self.config.num_leaves),
            -1.0,
            dtype=torch.float32,
            device=torch_device,
        )
        logits[
            torch.arange(usable_ids.size, device=torch_device),
            torch.as_tensor(initial, dtype=torch.long, device=torch_device),
        ] = 1.0
        logits = torch.nn.Parameter(logits)
        optimizer = torch.optim.Adam([logits], lr=self.config.learning_rate)
        history: List[Dict[str, float]] = []
        context_count = int(probabilities.shape[0])
        context_batch_size = self.config.context_batch_size or context_count
        context_batch_size = min(context_batch_size, context_count)
        context_rng = np.random.default_rng(stable_seed("partition-contexts", self.config.seed))
        context_order = context_rng.permutation(context_count)
        context_cursor = 0

        def next_context_indices() -> np.ndarray:
            nonlocal context_cursor, context_order
            if context_batch_size >= context_count:
                return np.arange(context_count, dtype=np.int64)
            selected: List[np.ndarray] = []
            remaining = context_batch_size
            while remaining > 0:
                available = context_count - context_cursor
                take = min(remaining, available)
                selected.append(context_order[context_cursor : context_cursor + take])
                context_cursor += take
                remaining -= take
                if context_cursor == context_count:
                    context_order = context_rng.permutation(context_count)
                    context_cursor = 0
            return np.concatenate(selected).astype(np.int64, copy=False)

        for step in range(self.config.steps):
            context_indices = next_context_indices()
            batch_probabilities = probabilities[context_indices][:, usable_ids]
            probs_tensor = torch.as_tensor(batch_probabilities, device=torch_device)
            probs_tensor = probs_tensor / probs_tensor.sum(dim=1, keepdim=True).clamp_min(1e-12)
            fraction = step / max(1, self.config.steps - 1)
            temperature = self.config.temperature_start * (
                self.config.temperature_end / self.config.temperature_start
            ) ** fraction
            assignment = torch.softmax(logits / temperature, dim=-1)
            leaf_masses = probs_tensor @ assignment
            mass_loss = torch.zeros((), device=torch_device)
            for frontier in self.config.frontiers:
                group_size = self.config.num_leaves // frontier
                frontier_masses = leaf_masses.reshape(leaf_masses.shape[0], frontier, group_size).sum(dim=-1)
                mass_loss = mass_loss + (frontier_masses - 1.0 / frontier).square().mean()

            count_loss = (assignment.mean(dim=0) - 1.0 / self.config.num_leaves).square().mean()
            semantic_sums = torch.zeros(
                (group_count, self.config.num_leaves),
                dtype=assignment.dtype,
                device=torch_device,
            )
            semantic_sums.index_add_(0, group_tensor, assignment)
            semantic_distribution = semantic_sums / group_sizes.unsqueeze(1)
            semantic_loss = (
                semantic_distribution - 1.0 / self.config.num_leaves
            ).square().mean()
            entropy_loss = -(assignment.clamp_min(1e-9) * assignment.clamp_min(1e-9).log()).sum(dim=-1).mean()
            loss = (
                self.config.mass_weight * mass_loss
                + self.config.count_weight * count_loss
                + self.config.semantic_weight * semantic_loss
                + self.config.entropy_weight * entropy_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step == 0 or step == self.config.steps - 1 or (step + 1) % max(1, self.config.steps // 10) == 0:
                history.append(
                    {
                        "step": float(step + 1),
                        "temperature": float(temperature),
                        "loss": float(loss.detach().cpu()),
                        "mass_loss": float(mass_loss.detach().cpu()),
                        "count_loss": float(count_loss.detach().cpu()),
                        "semantic_loss": float(semantic_loss.detach().cpu()),
                        "entropy": float(entropy_loss.detach().cpu()),
                    }
                )

        rounding_context_count = self.config.rounding_context_count or context_count
        rounding_context_count = min(rounding_context_count, context_count)
        if rounding_context_count < context_count:
            rounding_rng = np.random.default_rng(stable_seed("partition-rounding", self.config.seed))
            rounding_indices = np.sort(
                rounding_rng.choice(context_count, size=rounding_context_count, replace=False)
            )
        else:
            rounding_indices = np.arange(context_count, dtype=np.int64)
        rounding_probabilities = probabilities[rounding_indices][:, usable_ids]
        rounded = _mass_aware_balanced_round(
            logits.detach().cpu().numpy(),
            rounding_probabilities,
            compact_groups,
            self.config.frontiers,
        )
        leaf_ids = np.full(vocab_size, -1, dtype=np.int16)
        leaf_ids[usable_ids] = rounded
        full_group_ids = np.full(vocab_size, -1, dtype=np.int32)
        full_group_ids[usable_ids] = compact_groups.astype(np.int32)
        partition = HierarchicalPartition(
            leaf_ids=leaf_ids,
            num_leaves=self.config.num_leaves,
            boundary_ids=boundary_ids,
            banned_ids=banned_ids,
            semantic_group_ids=full_group_ids,
            metadata={
                **(metadata or {}),
                "optimizer": asdict(self.config),
                "num_contexts": int(probabilities.shape[0]),
                "rounding_contexts": int(rounding_context_count),
                "num_usable_tokens": int(usable_ids.size),
            },
        )
        return partition, history
