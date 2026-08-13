from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


RULE_ECC = 0
RULE_KGW = 1
RULE_BINARY = 2
RULE_CW_ABCD = 3
RULE_ATOMIC_KGW = 4
RULE_UNIGRAM = 5
RULE_ATOMIC_KGW25 = 6
RULE_ENTROPY_KGW = 7
RULE_COMPOSITION_2 = 8
RULE_COMPOSITION_4 = 9
RULE_COMPOSITION_7 = 10
RULE_COMPOSITION_8 = 11
RULE_NAMES = {
    RULE_ECC: "ecc",
    RULE_KGW: "kgw",
    RULE_BINARY: "binary",
    RULE_CW_ABCD: "cw_abcd",
    RULE_ATOMIC_KGW: "atomic_kgw",
    RULE_UNIGRAM: "unigram",
    RULE_ATOMIC_KGW25: "atomic_kgw25",
    RULE_ENTROPY_KGW: "entropy_kgw",
}
RULE_IDS = {name: rule_id for rule_id, name in RULE_NAMES.items()}
DYNAMIC_RULE_IDS = {
    RULE_KGW,
    RULE_ATOMIC_KGW,
    RULE_ATOMIC_KGW25,
    RULE_ENTROPY_KGW,
}
COMPOSITION_RULE_BY_CARDINALITY = {
    2: RULE_COMPOSITION_2,
    4: RULE_COMPOSITION_4,
    7: RULE_COMPOSITION_7,
    8: RULE_COMPOSITION_8,
}
COMPOSITION_CARDINALITY_BY_RULE = {
    rule_id: cardinality
    for cardinality, rule_id in COMPOSITION_RULE_BY_CARDINALITY.items()
}


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def green_mask_for_seed(seed: int, vocab_size: int) -> np.ndarray:
    """Match the repository-native KGW green-list construction exactly."""
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    return rng.integers(
        0, 2, size=int(vocab_size), endpoint=False, dtype=np.int8
    ).astype(bool)


def atomic_green_mask_for_seed(
    seed: int,
    leaf_ids: np.ndarray,
    *,
    selected_atoms: int,
    num_atoms: int,
) -> np.ndarray:
    """Select complete atomic buckets for a prefix/key-dependent green list."""
    selected_atoms = int(selected_atoms)
    num_atoms = int(num_atoms)
    if not 0 < selected_atoms < num_atoms:
        raise ValueError("selected_atoms must be between zero and num_atoms.")
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    atoms = rng.choice(num_atoms, size=selected_atoms, replace=False)
    return np.isin(leaf_ids, atoms)


def row_seed_from_suffix(
    generated_ids: Sequence[int], seed_count: int, seed_offset: int
) -> int:
    """Match watermark_project.kgw_generator.row_seed_from_suffix."""
    end = max(0, len(generated_ids) - 1)
    start = max(0, end - int(seed_count))
    return int(sum(int(x) for x in generated_ids[start:end]) + int(seed_offset))


def kgw_row_seed(
    generated_ids: Sequence[int],
    *,
    key: int | str,
    seed_count: int,
    seed_offset: int,
) -> int:
    key_offset = stable_seed("kgw-key", key) & 0xFFFFFFFF
    return row_seed_from_suffix(
        generated_ids,
        int(seed_count),
        (int(seed_offset) + key_offset) & 0xFFFFFFFF,
    )


def make_rule_assignments(count: int, kgw_fraction: float, *, seed: int) -> np.ndarray:
    fraction = float(kgw_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("kgw_fraction must be in [0, 1].")
    assignments = np.full(int(count), RULE_ECC, dtype=np.uint8)
    dynamic_count = int(round(int(count) * fraction))
    if dynamic_count:
        rng = np.random.default_rng(stable_seed("rule-assignment", seed, count, fraction))
        dynamic_rows = rng.choice(int(count), size=dynamic_count, replace=False)
        assignments[dynamic_rows] = RULE_KGW
    return assignments


def _exhaustive_combination_indices(
    count: int,
    *,
    num_atoms: int,
    cardinality: int,
) -> np.ndarray:
    """Enumerate every union canonically before starting another full pass."""
    total = math.comb(int(num_atoms), int(cardinality))
    return np.resize(np.arange(total, dtype=np.int32), int(count))


def make_composition_curriculum(
    prompts: Sequence[str],
    *,
    repeats: int,
    generated_tokens: int,
    cardinalities: Sequence[int],
    num_atoms: int,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build watermark-agnostic trajectories over requested atom unions."""
    cardinalities = tuple(int(value) for value in cardinalities)
    unknown = sorted(set(cardinalities) - set(COMPOSITION_RULE_BY_CARDINALITY))
    if unknown:
        raise ValueError(f"Unsupported composition cardinalities: {unknown}")
    if not cardinalities:
        raise ValueError("At least one composition cardinality is required.")
    if int(repeats) <= 0:
        raise ValueError("Composition repeats must be positive.")

    expanded_prompts = [
        str(prompt)
        for _ in range(int(repeats))
        for prompt in prompts
    ]
    sequence_count = len(expanded_prompts)
    key_indices = np.zeros(sequence_count, dtype=np.uint16)
    target_codes = np.zeros((sequence_count, int(generated_tokens)), dtype=np.int8)
    rule_ids = np.empty(sequence_count, dtype=np.uint8)
    combination_indices = np.empty(
        (sequence_count, int(generated_tokens)), dtype=np.int32
    )

    prompt_count = len(prompts)
    for repeat_index in range(int(repeats)):
        for prompt_index in range(prompt_count):
            sequence_index = repeat_index * prompt_count + prompt_index
            cardinality = cardinalities[
                (prompt_index + repeat_index) % len(cardinalities)
            ]
            rule_ids[sequence_index] = COMPOSITION_RULE_BY_CARDINALITY[cardinality]

    for cardinality in cardinalities:
        rows = np.flatnonzero(rule_ids == COMPOSITION_RULE_BY_CARDINALITY[cardinality])
        values = _exhaustive_combination_indices(
            rows.size * int(generated_tokens),
            num_atoms=int(num_atoms),
            cardinality=cardinality,
        ).reshape(rows.size, int(generated_tokens))
        combination_indices[rows] = values

    return (
        expanded_prompts,
        key_indices,
        target_codes,
        rule_ids,
        combination_indices,
    )


def feasible_codewords(block_len: int = 7, vt_a: int = 6) -> list[tuple[int, ...]]:
    parity_columns = [tuple((index >> bit) & 1 for bit in range(3)) for index in range(1, 8)]
    feasible: list[tuple[int, ...]] = []
    for value in range(1 << block_len):
        bits = tuple((value >> index) & 1 for index in range(block_len))
        vt_ok = sum((index + 1) * bit for index, bit in enumerate(bits)) % (block_len + 1) == vt_a
        syndrome = [sum(bits[index] * parity_columns[index][row] for index in range(7)) % 2 for row in range(3)]
        if vt_ok and not any(syndrome):
            feasible.append(bits)
    if not feasible:
        raise RuntimeError("The VT-Hamming feasible codebook is empty.")
    return feasible


@dataclass
class Partition:
    leaf_ids: np.ndarray
    semantic_group_ids: np.ndarray
    boundary_ids: np.ndarray
    banned_ids: np.ndarray
    num_leaves: int
    ecc_payload_0_atoms: tuple[int, ...] | None = None
    ecc_payload_1_atoms: tuple[int, ...] | None = None
    ecc_boundary_atoms: tuple[int, ...] | None = None
    binary_0_atoms: tuple[int, ...] | None = None
    binary_1_atoms: tuple[int, ...] | None = None
    cw_atoms: tuple[tuple[int, ...], ...] | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.leaf_ids = np.asarray(self.leaf_ids, dtype=np.int16)
        self.semantic_group_ids = np.asarray(
            self.semantic_group_ids, dtype=np.int32
        )
        self.boundary_ids = np.asarray(self.boundary_ids, dtype=np.int64)
        self.banned_ids = np.asarray(self.banned_ids, dtype=np.int64)
        if self.leaf_ids.ndim != 1:
            raise ValueError("leaf_ids must be one-dimensional.")
        if self.semantic_group_ids.shape != self.leaf_ids.shape:
            raise ValueError("semantic_group_ids must match leaf_ids shape.")
        if self.num_leaves <= 0 or self.num_leaves & (self.num_leaves - 1):
            raise ValueError("num_leaves must be a positive power of two.")
        usable = self.leaf_ids >= 0
        if not bool(usable.any()):
            raise ValueError("Partition contains no usable vocabulary tokens.")
        if int(self.leaf_ids[usable].max()) >= int(self.num_leaves):
            raise ValueError("leaf_ids contains an out-of-range atom index.")
        for name, ids in (
            ("boundary_ids", self.boundary_ids),
            ("banned_ids", self.banned_ids),
        ):
            if ids.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")
            if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= self.vocab_size):
                raise ValueError(f"{name} contains an out-of-range token ID.")
        self._key_cache: dict[int, np.ndarray] = {}

    @classmethod
    def load(cls, path: str | Path) -> "Partition":
        root = Path(path)
        config = json.loads((root / "hierarchy_config.json").read_text(encoding="utf-8"))
        composition_path = root / "watermark_compositions.json"
        compositions = (
            json.loads(composition_path.read_text(encoding="utf-8"))
            if composition_path.exists()
            else None
        )
        ecc = compositions.get("ecc") if compositions is not None else None
        binary = compositions.get("binary") if compositions is not None else None
        cw = compositions.get("cw_abcd") if compositions is not None else None
        return cls(
            leaf_ids=np.load(root / "leaf_ids.npy", allow_pickle=False).astype(np.int16),
            semantic_group_ids=np.load(root / "semantic_group_ids.npy", allow_pickle=False).astype(np.int32),
            boundary_ids=np.load(root / "boundary_ids.npy", allow_pickle=False).astype(np.int64),
            banned_ids=np.load(root / "banned_ids.npy", allow_pickle=False).astype(np.int64),
            num_leaves=int(config["num_leaves"]),
            ecc_payload_0_atoms=(
                tuple(int(value) for value in ecc["payload_0"])
                if ecc is not None
                else None
            ),
            ecc_payload_1_atoms=(
                tuple(int(value) for value in ecc["payload_1"])
                if ecc is not None
                else None
            ),
            ecc_boundary_atoms=(
                tuple(int(value) for value in ecc["boundary"])
                if ecc is not None
                else None
            ),
            binary_0_atoms=(
                tuple(int(value) for value in binary["bucket_0"])
                if binary is not None
                else None
            ),
            binary_1_atoms=(
                tuple(int(value) for value in binary["bucket_1"])
                if binary is not None
                else None
            ),
            cw_atoms=(
                tuple(
                    tuple(int(value) for value in cw[label])
                    for label in ("A", "B", "C", "D")
                )
                if cw is not None
                else None
            ),
            metadata=dict(config.get("metadata") or {}),
        )

    @property
    def vocab_size(self) -> int:
        return int(self.leaf_ids.shape[0])

    def keyed_leaf_ids(self, key: int) -> np.ndarray:
        if self.ecc_payload_0_atoms is not None:
            return self.leaf_ids
        cached = self._key_cache.get(int(key))
        if cached is not None:
            return cached
        out = self.leaf_ids.copy()
        usable = out >= 0
        for group_id in np.unique(self.semantic_group_ids[usable]).tolist():
            indices = np.flatnonzero((self.semantic_group_ids == group_id) & usable)
            if indices.size <= 1:
                continue
            rng = np.random.default_rng(stable_seed("partition-key", key, int(group_id)))
            values = out[indices].copy()
            rng.shuffle(values)
            out[indices] = values
        self._key_cache[int(key)] = out
        return out

    def bucket_ids(self, key: int) -> np.ndarray:
        leaves = self.keyed_leaf_ids(int(key))
        if self.ecc_payload_0_atoms is not None:
            buckets = np.full(leaves.shape, -1, dtype=np.int8)
            buckets[np.isin(leaves, self.ecc_payload_0_atoms)] = 0
            buckets[np.isin(leaves, self.ecc_payload_1_atoms)] = 1
            return buckets
        group_size = self.num_leaves // 2
        buckets = np.full(leaves.shape, -1, dtype=np.int8)
        valid = leaves >= 0
        buckets[valid] = (leaves[valid] // group_size).astype(np.int8)
        return buckets


class TargetRegistry:
    """Materialize the two payload buckets and boundary target for each key."""

    def __init__(self, partition: Partition, keys: Sequence[int], embedding_weight: torch.Tensor):
        self.partition = partition
        self.keys = [int(key) for key in keys]
        normalized = torch.nn.functional.normalize(embedding_weight.detach().float().cpu(), dim=-1)
        masks: list[np.ndarray] = []
        summaries: list[torch.Tensor] = []
        bucket_ids: list[np.ndarray] = []
        for key in self.keys:
            buckets = partition.bucket_ids(key)
            bucket_ids.append(buckets)
            key_masks = [buckets == 0, buckets == 1, buckets >= 0]
            boundary = np.zeros(partition.vocab_size, dtype=bool)
            boundary[partition.boundary_ids] = True
            key_masks.append(boundary)
            for mask in key_masks:
                masks.append(mask)
                ids = torch.from_numpy(np.flatnonzero(mask).astype(np.int64))
                if ids.numel():
                    summaries.append(normalized.index_select(0, ids).mean(dim=0))
                else:
                    summaries.append(torch.zeros(normalized.shape[1]))
        self.masks = torch.from_numpy(np.stack(masks, axis=0))
        self.summaries = torch.stack(summaries, dim=0)
        self.bucket_ids_by_key = np.stack(bucket_ids, axis=0)
        eligible = np.ones(partition.vocab_size, dtype=bool)
        eligible[partition.banned_ids] = False
        self.kgw_eligible_mask = eligible
        self.binary_masks = self._composition_masks(
            (partition.binary_0_atoms, partition.binary_1_atoms)
            if partition.binary_0_atoms is not None
            and partition.binary_1_atoms is not None
            else None
        )
        self.cw_masks = self._composition_masks(partition.cw_atoms)
        self.generic_combinations = {
            rule_id: np.asarray(
                list(itertools.combinations(range(partition.num_leaves), cardinality)),
                dtype=np.int16,
            )
            for rule_id, cardinality in COMPOSITION_CARDINALITY_BY_RULE.items()
            if cardinality < partition.num_leaves
        }
        self.atom_bit_by_token = np.zeros(partition.vocab_size, dtype=np.uint32)
        valid_atoms = partition.leaf_ids >= 0
        self.atom_bit_by_token[valid_atoms] = np.left_shift(
            np.uint32(1), partition.leaf_ids[valid_atoms].astype(np.uint32)
        )
        self.generic_combination_bits = {
            rule_id: np.bitwise_or.reduce(
                np.left_shift(np.uint32(1), combinations.astype(np.uint32)),
                axis=1,
            )
            for rule_id, combinations in self.generic_combinations.items()
        }

    def _composition_masks(
        self, atom_groups: Sequence[Sequence[int] | None] | None
    ) -> torch.Tensor | None:
        if atom_groups is None or any(group is None for group in atom_groups):
            return None
        masks = [
            np.isin(self.partition.leaf_ids, tuple(int(value) for value in group))
            & self.kgw_eligible_mask
            for group in atom_groups
        ]
        if any(not bool(mask.any()) for mask in masks):
            raise ValueError("A configured target composition has no usable tokens.")
        return torch.from_numpy(np.stack(masks, axis=0))

    @staticmethod
    def descriptor_id(key_index: int, target_code: int) -> int:
        return int(key_index) * 4 + int(target_code)

    def batch(self, key_indices: torch.Tensor, target_codes: torch.Tensor, device: torch.device):
        descriptor_ids = key_indices.long() * 4 + target_codes.long()
        flat = descriptor_ids.reshape(-1).cpu()
        masks = self.masks.index_select(0, flat).reshape(*descriptor_ids.shape, -1).to(device)
        summaries = self.summaries.index_select(0, flat).reshape(*descriptor_ids.shape, -1).to(device)
        return masks, summaries

    def mixed_masks(
        self,
        key_indices: torch.Tensor,
        target_codes: torch.Tensor,
        rule_ids: torch.Tensor,
        kgw_seeds: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Resolve static descriptors and dynamic KGW seeds into exact masks."""
        key_indices = key_indices.reshape(-1).long().cpu()
        target_codes = target_codes.reshape(-1).long().cpu()
        rule_ids = rule_ids.reshape(-1).long().cpu()
        kgw_seeds = kgw_seeds.reshape(-1).long().cpu()
        if not (
            key_indices.numel()
            == target_codes.numel()
            == rule_ids.numel()
            == kgw_seeds.numel()
        ):
            raise ValueError("Mixed target descriptor lengths do not match.")
        resolved = torch.empty(
            (rule_ids.numel(), self.partition.vocab_size), dtype=torch.bool
        )
        eligible = self.kgw_eligible_mask
        generic_rows = np.flatnonzero(
            np.isin(rule_ids.numpy(), tuple(COMPOSITION_CARDINALITY_BY_RULE))
        )
        for rule_id in COMPOSITION_CARDINALITY_BY_RULE:
            rows = generic_rows[rule_ids.numpy()[generic_rows] == int(rule_id)]
            if rows.size == 0:
                continue
            combination_indices = kgw_seeds.numpy()[rows]
            if (
                (combination_indices < 0).any()
                or (combination_indices >= len(self.generic_combination_bits[rule_id])).any()
            ):
                raise ValueError(
                    f"Invalid composition index for generic rule {rule_id}."
                )
            bitmasks = self.generic_combination_bits[rule_id][combination_indices]
            masks = (
                np.bitwise_and(bitmasks[:, None], self.atom_bit_by_token[None, :])
                != 0
            )
            masks &= eligible[None, :]
            resolved[torch.from_numpy(rows)] = torch.from_numpy(masks)
        for row in range(rule_ids.numel()):
            rule_id = int(rule_ids[row])
            code = int(target_codes[row])
            if rule_id in COMPOSITION_CARDINALITY_BY_RULE:
                continue
            if rule_id == RULE_ECC:
                descriptor = int(key_indices[row]) * 4 + code
                resolved[row] = self.masks[descriptor]
            elif rule_id == RULE_BINARY:
                if self.binary_masks is None:
                    raise ValueError("Partition has no binary composition.")
                if code not in (0, 1):
                    raise ValueError(f"Invalid binary target code: {code}")
                resolved[row] = self.binary_masks[code]
            elif rule_id == RULE_UNIGRAM:
                if self.binary_masks is None:
                    raise ValueError("Partition has no balanced binary composition.")
                resolved[row] = self.binary_masks[0]
            elif rule_id == RULE_CW_ABCD:
                if self.cw_masks is None:
                    raise ValueError("Partition has no CW-ABCD composition.")
                if code not in (0, 1, 2, 3):
                    raise ValueError(f"Invalid CW target code: {code}")
                resolved[row] = self.cw_masks[code]
            elif rule_id in DYNAMIC_RULE_IDS:
                kgw_seed_value = int(kgw_seeds[row])
                if kgw_seed_value < 0:
                    raise ValueError("Dynamic KGW target is missing its row seed.")
                if rule_id in (RULE_ATOMIC_KGW, RULE_ATOMIC_KGW25, RULE_ENTROPY_KGW):
                    selected_atoms = (
                        self.partition.num_leaves // 4
                        if rule_id == RULE_ATOMIC_KGW25
                        else self.partition.num_leaves // 2
                    )
                    mask = atomic_green_mask_for_seed(
                        kgw_seed_value,
                        self.partition.leaf_ids,
                        selected_atoms=selected_atoms,
                        num_atoms=self.partition.num_leaves,
                    )
                else:
                    mask = green_mask_for_seed(
                        kgw_seed_value, self.partition.vocab_size
                    )
                resolved[row] = torch.from_numpy(mask & eligible)
            else:
                raise ValueError(f"Unknown target rule id: {rule_id}")
        return resolved.to(device)


def make_fixed_ecc_schedule(
    count: int,
    generated_tokens: int,
    *,
    seed: int,
    key_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if generated_tokens % 8:
        raise ValueError("generated_tokens must be divisible by 8 (7 payload + 1 boundary).")
    codebook = feasible_codewords()
    target_codes = np.empty((count, generated_tokens), dtype=np.int8)
    key_indices = np.empty(count, dtype=np.uint16)
    for row in range(count):
        rng = np.random.default_rng(stable_seed("ecc-schedule", seed, row))
        key_indices[row] = int(rng.integers(0, key_count))
        position = 0
        while position < generated_tokens:
            codeword = codebook[int(rng.integers(0, len(codebook)))]
            target_codes[row, position : position + 7] = np.asarray(codeword, dtype=np.uint8)
            target_codes[row, position + 7] = 3
            position += 8
    return key_indices, target_codes


def make_rule_schedule(
    rule_name: str,
    count: int,
    generated_tokens: int,
    *,
    seed: int,
    key_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the native target sequence for one supported watermark rule."""
    rule_name = str(rule_name).lower()
    if rule_name == "ecc":
        return make_fixed_ecc_schedule(
            count,
            generated_tokens,
            seed=seed,
            key_count=key_count,
        )

    key_indices = np.empty(int(count), dtype=np.uint16)
    for row in range(int(count)):
        rng = np.random.default_rng(stable_seed(rule_name, seed, row))
        key_indices[row] = int(rng.integers(0, key_count))

    if rule_name == "binary":
        pattern = np.asarray([0, 1], dtype=np.uint8)
    elif rule_name == "unigram":
        # A fixed balanced green list at every position (Unigram-style target).
        pattern = np.asarray([0], dtype=np.uint8)
    elif rule_name == "cw_abcd":
        # Original combinatorial pattern ACADBCBD with A/B/C/D = 0/1/2/3.
        pattern = np.asarray([0, 2, 0, 3, 1, 2, 1, 3], dtype=np.uint8)
    elif rule_name in {"kgw", "atomic_kgw", "atomic_kgw25", "entropy_kgw"}:
        pattern = np.asarray([0], dtype=np.uint8)
    else:
        raise ValueError(f"Unknown watermark rule: {rule_name}")

    tiled = np.resize(pattern, int(generated_tokens)).astype(np.uint8)
    target_codes = np.repeat(tiled[None, :], int(count), axis=0)
    return key_indices, target_codes
