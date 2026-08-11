from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


@dataclass
class HierarchicalPartition:
    """A materialized token-to-leaf hierarchy with deterministic keyed variants."""

    leaf_ids: np.ndarray
    num_leaves: int
    boundary_ids: Sequence[int] = field(default_factory=list)
    banned_ids: Sequence[int] = field(default_factory=list)
    semantic_group_ids: np.ndarray | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    _key_cache: Dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.leaf_ids = np.asarray(self.leaf_ids, dtype=np.int16)
        self.boundary_ids = [int(x) for x in self.boundary_ids]
        self.banned_ids = [int(x) for x in self.banned_ids]
        if self.semantic_group_ids is not None:
            self.semantic_group_ids = np.asarray(self.semantic_group_ids, dtype=np.int32)
            if self.semantic_group_ids.shape != self.leaf_ids.shape:
                raise ValueError("semantic_group_ids must match leaf_ids shape.")
        valid = self.leaf_ids[self.leaf_ids >= 0]
        if valid.size and (valid.max() >= self.num_leaves or valid.min() < 0):
            raise ValueError("leaf_ids contains a leaf outside the configured hierarchy.")
        if self.num_leaves <= 0 or self.num_leaves & (self.num_leaves - 1):
            raise ValueError("num_leaves must be a positive power of two.")

    @property
    def vocab_size(self) -> int:
        return int(self.leaf_ids.shape[0])

    @property
    def usable_mask(self) -> np.ndarray:
        return self.leaf_ids >= 0

    def bucket_ids(self, num_buckets: int, leaf_ids: np.ndarray | None = None) -> np.ndarray:
        if num_buckets <= 0 or self.num_leaves % int(num_buckets) != 0:
            raise ValueError(f"num_buckets={num_buckets} must divide num_leaves={self.num_leaves}.")
        leaves = self.leaf_ids if leaf_ids is None else np.asarray(leaf_ids, dtype=np.int16)
        group_size = self.num_leaves // int(num_buckets)
        out = np.full(leaves.shape, -1, dtype=np.int16)
        usable = leaves >= 0
        out[usable] = leaves[usable] // group_size
        return out

    def target_mask(
        self,
        num_buckets: int,
        target_buckets: Iterable[int],
        *,
        key: int | str | None = None,
    ) -> np.ndarray:
        leaves = self.keyed_leaf_ids(key) if key is not None else self.leaf_ids
        buckets = self.bucket_ids(num_buckets, leaves)
        targets = np.asarray(sorted(set(int(x) for x in target_buckets)), dtype=np.int16)
        return np.isin(buckets, targets) & (buckets >= 0)

    def atom_target_mask(
        self,
        target_atoms: Iterable[int],
        *,
        key: int | str | None = None,
    ) -> np.ndarray:
        """Return the exact vocabulary mask for an arbitrary union of atomic leaves."""
        atoms = np.asarray(sorted(set(int(x) for x in target_atoms)), dtype=np.int16)
        if atoms.size == 0:
            return np.zeros(self.leaf_ids.shape, dtype=bool)
        if atoms.min() < 0 or atoms.max() >= self.num_leaves:
            raise ValueError(f"target_atoms must lie in [0, {self.num_leaves}).")
        leaves = self.keyed_leaf_ids(key) if key is not None else self.leaf_ids
        return np.isin(leaves, atoms) & (leaves >= 0)

    def keyed_leaf_ids(self, key: int | str | None) -> np.ndarray:
        """Permute assignments within matched semantic groups without changing group counts."""
        if key is None:
            return self.leaf_ids.copy()
        cache_key = f"{type(key).__name__}:{key}"
        cached = self._key_cache.get(cache_key)
        if cached is not None:
            return cached
        groups = self.semantic_group_ids
        if groups is None:
            groups = np.zeros(self.vocab_size, dtype=np.int32)
        out = self.leaf_ids.copy()
        usable_groups = np.unique(groups[self.usable_mask])
        for group_id in usable_groups.tolist():
            indices = np.flatnonzero((groups == group_id) & self.usable_mask)
            if indices.size <= 1:
                continue
            rng = np.random.default_rng(stable_seed("partition-key", key, int(group_id)))
            values = out[indices].copy()
            rng.shuffle(values)
            out[indices] = values
        self._key_cache[cache_key] = out
        return out

    def save(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "leaf_ids.npy", self.leaf_ids.astype(np.int16))
        np.save(path / "boundary_ids.npy", np.asarray(self.boundary_ids, dtype=np.int32))
        np.save(path / "banned_ids.npy", np.asarray(self.banned_ids, dtype=np.int32))
        if self.semantic_group_ids is not None:
            np.save(path / "semantic_group_ids.npy", self.semantic_group_ids.astype(np.int32))
        payload = {
            "num_leaves": int(self.num_leaves),
            "vocab_size": self.vocab_size,
            "metadata": self.metadata,
        }
        (path / "hierarchy_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, input_dir: str | Path) -> "HierarchicalPartition":
        path = Path(input_dir)
        config = json.loads((path / "hierarchy_config.json").read_text(encoding="utf-8"))
        semantic_path = path / "semantic_group_ids.npy"
        return cls(
            leaf_ids=np.load(path / "leaf_ids.npy", allow_pickle=False),
            num_leaves=int(config["num_leaves"]),
            boundary_ids=np.load(path / "boundary_ids.npy", allow_pickle=False).tolist(),
            banned_ids=np.load(path / "banned_ids.npy", allow_pickle=False).tolist(),
            semantic_group_ids=(
                np.load(semantic_path, allow_pickle=False) if semantic_path.exists() else None
            ),
            metadata=dict(config.get("metadata") or {}),
        )
