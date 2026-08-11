from __future__ import annotations

import unittest

import numpy as np
import torch

from targets import (
    RULE_ATOMIC_KGW,
    RULE_ATOMIC_KGW25,
    RULE_BINARY,
    RULE_CW_ABCD,
    RULE_ENTROPY_KGW,
    RULE_UNIGRAM,
    Partition,
    TargetRegistry,
    atomic_green_mask_for_seed,
    make_rule_schedule,
)


class Atomic16TargetTests(unittest.TestCase):
    @staticmethod
    def full_partition() -> Partition:
        return Partition(
            leaf_ids=np.arange(16, dtype=np.int16),
            semantic_group_ids=np.zeros(16, dtype=np.int32),
            boundary_ids=np.asarray([14, 15], dtype=np.int64),
            banned_ids=np.asarray([], dtype=np.int64),
            num_leaves=16,
            ecc_payload_0_atoms=tuple(range(0, 7)),
            ecc_payload_1_atoms=tuple(range(7, 14)),
            ecc_boundary_atoms=(14, 15),
            binary_0_atoms=tuple(range(0, 8)),
            binary_1_atoms=tuple(range(8, 16)),
            cw_atoms=(
                (0, 1, 2, 3),
                (4, 5, 6, 7),
                (8, 9, 10, 11),
                (12, 13, 14, 15),
            ),
        )

    def test_ecc_composition_is_materialized_without_key_reshuffling(self) -> None:
        partition = Partition(
            leaf_ids=np.arange(16, dtype=np.int16),
            semantic_group_ids=np.zeros(16, dtype=np.int32),
            boundary_ids=np.asarray([14, 15], dtype=np.int64),
            banned_ids=np.asarray([], dtype=np.int64),
            num_leaves=16,
            ecc_payload_0_atoms=tuple(range(0, 7)),
            ecc_payload_1_atoms=tuple(range(7, 14)),
            ecc_boundary_atoms=(14, 15),
        )
        buckets_a = partition.bucket_ids(1000)
        buckets_b = partition.bucket_ids(900000)

        np.testing.assert_array_equal(buckets_a, buckets_b)
        self.assertTrue(np.all(buckets_a[:7] == 0))
        self.assertTrue(np.all(buckets_a[7:14] == 1))
        self.assertTrue(np.all(buckets_a[14:] == -1))

    def test_binary_and_cw_schedules_match_native_patterns(self) -> None:
        _, binary = make_rule_schedule(
            "binary", 2, 16, seed=7, key_count=1
        )
        _, cw = make_rule_schedule("cw_abcd", 2, 16, seed=7, key_count=1)
        np.testing.assert_array_equal(binary[0], np.resize([0, 1], 16))
        np.testing.assert_array_equal(
            cw[0], np.resize([0, 2, 0, 3, 1, 2, 1, 3], 16)
        )
        _, unigram = make_rule_schedule("unigram", 2, 16, seed=7, key_count=1)
        _, kgw25 = make_rule_schedule("atomic_kgw25", 2, 16, seed=7, key_count=1)
        _, entropy = make_rule_schedule("entropy_kgw", 2, 16, seed=7, key_count=1)
        np.testing.assert_array_equal(unigram[0], np.zeros(16, dtype=np.int8))
        np.testing.assert_array_equal(kgw25[0], np.zeros(16, dtype=np.int8))
        np.testing.assert_array_equal(entropy[0], np.zeros(16, dtype=np.int8))

    def test_registry_materializes_binary_and_cw_atom_unions(self) -> None:
        registry = TargetRegistry(self.full_partition(), [101], torch.eye(16))
        masks = registry.mixed_masks(
            torch.tensor([0, 0, 0]),
            torch.tensor([1, 2, 3]),
            torch.tensor([RULE_BINARY, RULE_CW_ABCD, RULE_CW_ABCD]),
            torch.tensor([-1, -1, -1]),
            device=torch.device("cpu"),
        ).numpy()
        np.testing.assert_array_equal(np.flatnonzero(masks[0]), np.arange(8, 16))
        np.testing.assert_array_equal(np.flatnonzero(masks[1]), np.arange(8, 12))
        np.testing.assert_array_equal(np.flatnonzero(masks[2]), np.arange(12, 16))

    def test_atomic_kgw_selects_complete_atoms(self) -> None:
        partition = self.full_partition()
        registry = TargetRegistry(partition, [101], torch.eye(16))
        mask = registry.mixed_masks(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([RULE_ATOMIC_KGW]),
            torch.tensor([12345]),
            device=torch.device("cpu"),
        )[0].numpy()
        expected = atomic_green_mask_for_seed(
            12345, partition.leaf_ids, selected_atoms=8, num_atoms=16
        )
        np.testing.assert_array_equal(mask, expected)
        self.assertEqual(int(mask.sum()), 8)

    def test_unigram_and_dynamic_variants_use_expected_atom_counts(self) -> None:
        partition = self.full_partition()
        registry = TargetRegistry(partition, [101], torch.eye(16))
        masks = registry.mixed_masks(
            torch.tensor([0, 0, 0]),
            torch.tensor([0, 0, 0]),
            torch.tensor([RULE_UNIGRAM, RULE_ATOMIC_KGW25, RULE_ENTROPY_KGW]),
            torch.tensor([-1, 12345, 12345]),
            device=torch.device("cpu"),
        ).numpy()
        np.testing.assert_array_equal(np.flatnonzero(masks[0]), np.arange(0, 8))
        self.assertEqual(int(masks[1].sum()), 4)
        self.assertEqual(int(masks[2].sum()), 8)


if __name__ == "__main__":
    unittest.main()
