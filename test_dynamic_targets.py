from __future__ import annotations

import unittest

import numpy as np
import torch

from modeling import topk_contrast_summaries
from targets import (
    RULE_ECC,
    RULE_KGW,
    Partition,
    TargetRegistry,
    green_mask_for_seed,
    kgw_row_seed,
    make_rule_assignments,
)


class DynamicTargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.partition = Partition(
            leaf_ids=np.asarray([0, 1, 2, 3, 0, 1, -1, -1], dtype=np.int16),
            semantic_group_ids=np.arange(8, dtype=np.int32),
            boundary_ids=np.asarray([6], dtype=np.int64),
            banned_ids=np.asarray([7], dtype=np.int64),
            num_leaves=4,
        )
        embeddings = torch.eye(8, dtype=torch.float32)
        self.registry = TargetRegistry(self.partition, [101], embeddings)

    def test_static_registry_path_is_unchanged(self) -> None:
        key_indices = torch.tensor([0, 0])
        target_codes = torch.tensor([0, 3])
        masks = self.registry.mixed_masks(
            key_indices,
            target_codes,
            torch.tensor([RULE_ECC, RULE_ECC]),
            torch.tensor([-1, -1]),
            device=torch.device("cpu"),
        )
        expected = self.registry.masks.index_select(0, torch.tensor([0, 3]))
        self.assertTrue(torch.equal(masks, expected))

    def test_dynamic_mask_is_exact_and_excludes_banned_tokens(self) -> None:
        seed = kgw_row_seed(
            [3, 5, 7], key=101, seed_count=1, seed_offset=3779
        )
        mask = self.registry.mixed_masks(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([RULE_KGW]),
            torch.tensor([seed]),
            device=torch.device("cpu"),
        )[0]
        expected = green_mask_for_seed(seed, 8)
        expected[7] = False
        self.assertTrue(np.array_equal(mask.numpy(), expected))

    def test_dynamic_seed_is_deterministic_and_prefix_dependent(self) -> None:
        first = kgw_row_seed([11, 13], key=7, seed_count=1, seed_offset=3779)
        repeated = kgw_row_seed([11, 13], key=7, seed_count=1, seed_offset=3779)
        changed = kgw_row_seed([11, 17, 19], key=7, seed_count=1, seed_offset=3779)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)

    def test_rule_assignment_has_requested_count(self) -> None:
        assignments = make_rule_assignments(20, 0.25, seed=9)
        self.assertEqual(int((assignments == RULE_KGW).sum()), 5)
        self.assertEqual(int((assignments == RULE_ECC).sum()), 15)

    def test_topk_contrast_summary_supports_unseen_masks(self) -> None:
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
        embeddings = torch.eye(4)
        left = topk_contrast_summaries(
            logits,
            torch.tensor([[True, False, True, False]]),
            embeddings,
            top_k=4,
        )
        right = topk_contrast_summaries(
            logits,
            torch.tensor([[False, True, False, True]]),
            embeddings,
            top_k=4,
        )
        self.assertEqual(tuple(left.shape), (1, 4))
        self.assertTrue(torch.isfinite(left).all())
        self.assertTrue(torch.allclose(left, -right, atol=1e-6))

    def test_matches_repository_native_kgw_fixture(self) -> None:
        seed = kgw_row_seed(
            [2, 9, 5, 12], key=31415, seed_count=1, seed_offset=3779
        )
        self.assertEqual(seed, 1082315466)
        np.testing.assert_array_equal(
            green_mask_for_seed(seed, 8),
            np.asarray([1, 0, 1, 1, 0, 1, 1, 1], dtype=bool),
        )


if __name__ == "__main__":
    unittest.main()
