import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_cross_rule_watermarkability import (
    fixed_bias_statistics,
    load_reference_texts,
)
from analyze_teacher_forced_watermarkability import minimum_intervention


class CrossRuleWatermarkabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_path = Path("test_cross_rule_reference_tmp.csv")

    def tearDown(self) -> None:
        self.reference_path.unlink(missing_ok=True)

    def test_fixed_bias_matches_binary_exponential_tilt(self) -> None:
        mass = np.asarray([0.2, 0.5, 0.8], dtype=np.float64)
        controlled, kl = fixed_bias_statistics(mass, 2.0)
        normalizer = 1.0 - mass + np.exp(2.0) * mass
        expected = np.exp(2.0) * mass / normalizer
        np.testing.assert_allclose(controlled, expected)
        np.testing.assert_allclose(kl, 2.0 * expected - np.log(normalizer))

    def test_fixed_bias_clips_float32_probability_roundoff(self) -> None:
        mass = np.asarray([1.0000001, 1.0, 0.0], dtype=np.float64)
        controlled, kl = fixed_bias_statistics(mass, 20.0)
        self.assertTrue(np.isfinite(controlled).all())
        self.assertTrue(np.isfinite(kl).all())
        self.assertTrue(((controlled >= 0.0) & (controlled <= 1.0)).all())
        self.assertTrue((kl >= -1e-12).all())

    def test_matched_frontier_is_zero_above_tau(self) -> None:
        bias, cost = minimum_intervention(np.asarray([0.99, 0.999]), 0.99)
        np.testing.assert_array_equal(bias, np.zeros(2))
        np.testing.assert_array_equal(cost, np.zeros(2))

    def test_reference_loader_requires_identical_shared_text(self) -> None:
        pd.DataFrame(
            [
                {"mode": "unwatermarked", "sequence_index": 0, "target_rule": "ecc", "generated_text": "same"},
                {"mode": "unwatermarked", "sequence_index": 0, "target_rule": "binary", "generated_text": "same"},
                {"mode": "unwatermarked", "sequence_index": 1, "target_rule": "ecc", "generated_text": "text"},
                {"mode": "unwatermarked", "sequence_index": 1, "target_rule": "binary", "generated_text": "text"},
            ]
        ).to_csv(self.reference_path, index=False)
        self.assertEqual(
            load_reference_texts(self.reference_path, 2), ["same", "text"]
        )

    def test_reference_loader_rejects_rule_specific_text(self) -> None:
        pd.DataFrame(
            [
                {"mode": "unwatermarked", "sequence_index": 0, "target_rule": "ecc", "generated_text": "a"},
                {"mode": "unwatermarked", "sequence_index": 0, "target_rule": "binary", "generated_text": "b"},
            ]
        ).to_csv(self.reference_path, index=False)
        with self.assertRaises(ValueError):
            load_reference_texts(self.reference_path, 1)


if __name__ == "__main__":
    unittest.main()
