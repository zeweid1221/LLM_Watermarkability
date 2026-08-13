import unittest

import numpy as np

from analyze_teacher_forced_watermarkability import minimum_intervention


class MinimumInterventionTests(unittest.TestCase):
    def test_matches_bernoulli_kl_projection(self) -> None:
        bias, kl = minimum_intervention(np.array([0.5]), tau=0.9)
        expected_bias = np.log(0.9 / 0.1)
        expected_kl = 0.9 * np.log(0.9 / 0.5) + 0.1 * np.log(0.1 / 0.5)
        self.assertAlmostEqual(float(bias[0]), expected_bias)
        self.assertAlmostEqual(float(kl[0]), expected_kl)

    def test_requires_no_control_above_target_mass(self) -> None:
        bias, kl = minimum_intervention(np.array([0.9, 0.95]), tau=0.9)
        np.testing.assert_array_equal(bias, np.zeros(2))
        np.testing.assert_array_equal(kl, np.zeros(2))

    def test_clips_small_probability_roundoff(self) -> None:
        bias, kl = minimum_intervention(np.array([1.0 + 1e-7]), tau=0.9)
        np.testing.assert_array_equal(bias, np.zeros(1))
        np.testing.assert_array_equal(kl, np.zeros(1))

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            minimum_intervention(np.array([-0.1]), tau=0.9)
        with self.assertRaises(ValueError):
            minimum_intervention(np.array([0.5]), tau=1.0)


if __name__ == "__main__":
    unittest.main()
