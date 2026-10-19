from __future__ import annotations

import argparse
import unittest

from demo_watermarkability import (
    DEFAULT_CHECKPOINT,
    DEFAULT_METADATA,
    DEFAULT_PARTITION,
    load_and_verify_metadata,
    validate_options,
)


class DemoWatermarkabilityTest(unittest.TestCase):
    def test_released_artifacts_match_metadata(self) -> None:
        metadata = load_and_verify_metadata(
            DEFAULT_METADATA, DEFAULT_CHECKPOINT, DEFAULT_PARTITION
        )
        self.assertEqual(metadata["model_name"], "facebook/opt-125m")
        self.assertEqual(metadata["paper_tau"], 0.99)

    def test_tau_must_be_a_probability(self) -> None:
        args = argparse.Namespace(
            tau=1.0,
            max_new_tokens=16,
            max_prompt_tokens=160,
            temperature=0.75,
            top_k=0,
            rule="binary",
        )
        with self.assertRaisesRegex(ValueError, "--tau"):
            validate_options(args)

    def test_ecc_length_must_contain_complete_blocks(self) -> None:
        args = argparse.Namespace(
            tau=0.99,
            max_new_tokens=15,
            max_prompt_tokens=160,
            temperature=0.75,
            top_k=0,
            rule="ecc",
        )
        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            validate_options(args)


if __name__ == "__main__":
    unittest.main()
