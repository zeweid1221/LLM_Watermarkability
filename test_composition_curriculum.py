from __future__ import annotations

import itertools
import unittest

import numpy as np
import torch
from transformers import OPTConfig, OPTForCausalLM

from modeling import TargetConditionedLoRA
from run_experiment import prepare_trajectories
from targets import (
    COMPOSITION_RULE_BY_CARDINALITY,
    Partition,
    TargetRegistry,
    make_composition_curriculum,
)


class CompositionCurriculumTests(unittest.TestCase):
    def test_small_composition_space_is_exhausted_before_repeating(self) -> None:
        prompts, keys, codes, rules, indices = make_composition_curriculum(
            ["prompt"],
            repeats=1,
            generated_tokens=6,
            cardinalities=[2],
            num_atoms=4,
        )
        self.assertEqual(prompts, ["prompt"])
        np.testing.assert_array_equal(keys, np.zeros(1, dtype=np.uint16))
        np.testing.assert_array_equal(codes, np.zeros((1, 6), dtype=np.int8))
        np.testing.assert_array_equal(
            rules, np.asarray([COMPOSITION_RULE_BY_CARDINALITY[2]], dtype=np.uint8)
        )
        np.testing.assert_array_equal(indices[0], np.arange(6, dtype=np.int32))

    def test_cardinality_assignment_is_deterministic_and_rotates_by_repeat(self) -> None:
        _, _, _, rules, _ = make_composition_curriculum(
            ["a", "b", "c", "d"],
            repeats=2,
            generated_tokens=1,
            cardinalities=[2, 4, 7, 8],
            num_atoms=16,
        )
        expected = [2, 4, 7, 8, 4, 7, 8, 2]
        actual = [
            next(
                cardinality
                for cardinality, rule_id in COMPOSITION_RULE_BY_CARDINALITY.items()
                if rule_id == int(value)
            )
            for value in rules
        ]
        self.assertEqual(actual, expected)

    def test_combination_index_materializes_the_exact_atom_union(self) -> None:
        partition = Partition(
            leaf_ids=np.arange(16, dtype=np.int16),
            semantic_group_ids=np.zeros(16, dtype=np.int32),
            boundary_ids=np.asarray([], dtype=np.int64),
            banned_ids=np.asarray([15], dtype=np.int64),
            num_leaves=16,
        )
        registry = TargetRegistry(partition, [0], torch.eye(16))
        combinations = list(itertools.combinations(range(16), 4))
        index = 137
        mask = registry.mixed_masks(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([COMPOSITION_RULE_BY_CARDINALITY[4]]),
            torch.tensor([index]),
            device=torch.device("cpu"),
        )[0].numpy()
        expected = np.zeros(16, dtype=bool)
        expected[list(combinations[index])] = True
        expected[15] = False
        np.testing.assert_array_equal(mask, expected)

    def test_full_protocol_balances_cardinalities_and_covers_each_space(self) -> None:
        _, _, _, rules, indices = make_composition_curriculum(
            [f"prompt-{index}" for index in range(5000)],
            repeats=4,
            generated_tokens=96,
            cardinalities=[2, 4, 7, 8],
            num_atoms=16,
        )
        counts = []
        for cardinality in (2, 4, 7, 8):
            selected = rules == COMPOSITION_RULE_BY_CARDINALITY[cardinality]
            counts.append(int(selected.sum()))
            self.assertEqual(
                np.unique(indices[selected]).size,
                len(list(itertools.combinations(range(16), cardinality))),
            )
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_preparation_uses_cached_composition_indices_exactly(self) -> None:
        config = OPTConfig(
            vocab_size=32,
            hidden_size=16,
            ffn_dim=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=32,
            word_embed_proj_dim=16,
            pad_token_id=1,
            bos_token_id=2,
            eos_token_id=2,
        )
        base_model = OPTForCausalLM(config)
        model = TargetConditionedLoRA(base_model, rank=4, alpha=8.0, dropout=0.0)
        partition = Partition(
            leaf_ids=np.arange(32, dtype=np.int16) % 16,
            semantic_group_ids=np.arange(32, dtype=np.int32),
            boundary_ids=np.asarray([31], dtype=np.int64),
            banned_ids=np.asarray([31], dtype=np.int64),
            num_leaves=16,
        )
        registry = TargetRegistry(
            partition, [0], base_model.get_input_embeddings().weight
        )

        class TinyTokenizer:
            padding_side = "left"

            def __call__(self, prompts, **_):
                batch = len(prompts)
                return {
                    "input_ids": torch.tensor([[2, 3]] * batch),
                    "attention_mask": torch.ones((batch, 2), dtype=torch.long),
                }

        preset = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
        generated, returned_indices = prepare_trajectories(
            model,
            TinyTokenizer(),
            ["prompt"],
            registry,
            np.asarray([0], dtype=np.uint16),
            np.zeros((1, 4), dtype=np.int8),
            np.asarray([COMPOSITION_RULE_BY_CARDINALITY[2]], dtype=np.uint8),
            preset,
            max_prompt_tokens=8,
            batch_size=1,
            temperature=1.0,
            kgw_seed_count=1,
            kgw_seed_offset=3779,
            entropy_gate_threshold=2.0,
            seed=9,
            device=torch.device("cpu"),
        )
        np.testing.assert_array_equal(returned_indices, preset)
        combinations = list(itertools.combinations(range(16), 2))
        for position, token_id in enumerate(generated[0].tolist()):
            self.assertIn(
                int(partition.leaf_ids[token_id]), combinations[int(preset[0, position])]
            )


if __name__ == "__main__":
    unittest.main()
