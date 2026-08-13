from __future__ import annotations

import unittest

import torch
import numpy as np
from types import SimpleNamespace
from transformers import OPTConfig, OPTForCausalLM

from modeling import ConditionProjector, TargetConditionedLoRA, target_geometry_features
from run_experiment import aggregate_active_loss, batch_objective
from targets import RULE_BINARY, RULE_ECC, Partition, TargetRegistry


class Qwen3DiagnosticTests(unittest.TestCase):
    def test_legacy_token_micro_is_plain_mean(self) -> None:
        values = torch.tensor([1.0, 3.0, 8.0])
        rules = torch.tensor([RULE_ECC, RULE_ECC, RULE_BINARY])
        codes = torch.tensor([0, 1, 0])
        result = aggregate_active_loss(
            values,
            rules,
            codes,
            aggregation="token_micro",
            ecc_boundary_weight=1.0,
        )
        torch.testing.assert_close(result, values.mean())

    def test_rule_macro_equalizes_rule_contributions(self) -> None:
        values = torch.tensor([1.0, 3.0, 8.0])
        rules = torch.tensor([RULE_ECC, RULE_ECC, RULE_BINARY])
        codes = torch.tensor([0, 1, 0])
        result = aggregate_active_loss(
            values,
            rules,
            codes,
            aggregation="rule_macro",
            ecc_boundary_weight=1.0,
        )
        self.assertAlmostEqual(float(result), 5.0)

    def test_ecc_boundary_weight_is_applied_only_to_boundary(self) -> None:
        values = torch.tensor([1.0, 3.0, 9.0])
        rules = torch.tensor([RULE_ECC, RULE_ECC, RULE_BINARY])
        codes = torch.tensor([0, 3, 3])
        result = aggregate_active_loss(
            values,
            rules,
            codes,
            aggregation="token_micro",
            ecc_boundary_weight=7.0,
        )
        self.assertAlmostEqual(float(result), (1.0 + 7.0 * 3.0 + 9.0) / 9.0)

    def test_geometry_features_are_bounded_and_rule_agnostic(self) -> None:
        logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
        masks = torch.tensor([[True, False, False], [True, True, False]])
        features = target_geometry_features(logits, masks)
        self.assertEqual(tuple(features.shape), (2, 3))
        self.assertTrue(bool(((features >= 0.0) & (features <= 1.0)).all()))
        torch.testing.assert_close(features[:, 1], torch.tensor([1 / 3, 2 / 3]))

    def test_condition_projector_accepts_geometry_features(self) -> None:
        projector = ConditionProjector(hidden_size=8, rank=2, feature_dim=3)
        summaries = torch.randn(4, 8)
        features = torch.rand(4, 3)
        output = projector(summaries, features)
        self.assertEqual(tuple(output.shape), (4, 8))
        with self.assertRaises(ValueError):
            projector(summaries)

    def test_geometry_conditioning_backpropagates_end_to_end(self) -> None:
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
        model = TargetConditionedLoRA(
            base_model,
            rank=4,
            alpha=8.0,
            dropout=0.0,
            condition_feature_dim=3,
        )
        partition = Partition(
            leaf_ids=(np.arange(32, dtype=np.int16) % 4),
            semantic_group_ids=np.arange(32, dtype=np.int32),
            boundary_ids=np.asarray([3, 7], dtype=np.int64),
            banned_ids=np.asarray([], dtype=np.int64),
            num_leaves=4,
        )
        registry = TargetRegistry(
            partition, [101], base_model.get_input_embeddings().weight
        )
        label = int(torch.nonzero(registry.masks[0], as_tuple=False)[0])
        batch = {
            "input_ids": torch.tensor([[2, 3, 4]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "labels": torch.tensor([[-100, label, -100]]),
            "target_codes": torch.tensor([[-1, 0, -1]]),
            "key_indices": torch.tensor([[-1, 0, -1]]),
            "rule_ids": torch.tensor([[-1, RULE_ECC, -1]]),
            "kgw_seeds": torch.tensor([[-1, -1, -1]]),
            "sequence_indices": torch.tensor([0]),
        }
        args = SimpleNamespace(
            training_targets_per_sequence=0,
            seed=7,
            condition_summary_mode="topk_contrast",
            condition_top_k=8,
            condition_feature_mode="geometry",
            objective_aggregation="rule_macro",
            ecc_boundary_loss_weight=1.0,
            target_adherence=0.9,
            residual_loss_weight=0.02,
            target_mass_loss_weight=0.0,
            kl_loss_weight=0.02,
        )
        loss, _ = batch_objective(
            model, registry, batch, args, torch.device("cpu")
        )
        loss.backward()
        feature_gradient = float(model.conditioner.feature_up.weight.grad.abs().sum())
        self.assertGreater(feature_gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
