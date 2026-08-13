from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from transformers import OPTConfig, OPTForCausalLM

from modeling import TargetConditionedLoRA
from run_experiment import batch_objective
from targets import (
    COMPOSITION_RULE_BY_CARDINALITY,
    RULE_ECC,
    RULE_KGW,
    Partition,
    TargetRegistry,
)


class MixedObjectiveIntegrationTests(unittest.TestCase):
    def test_static_and_dynamic_targets_backpropagate_together(self) -> None:
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
        )
        leaf_ids = np.full(32, -1, dtype=np.int16)
        leaf_ids[:30] = np.arange(30, dtype=np.int16) % 4
        partition = Partition(
            leaf_ids=leaf_ids,
            semantic_group_ids=np.arange(32, dtype=np.int32),
            boundary_ids=np.asarray([30], dtype=np.int64),
            banned_ids=np.asarray([31], dtype=np.int64),
            num_leaves=4,
        )
        registry = TargetRegistry(
            partition,
            [101],
            base_model.get_input_embeddings().weight,
        )
        dynamic_mask = registry.mixed_masks(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([RULE_KGW]),
            torch.tensor([12345]),
            device=torch.device("cpu"),
        )[0]
        dynamic_label = int(torch.nonzero(dynamic_mask, as_tuple=False)[0])
        static_label = int(torch.nonzero(registry.masks[0], as_tuple=False)[0])
        batch = {
            "input_ids": torch.tensor([[2, 3, 4, 5]]),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "labels": torch.tensor([[-100, static_label, dynamic_label, -100]]),
            "target_codes": torch.tensor([[-1, 0, 0, -1]]),
            "key_indices": torch.tensor([[-1, 0, 0, -1]]),
            "rule_ids": torch.tensor([[-1, RULE_ECC, RULE_KGW, -1]]),
            "kgw_seeds": torch.tensor([[-1, -1, 12345, -1]]),
            "sequence_indices": torch.tensor([0]),
        }
        args = SimpleNamespace(
            training_targets_per_sequence=0,
            seed=7,
            condition_summary_mode="topk_contrast",
            condition_top_k=8,
            target_adherence=0.9,
            residual_loss_weight=0.02,
            target_mass_loss_weight=0.1,
            kl_loss_weight=0.02,
        )
        loss, metrics = batch_objective(
            model,
            registry,
            batch,
            args,
            torch.device("cpu"),
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradient = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.trainable_parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient, 0.0)
        self.assertGreater(metrics["native_target_mass"], 0.0)

    def test_generic_composition_target_backpropagates_without_watermark_rule(self) -> None:
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
            boundary_ids=np.asarray([], dtype=np.int64),
            banned_ids=np.asarray([], dtype=np.int64),
            num_leaves=16,
        )
        registry = TargetRegistry(
            partition, [0], base_model.get_input_embeddings().weight
        )
        rule_id = COMPOSITION_RULE_BY_CARDINALITY[7]
        mask = registry.mixed_masks(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([rule_id]),
            torch.tensor([314]),
            device=torch.device("cpu"),
        )[0]
        label = int(torch.nonzero(mask, as_tuple=False)[0])
        batch = {
            "input_ids": torch.tensor([[2, 3, 4]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "labels": torch.tensor([[-100, label, -100]]),
            "target_codes": torch.tensor([[-1, 0, -1]]),
            "key_indices": torch.tensor([[-1, 0, -1]]),
            "rule_ids": torch.tensor([[-1, rule_id, -1]]),
            "kgw_seeds": torch.tensor([[-1, 314, -1]]),
            "sequence_indices": torch.tensor([0]),
        }
        args = SimpleNamespace(
            training_targets_per_sequence=0,
            seed=7,
            condition_summary_mode="topk_contrast",
            condition_top_k=8,
            condition_feature_mode="none",
            target_adherence=0.9,
            residual_loss_weight=0.02,
            target_mass_loss_weight=0.1,
            kl_loss_weight=0.02,
            objective_aggregation="token_micro",
            ecc_boundary_loss_weight=1.0,
        )
        loss, _ = batch_objective(
            model, registry, batch, args, torch.device("cpu")
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(
            sum(
                float(parameter.grad.abs().sum())
                for parameter in model.trainable_parameters()
                if parameter.grad is not None
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
