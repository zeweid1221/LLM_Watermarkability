from __future__ import annotations

import io
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from transformers import OPTConfig, OPTForCausalLM

from modeling import TargetConditionedLoRA
from build_atomic16_partition import collect_last_logits, derive_banned_ids
from reproducibility import encode_prompt, validate_tokenizer_identity
from run_experiment import (
    TrajectoryDataset,
    load_adapter,
    validate_trajectory_cache_metadata,
    weighted_metrics,
)
from targets import Partition


class FakeTokenizer:
    name_or_path = "fake/tokenizer"
    eos_token_id = 2

    def __len__(self) -> int:
        return 5

    def get_vocab(self) -> dict[str, int]:
        return {"<pad>": 0, "a": 1, "</s>": 2, "b": 3, "c": 4}

    def encode(self, _prompt: str, *, add_special_tokens: bool, **_) -> list[int]:
        return [2, 1, 3] if add_special_tokens else [1, 3]


class ReproducibilityTests(unittest.TestCase):
    def test_prompt_tokenization_keeps_model_special_tokens(self) -> None:
        tokenizer = FakeTokenizer()
        self.assertEqual(encode_prompt(tokenizer, "prompt", max_length=8), [2, 1, 3])
        dataset = TrajectoryDataset(
            ["prompt"],
            np.asarray([[4, 3]], dtype=np.int32),
            np.asarray([0], dtype=np.uint16),
            np.asarray([[0, 1]], dtype=np.int8),
            np.asarray([0], dtype=np.uint8),
            np.asarray([[-1, -1]], dtype=np.int64),
            tokenizer,
            8,
        )
        self.assertEqual(dataset.prompt_ids, [[2, 1, 3]])

    def test_cache_without_or_with_changed_metadata_is_rejected(self) -> None:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, generated_ids=np.zeros((1, 1), dtype=np.int32))
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as cache:
            with self.assertRaisesRegex(ValueError, "no reproducibility metadata"):
                validate_trajectory_cache_metadata(cache, {"seed": 1})

        buffer = io.BytesIO()
        np.savez_compressed(buffer, metadata_json=np.asarray('{"seed":1}'))
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as cache:
            validate_trajectory_cache_metadata(cache, {"seed": 1})
            with self.assertRaisesRegex(ValueError, "changed fields"):
                validate_trajectory_cache_metadata(cache, {"seed": 2})

    def test_partition_identity_rejects_same_size_wrong_tokenizer(self) -> None:
        tokenizer = FakeTokenizer()
        with self.assertRaisesRegex(ValueError, "tokenizer_vocab_sha256"):
            validate_tokenizer_identity(
                {
                    "tokenizer_vocab_size": 5,
                    "tokenizer_vocab_sha256": "wrong",
                },
                tokenizer,
                context="test partition",
            )

    def test_partition_validates_shapes_and_token_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "semantic_group_ids"):
            Partition(
                leaf_ids=np.asarray([0, 1], dtype=np.int16),
                semantic_group_ids=np.asarray([0], dtype=np.int32),
                boundary_ids=np.asarray([], dtype=np.int64),
                banned_ids=np.asarray([], dtype=np.int64),
                num_leaves=2,
            )

    def test_partition_builder_bans_special_and_padding_rows(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.all_special_ids = [0, 2]
        banned, special, padded = derive_banned_ids(
            tokenizer, 8, np.asarray([1], dtype=np.int64)
        )
        np.testing.assert_array_equal(special, [0, 2])
        np.testing.assert_array_equal(padded, [5, 6, 7])
        np.testing.assert_array_equal(banned, [0, 1, 2, 5, 6, 7])

    def test_partition_logits_select_last_attended_token(self) -> None:
        class PaddingTokenizer:
            def __call__(self, _batch, **_):
                return {
                    "input_ids": torch.tensor([[4, 5, 0], [0, 6, 7]]),
                    "attention_mask": torch.tensor([[1, 1, 0], [0, 1, 1]]),
                }

        class PositionModel:
            def __call__(self, input_ids, **_):
                positions = torch.arange(input_ids.shape[1], dtype=torch.float32)
                logits = positions.view(1, -1, 1).repeat(input_ids.shape[0], 1, 2)
                return SimpleNamespace(logits=logits)

        logits = collect_last_logits(
            PositionModel(),
            PaddingTokenizer(),
            ["right padded", "left padded"],
            batch_size=2,
            max_length=8,
            device=torch.device("cpu"),
        )
        np.testing.assert_array_equal(logits[:, 0], [1.0, 2.0])

    def test_metric_aggregation_is_position_weighted(self) -> None:
        rows = [
            {"loss": 1.0, "metric_weight": 1.0},
            {"loss": 3.0, "metric_weight": 3.0},
        ]
        self.assertAlmostEqual(weighted_metrics(rows)["loss"], 2.5)

    def test_adapter_loader_rejects_incomplete_checkpoint(self) -> None:
        config = OPTConfig(
            vocab_size=16,
            hidden_size=8,
            ffn_dim=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=16,
            word_embed_proj_dim=8,
            pad_token_id=1,
            bos_token_id=2,
            eos_token_id=2,
        )
        model = TargetConditionedLoRA(
            OPTForCausalLM(config), rank=2, alpha=4.0, dropout=0.0
        )
        state = model.adapter_state_dict()
        state.pop(next(iter(state)))
        with patch("run_experiment.torch.load", return_value=state):
            with self.assertRaisesRegex(RuntimeError, "do not match"):
                load_adapter(model, Path("adapter.pt"))


if __name__ == "__main__":
    unittest.main()
