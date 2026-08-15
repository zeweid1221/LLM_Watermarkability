from __future__ import annotations

import json
import unittest
from pathlib import Path

from run_experiment import parse_args
from run_paper_protocol import load_protocol, protocol_commands


ROOT = Path(__file__).resolve().parent
CONFIGS = {
    "opt_composition_pool.json": ("composition_pool", "facebook/opt-125m", 20000),
    "qwen3_composition_pool.json": ("composition_pool", "Qwen/Qwen3-8B", 20000),
    "opt_schedule_aware.json": ("schedule_pool", "facebook/opt-125m", 35000),
    "qwen3_schedule_aware.json": ("schedule_pool", "Qwen/Qwen3-8B", 35000),
}


class ProtocolConfigTests(unittest.TestCase):
    def test_configs_materialize_expected_protocols(self) -> None:
        for filename, (protocol, model, expected_training_rows) in CONFIGS.items():
            with self.subTest(filename=filename):
                path = ROOT / "configs" / filename
                payload = load_protocol(path)
                args = parse_args(["--config", str(path)])
                self.assertEqual(args.training_protocol, protocol)
                self.assertEqual(args.model_name, model)
                rows = (
                    args.train_count * args.composition_repeats
                    if protocol == "composition_pool"
                    else args.train_count * len(args.train_rules.split(","))
                )
                self.assertEqual(rows, expected_training_rows)
                self.assertEqual(args.test_count, 256)
                self.assertEqual(args.epochs, 6)
                self.assertEqual(args.lora_rank, 16)
                self.assertEqual(len(protocol_commands(path, payload)), 3)

    def test_cli_overrides_config_defaults(self) -> None:
        path = ROOT / "configs" / "opt_composition_pool.json"
        args = parse_args(["--config", str(path), "--epochs", "1"])
        self.assertEqual(args.epochs, 1)

    def test_unknown_config_key_is_rejected(self) -> None:
        path = ROOT / "configs" / "opt_composition_pool.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["experiment"]["not_a_real_argument"] = 1
        temp = ROOT / "temp_protocol_config_test.json"
        temp.write_text(json.dumps(payload), encoding="utf-8")
        try:
            with self.assertRaisesRegex(ValueError, "not_a_real_argument"):
                parse_args(["--config", str(temp)])
        finally:
            temp.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
