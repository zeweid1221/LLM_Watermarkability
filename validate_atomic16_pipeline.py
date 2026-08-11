from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the atomic-16 OPT pipeline.")
    parser.add_argument("--partition-dir", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--expected-rules", default="ecc")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    partition_dir = Path(args.partition_dir)
    experiment_dir = Path(args.experiment_dir)

    leaf_ids = np.load(partition_dir / "leaf_ids.npy", allow_pickle=False)
    boundary_ids = np.load(partition_dir / "boundary_ids.npy", allow_pickle=False)
    config = json.loads((partition_dir / "hierarchy_config.json").read_text())
    compositions = json.loads(
        (partition_dir / "watermark_compositions.json").read_text()
    )
    report = json.loads((partition_dir / "atomic16_report.json").read_text())

    counts = np.bincount(leaf_ids[leaf_ids >= 0], minlength=16)
    ecc = compositions["ecc"]
    groups = [set(ecc[name]) for name in ("payload_0", "payload_1", "boundary")]
    assert int(config["num_leaves"]) == 16
    assert int(report["calibration_contexts"]) == 5000
    assert int(report["validation_contexts"]) == 500
    assert int(counts.max() - counts.min()) <= 1
    assert [len(group) for group in groups] == [7, 7, 2]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert set.union(*groups) == set(range(16))
    expected_boundary = np.flatnonzero(np.isin(leaf_ids, list(groups[2])))
    np.testing.assert_array_equal(boundary_ids, expected_boundary)

    summary = pd.read_csv(experiment_dir / "evaluation_summary.csv")
    details = pd.read_csv(experiment_dir / "evaluation_details.csv")
    history = pd.read_csv(experiment_dir / "training_history.csv")
    run_config = json.loads((experiment_dir / "run_config.json").read_text())
    expected_rules = {
        value.strip() for value in args.expected_rules.split(",") if value.strip()
    }
    expected_modes = {
        "unwatermarked",
        "fixed_delta20",
        "closed_form",
        "lora_only",
        "lora_residual",
    }
    assert set(summary["target_rule"]) == expected_rules
    assert set(summary["mode"]) == expected_modes
    assert len(summary) == 5 * len(expected_rules)
    assert len(details) == 256 * 5 * len(expected_rules)
    assert len(history) == 6
    assert np.isfinite(summary["base_ppl_from_corpus_nll"]).all()
    assert "active_token_rate" in summary.columns
    if "entropy_kgw" in expected_rules:
        entropy_rows = summary[summary["target_rule"] == "entropy_kgw"]
        assert (entropy_rows["active_token_rate"] > 0.0).all()
        assert (entropy_rows["active_token_rate"] <= 1.0).all()
    resolved_eval_rules = run_config.get(
        "resolved_eval_rules", str(run_config.get("eval_rules", "ecc")).split(",")
    )
    assert set(resolved_eval_rules) == expected_rules
    if (
        "effective_train_sequences" in run_config
        and float(run_config.get("kgw_train_fraction", 0.0)) == 0.0
    ):
        assert set(run_config["resolved_train_rules"]) == expected_rules
        assert int(run_config["effective_train_sequences"]) == (
            int(run_config["train_count"]) * len(expected_rules)
        )
        assert int(run_config["effective_validation_sequences"]) == (
            int(run_config["validation_count"]) * len(expected_rules)
        )

    print("ATOMIC16 PARTITION AND MULTIRULE EXPERIMENT PASSED")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
