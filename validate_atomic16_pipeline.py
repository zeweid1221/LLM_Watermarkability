from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise ValueError(message)


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
    require(int(config["num_leaves"]) == 16, "Expected a 16-atom partition.")
    require(
        int(report["calibration_contexts"]) == 5000,
        "Expected 5,000 calibration contexts.",
    )
    require(
        int(report["validation_contexts"]) == 500,
        "Expected 500 validation contexts.",
    )
    require(int(counts.max() - counts.min()) <= 1, "Atom counts are not balanced.")
    require(
        [len(group) for group in groups] == [7, 7, 2],
        "ECC composition must use 7/7/2 atoms.",
    )
    require(
        not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]),
        "ECC atom groups overlap.",
    )
    require(set.union(*groups) == set(range(16)), "ECC atoms do not cover all atoms.")
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
    require(set(summary["target_rule"]) == expected_rules, "Rule set mismatch.")
    require(set(summary["mode"]) == expected_modes, "Evaluation mode mismatch.")
    require(len(summary) == 5 * len(expected_rules), "Summary row count mismatch.")
    require(
        len(details) == 256 * 5 * len(expected_rules),
        "Evaluation detail row count mismatch.",
    )
    require(len(history) == 6, "Training history must contain six epochs.")
    require(
        np.isfinite(summary["base_ppl_from_corpus_nll"]).all(),
        "PPL summary contains nonfinite values.",
    )
    require("active_token_rate" in summary.columns, "Missing active_token_rate.")
    if "entropy_kgw" in expected_rules:
        entropy_rows = summary[summary["target_rule"] == "entropy_kgw"]
        require(
            (entropy_rows["active_token_rate"] > 0.0).all(),
            "Entropy-gated rule has no active positions.",
        )
        require(
            (entropy_rows["active_token_rate"] <= 1.0).all(),
            "Entropy-gated active rate exceeds one.",
        )
    resolved_eval_rules = run_config.get(
        "resolved_eval_rules", str(run_config.get("eval_rules", "ecc")).split(",")
    )
    require(set(resolved_eval_rules) == expected_rules, "Resolved rule set mismatch.")
    if run_config.get("training_protocol") == "composition_pool":
        require(
            run_config["resolved_train_rules"] == [],
            "Composition training must not contain watermark rule names.",
        )
        repeats = int(run_config["composition_repeats"])
        require(
            int(run_config["effective_train_sequences"])
            == int(run_config["train_count"]) * repeats,
            "Effective training sequence count mismatch.",
        )
        require(
            int(run_config["effective_validation_sequences"])
            == int(run_config["validation_count"]) * repeats,
            "Effective validation sequence count mismatch.",
        )
        coverage = json.loads(
            (experiment_dir / "composition_coverage.json").read_text()
        )
        expected_cardinalities = set(run_config["composition_cardinalities"])
        for split in ("train", "validation"):
            rows = coverage[split]
            require(
                {int(row["cardinality"]) for row in rows}
                == expected_cardinalities,
                f"{split} cardinality set mismatch.",
            )
            for row in rows:
                require(
                    int(row["observed_unions"]) == int(row["possible_unions"]),
                    f"{split} does not cover every requested atom union.",
                )
                require(
                    float(row["coverage"]) == 1.0,
                    f"{split} composition coverage is not one.",
                )
    elif (
        "effective_train_sequences" in run_config
        and float(run_config.get("kgw_train_fraction", 0.0)) == 0.0
    ):
        require(
            set(run_config["resolved_train_rules"]) == expected_rules,
            "Schedule-aware train rules do not match evaluation rules.",
        )
        require(
            int(run_config["effective_train_sequences"])
            == int(run_config["train_count"]) * len(expected_rules),
            "Schedule-aware training sequence count mismatch.",
        )
        require(
            int(run_config["effective_validation_sequences"])
            == int(run_config["validation_count"]) * len(expected_rules),
            "Schedule-aware validation sequence count mismatch.",
        )

    print("ATOMIC16 PARTITION AND MULTIRULE EXPERIMENT PASSED")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
