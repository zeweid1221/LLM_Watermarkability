"""Validate and combine cross-rule watermarkability result directories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_cross_rule_watermarkability import DEFAULT_RULES, fixed_bias_statistics
from analyze_teacher_forced_watermarkability import minimum_intervention


def validate_directory(
    path: Path,
    *,
    expected_rules: tuple[str, ...] = DEFAULT_RULES,
    require_reference_match: bool = True,
) -> pd.DataFrame:
    required = (
        "cross_rule_summary.csv",
        "cross_rule_sequence_summary.csv",
        "cross_rule_steps.npz",
        "shared_trajectories.csv",
        "run_metadata.json",
    )
    for name in required:
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)

    summary = pd.read_csv(path / "cross_rule_summary.csv")
    sequences = pd.read_csv(path / "cross_rule_sequence_summary.csv")
    metadata = json.loads((path / "run_metadata.json").read_text(encoding="utf-8"))
    steps = np.load(path / "cross_rule_steps.npz", allow_pickle=False)
    rules = steps["rule_names"].astype(str).tolist()
    if rules != list(expected_rules):
        raise ValueError(f"Unexpected rule order: {rules}")
    if metadata["protocol_version"] != "cross_rule_shared_trajectory_v1":
        raise ValueError("Unexpected protocol version")
    if require_reference_match and metadata["reference_text_exact_match"] is not True:
        raise ValueError("Formal unwatermarked trajectory match was not established")

    expected_shape = (
        len(rules),
        int(metadata["test_count"]),
        int(metadata["generated_tokens"]),
    )
    for name in (
        "native_target_mass",
        "required_bias_at_tau",
        "watermarkability_at_tau",
        "fixed_controlled_mass",
        "fixed_bias_kl",
        "target_vocab_fraction",
        "observed_membership",
    ):
        if steps[name].shape != expected_shape:
            raise ValueError(
                f"{name} has shape {steps[name].shape}, expected {expected_shape}"
            )
    if steps["generated_token_ids"].shape != expected_shape[1:]:
        raise ValueError("Generated-token shape mismatch")
    if steps["target_codes"].shape != expected_shape:
        raise ValueError("Target-code shape mismatch")
    if steps["key_indices"].shape != expected_shape[:2]:
        raise ValueError("Key-index shape mismatch")
    generated_hash = hashlib.sha256(
        steps["generated_token_ids"].tobytes()
    ).hexdigest()
    if generated_hash != metadata["generated_token_ids_sha256"]:
        raise ValueError("Generated-token hash mismatch")
    trajectories = pd.read_csv(path / "shared_trajectories.csv")
    if len(trajectories) != expected_shape[1]:
        raise ValueError("Shared-trajectory row count mismatch")
    if not np.array_equal(
        trajectories["prompt_sha256"].astype(str).to_numpy(),
        steps["prompt_sha256"].astype(str),
    ):
        raise ValueError("Prompt hashes differ between trajectory artifacts")
    mass = steps["native_target_mass"].astype(np.float64)
    numerical_tolerance = 1e-5
    if not np.isfinite(mass).all() or np.any(
        (mass < -numerical_tolerance) | (mass > 1.0 + numerical_tolerance)
    ):
        raise ValueError("Native target mass contains invalid values")
    bias, frontier = minimum_intervention(mass, float(metadata["target_mass"]))
    controlled, fixed_kl = fixed_bias_statistics(mass, float(metadata["fixed_bias"]))
    np.testing.assert_allclose(
        steps["required_bias_at_tau"], bias, rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        steps["watermarkability_at_tau"], frontier, rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        steps["fixed_controlled_mass"], controlled, rtol=2e-5, atol=2e-5
    )
    np.testing.assert_allclose(
        steps["fixed_bias_kl"], fixed_kl, rtol=2e-5, atol=2e-5
    )
    if len(summary) != len(rules):
        raise ValueError("Summary rule count mismatch")
    if len(sequences) != len(rules) * expected_shape[1]:
        raise ValueError("Sequence summary row count mismatch")
    if summary["target_rule"].tolist() != rules:
        raise ValueError("Summary rule order mismatch")
    aggregate_fields = {
        "mean_native_target_mass": "native_target_mass",
        "mean_required_bias_at_tau": "required_bias_at_tau",
        "empirical_watermarkability_at_tau": "watermarkability_at_tau",
        "mean_fixed_bias_kl": "fixed_bias_kl",
        "mean_fixed_controlled_mass": "fixed_controlled_mass",
        "mean_target_vocab_fraction": "target_vocab_fraction",
        "unwatermarked_membership_rate": "observed_membership",
    }
    for rule_index, row in summary.iterrows():
        for column, array_name in aggregate_fields.items():
            expected = float(steps[array_name][rule_index].mean())
            if not np.isclose(float(row[column]), expected, rtol=2e-6, atol=2e-6):
                raise ValueError(
                    f"Summary mismatch for {row['target_rule']}:{column}"
                )
    return summary


def correlation_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    def correlation(left: pd.Series, right: pd.Series) -> float:
        if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
            return float("nan")
        return float(np.corrcoef(left.to_numpy(), right.to_numpy())[0, 1])

    rows: list[dict[str, object]] = []
    groups = [(str(model), group) for model, group in frame.groupby("model")]
    groups.append(("pooled", frame))
    for model, group in groups:
        for predictor in (
            "empirical_watermarkability_at_tau",
            "mean_required_bias_at_tau",
            "mean_fixed_bias_kl",
        ):
            selected = group[[predictor, "reference_log_ppl_increase"]].dropna()
            pearson = correlation(
                selected[predictor], selected["reference_log_ppl_increase"]
            )
            spearman = correlation(
                selected[predictor].rank(method="average"),
                selected["reference_log_ppl_increase"].rank(method="average"),
            )
            rows.append(
                {
                    "model": model,
                    "predictor": predictor,
                    "rule_points": len(selected),
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rules", default=",".join(DEFAULT_RULES))
    parser.add_argument(
        "--require-reference-match",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    expected_rules = tuple(
        value.strip() for value in args.expected_rules.split(",") if value.strip()
    )
    if not expected_rules:
        raise ValueError("--expected-rules must not be empty.")
    frames = [
        validate_directory(
            path.resolve(),
            expected_rules=expected_rules,
            require_reference_match=args.require_reference_match,
        )
        for path in args.result_dir
    ]
    combined = pd.concat(frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "cross_model_rule_summary.csv", index=False)
    correlations = pd.DataFrame(correlation_rows(combined))
    correlations.to_csv(args.output_dir / "cross_rule_correlations.csv", index=False)
    report = {
        "status": "passed",
        "models": sorted(combined["model"].unique().tolist()),
        "rules": list(expected_rules),
        "summary_rows": len(combined),
    }
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(combined.to_string(index=False))
    print("\nCross-rule correlations:")
    print(correlations.to_string(index=False))
    print("\nCROSS-RULE WATERMARKABILITY VALIDATION PASSED")


if __name__ == "__main__":
    main()
