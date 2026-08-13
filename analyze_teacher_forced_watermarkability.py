"""Estimate the watermarkability frontier from teacher-forced step records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reproducibility import sha256_file


def minimum_intervention(
    native_mass: np.ndarray, tau: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the minimum uniform bias and KL needed to reach target mass tau."""
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must lie strictly between zero and one")

    mass = np.asarray(native_mass, dtype=np.float64)
    numerical_tolerance = 1e-5
    if np.any(~np.isfinite(mass)) or np.any(
        (mass < -numerical_tolerance) | (mass > 1.0 + numerical_tolerance)
    ):
        raise ValueError("native target masses must be finite probabilities")

    finfo = np.finfo(np.float64)
    clipped = np.clip(mass, finfo.tiny, 1.0 - finfo.eps)
    logit_tau = np.log(tau) - np.log1p(-tau)
    logit_mass = np.log(clipped) - np.log1p(-clipped)
    required_bias = np.maximum(logit_tau - logit_mass, 0.0)

    # The I-projection changes only the total probability of the requested set.
    kl = np.zeros_like(clipped)
    needs_control = clipped < tau
    controlled_mass = clipped[needs_control]
    kl[needs_control] = (
        tau * np.log(tau / controlled_mass)
        + (1.0 - tau)
        * np.log((1.0 - tau) / (1.0 - controlled_mass))
    )
    return required_bias, kl


def summarize_result_directory(
    model: str, result_dir: Path, tau: float
) -> tuple[pd.DataFrame, dict[str, float | str]]:
    summary_path = result_dir / "bucket_compatibility_summary.csv"
    steps_path = result_dir / "bucket_compatibility_steps.parquet"
    summary = pd.read_csv(summary_path)
    steps = pd.read_parquet(steps_path)

    required = {
        "logit_bias",
        "allowed_mass",
        "compatibility_nll",
        "intervention_kl_nats",
        "step_type",
    }
    missing = required.difference(steps.columns)
    if missing:
        raise ValueError(f"{steps_path} is missing columns: {sorted(missing)}")

    required_bias, frontier_kl = minimum_intervention(
        steps["allowed_mass"].to_numpy(), tau
    )
    steps = steps.assign(
        required_bias_at_tau=required_bias,
        watermarkability_at_tau=frontier_kl,
        native_compatibility_nll=-np.log(
            np.clip(
                steps["allowed_mass"].to_numpy(dtype=np.float64),
                np.finfo(np.float64).tiny,
                1.0,
            )
        ),
    )

    rows: list[dict[str, float | int | str]] = []
    for _, source_row in summary.sort_values("logit_bias").iterrows():
        bias = float(source_row["logit_bias"])
        group = steps.loc[steps["logit_bias"] == bias]
        if len(group) != int(source_row["num_steps"]):
            raise ValueError(f"step-count mismatch for {model}, bias={bias:g}")
        boundary = group["step_type"].eq("boundary")
        rows.append(
            {
                "model": model,
                "logit_bias": int(bias) if bias.is_integer() else bias,
                "target_mass_tau": tau,
                "log_ppl_increase": source_row[
                    "log_ppl_increase_vs_unwatermarked"
                ],
                "intervention_kl": source_row["mean_intervention_kl_nats"],
                "step_spearman": source_row[
                    "step_spearman_compatibility_vs_token_nll"
                ],
                "mean_native_target_mass": group["allowed_mass"].mean(),
                "mean_required_bias_at_tau": group[
                    "required_bias_at_tau"
                ].mean(),
                "empirical_watermarkability_at_tau": group[
                    "watermarkability_at_tau"
                ].mean(),
                "boundary_step_fraction": boundary.mean(),
                "boundary_incompatibility_share": _share(
                    group, boundary, "native_compatibility_nll"
                ),
                "boundary_watermarkability_share": _share(
                    group, boundary, "watermarkability_at_tau"
                ),
            }
        )

    boundary = steps["step_type"].eq("boundary")
    pooled: dict[str, float | str] = {
        "model": model,
        "target_mass_tau": tau,
        "mean_native_target_mass": steps["allowed_mass"].mean(),
        "mean_required_bias_at_tau": steps["required_bias_at_tau"].mean(),
        "empirical_watermarkability_at_tau": steps[
            "watermarkability_at_tau"
        ].mean(),
        "boundary_step_fraction": boundary.mean(),
        "boundary_incompatibility_share": _share(
            steps, boundary, "native_compatibility_nll"
        ),
        "boundary_watermarkability_share": _share(
            steps, boundary, "watermarkability_at_tau"
        ),
    }
    return pd.DataFrame(rows), pooled


def _share(frame: pd.DataFrame, mask: pd.Series, column: str) -> float:
    total = float(frame[column].sum())
    return float(frame.loc[mask, column].sum()) / total if total > 0.0 else 0.0


def parse_result(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("use MODEL=RESULT_DIRECTORY")
    model, path = value.split("=", 1)
    if not model or not path:
        raise argparse.ArgumentTypeError("use MODEL=RESULT_DIRECTORY")
    return model, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        type=parse_result,
        metavar="MODEL=DIR",
    )
    parser.add_argument("--tau", type=float, default=0.99)
    parser.add_argument("--output-detail", type=Path, required=True)
    parser.add_argument("--output-pooled", type=Path, required=True)
    args = parser.parse_args()

    detail_frames = []
    pooled_rows = []
    input_fingerprints: dict[str, dict[str, str]] = {}
    for model, result_dir in args.result:
        detail, pooled = summarize_result_directory(model, result_dir, args.tau)
        detail_frames.append(detail)
        pooled_rows.append(pooled)
        input_fingerprints[model] = {
            "bucket_compatibility_summary.csv": sha256_file(
                result_dir / "bucket_compatibility_summary.csv"
            ),
            "bucket_compatibility_steps.parquet": sha256_file(
                result_dir / "bucket_compatibility_steps.parquet"
            ),
        }

    detail_output = pd.concat(detail_frames, ignore_index=True)
    pooled_output = pd.DataFrame(pooled_rows)
    args.output_detail.parent.mkdir(parents=True, exist_ok=True)
    args.output_pooled.parent.mkdir(parents=True, exist_ok=True)
    detail_output.to_csv(args.output_detail, index=False)
    pooled_output.to_csv(args.output_pooled, index=False)
    metadata_path = args.output_detail.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "target_mass_tau": args.tau,
                "inputs": input_fingerprints,
                "output_detail": str(args.output_detail),
                "output_pooled": str(args.output_pooled),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(detail_output.to_string(index=False))
    print("\nPooled at matched reliability:")
    print(pooled_output.to_string(index=False))


if __name__ == "__main__":
    main()
