"""Paired PPL analysis for composition-pool evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reproducibility import sha256_file


DEFAULT_RULES = (
    "ecc",
    "binary",
    "cw_abcd",
    "unigram",
    "atomic_kgw",
    "atomic_kgw25",
)


def relative_ppl_reduction(fixed_nll: np.ndarray, adapted_nll: np.ndarray) -> np.ndarray:
    fixed_ppl = np.exp(fixed_nll)
    adapted_ppl = np.exp(adapted_nll)
    return 100.0 * (fixed_ppl - adapted_ppl) / fixed_ppl


def analyze(
    result_dir: Path,
    rules: tuple[str, ...],
    resamples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(result_dir / "evaluation_summary.csv")
    details = pd.read_csv(result_dir / "evaluation_details.csv")
    rng = np.random.default_rng(seed)

    sequence_ids = sorted(
        details.loc[
            (details["target_rule"] == rules[0])
            & (details["mode"] == "fixed_delta20"),
            "sequence_index",
        ].unique()
    )
    sample_indices = rng.integers(
        0, len(sequence_ids), size=(resamples, len(sequence_ids))
    )

    rows = []
    bootstrap_reductions = []
    for rule in rules:
        rule_details = details.loc[
            (details["target_rule"] == rule)
            & details["mode"].isin(("fixed_delta20", "lora_residual"))
        ]
        paired = rule_details.pivot(
            index="sequence_index", columns="mode", values="mean_base_nll"
        ).reindex(sequence_ids)
        if paired.isna().any().any():
            raise ValueError(f"incomplete fixed/adapted pairing for {rule}")

        fixed_boot = paired["fixed_delta20"].to_numpy()[sample_indices].mean(axis=1)
        adapted_boot = paired["lora_residual"].to_numpy()[sample_indices].mean(axis=1)
        reduction_boot = relative_ppl_reduction(fixed_boot, adapted_boot)
        bootstrap_reductions.append(reduction_boot)

        fixed = summary.loc[
            (summary["target_rule"] == rule)
            & (summary["mode"] == "fixed_delta20")
        ].iloc[0]
        adapted = summary.loc[
            (summary["target_rule"] == rule)
            & (summary["mode"] == "lora_residual")
        ].iloc[0]
        reduction = 100.0 * (
            fixed["base_ppl_from_corpus_nll"]
            - adapted["base_ppl_from_corpus_nll"]
        ) / fixed["base_ppl_from_corpus_nll"]
        low, high = np.quantile(reduction_boot, (0.025, 0.975))
        rows.append(
            {
                "target_rule": rule,
                "fixed_ppl": fixed["base_ppl_from_corpus_nll"],
                "lora_controller_ppl": adapted["base_ppl_from_corpus_nll"],
                "relative_ppl_reduction_pct": reduction,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "token_adherence": adapted["token_adherence"],
                "exact_block_rate": adapted["exact_block_rate"],
                "mean_residual_bias": adapted["mean_residual_bias"],
            }
        )

    result = pd.DataFrame(rows)
    macro_boot = np.stack(bootstrap_reductions).mean(axis=0)
    macro_low, macro_high = np.quantile(macro_boot, (0.025, 0.975))
    macro = pd.DataFrame(
        [
            {
                "rules": len(rules),
                "mean_per_rule_ppl_reduction_pct": result[
                    "relative_ppl_reduction_pct"
                ].mean(),
                "bootstrap_ci_low": macro_low,
                "bootstrap_ci_high": macro_high,
                "minimum_token_adherence": result["token_adherence"].min(),
                "mean_token_adherence": result["token_adherence"].mean(),
            }
        ]
    )
    return result, macro


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-rules", type=Path, required=True)
    parser.add_argument("--output-macro", type=Path, required=True)
    parser.add_argument("--rules", default=",".join(DEFAULT_RULES))
    parser.add_argument("--resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()

    rules = tuple(value.strip() for value in args.rules.split(",") if value.strip())
    result, macro = analyze(args.result_dir, rules, args.resamples, args.seed)
    args.output_rules.parent.mkdir(parents=True, exist_ok=True)
    args.output_macro.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_rules, index=False)
    macro.to_csv(args.output_macro, index=False)
    metadata = {
        "rules": list(rules),
        "bootstrap_resamples": args.resamples,
        "seed": args.seed,
        "inputs": {
            name: sha256_file(args.result_dir / name)
            for name in ("evaluation_summary.csv", "evaluation_details.csv")
        },
    }
    args.output_rules.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))
    print("\nMacro result:")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
