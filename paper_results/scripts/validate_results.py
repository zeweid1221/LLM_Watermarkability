from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / "data" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def pearson(xs: list[float], ys: list[float]) -> float:
    x_bar, y_bar = mean(xs), mean(ys)
    numerator = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_bar) ** 2 for x in xs) * sum((y - y_bar) ** 2 for y in ys)
    )
    return numerator / denominator


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    for rank, index in enumerate(order, start=1):
        result[index] = float(rank)
    return result


def by_mode(rows: list[dict[str, str]], mode: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["mode"] == mode]
    assert len(selected) == 7
    return selected


def check_model(
    filename: str,
    expected_full_reduction: float,
    expected_full_bias: float,
    expected_primary_reduction: float,
    expected_primary_bias: float,
    minimum_adherence: float = 0.9979,
) -> None:
    rows = read_csv(filename)
    assert len(rows) == 35
    fixed = {row["target_rule"]: row for row in by_mode(rows, "fixed_delta20")}
    final = {row["target_rule"]: row for row in by_mode(rows, "lora_residual")}
    controller = by_mode(rows, "closed_form")

    reductions = []
    for rule in fixed:
        fixed_ppl = float(fixed[rule]["base_ppl_from_corpus_nll"])
        final_ppl = float(final[rule]["base_ppl_from_corpus_nll"])
        reductions.append((fixed_ppl - final_ppl) / fixed_ppl)
    assert math.isclose(mean(reductions), expected_full_reduction, abs_tol=5e-5)
    assert (
        min(float(row["token_adherence"]) for row in final.values())
        >= minimum_adherence
    )
    controller_bias = mean([float(row["mean_residual_bias"]) for row in controller])
    final_bias = mean([float(row["mean_residual_bias"]) for row in final.values()])
    assert math.isclose(1.0 - final_bias / controller_bias, expected_full_bias, abs_tol=5e-4)

    # The main paper reports six primary schedules; the entropy-gated auxiliary
    # schedule remains in the complete appendix tables and source archive.
    primary_rules = set(fixed) - {"entropy_kgw"}
    assert len(primary_rules) == 6
    primary_reductions = []
    for rule in primary_rules:
        fixed_ppl = float(fixed[rule]["base_ppl_from_corpus_nll"])
        final_ppl = float(final[rule]["base_ppl_from_corpus_nll"])
        primary_reductions.append((fixed_ppl - final_ppl) / fixed_ppl)
    assert math.isclose(
        mean(primary_reductions), expected_primary_reduction, abs_tol=5e-5
    )
    primary_controller_bias = mean(
        [
            float(row["mean_residual_bias"])
            for row in controller
            if row["target_rule"] in primary_rules
        ]
    )
    primary_final_bias = mean(
        [float(final[rule]["mean_residual_bias"]) for rule in primary_rules]
    )
    assert math.isclose(
        1.0 - primary_final_bias / primary_controller_bias,
        expected_primary_bias,
        abs_tol=5e-4,
    )


def check_compositions() -> None:
    rows = read_csv("composition_coverage.csv")
    assert len(rows) == 4
    for row in rows:
        observed = int(row["observed_unions"])
        possible = int(row["possible_unions"])
        assert math.isclose(float(row["coverage"]), observed / possible, abs_tol=1e-12)


def check_teacher_diagnostic() -> None:
    rows = read_csv("teacher_forced_ecc_diagnostic.csv")
    x = [float(row["intervention_kl"]) for row in rows]
    y = [float(row["log_ppl_increase"]) for row in rows]
    assert math.isclose(pearson(x, y), 0.987547, abs_tol=1e-5)
    assert math.isclose(pearson(ranks(x), ranks(y)), 1.0, abs_tol=1e-5)
    assert all(math.isclose(float(row["boundary_step_fraction"]), 0.125) for row in rows)
    assert all(math.isclose(float(row["target_mass_tau"]), 0.99) for row in rows)

    pooled = {row["model"]: row for row in read_csv(
        "teacher_forced_ecc_watermarkability_pooled.csv"
    )}
    assert math.isclose(
        float(pooled["opt-125m"]["empirical_watermarkability_at_tau"]),
        1.376207,
        abs_tol=1e-6,
    )
    assert math.isclose(
        float(pooled["qwen3-8b"]["empirical_watermarkability_at_tau"]),
        3.464128,
        abs_tol=1e-6,
    )


def check_cross_rule_diagnostic() -> None:
    rows = read_csv("cross_rule_watermarkability_summary.csv")
    assert len(rows) == 12
    expected_models = {"facebook/opt-125m", "Qwen/Qwen3-8B"}
    expected_rules = {
        "ecc",
        "binary",
        "cw_abcd",
        "unigram",
        "atomic_kgw",
        "atomic_kgw25",
    }
    assert {row["model"] for row in rows} == expected_models
    assert {row["target_rule"] for row in rows} == expected_rules
    assert all(int(row["sequences"]) == 256 for row in rows)
    assert all(int(row["positions_per_sequence"]) == 96 for row in rows)

    correlations: dict[str, tuple[float, float]] = {}
    for model in expected_models:
        selected = [row for row in rows if row["model"] == model]
        frontier = [
            float(row["empirical_watermarkability_at_tau"]) for row in selected
        ]
        ppl_cost = [float(row["reference_log_ppl_increase"]) for row in selected]
        correlations[model] = (
            pearson(frontier, ppl_cost),
            pearson(ranks(frontier), ranks(ppl_cost)),
        )
    frontier = [float(row["empirical_watermarkability_at_tau"]) for row in rows]
    ppl_cost = [float(row["reference_log_ppl_increase"]) for row in rows]
    correlations["pooled"] = (
        pearson(frontier, ppl_cost),
        pearson(ranks(frontier), ranks(ppl_cost)),
    )
    expected = {
        "facebook/opt-125m": (0.741077, 0.600000),
        "Qwen/Qwen3-8B": (0.954550, 0.942857),
        "pooled": (0.928631, 0.930070),
    }
    for model, target in expected.items():
        for observed, value in zip(correlations[model], target):
            assert math.isclose(observed, value, abs_tol=1e-6)

    ppl_ratios = {
        model: [
            math.exp(float(row["reference_log_ppl_increase"]))
            for row in rows
            if row["model"] == model
        ]
        for model in expected_models
    }
    assert math.isclose(
        min(ppl_ratios["facebook/opt-125m"]), 0.913693485, abs_tol=1e-8
    )
    assert math.isclose(
        max(ppl_ratios["facebook/opt-125m"]), 3.683712419, abs_tol=1e-8
    )
    assert math.isclose(
        min(ppl_ratios["Qwen/Qwen3-8B"]), 3.575367887, abs_tol=1e-8
    )
    assert math.isclose(
        max(ppl_ratios["Qwen/Qwen3-8B"]), 13.174520829, abs_tol=1e-8
    )

def check_qwen_composition_uncertainty() -> None:
    paired = read_csv("qwen3_composition_pool_corrected_paired_ppl.csv")
    expected = {
        "ecc": (-12.690277, -18.461411, -7.322204),
        "binary": (10.139783, 2.158042, 17.757861),
        "cw_abcd": (7.961975, -6.242443, 20.326203),
        "unigram": (26.649208, 16.608842, 35.502961),
        "atomic_kgw": (4.170439, -6.093764, 13.389598),
        "atomic_kgw25": (-5.226091, -17.518562, 5.713941),
    }
    assert len(paired) == len(expected)
    for row in paired:
        rule = row["target_rule"]
        assert rule in expected
        actual = (
            float(row["relative_ppl_reduction_pct"]),
            float(row["bootstrap_ci_low"]),
            float(row["bootstrap_ci_high"]),
        )
        for observed, target in zip(actual, expected[rule]):
            assert math.isclose(observed, target, abs_tol=1e-6)

    macro = read_csv("qwen3_composition_pool_corrected_macro.csv")
    assert len(macro) == 1
    row = macro[0]
    assert int(row["rules"]) == 6
    for column, target in (
        ("mean_per_rule_ppl_reduction_pct", 5.167506),
        ("bootstrap_ci_low", 0.582826),
        ("bootstrap_ci_high", 9.322580),
        ("minimum_token_adherence", 0.997965),
    ):
        assert math.isclose(float(row[column]), target, abs_tol=1e-6)


if __name__ == "__main__":
    check_model(
        "opt_composition_pool_corrected_evaluation_summary.csv",
        0.460592,
        0.222231,
        0.462794,
        0.237460,
        minimum_adherence=0.9979,
    )
    check_model(
        "qwen3_composition_pool_corrected_evaluation_summary.csv",
        0.030435,
        0.169989,
        0.051675,
        0.192305,
        minimum_adherence=0.9967,
    )
    check_model(
        "opt_schedule_aware_corrected_evaluation_summary.csv",
        0.475367,
        0.319061,
        0.484668,
        0.342642,
        minimum_adherence=0.9976,
    )
    check_model(
        "qwen3_schedule_aware_legacy_evaluation_summary.csv",
        0.0593,
        0.2283,
        0.08235,
        0.25403,
    )
    check_compositions()
    check_teacher_diagnostic()
    check_cross_rule_diagnostic()
    check_qwen_composition_uncertainty()
    print("ALL RELEASED RESULTS PASSED")
