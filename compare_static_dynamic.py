from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = {
    "static_only": ROOT / "outputs/opt125m_lora_20k_static_only_topk_seed20260810",
    "mixed_kgw25": ROOT / "outputs/opt125m_lora_20k_mixed_kgw25_topk_seed20260810",
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: str) -> float | None:
    return None if value in {"", "None"} else float(value)


def main() -> None:
    report: dict[str, object] = {}
    for label, run_dir in RUNS.items():
        summary_path = run_dir / "evaluation_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        rows = load_rows(summary_path)
        if len(rows) != 10:
            raise RuntimeError(f"Expected 10 summary rows in {summary_path}; found {len(rows)}")
        indexed = {(row["target_rule"], row["mode"]): row for row in rows}
        selected: dict[str, object] = {}
        for rule in ("ecc", "kgw"):
            for mode in ("lora_only", "lora_residual"):
                row = indexed[(rule, mode)]
                selected[f"{rule}_{mode}"] = {
                    "token_adherence": as_float(row["token_adherence"]),
                    "exact_block_rate": as_float(row["exact_block_rate"]),
                    "mean_kgw_z_score": as_float(row["mean_kgw_z_score"]),
                    "base_ppl": as_float(row["base_ppl_from_corpus_nll"]),
                    "mean_residual_bias": as_float(row["mean_residual_bias"]),
                }
        report[label] = selected

    static = report["static_only"]
    mixed = report["mixed_kgw25"]
    report["mixed_minus_static"] = {
        "ecc_lora_only_adherence": (
            mixed["ecc_lora_only"]["token_adherence"]
            - static["ecc_lora_only"]["token_adherence"]
        ),
        "ecc_lora_residual_exact_block_rate": (
            mixed["ecc_lora_residual"]["exact_block_rate"]
            - static["ecc_lora_residual"]["exact_block_rate"]
        ),
        "kgw_lora_only_adherence": (
            mixed["kgw_lora_only"]["token_adherence"]
            - static["kgw_lora_only"]["token_adherence"]
        ),
        "kgw_lora_residual_bias": (
            mixed["kgw_lora_residual"]["mean_residual_bias"]
            - static["kgw_lora_residual"]["mean_residual_bias"]
        ),
    }
    output = ROOT / "outputs/opt125m_lora_20k_static_dynamic_comparison.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
