from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = {
    "current_residual_0p02": ROOT / "outputs/opt125m_lora_20k_current_residual_0p02_seed20260810",
    "strong_residual_0p10": ROOT / "outputs/opt125m_lora_20k_strong_residual_0p10_seed20260810",
    "direct_target_0p50": ROOT / "outputs/opt125m_lora_20k_direct_target_0p50_seed20260810",
}


def main() -> None:
    rows = []
    for objective, run_dir in RUNS.items():
        summary_path = run_dir / "evaluation_summary.csv"
        history_path = run_dir / "training_history.csv"
        config_path = run_dir / "run_config.json"
        if not summary_path.exists() or not history_path.exists() or not config_path.exists():
            raise FileNotFoundError(f"Incomplete run: {run_dir}")
        with summary_path.open(newline="", encoding="utf-8") as handle:
            summaries = {row["mode"]: row for row in csv.DictReader(handle)}
        with history_path.open(newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        best = min(history, key=lambda row: float(row["validation_loss"]))
        for mode in ("lora_only", "lora_residual"):
            summary = summaries[mode]
            rows.append(
                {
                    "objective": objective,
                    "mode": mode,
                    "residual_loss_weight": config["residual_loss_weight"],
                    "target_mass_loss_weight": config["target_mass_loss_weight"],
                    "best_epoch": best["epoch"],
                    "validation_native_target_mass": best["validation_native_target_mass"],
                    "validation_mean_residual_bias": best["validation_mean_residual_bias"],
                    "token_adherence": summary["token_adherence"],
                    "exact_block_rate": summary["exact_block_rate"],
                    "base_ppl_from_corpus_nll": summary["base_ppl_from_corpus_nll"],
                    "mean_residual_bias": summary["mean_residual_bias"],
                }
            )
    output = ROOT / "outputs/opt125m_lora_20k_adherence_ablation_comparison.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    for row in rows:
        print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
