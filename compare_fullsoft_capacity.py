import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
RUNS = (
    ("rank8_epochs3", 8, 3),
    ("rank8_epochs6", 8, 6),
    ("rank16_epochs3", 16, 3),
    ("rank16_epochs6", 16, 6),
)


def row_for(summary: pd.DataFrame, mode: str) -> pd.Series:
    rows = summary.loc[summary["mode"] == mode]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one {mode!r} row, found {len(rows)}")
    return rows.iloc[0]


records = []
for label, rank, epochs in RUNS:
    run_dir = OUTPUTS / f"opt125m_lora_20k_fullsoft_{label}_seed20260810"
    summary = pd.read_csv(run_dir / "evaluation_summary.csv")
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    if int(config["top_k"]) != 0:
        raise RuntimeError(f"{label} is not a full-softmax run")
    if int(config["train_count"]) != 20000:
        raise RuntimeError(f"{label} does not use 20k training samples")
    if int(config["lora_rank"]) != rank or int(config["epochs"]) != epochs:
        raise RuntimeError(f"Unexpected rank/epoch configuration for {label}")

    closed = row_for(summary, "closed_form")
    residual = row_for(summary, "lora_residual")
    lora_only = row_for(summary, "lora_only")
    fixed = row_for(summary, "fixed_delta20")

    closed_ppl = float(closed["base_ppl_from_corpus_nll"])
    residual_ppl = float(residual["base_ppl_from_corpus_nll"])
    fixed_ppl = float(fixed["base_ppl_from_corpus_nll"])
    records.append(
        {
            "label": label,
            "rank": rank,
            "epochs": epochs,
            "trainable_parameters": int(config["trainable_parameters"]),
            "lora_only_adherence": float(lora_only["token_adherence"]),
            "lora_residual_adherence": float(residual["token_adherence"]),
            "lora_residual_exact_block_rate": float(residual["exact_block_rate"]),
            "mean_residual_bias": float(residual["mean_residual_bias"]),
            "closed_form_ppl": closed_ppl,
            "fixed_delta20_ppl": fixed_ppl,
            "lora_residual_ppl": residual_ppl,
            "ppl_reduction_vs_closed_pct": 100.0
            * (closed_ppl - residual_ppl)
            / closed_ppl,
            "ppl_reduction_vs_fixed_pct": 100.0
            * (fixed_ppl - residual_ppl)
            / fixed_ppl,
        }
    )

comparison = pd.DataFrame(records).sort_values(["rank", "epochs"])
csv_path = OUTPUTS / "opt125m_lora_20k_fullsoft_capacity_comparison.csv"
json_path = OUTPUTS / "opt125m_lora_20k_fullsoft_capacity_comparison.json"
comparison.to_csv(csv_path, index=False)
json_path.write_text(
    json.dumps(comparison.to_dict(orient="records"), indent=2),
    encoding="utf-8",
)

print(comparison.to_string(index=False))
print(f"\nSaved: {csv_path}")
print(f"Saved: {json_path}")
