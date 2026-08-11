from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/opt125m_lora_5k_seed20260810")
    args = parser.parse_args()
    output = Path(args.output_dir)
    required = [
        "adapter_best.pt",
        "training_history.csv",
        "evaluation_details.csv",
        "evaluation_summary.csv",
        "run_config.json",
        "RESULTS.json",
    ]
    missing = [name for name in required if not (output / name).exists()]
    if missing:
        raise SystemExit(f"Missing outputs: {missing}")
    summary = pd.read_csv(output / "evaluation_summary.csv")
    details = pd.read_csv(output / "evaluation_details.csv")
    if set(summary["mode"]) != {"unwatermarked", "fixed_delta20", "closed_form", "lora_residual"}:
        raise SystemExit("Evaluation modes are incomplete.")
    numeric = summary.select_dtypes(include="number")
    if not numeric.apply(lambda column: column.map(lambda value: pd.notna(value))).all().all():
        raise SystemExit("Non-finite summary value detected.")
    if len(details) != 4 * 256:
        raise SystemExit(f"Expected 1024 detail rows, found {len(details)}.")
    print(summary.to_string(index=False))
    config = json.loads((output / "run_config.json").read_text())
    print("runtime_seconds", config["runtime_seconds"])
    print("trainable_parameters", config["trainable_parameters"])
    print("ALL WATERMARKABILITY LORA CHECKS PASSED")


if __name__ == "__main__":
    main()
