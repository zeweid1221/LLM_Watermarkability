"""Run a paper protocol from a scheduler-independent JSON configuration."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_protocol(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("experiment"), dict):
        raise ValueError("Protocol config must contain an 'experiment' object.")
    experiment = payload["experiment"]
    for key in ("partition_dir", "output_dir", "eval_rules"):
        if not experiment.get(key):
            raise ValueError(f"Protocol experiment is missing {key!r}.")
    return payload


def protocol_commands(config_path: Path, payload: dict[str, object]) -> list[list[str]]:
    experiment = payload["experiment"]
    output_dir = Path(str(experiment["output_dir"]))
    partition_dir = str(experiment["partition_dir"])
    expected_rules = str(experiment["eval_rules"])

    commands = [
        [sys.executable, "run_experiment.py", "--config", str(config_path)],
        [
            sys.executable,
            "validate_atomic16_pipeline.py",
            "--partition-dir",
            partition_dir,
            "--experiment-dir",
            str(output_dir),
            "--expected-rules",
            expected_rules,
        ],
    ]

    paired = payload.get("paired_analysis")
    if paired:
        if not isinstance(paired, dict):
            raise ValueError("Protocol 'paired_analysis' must be a JSON object.")
        commands.append(
            [
                sys.executable,
                "analyze_composition_pool_paired.py",
                "--result-dir",
                str(output_dir),
                "--output-rules",
                str(output_dir / str(paired["output_rules"])),
                "--output-macro",
                str(output_dir / str(paired["output_macro"])),
                "--rules",
                str(paired["rules"]),
                "--resamples",
                str(paired.get("resamples", 20_000)),
                "--seed",
                str(paired.get("seed", 20260812)),
            ]
        )
    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Permit an existing output directory, for an intentional resume.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    payload = load_protocol(config_path)
    experiment = payload["experiment"]
    output_dir = ROOT / str(experiment["output_dir"])
    if output_dir.exists() and not args.allow_existing_output and not args.dry_run:
        raise FileExistsError(
            f"Refusing to reuse existing output directory: {output_dir}. "
            "Pass --allow-existing-output only for an intentional resume."
        )

    commands = protocol_commands(config_path, payload)
    if args.dry_run:
        for command in commands:
            print(shlex.join(command))
        return

    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
