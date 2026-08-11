from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from build_manifest import SOURCES, acceptable, clean, digest, parse_source_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extend a fixed manifest while preserving its validation and test splits."
    )
    parser.add_argument("--base", default="data/mixed_prompts_5756.jsonl")
    parser.add_argument("--output", default="data/mixed_prompts_22256.jsonl")
    parser.add_argument("--train-count", type=int, default=20000)
    parser.add_argument(
        "--train-source-weights",
        default="ultrachat:0.40,lfqa:0.30,oasst1:0.10,fineweb_edu:0.20",
    )
    parser.add_argument("--seed", type=int, default=20260810)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_signature(rows: list[dict[str, object]], split: str) -> str:
    payload = [
        {
            "source": row["source"],
            "source_id": row["source_id"],
            "prompt": row["prompt"],
        }
        for row in rows
        if row["split"] == split
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def allocate(total: int, weights: dict[str, float]) -> dict[str, int]:
    counts = {name: int(total * weight) for name, weight in weights.items()}
    remainder = total - sum(counts.values())
    for name in weights:
        if remainder <= 0:
            break
        counts[name] += 1
        remainder -= 1
    return counts


def main() -> None:
    args = parse_args()
    base_path = Path(args.base)
    base_rows = load_rows(base_path)
    weights = parse_source_weights(args.train_source_weights)
    current_train = Counter(
        str(row["source"]) for row in base_rows if row["split"] == "train"
    )
    target_train = allocate(args.train_count, weights)
    additions_needed = {
        source: target_train[source] - current_train[source] for source in SOURCES
    }
    if any(count < 0 for count in additions_needed.values()):
        raise ValueError(
            f"Target source counts cannot retain the base training set: {additions_needed}"
        )
    if sum(current_train.values()) + sum(additions_needed.values()) != args.train_count:
        raise RuntimeError("Extended training count is inconsistent.")

    seen = {digest(clean(row["prompt"])) for row in base_rows}
    additions: list[dict[str, object]] = []
    for source_name, source_fn in SOURCES.items():
        target = additions_needed[source_name]
        source_rows: list[dict[str, object]] = []
        for source_id, prompt in source_fn():
            prompt = clean(prompt)
            key = digest(prompt)
            if key in seen or not acceptable(prompt):
                continue
            seen.add(key)
            source_rows.append(
                {
                    "split": "train",
                    "source": source_name,
                    "source_id": source_id,
                    "prompt": prompt,
                }
            )
            if len(source_rows) >= target:
                break
        if len(source_rows) != target:
            raise RuntimeError(
                f"Collected {len(source_rows)}/{target} additional rows from {source_name}."
            )
        random.Random(args.seed + len(source_name)).shuffle(source_rows)
        additions.extend(source_rows)
    random.Random(args.seed).shuffle(additions)

    output_rows = [
        {key: value for key, value in row.items() if key != "manifest_index"}
        for row in base_rows
    ] + additions
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(output_rows):
            handle.write(
                json.dumps({"manifest_index": index, **row}, ensure_ascii=True) + "\n"
            )

    extended_rows = load_rows(output)
    for split in ("validation", "test"):
        if split_signature(base_rows, split) != split_signature(extended_rows, split):
            raise RuntimeError(f"The {split} split changed while extending the manifest.")
    if [row["prompt"] for row in base_rows if row["split"] == "train"] != [
        row["prompt"] for row in extended_rows if row["split"] == "train"
    ][: sum(current_train.values())]:
        raise RuntimeError("The original 5k training prefix was not preserved.")

    report = {
        "output": str(output),
        "base_manifest": str(base_path),
        "base_manifest_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "train_count": args.train_count,
        "base_train_count": sum(current_train.values()),
        "additional_train_count": len(additions),
        "target_train_source_counts": target_train,
        "additional_train_source_counts": additions_needed,
        "validation_signature": split_signature(base_rows, "validation"),
        "test_signature": split_signature(base_rows, "test"),
        "heldout_splits_preserved": True,
    }
    output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
