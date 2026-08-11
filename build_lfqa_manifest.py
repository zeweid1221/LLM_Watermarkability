from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset

from build_manifest import acceptable, clean, digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic LFQA-only manifest.")
    parser.add_argument("--output", default="data/lfqa_prompts_5756.jsonl")
    parser.add_argument("--train-count", type=int, default=5000)
    parser.add_argument("--validation-count", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--arrow-files",
        nargs="*",
        default=None,
        help="Optional cached lfqa-train Arrow shards; auto-discovered when omitted.",
    )
    return parser.parse_args()


def discover_train_arrows() -> list[Path]:
    cache_root = Path(
        os.environ.get("HF_DATASETS_CACHE", Path.home() / ".cache" / "huggingface" / "datasets")
    )
    return sorted(cache_root.glob("vblagoje___lfqa/default/*/*/lfqa-train*.arrow"))


def load_lfqa_train(arrow_files: list[str] | None):
    files = [Path(value) for value in arrow_files] if arrow_files else discover_train_arrows()
    if files and all(path.is_file() for path in files):
        return concatenate_datasets([Dataset.from_file(str(path)) for path in files])
    return load_dataset("vblagoje/lfqa", split="train", streaming=False)


def main() -> None:
    args = parse_args()
    required = args.train_count + args.validation_count + args.test_count
    dataset = load_lfqa_train(args.arrow_files)
    pool: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in dataset:
        title = clean(row.get("title"))
        context = clean(row.get("selftext"))
        prompt = title if not context else f"{title} Context: {context}"
        key = digest(prompt)
        if key in seen or not acceptable(prompt):
            continue
        seen.add(key)
        pool.append(
            {
                "source": "lfqa",
                "source_id": str(row.get("q_id") or key),
                "prompt": prompt,
            }
        )
        if len(pool) == required:
            break
    if len(pool) != required:
        raise RuntimeError(f"Collected {len(pool)}/{required} acceptable LFQA prompts.")

    random.Random(args.seed).shuffle(pool)
    split_counts = {
        "train": args.train_count,
        "validation": args.validation_count,
        "test": args.test_count,
    }
    rows: list[dict[str, object]] = []
    offset = 0
    for split, count in split_counts.items():
        for row in pool[offset : offset + count]:
            rows.append({"split": split, **row})
        offset += count

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            handle.write(json.dumps({"manifest_index": index, **row}, ensure_ascii=True) + "\n")
    report = {
        "output": str(output),
        "dataset": "vblagoje/lfqa",
        "seed": args.seed,
        "split_counts": split_counts,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
