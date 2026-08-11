from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Callable, Iterable

from datasets import load_dataset


SOURCE_WEIGHTS = {
    "ultrachat": 0.40,
    "lfqa": 0.30,
    "oasst1": 0.20,
    "fineweb_edu": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic mixed English prompt manifest.")
    parser.add_argument("--output", default="data/mixed_prompts_5756.jsonl")
    parser.add_argument("--train-count", type=int, default=5000)
    parser.add_argument("--validation-count", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--source-weights",
        default=",".join(f"{name}:{weight}" for name, weight in SOURCE_WEIGHTS.items()),
        help="Comma-separated source:weight pairs summing to one.",
    )
    return parser.parse_args()


def parse_source_weights(value: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in value.split(","):
        name, raw_weight = item.split(":", 1)
        name = name.strip()
        if name not in SOURCES:
            raise ValueError(f"Unknown source in --source-weights: {name}")
        weights[name] = float(raw_weight)
    if set(weights) != set(SOURCES):
        raise ValueError("--source-weights must name every configured source.")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("Source weights must be non-negative.")
    if abs(sum(weights.values()) - 1.0) > 1e-8:
        raise ValueError("Source weights must sum to one.")
    return weights


def clean(text: object) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    return value


def acceptable(text: str) -> bool:
    words = text.split()
    if not 12 <= len(words) <= 320:
        return False
    ascii_ratio = sum(ord(char) < 128 for char in text) / max(1, len(text))
    letter_ratio = sum(char.isalpha() for char in text) / max(1, len(text))
    return ascii_ratio >= 0.97 and letter_ratio >= 0.55


def digest(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def ultrachat_rows() -> Iterable[tuple[str, str]]:
    rows = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    for row in rows:
        messages = row.get("messages") or []
        prompt = next(
            (clean(message.get("content")) for message in messages if message.get("role") == "user"),
            "",
        )
        yield str(row.get("prompt_id") or digest(prompt)), prompt


def lfqa_rows() -> Iterable[tuple[str, str]]:
    rows = load_dataset("vblagoje/lfqa", split="train", streaming=True)
    for row in rows:
        title = clean(row.get("title"))
        context = clean(row.get("selftext"))
        prompt = title if not context else f"{title} Context: {context}"
        yield str(row.get("q_id") or digest(prompt)), prompt


def oasst_rows() -> Iterable[tuple[str, str]]:
    rows = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
    for row in rows:
        if row.get("lang") != "en" or row.get("role") != "prompter" or row.get("parent_id"):
            continue
        prompt = clean(row.get("text"))
        yield str(row.get("message_id") or digest(prompt)), prompt


def fineweb_rows() -> Iterable[tuple[str, str]]:
    rows = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    for row in rows:
        text = clean(row.get("text"))
        words = text.split()
        if len(words) < 48:
            continue
        prefix = " ".join(words[: min(180, len(words))])
        prompt = f"Continue the following passage in the same style: {prefix}"
        yield str(row.get("id") or digest(prompt)), prompt


SOURCES: dict[str, Callable[[], Iterable[tuple[str, str]]]] = {
    "ultrachat": ultrachat_rows,
    "lfqa": lfqa_rows,
    "oasst1": oasst_rows,
    "fineweb_edu": fineweb_rows,
}


def allocate(total: int, source_weights: dict[str, float]) -> dict[str, int]:
    counts = {name: int(total * weight) for name, weight in source_weights.items()}
    remainder = total - sum(counts.values())
    for name in source_weights:
        if remainder <= 0:
            break
        counts[name] += 1
        remainder -= 1
    return counts


def main() -> None:
    args = parse_args()
    source_weights = parse_source_weights(args.source_weights)
    split_counts = {
        "train": int(args.train_count),
        "validation": int(args.validation_count),
        "test": int(args.test_count),
    }
    needed_by_source = {name: 0 for name in SOURCES}
    allocations: dict[str, dict[str, int]] = {}
    for split, count in split_counts.items():
        allocations[split] = allocate(count, source_weights)
        for source, amount in allocations[split].items():
            needed_by_source[source] += amount

    seen: set[str] = set()
    pools: dict[str, list[dict[str, object]]] = {}
    for source_name, source_fn in SOURCES.items():
        target = needed_by_source[source_name]
        if target == 0:
            pools[source_name] = []
            continue
        pool: list[dict[str, object]] = []
        for source_id, prompt in source_fn():
            prompt = clean(prompt)
            key = digest(prompt)
            if key in seen or not acceptable(prompt):
                continue
            seen.add(key)
            pool.append({"source": source_name, "source_id": source_id, "prompt": prompt})
            if len(pool) >= target:
                break
        if len(pool) != target:
            raise RuntimeError(f"Collected {len(pool)}/{target} rows from {source_name}.")
        random.Random(args.seed + len(source_name)).shuffle(pool)
        pools[source_name] = pool

    output_rows: list[dict[str, object]] = []
    offsets = {name: 0 for name in SOURCES}
    for split in ("train", "validation", "test"):
        for source, count in allocations[split].items():
            start = offsets[source]
            selected = pools[source][start : start + count]
            offsets[source] += count
            for row in selected:
                output_rows.append({"split": split, **row})
    random.Random(args.seed).shuffle(output_rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(output_rows):
            handle.write(json.dumps({"manifest_index": index, **row}, ensure_ascii=True) + "\n")
    report = {
        "output": str(output),
        "seed": args.seed,
        "split_counts": split_counts,
        "source_weights": source_weights,
        "allocations": allocations,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
