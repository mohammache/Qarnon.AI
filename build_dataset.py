#!/usr/bin/env python3
"""
Merge all instruction JSONL files → train/val split.
Output:
  dataset/train.jsonl   (90%)
  dataset/val.jsonl     (10%)
  dataset/dataset_stats.json
"""

import json
import random
from pathlib import Path
from collections import Counter

OUT_DIR = Path("dataset")

SOURCES = [
    ("darija_foundation",    OUT_DIR / "darija_foundation.jsonl"),
    ("tatoeba_ary",          OUT_DIR / "tatoeba_ary.jsonl"),
    ("hespress_instructions", OUT_DIR / "hespress_instructions.jsonl"),
]

VAL_RATIO = 0.10
RANDOM_SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, records: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    random.seed(RANDOM_SEED)

    all_examples: list[dict] = []
    source_counts: dict[str, int] = {}

    for name, path in SOURCES:
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        data = load_jsonl(path)
        # Tag each example with source for stats
        for ex in data:
            ex["_source"] = name
        all_examples.extend(data)
        source_counts[name] = len(data)
        print(f"  {name:<30} {len(data):>7} examples")

    total = len(all_examples)
    print(f"\n  {'TOTAL':<30} {total:>7} examples")

    # Shuffle
    random.shuffle(all_examples)

    # Split
    val_n = int(total * VAL_RATIO)
    val_data  = all_examples[:val_n]
    train_data = all_examples[val_n:]

    # Remove internal _source tag before writing
    def clean(records):
        out = []
        for r in records:
            r2 = {k: v for k, v in r.items() if k != "_source"}
            out.append(r2)
        return out

    write_jsonl(OUT_DIR / "train.jsonl", clean(train_data))
    write_jsonl(OUT_DIR / "val.jsonl",   clean(val_data))

    # Stats
    stats = {
        "total": total,
        "train": len(train_data),
        "val": len(val_data),
        "val_ratio": VAL_RATIO,
        "sources": source_counts,
        "avg_messages_per_example": 3,
    }
    with open(OUT_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Message length stats
    all_lens = [
        len(msg["content"])
        for ex in all_examples
        for msg in ex["messages"]
        if msg["role"] in ("user", "assistant")
    ]
    all_lens.sort()
    p50 = all_lens[len(all_lens) // 2]
    p90 = all_lens[int(len(all_lens) * 0.9)]
    p99 = all_lens[int(len(all_lens) * 0.99)]

    print(f"\n  Train : {len(train_data):,}")
    print(f"  Val   : {len(val_data):,}")
    print(f"\n  Message length (user+assistant):")
    print(f"    median : {p50} chars")
    print(f"    p90    : {p90} chars")
    print(f"    p99    : {p99} chars")
    print(f"\n  Output: {OUT_DIR}/train.jsonl, val.jsonl")


if __name__ == "__main__":
    main()
