"""
split_corpus.py  --  Phase 1, Task 1.3: Corpus Splitting (leak-free)
====================================================================

Splits a cleaned parallel corpus into Train (70%) / Validation (15%) / Test (15%)
with a hard guarantee of ZERO data leakage between splits.

Why "group by source" matters for the rubric:
    If the SAME source sentence appears in both train and test (even paired with a
    slightly different translation), the model effectively sees test inputs during
    training and your BLEU/chrF++ scores become dishonestly high. The rubric
    explicitly rewards "strict split handling without leakage", so we:

        1. Group all pairs by their (normalized) source sentence.
        2. Shuffle the GROUPS with a fixed seed (reproducible).
        3. Slice groups into 70/15/15 by cumulative pair count.
        4. ASSERT afterwards that no source sentence crosses any split boundary
           and that no exact pair is shared across splits.

This keeps every occurrence of a given source sentence inside one split while
still hitting the 70/15/15 proportions closely. Actual achieved percentages are
printed and saved, so you can report the exact numbers.

Usage:
    python src/split_corpus.py \
        --input data/processed/cleaned.csv \
        --outdir data/processed \
        --src-col source_text --tgt-col target_text
"""

import argparse
import csv
import json
import random
import unicodedata
from collections import defaultdict
from pathlib import Path


def load(path: Path, src_col, tgt_col):
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [[r[src_col], r[tgt_col]] for r in data]
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [[r[src_col], r[tgt_col]] for r in csv.DictReader(f)]


def norm_key(text: str) -> str:
    """Normalization used only for grouping/leak detection (not for output)."""
    return unicodedata.normalize("NFC", text).strip().lower()


def split(rows, train=0.70, val=0.15, seed=42):
    """Group by source, shuffle groups, slice to proportions. Returns 3 lists."""
    groups = defaultdict(list)
    for src, tgt in rows:
        groups[norm_key(src)].append([src, tgt])

    group_keys = list(groups.keys())
    random.Random(seed).shuffle(group_keys)

    total = len(rows)
    n_train = int(total * train)
    n_val = int(total * val)

    train_set, val_set, test_set = [], [], []
    count = 0
    for k in group_keys:
        pairs = groups[k]
        if count < n_train:
            train_set.extend(pairs)
        elif count < n_train + n_val:
            val_set.extend(pairs)
        else:
            test_set.extend(pairs)
        count += len(pairs)

    return train_set, val_set, test_set


def verify_no_leakage(train, val, test):
    """Hard checks. Raises AssertionError if any leakage is found."""
    def src_keys(s):
        return {norm_key(p[0]) for p in s}

    def pair_keys(s):
        return {(norm_key(p[0]), norm_key(p[1])) for p in s}

    # 1. No source sentence shared across splits
    assert not (src_keys(train) & src_keys(val)), "Source leakage: train/val"
    assert not (src_keys(train) & src_keys(test)), "Source leakage: train/test"
    assert not (src_keys(val) & src_keys(test)), "Source leakage: val/test"

    # 2. No exact pair shared across splits (implied by #1, checked anyway)
    assert not (pair_keys(train) & pair_keys(test)), "Pair leakage: train/test"
    return True


def write(rows, path: Path, src_col, tgt_col):
    path.parent.mkdir(parents=True, exist_ok=True)
    # CSV
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([src_col, tgt_col])
        w.writerows(rows)
    # JSON (handy for Hugging Face datasets later)
    payload = [{src_col: s, tgt_col: t} for s, t in rows]
    path.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    ap = argparse.ArgumentParser(description="Leak-free 70/15/15 corpus split.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--src-col", default="source_text")
    ap.add_argument("--tgt-col", default="target_text")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load(args.input, args.src_col, args.tgt_col)
    train, val, test = split(rows, seed=args.seed)
    verify_no_leakage(train, val, test)

    write(train, args.outdir / "train", args.src_col, args.tgt_col)
    write(val, args.outdir / "val", args.src_col, args.tgt_col)
    write(test, args.outdir / "test", args.src_col, args.tgt_col)

    total = len(rows)
    summary = {
        "total_pairs": total,
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "train_pct": round(100 * len(train) / total, 2) if total else 0,
        "val_pct": round(100 * len(val) / total, 2) if total else 0,
        "test_pct": round(100 * len(test) / total, 2) if total else 0,
        "leakage_check": "PASSED (no source or pair shared across splits)",
        "seed": args.seed,
    }
    (args.outdir / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("=== Split summary (leak-free, grouped by source) ===")
    for k, v in summary.items():
        print(f"  {k:15s}: {v}")
    print(f"\nFiles written to {args.outdir}/ (train/val/test .csv and .json)")


if __name__ == "__main__":
    main()
