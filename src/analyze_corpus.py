"""
analyze_corpus.py  --  Phase 1, Task 1.4: Baseline Vocabulary Analysis
======================================================================

Computes the descriptive statistics the rubric asks for, over the cleaned corpus
(and optionally per split):

    * Sentence-length distributions (tokens per sentence) for both languages:
      count, mean, median, min, max, and a small histogram (bucketed counts).
    * Type-Token Ratio (TTR) per language  =  unique tokens / total tokens.
      (Lower TTR = more repetition; higher TTR = more lexical variety.)
    * Vocabulary overlap against an optional reference word list
      (e.g., a Central Bikol or Tagalog vocabulary) -- Task 1.4's
      "vocabulary overlap against standard Central Bikol or Tagalog datasets".
    * Cross-side type overlap: how many source (Bikol) types also appear on the
      target side -- a rough indicator of loanwords / shared vocabulary.

Outputs a JSON report and prints a readable summary. Optionally saves simple
PNG histograms if matplotlib is installed (--plots).

Usage:
    python src/analyze_corpus.py \
        --input data/processed/cleaned.csv \
        --src-col source_text --tgt-col target_text \
        --reference-vocab config/central_bikol_vocab.txt   # optional
"""

import argparse
import csv
import json
import statistics as stats
from collections import Counter
from pathlib import Path


def load(path: Path, src_col, tgt_col):
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return [[r[src_col], r[tgt_col]] for r in data]
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [[r[src_col], r[tgt_col]] for r in csv.DictReader(f)]


def tokenize(text: str):
    """Simple, language-agnostic whitespace tokenization, lowercased.

    (For the real NMT model you'll use the NLLB subword tokenizer; this plain
    tokenizer is only for descriptive corpus statistics.)
    """
    return text.lower().split()


def length_stats(lengths):
    if not lengths:
        return {}
    # Histogram buckets for sentence length
    buckets = {"1-5": 0, "6-10": 0, "11-20": 0, "21-40": 0, "41+": 0}
    for n in lengths:
        if n <= 5:
            buckets["1-5"] += 1
        elif n <= 10:
            buckets["6-10"] += 1
        elif n <= 20:
            buckets["11-20"] += 1
        elif n <= 40:
            buckets["21-40"] += 1
        else:
            buckets["41+"] += 1
    return {
        "sentences": len(lengths),
        "mean_tokens": round(stats.mean(lengths), 2),
        "median_tokens": stats.median(lengths),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "length_histogram": buckets,
    }


def ttr(tokens):
    """Type-Token Ratio = unique tokens / total tokens."""
    total = len(tokens)
    return {
        "total_tokens": total,
        "unique_types": len(set(tokens)),
        "ttr": round(len(set(tokens)) / total, 4) if total else 0.0,
    }


def load_reference_vocab(path: Path):
    """One word per line. Returns a lowercase set, or None if no file."""
    if path is None or not path.exists():
        return None
    words = {
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    return words


def analyze(rows, reference_vocab):
    src_lens, tgt_lens = [], []
    src_tokens, tgt_tokens = [], []

    for src, tgt in rows:
        s, t = tokenize(src), tokenize(tgt)
        src_lens.append(len(s))
        tgt_lens.append(len(t))
        src_tokens.extend(s)
        tgt_tokens.extend(t)

    src_types = set(src_tokens)
    tgt_types = set(tgt_tokens)

    report = {
        "pairs": len(rows),
        "source": {"length": length_stats(src_lens), "vocab": ttr(src_tokens)},
        "target": {"length": length_stats(tgt_lens), "vocab": ttr(tgt_tokens)},
        "cross_side_type_overlap": {
            "shared_types": len(src_types & tgt_types),
            "pct_of_source_types_shared": (
                round(100 * len(src_types & tgt_types) / len(src_types), 2)
                if src_types else 0.0
            ),
            "note": "Source types also seen on target side (loanwords / shared words).",
        },
        "top_source_tokens": Counter(src_tokens).most_common(20),
        "top_target_tokens": Counter(tgt_tokens).most_common(20),
    }

    # Task 1.4: vocabulary overlap vs a reference (Central Bikol / Tagalog)
    if reference_vocab is not None:
        shared = src_types & reference_vocab
        report["reference_vocab_overlap"] = {
            "reference_size": len(reference_vocab),
            "source_types": len(src_types),
            "shared_types": len(shared),
            "pct_of_source_in_reference": (
                round(100 * len(shared) / len(src_types), 2) if src_types else 0.0
            ),
            "note": (
                "How much of your Bikol vocabulary already exists in the reference "
                "(Central Bikol / Tagalog). Lower overlap = more genuinely "
                "distinct local vocabulary = stronger justification for the project."
            ),
        }
    else:
        report["reference_vocab_overlap"] = (
            "No reference vocab supplied. Pass --reference-vocab to compute this "
            "(Task 1.4). A Tagalog or Central Bikol word list, one word per line."
        )

    return report


def maybe_plot(rows, outdir: Path):
    """Save simple length-distribution histograms if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[info] matplotlib not installed; skipping plots (pip install matplotlib)")
        return

    src_lens = [len(tokenize(s)) for s, _ in rows]
    tgt_lens = [len(tokenize(t)) for _, t in rows]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(src_lens, bins=20, color="#2b6cb0")
    ax[0].set_title("Source (Bikol) sentence length")
    ax[0].set_xlabel("tokens"); ax[0].set_ylabel("sentences")
    ax[1].hist(tgt_lens, bins=20, color="#dd6b20")
    ax[1].set_title("Target sentence length")
    ax[1].set_xlabel("tokens")
    fig.tight_layout()
    out = outdir / "length_distributions.png"
    fig.savefig(out, dpi=120)
    print(f"Saved length histograms -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Baseline vocabulary & length analysis.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--src-col", default="source_text")
    ap.add_argument("--tgt-col", default="target_text")
    ap.add_argument("--reference-vocab", type=Path, default=None,
                    help="Optional word list (one per line) for overlap analysis.")
    ap.add_argument("--plots", action="store_true", help="Save PNG histograms.")
    args = ap.parse_args()

    rows = load(args.input, args.src_col, args.tgt_col)
    reference_vocab = load_reference_vocab(args.reference_vocab)
    report = analyze(rows, reference_vocab)

    out = args.input.with_name(args.input.stem + "_analysis.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Readable summary
    print("=== Corpus analysis ===")
    print(f"  pairs: {report['pairs']}")
    for side in ("source", "target"):
        L, V = report[side]["length"], report[side]["vocab"]
        print(f"  [{side}] mean_len={L.get('mean_tokens')} "
              f"median={L.get('median_tokens')} max={L.get('max_tokens')} "
              f"| TTR={V['ttr']} ({V['unique_types']} types / {V['total_tokens']} tokens)")
    if isinstance(report["reference_vocab_overlap"], dict):
        o = report["reference_vocab_overlap"]
        print(f"  reference overlap: {o['pct_of_source_in_reference']}% of Bikol "
              f"types appear in the reference vocab")
    print(f"\nFull report -> {out}")

    if args.plots:
        maybe_plot(rows, args.input.parent)


if __name__ == "__main__":
    main()
