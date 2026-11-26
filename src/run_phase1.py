"""
run_phase1.py  --  One command to run the whole Phase 1 pipeline
================================================================

Clean  ->  Split (leak-free 70/15/15)  ->  Analyze

Once you have your collected raw corpus (a CSV/JSON with a Bikol column and an
English/Tagalog column), this produces every Phase 1 deliverable in one go:

    data/processed/cleaned.csv                 (+ _cleaning_report.json)
    data/processed/train.csv / val.csv / test.csv  (+ .json, split_summary.json)
    data/processed/cleaned_analysis.json       (+ length_distributions.png)

Usage:
    python src/run_phase1.py \
        --input data/raw/collected.csv \
        --src-col bikol --tgt-col english \
        --reference-vocab config/central_bikol_vocab.txt   # optional
"""

import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent


def run(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run([sys.executable, *cmd])
    if result.returncode != 0:
        sys.exit(f"[error] step failed: {cmd}")


def main():
    ap = argparse.ArgumentParser(description="Run the full Phase 1 corpus pipeline.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--outdir", default=Path("data/processed"), type=Path)
    ap.add_argument("--src-col", default="source_text")
    ap.add_argument("--tgt-col", default="target_text")
    ap.add_argument("--spelling-map", default=Path("config/spelling_map.csv"), type=Path)
    ap.add_argument("--reference-vocab", default=None, type=Path)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    cleaned = args.outdir / "cleaned.csv"

    # 1. Clean
    run([SRC / "clean_corpus.py",
         "--input", args.input, "--output", cleaned,
         "--src-col", args.src_col, "--tgt-col", args.tgt_col,
         "--spelling-map", args.spelling_map])

    # 2. Split (note: output columns are now the same src/tgt names)
    run([SRC / "split_corpus.py",
         "--input", cleaned, "--outdir", args.outdir,
         "--src-col", args.src_col, "--tgt-col", args.tgt_col])

    # 3. Analyze
    analyze_cmd = [SRC / "analyze_corpus.py",
                   "--input", cleaned,
                   "--src-col", args.src_col, "--tgt-col", args.tgt_col,
                   "--plots"]
    if args.reference_vocab:
        analyze_cmd += ["--reference-vocab", args.reference_vocab]
    run(analyze_cmd)

    print("\n=== Phase 1 complete ===")
    print(f"Deliverables are in {args.outdir}/")
    print("Next: fill in docs/CORPUS_DOCUMENTATION.md with your sourcing + the")
    print("numbers from the *_report.json / split_summary.json / *_analysis.json files.")


if __name__ == "__main__":
    main()
