"""
clean_corpus.py  --  Phase 1, Task 1.2: Data Normalization & Cleaning
=====================================================================

Takes a raw collected corpus (any of CSV / TSV / JSON) and produces a cleaned,
de-duplicated parallel corpus plus a cleaning report that documents exactly how
many rows were removed at each step (useful for the Phase 1 documentation and
for the rubric's "data quality" points).

What it does, in order:
    1. Load raw pairs (flexible column names / file formats).
    2. Strip HTML tags and unescape HTML entities.
    3. Unicode-normalize (NFC) and fix whitespace / invisible control chars.
    4. Optional diacritic standardization.
    5. Apply a user-editable spelling-standardization map (config/spelling_map.csv).
    6. Drop empties, drop untranslated rows (source == target), drop length outliers.
    7. Remove exact duplicate pairs.

It NEVER invents data. It only cleans what you give it.

Usage:
    python src/clean_corpus.py \
        --input data/raw/collected.csv \
        --output data/processed/cleaned.csv \
        --src-col source_text --tgt-col target_text \
        --spelling-map config/spelling_map.csv

Run with -h for all options.
"""

import argparse
import csv
import html
import json
import re
import sys
import unicodedata
from pathlib import Path


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #
def load_rows(path: Path, src_col: str, tgt_col: str):
    """Load (source, target) pairs from CSV, TSV, or JSON.

    JSON may be a list of objects or an object with a 'data' list.
    Returns a list of [source, target] lists.
    """
    suffix = path.suffix.lower()
    rows = []

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if src_col not in reader.fieldnames or tgt_col not in reader.fieldnames:
                sys.exit(
                    f"[error] Columns '{src_col}' / '{tgt_col}' not found. "
                    f"Available columns: {reader.fieldnames}"
                )
            for r in reader:
                rows.append([r.get(src_col, ""), r.get(tgt_col, "")])

    elif suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("data", [])
        for r in data:
            rows.append([r.get(src_col, ""), r.get(tgt_col, "")])

    else:
        sys.exit(f"[error] Unsupported file type: {suffix} (use .csv/.tsv/.json)")

    return rows


# --------------------------------------------------------------------------- #
# Cleaning primitives                                                          #
# --------------------------------------------------------------------------- #
_TAG_RE = re.compile(r"<[^>]+>")            # crude HTML tag remover
_CTRL_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")
_WS_RE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Remove HTML tags and unescape entities (&amp; -> &, &#241; -> ñ)."""
    text = html.unescape(text)
    text = _TAG_RE.sub(" ", text)
    return text


def normalize_unicode(text: str) -> str:
    """NFC-normalize, drop control chars, collapse whitespace, trim."""
    text = unicodedata.normalize("NFC", text)
    text = _CTRL_RE.sub("", text)
    text = text.replace("​", "").replace("﻿", "")  # zero-width / BOM
    text = _WS_RE.sub(" ", text)
    return text.strip()


def strip_diacritics(text: str) -> str:
    """Optional: fold accented letters to their base form (á -> a, à -> a).

    Spoken Catanduanes Bikol is often written without stress marks, so folding
    them can reduce spurious vocabulary variants. BUT this also merges ñ -> n,
    which you may NOT want. Only enable with --fold-diacritics after checking
    your data. Off by default.
    """
    decomposed = unicodedata.normalize("NFD", text)
    folded = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", folded)


def load_spelling_map(path: Path):
    """Load a CSV of spelling variants -> standard form.

    Expected columns: variant, standard  (case-insensitive, whole-word match).
    Returns a dict. Missing file -> empty map (no standardization applied).
    """
    mapping = {}
    if path is None or not path.exists():
        return mapping
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            variant = (r.get("variant") or "").strip().lower()
            standard = (r.get("standard") or "").strip()
            # Skip blank rows and comment lines (variant starting with '#').
            if not variant or variant.startswith("#"):
                continue
            if variant and standard:
                mapping[variant] = standard
    return mapping


def apply_spelling_map(text: str, mapping: dict) -> str:
    """Replace whole-word spelling variants with their standard form."""
    if not mapping:
        return text

    def repl(m):
        word = m.group(0)
        std = mapping.get(word.lower())
        if std is None:
            return word
        # preserve capitalization of the first letter
        return std.capitalize() if word[:1].isupper() else std

    return re.sub(r"\b\w+\b", repl, text, flags=re.UNICODE)


# --------------------------------------------------------------------------- #
# Main cleaning pipeline                                                       #
# --------------------------------------------------------------------------- #
def clean(rows, spelling_map, fold_diacritics, min_len, max_len, max_ratio):
    """Run every row through the cleaning steps and record what was dropped."""
    report = {
        "input_rows": len(rows),
        "dropped_empty": 0,
        "dropped_untranslated": 0,     # source == target (likely not translated)
        "dropped_too_short_or_long": 0,
        "dropped_bad_length_ratio": 0,
        "dropped_exact_duplicates": 0,
    }

    cleaned = []
    for src_raw, tgt_raw in rows:
        src = apply_spelling_map(
            normalize_unicode(strip_html(src_raw or "")), spelling_map
        )
        tgt = normalize_unicode(strip_html(tgt_raw or ""))
        if fold_diacritics:
            src = strip_diacritics(src)

        # Drop empties
        if not src or not tgt:
            report["dropped_empty"] += 1
            continue

        # Drop untranslated (identical both sides -> speaker skipped the cell)
        if src.lower() == tgt.lower():
            report["dropped_untranslated"] += 1
            continue

        # Length filters (token counts, whitespace tokenization)
        src_len, tgt_len = len(src.split()), len(tgt.split())
        if not (min_len <= src_len <= max_len and min_len <= tgt_len <= max_len):
            report["dropped_too_short_or_long"] += 1
            continue

        # Length-ratio filter: wildly mismatched lengths signal misalignment
        ratio = max(src_len, tgt_len) / max(1, min(src_len, tgt_len))
        if ratio > max_ratio:
            report["dropped_bad_length_ratio"] += 1
            continue

        cleaned.append([src, tgt])

    # Exact-duplicate removal (keep first occurrence, preserve order)
    seen, deduped = set(), []
    for pair in cleaned:
        key = (pair[0].lower(), pair[1].lower())
        if key in seen:
            report["dropped_exact_duplicates"] += 1
            continue
        seen.add(key)
        deduped.append(pair)

    report["output_rows"] = len(deduped)
    return deduped, report


# --------------------------------------------------------------------------- #
# Writing                                                                      #
# --------------------------------------------------------------------------- #
def write_output(rows, path: Path, src_col, tgt_col):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        payload = [{src_col: s, tgt_col: t} for s, t in rows]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([src_col, tgt_col])
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Clean & normalize a raw parallel corpus.")
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--src-col", default="source_text", help="Bikol (source) column name")
    ap.add_argument("--tgt-col", default="target_text", help="English/Tagalog (target) column name")
    ap.add_argument("--spelling-map", type=Path, default=Path("config/spelling_map.csv"))
    ap.add_argument("--fold-diacritics", action="store_true",
                    help="Fold accents (á->a). Off by default; also merges ñ->n.")
    ap.add_argument("--min-len", type=int, default=1, help="Min tokens per side")
    ap.add_argument("--max-len", type=int, default=80, help="Max tokens per side")
    ap.add_argument("--max-ratio", type=float, default=3.0,
                    help="Max length ratio between sides before a pair is dropped")
    args = ap.parse_args()

    rows = load_rows(args.input, args.src_col, args.tgt_col)
    spelling_map = load_spelling_map(args.spelling_map)
    cleaned, report = clean(
        rows, spelling_map, args.fold_diacritics,
        args.min_len, args.max_len, args.max_ratio,
    )
    write_output(cleaned, args.output, args.src_col, args.tgt_col)

    # Save + print the cleaning report (document this in CORPUS_DOCUMENTATION.md)
    report_path = args.output.with_name(args.output.stem + "_cleaning_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Cleaning report ===")
    for k, v in report.items():
        print(f"  {k:30s}: {v}")
    print(f"\nCleaned corpus -> {args.output}")
    print(f"Report         -> {report_path}")
    if spelling_map:
        print(f"Spelling variants standardized: {len(spelling_map)} entries")


if __name__ == "__main__":
    main()
