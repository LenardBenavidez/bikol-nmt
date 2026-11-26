# Corpus Documentation — Phase 1 Deliverable (Southern Catanduanes Bikol)

## 1. Overview

- **Project track:** Low-Resource NMT
- **Language variant:** Southern Catanduanes Bikol (Virac area)
- **Language pair & direction:** Southern Catanduanes Bikol ↔ English (bidirectional-capable; evaluated Bikol → English)
- **ISO 639-3 codes:** Southern Catanduanes Bikol = `bln`; Northern/Pandan = `cts`; Central Bikol (reference) = `bcl`.
  - NLLB-200 supports Central Bikol (`bcl_Latn`) but **not** `bln`/`cts` — the core motivation for this project.
- **Primary deliverable:** 1,877 aligned parallel sentence/phrase pairs (within the required 1,500–3,000 range)

## 2. Data Sourcing (Task 1.1)

| Batch | Source | Method | # rows | Notes |
|-------|--------|--------|--------|-------|
| 1 | `12Manman12/eng-to-bcl-vrc` (Hugging Face) | Downloaded, cleaned, filtered | 5,961 raw | Publicly available Virac (southern Catanduanes) Bikol↔English dataset, used as a cited third-party source (see §8). |

- **Variant confirmation:** content is clearly Virac/Catanduanes (e.g., *"Virac ay an sentrong munisipalidad kan Catanduanes…"*, heavy abaca/*bandala* vocabulary — Catanduanes is the abaca capital).
- **Reference data (NOT part of the target corpus):** Central Bikol parallel data from `yam-1111/bicolingo` (AGPL-3.0) + Wikimedia-aligned sentences (CC-BY-SA), 28,465 pairs, used only (a) as the Task 1.4 vocabulary-overlap reference and (b) as the planned Phase 2 transfer-learning base. Both cited in §8.
- **Privacy (RA 10173):** no personal data of private individuals is present.

## 3. Data Cleaning & Normalization (Task 1.2)

Applied by `clean_corpus.py`: HTML strip, Unicode NFC, control/zero-width removal, whitespace
collapse, spelling-map standardization, drop empties/untranslated/length-outliers/bad-ratio, exact dedup.
**Sentence corpus** additionally requires ≥2 tokens on both sides (drops the single-word dictionary).

| Step | Rows removed (sentence corpus) |
|------|-------------------------------|
| Input rows | 5,961 |
| Dropped empty | 38 |
| Dropped untranslated (src == tgt, e.g. loanwords like *bakery/bakery*) | 427 |
| Dropped too short/long (single words, ≥2-token filter) | 3,601 |
| Dropped bad length ratio | 11 |
| Dropped exact duplicates | 7 |
| **Output (sentence corpus)** | **1,877** |

*(Full corpus without the ≥2-token filter = 5,084 pairs: 38 empty, 427 untranslated, 218 bad-ratio, 194 dup removed.)*

**Known quality limitations:** some English typos (*aftenoon, blancket, cutted*), a few rows with
columns swapped (English in the Bikol field), and inconsistent spelling. These are inherited from the
source dataset; a manual QA pass with a native speaker is recommended for future iterations.

## 4. Corpus Splitting (Task 1.3)

Sentence corpus, from `split_summary.json` (grouped by source → **zero leakage**, verified; seed = 42):

| Split | Pairs | % |
|-------|-------|---|
| Train | 1,313 | 69.95% |
| Validation | 281 | 14.97% |
| Test | 283 | 15.08% |
| **Total** | **1,877** | 100% |

`leakage_check: PASSED` — no source sentence or exact pair appears in more than one split.

## 5. Baseline Vocabulary Analysis (Task 1.4)

Sentence corpus, from `cleaned_analysis.json`:

- **Sentence length — Bikol:** mean 4.74, median 4, max 27 tokens
- **Sentence length — English:** mean 4.83, median 4, max 30 tokens
- **Type-Token Ratio:** Bikol 0.279 (2,485 types / 8,897 tokens); English 0.246 (2,230 types / 9,072 tokens)
- **Vocabulary overlap vs Central Bikol (`bcl`) reference:** **54.1%** of Bikol word types appear in the
  28,465-pair Central Bikol reference — i.e. **~46% of Virac Bikol vocabulary is distinct from Central Bikol.**
  - **This is the key empirical justification for the project:** nearly half the vocabulary is not shared
    with the Bikol variety that NLLB already supports, confirming that the Catanduanes variant is genuinely
    under-served and needs its own corpus and fine-tuning.
- See `length_distributions.png` for histograms.

## 6. Known Limitations

- Corpus includes both full sentences (≥3 tokens both sides, ~1,164) and shorter phrase pairs; phrasebook-style entries are an accepted source type per Task 1.1. Adding native-speaker sentences in later iterations would strengthen it further.
- Third-party source; cited in §8.
- Domain skews toward everyday vocabulary and short sentences.
- Southern (Virac) variant only; Northern/Pandan (`cts`) not covered.

## 7. Reproducibility

```bash
# sentence corpus (primary deliverable)
python src/clean_corpus.py --input data/raw/virac_bikol.csv --output data/processed_sentences/cleaned.csv \
    --src-col Bicol --tgt-col English --min-len 2
python src/split_corpus.py --input data/processed_sentences/cleaned.csv --outdir data/processed_sentences \
    --src-col Bicol --tgt-col English
python src/analyze_corpus.py --input data/processed_sentences/cleaned.csv --src-col Bicol --tgt-col English \
    --reference-vocab config/central_bikol_vocab.txt --plots
```
Seed = 42 throughout.

## 8. Attribution / Citations

- Source corpus: 12Manman12, *eng-to-bcl-vrc* dataset, Hugging Face.
- Central Bikol reference/transfer data: yam-1111, *bicolingo* (AGPL-3.0); Wikimedia Foundation (CC-BY-SA).
