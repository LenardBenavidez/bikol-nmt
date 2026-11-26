# Low-Resource NMT for Catanduanes Bikol

Fine-tuning a multilingual seq2seq model (NLLB-200 / mBART) with LoRA to translate
Northern (Pandan) / Southern Catanduanes Bikol ↔ English, served through a
FastAPI + web interface. 8-week capstone project.

> **Status: Phase 1 (Corpus) — COMPLETE.** This repo contains the finished Phase 1
> corpus, splits, analysis, documentation, and the full reproducible pipeline.
> Phases 2–4 (LoRA fine-tuning, API, evaluation) come later.

## Phase 1 result (Southern Catanduanes / Virac Bikol)

| Item | Value |
|------|-------|
| Aligned parallel pairs | **1,877** (within required 1,500–3,000) |
| Train / Val / Test | 1,313 / 281 / 283 (70/15/15) |
| Data leakage | **None** — verified, no source or pair crosses splits |
| Bikol vocab distinct from Central Bikol | **~46%** (key justification for the project) |
| Source | `12Manman12/eng-to-bcl-vrc` (Virac Bikol↔English), cleaned & filtered — cited |

See `docs/CORPUS_DOCUMENTATION.md` for the full writeup with all numbers.

## Where the deliverables are

```
bikol-nmt/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/virac_bikol.csv                 # source dataset (5,961 raw rows)
│   └── processed_sentences/                # ★ PRIMARY DELIVERABLE
│       ├── cleaned.csv                      #   1,877 cleaned aligned pairs
│       ├── train.csv / val.csv / test.csv   #   70/15/15 split (+ .json)
│       ├── split_summary.json               #   leakage check: PASSED
│       ├── cleaned_analysis.json            #   length / TTR / vocab overlap
│       └── length_distributions.png         #   histograms
├── src/
│   ├── clean_corpus.py     # Task 1.2  normalization & cleaning
│   ├── split_corpus.py     # Task 1.3  leak-free 70/15/15 split
│   ├── analyze_corpus.py   # Task 1.4  length / TTR / vocab overlap
│   └── run_phase1.py       # runs all three in one command
├── config/
│   ├── spelling_map.csv          # variant→standard spelling rules (Task 1.2)
│   └── central_bikol_vocab.txt   # Central Bikol reference vocab (Task 1.4)
└── docs/
    └── CORPUS_DOCUMENTATION.md          # ★ the Phase 1 writeup (real numbers)
```

## Reproduce the pipeline

```bash
pip install -r requirements.txt
python src/clean_corpus.py   --input data/raw/virac_bikol.csv --output data/processed_sentences/cleaned.csv \
    --src-col Bicol --tgt-col English --min-len 2
python src/split_corpus.py   --input data/processed_sentences/cleaned.csv --outdir data/processed_sentences \
    --src-col Bicol --tgt-col English
python src/analyze_corpus.py --input data/processed_sentences/cleaned.csv --src-col Bicol --tgt-col English \
    --reference-vocab config/central_bikol_vocab.txt --plots
```
Seed = 42 throughout (fully reproducible).

## Phase 1 deliverable checklist

- [x] Cleaned parallel corpus (CSV/JSON) — `data/processed_sentences/cleaned.csv`
- [x] 70/15/15 split with verified zero leakage — `split_summary.json`
- [x] Vocabulary analysis — `cleaned_analysis.json`, `length_distributions.png`
- [x] Documentation of sourcing + cleaning — `docs/CORPUS_DOCUMENTATION.md`

## Design notes

- **No fabricated data.** The Bikol side comes from a real cited source; scripts only
  clean and measure — nothing is invented.
- **Leak-free split** groups by source sentence so no source crosses splits — honest
  evaluation later.
- **NLLB gap:** NLLB-200 covers Central Bikol (`bcl_Latn`) but not the Catanduanes
  variants — the motivation for fine-tuning in Phase 2.
