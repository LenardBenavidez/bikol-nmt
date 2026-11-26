# Low-Resource NMT for Catanduanes (Virac) Bikol → English

A capstone project that fine-tunes Meta's **NLLB-200** with **LoRA** to translate
**Southern Catanduanes (Virac) Bikol ↔ English** — a variant NLLB does not natively
support — and serves it through a **FastAPI** API and web interface.

> **Status: all four phases complete.** Corpus, trained LoRA adapter, API + web app,
> and evaluation are in this repository.

## Results (held-out test set, 283 pairs, SacreBLEU 2.4.3)

| Model | BLEU ↑ | chrF++ ↑ | TER ↓ |
|-------|-------|---------|------|
| Zero-shot base NLLB-200 (un-tuned) | 1.91 | 21.08 | 141.63 |
| **LoRA fine-tuned (this work)** | **19.71** | **40.45** | **67.16** |
| Difference | +17.80 | +19.37 | −74.47 |

Fine-tuning lifts an unusable baseline (BLEU ~2) to usable quality (BLEU ~20).
~46% of Virac Bikol vocabulary is distinct from Central Bikol — the motivation for
building a dedicated corpus and adapter.

## Repository structure

```
bikol-nmt/
├── README.md
├── requirements.txt
├── train.py                     # Phase 2: LoRA fine-tuning script
├── main.py                      # Phase 3: FastAPI API + web UI + feedback logging
├── src/                         # Phase 1: corpus pipeline
│   ├── clean_corpus.py          #   cleaning & normalization (Task 1.2)
│   ├── split_corpus.py          #   leak-free 70/15/15 split (Task 1.3)
│   ├── analyze_corpus.py        #   length / TTR / vocab overlap (Task 1.4)
│   └── run_phase1.py            #   runs all three in one command
├── config/                      # spelling map, Central Bikol reference vocab
├── data/
│   ├── raw/virac_bikol.csv       #   source dataset (5,961 raw rows)
│   └── processed_sentences/      #   cleaned corpus + train/val/test (CSV + JSON)
├── models/nllb_bikol_lora/      # trained LoRA adapter (~9.5 MB)
├── results/                     # metrics.json, error_analysis.csv, training_loss_curve.png
├── docs/CORPUS_DOCUMENTATION.md # corpus writeup
└── notebooks/                   # Colab notebooks (training / serving / evaluation)
```

## Setup

Requires **Python 3.10+**. Training and serving the 600M model need a **GPU**
(e.g. Google Colab T4); the corpus pipeline runs on CPU.

```bash
pip install -r requirements.txt
```

The base model `facebook/nllb-200-distilled-600M` downloads from Hugging Face on
first use (no API key needed for this public model).

## Reproduce each phase

**Phase 1 — Corpus** (CPU, seconds):
```bash
python src/clean_corpus.py   --input data/raw/virac_bikol.csv --output data/processed_sentences/cleaned.csv \
    --src-col Bicol --tgt-col English --min-len 2
python src/split_corpus.py   --input data/processed_sentences/cleaned.csv --outdir data/processed_sentences \
    --src-col Bicol --tgt-col English
python src/analyze_corpus.py --input data/processed_sentences/cleaned.csv --src-col Bicol --tgt-col English \
    --reference-vocab config/central_bikol_vocab.txt --plots
```

**Phase 2 — Train the LoRA adapter** (GPU):
```bash
python train.py --train data/processed_sentences/train.csv \
                --val   data/processed_sentences/val.csv \
                --outdir models/nllb_bikol_lora
```
Writes the adapter to `models/nllb_bikol_lora/` and the loss curve to
`results/training_loss_curve.png`. Hyperparameters (q_proj/v_proj, r=16, α=32,
dropout 0.05, lr 3e-4, AdamW, linear decay, ≤20 epochs with early stopping, seed 42)
match the project brief and can be overridden via command-line flags.

**Phase 3 — Run the API + web app** (GPU):
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
Then open `http://localhost:8000/` for the web interface, or POST to
`/api/v1/translate`:
```json
{ "source_text": "nagkaon kana ba", "source_lang": "bcl_North", "target_lang": "eng_Latn" }
```
Feedback reports are logged to `results/feedback.db` (SQLite).

**Phase 4 — Evaluate** (GPU): run `notebooks/phase4_evaluation.ipynb`, which scores
the base vs. fine-tuned model on the test set and writes `results/metrics.json` and
`results/error_analysis.csv`.

### Running on Google Colab
The notebooks in `notebooks/` wrap the same steps for a free Colab T4 GPU:
`phase2_train_nllb_lora.ipynb` (train), `run_api_colab.ipynb` (serve with a public
URL), and `phase4_evaluation.ipynb` (evaluate). Each clones this repo, so they run
end-to-end with no local setup.

## Notes

- **Language codes:** NLLB has no code for Catanduanes Bikol, so the closest supported
  code `bcl_Latn` (Central Bikol) is reused as the source tag and the model is
  fine-tuned to adapt it to the Virac variant.
- **Decoding:** inference uses greedy decoding, matching the metrics reported above.
- **No fabricated data:** the Bikol side comes from a cited third-party source
  (see `docs/CORPUS_DOCUMENTATION.md`); scripts only clean, train, and measure.
- **Reproducibility:** seed = 42 throughout.
