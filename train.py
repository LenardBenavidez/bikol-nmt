"""
train.py — LoRA fine-tuning of NLLB-200 for Catanduanes (Virac) Bikol -> English.

This is the runnable training script for Phase 2. It reproduces the fine-tuning
exactly: loads the cleaned 70/15/15 corpus, applies LoRA to NLLB-200, trains with
the project's hyperparameters, plots the training/validation loss curve, and saves
the adapter.

Usage (from the repo root, on a GPU machine / Colab):
    pip install -r requirements.txt
    python train.py \
        --train data/processed_sentences/train.csv \
        --val   data/processed_sentences/val.csv \
        --outdir models/nllb_bikol_lora

All hyperparameters match the project brief (q_proj/v_proj; r in [8,16];
alpha in [16,32]; dropout 0.05; lr 3e-4; AdamW; linear decay + warmup; 15-30 epochs).
Seed is fixed at 42 for reproducibility. Decoding at inference is greedy, matching
the deployed API (main.py).

Note on language codes: NLLB has no code for Catanduanes Bikol, so the closest
supported code (bcl_Latn, Central Bikol) is reused as the source tag and the model
is fine-tuned to adapt it to the Virac variant.
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoRA fine-tune NLLB-200 for Virac Bikol -> English")
    p.add_argument("--train", default="data/processed_sentences/train.csv")
    p.add_argument("--val", default="data/processed_sentences/val.csv")
    p.add_argument("--outdir", default="models/nllb_bikol_lora")
    p.add_argument("--base-model", default="facebook/nllb-200-distilled-600M")
    p.add_argument("--src-col", default="Bicol")
    p.add_argument("--tgt-col", default="English")
    p.add_argument("--src-lang", default="bcl_Latn")   # closest supported code, reused for Virac
    p.add_argument("--tgt-lang", default="eng_Latn")
    p.add_argument("--max-len", type=int, default=128)
    # LoRA (brief: q_proj/v_proj; r in [8,16]; alpha in [16,32]; dropout 0.05)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    # Training (brief: lr 3e-4; AdamW; linear decay + warmup; 15-30 epochs)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=3, help="early-stopping patience (epochs)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                              Seq2SeqTrainingArguments, Seq2SeqTrainer,
                              DataCollatorForSeq2Seq, EarlyStoppingCallback)
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ---- data ----
    train_df = pd.read_csv(args.train).dropna(subset=[args.src_col, args.tgt_col])
    val_df = pd.read_csv(args.val).dropna(subset=[args.src_col, args.tgt_col])
    print(f"train={len(train_df)}  val={len(val_df)}")

    # ---- model + tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(args.tgt_lang)

    # ---- LoRA: freeze base weights, train only low-rank matrices on q_proj/v_proj ----
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- tokenize ----
    def preprocess(batch):
        model_inputs = tokenizer(batch[args.src_col], max_length=args.max_len, truncation=True)
        labels = tokenizer(text_target=batch[args.tgt_col], max_length=args.max_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = Dataset.from_pandas(train_df[[args.src_col, args.tgt_col]]).map(
        preprocess, batched=True, remove_columns=[args.src_col, args.tgt_col])
    val_ds = Dataset.from_pandas(val_df[[args.src_col, args.tgt_col]]).map(
        preprocess, batched=True, remove_columns=[args.src_col, args.tgt_col])

    # ---- train ----
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    targs = Seq2SeqTrainingArguments(
        output_dir=args.outdir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=(device == "cuda"),
        seed=args.seed,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collator, tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()

    # ---- loss curve (Figure 1 in the report) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        hist = trainer.state.log_history
        tr = {e["epoch"]: e["loss"] for e in hist if "loss" in e}
        va = {e["epoch"]: e["eval_loss"] for e in hist if "eval_loss" in e}
        plt.figure(figsize=(7, 4.5))
        plt.plot(list(tr.keys()), list(tr.values()), marker="o", label="Training loss")
        plt.plot(list(va.keys()), list(va.values()), marker="s", label="Validation loss")
        if va:
            best = min(va, key=va.get)
            plt.axvline(best, color="gray", ls="--", lw=1)
        plt.xlabel("Epoch"); plt.ylabel("Cross-entropy loss")
        plt.title("Training and validation loss per epoch")
        plt.legend(); plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/training_loss_curve.png", dpi=150)
        print("saved results/training_loss_curve.png")
    except Exception as e:
        print("skipped loss plot:", e)

    # ---- save adapter ----
    os.makedirs(args.outdir, exist_ok=True)
    model.save_pretrained(args.outdir)
    tokenizer.save_pretrained(args.outdir)
    print(f"adapter saved to {args.outdir}/")


if __name__ == "__main__":
    main()
