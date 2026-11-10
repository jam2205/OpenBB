"""OPTIONAL: fine-tune ModernBERT heads for tone and driver classification.

The defaults (FinBERT tone + pre-trained ModernBERT zero-shot drivers) need no training.
Fine-tune only when you have labelled data and want higher accuracy or 8k-context tone.

Labels, best first:
  1. Your own labelled headlines (CSV with columns text,label).
  2. Weak labels: FinBERT as teacher for tone, config `driver_keywords` for drivers.
     The student adds 8k context and your domain; it inherits the teacher's errors, so
     replace weak labels with human labels where it matters.
  3. Public: `takala/financial_phrasebank` (tone) can be mixed in.

Colab:  python -m mnp.finetune tone   --csv data/headlines.csv --out heads/tone
        python -m mnp.finetune driver --csv data/headlines.csv --out heads/driver
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd


def weak_tone_labels(texts: list[str], teacher: str = "yiyanghkust/finbert-tone", min_conf: float = 0.8) -> pd.DataFrame:
    from .nlp import NLP

    pipe = NLP({"device": "auto"})._clf(teacher, 512)
    res = pd.DataFrame(pipe(texts, batch_size=32))
    df = pd.DataFrame({"text": texts, "label": res["label"].str.lower(), "conf": res["score"]})
    return df[df.conf >= min_conf].drop(columns="conf")


def weak_driver_labels(texts: list[str], keywords: dict[str, list[str]]) -> pd.DataFrame:
    """Label a headline only if exactly one driver's keywords match (keeps labels clean)."""
    pats = {d: re.compile("|".join(map(re.escape, kws)), re.I) for d, kws in keywords.items()}
    rows = []
    for t in texts:
        hits = [d for d, p in pats.items() if p.search(t)]
        if len(hits) == 1:
            rows.append((t, hits[0]))
    return pd.DataFrame(rows, columns=["text", "label"])


def train(df: pd.DataFrame, out_dir: str, base: str = "answerdotai/ModernBERT-large",
          max_length: int = 512, epochs: int = 3, lr: float = 3e-5, batch_size: int = 16,
          eval_frac: float = 0.15, seed: int = 42) -> dict:
    """Standard BERT-style fine-tune. Returns eval metrics. Saves model + tokenizer to out_dir."""
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments)

    from .nlp import modernbert_kwargs

    df = df.dropna(subset=["text", "label"]).drop_duplicates("text")
    labels = sorted(df["label"].unique())
    l2i = {l: i for i, l in enumerate(labels)}
    df = df.assign(labels=df["label"].map(l2i))
    ds = Dataset.from_pandas(df[["text", "labels"]], preserve_index=False).train_test_split(
        test_size=eval_frac, seed=seed, stratify_by_column=None)

    tok = AutoTokenizer.from_pretrained(base)
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_length,
                              return_token_type_ids=False), batched=True, remove_columns=["text"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kw = modernbert_kwargs(device)
    kw.pop("torch_dtype", None)  # train in fp32 master weights; Trainer handles mixed precision
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=l2i, **kw)

    bf16 = device == "cuda" and torch.cuda.get_device_capability()[0] >= 8

    def metrics(p):
        pred = p.predictions.argmax(-1)
        acc = float((pred == p.label_ids).mean())
        f1s = []
        for k in range(len(labels)):
            tp = ((pred == k) & (p.label_ids == k)).sum()
            prec = tp / max((pred == k).sum(), 1)
            rec = tp / max((p.label_ids == k).sum(), 1)
            f1s.append(0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
        return {"accuracy": acc, "macro_f1": float(np.mean(f1s))}

    args = TrainingArguments(
        out_dir, num_train_epochs=epochs, learning_rate=lr, per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2, warmup_ratio=0.1, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", save_total_limit=1, bf16=bf16, fp16=device == "cuda" and not bf16,
        report_to=[], seed=seed)
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"],
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    trainer.train()
    result = trainer.evaluate()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    return result


def main():
    from .run import load_config

    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["tone", "driver"])
    p.add_argument("--csv", required=True, help="CSV with a 'text' column; a 'label' column skips weak labelling")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-length", type=int, default=512)
    a = p.parse_args()
    cfg = load_config()
    df = pd.read_csv(a.csv)
    if "label" not in df:
        texts = df["text"].astype(str).tolist()
        df = weak_tone_labels(texts) if a.task == "tone" else weak_driver_labels(texts, cfg["driver_keywords"])
        print(f"weak-labelled {len(df)} / {len(texts)} rows:\n{df.label.value_counts()}")
    print(train(df, a.out, base=cfg["models"]["base"], max_length=a.max_length, epochs=a.epochs))


if __name__ == "__main__":
    main()
