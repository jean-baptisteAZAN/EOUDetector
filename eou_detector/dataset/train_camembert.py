"""Fine-tune CamemBERT as the semantic (lexical-only) EOU baseline.

Trains on the fini/pas_fini dataset (built by build_lexical_dataset.py) and reports
the same metrics as the classical baseline, so the numbers are directly comparable.

Runs on CPU but is slow — use a GPU (Colab). ~20-40 min on a T4.

Usage:
  python -m eou_detector.dataset.train_camembert --data data/lexical --out models/camembert-eou
  python -m eou_detector.dataset.train_camembert --data data/lexical --epochs 4 --model almanach/camembert-base
"""
import argparse
import json
import os

import numpy as np


def load_jsonl(path):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["text"])
            labels.append(int(r["label"]))
    return texts, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lexical")
    ap.add_argument("--out", default="models/camembert-eou")
    ap.add_argument("--model", default="almanach/camembert-base")
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=64)
    args = ap.parse_args()

    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer)

    tr_x, tr_y = load_jsonl(os.path.join(args.data, "train.jsonl"))
    te_x, te_y = load_jsonl(os.path.join(args.data, "test.jsonl"))
    print(f"train={len(tr_x)}  test={len(te_x)}")

    tok = AutoTokenizer.from_pretrained(args.model)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    dtr = Dataset.from_dict({"text": tr_x, "label": tr_y}).map(tokenize, batched=True)
    dte = Dataset.from_dict({"text": te_x, "label": te_y}).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2,
        id2label={0: "pas_fini", 1: "fini"}, label2id={"pas_fini": 0, "fini": 1})

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred = np.argmax(logits, axis=-1)
        tp = int(((pred == 1) & (labels == 1)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        acc = (tp + tn) / max(1, len(labels))
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        return {"accuracy": acc, "f1": f1,
                "cut_in": fp / max(1, fp + tn),   # fini predicted on pas_fini
                "miss": fn / max(1, fn + tp)}

    targs = TrainingArguments(
        output_dir=args.out,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        report_to="none",
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=dtr, eval_dataset=dte,
        processing_class=tok, compute_metrics=compute_metrics,
    )
    trainer.train()
    print("\n=== FINAL (best model) ===")
    print(trainer.evaluate())
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
