"""In-distribution fusion with a CamemBERT-embedding lexical (vs the TF-IDF proxy).

Frozen CamemBERT-base -> mean-pooled embedding of the causal text -> k-fold LogReg
= a stronger in-distribution lexical than TF-IDF, no fine-tuning (CPU-feasible).
Fuse with SmartTurn p_ac and measure the frontier.

Usage:
  python -m eou_detector.eval.indist_fusion_camembert --span-set <fr dir> --ref <smartturn dir>
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel

from .indist_fusion import build_text_df, per_span_ac


def embed_texts(texts, model_id="almanach/camembert-base", batch=32, max_len=64):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = [t if t else "" for t in texts[i:i + batch]]
            enc = tok(chunk, return_tensors="pt", truncation=True, max_length=max_len, padding=True)
            h = model(**enc).last_hidden_state           # (B, T, 768)
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)   # masked mean-pool
            out.append(pooled.cpu().numpy())
            print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.concatenate(out, axis=0)


def oof(fit_predict, folds, n):
    p = np.zeros(n)
    for tr, te in folds:
        p[te] = fit_predict(tr, te)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span-set", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out-name", default="indist_fusion_camembert")
    args = ap.parse_args()

    ac = per_span_ac(os.path.join(args.span_set, args.ref, "predictions.parquet"))
    txt = build_text_df()
    df = ac.merge(txt, on=["id", "span_index"], how="inner")
    df["y"] = df["y"].astype(int)
    print(f"spans: {len(df)} (eot={df.y.sum()} hold={(df.y==0).sum()})")

    print("embedding causal text with frozen CamemBERT-base ...")
    E = embed_texts(df["text"].tolist())

    folds = list(GroupKFold(5).split(E, df["y"], df["id"].astype(str)))
    y = df["y"].values

    def lex_fp(tr, te):
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(E[tr], y[tr])
        return lr.predict_proba(E[te])[:, 1]
    df["p_lex"] = oof(lex_fp, folds, len(df))

    Xf = np.c_[df["p_ac"], df["p_lex"], df["p_ac"] * df["p_lex"]]
    models, id_fold = [], {}

    def fus_fp(tr, te):
        lr = LogisticRegression(max_iter=1000).fit(Xf[tr], y[tr])
        models.append(lr)
        for i in df.iloc[te]["id"].astype(str).unique():
            id_fold[i] = len(models) - 1
        return lr.predict_proba(Xf[te])[:, 1]
    df["p_fus"] = oof(fus_fp, folds, len(df))

    print("\n=== per-span AUC (in-distribution) ===")
    print(f"  p_ac  SmartTurn            = {roc_auc_score(y, df.p_ac):.4f}")
    print(f"  p_lex CamemBERT-emb        = {roc_auc_score(y, df.p_lex):.4f}")
    print(f"  fusion (CamemBERT)         = {roc_auc_score(y, df.p_fus):.4f}")

    # full-trajectory predictions for the real frontier
    full = pd.read_parquet(os.path.join(args.span_set, args.ref, "predictions.parquet"))
    lex_map = df.set_index(["id", "span_index"])["p_lex"]
    full = full.merge(lex_map.rename("p_lex"), on=["id", "span_index"], how="inner")
    full["fold"] = full["id"].astype(str).map(id_fold)
    full = full.dropna(subset=["fold"])
    pa = full["p_eot"].values
    Xff = np.c_[pa, full["p_lex"].values, pa * full["p_lex"].values]
    pe = np.zeros(len(full))
    for k, lr in enumerate(models):
        mk = full["fold"].values == k
        if mk.any():
            pe[mk] = lr.predict_proba(Xff[mk])[:, 1]
    out = full[["id", "language", "span_index", "timestamp", "silence_dur", "label"]].copy()
    out["p_eot"] = pe
    d = os.path.join(args.span_set, args.out_name)
    os.makedirs(d, exist_ok=True)
    out.to_parquet(os.path.join(d, "predictions.parquet"), index=False)
    man = json.load(open(os.path.join(args.span_set, args.ref, "manifest.json")))
    man["adapter_id"] = "indist-fusion-cv-st-camembert-emb"
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"))
    print(f"\nwrote {d}/predictions.parquet -> compute-metrics")


if __name__ == "__main__":
    main()
