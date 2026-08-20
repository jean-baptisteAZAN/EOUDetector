"""In-distribution lexical + fusion on eot-bench-fr (no gold needed).

Tests the hypothesis: our medical CamemBERT was OOD on generic French, so fusion
didn't help. If we train the lexical ON eot-bench French text (k-fold, no leakage),
is it in-distribution -> does fusion then beat acoustic-only?

  causal text @ span = the caller words spoken before the pause (from `words`).
  label               = last silence span = eot, earlier = hold.
  p_lex  = out-of-fold TF-IDF+LogReg on the causal text.
  p_eot  = out-of-fold LogReg fusion of [p_ac (SmartTurn), p_lex, p_ac*p_lex].

Writes a predictions.parquet (in the eot-bench schema) so `eot-harness
compute-metrics` gives the real Pareto frontier, comparable to the baselines.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


def causal_text(words, cut_s):
    return " ".join((w.get("word") or "") for w in words if w.get("end", 1e9) <= cut_s).strip()


def build_text_df():
    ds = load_dataset("livekit/eot-bench-data", "fr", split="validation").remove_columns(["audio"])
    rows = []
    for r in ds:
        spans = r["silence_spans"]
        words = r.get("words") or []
        last = len(spans) - 1
        for i, sp in enumerate(spans):
            rows.append({
                "id": r["id"], "span_index": i,
                "text": causal_text(words, sp["start"]),
                "y": 0 if i < last else 1,   # last span = eot
            })
    return pd.DataFrame(rows)


def per_span_ac(pred_path, score_point=0.2):
    p = pd.read_parquet(pred_path).copy()
    p["_d"] = (p["silence_dur"] - score_point).abs()
    sp = p.sort_values("_d").groupby(["id", "span_index"], as_index=False).first()
    return sp[["id", "span_index", "timestamp", "silence_dur", "language", "label", "p_eot"]].rename(
        columns={"p_eot": "p_ac"})


def oof_lexical(df, folds):
    p_lex = np.zeros(len(df))
    for tr, te in folds:
        clf = Pipeline([("tf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)),
                        ("lr", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))])
        clf.fit(df.iloc[tr]["text"], df.iloc[tr]["y"])
        p_lex[te] = clf.predict_proba(df.iloc[te]["text"])[:, 1]
    return p_lex


def oof_fusion(df, folds):
    """Return OOF fused prob + per-fold models + each id's fold (to score full points)."""
    p = np.zeros(len(df))
    X = np.c_[df["p_ac"], df["p_lex"], df["p_ac"] * df["p_lex"]]
    models, id_fold = [], {}
    for k, (tr, te) in enumerate(folds):
        lr = LogisticRegression(max_iter=1000).fit(X[tr], df.iloc[tr]["y"])
        p[te] = lr.predict_proba(X[te])[:, 1]
        models.append(lr)
        for i in df.iloc[te]["id"].astype(str).unique():
            id_fold[i] = k
    return p, models, id_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span-set", required=True)
    ap.add_argument("--ref", required=True, help="SmartTurn run dir name (for p_ac)")
    ap.add_argument("--out-name", default="indist_fusion_cv")
    args = ap.parse_args()

    ac = per_span_ac(os.path.join(args.span_set, args.ref, "predictions.parquet"))
    txt = build_text_df()
    df = ac.merge(txt, on=["id", "span_index"], how="inner")
    df["y"] = df["y"].astype(int)
    print(f"spans merged: {len(df)}  (eot={df.y.sum()} hold={(df.y==0).sum()})")

    gk = GroupKFold(n_splits=5)
    folds = list(gk.split(df, df["y"], df["id"].astype(str)))
    df["p_lex"] = oof_lexical(df, folds)
    df["p_fus"], models, id_fold = oof_fusion(df, folds)

    print("\n=== per-span AUC (in-distribution) ===")
    print(f"  p_ac  SmartTurn        = {roc_auc_score(df.y, df.p_ac):.4f}")
    print(f"  p_lex in-dist lexical  = {roc_auc_score(df.y, df.p_lex):.4f}")
    print(f"  fusion (learned)       = {roc_auc_score(df.y, df.p_fus):.4f}")

    # FULL-trajectory predictions: fuse at EVERY prediction point (per-point p_ac +
    # per-span p_lex, via that id's OOF fold model) so the harness computes real latency.
    full = pd.read_parquet(os.path.join(args.span_set, args.ref, "predictions.parquet"))
    lex_map = df.set_index(["id", "span_index"])["p_lex"]
    full = full.merge(lex_map.rename("p_lex"), on=["id", "span_index"], how="inner")
    full["fold"] = full["id"].astype(str).map(id_fold)
    full = full.dropna(subset=["fold"])
    p_ac = full["p_eot"].values
    Xf = np.c_[p_ac, full["p_lex"].values, p_ac * full["p_lex"].values]
    pe = np.zeros(len(full))
    for k, lr in enumerate(models):
        mask = (full["fold"].values == k)
        if mask.any():
            pe[mask] = lr.predict_proba(Xf[mask])[:, 1]
    out = full[["id", "language", "span_index", "timestamp", "silence_dur", "label"]].copy()
    out["p_eot"] = pe
    d = os.path.join(args.span_set, args.out_name)
    os.makedirs(d, exist_ok=True)
    out.to_parquet(os.path.join(d, "predictions.parquet"), index=False)
    man = json.load(open(os.path.join(args.span_set, args.ref, "manifest.json")))
    man["adapter_id"] = "indist-fusion-cv-st-tfidf"
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"))
    print(f"\nwrote {d}/predictions.parquet  -> run compute-metrics + compare-models")


if __name__ == "__main__":
    main()
