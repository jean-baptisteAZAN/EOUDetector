"""In-distribution fusion, lexical = [agent context] + [caller causal text].

Adds the agent's prior turn(s) as context to the causal caller text before
embedding with CamemBERT. Rationale: whether "oui" ends a turn depends on what
the agent just asked -> context should make the completion signal both stronger
and earlier-confident (lower latency).

Compares to the no-context version (indist_fusion_camembert).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from .indist_fusion import per_span_ac, causal_text
from .indist_fusion_camembert import embed_texts, oof


def build_text_context_df(max_ctx_chars=240):
    from datasets import load_dataset
    ds = load_dataset("livekit/eot-bench-data", "fr", split="validation").remove_columns(["audio"])
    rows = []
    for r in ds:
        spans = r["silence_spans"]
        words = r.get("words") or []
        ctx = " ".join((m.get("content") or "") for m in (r.get("messages") or [])).strip()
        ctx = ctx[-max_ctx_chars:]
        last = len(spans) - 1
        for i, sp in enumerate(spans):
            caller = causal_text(words, sp["start"])
            text = (f"{ctx} [AGENT] {caller}" if ctx else caller).strip()
            rows.append({"id": r["id"], "span_index": i, "text": text,
                         "y": 0 if i < last else 1})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span-set", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out-name", default="indist_fusion_context")
    args = ap.parse_args()

    ac = per_span_ac(os.path.join(args.span_set, args.ref, "predictions.parquet"))
    df = ac.merge(build_text_context_df(), on=["id", "span_index"], how="inner")
    df["y"] = df["y"].astype(int)
    print(f"spans: {len(df)} (eot={df.y.sum()} hold={(df.y==0).sum()})")

    print("embedding [context + caller] with frozen CamemBERT-base ...")
    E = embed_texts(df["text"].tolist(), max_len=128)

    folds = list(GroupKFold(5).split(E, df["y"], df["id"].astype(str)))
    y = df["y"].values
    df["p_lex"] = oof(lambda tr, te: LogisticRegression(max_iter=2000).fit(E[tr], y[tr]).predict_proba(E[te])[:, 1], folds, len(df))

    Xf = np.c_[df["p_ac"], df["p_lex"], df["p_ac"] * df["p_lex"]]
    models, id_fold = [], {}

    def fus(tr, te):
        lr = LogisticRegression(max_iter=1000).fit(Xf[tr], y[tr])
        models.append(lr)
        for i in df.iloc[te]["id"].astype(str).unique():
            id_fold[i] = len(models) - 1
        return lr.predict_proba(Xf[te])[:, 1]
    df["p_fus"] = oof(fus, folds, len(df))

    print("\n=== per-span AUC (context-augmented, in-distribution) ===")
    print(f"  p_ac  SmartTurn                  = {roc_auc_score(y, df.p_ac):.4f}")
    print(f"  p_lex CamemBERT-emb + context    = {roc_auc_score(y, df.p_lex):.4f}")
    print(f"  fusion                            = {roc_auc_score(y, df.p_fus):.4f}")

    full = pd.read_parquet(os.path.join(args.span_set, args.ref, "predictions.parquet"))
    full = full.merge(df.set_index(["id", "span_index"])["p_lex"].rename("p_lex"),
                      on=["id", "span_index"], how="inner")
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
    man["adapter_id"] = "indist-fusion-cv-st-camembert-context"
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"))
    print(f"\nwrote {d}/predictions.parquet -> compute-metrics")


if __name__ == "__main__":
    main()
