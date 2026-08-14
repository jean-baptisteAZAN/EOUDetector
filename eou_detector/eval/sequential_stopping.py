"""Sequential optimal-stopping via a time-aware calibrated probability (model-agnostic).

The eot-bench oracle reaches low interruptions by combining a threshold with
action-delay + timeout, i.e. a *temporal* policy. A flat threshold on the raw score
can't. We fold time into the probability: fit g([score, silence_dur]) -> P(eot),
so a single threshold on g is equivalent to a time-varying threshold on the raw
score = a sequential stopping rule. Works on ANY model's predictions.

Writes a new run with p_eot = g so `eot-harness compute-metrics` gives the frontier.

Usage:
  python -m eou_detector.eval.sequential_stopping --span-set <fr dir> --ref <model_dir>
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


def load_points(pred_path):
    p = pd.read_parquet(pred_path).copy()
    p["y"] = (p["label"].astype(str).str.lower() == "eot").astype(int)
    return p


def time_aware_oof(p):
    """OOF P(eot | score, silence_dur) via k-fold logistic (time folded into the prob)."""
    s = p["p_eot"].values.astype(float)
    d = p["silence_dur"].values.astype(float)
    X = np.c_[s, d, s * d, d * d]
    y = p["y"].values
    g = np.zeros(len(p))
    for tr, te in GroupKFold(5).split(X, y, p["id"].astype(str)):
        lr = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
        g[te] = lr.predict_proba(X[te])[:, 1]
    return g


def per_span_auc(p, col):
    """AUC at the score point (0.2s) per span, matching the harness metric."""
    q = p.copy()
    q["_d"] = (q["silence_dur"] - 0.2).abs()
    sp = q.sort_values("_d").groupby(["id", "span_index"], as_index=False).first()
    return roc_auc_score(sp["y"], sp[col])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span-set", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    pred = os.path.join(args.span_set, args.ref, "predictions.parquet")
    p = load_points(pred)
    p["g"] = time_aware_oof(p)

    print(f"model: {args.ref}")
    print(f"  per-span AUC  raw score      = {per_span_auc(p, 'p_eot'):.4f}")
    print(f"  per-span AUC  time-aware (g)  = {per_span_auc(p, 'g'):.4f}")

    out = p[["id", "language", "span_index", "timestamp", "silence_dur", "label"]].copy()
    out["p_eot"] = p["g"].values
    name = args.out_name or f"seqstop__{args.ref.split('__')[0]}"
    d = os.path.join(args.span_set, name)
    os.makedirs(d, exist_ok=True)
    out.to_parquet(os.path.join(d, "predictions.parquet"), index=False)
    man = json.load(open(os.path.join(args.span_set, args.ref, "manifest.json")))
    man["adapter_id"] = f"seqstop-timeaware-{man.get('adapter_id', args.ref)}"
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"))
    print(f"  wrote {d}/predictions.parquet -> compute-metrics for the frontier")


if __name__ == "__main__":
    main()
