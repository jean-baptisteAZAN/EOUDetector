"""Per-hard-case breakdown on the public eot-bench French subset.

The dissertation motivates the work with three cases in which silence is a poor
end-of-turn cue: a name being spelled out, digits given in blocks, and a date.
This script measures whether the acoustic baseline and the fusion actually differ
on those cases, by tagging each silence span from the causal caller text that
precedes it and computing a per-span AUC within each subgroup.

Subgroup sizes are reported alongside every AUC: on generic French these cases
are rare, so several groups are descriptive rather than conclusive.

Usage:
  python -m eou_detector.eval.hardcase_breakdown --span-set <SS> --ref <smart_turn_dir>
"""
import argparse
import os
import re

import pandas as pd
from sklearn.metrics import roc_auc_score

from eou_detector.dataset.build_lexical_dataset import hardcase_tag, normalize
from eou_detector.eval.indist_fusion import build_text_df, per_span_ac

# a date is a month name, or a day+month pattern, in the trailing text
_MONTHS = ("janvier fevrier février mars avril mai juin juillet aout août "
           "septembre octobre novembre decembre décembre").split()
_DATE = re.compile(r"\b(" + "|".join(_MONTHS) + r")\b")


def tag(text: str) -> str:
    t = normalize(text or "")
    if not t:
        return "empty"
    h = hardcase_tag(t)
    if h:
        return h
    if _DATE.search(t):
        return "date"
    return "plain"


def auc(df, col):
    if df["y"].nunique() < 2:
        return None
    return roc_auc_score(df["y"], df[col])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span-set", required=True)
    ap.add_argument("--ref", required=True, help="smart_turn run dir name inside --span-set")
    ap.add_argument("--fusion", default="indist_fusion_camembert")
    ap.add_argument("--score-point", type=float, default=0.2)
    a = ap.parse_args()

    txt = build_text_df()
    txt["case"] = txt["text"].map(tag)

    ac = per_span_ac(os.path.join(a.span_set, a.ref, "predictions.parquet"), a.score_point)
    fu = per_span_ac(os.path.join(a.span_set, a.fusion, "predictions.parquet"), a.score_point)
    fu = fu.rename(columns={"p_ac": "p_fused"})[["id", "span_index", "p_fused"]]

    df = txt.merge(ac[["id", "span_index", "p_ac"]], on=["id", "span_index"])
    df = df.merge(fu, on=["id", "span_index"])

    print(f"\nspans matched: {len(df)}   (score point {a.score_point}s)\n")
    print(f"{'case':10} {'n':>5} {'eot':>5} {'hold':>5} {'AUC acoustic':>13} {'AUC fusion':>11}")
    print("-" * 56)
    order = ["spelling", "number", "date", "plain", "empty"]
    for c in order + sorted(set(df["case"]) - set(order)):
        d = df[df["case"] == c]
        if d.empty:
            continue
        aa, af = auc(d, "p_ac"), auc(d, "p_fused")
        fmt = lambda v: f"{v:.3f}" if v is not None else "n/a"
        print(f"{c:10} {len(d):5d} {int(d['y'].sum()):5d} {int((1 - d['y']).sum()):5d} "
              f"{fmt(aa):>13} {fmt(af):>11}")
    print("-" * 56)
    aa, af = auc(df, "p_ac"), auc(df, "p_fused")
    print(f"{'ALL':10} {len(df):5d} {int(df['y'].sum()):5d} {int((1 - df['y']).sum()):5d} "
          f"{aa:13.3f} {af:11.3f}")


if __name__ == "__main__":
    main()
