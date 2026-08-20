"""Semantic (lexical-only) baselines on the fini/pas_fini dataset.

Two CPU baselines, runnable in seconds (no GPU):
  1. FR heuristic veto: rule-based floor (spelling / trailing connector / open number).
  2. TF-IDF + LogReg: classical ML baseline.

Metrics (endpoint = predict "fini"):
  accuracy, F1(fini),
  cut-in rate  = P(predict fini | true pas_fini)  -> interruptions, LOWER better
  miss rate    = P(predict pas_fini | true fini)  -> over-holding
  hardcase hold-rate = recall of pas_fini on spelling/number prefixes (LOWER = cuts them)

Usage:
  python -m eou_detector.dataset.baselines --data data/lexical
"""
import argparse
import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .build_lexical_dataset import tokens, _TRAILING, _NUMBERS


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def heuristic_predict(text: str) -> int:
    """1 = fini (endpoint), 0 = pas_fini (hold). Veto -> hold."""
    toks = tokens(text)
    if not toks:
        return 1
    singles = 0
    for w in reversed(toks):
        if len(w) == 1 and w.isalpha():
            singles += 1
        else:
            break
    if singles >= 2:
        return 0
    if toks[-1] in _TRAILING:
        return 0
    if toks[-1] in _NUMBERS or toks[-1].isdigit():
        return 0
    return 1


def metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    cut_in = fp / max(1, fp + tn)   # predicted fini on a pas_fini
    miss = fn / max(1, fn + tp)     # predicted pas_fini on a fini
    return dict(acc=acc, f1=f1, cut_in=cut_in, miss=miss,
                tp=tp, tn=tn, fp=fp, fn=fn)


def hardcase_holdrate(rows, y_pred):
    """Among pas_fini hardcase prefixes, fraction correctly held (pred=0)."""
    out = {}
    for tag in ("spelling", "number"):
        idx = [i for i, r in enumerate(rows) if r["label"] == 0 and r["hardcase"] == tag]
        if idx:
            held = sum(1 for i in idx if y_pred[i] == 0)
            out[tag] = (held / len(idx), len(idx))
    return out


def show(name, rows, y_pred):
    y = [r["label"] for r in rows]
    m = metrics(y, y_pred)
    hc = hardcase_holdrate(rows, y_pred)
    print(f"\n=== {name} ===")
    print(f"accuracy      : {m['acc']*100:5.1f}%")
    print(f"F1 (fini)     : {m['f1']*100:5.1f}%")
    print(f"cut-in rate   : {m['cut_in']*100:5.1f}%   (predict fini on a pas_fini -> interruption)")
    print(f"miss rate     : {m['miss']*100:5.1f}%   (predict pas_fini on a fini -> over-hold)")
    print(f"confusion     : tp={m['tp']} tn={m['tn']} fp={m['fp']} fn={m['fn']}")
    for tag, (rate, n) in hc.items():
        print(f"hold-rate {tag:8}: {rate*100:5.1f}%  (n={n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lexical")
    args = ap.parse_args()

    train = load(os.path.join(args.data, "train.jsonl"))
    test = load(os.path.join(args.data, "test.jsonl"))
    print(f"train={len(train)}  test={len(test)}  (held-out calls)")

    # 1) heuristic floor
    y_pred_h = [heuristic_predict(r["text"]) for r in test]
    show("FR heuristic veto (floor)", test, y_pred_h)

    # 2) TF-IDF + Logistic Regression
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)),
        ("lr", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
    ])
    clf.fit([r["text"] for r in train], [r["label"] for r in train])
    y_pred_lr = clf.predict([r["text"] for r in test]).tolist()
    show("TF-IDF + LogReg (semantic baseline)", test, y_pred_lr)


if __name__ == "__main__":
    main()
