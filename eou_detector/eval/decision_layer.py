"""Decision layer: calibration (ECE) + cost-sensitive operating point.

The eot-bench harness sweeps a policy and reports the Pareto frontier; it never
calibrates and never *derives* the operating point from a cost. This module adds
both, offline, from an existing model run:

  A. CALIBRATION: fit isotonic on a held-out split, report ECE before/after.
     (None of the leaderboard models report calibration.)
  B. COST-SENSITIVE OPERATING POINT: for a range of interruption/latency cost
     ratios, pick the min-cost point from the swept trade-off = the endpoint
     "dynamically" chosen by the cost, not a fixed magic threshold.

Usage:
  python -m eou_detector.eval.decision_layer \
    --run output/livekit__eot-bench-data__validation__min_silence_100ms/fr/smart_turn_audio_adapter__cb4f57229a
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


def ece(probs, labels, n_bins=10):
    """Expected Calibration Error (equal-width bins)."""
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        e += (mask.sum() / len(probs)) * abs(acc - conf)
    return float(e)


def per_span_scores(pred_path, score_point=0.2):
    """One (score, label) per silence span, taken at the score point."""
    p = pd.read_parquet(pred_path)
    p = p.copy()
    p["_d"] = (p["silence_dur"] - score_point).abs()
    sp = (p.sort_values("_d")
            .groupby(["id", "span_index"], as_index=False)
            .first())
    sp["y"] = (sp["label"].astype(str).str.lower() == "eot").astype(int)
    return sp[["id", "span_index", "p_eot", "y"]]


def calibrate_report(sp, seed=0):
    ids = sp["id"].astype(str).unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    test_ids = set(ids[: len(ids) // 5])
    tr = sp[~sp["id"].astype(str).isin(test_ids)]
    te = sp[sp["id"].astype(str).isin(test_ids)]

    iso = IsotonicRegression(out_of_bounds="clip").fit(tr["p_eot"], tr["y"])
    raw = te["p_eot"].values
    cal = iso.predict(raw)
    return {
        "n_train_spans": len(tr), "n_test_spans": len(te),
        "ece_raw": ece(raw, te["y"]), "ece_cal": ece(cal, te["y"]),
        "brier_raw": float(np.mean((raw - te["y"]) ** 2)),
        "brier_cal": float(np.mean((cal - te["y"]) ** 2)),
    }, iso


def cost_operating_points(tradeoff_path, ratios=(1, 3, 5, 10, 20, 50)):
    """Min-cost operating point per cost ratio lambda = C_cut / C_lat.

    total cost = C_cut * cutoff_rate + C_lat * mean_latency  (C_lat = 1)."""
    t = pd.read_parquet(tradeoff_path)
    rows = []
    for lam in ratios:
        c = lam * t["cutoff_rate"] + t["mean_latency"]
        b = t.loc[c.idxmin()]
        rows.append({
            "lambda": lam, "threshold": float(b["threshold"]),
            "action_delay": float(b["action_delay"]), "timeout": float(b["timeout"]),
            "cutoff_%": float(b["cutoff_rate"]) * 100,
            "latency_ms": float(b["mean_latency"]) * 1000,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a model run dir (with predictions.parquet + metrics/tradeoff.parquet)")
    args = ap.parse_args()

    pred = os.path.join(args.run, "predictions.parquet")
    trade = os.path.join(args.run, "metrics", "tradeoff.parquet")

    print(f"=== A. Calibration (held-out): {os.path.basename(args.run)} ===")
    sp = per_span_scores(pred)
    rep, _ = calibrate_report(sp)
    print(f"spans: train={rep['n_train_spans']}  test={rep['n_test_spans']}")
    print(f"ECE   raw = {rep['ece_raw']:.4f}   ->  calibrated = {rep['ece_cal']:.4f}"
          f"   ({100*(1-rep['ece_cal']/max(rep['ece_raw'],1e-9)):.0f}% lower)")
    print(f"Brier raw = {rep['brier_raw']:.4f}   ->  calibrated = {rep['brier_cal']:.4f}")

    print("\n=== B. Cost-sensitive operating point (from the swept trade-off) ===")
    print("higher lambda = interruptions more costly -> more conservative (higher threshold, lower cut-in, higher latency)")
    df = cost_operating_points(trade)
    print(df.to_string(index=False, formatters={
        "cutoff_%": "{:.1f}".format, "latency_ms": "{:.0f}".format,
        "threshold": "{:.2f}".format, "action_delay": "{:.1f}".format,
        "timeout": "{:.1f}".format}))


if __name__ == "__main__":
    main()
