"""Calibration study across the eot-bench leaderboard + a cost-optimal threshold check.

Point 1: ECE (raw vs calibrated) for EVERY model run in a span-set -> shows
         miscalibration is universal (none of the leaderboard models calibrate).
Point 2: for a reference model, check that a calibrated cost-derived threshold
         tracks the swept-oracle threshold (so the operating point can be chosen
         from a cost, not found by sweeping).

Usage:
  python -m eou_detector.eval.calibration_study \
    --span-set eot-bench/output/livekit__eot-bench-data__validation__min_silence_100ms/fr \
    --ref smart_turn_audio_adapter__cb4f57229a
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from .decision_layer import ece, per_span_scores


def _calibrated_ece(sp, seed=0):
    from sklearn.isotonic import IsotonicRegression
    ids = sp["id"].astype(str).unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    test = set(ids[: max(1, len(ids) // 5)])
    tr = sp[~sp["id"].astype(str).isin(test)]
    te = sp[sp["id"].astype(str).isin(test)]
    iso = IsotonicRegression(out_of_bounds="clip").fit(tr["p_eot"], tr["y"])
    return ece(te["p_eot"], te["y"]), ece(iso.predict(te["p_eot"]), te["y"]), iso


def point1_all_models(span_set):
    rows = []
    for name in sorted(os.listdir(span_set)):
        pred = os.path.join(span_set, name, "predictions.parquet")
        if not os.path.isfile(pred):
            continue
        try:
            man = json.load(open(os.path.join(span_set, name, "manifest.json")))
            label = man.get("adapter_id", name)
        except Exception:
            label = name
        try:
            sp = per_span_scores(pred)
            if sp["y"].nunique() < 2 or len(sp) < 20:
                continue
            raw, cal, _ = _calibrated_ece(sp)
            rows.append({"model": label, "ece_raw": raw, "ece_cal": cal, "n": len(sp)})
        except Exception as e:  # noqa: BLE001
            print(f"  (skip {name}: {e})")
    return pd.DataFrame(rows).sort_values("ece_raw", ascending=False)


def point2_cost_threshold(span_set, ref):
    sp = per_span_scores(os.path.join(span_set, ref, "predictions.parquet"))
    _, _, iso = _calibrated_ece(sp)
    # invert isotonic: smallest raw score whose calibrated prob >= target
    grid = np.linspace(0, 1, 1001)
    cal_grid = iso.predict(grid)

    def raw_for_calibrated(target):
        j = np.searchsorted(cal_grid, target)
        return float(grid[min(j, len(grid) - 1)])

    t = pd.read_parquet(os.path.join(span_set, ref, "metrics", "tradeoff.parquet"))
    rows = []
    for lam in (2, 3, 5, 10, 20):
        tau_cal = float(np.clip(1 - 1.0 / lam, 0, 1))     # calibrated-prob decision threshold
        raw_star = raw_for_calibrated(tau_cal)             # -> raw-score threshold
        c = lam * t["cutoff_rate"] + t["mean_latency"]
        oracle = t.loc[c.idxmin()]
        rows.append({
            "lambda": lam,
            "tau*_calib": round(tau_cal, 3),
            "thr_derived(raw)": round(raw_star, 3),
            "thr_oracle(raw)": round(float(oracle["threshold"]), 3),
            "oracle_cutoff_%": round(float(oracle["cutoff_rate"]) * 100, 1),
            "oracle_lat_ms": round(float(oracle["mean_latency"]) * 1000),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span-set", required=True)
    ap.add_argument("--ref", required=True)
    args = ap.parse_args()

    print("=== POINT 1 · calibration across the leaderboard (held-out ECE) ===")
    df = point1_all_models(args.span_set)
    print(df.to_string(index=False, formatters={
        "ece_raw": "{:.3f}".format, "ece_cal": "{:.3f}".format}))
    print(f"\n-> {(df['ece_raw'] > 0.05).sum()}/{len(df)} models have ECE_raw > 0.05 (miscalibrated); "
          f"calibration lowers ECE for {(df['ece_cal'] < df['ece_raw']).sum()}/{len(df)}.")

    print("\n=== POINT 2 · calibrated cost-derived threshold vs swept oracle ===")
    print("tau* = 1 - 1/lambda on the CALIBRATED prob, inverted to a raw-score threshold, vs the sweep's cost-optimal threshold:")
    print(point2_cost_threshold(args.span_set, args.ref).to_string(index=False))


if __name__ == "__main__":
    main()
