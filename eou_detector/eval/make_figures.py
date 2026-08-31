"""Generate the dissertation figures from the eot-bench outputs.

Three vector figures, written to dissertation/figures/:
  fig_pareto.pdf       latency vs cut-in frontier, acoustic / fusion / best commercial
  fig_reliability.pdf  reliability diagram of the fused probability, raw and calibrated
  fig_cost.pdf         cost-derived operating points, on the same axes as the frontier

Usage:
  python -m eou_detector.eval.make_figures --span-set <SS> --out dissertation/figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from eou_detector.eval.indist_fusion import per_span_ac

# validated categorical slots 1-3 (all-pairs, light surface)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1a1a19", "#5c5b55", "#dcdcd6"

RUNS = [
    ("smart_turn_audio_adapter__cb4f57229a", "Smart Turn v3 (acoustic)", BLUE,  "o"),
    ("indist_fusion_camembert",              "Fusion (this work)",       ORANGE, "s"),
    ("live_kit_turn_detector_adapter__turn_detector_v1",
                                             "LiveKit v1 (commercial)",  AQUA,  "^"),
]


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def frontier(path):
    """Pareto-optimal (cut-in rate, latency) points, ascending in cut-in rate."""
    d = pd.read_parquet(path)
    d = d[d["is_pareto"]].sort_values("cutoff_rate")
    return d["cutoff_rate"] * 100.0, d["mean_latency"] * 1000.0


def fig_pareto(ss, out):
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for run, label, colour, marker in RUNS:
        p = os.path.join(ss, run, "metrics", "tradeoff.parquet")
        if not os.path.isfile(p):
            continue
        x, y = frontier(p)
        ax.plot(x, y, lw=1.6, color=colour, label=label, zorder=3)
        ax.plot(x[::6], y[::6], marker, ms=4.0, color=colour, mec="white", mew=0.7, zorder=4)
    ax.axvline(5, lw=0.9, ls=(0, (3, 3)), color=MUTED, zorder=1)
    ax.text(4.8, 1700, "5% budget", fontsize=7.5, color=MUTED, ha="right")
    ax.set_xlim(0, 20); ax.set_ylim(200, 1800)
    ax.set_xlabel("Cut-in rate (%)  →  worse")
    ax.set_ylabel("Mean latency (ms)  →  worse")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.savefig(os.path.join(out, "fig_pareto.pdf"))
    plt.close(fig)


def fig_reliability(ss, out, bins=10):
    sp = per_span_ac(os.path.join(ss, "indist_fusion_camembert", "predictions.parquet"))
    y = (sp["label"] == "eot").astype(int).to_numpy()
    p = sp["p_ac"].to_numpy()                      # per_span_ac names the score column p_ac

    rng = np.random.default_rng(0)                 # held-out split for the calibrator
    idx = rng.permutation(len(p))
    fit, held = idx[: len(p) // 2], idx[len(p) // 2:]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p[fit], y[fit])
    q = iso.predict(p[held])

    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    ax.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=1.0, color=MUTED, zorder=2)
    ax.text(0.42, 0.33, "perfect calibration", fontsize=7.5, color=MUTED, rotation=45)

    for probs, truth, label, colour, marker in [
        (p[held], y[held], "Raw fused probability", BLUE, "o"),
        (q,       y[held], "After isotonic recalibration", ORANGE, "s"),
    ]:
        edges = np.linspace(0, 1, bins + 1)
        b = np.clip(np.digitize(probs, edges) - 1, 0, bins - 1)
        xs, ys = [], []
        for k in range(bins):
            m = b == k
            if m.sum() >= 10:
                xs.append(probs[m].mean())
                ys.append(truth[m].mean())
        ax.plot(xs, ys, lw=1.6, color=colour, label=label, zorder=3)
        ax.plot(xs, ys, marker, ms=4.5, color=colour, mec="white", mew=0.7, zorder=4)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("Predicted probability of end of turn")
    ax.set_ylabel("Observed frequency")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.savefig(os.path.join(out, "fig_reliability.pdf"))
    plt.close(fig)


def fig_cost(ss, out, lambdas=(1, 5, 10, 50)):
    d = pd.read_parquet(os.path.join(ss, "indist_fusion_camembert", "metrics", "tradeoff.parquet"))
    fx, fy = frontier(os.path.join(ss, "indist_fusion_camembert", "metrics", "tradeoff.parquet"))

    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    ax.plot(fx, fy, lw=1.4, color=GRID, zorder=2)
    ax.text(13.5, 560, "swept frontier", fontsize=7.5, color=MUTED)

    xs, ys = [], []
    for lam in lambdas:
        j = (lam * d["cutoff_rate"] + d["mean_latency"]).idxmin()
        xs.append(d.loc[j, "cutoff_rate"] * 100.0)
        ys.append(d.loc[j, "mean_latency"] * 1000.0)
    ax.plot(xs, ys, lw=1.6, color=ORANGE, zorder=3)
    ax.plot(xs, ys, "s", ms=5, color=ORANGE, mec="white", mew=0.8, zorder=4)
    for lam, x, y in zip(lambdas, xs, ys):
        ax.annotate(rf"$\lambda={lam}$", (x, y), textcoords="offset points",
                    xytext=(9, 4), fontsize=8.5, color=INK)

    ax.set_xlim(0, 20); ax.set_ylim(200, 1750)
    ax.set_xlabel("Cut-in rate (%)  →  worse")
    ax.set_ylabel("Mean latency (ms)  →  worse")
    fig.savefig(os.path.join(out, "fig_cost.pdf"))
    plt.close(fig)
    for lam, x, y in zip(lambdas, xs, ys):
        print(f"  lambda={lam:<3} cut-in {x:5.1f}%   latency {y:6.0f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--span-set", required=True)
    ap.add_argument("--out", default="dissertation/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    style()
    fig_pareto(a.span_set, a.out)
    fig_reliability(a.span_set, a.out)
    fig_cost(a.span_set, a.out)
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
