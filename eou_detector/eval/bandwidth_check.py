"""Two checks behind the in-domain bandwidth explanation of Section 5.7.

(A) Saturation. The dissertation claims Smart Turn "saturates upward" on the
    production audio: it answers "finished" for almost every pause. This compares
    the distribution of p_ac in domain against the same model's distribution on
    the public benchmark, where it discriminates normally.

(B) Spectral content. Production recordings are telephone-band. This measures how
    much energy sits above the telephone cut-off, which is what the acoustic model
    was trained to hear and does not receive here.

Usage:
  python -m eou_detector.eval.bandwidth_check --limit 150
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import soundfile as sf

from eou_detector.eval.score_gold import collect, load_caller_16k
from eou_detector.eval.indist_fusion import per_span_ac


def describe(name, p):
    p = np.asarray(p, dtype=float)
    print(f"  {name:34} n={len(p):5d}  mean {p.mean():.3f}  sd {p.std():.3f}  "
          f"median {np.median(p):.3f}  >0.9 {100*(p > 0.9).mean():5.1f}%  "
          f"<0.1 {100*(p < 0.1).mean():4.1f}%  IQR {np.subtract(*np.percentile(p,[75,25])):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="eou_calls.jsonl")
    ap.add_argument("--audio-dir", default="data/gold/audio")
    ap.add_argument("--span-set", default="eot-bench/output/"
                    "livekit__eot-bench-data__validation__min_silence_100ms/fr")
    ap.add_argument("--ref", default="smart_turn_audio_adapter__cb4f57229a")
    ap.add_argument("--limit", type=int, default=150)
    a = ap.parse_args()

    print("\n=== (A) distribution of p_ac from Smart Turn v3 ===")
    _, p_gold, _, y_gold, n_calls, _ = collect(a.export, a.audio_dir, 8.0, a.limit, 100.0)
    bench = per_span_ac(os.path.join(a.span_set, a.ref, "predictions.parquet"))
    describe("benchmark (16 kHz, wideband)", bench["p_ac"])
    describe(f"in domain (8 kHz, {n_calls} calls)", p_gold)

    y_gold = np.asarray(y_gold)
    p_gold = np.asarray(p_gold, dtype=float)
    yb = (bench["label"] == "eot").astype(int).to_numpy()
    pb = bench["p_ac"].to_numpy()
    print("\n  separation between the two classes (mean p_ac):")
    print(f"    benchmark   eot {pb[yb==1].mean():.3f}   hold {pb[yb==0].mean():.3f}   "
          f"gap {pb[yb==1].mean()-pb[yb==0].mean():+.3f}")
    print(f"    in domain   eot {p_gold[y_gold==1].mean():.3f}   hold {p_gold[y_gold==0].mean():.3f}   "
          f"gap {p_gold[y_gold==1].mean()-p_gold[y_gold==0].mean():+.3f}")

    print("\n=== (B) spectral content of the production audio ===")
    wavs = sorted(glob.glob(os.path.join(a.audio_dir, "*.wav")))[:60]
    rows = []
    for w in wavs:
        info = sf.info(w)
        x = load_caller_16k(w)
        if len(x) < 16000:
            continue
        x = x[: 16000 * 20]
        f = np.fft.rfftfreq(len(x), 1 / 16000)
        P = np.abs(np.fft.rfft(x)) ** 2
        tot = P.sum()
        if tot <= 0:
            continue
        rows.append({"sr_file": info.samplerate,
                     "above_3400": P[f > 3400].sum() / tot,
                     "above_4000": P[f > 4000].sum() / tot,
                     "above_6000": P[f > 6000].sum() / tot})
    d = pd.DataFrame(rows)
    print(f"  {len(d)} recordings sampled")
    print(f"  native sample rate on disk : {sorted(d.sr_file.unique())}")
    for col, lab in (("above_3400", "above 3.4 kHz (telephone cut-off)"),
                     ("above_4000", "above 4 kHz  (Nyquist of an 8 kHz signal)"),
                     ("above_6000", "above 6 kHz")):
        print(f"  share of energy {lab:42} {100*d[col].mean():6.3f}%  (max {100*d[col].max():.3f}%)")


if __name__ == "__main__":
    main()
