"""Standalone in-domain (gold) scoring, without the eot-bench harness.

The harness path needs the HuggingFace Audio feature, which in datasets 5.x
requires torchcodec; torchcodec does not load against ffmpeg 8. To avoid that,
this script reads the caller wavs directly with soundfile and scores each pause.

Per pause = one decision point. p_ac comes from Smart Turn on the caller audio
window ending at t_decision_ms; p_lex comes from a frozen CamemBERT embedding and
a k-fold logistic head on the caller text; the two are fused by a k-fold logistic.
Label: eot if the caller did not resume after the pause, else hold. Folds are split
by call so no call is in both train and test. Reports per-pause AUC for
acoustic-only, lexical-only and fusion, on the in-domain gold set.

Usage:
  python -m eou_detector.eval.score_gold --export eou_calls.jsonl \
    --audio-dir data/gold/audio [--limit N]
"""
import argparse
import json
import os
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from eou_detector.eou.acoustic import SmartTurnV3, find_smart_turn_model
from eou_detector.eval.indist_fusion_camembert import embed_texts, oof


def load_caller_16k(wav):
    """Caller channel (left of the stereo customer+agent recording) as float32 at 16 kHz."""
    data, sr = sf.read(wav, dtype="float32", always_2d=True)
    caller = data[:, 0]
    if sr != 16000:
        g = gcd(sr, 16000)
        caller = resample_poly(caller, 16000 // g, sr // g).astype(np.float32)
    return caller


def collect(export, audio_dir, window_s, limit, min_silence_ms):
    st = SmartTurnV3(find_smart_turn_model(), 16000, window_s)
    win = int(window_s * 16000)
    calls = [json.loads(l) for l in open(export, encoding="utf-8") if l.strip()]
    if limit:
        calls = calls[:limit]
    ids, p_ac, texts, y = [], [], [], []
    n_calls = n_missing = 0
    for c in calls:
        uuid = str(c.get("uuid"))
        wav = os.path.join(audio_dir, f"{uuid}.wav")
        if not os.path.isfile(wav):
            n_missing += 1
            continue
        audio = load_caller_16k(wav)
        n_calls += 1
        cumulative = []                       # causal caller text of the CURRENT turn
        for t in (c.get("turns") or []):
            frag = (t.get("caller_text") or "").strip()
            if frag:
                cumulative.append(frag)
            resumed = t.get("caller_resumed_after_ms") is not None
            td = t.get("t_decision_ms")
            tp = t.get("t_pause_start_ms")
            sil = t.get("silence_ms")
            if sil is None and td is not None and tp is not None:
                sil = td - tp
            # only real endpointing decisions: a genuine pause, not a micro-gap
            if td is None or sil is None or sil < min_silence_ms:
                if not resumed:
                    cumulative = []           # turn ended: reset even if not scored
                continue
            end = int(td / 1000.0 * 16000)
            start = max(0, end - win)
            window = audio[start:end]
            if window.size >= 800:
                pcm16 = np.clip(window * 32768.0, -32768, 32767).astype(np.int16)
                ids.append(uuid)
                p_ac.append(float(st.predict(pcm16)))
                texts.append(" ".join(cumulative))   # causal text of the current turn
                y.append(0 if resumed else 1)
            if not resumed:
                cumulative = []               # eot: next text is a new utterance
    return (np.array(ids), np.array(p_ac, float), texts, np.array(y, int),
            n_calls, n_missing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="eou_calls.jsonl")
    ap.add_argument("--audio-dir", default="data/gold/audio")
    ap.add_argument("--window-s", type=float, default=8.0)
    ap.add_argument("--min-silence-ms", type=float, default=100.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ids, p_ac, texts, y, n_calls, n_missing = collect(
        args.export, args.audio_dir, args.window_s, args.limit, args.min_silence_ms)
    print(f"calls used: {n_calls} (missing audio: {n_missing}) | pauses: {len(y)} "
          f"(eot={int(y.sum())} hold={int((y == 0).sum())})")
    if len(set(y)) < 2:
        print("only one class present; cannot score AUC")
        return

    folds = list(GroupKFold(5).split(p_ac, y, ids))

    # lexical: frozen CamemBERT embedding + k-fold logistic
    print("embedding caller text with frozen CamemBERT ...")
    E = embed_texts(texts, max_len=64)
    p_lex = oof(lambda tr, te: LogisticRegression(max_iter=2000)
                .fit(E[tr], y[tr]).predict_proba(E[te])[:, 1], folds, len(y))

    # fusion: k-fold logistic on [p_ac, p_lex, p_ac*p_lex]
    Xf = np.c_[p_ac, p_lex, p_ac * p_lex]
    p_fus = oof(lambda tr, te: LogisticRegression(max_iter=1000)
                .fit(Xf[tr], y[tr]).predict_proba(Xf[te])[:, 1], folds, len(y))

    print("\n=== in-domain (gold) per-pause AUC ===")
    print(f"  acoustic only (Smart Turn)      = {roc_auc_score(y, p_ac):.4f}")
    print(f"  lexical only (CamemBERT emb)     = {roc_auc_score(y, p_lex):.4f}")
    print(f"  fusion                            = {roc_auc_score(y, p_fus):.4f}")


if __name__ == "__main__":
    main()
