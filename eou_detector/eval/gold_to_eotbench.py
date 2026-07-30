"""Convert our production gold export -> eot-bench dataset schema.

So the SAME harness (adapters, metrics, Pareto sweep) that runs on the public
eot-bench-data also runs on our French-medical gold set — no new eval code.

Input  : eou_calls.jsonl (from scripts/export_eou_dataset.sh) + audio/<uuid>.wav
Output : an eot-bench-format HuggingFace dataset (id, language, audio, silence_spans,
         messages), loadable by the harness with --path <out> --name fr.

eot-bench schema (per row = one user turn):
  id, language, audio {array, sampling_rate}, silence_spans [{start, end}] (last = eot,
  earlier = hold, sub-0.1s dropped), optional messages [{role, content}].

Label derivation from our raw signals (weak supervision, no manual annotation):
  a pause where the caller RESUMED (caller_resumed_after_ms not null) = `hold`;
  the pause that actually ended the turn (agent replied, no resume)   = `eot`.
  Consecutive holds + their terminating eot form one eot-bench "turn".

⚠️ UNTESTED until the first real gold export exists — validate field names + the
turn-grouping against a real sample, and confirm the harness's local `--path`
loader accepts the produced format (save_to_disk vs parquet vs push_to_hub).

Usage:
  python -m eou_detector.eval.gold_to_eotbench \
    --export data/eou_export_<ts>/eou_calls.jsonl \
    --audio-dir data/eou_export_<ts>/audio \
    --out data/gold_eotbench_fr
"""
import argparse
import json
import os

MIN_SILENCE_S = 0.1   # eot-bench drops spans shorter than this


def load_calls(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def group_turns(turns):
    """Group per-pause records into eot-bench turns: runs of `hold` ending in `eot`.

    A turn record is `hold` if the caller resumed after the decision, else `eot`."""
    groups, cur = [], []
    for t in sorted(turns, key=lambda x: x.get("turn_index", 0)):
        resumed = t.get("caller_resumed_after_ms") is not None
        cur.append(t)
        if not resumed:            # this pause ended the turn -> close the group
            groups.append(cur)
            cur = []
    if cur:                        # trailing holds with no closing eot -> still a group
        groups.append(cur)
    return groups


def silence_spans(group):
    """[{start,end}] in seconds; last span is the eot, earlier ones hold.

    Uses t_pause_start_ms -> t_decision_ms per pause (relative to call start)."""
    spans = []
    for t in group:
        start = t.get("t_pause_start_ms")
        end = t.get("t_decision_ms")
        if start is None or end is None:
            continue
        s, e = start / 1000.0, end / 1000.0
        if e - s >= MIN_SILENCE_S:
            spans.append({"start": round(s, 3), "end": round(e, 3)})
    return spans


def messages_from(group):
    """Causal text context = the caller_text of the pauses so far (last = current)."""
    msgs = []
    for t in group:
        txt = (t.get("caller_text") or "").strip()
        if txt:
            msgs.append({"role": "user", "content": txt})
    return msgs


def load_caller_audio(wav_path):
    """Caller (customer) channel as mono float32 + sampling rate."""
    import numpy as np
    import soundfile as sf
    data, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    caller = data[:, 0]          # customer channel = left (stereo customer+agent)
    return np.asarray(caller, dtype="float32"), int(sr)


def build_rows(export_path, audio_dir):
    rows = []
    for call in load_calls(export_path):
        uuid = call.get("uuid")
        turns = call.get("turns") or []
        wav = os.path.join(audio_dir, f"{uuid}.wav")
        audio = None
        if os.path.isfile(wav):
            arr, sr = load_caller_audio(wav)
            audio = {"array": arr, "sampling_rate": sr}
        for gi, group in enumerate(group_turns(turns)):
            spans = silence_spans(group)
            if not spans:
                continue
            rows.append({
                "id": f"{uuid}__{gi}",
                "language": "fr",
                "audio": audio,                 # None if the wav is missing
                "silence_spans": spans,          # last = eot, earlier = hold
                "messages": messages_from(group),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="eou_calls.jsonl")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", default="data/gold_eotbench_fr")
    args = ap.parse_args()

    rows = build_rows(os.path.expanduser(args.export), os.path.expanduser(args.audio_dir))
    n_audio = sum(1 for r in rows if r["audio"] is not None)
    print(f"built {len(rows)} eot-bench turns ({n_audio} with audio) from {args.export}")
    if not rows:
        print("no rows — check the export/audio paths and the turn signals")
        return

    from datasets import Dataset, Audio, Features, Value, Sequence
    features = Features({
        "id": Value("string"),
        "language": Value("string"),
        "audio": Audio(),
        "silence_spans": Sequence({"start": Value("float32"), "end": Value("float32")}),
        "messages": Sequence({"role": Value("string"), "content": Value("string")}),
    })
    # datasets.Audio expects paths or {array, sampling_rate}; keep as-is
    ds = Dataset.from_list(rows, features=features)
    os.makedirs(args.out, exist_ok=True)
    ds.save_to_disk(args.out)
    print(f"saved -> {args.out}  (validate load path with the harness on a real sample)")


if __name__ == "__main__":
    main()
