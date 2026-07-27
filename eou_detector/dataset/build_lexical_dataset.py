"""Build a lexical fini/pas_fini dataset from historical call transcripts.

Input: the psql transcript export CSV (no header):
  col0 = row index, col1 = call uuid, col2 = internal id, col3 = callDate,
  col4 = transcript JSON  ([{role, message}, ...], roles: system|bot|user|tool_*).

Labelling = prefix-truncation on the *caller* (user) turns:
  - the full user turn  -> fini      (label 1)   (the turn ended, agent/tool followed)
  - sampled prefixes    -> pas_fini  (label 0)   (still speaking)

Normalisation mimics the Azure `recognizing` partials the model sees at inference:
lowercase, punctuation stripped, disfluencies kept. Light de-contamination strips
a leading verbatim echo of the previous bot turn (a known ASR artefact on some calls).

Weak supervision caveat: the "fini" label = where the CURRENT system ended the turn,
so it inherits that system's errors. Good enough for a first baseline; the definitive
labels come from the gold audio set.

Usage:
  python -m eou_detector.dataset.build_lexical_dataset --csv ~/Downloads/export_tr_complete.csv \
      --out data/lexical --max-prefixes 3
"""
import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter

# some transcript fields are large; set the field limit as high as the platform allows
_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(_limit)
        break
    except OverflowError:
        _limit = int(_limit / 10)
random.seed(0)

_KEEP = "abcdefghijklmnopqrstuvwxyzàâäéèêëïîôöùûüç0-9'\\- "
_PUNCT = re.compile(rf"[^{_KEEP}]")
_WS = re.compile(r"\s+")

# French connectors / hesitations: a turn trailing on one of these is unfinished.
_TRAILING = {
    "euh", "heu", "hum", "et", "ou", "donc", "alors", "mais", "car", "puis",
    "que", "qui", "de", "du", "le", "la", "les", "un", "une", "mon", "ma",
    "mes", "je", "j", "c'est", "a", "à", "au", "aux", "pour", "avec", "dans",
    "mon", "ton", "son",
}
_NUMBERS = {
    "zero", "zéro", "un", "une", "deux", "trois", "quatre", "cinq", "six",
    "sept", "huit", "neuf", "dix", "onze", "douze", "treize", "quatorze",
    "quinze", "seize", "vingt", "trente", "quarante", "cinquante", "soixante",
    "cent", "cents", "mille",
}


def normalize(text: str) -> str:
    t = _PUNCT.sub(" ", text.lower())
    return _WS.sub(" ", t).strip()


def tokens(text: str):
    return text.split()


def hardcase_tag(text: str):
    toks = tokens(text)
    if not toks:
        return None
    singles = 0
    for w in reversed(toks):
        if len(w) == 1 and w.isalpha():
            singles += 1
        else:
            break
    if singles >= 2:
        return "spelling"
    if any(w.isdigit() for w in toks) or (toks[-1] in _NUMBERS):
        return "number"
    return None


def decontaminate(user_norm: str, prev_bot_norm: str) -> str:
    """Strip a leading verbatim echo of the previous bot turn (ASR bleed)."""
    if not prev_bot_norm:
        return user_norm
    u, b = tokens(user_norm), tokens(prev_bot_norm)
    if len(b) >= 3 and len(u) > len(b) and u[: len(b)] == b:
        return " ".join(u[len(b):])
    return user_norm


def load_csv(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.reader(f):
            if len(row) < 5:
                continue
            uuid = row[1]
            tj = row[4] if len(row) == 5 else ",".join(row[4:])
            try:
                turns = json.loads(tj)
            except Exception:
                continue
            yield uuid, turns


def load_xlsx(path):
    import pandas as pd
    df = pd.ExcelFile(path).parse(0)
    col = "transcript" if "transcript" in df.columns else df.columns[-1]
    for i, v in enumerate(df[col]):
        if not isinstance(v, str):
            continue
        try:
            turns = json.loads(v)
        except Exception:
            continue
        yield f"xlsx:{os.path.basename(path)}:{i}", turns


def load_calls(csv_path=None, xlsx_path=None):
    if csv_path:
        yield from load_csv(csv_path)
    if xlsx_path:
        yield from load_xlsx(xlsx_path)


def prefixes(words, k):
    """Up to k informative prefix cut points, preferring near-the-end."""
    n = len(words)
    cands = [n - 1, n - 2, n // 2, 1]
    seen, out = set(), []
    for c in cands:
        if 1 <= c < n and c not in seen:
            seen.add(c)
            out.append(c)
        if len(out) >= k:
            break
    return out


def build(max_prefixes, csv_path=None, xlsx_path=None):
    examples = []  # (text, label, hardcase, uuid, kind)
    n_calls = n_user = n_contam = 0
    for uuid, turns in load_calls(csv_path, xlsx_path):
        n_calls += 1
        prev_bot = ""
        for t in turns:
            role = t.get("role")
            msg = t.get("message") or t.get("content") or ""
            if role == "bot":
                prev_bot = normalize(str(msg))
                continue
            if role != "user":
                continue
            n_user += 1
            norm = normalize(str(msg))
            clean = decontaminate(norm, prev_bot)
            if clean != norm:
                n_contam += 1
            words = tokens(clean)
            if len(words) < 2:
                continue
            # fini = the full turn
            examples.append((clean, 1, hardcase_tag(clean), uuid, "full"))
            # pas_fini = sampled prefixes
            for c in prefixes(words, max_prefixes):
                pref = " ".join(words[:c])
                examples.append((pref, 0, hardcase_tag(pref), uuid, "prefix"))
    return examples, dict(n_calls=n_calls, n_user=n_user, n_contam=n_contam)


def balance(examples):
    pos = [e for e in examples if e[1] == 1]
    neg = [e for e in examples if e[1] == 0]
    random.shuffle(neg)
    neg = neg[: len(pos)]  # downsample the majority (pas_fini)
    out = pos + neg
    random.shuffle(out)
    return out


def split_by_call(examples, test_frac=0.2):
    """Split by uuid so prefixes of one call never straddle train/test."""
    uuids = sorted({e[3] for e in examples})
    random.shuffle(uuids)
    n_test = int(len(uuids) * test_frac)
    test_ids = set(uuids[:n_test])
    train = [e for e in examples if e[3] not in test_ids]
    test = [e for e in examples if e[3] in test_ids]
    return train, test


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for text, label, hc, uuid, kind in rows:
            f.write(json.dumps({"text": text, "label": label,
                                "hardcase": hc, "uuid": uuid, "kind": kind},
                               ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--xlsx", default=None)
    ap.add_argument("--out", default="data/lexical")
    ap.add_argument("--max-prefixes", type=int, default=3)
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    if not args.csv and not args.xlsx:
        ap.error("provide --csv and/or --xlsx")
    csv_path = os.path.expanduser(args.csv) if args.csv else None
    xlsx_path = os.path.expanduser(args.xlsx) if args.xlsx else None
    os.makedirs(args.out, exist_ok=True)

    examples, stats = build(args.max_prefixes, csv_path, xlsx_path)
    bal = balance(examples)
    train, test = split_by_call(bal, args.test_frac)

    write_jsonl(os.path.join(args.out, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out, "test.jsonl"), test)

    def dist(rows):
        c = Counter(l for _, l, *_ in rows)
        return f"fini={c[1]} pas_fini={c[0]}"

    hc = Counter(e[2] for e in examples if e[2])
    print("=== BUILD STATS ===")
    print(f"calls parsed        : {stats['n_calls']}")
    print(f"user turns           : {stats['n_user']}")
    print(f"echo-decontaminated  : {stats['n_contam']}")
    print(f"raw examples         : {len(examples)}  ({dist(examples)})")
    print(f"balanced examples    : {len(bal)}  ({dist(bal)})")
    print(f"hardcases (raw)      : {dict(hc)}")
    print(f"train                : {len(train)}  ({dist(train)})")
    print(f"test (held-out calls): {len(test)}  ({dist(test)})")
    print(f"written              : {args.out}/train.jsonl , test.jsonl")


if __name__ == "__main__":
    main()
