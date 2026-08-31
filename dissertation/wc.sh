#!/bin/bash
# Word count of the dissertation body (chapters 1-6).
# Excludes: tables, tikz, verbatim, equations, captions, LaTeX commands, comments.
cd "$(dirname "$0")"
python3 - <<'PY'
import re, io, glob
tot = 0
for f in sorted(glob.glob('chapters/ch*.tex')):
    t = io.open(f, encoding='utf-8').read()
    t = re.sub(r'^%.*$', '', t, flags=re.M)                                   # commentaires
    t = re.sub(r'\\begin\{(tabular|tikzpicture|verbatim|equation)\}.*?\\end\{\1\}', '', t, flags=re.S)
    t = re.sub(r'\\caption(\[[^\]]*\])?\{(?:[^{}]|\{[^{}]*\})*\}', '', t, flags=re.S)  # legendes
    t = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?', ' ', t)            # commandes
    n = len([w for w in t.split() if any(c.isalpha() for c in w)])
    tot += n
    print(f"  {f.split('/')[-1]:26} {n:5d}")
print(f"  {'-'*32}")
print(f"  {'CORPS (ch. 1-6)':26} {tot:5d}")
print(f"\n  Kent : 7 000 a 10 000, hors abstract, bibliographie et annexes.")
PY
