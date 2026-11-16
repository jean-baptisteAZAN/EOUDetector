#!/bin/bash
# Compile the dissertation and report anything worth looking at.
export PATH="/Library/TeX/texbin:/opt/homebrew/bin:/usr/local/bin:$PATH"
cd "$(dirname "$0")" || exit 1
rm -f diss.aux diss.bbl diss.blg diss.out diss.toc diss.lof diss.lot

pdflatex -interaction=nonstopmode -shell-escape diss.tex >/dev/null 2>&1
bibtex diss >bibtex.log 2>&1
pdflatex -interaction=nonstopmode -shell-escape diss.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode -shell-escape diss.tex >build.log 2>&1

echo "  pages           : $(grep -o '([0-9]* pages' build.log | tr -d '(')"
echo "  erreurs         : $(grep -c '^!' build.log)"
echo "  renvois ??      : $(grep -ci undefined build.log)"
echo "  avertissements  : $(grep -c 'Warning' build.log)"
echo "  debordement max : $(grep 'Overfull \\hbox' build.log | grep -oE '[0-9.]+pt too wide' | sort -rn | head -1)"
grep -m3 '^!' build.log

pdftotext -layout diss.pdf /tmp/diss.txt 2>/dev/null && python3 - <<'PY'
import re
pages=open('/tmp/diss.txt',encoding='utf-8',errors='ignore').read().split('\f')
bad=[]
for i,p in enumerate(pages,1):
    L=[l.rstrip() for l in p.split('\n') if l.strip()]
    P=[l.rstrip() for l in pages[i-2].split('\n') if l.strip()] if i>1 else []
    # ignorer les numeros de page isoles en fin de page
    while P and re.match(r'^\s*\d+\s*$', P[-1]): P.pop()
    if not L or not P: continue
    # le flottant doit venir AVANT la reprise du texte : on ne regarde que la
    # premiere ligne de contenu, apres l'en-tete courant
    body=[l for l in L if not re.match(r'^(Chapter|Appendix)\s', l)]
    if P[-1] and not re.search(r'[.!?:;]["”)]?\s*$',P[-1]) and body \
       and re.match(r'^(Table|Figure)\s+\d', body[0]): bad.append(f"{i-1}->{i}")
print("  phrase coupee par un flottant :", ", ".join(bad) if bad else "aucune")
PY
# --- abstract en fichier separe (livrable Moodle) ---
python3 - <<'PYEOF'
import io, re
s = io.open('diss.tex', encoding='utf-8').read()
i = s.index('\\section*{Abstract}') + len('\\section*{Abstract}')
j = s.index('\\cleardoublepage', i)
t = s[i:j]
t = re.sub(r'^%.*$', '', t, flags=re.M)
t = re.sub(r'\\citep?t?\{[^}]*\}', '', t)
t = re.sub(r'\$([^$]*)\$', r'\1', t)                       # math inline
t = (t.replace('\\,', ' ').replace('\\%', '%').replace('\\&', '&')
       .replace('\\times', 'x').replace('\\emph{', '').replace('~', ' '))
t = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{([^}]*)\})?', r'\3', t)
t = re.sub(r'[{}]', '', t)
paras = [re.sub(r'\s+', ' ', p).strip() for p in re.split(r'\n\s*\n', t) if p.strip()]
n = sum(len([w for w in p.split() if any(c.isalpha() for c in w)]) for p in paras)
out = ("Towards Semantically-Aware End-of-Utterance Detection for Conversational AI Systems\n"
       "Jean-Baptiste Azan - MSc Artificial Intelligence - University of Kent\n\n"
       "ABSTRACT\n\n" + "\n\n".join(paras) + "\n")
io.open('abstract.txt', 'w', encoding='utf-8').write(out)
print(f"  abstract.txt    : {n} mots (limite 250)")
PYEOF

# --- abstract en PDF (meme typographie que le memoire) ---
python3 - <<'PYEOF'
import io, re
s = io.open('diss.tex', encoding='utf-8').read()
i = s.index('\\section*{Abstract}') + len('\\section*{Abstract}')
body = s[i:s.index('\\cleardoublepage', i)]
body = re.sub(r'^%.*$', '', body, flags=re.M).strip()
tex = r"""\documentclass[12pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[a4paper,margin=25mm]{geometry}
\usepackage{amsmath}
\renewcommand\baselinestretch{1.15}
\pagestyle{empty}
\begin{document}
\begin{center}
{\large\bfseries Towards Semantically-Aware End-of-Utterance Detection\\for Conversational AI Systems}

\medskip
Jean-Baptiste Azan\\
MSc Artificial Intelligence, University of Kent
\end{center}

\vspace{6mm}
\begin{center}\textbf{Abstract}\end{center}

""" + body + "\n\\end{document}\n"
io.open('abstract.tex', 'w', encoding='utf-8').write(tex)
PYEOF
pdflatex -interaction=nonstopmode abstract.tex >/dev/null 2>&1
rm -f abstract.aux abstract.log
[ -f abstract.pdf ] && echo "  abstract.pdf    : ok"

echo "  -> $(pwd)/diss.pdf"
