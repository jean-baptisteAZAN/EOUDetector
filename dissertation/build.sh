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
echo "  -> $(pwd)/diss.pdf"
