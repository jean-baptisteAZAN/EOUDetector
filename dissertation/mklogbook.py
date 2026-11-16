# logbook.md -> logbook.tex -> logbook.pdf (meme typographie que le memoire)
import io, re

src = io.open('logbook.md', encoding='utf-8').read()

# une entree datee, un intitule d'evaluation ou un label ouvre sa propre ligne
LOGICAL = re.compile(r'\*\*|(?:Goal|Written|Result|Next|Obstacle|Diagnosis|Files|Note):')

def esc(s):
    for a, b in (('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('$', r'\$'),
                 ('#', r'\#'), ('_', r'\_'), ('{', r'\{'), ('}', r'\}'),
                 ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}'),
                 ('·', r'$\cdot$'), ('—', '---')):
        s = s.replace(a, b)
    return s

def inline(s):
    code = []
    s = re.sub(r'`([^`]*)`', lambda m: code.append(m.group(1)) or '\x00%d\x00' % (len(code) - 1), s)
    s = esc(s)
    s = re.sub(r'\*\*(.+?)\*\*', lambda m: r'\textbf{' + m.group(1) + '}', s)
    s = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', lambda m: r'\textit{' + m.group(1) + '}', s)
    s = re.sub(r'"([^"]*)"', lambda m: '``' + m.group(1) + "''", s)
    return re.sub('\x00(\\d+)\x00',
                  lambda m: r'\texttt{' + esc(code[int(m.group(1))]) + '}', s)

out, lines, i = [], src.split('\n'), 0
while i < len(lines):
    ln = lines[i]
    if ln.startswith('# '):
        out.append(r'\section*{' + inline(ln[2:]) + '}')
    elif ln.startswith('## '):
        out.append(r'\subsection*{' + inline(ln[3:]) + '}')
    elif ln.startswith('---'):
        out.append(r'\vspace{1mm}\hrule\vspace{3mm}')
    elif ln.startswith('> '):
        buf = []
        while i < len(lines) and lines[i].startswith('> '):
            buf.append(lines[i][2:]); i += 1
        out.append(r'\begin{quote}\small ' + inline(' '.join(buf)) + r'\end{quote}')
        continue
    elif ln.startswith('|'):
        rows = []
        while i < len(lines) and lines[i].startswith('|'):
            cells = [c.strip() for c in lines[i].strip('|').split('|')]
            if not all(set(c) <= set('-: ') for c in cells):
                rows.append(cells)
            i += 1
        out.append(r'\begin{center}\begin{tabular}{lll}\hline')
        out.append(' & '.join(r'\textbf{' + inline(c) + '}' for c in rows[0]) + r' \\ \hline')
        for r in rows[1:]:
            out.append(' & '.join(inline(c) for c in r) + r' \\')
        out.append(r'\hline\end{tabular}\end{center}')
        continue
    elif ln.strip() == '':
        out.append('')
    else:
        buf = []
        while i < len(lines) and lines[i].strip() and lines[i][0] not in '#|>-':
            buf.append(lines[i]); i += 1
        groups = []
        for ln2 in buf:
            if not groups or LOGICAL.match(ln2):
                groups.append([ln2])
            else:
                groups[-1].append(ln2)
        out.append(' \\\\\n'.join(inline(' '.join(g)) for g in groups) + r'\par\medskip')
        continue
    i += 1

tex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[a4paper,margin=22mm]{geometry}
\usepackage{parskip}
\usepackage{fancyhdr}
\setlength{\headheight}{14.5pt}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small Log book --- Jean-Baptiste Azan}
\fancyhead[R]{\small COMP7300}
\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\sloppy
\begin{document}
""" + '\n'.join(out) + "\n\\end{document}\n"

io.open('logbook.tex', 'w', encoding='utf-8').write(tex)
print('logbook.tex written')
