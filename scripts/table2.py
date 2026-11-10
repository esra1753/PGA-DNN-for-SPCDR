# -*- coding: utf-8 -*-
"""Table 2: breakdown of the offline cost of PGA-DNN and PA-DNN (N_a = 200) for TP1, seed 1234."""

import csv
import os

from utils import run_dir, save_tex

RUNS = [('PGA-DNN', run_dir('TP1', 'PGA-DNN', 1234)), ('PA-DNN', run_dir('TP1', 'PA-DNN_Na200', 1234))]
data = {}
for name, d in RUNS:
    with open(os.path.join(d, 'timings.csv'), encoding='utf-8') as f:
        data[name] = {r['stage']: float(r['time_s']) for r in csv.DictReader(f)}

ROWS = [  # (stage in timings.csv, label in the table, bold)
    ('1) Initial data generation', '1) Initial data generation', True),
    ('a) IELDTM, 10 eps values x 500 points', '\\quad a) IELDTM, 10 $\\epsilon$ values $\\times$ 500 points', False),
    ('b) Collocation points (N_f = 10000)', '\\quad b) Collocation points ($N_f = 10000$)', False),
    ('2) Pre-training (A = 10000 iterations)', '2) Pre-training ($A = 10000$ iterations)', True),
    ('3) Adaptive training (90 steps)', '3) Adaptive training (90 steps)', True),
    ('a) Residual evaluation / selection of eps*', '\\quad a) Residual evaluation / selection of $\\epsilon^*$', False),
    ('b) Mesh generation', '\\quad b) Mesh generation', False),
    ('c) IELDTM solutions (90 solves, 18000 labels)', '\\quad c) IELDTM solutions (90 solves, 18000 labels)', False),
    ('d) Network updates (90000 iterations)', '\\quad d) Network updates (90000 iterations)', False),
    ('Total IELDTM time', 'Total IELDTM time', False),
    ('Total network training time', 'Total network training time', False),
    ('Total offline cost', 'Total offline cost', True),
]
lines = []
for key, label, bold in ROWS:
    v = [data[n][key] for n, _ in RUNS]
    cell = (lambda t: '%.3f' % t) if max(v) < 0.01 else (lambda t: '%.2f' % t)
    c = [cell(t) for t in v]
    if bold:
        label, c = '\\textbf{%s}' % label, ['\\textbf{%s}' % t for t in c]
    lines.append('%s & %s & %s \\\\' % (label, c[0], c[1]))
    if key == 'd) Network updates (90000 iterations)':
        lines.append('\\hline')
    print('%-50s %10.2f %10.2f' % (key, *v))

save_tex('table2.tex', r'''\begin{table}[ht]
\centering
\caption{Breakdown of the offline computational cost (in seconds) of the PGA-DNN and PA-DNN approaches for TP1.
Both models use the same network ($60\times3$, tanh), the same optimizer settings ($A = 10000$, $k = 1000$,
$k_{\max} = 100000$), and the same number of labeled data added during the adaptive phase (90 adaptive steps,
18000 labels): $N_a = 100$, $M = 100$ adaptive points per step for PGA-DNN and $N_a = 200$ uniform points per
step for PA-DNN.}
\label{tab:cost_tp1}
\begin{tabular}{lcc}
\hline
 & PGA-DNN & PA-DNN \\
Stage & ($N_a = 100$, $M = 100$, adaptive) & ($N_a = 200$, uniform) \\
\hline
''' + '\n'.join(lines) + r'''
\hline
\end{tabular}
\end{table}
''')
