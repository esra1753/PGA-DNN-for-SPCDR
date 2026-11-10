# -*- coding: utf-8 -*-
"""Table 1: ||E||_inf and ||E||_2 of PGA-DNN and PA-DNN (mean +- sample standard deviation over 4 seeds)."""

import numpy as np

from utils import P, errors, load_model, save_tex, seed_dirs, tex_eps, tex_num


def pm(m, s):
    """Mean +- std, the std written with the exponent of the mean (e.g. 5.63 x 10^-1 +- 0.64 x 10^-1)."""
    e = int(np.floor(np.log10(m)))
    if s >= 10 ** e:
        return '$%s \\pm %s$' % (tex_num(m), tex_num(s))
    return '$%s \\pm %.2f\\times10^{%d}$' % (tex_num(m), s / 10 ** e, e)


rows = []
for problem in ('TP1', 'TP2', 'TP3'):
    eps_list = P.SETTINGS[problem]['eps_test']
    res = {}
    for method in ('PGA-DNN', 'PA-DNN'):
        dirs = seed_dirs(problem, method)
        res[method] = np.array([[errors(problem, load_model(d), e) for e in eps_list] for d in dirs])
        print('%s %s: %d seeds' % (problem, method, len(dirs)))
    rows += ['\\hline', '\\multicolumn{5}{l}{\\textbf{%s}} \\\\' % problem, '\\hline']
    for j, e in enumerate(eps_list):
        cells = []
        for method in ('PGA-DNN', 'PA-DNN'):
            for n in (0, 1):
                v = res[method][:, j, n]
                cells.append(pm(v.mean(), v.std(ddof=1)))
        rows.append('%s & %s \\\\' % (tex_eps(e), ' & '.join(cells)))
        print('  eps %.0e  PGA %.2e  PA %.2e' % (e, res['PGA-DNN'][:, j, 0].mean(), res['PA-DNN'][:, j, 0].mean()))

save_tex('table1.tex', r'''\begin{table}[ht]
\centering
\caption{Maximum absolute errors $\|E\|_\infty$ and $L_2$ errors $\|E\|_2$ of PGA-DNN and PA-DNN for TP1, TP2 and TP3
(mean $\pm$ standard deviation over 4 independent runs with random seeds 1234, 2468, 4123 and 5678).
The errors are evaluated with respect to the exact solution on a grid that resolves the boundary layer(s).}
\label{tab:dnn_all}
\resizebox{\linewidth}{!}{%
\begin{tabular}{ccccc}
\hline
 & \multicolumn{2}{c}{PGA-DNN} & \multicolumn{2}{c}{PA-DNN} \\
\cline{2-3}\cline{4-5}
$\epsilon$ & $\|E\|_\infty$ & $\|E\|_2$ & $\|E\|_\infty$ & $\|E\|_2$ \\
''' + '\n'.join(rows[1:]) + r'''
\hline
\end{tabular}%
}
\end{table}
''')
