# -*- coding: utf-8 -*-
"""Table 4: PGA-DNN compared with a parametric PINN and a data-driven DNN trained on fixed data (DD).
For each problem all models use the same architecture, optimizer settings, iterations and random seed
(the seed with the smallest mean log10 ||E||_inf of PGA-DNN)."""

from utils import P, errors, load_model, run_dir, save_tex, tex_eps, tex_num

SEED = {'TP1': 4123, 'TP2': 5678, 'TP3': 2468}
METHODS = ('PGA-DNN', 'PINN', 'DD')

rows = []
for problem in ('TP1', 'TP2', 'TP3'):
    nets = {m: load_model(run_dir(problem, m, SEED[problem])) for m in METHODS}
    rows += ['\\hline', '\\multicolumn{7}{l}{\\textbf{%s}} \\\\' % problem, '\\hline']
    print('%s (seed %d)' % (problem, SEED[problem]))
    for e in P.SETTINGS[problem]['eps_test']:
        res = {m: errors(problem, nets[m], e) for m in METHODS}
        rows.append('%s & %s \\\\' % (tex_eps(e), ' & '.join('$%s$' % tex_num(v) for m in METHODS for v in res[m])))
        print('  eps %.0e  ' % e + '  '.join('%s %.2e' % (m, res[m][0]) for m in METHODS))

save_tex('table4.tex', r'''\begin{table}[ht]
\centering
\caption{Maximum absolute errors $\|E\|_\infty$ and $L_2$ errors $\|E\|_2$ of PGA-DNN, a conventional PINN
and a data-driven DNN trained on a fixed data set (DD) for TP1, TP2 and TP3. For each problem, all models
are trained with the same network architecture, optimizer settings, number of iterations and random seed.}
\label{tab:comparison}
\resizebox{\linewidth}{!}{%
\begin{tabular}{ccccccc}
\hline
 & \multicolumn{2}{c}{PGA-DNN} & \multicolumn{2}{c}{PINN} & \multicolumn{2}{c}{DD} \\
\cline{2-3}\cline{4-5}\cline{6-7}
$\epsilon$ & $\|E\|_\infty$ & $\|E\|_2$ & $\|E\|_\infty$ & $\|E\|_2$ & $\|E\|_\infty$ & $\|E\|_2$ \\
''' + '\n'.join(rows[1:]) + r'''
\hline
\end{tabular}%
}
\end{table}
''')
