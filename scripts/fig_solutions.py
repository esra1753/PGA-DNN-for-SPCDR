# -*- coding: utf-8 -*-
"""Fig. 7: exact solution, PINN and PGA-DNN predictions for TP1, TP2 and TP3 (values of eps in the training
range; right column = smallest eps of the training range). The same models as in Table 4.
Also writes figures/solutions_PGA-DNN_PA-DNN.(png|pdf): PGA-DNN and PA-DNN predictions (seed 1234)."""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utils import FIGURES, P, load_model, predict, run_dir  # noqa: E402

plt.rcParams.update({'font.size': 14, 'axes.titlesize': 15, 'axes.labelsize': 15, 'xtick.labelsize': 13,
                     'ytick.labelsize': 13, 'legend.fontsize': 13, 'axes.titleweight': 'bold',
                     'axes.labelweight': 'bold'})
ROWS = [('TP1', [1e-1, 3e-3, 4e-4], 4123), ('TP2', [1e-1, 3e-3, 7e-4], 5678), ('TP3', [1e-1, 1e-3, 1e-5], 2468)]


def eps_label(e):
    k = int(np.floor(np.log10(e) + 1e-9))
    m = int(round(e / 10 ** k))
    if (k, m) == (-1, 1):
        return '0.1'
    return r'$10^{%d}$' % k if m == 1 else r'$%d\times10^{%d}$' % (m, k)


def plot(curves, name, seed_of):
    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    for r, (problem, eps_list, seed) in enumerate(ROWS):
        nets = [(label, load_model(run_dir(problem, method, seed_of(seed))), color, style)
                for label, method, color, style in curves]
        for c, e in enumerate(eps_list):
            ax = axes[r, c]
            w = 20 * np.sqrt(e) if problem == 'TP3' else 20 * e
            x = np.unique(np.concatenate((np.linspace(0, 1, 4001), np.linspace(0, min(w, 1), 2001),
                                          np.linspace(max(1 - w, 0), 1, 2001) if problem == 'TP2' else [])))
            ax.plot(x, P.u_exact(problem, x, e), '-', color='red', lw=4, label='Exact')
            for label, net, color, style in nets:
                ax.plot(x, predict(net, x, e), '--', color=color, lw=3, label=label,
                        **({'dashes': style} if style else {}))
            ax.set_title(r'%s for $\epsilon$=%s' % (problem, eps_label(e)))
            ax.set_xlim(0, 1)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
            ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(['0', '0.5', '1'])
            ax.set_xlabel('x')
            ax.set_ylabel('u(x)')
            ax.grid(True, color='0.88', lw=0.8)
            ax.legend(loc='upper center' if problem == 'TP2' else 'upper right', framealpha=1, edgecolor='0.3',
                      handlelength=2.5)
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    for ext, kw in (('png', {'dpi': 300}), ('pdf', {})):
        fig.savefig(os.path.join(FIGURES, '%s.%s' % (name, ext)), bbox_inches='tight', **kw)
    plt.close(fig)
    print('saved figures/%s.png / .pdf' % name)


plot([('PINN', 'PINN', '#EDB120', None), ('PGA-DNN', 'PGA-DNN', 'blue', (4, 3))],
     'solutions_PINN_PGA-DNN', lambda s: s)
plot([('PA-DNN', 'PA-DNN', '#EDB120', None), ('PGA-DNN', 'PGA-DNN', 'blue', (4, 3))],
     'solutions_PA-DNN_PGA-DNN', lambda s: 1234)
