# -*- coding: utf-8 -*-
"""Convergence history of the training loss L_d (Fig. 6 of the paper and the corresponding PA-DNN figure).

Solid line: geometric mean over the 4 seeds; shaded region: range over the seeds; dashed line: end of
pre-training. Outputs: figures/convergence_PGA-DNN.(png|pdf) (Fig. 6), figures/convergence_PA-DNN.(png|pdf)
and figures/convergence_PGA-DNN_vs_PA-DNN.(png|pdf).
"""

import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from utils import FIGURES, seed_dirs  # noqa: E402

COLORS = {'PGA-DNN': '#2a78d6', 'PA-DNN': '#eb6834'}
plt.rcParams.update({
    'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 15,
    'ytick.labelsize': 15, 'legend.fontsize': 16, 'font.weight': 'bold', 'axes.labelweight': 'bold',
    'axes.titleweight': 'bold', 'mathtext.default': 'bf', 'mathtext.fontset': 'dejavusans',
    'axes.edgecolor': '#c3c2b7', 'axes.linewidth': 0.8,
})


def read_history(folder):
    with open(os.path.join(folder, 'convergence_history.csv'), encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    it = np.array([int(r['iteration']) for r in rows])
    L_d = np.array([float(r['L_d']) for r in rows])
    a_pre = max(int(r['iteration']) for r in rows if r['phase'] == 'pretraining')
    return it, L_d, a_pre


def history(problem, method):
    data = [read_history(d) for d in seed_dirs(problem, method)]
    it, a_pre = data[0][0], data[0][2]
    Y = np.vstack([d[1] for d in data])
    return it, a_pre, np.exp(np.log(Y).mean(axis=0)), Y.min(axis=0), Y.max(axis=0), len(data)


def figure(methods, name):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5.4))
    for k, (problem, ax) in enumerate(zip(('TP1', 'TP2', 'TP3'), axs)):
        for m in methods:
            it, a_pre, gm, lo, hi, n = history(problem, m)
            ax.fill_between(it, lo, hi, color=COLORS[m], alpha=0.18, lw=0)
            ax.plot(it, gm, color=COLORS[m], lw=1.4)
        ax.axvline(a_pre, color='#898781', ls='--', lw=1.0)
        ax.set_yscale('log')
        ax.set_xlim(0, it.max())
        ax.set_xticks([0, 20000, 40000, 60000, 80000, 100000])
        ax.set_xlabel('Iteration')
        if k == 0:
            ax.set_ylabel(r'Training loss $L_d$')
        ax.grid(True, color='#e1e0d9', lw=0.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.set_title('(%s) %s' % ('abc'[k], problem), loc='left')
    handles, labels = [], []
    for m in methods:
        handles += [Line2D([0], [0], color=COLORS[m], lw=1.6)]
        labels += [(r'Training loss $L_d$ (geometric mean over %d seeds)' % n) if len(methods) == 1 else m]
    handles += [plt.Rectangle((0, 0), 1, 1, color='#898781', alpha=0.25, lw=0),
                Line2D([0], [0], color='#898781', ls='--', lw=1.2)]
    labels += ['Range over seeds', 'End of pre-training']
    fig.legend(handles, labels, loc='lower center', ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    os.makedirs(FIGURES, exist_ok=True)
    for ext, kw in (('png', {'dpi': 300}), ('pdf', {})):
        fig.savefig(os.path.join(FIGURES, '%s.%s' % (name, ext)), bbox_inches='tight', **kw)
    plt.close(fig)
    print('saved figures/%s.png / .pdf' % name)


figure(['PGA-DNN'], 'convergence_PGA-DNN')
figure(['PA-DNN'], 'convergence_PA-DNN')
figure(['PGA-DNN', 'PA-DNN'], 'convergence_PGA-DNN_vs_PA-DNN')
