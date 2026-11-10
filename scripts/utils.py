# -*- coding: utf-8 -*-
"""Helpers shared by the table and figure scripts."""

import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
import problems as P  # noqa: E402

RESULTS = os.path.join(ROOT, 'results')
TABLES = os.path.join(ROOT, 'tables')
FIGURES = os.path.join(ROOT, 'figures')
SEEDS = (1234, 2468, 4123, 5678)


class Net(nn.Module):
    def __init__(self, layers=(2, 60, 60, 60, 1)):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

    def forward(self, x):
        for layer in self.linears[:-1]:
            x = torch.tanh(layer(x))
        return self.linears[-1](x)


def run_dir(problem, method, seed):
    return os.path.join(RESULTS, problem, method, 'seed%d' % seed)


def seed_dirs(problem, method):
    return sorted(glob.glob(os.path.join(RESULTS, problem, method, 'seed*')))


def load_model(folder, ckpt='it_100000.pt'):
    net = Net()
    net.load_state_dict(torch.load(os.path.join(folder, 'checkpoints', ckpt))['model'])
    net.eval()
    return net


def predict(net, x, eps):
    with torch.no_grad():
        return net(torch.from_numpy(np.column_stack((x, eps * np.ones_like(x)))).float()).numpy().ravel()


def errors(problem, net, eps):
    """||E||_inf and ||E||_2 (Eqs. 35-36, trapezoidal rule) on the test grid of Table 1."""
    x = P.test_grid(problem, eps)
    err = P.u_exact(problem, x, eps) - predict(net, x, eps)
    return np.max(np.abs(err)), np.sqrt(np.trapezoid(err ** 2, x))


def tex_num(v, digits=2):
    if v >= 1:
        return '%.*f' % (digits, v)
    e = int(np.floor(np.log10(v)))
    m = v / 10 ** e
    if round(m, digits) >= 10:
        m, e = m / 10, e + 1
    return '%.*f\\times10^{%d}' % (digits, m, e)


def tex_eps(e):
    k = int(np.floor(np.log10(e) + 1e-9))
    m = int(round(e / 10 ** k))
    return '$10^{%d}$' % k if m == 1 else '$%d\\times10^{%d}$' % (m, k)


def save_tex(name, text):
    os.makedirs(TABLES, exist_ok=True)
    path = os.path.join(TABLES, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print('saved', os.path.relpath(path, ROOT))
