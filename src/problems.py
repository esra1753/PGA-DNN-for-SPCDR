# -*- coding: utf-8 -*-
"""
Test problems TP1-TP3 and the IELDTM solvers used to generate the labelled data.

    TP1: -eps u'' - 2u' = 0,                     u(0) = 1, u(1) = 0
    TP2: -eps u'' + 2(2x - 1)u' + 4u = 0,         u(0) = 1, u(1) = 1
    TP3: -eps u'' + u(1 + u) = exp(-2x/sqrt(eps)), u(0) = 1, u(1) = exp(-1/sqrt(eps))

The IELDTM uses local Taylor expansions of order K = 12 at every mesh node and enforces C0 and C1
continuity at x_i + (1 - theta) dx_i with theta = 0.5 (Section 2.2 of the paper).
"""

from math import factorial

import numpy as np
import torch

TAYLOR_ORDER = 12
THETA = 0.5


# ----------------------------------------------------------------------------------------------
# IELDTM solvers
# ----------------------------------------------------------------------------------------------

def ieldtm_tp1(x, eps, order=TAYLOR_ORDER):
    """IELDTM for TP1. Returns the nodal values u(x_i).

    For TP1 the recursion (Eq. 11) gives U_i(k) = c_k U_i(1) for k >= 1, so the linear system for the
    unknowns [U_0(1), U_1(0), U_1(1), ..., U_{M-1}(0), U_{M-1}(1), U_M(1)] is assembled directly.
    Rows 0..M-1: C0 continuity (Eq. 9); rows M..2M-1: C1 continuity (Eq. 10).
    """
    p = np.sort(x)
    dx = np.diff(x)
    M = len(x) - 1

    c = np.zeros(order + 1)
    c[1] = 1.0
    for n in range(order - 1):
        c[n + 2] = (2 * (n + 1) * c[n + 1]) / (-eps * (n + 2) * (n + 1))

    ha = -THETA * dx               # expansion around x_{i+1}: (-theta dx_i)
    hb = (1 - THETA) * dx          # expansion around x_i: ((1 - theta) dx_i)
    S1a = np.zeros(M); S1b = np.zeros(M)
    S2a = np.zeros(M); S2b = np.zeros(M)
    for j in range(order + 1):
        S1a += c[j] * (ha ** j)
        S1b += c[j] * (hb ** j)
    for j in range(order):
        S2a += c[j + 1] * (j + 1) * (ha ** j)
        S2b += c[j + 1] * (j + 1) * (hb ** j)

    col0 = np.arange(M + 1) * 2 - 1                               # column of U_i(0), i = 1..M-1
    col1 = np.concatenate(([0], np.arange(1, M) * 2, [2 * M - 1]))  # column of U_i(1)
    A = np.zeros((2 * M, 2 * M))
    rows = np.arange(M)
    inner_l = rows >= 1
    inner_r = rows + 1 <= M - 1
    A[rows[inner_l], col0[rows[inner_l]]] = -1.0
    A[rows[inner_r], col0[rows[inner_r] + 1]] = 1.0
    A[rows, col1[rows + 1]] += S1a
    A[rows, col1[rows]] -= S1b
    A[M + rows, col1[rows + 1]] += S2a
    A[M + rows, col1[rows]] -= S2b

    b = np.zeros(2 * M)
    b[0] = 1                       # u(0) = 1
    b[M - 1] = 0
    z = np.linalg.solve(A, b)

    u = np.zeros(M + 1)
    u[1:] = z[1:M * 2:2]
    u[0] = 1
    u[M] = 0
    return u


def ieldtm_linear(p, eps, P, Q, F, uL, uR, order=TAYLOR_ORDER, theta=THETA):
    """General linear IELDTM:  -eps u'' + p(x) u' + q(x) u = f(x),  u(0) = uL, u(1) = uR.

    P, Q, F: (n, order + 1) arrays with the Taylor coefficients of p, q and f at every node.
    Recursion (Eq. 11): eps (k+1)(k+2) U(k+2) = sum_m P(m)(k-m+1) U(k-m+1) + sum_m Q(m) U(k-m) - F(k).
    Returns the (n, order + 1) local Taylor coefficients U_i(k).
    """
    n = len(p)
    al = np.zeros((n, order + 1)); be = np.zeros((n, order + 1)); ga = np.zeros((n, order + 1))
    al[:, 0] = 1.0
    be[:, 1] = 1.0
    for k in range(order - 1):
        sa = np.zeros(n); sb = np.zeros(n); sg = -F[:, k].copy()
        for m in range(k + 1):
            sa += P[:, m] * (k - m + 1) * al[:, k - m + 1] + Q[:, m] * al[:, k - m]
            sb += P[:, m] * (k - m + 1) * be[:, k - m + 1] + Q[:, m] * be[:, k - m]
            sg += P[:, m] * (k - m + 1) * ga[:, k - m + 1] + Q[:, m] * ga[:, k - m]
        d = eps * (k + 1) * (k + 2)
        al[:, k + 2] = sa / d
        be[:, k + 2] = sb / d
        ga[:, k + 2] = sg / d

    dx = np.diff(p)
    hb = (1 - theta) * dx
    ha = -theta * dx
    kk = np.arange(order + 1)

    def val_der(coef, h):
        pw = h[:, None] ** kk[None, :]
        dpw = np.zeros_like(pw)
        dpw[:, 1:] = kk[None, 1:] * h[:, None] ** (kk[None, 1:] - 1)
        return (coef * pw).sum(1), (coef * dpw).sum(1)

    M = n - 1
    A = np.zeros((2 * n, 2 * n))
    b = np.zeros(2 * n)
    i = np.arange(M)
    for coef, off in ((al, 0), (be, 1)):
        vL, dL = val_der(coef[:-1], hb)
        vR, dR = val_der(coef[1:], ha)
        A[2 * i, 2 * i + off] += vL
        A[2 * i, 2 * (i + 1) + off] -= vR
        A[2 * i + 1, 2 * i + off] += dL
        A[2 * i + 1, 2 * (i + 1) + off] -= dR
    vL, dL = val_der(ga[:-1], hb)
    vR, dR = val_der(ga[1:], ha)
    b[2 * i] = -(vL - vR)
    b[2 * i + 1] = -(dL - dR)
    A[2 * M, 0] = 1.0
    b[2 * M] = uL
    A[2 * M + 1, 2 * M] = 1.0
    b[2 * M + 1] = uR
    z = np.linalg.solve(A, b)
    return al * z[0::2][:, None] + be * z[1::2][:, None] + ga


def ieldtm_coefficients(problem, p, eps, order=TAYLOR_ORDER, tol=1e-12, maxit=50):
    """Local Taylor coefficients (n, order + 1) of the IELDTM solution on the sorted mesh p."""
    n = len(p)
    P = np.zeros((n, order + 1)); Q = np.zeros((n, order + 1)); F = np.zeros((n, order + 1))
    if problem == 'TP1':
        P[:, 0] = -2.0
        return ieldtm_linear(p, eps, P, Q, F, 1.0, 0.0, order)
    if problem == 'TP2':
        P[:, 0] = 2 * (2 * p - 1)
        P[:, 1] = 4.0
        Q[:, 0] = 4.0
        return ieldtm_linear(p, eps, P, Q, F, 1.0, 1.0, order)
    # TP3: Newton linearisation  -eps u'' + (1 + 2 u_n) u = exp(-2x/sqrt(eps)) + u_n^2,  u_0 = 0.
    # The Taylor coefficients of u_n are those of the previous iterate (products = Cauchy products).
    s = np.sqrt(eps)
    fk = np.array([(-2 / s) ** k / factorial(k) for k in range(order + 1)], dtype=np.float64)
    F0 = np.exp(-2 * p / s)[:, None] * fk[None, :]
    Un = np.zeros((n, order + 1))
    for _ in range(maxit):
        Q = 2 * Un
        Q[:, 0] += 1.0
        F = F0.copy()
        for k in range(order + 1):
            F[:, k] += (Un[:, :k + 1] * Un[:, k::-1]).sum(1)
        U = ieldtm_linear(p, eps, P, Q, F, 1.0, np.exp(-1 / s), order)
        done = np.max(np.abs(U[:, 0] - Un[:, 0])) < tol
        Un = U
        if done:
            break
    return Un


def ieldtm(problem, x, eps):
    """IELDTM nodal solution u(x_i) of the given problem on the mesh x (used for the training labels)."""
    if problem == 'TP1':
        return ieldtm_tp1(np.asarray(x, dtype=np.float64), eps)
    p = np.sort(np.asarray(x, dtype=np.float64))
    return ieldtm_coefficients(problem, p, eps)[:, 0]


def evaluate_taylor(p, U, xq, theta=THETA):
    """Evaluates the piecewise Taylor representation of an IELDTM solution at the points xq."""
    i = np.clip(np.searchsorted(p, xq, side='right') - 1, 0, len(p) - 2)
    split = p[i] + (1 - theta) * (p[i + 1] - p[i])
    j = np.where(xq <= split, i, i + 1)
    return (U[j] * (xq - p[j])[:, None] ** np.arange(U.shape[1])[None, :]).sum(1)


# ----------------------------------------------------------------------------------------------
# Problem definitions
# ----------------------------------------------------------------------------------------------

def u_exact(problem, x, eps):
    if problem == 'TP1':
        return (np.exp(-2 * x / eps) - np.exp(-2 / eps)) / (1 - np.exp(-2 / eps))
    if problem == 'TP2':
        return np.exp(-2 * x * (1 - x) / eps)
    return np.exp(-x / np.sqrt(eps))


def residual(problem, g, u, u_x, u_xx):
    """Pointwise residual of the governing equation (Eq. 21); g[:, 0] = x, g[:, 1] = eps."""
    eps = g[:, [1]]
    x = g[:, [0]]
    if problem == 'TP1':
        return -eps * u_xx - 2 * u_x
    if problem == 'TP2':
        return -eps * u_xx + 2 * (2 * x - 1) * u_x + 4 * u
    return -eps * u_xx + u * (1 + u) - torch.exp(-2 * x / torch.sqrt(eps))


def boundary_values(problem, eps):
    """u(0) and u(1) for an array of eps values (column vectors)."""
    left = np.ones_like(eps)
    if problem == 'TP1':
        return left, np.zeros_like(eps)
    if problem == 'TP2':
        return left, np.ones_like(eps)
    return left, np.exp(-1 / np.sqrt(eps))


def test_grid(problem, eps):
    """Test grid of Table 1: 1001 uniform points + 401 uniform points in each boundary layer region."""
    u = np.linspace(0, 1, 1001)
    if problem == 'TP1':
        return np.unique(np.concatenate((u, np.linspace(0, min(20 * eps, 1.0), 401))))
    if problem == 'TP2':
        w = min(20 * eps, 1.0)
        return np.unique(np.concatenate((u, np.linspace(0, w, 401), np.linspace(1 - w, 1, 401))))
    return np.unique(np.concatenate((u, np.linspace(0, min(20 * np.sqrt(eps), 1.0), 401))))


# Settings of the paper (Section 3)
SETTINGS = {
    'TP1': dict(eps_min=4e-4,
                eps_init=[0.0004, 0.0013, 0.03, 0.13, 0.013, 1, 0.007, 0.5, 0.0009, 0.0012],
                na_pga=100,         # PGA-DNN: Na = M = 100 (200-point adaptive meshes)
                na_pa=500,          # PA-DNN: Na uniform points per adaptive step
                n_dd=200,           # DD: uniform points per additional eps
                lr_decay=False,
                eps_test=[3e-1, 7e-2, 2e-2, 3e-3, 6e-4]),
    'TP2': dict(eps_min=7e-4,
                eps_init=[0.0007, 0.0013, 0.03, 0.13, 0.013, 1, 0.007, 0.5, 0.25, 0.75],
                na_pga=100, na_pa=200, n_dd=200,
                lr_decay=True,      # 1e-3 (it < 30000), 1e-4 (it < 70000), 1e-5
                eps_test=[3e-1, 7e-2, 2e-2, 3e-3, 1e-3]),
    'TP3': dict(eps_min=1e-5,
                eps_init=[1e-5, 3e-5, 0.00013, 0.0005, 0.0013, 0.007, 0.013, 0.13, 0.5, 1],
                na_pga=50, na_pa=100, n_dd=100,
                lr_decay=False,
                eps_test=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5]),
}
