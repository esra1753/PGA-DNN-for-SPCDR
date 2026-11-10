# -*- coding: utf-8 -*-
"""
Training of PGA-DNN, PA-DNN and the two baselines (DD, PINN) for TP1, TP2 and TP3.

    python src/train.py --problem TP1 --method PGA --seed 1234

Methods
    PGA  parameter-geometry-adaptive DNN (Algorithm 2): eps* from the network residual (Eqs. 20-23),
         IELDTM labels on a residual-guided nonuniform mesh of Na + M points (Eqs. 26-34).
    PA   parameter-adaptive DNN (Algorithm 1): eps* as in PGA, IELDTM labels on a uniform mesh of Na points.
    DD   data-driven DNN trained on a fixed data set: the initial data plus 90 log-uniformly sampled eps
         values on uniform meshes (the same number of IELDTM solutions as PGA-DNN), no adaptive steps.
    PINN parametric PINN: loss = PDE residual + boundary conditions (equal weights), no labelled data.
         N_f = 10^4 collocation points and 50 boundary points, eps log-uniform in [eps_min, 1].

All models: 3 hidden layers of 60 neurons, tanh, Xavier normal weights / zero biases, Adam (full batch,
lr = 1e-3; for TP2 1e-4 after 30000 and 1e-5 after 70000 iterations), k_max = 100000 iterations,
pre-training A = 10000 iterations, one adaptive step every k = 1000 iterations (90 steps). The initial
labelled data consist of 500 uniform points for 10 values of eps (Eq. 17). No early stopping.

Outputs (results/<problem>/<method>/seed<seed>/):
    convergence_history.csv   L_d, max PDE residual and offline time every 500 iterations
    adaptive_steps.csv        eps*, mesh size and timings of every adaptive step (PGA, PA)
    timings.csv               breakdown of the offline computational cost (Table 2)
    checkpoints/it_XXXXXX.pt  network parameters every 500 iterations (it_100000.pt = final model)
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import problems as P

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--problem', required=True, choices=('TP1', 'TP2', 'TP3'))
ap.add_argument('--method', required=True, choices=('PGA', 'PA', 'DD', 'PINN'))
ap.add_argument('--seed', type=int, default=1234)
ap.add_argument('--na', type=int, default=None,
                help='PA-DNN: number of uniform points per adaptive step (default: value of the paper)')
ap.add_argument('--out', default=None, help='output folder (default: results/<problem>/<method>-DNN/seed<seed>)')
ap.add_argument('--ckpt-every', type=int, default=500, help='checkpoint interval (0: final model only)')
ap.add_argument('--A', type=int, default=10000, help='pre-training iterations')
ap.add_argument('--k', type=int, default=1000, help='iterations between adaptive steps')
ap.add_argument('--kmax', type=int, default=100000, help='total number of iterations')
args = ap.parse_args()

PROBLEM, METHOD, SEED = args.problem, args.method, args.seed
S = P.SETTINGS[PROBLEM]
EPS_MIN = S['eps_min']
NA_PA = args.na if args.na is not None else S['na_pa']

torch.set_default_dtype(torch.float)
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

method_dir = {'PGA': 'PGA-DNN', 'PA': 'PA-DNN', 'DD': 'DD', 'PINN': 'PINN'}[METHOD]
if METHOD == 'PA' and args.na is not None:
    method_dir += '_Na%d' % NA_PA
OUT_DIR = args.out or os.path.join('results', PROBLEM, method_dir, 'seed%d' % SEED)
CKPT_DIR = os.path.join(OUT_DIR, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------------------------
# Timing of the offline stages
# ----------------------------------------------------------------------------------------------
timings = {}
MONITOR_KEYS = ('log_initial', 'log_pretraining', 'log_adaptive', 'monitor_adaptive')


class Timer:
    """with Timer('name'): ...  adds the wall-clock time of the block to timings['name']."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        self.dt = time.perf_counter() - self.t0
        timings[self.name] = timings.get(self.name, 0.0) + self.dt


# ----------------------------------------------------------------------------------------------
# Initial labelled data d (Eq. 17): 10 values of eps x 500 uniform points, IELDTM labels
# ----------------------------------------------------------------------------------------------
x_init = np.linspace(0, 1, 500)
with Timer('initial_data_ieldtm'):
    eps_init = np.array(S['eps_init'], dtype=np.float64)
    nodes = np.sort(x_init)
    u_init = np.column_stack([P.ieldtm(PROBLEM, x_init, e) for e in eps_init])
    X_d_train = np.vstack((np.tile(nodes, len(eps_init)), np.repeat(eps_init, len(nodes)))).T
    u_d_train = u_init.flatten('F')[:, None]


# ----------------------------------------------------------------------------------------------
# Boundary and collocation points (used in the loss only by the PINN)
# ----------------------------------------------------------------------------------------------
def boundary_and_collocation_points(n_f):
    n_b = 50
    x_b = np.vstack([np.zeros((n_b // 2, 1)), np.ones((n_b // 2, 1))])
    if METHOD == 'PINN':
        eps_b = 10 ** np.random.uniform(np.log10(EPS_MIN), 0, size=(n_b // 2, 1))
    else:
        eps_b = np.random.random(size=(n_b // 2, 1))
    X_b = np.column_stack((x_b, np.vstack([eps_b, eps_b]))).astype(np.float64)
    u_left, u_right = P.boundary_values(PROBLEM, eps_b)
    u_b = np.vstack([u_left, u_right])

    x_f = np.random.uniform(0, 1, n_f).reshape(-1, 1)
    if METHOD == 'PINN':
        eps_f = (10 ** np.random.uniform(np.log10(EPS_MIN), 0, n_f)).reshape(-1, 1)
    else:
        eps_f = np.linspace(EPS_MIN, 1, n_f).reshape(-1, 1)
    X_f = np.vstack((np.hstack((x_f, eps_f)), X_b))   # boundary points appended to the collocation points
    return X_f, X_b, u_b


N_f = 10000
with Timer('collocation_points'):
    X_f_np, X_b_np, u_b_np = boundary_and_collocation_points(N_f)

X_f_train = torch.from_numpy(X_f_np).float().to(device)
X_b_train = torch.from_numpy(X_b_np).float().to(device)
u_b_train = torch.from_numpy(u_b_np).float().to(device)
X_d_train_tensor = torch.from_numpy(X_d_train).float().to(device)
u_d_train_tensor = torch.from_numpy(u_d_train).float().to(device)
n_bc = X_b_train.shape[0]


# ----------------------------------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------------------------------
class Network(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, x):
        a = x.float()
        for i in range(len(self.linears) - 1):
            a = self.activation(self.linears[i](a))
        return self.linears[-1](a)

    def loss_data(self, x_d, y_d):
        """Supervised loss L_d (Eq. 18)."""
        return self.loss_function(self.forward(x_d), y_d)

    def pde_residual(self, x):
        """Pointwise residual (Eq. 21); derivatives with respect to x only."""
        g = x.clone().float().requires_grad_(True)
        u = self.forward(g)
        u_x = autograd.grad(u, g, torch.ones_like(u), create_graph=True)[0][:, [0]]
        u_xx = autograd.grad(u_x, g, torch.ones_like(u_x), create_graph=True)[0][:, [0]]
        return P.residual(PROBLEM, g, u, u_x, u_xx)

    def loss(self, x_d, y_d, x_b, y_b, x_f, lambda_pde, lambda_data):
        loss_data = self.loss_data(x_d, y_d)
        if lambda_pde == 0:
            loss_pde = torch.zeros((), device=x_d.device)
        else:
            loss_bc = self.loss_function(self.forward(x_b), y_b)
            f = self.pde_residual(x_f)
            loss_pde = loss_bc + self.loss_function(f, torch.zeros(x_f.shape[0], 1, device=x_f.device))
        return lambda_pde * loss_pde + lambda_data * loss_data


# ----------------------------------------------------------------------------------------------
# Adaptive steps
# ----------------------------------------------------------------------------------------------
def select_eps_star(model, n_eps=100, n_x=100):
    """eps* (Eqs. 20-23): a new set of 100 log-uniform eps values x 100 random x is drawn at every
    adaptive step; L_physics is the mean squared residual for each eps, eps* = argmax L_physics."""
    eps = 10 ** np.random.uniform(np.log10(EPS_MIN), 0.0, n_eps)
    x = np.random.uniform(0, 1, (n_eps, n_x))
    X = torch.from_numpy(np.column_stack((x.ravel(), np.repeat(eps, n_x)))).float().to(device)
    R2 = model.pde_residual(X).detach().pow(2).cpu().double().numpy().reshape(n_eps, n_x)
    L_phys = R2.mean(axis=1)
    return to_float32(eps[int(np.argmax(L_phys))])


def to_float32(v):
    """eps* is stored in single precision, as the network input."""
    return float(torch.tensor(float(v), dtype=torch.float32))


def adaptive_mesh(model, eps_star, n_a, m):
    """Nonuniform mesh of PGA-DNN (Eqs. 26-32): n_a uniform nodes plus m points distributed over the
    n_a - 1 subintervals proportionally to log(1 + mean squared residual), largest remainder rounding,
    placed equidistantly inside each subinterval."""
    x = np.linspace(0, 1, n_a)
    X = torch.from_numpy(np.vstack((x, eps_star * np.ones_like(x))).T).float().to(device)
    E = model.pde_residual(X).detach().cpu().double().pow(2).numpy().ravel()     # Eq. 26
    E_avg = (E[:-1] + E[1:]) / 2                                                 # Eq. 27
    w = np.log1p(E_avg)                                                          # Eq. 28
    if np.isfinite(w).all() and w.sum() > 0:
        share = w / w.sum() * m
    else:
        share = np.full(len(w), m / len(w))
    counts = np.floor(share).astype(int)                                         # Eq. 29
    remaining = m - counts.sum()
    if remaining > 0:
        order = np.argsort(-(share - counts), kind='stable')
        counts[order[:remaining]] += 1
    new_points = [x[i] + (x[i + 1] - x[i]) * np.arange(1, n + 1) / (n + 1)     # Eq. 31
                  for i, n in enumerate(counts) if n > 0]
    return np.sort(np.concatenate([x] + new_points))                             # Eq. 32


def ieldtm_labels(eps, mesh):
    """IELDTM labels for one value of eps on the given mesh (Eqs. 25 and 34), in random order."""
    nodes = np.sort(mesh)
    u = P.ieldtm(PROBLEM, mesh, eps)
    X = np.hstack([nodes.reshape(-1, 1), eps * np.ones((len(nodes), 1))])
    idx = np.random.choice(len(nodes), len(nodes), replace=False)
    return (torch.from_numpy(X[idx]).float().to(device),
            torch.from_numpy(u[idx].reshape(-1, 1)).float().to(device))


# DD: the labels that PGA-DNN generates during the adaptive phase are generated at the beginning,
# for 90 log-uniform random values of eps on uniform meshes.
if METHOD == 'DD':
    eps_dd = 10 ** np.random.RandomState(SEED + 7).uniform(np.log10(EPS_MIN), 0, 90)
    x_dd = np.linspace(0, 1, S['n_dd'])
    with Timer('dd_data_ieldtm'):
        for e in eps_dd:
            X_new, u_new = ieldtm_labels(to_float32(e), x_dd)
            X_d_train_tensor = torch.cat([X_d_train_tensor, X_new], dim=0)
            u_d_train_tensor = torch.cat([u_d_train_tensor, u_new], dim=0)
    np.savetxt(os.path.join(OUT_DIR, 'dd_eps.csv'), eps_dd)

# ----------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------
lambda_pde, lambda_data = (1, 0) if METHOD == 'PINN' else (0, 1)
adaptive = METHOD in ('PGA', 'PA')

model = Network(np.array([2, 60, 60, 60, 1])).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                       amsgrad=False)
A_pre, k_adapt, max_iter = args.A, args.k, args.kmax


def lr_at(it):
    if not S['lr_decay'] or it < 30000:
        return 1e-3
    return 1e-4 if it < 70000 else 1e-5


def train_step():
    loss = model.loss(X_d_train_tensor, u_d_train_tensor, X_b_train, u_b_train, X_f_train,
                      lambda_pde, lambda_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


n_data_initial = X_d_train_tensor.shape[0]
t_initial = sum(timings.get(k, 0.0) for k in ('initial_data_ieldtm', 'collocation_points', 'dd_data_ieldtm'))

LOG_EVERY = 500
f_hist = open(os.path.join(OUT_DIR, 'convergence_history.csv'), 'w', newline='', encoding='utf-8')
w_hist = csv.writer(f_hist)
w_hist.writerow(['iteration', 'phase', 'n_data', 'L_d', 'max_residual', 'offline_time_s'])


def offline_elapsed():
    """Offline cost so far (initial data + training), without logging and monitoring."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return t_initial + (time.perf_counter() - t_train_start) - sum(timings.get(k, 0.0) for k in MONITOR_KEYS)


def log_history(n_iter, phase, t_off):
    with torch.no_grad():
        L_d = model.loss_data(X_d_train_tensor, u_d_train_tensor).item()
    max_res = model.pde_residual(X_f_train[:X_f_train.shape[0] - n_bc]).detach().abs().max().item()
    w_hist.writerow([n_iter, phase, X_d_train_tensor.shape[0], '%.6e' % L_d, '%.6e' % max_res, '%.3f' % t_off])
    f_hist.flush()
    if (args.ckpt_every and n_iter % args.ckpt_every == 0) or n_iter == max_iter:
        torch.save({'iteration': n_iter, 'model': model.state_dict(), 'n_data': X_d_train_tensor.shape[0]},
                   os.path.join(CKPT_DIR, 'it_%06d.pt' % n_iter))


print('%s %s seed %d -> %s' % (PROBLEM, method_dir, SEED, OUT_DIR))
t_train_start = time.perf_counter()
with Timer('log_initial'):
    log_history(0, 'initial', t_initial)

# Pre-training with the initial data d
with Timer('pretraining'):
    for it in range(A_pre):
        loss = train_step()
        if (it + 1) % LOG_EVERY == 0:
            t_off = offline_elapsed()
            with Timer('log_pretraining'):
                log_history(it + 1, 'pretraining', t_off)
print('pre-training finished: loss %.3e' % loss.item())

# Adaptive training
adaptive_log = []
with Timer('adaptive_training'):
    for it in range(A_pre, max_iter):
        for g in optimizer.param_groups:
            g['lr'] = lr_at(it)
        loss = train_step()

        if adaptive and it % k_adapt == 0:
            with Timer('eps_selection') as t_a:
                eps_star = select_eps_star(model)
            with Timer('mesh_generation') as t_b:
                if METHOD == 'PGA':
                    mesh = adaptive_mesh(model, eps_star, S['na_pga'], S['na_pga'])
                else:
                    mesh = np.linspace(0, 1, NA_PA)
            with Timer('ieldtm_labels') as t_c:
                X_new, u_new = ieldtm_labels(eps_star, mesh)
            X_d_train_tensor = torch.cat([X_d_train_tensor, X_new], dim=0)
            u_d_train_tensor = torch.cat([u_d_train_tensor, u_new], dim=0)
            adaptive_log.append((it, eps_star, len(mesh), t_a.dt, t_b.dt, t_c.dt, X_d_train_tensor.shape[0]))
            with Timer('monitor_adaptive'):
                print('it %6d  loss %.3e  eps* %.3e  n_data %d' % (it, loss.item(), eps_star,
                                                                   X_d_train_tensor.shape[0]))

        if (it + 1) % LOG_EVERY == 0:
            t_off = offline_elapsed()
            with Timer('log_adaptive'):
                log_history(it + 1, 'adaptive', t_off)
f_hist.close()

# ----------------------------------------------------------------------------------------------
# Offline cost (Table 2); logging and monitoring are excluded
# ----------------------------------------------------------------------------------------------
for k in ('eps_selection', 'mesh_generation', 'ieldtm_labels', 'monitor_adaptive', 'dd_data_ieldtm'):
    timings.setdefault(k, 0.0)
t_pre = timings['pretraining'] - timings.get('log_pretraining', 0.0)
t_adapt = timings['adaptive_training'] - timings['monitor_adaptive'] - timings.get('log_adaptive', 0.0)
t_updates = t_adapt - timings['eps_selection'] - timings['mesh_generation'] - timings['ieldtm_labels']
t_ieldtm = timings['initial_data_ieldtm'] + timings['dd_data_ieldtm'] + timings['ieldtm_labels']
n_adapt = len(adaptive_log)

rows = [
    ('1) Initial data generation', t_initial),
    ('   a) IELDTM, %d eps values x %d points' % (len(eps_init), len(x_init)), timings['initial_data_ieldtm']),
    ('   b) Collocation points (N_f = %d)' % N_f, timings['collocation_points']),
    ('   c) IELDTM, fixed DD data (90 eps values)', timings['dd_data_ieldtm']),
    ('2) Pre-training (A = %d iterations)' % A_pre, t_pre),
    ('3) Adaptive training (%d steps)' % n_adapt, t_adapt),
    ('   a) Residual evaluation / selection of eps*', timings['eps_selection']),
    ('   b) Mesh generation', timings['mesh_generation']),
    ('   c) IELDTM solutions (%d solves, %d labels)' % (n_adapt, X_d_train_tensor.shape[0] - n_data_initial
                                                       if adaptive else 0), timings['ieldtm_labels']),
    ('   d) Network updates (%d iterations)' % (max_iter - A_pre), t_updates),
    ('Total IELDTM time', t_ieldtm),
    ('Total network training time', t_pre + t_updates),
    ('Total offline cost', t_initial + t_pre + t_adapt),
]
print('\n' + '=' * 72)
print('OFFLINE COST  (device: %s, torch %s, CPU threads: %d)' % (device, torch.__version__,
                                                                 torch.get_num_threads()))
print('=' * 72)
for name, t in rows:
    print('%-58s %10.2f s' % (name, t))
print('=' * 72)

with open(os.path.join(OUT_DIR, 'timings.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['stage', 'time_s'])
    for name, t in rows:
        w.writerow([name.strip(), '%.4f' % t])

if adaptive:
    with open(os.path.join(OUT_DIR, 'adaptive_steps.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['iteration', 'eps_star', 'n_mesh_points', 'time_selection_s', 'time_mesh_s',
                    'time_ieldtm_s', 'n_data'])
        for r in adaptive_log:
            w.writerow([r[0], '%.6e' % r[1], r[2], '%.6f' % r[3], '%.6f' % r[4], '%.6f' % r[5], r[6]])
