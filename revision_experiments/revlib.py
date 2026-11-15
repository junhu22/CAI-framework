"""
revlib.py — Shared loaders, model definitions, observations, and diagnostics
for the CAI revision experiments (E1-E6 + E-th).

Re-uses the exact architectures/conventions of:
  - cai_lib.py           (D=6 sailboat, D=6 broad, coupled D=3; theta-space flow)
  - d12_validation.py    (D=12; log-theta-space flow)
  - run_v3_mm.py         (mass-matrix NeuTra-HMC)

Global seed = 42. All models run on CPU (matches the paper's timing convention),
except callers may move D=12 or SNPE to GPU explicitly.
"""
import sys, math, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

SURR = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
NFP  = Path(r"C:\nf-pmcmc\experiments")
DEV  = torch.device("cpu")

# ── D=6 primitives from cai_lib (theta-space) ───────────────────────────
sys.path.insert(0, str(SURR / "experiments"))
from cai_lib import (                                    # noqa: E402
    Surrogate, ConditionalRealNVP, KModeLogPosterior,
    NeuTraHMC_Conditional, HMCSampler,
    effective_sample_size, importance_weight_ess,
    load_surrogate, load_cnf,
)

# ── Weight paths ────────────────────────────────────────────────────────
D6_SURR_DIR = SURR / "surrogate_models"
D6_CNF      = NFP / "results" / "conditional_flow.pt"
BROAD_SURR_DIR = SURR / "surrogate_models_broad"
BROAD_CNF   = SURR / "experiments" / "results" / "conditional_flow_broad.pt"
COUP_SURR   = SURR / "surrogate_models_coupled" / "member_0.pt"
COUP_CNF    = SURR / "experiments" / "results" / "conditional_flow_coupled.pt"
D12_PT      = NFP / "results" / "d12_validation.pt"
D12_PT_V2   = NFP / "results" / "d12_validation_v2.pt"

# ── Config constants ────────────────────────────────────────────────────
CFG = {
    "d6":      dict(kappa_sq=1e-3, prior_sigma=0.30, K=20, D=6),
    "broad":   dict(kappa_sq=1e-3, prior_sigma=0.55, K=6,  D=6),
    "coupled": dict(kappa_sq=9e-4, prior_sigma=0.50, K=6,  D=3),
    "d12":     dict(kappa=0.03,    sigma_prior=0.50, K=20, D=12),
}
FIELD_MEASURED = [0.91, 1.08, 1.47, 2.55, 3.16, 3.81]
COUP_MEASURED  = [2.06, 2.30, 2.62, 6.14, 6.53, 6.76]
FIELD_MASS = [1.5482755899429321, 0.9718354940414429, 0.34744009375572205,
              3.2696502208709717, 0.5412551164627075, 0.34921297430992126]
COUP_MASS  = [0.9902399778366089, 0.9208102822303772, 1.010308861732483]
FIELD_CENTERS = {   # exact-mode posterior means (mode centers) for E5/E1-B
    "narrow":      [1.2207874, 2.6167192, 2.1256261, 0.9051405, 0.9775811, 1.4227651],
    "broad":       [1.2868387, 2.4812047, 1.7759950, 0.8944345, 1.0234228, 1.0731375],
    "broad_mass":  [1.3992568, 2.4102411, 1.8381885, 0.9351584, 0.9421184, 1.1927426],
}
FIELD_FASTMEAN = [1.2977122, 1.3546389, 1.3393221, 1.1355579, 1.0362222, 1.3487413]
COUP_THETA_MAP  = [3.4315545558929443, 0.7805843949317932, 2.8936502933502197]
COUP_NORMAL_MEAN = [3.227696657180786, 1.1847851276397705, 3.300145387649536]
COUP_DAMAGE_MEAN = [0.5070630311965942, 1.1409382820129395, 0.9872524738311768]
# Full 20-mode y used to condition the coupled CNF (from coupled_results.json)
COUP_NORMAL_YFULL = [2.06, 2.30, 2.62, 6.14, 6.53, 6.76, 7.233523, 8.926735,
    9.065089, 10.551189, 10.581640, 10.894555, 11.742413, 12.575490, 12.690943,
    12.883026, 13.133237, 13.473139, 13.918221, 13.985258]
COUP_DAMAGE_YFULL = [0.927721, 1.057796, 1.069256, 2.702219, 3.085872, 3.217286,
    3.285917, 4.048956, 4.249094, 4.381341, 4.693439, 4.894359, 4.955325, 5.267453,
    5.318134, 5.442022, 5.515037, 5.707598, 5.931717, 6.051433]

def field_yfull(broad_surrogate):
    """20-mode hybrid y for the field CNF: [measured_6, surrogate(theta=1)[6:20]]."""
    with torch.no_grad():
        f_nom = broad_surrogate(torch.ones(1, 6)).squeeze(0).numpy()
    return np.concatenate([FIELD_MEASURED, f_nom[6:]]).astype(np.float32)

# D=6 robustness thetas (from robustness_test.py), Obs2 theta (corrected_amortized.py)
ROBUST_THETAS = [
    [0.7, 1.2, 1.8, 1.0, 0.8, 1.1], [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    [1.5, 1.4, 1.3, 1.2, 1.1, 1.0], [0.6, 1.8, 0.7, 1.5, 0.9, 1.3],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.8, 0.9, 1.9, 0.7, 1.4, 0.6],
    [1.3, 1.3, 1.3, 1.3, 1.3, 1.3], [0.55, 1.0, 1.5, 0.85, 1.2, 0.75],
    [1.1, 0.7, 1.6, 1.4, 0.5, 1.8], [0.9, 1.1, 1.0, 0.9, 1.1, 1.0],
]
OBS2_THETA = [0.9, 1.3, 1.5, 0.8, 1.1, 1.2]
D12_SCENARIOS = {
    "Near nominal": [1.0]*12,
    "Mixed damage": [0.7, 1.0, 1.3, 0.8, 1.1, 0.9, 1.2, 0.75, 1.0, 1.15, 0.85, 1.05],
    "Heavy damage": [0.55, 0.65, 0.7, 0.6, 0.75, 0.8, 0.55, 0.6, 0.7, 0.65, 0.75, 0.5],
}

# ════════════════════════════════════════════════════════════════════════
# D=6 / broad / coupled loaders (theta-space)
# ════════════════════════════════════════════════════════════════════════
def load_d6(member=0):
    return load_surrogate(D6_SURR_DIR / f"member_{member}.pt")

def load_d6_ensemble():
    return [load_surrogate(D6_SURR_DIR / f"member_{i}.pt") for i in range(5)]

def load_d6_cnf():
    return load_cnf(D6_CNF, dim_theta=6, dim_y=20, n_layers=8, hidden_dim=128)

def load_broad(member=0):
    return load_surrogate(BROAD_SURR_DIR / f"member_{member}.pt")

def load_broad_ensemble():
    return [load_surrogate(BROAD_SURR_DIR / f"member_{i}.pt") for i in range(5)]

def load_broad_cnf():
    return load_cnf(BROAD_CNF, dim_theta=6, dim_y=20, n_layers=8, hidden_dim=128)

# ── Round-2 cleaned/retrained broad models ──────────────────────────────
BROAD_SURR_CLEAN_DIR = SURR / "surrogate_models_broad_clean"
BROAD_CNF_CLEAN = SURR / "experiments" / "results" / "conditional_flow_broad_clean.pt"
def load_broad_clean(member=0):
    return load_surrogate(BROAD_SURR_CLEAN_DIR / f"member_{member}.pt")
def load_broad_clean_ensemble():
    return [load_surrogate(BROAD_SURR_CLEAN_DIR / f"member_{i}.pt") for i in range(5)]
def load_broad_clean_cnf():
    return load_cnf(BROAD_CNF_CLEAN, dim_theta=6, dim_y=20, n_layers=8, hidden_dim=128)

# ── Round-3 domain-extended broad models ([0.5,3.5]) ────────────────────
BROAD_SURR_EXT_DIR = SURR / "surrogate_models_broad_ext"
BROAD_CNF_EXT = SURR / "experiments" / "results" / "conditional_flow_broad_ext.pt"
def load_broad_ext(member=0):
    return load_surrogate(BROAD_SURR_EXT_DIR / f"member_{member}.pt")
def load_broad_ext_ensemble():
    return [load_surrogate(BROAD_SURR_EXT_DIR / f"member_{i}.pt") for i in range(5)]
def load_broad_ext_cnf():
    return load_cnf(BROAD_CNF_EXT, dim_theta=6, dim_y=20, n_layers=8, hidden_dim=128)

def load_coupled():
    m = Surrogate(nin=3, nout=20)
    m.load_state_dict(torch.load(COUP_SURR, map_location=DEV, weights_only=True))
    m.eval()
    return m

def load_coupled_cnf():
    # Coupled CNF is cai_lib's class but with a 20->64->20 y-embed (hidden 64).
    cnf = ConditionalRealNVP(dim_theta=3, dim_y=20, n_layers=8, hidden_dim=128).to(DEV)
    cnf.y_embed = nn.Sequential(nn.Linear(20, 64), nn.SiLU(), nn.Linear(64, 20))
    cnf.load_state_dict(torch.load(COUP_CNF, map_location=DEV, weights_only=True))
    cnf.eval()
    return cnf

def make_log_post(surrogate, y_obs, cfgname):
    """KModeLogPosterior in theta-space. y_obs length = K (uses first K modes)."""
    c = CFG[cfgname]
    return KModeLogPosterior(surrogate, list(y_obs),
                             kappa_sq=c["kappa_sq"], prior_sigma=c["prior_sigma"]).to(DEV)

# ════════════════════════════════════════════════════════════════════════
# D=12 models (log-theta space) — mirrors d12_validation.py
# ════════════════════════════════════════════════════════════════════════
class ResBlock12(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, dim),
            nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, dim))
    def forward(self, x): return x + self.net(x)

class NNSurrogate12(nn.Module):
    def __init__(self, D=12, M=20, hidden=256, n_blocks=6):
        super().__init__()
        layers = [nn.Linear(D, hidden)]
        for _ in range(n_blocks): layers.append(ResBlock12(hidden))
        layers.extend([nn.SiLU(), nn.Linear(hidden, M)])
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class CondAffineCoupling12(nn.Module):
    def __init__(self, D, cond_dim, hidden=128, mask_even=True):
        super().__init__()
        self.D, self.mask_even = D, mask_even
        d_in = D // 2 + cond_dim
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * (D - D // 2)))
    def _split(self, x):
        return (x[..., ::2], x[..., 1::2]) if self.mask_even else (x[..., 1::2], x[..., ::2])
    def _merge(self, xk, xt):
        out = torch.empty_like(torch.cat([xk, xt], dim=-1))
        if self.mask_even: out[..., ::2] = xk; out[..., 1::2] = xt
        else:              out[..., 1::2] = xk; out[..., ::2] = xt
        return out
    def forward(self, x, cond):
        xk, xt = self._split(x)
        s, t = self.net(torch.cat([xk, cond], dim=-1)).chunk(2, dim=-1)
        s = torch.tanh(s) * 2
        return self._merge(xk, xt * torch.exp(s) + t), s.sum(dim=-1)
    def inverse(self, z, cond):
        zk, zt = self._split(z)
        s, t = self.net(torch.cat([zk, cond], dim=-1)).chunk(2, dim=-1)
        s = torch.tanh(s) * 2
        return self._merge(zk, (zt - t) * torch.exp(-s))

class CondRealNVP12(nn.Module):
    def __init__(self, D=12, M=20, n_layers=12, hidden=192, embed_dim=32):
        super().__init__()
        self.D = D
        self.embed = nn.Sequential(nn.Linear(M, embed_dim*2), nn.SiLU(),
                                   nn.Linear(embed_dim*2, embed_dim))
        self.layers = nn.ModuleList([
            CondAffineCoupling12(D, embed_dim, hidden, mask_even=(i % 2 == 0))
            for i in range(n_layers)])
    def forward(self, theta, y):
        cond = self.embed(y); z = theta; log_det = 0
        for layer in self.layers:
            z, ld = layer(z, cond); log_det = log_det + ld
        return z, log_det
    def inverse(self, z, y):
        cond = self.embed(y); x = z
        for layer in reversed(self.layers): x = layer.inverse(x, cond)
        return x
    def log_prob(self, theta, y):
        z, log_det = self.forward(theta, y)
        log_pz = -0.5 * (z**2 + np.log(2*np.pi)).sum(dim=-1)
        return log_pz + log_det

def load_d12(pt_path=D12_PT):
    blob = torch.load(pt_path, map_location=DEV, weights_only=False)
    nn_sur = NNSurrogate12(12, 20, 256, 6).double()
    nn_sur.load_state_dict(blob["nn_state_dict"]); nn_sur.eval()
    flow = CondRealNVP12(12, 20, 12, 192, 32).double()
    flow.load_state_dict(blob["flow_state_dict"]); flow.eval()
    return nn_sur, flow, blob

def d12_log_posterior(log_theta, y_obs, nn_sur, kappa=0.03, sigma_prior=0.5):
    theta = torch.exp(log_theta)
    y_pred = nn_sur(theta)
    rel = (y_pred - y_obs) / y_obs
    return -0.5 * (rel**2).sum(dim=-1) / kappa**2 - 0.5 * (log_theta**2).sum(dim=-1) / sigma_prior**2

def d12_fast_sample(flow, y_obs, N=2000):
    with torch.no_grad():
        z = torch.randn(N, flow.D, dtype=torch.float64)
        lt = flow.inverse(z, y_obs.unsqueeze(0).expand(N, -1))
    return torch.exp(lt), lt

def d12_ess_is(flow, nn_sur, y_obs, N=2000, kappa=0.03, sigma_prior=0.5):
    theta, lt = d12_fast_sample(flow, y_obs, N)
    yb = y_obs.unsqueeze(0).expand(N, -1)
    with torch.no_grad():
        log_tgt = d12_log_posterior(lt, y_obs, nn_sur, kappa, sigma_prior)
        log_q = flow.log_prob(lt, yb)
        lw = log_tgt - log_q; lw = lw - lw.max()
        w = torch.exp(lw); w = w / w.sum()
        ess = float(1.0 / (w**2).sum().item())
    return ess, ess / N, theta

# ════════════════════════════════════════════════════════════════════════
# Mass-matrix NeuTra-HMC (mirrors run_v3_mm.py NeuTraHMC_MassDiag)
# ════════════════════════════════════════════════════════════════════════
class NeuTraHMC_MassDiag(NeuTraHMC_Conditional):
    def __init__(self, log_post_fn, cond_flow, y_obs, step_size, n_leapfrog, mass_diag):
        super().__init__(log_post_fn, cond_flow, y_obs,
                         step_size=step_size, n_leapfrog=n_leapfrog)
        m = torch.as_tensor(mass_diag, dtype=torch.float32).clamp(min=1e-8)
        self.mass, self.inv_mass, self.sqrt_mass = m, 1.0/m, torch.sqrt(m)
    def _leapfrog_mass(self, q, p, eps, grad_init=None):
        g = grad_init if grad_init is not None else self._grad_log_prob_z(q)[1]
        p = p + 0.5 * eps * g
        for _ in range(self.n_leapfrog - 1):
            q = q + eps * (p * self.inv_mass)
            _, g = self._grad_log_prob_z(q); p = p + eps * g
        q = q + eps * (p * self.inv_mass)
        lp, g = self._grad_log_prob_z(q); p = p + 0.5 * eps * g
        return q, p, lp, g
    def run(self, theta0, n_samples, n_burnin=100, seed=42):
        torch.manual_seed(seed)
        self.flow.eval()
        with torch.no_grad():
            z, _ = self.flow(theta0.unsqueeze(0), self._y_batch(1))
        z = z.squeeze(0)
        z_chain, n_acc = [], 0
        eps = self.step_size
        lp_current, grad_current = self._grad_log_prob_z(z)
        for i in range(-n_burnin, n_samples):
            p = self.sqrt_mass * torch.randn_like(z)
            H0 = -lp_current + 0.5 * (p**2 * self.inv_mass).sum()
            q_prop, p_prop, lp_prop, grad_prop = self._leapfrog_mass(z, p, eps, grad_current)
            H1 = -lp_prop + 0.5 * (p_prop**2 * self.inv_mass).sum()
            if torch.log(torch.rand(1)).item() < (H0 - H1).item():
                z, lp_current, grad_current = q_prop, lp_prop, grad_prop
                if i >= 0: n_acc += 1
            if i >= 0: z_chain.append(z.clone())
        z_stack = torch.stack(z_chain)
        with torch.no_grad():
            theta_chain, _ = self.flow.inverse(z_stack, self._y_batch(len(z_chain)))
        return theta_chain, n_acc / n_samples

# ════════════════════════════════════════════════════════════════════════
# Observation generation (deterministic)
# ════════════════════════════════════════════════════════════════════════
def gen_noisy_y(surrogate, theta, kappa_sq, seed):
    """y = f(theta) + sqrt(kappa_sq) * f * N(0,1), deterministic via local generator."""
    th = torch.as_tensor(theta, dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        f = surrogate(th).squeeze(0)
    g = torch.Generator().manual_seed(int(seed))
    noise = math.sqrt(kappa_sq) * f * torch.randn(f.shape, generator=g)
    return (f + noise).numpy(), f.numpy()

def load_obs1():
    """Obs1 = row 0 of dataset_test.csv: (theta_true[6], y[20])."""
    import csv
    with open(SURR / "dataset_test.csv") as fh:
        row = next(csv.DictReader(fh))
    theta = np.array([float(row[n]) for n in ["e1","e2","e3","e4","e5","e6"]])
    y = np.array([float(row[f"f{j}"]) for j in range(1, 21)])
    return theta, y

def split_rhat(chains):
    """Split-Rhat per dimension. chains: list of (N, D) arrays (equal N)."""
    chains = [np.asarray(c) for c in chains]
    N = min(c.shape[0] for c in chains); D = chains[0].shape[1]
    half = N // 2
    segs = []
    for c in chains:
        segs.append(c[:half]); segs.append(c[half:2*half])
    m = len(segs); n = half
    rhat = np.zeros(D)
    for d in range(D):
        means = np.array([s[:, d].mean() for s in segs])
        vars  = np.array([s[:, d].var(ddof=1) for s in segs])
        W = vars.mean()
        B = n * means.var(ddof=1)
        var_hat = (n - 1) / n * W + B / n
        rhat[d] = np.sqrt(var_hat / W) if W > 1e-30 else np.nan
    return rhat
