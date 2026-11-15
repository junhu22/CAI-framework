#!/usr/bin/env python3
"""
Stage 3+4 — CAI inference on the coupled building (D=3, M=20).

Three inference modes on the Hu et al. 2018 measured frequencies:
  (a) Fast mode:           2000 amortized samples from the conditional flow
  (b) Exact mode:          NeuTra-HMC with diagonal mass matrix,
                           step_size=0.04, L=40, 500 samples + 100 burnin
  (c) Standard HMC:        500 samples + 100 burnin (baseline, no flow)

Hybrid padding: y_full = [y_meas (6 modes), surrogate(θ=1)[6:20]]

Also (Stage 4): damage detection — compare IS-ESS/N for
  Normal scenario  → y_meas (real Hu et al. 2018)
  Damage scenario  → e1 = 0.5, others = 1.0   (column stiffness halved)

Outputs:
  ../../results/coupled_building/coupled_results.json
  ../../figures/fig_coupled_posterior.{pdf,png}
  ../../figures/fig_coupled_damage.{pdf,png}
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SURR_DIR = HERE.parent.parent  # surrogate_training/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SURR_DIR / "experiments"))

from config_coupled import (  # noqa: E402
    N_PARAMS, N_MODES, PARAM_NAMES, MEASURED_FREQS,
    THETA_LO, THETA_HI,
)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] using {DEV}", flush=True)

D = N_PARAMS
M = N_MODES
N_MEAS = len(MEASURED_FREQS)

SURR_PATH = SURR_DIR / "surrogate_models_coupled/member_0.pt"
CNF_PATH = SURR_DIR / "experiments/results/conditional_flow_coupled.pt"
OUT_RES = SURR_DIR / "results/coupled_building"
OUT_FIG = SURR_DIR / "figures"
OUT_RES.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

STEP_SIZE = 0.04
N_LEAPFROG = 40
N_SAMPLES = 500
N_BURNIN = 100
N_FAST = 2000
KAPPA_SQ = 0.03 ** 2     # κ=0.03 ⇒ κ²=9e-4
PRIOR_SIGMA = 0.5


# ── Surrogate ─────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, d, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.SiLU(), nn.Linear(d, d),
            nn.LayerNorm(d), nn.SiLU(), nn.Dropout(drop), nn.Linear(d, d))

    def forward(self, x):
        return x + self.net(x)


class Surrogate(nn.Module):
    def __init__(self, nin=D, nout=M, hid=256, nb=4):
        super().__init__()
        self.nin, self.nout = nin, nout
        self.inp = nn.Sequential(nn.Linear(nin, hid), nn.SiLU())
        self.blk = nn.ModuleList([ResBlock(hid) for _ in range(nb)])
        self.out = nn.Sequential(
            nn.LayerNorm(hid), nn.SiLU(), nn.Linear(hid, nout))
        for n in ("xm", "xs", "ym", "ys"):
            self.register_buffer(n, torch.zeros(nin if "x" in n else nout))

    def _fwd_n(self, xn):
        h = self.inp(xn)
        for b in self.blk:
            h = b(h)
        return self.out(h)

    def forward(self, x):
        return self._fwd_n((x - self.xm) / self.xs) * self.ys + self.ym


# ── Conditional RealNVP (must match 04_train_cnf_coupled.py exactly) ─
class ConditionalAffineCoupling(nn.Module):
    def __init__(self, dim_theta, dim_y, hidden_dim=128, mask_even=True):
        super().__init__()
        self.dim_theta = dim_theta
        mask = torch.zeros(dim_theta)
        if mask_even:
            mask[::2] = 1.0
        else:
            mask[1::2] = 1.0
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dim_theta + dim_y, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, dim_theta * 2),
        )
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x, y):
        x_masked = x * self.mask
        inp = torch.cat([x_masked, y], dim=-1)
        out = self.net(inp)
        log_s, t = out.chunk(2, dim=-1)
        log_s = log_s * (1 - self.mask)
        t = t * (1 - self.mask)
        log_s = torch.tanh(log_s) * 2.0
        z = x * torch.exp(log_s) + t
        return z, log_s.sum(dim=-1)

    def inverse(self, z, y):
        z_masked = z * self.mask
        inp = torch.cat([z_masked, y], dim=-1)
        out = self.net(inp)
        log_s, t = out.chunk(2, dim=-1)
        log_s = log_s * (1 - self.mask)
        t = t * (1 - self.mask)
        log_s = torch.tanh(log_s) * 2.0
        x = (z - t) * torch.exp(-log_s)
        return x, -log_s.sum(dim=-1)


class ConditionalRealNVP(nn.Module):
    def __init__(self, dim_theta=D, dim_y=M, n_layers=8, hidden_dim=128):
        super().__init__()
        self.dim_theta, self.dim_y = dim_theta, dim_y
        self.layers = nn.ModuleList([
            ConditionalAffineCoupling(dim_theta, dim_y, hidden_dim,
                                      mask_even=(i % 2 == 0))
            for i in range(n_layers)
        ])
        self.y_embed = nn.Sequential(
            nn.Linear(dim_y, 64), nn.SiLU(),
            nn.Linear(64, dim_y),
        )

    def _embed_y(self, y):
        return self.y_embed(y)

    def forward(self, theta, y):
        y_emb = self._embed_y(y)
        log_det = torch.zeros(theta.shape[0], device=theta.device)
        z = theta
        for layer in self.layers:
            z, ld = layer(z, y_emb)
            log_det += ld
        return z, log_det

    def inverse(self, z, y):
        y_emb = self._embed_y(y)
        log_det = torch.zeros(z.shape[0], device=z.device)
        x = z
        for layer in reversed(self.layers):
            x, ld = layer.inverse(x, y_emb)
            log_det += ld
        return x, log_det

    def sample(self, y, n_samples=1000):
        y_batch = y.unsqueeze(0).expand(n_samples, -1)
        z = torch.randn(n_samples, self.dim_theta, device=y.device)
        with torch.no_grad():
            theta, _ = self.inverse(z, y_batch)
        return theta

    def log_prob(self, theta, y):
        if y.dim() == 1:
            y_batch = y.unsqueeze(0).expand(theta.shape[0], -1)
        else:
            y_batch = y
        z, log_det = self.forward(theta, y_batch)
        log_pz = (-0.5 * (z ** 2).sum(dim=-1)
                  - 0.5 * self.dim_theta * math.log(2 * math.pi))
        return log_pz + log_det


# ── K-mode log posterior (LogNormal prior on θ) ───────────────────────
class KModeLogPosterior(nn.Module):
    def __init__(self, surrogate, observed_freqs, kappa_sq=KAPPA_SQ,
                 prior_sigma=PRIOR_SIGMA):
        super().__init__()
        self.surrogate = surrogate
        self.K = len(observed_freqs)
        self.register_buffer("f_obs",
                             torch.tensor(observed_freqs, dtype=torch.float32))
        self.kappa_sq = kappa_sq
        self.prior_sigma = prior_sigma

    def forward(self, theta):
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        f_pred = self.surrogate(theta)[:, :self.K]
        resid = (f_pred - self.f_obs) / self.f_obs
        ll = -0.5 / self.kappa_sq * (resid ** 2).sum(dim=-1)
        if (theta <= 0).any(dim=-1).any():
            lp = torch.full_like(ll, -1e30)
            valid = (theta > 0).all(dim=-1)
            log_t = torch.log(theta[valid].clamp(min=1e-30))
            lp[valid] = ll[valid] + (
                -0.5 * (log_t / self.prior_sigma) ** 2 - log_t
            ).sum(dim=-1)
            return lp
        log_t = torch.log(theta)
        lp_prior = (-0.5 * (log_t / self.prior_sigma) ** 2 - log_t).sum(dim=-1)
        return ll + lp_prior

    def grad_log_prob(self, theta):
        theta = theta.detach().requires_grad_(True)
        lp = self.forward(theta).sum()
        g = torch.autograd.grad(lp, theta)[0]
        return lp.detach(), g.detach()


# ── NeuTra-HMC with diagonal mass matrix ──────────────────────────────
class NeuTraHMC_MassDiag:
    def __init__(self, log_post_fn, cond_flow, y_obs,
                 step_size, n_leapfrog, mass_diag):
        self.log_post = log_post_fn
        self.flow = cond_flow
        self.y_obs = y_obs
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        m = mass_diag.to(y_obs.device).clamp(min=1e-8)
        self.mass = m
        self.inv_mass = 1.0 / m
        self.sqrt_mass = torch.sqrt(m)

    def _y_batch(self, n=1):
        return self.y_obs.unsqueeze(0).expand(n, -1)

    def _grad_log_prob_z(self, z):
        z = z.detach().requires_grad_(True)
        theta, log_det = self.flow.inverse(z.unsqueeze(0), self._y_batch(1))
        theta = theta.squeeze(0)
        log_det = log_det.squeeze()
        lp = self.log_post(theta.unsqueeze(0)).squeeze()
        total = lp + log_det
        grad = torch.autograd.grad(total, z)[0]
        return total.detach(), grad.detach()

    def _leapfrog(self, q, p, eps, grad_init=None):
        g = grad_init if grad_init is not None else self._grad_log_prob_z(q)[1]
        p = p + 0.5 * eps * g
        for _ in range(self.n_leapfrog - 1):
            q = q + eps * (p * self.inv_mass)
            _, g = self._grad_log_prob_z(q)
            p = p + eps * g
        q = q + eps * (p * self.inv_mass)
        lp_final, g = self._grad_log_prob_z(q)
        p = p + 0.5 * eps * g
        return q, p, lp_final, g

    def run(self, theta0, n_samples, n_burnin=100):
        self.flow.eval()
        with torch.no_grad():
            z, _ = self.flow(theta0.unsqueeze(0), self._y_batch(1))
        z = z.squeeze(0)
        z_chain = []
        n_acc = 0
        eps = self.step_size
        lp_current, grad_current = self._grad_log_prob_z(z)
        for i in range(-n_burnin, n_samples):
            p = self.sqrt_mass * torch.randn_like(z)
            H0 = -lp_current + 0.5 * (p ** 2 * self.inv_mass).sum()
            q_prop, p_prop, lp_prop, grad_prop = self._leapfrog(
                z, p, eps, grad_init=grad_current)
            H1 = -lp_prop + 0.5 * (p_prop ** 2 * self.inv_mass).sum()
            if torch.log(torch.rand(1)).item() < (H0 - H1).item():
                z = q_prop
                lp_current = lp_prop
                grad_current = grad_prop
                if i >= 0:
                    n_acc += 1
            if i >= 0:
                z_chain.append(z.clone())
        z_stack = torch.stack(z_chain)
        with torch.no_grad():
            y_batch = self._y_batch(len(z_chain))
            theta_chain, _ = self.flow.inverse(z_stack, y_batch)
        return theta_chain, n_acc / n_samples, z_stack


# ── Standard HMC (no flow) ────────────────────────────────────────────
class HMCSampler:
    def __init__(self, grad_log_prob_fn, step_size=0.01, n_leapfrog=20):
        self.grad_lp = grad_log_prob_fn
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog

    def _leapfrog(self, q, p, eps, grad_init=None):
        g = grad_init if grad_init is not None else self.grad_lp(q)[1]
        p = p + 0.5 * eps * g
        for _ in range(self.n_leapfrog - 1):
            q = q + eps * p
            _, g = self.grad_lp(q)
            p = p + eps * g
        q = q + eps * p
        lp_final, g = self.grad_lp(q)
        p = p + 0.5 * eps * g
        return q, p, lp_final, g

    def run(self, x0, n_samples, n_burnin=500):
        x = x0.clone()
        eps = self.step_size
        chain = []
        n_acc = 0
        n_acc_adapt = 0
        lp_current, grad_current = self.grad_lp(x)
        for i in range(-n_burnin, n_samples):
            p = torch.randn_like(x)
            H0 = -lp_current + 0.5 * (p ** 2).sum()
            q_prop, p_prop, lp_prop, grad_prop = self._leapfrog(
                x, p, eps, grad_init=grad_current)
            H1 = -lp_prop + 0.5 * (p_prop ** 2).sum()
            if torch.log(torch.rand(1)).item() < (H0 - H1).item():
                x = q_prop
                lp_current = lp_prop
                grad_current = grad_prop
                if i >= 0:
                    n_acc += 1
                n_acc_adapt += 1
            if i >= 0:
                chain.append(x.clone())
            if i < 0 and (i + n_burnin) % 50 == 49:
                rate = n_acc_adapt / 50
                eps *= math.exp(0.5 * (rate - 0.75))
                eps = max(1e-5, min(eps, 1.0))
                n_acc_adapt = 0
        return torch.stack(chain), n_acc / n_samples


# ── Diagnostics ───────────────────────────────────────────────────────
def effective_sample_size(chain):
    if isinstance(chain, torch.Tensor):
        chain = chain.cpu()
    N, D_ = chain.shape
    ess = torch.zeros(D_)
    for d in range(D_):
        x = chain[:, d].numpy()
        x = x - x.mean()
        var = np.var(x)
        if var < 1e-30:
            ess[d] = 1.0
            continue
        f = np.fft.fft(x, n=2 * N)
        acf = np.fft.ifft(f * np.conj(f)).real[:N] / (N * var)
        t = 1
        while t < N - 1 and acf[t] + acf[t + 1] > 0:
            t += 2
        ess[d] = float(N / (1 + 2 * acf[1:t].sum()))
    return ess


def importance_weight_ess(theta_samples, log_post_fn, cnf, y_obs):
    with torch.no_grad():
        log_target = log_post_fn(theta_samples)
        log_q = cnf.log_prob(theta_samples, y_obs)
        log_w = log_target - log_q
        log_w = log_w - log_w.max()
        w = torch.exp(log_w)
        w = w / w.sum()
        ess_is = float(1.0 / (w ** 2).sum().item())
    return ess_is, ess_is / theta_samples.shape[0]


# ── Loaders ───────────────────────────────────────────────────────────
def load_surrogate():
    m = Surrogate().to(DEV)
    state = torch.load(SURR_PATH, map_location=DEV, weights_only=True)
    m.load_state_dict(state)
    m.eval()
    return m


def load_cnf():
    f = ConditionalRealNVP().to(DEV)
    state = torch.load(CNF_PATH, map_location=DEV, weights_only=True)
    f.load_state_dict(state)
    f.eval()
    return f


# ── θ_MAP via gradient-based optimization on the K-mode posterior ─────
def find_theta_map(surrogate, y_meas, kappa_sq=KAPPA_SQ,
                   prior_sigma=PRIOR_SIGMA, lo=THETA_LO, hi=THETA_HI,
                   n_iters=300, lr=0.08, seed=0):
    """Maximize log-posterior over K measured modes by Adam on log θ
    (so positivity is automatic). Runs all restarts as a batch on GPU.
    """
    torch.manual_seed(seed)
    y_t = torch.tensor(y_meas, dtype=torch.float32, device=DEV)
    K = len(y_meas)

    # Restart bank: 3-point grid per dim (27 inits) + 5 random LogNormal
    grid = torch.linspace(lo, hi, 3, device=DEV)
    g1, g2, g3 = torch.meshgrid(grid, grid, grid, indexing="ij")
    grid_init = torch.stack([g1.flatten(), g2.flatten(), g3.flatten()], 1)
    rand_init = torch.exp(prior_sigma *
                          torch.randn(5, D, device=DEV)).clamp(min=lo, max=hi)
    inits = torch.cat([grid_init, rand_init], dim=0)  # (32, D)

    log_theta = torch.log(inits.clamp(min=lo * 0.5)).clone().detach()
    log_theta.requires_grad_(True)
    opt = torch.optim.Adam([log_theta], lr=lr)
    for _ in range(n_iters):
        opt.zero_grad()
        theta = torch.exp(log_theta)
        f_pred = surrogate(theta)[:, :K]
        resid = (f_pred - y_t) / y_t
        nll_per = 0.5 / kappa_sq * (resid ** 2).sum(dim=1)
        nlp_per = (0.5 * (log_theta / prior_sigma).pow(2).sum(dim=1)
                   + log_theta.sum(dim=1))
        (nll_per + nlp_per).sum().backward()
        opt.step()

    with torch.no_grad():
        theta = torch.exp(log_theta).clamp(min=lo, max=hi)
        f_pred = surrogate(theta)[:, :K]
        resid = (f_pred - y_t) / y_t
        nll_per = 0.5 / kappa_sq * (resid ** 2).sum(dim=1)
        best_k = int(nll_per.argmin().item())
    return theta[best_k].cpu().numpy().copy(), float(nll_per[best_k].item())


# ── Mass diag estimate from fast-mode samples ─────────────────────────
def estimate_mass_diag(cnf, theta_samples, y_obs):
    """Empirical z-space variance from fast-mode θ samples.

    Mass diag M_d = Var(z_d) where z = T(θ; y) under the fixed flow.
    For a well-calibrated flow, this should be close to 1.0 per dim.
    """
    with torch.no_grad():
        y_batch = y_obs.unsqueeze(0).expand(theta_samples.shape[0], -1)
        z, _ = cnf(theta_samples, y_batch)
        var = z.var(dim=0).clamp(min=1e-3)
    return var


# ── Main pipeline ─────────────────────────────────────────────────────
def run_scenario(name, y_full_np, surrogate, cnf, do_exact=True, do_hmc=True,
                 theta_init_hmc=None):
    """Run fast / exact / std-HMC for a single y_full padding."""
    print(f"\n{'='*70}\n  Scenario: {name}\n{'='*70}", flush=True)
    y_full_t = torch.tensor(y_full_np, dtype=torch.float32, device=DEV)
    log_post = KModeLogPosterior(
        surrogate, y_full_np[:N_MEAS],
        kappa_sq=KAPPA_SQ, prior_sigma=PRIOR_SIGMA).to(DEV)

    # (a) Fast mode
    print("\n[a] Fast mode (CNF amortized, N=2000) ...", flush=True)
    torch.manual_seed(42)
    t0 = time.time()
    with torch.no_grad():
        theta_fast = cnf.sample(y_full_t, n_samples=N_FAST)
    t_fast = time.time() - t0
    ess_is, ratio = importance_weight_ess(
        theta_fast, log_post.forward, cnf, y_full_t)
    fast_mean = theta_fast.mean(0).cpu().numpy()
    fast_std = theta_fast.std(0).cpu().numpy()
    print(f"    time={t_fast*1000:.1f} ms   ESS_IS={ess_is:.2f}  "
          f"ratio={ratio:.4f}", flush=True)
    print(f"    mean = {[round(float(x), 4) for x in fast_mean.tolist()]}",
          flush=True)
    print(f"    std  = {[round(float(x), 4) for x in fast_std.tolist()]}",
          flush=True)

    result = {
        "name": name,
        "y_full": [float(v) for v in y_full_np.tolist()],
        "fast_mode": {
            "time_ms": float(t_fast * 1000),
            "posterior_mean": [float(v) for v in fast_mean.tolist()],
            "posterior_std": [float(v) for v in fast_std.tolist()],
            "ess_is": float(ess_is),
            "ess_is_ratio": float(ratio),
        },
    }
    samples = {"fast": theta_fast.detach().cpu().numpy()}

    if do_exact:
        mass_diag = estimate_mass_diag(cnf, theta_fast, y_full_t)
        print(f"\n[b] Exact mode (NeuTra-HMC mass-diag, "
              f"eps={STEP_SIZE}, L={N_LEAPFROG}, N={N_SAMPLES}, "
              f"burn={N_BURNIN}) ...", flush=True)
        print(f"    mass_diag = "
              f"{[round(float(x), 4) for x in mass_diag.cpu().tolist()]}",
              flush=True)
        theta_init = torch.tensor(fast_mean, dtype=torch.float32, device=DEV)
        torch.manual_seed(42)
        sampler = NeuTraHMC_MassDiag(
            log_post.forward, cnf, y_full_t,
            step_size=STEP_SIZE, n_leapfrog=N_LEAPFROG, mass_diag=mass_diag)
        t0 = time.time()
        chain_exact, ar_exact, _ = sampler.run(
            theta_init, n_samples=N_SAMPLES, n_burnin=N_BURNIN)
        t_exact = time.time() - t0
        ess_exact = effective_sample_size(chain_exact)
        exact_mean = chain_exact.mean(0).cpu().numpy()
        exact_std = chain_exact.std(0).cpu().numpy()
        print(f"    time={t_exact:.1f}s  AR={ar_exact*100:.1f}%  "
              f"ESS_min={float(ess_exact.min()):.2f}", flush=True)
        print(f"    mean = {[round(float(x), 4) for x in exact_mean.tolist()]}",
              flush=True)
        print(f"    std  = {[round(float(x), 4) for x in exact_std.tolist()]}",
              flush=True)

        result["exact_mode"] = {
            "step_size": STEP_SIZE,
            "n_leapfrog": N_LEAPFROG,
            "n_samples": N_SAMPLES,
            "n_burnin": N_BURNIN,
            "mass_diag": [float(v) for v in mass_diag.cpu().tolist()],
            "time_s": float(t_exact),
            "acceptance_rate": float(ar_exact),
            "ess_min": float(ess_exact.min()),
            "ess_per_dim": [float(v) for v in ess_exact.tolist()],
            "posterior_mean": [float(v) for v in exact_mean.tolist()],
            "posterior_std": [float(v) for v in exact_std.tolist()],
        }
        samples["exact"] = chain_exact.detach().cpu().numpy()

    if do_hmc:
        print(f"\n[c] Standard HMC baseline (N={N_SAMPLES}, "
              f"burn={N_BURNIN}) ...", flush=True)
        torch.manual_seed(42)
        hmc = HMCSampler(log_post.grad_log_prob,
                         step_size=0.01, n_leapfrog=20)
        t0 = time.time()
        x0_hmc = (theta_init_hmc.clone() if theta_init_hmc is not None
                  else torch.ones(D, device=DEV))
        chain_hmc, ar_hmc = hmc.run(
            x0_hmc, n_samples=N_SAMPLES, n_burnin=N_BURNIN)
        t_hmc = time.time() - t0
        ess_hmc = effective_sample_size(chain_hmc)
        hmc_mean = chain_hmc.mean(0).cpu().numpy()
        hmc_std = chain_hmc.std(0).cpu().numpy()
        print(f"    time={t_hmc:.1f}s  AR={ar_hmc*100:.1f}%  "
              f"ESS_min={float(ess_hmc.min()):.2f}", flush=True)
        print(f"    mean = {[round(float(x), 4) for x in hmc_mean.tolist()]}",
              flush=True)

        result["std_hmc"] = {
            "time_s": float(t_hmc),
            "acceptance_rate": float(ar_hmc),
            "ess_min": float(ess_hmc.min()),
            "ess_per_dim": [float(v) for v in ess_hmc.tolist()],
            "posterior_mean": [float(v) for v in hmc_mean.tolist()],
            "posterior_std": [float(v) for v in hmc_std.tolist()],
        }
        samples["std_hmc"] = chain_hmc.detach().cpu().numpy()

    return result, samples


def fit_quality(surrogate, theta_mean, y_meas):
    """Report surrogate(θ_mean) vs measured freqs for the first 6 modes."""
    with torch.no_grad():
        f_pred = surrogate(
            torch.tensor(theta_mean, dtype=torch.float32,
                         device=DEV).unsqueeze(0)
        ).squeeze().cpu().numpy()
    rel_err = (f_pred[:N_MEAS] - y_meas) / y_meas
    rms = float(np.sqrt((rel_err ** 2).mean()) * 100)
    return f_pred, rel_err, rms


def main():
    t_total = time.time()
    print(f"[load] surrogate: {SURR_PATH}", flush=True)
    print(f"[load] cnf:       {CNF_PATH}", flush=True)
    surrogate = load_surrogate()
    cnf = load_cnf()

    # Sanity surrogate(θ=1)
    with torch.no_grad():
        f_nominal = surrogate(torch.ones(1, D, device=DEV)).squeeze().cpu().numpy()
    print(f"[check] surrogate(θ=1) modes 1-6: "
          f"{[round(float(v), 3) for v in f_nominal[:6].tolist()]}", flush=True)
    print(f"[check] measured  modes 1-6:     "
          f"{MEASURED_FREQS}", flush=True)

    # Find θ_MAP first so y_full is self-consistent with the manifold the
    # CNF saw during training. Padding with surrogate(θ=1)[6:] when the
    # measured modes 1-6 imply θ≈3 puts y_full off-manifold and breaks the
    # CNF (whose conditional inversion is only valid near the training data).
    y_meas = np.asarray(MEASURED_FREQS, dtype=np.float32)
    print("\n[pad] gradient-based θ_MAP on the 6 measured modes ...",
          flush=True)
    t_map0 = time.time()
    theta_map, map_nll = find_theta_map(surrogate, y_meas)
    print(f"  theta_MAP = {[round(float(v), 4) for v in theta_map.tolist()]}  "
          f"(NLL={map_nll:.3f}, time={time.time()-t_map0:.2f}s)", flush=True)
    with torch.no_grad():
        f_at_map = surrogate(torch.tensor(theta_map, dtype=torch.float32,
                                          device=DEV).unsqueeze(0))[0].cpu().numpy()
    rms_map = float(np.sqrt(np.mean(
        ((f_at_map[:N_MEAS] - y_meas) / y_meas) ** 2)) * 100)
    print(f"  surrogate(theta_MAP) modes 1-6: "
          f"{[round(float(v), 3) for v in f_at_map[:N_MEAS].tolist()]}",
          flush=True)
    print(f"  RMS rel err at theta_MAP: {rms_map:.2f}%", flush=True)

    # Self-consistent padding using θ_MAP
    y_full = np.concatenate([y_meas, f_at_map[N_MEAS:]]).astype(np.float32)
    print(f"  y_full[0:6]  (measured): "
          f"{[round(float(v), 3) for v in y_full[:N_MEAS].tolist()]}",
          flush=True)
    print(f"  y_full[6:10] (padding):  "
          f"{[round(float(v), 3) for v in y_full[N_MEAS:N_MEAS+4].tolist()]}"
          f" ...", flush=True)

    theta_map_t = torch.tensor(theta_map, dtype=torch.float32, device=DEV)
    # ── Stage 3: normal scenario ──────────────────────────────────────
    res_normal, smp_normal = run_scenario(
        "Normal (Hu et al. 2018 measured)", y_full, surrogate, cnf,
        do_exact=True, do_hmc=True, theta_init_hmc=theta_map_t)

    # Fit at fast / exact posterior mean
    print("\n[fit] surrogate(θ_post) vs measured freqs (Normal scenario)",
          flush=True)
    fit_summary = {}
    for label, key in [("fast", "fast_mode"), ("exact", "exact_mode")]:
        if key not in res_normal:
            continue
        f_pred, rel, rms = fit_quality(
            surrogate, np.asarray(res_normal[key]["posterior_mean"]), y_meas)
        print(f"  {label} mean → RMS rel err = {rms:.2f}%", flush=True)
        for i in range(N_MEAS):
            print(f"    f{i+1}: meas={y_meas[i]:.3f}  "
                  f"pred={f_pred[i]:.3f}  err={rel[i]*100:+.2f}%", flush=True)
        fit_summary[label] = {
            "predicted_freqs": [float(v) for v in f_pred[:N_MEAS].tolist()],
            "rel_error_pct": [float(v * 100) for v in rel.tolist()],
            "rms_rel_error_pct": rms,
        }
    res_normal["fit_quality"] = fit_summary

    # Posterior figure (3 dims × histograms for fast/exact/hmc)
    fig, axes = plt.subplots(1, D, figsize=(4 * D, 4))
    for i, ax in enumerate(axes):
        ax.hist(smp_normal["fast"][:, i], bins=40, density=True,
                alpha=0.45, label="Fast (CNF)")
        if "exact" in smp_normal:
            ax.hist(smp_normal["exact"][:, i], bins=40, density=True,
                    alpha=0.45, label="Exact (NeuTra-HMC)")
        if "std_hmc" in smp_normal:
            ax.hist(smp_normal["std_hmc"][:, i], bins=40, density=True,
                    alpha=0.45, label="Std HMC")
        ax.set_xlabel(PARAM_NAMES[i])
        if i == 0:
            ax.set_ylabel("density")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    plt.suptitle(f"Coupled building posterior (Hu et al. 2018) — "
                 f"eps={STEP_SIZE}, L={N_LEAPFROG}, diag-mass")
    plt.tight_layout()
    pdf_path = OUT_FIG / "fig_coupled_posterior.pdf"
    png_path = OUT_FIG / "fig_coupled_posterior.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[save] {pdf_path}", flush=True)
    print(f"[save] {png_path}", flush=True)

    # ── Stage 4: damage detection ─────────────────────────────────────
    print(f"\n{'='*70}\n  Stage 4: damage detection\n{'='*70}", flush=True)
    theta_damage = torch.tensor([[0.5, 1.0, 1.0]],
                                device=DEV, dtype=torch.float32)
    with torch.no_grad():
        f_damage = surrogate(theta_damage).squeeze().cpu().numpy()
    y_damage = f_damage.astype(np.float32)

    res_damage, smp_damage = run_scenario(
        "Damage (e1=0.5, others=1.0)", y_damage, surrogate, cnf,
        do_exact=False, do_hmc=False)

    # Comparison plot — fast-mode densities, normal vs damage
    fig, axes = plt.subplots(1, D, figsize=(4 * D, 4))
    for i, ax in enumerate(axes):
        ax.hist(smp_normal["fast"][:, i], bins=40, density=True,
                alpha=0.45, label="Normal")
        ax.hist(smp_damage["fast"][:, i], bins=40, density=True,
                alpha=0.45, label="Damage (e1=0.5)")
        ax.axvline(1.0, color="gray", ls="--", alpha=0.5,
                   label="theta = 1.0")
        if i == 0:
            ax.axvline(0.5, color="red", ls="--", alpha=0.5, label="true e1")
        ax.set_xlabel(PARAM_NAMES[i])
        if i == 0:
            ax.set_ylabel("density")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    plt.suptitle("Coupled building damage detection — fast-mode posteriors")
    plt.tight_layout()
    pdf_d = OUT_FIG / "fig_coupled_damage.pdf"
    png_d = OUT_FIG / "fig_coupled_damage.png"
    plt.savefig(pdf_d, bbox_inches="tight")
    plt.savefig(png_d, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[save] {pdf_d}", flush=True)
    print(f"[save] {png_d}", flush=True)

    # ── Save results ──────────────────────────────────────────────────
    out = {
        "config": {
            "D": D, "M": M, "n_measured": N_MEAS,
            "measured_freqs_hz": [float(v) for v in MEASURED_FREQS],
            "kappa_sq": KAPPA_SQ,
            "prior_sigma": PRIOR_SIGMA,
            "theta_lo": THETA_LO, "theta_hi": THETA_HI,
            "step_size": STEP_SIZE, "n_leapfrog": N_LEAPFROG,
            "n_samples": N_SAMPLES, "n_burnin": N_BURNIN,
            "n_fast": N_FAST,
            "surrogate_path": str(SURR_PATH),
            "cnf_path": str(CNF_PATH),
        },
        "theta_map_padding": {
            "theta_map": [float(v) for v in theta_map.tolist()],
            "map_nll": float(map_nll),
            "rms_rel_err_pct": rms_map,
            "surrogate_at_map_modes_1_6": [float(v) for v in f_at_map[:N_MEAS].tolist()],
        },
        "normal_scenario": res_normal,
        "damage_scenario": res_damage,
    }
    out_json = OUT_RES / "coupled_results.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n[save] {out_json}", flush=True)

    # Final summary
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}", flush=True)
    print(f"  Normal scenario (y = Hu et al. 2018 measured):", flush=True)
    print(f"    fast    mean = {[round(v, 4) for v in res_normal['fast_mode']['posterior_mean']]}",
          flush=True)
    print(f"    fast    ESS_IS/N = {res_normal['fast_mode']['ess_is_ratio']:.4f}",
          flush=True)
    if "exact_mode" in res_normal:
        print(f"    exact   mean = {[round(v, 4) for v in res_normal['exact_mode']['posterior_mean']]}",
              flush=True)
        print(f"    exact   AR = {res_normal['exact_mode']['acceptance_rate']*100:.1f}%   "
              f"ESS_min = {res_normal['exact_mode']['ess_min']:.2f}",
              flush=True)
    if "std_hmc" in res_normal:
        print(f"    std-hmc mean = {[round(v, 4) for v in res_normal['std_hmc']['posterior_mean']]}",
              flush=True)
        print(f"    std-hmc AR = {res_normal['std_hmc']['acceptance_rate']*100:.1f}%   "
              f"ESS_min = {res_normal['std_hmc']['ess_min']:.2f}",
              flush=True)
    if "fit_quality" in res_normal and "exact" in res_normal["fit_quality"]:
        print(f"    fit RMS rel err = "
              f"{res_normal['fit_quality']['exact']['rms_rel_error_pct']:.2f}% (exact)",
              flush=True)

    print(f"\n  Damage scenario (e1=0.5, others=1):", flush=True)
    print(f"    fast    mean = {[round(v, 4) for v in res_damage['fast_mode']['posterior_mean']]}",
          flush=True)
    print(f"    fast    ESS_IS/N = {res_damage['fast_mode']['ess_is_ratio']:.4f}",
          flush=True)

    print(f"\n[done] total time: {(time.time()-t_total)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
