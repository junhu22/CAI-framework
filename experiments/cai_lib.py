"""
Shared CAI surrogate, flow, likelihood and sampler definitions.

Architectures match experiments/corrected_amortized.py and the trained
state dictionaries described in REPRODUCIBILITY.md. Importing this module
does not launch an experiment.
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DEV = torch.device("cpu")


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
    """ResNet surrogate: θ (D) → predicted modal frequencies (M)."""

    def __init__(self, nin=6, nout=20, hid=256, nb=4):
        super().__init__()
        self.nin, self.nout = nin, nout
        self.inp = nn.Sequential(nn.Linear(nin, hid), nn.SiLU())
        self.blk = nn.ModuleList([ResBlock(hid) for _ in range(nb)])
        self.out = nn.Sequential(
            nn.LayerNorm(hid), nn.SiLU(), nn.Linear(hid, nout))
        for n in ("xm", "xs", "ym", "ys"):
            self.register_buffer(n, torch.zeros(nin if "x" in n else nout))

    def _fwd_n(self, xn):
        """Forward pass on already-normalized inputs; returns normalized output.
        Mirrors 03_train_surrogate.py:_fwd_n so training MSE is computed in
        normalized space and every mode contributes equally to the loss."""
        h = self.inp(xn)
        for b in self.blk:
            h = b(h)
        return self.out(h)

    def set_norm(self, X, Y):
        self.xm, self.xs = X.mean(0), X.std(0).clamp(min=1e-8)
        self.ym, self.ys = Y.mean(0), Y.std(0).clamp(min=1e-8)

    def forward(self, x):
        return self._fwd_n((x - self.xm) / self.xs) * self.ys + self.ym


def load_surrogate(member_path):
    m = Surrogate().to(DEV)
    state = torch.load(member_path, map_location=DEV, weights_only=True)
    m.load_state_dict(state)
    m.eval()
    return m


# ── Conditional RealNVP ───────────────────────────────────────────────
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
    def __init__(self, dim_theta=6, dim_y=20, n_layers=8, hidden_dim=128):
        super().__init__()
        self.dim_theta, self.dim_y = dim_theta, dim_y
        self.layers = nn.ModuleList([
            ConditionalAffineCoupling(dim_theta, dim_y, hidden_dim,
                                      mask_even=(i % 2 == 0))
            for i in range(n_layers)
        ])
        self.y_embed = nn.Sequential(
            nn.Linear(dim_y, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, dim_y),
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
        """log q(θ|y) — needed for importance-weight diagnostics."""
        if y.dim() == 1:
            y_batch = y.unsqueeze(0).expand(theta.shape[0], -1)
        else:
            y_batch = y
        z, log_det = self.forward(theta, y_batch)
        log_pz = (-0.5 * (z ** 2).sum(dim=-1)
                  - 0.5 * self.dim_theta * math.log(2 * math.pi))
        return log_pz + log_det


def load_cnf(cnf_path, dim_theta=6, dim_y=20, n_layers=8, hidden_dim=128):
    cnf = ConditionalRealNVP(dim_theta, dim_y, n_layers, hidden_dim).to(DEV)
    state = torch.load(cnf_path, map_location=DEV, weights_only=True)
    cnf.load_state_dict(state)
    cnf.eval()
    return cnf


# ── Log-posterior on K (≤ M) measured modes ──────────────────────────
class KModeLogPosterior(nn.Module):
    """Log posterior using only the first K modes from the surrogate output.

    log π(θ|y) = -0.5/κ² Σ_{i=1..K} ((f_pred_i - y_i) / y_i)² + log-normal prior
    """

    def __init__(self, surrogate, observed_freqs, kappa_sq=1e-3, prior_sigma=0.3):
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


# ── NeuTra-HMC with conditional flow ─────────────────────────────────
class NeuTraHMC_Conditional:
    def __init__(self, log_post_fn, cond_flow, y_obs,
                 step_size=0.05, n_leapfrog=20):
        self.log_post = log_post_fn
        self.flow = cond_flow
        self.y_obs = y_obs
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog

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
            q = q + eps * p
            _, g = self._grad_log_prob_z(q)
            p = p + eps * g
        q = q + eps * p
        lp_final, g = self._grad_log_prob_z(q)
        p = p + 0.5 * eps * g
        return q, p, lp_final, g

    def run(self, theta0, n_samples, n_burnin=500):
        self.flow.eval()
        with torch.no_grad():
            z, _ = self.flow(theta0.unsqueeze(0), self._y_batch(1))
        z = z.squeeze(0)
        z_chain = []
        n_acc = 0
        n_acc_adapt = 0
        eps = self.step_size
        lp_current, grad_current = self._grad_log_prob_z(z)
        for i in range(-n_burnin, n_samples):
            p = torch.randn_like(z)
            H0 = -lp_current + 0.5 * (p ** 2).sum()
            q_prop, p_prop, lp_prop, grad_prop = self._leapfrog(
                z, p, eps, grad_init=grad_current)
            H1 = -lp_prop + 0.5 * (p_prop ** 2).sum()
            if torch.log(torch.rand(1)).item() < (H0 - H1).item():
                z = q_prop
                lp_current = lp_prop
                grad_current = grad_prop
                if i >= 0:
                    n_acc += 1
                n_acc_adapt += 1
            if i >= 0:
                z_chain.append(z.clone())
            if i < 0 and (i + n_burnin) % 50 == 49:
                rate = n_acc_adapt / 50
                eps *= math.exp(0.5 * (rate - 0.80))
                eps = max(1e-5, min(eps, 1.0))
                n_acc_adapt = 0
        z_stack = torch.stack(z_chain)
        with torch.no_grad():
            y_batch = self._y_batch(len(z_chain))
            theta_chain, _ = self.flow.inverse(z_stack, y_batch)
        return theta_chain, n_acc / n_samples


# ── Standard HMC (no flow) ───────────────────────────────────────────
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


# ── Diagnostics ──────────────────────────────────────────────────────
def effective_sample_size(chain):
    if isinstance(chain, torch.Tensor):
        chain = chain.cpu()
    N, D = chain.shape
    ess = torch.zeros(D)
    for d in range(D):
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
    """Self-normalized IS ESS for an amortized-fit diagnostic.

    Returns ESS_IS and ESS_IS / N. Closer to 1 = fast-mode well-calibrated.
    """
    with torch.no_grad():
        log_target = log_post_fn(theta_samples)
        log_q = cnf.log_prob(theta_samples, y_obs)
        log_w = log_target - log_q
        log_w = log_w - log_w.max()
        w = torch.exp(log_w)
        w = w / w.sum()
        ess_is = float(1.0 / (w ** 2).sum().item())
    return ess_is, ess_is / theta_samples.shape[0]
