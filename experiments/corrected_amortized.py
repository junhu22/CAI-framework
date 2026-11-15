#!/usr/bin/env python3
"""
Corrected Amortized Inference for Bayesian Structural Identification.

A conditional flow provides an approximate posterior and a coordinate
transformation for NeuTra-HMC. Corrected samples target the surrogate-based
posterior under the usual MCMC convergence conditions. The transport route
is demonstrated on analytical and synthetic cases; the two field targets
in the article use adaptive NUTS.

Architecture:
  Conditional RealNVP with an analytic inverse;
  a differentiable neural surrogate for modal frequencies;
  HMC in transformed coordinates.

Historical experiment driver. Configure paths below before running.
See README.md and REPRODUCIBILITY.md for the published result mapping.
"""

import sys
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SURROGATE_DIR = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
MODEL_DIR = SURROGATE_DIR / "surrogate_models"
TEST_CSV = SURROGATE_DIR / "dataset_test.csv"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SURROGATE_DIR))

from config import N_PARAMS, N_MODES, PARAM_NAMES

DEV = torch.device("cpu")
D = N_PARAMS   # 6
M = N_MODES    # 20


# ═══════════════════════════════════════════════════════════════════
#  Section 1: NN Surrogate (same as before, loads trained model)
# ═══════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, d, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.SiLU(), nn.Linear(d, d),
            nn.LayerNorm(d), nn.SiLU(), nn.Dropout(drop), nn.Linear(d, d))
    def forward(self, x): return x + self.net(x)

class Surrogate(nn.Module):
    def __init__(self, nin=D, nout=M, hid=256, nb=4):
        super().__init__()
        self.nin, self.nout = nin, nout
        self.inp = nn.Sequential(nn.Linear(nin, hid), nn.SiLU())
        self.blk = nn.ModuleList([ResBlock(hid) for _ in range(nb)])
        self.out = nn.Sequential(nn.LayerNorm(hid), nn.SiLU(), nn.Linear(hid, nout))
        for n in ("xm","xs","ym","ys"):
            self.register_buffer(n, torch.zeros(nin if "x" in n else nout))

    def set_norm(self, X, Y):
        self.xm, self.xs = X.mean(0), X.std(0).clamp(min=1e-8)
        self.ym, self.ys = Y.mean(0), Y.std(0).clamp(min=1e-8)

    def forward(self, x):
        xn = (x - self.xm) / self.xs
        h = self.inp(xn)
        for b in self.blk: h = b(h)
        return self.out(h) * self.ys + self.ym


def load_nn_surrogate(member=0):
    model = Surrogate(nin=D, nout=M).to(DEV)
    state = torch.load(MODEL_DIR / f"member_{member}.pt",
                       map_location=DEV, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════
#  Section 2: Log-Posterior
# ═══════════════════════════════════════════════════════════════════

class NNLogPosterior(nn.Module):
    def __init__(self, surrogate, observed_freqs_hz, kappa_sq=1e-3,
                 prior_sigma=0.3):
        super().__init__()
        self.surrogate = surrogate
        self.register_buffer("f_obs",
            torch.tensor(observed_freqs_hz, dtype=torch.float32))
        self.kappa_sq = kappa_sq
        self.prior_sigma = prior_sigma

    def forward(self, theta):
        """theta: (B, D) or (D,) → log π(θ|y)"""
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        f_pred = self.surrogate(theta)
        resid = (f_pred - self.f_obs) / self.f_obs
        ll = -0.5 / self.kappa_sq * (resid ** 2).sum(dim=-1)
        if (theta <= 0).any(dim=-1).any():
            lp = torch.full_like(ll, -1e30)
            valid = (theta > 0).all(dim=-1)
            log_t = torch.log(theta[valid].clamp(min=1e-30))
            lp[valid] = ll[valid] + (
                -0.5 * (log_t / self.prior_sigma)**2 - log_t
            ).sum(dim=-1)
            return lp
        log_t = torch.log(theta)
        lp_prior = (-0.5 * (log_t / self.prior_sigma)**2 - log_t).sum(dim=-1)
        return ll + lp_prior

    def grad_log_prob(self, theta):
        theta = theta.detach().requires_grad_(True)
        lp = self.forward(theta).sum()
        grad = torch.autograd.grad(lp, theta)[0]
        return lp.detach(), grad.detach()


# ═══════════════════════════════════════════════════════════════════
#  Section 3: ★ Conditional RealNVP ★
# ═══════════════════════════════════════════════════════════════════

class ConditionalAffineCoupling(nn.Module):
    """
    Affine coupling layer conditioned on observation y.

    The scale/translation network takes [x_masked ; y] as input,
    so the transform adapts to each observation automatically.
    """
    def __init__(self, dim_theta, dim_y, hidden_dim=128, mask_even=True):
        super().__init__()
        self.dim_theta = dim_theta
        mask = torch.zeros(dim_theta)
        if mask_even:
            mask[::2] = 1.0
        else:
            mask[1::2] = 1.0
        self.register_buffer("mask", mask)

        # Input: masked θ (dim_theta) + observation y (dim_y)
        self.net = nn.Sequential(
            nn.Linear(dim_theta + dim_y, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, dim_theta * 2),
        )
        # Initialize near identity
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()

    def forward(self, x, y):
        """x: (B, D), y: (B, M) → z: (B, D), log_det: (B,)"""
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
        """z: (B, D), y: (B, M) → x: (B, D), log_det: (B,)"""
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
    """
    Conditional RealNVP: T(θ; y) → z  and  T⁻¹(z; y) → θ

    Given observation y, this flow maps posterior samples θ ~ π(·|y)
    to approximately standard Gaussian z ≈ N(0, I).

    Key properties:
      - Analytic O(1) inverse (crucial for NeuTra-HMC)
      - Observation-conditioned (enables amortized inference)
      - Exact log-determinant computation
    """
    def __init__(self, dim_theta, dim_y, n_layers=8, hidden_dim=128):
        super().__init__()
        self.layers = nn.ModuleList([
            ConditionalAffineCoupling(
                dim_theta, dim_y, hidden_dim, mask_even=(i % 2 == 0))
            for i in range(n_layers)
        ])
        self.dim_theta = dim_theta
        self.dim_y = dim_y

        # Learnable observation embedding (compresses y into a summary)
        self.y_embed = nn.Sequential(
            nn.Linear(dim_y, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, dim_y),
        )

    def _embed_y(self, y):
        return self.y_embed(y)

    def forward(self, theta, y):
        """θ → z conditioned on y. Returns (z, log_det)."""
        y_emb = self._embed_y(y)
        log_det = torch.zeros(theta.shape[0], device=theta.device)
        z = theta
        for layer in self.layers:
            z, ld = layer(z, y_emb)
            log_det += ld
        return z, log_det

    def inverse(self, z, y):
        """z → θ conditioned on y. Returns (θ, log_det_inv)."""
        y_emb = self._embed_y(y)
        log_det = torch.zeros(z.shape[0], device=z.device)
        x = z
        for layer in reversed(self.layers):
            x, ld = layer.inverse(x, y_emb)
            log_det += ld
        return x, log_det

    def sample(self, y, n_samples=1000):
        """
        FAST MODE: Draw approximate posterior samples for observation y.
        No MCMC needed — direct ancestral sampling through inverse flow.

        y: (M,) single observation → returns (n_samples, D) θ-samples
        """
        y_batch = y.unsqueeze(0).expand(n_samples, -1)
        z = torch.randn(n_samples, self.dim_theta, device=y.device)
        with torch.no_grad():
            theta, _ = self.inverse(z, y_batch)
        return theta

    def log_prob(self, theta, y):
        """Evaluate log q(θ|y) = log N(z;0,I) + log|det dT/dθ|."""
        z, log_det = self.forward(theta, y)
        log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * self.dim_theta * math.log(2 * math.pi)
        return log_pz + log_det


# ═══════════════════════════════════════════════════════════════════
#  Section 4: Training Data Generation
# ═══════════════════════════════════════════════════════════════════

def generate_training_data(surrogate, n_train=50000, n_val=5000,
                           kappa=math.sqrt(1e-3), prior_sigma=0.3):
    """
    Generate (θ, y) pairs by:
      1. Sample θ ~ p(θ) = LogNormal(0, σ²)
      2. Compute y = f̂(θ) + ε,  ε ~ N(0, κ²·diag(y²))

    This creates a dataset of observation-parameter pairs that the
    conditional NF will learn to invert.
    """
    print(f"  Generating {n_train + n_val} (θ, y) pairs...")

    all_theta = []
    all_y = []

    total = n_train + n_val
    batch_size = 2000

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            B = end - start

            # Sample θ from log-normal prior
            log_theta = prior_sigma * torch.randn(B, D, device=DEV)
            theta = torch.exp(log_theta)

            # Compute clean frequencies
            f_clean = surrogate(theta)  # (B, M)

            # Add relative Gaussian noise
            noise = kappa * f_clean * torch.randn_like(f_clean)
            y = f_clean + noise

            all_theta.append(theta)
            all_y.append(y)

    theta_all = torch.cat(all_theta, dim=0)
    y_all = torch.cat(all_y, dim=0)

    # Split
    theta_train, theta_val = theta_all[:n_train], theta_all[n_train:]
    y_train, y_val = y_all[:n_train], y_all[n_train:]

    print(f"  θ range: [{theta_all.min():.3f}, {theta_all.max():.3f}]")
    print(f"  y range: [{y_all.min():.3f}, {y_all.max():.3f}]")

    return theta_train, y_train, theta_val, y_val


# ═══════════════════════════════════════════════════════════════════
#  Section 5: Conditional NF Training
# ═══════════════════════════════════════════════════════════════════

def train_conditional_nf(flow, theta_train, y_train, theta_val, y_val,
                         n_epochs=500, lr=1e-3, batch_size=512):
    """
    Train conditional NF via forward KL:
      L = -E_{(θ,y)~data}[ log q(θ|y) ]
        = -E[ log N(T(θ;y); 0,I) + log|det J_T| ]
    """
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    dataset = torch.utils.data.TensorDataset(theta_train, y_train)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    best_val_loss = float("inf")
    best_state = None
    patience = 50
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        # Train
        flow.train()
        total_loss = 0; n = 0
        for theta_b, y_b in loader:
            z, log_det = flow(theta_b, y_b)
            log_pz = -0.5 * (z ** 2).sum(dim=-1)
            loss = -(log_pz + log_det).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() * theta_b.shape[0]
            n += theta_b.shape[0]
        scheduler.step()

        train_loss = total_loss / n

        # Validate
        flow.eval()
        with torch.no_grad():
            z_val, ld_val = flow(theta_val, y_val)
            log_pz_val = -0.5 * (z_val ** 2).sum(dim=-1)
            val_loss = -(log_pz_val + ld_val).mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in flow.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}: train={train_loss:.4f}, "
                  f"val={val_loss:.4f}, best={best_val_loss:.4f}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Restore best
    flow.load_state_dict(best_state)
    flow.eval()
    print(f"  Best validation loss: {best_val_loss:.4f}")
    return flow


# ═══════════════════════════════════════════════════════════════════
#  Section 6: NeuTra-HMC with Conditional Flow (Exact Mode)
# ═══════════════════════════════════════════════════════════════════

class NeuTraHMC_Conditional:
    """
    NeuTra-HMC using a Conditional NF as transport map.

    Given observation y:
      - Maps θ → z via T(θ; y)
      - Runs HMC in z-space targeting π_z(z|y)
      - Maps z → θ via T⁻¹(z; y)

    The conditional NF is FIXED during sampling (Proposition 2),
    guaranteeing convergence to the exact posterior.
    """
    def __init__(self, log_post_fn, cond_flow, y_obs,
                 step_size=0.05, n_leapfrog=20):
        self.log_post = log_post_fn
        self.flow = cond_flow
        self.y_obs = y_obs          # (M,) observation
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog

    def _y_batch(self, n=1):
        return self.y_obs.unsqueeze(0).expand(n, -1)

    def _grad_log_prob_z(self, z):
        """Gradient of log π_z(z|y) w.r.t. z."""
        z = z.detach().requires_grad_(True)
        theta, log_det = self.flow.inverse(z.unsqueeze(0), self._y_batch(1))
        theta = theta.squeeze(0); log_det = log_det.squeeze()
        lp = self.log_post(theta.unsqueeze(0)).squeeze()
        total = lp + log_det
        grad = torch.autograd.grad(total, z)[0]
        return total.detach(), grad.detach()

    def _leapfrog(self, q, p, eps, grad_init=None):
        if grad_init is not None:
            g = grad_init
        else:
            _, g = self._grad_log_prob_z(q)
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

        # Map θ₀ → z₀
        with torch.no_grad():
            z, _ = self.flow(theta0.unsqueeze(0), self._y_batch(1))
        z = z.squeeze(0)

        z_chain = []
        n_acc = 0; n_acc_adapt = 0
        eps = self.step_size

        lp_current, grad_current = self._grad_log_prob_z(z)

        for i in range(-n_burnin, n_samples):
            p = torch.randn_like(z)
            H0 = -lp_current + 0.5 * (p**2).sum()

            q_prop, p_prop, lp_prop, grad_prop = self._leapfrog(
                z, p, eps, grad_init=grad_current)
            H1 = -lp_prop + 0.5 * (p_prop**2).sum()

            if torch.log(torch.rand(1)).item() < (H0 - H1).item():
                z = q_prop
                lp_current = lp_prop
                grad_current = grad_prop
                if i >= 0: n_acc += 1
                n_acc_adapt += 1

            if i >= 0:
                z_chain.append(z.clone())

            if i < 0 and (i + n_burnin) % 50 == 49:
                rate = n_acc_adapt / 50
                eps *= math.exp(0.5 * (rate - 0.80))
                eps = max(1e-5, min(eps, 1.0))
                n_acc_adapt = 0

        # Batch inverse
        z_stack = torch.stack(z_chain)
        with torch.no_grad():
            y_batch = self._y_batch(len(z_chain))
            theta_chain, _ = self.flow.inverse(z_stack, y_batch)

        return theta_chain, n_acc / n_samples


# ═══════════════════════════════════════════════════════════════════
#  Section 7: Standard HMC (no flow, for comparison)
# ═══════════════════════════════════════════════════════════════════

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
        x = x0.clone(); eps = self.step_size
        chain = []; n_acc = 0; n_acc_adapt = 0
        lp_current, grad_current = self.grad_lp(x)

        for i in range(-n_burnin, n_samples):
            p = torch.randn_like(x)
            H0 = -lp_current + 0.5 * (p**2).sum()
            q_prop, p_prop, lp_prop, grad_prop = self._leapfrog(
                x, p, eps, grad_init=grad_current)
            H1 = -lp_prop + 0.5 * (p_prop**2).sum()
            if torch.log(torch.rand(1)).item() < (H0 - H1).item():
                x = q_prop; lp_current = lp_prop; grad_current = grad_prop
                if i >= 0: n_acc += 1
                n_acc_adapt += 1
            if i >= 0: chain.append(x.clone())
            if i < 0 and (i + n_burnin) % 50 == 49:
                rate = n_acc_adapt / 50
                eps *= math.exp(0.5 * (rate - 0.75))
                eps = max(1e-5, min(eps, 1.0))
                n_acc_adapt = 0

        return torch.stack(chain), n_acc / n_samples


# ═══════════════════════════════════════════════════════════════════
#  Section 8: ESS Computation
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
#  Section 9: Main Experiment
# ═══════════════════════════════════════════════════════════════════

def load_observed_freqs():
    import csv
    with open(TEST_CSV) as f:
        reader = csv.DictReader(f)
        row = next(reader)
        freqs = np.array([float(row[f"f{j+1}"]) for j in range(M)])
        theta_true = np.array([float(row[n]) for n in PARAM_NAMES])
    return freqs, theta_true


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    N_SAMPLES = 2000
    N_BURNIN = 500

    print("=" * 70)
    print("  Corrected Amortized Inference — Sailboat Building D=6")
    print("=" * 70)

    # ── 1. Load surrogate & observation ────────────────────────────
    print("\n[1] Loading NN surrogate and observation...")
    surrogate = load_nn_surrogate(member=0).to(DEV)
    f_obs, theta_true = load_observed_freqs()
    f_obs_t = torch.tensor(f_obs, dtype=torch.float32, device=DEV)
    print(f"  θ_true = {theta_true.round(4)}")

    log_post = NNLogPosterior(surrogate, f_obs, kappa_sq=1e-3).to(DEV)

    # ── 2. Generate training data ─────────────────────────────────
    print("\n[2] Generating training data...")
    theta_train, y_train, theta_val, y_val = generate_training_data(
        surrogate, n_train=50000, n_val=5000)
    print(f"  Train: {theta_train.shape}, Val: {theta_val.shape}")

    # ── 3. Build and train Conditional RealNVP ────────────────────
    print("\n[3] Training Conditional RealNVP (8 layers, 128 hidden)...")
    cond_flow = ConditionalRealNVP(
        dim_theta=D, dim_y=M, n_layers=8, hidden_dim=128).to(DEV)

    n_params_flow = sum(p.numel() for p in cond_flow.parameters())
    print(f"  Flow parameters: {n_params_flow:,}")

    cond_flow = train_conditional_nf(
        cond_flow, theta_train, y_train, theta_val, y_val,
        n_epochs=500, lr=1e-3, batch_size=512)

    # Flow quality check
    with torch.no_grad():
        z_test, ld_test = cond_flow(
            theta_val[:100], y_val[:100])
        x_back, _ = cond_flow.inverse(z_test, y_val[:100])
        recon_err = (x_back - theta_val[:100]).abs().max().item()
    print(f"  Reconstruction error: {recon_err:.2e}")

    # ── 4. Fast Mode: Amortized posterior ─────────────────────────
    print(f"\n[4] FAST MODE: Amortized posterior (N={N_SAMPLES})...")
    t0 = time.time()
    theta_fast = cond_flow.sample(f_obs_t, n_samples=N_SAMPLES)
    t_fast = time.time() - t0
    print(f"  Time: {t_fast*1000:.1f} ms")
    print(f"  Mean: {theta_fast.mean(0).numpy().round(4)}")
    print(f"  Std:  {theta_fast.std(0).numpy().round(4)}")

    # ── 5. Exact Mode: NeuTra-HMC correction ─────────────────────
    print(f"\n[5] EXACT MODE: NeuTra-HMC correction (N={N_SAMPLES})...")
    x0 = torch.ones(D, dtype=torch.float32, device=DEV)

    t0 = time.time()
    neutra_hmc = NeuTraHMC_Conditional(
        log_post.forward, cond_flow, f_obs_t,
        step_size=0.05, n_leapfrog=20)
    chain_exact, ar_exact = neutra_hmc.run(x0, N_SAMPLES, N_BURNIN)
    t_exact = time.time() - t0
    ess_exact = effective_sample_size(chain_exact)
    print(f"  Accept: {ar_exact*100:.1f}%")
    print(f"  Time: {t_exact:.1f}s")
    print(f"  ESS_min: {ess_exact.min():.0f}")
    print(f"  Mean: {chain_exact.mean(0).numpy().round(4)}")

    # ── 6. Baseline: Standard HMC (no flow) ───────────────────────
    print(f"\n[6] BASELINE: Standard HMC (N={N_SAMPLES})...")
    t0 = time.time()
    chain_hmc, ar_hmc = HMCSampler(
        log_post.grad_log_prob, step_size=0.01, n_leapfrog=20
    ).run(x0, N_SAMPLES, N_BURNIN)
    t_hmc = time.time() - t0
    ess_hmc = effective_sample_size(chain_hmc)
    print(f"  Accept: {ar_hmc*100:.1f}%")
    print(f"  Time: {t_hmc:.1f}s")
    print(f"  ESS_min: {ess_hmc.min():.0f}")
    print(f"  Mean: {chain_hmc.mean(0).numpy().round(4)}")

    # ── 7. NEW TEST: Amortized on a DIFFERENT observation ─────────
    print(f"\n[7] GENERALIZATION: Amortized on unseen observation...")
    # Generate a new θ and corresponding y
    theta_new = torch.tensor([0.9, 1.3, 1.5, 0.8, 1.1, 1.2],
                              dtype=torch.float32, device=DEV)
    with torch.no_grad():
        f_new = surrogate(theta_new.unsqueeze(0)).squeeze()
        noise = math.sqrt(1e-3) * f_new * torch.randn_like(f_new)
        y_new = f_new + noise

    t0 = time.time()
    theta_amort_new = cond_flow.sample(y_new, n_samples=N_SAMPLES)
    t_amort_new = time.time() - t0
    print(f"  True θ_new:  {theta_new.numpy().round(4)}")
    print(f"  Amort mean:  {theta_amort_new.mean(0).numpy().round(4)}")
    print(f"  Amort std:   {theta_amort_new.std(0).numpy().round(4)}")
    print(f"  Time: {t_amort_new*1000:.1f} ms (ZERO retraining!)")

    # Correct the new observation with NeuTra-HMC
    log_post_new = NNLogPosterior(surrogate, y_new.numpy(), kappa_sq=1e-3).to(DEV)
    t0 = time.time()
    neutra_new = NeuTraHMC_Conditional(
        log_post_new.forward, cond_flow, y_new,
        step_size=0.05, n_leapfrog=20)
    chain_new, ar_new = neutra_new.run(
        theta_new, N_SAMPLES, N_BURNIN)
    t_new = time.time() - t0
    ess_new = effective_sample_size(chain_new)
    print(f"  Corrected mean: {chain_new.mean(0).numpy().round(4)}")
    print(f"  Corrected ESS:  {ess_new.min():.0f}")
    print(f"  Corrected time: {t_new:.1f}s")

    # ── 8. Summary table ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Method':<28} {'Time':>10} {'ESS_min':>8} {'ESS/s':>8}")
    print("─" * 70)

    # Fast mode
    # ESS for amortized = N (iid samples by construction)
    print(f"{'Amortized (fast mode)':<28} "
          f"{t_fast*1000:>8.1f}ms {N_SAMPLES:>8} "
          f"{N_SAMPLES/t_fast:>7.0f}")

    # Exact mode
    print(f"{'NeuTra-HMC (exact mode)':<28} "
          f"{t_exact:>8.1f}s  {ess_exact.min():>7.0f} "
          f"{ess_exact.min()/t_exact:>7.1f}")

    # HMC baseline
    print(f"{'Standard HMC (baseline)':<28} "
          f"{t_hmc:>8.1f}s  {ess_hmc.min():>7.0f} "
          f"{ess_hmc.min()/t_hmc:>7.1f}")
    print("─" * 70)

    # ── 9. Save ───────────────────────────────────────────────────
    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "corrected_amortized.pt"
    torch.save({
        "theta_true": torch.tensor(theta_true),
        "f_obs": torch.tensor(f_obs),
        "fast_mode": {"chain": theta_fast, "time": t_fast},
        "exact_mode": {"chain": chain_exact, "accept_rate": ar_exact,
                       "time": t_exact, "ess": ess_exact},
        "hmc_baseline": {"chain": chain_hmc, "accept_rate": ar_hmc,
                         "time": t_hmc, "ess": ess_hmc},
        "generalization": {
            "theta_true_new": theta_new,
            "y_new": y_new,
            "amortized": {"chain": theta_amort_new, "time": t_amort_new},
            "corrected": {"chain": chain_new, "accept_rate": ar_new,
                          "time": t_new, "ess": ess_new},
        },
    }, save_path)
    print(f"\n  Results saved to {save_path}")

    # Save the trained conditional flow
    flow_path = out_dir / "conditional_flow.pt"
    torch.save(cond_flow.state_dict(), flow_path)
    print(f"  Flow saved to {flow_path}")

    print(f"\n{'='*70}")
    print("POSTERIOR SUMMARY")
    print(f"{'='*70}")
    print(f"  True θ:       {theta_true.round(4)}")
    print(f"  Amortized:    {theta_fast.mean(0).numpy().round(4)}")
    print(f"  NeuTra-HMC:   {chain_exact.mean(0).numpy().round(4)}")
    print(f"  Standard HMC: {chain_hmc.mean(0).numpy().round(4)}")


if __name__ == "__main__":
    main()
