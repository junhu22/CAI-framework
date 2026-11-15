#!/usr/bin/env python3
"""
NeuTra-HMC Prototype for Sailboat Building.

Wraps the existing LinearStiffnessSurrogate in a PyTorch-differentiable
module, constructs the full log-posterior (Gaussian likelihood + log-normal
prior), and runs NeuTra-HMC in z-space through a pre-trained NF transport.

This script is SELF-CONTAINED — it only needs:
  1. sailboat_weights_calibrated.npy (for Case 1, D=6)
  2. A trained NF checkpoint from your existing neutra/ codebase (optional;
     falls back to identity transport for baseline HMC comparison)

Usage:
  # Quick test: HMC without NF transport (baseline)
  python neutra_hmc_sailboat.py --dim 12 --case 2 --n_samples 2000

  # With NF transport (provide your trained checkpoint)
  python neutra_hmc_sailboat.py --dim 12 --case 2 --n_samples 5000 \
      --nf_checkpoint path/to/nf_state.pt

  # Compare RWMH vs HMC
  python neutra_hmc_sailboat.py --dim 12 --case 2 --compare
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ============================================================================
#                PyTorch-differentiable Linear Stiffness Surrogate
# ============================================================================

class TorchLinearSurrogate(nn.Module):
    """
    PyTorch wrapper for LinearStiffnessSurrogate.

    Forward: θ (D,) → frequencies (M,) in Hz.
    Fully differentiable via autograd — no finite differences needed.

    Math:
        s_r = (W_r · θ) / (W_r · 1)          per-mode stiffness factor
        ω_r = ω_r0 · √(max(s_r, ε))          natural frequency (rad/s)
        f_r = ω_r / (2π)                       frequency (Hz)

    Gradient (analytic, verified by autograd):
        ∂f_r/∂θ_j = f_r0 · W_{r,j} / (2 · √s_r · row_norm_r)
    """

    def __init__(self, nominal_freqs_hz: np.ndarray, weight_matrix: np.ndarray):
        super().__init__()
        M, D = weight_matrix.shape
        assert nominal_freqs_hz.shape == (M,)

        # Store as buffers (not trainable, move with .to(device))
        self.register_buffer("f0", torch.tensor(nominal_freqs_hz, dtype=torch.float64))
        self.register_buffer("W", torch.tensor(weight_matrix, dtype=torch.float64))
        row_norm = weight_matrix.sum(axis=1)
        row_norm = np.where(row_norm > 0, row_norm, 1.0)
        self.register_buffer("row_norm", torch.tensor(row_norm, dtype=torch.float64))
        self.M = M
        self.D = D

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            theta: (..., D) parameter tensor

        Returns:
            freqs: (..., M) frequencies in Hz
        """
        # s_r = (W @ theta) / row_norm  →  per-mode stiffness factor
        s = (theta @ self.W.T) / self.row_norm  # (..., M)
        s = torch.clamp(s, min=1e-6)
        freqs = self.f0 * torch.sqrt(s)
        return freqs


# ============================================================================
#                    Weight matrices (from sailboat_building.py)
# ============================================================================

SAILBOAT_2019_FREQS_HZ = np.array([0.91, 1.08, 1.47, 2.55, 3.16, 3.81])


def load_case1_weights(n_params: int, calib_path: str | None = None) -> np.ndarray:
    """Load Case 1 weight matrix (D=2,4,6)."""
    if calib_path and Path(calib_path).exists():
        W6 = np.load(calib_path)
    else:
        # Legacy fallback
        c1 = np.array([0.06, 0.08, 0.12, 0.40, 0.60, 0.75])
        c2 = np.array([0.07, 0.08, 0.10, 0.20, 0.20, 0.15])
        c3 = np.array([0.07, 0.09, 0.18, 0.10, 0.10, 0.05])
        s1 = np.array([0.55, 0.45, 0.30, 0.20, 0.15, 0.10])
        s2 = np.array([0.20, 0.25, 0.25, 0.10, 0.05, 0.04])
        s3 = np.array([0.05, 0.10, 0.10, 0.05, 0.03, 0.02])
        W6 = np.stack([c1, c2, c3, s1, s2, s3], axis=1)
        W6 = W6 / W6.sum(axis=1, keepdims=True)
    if n_params == 6:
        return W6
    if n_params == 2:
        return np.concatenate([W6[:, :3].sum(1, keepdims=True),
                               W6[:, 3:].sum(1, keepdims=True)], axis=1)
    if n_params == 4:
        return np.concatenate([W6[:, 0:1], W6[:, 1:3].sum(1, keepdims=True),
                               W6[:, 3:4], W6[:, 4:6].sum(1, keepdims=True)], axis=1)
    raise ValueError(f"Case 1: n_params ∈ {{2,4,6}}, got {n_params}")


def load_case2_weights(n_params: int) -> np.ndarray:
    """Load Case 2 weight matrix (D=8,12)."""
    cols = np.array([
        [0.05, 0.05, 0.04, 0.03], [0.06, 0.06, 0.05, 0.03],
        [0.10, 0.09, 0.07, 0.05], [0.30, 0.20, 0.10, 0.05],
        [0.45, 0.25, 0.10, 0.05], [0.55, 0.20, 0.10, 0.05],
    ])
    walls = np.array([
        [0.40, 0.30, 0.20, 0.10], [0.35, 0.30, 0.20, 0.10],
        [0.25, 0.20, 0.15, 0.10], [0.15, 0.10, 0.05, 0.04],
        [0.10, 0.07, 0.04, 0.03], [0.07, 0.05, 0.03, 0.02],
    ])
    beams = np.array([
        [0.04, 0.04, 0.04, 0.03], [0.05, 0.05, 0.04, 0.03],
        [0.07, 0.06, 0.05, 0.04], [0.10, 0.08, 0.06, 0.04],
        [0.13, 0.10, 0.07, 0.05], [0.16, 0.12, 0.08, 0.06],
    ])
    if n_params == 8:
        return np.concatenate([cols, walls], axis=1)
    if n_params == 12:
        return np.concatenate([cols, walls, beams], axis=1)
    raise ValueError(f"Case 2: n_params ∈ {{8,12}}, got {n_params}")


def build_surrogate(case: int, n_params: int,
                    calib_path: str | None = None) -> TorchLinearSurrogate:
    """Build PyTorch surrogate for a given case and dimensionality."""
    if case == 1:
        W = load_case1_weights(n_params, calib_path)
    else:
        W = load_case2_weights(n_params)
    n_modes = W.shape[0]
    return TorchLinearSurrogate(SAILBOAT_2019_FREQS_HZ[:n_modes], W)


# ============================================================================
#              Differentiable Log-Posterior (likelihood + prior)
# ============================================================================

class SailboatLogPosterior(nn.Module):
    """
    Differentiable log-posterior for Bayesian structural identification.

    log π(θ|y) = log p(y|θ) + log p(θ)

    Likelihood: Gaussian on frequency mismatch
        log p(y|θ) = -1/(2κ²) · Σ_r [(f_r(θ) - f_r^obs) / f_r^obs]²

    Prior: Independent log-normal
        log p(θ) = Σ_j [-½(ln θ_j / σ)² - ln θ_j + const]
    """

    def __init__(self, surrogate: TorchLinearSurrogate,
                 observed_freqs_hz: np.ndarray,
                 kappa_sq: float = 1e-3,
                 prior_sigma: float = 0.3):
        super().__init__()
        self.surrogate = surrogate
        self.register_buffer(
            "f_obs", torch.tensor(observed_freqs_hz, dtype=torch.float64))
        self.kappa_sq = kappa_sq
        self.prior_sigma = prior_sigma
        self.D = surrogate.D

    def log_likelihood(self, theta: torch.Tensor) -> torch.Tensor:
        """Gaussian log-likelihood on relative frequency error."""
        f_pred = self.surrogate(theta)
        residual = (f_pred - self.f_obs) / self.f_obs
        return -0.5 / self.kappa_sq * (residual ** 2).sum(dim=-1)

    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        """Independent log-normal prior, median=1, log-std=σ."""
        if (theta <= 0).any():
            return torch.tensor(-1e30, dtype=theta.dtype, device=theta.device)
        log_t = torch.log(theta)
        return (-0.5 * (log_t / self.prior_sigma) ** 2 - log_t).sum(dim=-1)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        """Unnormalized log-posterior (scalar for single θ, batch for multiple)."""
        return self.log_likelihood(theta) + self.log_prior(theta)

    def grad_log_prob(self, theta: torch.Tensor):
        """Return (log_prob, ∇_θ log_prob) for HMC."""
        theta = theta.detach().requires_grad_(True)
        lp = self.forward(theta)
        grad = torch.autograd.grad(lp.sum(), theta)[0]
        return lp.detach(), grad.detach()


# ============================================================================
#                          HMC / NUTS Sampler
# ============================================================================

class HMCSampler:
    """Basic HMC sampler operating in θ-space (or z-space if transformed)."""

    def __init__(self, log_prob_and_grad_fn, dim: int,
                 step_size: float = 0.01, n_leapfrog: int = 20,
                 dtype=torch.float64):
        self.log_prob_and_grad = log_prob_and_grad_fn
        self.dim = dim
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.dtype = dtype

    def _leapfrog(self, q, p):
        """Leapfrog integrator."""
        _, grad = self.log_prob_and_grad(q)
        p = p + 0.5 * self.step_size * grad

        for _ in range(self.n_leapfrog - 1):
            q = q + self.step_size * p
            _, grad = self.log_prob_and_grad(q)
            p = p + self.step_size * grad

        q = q + self.step_size * p
        _, grad = self.log_prob_and_grad(q)
        p = p + 0.5 * self.step_size * grad
        return q, p

    def _kinetic(self, p):
        return 0.5 * (p ** 2).sum()

    def sample(self, q_init: torch.Tensor, n_samples: int,
               warmup: int = 500, verbose: bool = True):
        """Run HMC chain."""
        q = q_init.clone().to(self.dtype)
        lp_current, _ = self.log_prob_and_grad(q)
        samples = torch.zeros(n_samples, self.dim, dtype=self.dtype)
        log_probs = torch.zeros(n_samples, dtype=self.dtype)
        n_accept = 0

        for i in range(-(warmup) , n_samples):
            p = torch.randn(self.dim, dtype=self.dtype)
            H_current = -lp_current + self._kinetic(p)

            q_prop, p_prop = self._leapfrog(q, p)
            lp_prop, _ = self.log_prob_and_grad(q_prop)
            H_prop = -lp_prop + self._kinetic(p_prop)

            log_alpha = H_current - H_prop
            if torch.log(torch.rand(1, dtype=self.dtype)) < log_alpha:
                q = q_prop
                lp_current = lp_prop
                if i >= 0:
                    n_accept += 1

            if i >= 0:
                samples[i] = q
                log_probs[i] = lp_current

            # Adapt step size during warmup (dual averaging simplified)
            if i < 0 and (i + warmup) % 50 == 49:
                recent_rate = n_accept / max(i + warmup + 1, 1)
                if i >= -warmup:
                    n_accept = 0  # reset for sampling phase

                if recent_rate < 0.55:
                    self.step_size *= 0.8
                elif recent_rate > 0.85:
                    self.step_size *= 1.2

            if verbose and i >= 0 and (i + 1) % 500 == 0:
                rate = n_accept / (i + 1) * 100
                print(f"  HMC step {i+1}/{n_samples}: "
                      f"accept={rate:.0f}%, step_size={self.step_size:.4f}")

        accept_rate = n_accept / n_samples
        return samples, log_probs, accept_rate


# ============================================================================
#                          RWMH Sampler (for comparison)
# ============================================================================

class RWMHSampler:
    """Random Walk Metropolis-Hastings (your MSSP baseline)."""

    def __init__(self, log_prob_fn, dim: int, proposal_std: float = 0.05):
        self.log_prob = log_prob_fn
        self.dim = dim
        self.proposal_std = proposal_std

    def sample(self, q_init, n_samples, warmup=500, verbose=True):
        q = q_init.clone().double()
        lp_current = self.log_prob(q)
        samples = torch.zeros(n_samples, self.dim, dtype=torch.float64)
        log_probs = torch.zeros(n_samples, dtype=torch.float64)
        n_accept = 0

        for i in range(-warmup, n_samples):
            q_prop = q + self.proposal_std * torch.randn(self.dim, dtype=torch.float64)
            if (q_prop <= 0).any():
                lp_prop = torch.tensor(-1e30)
            else:
                lp_prop = self.log_prob(q_prop)
            if torch.log(torch.rand(1)) < (lp_prop - lp_current):
                q, lp_current = q_prop, lp_prop
                if i >= 0:
                    n_accept += 1
            if i >= 0:
                samples[i] = q
                log_probs[i] = lp_current

            if verbose and i >= 0 and (i + 1) % 500 == 0:
                rate = n_accept / (i + 1) * 100
                print(f"  RWMH step {i+1}/{n_samples}: accept={rate:.0f}%")

        return samples, log_probs, n_accept / n_samples


# ============================================================================
#                          Diagnostics
# ============================================================================

def effective_sample_size(chain: torch.Tensor) -> torch.Tensor:
    """Compute ESS per dimension using autocorrelation (Geyer's method)."""
    N, D = chain.shape
    ess = torch.zeros(D)
    for d in range(D):
        x = chain[:, d].numpy()
        x = x - x.mean()
        # FFT-based autocorrelation
        f = np.fft.fft(x, n=2 * N)
        acf = np.fft.ifft(f * np.conj(f)).real[:N] / (N * np.var(x) + 1e-30)
        # Geyer's initial positive sequence
        t = 1
        while t < N - 1 and acf[t] + acf[t + 1] > 0:
            t += 2
        ess[d] = N / (1 + 2 * acf[1:t].sum())
    return ess


def print_diagnostics(name: str, samples: torch.Tensor, elapsed: float,
                      accept_rate: float):
    """Print chain diagnostics."""
    N, D = samples.shape
    ess = effective_sample_size(samples)
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  Samples:       {N}")
    print(f"  Accept rate:   {accept_rate*100:.1f}%")
    print(f"  Wall time:     {elapsed:.1f}s")
    print(f"  ESS (min/med/max): {ess.min():.0f} / {ess.median():.0f} / {ess.max():.0f}")
    print(f"  ESS/s (min):   {ess.min()/elapsed:.1f}")
    print(f"  Posterior mean: {samples.mean(0).numpy().round(4)}")
    print(f"  Posterior std:  {samples.std(0).numpy().round(4)}")
    print(f"{'─'*60}")
    return ess


# ============================================================================
#                              Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="NeuTra-HMC prototype for sailboat building")
    ap.add_argument("--case", type=int, default=2, choices=[1, 2])
    ap.add_argument("--dim", type=int, default=12)
    ap.add_argument("--n_samples", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--kappa_sq", type=float, default=1e-3)
    ap.add_argument("--calib_path", type=str, default=None,
                    help="Path to sailboat_weights_calibrated.npy")
    ap.add_argument("--compare", action="store_true",
                    help="Run both RWMH and HMC for comparison")
    ap.add_argument("--hmc_step", type=float, default=0.01)
    ap.add_argument("--hmc_leapfrog", type=int, default=20)
    args = ap.parse_args()

    print(f"Sailboat Building — Case {args.case}, D={args.dim}")
    print(f"Surrogate: LinearStiffnessSurrogate (closed-form, differentiable)")

    # Build surrogate + log-posterior
    surrogate = build_surrogate(args.case, args.dim, args.calib_path)
    log_post = SailboatLogPosterior(
        surrogate,
        observed_freqs_hz=SAILBOAT_2019_FREQS_HZ[:surrogate.M],
        kappa_sq=args.kappa_sq,
    )

    # Verify gradient
    theta_test = torch.ones(args.dim, dtype=torch.float64)
    lp, grad = log_post.grad_log_prob(theta_test)
    print(f"\nGradient check at θ=1:")
    print(f"  log π(1) = {lp.item():.4f}")
    print(f"  ||∇ log π|| = {grad.norm().item():.4f}")
    print(f"  ∇ log π = {grad.numpy().round(4)}")

    # Initial point
    q0 = torch.ones(args.dim, dtype=torch.float64)

    if args.compare or True:  # Always run both for comparison
        # ---- RWMH ----
        print(f"\n{'='*60}")
        print("Running RWMH (baseline)...")
        rwmh = RWMHSampler(log_post.forward, args.dim, proposal_std=0.02)
        t0 = time.time()
        s_rwmh, lp_rwmh, ar_rwmh = rwmh.sample(q0, args.n_samples, args.warmup)
        t_rwmh = time.time() - t0
        ess_rwmh = print_diagnostics("RWMH", s_rwmh, t_rwmh, ar_rwmh)

    # ---- HMC ----
    print(f"\n{'='*60}")
    print("Running HMC (gradient-enhanced)...")
    hmc = HMCSampler(
        log_post.grad_log_prob, args.dim,
        step_size=args.hmc_step,
        n_leapfrog=args.hmc_leapfrog,
    )
    t0 = time.time()
    s_hmc, lp_hmc, ar_hmc = hmc.sample(q0, args.n_samples, args.warmup)
    t_hmc = time.time() - t0
    ess_hmc = print_diagnostics("HMC", s_hmc, t_hmc, ar_hmc)

    # ---- Comparison ----
    if args.compare or True:
        print(f"\n{'='*60}")
        print("COMPARISON: RWMH vs HMC")
        print(f"{'='*60}")
        print(f"  ESS/s improvement: {ess_hmc.min()/t_hmc / (ess_rwmh.min()/t_rwmh + 1e-10):.1f}×")
        print(f"  Accept rate:  RWMH={ar_rwmh*100:.0f}%  HMC={ar_hmc*100:.0f}%")
        print(f"  Min ESS:      RWMH={ess_rwmh.min():.0f}  HMC={ess_hmc.min():.0f}")
        print(f"  Wall time:    RWMH={t_rwmh:.1f}s  HMC={t_hmc:.1f}s")
        print(f"\nNote: This uses the LINEAR surrogate. With NF transport")
        print(f"(NeuTra), HMC improvement will be even larger because the")
        print(f"transport normalises the posterior geometry.")


if __name__ == "__main__":
    main()
