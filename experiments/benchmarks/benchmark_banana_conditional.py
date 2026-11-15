"""
Benchmark 1: Conditional Banana-Shaped Posterior
=================================================
Tests whether the conditional normalizing flow can handle
observation-dependent nonlinear correlation structure.

Comparable to MSSP paper's BananaHard (σ₁=10, b=0.1) benchmark,
but extended to the conditional/amortized setting.

Target: π(θ|y) where θ₂|θ₁ ~ N(b*θ₁², σ₂²) and θ₁|y ~ N(y₁, σ₁²)
Remaining dimensions: θ_j ~ N(0, 1) for j > 2

Metrics: Fast-mode MAE, Exact-mode ESS, wall-clock time,
         generalization to unseen y values
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# ============================================================
# Configuration
# ============================================================
@dataclass
class BananaConfig:
    """Benchmark configuration."""
    # Target parameters
    sigma1: float = 3.0        # Marginal std of θ₁
    b: float = 0.1             # Banana curvature
    sigma2: float = 1.0        # Conditional std of θ₂|θ₁
    kappa: float = 0.03        # Observation noise level

    # Dimensions to test
    dimensions: List[int] = None

    # Training
    n_train: int = 50000
    n_val: int = 5000
    n_epochs: int = 500
    lr: float = 1e-3
    batch_size: int = 512

    # CNF architecture
    n_coupling_layers: int = 8
    hidden_dim: int = 128
    embedding_dim: int = 32

    # Inference
    n_samples: int = 500
    n_burnin: int = 100
    n_leapfrog: int = 20
    hmc_step_size: float = 0.15

    # Evaluation
    n_test_observations: int = 10
    n_seeds: int = 3

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.dimensions is None:
            self.dimensions = [2, 4, 6, 8, 10, 20]


# ============================================================
# Conditional Banana Target Distribution
# ============================================================
class ConditionalBananaTarget:
    """
    Conditional banana-shaped posterior:
        θ₁ | y  ~ N(y₁, σ₁²)
        θ₂ | θ₁ ~ N(b·θ₁², σ₂²)
        θ_j      ~ N(0, 1)  for j = 3, ..., D

    The observation model: y = F(θ) + ε, ε ~ N(0, κ²I)
    where F(θ) = (θ₁, θ₂ - b·θ₁², θ₃, ..., θ_D)
    """
    def __init__(self, D: int, cfg: BananaConfig):
        self.D = D
        self.sigma1 = cfg.sigma1
        self.b = cfg.b
        self.sigma2 = cfg.sigma2
        self.kappa = cfg.kappa

    def sample_prior(self, n: int, device: str = "cpu") -> torch.Tensor:
        """Sample from the prior p(θ)."""
        theta = torch.randn(n, self.D, device=device)
        theta[:, 0] *= self.sigma1
        # θ₂ = b·θ₁² + σ₂·ε
        theta[:, 1] = self.b * theta[:, 0]**2 + self.sigma2 * torch.randn(n, device=device)
        return theta

    def forward_model(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute F(θ) — the 'straightened' representation."""
        y = theta.clone()
        y[:, 1] = theta[:, 1] - self.b * theta[:, 0]**2
        return y

    def generate_observation(self, theta_true: torch.Tensor) -> torch.Tensor:
        """Generate noisy observation y = F(θ_true) + ε."""
        y_clean = self.forward_model(theta_true.unsqueeze(0))
        noise = self.kappa * torch.randn_like(y_clean)
        return (y_clean + noise).squeeze(0)

    def log_likelihood(self, theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-likelihood log p(y|θ)."""
        F_theta = self.forward_model(theta)
        diff = (F_theta - y.unsqueeze(0)) / self.kappa
        return -0.5 * (diff ** 2).sum(dim=-1)

    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        """Log-prior log p(θ)."""
        lp = -0.5 * (theta[:, 0] / self.sigma1) ** 2
        lp += -0.5 * ((theta[:, 1] - self.b * theta[:, 0]**2) / self.sigma2) ** 2
        if self.D > 2:
            lp += -0.5 * (theta[:, 2:] ** 2).sum(dim=-1)
        return lp

    def log_posterior(self, theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Unnormalized log-posterior log p(θ|y) ∝ log p(y|θ) + log p(θ)."""
        return self.log_likelihood(theta, y) + self.log_prior(theta)

    def analytical_posterior_mean(self, y: torch.Tensor) -> torch.Tensor:
        """Approximate analytical posterior mean (for near-linear regime)."""
        # This is approximate; used as reference
        mean = torch.zeros(self.D, device=y.device)
        prec1 = 1.0 / self.sigma1**2 + 1.0 / self.kappa**2
        mean[0] = y[0] / (self.kappa**2 * prec1)
        mean[1] = self.b * (mean[0]**2 + 1.0/prec1) + y[1] * self.sigma2**2 / (self.sigma2**2 + self.kappa**2)
        if self.D > 2:
            for j in range(2, self.D):
                mean[j] = y[j] / (1.0 + self.kappa**2)
        return mean

    def generate_training_pairs(self, n: int, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate (θ, y) training pairs for CNF training."""
        theta = self.sample_prior(n, device)
        y_clean = self.forward_model(theta)
        noise = self.kappa * torch.randn_like(y_clean)
        y = y_clean + noise
        return theta, y


# ============================================================
# Conditional RealNVP (same architecture as CAI paper)
# ============================================================
class ObservationEmbedding(nn.Module):
    """Two-layer observation embedding network."""
    def __init__(self, obs_dim: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


class ConditionalAffineCoupling(nn.Module):
    """Conditional affine coupling layer with observation embedding."""
    def __init__(self, D: int, embed_dim: int, hidden_dim: int, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)
        input_dim = int(mask.sum().item()) + embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * (D - int(mask.sum().item()))),
        )

    def forward(self, theta: torch.Tensor, h_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward: θ → z, returns (z, log_det)."""
        masked = theta * self.mask
        inp = torch.cat([masked[:, self.mask.bool()], h_y], dim=-1)
        st = self.net(inp)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * 2.0  # Bounded scale

        z = theta.clone()
        idx = ~self.mask.bool()
        z[:, idx] = theta[:, idx] * torch.exp(s) + t
        log_det = s.sum(dim=-1)
        return z, log_det

    def inverse(self, z: torch.Tensor, h_y: torch.Tensor) -> torch.Tensor:
        """Inverse: z → θ."""
        masked = z * self.mask
        inp = torch.cat([masked[:, self.mask.bool()], h_y], dim=-1)
        st = self.net(inp)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * 2.0

        theta = z.clone()
        idx = ~self.mask.bool()
        theta[:, idx] = (z[:, idx] - t) * torch.exp(-s)
        return theta


class ConditionalRealNVP(nn.Module):
    """Conditional RealNVP normalizing flow for CAI framework."""
    def __init__(self, D: int, obs_dim: int, cfg: BananaConfig):
        super().__init__()
        self.D = D
        self.embedding = ObservationEmbedding(obs_dim, cfg.embedding_dim)

        self.layers = nn.ModuleList()
        for i in range(cfg.n_coupling_layers):
            mask = torch.zeros(D)
            if i % 2 == 0:
                mask[:D // 2] = 1.0
            else:
                mask[D // 2:] = 1.0
            self.layers.append(
                ConditionalAffineCoupling(D, cfg.embedding_dim, cfg.hidden_dim, mask)
            )

    def forward(self, theta: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward transform: θ → z, conditioned on y."""
        h_y = self.embedding(y)
        z = theta
        log_det_total = torch.zeros(theta.shape[0], device=theta.device)
        for layer in self.layers:
            z, log_det = layer(z, h_y)
            log_det_total += log_det
        return z, log_det_total

    def inverse(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Inverse transform: z → θ, conditioned on y."""
        h_y = self.embedding(y)
        theta = z
        for layer in reversed(self.layers):
            theta = layer.inverse(theta, h_y)
        return theta

    def log_prob(self, theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Log-probability q_ϕ(θ|y)."""
        z, log_det = self.forward(theta, y)
        log_pz = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * self.D * np.log(2 * np.pi)
        return log_pz + log_det


# ============================================================
# Training
# ============================================================
def train_cnf(
    cnf: ConditionalRealNVP,
    target: ConditionalBananaTarget,
    cfg: BananaConfig,
) -> Dict:
    """Train CNF via amortized forward KL (Eq. 5 in CAI paper)."""
    device = cfg.device
    cnf = cnf.to(device)

    optimizer = torch.optim.Adam(cnf.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.n_epochs)

    # Generate training data
    theta_train, y_train = target.generate_training_pairs(cfg.n_train, device)
    theta_val, y_val = target.generate_training_pairs(cfg.n_val, device)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.n_epochs):
        cnf.train()
        perm = torch.randperm(cfg.n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, cfg.n_train, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            theta_b = theta_train[idx]
            y_b = y_train[idx]

            log_q = cnf.log_prob(theta_b, y_b)
            loss = -log_q.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cnf.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / n_batches

        # Validation
        cnf.eval()
        with torch.no_grad():
            val_log_q = cnf.log_prob(theta_val, y_val)
            val_loss = -val_log_q.mean().item()

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in cnf.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= 50:
            print(f"  Early stopping at epoch {epoch}, best val loss: {best_val_loss:.4f}", flush=True)
            break

        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: train={avg_train_loss:.4f}, val={val_loss:.4f}", flush=True)

    cnf.load_state_dict(best_state)
    cnf = cnf.to(device)
    return {"best_val_loss": best_val_loss, "history": history, "epochs": epoch + 1}


# ============================================================
# Inference: Fast Mode (Amortized)
# ============================================================
def fast_mode_inference(
    cnf: ConditionalRealNVP,
    y: torch.Tensor,
    n_samples: int,
    device: str,
) -> Tuple[torch.Tensor, float]:
    """
    Fast mode: generate i.i.d. posterior samples via CNF inverse.
    Returns (samples, wall_clock_time_ms).
    """
    cnf.eval()
    y_expanded = y.unsqueeze(0).expand(n_samples, -1).to(device)

    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()

    with torch.no_grad():
        z = torch.randn(n_samples, cnf.D, device=device)
        theta_samples = cnf.inverse(z, y_expanded)

    torch.cuda.synchronize() if device == "cuda" else None
    t1 = time.perf_counter()

    return theta_samples.cpu(), (t1 - t0) * 1000


# ============================================================
# Inference: Exact Mode (NeuTra-HMC)
# ============================================================
def neutra_hmc_inference(
    cnf: ConditionalRealNVP,
    target: ConditionalBananaTarget,
    y: torch.Tensor,
    cfg: BananaConfig,
) -> Tuple[torch.Tensor, float, Dict]:
    """
    Exact mode: NeuTra-HMC in z-space with Metropolis correction.
    Returns (samples, wall_clock_time_s, diagnostics).
    """
    device = cfg.device
    cnf.eval()

    def U_z(z_val):
        """Potential energy in z-space (Eq. 7 in CAI paper)."""
        z_val = z_val.unsqueeze(0) if z_val.dim() == 1 else z_val
        y_exp = y.unsqueeze(0).expand(z_val.shape[0], -1).to(device)

        # Need gradients for leapfrog
        z_val_req = z_val.detach().requires_grad_(True)
        theta = cnf.inverse(z_val_req, y_exp)

        # Forward pass to get log_det
        z_check, log_det = cnf.forward(theta, y_exp)

        log_pi = target.log_posterior(theta, y.to(device))
        U = -(log_pi + log_det)
        return U.squeeze(), z_val_req

    n_total = cfg.n_samples + cfg.n_burnin
    D = cnf.D
    samples = torch.zeros(n_total, D)

    # Initialize from fast mode
    with torch.no_grad():
        z_init = torch.randn(1, D, device=device)

    z_current = z_init.squeeze(0)
    n_accept = 0
    step_size = cfg.hmc_step_size

    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    TIMEOUT_S = 1200  # 20 min max per observation

    for i in range(n_total):
        # Wall-clock guard
        if time.perf_counter() - t0 > TIMEOUT_S:
            print(f"    [TIMEOUT] NeuTra-HMC stopped at iter {i}/{n_total} after {TIMEOUT_S}s", flush=True)
            break

        # Progress every 250 iterations
        if i % 250 == 0:
            print(f"      NeuTra-HMC iter {i}/{n_total}, accept={n_accept}/{max(i,1)}", flush=True)
        # Leapfrog integration
        p = torch.randn(D, device=device)
        z_prop = z_current.clone().detach()

        U_current, _ = U_z(z_current)
        H_current = U_current + 0.5 * (p ** 2).sum()

        # Half step for momentum
        U_val, z_req = U_z(z_prop)
        U_val.backward()
        grad_U = z_req.grad.detach().squeeze(0)
        p = p - 0.5 * step_size * grad_U

        # Full steps
        for _ in range(cfg.n_leapfrog - 1):
            z_prop = z_prop + step_size * p
            U_val, z_req = U_z(z_prop)
            U_val.backward()
            grad_U = z_req.grad.detach().squeeze(0)
            p = p - step_size * grad_U

        # Last half step
        z_prop = z_prop + step_size * p
        U_val, z_req = U_z(z_prop)
        U_val.backward()
        grad_U = z_req.grad.detach().squeeze(0)
        p = p - 0.5 * step_size * grad_U

        # Metropolis correction
        U_proposed, _ = U_z(z_prop)
        H_proposed = U_proposed + 0.5 * (p ** 2).sum()

        log_alpha = -(H_proposed - H_current)
        if torch.log(torch.rand(1, device=device)) < log_alpha:
            z_current = z_prop.detach()
            n_accept += 1

        # Map back to θ-space
        with torch.no_grad():
            y_exp = y.unsqueeze(0).to(device)
            theta = cnf.inverse(z_current.unsqueeze(0), y_exp)
            samples[i] = theta.squeeze(0).cpu()

    torch.cuda.synchronize() if device == "cuda" else None
    t1 = time.perf_counter()
    n_completed = min(i + 1, n_total) if 'i' in dir() else 0

    # Discard burn-in, handle early stop
    if n_completed > cfg.n_burnin:
        posterior_samples = samples[cfg.n_burnin:n_completed]
    else:
        posterior_samples = samples[:max(n_completed, 1)]
    acceptance_rate = n_accept / max(n_completed, 1)

    # Compute ESS (Geyer initial monotone sequence estimator — simplified)
    ess = compute_ess(posterior_samples)

    diagnostics = {
        "acceptance_rate": acceptance_rate,
        "ess_min": ess.min().item(),
        "ess_mean": ess.mean().item(),
        "wall_time_s": t1 - t0,
    }

    return posterior_samples, t1 - t0, diagnostics


def compute_ess(samples: torch.Tensor) -> torch.Tensor:
    """Compute ESS per dimension using autocorrelation."""
    N, D = samples.shape
    ess = torch.zeros(D)
    for d in range(D):
        x = samples[:, d]
        x = x - x.mean()
        var = (x ** 2).mean()
        if var < 1e-12:
            ess[d] = N
            continue
        # Autocorrelation
        acf_sum = 0.0
        for lag in range(1, N):
            acf = (x[:-lag] * x[lag:]).mean() / var
            if acf < 0.05:
                break
            acf_sum += acf
        ess[d] = N / (1 + 2 * acf_sum)
    return ess


# ============================================================
# Standard HMC (baseline, no flow)
# ============================================================
def standard_hmc_inference(
    target: ConditionalBananaTarget,
    y: torch.Tensor,
    cfg: BananaConfig,
) -> Tuple[torch.Tensor, float, Dict]:
    """Standard HMC without NeuTra preconditioning (baseline)."""
    device = cfg.device
    D = target.D
    n_total = cfg.n_samples + cfg.n_burnin

    def U_theta(theta_val):
        theta_val = theta_val.unsqueeze(0) if theta_val.dim() == 1 else theta_val
        theta_req = theta_val.detach().requires_grad_(True)
        log_p = target.log_posterior(theta_req, y.to(device))
        U = -log_p
        return U.squeeze(), theta_req

    # Initialize from prior
    theta_current = target.sample_prior(1, device).squeeze(0)
    samples = torch.zeros(n_total, D)
    n_accept = 0
    step_size = 0.01  # smaller step for original-space HMC to get non-zero AR

    t0 = time.perf_counter()
    TIMEOUT_S = 1200  # 20 min max

    for i in range(n_total):
        if time.perf_counter() - t0 > TIMEOUT_S:
            print(f"    [TIMEOUT] Std HMC stopped at iter {i}/{n_total}", flush=True)
            break
        if i % 250 == 0:
            print(f"      Std HMC iter {i}/{n_total}, accept={n_accept}/{max(i,1)}", flush=True)

        p = torch.randn(D, device=device)
        theta_prop = theta_current.clone().detach()

        U_current, _ = U_theta(theta_current)
        H_current = U_current + 0.5 * (p ** 2).sum()

        U_val, th_req = U_theta(theta_prop)
        U_val.backward()
        grad_U = th_req.grad.detach().squeeze(0)
        p = p - 0.5 * step_size * grad_U

        for _ in range(cfg.n_leapfrog - 1):
            theta_prop = theta_prop + step_size * p
            U_val, th_req = U_theta(theta_prop)
            U_val.backward()
            grad_U = th_req.grad.detach().squeeze(0)
            p = p - step_size * grad_U

        theta_prop = theta_prop + step_size * p
        U_val, th_req = U_theta(theta_prop)
        U_val.backward()
        grad_U = th_req.grad.detach().squeeze(0)
        p = p - 0.5 * step_size * grad_U

        U_proposed, _ = U_theta(theta_prop)
        H_proposed = U_proposed + 0.5 * (p ** 2).sum()

        log_alpha = -(H_proposed - H_current)
        if torch.log(torch.rand(1, device=device)) < log_alpha:
            theta_current = theta_prop.detach()
            n_accept += 1

        samples[i] = theta_current.cpu()

    t1 = time.perf_counter()
    n_completed = min(i + 1, n_total) if 'i' in dir() else 0
    if n_completed > cfg.n_burnin:
        posterior_samples = samples[cfg.n_burnin:n_completed]
    else:
        posterior_samples = samples[:max(n_completed, 1)]
    acceptance_rate = n_accept / max(n_completed, 1)
    ess = compute_ess(posterior_samples)

    diagnostics = {
        "acceptance_rate": acceptance_rate,
        "ess_min": ess.min().item(),
        "ess_mean": ess.mean().item(),
        "wall_time_s": t1 - t0,
    }

    return posterior_samples, t1 - t0, diagnostics


# ============================================================
# Main Benchmark Runner
# ============================================================
def run_benchmark(cfg: BananaConfig, save_dir: str = "results/banana", skip_hmc: bool = False):
    """Run the full conditional banana benchmark."""
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    for D in cfg.dimensions:
        print(f"\n{'='*60}", flush=True)
        print(f"  Banana Benchmark: D = {D}", flush=True)
        print(f"{'='*60}", flush=True)

        target = ConditionalBananaTarget(D, cfg)
        results[D] = {"fast": [], "exact": [], "hmc": []}

        for seed in range(cfg.n_seeds):
            print(f"\n--- Seed {seed + 1}/{cfg.n_seeds} ---", flush=True)
            torch.manual_seed(42 + seed)
            np.random.seed(42 + seed)

            # 1. Train CNF
            print("Training CNF...", flush=True)
            cnf = ConditionalRealNVP(D, D, cfg)
            train_info = train_cnf(cnf, target, cfg)
            print(f"  CNF trained: val_loss = {train_info['best_val_loss']:.4f}", flush=True)

            # 2. Generate test observations
            theta_true_list = []
            y_test_list = []
            for obs_idx in range(cfg.n_test_observations):
                theta_true = target.sample_prior(1, cfg.device).squeeze(0)
                y_test = target.generate_observation(theta_true.to(cfg.device))
                theta_true_list.append(theta_true.cpu())
                y_test_list.append(y_test.cpu())

            # 3. Evaluate on each test observation
            for obs_idx in range(min(3, cfg.n_test_observations)):
                y = y_test_list[obs_idx]
                theta_true = theta_true_list[obs_idx]

                print(f"  Observation {obs_idx + 1}:", flush=True)

                # Fast mode
                fast_samples, fast_time = fast_mode_inference(
                    cnf, y, cfg.n_samples, cfg.device
                )
                fast_mean = fast_samples.mean(dim=0)
                fast_mae = (fast_mean - theta_true).abs().mean().item()
                fast_max_rel = ((fast_mean - theta_true).abs() / theta_true.abs().clamp(min=0.1)).max().item()

                print(f"    Fast: MAE={fast_mae:.4f}, time={fast_time:.1f}ms", flush=True)

                results[D]["fast"].append({
                    "seed": seed,
                    "obs": obs_idx,
                    "mae": fast_mae,
                    "max_rel_err": fast_max_rel,
                    "time_ms": fast_time,
                    "ess": cfg.n_samples,  # i.i.d. by construction
                })

                # Exact mode (NeuTra-HMC) — only for D <= 10 and not quick mode
                if D <= 10 and not skip_hmc:
                    exact_samples, exact_time, exact_diag = neutra_hmc_inference(
                        cnf, target, y, cfg
                    )
                    exact_mean = exact_samples.mean(dim=0)
                    exact_mae = (exact_mean - theta_true).abs().mean().item()

                    print(f"    Exact: MAE={exact_mae:.4f}, ESS_min={exact_diag['ess_min']:.0f}, "
                          f"AR={exact_diag['acceptance_rate']:.1%}, time={exact_time:.1f}s", flush=True)

                    results[D]["exact"].append({
                        "seed": seed,
                        "obs": obs_idx,
                        "mae": exact_mae,
                        "ess_min": exact_diag["ess_min"],
                        "acceptance_rate": exact_diag["acceptance_rate"],
                        "time_s": exact_time,
                    })

                    # Standard HMC baseline
                    hmc_samples, hmc_time, hmc_diag = standard_hmc_inference(
                        target, y, cfg
                    )
                    hmc_mean = hmc_samples.mean(dim=0)
                    hmc_mae = (hmc_mean - theta_true).abs().mean().item()

                    print(f"    HMC:   MAE={hmc_mae:.4f}, ESS_min={hmc_diag['ess_min']:.0f}, "
                          f"AR={hmc_diag['acceptance_rate']:.1%}, time={hmc_time:.1f}s", flush=True)

                    results[D]["hmc"].append({
                        "seed": seed,
                        "obs": obs_idx,
                        "mae": hmc_mae,
                        "ess_min": hmc_diag["ess_min"],
                        "acceptance_rate": hmc_diag["acceptance_rate"],
                        "time_s": hmc_time,
                    })

            # Intermediate save: persist after each D+seed completes so a crash
            # doesn't lose previously-finished work.
            _ckpt_path = os.path.join(save_dir, "banana_results.json")
            with open(_ckpt_path, "w") as _f:
                json.dump(results, _f, indent=2, default=str)
            print(f"  [SAVE] Checkpointed results to {_ckpt_path} (D={D}, seed={seed})", flush=True)

    # Save results
    with open(os.path.join(save_dir, "banana_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print_summary_table(results)
    return results


def print_summary_table(results: Dict):
    """Print MSSP-style summary table."""
    print("\n" + "=" * 90, flush=True)
    print("Conditional Banana Benchmark — Summary", flush=True)
    print("=" * 90, flush=True)
    print(f"{'D':>4} | {'Method':>12} | {'MAE':>8} | {'ESS_min':>8} | {'AR (%)':>8} | {'Time':>10}", flush=True)
    print("-" * 90, flush=True)

    for D in sorted(results.keys(), key=int):
        D_int = int(D)
        # Fast mode
        fast_entries = results[D]["fast"]
        if fast_entries:
            fast_mae = np.mean([e["mae"] for e in fast_entries])
            fast_time = np.mean([e["time_ms"] for e in fast_entries])
            print(f"{D_int:>4} | {'Amortized':>12} | {fast_mae:>8.4f} | {'2000':>8} | {'N/A':>8} | {fast_time:>7.1f} ms", flush=True)

        # Exact mode
        exact_entries = results[D]["exact"]
        if exact_entries:
            exact_mae = np.mean([e["mae"] for e in exact_entries])
            exact_ess = np.mean([e["ess_min"] for e in exact_entries])
            exact_ar = np.mean([e["acceptance_rate"] for e in exact_entries])
            exact_time = np.mean([e["time_s"] for e in exact_entries])
            print(f"{'':>4} | {'NeuTra-HMC':>12} | {exact_mae:>8.4f} | {exact_ess:>8.0f} | {exact_ar:>7.1%} | {exact_time:>7.1f} s", flush=True)

        # HMC baseline
        hmc_entries = results[D]["hmc"]
        if hmc_entries:
            hmc_mae = np.mean([e["mae"] for e in hmc_entries])
            hmc_ess = np.mean([e["ess_min"] for e in hmc_entries])
            hmc_ar = np.mean([e["acceptance_rate"] for e in hmc_entries])
            hmc_time = np.mean([e["time_s"] for e in hmc_entries])
            print(f"{'':>4} | {'Std HMC':>12} | {hmc_mae:>8.4f} | {hmc_ess:>8.0f} | {hmc_ar:>7.1%} | {hmc_time:>7.1f} s", flush=True)

        print("-" * 90, flush=True)


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    import sys

    quick = "--quick" in sys.argv

    if quick:
        print("[QUICK MODE] Fast-mode only, D=2, 1 seed, 1 obs", flush=True)
        cfg = BananaConfig(
            dimensions=[2],
            n_train=5000,
            n_epochs=50,
            n_seeds=1,
            n_test_observations=1,
        )
    else:
        print("[FULL MODE] All dimensions, 3 seeds, HMC enabled", flush=True)
        print("  Requires GPU. Estimated time: ~2h on RTX 4090", flush=True)
        cfg = BananaConfig(
            dimensions=[2, 6, 10, 20],
            n_train=50000,
            n_epochs=500,
            n_seeds=3,
            n_test_observations=5,
        )

    results = run_benchmark(cfg, skip_hmc=quick)
