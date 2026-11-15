"""
Benchmark 3: Dimension Scaling on Linear-Gaussian Inverse Problem
=================================================================
Uses a linear-Gaussian inverse problem with KNOWN analytical posterior
to test dimension scaling from D=2 to D=50.

This is the cleanest scalability test because:
1. The analytical posterior is available for exact error computation
2. No surrogate error confounds the flow approximation error
3. Directly comparable to MSSP's AR(1) Gaussian scaling (Table 1)

Forward model: y = A·θ + ε, ε ~ N(0, Σ_ε)
Prior:         θ ~ N(μ₀, Σ₀)
Posterior:     θ|y ~ N(μ_post, Σ_post) [exact Gaussian]

This tests the CAI framework's ability to scale, separate from
the structural model complexity.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import os
from typing import Dict, Tuple
from dataclasses import dataclass, field
from benchmark_banana_conditional import (
    ConditionalRealNVP,
    compute_ess,
)


@dataclass
class ScalingConfig:
    correlation: float = 0.9      # AR(1) correlation in prior
    obs_noise_std: float = 0.1    # Observation noise std
    dimensions: list = field(default_factory=lambda: [2, 4, 6, 8, 12, 20, 50])
    n_train: int = 100000
    n_val: int = 5000
    n_epochs: int = 500
    lr: float = 1e-3
    batch_size: int = 512
    n_coupling_layers: int = 8
    hidden_dim: int = 128
    embedding_dim: int = 64
    n_samples: int = 2000
    n_burnin: int = 500
    n_leapfrog: int = 20
    hmc_step_size: float = 0.15
    n_test_observations: int = 10
    n_seeds: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LinearGaussianTarget:
    """
    Linear-Gaussian inverse problem with analytical posterior.

    Prior: θ ~ N(0, Σ₀) where Σ₀ has AR(1) structure
    Forward: y = A·θ + ε, ε ~ N(0, σ²I)
    Posterior: θ|y ~ N(μ_post, Σ_post)
    """
    def __init__(self, D: int, cfg: ScalingConfig):
        self.D = D
        self.M = D  # Same number of observations as parameters

        # AR(1) prior covariance
        rho = cfg.correlation
        Sigma0 = torch.zeros(D, D)
        for i in range(D):
            for j in range(D):
                Sigma0[i, j] = rho ** abs(i - j)
        self.Sigma0 = Sigma0
        self.Sigma0_inv = torch.linalg.inv(Sigma0)

        # Forward model: A = random orthogonal for well-conditioning
        torch.manual_seed(0)
        Q, _ = torch.linalg.qr(torch.randn(D, D))
        self.A = Q  # Orthogonal forward model

        # Observation noise
        self.sigma_eps = cfg.obs_noise_std
        self.Sigma_eps = (cfg.obs_noise_std ** 2) * torch.eye(self.M)
        self.Sigma_eps_inv = torch.eye(self.M) / (cfg.obs_noise_std ** 2)

        # Precompute analytical posterior covariance
        # Σ_post = (Σ₀⁻¹ + Aᵀ Σ_ε⁻¹ A)⁻¹
        self.Sigma_post = torch.linalg.inv(
            self.Sigma0_inv + self.A.T @ self.Sigma_eps_inv @ self.A
        )

    def analytical_posterior(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute analytical posterior mean and covariance given y."""
        # μ_post = Σ_post · Aᵀ Σ_ε⁻¹ y
        mu_post = self.Sigma_post @ self.A.T @ self.Sigma_eps_inv @ y
        return mu_post, self.Sigma_post

    def sample_prior(self, n: int, device: str = "cpu") -> torch.Tensor:
        L = torch.linalg.cholesky(self.Sigma0.to(device))
        z = torch.randn(n, self.D, device=device)
        return z @ L.T

    def forward_model(self, theta: torch.Tensor) -> torch.Tensor:
        return theta @ self.A.to(theta.device).T

    def generate_observation(self, theta_true: torch.Tensor) -> torch.Tensor:
        y = self.forward_model(theta_true.unsqueeze(0)).squeeze(0)
        return y + self.sigma_eps * torch.randn_like(y)

    def log_likelihood(self, theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pred = self.forward_model(theta)
        diff = pred - y.unsqueeze(0)
        return -0.5 * (diff @ self.Sigma_eps_inv.to(theta.device) @ diff.T).diag()

    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        return -0.5 * (theta @ self.Sigma0_inv.to(theta.device) @ theta.T).diag()

    def log_posterior(self, theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.log_likelihood(theta, y) + self.log_prior(theta)

    def generate_training_pairs(self, n: int, device: str = "cpu"):
        theta = self.sample_prior(n, device)
        y = self.forward_model(theta) + self.sigma_eps * torch.randn(n, self.M, device=device)
        return theta, y


def run_scaling_benchmark(cfg: ScalingConfig, save_dir: str = "results/scaling"):
    """Run dimension scaling benchmark with analytical ground truth."""
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    for D in cfg.dimensions:
        print(f"\n{'='*60}", flush=True)
        print(f"  Scaling Benchmark: D = {D}", flush=True)
        print(f"{'='*60}", flush=True)

        target = LinearGaussianTarget(D, cfg)
        results[D] = {"fast": [], "analytical_comparison": []}

        # Adjust architecture for higher D
        actual_layers = max(cfg.n_coupling_layers, D // 2 + 4)
        actual_hidden = max(cfg.hidden_dim, D * 8)
        actual_embed = max(cfg.embedding_dim, D * 4)

        for seed in range(cfg.n_seeds):
            print(f"\n--- Seed {seed + 1}/{cfg.n_seeds} ---", flush=True)
            torch.manual_seed(42 + seed)

            # Create and train CNF
            cnf_cfg_dict = {
                "n_coupling_layers": actual_layers,
                "hidden_dim": actual_hidden,
                "embedding_dim": actual_embed,
            }
            print(f"  Architecture: {actual_layers} layers, hidden={actual_hidden}, embed={actual_embed}", flush=True)

            # Monkey-patch config for this D
            cfg_copy = ScalingConfig(
                dimensions=[D],
                n_coupling_layers=actual_layers,
                hidden_dim=actual_hidden,
                embedding_dim=actual_embed,
                n_train=cfg.n_train,
                n_val=cfg.n_val,
                n_epochs=cfg.n_epochs,
                lr=cfg.lr,
                batch_size=cfg.batch_size,
                n_samples=cfg.n_samples,
                device=cfg.device,
            )

            cnf = ConditionalRealNVP(D, D, cfg_copy)
            cnf = cnf.to(cfg.device)

            # Training
            optimizer = torch.optim.Adam(cnf.parameters(), lr=cfg.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.n_epochs)
            theta_train, y_train = target.generate_training_pairs(cfg.n_train, cfg.device)
            theta_val, y_val = target.generate_training_pairs(cfg.n_val, cfg.device)

            best_val = float("inf")
            patience = 0
            best_state = None

            for epoch in range(cfg.n_epochs):
                cnf.train()
                perm = torch.randperm(cfg.n_train, device=cfg.device)
                for i in range(0, cfg.n_train, cfg.batch_size):
                    idx = perm[i:i+cfg.batch_size]
                    loss = -cnf.log_prob(theta_train[idx], y_train[idx]).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(cnf.parameters(), 5.0)
                    optimizer.step()
                scheduler.step()

                cnf.eval()
                with torch.no_grad():
                    vl = -cnf.log_prob(theta_val, y_val).mean().item()
                if vl < best_val:
                    best_val = vl
                    patience = 0
                    best_state = {k: v.cpu().clone() for k, v in cnf.state_dict().items()}
                else:
                    patience += 1
                    if patience >= 50:
                        break
                if epoch % 100 == 0:
                    print(f"  Epoch {epoch}: val_loss={vl:.4f}", flush=True)

            cnf.load_state_dict(best_state)
            cnf = cnf.to(cfg.device)
            print(f"  Training done, best val loss: {best_val:.4f}", flush=True)

            # Test on multiple observations
            for obs_idx in range(cfg.n_test_observations):
                theta_true = target.sample_prior(1, cfg.device).squeeze(0)
                y = target.generate_observation(theta_true.to(cfg.device))

                # Analytical posterior
                mu_exact, Sigma_exact = target.analytical_posterior(y.cpu())

                # Fast mode
                y_cpu = y.cpu()
                cnf.eval()
                y_exp = y_cpu.unsqueeze(0).expand(cfg.n_samples, -1).to(cfg.device)
                torch.cuda.synchronize() if cfg.device == "cuda" else None
                t0 = time.perf_counter()
                with torch.no_grad():
                    z = torch.randn(cfg.n_samples, D, device=cfg.device)
                    fast_samples = cnf.inverse(z, y_exp).cpu()
                torch.cuda.synchronize() if cfg.device == "cuda" else None
                fast_time = (time.perf_counter() - t0) * 1000

                fast_mean = fast_samples.mean(0)
                mae_vs_analytical = (fast_mean - mu_exact).abs().mean().item()
                mae_vs_true = (fast_mean - theta_true.cpu()).abs().mean().item()

                # Covariance comparison
                fast_cov = torch.cov(fast_samples.T)
                cov_frobenius = torch.norm(fast_cov - Sigma_exact, p="fro").item()
                cov_frobenius_rel = cov_frobenius / torch.norm(Sigma_exact, p="fro").item()

                results[D]["fast"].append({
                    "seed": seed, "obs": obs_idx,
                    "mae_vs_analytical": mae_vs_analytical,
                    "mae_vs_true": mae_vs_true,
                    "cov_frobenius_rel": cov_frobenius_rel,
                    "time_ms": fast_time,
                })

            # Summary for this D + seed
            fast_entries = results[D]["fast"]
            recent = [e for e in fast_entries if e["seed"] == seed]
            avg_mae = np.mean([e["mae_vs_analytical"] for e in recent])
            avg_cov = np.mean([e["cov_frobenius_rel"] for e in recent])
            avg_time = np.mean([e["time_ms"] for e in recent])
            print(f"  D={D}, seed={seed}: MAE(vs analytical)={avg_mae:.5f}, "
                  f"Cov err={avg_cov:.4f}, Time={avg_time:.1f}ms", flush=True)

    # Print final table
    print("\n" + "=" * 80, flush=True)
    print("Dimension Scaling — Summary (Fast Mode vs Analytical Posterior)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'D':>4} | {'MAE (mean)':>12} | {'MAE (std)':>10} | "
          f"{'Cov Err':>10} | {'Time (ms)':>10}", flush=True)
    print("-" * 80, flush=True)

    scaling_data = []
    for D in sorted(results.keys(), key=int):
        entries = results[D]["fast"]
        if entries:
            maes = [e["mae_vs_analytical"] for e in entries]
            covs = [e["cov_frobenius_rel"] for e in entries]
            times = [e["time_ms"] for e in entries]
            row = {
                "D": int(D),
                "mae_mean": np.mean(maes),
                "mae_std": np.std(maes),
                "cov_err": np.mean(covs),
                "time_ms": np.mean(times),
            }
            scaling_data.append(row)
            print(f"{int(D):>4} | {row['mae_mean']:>12.5f} | {row['mae_std']:>10.5f} | "
                  f"{row['cov_err']:>10.4f} | {row['time_ms']:>10.1f}")

    with open(os.path.join(save_dir, "scaling_results.json"), "w") as f:
        json.dump({"per_obs": results, "summary": scaling_data}, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv
    if quick:
        print("[QUICK] Scaling: D=2 only, 1 seed, 5000 train", flush=True)
        cfg = ScalingConfig(
            dimensions=[2],
            n_train=5000,
            n_epochs=50,
            n_seeds=1,
            n_test_observations=2,
        )
    else:
        cfg = ScalingConfig(
            dimensions=[2, 4, 6, 8, 12, 20],
            n_train=100000,
            n_epochs=500,
            n_seeds=3,
        )
    run_scaling_benchmark(cfg)
