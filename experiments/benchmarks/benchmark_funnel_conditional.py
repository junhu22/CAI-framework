"""
Benchmark 2: Conditional Heteroscedastic Funnel
================================================
Tests whether the conditional normalizing flow handles
observation-dependent variance structure (Neal's funnel
extended to the conditional/amortized setting).

Target: π(θ|y) where
    v     ~ N(0, σ_v²)
    θ_j|v ~ N(0, exp(v))  for j = 1, ..., D-1
    Observation: y = (v, θ₁, ..., θ_{D-1}) + noise

This is the hardest test for KDE-based methods (MSSP showed
KDE acceptance collapses to 0.2% at T=1 on D=10 Funnel).
"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Reuse components from benchmark_banana_conditional
from benchmark_banana_conditional import (
    ConditionalRealNVP,
    compute_ess,
    fast_mode_inference,
)


@dataclass
class FunnelConfig:
    sigma_v: float = 3.0
    kappa: float = 0.05
    dimensions: List[int] = None
    n_train: int = 50000
    n_val: int = 5000
    n_epochs: int = 500
    lr: float = 1e-3
    batch_size: int = 512
    n_coupling_layers: int = 8
    hidden_dim: int = 128
    embedding_dim: int = 32
    n_samples: int = 2000
    n_burnin: int = 500
    n_leapfrog: int = 20
    hmc_step_size: float = 0.1
    n_test_observations: int = 5
    n_seeds: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.dimensions is None:
            self.dimensions = [2, 6, 10]


class ConditionalFunnelTarget:
    """
    Conditional funnel posterior:
        v     ~ N(0, σ_v²)
        θ_j|v ~ N(0, exp(v))  for j = 1, ..., D-1
        y = (v, θ₁, ..., θ_{D-1}) + ε, ε ~ N(0, κ²I)

    The key challenge: the conditional variance exp(v) varies over
    many orders of magnitude, defeating any fixed-covariance proposal.
    """
    def __init__(self, D: int, cfg: FunnelConfig):
        self.D = D
        self.sigma_v = cfg.sigma_v
        self.kappa = cfg.kappa

    def sample_prior(self, n: int, device: str = "cpu") -> torch.Tensor:
        theta = torch.zeros(n, self.D, device=device)
        v = self.sigma_v * torch.randn(n, device=device)
        theta[:, 0] = v
        std = torch.exp(0.5 * v).unsqueeze(1)
        theta[:, 1:] = std * torch.randn(n, self.D - 1, device=device)
        return theta

    def forward_model(self, theta: torch.Tensor) -> torch.Tensor:
        return theta.clone()

    def generate_observation(self, theta_true: torch.Tensor) -> torch.Tensor:
        y_clean = self.forward_model(theta_true.unsqueeze(0))
        noise = self.kappa * torch.randn_like(y_clean)
        return (y_clean + noise).squeeze(0)

    def log_prior(self, theta: torch.Tensor) -> torch.Tensor:
        v = theta[:, 0]
        lp = -0.5 * (v / self.sigma_v) ** 2
        if self.D > 1:
            x = theta[:, 1:]
            lp += -0.5 * (self.D - 1) * v  # -0.5 * log(exp(v))^{D-1}
            lp += -0.5 * (x ** 2).sum(dim=-1) * torch.exp(-v)
        return lp

    def log_likelihood(self, theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = (theta - y.unsqueeze(0)) / self.kappa
        return -0.5 * (diff ** 2).sum(dim=-1)

    def log_posterior(self, theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.log_likelihood(theta, y) + self.log_prior(theta)

    def generate_training_pairs(self, n: int, device: str = "cpu"):
        theta = self.sample_prior(n, device)
        y = theta + self.kappa * torch.randn_like(theta)
        return theta, y


def train_cnf_funnel(cnf, target, cfg):
    """Train CNF on funnel target."""
    device = cfg.device
    cnf = cnf.to(device)
    optimizer = torch.optim.Adam(cnf.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.n_epochs)

    theta_train, y_train = target.generate_training_pairs(cfg.n_train, device)
    theta_val, y_val = target.generate_training_pairs(cfg.n_val, device)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg.n_epochs):
        cnf.train()
        perm = torch.randperm(cfg.n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, cfg.n_train, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            log_q = cnf.log_prob(theta_train[idx], y_train[idx])
            loss = -log_q.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(cnf.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        cnf.eval()
        with torch.no_grad():
            val_loss = -cnf.log_prob(theta_val, y_val).mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in cnf.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 50:
                break

        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: val_loss={val_loss:.4f}", flush=True)

    cnf.load_state_dict(best_state)
    cnf = cnf.to(device)
    return {"best_val_loss": best_val_loss}


def run_funnel_benchmark(cfg: FunnelConfig, save_dir: str = "results/funnel"):
    """Run the conditional funnel benchmark."""
    os.makedirs(save_dir, exist_ok=True)
    results = {}

    for D in cfg.dimensions:
        print(f"\n{'='*60}", flush=True)
        print(f"  Funnel Benchmark: D = {D}", flush=True)
        print(f"{'='*60}", flush=True)

        target = ConditionalFunnelTarget(D, cfg)
        results[D] = {"fast": [], "exact": []}

        for seed in range(cfg.n_seeds):
            print(f"\n--- Seed {seed + 1}/{cfg.n_seeds} ---", flush=True)
            torch.manual_seed(42 + seed)

            cnf = ConditionalRealNVP(D, D, cfg)
            train_info = train_cnf_funnel(cnf, target, cfg)
            print(f"  CNF trained: val_loss = {train_info['best_val_loss']:.4f}", flush=True)

            for obs_idx in range(min(3, cfg.n_test_observations)):
                theta_true = target.sample_prior(1, cfg.device).squeeze(0)
                y = target.generate_observation(theta_true.to(cfg.device))

                # Fast mode
                fast_samples, fast_time = fast_mode_inference(cnf, y.cpu(), cfg.n_samples, cfg.device)
                fast_mae = (fast_samples.mean(0) - theta_true.cpu()).abs().mean().item()

                print(f"  Obs {obs_idx}: Fast MAE={fast_mae:.4f}, time={fast_time:.1f}ms", flush=True)

                results[D]["fast"].append({
                    "seed": seed, "obs": obs_idx,
                    "mae": fast_mae, "time_ms": fast_time,
                    "v_true": theta_true[0].item(),
                })

    # Print summary
    print("\n" + "=" * 70, flush=True)
    print("Conditional Funnel Benchmark — Summary", flush=True)
    print("=" * 70, flush=True)
    for D in sorted(results.keys(), key=int):
        entries = results[D]["fast"]
        if entries:
            mae = np.mean([e["mae"] for e in entries])
            t = np.mean([e["time_ms"] for e in entries])
            print(f"D={int(D):>3}: Fast MAE = {mae:.4f}, Time = {t:.1f} ms", flush=True)

    with open(os.path.join(save_dir, "funnel_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv
    if quick:
        print("[QUICK] Funnel: D=2, 1 seed", flush=True)
        cfg = FunnelConfig(dimensions=[2], n_seeds=1, n_train=5000, n_epochs=50)
    else:
        cfg = FunnelConfig(dimensions=[2, 6, 10], n_seeds=3)
    run_funnel_benchmark(cfg)
