"""HMC and NeuTra-HMC samplers — drop into neutra/samplers.py.

New classes (append to existing samplers.py):
  - HMCSampler          : gradient-based HMC, parallel to MHSampler
  - NeuTraMHSampler     : NF transport + RWMH in z-space  (your MSSP paper)
  - NeuTraHMCSampler    : NF transport + HMC in z-space   (NEW — the paper's core)

All follow the same interface:
  sampler = XxxSampler(log_prob_fn, ...)
  result  = sampler.run(x0, n_samples, n_burnin=1000)
  # result.chain : (N, D),  result.acceptance_rate : float

Usage:
  from neutra.samplers import NeuTraHMCSampler

  sampler = NeuTraHMCSampler(
      log_prob_fn=target.log_prob,
      flow=trained_flow,
      step_size=0.1,
      n_leapfrog=20,
  )
  result = sampler.run(x0, n_samples=5000, n_burnin=500)
"""

import math
import time
import torch
from nflows.flows import Flow
from typing import Callable
from dataclasses import dataclass

# Assumes SamplerResult is already defined in samplers.py
# from .samplers import SamplerResult


# ============================================================================
#                     Standalone HMC (no NF transport)
# ============================================================================

class HMCSampler:
    """Hamiltonian Monte Carlo — requires differentiable log_prob_fn.

    The log_prob_fn must accept a (1, D) tensor and return a scalar that
    supports torch.autograd.grad().  For the linear surrogate this works
    out of the box via TorchLinearSurrogate; for the NN surrogate it
    works via PyTorch autograd.
    """

    def __init__(
        self,
        log_prob_fn: Callable,
        step_size: float = 0.05,
        n_leapfrog: int = 20,
        adapt_step: bool = True,
        target_accept: float = 0.70,
    ):
        self.log_prob_fn = log_prob_fn
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.adapt_step = adapt_step
        self.target_accept = target_accept

    def _grad_log_prob(self, x: torch.Tensor):
        """Compute (log_prob, grad_log_prob) for a single point."""
        x = x.detach().requires_grad_(True)
        lp = self.log_prob_fn(x.unsqueeze(0)).squeeze()
        grad = torch.autograd.grad(lp, x)[0]
        return lp.detach(), grad.detach()

    def _leapfrog(self, q, p, step_size):
        """Leapfrog integrator: (q, p) -> (q', p')."""
        _, grad = self._grad_log_prob(q)
        p = p + 0.5 * step_size * grad

        for _ in range(self.n_leapfrog - 1):
            q = q + step_size * p
            _, grad = self._grad_log_prob(q)
            p = p + step_size * grad

        q = q + step_size * p
        _, grad = self._grad_log_prob(q)
        p = p + 0.5 * step_size * grad

        return q, p

    def _adapt(self, x, n_steps, device):
        """Burn-in with step-size adaptation via Robbins-Monro."""
        step = self.step_size
        n_acc = 0

        for i in range(1, n_steps + 1):
            p = torch.randn_like(x)
            lp_x, _ = self._grad_log_prob(x)
            H_current = -lp_x + 0.5 * (p ** 2).sum()

            q_prop, p_prop = self._leapfrog(x, p, step)
            lp_prop, _ = self._grad_log_prob(q_prop)
            H_prop = -lp_prop + 0.5 * (p_prop ** 2).sum()

            log_alpha = H_current - H_prop
            if torch.log(torch.rand(1, device=device)).item() < log_alpha.item():
                x = q_prop
                n_acc += 1

            if i % 50 == 0:
                rate = n_acc / 50
                step *= math.exp(0.5 * (rate - self.target_accept))
                step = max(1e-5, min(step, 1.0))
                n_acc = 0

        self.step_size = step
        return x

    def run(self, x0, n_samples, n_burnin=1000, device=None):
        if device is None:
            device = x0.device
        x = x0.to(device).clone()
        chain, accepted = [], []

        if self.adapt_step and n_burnin > 0:
            x = self._adapt(x, n_burnin, device)

        t0 = time.perf_counter()
        for _ in range(n_samples):
            p = torch.randn_like(x)
            lp_x, _ = self._grad_log_prob(x)
            H_current = -lp_x + 0.5 * (p ** 2).sum()

            q_prop, p_prop = self._leapfrog(x, p, self.step_size)
            lp_prop, _ = self._grad_log_prob(q_prop)
            H_prop = -lp_prop + 0.5 * (p_prop ** 2).sum()

            log_alpha = H_current - H_prop
            acc = torch.log(torch.rand(1, device=device)).item() < log_alpha.item()
            if acc:
                x = q_prop
            chain.append(x.clone())
            accepted.append(acc)

        return SamplerResult(
            chain=torch.stack(chain),
            accepted=accepted,
            elapsed_sec=time.perf_counter() - t0,
            label="HMC",
        )


# ============================================================================
#                NeuTra-MH: NF transport + RWMH in z-space
#              (this is what your MSSP paper does — baseline)
# ============================================================================

class NeuTraMHSampler:
    """NF-transported Random Walk MH (MSSP paper, Theorem 3.2).

    Coordinate transport T: θ → z.  Run RWMH on the transported density:
        π_z(z) = π(T⁻¹(z)) · |det J_{T⁻¹}(z)|

    The flow is FIXED after warmup — Proposition 3.4 guarantees correctness.
    """

    def __init__(
        self,
        log_prob_fn: Callable,
        flow: Flow,
        step_size: float = 0.5,
    ):
        self.log_prob_fn = log_prob_fn
        self.flow = flow
        self.step_size = step_size

    def _theta_to_z(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """θ → z, returns (z, log_det_forward)."""
        with torch.no_grad():
            z, log_det = self.flow._transform(theta.unsqueeze(0))
        return z.squeeze(0), log_det.squeeze()

    def _z_to_theta(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z → θ, returns (θ, log_det_inverse)."""
        with torch.no_grad():
            theta, log_det = self.flow._transform.inverse(z.unsqueeze(0))
        return theta.squeeze(0), log_det.squeeze()

    def _log_prob_z(self, z: torch.Tensor) -> torch.Tensor:
        """Transported log-density: log π_z(z) = log π(T⁻¹(z)) + log|det J_{T⁻¹}|."""
        theta, log_det_inv = self._z_to_theta(z)
        log_pi = self.log_prob_fn(theta.unsqueeze(0)).squeeze()
        return log_pi + log_det_inv

    def run(self, x0, n_samples, n_burnin=1000, device=None):
        if device is None:
            device = x0.device
        self.flow = self.flow.to(device).eval()

        # Map initial point to z-space
        z, _ = self._theta_to_z(x0.to(device))
        chain, accepted = [], []
        step = self.step_size

        # Burn-in with adaptation
        lp_z = self._log_prob_z(z)
        n_acc = 0
        for i in range(1, n_burnin + 1):
            z_prop = z + step * torch.randn_like(z)
            lp_prop = self._log_prob_z(z_prop)
            if torch.log(torch.rand(1, device=device)).item() < (lp_prop - lp_z).item():
                z, lp_z = z_prop, lp_prop
                n_acc += 1
            if i % 50 == 0:
                rate = n_acc / 50
                step *= math.exp(rate - 0.234)
                step = max(1e-4, min(step, 5.0))
                n_acc = 0

        # Sampling
        t0 = time.perf_counter()
        for _ in range(n_samples):
            z_prop = z + step * torch.randn_like(z)
            lp_prop = self._log_prob_z(z_prop)
            acc = torch.log(torch.rand(1, device=device)).item() < (lp_prop - lp_z).item()
            if acc:
                z, lp_z = z_prop, lp_prop
            # Map back to θ-space for storage
            theta, _ = self._z_to_theta(z)
            chain.append(theta.clone())
            accepted.append(acc)

        return SamplerResult(
            chain=torch.stack(chain),
            accepted=accepted,
            elapsed_sec=time.perf_counter() - t0,
            label="NeuTra-MH",
        )


# ============================================================================
#         NeuTra-HMC: NF transport + HMC in z-space  ★ THE NEW METHOD ★
# ============================================================================

class NeuTraHMCSampler:
    """NF-transported HMC — the core contribution for the next paper.

    Coordinate transport T: θ → z.  Run HMC on the transported density:
        π_z(z) = π(T⁻¹(z)) · |det J_{T⁻¹}(z)|

    The gradient ∇_z log π_z(z) is computed via autograd through the
    inverse flow T⁻¹ and the (differentiable) log-posterior.

    Requirements:
        - log_prob_fn must be differentiable (NN surrogate or TorchLinearSurrogate)
        - flow must be an nflows.Flow with differentiable inverse

    Theory:
        - Detailed balance for transported HMC: follows from symplecticity
          of leapfrog + volume preservation of T (Theorem X.1 in new paper)
        - NF reuse validity: Proposition 3.4 from MSSP paper carries over
          (the flow is fixed; HMC kernel is π_z-invariant)
    """

    def __init__(
        self,
        log_prob_fn: Callable,
        flow: Flow,
        step_size: float = 0.1,
        n_leapfrog: int = 20,
        adapt_step: bool = True,
        target_accept: float = 0.70,
    ):
        self.log_prob_fn = log_prob_fn
        self.flow = flow
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.adapt_step = adapt_step
        self.target_accept = target_accept

    def _grad_log_prob_z(self, z: torch.Tensor):
        """Compute log π_z(z) and ∇_z log π_z(z) via autograd.

        The gradient flows through:
            z → T⁻¹(z) = θ → log π(θ) + log|det J_{T⁻¹}(z)|
        Both the inverse flow and log_prob_fn must be differentiable.
        """
        z = z.detach().requires_grad_(True)

        # Inverse flow: z → θ (with grad tracking)
        theta, log_det_inv = self.flow._transform.inverse(z.unsqueeze(0))
        theta = theta.squeeze(0)
        log_det_inv = log_det_inv.squeeze()

        # Log-posterior in θ-space
        log_pi = self.log_prob_fn(theta.unsqueeze(0)).squeeze()

        # Transported log-density
        log_prob_z = log_pi + log_det_inv

        # Gradient via autograd
        grad = torch.autograd.grad(log_prob_z, z)[0]

        return log_prob_z.detach(), grad.detach()

    def _leapfrog(self, q, p, step_size):
        """Leapfrog integrator in z-space."""
        _, grad = self._grad_log_prob_z(q)
        p = p + 0.5 * step_size * grad

        for _ in range(self.n_leapfrog - 1):
            q = q + step_size * p
            _, grad = self._grad_log_prob_z(q)
            p = p + step_size * grad

        q = q + step_size * p
        _, grad = self._grad_log_prob_z(q)
        p = p + 0.5 * step_size * grad

        return q, p

    def _theta_to_z(self, theta: torch.Tensor) -> torch.Tensor:
        """θ → z via forward flow (no grad needed)."""
        with torch.no_grad():
            z, _ = self.flow._transform(theta.unsqueeze(0))
        return z.squeeze(0)

    def _z_to_theta(self, z: torch.Tensor) -> torch.Tensor:
        """z → θ via inverse flow (no grad needed, for storing chain)."""
        with torch.no_grad():
            theta, _ = self.flow._transform.inverse(z.unsqueeze(0))
        return theta.squeeze(0)

    def _adapt(self, z, n_steps, device):
        """Burn-in with step-size adaptation."""
        step = self.step_size
        n_acc = 0

        for i in range(1, n_steps + 1):
            p = torch.randn_like(z)
            lp_z, _ = self._grad_log_prob_z(z)
            H_current = -lp_z + 0.5 * (p ** 2).sum()

            q_prop, p_prop = self._leapfrog(z, p, step)
            lp_prop, _ = self._grad_log_prob_z(q_prop)
            H_prop = -lp_prop + 0.5 * (p_prop ** 2).sum()

            if torch.log(torch.rand(1, device=device)).item() < (H_current - H_prop).item():
                z = q_prop
                n_acc += 1

            if i % 50 == 0:
                rate = n_acc / 50
                step *= math.exp(0.5 * (rate - self.target_accept))
                step = max(1e-5, min(step, 1.0))
                n_acc = 0

        self.step_size = step
        return z

    def run(self, x0, n_samples, n_burnin=1000, device=None):
        """Run NeuTra-HMC chain.

        Args:
            x0: Initial point in θ-space (D,)
            n_samples: Number of post-burnin samples
            n_burnin: Adaptation steps (step-size tuning)

        Returns:
            SamplerResult with chain in θ-space (N, D)
        """
        if device is None:
            device = x0.device
        self.flow = self.flow.to(device).eval()

        # Map initial θ to z-space
        z = self._theta_to_z(x0.to(device))

        # Adapt step size during burn-in
        if self.adapt_step and n_burnin > 0:
            z = self._adapt(z, n_burnin, device)

        # Sampling
        chain, accepted = [], []
        t0 = time.perf_counter()

        for _ in range(n_samples):
            p = torch.randn_like(z)
            lp_z, _ = self._grad_log_prob_z(z)
            H_current = -lp_z + 0.5 * (p ** 2).sum()

            q_prop, p_prop = self._leapfrog(z, p, self.step_size)
            lp_prop, _ = self._grad_log_prob_z(q_prop)
            H_prop = -lp_prop + 0.5 * (p_prop ** 2).sum()

            log_alpha = H_current - H_prop
            acc = torch.log(torch.rand(1, device=device)).item() < log_alpha.item()
            if acc:
                z = q_prop

            # Map z → θ for storage
            theta = self._z_to_theta(z)
            chain.append(theta.clone())
            accepted.append(acc)

        return SamplerResult(
            chain=torch.stack(chain),
            accepted=accepted,
            elapsed_sec=time.perf_counter() - t0,
            label="NeuTra-HMC",
        )


# ---------------------------------------------------------------------------
# Convenience: comparison runner
# ---------------------------------------------------------------------------

def compare_samplers(
    log_prob_fn: Callable,
    flow: Flow,
    x0: torch.Tensor,
    n_samples: int = 5000,
    n_burnin: int = 1000,
    device: torch.device | None = None,
    hmc_step: float = 0.05,
    hmc_leapfrog: int = 20,
    mh_step: float = 0.1,
) -> dict[str, SamplerResult]:
    """Run all four samplers and return results dict.

    Returns dict with keys: 'MH', 'NeuTra-MH', 'HMC', 'NeuTra-HMC'
    """
    results = {}

    # 1. Standard RWMH
    print("Running MH...", flush=True)
    results["MH"] = MHSampler(log_prob_fn, step_size=mh_step).run(
        x0, n_samples, n_burnin, device)

    # 2. NeuTra-MH (MSSP baseline)
    print("Running NeuTra-MH...", flush=True)
    results["NeuTra-MH"] = NeuTraMHSampler(log_prob_fn, flow, step_size=0.5).run(
        x0, n_samples, n_burnin, device)

    # 3. Standalone HMC
    print("Running HMC...", flush=True)
    results["HMC"] = HMCSampler(log_prob_fn, step_size=hmc_step, n_leapfrog=hmc_leapfrog).run(
        x0, n_samples, n_burnin, device)

    # 4. NeuTra-HMC (NEW)
    print("Running NeuTra-HMC...", flush=True)
    results["NeuTra-HMC"] = NeuTraHMCSampler(
        log_prob_fn, flow, step_size=hmc_step, n_leapfrog=hmc_leapfrog,
    ).run(x0, n_samples, n_burnin, device)

    # Print summary
    print(f"\n{'Sampler':<15} {'Accept%':>8} {'ESS/s':>8} {'Time':>8}")
    print("─" * 45)
    for name, r in results.items():
        from .diagnostics import effective_sample_size
        ess = effective_sample_size(r.chain).min().item()
        print(f"{name:<15} {r.acceptance_rate*100:>7.1f}% {ess/r.elapsed_sec:>7.1f} {r.elapsed_sec:>7.1f}s")

    return results
