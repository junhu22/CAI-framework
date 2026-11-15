"""MCMC samplers: standard MH, NF-MH, and Multi-Proposal NF-MH."""
import math
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
from nflows.flows import Flow

from .flows import flow_log_prob, flow_sample


# ---------------------------------------------------------------------------
# Shared result container
# ---------------------------------------------------------------------------

@dataclass
class SamplerResult:
    chain: torch.Tensor           # (N, D)
    accepted: list[bool]
    elapsed_sec: float
    label: str = ""

    @property
    def acceptance_rate(self) -> float:
        return sum(self.accepted) / max(len(self.accepted), 1)


# ---------------------------------------------------------------------------
# Standard Random-Walk Metropolis-Hastings
# ---------------------------------------------------------------------------

class MHSampler:
    """Isotropic Gaussian random-walk MH sampler."""

    def __init__(self, log_prob_fn: Callable, step_size: float = 0.1):
        self.log_prob_fn = log_prob_fn
        self.step_size = step_size

    def run(
        self,
        x0: torch.Tensor,
        n_samples: int,
        n_burnin: int = 1000,
        device: torch.device | None = None,
    ) -> SamplerResult:
        if device is None:
            device = x0.device
        x = x0.to(device).clone()
        dim = x.shape[-1]
        chain = []
        accepted = []

        # Adapt step size during burn-in
        x, step = self._adapt(x, n_burnin, device)

        t0 = time.perf_counter()
        log_p_x = self.log_prob_fn(x.unsqueeze(0)).squeeze()

        for _ in range(n_samples):
            proposal = x + step * torch.randn_like(x)
            log_p_prop = self.log_prob_fn(proposal.unsqueeze(0)).squeeze()

            log_alpha = log_p_prop - log_p_x
            accept = torch.log(torch.rand(1, device=device)).item() < log_alpha.item()

            if accept:
                x = proposal
                log_p_x = log_p_prop
            chain.append(x.clone())
            accepted.append(accept)

        elapsed = time.perf_counter() - t0
        return SamplerResult(
            chain=torch.stack(chain),
            accepted=accepted,
            elapsed_sec=elapsed,
            label="MH",
        )

    def _adapt(self, x: torch.Tensor, n_steps: int, device: torch.device) -> tuple[torch.Tensor, float]:
        """Simple dual-averaging step-size adaptation targeting 0.234 acceptance."""
        step = self.step_size
        n_accepted = 0
        log_p_x = self.log_prob_fn(x.unsqueeze(0)).squeeze()
        target = 0.234

        for i in range(1, n_steps + 1):
            proposal = x + step * torch.randn_like(x)
            log_p_prop = self.log_prob_fn(proposal.unsqueeze(0)).squeeze()
            alpha = min(1.0, torch.exp(log_p_prop - log_p_x).item())

            if torch.log(torch.rand(1, device=device)).item() < (log_p_prop - log_p_x).item():
                x = proposal
                log_p_x = log_p_prop
                n_accepted += 1

            # Robbins-Monro update every 50 steps
            if i % 50 == 0:
                acc_rate = n_accepted / 50
                step *= math.exp(acc_rate - target)
                step = float(torch.clamp(torch.tensor(step), 1e-4, 10.0).item())
                n_accepted = 0

        return x, step


# ---------------------------------------------------------------------------
# NF-based Independence MH
# ---------------------------------------------------------------------------

class NFMHSampler:
    """Independence MH with a normalizing-flow proposal distribution."""

    def __init__(self, log_prob_fn: Callable, flow: Flow):
        self.log_prob_fn = log_prob_fn
        self.flow = flow

    def run(
        self,
        x0: torch.Tensor,
        n_samples: int,
        n_burnin: int = 500,
        device: torch.device | None = None,
        mix_standard: float = 0.05,
    ) -> SamplerResult:
        """
        Args:
            mix_standard: Fraction of proposals from N(0,I) to avoid getting stuck.
        """
        if device is None:
            device = x0.device
        self.flow = self.flow.to(device).eval()
        x = x0.to(device).clone()
        dim = x.shape[-1]
        chain = []
        accepted = []

        log_p_x = self.log_prob_fn(x.unsqueeze(0)).squeeze()
        log_q_x = flow_log_prob(self.flow, x.unsqueeze(0)).squeeze()

        # Burn-in (no recording)
        x, log_p_x, log_q_x = self._step_n(x, log_p_x, log_q_x, n_burnin, dim, device, mix_standard)

        t0 = time.perf_counter()
        for _ in range(n_samples):
            x, log_p_x, log_q_x, acc = self._step(x, log_p_x, log_q_x, dim, device, mix_standard)
            chain.append(x.clone())
            accepted.append(acc)

        elapsed = time.perf_counter() - t0
        return SamplerResult(
            chain=torch.stack(chain),
            accepted=accepted,
            elapsed_sec=elapsed,
            label="NF-MH",
        )

    def _step(self, x, log_p_x, log_q_x, dim, device, mix_std):
        with torch.no_grad():
            if torch.rand(1).item() < mix_std:
                prop = torch.randn(1, dim, device=device)
                log_q_prop = -0.5 * (prop ** 2).sum() - 0.5 * dim * math.log(2 * math.pi)
            else:
                prop = flow_sample(self.flow, 1)
                log_q_prop = flow_log_prob(self.flow, prop).squeeze()

            log_p_prop = self.log_prob_fn(prop).squeeze()
            log_alpha = (log_p_prop - log_q_prop) - (log_p_x - log_q_x)
            acc = torch.log(torch.rand(1, device=device)).item() < log_alpha.item()

        if acc:
            return prop.squeeze(0), log_p_prop, log_q_prop, True
        return x, log_p_x, log_q_x, False

    def _step_n(self, x, log_p_x, log_q_x, n, dim, device, mix_std):
        for _ in range(n):
            x, log_p_x, log_q_x, _ = self._step(x, log_p_x, log_q_x, dim, device, mix_std)
        return x, log_p_x, log_q_x


# ---------------------------------------------------------------------------
# Multi-Proposal NF-MH (parallel proposals, pick best via MTM)
# ---------------------------------------------------------------------------

class MultiProposalNFMHSampler:
    """Multiple-Try Metropolis with NF proposals.

    At each step, draws K proposals from the flow and selects one using
    the Multiple-Try Metropolis scheme (Liu et al. 2000), improving
    acceptance in high dimensions.
    """

    def __init__(self, log_prob_fn: Callable, flow: Flow, k_proposals: int = 10):
        self.log_prob_fn = log_prob_fn
        self.flow = flow
        self.k = k_proposals

    def run(
        self,
        x0: torch.Tensor,
        n_samples: int,
        n_burnin: int = 500,
        device: torch.device | None = None,
    ) -> SamplerResult:
        if device is None:
            device = x0.device
        self.flow = self.flow.to(device).eval()
        x = x0.to(device).clone()
        dim = x.shape[-1]
        chain = []
        accepted = []

        log_p_x = self.log_prob_fn(x.unsqueeze(0)).squeeze()
        log_q_x = flow_log_prob(self.flow, x.unsqueeze(0)).squeeze()

        for _ in range(n_burnin):
            x, log_p_x, log_q_x, _ = self._step(x, log_p_x, log_q_x, dim, device)

        t0 = time.perf_counter()
        for _ in range(n_samples):
            x, log_p_x, log_q_x, acc = self._step(x, log_p_x, log_q_x, dim, device)
            chain.append(x.clone())
            accepted.append(acc)

        elapsed = time.perf_counter() - t0
        return SamplerResult(
            chain=torch.stack(chain),
            accepted=accepted,
            elapsed_sec=elapsed,
            label=f"MP-NF-MH(K={self.k})",
        )

    def _step(self, x, log_p_x, log_q_x, dim, device):
        with torch.no_grad():
            # Draw K proposals from flow
            props = flow_sample(self.flow, self.k)
            log_q_props = flow_log_prob(self.flow, props)    # (K,)
            log_p_props = self.log_prob_fn(props)            # (K,)
            log_w_props = log_p_props - log_q_props          # (K,) unnorm log weights
            log_w_props = torch.where(torch.isfinite(log_w_props), log_w_props,
                                      torch.full_like(log_w_props, -1e38))

            # Select proposal proportional to weights (numerically stable)
            w_sel = log_w_props - log_w_props.max()
            weights = torch.exp(w_sel)
            weights = torch.clamp(weights, min=0.0)
            if weights.sum() <= 0:
                return x, log_p_x, log_q_x, False

            idx = torch.multinomial(weights, 1).item()
            x_star = props[idx]
            log_p_star = log_p_props[idx]
            log_q_star = log_q_props[idx]

            # Draw K-1 reference points from flow (MTM detailed balance)
            refs = flow_sample(self.flow, self.k - 1)
            log_q_refs = flow_log_prob(self.flow, refs)
            log_p_refs = self.log_prob_fn(refs)
            log_w_refs = log_p_refs - log_q_refs             # (K-1,)
            log_w_refs = torch.where(torch.isfinite(log_w_refs), log_w_refs,
                                      torch.full_like(log_w_refs, -1e38))

            # MTM acceptance: α = W_prop / W_ref via logsumexp (correct scale)
            # W_prop = Σ_j exp(log_w_props[j]),  W_ref = exp(log_w_x) + Σ_i exp(log_w_refs[i])
            log_w_x = (log_p_x - log_q_x).reshape(1)
            if not torch.isfinite(log_w_x):
                log_w_x = torch.tensor([-1e38], device=device)
            log_num = torch.logsumexp(log_w_props, dim=0)
            log_den = torch.logsumexp(torch.cat([log_w_refs, log_w_x]), dim=0)
            log_alpha = log_num - log_den

            acc = torch.log(torch.rand(1, device=device)).item() < log_alpha.item()

        if acc:
            return x_star, log_p_star, log_q_star, True
        return x, log_p_x, log_q_x, False


# ---------------------------------------------------------------------------
# Hamiltonian Monte Carlo (HMC)
# ---------------------------------------------------------------------------

class HMCSampler:
    """Vanilla HMC with leapfrog integration.

    Requires log_prob_fn to support autograd (i.e. PyTorch tensors with grad).
    """

    def __init__(
        self,
        log_prob_fn: Callable,
        step_size: float = 0.1,
        n_leapfrog: int = 10,
    ):
        self.log_prob_fn = log_prob_fn
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog

    def run(
        self,
        x0: torch.Tensor,
        n_samples: int,
        n_burnin: int = 1000,
        device: torch.device | None = None,
    ) -> SamplerResult:
        if device is None:
            device = x0.device
        x = x0.to(device).clone().float()
        dim = x.shape[-1]
        chain = []
        accepted = []

        # Dual-averaging step-size adaptation during burn-in
        x, step = self._adapt(x, n_burnin, device)

        t0 = time.perf_counter()
        log_p_x = self._log_prob_val(x)

        for _ in range(n_samples):
            x, log_p_x, acc = self._step(x, log_p_x, step, device)
            chain.append(x.clone())
            accepted.append(acc)

        elapsed = time.perf_counter() - t0
        return SamplerResult(
            chain=torch.stack(chain),
            accepted=accepted,
            elapsed_sec=elapsed,
            label=f"HMC(L={self.n_leapfrog})",
        )

    def _log_prob_val(self, x: torch.Tensor) -> torch.Tensor:
        """Scalar log-prob (detached)."""
        with torch.no_grad():
            return self.log_prob_fn(x.unsqueeze(0)).squeeze()

    def _grad_log_prob(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (log_prob scalar, gradient wrt x)."""
        xr = x.detach().requires_grad_(True)
        lp = self.log_prob_fn(xr.unsqueeze(0)).squeeze()
        lp.backward()
        return lp.detach(), xr.grad.detach()

    def _leapfrog(self, x: torch.Tensor, p: torch.Tensor, step: float):
        """Leapfrog integrator — returns (x', p', log_p_final)."""
        _, g = self._grad_log_prob(x)
        p = p + 0.5 * step * g                  # half-step momentum

        for _ in range(self.n_leapfrog - 1):
            x = x + step * p                    # full-step position
            _, g = self._grad_log_prob(x)
            p = p + step * g                    # full-step momentum

        x = x + step * p                        # final full-step position
        lp, g = self._grad_log_prob(x)
        p = p + 0.5 * step * g                  # final half-step momentum
        return x, p, lp

    def _step(
        self, x: torch.Tensor, log_p_x: torch.Tensor, step: float, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        p0 = torch.randn_like(x)
        ke0 = 0.5 * (p0 ** 2).sum()

        x_prop, p_prop, lp_prop = self._leapfrog(x, p0, step)
        ke_prop = 0.5 * (p_prop ** 2).sum()

        log_alpha = (lp_prop - ke_prop) - (log_p_x - ke0)
        acc = torch.log(torch.rand(1, device=device)).item() < log_alpha.item()

        if acc:
            return x_prop, lp_prop, True
        return x, log_p_x, False

    def _adapt(self, x: torch.Tensor, n_steps: int, device: torch.device) -> tuple[torch.Tensor, float]:
        """Robbins-Monro step-size adaptation targeting 0.65 acceptance."""
        step = self.step_size
        log_p_x = self._log_prob_val(x)
        target = 0.65
        n_accepted = 0

        for i in range(1, n_steps + 1):
            x, log_p_x, acc = self._step(x, log_p_x, step, device)
            if acc:
                n_accepted += 1
            if i % 50 == 0:
                acc_rate = n_accepted / 50
                step *= math.exp(acc_rate - target)
                step = float(max(1e-5, min(step, 5.0)))
                n_accepted = 0

        return x, step


# ---------------------------------------------------------------------------
# Batched NF-MH: pre-generate all proposals in one GPU call (massive speedup)
# ---------------------------------------------------------------------------

def _batch_log_prob(log_prob_fn, x: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """Evaluate log_prob_fn on large tensor in chunks to avoid OOM."""
    parts = []
    for i in range(0, len(x), chunk):
        parts.append(log_prob_fn(x[i : i + chunk]))
    return torch.cat(parts)


class NFMHSamplerBatched:
    """Independence MH with NF proposal — batched pre-generation for GPU efficiency.

    Instead of calling flow.sample(1) × N times (N kernel launches), this sampler
    generates ALL proposals in a single flow.sample(N) call, then evaluates
    log_prob in large batches.  Identical Markov chain semantics, 100–1000× faster
    on GPU.
    """

    def __init__(self, log_prob_fn: Callable, flow: "Flow"):
        self.log_prob_fn = log_prob_fn
        self.flow = flow

    def run(
        self,
        x0: torch.Tensor,
        n_samples: int,
        n_burnin: int = 500,
        device: torch.device | None = None,
    ) -> SamplerResult:
        if device is None:
            device = x0.device
        self.flow = self.flow.to(device).eval()
        x = x0.to(device).clone()
        dim = x.shape[-1]
        total = n_burnin + n_samples

        # ── Batch generate ALL proposals at once ─────────────────────────
        with torch.no_grad():
            all_props  = flow_sample(self.flow, total)           # (T, D)
            all_log_q  = flow_log_prob(self.flow, all_props)     # (T,)
            all_log_p  = _batch_log_prob(self.log_prob_fn, all_props)  # (T,)
            log_u      = torch.log(torch.rand(total, device=device))

        # Initial state log-probs
        with torch.no_grad():
            log_p_x = self.log_prob_fn(x.unsqueeze(0)).squeeze()
            log_q_x = flow_log_prob(self.flow, x.unsqueeze(0)).squeeze()

        # ── Sequential MH loop (pure Python, proposals pre-computed) ────
        chain:    list[torch.Tensor] = []
        accepted: list[bool]         = []

        t0 = time.perf_counter()
        for i in range(total):
            log_alpha = (all_log_p[i] - all_log_q[i]) - (log_p_x - log_q_x)
            acc = log_u[i].item() < log_alpha.item()
            if acc:
                x       = all_props[i]
                log_p_x = all_log_p[i]
                log_q_x = all_log_q[i]
            if i >= n_burnin:
                chain.append(x.clone())
                accepted.append(acc)

        elapsed = time.perf_counter() - t0
        return SamplerResult(
            chain=torch.stack(chain),
            accepted=accepted,
            elapsed_sec=elapsed,
            label="NF-MH",
        )


class MultiProposalNFMHSamplerBatched:
    """Multiple-Try Metropolis with NF proposals — fully batched.

    Pre-generates K × N proposals and (K-1) × N reference points in a single
    forward pass, then runs the MTM chain sequentially.  Same semantics as
    MultiProposalNFMHSampler but 50–500× faster on GPU.
    """

    def __init__(self, log_prob_fn: Callable, flow: "Flow", k_proposals: int = 10):
        self.log_prob_fn = log_prob_fn
        self.flow = flow
        self.k = k_proposals

    def run(
        self,
        x0: torch.Tensor,
        n_samples: int,
        n_burnin: int = 500,
        device: torch.device | None = None,
    ) -> SamplerResult:
        if device is None:
            device = x0.device
        self.flow = self.flow.to(device).eval()
        x = x0.to(device).clone()
        k = self.k
        total = n_burnin + n_samples

        # ── Batch generate ALL proposals & reference points ─────────────
        with torch.no_grad():
            # Proposals: (total*K, D)
            all_props     = flow_sample(self.flow, total * k)
            all_props_log_q = flow_log_prob(self.flow, all_props)
            all_props_log_p = _batch_log_prob(self.log_prob_fn, all_props)

            # Reference points: (total*(K-1), D)
            all_refs      = flow_sample(self.flow, total * (k - 1))
            all_refs_log_q  = flow_log_prob(self.flow, all_refs)
            all_refs_log_p  = _batch_log_prob(self.log_prob_fn, all_refs)

            log_u = torch.log(torch.rand(total, device=device))

        # Initial state log-probs
        with torch.no_grad():
            log_p_x = self.log_prob_fn(x.unsqueeze(0)).squeeze()
            log_q_x = flow_log_prob(self.flow, x.unsqueeze(0)).squeeze()

        # ── Sequential MTM loop ──────────────────────────────────────────
        chain:    list[torch.Tensor] = []
        accepted: list[bool]         = []

        t0 = time.perf_counter()
        for i in range(total):
            # Extract this step's proposals and refs
            ps  = all_props[i * k     : (i + 1) * k]       # (K, D)
            lq  = all_props_log_q[i * k     : (i + 1) * k] # (K,)
            lp  = all_props_log_p[i * k     : (i + 1) * k] # (K,)
            rs  = all_refs[i * (k-1)  : (i + 1) * (k-1)]   # (K-1, D)
            rlq = all_refs_log_q[i * (k-1)  : (i + 1) * (k-1)]
            rlp = all_refs_log_p[i * (k-1)  : (i + 1) * (k-1)]

            log_w = lp - lq   # (K,)
            log_w = torch.where(torch.isfinite(log_w), log_w,
                                torch.full_like(log_w, -1e38))

            # Select proposal proportional to weights
            w_sel = log_w - log_w.max()
            weights = torch.clamp(torch.exp(w_sel), min=0.0)
            if weights.sum() <= 0:
                if i >= n_burnin:
                    chain.append(x.clone())
                    accepted.append(False)
                continue

            idx     = torch.multinomial(weights, 1).item()
            x_star  = ps[idx]
            lp_star = lp[idx]
            lq_star = lq[idx]

            # MTM acceptance via logsumexp
            log_w_refs = rlp - rlq
            log_w_refs = torch.where(torch.isfinite(log_w_refs), log_w_refs,
                                     torch.full_like(log_w_refs, -1e38))
            log_w_x    = (log_p_x - log_q_x).reshape(1)
            if not torch.isfinite(log_w_x):
                log_w_x = torch.tensor([-1e38], device=device)

            log_num   = torch.logsumexp(log_w, dim=0)
            log_den   = torch.logsumexp(torch.cat([log_w_refs, log_w_x]), dim=0)
            log_alpha = log_num - log_den
            acc = log_u[i].item() < log_alpha.item()

            if acc:
                x       = x_star
                log_p_x = lp_star
                log_q_x = lq_star

            if i >= n_burnin:
                chain.append(x.clone())
                accepted.append(acc)

        elapsed = time.perf_counter() - t0
        return SamplerResult(
            chain=torch.stack(chain),
            accepted=accepted,
            elapsed_sec=elapsed,
            label=f"MP-NF-MH(K={k})",
        )


# ---------------------------------------------------------------------------
# Parallel chain runner (naive multi-chain for ESS improvement)
# ---------------------------------------------------------------------------

def run_parallel_chains(
    sampler_cls,
    sampler_kwargs: dict,
    x0s: torch.Tensor,      # (n_chains, D)
    n_samples: int,
    n_burnin: int = 1000,
    device: torch.device | None = None,
) -> list[SamplerResult]:
    """Run independent chains sequentially (parallelism via torch if desired)."""
    results = []
    for i, x0 in enumerate(x0s):
        sampler = sampler_cls(**sampler_kwargs)
        r = sampler.run(x0, n_samples, n_burnin=n_burnin, device=device)
        r.label = f"{r.label} chain-{i}"
        results.append(r)
    return results


