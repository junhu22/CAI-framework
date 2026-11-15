#!/usr/bin/env python3
"""
D=12 High-Dimensional Validation for CAI Framework
====================================================
Validates the CAI framework on a 12-parameter variant of the Sailboat Building
using the LinearStiffnessSurrogate (analytic eigenvalue model).

Run from: C:\nf-pmcmc\
Usage: python experiments/d12_validation.py

Output: experiments/results/d12_validation.pt
"""

import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'neutra' / 'targets'))

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LinearStiffnessSurrogate for D=12
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from sailboat_building import LinearStiffnessSurrogate
    print("✓ Loaded LinearStiffnessSurrogate from sailboat_building.py")
except ImportError:
    print("⚠ Cannot import LinearStiffnessSurrogate, using built-in fallback")
    
    class LinearStiffnessSurrogate(nn.Module):
        """
        Analytic linear stiffness surrogate for D=12.
        Maps 12 stiffness scaling factors → 20 natural frequencies
        via a pre-computed sensitivity matrix: f(θ) ≈ f₀ + J·(θ - 1)
        
        This provides a differentiable, realistic-complexity target for
        testing the CAI framework at higher dimensions.
        """
        def __init__(self, D=12, M=20, seed=42):
            super().__init__()
            self.D = D
            self.M = M
            torch.manual_seed(seed)
            
            # Base frequencies (Hz) - typical for 19-storey RC building
            f0 = torch.tensor([
                0.805, 0.830, 1.210, 2.397, 2.462, 3.647,
                4.175, 4.393, 5.260, 5.524, 6.823, 7.063,
                7.459, 8.227, 9.018, 9.371, 10.244, 11.053,
                11.687, 12.341
            ], dtype=torch.float64)
            self.register_buffer('f0', f0)
            
            # Sensitivity matrix J ∈ R^{M×D}: ∂f/∂θ evaluated at θ=1
            # Each column represents how a member group affects each mode
            J = torch.randn(M, D, dtype=torch.float64) * 0.15
            # Make it physically plausible: positive diagonal-dominant
            for j in range(min(D, M)):
                J[j, j] += 0.3 + 0.1 * torch.rand(1, dtype=torch.float64).item()
            # Ensure all frequencies increase with stiffness
            J = J.abs() * 0.2
            self.register_buffer('J', J)
            
        def forward(self, theta):
            """theta: (..., D) → frequencies: (..., M)"""
            delta = theta - 1.0
            return self.f0 + delta @ self.J.T


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NN Surrogate for D=12 (deeper architecture)
# ═══════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
    def forward(self, x):
        return x + self.net(x)

class NNSurrogate(nn.Module):
    """NN surrogate: D → M with residual blocks."""
    def __init__(self, D=12, M=20, hidden=256, n_blocks=6):
        super().__init__()
        layers = [nn.Linear(D, hidden)]
        for _ in range(n_blocks):
            layers.append(ResBlock(hidden))
        layers.extend([nn.SiLU(), nn.Linear(hidden, M)])
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


def train_nn_surrogate(surrogate_true, D=12, M=20, n_train=10000, n_test=2000,
                       epochs=3000, lr=1e-3, device='cpu'):
    """Train NN surrogate on data from the linear stiffness model."""
    print(f"\n{'='*60}")
    print(f"Training NN Surrogate (D={D}, M={M})")
    print(f"{'='*60}")
    
    # Generate training data via LHS
    from torch.quasirandom import SobolEngine
    sobol = SobolEngine(D, scramble=True, seed=42)
    theta_train = sobol.draw(n_train).double() * 1.5 + 0.5  # [0.5, 2.0]^D
    theta_test = sobol.draw(n_test).double() * 1.5 + 0.5
    
    with torch.no_grad():
        y_train = surrogate_true(theta_train)
        y_test = surrogate_true(theta_test)
    
    # Create model
    model = NNSurrogate(D, M, hidden=256, n_blocks=6).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    theta_train = theta_train.to(device)
    y_train = y_train.to(device)
    theta_test = theta_test.to(device)
    y_test = y_test.to(device)
    
    best_test_loss = float('inf')
    best_state = None
    
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        pred = model(theta_train)
        loss = ((pred - y_train) ** 2).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch + 1) % 500 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                test_pred = model(theta_test)
                test_loss = ((test_pred - y_test) ** 2).mean()
                rel_err = ((test_pred - y_test).abs() / y_test.abs()).mean()
                
                ss_res = ((test_pred - y_test) ** 2).sum()
                ss_tot = ((y_test - y_test.mean(0)) ** 2).sum()
                r2 = 1 - ss_res / ss_tot
                
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                
                print(f"  Epoch {epoch+1:4d}: train_MSE={loss.item():.2e}, "
                      f"test_MSE={test_loss.item():.2e}, R²={r2.item():.8f}, "
                      f"MAPE={rel_err.item()*100:.4f}%")
    
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(theta_test)
        ss_res = ((test_pred - y_test) ** 2).sum()
        ss_tot = ((y_test - y_test.mean(0)) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        mape = ((test_pred - y_test).abs() / y_test.abs()).mean() * 100
        max_rel = ((test_pred - y_test).abs() / y_test.abs()).max()
    
    print(f"\n  Final: R²={r2.item():.8f}, MAPE={mape.item():.4f}%, "
          f"max_rel_err={max_rel.item():.6f}")
    print(f"  Training time: {train_time:.1f}s")
    
    return model, {
        'r2': r2.item(), 'mape': mape.item(),
        'max_rel_err': max_rel.item(), 'train_time': train_time
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Conditional Normalizing Flow for D=12
# ═══════════════════════════════════════════════════════════════════════════════

class ConditionalAffineCoupling(nn.Module):
    """Single conditional affine coupling layer with observation embedding."""
    def __init__(self, D, cond_dim, hidden=128, mask_even=True):
        super().__init__()
        self.D = D
        self.mask_even = mask_even
        
        d_in = D // 2 + cond_dim  # masked params + observation embedding
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * (D - D // 2)),  # scale + translate
        )
        # Initialize near identity
        self.net[-1].weight.data.mul_(0.01)
        self.net[-1].bias.data.zero_()
        
    def _split(self, x):
        if self.mask_even:
            return x[..., ::2], x[..., 1::2]
        else:
            return x[..., 1::2], x[..., ::2]
    
    def _merge(self, x_keep, x_trans):
        out = torch.empty_like(torch.cat([x_keep, x_trans], dim=-1))
        if self.mask_even:
            out[..., ::2] = x_keep
            out[..., 1::2] = x_trans
        else:
            out[..., 1::2] = x_keep
            out[..., ::2] = x_trans
        return out
    
    def forward(self, x, cond):
        """Forward: x → z, returns (z, log_det)"""
        x_keep, x_trans = self._split(x)
        inp = torch.cat([x_keep, cond], dim=-1)
        st = self.net(inp)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * 2  # bounded scale
        
        z_trans = x_trans * torch.exp(s) + t
        log_det = s.sum(dim=-1)
        return self._merge(x_keep, z_trans), log_det
    
    def inverse(self, z, cond):
        """Inverse: z → x"""
        z_keep, z_trans = self._split(z)
        inp = torch.cat([z_keep, cond], dim=-1)
        st = self.net(inp)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * 2
        
        x_trans = (z_trans - t) * torch.exp(-s)
        return self._merge(z_keep, x_trans)


class ConditionalRealNVP(nn.Module):
    """Conditional RealNVP for D=12 with observation embedding."""
    def __init__(self, D=12, M=20, n_layers=12, hidden=192, embed_dim=32):
        super().__init__()
        self.D = D
        
        # Observation embedding
        self.embed = nn.Sequential(
            nn.Linear(M, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        
        # Coupling layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                ConditionalAffineCoupling(D, embed_dim, hidden, mask_even=(i % 2 == 0))
            )
    
    def forward(self, theta, y):
        """theta → z, returns (z, log_det_total)"""
        cond = self.embed(y)
        z = theta
        log_det = 0
        for layer in self.layers:
            z, ld = layer(z, cond)
            log_det = log_det + ld
        return z, log_det
    
    def inverse(self, z, y):
        """z → theta"""
        cond = self.embed(y)
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x, cond)
        return x
    
    def log_prob(self, theta, y):
        """Log probability under the flow: log q(theta | y)"""
        z, log_det = self.forward(theta, y)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + log_det


def train_cnf(nn_surrogate, D=12, M=20, n_train=80000, n_val=8000,
              kappa=0.03, sigma_prior=0.5, n_layers=12, hidden=192,
              epochs=800, lr=1e-3, patience=80, device='cpu'):
    """Train conditional NF on surrogate-generated (theta, y) pairs."""
    print(f"\n{'='*60}")
    print(f"Training Conditional NF (D={D}, {n_layers} layers)")
    print(f"{'='*60}")
    
    # Generate training pairs
    print("  Generating training pairs...")
    theta_all = torch.exp(torch.randn(n_train + n_val, D, dtype=torch.float64) * sigma_prior)
    theta_all = theta_all.clamp(0.5, 2.0)
    
    nn_surrogate.eval()
    with torch.no_grad():
        y_clean = nn_surrogate(theta_all)
        noise = torch.randn_like(y_clean) * kappa * y_clean.abs()
        y_noisy = y_clean + noise
    
    # Work in log-space for theta
    log_theta = torch.log(theta_all)
    
    # Split
    lt_train, lt_val = log_theta[:n_train], log_theta[n_train:]
    y_train, y_val = y_noisy[:n_train], y_noisy[n_train:]
    
    lt_train = lt_train.to(device)
    y_train = y_train.to(device)
    lt_val = lt_val.to(device)
    y_val = y_val.to(device)
    
    # Create flow
    flow = ConditionalRealNVP(D, M, n_layers, hidden, embed_dim=32).double().to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_val_loss = float('inf')
    best_state = None
    wait = 0
    
    batch_size = 2048
    n_batches = (n_train + batch_size - 1) // batch_size
    
    t0 = time.time()
    for epoch in range(epochs):
        flow.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0
        
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            lt_b = lt_train[idx]
            y_b = y_train[idx]
            
            loss = -flow.log_prob(lt_b, y_b).mean()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        epoch_loss /= n_batches
        
        # Validation
        if (epoch + 1) % 10 == 0 or epoch == 0:
            flow.eval()
            with torch.no_grad():
                val_loss = -flow.log_prob(lt_val, y_val).mean().item()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in flow.state_dict().items()}
                wait = 0
                marker = ' *'
            else:
                wait += 10
                marker = ''
            
            if (epoch + 1) % 100 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:4d}: train={epoch_loss:.4f}, "
                      f"val={val_loss:.4f}{marker}")
            
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    train_time = time.time() - t0
    flow.load_state_dict(best_state)
    print(f"  Best val loss: {best_val_loss:.4f} nats")
    print(f"  Training time: {train_time:.1f}s")
    
    return flow, {'val_loss': best_val_loss, 'train_time': train_time}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Inference Engine (Fast + Exact modes)
# ═══════════════════════════════════════════════════════════════════════════════

def log_posterior(log_theta, y_obs, nn_surrogate, kappa=0.03, sigma_prior=0.5):
    """Compute log posterior: log p(theta | y) up to constant."""
    theta = torch.exp(log_theta)
    y_pred = nn_surrogate(theta)
    
    # Relative-error Gaussian likelihood
    rel_err = (y_pred - y_obs) / y_obs
    log_lik = -0.5 * (rel_err ** 2).sum(dim=-1) / kappa ** 2
    
    # Log-normal prior
    log_prior = -0.5 * (log_theta ** 2).sum(dim=-1) / sigma_prior ** 2
    
    return log_lik + log_prior


def fast_mode(flow, y_obs, N=2000, device='cpu'):
    """Amortized inference: generate N i.i.d. samples in one pass."""
    flow.eval()
    t0 = time.time()
    with torch.no_grad():
        z = torch.randn(N, flow.D, dtype=torch.float64, device=device)
        y_batch = y_obs.unsqueeze(0).expand(N, -1)
        log_theta = flow.inverse(z, y_batch)
        theta = torch.exp(log_theta)
    elapsed = time.time() - t0
    return theta, {'time': elapsed, 'ess': N}


def neutra_hmc(flow, nn_surrogate, y_obs, N=2000, n_warmup=500,
               n_leapfrog=20, step_size=0.15, kappa=0.03,
               sigma_prior=0.5, device='cpu'):
    """NeuTra-HMC: exact mode with flow-transported sampling."""
    flow.eval()
    D = flow.D
    
    def U_z(z_):
        """Potential energy in z-space."""
        z_ = z_.detach().requires_grad_(True)
        y_batch = y_obs.unsqueeze(0)
        log_theta = flow.inverse(z_.unsqueeze(0), y_batch).squeeze(0)
        
        # Log posterior
        lp = log_posterior(log_theta, y_obs, nn_surrogate, kappa, sigma_prior)
        
        # Jacobian correction
        z_fwd, log_det = flow.forward(log_theta.unsqueeze(0), y_batch)
        
        U = -(lp - log_det.squeeze(0))
        U.backward()
        return U.detach(), z_.grad.detach()
    
    # Initialize from flow
    with torch.no_grad():
        z0 = torch.randn(1, D, dtype=torch.float64, device=device)
        log_theta0 = flow.inverse(z0, y_obs.unsqueeze(0))
    z_current = z0.squeeze(0)
    
    samples = []
    accepts = 0
    total = 0
    
    t0 = time.time()
    for i in range(N + n_warmup):
        # Leapfrog
        p = torch.randn(D, dtype=torch.float64, device=device)
        z = z_current.clone()
        
        U_val, grad_U = U_z(z)
        H_old = U_val + 0.5 * (p ** 2).sum()
        
        p = p - 0.5 * step_size * grad_U
        for l in range(n_leapfrog - 1):
            z = z + step_size * p
            _, grad_U = U_z(z)
            p = p - step_size * grad_U
        z = z + step_size * p
        U_new, grad_U = U_z(z)
        p = p - 0.5 * step_size * grad_U
        
        H_new = U_new + 0.5 * (p ** 2).sum()
        
        # Metropolis correction
        log_alpha = -(H_new - H_old)
        if torch.log(torch.rand(1, dtype=torch.float64)) < log_alpha:
            z_current = z
            if i >= n_warmup:
                accepts += 1
        
        if i >= n_warmup:
            total += 1
            with torch.no_grad():
                lt = flow.inverse(z_current.unsqueeze(0), y_obs.unsqueeze(0))
                samples.append(torch.exp(lt.squeeze(0)))
    
    elapsed = time.time() - t0
    samples = torch.stack(samples)
    ar = accepts / total
    
    # Compute ESS
    ess_per_dim = []
    for d in range(D):
        chain = samples[:, d].numpy()
        n = len(chain)
        mean = chain.mean()
        var = chain.var()
        if var < 1e-15:
            ess_per_dim.append(float(n))
            continue
        autocorr = np.correlate(chain - mean, chain - mean, 'full')[n-1:]
        autocorr /= autocorr[0]
        # Truncate at first negative
        tau = 1.0
        for k in range(1, n):
            if autocorr[k] < 0:
                break
            tau += 2 * autocorr[k]
        ess_per_dim.append(n / tau)
    
    min_ess = min(ess_per_dim)
    mean_ess = np.mean(ess_per_dim)
    
    return samples, {
        'time': elapsed, 'accept_rate': ar,
        'min_ess': min_ess, 'mean_ess': mean_ess,
        'ess_per_dim': ess_per_dim
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Main Validation
# ═══════════════════════════════════════════════════════════════════════════════

def run_d12_validation():
    device = 'cpu'  # Blackwell sm_120 not yet supported
    D, M = 12, 20
    kappa, sigma_prior = 0.03, 0.5
    
    print("=" * 70)
    print("  D=12 HIGH-DIMENSIONAL VALIDATION FOR CAI FRAMEWORK")
    print("=" * 70)
    
    # --- Step 1: Create ground-truth surrogate ---
    surrogate_true = LinearStiffnessSurrogate(D=D, M=M).double()
    
    # --- Step 2: Train NN surrogate ---
    nn_model, nn_stats = train_nn_surrogate(
        surrogate_true, D=D, M=M,
        n_train=15000, n_test=3000,
        epochs=3000, device=device
    )
    
    # --- Step 3: Train conditional NF (deeper for D=12) ---
    flow, cnf_stats = train_cnf(
        nn_model, D=D, M=M,
        n_train=80000, n_val=8000,
        n_layers=12, hidden=192,
        epochs=800, patience=80,
        device=device
    )
    
    # --- Step 4: Generate test observations ---
    print(f"\n{'='*60}")
    print("D=12 Inference Experiments")
    print(f"{'='*60}")
    
    # 3 test observations spanning the prior
    test_configs = [
        ("Near nominal", torch.ones(D, dtype=torch.float64)),
        ("Mixed damage", torch.tensor([0.7, 1.0, 1.3, 0.8, 1.1, 0.9,
                                        1.2, 0.75, 1.0, 1.15, 0.85, 1.05],
                                       dtype=torch.float64)),
        ("Heavy damage", torch.tensor([0.55, 0.65, 0.7, 0.6, 0.75, 0.8,
                                        0.55, 0.6, 0.7, 0.65, 0.75, 0.5],
                                       dtype=torch.float64)),
    ]
    
    all_results = {}
    
    for name, theta_true in test_configs:
        print(f"\n--- {name}: θ = {theta_true.tolist()[:6]}... ---")
        
        # Generate observation
        with torch.no_grad():
            y_clean = nn_model(theta_true.unsqueeze(0)).squeeze(0)
            y_obs = y_clean + torch.randn_like(y_clean) * kappa * y_clean.abs()
        
        # Fast mode
        theta_fast, fast_stats = fast_mode(flow, y_obs, N=2000, device=device)
        fast_mean = theta_fast.mean(dim=0)
        fast_mae = (fast_mean - theta_true).abs().mean().item()
        fast_max_rel = ((fast_mean - theta_true).abs() / theta_true).max().item()
        
        print(f"  Fast mode: time={fast_stats['time']*1000:.1f}ms, "
              f"MAE={fast_mae:.4f}, max_rel_err={fast_max_rel*100:.1f}%")
        
        # Exact mode (NeuTra-HMC)
        theta_exact, exact_stats = neutra_hmc(
            flow, nn_model, y_obs, N=2000, n_warmup=500,
            n_leapfrog=20, step_size=0.10, device=device
        )
        exact_mean = theta_exact.mean(dim=0)
        exact_mae = (exact_mean - theta_true).abs().mean().item()
        exact_max_rel = ((exact_mean - theta_true).abs() / theta_true).max().item()
        
        print(f"  Exact mode: time={exact_stats['time']:.1f}s, "
              f"AR={exact_stats['accept_rate']:.1%}, "
              f"ESS_min={exact_stats['min_ess']:.0f}, "
              f"MAE={exact_mae:.4f}, max_rel_err={exact_max_rel*100:.1f}%")
        
        # Cross-check: fast vs exact
        cross_mae = (fast_mean - exact_mean).abs().mean().item()
        print(f"  Fast↔Exact MAE: {cross_mae:.4f}")
        
        all_results[name] = {
            'theta_true': theta_true,
            'y_obs': y_obs,
            'fast': {'samples': theta_fast, 'stats': fast_stats,
                     'mae': fast_mae, 'max_rel': fast_max_rel},
            'exact': {'samples': theta_exact, 'stats': exact_stats,
                      'mae': exact_mae, 'max_rel': exact_max_rel},
            'cross_mae': cross_mae,
        }
    
    # --- Step 5: Standard HMC baseline (for first observation only) ---
    print(f"\n--- Standard HMC baseline (first observation) ---")
    name0 = list(test_configs)[0][0]
    y_obs0 = all_results[name0]['y_obs']
    theta_true0 = all_results[name0]['theta_true']
    
    # Run standard HMC (no flow transport)
    t0 = time.time()
    hmc_samples = []
    hmc_accepts = 0
    log_theta_curr = torch.zeros(D, dtype=torch.float64)
    
    n_hmc = 2000
    n_warmup_hmc = 500
    
    for i in range(n_hmc + n_warmup_hmc):
        lt = log_theta_curr.detach().requires_grad_(True)
        lp = log_posterior(lt, y_obs0, nn_model, kappa, sigma_prior)
        lp.backward()
        grad = lt.grad.clone()
        
        p = torch.randn(D, dtype=torch.float64)
        H_old = -lp.detach() + 0.5 * (p ** 2).sum()
        
        lt_prop = lt.detach().clone()
        p_prop = p - 0.5 * 0.05 * (-grad)
        
        for _ in range(20):
            lt_prop = lt_prop + 0.05 * p_prop
            lt_prop = lt_prop.detach().requires_grad_(True)
            lp_prop = log_posterior(lt_prop, y_obs0, nn_model, kappa, sigma_prior)
            lp_prop.backward()
            p_prop = p_prop - 0.05 * (-lt_prop.grad.clone())
        
        H_new = -lp_prop.detach() + 0.5 * (p_prop ** 2).sum()
        
        if torch.log(torch.rand(1, dtype=torch.float64)) < -(H_new - H_old):
            log_theta_curr = lt_prop.detach()
            if i >= n_warmup_hmc:
                hmc_accepts += 1
        
        if i >= n_warmup_hmc:
            hmc_samples.append(torch.exp(log_theta_curr))
    
    hmc_time = time.time() - t0
    hmc_samples = torch.stack(hmc_samples)
    hmc_ar = hmc_accepts / n_hmc
    hmc_mean = hmc_samples.mean(0)
    hmc_mae = (hmc_mean - theta_true0).abs().mean().item()
    
    print(f"  Standard HMC: time={hmc_time:.1f}s, AR={hmc_ar:.1%}, MAE={hmc_mae:.4f}")
    
    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"  D=12 VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(f"  NN Surrogate: R²={nn_stats['r2']:.8f}, MAPE={nn_stats['mape']:.4f}%")
    print(f"  CNF: val_loss={cnf_stats['val_loss']:.4f} nats")
    print()
    
    for name, res in all_results.items():
        print(f"  {name}:")
        print(f"    Fast:  MAE={res['fast']['mae']:.4f}, "
              f"max_rel={res['fast']['max_rel']*100:.1f}%, "
              f"time={res['fast']['stats']['time']*1000:.1f}ms")
        print(f"    Exact: MAE={res['exact']['mae']:.4f}, "
              f"max_rel={res['exact']['max_rel']*100:.1f}%, "
              f"ESS_min={res['exact']['stats']['min_ess']:.0f}, "
              f"AR={res['exact']['stats']['accept_rate']:.1%}")
    
    print(f"\n  Baseline HMC (D=12): time={hmc_time:.1f}s, AR={hmc_ar:.1%}")
    
    # --- Save results ---
    save_path = Path('experiments/results/d12_validation.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_data = {
        'D': D, 'M': M,
        'nn_stats': nn_stats,
        'cnf_stats': cnf_stats,
        'results': {k: {
            'theta_true': v['theta_true'],
            'fast_mean': v['fast']['samples'].mean(0),
            'fast_stats': v['fast']['stats'],
            'fast_mae': v['fast']['mae'],
            'exact_mean': v['exact']['samples'].mean(0),
            'exact_stats': v['exact']['stats'],
            'exact_mae': v['exact']['mae'],
        } for k, v in all_results.items()},
        'hmc_baseline': {
            'time': hmc_time, 'accept_rate': hmc_ar, 'mae': hmc_mae
        },
        'nn_state_dict': nn_model.state_dict(),
        'flow_state_dict': flow.state_dict(),
    }
    torch.save(save_data, save_path)
    print(f"\n  Results saved to: {save_path}")
    

if __name__ == '__main__':
    run_d12_validation()
