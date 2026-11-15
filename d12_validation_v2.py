#!/usr/bin/env python3
"""
D=12 High-Dimensional Validation for CAI Framework — v2.1 (Device-Fixed)
=========================================================================
Fixes all CPU/GPU tensor device mismatches from v2.

Run from: C:\\nf-pmcmc\\
Usage:
  conda activate base
  python experiments/d12_validation_v2.py
"""

import sys
import os
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch.set_default_dtype(torch.float64)

# Auto-select device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    torch.backends.cudnn.benchmark = True
    print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f} GB")
else:
    print("  Using CPU")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LinearStiffnessSurrogate for D=12
# ═══════════════════════════════════════════════════════════════════════════════

class LinearStiffnessSurrogate(nn.Module):
    def __init__(self, D=12, M=20, seed=42):
        super().__init__()
        self.D = D
        self.M = M
        torch.manual_seed(seed)

        f0 = torch.tensor([
            0.805, 0.830, 1.210, 2.397, 2.462, 3.647,
            4.175, 4.393, 5.260, 5.524, 6.823, 7.063,
            7.459, 8.227, 9.018, 9.371, 10.244, 11.053,
            11.687, 12.341
        ])
        self.register_buffer('f0', f0)

        J = torch.zeros(M, D)
        for j in range(D):
            J[j, j] = 0.25 + 0.05 * torch.rand(1).item()
            if j + D < M:
                J[j + D, j] = 0.10 + 0.03 * torch.rand(1).item()
            for k in range(D):
                if k != j:
                    dist = abs(k - j)
                    J[j, k] = 0.02 * torch.exp(torch.tensor(-0.5 * dist))
                    if j + D < M:
                        J[j + D, k] = 0.01 * torch.exp(torch.tensor(-0.5 * dist))
            for m_idx in range(D, M):
                if m_idx != j + D:
                    J[m_idx, j] = 0.005 * torch.randn(1).abs().item()

        self.register_buffer('J', J)
        print(f"  LinearStiffnessSurrogate: D={D}, M={M}")
        print(f"  J condition number: {torch.linalg.cond(J[:D,:D]).item():.1f}")
        print(f"  J diagonal dominance: {(J.diag()[:D].abs() / J[:D,:].abs().sum(1)).mean().item():.3f}")

    def forward(self, theta):
        delta = theta - 1.0
        return self.f0 * (1.0 + delta @ self.J.T)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. NN Surrogate for D=12
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
        self.net[-1].weight.data.mul_(0.01)
        self.net[-1].bias.data.zero_()

    def forward(self, x):
        return x + self.net(x)


class NNSurrogate(nn.Module):
    def __init__(self, D=12, M=20, hidden=512, n_blocks=8):
        super().__init__()
        layers = [nn.Linear(D, hidden)]
        for _ in range(n_blocks):
            layers.append(ResBlock(hidden))
        layers.extend([nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, M)])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NNSurrogateWrapper(nn.Module):
    def __init__(self, model, theta_mean, theta_std, y_mean, y_std):
        super().__init__()
        self.model = model
        self.register_buffer('theta_mean', theta_mean)
        self.register_buffer('theta_std', theta_std)
        self.register_buffer('y_mean', y_mean)
        self.register_buffer('y_std', y_std)

    def forward(self, theta):
        theta_n = (theta - self.theta_mean) / self.theta_std
        y_n = self.model(theta_n)
        return y_n * self.y_std + self.y_mean


def train_nn_surrogate(surrogate_true, D=12, M=20, n_train=50000, n_test=5000,
                       epochs=5000, lr=5e-4, device='cpu'):
    print(f"\n{'='*60}")
    print(f"Training NN Surrogate (D={D}, M={M}, N={n_train})")
    print(f"{'='*60}")

    from torch.quasirandom import SobolEngine
    sobol = SobolEngine(D, scramble=True, seed=42)
    theta_train = (sobol.draw(n_train) * 1.5 + 0.5).to(device)
    sobol2 = SobolEngine(D, scramble=True, seed=123)
    theta_test = (sobol2.draw(n_test) * 1.5 + 0.5).to(device)

    with torch.no_grad():
        y_train = surrogate_true(theta_train)
        y_test = surrogate_true(theta_test)

    theta_mean = theta_train.mean(0)
    theta_std = theta_train.std(0)
    y_mean = y_train.mean(0)
    y_std = y_train.std(0).clamp(min=1e-8)

    theta_train_n = (theta_train - theta_mean) / theta_std
    theta_test_n = (theta_test - theta_mean) / theta_std
    y_train_n = (y_train - y_mean) / y_std
    y_test_n = (y_test - y_mean) / y_std

    model = NNSurrogate(D, M, hidden=512, n_blocks=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_test_loss = float('inf')
    best_state = None
    batch_size = 4096

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            pred = model(theta_train_n[idx])
            loss = ((pred - y_train_n[idx]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss /= n_batches

        if (epoch + 1) % 500 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                test_pred_n = model(theta_test_n)
                test_pred = test_pred_n * y_std + y_mean
                ss_res = ((test_pred - y_test) ** 2).sum()
                ss_tot = ((y_test - y_test.mean(0)) ** 2).sum()
                r2 = 1 - ss_res / ss_tot
                rel_err = ((test_pred - y_test).abs() / y_test.abs())
                mape = rel_err.mean() * 100
                max_rel = rel_err.max()
                test_mse = ((test_pred - y_test) ** 2).mean()

                if test_mse < best_test_loss:
                    best_test_loss = test_mse
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

                print(f"  Epoch {epoch+1:5d}: train_MSE={epoch_loss:.2e}, "
                      f"R\u00b2={r2.item():.8f}, MAPE={mape.item():.4f}%, "
                      f"max_rel={max_rel.item():.6f}")

    train_time = time.time() - t0
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_pred_n = model(theta_test_n)
        test_pred = test_pred_n * y_std + y_mean
        ss_res = ((test_pred - y_test) ** 2).sum()
        ss_tot = ((y_test - y_test.mean(0)) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        mape = ((test_pred - y_test).abs() / y_test.abs()).mean() * 100
        max_rel = ((test_pred - y_test).abs() / y_test.abs()).max()

    print(f"\n  Final: R\u00b2={r2.item():.8f}, MAPE={mape.item():.4f}%, "
          f"max_rel={max_rel.item():.6f}")
    print(f"  Training time: {train_time:.1f}s")

    return model, {
        'r2': r2.item(), 'mape': mape.item(),
        'max_rel_err': max_rel.item(), 'train_time': train_time,
        'theta_mean': theta_mean, 'theta_std': theta_std,
        'y_mean': y_mean, 'y_std': y_std,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Conditional Normalizing Flow for D=12
# ═══════════════════════════════════════════════════════════════════════════════

class ConditionalAffineCoupling(nn.Module):
    def __init__(self, D, cond_dim, hidden=256, mask_even=True):
        super().__init__()
        self.D = D
        self.mask_even = mask_even
        d_keep = D // 2
        d_trans = D - d_keep
        d_in = d_keep + cond_dim

        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * d_trans),
        )
        self.net[-1].weight.data.mul_(0.001)
        self.net[-1].bias.data.zero_()

    def _split(self, x):
        if self.mask_even:
            return x[..., ::2], x[..., 1::2]
        else:
            return x[..., 1::2], x[..., ::2]

    def _merge(self, x_keep, x_trans):
        D = x_keep.shape[-1] + x_trans.shape[-1]
        out = torch.empty(*x_keep.shape[:-1], D, dtype=x_keep.dtype, device=x_keep.device)
        if self.mask_even:
            out[..., ::2] = x_keep
            out[..., 1::2] = x_trans
        else:
            out[..., 1::2] = x_keep
            out[..., ::2] = x_trans
        return out

    def forward(self, x, cond):
        x_keep, x_trans = self._split(x)
        inp = torch.cat([x_keep, cond], dim=-1)
        st = self.net(inp)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * 2.0
        z_trans = x_trans * torch.exp(s) + t
        log_det = s.sum(dim=-1)
        return self._merge(x_keep, z_trans), log_det

    def inverse(self, z, cond):
        z_keep, z_trans = self._split(z)
        inp = torch.cat([z_keep, cond], dim=-1)
        st = self.net(inp)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * 2.0
        x_trans = (z_trans - t) * torch.exp(-s)
        return self._merge(z_keep, x_trans)


class ConditionalRealNVP(nn.Module):
    def __init__(self, D=12, M=20, n_layers=12, hidden=256, embed_dim=64):
        super().__init__()
        self.D = D
        self.embed = nn.Sequential(
            nn.Linear(M, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                ConditionalAffineCoupling(D, embed_dim, hidden, mask_even=(i % 2 == 0))
            )

    def forward(self, theta, y):
        cond = self.embed(y)
        z = theta
        log_det = 0
        for layer in self.layers:
            z, ld = layer(z, cond)
            log_det = log_det + ld
        return z, log_det

    def inverse(self, z, y):
        cond = self.embed(y)
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x, cond)
        return x

    def log_prob(self, theta, y):
        z, log_det = self.forward(theta, y)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + log_det


def train_cnf(nn_surrogate, D=12, M=20, n_train=120000, n_val=12000,
              kappa=0.03, sigma_prior=0.5, n_layers=12, hidden=256,
              epochs=1200, lr=5e-4, patience=120, device='cpu'):
    print(f"\n{'='*60}")
    print(f"Training Conditional NF (D={D}, {n_layers} layers, {hidden} hidden)")
    print(f"{'='*60}")

    print("  Generating training pairs...")
    theta_all = torch.exp(torch.randn(n_train + n_val, D) * sigma_prior)
    theta_all = theta_all.clamp(0.5, 2.0).to(device)

    nn_surrogate.eval()
    with torch.no_grad():
        chunk = 10000
        y_list = []
        for i in range(0, n_train + n_val, chunk):
            y_clean = nn_surrogate(theta_all[i:i+chunk])
            noise = torch.randn_like(y_clean) * kappa * y_clean.abs()
            y_list.append(y_clean + noise)
        y_noisy = torch.cat(y_list, dim=0)

    log_theta = torch.log(theta_all)

    lt_mean = log_theta.mean(0)
    lt_std = log_theta.std(0).clamp(min=1e-6)
    log_theta_n = (log_theta - lt_mean) / lt_std

    y_mean = y_noisy.mean(0)
    y_std = y_noisy.std(0).clamp(min=1e-6)
    y_n = (y_noisy - y_mean) / y_std

    lt_train, lt_val = log_theta_n[:n_train], log_theta_n[n_train:]
    y_train, y_val = y_n[:n_train], y_n[n_train:]

    flow = ConditionalRealNVP(D, M, n_layers, hidden, embed_dim=64).to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val_loss = float('inf')
    best_state = None
    wait = 0
    batch_size = 4096

    t0 = time.time()
    for epoch in range(epochs):
        flow.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            loss = -flow.log_prob(lt_train[idx], y_train[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss /= n_batches

        if (epoch + 1) % 20 == 0 or epoch == 0:
            flow.eval()
            with torch.no_grad():
                val_loss = -flow.log_prob(lt_val, y_val).mean().item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in flow.state_dict().items()}
                wait = 0
                marker = ' *'
            else:
                wait += 20
                marker = ''

            if (epoch + 1) % 200 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:5d}: train={epoch_loss:.4f}, "
                      f"val={val_loss:.4f}{marker}")

            if wait >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    train_time = time.time() - t0
    flow.load_state_dict(best_state)
    print(f"  Best val loss: {best_val_loss:.4f} nats")
    print(f"  Training time: {train_time:.1f}s")

    return flow, {
        'val_loss': best_val_loss, 'train_time': train_time,
        'lt_mean': lt_mean, 'lt_std': lt_std,
        'y_mean': y_mean, 'y_std': y_std,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Inference Engine
# ═══════════════════════════════════════════════════════════════════════════════

def log_posterior(log_theta, y_obs, nn_surrogate, kappa=0.03, sigma_prior=0.5):
    theta = torch.exp(log_theta)
    y_pred = nn_surrogate(theta)
    rel_err = (y_pred - y_obs) / y_obs
    log_lik = -0.5 * (rel_err ** 2).sum(dim=-1) / kappa ** 2
    log_prior = -0.5 * (log_theta ** 2).sum(dim=-1) / sigma_prior ** 2
    return log_lik + log_prior


def fast_mode(flow, y_obs, cnf_stats, N=2000, device='cpu'):
    flow.eval()
    lt_mean = cnf_stats['lt_mean'].to(device)
    lt_std = cnf_stats['lt_std'].to(device)
    y_mean = cnf_stats['y_mean'].to(device)
    y_std = cnf_stats['y_std'].to(device)

    t0 = time.time()
    with torch.no_grad():
        z = torch.randn(N, flow.D, device=device)
        y_n = ((y_obs.to(device) - y_mean) / y_std).unsqueeze(0).expand(N, -1)
        log_theta_n = flow.inverse(z, y_n)
        log_theta = log_theta_n * lt_std + lt_mean
        theta = torch.exp(log_theta)
    elapsed = time.time() - t0
    return theta, {'time': elapsed, 'ess': N}


def neutra_hmc(flow, nn_surrogate, y_obs, cnf_stats,
               N=2000, n_warmup=500, n_leapfrog=25,
               init_step_size=0.1, kappa=0.03, sigma_prior=0.5,
               device='cpu'):
    flow.eval()
    D = flow.D
    lt_mean = cnf_stats['lt_mean'].to(device)
    lt_std = cnf_stats['lt_std'].to(device)
    y_mean = cnf_stats['y_mean'].to(device)
    y_std = cnf_stats['y_std'].to(device)
    y_obs = y_obs.to(device)
    y_n = ((y_obs - y_mean) / y_std).unsqueeze(0)

    def U_z(z_):
        z_ = z_.detach().requires_grad_(True)
        log_theta_n = flow.inverse(z_.unsqueeze(0), y_n).squeeze(0)
        log_theta = log_theta_n * lt_std + lt_mean
        lp = log_posterior(log_theta, y_obs, nn_surrogate, kappa, sigma_prior)
        z_fwd, log_det = flow.forward(log_theta_n.unsqueeze(0), y_n)
        log_det_scale = lt_std.abs().log().sum()
        U = -(lp - log_det.squeeze(0) - log_det_scale)
        U.backward()
        return U.detach(), z_.grad.detach()

    with torch.no_grad():
        z0 = torch.randn(1, D, device=device)
    z_current = z0.squeeze(0)

    samples = []
    accepts = 0
    total = 0
    step_size = init_step_size
    target_ar = 0.65
    warmup_accepts = 0
    warmup_total = 0

    t0 = time.time()
    for i in range(N + n_warmup):
        p = torch.randn(D, device=device)
        z = z_current.clone()

        try:
            U_val, grad_U = U_z(z)
            if torch.isnan(U_val) or torch.isnan(grad_U).any():
                if i >= n_warmup:
                    with torch.no_grad():
                        lt_n = flow.inverse(z_current.unsqueeze(0), y_n).squeeze(0)
                        samples.append(torch.exp(lt_n * lt_std + lt_mean))
                    total += 1
                continue

            H_old = U_val + 0.5 * (p ** 2).sum()

            p = p - 0.5 * step_size * grad_U
            for l in range(n_leapfrog - 1):
                z = z + step_size * p
                _, grad_U = U_z(z)
                if torch.isnan(grad_U).any():
                    break
                p = p - step_size * grad_U
            z = z + step_size * p
            U_new, grad_U = U_z(z)
            p = p - 0.5 * step_size * grad_U

            H_new = U_new + 0.5 * (p ** 2).sum()
            log_alpha = -(H_new - H_old)
            accepted = not torch.isnan(log_alpha) and torch.log(torch.rand(1, device=device)) < log_alpha

        except Exception:
            accepted = False

        if accepted:
            z_current = z.detach()
            if i < n_warmup:
                warmup_accepts += 1
            else:
                accepts += 1

        if i < n_warmup:
            warmup_total += 1
            if warmup_total % 50 == 0 and warmup_total > 0:
                ar = warmup_accepts / warmup_total
                if ar < target_ar - 0.1:
                    step_size *= 0.8
                elif ar > target_ar + 0.1:
                    step_size *= 1.2
                step_size = max(0.005, min(step_size, 0.5))

        if i >= n_warmup:
            total += 1
            with torch.no_grad():
                lt_n = flow.inverse(z_current.unsqueeze(0), y_n).squeeze(0)
                lt = lt_n * lt_std + lt_mean
                samples.append(torch.exp(lt))

        if (i + 1) % 500 == 0:
            if i < n_warmup:
                print(f"    Warmup {i+1}/{n_warmup}: AR={warmup_accepts/max(1,warmup_total):.1%}, "
                      f"step_size={step_size:.4f}")
            else:
                print(f"    Sample {i+1-n_warmup}/{N}: AR={accepts/max(1,total):.1%}")

    elapsed = time.time() - t0

    if len(samples) == 0:
        print("  WARNING: No samples collected!")
        return torch.ones(1, D, device=device), {
            'time': elapsed, 'accept_rate': 0,
            'min_ess': 0, 'mean_ess': 0, 'ess_per_dim': [0]*D}

    samples = torch.stack(samples)
    ar = accepts / max(1, total)

    samples_cpu = samples.cpu().numpy()
    ess_per_dim = []
    for d in range(D):
        chain = samples_cpu[:, d]
        n = len(chain)
        if n < 10:
            ess_per_dim.append(1.0)
            continue
        var = chain.var()
        if var < 1e-15:
            ess_per_dim.append(float(n))
            continue
        mean = chain.mean()
        autocorr = np.correlate(chain - mean, chain - mean, 'full')[n-1:]
        autocorr /= autocorr[0]
        tau = 1.0
        for k in range(1, min(n // 3, 500)):
            if autocorr[k] < 0:
                break
            tau += 2 * autocorr[k]
        ess_per_dim.append(max(1.0, n / tau))

    return samples, {
        'time': elapsed, 'accept_rate': ar,
        'min_ess': min(ess_per_dim), 'mean_ess': np.mean(ess_per_dim),
        'ess_per_dim': ess_per_dim, 'final_step_size': step_size
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_d12_validation():
    device = DEVICE
    D, M = 12, 20
    kappa, sigma_prior = 0.03, 0.5

    print("=" * 70)
    print("  D=12 HIGH-DIMENSIONAL VALIDATION \u2014 v2.1 (Device-Fixed)")
    print("=" * 70)

    # Step 1
    surrogate_true = LinearStiffnessSurrogate(D=D, M=M).to(device)

    # Step 2
    nn_model, nn_stats = train_nn_surrogate(
        surrogate_true, D=D, M=M,
        n_train=50000, n_test=5000,
        epochs=5000, lr=5e-4, device=device
    )

    nn_wrapped = NNSurrogateWrapper(
        nn_model,
        nn_stats['theta_mean'], nn_stats['theta_std'],
        nn_stats['y_mean'], nn_stats['y_std']
    ).to(device)

    print("\n  Verifying surrogate on 1000 random samples...")
    with torch.no_grad():
        theta_check = (torch.rand(1000, D) * 1.5 + 0.5).to(device)
        y_true = surrogate_true(theta_check)
        y_nn = nn_wrapped(theta_check)
        check_r2 = 1 - ((y_nn - y_true)**2).sum() / ((y_true - y_true.mean(0))**2).sum()
        check_mape = ((y_nn - y_true).abs() / y_true.abs()).mean() * 100
        print(f"  Verification: R\u00b2={check_r2.item():.8f}, MAPE={check_mape.item():.4f}%")

    if check_r2 < 0.9999:
        print("  WARNING: Surrogate R\u00b2 < 0.9999.")

    # Step 3
    flow, cnf_stats = train_cnf(
        nn_wrapped, D=D, M=M,
        n_train=120000, n_val=12000,
        n_layers=12, hidden=256,
        epochs=1200, lr=5e-4, patience=120,
        device=device
    )

    # Step 4
    print(f"\n{'='*60}")
    print("D=12 Inference Experiments")
    print(f"{'='*60}")

    test_configs = [
        ("Near nominal", torch.ones(D, device=device)),
        ("Mixed damage", torch.tensor([0.7, 1.0, 1.3, 0.8, 1.1, 0.9,
                                        1.2, 0.75, 1.0, 1.15, 0.85, 1.05],
                                       device=device)),
        ("Heavy damage", torch.tensor([0.55, 0.65, 0.7, 0.6, 0.75, 0.8,
                                        0.55, 0.6, 0.7, 0.65, 0.75, 0.5],
                                       device=device)),
    ]

    all_results = {}
    for name, theta_true in test_configs:
        print(f"\n--- {name}: theta_true[:6] = {theta_true[:6].cpu().tolist()} ---")

        with torch.no_grad():
            y_clean = nn_wrapped(theta_true.unsqueeze(0)).squeeze(0)
            y_obs = y_clean + torch.randn_like(y_clean) * kappa * y_clean.abs()

        theta_fast, fast_stats = fast_mode(flow, y_obs, cnf_stats, N=2000, device=device)
        fast_mean = theta_fast.mean(0)
        fast_mae = (fast_mean - theta_true).abs().mean().item()
        fast_max_rel = ((fast_mean - theta_true).abs() / theta_true).max().item()
        print(f"  Fast mode: time={fast_stats['time']*1000:.1f}ms, "
              f"MAE={fast_mae:.4f}, max_rel={fast_max_rel*100:.1f}%")

        theta_exact, exact_stats = neutra_hmc(
            flow, nn_wrapped, y_obs, cnf_stats,
            N=2000, n_warmup=500, n_leapfrog=25,
            init_step_size=0.1, device=device
        )
        exact_mean = theta_exact.mean(0)
        exact_mae = (exact_mean - theta_true).abs().mean().item()
        exact_max_rel = ((exact_mean - theta_true).abs() / theta_true).max().item()
        print(f"  Exact mode: time={exact_stats['time']:.1f}s, "
              f"AR={exact_stats['accept_rate']:.1%}, "
              f"ESS_min={exact_stats['min_ess']:.0f}, "
              f"MAE={exact_mae:.4f}, max_rel={exact_max_rel*100:.1f}%")

        cross_mae = (fast_mean - exact_mean).abs().mean().item()
        print(f"  Fast<->Exact MAE: {cross_mae:.4f}")

        all_results[name] = {
            'theta_true': theta_true.cpu(),
            'fast': {'mae': fast_mae, 'max_rel': fast_max_rel, 'stats': fast_stats,
                     'samples': theta_fast.cpu()},
            'exact': {'mae': exact_mae, 'max_rel': exact_max_rel, 'stats': exact_stats,
                      'samples': theta_exact.cpu()},
            'cross_mae': cross_mae,
        }

    # Step 5: Standard HMC baseline
    print(f"\n--- Standard HMC baseline (Near nominal) ---")
    theta_true0 = test_configs[0][1]
    with torch.no_grad():
        y_obs0 = nn_wrapped(theta_true0.unsqueeze(0)).squeeze(0)
        y_obs0 = y_obs0 + torch.randn_like(y_obs0) * kappa * y_obs0.abs()

    t0 = time.time()
    hmc_samples = []
    hmc_accepts = 0
    log_theta_curr = torch.zeros(D, device=device)
    n_hmc, n_warmup_hmc = 2000, 500
    hmc_step = 0.02

    for i in range(n_hmc + n_warmup_hmc):
        lt = log_theta_curr.detach().requires_grad_(True)
        lp = log_posterior(lt, y_obs0, nn_wrapped, kappa, sigma_prior)
        lp.backward()
        grad = lt.grad.clone()

        p = torch.randn(D, device=device)
        H_old = -lp.detach() + 0.5 * (p ** 2).sum()

        lt_prop = lt.detach().clone()
        p_prop = p + 0.5 * hmc_step * grad

        for _ in range(30):
            lt_prop = lt_prop + hmc_step * p_prop
            lt_prop = lt_prop.detach().requires_grad_(True)
            lp_prop = log_posterior(lt_prop, y_obs0, nn_wrapped, kappa, sigma_prior)
            lp_prop.backward()
            p_prop = p_prop + hmc_step * lt_prop.grad.clone()

        p_prop = p_prop + 0.5 * hmc_step * lt_prop.grad.clone()
        H_new = -lp_prop.detach() + 0.5 * (p_prop ** 2).sum()

        if torch.log(torch.rand(1, device=device)) < -(H_new - H_old):
            log_theta_curr = lt_prop.detach()
            if i >= n_warmup_hmc:
                hmc_accepts += 1

        if i >= n_warmup_hmc:
            hmc_samples.append(torch.exp(log_theta_curr).cpu())

    hmc_time = time.time() - t0
    hmc_samples_t = torch.stack(hmc_samples)
    hmc_ar = hmc_accepts / n_hmc
    hmc_mean = hmc_samples_t.mean(0)
    hmc_mae = (hmc_mean - theta_true0.cpu()).abs().mean().item()
    print(f"  Standard HMC: time={hmc_time:.1f}s, AR={hmc_ar:.1%}, MAE={hmc_mae:.4f}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  D=12 VALIDATION SUMMARY (v2.1)")
    print(f"{'='*70}")
    print(f"  NN Surrogate: R\u00b2={nn_stats['r2']:.8f}, MAPE={nn_stats['mape']:.4f}%")
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

    print(f"\n  Baseline HMC (D=12): time={hmc_time:.1f}s, AR={hmc_ar:.1%}, MAE={hmc_mae:.4f}")

    # Save
    save_path = Path('experiments/results/d12_validation_v2.pt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'D': D, 'M': M,
        'nn_stats': {k: v.cpu() if isinstance(v, torch.Tensor) else v
                     for k, v in nn_stats.items()},
        'cnf_stats': {k: v.cpu() if isinstance(v, torch.Tensor) else v
                      for k, v in cnf_stats.items()},
        'results': {k: {
            'theta_true': v['theta_true'],
            'fast_mae': v['fast']['mae'], 'fast_max_rel': v['fast']['max_rel'],
            'fast_time': v['fast']['stats']['time'],
            'exact_mae': v['exact']['mae'], 'exact_max_rel': v['exact']['max_rel'],
            'exact_ess_min': v['exact']['stats']['min_ess'],
            'exact_ar': v['exact']['stats']['accept_rate'],
            'exact_time': v['exact']['stats']['time'],
        } for k, v in all_results.items()},
        'hmc_baseline': {'time': hmc_time, 'ar': hmc_ar, 'mae': hmc_mae},
    }, save_path)
    print(f"\n  Results saved to: {save_path}")


if __name__ == '__main__':
    run_d12_validation()
