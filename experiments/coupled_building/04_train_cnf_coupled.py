#!/usr/bin/env python3
"""
Stage 2 — Train conditional RealNVP for the coupled building (D=3, dim_y=20).

Training data:
  θ ~ LogNormal(0, σ=0.5) clipped to [0.5, 4.0]^3
  y = surrogate(θ) + noise,  noise ~ N(0, κ²·diag(y²)),  κ=0.03
  50 000 train + 5 000 val pairs

Arch:    8 coupling layers, 128 hidden, y-embed 20→64→20
Train:   Adam lr=1e-3, cosine annealing, 500 epochs, early stop pat=50
Device:  GPU (cuda)
Output:  ../results/conditional_flow_coupled.pt
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
SURR_DIR = HERE.parent.parent  # surrogate_training/
sys.path.insert(0, str(HERE))

from config_coupled import N_PARAMS, N_MODES, THETA_LO, THETA_HI  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] using {DEV}", flush=True)

D = N_PARAMS          # 3
M = N_MODES           # 20

SURR_PATH = SURR_DIR / "surrogate_models_coupled/member_0.pt"
OUT_DIR = SURR_DIR / "experiments/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FLOW = OUT_DIR / "conditional_flow_coupled.pt"


# ── Surrogate class (must match training script) ──────────────────────
class ResBlock(nn.Module):
    def __init__(self, d, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.SiLU(), nn.Linear(d, d),
            nn.LayerNorm(d), nn.SiLU(), nn.Dropout(drop), nn.Linear(d, d))

    def forward(self, x):
        return x + self.net(x)


class Surrogate(nn.Module):
    def __init__(self, nin=D, nout=M, hid=256, nb=4):
        super().__init__()
        self.nin, self.nout = nin, nout
        self.inp = nn.Sequential(nn.Linear(nin, hid), nn.SiLU())
        self.blk = nn.ModuleList([ResBlock(hid) for _ in range(nb)])
        self.out = nn.Sequential(
            nn.LayerNorm(hid), nn.SiLU(), nn.Linear(hid, nout))
        for n in ("xm", "xs", "ym", "ys"):
            self.register_buffer(n, torch.zeros(nin if "x" in n else nout))

    def _fwd_n(self, xn):
        h = self.inp(xn)
        for b in self.blk:
            h = b(h)
        return self.out(h)

    def forward(self, x):
        return self._fwd_n((x - self.xm) / self.xs) * self.ys + self.ym


def load_surrogate():
    m = Surrogate().to(DEV)
    state = torch.load(SURR_PATH, map_location=DEV, weights_only=True)
    m.load_state_dict(state)
    m.eval()
    return m


# ── Conditional RealNVP (must match cai_lib.py for later loading) ─────
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
    def __init__(self, dim_theta=D, dim_y=M, n_layers=8, hidden_dim=128):
        super().__init__()
        self.dim_theta, self.dim_y = dim_theta, dim_y
        self.layers = nn.ModuleList([
            ConditionalAffineCoupling(dim_theta, dim_y, hidden_dim,
                                      mask_even=(i % 2 == 0))
            for i in range(n_layers)
        ])
        # NOTE: hidden width of the y-embed MLP is fixed to 64 per spec
        # ("embedding network: 20→64→20"), independent of hidden_dim above.
        self.y_embed = nn.Sequential(
            nn.Linear(dim_y, 64), nn.SiLU(),
            nn.Linear(64, dim_y),
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
        if y.dim() == 1:
            y_batch = y.unsqueeze(0).expand(theta.shape[0], -1)
        else:
            y_batch = y
        z, log_det = self.forward(theta, y_batch)
        log_pz = (-0.5 * (z ** 2).sum(dim=-1)
                  - 0.5 * self.dim_theta * math.log(2 * math.pi))
        return log_pz + log_det


# ── Training-data generation (truncated LogNormal prior on θ) ─────────
def generate_training_data(surrogate, n_train=50000, n_val=5000,
                           kappa=0.03, prior_sigma=0.5,
                           lo=THETA_LO, hi=THETA_HI, seed=42):
    torch.manual_seed(seed)
    total = n_train + n_val
    print(f"  Generating {total} (θ, y) pairs "
          f"(κ={kappa}, σ_prior={prior_sigma}, clip=[{lo},{hi}])",
          flush=True)
    all_theta, all_y = [], []
    batch_size = 5000

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            B = end - start
            # Rejection-clip: oversample, then clip *and* drop OOB rows.
            # To keep things simple and deterministic, draw LogNormal then clamp.
            # Clamping is preferred over rejection so prior remains analytic.
            log_theta = prior_sigma * torch.randn(B, D, device=DEV)
            theta = torch.exp(log_theta).clamp(min=lo, max=hi)
            f_clean = surrogate(theta)
            noise = kappa * f_clean * torch.randn_like(f_clean)
            y = f_clean + noise
            all_theta.append(theta)
            all_y.append(y)

    theta_all = torch.cat(all_theta, dim=0)
    y_all = torch.cat(all_y, dim=0)
    theta_tr, theta_val = theta_all[:n_train], theta_all[n_train:]
    y_tr, y_val = y_all[:n_train], y_all[n_train:]

    print(f"  θ range: [{float(theta_all.min()):.3f}, "
          f"{float(theta_all.max()):.3f}]", flush=True)
    print(f"  y range: [{float(y_all.min()):.3f}, "
          f"{float(y_all.max()):.3f}]", flush=True)
    return theta_tr, y_tr, theta_val, y_val


# ── Training loop ─────────────────────────────────────────────────────
def train_flow(flow, theta_tr, y_tr, theta_val, y_val,
               n_epochs=500, lr=1e-3, batch_size=512, patience=50):
    opt = torch.optim.Adam(flow.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    dataset = torch.utils.data.TensorDataset(theta_tr, y_tr)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    best, bst, no_imp = float("inf"), None, 0
    t0 = time.time()

    for ep in range(1, n_epochs + 1):
        flow.train()
        tot, n = 0.0, 0
        for tb, yb in loader:
            z, ld = flow(tb, yb)
            log_pz = -0.5 * (z ** 2).sum(dim=-1)
            loss = -(log_pz + ld).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
            tot += loss.item() * tb.shape[0]
            n += tb.shape[0]
        sch.step()
        train_loss = tot / n

        flow.eval()
        with torch.no_grad():
            zv, ldv = flow(theta_val, y_val)
            log_pzv = -0.5 * (zv ** 2).sum(dim=-1)
            val_loss = -(log_pzv + ldv).mean().item()

        if val_loss < best:
            best = val_loss
            bst = {k: v.clone() for k, v in flow.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1

        if ep == 1 or ep % 25 == 0:
            print(f"  ep{ep:4d}  train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  best={best:.4f}", flush=True)

        if no_imp >= patience:
            print(f"  early stop @{ep} (no improvement for {patience} epochs)",
                  flush=True)
            break

    print(f"  training time: {time.time()-t0:.1f}s", flush=True)
    flow.load_state_dict(bst)
    flow.eval()
    return flow, best


def main():
    print(f"[load] surrogate: {SURR_PATH}", flush=True)
    surrogate = load_surrogate()

    # Sanity check surrogate
    with torch.no_grad():
        f1 = surrogate(torch.ones(1, D, device=DEV)).squeeze()
    print(f"  surrogate(θ=1) first-6 modes: "
          f"{[float(x) for x in f1[:6].cpu().tolist()]}", flush=True)

    print("\n[data] generating training pairs ...", flush=True)
    theta_tr, y_tr, theta_val, y_val = generate_training_data(
        surrogate, n_train=50000, n_val=5000,
        kappa=0.03, prior_sigma=0.5,
        lo=THETA_LO, hi=THETA_HI, seed=42)
    print(f"  train: {theta_tr.shape}, {y_tr.shape}", flush=True)
    print(f"  val:   {theta_val.shape}, {y_val.shape}", flush=True)

    print("\n[build] ConditionalRealNVP (D=3, dim_y=20, "
          "8 layers, 128 hidden, y-emb 20→64→20)", flush=True)
    flow = ConditionalRealNVP(dim_theta=D, dim_y=M,
                              n_layers=8, hidden_dim=128).to(DEV)
    n_p = sum(p.numel() for p in flow.parameters())
    print(f"  total params: {n_p:,}", flush=True)

    print("\n[train] starting ...", flush=True)
    flow, best = train_flow(flow, theta_tr, y_tr, theta_val, y_val,
                            n_epochs=500, lr=1e-3, batch_size=512, patience=50)

    # Reconstruction error
    with torch.no_grad():
        zv, _ = flow(theta_val[:200], y_val[:200])
        x_back, _ = flow.inverse(zv, y_val[:200])
        recon_err = (x_back - theta_val[:200]).abs().max().item()
    print(f"\n[check] best val NLL = {best:.4f}", flush=True)
    print(f"[check] max recon |θ - T⁻¹(T(θ))| = {recon_err:.2e}", flush=True)

    torch.save(flow.state_dict(), OUT_FLOW)
    print(f"\n[save] {OUT_FLOW}", flush=True)


if __name__ == "__main__":
    main()
