#!/usr/bin/env python3
"""
Stage 1 - Train NN surrogate for the coupled building (D=3 -> 20 modes).

Dataset:  ../../dataset_coupled_train.csv  (5000 rows)
Split:    4000 train / 1000 test (shuffled, seed=42)
Arch:     ResNet (4 residual blocks, 256 hidden, LayerNorm+SiLU)
Device:   GPU (cuda)
Output:   ../../surrogate_models_coupled/member_0.pt
          ../../surrogate_models_coupled/metrics.json
"""
import csv as csv_mod
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
SURR_DIR = HERE.parent.parent  # surrogate_training/
sys.path.insert(0, str(HERE))

from config_coupled import (  # noqa: E402
    N_PARAMS, N_MODES, PARAM_NAMES, FREQ_NAMES,
)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] using {DEV}", flush=True)

TRAIN_CSV = SURR_DIR / "dataset_coupled_train.csv"
OUT_DIR = SURR_DIR / "surrogate_models_coupled"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Model (identical to sailboat 03_train_surrogate.py, but D=3 → M=20) ──
class ResBlock(nn.Module):
    def __init__(self, d, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.SiLU(), nn.Linear(d, d),
            nn.LayerNorm(d), nn.SiLU(), nn.Dropout(drop), nn.Linear(d, d))

    def forward(self, x):
        return x + self.net(x)


class Surrogate(nn.Module):
    def __init__(self, nin=N_PARAMS, nout=N_MODES, hid=256, nb=4):
        super().__init__()
        self.nin, self.nout = nin, nout
        self.inp = nn.Sequential(nn.Linear(nin, hid), nn.SiLU())
        self.blk = nn.ModuleList([ResBlock(hid) for _ in range(nb)])
        self.out = nn.Sequential(
            nn.LayerNorm(hid), nn.SiLU(), nn.Linear(hid, nout))
        for n in ("xm", "xs", "ym", "ys"):
            self.register_buffer(n, torch.zeros(nin if "x" in n else nout))

    def set_norm(self, X, Y):
        self.xm, self.xs = X.mean(0), X.std(0).clamp(min=1e-8)
        self.ym, self.ys = Y.mean(0), Y.std(0).clamp(min=1e-8)

    def _fwd_n(self, xn):
        h = self.inp(xn)
        for b in self.blk:
            h = b(h)
        return self.out(h)

    def forward(self, x):
        return self._fwd_n((x - self.xm) / self.xs) * self.ys + self.ym


def load_data(p):
    X, Y = [], []
    with open(p) as f:
        for r in csv_mod.DictReader(f):
            X.append([float(r[n]) for n in PARAM_NAMES])
            Y.append([float(r[f"f{j+1}"]) for j in range(N_MODES)])
    return np.array(X), np.array(Y)


def train(Xtr, Ytr, Xte, Yte, hid=256, nb=4, lr=1e-3, ep=2000, bs=256,
          pat=200, seed=42):
    torch.manual_seed(seed)
    xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(Ytr, dtype=torch.float32, device=DEV)
    xv = torch.tensor(Xte, dtype=torch.float32, device=DEV)
    yv = torch.tensor(Yte, dtype=torch.float32, device=DEV)
    m = Surrogate(hid=hid, nb=nb).to(DEV)
    m.set_norm(xt, yt)
    xtn = (xt - m.xm) / m.xs
    ytn = (yt - m.ym) / m.ys
    xvn = (xv - m.xm) / m.xs
    yvn = (yv - m.ym) / m.ys
    ld = DataLoader(TensorDataset(xtn, ytn), batch_size=bs, shuffle=True)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    best, bst, w = 1e9, None, 0
    t0 = time.time()
    for e in range(1, ep + 1):
        m.train()
        tl = 0
        nb_ = 0
        for xb, yb in ld:
            l = nn.functional.mse_loss(m._fwd_n(xb), yb)
            opt.zero_grad()
            l.backward()
            opt.step()
            tl += l.item()
            nb_ += 1
        sch.step()
        m.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(m._fwd_n(xvn), yvn).item()
        if vl < best:
            best = vl
            bst = {k: v.clone() for k, v in m.state_dict().items()}
            w = 0
        else:
            w += 1
        if e % 100 == 0 or e == 1:
            print(f"  ep{e:4d}  train={tl/nb_:.6e}  test={vl:.6e}  best={best:.6e}",
                  flush=True)
        if w >= pat:
            print(f"  early stop @{e} (no improvement for {pat} epochs)",
                  flush=True)
            break
    print(f"  training time: {time.time()-t0:.1f}s", flush=True)
    m.load_state_dict(bst)
    m.eval()
    with torch.no_grad():
        yp = m(xv)
        ae = (yp - yv).abs()
        re = ae / yv.abs().clamp(min=1e-8)
    r2 = []
    for j in range(N_MODES):
        sr = ((yv[:, j] - yp[:, j]) ** 2).sum().item()
        st = ((yv[:, j] - yv[:, j].mean()) ** 2).sum().item()
        r2.append(1 - sr / max(st, 1e-12))
    return m, {
        "mae_hz": float(ae.mean().item()),
        "mape_pct": float(re.mean().item() * 100),
        "max_rel_err_pct": float(re.max().item() * 100),
        "r2_mean": float(np.mean(r2)),
        "r2_min": float(np.min(r2)),
        "r2_per_mode": [float(x) for x in r2],
    }


def main():
    print(f"[load] {TRAIN_CSV}", flush=True)
    X, Y = load_data(TRAIN_CSV)
    print(f"  X={X.shape}  Y={Y.shape}", flush=True)
    print(f"  theta range per dim: "
          f"{[(float(X[:,i].min()), float(X[:,i].max())) for i in range(N_PARAMS)]}",
          flush=True)

    # 4000 train / 1000 test (deterministic)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    n_tr = 4000
    Xtr, Ytr = X[idx[:n_tr]], Y[idx[:n_tr]]
    Xte, Yte = X[idx[n_tr:]], Y[idx[n_tr:]]
    print(f"[split] train={Xtr.shape}  test={Xte.shape}", flush=True)

    print(f"\n=== Training single member (D={N_PARAMS}, M={N_MODES}) ===",
          flush=True)
    model, met = train(Xtr, Ytr, Xte, Yte, hid=256, nb=4, lr=1e-3, ep=2000,
                       bs=256, pat=200, seed=42)
    print(f"\n[metrics]", flush=True)
    print(f"  R2 mean = {met['r2_mean']:.6f}  (min={met['r2_min']:.6f})",
          flush=True)
    print(f"  MAE    = {met['mae_hz']:.4f} Hz", flush=True)
    print(f"  MAPE   = {met['mape_pct']:.4f}%", flush=True)
    print(f"  max rel err = {met['max_rel_err_pct']:.4f}%", flush=True)

    torch.save(model.state_dict(), OUT_DIR / "member_0.pt")
    print(f"\n[save] {OUT_DIR/'member_0.pt'}", flush=True)
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(met, f, indent=2)
    print(f"[save] {OUT_DIR/'metrics.json'}", flush=True)

    # Gradient sanity check (autograd vs finite-difference Jacobian)
    print("\n[grad-check] autograd vs finite-difference Jacobian", flush=True)
    eps = 1e-4
    for idx_s in [0, len(Xte) // 2, len(Xte) - 1]:
        x = torch.tensor(Xte[idx_s:idx_s + 1], dtype=torch.float32, device=DEV)
        x_g = x.clone().detach().requires_grad_(True)
        f_g = model(x_g)
        ja = np.zeros((N_MODES, N_PARAMS))
        for i in range(N_MODES):
            model.zero_grad()
            if x_g.grad is not None:
                x_g.grad.zero_()
            f_g[0, i].backward(retain_graph=True)
            ja[i] = x_g.grad[0].cpu().numpy()
        jf = np.zeros((N_MODES, N_PARAMS))
        for j in range(N_PARAMS):
            xp, xm_ = Xte[idx_s].copy(), Xte[idx_s].copy()
            xp[j] += eps
            xm_[j] -= eps
            with torch.no_grad():
                fp = model(torch.tensor(xp[None], dtype=torch.float32,
                                        device=DEV))[0].cpu().numpy()
                fm = model(torch.tensor(xm_[None], dtype=torch.float32,
                                        device=DEV))[0].cpu().numpy()
            jf[:, j] = (fp - fm) / (2 * eps)
        sc = np.maximum(np.abs(ja), np.abs(jf)).clip(1e-10)
        err = (np.abs(ja - jf) / sc).max()
        flag = "OK" if err < 1e-2 else "WARN"
        print(f"  sample {idx_s}: max rel Jacobian err = {err:.2e}  [{flag}]",
              flush=True)


if __name__ == "__main__":
    main()
