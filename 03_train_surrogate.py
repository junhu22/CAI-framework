#!/usr/bin/env python3
"""Step 3: Train MLP surrogate — θ(6) → f(20). Ensemble + gradient check."""
import json, time, argparse, csv as csv_mod
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from config import *

DEV = torch.device("cpu")

class ResBlock(nn.Module):
    def __init__(self, d, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d), nn.SiLU(), nn.Linear(d, d),
            nn.LayerNorm(d), nn.SiLU(), nn.Dropout(drop), nn.Linear(d, d))
    def forward(self, x): return x + self.net(x)

class Surrogate(nn.Module):
    def __init__(self, nin=N_PARAMS, nout=N_MODES, hid=256, nb=4):
        super().__init__()
        self.nin, self.nout = nin, nout
        self.inp = nn.Sequential(nn.Linear(nin, hid), nn.SiLU())
        self.blk = nn.ModuleList([ResBlock(hid) for _ in range(nb)])
        self.out = nn.Sequential(nn.LayerNorm(hid), nn.SiLU(), nn.Linear(hid, nout))
        for n in ("xm","xs","ym","ys"):
            self.register_buffer(n, torch.zeros(nin if "x" in n else nout))

    def set_norm(self, X, Y):
        self.xm, self.xs = X.mean(0), X.std(0).clamp(min=1e-8)
        self.ym, self.ys = Y.mean(0), Y.std(0).clamp(min=1e-8)

    def _fwd_n(self, xn):
        h = self.inp(xn)
        for b in self.blk: h = b(h)
        return self.out(h)

    def forward(self, x):
        return self._fwd_n((x - self.xm)/self.xs) * self.ys + self.ym

    def predict_with_grad(self, x):
        x = x.detach().requires_grad_(True)
        f = self.forward(x)
        jac = [torch.autograd.grad(f[:,i].sum(), x, create_graph=True)[0]
               for i in range(self.nout)]
        return f, torch.stack(jac, 1)

def load_data(p):
    X, Y = [], []
    with open(p) as f:
        for r in csv_mod.DictReader(f):
            X.append([float(r[n]) for n in PARAM_NAMES])
            Y.append([float(r[f"f{j+1}"]) for j in range(N_MODES)])
    return np.array(X), np.array(Y)

def train(Xtr, Ytr, Xte, Yte, hid=256, nb=4, lr=1e-3, ep=2000, bs=256, pat=200, seed=42):
    torch.manual_seed(seed)
    xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(Ytr, dtype=torch.float32, device=DEV)
    xv = torch.tensor(Xte, dtype=torch.float32, device=DEV)
    yv = torch.tensor(Yte, dtype=torch.float32, device=DEV)
    m = Surrogate(hid=hid, nb=nb).to(DEV); m.set_norm(xt, yt)
    xtn = (xt-m.xm)/m.xs; ytn = (yt-m.ym)/m.ys
    xvn = (xv-m.xm)/m.xs; yvn = (yv-m.ym)/m.ys
    ld = DataLoader(TensorDataset(xtn, ytn), batch_size=bs, shuffle=True)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    best, bst, w = 1e9, None, 0
    for e in range(1, ep+1):
        m.train(); tl=0; nb_=0
        for xb,yb in ld:
            l = nn.functional.mse_loss(m._fwd_n(xb), yb)
            opt.zero_grad(); l.backward(); opt.step(); tl+=l.item(); nb_+=1
        sch.step(); m.eval()
        with torch.no_grad(): vl = nn.functional.mse_loss(m._fwd_n(xvn), yvn).item()
        if vl < best: best=vl; bst={k:v.clone() for k,v in m.state_dict().items()}; w=0
        else: w+=1
        if e%200==0: print(f"  ep{e:4d} train={tl/nb_:.6f} test={vl:.6f}")
        if w>=pat: print(f"  early stop @{e}"); break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad():
        yp = m(xv); ae=(yp-yv).abs(); re=ae/yv.abs().clamp(min=1e-8)
    r2 = []
    for j in range(N_MODES):
        sr = ((yv[:,j]-yp[:,j])**2).sum().item()
        st = ((yv[:,j]-yv[:,j].mean())**2).sum().item()
        r2.append(1-sr/max(st,1e-12))
    return m, {"mae":ae.mean().item(), "mape":re.mean().item()*100,
               "r2_mean":np.mean(r2), "r2_min":np.min(r2), "r2":r2}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--n_ensemble", type=int, default=5)
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--nb", type=int, default=4)
    ap.add_argument("--out", default="surrogate_models")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    Xtr, Ytr = load_data(args.train)
    Xte, Yte = load_data(args.test)
    print(f"Train: {Xtr.shape}  Test: {Xte.shape}  Device: {DEV}")

    models, metrics = [], []
    for i in range(args.n_ensemble):
        print(f"\n=== Ensemble {i+1}/{args.n_ensemble} ===")
        m, met = train(Xtr, Ytr, Xte, Yte, hid=args.hid, nb=args.nb, seed=42+i*1000)
        print(f"  R²={met['r2_mean']:.6f} (min={met['r2_min']:.6f}) MAE={met['mae']:.4f}Hz MAPE={met['mape']:.4f}%")
        torch.save(m.state_dict(), out / f"member_{i}.pt")
        models.append(m); metrics.append(met)

    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Gradient check
    best_i = int(np.argmax([met["r2_mean"] for met in metrics]))
    bm = models[best_i]; bm.eval()
    print("\n--- Gradient check ---")
    eps = 1e-4
    check_indices = [0, len(Xte)//2, len(Xte)-1]  # FIX: -1 caused empty slice
    for idx in check_indices:
        x = torch.tensor(Xte[idx:idx+1], dtype=torch.float32, device=DEV)
        # Autograd Jacobian
        x_g = x.clone().detach().requires_grad_(True)
        f_g = bm(x_g)
        ja = np.zeros((N_MODES, N_PARAMS))
        for i in range(N_MODES):
            bm.zero_grad()
            if x_g.grad is not None:
                x_g.grad.zero_()
            f_g[0, i].backward(retain_graph=True)
            ja[i] = x_g.grad[0].cpu().numpy()
        # Finite difference Jacobian
        jf = np.zeros((N_MODES, N_PARAMS))
        for j in range(N_PARAMS):
            xp, xm_ = Xte[idx].copy(), Xte[idx].copy()
            xp[j]+=eps; xm_[j]-=eps
            with torch.no_grad():
                fp = bm(torch.tensor(xp[None],dtype=torch.float32,device=DEV))[0].cpu().numpy()
                fm = bm(torch.tensor(xm_[None],dtype=torch.float32,device=DEV))[0].cpu().numpy()
            jf[:,j] = (fp-fm)/(2*eps)
        sc = np.maximum(np.abs(ja), np.abs(jf)).clip(1e-10)
        err = (np.abs(ja-jf)/sc).max()
        print(f"  sample {idx}: max rel Jacobian err = {err:.2e} {'✓' if err<1e-2 else '✗'}")
        if err >= 1e-2:
            print(f"    autograd norm={np.linalg.norm(ja):.4e}, FD norm={np.linalg.norm(jf):.4e}")

    best = max(metrics, key=lambda m: m["r2_mean"])
    print(f"\n--- Quality gate ---")
    print(f"  R² mean: {best['r2_mean']:.6f} {'✓' if best['r2_mean']>0.999 else '✗'}")
    print(f"  MAPE:    {best['mape']:.4f}%  {'✓' if best['mape']<0.1 else '✗'}")

if __name__ == "__main__":
    main()
