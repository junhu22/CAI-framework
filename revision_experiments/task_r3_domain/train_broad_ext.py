"""
R3-T1.3: retrain the broad surrogate ensemble on the DOMAIN-EXTENDED clean dataset
([0.5,3.5]). Same recipe as the round-2 clean retrain, but the ResBlock now includes the
identity Dropout(0.0) so keys match cai_lib (net.6) directly — no remap needed.
Gate: held-out (incl. extended region) R2_min >= 0.9999.
"""
import json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SURR = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
DATA = SURR / "dataset_train_broad_ext.csv"
OUT = SURR / "surrogate_models_broad_ext"; OUT.mkdir(exist_ok=True)
HERE = Path(__file__).parent
N_MODES = 20; PN = [f"e{i}" for i in range(1, 7)]

class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.SiLU(), nn.Linear(d, d),
                                 nn.LayerNorm(d), nn.SiLU(), nn.Dropout(0.0), nn.Linear(d, d))
    def forward(self, x): return x + self.net(x)

class Surrogate(nn.Module):
    def __init__(self, nin=6, nout=20, hid=256, nb=4):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nin, hid), nn.SiLU())
        self.blk = nn.ModuleList([ResBlock(hid) for _ in range(nb)])
        self.out = nn.Sequential(nn.LayerNorm(hid), nn.SiLU(), nn.Linear(hid, nout))
        for n in ("xm", "xs", "ym", "ys"):
            self.register_buffer(n, torch.zeros(nin if "x" in n else nout))
    def set_norm(self, X, Y):
        self.xm, self.xs = X.mean(0), X.std(0).clamp(min=1e-8)
        self.ym, self.ys = Y.mean(0), Y.std(0).clamp(min=1e-8)
    def _fwd_n(self, xn):
        h = self.inp(xn)
        for b in self.blk: h = b(h)
        return self.out(h)
    def forward(self, x): return self._fwd_n((x - self.xm) / self.xs) * self.ys + self.ym

def load(p):
    import csv
    X, Y = [], []
    for r in csv.DictReader(open(p)):
        X.append([float(r[n]) for n in PN]); Y.append([float(r[f"f{j+1}"]) for j in range(N_MODES)])
    return np.array(X), np.array(Y)

def r2_stats(yv, yp):
    r2 = [1 - ((yv[:, j] - yp[:, j]) ** 2).sum() / max(((yv[:, j] - yv[:, j].mean()) ** 2).sum(), 1e-12)
          for j in range(N_MODES)]
    ae = np.abs(yp - yv); re = ae / np.abs(yv).clip(1e-8)
    return dict(mae=float(ae.mean()), mape=float(re.mean() * 100),
                r2_mean=float(np.mean(r2)), r2_min=float(np.min(r2)))

def train_member(Xtr, Ytr, Xv, Yv, seed, ep=2500, bs=2048, pat=300):
    torch.manual_seed(seed)
    xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV); yt = torch.tensor(Ytr, dtype=torch.float32, device=DEV)
    xv = torch.tensor(Xv, dtype=torch.float32, device=DEV); yv = torch.tensor(Yv, dtype=torch.float32, device=DEV)
    m = Surrogate().to(DEV); m.set_norm(xt, yt)
    xtn, ytn = (xt - m.xm) / m.xs, (yt - m.ym) / m.ys
    xvn, yvn = (xv - m.xm) / m.xs, (yv - m.ym) / m.ys
    ld = DataLoader(TensorDataset(xtn, ytn), batch_size=bs, shuffle=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep)
    best, bst, w = 1e9, None, 0
    for e in range(1, ep + 1):
        m.train()
        for xb, yb in ld:
            l = nn.functional.mse_loss(m._fwd_n(xb), yb)
            opt.zero_grad(); l.backward(); opt.step()
        sch.step(); m.eval()
        with torch.no_grad(): vl = nn.functional.mse_loss(m._fwd_n(xvn), yvn).item()
        if vl < best: best, bst, w = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else: w += 1
        if w >= pat: break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad(): yp = m(xv).cpu().numpy()
    return m, r2_stats(Yv, yp), e

def main():
    t0 = time.time(); X, Y = load(DATA)
    rng = np.random.RandomState(42); idx = rng.permutation(len(X)); nv = len(X) // 10
    Xtr, Ytr, Xv, Yv = X[idx[nv:]], Y[idx[nv:]], X[idx[:nv]], Y[idx[:nv]]
    # ensure some held-out points are in the extended region
    n_ext_val = int((Xv.max(1) > 3.0).sum())
    print(f"ext rows={len(X)} train={len(Xtr)} val={len(Xv)} (val in [3.0,3.5]: {n_ext_val}) dev={DEV}", flush=True)
    preds, metrics = [], []
    for i in range(5):
        tm = time.time(); m, met, ep = train_member(Xtr, Ytr, Xv, Yv, seed=42 + i * 1000)
        torch.save({k: v.cpu() for k, v in m.state_dict().items()}, OUT / f"member_{i}.pt")
        with torch.no_grad(): preds.append(m(torch.tensor(Xv, dtype=torch.float32, device=DEV)).cpu().numpy())
        metrics.append(met)
        print(f"member {i}: ep={ep} R2_min={met['r2_min']:.6f} MAE={met['mae']:.4f}Hz ({time.time()-tm:.0f}s)", flush=True)
    ens = r2_stats(Yv, np.mean(preds, axis=0))
    gate = ens["r2_min"] >= 0.9999
    json.dump({"ext_rows": len(X), "members": metrics, "ensemble": ens, "gate_pass": bool(gate),
               "n_val_extended": n_ext_val, "elapsed_s": time.time() - t0},
              open(HERE / "train_ext_report.json", "w"), indent=2)
    json.dump(metrics, open(OUT / "metrics.json", "w"), indent=2)
    print(f"\nENSEMBLE held-out R2_min={ens['r2_min']:.6f} MAE={ens['mae']:.4f}Hz  GATE {'PASS' if gate else 'FAIL'}", flush=True)

if __name__ == "__main__":
    main()
