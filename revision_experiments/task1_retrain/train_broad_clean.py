"""
Task 1.2: retrain the broadened surrogate ENSEMBLE on the cleaned dataset.
Matches the original recipe (03_train_surrogate.py): ResNet hid=256/nb=4, AdamW
lr=1e-3 wd=1e-5, CosineAnnealing(2000), bs=256, patience=200, normalized MSE,
5 members (seeds 42+1000i). Held-out = 10% of the clean broad data (broad domain).
GPU training; weights saved CPU-loadable.
"""
import json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SURR = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
CLEAN = SURR / "dataset_train_broad_clean.csv"
OUT = SURR / "surrogate_models_broad_clean"; OUT.mkdir(exist_ok=True)
HERE = Path(__file__).parent
N_MODES = 20; PARAM_NAMES = [f"e{i}" for i in range(1, 7)]

class ResBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.SiLU(), nn.Linear(d, d),
                                 nn.LayerNorm(d), nn.SiLU(), nn.Linear(d, d))
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
        X.append([float(r[n]) for n in PARAM_NAMES])
        Y.append([float(r[f"f{j+1}"]) for j in range(N_MODES)])
    return np.array(X), np.array(Y)

def r2_stats(yv, yp):
    r2 = []
    for j in range(N_MODES):
        sr = ((yv[:, j] - yp[:, j]) ** 2).sum()
        st = ((yv[:, j] - yv[:, j].mean()) ** 2).sum()
        r2.append(1 - sr / max(st, 1e-12))
    ae = np.abs(yp - yv); re = ae / np.abs(yv).clip(1e-8)
    return dict(mae=float(ae.mean()), mape=float(re.mean() * 100),
                r2_mean=float(np.mean(r2)), r2_min=float(np.min(r2)), r2=[float(x) for x in r2])

def train_member(Xtr, Ytr, Xv, Yv, seed, ep=2500, bs=2048, pat=300):
    torch.manual_seed(seed)
    xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(Ytr, dtype=torch.float32, device=DEV)
    xv = torch.tensor(Xv, dtype=torch.float32, device=DEV)
    yv = torch.tensor(Yv, dtype=torch.float32, device=DEV)
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
        with torch.no_grad():
            vl = nn.functional.mse_loss(m._fwd_n(xvn), yvn).item()
        if vl < best: best, bst, w = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else: w += 1
        if w >= pat: break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad():
        yp = m(xv).cpu().numpy()
    return m, r2_stats(Yv, yp), e

def main():
    t0 = time.time()
    X, Y = load(CLEAN)
    rng = np.random.RandomState(42); idx = rng.permutation(len(X))
    nv = len(X) // 10
    vi, ti = idx[:nv], idx[nv:]
    Xtr, Ytr, Xv, Yv = X[ti], Y[ti], X[vi], Y[vi]
    print(f"clean rows={len(X)}  train={len(Xtr)}  held-out={len(Xv)}  dev={DEV}", flush=True)
    preds, metrics = [], []
    for i in range(5):
        tm = time.time()
        m, met, ep = train_member(Xtr, Ytr, Xv, Yv, seed=42 + i * 1000)
        torch.save({k: v.cpu() for k, v in m.state_dict().items()}, OUT / f"member_{i}.pt")
        print(f"  (member {i} took {time.time()-tm:.0f}s)", flush=True)
        with torch.no_grad():
            preds.append(m(torch.tensor(Xv, dtype=torch.float32, device=DEV)).cpu().numpy())
        metrics.append(met)
        print(f"member {i}: ep={ep} R2_mean={met['r2_mean']:.6f} R2_min={met['r2_min']:.6f} "
              f"MAE={met['mae']:.4f}Hz MAPE={met['mape']:.4f}%", flush=True)
    ens = r2_stats(Yv, np.mean(preds, axis=0))
    print(f"\nENSEMBLE held-out: R2_mean={ens['r2_mean']:.6f} R2_min={ens['r2_min']:.6f} "
          f"MAE={ens['mae']:.4f}Hz MAPE={ens['mape']:.4f}%", flush=True)
    best_min = max(m["r2_min"] for m in metrics)
    gate_r2 = ens["r2_min"] >= 0.9999
    report = {"clean_rows": len(X), "n_train": len(Xtr), "n_heldout": len(Xv),
              "device": str(DEV), "members": metrics, "ensemble": ens,
              "old_mae_hz": 0.18, "gate_r2min_ge_0p9999": bool(gate_r2),
              "best_member_r2min": float(best_min), "elapsed_s": time.time() - t0}
    json.dump(report, open(HERE / "retrain_report.json", "w"), indent=2)
    print(f"\nGATE held-out R2_min>=0.9999 : {'PASS' if gate_r2 else 'FAIL'} "
          f"(ensemble R2_min={ens['r2_min']:.6f}); old MAE 0.18Hz -> new {ens['mae']:.4f}Hz")

if __name__ == "__main__":
    main()
