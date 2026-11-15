"""
Task 1.4: retrain the broadened conditional RealNVP against the CLEAN member_0
surrogate. Matches the original recipe (retrain_surrogate_broad.py stage 3):
8 layers / 128 hidden, y-embed 20->128->20 (cai_lib arch), Adam lr=1e-3 wd=1e-5,
CosineAnnealing(500), batch 512, max-likelihood loss, patience 50; theta~lognormal(0,0.55)
clamp[0.5,3.0], kappa=sqrt(1e-3). GPU. Records the val-loss curve.
"""
import sys, json, math, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, r"C:\neutra-hmc-surrogate\surrogate_training\revision_experiments")
import revlib as R

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SURR = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
CLEAN_SURR = SURR / "surrogate_models_broad_clean" / "member_0.pt"
OUT = SURR / "experiments" / "results" / "conditional_flow_broad_clean.pt"
HERE = Path(__file__).parent
torch.manual_seed(42)

# clean member_0 (post-remap, cai_lib-loadable), on GPU
sur = R.Surrogate(nin=6, nout=20)
sur.load_state_dict(torch.load(CLEAN_SURR, map_location="cpu", weights_only=True))
sur.eval().to(DEV)

N_TRAIN, N_VAL = 50000, 5000
SIGMA, KAPPA = 0.55, math.sqrt(1e-3)
th, ys = [], []
with torch.no_grad():
    for s in range(0, N_TRAIN + N_VAL, 2000):
        B = min(2000, N_TRAIN + N_VAL - s)
        theta = torch.exp(SIGMA * torch.randn(B, 6, device=DEV)).clamp(0.5, 3.0)
        f = sur(theta)
        y = f + KAPPA * f * torch.randn_like(f)
        th.append(theta); ys.append(y)
theta_all, y_all = torch.cat(th), torch.cat(ys)
th_tr, th_va = theta_all[:N_TRAIN], theta_all[N_TRAIN:]
y_tr, y_va = y_all[:N_TRAIN], y_all[N_TRAIN:]
print(f"pairs: train {th_tr.shape} val {th_va.shape}  theta[{theta_all.min():.2f},{theta_all.max():.2f}]  dev={DEV}", flush=True)

flow = R.ConditionalRealNVP(dim_theta=6, dim_y=20, n_layers=8, hidden_dim=128).to(DEV)
opt = torch.optim.Adam(flow.parameters(), lr=1e-3, weight_decay=1e-5)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 500)
ds = torch.utils.data.TensorDataset(th_tr, y_tr)
dl = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True, drop_last=True)
best, best_state, no_imp, curve = float("inf"), None, 0, []
t0 = time.time()
for ep in range(1, 501):
    flow.train()
    for thb, yb in dl:
        z, ld = flow(thb, yb)
        loss = -(-0.5 * (z ** 2).sum(-1) + ld).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0); opt.step()
    sch.step(); flow.eval()
    with torch.no_grad():
        z, ld = flow(th_va, y_va)
        vloss = -(-0.5 * (z ** 2).sum(-1) + ld).mean().item()
    curve.append(vloss)
    if vloss < best: best, best_state, no_imp = vloss, {k: v.clone() for k, v in flow.state_dict().items()}, 0
    else: no_imp += 1
    if ep % 50 == 0 or ep == 1: print(f"  epoch {ep:4d}: val={vloss:.4f} best={best:.4f}", flush=True)
    if no_imp >= 50: print(f"  early stop @ {ep}", flush=True); break
flow.load_state_dict(best_state)
torch.save({k: v.cpu() for k, v in flow.state_dict().items()}, OUT)
json.dump({"best_val_loss": best, "epochs_ran": len(curve), "val_curve": curve,
           "elapsed_s": time.time() - t0, "out": str(OUT)},
          open(HERE / "cnf_retrain_report.json", "w"), indent=2)
print(f"\nCNF retrained: best val loss={best:.4f} in {len(curve)} epochs -> {OUT.name}", flush=True)
