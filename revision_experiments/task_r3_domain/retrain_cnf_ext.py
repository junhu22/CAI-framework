"""
R3-T1.3: retrain the broad CNF against the domain-EXTENDED member_0 surrogate.
theta ~ lognormal(0,0.55) clamp [0.5,3.5] (extended); kappa=sqrt(1e-3). Same arch/recipe.
"""
import sys, json, math, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, r"C:\neutra-hmc-surrogate\surrogate_training\revision_experiments")
import revlib as R

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SURR = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
OUT = SURR / "experiments" / "results" / "conditional_flow_broad_ext.pt"
HERE = Path(__file__).parent
torch.manual_seed(42)

sur = R.load_broad_ext().to(DEV).eval()   # cai_lib-compatible ext member_0
N_TRAIN, N_VAL = 50000, 5000
SIGMA, KAPPA = 0.55, math.sqrt(1e-3)
th, ys = [], []
with torch.no_grad():
    for s in range(0, N_TRAIN + N_VAL, 2000):
        B = min(2000, N_TRAIN + N_VAL - s)
        theta = torch.exp(SIGMA * torch.randn(B, 6, device=DEV)).clamp(0.5, 3.5)  # extended
        f = sur(theta); ys.append(f + KAPPA * f * torch.randn_like(f)); th.append(theta)
theta_all, y_all = torch.cat(th), torch.cat(ys)
th_tr, y_tr = theta_all[:N_TRAIN], y_all[:N_TRAIN]
th_va, y_va = theta_all[N_TRAIN:], y_all[N_TRAIN:]
print(f"pairs train {th_tr.shape} theta[{theta_all.min():.2f},{theta_all.max():.2f}] dev={DEV}", flush=True)

flow = R.ConditionalRealNVP(dim_theta=6, dim_y=20, n_layers=8, hidden_dim=128).to(DEV)
opt = torch.optim.Adam(flow.parameters(), lr=1e-3, weight_decay=1e-5)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 500)
dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(th_tr, y_tr),
                                 batch_size=512, shuffle=True, drop_last=True)
best, best_state, no_imp, curve = 1e9, None, 0, []
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
        z, ld = flow(th_va, y_va); vloss = -(-0.5 * (z ** 2).sum(-1) + ld).mean().item()
    curve.append(vloss)
    if vloss < best: best, best_state, no_imp = vloss, {k: v.clone() for k, v in flow.state_dict().items()}, 0
    else: no_imp += 1
    if ep % 50 == 0: print(f"  epoch {ep}: val={vloss:.4f} best={best:.4f}", flush=True)
    if no_imp >= 50: print(f"  early stop @ {ep}", flush=True); break
flow.load_state_dict(best_state)
torch.save({k: v.cpu() for k, v in flow.state_dict().items()}, OUT)
json.dump({"best_val_loss": best, "epochs": len(curve), "elapsed_s": time.time() - t0},
          open(HERE / "cnf_ext_report.json", "w"), indent=2)
print(f"\nCNF-ext best val={best:.4f} in {len(curve)} epochs -> {OUT.name}", flush=True)
