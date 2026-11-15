"""
Task 4.3: was the E6 SNPE 30-epoch run a truncation? Train SNPE-C (NSF) to natural
early-stopping (patience 20, no low epoch cap), log the val-loss curve, and re-evaluate
the same 12 observations vs common HMC ground truth. Compare MAE to the truncated result
(fast 0.0234 / exact 0.0223 / snpe 0.0355).
"""
import sys, json, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, r"C:\neutra-hmc-surrogate\surrogate_training\revision_experiments")
import revlib as R
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sbi.inference import SNPE

HERE = Path(__file__).parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42); np.random.seed(42)
SIG, KSQ, N = 0.30, 1e-3, 50000
sur6 = R.load_d6(); cnf6 = R.load_d6_cnf()

g = torch.Generator().manual_seed(42)
theta_tr = torch.exp(SIG * torch.randn(N, 6, generator=g)).clamp(0.5, 2.0)
with torch.no_grad(): f = sur6(theta_tr)
y_tr = f + (KSQ ** 0.5) * f.abs() * torch.randn(f.shape, generator=g)

prior = torch.distributions.Independent(
    torch.distributions.LogNormal(torch.zeros(6, device=DEV), SIG * torch.ones(6, device=DEV)), 1)
t0 = time.time()
inf = SNPE(prior=prior, density_estimator="nsf", device=DEV)
inf.append_simulations(theta_tr, y_tr)
de = inf.train(training_batch_size=1024, max_num_epochs=400, stop_after_epochs=20,
               show_train_summary=True)
post = inf.build_posterior(de); ttrain = time.time() - t0
# val-loss curve
try:
    curve = [float(x) for x in inf.summary["validation_loss"]]
except Exception:
    curve = [float(x) for x in getattr(inf, "_summary", {}).get("validation_loss", [])]
converged = len(curve) < 400 - 1     # early-stopped => converged
print(f"SNPE converged in {len(curve)} epochs ({ttrain:.0f}s); early_stop={converged}", flush=True)

if curve:
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(range(1, len(curve) + 1), curve, lw=1.2)
    ax.axvline(30, color="r", ls=":", label="round-1 cap (30)")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation loss"); ax.legend(fontsize=8)
    ax.set_title(f"E6 SNPE-C val loss ({'converged @'+str(len(curve)) if converged else 'ran full'})")
    fig.tight_layout(); fig.savefig(HERE / "e6_val_curve.png", dpi=150)

# re-eval same 12 obs
th1, _ = R.load_obs1()
cases = [("Obs1", th1.tolist()), ("Obs2", R.OBS2_THETA)] + [(f"Rob{i+1}", t) for i, t in enumerate(R.ROBUST_THETAS)]
def hmc_gt(y):
    lp = R.make_log_post(sur6, y, "d6")
    return R.HMCSampler(lp.grad_log_prob, 0.01, 20).run(torch.ones(6), 800, 300)[0].numpy().mean(0)
rows = []
for i, (nm, th) in enumerate(cases):
    y, _ = R.gen_noisy_y(sur6, th, KSQ, seed=700 + i)
    gt = hmc_gt(y)
    mf = cnf6.sample(torch.tensor(y, dtype=torch.float32), 2000).numpy().mean(0)
    ms = post.sample((2000,), x=torch.tensor(y, dtype=torch.float32).to(DEV), show_progress_bars=False).cpu().numpy().mean(0)
    rows.append(dict(name=nm, mae_fast=float(np.abs(mf - gt).mean()), mae_snpe=float(np.abs(ms - gt).mean())))
    print(f"[{nm}] fast={rows[-1]['mae_fast']:.4f} snpe={rows[-1]['mae_snpe']:.4f}", flush=True)

msnpe = float(np.mean([r["mae_snpe"] for r in rows]))
mfast = float(np.mean([r["mae_fast"] for r in rows]))
out = {"epochs_to_converge": len(curve), "early_stopped": bool(converged), "train_time_s": ttrain,
       "mae_snpe_converged_mean": msnpe, "mae_fast_mean": mfast,
       "round1_truncated": {"snpe": 0.0355, "fast": 0.0234, "epochs": 30}, "rows": rows}
json.dump(out, open(HERE / "e6_convergence.json", "w"), indent=2)
print(f"\nCONVERGED SNPE mae={msnpe:.4f} (round-1 truncated 0.0355); CAI-fast {mfast:.4f}", flush=True)
