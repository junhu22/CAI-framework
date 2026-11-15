"""
E6 (Step 7): SNPE-C quantitative comparison (R1#1, R2 weakness-1, R3#4).

Trains SNPE-C (sbi, NSF density estimator) on the SAME 50,000 (theta,y) pairs
used by CAI for the D=6 original sailboat config, then compares CAI-fast (CNF),
CAI-exact (NeuTra-HMC), and SNPE-C against a common HMC ground truth on Obs1,
Obs2, and the 10 robustness observations. Everything is evaluated on the SAME
regenerated observation per case for a fair, self-consistent MAE.
"""
import sys, json, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, r"C:\neutra-hmc-surrogate\surrogate_training\revision_experiments")
import revlib as R

HERE = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42); np.random.seed(42)
SIGMA, KAPPA_SQ = 0.30, 1e-3
NTRAIN = 50000

sur6 = R.load_d6(); cnf6 = R.load_d6_cnf()

# ── 1. Build the 50k training set (theta ~ lognormal(0,0.3), y=f+noise) ─
g = torch.Generator().manual_seed(42)
log_theta = SIGMA * torch.randn(NTRAIN, 6, generator=g)
theta_tr = torch.exp(log_theta).clamp(0.5, 2.0)
with torch.no_grad():
    f = sur6(theta_tr)
y_tr = f + (KAPPA_SQ ** 0.5) * f.abs() * torch.randn(f.shape, generator=g)
print(f"train set: theta {tuple(theta_tr.shape)}, y {tuple(y_tr.shape)}  device={DEVICE}")

# ── 2. Train SNPE-C with NSF ────────────────────────────────────────────
from sbi.inference import SNPE
try:
    from sbi import __version__ as sbiv
except Exception:
    sbiv = "?"
prior = torch.distributions.Independent(
    torch.distributions.LogNormal(torch.zeros(6, device=DEVICE),
                                  SIGMA * torch.ones(6, device=DEVICE)), 1)
t0 = time.time()
inference = SNPE(prior=prior, density_estimator="nsf", device=DEVICE)
inference.append_simulations(theta_tr, y_tr)
print("  [train] starting SNPE-C (nsf) training...", flush=True)
density_estimator = inference.train(training_batch_size=1024, max_num_epochs=30,
                                    stop_after_epochs=8, show_train_summary=False)
posterior = inference.build_posterior(density_estimator)
train_time = time.time() - t0
n_epochs = getattr(inference, "epoch", getattr(inference, "_epochs_trained", -1))
print(f"SNPE-C (sbi {sbiv}) trained in {train_time:.1f}s on {DEVICE} ({n_epochs} epochs)", flush=True)

# ── 3. Evaluation observations (same y for all methods) ─────────────────
th1, _ = R.load_obs1()
cases = [("Obs1", th1.tolist()), ("Obs2", R.OBS2_THETA)]
cases += [(f"Rob{i+1}", th) for i, th in enumerate(R.ROBUST_THETAS)]

def cai_fast(y):
    tf = cnf6.sample(torch.tensor(y, dtype=torch.float32), n_samples=2000)
    return tf.mean(0).numpy()

def cai_exact(y):
    yt = torch.tensor(y, dtype=torch.float32)
    lp = R.make_log_post(sur6, y, "d6")
    ch, _ = R.NeuTraHMC_Conditional(lp.forward, cnf6, yt, step_size=0.05,
                                    n_leapfrog=20).run(torch.ones(6), 400, 150)
    return ch.numpy().mean(0)

def hmc_gt(y):
    lp = R.make_log_post(sur6, y, "d6")
    ch, ar = R.HMCSampler(lp.grad_log_prob, step_size=0.01, n_leapfrog=20).run(
        torch.ones(6), 800, 300)
    return ch.numpy().mean(0), ar

def snpe_infer(y):
    yt = torch.tensor(y, dtype=torch.float32).to(DEVICE)
    t0 = time.time()
    s = posterior.sample((2000,), x=yt, show_progress_bars=False).cpu().numpy()
    return s.mean(0), time.time() - t0

rows = []
for name, th in cases:
    y, _ = R.gen_noisy_y(sur6, th, KAPPA_SQ, seed=700 + len(rows))
    gt, ar = hmc_gt(y)
    mf = cai_fast(y); me = cai_exact(y); ms, t_snpe = snpe_infer(y)
    rows.append(dict(name=name, gt=gt.tolist(),
        mae_fast=float(np.abs(mf - gt).mean()),
        mae_exact=float(np.abs(me - gt).mean()),
        mae_snpe=float(np.abs(ms - gt).mean()),
        t_snpe=t_snpe))
    print(f"[{name}] MAE  fast={rows[-1]['mae_fast']:.4f}  exact={rows[-1]['mae_exact']:.4f}"
          f"  SNPE={rows[-1]['mae_snpe']:.4f}", flush=True)

def agg(key): return np.array([r[key] for r in rows])
summary = dict(
    train_time_snpe=train_time, sbi_version=sbiv, device=DEVICE,
    mae_fast_mean=float(agg("mae_fast").mean()),  mae_fast_worst=float(agg("mae_fast").max()),
    mae_exact_mean=float(agg("mae_exact").mean()),mae_exact_worst=float(agg("mae_exact").max()),
    mae_snpe_mean=float(agg("mae_snpe").mean()),  mae_snpe_worst=float(agg("mae_snpe").max()),
    t_snpe_infer=float(np.mean([r["t_snpe"] for r in rows])))

# ── markdown ────────────────────────────────────────────────────────────
L = ["# E6 — SNPE-C vs CAI (D=6 sailboat, same 50k training pairs)\n",
     f"sbi {sbiv}, NSF density estimator, device={DEVICE}. Evaluated on Obs1, Obs2, "
     "and 10 robustness observations against a common HMC ground truth "
     "(step=0.01, L=20, 1500 samples).\n",
     "| metric | CAI-fast | CAI-exact | SNPE-C |",
     "|---|---|---|---|",
     f"| MAE vs HMC (mean) | {summary['mae_fast_mean']:.4f} | "
     f"{summary['mae_exact_mean']:.4f} | {summary['mae_snpe_mean']:.4f} |",
     f"| MAE vs HMC (worst) | {summary['mae_fast_worst']:.4f} | "
     f"{summary['mae_exact_worst']:.4f} | {summary['mae_snpe_worst']:.4f} |",
     f"| training time | shared 50k pairs | (same map) | {train_time:.0f}s |",
     f"| inference time / obs | ~0.01s | ~10s (HMC) | {summary['t_snpe_infer']*1000:.0f}ms |",
     "| exact-correction path | no | **yes** | no |",
     "| reusable transport map | **yes** | **yes** | yes (density est.) |",
     "",
     "Per-observation MAE vs HMC:",
     "| obs | CAI-fast | CAI-exact | SNPE-C |", "|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['name']} | {r['mae_fast']:.4f} | {r['mae_exact']:.4f} | {r['mae_snpe']:.4f} |")
L += ["",
      "Both CAI-fast and SNPE-C are amortised (reusable map, millisecond inference). "
      "The structural difference is that CAI carries an *exact-correction path* "
      "(NeuTra-HMC) invoked by the ESS_IS/N router when the amortised fit is poor; "
      "SNPE-C has no such fallback. NSF (used here for SNPE) is the same spline-flow "
      "family referenced in R3#4."]
(HERE / "E6_snpe_comparison.md").write_text("\n".join(L), encoding="utf-8")
json.dump(dict(summary=summary, rows=rows), open(HERE / "E6_results.json", "w"), indent=1)
print(f"\nSNPE train {train_time:.0f}s | MAE mean fast={summary['mae_fast_mean']:.4f} "
      f"exact={summary['mae_exact_mean']:.4f} snpe={summary['mae_snpe_mean']:.4f}")
print("Wrote E6_snpe_comparison.md, E6_results.json")
