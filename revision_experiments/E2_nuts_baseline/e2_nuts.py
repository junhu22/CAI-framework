"""
E2 (Step 4): well-tuned NUTS baseline (R1#8).

Wraps each surrogate log-posterior as a pyro potential_fn in unconstrained
log-theta space (theta=exp(z); N(0,sigma) prior on z == lognormal prior on theta),
runs pyro NUTS with dual-averaging warmup + diagonal mass adaptation
(warmup=500, target_accept=0.8, 2000 samples).

Targets: D=6 Obs1, D=12 (Near nominal / Mixed / Heavy), field data.
Reports wall-clock, ESS_min, ESS/s, MAE vs truth, mean accept prob.
"""
import sys, json, time
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(4)
sys.path.insert(0, r"C:\neutra-hmc-surrogate\surrogate_training\revision_experiments")
import revlib as R
from pyro.infer import MCMC, NUTS

HERE = Path(__file__).resolve().parent
torch.manual_seed(42)

def run_nuts(potential_fn, D, dtype=torch.float32, warmup=500, nsamp=2000):
    init = {"z": torch.zeros(D, dtype=dtype)}
    kernel = NUTS(potential_fn=potential_fn, target_accept_prob=0.8,
                  full_mass=False, jit_compile=False)
    mcmc = MCMC(kernel, num_samples=nsamp, warmup_steps=warmup,
                initial_params=init, disable_progbar=True)
    t0 = time.time(); mcmc.run(); dt = time.time() - t0
    z = mcmc.get_samples()["z"]
    theta = torch.exp(z)
    ess = R.effective_sample_size(theta).numpy()
    # mean accept prob over sampling phase
    try:
        ap = float(np.mean(mcmc.diagnostics()["acceptance rate"]["z"]))
    except Exception:
        ap = float("nan")
    return theta.detach().numpy(), ess, dt, ap

def d6_potential(y, sur, kappa_sq, sigma):
    yt = torch.tensor(y, dtype=torch.float32); K = len(y)
    def U(params):
        z = params["z"]; theta = torch.exp(z)
        f = sur(theta.unsqueeze(0)).squeeze(0)[:K]
        rel = (f - yt) / yt
        ll = -0.5 * (rel ** 2).sum() / kappa_sq
        lp = -0.5 * (z ** 2).sum() / sigma ** 2
        return -(ll + lp)
    return U

def d12_potential(y, nn_sur, kappa, sigma):
    yt = y.double()
    def U(params):
        z = params["z"]; theta = torch.exp(z)
        f = nn_sur(theta.unsqueeze(0)).squeeze(0)
        rel = (f - yt) / yt
        ll = -0.5 * (rel ** 2).sum() / kappa ** 2
        lp = -0.5 * (z ** 2).sum() / sigma ** 2
        return -(ll + lp)
    return U

results = {}

# ── D=6 Obs1 ────────────────────────────────────────────────────────────
sur6 = R.load_d6()
th1, y1 = R.load_obs1()
theta, ess, dt, ap = run_nuts(d6_potential(y1, sur6, 1e-3, 0.3), 6)
mae = float(np.abs(theta.mean(0) - th1).mean())
results["D6_Obs1"] = dict(D=6, ess_min=float(ess.min()), ess_per_s=float(ess.min()/dt),
    time_s=dt, accept=ap, mae_vs_truth=mae, post_mean=theta.mean(0).tolist(),
    truth=th1.tolist())
print(f"[D6 Obs1] t={dt:.1f}s ESS_min={ess.min():.0f} ESS/s={ess.min()/dt:.2f} "
      f"AR={ap:.2f} MAE={mae:.4f}")

# ── D=12 three scenarios ────────────────────────────────────────────────
nn12, flow12, _ = R.load_d12()
for name, th in R.D12_SCENARIOS.items():
    tht = torch.tensor(th, dtype=torch.float64)
    with torch.no_grad():
        yc = nn12(tht.unsqueeze(0)).squeeze(0)
    g = torch.Generator().manual_seed(500)
    y12 = yc + torch.randn(yc.shape, generator=g, dtype=torch.float64) * 0.03 * yc.abs()
    theta, ess, dt, ap = run_nuts(d12_potential(y12, nn12, 0.03, 0.5), 12, dtype=torch.float64)
    mae = float(np.abs(theta.mean(0) - np.array(th)).mean())
    key = f"D12_{name.replace(' ','_')}"
    results[key] = dict(D=12, ess_min=float(ess.min()), ess_per_s=float(ess.min()/dt),
        time_s=dt, accept=ap, mae_vs_truth=mae, post_mean=theta.mean(0).tolist(),
        truth=list(th))
    print(f"[D12 {name}] t={dt:.1f}s ESS_min={ess.min():.0f} ESS/s={ess.min()/dt:.2f} "
          f"AR={ap:.2f} MAE={mae:.4f}")

# ── Field data (no truth: compare mean to exact-mode NeuTra mean) ───────
surB = R.load_broad()
theta, ess, dt, ap = run_nuts(d6_potential(R.FIELD_MEASURED, surB, 1e-3, 0.55), 6)
ref = np.array(R.FIELD_CENTERS["broad_mass"])
mae = float(np.abs(theta.mean(0) - ref).mean())
results["Field"] = dict(D=6, ess_min=float(ess.min()), ess_per_s=float(ess.min()/dt),
    time_s=dt, accept=ap, mae_vs_exactmean=mae, post_mean=theta.mean(0).tolist())
print(f"[Field] t={dt:.1f}s ESS_min={ess.min():.0f} ESS/s={ess.min()/dt:.2f} "
      f"AR={ap:.2f} MAE_vs_exactmean={mae:.4f}")

# ── markdown ────────────────────────────────────────────────────────────
L = ["# E2 — Well-tuned NUTS baseline (pyro, dual-averaging + diag mass)\n",
     "warmup=500, target_accept=0.8, 2000 samples, log-theta parameterisation.\n",
     "| target | D | wall-clock (s) | ESS_min | ESS/s | accept | MAE vs truth |",
     "|---|---|---|---|---|---|---|"]
for k, r in results.items():
    mae = r.get("mae_vs_truth", r.get("mae_vs_exactmean"))
    L.append(f"| {k} | {r['D']} | {r['time_s']:.1f} | {r['ess_min']:.0f} | "
             f"{r['ess_per_s']:.2f} | {r['accept']:.2f} | {mae:.4f} |")
L += ["",
      "Compare against NeuTra-HMC exact-mode (paper Tables 5/10/13). NUTS is a strong "
      "reference sampler; the CAI contribution is amortisation + routing (reusable "
      "transport map, ~ms fast mode, exact correction only when the router flags it), "
      "not raw single-observation sampling speed. Field data has no ground-truth theta "
      "(MAE reported vs the exact-mode posterior mean)."]
(HERE / "E2_nuts_comparison.md").write_text("\n".join(L), encoding="utf-8")
json.dump(results, open(HERE / "E2_results.json", "w"), indent=1)
print("\nWrote E2_nuts_comparison.md, E2_results.json")
