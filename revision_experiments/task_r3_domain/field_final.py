"""
R3-T2 + T3 (run after the extended ext surrogate+CNF exist).
T2: re-tune the NeuTra mass matrix per paper 6.10 (z-space 200-iter pre-run -> Mi=1/Var(zi),
    eps=0.04, L=40), re-run field exact mode -> Table-13 row (Time/ESS_min/AR). Fallback NUTS if AR bad.
T3: three prior configs (narrow sigma=0.30, broad sigma=0.55, broad+mass) -> posterior means +
    fit RMS + basin table; decide same-basin (robust) vs narrow prior-pull (prior sensitivity).
Also draws 50 diverse Case-B points from the broad NUTS for the ext-domain FE gate.
"""
import sys, csv, json, time
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(6)
sys.path.insert(0, r"C:\neutra-hmc-surrogate\surrogate_training\revision_experiments")
import revlib as R
from pyro.infer import MCMC, NUTS
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).parent
torch.manual_seed(42); np.random.seed(42)
sur = R.load_broad_ext(); ens = R.load_broad_ext_ensemble(); cnf = R.load_broad_ext_cnf()
y6 = R.FIELD_MEASURED
y20 = torch.tensor(R.field_yfull(sur))
meas = np.array(y6)

def fit_rms(theta):
    with torch.no_grad():
        f = sur(torch.tensor(theta, dtype=torch.float32).reshape(1, -1)).squeeze(0).numpy()[:6]
    return float(np.sqrt((((f - meas) / meas) ** 2).mean()) * 100)

# ── T2: mass-matrix re-tune (z-space pre-run) ───────────────────────────
lp = R.make_log_post(sur, y6, "broad")
pre = R.NeuTraHMC_Conditional(lp.forward, cnf, y20, step_size=0.04, n_leapfrog=40)
theta_pre, ar_pre = pre.run(torch.tensor(R.FIELD_FASTMEAN, dtype=torch.float32), n_samples=200, n_burnin=100)
with torch.no_grad():
    z_pre, _ = cnf.forward(theta_pre, y20.unsqueeze(0).expand(len(theta_pre), -1))
var_z = z_pre.var(0).clamp(min=1e-6).numpy()
mass = (1.0 / var_z)
print(f"[T2] pre-run AR={ar_pre:.2f}  Var(z)={var_z.round(3)}  mass=1/Var(z)={mass.round(3)}", flush=True)

t0 = time.time()
smp = R.NeuTraHMC_MassDiag(lp.forward, cnf, y20, step_size=0.04, n_leapfrog=40, mass_diag=mass.tolist())
chain_m, ar_m = smp.run(torch.tensor(R.FIELD_FASTMEAN, dtype=torch.float32), n_samples=500, n_burnin=100, seed=42)
t_exact = time.time() - t0
ess_m = float(R.effective_sample_size(chain_m).min())
mean_m = chain_m.numpy().mean(0)
print(f"[T2] mass-NeuTra AR={ar_m:.3f} ESS_min={ess_m:.1f} time={t_exact:.1f}s mean e2={mean_m[1]:.2f}", flush=True)
t2_method = "mass-NeuTra-HMC (retuned)"
if ar_m < 0.2:
    kern = NUTS(potential_fn=lambda p: -(-0.5*(((sur(torch.exp(p['z']).unsqueeze(0)).squeeze(0)[:6]-torch.tensor(y6))/torch.tensor(y6))**2).sum()/1e-3 - 0.5*(p['z']**2).sum()/0.55**2), target_accept_prob=0.8)
    mc = MCMC(kern, num_samples=500, warmup_steps=500, initial_params={"z": torch.log(torch.tensor(R.FIELD_FASTMEAN, dtype=torch.float32))}, disable_progbar=True)
    t0 = time.time(); mc.run(); t_exact = time.time() - t0
    chain_m = torch.exp(mc.get_samples()["z"]); ess_m = float(R.effective_sample_size(chain_m).min())
    mean_m = chain_m.numpy().mean(0); ar_m = float("nan"); t2_method = "NUTS (fallback: retuned mass still AR<0.2)"
    print(f"[T2] FALLBACK NUTS ESS_min={ess_m:.1f} time={t_exact:.1f}s", flush=True)

# ── T3: three prior configs ─────────────────────────────────────────────
def nuts_field(sigma, init):
    yt = torch.tensor(y6)
    def pot(params):
        z = params["z"]; th = torch.exp(z); f = sur(th.unsqueeze(0)).squeeze(0)[:6]
        return -(-0.5*(((f-yt)/yt)**2).sum()/1e-3 - 0.5*(z**2).sum()/sigma**2)
    mc = MCMC(NUTS(potential_fn=pot, target_accept_prob=0.8), num_samples=1500, warmup_steps=500,
              initial_params={"z": torch.log(torch.tensor(init, dtype=torch.float32))}, disable_progbar=True)
    mc.run(); return torch.exp(mc.get_samples()["z"]).numpy()

configs = {}
for name, sig in [("narrow(sig0.30)", 0.30), ("broad(sig0.55)", 0.55)]:
    th = nuts_field(sig, R.FIELD_FASTMEAN)
    m = th.mean(0); configs[name] = dict(mean=m.tolist(), fit_rms=fit_rms(m),
        frac_low_e2=float((th[:, 1] < 1.9).mean()), e1_mean=float(m[0]))
    print(f"[T3] {name}: mean={m.round(2)} fit={configs[name]['fit_rms']:.1f}% frac_low_e2={configs[name]['frac_low_e2']:.2f}", flush=True)
configs["broad+mass"] = dict(mean=mean_m.tolist(), fit_rms=fit_rms(mean_m),
    frac_low_e2=None, e1_mean=float(mean_m[0]))

# basin decision: compare narrow vs broad e1 (data-informed dim)
e1n, e1b = configs["narrow(sig0.30)"]["e1_mean"], configs["broad(sig0.55)"]["e1_mean"]
prior_pull = abs(e1n - e1b) > 0.5   # narrow pulled toward theta=1 vs broad's high-e1
verdict = ("PRIOR-SENSITIVE: narrow prior pulls e1 from %.2f (broad) to %.2f (narrow) — "
           "prior sensitivity, NOT multimodality" % (e1b, e1n)) if prior_pull else \
          ("ROBUST: narrow and broad agree on e1 (%.2f vs %.2f) — posterior prior-robust" % (e1n, e1b))
print(f"[T3] {verdict}", flush=True)

# ── Case-B points from broad NUTS ───────────────────────────────────────
thB = nuts_field(0.55, R.FIELD_FASTMEAN)
idx = np.random.choice(len(thB), 47, replace=False)
pts = [np.clip(thB[i], 0.3, 3.6) for i in idx]
pts.append(np.clip(thB.mean(0), 0.3, 3.6))
lpB = R.make_log_post(sur, y6, "broad"); lps = lpB.forward(torch.tensor(thB, dtype=torch.float32)).detach().numpy()
pts.append(np.clip(thB[int(lps.argmax())], 0.3, 3.6))
pts.append(np.clip(thB[int(lps.argmax())] + 0.03*np.random.randn(6), 0.3, 3.6))
kinds = ["random"]*47 + ["post_mean", "MAP", "near_MAP"]
tt = torch.tensor(np.array(pts), dtype=torch.float32)
with torch.no_grad(): fhat = np.mean([m(tt).numpy() for m in ens], axis=0)
with open(HERE / "samples_caseB_ext.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow([f"e{i}" for i in range(1, 7)])
    for p in pts: w.writerow([f"{v:.10f}" for v in p])
json.dump([dict(case="Bext", local_idx=i, kind=kinds[i], theta=[float(v) for v in pts[i]],
                fhat=[float(v) for v in fhat[i]], y_obs=list(y6), K=6) for i in range(len(pts))],
          open(HERE / "points_metadata_ext.json", "w"), indent=1)

json.dump({"T2": {"method": t2_method, "ar": ar_m, "ess_min": ess_m, "time_s": t_exact,
                  "mean": mean_m.tolist(), "mass": mass.tolist(), "var_z": var_z.tolist(), "pre_ar": ar_pre},
           "T3": {"configs": configs, "prior_pull": bool(prior_pull), "verdict": verdict}},
          open(HERE / "field_final_results.json", "w"), indent=2)
print("Wrote field_final_results.json, samples_caseB_ext.csv", flush=True)
