"""
FREEZE ROUND: all final field numbers under the frozen config
  = narrow prior sigma=0.30 + extended [0.5,3.5] surrogate/CNF (ext).
F1: Table 13 (re-tuned mass NeuTra-HMC N=500 + tuned NUTS + 3-chain split-Rhat).
F2.1: fast-mode field ESS_IS/N (sigma=0.30).
F2.2: fit RMS at the narrow posterior mean.
F3.2: narrow-vs-broad posterior marginal overlay figure.
"""
import sys, json, time
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(6)
sys.path.insert(0, r"C:\neutra-hmc-surrogate\surrogate_training\revision_experiments")
import revlib as R
from cai_lib import KModeLogPosterior
from pyro.infer import MCMC, NUTS
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).parent
torch.manual_seed(42); np.random.seed(42)
sur = R.load_broad_ext(); cnf = R.load_broad_ext_cnf()
y6 = R.FIELD_MEASURED; meas = np.array(y6)
y20 = torch.tensor(R.field_yfull(sur))
SIG_N, SIG_B = 0.30, 0.55
lp_n = KModeLogPosterior(sur, list(y6), kappa_sq=1e-3, prior_sigma=SIG_N).to(R.DEV)  # narrow (frozen)

def fit_rms(theta):
    with torch.no_grad():
        f = sur(torch.tensor(theta, dtype=torch.float32).reshape(1, -1)).squeeze(0).numpy()[:6]
    return float(np.sqrt((((f - meas) / meas) ** 2).mean()) * 100)

# ── F2.1 fast-mode ESS (frozen: sigma=0.30, ext CNF) ────────────────────
ess_vals = []
for s in [42, 43, 44, 45, 46]:
    torch.manual_seed(s); tf = cnf.sample(y20, 2000)
    _, r = R.importance_weight_ess(tf, lp_n.forward, cnf, y20); ess_vals.append(r)
fast_ess = float(np.mean(ess_vals)); fast_ess_sd = float(np.std(ess_vals))
print(f"[F2.1] fast ESS_IS/N (sigma=0.30) = {fast_ess:.4f} +- {fast_ess_sd:.4f}", flush=True)

# ── F1 mass re-tune (z-space pre-run) + mass-NeuTra N=500 ───────────────
pre = R.NeuTraHMC_Conditional(lp_n.forward, cnf, y20, step_size=0.04, n_leapfrog=40)
th_pre, _ = pre.run(torch.tensor(R.FIELD_FASTMEAN, dtype=torch.float32), n_samples=200, n_burnin=100)
with torch.no_grad():
    z_pre, _ = cnf.forward(th_pre, y20.unsqueeze(0).expand(len(th_pre), -1))
mass = (1.0 / z_pre.var(0).clamp(min=1e-6)).numpy()
t0 = time.time()
smp = R.NeuTraHMC_MassDiag(lp_n.forward, cnf, y20, step_size=0.04, n_leapfrog=40, mass_diag=mass.tolist())
ch_ne, ar_ne = smp.run(torch.tensor(R.FIELD_FASTMEAN, dtype=torch.float32), n_samples=500, n_burnin=100, seed=42)
t_ne = time.time() - t0
ess_ne = float(R.effective_sample_size(ch_ne).min()); mean_ne = ch_ne.numpy().mean(0)
print(f"[F1] mass-NeuTra: AR={ar_ne:.3f} ESS_min={ess_ne:.1f} time={t_ne:.1f}s mean e1={mean_ne[0]:.2f}", flush=True)

# ── F1 tuned NUTS (Pyro warmup 500 ta 0.8) + 3-chain Rhat ───────────────
def nuts_narrow(init, nsamp=500, seed=0):
    yt = torch.tensor(y6)
    def U(p):
        z = p["z"]; f = sur(torch.exp(z).unsqueeze(0)).squeeze(0)[:6]
        return -(-0.5 * (((f - yt) / yt) ** 2).sum() / 1e-3 - 0.5 * (z ** 2).sum() / SIG_N ** 2)
    torch.manual_seed(seed)
    mc = MCMC(NUTS(potential_fn=U, target_accept_prob=0.8), num_samples=nsamp, warmup_steps=500,
              initial_params={"z": torch.log(torch.tensor(init, dtype=torch.float32))}, disable_progbar=True)
    t = time.time(); mc.run(); dt = time.time() - t
    th = torch.exp(mc.get_samples()["z"])
    try: ap = float(np.mean(mc.diagnostics()["acceptance rate"]["z"]))
    except Exception: ap = 0.8
    return th, dt, ap

inits = [R.FIELD_FASTMEAN, np.ones(6), (np.array(R.FIELD_FASTMEAN) * 1.3)]
nuts_chains = []
for ci, init in enumerate(inits):
    th, dt, ap = nuts_narrow(init, 500, seed=42 + ci)
    nuts_chains.append(th.numpy())
    if ci == 0:
        t_nu, ess_nu, mean_nu = dt, float(R.effective_sample_size(th).min()), th.numpy().mean(0)
        print(f"[F1] NUTS: ESS_min={ess_nu:.1f} time={dt:.1f}s AR~{ap:.2f} mean e1={mean_nu[0]:.2f}", flush=True)
rhat = R.split_rhat(nuts_chains)
print(f"[F1] 3-chain split-Rhat (narrow): max={np.nanmax(rhat):.3f} per-dim={rhat.round(3).tolist()}", flush=True)

# means-agree convergence evidence
mean_agree = float(np.abs(mean_ne - mean_nu).mean())
print(f"[F1] NeuTra vs NUTS mean MAE = {mean_agree:.4f} (converged if small)", flush=True)

# ── F3.2 broad NUTS (for overlay) ───────────────────────────────────────
def nuts_broad(init, nsamp=1500, seed=7):
    yt = torch.tensor(y6)
    def U(p):
        z = p["z"]; f = sur(torch.exp(z).unsqueeze(0)).squeeze(0)[:6]
        return -(-0.5 * (((f - yt) / yt) ** 2).sum() / 1e-3 - 0.5 * (z ** 2).sum() / SIG_B ** 2)
    torch.manual_seed(seed)
    mc = MCMC(NUTS(potential_fn=U, target_accept_prob=0.8), num_samples=nsamp, warmup_steps=500,
              initial_params={"z": torch.log(torch.tensor(init, dtype=torch.float32))}, disable_progbar=True)
    mc.run(); return torch.exp(mc.get_samples()["z"]).numpy()
th_broad = nuts_broad(R.FIELD_FASTMEAN)
th_narrow = np.concatenate(nuts_chains)

# overlay figure (all 6 marginals)
fig, axes = plt.subplots(2, 3, figsize=(11, 5.5))
for d, ax in enumerate(axes.ravel()):
    ax.hist(th_narrow[:, d], bins=40, density=True, alpha=0.6, color="#2a7", label="narrow σ=0.30 (frozen)")
    ax.hist(th_broad[:, d], bins=40, density=True, alpha=0.5, color="#c44", label="broad σ=0.55 (comparison)")
    ax.axvline(3.5, color="grey", ls=":", lw=0.8)
    ax.set_title(f"e{d+1}", fontsize=9)
    if d == 0: ax.legend(fontsize=7)
fig.suptitle("Field posterior marginals — prior sensitivity (narrow frozen vs broad comparison)")
fig.tight_layout(); fig.savefig(HERE / "fig_field_priorconfig_v2.png", dpi=150)

# ── save ────────────────────────────────────────────────────────────────
out = {"frozen_config": "narrow prior sigma=0.30 + ext surrogate/CNF [0.5,3.5]",
       "fast_ess_mean": fast_ess, "fast_ess_std": fast_ess_sd,
       "table13": {
           "mass_neutra": {"AR": ar_ne, "ESS_min": ess_ne, "time_s": t_ne, "mean": mean_ne.tolist(),
                           "mass_diag": mass.tolist()},
           "nuts": {"ESS_min": ess_nu, "time_s": t_nu, "mean": mean_nu.tolist()},
           "split_rhat_3chain": rhat.tolist(), "neutra_vs_nuts_mean_mae": mean_agree,
           "neutra_ess_min_lt_20": bool(ess_ne < 20)},
       "fit_rms_narrow_mean": fit_rms(mean_nu),
       "narrow_e1_mean": float(mean_nu[0]), "broad_e1_mean": float(th_broad[:, 0].mean()),
       "broad_fit_rms": fit_rms(th_broad.mean(0))}
json.dump(out, open(HERE / "field_freeze_results.json", "w"), indent=2)
print(f"[F2.2] fit RMS (narrow mean) = {out['fit_rms_narrow_mean']:.2f}%", flush=True)
print("Wrote field_freeze_results.json, fig_field_priorconfig_v2.png", flush=True)
