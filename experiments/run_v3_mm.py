"""
run_v3_mm.py — Final v3_mm full experiment.

Config (best-on-Pareto from tune_v2_hmc_b sweep):
    step_size      = 0.04
    n_leapfrog     = 40
    mass_diag      = reused from results/tune_v2_hmc_b/part_b.json
    no step-size adaptation

Outputs:
    results/real_data_v3_mm/real_data_validation_v3_mm.json
    results/damage_detection_v3_mm/damage_results_v3_mm.json
    figures/fig_real_data_posterior_v3_mm.{pdf,png}
    figures/fig_damage_detection_v3_mm.{pdf,png}
    logs/run_v3_mm.log
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] using {device}", flush=True)

SURR_DIR = Path(r"C:/neutra-hmc-surrogate/surrogate_training")
sys.path.insert(0, str(SURR_DIR))
sys.path.insert(0, str(SURR_DIR / "experiments"))

import cai_lib
cai_lib.DEV = device

from config import N_PARAMS, N_MODES, PARAM_NAMES  # noqa: E402
from cai_lib import (  # noqa: E402
    load_surrogate, load_cnf, KModeLogPosterior, NeuTraHMC_Conditional,
    HMCSampler, effective_sample_size, importance_weight_ess,
)

D, M = N_PARAMS, N_MODES
F_MEASURED_6 = np.array([0.91, 1.08, 1.47, 2.55, 3.16, 3.81], dtype=np.float32)

SURR_PATH = SURR_DIR / "surrogate_models_broad/member_0.pt"
CNF_PATH = SURR_DIR / "experiments/results/conditional_flow_broad.pt"
PART_B_JSON = SURR_DIR / "results/tune_v2_hmc_b/part_b.json"

OUT_REAL_MM = SURR_DIR / "results/real_data_v3_mm"
OUT_DAMAGE_MM = SURR_DIR / "results/damage_detection_v3_mm"
OUT_FIGS = SURR_DIR / "figures"
OUT_LOG = SURR_DIR / "logs/run_v3_mm.log"
for d in (OUT_REAL_MM, OUT_DAMAGE_MM, OUT_FIGS, OUT_LOG.parent):
    d.mkdir(parents=True, exist_ok=True)

# Final config
STEP_SIZE = 0.04
N_LEAPFROG = 40


class NeuTraHMC_MassDiag(NeuTraHMC_Conditional):
    def __init__(self, log_post_fn, cond_flow, y_obs,
                 step_size, n_leapfrog, mass_diag):
        super().__init__(log_post_fn, cond_flow, y_obs,
                         step_size=step_size, n_leapfrog=n_leapfrog)
        m = mass_diag.to(y_obs.device).clamp(min=1e-8)
        self.mass = m
        self.inv_mass = 1.0 / m
        self.sqrt_mass = torch.sqrt(m)

    def _leapfrog_mass(self, q, p, eps, grad_init=None):
        g = grad_init if grad_init is not None else self._grad_log_prob_z(q)[1]
        p = p + 0.5 * eps * g
        for _ in range(self.n_leapfrog - 1):
            q = q + eps * (p * self.inv_mass)
            _, g = self._grad_log_prob_z(q)
            p = p + eps * g
        q = q + eps * (p * self.inv_mass)
        lp_final, g = self._grad_log_prob_z(q)
        p = p + 0.5 * eps * g
        return q, p, lp_final, g

    def run(self, theta0, n_samples, n_burnin=100):
        self.flow.eval()
        with torch.no_grad():
            z, _ = self.flow(theta0.unsqueeze(0), self._y_batch(1))
        z = z.squeeze(0)
        z_chain = []
        n_acc = 0
        eps = self.step_size
        lp_current, grad_current = self._grad_log_prob_z(z)
        for i in range(-n_burnin, n_samples):
            p = self.sqrt_mass * torch.randn_like(z)
            H0 = -lp_current + 0.5 * (p ** 2 * self.inv_mass).sum()
            q_prop, p_prop, lp_prop, grad_prop = self._leapfrog_mass(
                z, p, eps, grad_init=grad_current)
            H1 = -lp_prop + 0.5 * (p_prop ** 2 * self.inv_mass).sum()
            if torch.log(torch.rand(1)).item() < (H0 - H1).item():
                z = q_prop
                lp_current = lp_prop
                grad_current = grad_prop
                if i >= 0:
                    n_acc += 1
            if i >= 0:
                z_chain.append(z.clone())
        z_stack = torch.stack(z_chain)
        with torch.no_grad():
            y_batch = self._y_batch(len(z_chain))
            theta_chain, _ = self.flow.inverse(z_stack, y_batch)
        return theta_chain, n_acc / n_samples, z_stack


def run_mass(log_post, cnf, y_full_t, theta_init, mass_diag,
             step_size, n_leapfrog, n_samples=500, n_burnin=100, seed=42):
    torch.manual_seed(seed)
    sampler = NeuTraHMC_MassDiag(log_post.forward, cnf, y_full_t,
                                 step_size=step_size, n_leapfrog=n_leapfrog,
                                 mass_diag=mass_diag)
    t0 = time.time()
    chain, ar, _ = sampler.run(theta_init, n_samples=n_samples, n_burnin=n_burnin)
    elapsed = time.time() - t0
    ess = effective_sample_size(chain)
    return {
        "step_size": float(step_size), "n_leapfrog": int(n_leapfrog),
        "trajectory_length": float(step_size * n_leapfrog),
        "n_samples": int(n_samples), "n_burnin": int(n_burnin),
        "acceptance_rate": float(ar),
        "ess_min": float(ess.min().item()),
        "ess_per_dim": [float(x) for x in ess.tolist()],
        "elapsed_s": float(elapsed),
        "posterior_mean": chain.mean(0).cpu().numpy().tolist(),
        "posterior_std": chain.std(0).cpu().numpy().tolist(),
    }, chain


def main():
    t0 = time.time()
    log_buf = []

    print("\n[setup] loading models...", flush=True)
    surrogate = load_surrogate(SURR_PATH)
    cnf = load_cnf(CNF_PATH)
    with torch.no_grad():
        f_nominal = surrogate(torch.ones(1, D, device=device)).squeeze().cpu().numpy()
    y_full = np.concatenate([F_MEASURED_6, f_nominal[6:]]).astype(np.float32)
    y_full_t = torch.tensor(y_full, dtype=torch.float32, device=device)
    log_post = KModeLogPosterior(
        surrogate, F_MEASURED_6, kappa_sq=1e-3, prior_sigma=0.55).to(device)

    part_b = json.loads(PART_B_JSON.read_text())
    mass = torch.tensor(part_b["mass_diag"], dtype=torch.float32, device=device)
    msg = f"[setup] mass = {[round(float(x), 3) for x in mass.cpu().tolist()]}"
    print(msg, flush=True); log_buf.append(msg)
    msg = f"[setup] step_size={STEP_SIZE}  n_leapfrog={N_LEAPFROG}  T={STEP_SIZE*N_LEAPFROG:.2f}"
    print(msg, flush=True); log_buf.append(msg)

    # Fast mode
    print("\n[Exp A] fast mode (N=2000)...", flush=True)
    torch.manual_seed(42)
    t_fm0 = time.time()
    with torch.no_grad():
        theta_fast = cnf.sample(y_full_t, n_samples=2000)
    t_fast = time.time() - t_fm0
    ess_is, ess_is_ratio = importance_weight_ess(
        theta_fast, log_post.forward, cnf, y_full_t)
    fast_mean = theta_fast.mean(0).cpu().numpy()
    fast_std = theta_fast.std(0).cpu().numpy()
    msg = (f"  ESS_IS={ess_is:.2f}  ratio={ess_is_ratio:.4f}  "
           f"time={t_fast*1000:.1f}ms")
    print(msg, flush=True); log_buf.append(msg)
    msg = f"  Mean: {[round(float(x), 4) for x in fast_mean.tolist()]}"
    print(msg, flush=True); log_buf.append(msg)

    # Exact mode (mass-HMC)
    theta_init = torch.tensor(fast_mean, dtype=torch.float32, device=device)
    print(f"\n[Exp A] exact mode (mass-HMC, eps={STEP_SIZE}, L={N_LEAPFROG}, "
          f"N=500, burn=100)...", flush=True)
    r_exact, chain_exact = run_mass(
        log_post, cnf, y_full_t, theta_init, mass,
        step_size=STEP_SIZE, n_leapfrog=N_LEAPFROG,
        n_samples=500, n_burnin=100, seed=42)
    msg = (f"  AR={r_exact['acceptance_rate']*100:.1f}%  "
           f"ESS_min={r_exact['ess_min']:.2f}  "
           f"time={r_exact['elapsed_s']:.1f}s")
    print(msg, flush=True); log_buf.append(msg)
    msg = f"  Mean: {[round(x, 4) for x in r_exact['posterior_mean']]}"
    print(msg, flush=True); log_buf.append(msg)
    msg = f"  Std : {[round(x, 4) for x in r_exact['posterior_std']]}"
    print(msg, flush=True); log_buf.append(msg)

    # Std HMC baseline
    print(f"\n[Exp A] std-HMC baseline (N=500, burn=100)...", flush=True)
    torch.manual_seed(42)
    hmc = HMCSampler(log_post.grad_log_prob, step_size=0.01, n_leapfrog=20)
    t_hmc0 = time.time()
    chain_hmc, ar_hmc = hmc.run(torch.ones(D, device=device),
                                n_samples=500, n_burnin=100)
    t_hmc = time.time() - t_hmc0
    ess_hmc = effective_sample_size(chain_hmc)
    hmc_mean = chain_hmc.mean(0).cpu().numpy()
    hmc_std = chain_hmc.std(0).cpu().numpy()
    msg = (f"  AR={ar_hmc*100:.1f}%  ESS_min={ess_hmc.min():.2f}  "
           f"time={t_hmc:.1f}s")
    print(msg, flush=True); log_buf.append(msg)

    # Fit at exact mean
    exact_mean = np.array(r_exact["posterior_mean"], dtype=np.float32)
    with torch.no_grad():
        f_pred = surrogate(torch.tensor(exact_mean, device=device).unsqueeze(0)).squeeze().cpu().numpy()
    rel_err = (f_pred[:6] - F_MEASURED_6) / F_MEASURED_6
    rms_pct = float(np.sqrt((rel_err ** 2).mean()) * 100)
    print(f"\n[Exp A] fit at exact-mode mean:", flush=True)
    for i in range(6):
        line = (f"  f{i+1}: measured={F_MEASURED_6[i]:.3f}, "
                f"predicted={f_pred[i]:.3f}, rel_err={rel_err[i]*100:+.2f}%")
        print(line, flush=True); log_buf.append(line)
    msg = f"  RMS rel err: {rms_pct:.2f}%"
    print(msg, flush=True); log_buf.append(msg)

    out_a = {
        "experiment": "Experiment A v3_mm (mass-matrix HMC, broad prior)",
        "tuning_config": {
            "step_size": STEP_SIZE,
            "n_leapfrog": N_LEAPFROG,
            "trajectory_length": STEP_SIZE * N_LEAPFROG,
            "mass_diag": [float(x) for x in mass.cpu().tolist()],
            "no_adapt": True,
            "selection_rationale": (
                "Best-on-Pareto for (AR, ESS_min) from sweeps in tune_v2_hmc_b/c/d. "
                "Smaller eps raised AR but collapsed ESS_min; this config gives "
                "highest ESS_min across all (eps, L, mass) tried. AR=~55% is "
                "consistent with a multi-modal posterior on the broadened prior."
            ),
        },
        "surrogate": str(SURR_PATH), "cnf": str(CNF_PATH),
        "measured_freqs_hz": F_MEASURED_6.tolist(),
        "fast_mode": {
            "time_ms": float(t_fast * 1000),
            "posterior_mean": fast_mean.tolist(),
            "posterior_std": fast_std.tolist(),
            "ess_is": float(ess_is), "ess_is_ratio": float(ess_is_ratio),
        },
        "exact_mode": r_exact,
        "std_hmc": {
            "time_s": float(t_hmc), "acceptance_rate": float(ar_hmc),
            "ess_min": float(ess_hmc.min().item()),
            "posterior_mean": hmc_mean.tolist(),
            "posterior_std": hmc_std.tolist(),
        },
        "fit_quality": {
            "predicted_freqs": f_pred[:6].tolist(),
            "rel_error_pct": (rel_err * 100).tolist(),
            "rms_rel_error_pct": rms_pct,
        },
    }
    out_json = OUT_REAL_MM / "real_data_validation_v3_mm.json"
    out_json.write_text(json.dumps(out_a, indent=2))
    print(f"\n  Saved: {out_json}", flush=True)

    # Figure A
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for i, ax in enumerate(axes.flat):
        ax.hist(theta_fast[:, i].cpu().numpy(), bins=40, density=True,
                alpha=0.45, label="Fast (CNF)")
        ax.hist(chain_exact[:, i].cpu().numpy(), bins=40, density=True,
                alpha=0.45, label="Exact (NeuTra-HMC)")
        ax.hist(chain_hmc[:, i].cpu().numpy(), bins=40, density=True,
                alpha=0.45, label="Std HMC")
        ax.set_xlabel(PARAM_NAMES[i])
        ax.set_ylabel("density" if i % 3 == 0 else "")
        # Legend on every subplot so readers don't need to look up at panel 0
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    plt.suptitle(f"CAI v3_mm posterior on Lam 2019 "
                 f"(eps={STEP_SIZE}, L={N_LEAPFROG}, diag-mass)")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "fig_real_data_posterior_v3_mm.pdf", bbox_inches="tight")
    plt.savefig(OUT_FIGS / "fig_real_data_posterior_v3_mm.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_FIGS / 'fig_real_data_posterior_v3_mm.pdf'}", flush=True)

    # Experiment B
    print("\n[Exp B] damage detection v3_mm ...", flush=True)
    with torch.no_grad():
        f_norm_pad = surrogate(torch.ones(1, D, device=device)).squeeze().cpu().numpy()
        theta_dmg = torch.tensor([[0.75, 1.0, 1.0, 1.0, 1.0, 1.0]],
                                 device=device, dtype=torch.float32)
        f_dmg_full = surrogate(theta_dmg).squeeze().cpu().numpy()
    y_normal = np.concatenate([F_MEASURED_6, f_norm_pad[6:]]).astype(np.float32)
    y_damage = f_dmg_full.astype(np.float32)

    scenarios = []
    fast_samples = {}
    for name, y_vec in [("Normal (real Lam 2019)", y_normal),
                        ("Damage (e1=0.75)", y_damage)]:
        y_t = torch.tensor(y_vec, dtype=torch.float32, device=device)
        lp = KModeLogPosterior(surrogate, y_vec[:6],
                               kappa_sq=1e-3, prior_sigma=0.55).to(device)
        torch.manual_seed(42)
        t_b0 = time.time()
        with torch.no_grad():
            tf = cnf.sample(y_t, n_samples=2000)
        t_fm = time.time() - t_b0
        ess_is, ratio = importance_weight_ess(tf, lp.forward, cnf, y_t)
        msg = (f"  {name:30s}  ESS_IS={ess_is:6.2f}  ratio={ratio:.4f}  "
               f"time={t_fm*1000:.1f}ms")
        print(msg, flush=True); log_buf.append(msg)
        fast_samples[name] = tf.cpu().numpy()
        scenarios.append({
            "name": name, "y_full": y_vec.tolist(),
            "posterior_mean": tf.mean(0).cpu().numpy().tolist(),
            "posterior_std": tf.std(0).cpu().numpy().tolist(),
            "time_ms": float(t_fm * 1000),
            "ess_is": float(ess_is), "ess_is_ratio": float(ratio),
        })

    out_b = {
        "experiment": "Experiment B v3_mm",
        "trigger_threshold": 0.5,
        "scenarios": scenarios,
    }
    out_json = OUT_DAMAGE_MM / "damage_results_v3_mm.json"
    out_json.write_text(json.dumps(out_b, indent=2))
    print(f"  Saved: {out_json}", flush=True)

    # Figure B
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for i, ax in enumerate(axes.flat):
        ax.hist(fast_samples["Normal (real Lam 2019)"][:, i], bins=40,
                density=True, alpha=0.45, label="Normal")
        ax.hist(fast_samples["Damage (e1=0.75)"][:, i], bins=40,
                density=True, alpha=0.45, label="Damage")
        ax.set_xlabel(PARAM_NAMES[i])
        ax.set_ylabel("density" if i % 3 == 0 else "")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    plt.suptitle("CAI v3_mm damage detection (fast-mode posteriors)")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "fig_damage_detection_v3_mm.pdf", bbox_inches="tight")
    plt.savefig(OUT_FIGS / "fig_damage_detection_v3_mm.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT_FIGS / 'fig_damage_detection_v3_mm.pdf'}", flush=True)

    log_buf.append(f"\nTotal elapsed: {(time.time() - t0)/60:.1f} min")
    OUT_LOG.write_text("\n".join(log_buf))
    print(f"\n[done] log written to {OUT_LOG}", flush=True)


if __name__ == "__main__":
    main()
