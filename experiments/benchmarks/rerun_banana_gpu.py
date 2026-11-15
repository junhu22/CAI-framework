"""
rerun_banana_gpu.py — verify NeuTra-HMC wall-clock for the banana benchmark
                      (Table 1) on GPU.

The base script benchmark_banana_conditional.py already sets
    BananaConfig.device = "cuda" if torch.cuda.is_available() else "cpu"
so the auto-detect should pick the RTX 4090.  This wrapper does three things:

  1) Quick timing test
       Average wall-clock for a single U_z forward+backward
       (one leapfrog gradient evaluation) on CPU vs GPU.  This gives the
       practical speedup ratio without needing a trained CNF.

  2) Train a small CNF for D=6 on GPU
       Same architecture and training schedule as the original benchmark
       (n_train=50000, n_epochs=500) so the wall-clock is directly
       comparable to the saved Table 1 values.

  3) Run NeuTra-HMC + Standard HMC for D=6 on GPU
       1 observation, 500 samples + 100 burn-in.  Report:
         - wall-clock (with cuda.synchronize for accurate timing)
         - acceptance rate
         - ESS_min, ESS_mean

Outputs: results/banana_gpu/banana_gpu_d6.json
         logs/rerun_banana_gpu.log
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] using {device}", flush=True)
if device.type == "cuda":
    print(f"[device]   name = {torch.cuda.get_device_name(0)}", flush=True)
    print(f"[device]   torch = {torch.__version__}", flush=True)

SURR_DIR = Path(r"C:/neutra-hmc-surrogate/surrogate_training")
sys.path.insert(0, str(SURR_DIR / "experiments/benchmarks"))

from benchmark_banana_conditional import (  # noqa: E402
    BananaConfig, ConditionalBananaTarget, ConditionalRealNVP,
    train_cnf, neutra_hmc_inference, standard_hmc_inference,
)

OUT_DIR = SURR_DIR / "results/banana_gpu"
OUT_LOG = SURR_DIR / "logs/rerun_banana_gpu.log"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_LOG.parent.mkdir(parents=True, exist_ok=True)

D = 6
CPU_BASELINE_D6_S = 770.0  # approximate from saved banana_results.json (range 763-779s)


def U_z_step(cnf, target, z, y, d):
    """One U_z forward+backward (one leapfrog gradient evaluation)."""
    z_req = z.detach().requires_grad_(True)
    y_exp = y.unsqueeze(0).to(d) if y.dim() == 1 else y.to(d)
    theta = cnf.inverse(z_req, y_exp)
    _, log_det = cnf.forward(theta, y_exp)
    log_pi = target.log_posterior(theta, y.to(d))
    U = -(log_pi + log_det).sum()
    U.backward()
    return U.detach(), z_req.grad.detach()


def timing_test(n_iter=100):
    """Measure per-step wall-clock for U_z on cpu vs cuda."""
    cfg = BananaConfig(dimensions=[D], n_test_observations=1, n_seeds=1)
    target = ConditionalBananaTarget(D, cfg)
    cnf = ConditionalRealNVP(D, D, cfg)

    results = {}
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")

    for dev_name in devs:
        d = torch.device(dev_name)
        cnf_d = ConditionalRealNVP(D, D, cfg).to(d)
        cnf_d.load_state_dict(cnf.state_dict())
        cnf_d.eval()

        # Fixed test data on this device
        torch.manual_seed(0)
        theta_true = target.sample_prior(1, dev_name).squeeze(0)
        y = target.generate_observation(theta_true.to(dev_name))
        z = torch.randn(1, D, device=d)

        # Warmup
        for _ in range(10):
            U_z_step(cnf_d, target, z, y, d)

        if dev_name == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            U_z_step(cnf_d, target, z, y, d)
        if dev_name == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) / n_iter * 1000
        results[dev_name] = elapsed_ms
        print(f"  [{dev_name:>4}] {elapsed_ms:7.3f} ms / U_z fwd+bwd  "
              f"(avg over {n_iter} iters)", flush=True)

    if "cuda" in results and "cpu" in results:
        speedup = results["cpu"] / results["cuda"]
        print(f"  GPU speedup: {speedup:.2f}x", flush=True)
        results["speedup"] = speedup
    return results


def full_rerun_gpu():
    """Train CNF on GPU and run NeuTra-HMC + Std-HMC for D=6."""
    cfg = BananaConfig(dimensions=[D], n_test_observations=1, n_seeds=1,
                      n_train=50000, n_epochs=500)
    cfg.device = str(device)
    print(f"  cfg.device = {cfg.device}", flush=True)

    torch.manual_seed(42)
    np.random.seed(42)
    target = ConditionalBananaTarget(D, cfg)

    print(f"\n  [train] CNF D={D}, n_train={cfg.n_train}, n_epochs={cfg.n_epochs} ...",
          flush=True)
    cnf = ConditionalRealNVP(D, D, cfg)
    t0 = time.perf_counter()
    info = train_cnf(cnf, target, cfg)
    t_train = time.perf_counter() - t0
    print(f"  [train] done in {t_train:.1f}s  "
          f"val_loss={info['best_val_loss']:.4f}  epochs={info['epochs']}",
          flush=True)

    # Single test observation
    theta_true = target.sample_prior(1, cfg.device).squeeze(0)
    y = target.generate_observation(theta_true.to(cfg.device))

    print(f"\n  [NeuTra-HMC] D={D}, N={cfg.n_samples}, burn={cfg.n_burnin}, "
          f"L={cfg.n_leapfrog}, eps={cfg.hmc_step_size} ...", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    exact_samples, exact_time, exact_diag = neutra_hmc_inference(cnf, target, y, cfg)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_neutra = time.perf_counter() - t0
    print(f"  [NeuTra-HMC] AR={exact_diag['acceptance_rate']:.1%}  "
          f"ESS_min={exact_diag['ess_min']:.0f}  "
          f"ESS_mean={exact_diag['ess_mean']:.0f}  "
          f"time={wall_neutra:.1f}s (internal: {exact_time:.1f}s)", flush=True)

    print(f"\n  [Std-HMC] D={D}, N={cfg.n_samples}, burn={cfg.n_burnin}, "
          f"L={cfg.n_leapfrog}, eps=0.01 ...", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    hmc_samples, hmc_time, hmc_diag = standard_hmc_inference(target, y, cfg)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_hmc = time.perf_counter() - t0
    print(f"  [Std-HMC]   AR={hmc_diag['acceptance_rate']:.1%}  "
          f"ESS_min={hmc_diag['ess_min']:.0f}  "
          f"ESS_mean={hmc_diag['ess_mean']:.0f}  "
          f"time={wall_hmc:.1f}s (internal: {hmc_time:.1f}s)", flush=True)

    return {
        "D": D,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "training_time_s": float(t_train),
        "best_val_loss": float(info["best_val_loss"]),
        "n_epochs_used": int(info["epochs"]),
        "neutra_hmc": {
            "wall_time_s": float(wall_neutra),
            "internal_time_s": float(exact_time),
            "ess_min": float(exact_diag["ess_min"]),
            "ess_mean": float(exact_diag["ess_mean"]),
            "acceptance_rate": float(exact_diag["acceptance_rate"]),
        },
        "std_hmc": {
            "wall_time_s": float(wall_hmc),
            "internal_time_s": float(hmc_time),
            "ess_min": float(hmc_diag["ess_min"]),
            "ess_mean": float(hmc_diag["ess_mean"]),
            "acceptance_rate": float(hmc_diag["acceptance_rate"]),
        },
    }


def main():
    overall_t0 = time.perf_counter()
    log_buf = []
    log_buf.append(f"[device] {device}  "
                   f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    log_buf.append(f"[torch] {torch.__version__}")

    print("\n" + "=" * 70, flush=True)
    print("  STEP 1: timing test (per U_z fwd+bwd)", flush=True)
    print("=" * 70, flush=True)
    timing = timing_test(n_iter=100)
    log_buf.append(f"\nTiming test (100 iters avg):")
    for k, v in timing.items():
        if isinstance(v, float):
            unit = "x" if k == "speedup" else " ms"
            log_buf.append(f"  {k}: {v:.3f}{unit}")

    print("\n" + "=" * 70, flush=True)
    print(f"  STEP 2: full GPU rerun (D=6, train + NeuTra-HMC + Std-HMC)", flush=True)
    print("=" * 70, flush=True)
    full = full_rerun_gpu()
    log_buf.append(f"\nFull rerun result:")
    log_buf.append(f"  Training: {full['training_time_s']:.1f}s")
    log_buf.append(f"  NeuTra-HMC: {full['neutra_hmc']['wall_time_s']:.1f}s  "
                   f"(CPU baseline {CPU_BASELINE_D6_S:.0f}s, "
                   f"speedup {CPU_BASELINE_D6_S / full['neutra_hmc']['wall_time_s']:.2f}x)")
    log_buf.append(f"  Std HMC:    {full['std_hmc']['wall_time_s']:.1f}s")

    out = {
        "timing_test": timing,
        "full_rerun": full,
        "cpu_baseline_d6_s": CPU_BASELINE_D6_S,
        "speedup_d6_neutra_hmc": (CPU_BASELINE_D6_S /
                                  full["neutra_hmc"]["wall_time_s"]),
    }
    out_json = OUT_DIR / "banana_gpu_d6.json"
    out_json.write_text(json.dumps(out, indent=2))
    print(f"\n  Saved: {out_json}", flush=True)

    log_buf.append(f"\nTotal elapsed: {(time.perf_counter() - overall_t0)/60:.1f} min")
    OUT_LOG.write_text("\n".join(log_buf))
    print(f"\n[done] log: {OUT_LOG}", flush=True)


if __name__ == "__main__":
    main()
