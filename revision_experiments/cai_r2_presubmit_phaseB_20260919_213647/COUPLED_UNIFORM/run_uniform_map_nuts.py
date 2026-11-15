import hashlib
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyro
import torch
import torch.nn.functional as F
from pyro.infer import MCMC, NUTS
from scipy.optimize import minimize

ROOT = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
OUT = ROOT / "revision_experiments/cai_r2_presubmit_phaseB_20260919_213647/COUPLED_UNIFORM"
SURROGATE_PATH = ROOT / "surrogate_models_coupled/member_0.pt"
SOURCE = ROOT / "experiments/coupled_building/04_train_cnf_coupled.py"
Y6_DECIMAL = [2.06, 2.30, 2.62, 6.14, 6.53, 6.76]
LO, HI = 0.5, 4.0
WIDTH = HI - LO
KAPPA_SQ = 0.03 ** 2
SEEDS = [811, 812, 813, 814]
WARMUP = 500
RETAINED = 600
TARGET_ACCEPT = 0.90


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path, obj):
    Path(path).write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_classes():
    sys.path.insert(0, str(ROOT / "experiments/coupled_building"))
    spec = importlib.util.spec_from_file_location("coupled_source_for_uniform_nuts", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "COUPLED_UNIFORM_MAP_NUTS_MANIFEST.json"
    if manifest_path.exists():
        raise RuntimeError("Refusing to overwrite coupled MAP/NUTS run")
    assert sha(SURROGATE_PATH).lower() == "6a1e4f530d4e3628e0b7fdbf08661b4dc32a71a0c2b538b26f5af9d7c07b9986"
    manifest = {
        "created_utc_before_map_or_sampling": now(),
        "target": "coupled field, first six Hu et al. measured modes",
        "observation_decimal": Y6_DECIMAL,
        "likelihood": "relative-error Gaussian potential, kappa_sq=0.0009",
        "theta_prior": "independent Uniform(0.5,4.0)^3",
        "nuts_transform": "theta=0.5+3.5*sigmoid(z), with sum log|dtheta/dz| included",
        "surrogate_path": str(SURROGATE_PATH),
        "surrogate_sha256": sha(SURROGATE_PATH),
        "map": {
            "objective": "box-constrained maximum likelihood; Uniform prior constant in box",
            "starts": "27-point interior grid plus 5 PCG64 Uniform-box draws",
            "start_seed": 20260919,
            "optimizer": "SciPy L-BFGS-B with explicit [0.5,4.0]^3 bounds and analytic autograd gradient",
        },
        "nuts": {
            "seeds_predeclared": SEEDS,
            "warmup_steps": WARMUP,
            "retained_per_chain": RETAINED,
            "target_accept_prob": TARGET_ACCEPT,
            "initialization_rule_predeclared": "all chains initialize at the accepted Uniform-target theta_MAP",
            "drop_or_replace_chain": False,
            "automatic_retries": 0,
        },
        "new_fe_calls": 0,
        "surrogate_training_calls": 0,
    }
    write_json(manifest_path, manifest)

    module = load_classes()
    surrogate64 = module.Surrogate().double().cpu()
    surrogate64.load_state_dict(torch.load(SURROGATE_PATH, map_location="cpu", weights_only=True))
    surrogate64.eval()
    y64 = torch.tensor(Y6_DECIMAL, dtype=torch.float32).double()

    def value_grad(theta_np):
        theta = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)
        pred = surrogate64(theta.unsqueeze(0)).squeeze(0)[:6]
        value = 0.5 / KAPPA_SQ * (((pred - y64) / y64) ** 2).sum()
        grad = torch.autograd.grad(value, theta)[0]
        return float(value.detach()), grad.detach().numpy().astype(np.float64)

    grid = [0.55, 2.25, 3.95]
    starts = np.asarray([[a, b, c] for a in grid for b in grid for c in grid], dtype=np.float64)
    rng = np.random.default_rng(20260919)
    starts = np.vstack([starts, rng.uniform(LO, HI, size=(5, 3))])
    assert starts.shape == (32, 3)
    map_records = []
    for index, start in enumerate(starts):
        result = minimize(
            fun=lambda x: value_grad(x), x0=start, jac=True,
            method="L-BFGS-B", bounds=[(LO, HI)] * 3,
            options={"maxiter": 1000, "ftol": 1e-15, "gtol": 1e-10, "maxls": 50},
        )
        map_records.append({
            "start_index": index,
            "theta_initial": start.tolist(),
            "theta_final": result.x.tolist(),
            "negative_log_likelihood": float(result.fun),
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "projected_gradient_reported": np.asarray(result.jac, dtype=float).tolist(),
            "within_bounds": bool(np.all(result.x >= LO) and np.all(result.x <= HI)),
        })
    map_records.sort(key=lambda r: r["negative_log_likelihood"])
    best = map_records[0]
    theta_map = np.asarray(best["theta_final"], dtype=np.float64)
    replicated = [
        r for r in map_records[1:]
        if abs(r["negative_log_likelihood"] - best["negative_log_likelihood"]) <= 1e-8
        and np.max(np.abs(np.asarray(r["theta_final"]) - theta_map)) <= 1e-4
    ]
    map_gate = bool(best["success"] and best["within_bounds"] and len(replicated) >= 1)
    with torch.no_grad():
        predicted = surrogate64(torch.tensor(theta_map).unsqueeze(0)).squeeze(0).numpy()
    map_output = {
        "optimization_gate": "PASS" if map_gate else "FAIL",
        "theta_MAP_uniform": theta_map.tolist(),
        "negative_log_likelihood": best["negative_log_likelihood"],
        "rms_relative_error_pct_first6": float(np.sqrt(np.mean(((predicted[:6] - y64.numpy()) / y64.numpy()) ** 2)) * 100),
        "surrogate_at_map_20_modes": predicted.tolist(),
        "replicated_candidates_excluding_best": len(replicated),
        "all_starts": map_records,
    }
    write_json(OUT / "COUPLED_UNIFORM_MAP_OPTIMIZATION.json", map_output)
    print("COUPLED_UNIFORM_MAP_GATE", map_output["optimization_gate"], theta_map.tolist(), best["negative_log_likelihood"], len(replicated), flush=True)
    if not map_gate:
        return

    surrogate32 = module.Surrogate().float().cpu()
    surrogate32.load_state_dict(torch.load(SURROGATE_PATH, map_location="cpu", weights_only=True))
    surrogate32.eval()
    y32 = torch.tensor(Y6_DECIMAL, dtype=torch.float32)
    scaled = np.clip((theta_map - LO) / WIDTH, 1e-6, 1 - 1e-6)
    z_init_np = np.log(scaled) - np.log1p(-scaled)
    z_init = torch.tensor(z_init_np, dtype=torch.float32)

    def potential(params):
        z = params["z"]
        s = torch.sigmoid(z)
        theta = LO + WIDTH * s
        pred = surrogate32(theta.unsqueeze(0)).squeeze(0)[:6]
        nll = 0.5 / KAPPA_SQ * (((pred - y32) / y32) ** 2).sum()
        log_jacobian = (math.log(WIDTH) + F.logsigmoid(z) + F.logsigmoid(-z)).sum()
        return nll - log_jacobian

    torch.set_num_threads(6)
    chains, records = [], []
    for chain_index, seed in enumerate(SEEDS):
        npz_path = OUT / f"COUPLED_UNIFORM_NUTS_CHAIN_seed{seed}.npz"
        json_path = OUT / f"COUPLED_UNIFORM_NUTS_CHAIN_seed{seed}.json"
        if npz_path.exists() or json_path.exists():
            raise RuntimeError(f"Refusing to overwrite NUTS seed {seed}")
        torch.manual_seed(seed)
        kernel = NUTS(potential_fn=potential, target_accept_prob=TARGET_ACCEPT)
        sampler = MCMC(
            kernel, num_samples=RETAINED, warmup_steps=WARMUP,
            initial_params={"z": z_init.clone()}, disable_progbar=True,
        )
        started = now()
        t0 = time.time()
        sampler.run()
        runtime = time.time() - t0
        z = sampler.get_samples()["z"].detach().cpu()
        theta = (LO + WIDTH * torch.sigmoid(z)).numpy()
        diagnostics = sampler.diagnostics()
        divergences = [int(i) for i in diagnostics["divergences"]["chain 0"]]
        acceptance = float(diagnostics["acceptance rate"]["chain 0"])
        assert theta.shape == (RETAINED, 3)
        assert np.isfinite(theta).all() and (theta >= LO).all() and (theta <= HI).all()
        np.savez_compressed(
            npz_path, theta=theta, z=z.numpy(), seed=np.asarray(seed),
            theta_init=theta_map, z_init=z_init_np,
            divergences=np.asarray(divergences, dtype=np.int64),
            acceptance_rate=np.asarray(acceptance), step_size=np.asarray(float(kernel.step_size)),
        )
        record = {
            "chain_index": chain_index,
            "seed": seed,
            "started_utc": started,
            "finished_utc": now(),
            "runtime_s": runtime,
            "acceptance_rate": acceptance,
            "divergence_count": len(divergences),
            "divergence_indices": divergences,
            "adapted_step_size": float(kernel.step_size),
            "inverse_mass_matrix": clean(kernel.inverse_mass_matrix),
            "raw_diagnostics": clean(diagnostics),
            "artifact_path": str(npz_path),
            "artifact_sha256": sha(npz_path),
        }
        write_json(json_path, record)
        chains.append(theta)
        records.append(record)
        print(f"COUPLED_UNIFORM_NUTS_CHAIN seed={seed} runtime={runtime:.2f}s acceptance={acceptance:.6f} divergences={len(divergences)}", flush=True)
    chains = np.stack(chains)
    combined = OUT / "COUPLED_UNIFORM_NUTS_REFERENCE.npz"
    np.savez_compressed(
        combined, chains=chains, pooled=chains.reshape(-1, 3), seeds=np.asarray(SEEDS),
        theta_map=theta_map, z_init=z_init_np,
        acceptance_rates=np.asarray([r["acceptance_rate"] for r in records]),
        divergence_counts=np.asarray([r["divergence_count"] for r in records]),
    )
    write_json(OUT / "COUPLED_UNIFORM_NUTS_RAW.json", {
        "status": "COMPLETED_ALL_PREDECLARED_CHAINS",
        "records": records,
        "combined_path": str(combined),
        "combined_sha256": sha(combined),
        "shape": list(chains.shape),
        "total_divergences": int(sum(r["divergence_count"] for r in records)),
        "theta_support_min": chains.min(axis=(0, 1)).tolist(),
        "theta_support_max": chains.max(axis=(0, 1)).tolist(),
        "theta_above_4_count": int((chains > 4).sum()),
        "theta_below_0_5_count": int((chains < 0.5).sum()),
        "no_chain_dropped_or_replaced": True,
        "automatic_retries": 0,
    })
    print("COUPLED_UNIFORM_NUTS_ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

