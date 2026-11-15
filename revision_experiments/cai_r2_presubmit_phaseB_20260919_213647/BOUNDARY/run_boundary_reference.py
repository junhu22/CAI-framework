import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyro
import torch
from pyro.infer import MCMC, NUTS

ROOT = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
OUT = ROOT / "revision_experiments/cai_r2_presubmit_phaseB_20260919_213647/BOUNDARY"
SOURCE = ROOT / "revision_experiments/cai_r2_stage1/final_ext_verification_20260918_091227"
ROUTING = ROOT / "revision_experiments/cai_r2_stage1/run_20260917_094722_21d144f3db844a6b8a2e8167c5282dbe/routing_arrays"
SURROGATE_PATH = ROOT / "surrogate_models_broad_ext/member_0.pt"
SEEDS = [711, 712, 713, 714]
NUM_SAMPLES = 600
WARMUP = 500
TARGET_ACCEPT = 0.95
SIGMA_LOG = 0.55
KAPPA_SQ = 0.001
THETA_INIT = np.exp(np.linspace(0.20, -0.20, 6)).astype(np.float64)

sys.path.insert(0, str(ROOT / "experiments"))
from cai_lib import load_surrogate  # noqa: E402


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


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "BOUNDARY_REFERENCE_RERUN_MANIFEST.json"
    if manifest_path.exists():
        raise RuntimeError("Refusing to overwrite an existing boundary run")
    source_manifest = json.loads((SOURCE / "reference_quality_4chain/REFERENCE4_sailboat_boundary_e1_075_RUN.json").read_text(encoding="utf-8"))["target_manifest"]
    with np.load(ROUTING / "sailboat_boundary_e1_075_seed42_N8000.npz", allow_pickle=False) as a:
        observation = a["observation"].astype(np.float64)
    assert observation.shape == (20,)
    assert np.array_equal(observation.astype(np.float32), np.asarray(source_manifest["observation_used_float32"], dtype=np.float32))
    assert sha(SURROGATE_PATH).lower() == "4821100192a768327212ed724090732e63483650f4e80421f89a8df3f984f447"
    for seed in [42, 43, 44, 45, 46]:
        with np.load(ROUTING / f"sailboat_boundary_e1_075_seed{seed}_N2000.npz", allow_pickle=False) as a:
            assert np.array_equal(a["observation"].astype(np.float32), observation.astype(np.float32))
    manifest = {
        "created_utc_before_sampling": now(),
        "case": "sailboat_boundary_e1_075",
        "scientific_target": {
            "observation": observation.tolist(),
            "observation_source": str(ROUTING / "sailboat_boundary_e1_075_seed42_N8000.npz"),
            "surrogate_path": str(SURROGATE_PATH),
            "surrogate_sha256": sha(SURROGATE_PATH),
            "likelihood": "relative-error Gaussian potential, kappa_sq=0.001, K=20",
            "prior": "independent untruncated LogNormal(log_mean=0, log_sigma=0.55)",
            "parameterization": "z=log(theta); LogNormal Jacobian cancels in z target",
        },
        "sampler": {
            "implementation": "Pyro NUTS",
            "target_accept_prob": TARGET_ACCEPT,
            "warmup_steps": WARMUP,
            "retained_per_chain": NUM_SAMPLES,
            "seeds_predeclared": SEEDS,
            "theta_init_all_chains": THETA_INIT.tolist(),
            "automatic_retries": 0,
            "drop_or_replace_chain": False,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "pyro": pyro.__version__,
            "device": "cpu",
            "torch_threads": 6,
        },
        "routing_samples_unchanged": True,
        "new_fast_or_routing_draws": 0,
    }
    write_json(manifest_path, manifest)

    torch.set_num_threads(6)
    surrogate = load_surrogate(SURROGATE_PATH)
    y = torch.tensor(observation, dtype=torch.float32)
    init = torch.log(torch.tensor(THETA_INIT, dtype=torch.float32))

    def potential(params):
        z = params["z"]
        f = surrogate(torch.exp(z).unsqueeze(0)).squeeze(0)[:20]
        nll = 0.5 * (((f - y) / y) ** 2).sum() / KAPPA_SQ
        neg_log_prior_z = 0.5 * (z ** 2).sum() / (SIGMA_LOG ** 2)
        return nll + neg_log_prior_z

    chains = []
    run_records = []
    for chain_index, seed in enumerate(SEEDS):
        chain_npz = OUT / f"BOUNDARY_CHAIN_seed{seed}.npz"
        chain_json = OUT / f"BOUNDARY_CHAIN_seed{seed}.json"
        if chain_npz.exists() or chain_json.exists():
            raise RuntimeError(f"Refusing to overwrite chain seed {seed}")
        torch.manual_seed(seed)
        kernel = NUTS(potential_fn=potential, target_accept_prob=TARGET_ACCEPT)
        mcmc = MCMC(
            kernel,
            num_samples=NUM_SAMPLES,
            warmup_steps=WARMUP,
            initial_params={"z": init.clone()},
            disable_progbar=True,
        )
        started = now()
        t0 = time.time()
        mcmc.run()
        runtime = time.time() - t0
        theta = torch.exp(mcmc.get_samples()["z"]).detach().cpu().numpy()
        diagnostics = mcmc.diagnostics()
        divergences = [int(i) for i in diagnostics["divergences"]["chain 0"]]
        acceptance = float(diagnostics["acceptance rate"]["chain 0"])
        step_size = float(kernel.step_size)
        inv_mass = clean(kernel.inverse_mass_matrix)
        assert theta.shape == (NUM_SAMPLES, 6)
        assert np.isfinite(theta).all() and (theta > 0).all()
        np.savez_compressed(
            chain_npz,
            theta=theta,
            seed=np.asarray(seed),
            theta_init=THETA_INIT,
            observation=observation,
            divergences=np.asarray(divergences, dtype=np.int64),
            acceptance_rate=np.asarray(acceptance),
            step_size=np.asarray(step_size),
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
            "adapted_step_size": step_size,
            "inverse_mass_matrix": inv_mass,
            "raw_diagnostics": clean(diagnostics),
            "artifact_path": str(chain_npz),
            "artifact_sha256": sha(chain_npz),
        }
        write_json(chain_json, record)
        run_records.append(record)
        chains.append(theta)
        print(f"BOUNDARY_CHAIN_COMPLETE seed={seed} runtime={runtime:.2f}s acceptance={acceptance:.6f} divergences={len(divergences)}", flush=True)

    chains = np.stack(chains, axis=0)
    combined_path = OUT / "BOUNDARY_REFERENCE_RERUN.npz"
    np.savez_compressed(
        combined_path,
        chains=chains,
        pooled=chains.reshape(-1, 6),
        seeds=np.asarray(SEEDS),
        theta_init=THETA_INIT,
        observation=observation,
        acceptance_rates=np.asarray([r["acceptance_rate"] for r in run_records]),
        divergence_counts=np.asarray([r["divergence_count"] for r in run_records]),
    )
    write_json(
        OUT / "BOUNDARY_REFERENCE_RERUN_RAW.json",
        {
            "status": "COMPLETED_ALL_PREDECLARED_CHAINS",
            "chains_shape": list(chains.shape),
            "run_records": run_records,
            "combined_artifact_path": str(combined_path),
            "combined_artifact_sha256": sha(combined_path),
            "total_divergences": int(sum(r["divergence_count"] for r in run_records)),
            "no_chain_dropped_or_replaced": True,
            "automatic_retries": 0,
        },
    )
    print("BOUNDARY_ALL_CHAINS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

