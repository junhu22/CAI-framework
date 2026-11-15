import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance

ROOT = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
OUT = ROOT / "revision_experiments/cai_r2_presubmit_phaseB_20260919_213647/COUPLED_UNIFORM"
SOURCE = ROOT / "experiments/coupled_building/04_train_cnf_coupled.py"
SURROGATE_PATH = ROOT / "surrogate_models_coupled/member_0.pt"
OLD_RESULTS = ROOT / "results/coupled_building/coupled_results.json"
REFERENCE_PATH = OUT / "COUPLED_UNIFORM_NUTS_REFERENCE.npz"
Y6 = [2.06, 2.30, 2.62, 6.14, 6.53, 6.76]
KAPPA_SQ = 0.03 ** 2
LO, HI = 0.5, 4.0
SEEDS = [42, 43, 44, 45, 46]
N = 2000
FLOWS = [
    (42, "primary", "conditional_flow_coupled_uniform_primary.pt"),
    (202, "secondary", "conditional_flow_coupled_uniform_secondary_seed202.pt"),
    (303, "secondary", "conditional_flow_coupled_uniform_secondary_seed303.pt"),
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_source():
    sys.path.insert(0, str(ROOT / "experiments/coupled_building"))
    spec = importlib.util.spec_from_file_location("coupled_uniform_fast_source", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log_target(surrogate, theta, observed_first6):
    pred = surrogate(theta)[:, :6]
    residual = (pred - observed_first6) / observed_first6
    ll = -0.5 / KAPPA_SQ * (residual ** 2).sum(dim=-1)
    valid = ((theta >= LO) & (theta <= HI)).all(dim=-1)
    result = torch.full_like(ll, -torch.inf)
    result[valid] = ll[valid] - 3.0 * math.log(HI - LO)
    return result


def raw_ess(logw):
    finite = np.isfinite(logw)
    if not finite.any():
        return 0.0
    shifted = logw[finite] - np.max(logw[finite])
    weights = np.zeros_like(logw, dtype=np.float64)
    weights[finite] = np.exp(shifted)
    weights /= weights.sum()
    return float(1.0 / np.sum(weights ** 2) / len(weights))


def main():
    raw_path = OUT / "COUPLED_UNIFORM_FAST_ROUTING_RAW.npz"
    if raw_path.exists():
        raise RuntimeError("Refusing to overwrite coupled fast/routing output")
    module = load_source()
    surrogate = module.Surrogate().float().cpu()
    surrogate.load_state_dict(torch.load(SURROGATE_PATH, map_location="cpu", weights_only=True))
    surrogate.eval()
    map_result = json.loads((OUT / "COUPLED_UNIFORM_MAP_OPTIMIZATION.json").read_text(encoding="utf-8"))
    assert map_result["optimization_gate"] == "PASS"
    y_field = np.asarray(Y6 + map_result["surrogate_at_map_20_modes"][6:20], dtype=np.float32)
    old = json.loads(OLD_RESULTS.read_text(encoding="utf-8"))
    y_damage = np.asarray(old["damage_scenario"]["y_full"], dtype=np.float32)
    assert y_damage.shape == (20,)
    scenarios = [("field", y_field), ("damage", y_damage)]
    with np.load(REFERENCE_PATH, allow_pickle=False) as a:
        reference = a["chains"].reshape(-1, 3).astype(np.float64)
    ref_mean = reference.mean(0)
    ref_q = np.quantile(reference, [0.025, 0.975], axis=0)

    latents = np.empty((5, N, 3), dtype=np.float32)
    for si, seed in enumerate(SEEDS):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        latents[si] = torch.randn((N, 3), dtype=torch.float32, generator=gen).numpy()
    all_theta = np.empty((3, 2, 5, N, 3), dtype=np.float32)
    all_logw = np.empty((3, 2, 5, N), dtype=np.float64)
    rows = []
    for fi, (training_seed, role, checkpoint_name) in enumerate(FLOWS):
        checkpoint = OUT / checkpoint_name
        flow = module.ConditionalRealNVP(dim_theta=3, dim_y=20, n_layers=8, hidden_dim=128).cpu()
        flow.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        flow.eval()
        for sci, (scenario, y_np) in enumerate(scenarios):
            y = torch.tensor(y_np, dtype=torch.float32)
            observed = y[:6]
            for si, seed in enumerate(SEEDS):
                z = torch.from_numpy(latents[si])
                ybatch = y.unsqueeze(0).expand(N, -1)
                t0 = time.perf_counter()
                with torch.no_grad():
                    theta, _ = flow.inverse(z, ybatch)
                inference_ms = (time.perf_counter() - t0) * 1000.0
                with torch.no_grad():
                    target = log_target(surrogate, theta, observed)
                    logq = flow.log_prob(theta, y)
                    logw = (target - logq).double().numpy()
                theta_np = theta.numpy().astype(np.float64)
                valid = ((theta_np >= LO) & (theta_np <= HI)).all(axis=1)
                valid_theta = theta_np[valid]
                all_theta[fi, sci, si] = theta.numpy()
                all_logw[fi, sci, si] = logw
                row = {
                    "flow_training_seed": training_seed,
                    "flow_role": role,
                    "checkpoint": checkpoint_name,
                    "checkpoint_sha256": sha(checkpoint),
                    "scenario": scenario,
                    "latent_seed": seed,
                    "N_proposals": N,
                    "inference_time_ms": inference_ms,
                    "in_support_count": int(valid.sum()),
                    "in_support_fraction": float(valid.mean()),
                    "below_0_5_count": int((theta_np < LO).any(axis=1).sum()),
                    "above_4_count": int((theta_np > HI).any(axis=1).sum()),
                    "raw_ess_ratio": raw_ess(logw),
                }
                for j in range(3):
                    row[f"fast_all_mean_e{j+1}"] = float(theta_np[:, j].mean())
                    row[f"fast_all_std_e{j+1}"] = float(theta_np[:, j].std(ddof=1))
                    row[f"fast_in_support_mean_e{j+1}"] = float(valid_theta[:, j].mean()) if len(valid_theta) else None
                    row[f"fast_in_support_std_e{j+1}"] = float(valid_theta[:, j].std(ddof=1)) if len(valid_theta) > 1 else None
                if scenario == "field" and len(valid_theta):
                    q = np.quantile(valid_theta, [0.025, 0.975], axis=0)
                    w1 = np.asarray([wasserstein_distance(valid_theta[:, j], reference[:, j]) for j in range(3)])
                    ratios = (q[1] - q[0]) / (ref_q[1] - ref_q[0])
                    row.update(
                        candidate_reference_mean_mae=float(np.abs(valid_theta.mean(0) - ref_mean).mean()),
                        candidate_reference_w1_mean=float(w1.mean()),
                        candidate_reference_w1_max=float(w1.max()),
                        candidate_reference_width_ratio_mean=float(ratios.mean()),
                        candidate_reference_width_ratio_min=float(ratios.min()),
                        candidate_reference_width_ratio_max=float(ratios.max()),
                        reference_qualified=False,
                    )
                rows.append(row)
    np.savez_compressed(
        raw_path,
        theta=all_theta,
        log_weights=all_logw,
        latent_z=latents,
        flow_training_seeds=np.asarray([f[0] for f in FLOWS]),
        scenario_names=np.asarray([s[0] for s in scenarios]),
        latent_seeds=np.asarray(SEEDS),
        y_field=y_field,
        y_damage=y_damage,
        reference_candidate=reference,
    )
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / "COUPLED_UNIFORM_FAST_ROUTING_BASE.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "paired_latent_protocol": "each seed's base z tensor generated once and reused for all three training seeds and both scenarios",
        "latent_seeds": SEEDS,
        "N": N,
        "field_padding": "measured first six plus frozen surrogate(theta_MAP_uniform) modes 7-20",
        "theta_MAP_uniform": map_result["theta_MAP_uniform"],
        "damage_observation_source": str(OLD_RESULTS),
        "damage_observation_sha256": sha(OLD_RESULTS),
        "damage_forward_data_regenerated": False,
        "target": "bounded Uniform(0.5,4)^3; target log density -inf outside",
        "reference_status": "UNQUALIFIED_CANDIDATE because NUTS max rank-Rhat exceeded 1.01",
        "surrogate_sha256": sha(SURROGATE_PATH),
        "new_fe_calls": 0,
    }
    (OUT / "COUPLED_UNIFORM_FAST_ROUTING_MANIFEST.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("COUPLED_UNIFORM_FAST_ROUTING_BASE_COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    main()

