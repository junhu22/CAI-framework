import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import platformdirs


# ArviZ 0.23.4 writes a once-daily warning stamp at import. Redirect only that
# cache lookup to the writable system temporary directory; statistical behavior
# is unchanged.
_platformdirs_user_cache_dir = platformdirs.user_cache_dir
platformdirs.user_cache_dir = lambda *args, **kwargs: str(
    Path(tempfile.gettempdir()) / "cai_r2_phase_c1_arviz_cache"
)
import arviz as az

platformdirs.user_cache_dir = _platformdirs_user_cache_dir

import pyro
import torch
import torch.nn.functional as F
from pyro.infer import MCMC, NUTS
from scipy.stats import wasserstein_distance


ROOT = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
OUT = ROOT / "revision_experiments/cai_r2_presubmit_phaseB_20260919_213647/COUPLED_UNIFORM"
SOURCE = ROOT / "experiments/coupled_building/04_train_cnf_coupled.py"
SURROGATE_PATH = ROOT / "surrogate_models_coupled/member_0.pt"
PRIMARY_CNF = OUT / "conditional_flow_coupled_uniform_primary.pt"
OLD_MANIFEST = OUT / "COUPLED_UNIFORM_MAP_NUTS_MANIFEST.json"
OLD_MAP = OUT / "COUPLED_UNIFORM_MAP_OPTIMIZATION.json"
OLD_REFERENCE = OUT / "COUPLED_UNIFORM_NUTS_REFERENCE.npz"
FAST_RAW = OUT / "COUPLED_UNIFORM_FAST_ROUTING_RAW.npz"
FAST_ROUTING = OUT / "COUPLED_UNIFORM_FAST_ROUTING.csv"
FAST_SUMMARY = OUT / "COUPLED_UNIFORM_FAST_ROUTING_SUMMARY.json"

REPORT_PATH = OUT / "PHASE_C1_COUPLED_EXTENSION_REPORT.md"
GATE_PATH = OUT / "COUPLED_NUTS_2000_GATE.json"
DIAGNOSTICS_PATH = OUT / "COUPLED_NUTS_2000_DIAGNOSTICS.csv"
PREFIX_PATH = OUT / "COUPLED_PREFIX_REPRODUCIBILITY.json"
REFERENCE_2000_PATH = OUT / "COUPLED_UNIFORM_NUTS_REFERENCE_2000.npz"
GAP_PATH = OUT / "COUPLED_QUALIFIED_GAP.csv"
QUALIFIED_AUDIT_PATH = OUT / "COUPLED_QUALIFIED_REFERENCE_AUDIT.md"

PARAMETERS = ["e1", "e2", "e3"]
NUTS_SEEDS = [811, 812, 813, 814]
LATENT_SEEDS = [42, 43, 44, 45, 46]
WARMUP = 500
OLD_RETAINED = 600
RETAINED = 2000
TARGET_ACCEPT = 0.90
LO, HI = 0.5, 4.0
WIDTH = HI - LO
KAPPA_SQ = 0.03**2
Y6_DECIMAL = [2.06, 2.30, 2.62, 6.14, 6.53, 6.76]

EXPECTED_SURROGATE_SHA256 = "6a1e4f530d4e3628e0b7fdbf08661b4dc32a71a0c2b538b26f5af9d7c07b9986"
EXPECTED_PRIMARY_CNF_SHA256 = "690d5dbd8ae507e10b7393590721faa7b78cb1d24919330062429e12d4381cc3"
EXPECTED_FAST_RAW_SHA256 = "0d306da8c48422425659481066cc6ce430931d2acb8faaaeb8f95e67d16f176e"
EXPECTED_OLD_REFERENCE_SHA256 = "8b2720358be4974a96a3e1467b1749193eac4271929873c67e82e9cc25ab9eee"
EXPECTED_OLD_CHAIN_SHA256 = {
    811: "881fb6e18aa1ce82f5e1ce6eb72b169b43f289651b28c18d586cd2e2810875e8",
    812: "8446f3e19db08a18f14fa504e1b49e15448af4be3c3014c9db8b865dd68bf07d",
    813: "18b226b670abebd42019e60d1c7943ec73b3e5e866b8570a03bf5e6c1228de2b",
    814: "755c4a764d54413a63f05dd8295966278d0273c13480efa4de79664b3da6dd18",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def json_text(value):
    return json.dumps(clean(value), indent=2, allow_nan=False) + "\n"


def write_text_exclusive(path, text):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        handle.write(text)


def write_csv_exclusive(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clean(rows))


def load_source_module():
    sys.path.insert(0, str(ROOT / "experiments/coupled_building"))
    spec = importlib.util.spec_from_file_location("coupled_phase_c1_source", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bitwise_equal(left, right):
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def exact_max_abs_difference(left, right):
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def in_support(theta):
    return ((theta >= LO) & (theta <= HI)).all(axis=-1)


def md_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def format_vector(values, digits=8):
    return "[" + ", ".join(f"{float(value):.{digits}f}" for value in values) + "]"


def gap_metrics(fast, reference):
    fast = np.asarray(fast, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if fast.ndim != 2 or fast.shape[1] != 3 or len(fast) < 2:
        raise ValueError(f"Invalid fast array shape {fast.shape}")
    reference_mean = reference.mean(axis=0)
    fast_mean = fast.mean(axis=0)
    absolute_mean_error = np.abs(fast_mean - reference_mean)
    w1 = np.asarray(
        [wasserstein_distance(fast[:, j], reference[:, j]) for j in range(3)],
        dtype=np.float64,
    )
    fast_q = np.quantile(fast, [0.025, 0.5, 0.975], axis=0)
    reference_q = np.quantile(reference, [0.025, 0.5, 0.975], axis=0)
    reference_width = reference_q[2] - reference_q[0]
    if not (reference_width > 0).all():
        raise ValueError("Qualified reference has a non-positive 95% interval width")
    width_ratio = (fast_q[2] - fast_q[0]) / reference_width
    result = {
        "active_mean_mae": float(absolute_mean_error.mean()),
        "all_mean_mae": float(absolute_mean_error.mean()),
        "w1_1d_mean": float(w1.mean()),
        "w1_1d_max": float(w1.max()),
        "q95_width_ratio_mean": float(width_ratio.mean()),
        "q95_width_ratio_min": float(width_ratio.min()),
        "q95_width_ratio_max": float(width_ratio.max()),
    }
    for j, parameter in enumerate(PARAMETERS):
        result.update(
            {
                f"fast_mean_{parameter}": float(fast_mean[j]),
                f"reference_mean_{parameter}": float(reference_mean[j]),
                f"mean_abs_error_{parameter}": float(absolute_mean_error[j]),
                f"w1_{parameter}": float(w1[j]),
                f"fast_q025_{parameter}": float(fast_q[0, j]),
                f"fast_q50_{parameter}": float(fast_q[1, j]),
                f"fast_q975_{parameter}": float(fast_q[2, j]),
                f"reference_q025_{parameter}": float(reference_q[0, j]),
                f"reference_q50_{parameter}": float(reference_q[1, j]),
                f"reference_q975_{parameter}": float(reference_q[2, j]),
                f"q95_width_ratio_{parameter}": float(width_ratio[j]),
            }
        )
    return result


def make_gap_row(
    scope,
    latent_seed,
    population,
    fast_used,
    raw_proposals,
    reference,
    routing,
    reference_sha,
):
    raw_proposals = np.asarray(raw_proposals, dtype=np.float64)
    valid = in_support(raw_proposals)
    outside = ~valid
    below = (raw_proposals < LO).any(axis=1)
    above = (raw_proposals > HI).any(axis=1)
    row = {
        "scope": scope,
        "flow_training_seed": 42,
        "flow_role": "primary",
        "scenario": "field",
        "latent_seed": latent_seed,
        "analysis_population": population,
        "fast_distribution_label": "q_phi proposal",
        "reference_distribution_label": "bounded-Uniform NUTS posterior",
        "active_coordinates": "e1;e2;e3",
        "all_coordinates": "e1;e2;e3",
        "N_fast_proposals_raw": int(len(raw_proposals)),
        "N_fast_used_for_gap": int(len(fast_used)),
        "N_reference": int(len(reference)),
        "field_proposal_in_support_count": int(valid.sum()),
        "field_proposal_support_leakage_count": int(outside.sum()),
        "field_proposal_support_leakage_percentage": float(100.0 * outside.mean()),
        "field_proposal_below_0_5_count": int(below.sum()),
        "field_proposal_above_4_count": int(above.sum()),
        "support_leakage_counts_are_unique_for_total": True,
        "outside_support_handling": (
            "retained in q_phi proposal comparison"
            if population == "all_q_phi_proposals"
            else "excluded only for explicitly labelled in-support proposal subset; not relabelled posterior"
        ),
        "raw_ess_ratio_reused": routing["raw_ess_ratio"],
        "psis_ess_ratio_reused": routing["psis_ess_ratio"],
        "pareto_k_reused": routing["pareto_k"],
        "raw_route_tau05_reused": routing["raw_route_tau05"],
        "psis_route_tau05_reused": routing["psis_route_tau05"],
        "routing_values_recomputed": False,
        "fast_draws_regenerated": False,
        "qualified_reference_sha256": reference_sha,
    }
    row.update(gap_metrics(fast_used, reference))
    return row


def preflight():
    output_paths = [
        REPORT_PATH,
        GATE_PATH,
        DIAGNOSTICS_PATH,
        PREFIX_PATH,
        REFERENCE_2000_PATH,
        GAP_PATH,
        QUALIFIED_AUDIT_PATH,
    ]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise RuntimeError(f"Refusing to rerun or overwrite Phase C1 outputs: {existing}")

    if az.__version__ != "0.23.4":
        raise RuntimeError(f"Expected ArviZ 0.23.4, found {az.__version__}")
    if pyro.__version__ != "1.9.1":
        raise RuntimeError(f"Expected Pyro 1.9.1, found {pyro.__version__}")
    if torch.__version__ != "2.6.0+cu124":
        raise RuntimeError(f"Expected torch 2.6.0+cu124, found {torch.__version__}")

    expected_hashes = {
        SURROGATE_PATH: EXPECTED_SURROGATE_SHA256,
        PRIMARY_CNF: EXPECTED_PRIMARY_CNF_SHA256,
        FAST_RAW: EXPECTED_FAST_RAW_SHA256,
        OLD_REFERENCE: EXPECTED_OLD_REFERENCE_SHA256,
    }
    for path, expected in expected_hashes.items():
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"Frozen input hash mismatch for {path}: {observed} != {expected}")
    for seed, expected in EXPECTED_OLD_CHAIN_SHA256.items():
        path = OUT / f"COUPLED_UNIFORM_NUTS_CHAIN_seed{seed}.npz"
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"Phase-B chain hash mismatch for seed {seed}: {observed} != {expected}")

    manifest = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    nuts = manifest["nuts"]
    expected_protocol = {
        "seeds_predeclared": NUTS_SEEDS,
        "warmup_steps": WARMUP,
        "retained_per_chain": OLD_RETAINED,
        "target_accept_prob": TARGET_ACCEPT,
        "drop_or_replace_chain": False,
        "automatic_retries": 0,
    }
    for key, expected in expected_protocol.items():
        if nuts[key] != expected:
            raise RuntimeError(f"Phase-B manifest mismatch for {key}: {nuts[key]} != {expected}")
    if manifest["theta_prior"] != "independent Uniform(0.5,4.0)^3":
        raise RuntimeError("Unexpected Phase-B prior")
    if "sum log|dtheta/dz| included" not in manifest["nuts_transform"]:
        raise RuntimeError("Phase-B transform/Jacobian declaration missing")

    map_result = json.loads(OLD_MAP.read_text(encoding="utf-8"))
    if map_result["optimization_gate"] != "PASS":
        raise RuntimeError("Frozen Phase-B MAP did not pass")
    theta_map = np.asarray(map_result["theta_MAP_uniform"], dtype=np.float64)
    scaled = np.clip((theta_map - LO) / WIDTH, 1e-6, 1 - 1e-6)
    z_init_recomputed = np.log(scaled) - np.log1p(-scaled)

    old_chains = {}
    with np.load(OLD_REFERENCE, allow_pickle=False) as data:
        reference_chains = data["chains"]
        reference_theta_map = data["theta_map"].astype(np.float64)
        reference_z_init = data["z_init"].astype(np.float64)
        reference_seeds = data["seeds"].astype(int)
    if reference_chains.shape != (4, OLD_RETAINED, 3):
        raise RuntimeError(f"Unexpected Phase-B reference shape {reference_chains.shape}")
    if reference_seeds.tolist() != NUTS_SEEDS:
        raise RuntimeError("Unexpected Phase-B reference seeds")
    if not np.array_equal(reference_theta_map, theta_map):
        raise RuntimeError("Frozen MAP and Phase-B reference initialization differ")
    if not np.allclose(reference_z_init, z_init_recomputed, rtol=0.0, atol=1e-15):
        raise RuntimeError("Frozen z initialization does not match the declared transform")

    for chain_index, seed in enumerate(NUTS_SEEDS):
        path = OUT / f"COUPLED_UNIFORM_NUTS_CHAIN_seed{seed}.npz"
        with np.load(path, allow_pickle=False) as data:
            old_theta = data["theta"].copy()
            old_z = data["z"].copy()
            old_seed = int(data["seed"])
            old_theta_init = data["theta_init"].astype(np.float64)
            old_z_init = data["z_init"].astype(np.float64)
            old_step_size = float(data["step_size"])
        if old_seed != seed or old_theta.shape != (OLD_RETAINED, 3) or old_z.shape != (OLD_RETAINED, 3):
            raise RuntimeError(f"Unexpected old chain content for seed {seed}")
        if not bitwise_equal(old_theta, reference_chains[chain_index]):
            raise RuntimeError(f"Old combined reference differs from old seed-{seed} chain")
        if not np.array_equal(old_theta_init, theta_map) or not np.array_equal(old_z_init, reference_z_init):
            raise RuntimeError(f"Old seed-{seed} initialization mismatch")
        old_chains[seed] = {
            "theta": old_theta,
            "z": old_z,
            "step_size": old_step_size,
            "path": path,
            "sha256": sha256(path),
        }

    return {
        "manifest": manifest,
        "map_result": map_result,
        "theta_map": theta_map,
        "z_init": reference_z_init,
        "old_chains": old_chains,
        "input_hashes": {
            "phase_b_manifest": sha256(OLD_MANIFEST),
            "phase_b_map": sha256(OLD_MAP),
            "phase_b_reference": sha256(OLD_REFERENCE),
            "surrogate": sha256(SURROGATE_PATH),
            "source_module": sha256(SOURCE),
            "primary_uniform_cnf": sha256(PRIMARY_CNF),
            "fast_routing_raw": sha256(FAST_RAW),
            "fast_routing_csv": sha256(FAST_ROUTING),
            "fast_routing_summary": sha256(FAST_SUMMARY),
        },
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = preflight()
    module = load_source_module()

    surrogate32 = module.Surrogate().float().cpu()
    surrogate32.load_state_dict(torch.load(SURROGATE_PATH, map_location="cpu", weights_only=True))
    surrogate32.eval()
    y32 = torch.tensor(Y6_DECIMAL, dtype=torch.float32)
    theta_map = frozen["theta_map"]
    z_init_np = frozen["z_init"]
    z_init = torch.tensor(z_init_np, dtype=torch.float32)

    def potential(params):
        z = params["z"]
        sigmoid_z = torch.sigmoid(z)
        theta = LO + WIDTH * sigmoid_z
        prediction = surrogate32(theta.unsqueeze(0)).squeeze(0)[:6]
        negative_log_likelihood = 0.5 / KAPPA_SQ * (((prediction - y32) / y32) ** 2).sum()
        log_jacobian = (math.log(WIDTH) + F.logsigmoid(z) + F.logsigmoid(-z)).sum()
        return negative_log_likelihood - log_jacobian

    torch.set_num_threads(6)
    execution_started = now()
    wall_start = time.time()
    chains = []
    records = []
    prefix_records = []

    print(
        "PHASE_C1_START single_predeclared_run seeds=811,812,813,814 warmup=500 retained=2000 target_accept=0.90",
        flush=True,
    )
    for chain_index, seed in enumerate(NUTS_SEEDS):
        torch.manual_seed(seed)
        kernel = NUTS(potential_fn=potential, target_accept_prob=TARGET_ACCEPT)
        sampler = MCMC(
            kernel,
            num_samples=RETAINED,
            warmup_steps=WARMUP,
            initial_params={"z": z_init.clone()},
            disable_progbar=True,
        )
        chain_started = now()
        chain_wall_start = time.time()
        sampler.run()
        runtime_s = time.time() - chain_wall_start

        z = sampler.get_samples()["z"].detach().cpu().numpy()
        theta = (LO + WIDTH * torch.sigmoid(torch.from_numpy(z))).numpy()
        diagnostics = sampler.diagnostics()
        divergences = [int(index) for index in diagnostics["divergences"]["chain 0"]]
        acceptance_rate = float(diagnostics["acceptance rate"]["chain 0"])
        if theta.shape != (RETAINED, 3) or z.shape != (RETAINED, 3):
            raise RuntimeError(f"Unexpected new chain shape for seed {seed}: theta={theta.shape}, z={z.shape}")
        if not np.isfinite(theta).all() or not np.isfinite(z).all():
            raise RuntimeError(f"Non-finite sample in seed {seed}")

        old = frozen["old_chains"][seed]
        new_theta_prefix = theta[:OLD_RETAINED]
        new_z_prefix = z[:OLD_RETAINED]
        theta_array_equal = bool(np.array_equal(new_theta_prefix, old["theta"]))
        z_array_equal = bool(np.array_equal(new_z_prefix, old["z"]))
        theta_bitwise_equal = bool(bitwise_equal(new_theta_prefix, old["theta"]))
        z_bitwise_equal = bool(bitwise_equal(new_z_prefix, old["z"]))
        prefix_record = {
            "chain_index": chain_index,
            "seed": seed,
            "comparison_retained_draws": OLD_RETAINED,
            "old_chain_path": str(old["path"]),
            "old_chain_sha256": old["sha256"],
            "theta_exact_max_abs_difference": exact_max_abs_difference(new_theta_prefix, old["theta"]),
            "theta_exact_array_equality": theta_array_equal,
            "theta_bitwise_equality": theta_bitwise_equal,
            "z_exact_max_abs_difference": exact_max_abs_difference(new_z_prefix, old["z"]),
            "z_exact_array_equality": z_array_equal,
            "z_bitwise_equality": z_bitwise_equal,
            "old_adapted_step_size": old["step_size"],
            "new_adapted_step_size": float(kernel.step_size),
            "adapted_step_size_exact_difference": float(kernel.step_size - old["step_size"]),
            "full_prefix_bitwise_equality": bool(theta_bitwise_equal and z_bitwise_equal),
        }
        prefix_records.append(prefix_record)
        records.append(
            {
                "chain_index": chain_index,
                "seed": seed,
                "started_utc": chain_started,
                "finished_utc": now(),
                "runtime_s": runtime_s,
                "acceptance_rate": acceptance_rate,
                "divergence_count": len(divergences),
                "divergence_indices": divergences,
                "adapted_step_size": float(kernel.step_size),
                "inverse_mass_matrix": clean(kernel.inverse_mass_matrix),
                "raw_diagnostics": clean(diagnostics),
                "prefix": prefix_record,
            }
        )
        chains.append(theta)
        print(
            f"PHASE_C1_CHAIN_COMPLETE seed={seed} runtime_s={runtime_s:.3f} "
            f"acceptance={acceptance_rate:.6f} divergences={len(divergences)} "
            f"prefix_bitwise={prefix_record['full_prefix_bitwise_equality']} "
            f"prefix_theta_max_abs={prefix_record['theta_exact_max_abs_difference']:.17g}",
            flush=True,
        )

    execution_finished = now()
    total_runtime_s = time.time() - wall_start
    chains = np.stack(chains)
    if chains.shape != (4, RETAINED, 3):
        raise RuntimeError(f"Unexpected stacked chain shape {chains.shape}")

    all_prefix_bitwise = all(record["full_prefix_bitwise_equality"] for record in prefix_records)
    run_classification = "CHAIN_LENGTH_EXTENSION" if all_prefix_bitwise else "LONGER_PREDECLARED_RERUN"
    mismatch_reason = None
    if not all_prefix_bitwise:
        mismatch_reason = (
            "At least one new chain's first 600 retained (theta,z) states was not bitwise identical to its "
            "Phase-B counterpart under the longer declared sampling horizon. The result is retained and evaluated "
            "without rerun, as required."
        )
    prefix_output = {
        "phase": "CAI R2 Phase C1 coupled bounded-Uniform NUTS chain-length extension",
        "classification": run_classification,
        "classification_rule": (
            "CHAIN_LENGTH_EXTENSION iff every seed's first 600 retained theta and z arrays are bitwise identical "
            "to Phase B; otherwise LONGER_PREDECLARED_RERUN"
        ),
        "old_retained_per_chain": OLD_RETAINED,
        "new_retained_per_chain": RETAINED,
        "seeds": NUTS_SEEDS,
        "warmup_steps_unchanged": WARMUP,
        "target_accept_prob_unchanged": TARGET_ACCEPT,
        "initialization_unchanged": True,
        "phase_b_reference_path": str(OLD_REFERENCE),
        "phase_b_reference_sha256": frozen["input_hashes"]["phase_b_reference"],
        "all_prefix_bitwise_equal": all_prefix_bitwise,
        "mismatch_reason": mismatch_reason,
        "chains": prefix_records,
        "rerun_after_prefix_check": False,
    }

    chains64 = chains.astype(np.float64)
    inference_data = az.from_dict(
        posterior={"theta": chains64},
        coords={"parameter": PARAMETERS},
        dims={"theta": ["parameter"]},
    )
    rank_rhat = np.asarray(az.rhat(inference_data, method="rank")["theta"].values, dtype=float)
    bulk_ess = np.asarray(
        az.ess(inference_data, method="bulk", relative=False)["theta"].values,
        dtype=float,
    )
    tail_ess = np.asarray(
        az.ess(inference_data, method="tail", relative=False)["theta"].values,
        dtype=float,
    )
    pooled = chains64.reshape(-1, 3)
    quantiles = np.quantile(pooled, [0.025, 0.5, 0.975], axis=0)
    posterior_mean = pooled.mean(axis=0)
    posterior_std = pooled.std(axis=0, ddof=1)
    divergence_counts = np.asarray([record["divergence_count"] for record in records], dtype=int)
    acceptance_rates = np.asarray([record["acceptance_rate"] for record in records], dtype=float)
    outside_coordinate_count = int(((chains64 < LO) | (chains64 > HI)).sum())
    outside_draw_count = int((~in_support(chains64)).sum())
    below_coordinate_count = int((chains64 < LO).sum())
    above_coordinate_count = int((chains64 > HI).sum())
    gates = {
        "rank_normalized_split_rhat": bool(np.isfinite(rank_rhat).all() and rank_rhat.max() <= 1.01),
        "bulk_ess": bool(np.isfinite(bulk_ess).all() and bulk_ess.min() >= 400.0),
        "tail_ess": bool(np.isfinite(tail_ess).all() and tail_ess.min() >= 400.0),
        "divergences": int(divergence_counts.sum()) == 0,
        "bounded_support": outside_coordinate_count == 0,
    }
    passed = all(gates.values())
    gate_status = "PASS" if passed else "FAIL"

    gate_output = {
        "phase": "CAI R2 Phase C1 coupled bounded-Uniform NUTS chain-length extension",
        "overall_gate": gate_status,
        "run_classification": run_classification,
        "execution": {
            "single_predeclared_run_count": 1,
            "automatic_retries": 0,
            "chains_dropped_or_replaced": 0,
            "started_utc": execution_started,
            "finished_utc": execution_finished,
            "total_serial_wall_runtime_s": total_runtime_s,
        },
        "protocol": {
            "target": "independent Uniform(0.5,4.0)^3",
            "observation": Y6_DECIMAL,
            "likelihood": "relative-error Gaussian potential, kappa_sq=0.0009",
            "transform": "theta=0.5+3.5*sigmoid(z)",
            "jacobian": "sum(log(3.5)+logsigmoid(z)+logsigmoid(-z)) included exactly",
            "seeds": NUTS_SEEDS,
            "warmup_steps": WARMUP,
            "retained_per_chain": RETAINED,
            "target_accept_prob": TARGET_ACCEPT,
            "initialization": "frozen accepted Phase-B Uniform-target theta_MAP for every chain",
            "theta_map": theta_map.tolist(),
            "z_init": z_init_np.tolist(),
        },
        "gates": {
            "rank_normalized_split_rhat": {
                "threshold": "<= 1.01 for every parameter",
                "values": rank_rhat.tolist(),
                "observed_max": float(rank_rhat.max()),
                "pass": gates["rank_normalized_split_rhat"],
            },
            "bulk_ess": {
                "threshold": ">= 400 for every parameter",
                "values": bulk_ess.tolist(),
                "observed_min": float(bulk_ess.min()),
                "pass": gates["bulk_ess"],
            },
            "tail_ess": {
                "threshold": ">= 400 for every parameter",
                "values": tail_ess.tolist(),
                "observed_min": float(tail_ess.min()),
                "pass": gates["tail_ess"],
            },
            "divergences": {
                "threshold": "= 0",
                "value": int(divergence_counts.sum()),
                "pass": gates["divergences"],
            },
            "theta_outside_0_5_4": {
                "threshold": "= 0",
                "outside_coordinate_count": outside_coordinate_count,
                "outside_draw_count": outside_draw_count,
                "below_0_5_coordinate_count": below_coordinate_count,
                "above_4_coordinate_count": above_coordinate_count,
                "pass": gates["bounded_support"],
            },
        },
        "rank_rhat": rank_rhat.tolist(),
        "bulk_ess": bulk_ess.tolist(),
        "tail_ess": tail_ess.tolist(),
        "max_rhat": float(rank_rhat.max()),
        "min_bulk_ess": float(bulk_ess.min()),
        "min_tail_ess": float(tail_ess.min()),
        "divergences": int(divergence_counts.sum()),
        "acceptance_rates": acceptance_rates.tolist(),
        "mean_acceptance_rate": float(acceptance_rates.mean()),
        "posterior_mean": posterior_mean.tolist(),
        "posterior_std": posterior_std.tolist(),
        "posterior_quantiles": {
            "q025": quantiles[0].tolist(),
            "q50": quantiles[1].tolist(),
            "q975": quantiles[2].tolist(),
        },
        "theta_support_min": pooled.min(axis=0).tolist(),
        "theta_support_max": pooled.max(axis=0).tolist(),
        "theta_outside_0_5_4_coordinate_count": outside_coordinate_count,
        "theta_outside_0_5_4_draw_count": outside_draw_count,
        "chain_records": records,
        "prefix_reproducibility": {
            "classification": run_classification,
            "all_prefix_bitwise_equal": all_prefix_bitwise,
        },
        "software": {
            "python": sys.version,
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "pyro": pyro.__version__,
            "numpy": np.__version__,
            "arviz": az.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
        "frozen_input_sha256": frozen["input_hashes"],
        "qualified_reference": None,
        "post_gate_actions_performed": False,
    }

    diagnostics_rows = []
    for j, parameter in enumerate(PARAMETERS):
        row = {
            "parameter": parameter,
            "chains": 4,
            "retained_per_chain": RETAINED,
            "total_retained_draws": int(pooled.shape[0]),
            "rank_normalized_split_rhat": float(rank_rhat[j]),
            "bulk_ess": float(bulk_ess[j]),
            "tail_ess": float(tail_ess[j]),
            "posterior_mean": float(posterior_mean[j]),
            "posterior_std_ddof1": float(posterior_std[j]),
            "q025": float(quantiles[0, j]),
            "q50": float(quantiles[1, j]),
            "q975": float(quantiles[2, j]),
            "min_sample": float(pooled[:, j].min()),
            "max_sample": float(pooled[:, j].max()),
            "below_0_5_coordinate_count": int((chains64[:, :, j] < LO).sum()),
            "above_4_coordinate_count": int((chains64[:, :, j] > HI).sum()),
            "acceptance_rate_seed811": float(acceptance_rates[0]),
            "acceptance_rate_seed812": float(acceptance_rates[1]),
            "acceptance_rate_seed813": float(acceptance_rates[2]),
            "acceptance_rate_seed814": float(acceptance_rates[3]),
            "mean_acceptance_rate": float(acceptance_rates.mean()),
            "divergences_seed811": int(divergence_counts[0]),
            "divergences_seed812": int(divergence_counts[1]),
            "divergences_seed813": int(divergence_counts[2]),
            "divergences_seed814": int(divergence_counts[3]),
            "total_divergences": int(divergence_counts.sum()),
            "rank_rhat_gate_pass": gates["rank_normalized_split_rhat"],
            "bulk_ess_gate_pass": gates["bulk_ess"],
            "tail_ess_gate_pass": gates["tail_ess"],
            "divergence_gate_pass": gates["divergences"],
            "support_gate_pass": gates["bounded_support"],
            "overall_gate": gate_status,
            "run_classification": run_classification,
        }
        diagnostics_rows.append(row)

    gap_rows = None
    qualified_audit = None
    reference_temp_path = None
    reference_sha = None
    table14_rows = None
    table14_rms_relative_error_pct = None

    if passed:
        temporary = tempfile.NamedTemporaryFile(
            prefix="coupled_uniform_nuts_reference_2000_",
            suffix=".npz",
            dir=Path(tempfile.gettempdir()),
            delete=False,
        )
        reference_temp_path = Path(temporary.name)
        temporary.close()
        np.savez_compressed(
            reference_temp_path,
            chains=chains,
            pooled=chains.reshape(-1, 3),
            seeds=np.asarray(NUTS_SEEDS, dtype=np.int64),
            theta_map=theta_map,
            z_init=z_init_np,
            acceptance_rates=acceptance_rates,
            divergence_counts=divergence_counts,
            warmup_steps=np.asarray(WARMUP, dtype=np.int64),
            retained_per_chain=np.asarray(RETAINED, dtype=np.int64),
            target_accept_prob=np.asarray(TARGET_ACCEPT, dtype=np.float64),
            run_classification=np.asarray(run_classification),
            phase_b_reference_sha256=np.asarray(frozen["input_hashes"]["phase_b_reference"]),
        )
        reference_sha = sha256(reference_temp_path)
        gate_output["qualified_reference"] = {
            "path": str(REFERENCE_2000_PATH),
            "sha256": reference_sha,
            "shape": list(chains.shape),
            "status": "QUALIFIED",
        }

        with np.load(FAST_RAW, allow_pickle=False) as data:
            fast_theta = data["theta"].astype(np.float64)
            flow_training_seeds = data["flow_training_seeds"].astype(int)
            scenario_names = data["scenario_names"].astype(str)
            latent_seeds = data["latent_seeds"].astype(int)
        if fast_theta.shape != (3, 2, 5, 2000, 3):
            raise RuntimeError(f"Unexpected frozen fast theta shape {fast_theta.shape}")
        if flow_training_seeds.tolist() != [42, 202, 303]:
            raise RuntimeError("Unexpected frozen flow-training seeds")
        if scenario_names.tolist() != ["field", "damage"]:
            raise RuntimeError("Unexpected frozen fast scenarios")
        if latent_seeds.tolist() != LATENT_SEEDS:
            raise RuntimeError("Unexpected frozen latent seeds")

        with FAST_ROUTING.open(encoding="utf-8", newline="") as handle:
            all_routing_rows = list(csv.DictReader(handle))
        field_routing_rows = {
            int(row["latent_seed"]): row
            for row in all_routing_rows
            if int(row["flow_training_seed"]) == 42 and row["scenario"] == "field"
        }
        damage_routing_rows = [
            row
            for row in all_routing_rows
            if int(row["flow_training_seed"]) == 42 and row["scenario"] == "damage"
        ]
        if sorted(field_routing_rows) != LATENT_SEEDS or len(damage_routing_rows) != 5:
            raise RuntimeError("Frozen primary routing rows are incomplete")

        primary_field = fast_theta[0, 0]
        primary_damage = fast_theta[0, 1]
        gap_rows = []
        for seed_index, seed in enumerate(LATENT_SEEDS):
            raw = primary_field[seed_index]
            valid = in_support(raw)
            source = field_routing_rows[seed]
            routing = {
                "raw_ess_ratio": float(source["raw_ess_ratio"]),
                "psis_ess_ratio": float(source["psis_ess_ratio"]),
                "pareto_k": float(source["pareto_k"]),
                "raw_route_tau05": source["raw_route_tau05"],
                "psis_route_tau05": source["psis_route_tau05"],
            }
            if int(source["N_proposals"]) != len(raw) or int(source["in_support_count"]) != int(valid.sum()):
                raise RuntimeError(f"Frozen routing/support mismatch for latent seed {seed}")
            gap_rows.append(
                make_gap_row(
                    "per_latent_seed",
                    seed,
                    "all_q_phi_proposals",
                    raw,
                    raw,
                    pooled,
                    routing,
                    reference_sha,
                )
            )
            gap_rows.append(
                make_gap_row(
                    "per_latent_seed",
                    seed,
                    "in_support_q_phi_proposal_subset",
                    raw[valid],
                    raw,
                    pooled,
                    routing,
                    reference_sha,
                )
            )

        pooled_raw = primary_field.reshape(-1, 3)
        pooled_valid = in_support(pooled_raw)
        existing_field = [field_routing_rows[seed] for seed in LATENT_SEEDS]
        pooled_routing = {
            "raw_ess_ratio": float(np.mean([float(row["raw_ess_ratio"]) for row in existing_field])),
            "psis_ess_ratio": float(np.mean([float(row["psis_ess_ratio"]) for row in existing_field])),
            "pareto_k": float(np.mean([float(row["pareto_k"]) for row in existing_field])),
            "raw_route_tau05": (
                f"{sum(row['raw_route_tau05'] == 'FAST' for row in existing_field)} FAST / "
                f"{sum(row['raw_route_tau05'] == 'CORRECTION' for row in existing_field)} CORRECTION"
            ),
            "psis_route_tau05": (
                f"{sum(row['psis_route_tau05'] == 'FAST' for row in existing_field)} FAST / "
                f"{sum(row['psis_route_tau05'] == 'CORRECTION' for row in existing_field)} CORRECTION"
            ),
        }
        gap_rows.append(
            make_gap_row(
                "pooled_latent_seeds_42_46",
                "pooled",
                "all_q_phi_proposals",
                pooled_raw,
                pooled_raw,
                pooled,
                pooled_routing,
                reference_sha,
            )
        )
        gap_rows.append(
            make_gap_row(
                "pooled_latent_seeds_42_46",
                "pooled",
                "in_support_q_phi_proposal_subset",
                pooled_raw[pooled_valid],
                pooled_raw,
                pooled,
                pooled_routing,
                reference_sha,
            )
        )

        surrogate64 = module.Surrogate().double().cpu()
        surrogate64.load_state_dict(torch.load(SURROGATE_PATH, map_location="cpu", weights_only=True))
        surrogate64.eval()
        with torch.no_grad():
            modal_prediction = (
                surrogate64(torch.as_tensor(posterior_mean, dtype=torch.float64).unsqueeze(0))
                .squeeze(0)
                .numpy()[:6]
            )
        observed = np.asarray(Y6_DECIMAL, dtype=np.float64)
        relative_error_pct = 100.0 * (modal_prediction - observed) / observed
        table14_rms_relative_error_pct = float(
            100.0 * np.sqrt(np.mean(((modal_prediction - observed) / observed) ** 2))
        )
        table14_rows = [
            {
                "mode": mode,
                "measured_hz": float(observed[mode - 1]),
                "surrogate_at_qualified_nuts_posterior_mean_hz": float(modal_prediction[mode - 1]),
                "relative_error_pct": float(relative_error_pct[mode - 1]),
            }
            for mode in range(1, 7)
        ]

        pooled_all_row = next(
            row
            for row in gap_rows
            if row["scope"] == "pooled_latent_seeds_42_46"
            and row["analysis_population"] == "all_q_phi_proposals"
        )
        pooled_valid_row = next(
            row
            for row in gap_rows
            if row["scope"] == "pooled_latent_seeds_42_46"
            and row["analysis_population"] == "in_support_q_phi_proposal_subset"
        )
        damage_raw = primary_damage.reshape(-1, 3)
        damage_valid = in_support(damage_raw)
        fast_summary = json.loads(FAST_SUMMARY.read_text(encoding="utf-8"))["42"]["damage"]

        prefix_table_rows = [
            [
                record["seed"],
                f"{record['theta_exact_max_abs_difference']:.17g}",
                str(record["theta_bitwise_equality"]),
                f"{record['z_exact_max_abs_difference']:.17g}",
                str(record["z_bitwise_equality"]),
            ]
            for record in prefix_records
        ]
        posterior_table_rows = [
            [
                parameter,
                f"{posterior_mean[j]:.10f}",
                f"{posterior_std[j]:.10f}",
                f"{quantiles[0, j]:.10f}",
                f"{quantiles[1, j]:.10f}",
                f"{quantiles[2, j]:.10f}",
                f"{rank_rhat[j]:.10f}",
                f"{bulk_ess[j]:.3f}",
                f"{tail_ess[j]:.3f}",
            ]
            for j, parameter in enumerate(PARAMETERS)
        ]
        chain_table_rows = [
            [
                record["seed"],
                f"{record['runtime_s']:.3f}",
                f"{record['acceptance_rate']:.6f}",
                record["divergence_count"],
                f"{record['adapted_step_size']:.10g}",
            ]
            for record in records
        ]
        leakage_table_rows = []
        for seed_index, seed in enumerate(LATENT_SEEDS):
            raw = primary_field[seed_index]
            valid = in_support(raw)
            leakage_table_rows.append(
                [
                    seed,
                    len(raw),
                    int((~valid).sum()),
                    f"{100.0 * (~valid).mean():.4f}%",
                    int((raw < LO).any(axis=1).sum()),
                    int((raw > HI).any(axis=1).sum()),
                ]
            )
        leakage_table_rows.append(
            [
                "pooled",
                len(pooled_raw),
                int((~pooled_valid).sum()),
                f"{100.0 * (~pooled_valid).mean():.4f}%",
                int((pooled_raw < LO).any(axis=1).sum()),
                int((pooled_raw > HI).any(axis=1).sum()),
            ]
        )
        table14_md_rows = [
            [
                row["mode"],
                f"{row['measured_hz']:.6f}",
                f"{row['surrogate_at_qualified_nuts_posterior_mean_hz']:.9f}",
                f"{row['relative_error_pct']:.6f}",
            ]
            for row in table14_rows
        ]

        qualified_lines = [
            "# Coupled qualified bounded-Uniform reference audit",
            "",
            "## Disposition",
            "",
            "**QUALIFIED (PASS).** This is the single predeclared Phase C1 extension to 4 × 2,000 retained draws. "
            "No chain was dropped or replaced and no retry was performed.",
            "",
            "## Frozen protocol",
            "",
            "- target: independent `Uniform(0.5,4.0)^3`;",
            "- observation: first six measured modes `[2.06, 2.30, 2.62, 6.14, 6.53, 6.76]`;",
            "- likelihood: unchanged relative-error Gaussian potential with `kappa_sq=0.0009`;",
            "- transform: `theta = 0.5 + 3.5*sigmoid(z)`, with the exact Jacobian included;",
            "- chains: seeds `811/812/813/814`, warmup `500`, retained `2000`, target acceptance `0.90`;",
            "- initialization: the frozen accepted Phase-B bounded-Uniform MAP for every chain;",
            "- frozen surrogate SHA-256: `" + frozen["input_hashes"]["surrogate"] + "`.",
            "",
            "Only retained chain length changed from 600 to 2,000.",
            "",
            "## Prefix reproducibility",
            "",
            f"Classification: **{run_classification}**.",
            "",
            md_table(
                ["seed", "theta max abs diff", "theta bitwise", "z max abs diff", "z bitwise"],
                prefix_table_rows,
            ),
            "",
            "The comparison uses the first 600 retained states of each new chain against the matching Phase-B chain. "
            "No rerun was performed after this comparison.",
            "",
            "## Qualification gate and posterior summary",
            "",
            md_table(
                ["parameter", "mean", "std", "2.5%", "50%", "97.5%", "rank R-hat", "bulk ESS", "tail ESS"],
                posterior_table_rows,
            ),
            "",
            f"- maximum rank-normalized split R-hat: `{rank_rhat.max():.10f}` (required `<=1.01`);",
            f"- minimum bulk ESS: `{bulk_ess.min():.6f}` (required `>=400`);",
            f"- minimum tail ESS: `{tail_ess.min():.6f}` (required `>=400`);",
            f"- divergences: `{int(divergence_counts.sum())}`;",
            f"- theta coordinates outside `[0.5,4]`: `{outside_coordinate_count}`;",
            f"- qualified reference: `{REFERENCE_2000_PATH.name}`;",
            f"- qualified reference SHA-256: `{reference_sha}`.",
            "",
            "### Chain acceptance",
            "",
            md_table(["seed", "runtime (s)", "acceptance", "divergences", "adapted step size"], chain_table_rows),
            "",
            f"Mean acceptance: `{acceptance_rates.mean():.8f}`.",
            "",
            "## Qualified field gap using frozen primary Uniform-CNF draws",
            "",
            "The primary seed-42 flow and the already stored latent-seed 42–46 draws were reused. No fast draws, "
            "importance weights, ESS values, Pareto-k values, or routes were regenerated. All three coupled "
            "coordinates are active, so active-coordinate and all-coordinate mean MAE are numerically identical.",
            "",
            md_table(
                ["population", "N", "active/all mean MAE", "mean 1D W1", "max 1D W1", "mean 95% width ratio"],
                [
                    [
                        "all q_phi proposals",
                        pooled_all_row["N_fast_used_for_gap"],
                        f"{pooled_all_row['active_mean_mae']:.8f}",
                        f"{pooled_all_row['w1_1d_mean']:.8f}",
                        f"{pooled_all_row['w1_1d_max']:.8f}",
                        f"{pooled_all_row['q95_width_ratio_mean']:.8f}",
                    ],
                    [
                        "explicit in-support q_phi subset (secondary audit)",
                        pooled_valid_row["N_fast_used_for_gap"],
                        f"{pooled_valid_row['active_mean_mae']:.8f}",
                        f"{pooled_valid_row['w1_1d_mean']:.8f}",
                        f"{pooled_valid_row['w1_1d_max']:.8f}",
                        f"{pooled_valid_row['q95_width_ratio_mean']:.8f}",
                    ],
                ],
            ),
            "",
            f"Reused routing means are raw ESS/N `{pooled_routing['raw_ess_ratio']:.8f}`, PSIS ESS/N "
            f"`{pooled_routing['psis_ess_ratio']:.8f}`, and Pareto k `{pooled_routing['pareto_k']:.8f}`; "
            f"routes remain `{pooled_routing['raw_route_tau05']}` raw and `{pooled_routing['psis_route_tau05']}` PSIS.",
            "",
            "`q_phi` is a fast proposal distribution on R^3. Neither its raw draws nor its explicitly labelled "
            "in-support subset is called an exact bounded posterior.",
            "",
            "## Figure 15 data preparation only",
            "",
            "No Figure 15 manuscript asset was generated or replaced. The future candidate data sources are:",
            "",
            f"- fast distribution: frozen `q_phi` proposal in `{FAST_RAW.name}` (primary training seed 42, latent seeds 42–46);",
            f"- exact distribution: qualified bounded-Uniform NUTS posterior in `{REFERENCE_2000_PATH.name}`.",
            "",
            md_table(
                ["latent seed", "proposal N", "unique out-of-box", "leakage", "any <0.5", "any >4"],
                leakage_table_rows,
            ),
            "",
            "The unique pooled field leakage is "
            f"`{int((~pooled_valid).sum())}/{len(pooled_raw)} = {100.0 * (~pooled_valid).mean():.4f}%`. "
            "The below/above counts are diagnostic marginals and can overlap. Future plotting may mark prior bounds "
            "0.5 and 4.0; it must not silently discard leaked proposal draws. The qualified NUTS posterior has zero "
            "support violations.",
            "",
            "## Recomputed Table 14 modal fit",
            "",
            "The modal prediction below uses only the qualified NUTS posterior mean and the frozen surrogate; no FE call was made.",
            "",
            md_table(["mode", "measured (Hz)", "surrogate at qualified mean (Hz)", "relative error (%)"], table14_md_rows),
            "",
            f"RMS relative error: `{table14_rms_relative_error_pct:.8f}%`.",
            "",
            "## Coupled damage disposition",
            "",
            f"No damage NUTS, recovery-performance calculation, or Figure 16 was produced. The frozen primary damage "
            f"proposal has `{int((~damage_valid).sum())}/{len(damage_raw)} = {100.0 * (~damage_valid).mean():.4f}%` "
            "unique out-of-support draws. Existing audit-only routing remains: mean raw ESS/N "
            f"`{float(fast_summary['raw_ess_ratio_mean']):.8f}`, mean PSIS ESS/N "
            f"`{float(fast_summary['psis_ess_ratio_mean']):.8f}`, mean Pareto k "
            f"`{float(fast_summary['pareto_k_mean']):.8f}`, with 5/5 CORRECTION for both routes. "
            "A support-filtered proposal subset is not a posterior and no damage-performance claim is made.",
            "",
            "## Scope controls",
            "",
            "No manuscript/response file was modified. No final integration, 30-case routing, AUC confidence interval, "
            "surrogate training, CNF training, FE solve, or additional experiment was started.",
            "",
        ]
        qualified_audit = "\n".join(qualified_lines)
        gate_output["post_gate_actions_performed"] = True
        gate_output["qualified_gap"] = {
            "path": str(GAP_PATH),
            "fast_draw_source_sha256": frozen["input_hashes"]["fast_routing_raw"],
            "routing_source_sha256": frozen["input_hashes"]["fast_routing_csv"],
            "fast_draws_regenerated": False,
            "routing_values_recomputed": False,
            "pooled_all_q_phi_proposals": {
                key: pooled_all_row[key]
                for key in [
                    "active_mean_mae",
                    "all_mean_mae",
                    "w1_1d_mean",
                    "w1_1d_max",
                    "q95_width_ratio_mean",
                    "q95_width_ratio_min",
                    "q95_width_ratio_max",
                    "field_proposal_support_leakage_count",
                    "field_proposal_support_leakage_percentage",
                ]
            },
            "pooled_in_support_q_phi_subset": {
                key: pooled_valid_row[key]
                for key in [
                    "active_mean_mae",
                    "all_mean_mae",
                    "w1_1d_mean",
                    "w1_1d_max",
                    "q95_width_ratio_mean",
                    "q95_width_ratio_min",
                    "q95_width_ratio_max",
                ]
            },
        }
        gate_output["table14_modal_fit"] = {
            "posterior_mean_source": "qualified bounded-Uniform NUTS 4x2000 posterior mean",
            "surrogate_sha256": frozen["input_hashes"]["surrogate"],
            "new_fe_calls": 0,
            "rows": table14_rows,
            "rms_relative_error_pct": table14_rms_relative_error_pct,
        }

    gate_table_rows = [
        ["rank-normalized split R-hat", "<= 1.01", f"max {rank_rhat.max():.10f}", "PASS" if gates["rank_normalized_split_rhat"] else "FAIL"],
        ["bulk ESS", ">= 400", f"min {bulk_ess.min():.6f}", "PASS" if gates["bulk_ess"] else "FAIL"],
        ["tail ESS", ">= 400", f"min {tail_ess.min():.6f}", "PASS" if gates["tail_ess"] else "FAIL"],
        ["divergences", "= 0", int(divergence_counts.sum()), "PASS" if gates["divergences"] else "FAIL"],
        ["theta outside [0.5,4]", "= 0", outside_coordinate_count, "PASS" if gates["bounded_support"] else "FAIL"],
    ]
    report_lines = [
        "# Phase C1 coupled chain-length extension report",
        "",
        f"Overall qualification gate: **{gate_status}**.",
        "",
        "Exactly one predeclared 4-chain run was executed. Seeds `811/812/813/814`, warmup `500`, target "
        "acceptance `0.90`, target, observation, prior, likelihood, transform/Jacobian, initialization, surrogate, "
        "and NUTS implementation were held fixed. Only retained length changed from `600` to `2000` per chain.",
        "",
        f"Prefix classification: **{run_classification}**. No chain was dropped or replaced and no retry was performed.",
        "",
        md_table(["gate", "requirement", "observed", "result"], gate_table_rows),
        "",
        f"Posterior mean: `{format_vector(posterior_mean, 10)}`.",
        "",
        f"Posterior std: `{format_vector(posterior_std, 10)}`.",
        "",
        f"Acceptance by seed: `{format_vector(acceptance_rates, 8)}`; mean `{acceptance_rates.mean():.8f}`.",
        "",
    ]
    if passed:
        report_lines.extend(
            [
                f"The qualified reference is `{REFERENCE_2000_PATH.name}` (SHA-256 `{reference_sha}`).",
                "",
                "The frozen primary Uniform-CNF field draws were compared with this qualified reference, existing "
                "raw/PSIS ESS, Pareto-k and routes were copied unchanged, Table 14 modal fit was recomputed with the "
                "frozen surrogate, and Figure 15 data provenance/support leakage was prepared without generating a figure.",
                "",
                "Damage was retained only as routing/support audit evidence; no damage NUTS, performance claim, or Figure 16 was created.",
                "",
                "Generated PASS outputs:",
                "",
                f"- `{REFERENCE_2000_PATH.name}`",
                f"- `{GAP_PATH.name}`",
                f"- `{QUALIFIED_AUDIT_PATH.name}`",
                "",
            ]
        )
    else:
        failed_names = [name for name, value in gates.items() if not value]
        report_lines.extend(
            [
                "The gate failed and processing stopped before creation of a qualified reference, qualified gap, "
                "Table 14 modal fit, or Figure 15 preparation.",
                "",
                "Failed gate components: `" + ", ".join(failed_names) + "`.",
                "",
            ]
        )
    report_lines.extend(
        [
            "Required audit outputs:",
            "",
            f"- `{REPORT_PATH.name}`",
            f"- `{GATE_PATH.name}`",
            f"- `{DIAGNOSTICS_PATH.name}`",
            f"- `{PREFIX_PATH.name}`",
            "",
            "STOP: no final integration, 30-case routing, AUC CI, manuscript/response edit, commit, or push was performed.",
            "",
        ]
    )
    report_text = "\n".join(report_lines)

    if passed:
        if reference_temp_path is None or not reference_temp_path.exists():
            raise RuntimeError("Qualified reference temporary artifact is missing")
        os.replace(reference_temp_path, REFERENCE_2000_PATH)
        reference_temp_path = None
        write_csv_exclusive(GAP_PATH, gap_rows)
        write_text_exclusive(QUALIFIED_AUDIT_PATH, qualified_audit)

    write_text_exclusive(PREFIX_PATH, json_text(prefix_output))
    write_csv_exclusive(DIAGNOSTICS_PATH, diagnostics_rows)
    write_text_exclusive(GATE_PATH, json_text(gate_output))
    write_text_exclusive(REPORT_PATH, report_text)

    print(
        f"PHASE_C1_COMPLETE gate={gate_status} classification={run_classification} "
        f"max_rhat={rank_rhat.max():.10f} min_bulk={bulk_ess.min():.6f} "
        f"min_tail={tail_ess.min():.6f} divergences={int(divergence_counts.sum())} "
        f"outside={outside_coordinate_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
