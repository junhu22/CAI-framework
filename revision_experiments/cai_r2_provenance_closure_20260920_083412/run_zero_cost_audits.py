import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch


ROOT = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
OUT = ROOT / "revision_experiments/cai_r2_provenance_closure_20260920_083412"
ROUTING_ARRAYS = ROOT / "revision_experiments/cai_r2_stage1/run_20260917_094722_21d144f3db844a6b8a2e8167c5282dbe/routing_arrays"
REFERENCE_GAP = ROOT / "revision_experiments/cai_r2_stage1/reference_gap_20260918_121500"
PHASE_B = ROOT / "revision_experiments/cai_r2_presubmit_phaseB_20260919_213647"
SEEDS = [42, 43, 44, 45, 46]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_ess_ratio(log_weights: np.ndarray) -> float:
    lw = np.asarray(log_weights, dtype=np.float64)
    finite = np.isfinite(lw)
    if not finite.any():
        return 0.0
    shifted = np.full_like(lw, -np.inf)
    shifted[finite] = lw[finite] - np.max(lw[finite])
    weights = np.exp(shifted)
    weights /= weights.sum()
    return float(1.0 / np.sum(weights * weights) / len(lw))


def load_case_arrays():
    cases = {}
    for number in range(1, 6):
        proposal = []
        logw = []
        outside = []
        for seed in SEEDS:
            path = ROUTING_ARRAYS / f"ablation_obs{number}_seed{seed}_N2000.npz"
            with np.load(path, allow_pickle=False) as archive:
                proposal.append(np.asarray(archive["theta"], dtype=np.float64))
                logw.append(np.asarray(archive["log_weights"], dtype=np.float64))
                outside.append(np.asarray(archive["target_support_outside"], dtype=bool))
        ref_path = REFERENCE_GAP / f"ablation_obs{number}_reference_chains_4chain.npz"
        with np.load(ref_path, allow_pickle=False) as archive:
            reference = np.asarray(archive["chains"], dtype=np.float64).reshape(-1, 6)
        cases[f"Obs. {number}"] = {
            "proposal": np.stack(proposal), "logw": np.stack(logw),
            "outside": np.stack(outside), "reference": reference,
            "support": "theta_j>0 for all j", "source": str(ref_path),
        }

    proposal = []
    logw = []
    outside = []
    for seed in SEEDS:
        path = ROUTING_ARRAYS / f"sailboat_boundary_e1_075_seed{seed}_N2000.npz"
        with np.load(path, allow_pickle=False) as archive:
            proposal.append(np.asarray(archive["theta"], dtype=np.float64))
            logw.append(np.asarray(archive["log_weights"], dtype=np.float64))
            outside.append(np.asarray(archive["target_support_outside"], dtype=bool))
    boundary_ref = PHASE_B / "BOUNDARY/BOUNDARY_REFERENCE_RERUN.npz"
    with np.load(boundary_ref, allow_pickle=False) as archive:
        reference = np.asarray(archive["pooled"], dtype=np.float64)
    cases["Near-threshold e1=0.75"] = {
        "proposal": np.stack(proposal), "logw": np.stack(logw),
        "outside": np.stack(outside), "reference": reference,
        "support": "theta_j>0 for all j", "source": str(boundary_ref),
    }

    sail_path = PHASE_B / "SAILBOAT_PADDING/SAILBOAT_PADDING_PAIRED_RAW.npz"
    with np.load(sail_path, allow_pickle=False) as archive:
        proposal = np.asarray(archive["theta"][1], dtype=np.float64)
        logw = np.asarray(archive["log_weights"][1], dtype=np.float64)
        reference = np.asarray(archive["reference"], dtype=np.float64)
    cases["Sailboat field"] = {
        "proposal": proposal, "logw": logw,
        "outside": ~np.all(proposal > 0.0, axis=-1), "reference": reference,
        "support": "theta_j>0 for all j", "source": str(sail_path),
    }

    coupled_raw = PHASE_B / "COUPLED_UNIFORM/COUPLED_UNIFORM_FAST_ROUTING_RAW.npz"
    coupled_ref = PHASE_B / "COUPLED_UNIFORM/COUPLED_UNIFORM_NUTS_REFERENCE_2000.npz"
    with np.load(coupled_raw, allow_pickle=False) as archive:
        scenario_names = archive["scenario_names"].tolist()
        theta = np.asarray(archive["theta"], dtype=np.float64)
        weights = np.asarray(archive["log_weights"], dtype=np.float64)
    with np.load(coupled_ref, allow_pickle=False) as archive:
        reference = np.asarray(archive["pooled"], dtype=np.float64)
    for label, scenario in (("Coupled field", "field"), ("Coupled boundary damage", "damage")):
        si = scenario_names.index(scenario)
        proposal = theta[0, si]
        logw = weights[0, si]
        cases[label] = {
            "proposal": proposal, "logw": logw,
            "outside": ~np.all((proposal >= 0.5) & (proposal <= 4.0), axis=-1),
            "reference": reference if scenario == "field" else None,
            "support": "0.5<=theta_j<=4.0 for all j", "source": str(coupled_raw),
        }
    return cases


def width_audit(cases):
    path = OUT / "ROUTING_GAP_WIDTH_RATIO.csv"
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite {path}")
    fields = ["case", "row_type", "proposal_seed", "coordinate", "fast_width_95", "reference_width_95", "width_ratio", "active_coordinate_mean_width_ratio", "aggregation", "proposal_source", "reference_source"]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case_name in [f"Obs. {i}" for i in range(1, 6)] + ["Near-threshold e1=0.75", "Sailboat field", "Coupled field"]:
            case = cases[case_name]
            proposal = case["proposal"]
            reference = case["reference"]
            ref_quantiles = np.quantile(reference[:, :3], [0.025, 0.975], axis=0)
            ref_width = ref_quantiles[1] - ref_quantiles[0]
            per_seed = []
            for seed_index, seed in enumerate(SEEDS):
                fast_quantiles = np.quantile(proposal[seed_index, :, :3], [0.025, 0.975], axis=0)
                fast_width = fast_quantiles[1] - fast_quantiles[0]
                ratio = fast_width / ref_width
                per_seed.append(ratio)
                for coordinate in range(3):
                    writer.writerow({
                        "case": case_name, "row_type": "seed_coordinate", "proposal_seed": seed,
                        "coordinate": f"e{coordinate + 1}", "fast_width_95": repr(float(fast_width[coordinate])),
                        "reference_width_95": repr(float(ref_width[coordinate])),
                        "width_ratio": repr(float(ratio[coordinate])),
                        "active_coordinate_mean_width_ratio": repr(float(ratio.mean())),
                        "aggregation": "central empirical quantiles; active e1-e3 arithmetic mean",
                        "proposal_source": case["source"], "reference_source": case["source"],
                    })
            per_seed = np.asarray(per_seed)
            coordinate_mean = per_seed.mean(axis=0)
            for coordinate in range(3):
                writer.writerow({
                    "case": case_name, "row_type": "case_coordinate_mean", "proposal_seed": "",
                    "coordinate": f"e{coordinate + 1}", "fast_width_95": "", "reference_width_95": repr(float(ref_width[coordinate])),
                    "width_ratio": repr(float(coordinate_mean[coordinate])),
                    "active_coordinate_mean_width_ratio": repr(float(coordinate_mean.mean())),
                    "aggregation": "mean across proposal seeds, then arithmetic mean across active coordinates",
                    "proposal_source": case["source"], "reference_source": case["source"],
                })
            writer.writerow({
                "case": case_name, "row_type": "case_active_mean", "proposal_seed": "", "coordinate": "active_mean",
                "fast_width_95": "", "reference_width_95": "", "width_ratio": repr(float(coordinate_mean.mean())),
                "active_coordinate_mean_width_ratio": repr(float(coordinate_mean.mean())),
                "aggregation": "mean of seedwise e1-e3 central-95% width ratios",
                "proposal_source": case["source"], "reference_source": case["source"],
            })
            # Primary Table-10 case-class summary: pool the five retained
            # proposal arrays before taking the central interval.  This is
            # distinct from averaging five seedwise width ratios above.
            pooled_fast = proposal.reshape(-1, proposal.shape[-1])
            pooled_quantiles = np.quantile(pooled_fast[:, :3], [0.025, 0.975], axis=0)
            pooled_width = pooled_quantiles[1] - pooled_quantiles[0]
            pooled_ratio = pooled_width / ref_width
            if case_name.startswith("Obs. "):
                proposal_source = "retained routing arrays"
                reference_source = f"ablation_obs{case_name.split()[-1]}_reference_chains_4chain.npz"
            elif case_name.startswith("Near-threshold"):
                proposal_source = "retained boundary routing arrays"
                reference_source = "BOUNDARY_REFERENCE_RERUN.npz"
            elif case_name == "Sailboat field":
                proposal_source = "SAILBOAT_PADDING_PAIRED_RAW.npz"
                reference_source = "embedded qualified reference"
            else:
                proposal_source = "COUPLED_UNIFORM_FAST_ROUTING_RAW.npz"
                reference_source = "COUPLED_UNIFORM_NUTS_REFERENCE_2000.npz"
            writer.writerow({
                "case": case_name, "row_type": "case_pooled_active_mean", "proposal_seed": "",
                "coordinate": "active_mean", "fast_width_95": "", "reference_width_95": "",
                "width_ratio": repr(float(pooled_ratio.mean())),
                "active_coordinate_mean_width_ratio": repr(float(pooled_ratio.mean())),
                "aggregation": "PRIMARY case-class summary: pool five retained 2000-draw proposals; central 95% widths per active coordinate; arithmetic mean",
                "proposal_source": proposal_source, "reference_source": reference_source,
            })
    return path


def support_audit(cases):
    path = OUT / "PROPOSAL_SUPPORT_AUDIT.csv"
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite {path}")
    fields = ["case", "N", "N_in_support", "N_out_of_support", "support_fraction_s", "raw_ess_ratio_r", "in_support_ess_ratio_r_over_s", "arithmetic_support_ceiling", "target_zero_outside_support", "operative_router", "support_definition", "source"]
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case_name, case in cases.items():
            outside = np.asarray(case["outside"], dtype=bool)
            n = outside.size
            n_out = int(outside.sum())
            n_in = n - n_out
            s = n_in / n
            r = float(np.mean([raw_ess_ratio(row) for row in case["logw"]]))
            writer.writerow({
                "case": case_name, "N": n, "N_in_support": n_in, "N_out_of_support": n_out,
                "support_fraction_s": repr(float(s)), "raw_ess_ratio_r": repr(r),
                "in_support_ess_ratio_r_over_s": repr(float(r / s)) if s > 0 else "",
                "arithmetic_support_ceiling": repr(float(s)), "target_zero_outside_support": True,
                "operative_router": "raw r with tau=0.5; r_in is secondary only",
                "support_definition": case["support"], "source": case["source"],
            })
    return path


def rank_audit():
    cases = ["Obs. 1", "Obs. 2", "Obs. 3", "Obs. 4", "Obs. 5", "Near-threshold", "Sailboat field", "Coupled field"]
    raw = np.array([0.7001, 0.6718, 0.8620, 0.7674, 0.6549, 0.4362, 0.0515, 0.0604])
    mae = np.array([0.0177, 0.0302, 0.0040, 0.0112, 0.0129, 0.0163, 0.0952, 0.0701])
    w1 = np.array([0.0217, 0.0304, 0.0068, 0.0131, 0.0134, 0.0360, 0.149, 0.136])

    def ordinal_rank(values):
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=int)
        ranks[order] = np.arange(1, len(values) + 1)
        return ranks

    rx, rw, rm = ordinal_rank(raw), ordinal_rank(w1), ordinal_rank(mae)
    dw, dm = rx - rw, rx - rm
    n = len(cases)
    rho_w = 1.0 - 6.0 * float(np.sum(dw * dw)) / (n * (n * n - 1))
    rho_m = 1.0 - 6.0 * float(np.sum(dm * dm)) / (n * (n * n - 1))
    rows = []
    for i, case in enumerate(cases):
        rows.append({
            "case": case, "raw_ess_displayed": float(raw[i]), "raw_rank_ascending": int(rx[i]),
            "mean_w1_displayed": float(w1[i]), "w1_rank_ascending": int(rw[i]), "d_w1": int(dw[i]), "d_w1_squared": int(dw[i] ** 2),
            "active_mae_displayed": float(mae[i]), "mae_rank_ascending": int(rm[i]), "d_mae": int(dm[i]), "d_mae_squared": int(dm[i] ** 2),
        })
    return {
        "classification": "DESCRIPTIVE_SELECTED_CASE_ASSOCIATION",
        "n_case_classes": n, "rows": rows,
        "w1": {"sum_d_squared": int(np.sum(dw * dw)), "rho_s": rho_w},
        "mae": {"sum_d_squared": int(np.sum(dm * dm)), "rho_s": rho_m},
        "separation": {
            "max_mean_w1_unanimous_fast": float(w1[:5].max()),
            "min_mean_w1_other_three": float(w1[5:].min()),
            "strict_nonoverlap": bool(w1[:5].max() < w1[5:].min()),
        },
        "near_threshold_shape": {
            "active_mae": float(mae[5]), "mean_w1": float(w1[5]),
            "interpretation_rule": "combine with retained central-95% interval width ratio; do not form W1/MAE ratio",
        },
    }


def coupled_identifiability():
    ref_path = PHASE_B / "COUPLED_UNIFORM/COUPLED_UNIFORM_NUTS_REFERENCE_2000.npz"
    with np.load(ref_path, allow_pickle=False) as archive:
        samples = np.asarray(archive["pooled"], dtype=np.float64)
    e2 = samples[:, 1]
    sorted_e2 = np.sort(e2)
    qgrid = (np.arange(len(e2), dtype=np.float64) + 0.5) / len(e2)
    uniform_quantiles = 0.5 + 3.5 * qgrid
    w1 = float(np.mean(np.abs(sorted_e2 - uniform_quantiles)))
    prior_sd = 3.5 / math.sqrt(12.0)
    quantiles = np.quantile(e2, [0.025, 0.5, 0.975])

    module_path = ROOT / "experiments/coupled_building/04_train_cnf_coupled.py"
    spec = importlib.util.spec_from_file_location("coupled_class_for_sensitivity", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    surrogate_path = ROOT / "surrogate_models_coupled/member_0.pt"
    surrogate = module.Surrogate().double().cpu()
    surrogate.load_state_dict(torch.load(surrogate_path, map_location="cpu", weights_only=True))
    surrogate.eval()
    posterior_mean = samples.mean(axis=0)
    theta = torch.tensor(posterior_mean, dtype=torch.float64).unsqueeze(0).requires_grad_(True)
    prediction = surrogate(theta)
    jacobian = np.zeros((6, 3), dtype=np.float64)
    for mode in range(6):
        gradient = torch.autograd.grad(prediction[0, mode], theta, retain_graph=True)[0][0]
        jacobian[mode] = gradient.detach().cpu().numpy()
    norms = np.linalg.norm(jacobian, axis=0)
    normalized = norms / norms.max()

    return {
        "reference_path": str(ref_path), "reference_sha256": sha256(ref_path), "N": int(len(samples)),
        "e2_posterior": {
            "mean": float(e2.mean()), "sample_sd": float(e2.std(ddof=1)),
            "quantiles_0.025_0.5_0.975": quantiles.tolist(),
        },
        "uniform_prior": {
            "lower": 0.5, "upper": 4.0, "mean": 2.25, "sd": prior_sd,
            "quantiles_0.025_0.5_0.975": [0.5875, 2.25, 3.9125],
        },
        "e2_vs_prior": {
            "w1_deterministic_quantile_grid": w1, "grid_size": int(len(e2)),
            "grid": "midpoint exact-Uniform quantiles versus sorted empirical posterior values",
            "mean_difference": float(e2.mean() - 2.25),
            "posterior_to_prior_sd_ratio": float(e2.std(ddof=1) / prior_sd),
        },
        "boundary_mass": {
            "P_e1_gt_3.8": float(np.mean(samples[:, 0] > 3.8)),
            "P_e3_gt_3.8": float(np.mean(samples[:, 2] > 3.8)),
            "P_e1_gt_3.9": float(np.mean(samples[:, 0] > 3.9)),
            "P_e3_gt_3.9": float(np.mean(samples[:, 2] > 3.9)),
            "counts": {
                "e1_gt_3.8": int(np.sum(samples[:, 0] > 3.8)), "e3_gt_3.8": int(np.sum(samples[:, 2] > 3.8)),
                "e1_gt_3.9": int(np.sum(samples[:, 0] > 3.9)), "e3_gt_3.9": int(np.sum(samples[:, 2] > 3.9)),
            },
        },
        "first_six_surrogate_sensitivity_at_posterior_mean": {
            "definition_reused": "Euclidean L2 norm of each surrogate Jacobian column; divide by maximum column norm, matching Figure 12(b)",
            "theta": posterior_mean.tolist(), "jacobian_6_by_3": jacobian.tolist(),
            "column_l2_norms": norms.tolist(), "normalized_by_max": normalized.tolist(),
            "surrogate_path": str(surrogate_path), "surrogate_sha256": sha256(surrogate_path),
        },
    }


def main():
    output_json = OUT / "ZERO_COST_ROUTING_IDENTIFIABILITY.json"
    if output_json.exists():
        raise RuntimeError(f"Refusing to overwrite {output_json}")
    cases = load_case_arrays()
    width_path = width_audit(cases)
    support_path = support_audit(cases)
    payload = {
        "schema": "cai-r2-zero-cost-routing-identifiability-v1",
        "new_samples_generated": False,
        "retained_arrays_only": True,
        "width_csv": {"path": str(width_path), "sha256": sha256(width_path)},
        "support_csv": {"path": str(support_path), "sha256": sha256(support_path)},
        "rank_audit": rank_audit(),
        "coupled_identifiability": coupled_identifiability(),
    }
    output_json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "width_sha256": payload["width_csv"]["sha256"],
        "support_sha256": payload["support_csv"]["sha256"],
        "rank_audit": payload["rank_audit"],
        "coupled_identifiability": payload["coupled_identifiability"],
    }, indent=2))


if __name__ == "__main__":
    main()
