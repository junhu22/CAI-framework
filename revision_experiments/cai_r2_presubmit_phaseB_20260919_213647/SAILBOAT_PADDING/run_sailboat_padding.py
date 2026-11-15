import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance

ROOT = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
OUT = ROOT / "revision_experiments/cai_r2_presubmit_phaseB_20260919_213647/SAILBOAT_PADDING"
SURROGATE_PATH = ROOT / "surrogate_models_broad_ext/member_0.pt"
CNF_PATH = ROOT / "experiments/results/conditional_flow_broad_ext.pt"
REFERENCE_PATH = ROOT / "revision_experiments/cai_r2_stage1/final_ext_verification_20260918_091227/reference_quality_4chain/REFERENCE4_sailboat_field_frozen.npz"
SEEDS = [42, 43, 44, 45, 46]
N_FAST = 2000
SIGMA = 0.30
KAPPA_SQ = 0.001
Y6_DECIMAL = [0.91, 1.08, 1.47, 2.55, 3.16, 3.81]

sys.path.insert(0, str(ROOT / "experiments"))
from cai_lib import KModeLogPosterior, load_cnf, load_surrogate  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def components(surrogate, z, y6):
    theta = torch.exp(z)
    pred = surrogate(theta)[:, :6]
    residual = (pred - y6) / y6
    ll = -0.5 / KAPPA_SQ * (residual ** 2).sum(dim=-1)
    prior = (-0.5 * (z / SIGMA) ** 2 - z).sum(dim=-1)
    return ll, prior, ll + prior


def grad_inf(surrogate, z_value, y6):
    z = z_value.detach().clone().requires_grad_(True)
    obj = components(surrogate, z.unsqueeze(0), y6)[2].sum()
    grad = torch.autograd.grad(obj, z)[0]
    return float(grad.abs().max().detach()), float(obj.detach())


def raw_ess(log_weights):
    lw = log_weights.astype(np.float64)
    lw = lw - np.max(lw)
    w = np.exp(lw)
    w /= w.sum()
    return float(1.0 / np.sum(w ** 2) / len(w))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "SAILBOAT_MAP_OPTIMIZATION.json"
    if json_path.exists():
        raise RuntimeError("Refusing to overwrite sailboat MAP results")
    assert sha(SURROGATE_PATH).lower() == "4821100192a768327212ed724090732e63483650f4e80421f89a8df3f984f447"
    assert sha(CNF_PATH).lower() == "403e36617eab8f9f57b54f7e728f078c536fa3256390eb4851c271431a3af045"

    # Float32 model is retained for exact replay of CNF conditioning and weights.
    surrogate_f32 = load_surrogate(SURROGATE_PATH)
    cnf = load_cnf(CNF_PATH)
    surrogate_f64 = load_surrogate(SURROGATE_PATH).double()
    y6_f32 = torch.tensor(Y6_DECIMAL, dtype=torch.float32)
    y6_f64 = y6_f32.double()  # preserve the exact values stored by KModeLogPosterior

    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260919)
    random_starts = torch.exp(SIGMA * torch.randn((31, 6), dtype=torch.float64, generator=generator))
    theta_init = torch.cat([torch.ones((1, 6), dtype=torch.float64), random_starts], dim=0)
    init_path = OUT / "SAILBOAT_MAP_INIT_BANK.npy"
    np.save(init_path, theta_init.numpy())

    z = torch.log(theta_init).clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.05)
    adam_trace = []
    for iteration in range(1, 1001):
        opt.zero_grad()
        ll, prior, lp = components(surrogate_f64, z, y6_f64)
        loss = -lp.sum()
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()
        if iteration == 1 or iteration % 50 == 0 or iteration == 1000:
            adam_trace.append({
                "iteration": iteration,
                "best_log_posterior": float(lp.max().detach()),
                "worst_log_posterior": float(lp.min().detach()),
                "all_finite": bool(torch.isfinite(lp).all()),
            })

    with torch.no_grad():
        ll_a, prior_a, lp_a = components(surrogate_f64, z, y6_f64)
        top_indices = torch.argsort(lp_a, descending=True)[:4].cpu().tolist()

    refined = []
    for rank, index in enumerate(top_indices):
        zr = z[index].detach().clone().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [zr], lr=1.0, max_iter=300, max_eval=375,
            tolerance_grad=1e-12, tolerance_change=1e-12,
            history_size=100, line_search_fn="strong_wolfe",
        )
        calls = {"n": 0}

        def closure():
            optimizer.zero_grad()
            lp = components(surrogate_f64, zr.unsqueeze(0), y6_f64)[2].sum()
            (-lp).backward()
            calls["n"] += 1
            return -lp

        optimizer.step(closure)
        with torch.no_grad():
            ll, prior, lp = components(surrogate_f64, zr.unsqueeze(0), y6_f64)
        gi, _ = grad_inf(surrogate_f64, zr, y6_f64)
        refined.append({
            "rank_from_adam": rank,
            "start_index": int(index),
            "theta": torch.exp(zr).detach().cpu().tolist(),
            "z": zr.detach().cpu().tolist(),
            "log_likelihood": float(ll.item()),
            "log_prior_theta_density": float(prior.item()),
            "log_posterior": float(lp.item()),
            "gradient_inf_norm": gi,
            "lbfgs_closure_calls": calls["n"],
            "finite": bool(torch.isfinite(zr).all() and torch.isfinite(lp).all()),
        })
    refined.sort(key=lambda r: r["log_posterior"], reverse=True)
    best = refined[0]
    best_theta = np.asarray(best["theta"], dtype=np.float64)
    replication = []
    for item in refined[1:]:
        theta = np.asarray(item["theta"], dtype=np.float64)
        replication.append({
            "start_index": item["start_index"],
            "log_posterior_abs_difference": abs(item["log_posterior"] - best["log_posterior"]),
            "max_relative_theta_difference": float(np.max(np.abs(theta - best_theta) / np.maximum(np.abs(best_theta), 1e-15))),
        })
    replicated = [r for r in replication if r["log_posterior_abs_difference"] <= 1e-6 and r["max_relative_theta_difference"] <= 1e-4]
    gate = bool(best["finite"] and best["gradient_inf_norm"] <= 1e-5 and len(replicated) >= 1)

    with torch.no_grad():
        ll0, pr0, lp0 = components(surrogate_f64, torch.log(theta_init), y6_f64)
        ll_adam, pr_adam, lp_adam = components(surrogate_f64, z.detach(), y6_f64)
    starts = []
    for i in range(32):
        gi, _ = grad_inf(surrogate_f64, z[i].detach(), y6_f64)
        starts.append({
            "start_index": i,
            "theta_initial": theta_init[i].tolist(),
            "initial_log_likelihood": float(ll0[i]),
            "initial_log_prior_theta_density": float(pr0[i]),
            "initial_log_posterior": float(lp0[i]),
            "adam_theta": torch.exp(z[i]).detach().cpu().tolist(),
            "adam_log_likelihood": float(ll_adam[i]),
            "adam_log_prior_theta_density": float(pr_adam[i]),
            "adam_log_posterior": float(lp_adam[i]),
            "adam_gradient_inf_norm": gi,
            "adam_finite": bool(torch.isfinite(z[i]).all() and torch.isfinite(lp_adam[i])),
        })
    result = {
        "optimization_gate": "PASS" if gate else "FAIL",
        "objective": "theta-space posterior density; z=log(theta) is numerical only; no transform Jacobian added",
        "precision": "float64 surrogate/objective/optimizers",
        "settings": {
            "n_starts": 32,
            "init_seed": 20260919,
            "adam_iterations": 1000,
            "adam_lr": 0.05,
            "lbfgs_refined_candidates": 4,
            "lbfgs_max_iter": 300,
            "lbfgs_line_search": "strong_wolfe",
            "kappa_sq": KAPPA_SQ,
            "prior": "independent LogNormal(log_mean=0, log_sigma=0.30), theta-space density",
        },
        "y6_decimal": Y6_DECIMAL,
        "y6_float32_values_used_by_final_target": y6_f32.tolist(),
        "init_bank_path": str(init_path),
        "init_bank_sha256": sha(init_path),
        "surrogate_sha256": sha(SURROGATE_PATH),
        "cnf_sha256": sha(CNF_PATH),
        "adam_trace": adam_trace,
        "all_starts": starts,
        "lbfgs_refinements": refined,
        "replication_comparisons": replication,
        "replicated_candidates_excluding_best": len(replicated),
        "theta_MAP": best["theta"],
        "theta_MAP_distance_from_one": {
            "l2": float(np.linalg.norm(best_theta - 1.0)),
            "max_abs": float(np.max(np.abs(best_theta - 1.0))),
            "mean_abs": float(np.mean(np.abs(best_theta - 1.0))),
        },
    }
    write_json(json_path, result)
    print("SAILBOAT_MAP_GATE", result["optimization_gate"], best["theta"], best["log_posterior"], best["gradient_inf_norm"], len(replicated), flush=True)
    if not gate:
        return

    theta_map_f32 = torch.tensor(best_theta, dtype=torch.float32)
    with torch.no_grad():
        f_nominal = surrogate_f32(torch.ones((1, 6), dtype=torch.float32)).squeeze(0)
        f_map = surrogate_f32(theta_map_f32.unsqueeze(0)).squeeze(0)
    y_nominal = torch.cat([y6_f32, f_nominal[6:20]])
    y_map = torch.cat([y6_f32, f_map[6:20]])
    high_diff = (y_map[6:] - y_nominal[6:]).detach().cpu().numpy().astype(np.float64)
    result["padding"] = {
        "y_nominalpad": y_nominal.tolist(),
        "y_MAPpad": y_map.tolist(),
        "higher_mode_difference": high_diff.tolist(),
        "higher_mode_l2_difference": float(np.linalg.norm(high_diff)),
        "higher_mode_max_abs_difference": float(np.max(np.abs(high_diff))),
        "higher_mode_mean_abs_difference": float(np.mean(np.abs(high_diff))),
        "higher_mode_relative_l2_difference": float(np.linalg.norm(high_diff) / np.linalg.norm(y_nominal[6:].numpy())),
    }
    write_json(json_path, result)

    log_post = KModeLogPosterior(surrogate_f32, y6_f32.tolist(), kappa_sq=KAPPA_SQ, prior_sigma=SIGMA)
    with np.load(REFERENCE_PATH, allow_pickle=False) as r:
        reference = r["chains"].reshape(-1, 6).astype(np.float64)
    ref_mean = reference.mean(axis=0)
    ref_q = np.quantile(reference, [0.025, 0.975], axis=0)
    all_theta = np.empty((2, len(SEEDS), N_FAST, 6), dtype=np.float32)
    all_logw = np.empty((2, len(SEEDS), N_FAST), dtype=np.float64)
    all_z = np.empty((len(SEEDS), N_FAST, 6), dtype=np.float32)
    base_rows = []
    for si, seed in enumerate(SEEDS):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        zbase = torch.randn((N_FAST, 6), dtype=torch.float32, generator=gen)
        all_z[si] = zbase.numpy()
        for pi, (padding_name, ypad) in enumerate([("nominal", y_nominal), ("MAP", y_map)]):
            ybatch = ypad.unsqueeze(0).expand(N_FAST, -1)
            with torch.no_grad():
                theta, _ = cnf.inverse(zbase, ybatch)
                log_target = log_post(theta)
                log_q = cnf.log_prob(theta, ypad)
                logw = (log_target - log_q).double().numpy()
            theta_np = theta.numpy().astype(np.float64)
            all_theta[pi, si] = theta.numpy()
            all_logw[pi, si] = logw
            fq = np.quantile(theta_np, [0.025, 0.975], axis=0)
            w1 = np.asarray([wasserstein_distance(theta_np[:, j], reference[:, j]) for j in range(6)])
            width_ratio = (fq[1] - fq[0]) / (ref_q[1] - ref_q[0])
            base_rows.append({
                "padding": padding_name,
                "seed": seed,
                "N": N_FAST,
                **{f"fast_mean_e{j+1}": theta_np[:, j].mean() for j in range(6)},
                **{f"fast_std_e{j+1}": theta_np[:, j].std(ddof=1) for j in range(6)},
                "raw_ess_ratio": raw_ess(logw),
                "active_mean_mae": float(np.abs(theta_np.mean(0)[:3] - ref_mean[:3]).mean()),
                "active_w1_mean": float(w1[:3].mean()),
                "active_w1_max": float(w1[:3].max()),
                "active_width_ratio_mean": float(width_ratio[:3].mean()),
                "active_width_ratio_min": float(width_ratio[:3].min()),
                "active_width_ratio_max": float(width_ratio[:3].max()),
            })
    np.savez_compressed(
        OUT / "SAILBOAT_PADDING_PAIRED_RAW.npz",
        theta=all_theta,
        log_weights=all_logw,
        latent_z=all_z,
        seeds=np.asarray(SEEDS),
        y_nominal=y_nominal.numpy(),
        y_map=y_map.numpy(),
        theta_map=best_theta,
        reference=reference,
    )
    fields = list(base_rows[0])
    with (OUT / "SAILBOAT_PADDING_BASE.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(base_rows)
    print("SAILBOAT_PAIRED_BASE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

