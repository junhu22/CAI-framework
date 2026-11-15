import csv
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.optimize import minimize


ROOT = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
OUT = ROOT / "revision_experiments/cai_r2_provenance_closure_20260920_083412"
PHASE_B = ROOT / "revision_experiments/cai_r2_presubmit_phaseB_20260919_213647"
TIMING_CSV = OUT / "MAP_COMPLETION_TIMING.csv"
RAW_JSON = OUT / "MAP_COMPLETION_TIMING_RAW.json"
N_WARMUP = 1
N_TIMED = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def max_relative_difference(actual, expected):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    return float(np.max(np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-15)))


def sailboat_benchmark():
    sys.path.insert(0, str(ROOT / "experiments"))
    from cai_lib import load_surrogate

    surrogate_path = ROOT / "surrogate_models_broad_ext/member_0.pt"
    init_path = PHASE_B / "SAILBOAT_PADDING/SAILBOAT_MAP_INIT_BANK.npy"
    retained_path = PHASE_B / "SAILBOAT_PADDING/SAILBOAT_MAP_OPTIMIZATION.json"
    retained = json.loads(retained_path.read_text(encoding="utf-8"))
    retained_theta = np.asarray(retained["theta_MAP"], dtype=np.float64)
    theta_init = np.load(init_path, allow_pickle=False).astype(np.float64)
    if theta_init.shape != (32, 6):
        raise RuntimeError(f"Unexpected Sailboat start-bank shape {theta_init.shape}")
    if sha256(init_path) != "07136b2bd25ba46c7d26d873aba96ad8293f12b65d6944dc2578448af89a911b":
        raise RuntimeError("Sailboat start-bank hash mismatch")
    surrogate_f32 = load_surrogate(surrogate_path)
    surrogate_f64 = load_surrogate(surrogate_path).double()
    y6_f32 = torch.tensor([0.91, 1.08, 1.47, 2.55, 3.16, 3.81], dtype=torch.float32)
    y6_f64 = y6_f32.double()
    sigma = 0.30
    kappa_sq = 0.001

    def components(z):
        theta = torch.exp(z)
        prediction = surrogate_f64(theta)[:, :6]
        residual = (prediction - y6_f64) / y6_f64
        ll = -0.5 / kappa_sq * (residual ** 2).sum(dim=-1)
        prior = (-0.5 * (z / sigma) ** 2 - z).sum(dim=-1)
        return ll, prior, ll + prior

    def grad_inf(z_value):
        z_value = z_value.detach().clone().requires_grad_(True)
        objective = components(z_value.unsqueeze(0))[2].sum()
        gradient = torch.autograd.grad(objective, z_value)[0]
        return float(gradient.abs().max().detach())

    def one_run():
        t0 = time.perf_counter()
        z = torch.log(torch.from_numpy(theta_init).double()).clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=0.05)
        for _ in range(1000):
            optimizer.zero_grad()
            lp = components(z)[2]
            loss = -lp.sum()
            if not torch.isfinite(loss):
                break
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            lp_adam = components(z)[2]
            top_indices = torch.argsort(lp_adam, descending=True)[:4].cpu().tolist()
        refined = []
        for index in top_indices:
            candidate = z[index].detach().clone().requires_grad_(True)
            lbfgs = torch.optim.LBFGS(
                [candidate], lr=1.0, max_iter=300, max_eval=375,
                tolerance_grad=1e-12, tolerance_change=1e-12,
                history_size=100, line_search_fn="strong_wolfe",
            )

            def closure():
                lbfgs.zero_grad()
                log_posterior = components(candidate.unsqueeze(0))[2].sum()
                (-log_posterior).backward()
                return -log_posterior

            lbfgs.step(closure)
            with torch.no_grad():
                log_posterior = float(components(candidate.unsqueeze(0))[2].item())
            refined.append({
                "theta": torch.exp(candidate).detach().cpu().numpy(),
                "z": candidate.detach().clone(),
                "log_posterior": log_posterior,
                "gradient_inf_norm": grad_inf(candidate),
            })
        refined.sort(key=lambda item: item["log_posterior"], reverse=True)
        best = refined[0]
        theta_map = best["theta"]
        replicated = sum(
            abs(item["log_posterior"] - best["log_posterior"]) <= 1e-6
            and max_relative_difference(item["theta"], theta_map) <= 1e-4
            for item in refined[1:]
        )
        with torch.no_grad():
            f_map = surrogate_f32(torch.tensor(theta_map, dtype=torch.float32).unsqueeze(0)).squeeze(0)
            conditioning = torch.cat([y6_f32, f_map[6:20]]).cpu().numpy()
        elapsed = time.perf_counter() - t0
        relative_difference = max_relative_difference(theta_map, retained_theta)
        internal_gate = bool(np.isfinite(theta_map).all() and best["gradient_inf_norm"] <= 1e-5 and replicated >= 1)
        return {
            "wall_time_s": elapsed, "theta_map": theta_map.tolist(),
            "max_relative_difference_from_retained": relative_difference,
            "conditioning_vector": conditioning.tolist(),
            "internal_optimization_gate": internal_gate,
            "gradient_inf_norm": best["gradient_inf_norm"],
            "replicated_candidates_excluding_best": int(replicated),
        }

    runs = []
    for run_index in range(N_WARMUP + N_TIMED):
        result = one_run()
        result.update({"run_index": run_index, "run_type": "warmup" if run_index == 0 else "timed"})
        runs.append(result)
        print(f"MAP_TIMING Sailboat {result['run_type']} {run_index} {result['wall_time_s']:.6f}s rel={result['max_relative_difference_from_retained']:.3e}", flush=True)
    return {
        "case": "Sailboat", "retained_theta_map": retained_theta.tolist(), "runs": runs,
        "algorithm": "retained 32-start float64 Adam(1000,lr=0.05), top-four PyTorch L-BFGS, float32 surrogate completion",
        "start_bank_path": str(init_path), "start_bank_sha256": sha256(init_path),
        "surrogate_path": str(surrogate_path), "surrogate_sha256": sha256(surrogate_path),
        "routine_path": str(PHASE_B / "SAILBOAT_PADDING/run_sailboat_padding.py"),
        "routine_sha256": sha256(PHASE_B / "SAILBOAT_PADDING/run_sailboat_padding.py"),
    }


def coupled_benchmark():
    source = ROOT / "experiments/coupled_building/04_train_cnf_coupled.py"
    spec = importlib.util.spec_from_file_location("coupled_source_for_timing", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    surrogate_path = ROOT / "surrogate_models_coupled/member_0.pt"
    retained_path = PHASE_B / "COUPLED_UNIFORM/COUPLED_UNIFORM_MAP_OPTIMIZATION.json"
    retained = json.loads(retained_path.read_text(encoding="utf-8"))
    retained_theta = np.asarray(retained["theta_MAP_uniform"], dtype=np.float64)
    surrogate = module.Surrogate().double().cpu()
    surrogate.load_state_dict(torch.load(surrogate_path, map_location="cpu", weights_only=True))
    surrogate.eval()
    y6 = torch.tensor([2.06, 2.30, 2.62, 6.14, 6.53, 6.76], dtype=torch.float32).double()
    lo, hi, kappa_sq = 0.5, 4.0, 0.03 ** 2
    grid = [0.55, 2.25, 3.95]
    starts = np.asarray([[a, b, c] for a in grid for b in grid for c in grid], dtype=np.float64)
    generator = np.random.default_rng(20260919)
    starts = np.vstack([starts, generator.uniform(lo, hi, size=(5, 3))])
    if starts.shape != (32, 3):
        raise RuntimeError(f"Unexpected coupled start-bank shape {starts.shape}")

    def value_grad(theta_np):
        theta = torch.tensor(theta_np, dtype=torch.float64, requires_grad=True)
        prediction = surrogate(theta.unsqueeze(0)).squeeze(0)[:6]
        value = 0.5 / kappa_sq * (((prediction - y6) / y6) ** 2).sum()
        gradient = torch.autograd.grad(value, theta)[0]
        return float(value.detach()), gradient.detach().numpy().astype(np.float64)

    def one_run():
        t0 = time.perf_counter()
        records = []
        for start in starts:
            result = minimize(
                fun=lambda x: value_grad(x), x0=start.copy(), jac=True,
                method="L-BFGS-B", bounds=[(lo, hi)] * 3,
                options={"maxiter": 1000, "ftol": 1e-15, "gtol": 1e-10, "maxls": 50},
            )
            records.append({
                "theta": np.asarray(result.x, dtype=np.float64), "value": float(result.fun),
                "success": bool(result.success), "within_bounds": bool(np.all(result.x >= lo) and np.all(result.x <= hi)),
            })
        records.sort(key=lambda item: item["value"])
        best = records[0]
        theta_map = best["theta"]
        replicated = sum(
            abs(item["value"] - best["value"]) <= 1e-8
            and np.max(np.abs(item["theta"] - theta_map)) <= 1e-4
            for item in records[1:]
        )
        with torch.no_grad():
            prediction = surrogate(torch.tensor(theta_map, dtype=torch.float64).unsqueeze(0)).squeeze(0).cpu().numpy()
        conditioning = np.asarray([2.06, 2.30, 2.62, 6.14, 6.53, 6.76] + prediction[6:20].tolist(), dtype=np.float32)
        elapsed = time.perf_counter() - t0
        relative_difference = max_relative_difference(theta_map, retained_theta)
        internal_gate = bool(best["success"] and best["within_bounds"] and replicated >= 1)
        return {
            "wall_time_s": elapsed, "theta_map": theta_map.tolist(),
            "max_relative_difference_from_retained": relative_difference,
            "conditioning_vector": conditioning.tolist(),
            "internal_optimization_gate": internal_gate,
            "negative_log_likelihood": best["value"],
            "replicated_candidates_excluding_best": int(replicated),
        }

    runs = []
    for run_index in range(N_WARMUP + N_TIMED):
        result = one_run()
        result.update({"run_index": run_index, "run_type": "warmup" if run_index == 0 else "timed"})
        runs.append(result)
        print(f"MAP_TIMING Coupled {result['run_type']} {run_index} {result['wall_time_s']:.6f}s rel={result['max_relative_difference_from_retained']:.3e}", flush=True)
    return {
        "case": "Coupled", "retained_theta_map": retained_theta.tolist(), "runs": runs,
        "algorithm": "retained 32-start float64 bounded L-BFGS-B with analytic autograd gradient; float64 stored-prediction completion cast to float32",
        "start_bank": starts.tolist(),
        "surrogate_path": str(surrogate_path), "surrogate_sha256": sha256(surrogate_path),
        "routine_path": str(PHASE_B / "COUPLED_UNIFORM/run_uniform_map_nuts.py"),
        "routine_sha256": sha256(PHASE_B / "COUPLED_UNIFORM/run_uniform_map_nuts.py"),
    }


def summarize(case):
    timed = np.asarray([run["wall_time_s"] for run in case["runs"] if run["run_type"] == "timed"], dtype=np.float64)
    q1, q3 = np.quantile(timed, [0.25, 0.75])
    all_pass = all(
        run["max_relative_difference_from_retained"] <= 1e-6 and run["internal_optimization_gate"]
        for run in case["runs"]
    )
    return {
        "n_warmup": N_WARMUP, "n_timed": N_TIMED, "mean_s": float(timed.mean()),
        "median_s": float(np.median(timed)), "min_s": float(timed.min()), "max_s": float(timed.max()),
        "q1_s": float(q1), "q3_s": float(q3), "iqr_s": float(q3 - q1),
        "max_relative_difference_across_all_runs": float(max(run["max_relative_difference_from_retained"] for run in case["runs"])),
        "gate": "PASS" if all_pass else "FAIL",
    }


def main():
    if TIMING_CSV.exists() or RAW_JSON.exists():
        raise RuntimeError("Refusing to overwrite MAP timing outputs")
    cases = [sailboat_benchmark(), coupled_benchmark()]
    for case in cases:
        case["summary"] = summarize(case)
    environment = {
        "platform": platform.platform(), "processor": platform.processor(),
        "python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
        "scipy": scipy.__version__, "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(), "device_used": "cpu",
        "timer": "time.perf_counter wall clock",
    }
    payload = {"schema": "actual-map-conditioning-completion-timing-v1", "environment": environment, "cases": cases}
    RAW_JSON.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    fields = ["case", "row_type", "run_index", "wall_time_s", "theta_map", "max_relative_difference_from_retained", "internal_optimization_gate", "mean_s", "median_s", "min_s", "max_s", "q1_s", "q3_s", "iqr_s", "gate"]
    with TIMING_CSV.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            for run in case["runs"]:
                writer.writerow({
                    "case": case["case"], "row_type": run["run_type"], "run_index": run["run_index"],
                    "wall_time_s": repr(float(run["wall_time_s"])),
                    "theta_map": json.dumps(run["theta_map"], separators=(",", ":")),
                    "max_relative_difference_from_retained": repr(float(run["max_relative_difference_from_retained"])),
                    "internal_optimization_gate": run["internal_optimization_gate"], "gate": "PASS" if run["max_relative_difference_from_retained"] <= 1e-6 and run["internal_optimization_gate"] else "FAIL",
                })
            writer.writerow({"case": case["case"], "row_type": "summary", **case["summary"]})
    print(json.dumps({case["case"]: case["summary"] for case in cases}, indent=2), flush=True)


if __name__ == "__main__":
    main()
