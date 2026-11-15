#!/usr/bin/env python3
"""Verify the released records without training, sampling or running ANSYS."""
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REV = ROOT / "revision_experiments"
HOST = REV / "cai_r2_caseC_HOST_RUN_20260920_125550"
CLOSURE = REV / "cai_r2_provenance_closure_20260920_083412"

def require(condition, message):
    if not condition:
        raise ValueError(message)

def close(actual, expected, label):
    require(math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-11),
            f"{label}: {actual} != {expected}")

def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def verify_manifest():
    entries = read_json(ROOT / "RELEASE_MANIFEST.json")["files"]
    for entry in entries:
        path = ROOT / entry["path"]
        require(path.is_file(), f"Missing file: {entry['path']}")
        require(path.stat().st_size == entry["bytes"], f"Size mismatch: {entry['path']}")
        require(digest(path) == entry["sha256"], f"Hash mismatch: {entry['path']}")
    print(f"PASS: {len(entries)} released file checksums")

def verify_datasets():
    expected = {"dataset_train.csv": 8009, "dataset_test.csv": 1995,
                "dataset_train_broad_ext.csv": 12323, "dataset_coupled_train.csv": 5000}
    theta_sets = {}
    for name, count in expected.items():
        data = rows(ROOT / name)
        require(len(data) == count, f"Dataset row count: {name}")
        keys = [key for key in data[0] if key.startswith("e") and key[1:2].isdigit()]
        for row in data:
            values = [float(row[f"f{i}"]) for i in range(1, 21)]
            require(all(math.isfinite(v) and v > 0 for v in values), f"Invalid mode in {name}")
        theta_sets[name] = {tuple(float(row[key]) for key in keys) for row in data}
    require(not (theta_sets["dataset_train.csv"] & theta_sets["dataset_test.csv"]),
            "Original training/test parameter sets overlap")
    print("PASS: four FE dataset counts, 20 positive modes, independent original train/test vectors")

def verify_case_c():
    output = HOST / "outputs"
    fe = rows(output / "HOST_CASE_C_FE.csv")
    points = rows(output / "HOST_CASE_C_METRICS_PER_POINT.csv")
    selected = rows(CLOSURE / "CASE_C_SELECTED_THETA.csv")
    metrics = read_json(output / "HOST_CASE_C_METRICS.json")
    summary = metrics["new_case_c"]
    require(len(fe) == len(points) == len(selected) == summary["n_points"] == 50,
            "Case C requires exactly 50 complete records")
    obs = metrics["target"]["observation_hz"]
    k2 = metrics["target"]["kappa_squared"]
    require(obs == [2.06, 2.30, 2.62, 6.14, 6.53, 6.76] and k2 == 0.0009,
            "Unexpected Case C likelihood")
    require(Counter(int(r["chain_seed"]) for r in fe) == {811: 13, 812: 13, 813: 12, 814: 12},
            "Case C chain allocation")
    for i, (f, p, s) in enumerate(zip(fe, points, selected)):
        require(int(f["selection_order"]) == int(p["selection_order"]) == int(s["selection_order"]) == i,
                f"Case C order mismatch at {i}")
        for key in ("chain_position", "chain_seed", "retained_index_zero_based", "e1", "e2", "e3"):
            close(f[key], p[key], f"per-point {i}/{key}")
            close(f[key], s[key], f"selection {i}/{key}")
        require(f["solver_status"] == "SUCCESS" and int(f["process_exit_code"]) == 0
                and int(f["modal_frequency_count"]) == 20, f"Unsuccessful solve: {i}")
        values = [float(f[f"fe_f{m:02d}_hz"]) for m in range(1, 21)]
        require(all(math.isfinite(v) and v > 0 for v in values), f"Invalid Case C modes: {i}")
        folder = output / f"case_{i:03d}"
        require(digest(folder / "run.mac") == f["input_run_mac_sha256"], f"APDL hash: {i}")
        require(digest(HOST / "inputs" / f"case_{i:03d}" / "run.mac") == f["input_run_mac_sha256"],
                f"Locked input hash: {i}")
        require(digest(folder / "modal_freqs.txt") == f["output_modal_freqs_sha256"], f"Modal hash: {i}")
        saved = [float(x) for x in (folder / "modal_freqs.txt").read_text().split()]
        require(saved == values, f"Modal CSV/text mismatch: {i}")
        residuals = [(v - y) / y for v, y in zip(values[:6], obs)]
        ell = -0.5 * sum(x*x for x in residuals) / k2
        close(ell, p["ell_fe_nat"], f"FE likelihood {i}")
        close(max(abs(x) for x in residuals), p["rho_point"], f"rho {i}")
        close(abs(float(p["ell_surrogate_nat"]) - ell), p["abs_delta_ell_nat"], f"loglik difference {i}")
    delta = max(float(r["first_six_max_relative_error"]) for r in points)
    rho = max(float(r["rho_point"]) for r in points)
    errors = [float(r["abs_delta_ell_nat"]) for r in points]
    expected = {
        "delta_post": delta, "rho": rho,
        "standard_loglik_bound_nat": 6 * delta * (2 + delta) / (2*k2),
        "residual_aware_loglik_bound_nat": 6 * delta * (2*rho + delta) / (2*k2),
        "max_abs_delta_ell_nat": max(errors), "mean_abs_delta_ell_nat": statistics.mean(errors),
        "first_six_max_relative_surrogate_error": delta,
        "all_20_max_relative_surrogate_error": max(float(r["all_20_max_relative_error"]) for r in points)}
    for key, value in expected.items():
        close(value, summary[key], key)
    hashes = {
        CLOSURE / "CASE_C_SELECTED_THETA.csv": "054bbbcbe4c75a604670faa5d6a63f9e8ec309c0a1dfdd853a968cfc2b427bf5",
        CLOSURE / "CASE_C_SELECTED_INDICES.json": "4188aca464376a1a4679206653f0281c686cbdff7c5e02d84637aca81a10817f",
        REV / "cai_r2_presubmit_phaseB_20260919_213647/COUPLED_UNIFORM/COUPLED_UNIFORM_NUTS_REFERENCE_2000.npz":
            "c3174b9f49aed3cf0cd2ef68afddd7290ad10f47f5c20744ea60491cc1819d60"}
    for path, expected_hash in hashes.items():
        require(digest(path) == expected_hash, f"Scientific input hash: {path.name}")
    for filename, key in (("HOST_CASE_C_FE.csv", "host_fe_csv_sha256"),
                          ("HOST_CASE_C_RUN_STATUS.json", "host_run_status_sha256"),
                          ("HOST_CASE_C_METRICS_PER_POINT.csv", "per_point_metrics_sha256")):
        require(digest(output / filename) == metrics["hashes"][key], f"Original metric hash: {filename}")
    print("PASS: Case C 50/50, selection/input/output hashes, first-six likelihood arithmetic and summary metrics")

def verify_timings():
    data = rows(CLOSURE / "MAP_COMPLETION_TIMING.csv")
    for case in ("Sailboat", "Coupled"):
        timed = [float(r["wall_time_s"]) for r in data if r["case"] == case and r["row_type"] == "timed"]
        summary = next(r for r in data if r["case"] == case and r["row_type"] == "summary")
        require(len(timed) == int(summary["n_timed"]) == 5, f"Timing count: {case}")
        close(statistics.median(timed), summary["median_s"], f"Timing median: {case}")
        close(statistics.mean(timed), summary["mean_s"], f"Timing mean: {case}")
    print("PASS: MAP-completion timing summaries")

def main():
    verify_manifest()
    verify_datasets()
    verify_case_c()
    verify_timings()
    print("Release verification passed. No FE solve, model evaluation, training or MCMC was run.")

if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
