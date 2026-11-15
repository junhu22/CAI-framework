from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch


sys.dont_write_bytecode = True

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
INPUTS = PACKAGE / "inputs"
OUTPUTS = PACKAGE / "outputs"
HOST_CSV = OUTPUTS / "HOST_CASE_C_FE.csv"
HOST_STATUS = OUTPUTS / "HOST_CASE_C_RUN_STATUS.json"
SELECTED = INPUTS / "CASE_C_SELECTED_THETA.csv"
SURROGATE = ROOT / "surrogate_models_coupled" / "member_0.pt"
SURROGATE_SOURCE = ROOT / "experiments" / "coupled_building" / "04_train_cnf_coupled.py"

EXPECTED_SELECTED_SHA256 = "054bbbcbe4c75a604670faa5d6a63f9e8ec309c0a1dfdd853a968cfc2b427bf5"
EXPECTED_SURROGATE_SHA256 = "6a1e4f530d4e3628e0b7fdbf08661b4dc32a71a0c2b538b26f5af9d7c07b9986"
EXPECTED_SURROGATE_SOURCE_SHA256 = "844446aff945285062f9b0b9a1e536252233718e6d5aebb3642ae53a2513369f"

OBSERVATION = np.asarray([2.06, 2.30, 2.62, 6.14, 6.53, 6.76], dtype=np.float64)
KAPPA = 0.03
KAPPA_SQUARED = 9.0e-4
M = 6

OLD_SUPERSEDED = {
    "classification": "OLD_SUPERSEDED_LOGNORMAL_REGION",
    "delta_post": 0.0017278678,
    "rho": 0.5915577748,
    "standard_loglik_bound_nat": 11.529070318712241,
    "residual_aware_loglik_bound_nat": 6.824175901102032,
    "max_abs_delta_ell_nat": 1.02402296,
    "mean_abs_delta_ell_nat": 0.03803122,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    metrics_path = OUTPUTS / "HOST_CASE_C_METRICS.json"
    per_point_path = OUTPUTS / "HOST_CASE_C_METRICS_PER_POINT.csv"
    for path in (metrics_path, per_point_path):
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite an existing metric artifact: {path}")

    if sha256(SELECTED) != EXPECTED_SELECTED_SHA256:
        raise RuntimeError("Locked theta hash mismatch")
    if sha256(SURROGATE) != EXPECTED_SURROGATE_SHA256:
        raise RuntimeError("Frozen coupled surrogate hash mismatch")
    if sha256(SURROGATE_SOURCE) != EXPECTED_SURROGATE_SOURCE_SHA256:
        raise RuntimeError("Frozen coupled surrogate source hash mismatch")

    status = json.loads(HOST_STATUS.read_text(encoding="utf-8-sig"))
    if status.get("CASE_C_HOST_RUN") != "PASS":
        raise RuntimeError("Metrics are forbidden unless CASE_C_HOST_RUN is PASS")
    if status.get("run_complete") is not True or status.get("valid_20_mode_outputs") != 50:
        raise RuntimeError("Metrics require a complete 50/50 host FE dataset")
    if status.get("retries") != 0 or status.get("replacements") != 0 or status.get("extra_points") != 0:
        raise RuntimeError("Run-status provenance violates the no-retry/no-replacement protocol")

    selected = load_csv(SELECTED)
    host = load_csv(HOST_CSV)
    if len(selected) != 50 or len(host) != 50:
        raise RuntimeError("Expected exactly 50 selected rows and 50 host FE rows")

    theta = np.empty((50, 3), dtype=np.float64)
    fe = np.empty((50, 20), dtype=np.float64)
    for index, (source_row, host_row) in enumerate(zip(selected, host)):
        if int(source_row["selection_order"]) != index or int(host_row["selection_order"]) != index:
            raise RuntimeError(f"Row order mismatch at position {index}")
        for key in ("chain_position", "chain_seed", "retained_index_zero_based", "e1", "e2", "e3"):
            if source_row[key] != host_row[key]:
                raise RuntimeError(f"Locked-source mismatch at row {index}, field {key}")
        if host_row["solver_status"] != "SUCCESS":
            raise RuntimeError(f"Non-success host row at position {index}")
        if int(host_row["modal_frequency_count"]) != 20:
            raise RuntimeError(f"Host row {index} does not contain 20 modes")
        theta[index] = [float(host_row["e1"]), float(host_row["e2"]), float(host_row["e3"])]
        fe[index] = [float(host_row[f"fe_f{mode:02d}_hz"]) for mode in range(1, 21)]

    if not np.all(np.isfinite(fe)) or not np.all(fe > 0.0):
        raise RuntimeError("FE frequencies must all be finite and positive")

    sys.path.insert(0, str(SURROGATE_SOURCE.parent))
    spec = importlib.util.spec_from_file_location("case_c_host_metric_surrogate", SURROGATE_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not import frozen surrogate source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Surrogate().float().cpu()
    model.load_state_dict(torch.load(SURROGATE, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        prediction = model(torch.tensor(theta, dtype=torch.float32)).cpu().numpy().astype(np.float64)
    if prediction.shape != (50, 20):
        raise RuntimeError(f"Unexpected surrogate prediction shape: {prediction.shape}")

    relative_error = np.abs(prediction - fe) / fe
    delta_post = float(np.max(relative_error[:, :6]))
    rho_by_point = np.max(np.abs((fe[:, :6] - OBSERVATION) / OBSERVATION), axis=1)
    rho = float(np.max(rho_by_point))

    normalized_fe_residual = (fe[:, :6] - OBSERVATION) / OBSERVATION
    normalized_surrogate_residual = (prediction[:, :6] - OBSERVATION) / OBSERVATION
    ell_fe = -0.5 / KAPPA_SQUARED * np.sum(normalized_fe_residual**2, axis=1)
    ell_surrogate = -0.5 / KAPPA_SQUARED * np.sum(normalized_surrogate_residual**2, axis=1)
    abs_delta_ell = np.abs(ell_surrogate - ell_fe)

    standard_bound = float(M * delta_post * (2.0 + delta_post) / (2.0 * KAPPA_SQUARED))
    residual_aware_bound = float(
        M * delta_post * (2.0 * rho + delta_post) / (2.0 * KAPPA_SQUARED)
    )
    first_six_max = float(np.max(relative_error[:, :6]))
    all_20_max = float(np.max(relative_error))

    point_fields = [
        "selection_order",
        "chain_position",
        "chain_seed",
        "retained_index_zero_based",
        "e1",
        "e2",
        "e3",
        "first_six_max_relative_error",
        "all_20_max_relative_error",
        "rho_point",
        "ell_fe_nat",
        "ell_surrogate_nat",
        "abs_delta_ell_nat",
    ]
    with per_point_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=point_fields, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(host):
            writer.writerow(
                {
                    **{key: row[key] for key in point_fields[:7]},
                    "first_six_max_relative_error": repr(float(np.max(relative_error[index, :6]))),
                    "all_20_max_relative_error": repr(float(np.max(relative_error[index]))),
                    "rho_point": repr(float(rho_by_point[index])),
                    "ell_fe_nat": repr(float(ell_fe[index])),
                    "ell_surrogate_nat": repr(float(ell_surrogate[index])),
                    "abs_delta_ell_nat": repr(float(abs_delta_ell[index])),
                }
            )

    new_current = {
        "classification": "NEW_CURRENT_BOUNDED_UNIFORM_REGION",
        "n_points": 50,
        "delta_post": delta_post,
        "rho": rho,
        "standard_loglik_bound_nat": standard_bound,
        "residual_aware_loglik_bound_nat": residual_aware_bound,
        "max_abs_delta_ell_nat": float(np.max(abs_delta_ell)),
        "mean_abs_delta_ell_nat": float(np.mean(abs_delta_ell)),
        "first_six_max_relative_surrogate_error": first_six_max,
        "all_20_max_relative_surrogate_error": all_20_max,
    }
    direction = {
        key: (
            "INCREASED"
            if new_current[key] > OLD_SUPERSEDED[key]
            else "DECREASED"
            if new_current[key] < OLD_SUPERSEDED[key]
            else "UNCHANGED"
        )
        for key in (
            "delta_post",
            "rho",
            "standard_loglik_bound_nat",
            "residual_aware_loglik_bound_nat",
            "max_abs_delta_ell_nat",
            "mean_abs_delta_ell_nat",
        )
    }
    metrics = {
        "schema": "cai-r2-case-c-host-metrics-v1",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "CASE_C_HOST_RUN": "PASS",
        "metrics_status": "COMPUTED_FROM_COMPLETE_50_POINT_HOST_RUN",
        "scientific_protocol": "UNCHANGED",
        "target": {
            "prior": "independent Uniform(0.5,4)^3",
            "likelihood_modes": 6,
            "observation_hz": OBSERVATION.tolist(),
            "kappa": KAPPA,
            "kappa_squared": KAPPA_SQUARED,
        },
        "old_case_c": OLD_SUPERSEDED,
        "new_case_c": new_current,
        "old_to_new_direction": direction,
        "hashes": {
            "selected_theta_sha256": sha256(SELECTED),
            "host_fe_csv_sha256": sha256(HOST_CSV),
            "host_run_status_sha256": sha256(HOST_STATUS),
            "surrogate_sha256": sha256(SURROGATE),
            "surrogate_source_sha256": sha256(SURROGATE_SOURCE),
            "per_point_metrics_sha256": sha256(per_point_path),
        },
        "no_accept_reject_threshold": True,
        "no_retry_or_point_substitution": True,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
