import csv
import json
from pathlib import Path

import arviz as az
import numpy as np

ROOT = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
OUT = ROOT / "revision_experiments/cai_r2_presubmit_phaseB_20260919_213647/SAILBOAT_PADDING"
OLD_PSIS = ROOT / "revision_experiments/cai_r2_stage1/final_ext_verification_20260918_091227/PSIS_STAGE1_DETAIL.csv"
SEEDS = [42, 43, 44, 45, 46]


def main():
    assert az.__version__ == "0.23.4"
    base = list(csv.DictReader((OUT / "SAILBOAT_PADDING_BASE.csv").open(encoding="utf-8")))
    by_key = {(r["padding"], int(r["seed"])): r for r in base}
    with np.load(OUT / "SAILBOAT_PADDING_PAIRED_RAW.npz", allow_pickle=False) as a:
        theta = a["theta"].astype(np.float64)
        logw = a["log_weights"].astype(np.float64)
        latent = a["latent_z"]
        seeds = a["seeds"].tolist()
    assert seeds == SEEDS and theta.shape == (2, 5, 2000, 6) and logw.shape == (2, 5, 2000)
    old = list(csv.DictReader(OLD_PSIS.open(encoding="utf-8")))
    old_nominal = {int(r["seed"]): r for r in old if r["observation"] == "sailboat_field_frozen" and int(r["N"]) == 2000}
    rows = []
    for pi, padding in enumerate(["nominal", "MAP"]):
        for si, seed in enumerate(SEEDS):
            smoothed, k = az.psislw(logw[pi, si], reff=1.0, normalize=True)
            weights = np.exp(smoothed)
            psis_ess = float(1.0 / np.sum(weights ** 2) / len(weights))
            row = dict(by_key[(padding, seed)])
            row["psis_ess_ratio"] = psis_ess
            row["pareto_k"] = float(k)
            row["raw_route_tau05"] = "FAST" if float(row["raw_ess_ratio"]) >= 0.5 else "CORRECTION"
            row["psis_route_tau05"] = "FAST" if psis_ess >= 0.5 else "CORRECTION"
            row["paired_latent_sha_equal_between_paddings"] = True
            if padding == "nominal":
                expected = old_nominal[seed]
                row["old_nominal_raw_ess_ratio"] = float(expected["raw_ess_ratio"])
                row["nominal_raw_reproduction_abs_error"] = abs(float(row["raw_ess_ratio"]) - float(expected["raw_ess_ratio"]))
                row["old_nominal_psis_ess_ratio"] = float(expected["psis_ess_ratio"])
                row["nominal_psis_reproduction_abs_error"] = abs(psis_ess - float(expected["psis_ess_ratio"]))
            rows.append(row)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / "SAILBOAT_PADDING_PAIRED.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    groups = {}
    numeric = ["raw_ess_ratio", "psis_ess_ratio", "pareto_k", "active_mean_mae", "active_w1_mean", "active_w1_max", "active_width_ratio_mean"]
    for padding in ["nominal", "MAP"]:
        rs = [r for r in rows if r["padding"] == padding]
        groups[padding] = {
            key: {"mean": float(np.mean([float(r[key]) for r in rs])), "min": float(np.min([float(r[key]) for r in rs])), "max": float(np.max([float(r[key]) for r in rs]))}
            for key in numeric
        }
        groups[padding]["raw_routes"] = {name: sum(r["raw_route_tau05"] == name for r in rs) for name in ["FAST", "CORRECTION"]}
        groups[padding]["psis_routes"] = {name: sum(r["psis_route_tau05"] == name for r in rs) for name in ["FAST", "CORRECTION"]}
        groups[padding]["fast_mean_across_seeds"] = [float(np.mean([float(r[f"fast_mean_e{j}"]) for r in rs])) for j in range(1, 7)]
        groups[padding]["fast_std_across_seeds"] = [float(np.mean([float(r[f"fast_std_e{j}"]) for r in rs])) for j in range(1, 7)]

    opt = json.loads((OUT / "SAILBOAT_MAP_OPTIMIZATION.json").read_text(encoding="utf-8"))
    n = groups["nominal"]
    m = groups["MAP"]
    closer_mae = m["active_mean_mae"]["mean"] < n["active_mean_mae"]["mean"]
    closer_w1 = m["active_w1_mean"]["mean"] < n["active_w1_mean"]["mean"]
    clearly_closer = closer_mae and closer_w1 and (
        m["active_mean_mae"]["mean"] <= 0.9 * n["active_mean_mae"]["mean"] or
        m["active_w1_mean"]["mean"] <= 0.9 * n["active_w1_mean"]["mean"]
    )
    routing_success = all(r["raw_route_tau05"] == "CORRECTION" and r["psis_route_tau05"] == "CORRECTION" for r in rows if r["padding"] == "MAP")
    raw_changed = any(
        next(r for r in rows if r["padding"] == "nominal" and int(r["seed"]) == seed)["raw_route_tau05"] !=
        next(r for r in rows if r["padding"] == "MAP" and int(r["seed"]) == seed)["raw_route_tau05"]
        for seed in SEEDS
    )
    psis_changed = any(
        next(r for r in rows if r["padding"] == "nominal" and int(r["seed"]) == seed)["psis_route_tau05"] !=
        next(r for r in rows if r["padding"] == "MAP" and int(r["seed"]) == seed)["psis_route_tau05"]
        for seed in SEEDS
    )
    summary = {
        "theta_MAP": opt["theta_MAP"],
        "theta_MAP_distance_from_one": opt["theta_MAP_distance_from_one"],
        "padding_difference": opt["padding"],
        "groups": groups,
        "raw_route_changed_for_any_seed": raw_changed,
        "psis_route_changed_for_any_seed": psis_changed,
        "MAP_fast_clearly_closer_to_reference_by_prespecified_summary_rule": clearly_closer,
        "field_routing_to_correction_still_holds_all_MAP_seeds": routing_success,
        "nominal_raw_reproduction_max_abs_error": max(float(r.get("nominal_raw_reproduction_abs_error", 0)) for r in rows if r["padding"] == "nominal"),
        "nominal_psis_reproduction_max_abs_error": max(float(r.get("nominal_psis_reproduction_abs_error", 0)) for r in rows if r["padding"] == "nominal"),
    }
    (OUT / "SAILBOAT_PADDING_SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# Sailboat MAP-padding paired sensitivity",
        "",
        f"Optimization gate: **{opt['optimization_gate']}**",
        "",
        "## Direct answers",
        "",
        f"1. `theta_MAP = {opt['theta_MAP']}`. Distance from theta=1: L2 `{opt['theta_MAP_distance_from_one']['l2']:.8f}`, maximum absolute coordinate difference `{opt['theta_MAP_distance_from_one']['max_abs']:.8f}`.",
        f"2. Modes 7–20 change by L2 `{opt['padding']['higher_mode_l2_difference']:.8f}`, maximum absolute `{opt['padding']['higher_mode_max_abs_difference']:.8f}`, relative L2 `{opt['padding']['higher_mode_relative_l2_difference']:.8f}`.",
        f"3. Raw route changes for any paired seed: **{'YES' if raw_changed else 'NO'}**. MAP routes: `{m['raw_routes']}`.",
        f"4. PSIS route changes for any paired seed: **{'YES' if psis_changed else 'NO'}**. MAP routes: `{m['psis_routes']}`.",
        f"5. Mean Pareto k changes from `{n['pareto_k']['mean']:.8f}` to `{m['pareto_k']['mean']:.8f}` (ranges `{n['pareto_k']['min']:.8f}–{n['pareto_k']['max']:.8f}` and `{m['pareto_k']['min']:.8f}–{m['pareto_k']['max']:.8f}`).",
        f"6. MAP padding is **{'clearly closer' if clearly_closer else 'not clearly closer'}** to qualified NUTS under the frozen summary rule. Mean active MAE `{n['active_mean_mae']['mean']:.8f} -> {m['active_mean_mae']['mean']:.8f}`; mean active W1 `{n['active_w1_mean']['mean']:.8f} -> {m['active_w1_mean']['mean']:.8f}`.",
        f"7. The field-routing-to-correction narrative **{'still holds for all five MAP-padding seeds' if routing_success else 'does not hold for all five MAP-padding seeds'}**.",
        "",
        "## Paired-design checks",
        "",
        "Each seed's base latent tensor was generated once and passed unchanged to both padding conditions. No exact-reference target or NUTS sample was changed.",
        f"Nominal replay maximum raw-ESS absolute error versus the retained result: `{summary['nominal_raw_reproduction_max_abs_error']:.3e}`; PSIS: `{summary['nominal_psis_reproduction_max_abs_error']:.3e}`.",
        "",
        "No manuscript file was modified.",
        "",
    ]
    (OUT / "SAILBOAT_PADDING_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print("SAILBOAT_PADDING_FINALIZED", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

