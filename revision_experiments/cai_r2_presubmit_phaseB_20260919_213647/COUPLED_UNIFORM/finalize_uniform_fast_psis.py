import csv
import json
from pathlib import Path

import arviz as az
import numpy as np

ROOT = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
OUT = ROOT / "revision_experiments/cai_r2_presubmit_phaseB_20260919_213647/COUPLED_UNIFORM"
SEEDS = [42, 43, 44, 45, 46]
TRAINING_SEEDS = [42, 202, 303]


def main():
    assert az.__version__ == "0.23.4"
    base = list(csv.DictReader((OUT / "COUPLED_UNIFORM_FAST_ROUTING_BASE.csv").open(encoding="utf-8")))
    keyed = {(int(r["flow_training_seed"]), r["scenario"], int(r["latent_seed"])): r for r in base}
    with np.load(OUT / "COUPLED_UNIFORM_FAST_ROUTING_RAW.npz", allow_pickle=False) as a:
        logw = a["log_weights"].astype(float)
    assert logw.shape == (3, 2, 5, 2000)
    rows = []
    for fi, training_seed in enumerate(TRAINING_SEEDS):
        for sci, scenario in enumerate(["field", "damage"]):
            for si, latent_seed in enumerate(SEEDS):
                smoothed, k = az.psislw(logw[fi, sci, si], reff=1.0, normalize=True)
                w = np.exp(smoothed)
                psis_ess = float(1.0 / np.sum(w ** 2) / len(w))
                row = dict(keyed[(training_seed, scenario, latent_seed)])
                row["psis_ess_ratio"] = psis_ess
                row["pareto_k"] = float(k)
                row["raw_route_tau05"] = "FAST" if float(row["raw_ess_ratio"]) >= 0.5 else "CORRECTION"
                row["psis_route_tau05"] = "FAST" if psis_ess >= 0.5 else "CORRECTION"
                rows.append(row)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / "COUPLED_UNIFORM_FAST_ROUTING.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for training_seed in TRAINING_SEEDS:
        role = "primary" if training_seed == 42 else "secondary"
        summary[str(training_seed)] = {"role": role}
        for scenario in ["field", "damage"]:
            rs = [r for r in rows if int(r["flow_training_seed"]) == training_seed and r["scenario"] == scenario]
            entry = {
                "fast_all_mean_across_latent_seeds": [float(np.mean([float(r[f"fast_all_mean_e{j}"]) for r in rs])) for j in range(1, 4)],
                "fast_all_std_across_latent_seeds": [float(np.mean([float(r[f"fast_all_std_e{j}"]) for r in rs])) for j in range(1, 4)],
                "in_support_fraction_mean": float(np.mean([float(r["in_support_fraction"]) for r in rs])),
                "raw_ess_ratio_mean": float(np.mean([float(r["raw_ess_ratio"]) for r in rs])),
                "raw_ess_ratio_range": [float(np.min([float(r["raw_ess_ratio"]) for r in rs])), float(np.max([float(r["raw_ess_ratio"]) for r in rs]))],
                "psis_ess_ratio_mean": float(np.mean([float(r["psis_ess_ratio"]) for r in rs])),
                "psis_ess_ratio_range": [float(np.min([float(r["psis_ess_ratio"]) for r in rs])), float(np.max([float(r["psis_ess_ratio"]) for r in rs]))],
                "pareto_k_mean": float(np.mean([float(r["pareto_k"]) for r in rs])),
                "pareto_k_range": [float(np.min([float(r["pareto_k"]) for r in rs])), float(np.max([float(r["pareto_k"]) for r in rs]))],
                "raw_routes": {name: sum(r["raw_route_tau05"] == name for r in rs) for name in ["FAST", "CORRECTION"]},
                "psis_routes": {name: sum(r["psis_route_tau05"] == name for r in rs) for name in ["FAST", "CORRECTION"]},
                "above_4_proposals_total": sum(int(r["above_4_count"]) for r in rs),
                "below_0_5_proposals_total": sum(int(r["below_0_5_count"]) for r in rs),
                "mean_inference_time_ms": float(np.mean([float(r["inference_time_ms"]) for r in rs])),
            }
            if scenario == "field":
                entry.update(
                    gap_against_unqualified_candidate=True,
                    candidate_reference_mean_mae_mean=float(np.mean([float(r["candidate_reference_mean_mae"]) for r in rs])),
                    candidate_reference_w1_mean=float(np.mean([float(r["candidate_reference_w1_mean"]) for r in rs])),
                    candidate_reference_w1_max_mean=float(np.mean([float(r["candidate_reference_w1_max"]) for r in rs])),
                    candidate_reference_width_ratio_mean=float(np.mean([float(r["candidate_reference_width_ratio_mean"]) for r in rs])),
                )
            summary[str(training_seed)][scenario] = entry
    (OUT / "COUPLED_UNIFORM_FAST_ROUTING_SUMMARY.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("COUPLED_UNIFORM_FAST_PSIS_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

