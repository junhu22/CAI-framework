"""Formal six-case panel from existing samples; no Torch/Pyro or routing calls."""
import json
import sys

import numpy as np
import scipy
from scipy.stats import wasserstein_distance

import common as C


def main():
    result_path = C.CONTROL / "QUALIFIED_ROUTING_EVALUATION.json"
    names = ["ROUTING_VS_GAP_4CHAIN_QUALIFIED.csv", "ROUTING_VS_GAP_4CHAIN_QUALIFIED_AUDIT.md", "BOUNDARY_REFERENCE_EXCLUSION_NOTE.md"]
    assert not result_path.exists() and all(not (C.OUT / name).exists() for name in names)
    snapshot = C.read_json(C.FROZEN_PATH)
    before = C.verify(snapshot)
    quality, targets = C.frozen_quality()
    table_path = C.HERE / "ROUTING_VS_GAP_TABLE.csv"
    routes = {row["target"]: row for row in C.read_csv(table_path)}
    rows = []
    for name in C.QUALIFIED:
        target = targets[name]
        old, pooled, mean, quantiles = C.qualified_reference(name, target)
        route = routes[name]
        fast = old["fast"]
        assert fast.shape == (40000, 6) and np.isfinite(fast).all()
        # Preserve the original fast posterior mean accumulation and all fast draws.
        fast_mean = old["fast_mean"].astype(np.float64)
        fast_q = np.quantile(fast, C.Q, axis=0)
        assert np.array_equal(fast_q, old["fast_quantiles"])
        reference3_mean, reference3_q = old["ref_mean"].astype(np.float64), old["ref_quantiles"].astype(np.float64)
        old_active = float(route["mean_mae_active"])
        assert np.isclose(old_active, float(np.abs(old["fast_mean"][:3]-old["ref_mean"][:3]).mean()), atol=1e-14, rtol=0)
        old_all6 = float(np.abs(old["fast_mean"]-old["ref_mean"]).mean())
        error = np.abs(fast_mean-mean)
        w1 = np.asarray([wasserstein_distance(fast[:, j], pooled[:, j]) for j in range(6)])
        ratio = (fast_q[-1]-fast_q[0])/(quantiles[-1]-quantiles[0])
        row = dict(target=name, reference_quality_gate="PASS", reference_chain_count=4, reference_draws=2400,
            raw_ESS_IS_over_N=route["raw_mean"], PSIS_ESS_over_N=route["psis_mean"], Pareto_k=route["k_mean"],
            raw_route=route["route_raw"], PSIS_route=route["route_psis"], routing_threshold_tau=.5,
            routing_results_read_unchanged=True, routing_threshold_recomputed=False,
            MAE_reference_active_e1_e3=float(error[:3].mean()), MAE_reference_all6=float(error.mean()),
            W1_active_e1_e3_mean=float(w1[:3].mean()), W1_active_e1_e3_max=float(w1[:3].max()),
            width95_fast_over_reference_active_mean=float(ratio[:3].mean()),
            width95_fast_over_reference_active_min=float(ratio[:3].min()), width95_fast_over_reference_active_max=float(ratio[:3].max()),
            active_MAE_3chain=old_active, active_MAE_4chain=float(error[:3].mean()),
            active_MAE_change=float(error[:3].mean()-old_active), active_MAE_absolute_change=float(abs(error[:3].mean()-old_active)),
            all6_MAE_3chain=old_all6, all6_MAE_change=float(error.mean()-old_all6),
            fast_draws=40000, negative_fast_proposal_draws=int((fast < 0).any(axis=1).sum()), fast_draws_filtered=0,
            original_reference_sha256=target["target_manifest"]["original_reference_sha256"], reference4_sha256=target["reference4_sha256"])
        for j in range(6):
            row.update({f"e{j+1}_fast_mean": float(fast_mean[j]), f"e{j+1}_reference3_mean": float(reference3_mean[j]),
                f"e{j+1}_reference4_mean": float(mean[j]), f"e{j+1}_Wasserstein_1D": float(w1[j]),
                f"e{j+1}_width95_ratio_fast_over_reference": float(ratio[j]),
                f"e{j+1}_fast_width95": float(fast_q[-1,j]-fast_q[0,j]),
                f"e{j+1}_reference4_width95": float(quantiles[-1,j]-quantiles[0,j])})
            for qi, label in enumerate(C.QL):
                row[f"e{j+1}_fast_{label}"] = float(fast_q[qi,j])
                row[f"e{j+1}_reference3_{label}"] = float(reference3_q[qi,j])
                row[f"e{j+1}_reference4_{label}"] = float(quantiles[qi,j])
        rows.append(row)
        print("QUALIFIED_ROUTING_CASE", name, row["raw_route"], row["PSIS_route"], row["MAE_reference_active_e1_e3"], row["W1_active_e1_e3_mean"], row["W1_active_e1_e3_max"], flush=True)
    fast_cases = rows[:5]
    assert all(row["raw_route"] == row["PSIS_route"] == "FAST" for row in fast_cases)
    field = rows[-1]
    assert field["target"] == "sailboat_field_frozen" and field["raw_route"] == field["PSIS_route"] == "CORRECTION"
    max_mae = max(row["MAE_reference_active_e1_e3"] for row in fast_cases)
    max_w1_mean = max(row["W1_active_e1_e3_mean"] for row in fast_cases)
    max_w1_parameter = max(row["W1_active_e1_e3_max"] for row in fast_cases)
    misses = [row["target"] for row in fast_cases if row["MAE_reference_active_e1_e3"] >= field["MAE_reference_active_e1_e3"] or row["W1_active_e1_e3_mean"] >= field["W1_active_e1_e3_mean"]]
    summary = dict(qualified_r2_1_targets=6, all_6_quality_gates_PASS=True,
        boundary_classification="EXCLUDED_BY_PRESPECIFIED_REFERENCE_QUALITY_GATE",
        five_FAST_cases_max_active_MAE=max_mae,
        five_FAST_cases_max_active_mean_Wasserstein=max_w1_mean,
        five_FAST_cases_max_active_parameter_Wasserstein=max_w1_parameter,
        five_FAST_cases_max_active_MAE_3chain=max(row["active_MAE_3chain"] for row in fast_cases),
        field_raw_route=field["raw_route"], field_PSIS_route=field["PSIS_route"],
        field_active_MAE=field["MAE_reference_active_e1_e3"], field_active_mean_Wasserstein=field["W1_active_e1_e3_mean"],
        field_active_max_Wasserstein=field["W1_active_e1_e3_max"],
        field_active_MAE_over_max_FAST=field["MAE_reference_active_e1_e3"]/max_mae,
        field_active_mean_Wasserstein_over_max_FAST=field["W1_active_e1_e3_mean"]/max_w1_mean,
        observed_high_ESS_field_scale_large_gap_misses=misses,
        any_high_ESS_field_scale_large_gap_missed_escalation=bool(misses),
        r2_1_main_qualitative_conclusion_changed=bool(misses),
        no_new_routing_or_gap_threshold=True, scope="Six qualified observations only; descriptive field-scale comparison",
        all_fast_samples_reused=True, new_flow_sample_calls=0, new_MCMC_calls=0)
    C.new_csv(C.OUT / names[0], rows)
    text = ["Formal reference panel: 6/6 complete quality gate PASS (five ablation observations and frozen field).",
        "Boundary is EXCLUDED_BY_PRESPECIFIED_REFERENCE_QUALITY_GATE and contributes no quantitative gap to this panel.",
        f"R2.1 main qualitative conclusion changed: {summary['r2_1_main_qualitative_conclusion_changed']}.", "",
        f"A. Five FAST cases: maximum active posterior-mean MAE={max_mae:.15g}; maximum case-mean active 1D Wasserstein={max_w1_mean:.15g}; maximum individual e1-e3 Wasserstein across cases={max_w1_parameter:.15g}.",
        f"Observed high-ESS FAST cases with field-scale large posterior gaps or missed escalation: {misses}.",
        f"B. Field: raw route={field['raw_route']}, PSIS route={field['PSIS_route']}; active MAE={field['MAE_reference_active_e1_e3']:.15g}; active mean/max Wasserstein={field['W1_active_e1_e3_mean']:.15g}/{field['W1_active_e1_e3_max']:.15g}.",
        f"Field gaps remain numerically much larger than the five FAST cases: active MAE ratio={summary['field_active_MAE_over_max_FAST']:.9g}, active mean Wasserstein ratio={summary['field_active_mean_Wasserstein_over_max_FAST']:.9g}. No formal statistical test is performed.", "",
        "C. Three-chain -> qualified four-chain active MAE:", "",
        "| Target | 3-chain | qualified 4-chain | signed change |",
        "| --- | ---: | ---: | ---: |"]
    for row in rows:
        text.append(f"| {row['target']} | {row['active_MAE_3chain']:.12g} | {row['active_MAE_4chain']:.12g} | {row['active_MAE_change']:.12g} |")
    text += ["", "The qualified-panel conclusion is retained when the five FAST cases show small observed gaps and the field case shows a much larger gap with CORRECTION routing. This is a descriptive comparison of six observations. It does not estimate a universal false-negative rate, a calibrated probability of error or universal threshold optimality.",
        "Raw IS-ESS/N, PSIS ESS/N, Pareto-k and both routes are read unchanged from the existing ROUTING_VS_GAP_TABLE.csv. Existing tau=0.5 is recorded, not recalculated or optimized. No routing calculation or threshold adjustment occurs.",
        "The original 40,000-draw fast array per case is reused without filtering. Original fast means are retained. Qualified reference means are float64 means of the original three chains plus the existing seed704 chain. NumPy linear quantiles use probabilities 0.025/0.25/0.50/0.75/0.975; SciPy computes 1D Wasserstein distances. The CSV includes all six parameter quantiles, distances and fast/reference 95 percent width ratios; active e1-e3 metrics are primary and all6 MAE is supplementary.",
        "The frozen original 3-chain active MAEs are preserved for the comparison. The old seven-target FAIL audit is retained; the six-target qualified subset is explicitly authorized and follows the same pre-specified gate. No target definition, chain, sampler or historical diagnostic is changed.",
        "No new MCMC, flow training/draws, seed, FE, damage grid, manuscript/response change or Git mutation was performed.", ""]
    C.new_bytes(C.OUT / names[1], "\n".join(text).encode("utf-8"))
    boundary = routes[C.BOUNDARY]
    boundary_record = next(record for record in targets[C.BOUNDARY]["historical_chain_records"] if record["manifest"]["seed"] == 701)
    note = ["Classification: EXCLUDED_BY_PRESPECIFIED_REFERENCE_QUALITY_GATE.",
        "Target: sailboat_boundary_e1_075. The pre-specified zero-divergence gate fails because historical chain index0 / seed701 has one divergence at original Pyro retained-draw index375. Its replay exactly reproduces the historical samples, so the count is attributable to the historical chain.", "",
        f"Existing reference-independent routing diagnostics remain quotable: raw IS-ESS/N={boundary['raw_mean']}; PSIS ESS/N={boundary['psis_mean']}; Pareto-k={boundary['k_mean']}; raw route={boundary['route_raw']}; PSIS route={boundary['route_psis']}; frozen tau=0.5.",
        "The original three/four-chain descriptive posterior gaps must not enter the formal qualified comparison. This note contains no posterior gap numbers. Exclusion neither establishes nor refutes conservative false-positive escalation.",
        "The target, all four chains and all historical replay/log/audit evidence are retained. No repair, target_accept change, additional chain or historical-chain replacement is performed. This auxiliary case is not added to a formal reviewer reply by this task; any later discussion is limited to a body discussion that requires threshold-borderline behavior.",
        f"Frozen quality evidence: {C.QUALITY_PATH}; SHA256={C.sha(C.QUALITY_PATH)}.",
        f"Exact historical replay evidence: {boundary_record['artifact_path']}; SHA256={boundary_record['artifact_sha256']}.", ""]
    C.new_bytes(C.OUT / names[2], "\n".join(note).encode("utf-8"))
    C.new_json(result_path, dict(created_utc=C.utc_now(), summary=summary, rows=rows,
        original_routing_source=dict(path=str(table_path), sha256=C.sha(table_path)),
        numerical_environment=dict(python=sys.executable, numpy=np.__version__, scipy=scipy.__version__),
        method=dict(quantiles="numpy.quantile method=linear", Wasserstein="scipy.stats.wasserstein_distance",
            reference_mean_accumulation_dtype="float64", fast_mean_source="Original frozen saved float32 mean"),
        preservation_before=before, preservation_after=C.verify(snapshot), controller_sha256=C.sha(__file__),
        outputs={name:dict(path=str(C.OUT / name), sha256=C.sha(C.OUT / name)) for name in names}))
    print("QUALIFIED_ROUTING_FINISHED", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
