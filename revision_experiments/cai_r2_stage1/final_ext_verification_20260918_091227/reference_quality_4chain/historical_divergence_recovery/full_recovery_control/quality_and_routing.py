"""Pinned ArviZ quality gate and conditional fixed-sample routing reevaluation."""
import importlib.util
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
from scipy.stats import wasserstein_distance

CONTROL = Path(__file__).resolve().parent
HOUT = CONTROL.parent
OUT = HOUT.parent
HERE = OUT.parent
spec = importlib.util.spec_from_file_location("frozen_quality_helpers", OUT / "control/diagnose_four_chains.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)  # Definitions only; original output-producing main is never called.
Q = [.025, .25, .5, .75, .975]
QL = ["q025", "q25", "q50", "q75", "q975"]
CALC = CONTROL / "RECOVERED_QUALITY_CALCULATION.json"


def inputs():
    snapshot = json.loads((CONTROL / "FROZEN_BEFORE_FULL_RECOVERY.json").read_text(encoding="utf-8"))
    history_path = HOUT / "HISTORICAL_DIAGNOSTICS_RECOVERY_INTEGRITY.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    assert history["summary"]["all_21_exact_reproduced"]
    assert history["summary"]["remaining_chain_calls"] == 20
    assert history["summary"]["pilot_chain_reruns"] == 0
    manifest = json.loads((OUT / "control/TARGET_MANIFEST.json").read_text(encoding="utf-8"))
    for record in history["chain_results"]:
        assert record["classification"] == "EXACT_HISTORICAL_CHAIN_REPRODUCTION"
        assert record["comparison"]["np_array_equal"] or record["comparison"]["max_abs_difference"] == 0
        assert D.sha(record["artifact_path"]) == record["artifact_sha256"]
        assert record["replay_diagnostics_attributed_to_historical_chain"]
        assert len(record["raw_mc_diagnostics"]["divergences"]["chain 0"]) == record["recovered_historical_divergence_count"]
    return snapshot, history, manifest, history_path


def calculate():
    assert az.__version__ == "0.23.4"
    assert not CALC.exists(), "Preserve an existing calculation."
    snapshot, history, manifest, history_path = inputs()
    before = D.verify_preservation(snapshot)
    lookup = {(r["manifest"]["target"], r["manifest"]["seed"]): r for r in history["chain_results"]}
    rows, results = [], []
    for target in manifest["targets"]:
        name = target["target"]
        run_path = OUT / f"REFERENCE4_{name}_RUN.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        assert run["status"] == "COMPLETED" and D.sha(run["artifact_path"]) == run["artifact_sha256"]
        counts = [lookup[name, seed]["recovered_historical_divergence_count"] for seed in [701, 702, 703]]
        with np.load(target["original_reference_path"], allow_pickle=False) as old, np.load(run["artifact_path"], allow_pickle=False) as saved:
            chains = saved["chains"].copy()
            assert chains.shape == (4, 600, 6) and np.array_equal(chains[:3], old["chains"])
            assert np.array_equal(chains[3], saved["theta_samples"]) and saved["seed"].item() == 704
            fourth_indices = saved["divergences"].tolist()
            raw704 = json.loads(saved["sampler_diagnostics_json"].item())
            assert fourth_indices == raw704["divergences"]["chain 0"] == run["fourth_chain_divergence_indices"]
            assert len(fourth_indices) == run["fourth_chain_divergence_count"]
            old_mean, old_quantiles = old["ref_mean"].astype(np.float64), old["ref_quantiles"].astype(np.float64)
        assert np.isfinite(chains).all()
        diagnostic = D.arviz_diagnostics(chains)
        pooled = chains.reshape(-1, 6).astype(np.float64)
        mean, quantiles = pooled.mean(0), np.quantile(pooled, Q, axis=0)
        total = sum(counts) + len(fourth_indices)
        gates = dict(chains="PASS" if chains.shape[0] >= 4 else "FAIL",
            rhat="PASS" if diagnostic["max_rhat"] is not None and diagnostic["max_rhat"] <= 1.01 else "FAIL",
            bulk_ess="PASS" if diagnostic["min_bulk_ess"] is not None and diagnostic["min_bulk_ess"] >= 400 else "FAIL",
            tail_ess="PASS" if diagnostic["min_tail_ess"] is not None and diagnostic["min_tail_ess"] >= 400 else "FAIL",
            divergences="PASS" if total == 0 else "FAIL")
        overall = "PASS" if all(v == "PASS" for v in gates.values()) else "FAIL"
        result = dict(target=name, target_manifest=target, diagnostics_4chain=diagnostic,
            historical_divergence_counts=counts, historical_3chain_divergences=sum(counts),
            seed704_divergence_count=len(fourth_indices), all_4chain_divergences=total,
            historical_chain_records=[lookup[name, seed] for seed in [701, 702, 703]],
            reference4_path=run["artifact_path"], reference4_sha256=run["artifact_sha256"],
            gates=gates, overall_quality_gate=overall,
            original_3chain_mean=old_mean.tolist(), reference_4chain_mean=mean.tolist(),
            mean_change=(mean - old_mean).tolist(), mean_mae_change_all6=float(np.abs(mean-old_mean).mean()),
            mean_mae_change_active_e1_e3=float(np.abs(mean[:3]-old_mean[:3]).mean()),
            original_3chain_quantiles=old_quantiles.tolist(), reference_4chain_quantiles=quantiles.tolist(),
            quantile_change=(quantiles-old_quantiles).tolist(), quantile_max_absolute_change=float(np.abs(quantiles-old_quantiles).max()))
        row = dict(target=name, chain_count=4, draws_per_chain=600,
            max_rank_normalized_split_rhat=diagnostic["max_rhat"], min_bulk_ess=diagnostic["min_bulk_ess"], min_tail_ess=diagnostic["min_tail_ess"],
            seed701_divergence_count=counts[0], seed702_divergence_count=counts[1], seed703_divergence_count=counts[2],
            historical_3chain_divergences=sum(counts), seed704_divergences=len(fourth_indices), all_4chain_divergences=total,
            all_three_historical_chains_exact_reproduced=True, overall_quality_gate=overall,
            mean_change_mae_all6=result["mean_mae_change_all6"], mean_change_mae_active_e1_e3=result["mean_mae_change_active_e1_e3"],
            quantile_max_absolute_change=result["quantile_max_absolute_change"],
            reference4_path=run["artifact_path"], reference4_sha256=run["artifact_sha256"])
        row.update({f"{key}_gate": value for key, value in gates.items()})
        for j in range(6):
            for key in ["rank_normalized_split_rhat", "bulk_ess", "tail_ess"]:
                row[f"e{j+1}_{key}"] = diagnostic[key][j]
            row[f"e{j+1}_mean_3chain"] = float(old_mean[j])
            row[f"e{j+1}_mean_4chain"] = float(mean[j])
            row[f"e{j+1}_mean_change"] = float(mean[j]-old_mean[j])
            for qi, label in enumerate(QL):
                for suffix, array in [("3chain", old_quantiles), ("4chain", quantiles), ("change", quantiles-old_quantiles)]:
                    row[f"e{j+1}_{label}_{suffix}"] = float(array[qi, j])
        rows.append(row)
        results.append(result)
        print("COMPLETE_FOUR_CHAIN_GATE", name, overall, "historical_div", sum(counts), "all_div", total, flush=True)
    summary = dict(exact_reproduced_historical_chains=21, historical_divergences_total=history["summary"]["historical_divergences_total"],
        all_28_chains_divergences_total=sum(r["all_4chain_divergences"] for r in results),
        quality_gate_passed_targets=sum(r["overall_quality_gate"] == "PASS" for r in results),
        all_7_complete_quality_gates_pass=all(r["overall_quality_gate"] == "PASS" for r in results),
        max_rank_normalized_split_rhat=max(r["diagnostics_4chain"]["max_rhat"] for r in results),
        min_bulk_ess=min(r["diagnostics_4chain"]["min_bulk_ess"] for r in results),
        min_tail_ess=min(r["diagnostics_4chain"]["min_tail_ess"] for r in results))
    calculation = dict(created_utc=D.utc_now(), summary=summary, target_rows=rows, targets=results,
        arviz_version=az.__version__, methods=dict(rhat="rank", ess_bulk="bulk", ess_tail="tail", relative=False),
        quality_gate=dict(chains_min=4, max_rank_rhat=1.01, min_bulk_ess=400, min_tail_ess=400, divergences=0),
        historical_integrity_path=str(history_path), historical_integrity_sha256=D.sha(history_path),
        preservation_before=before, preservation_after=D.verify_preservation(snapshot))
    D.write_new_json(CALC, calculation)
    if summary["all_7_complete_quality_gates_pass"]:
        routing(calculation)
    else:
        finalize()


def routing(calculation):
    import csv
    assert calculation["summary"]["all_7_complete_quality_gates_pass"]
    table_path = HERE / "ROUTING_VS_GAP_TABLE.csv"
    with table_path.open(encoding="utf-8-sig", newline="") as stream:
        original_routes = {row["target"]: row for row in csv.DictReader(stream)}
    rows, reference_rows = [], []
    for result in calculation["targets"]:
        name = result["target"]
        original_route = original_routes[name]
        with np.load(result["target_manifest"]["original_reference_path"], allow_pickle=False) as old, np.load(result["reference4_path"], allow_pickle=False) as saved:
            fast = old["fast"].copy()
            old_mean, fast_mean = old["ref_mean"].astype(np.float64), old["fast_mean"].astype(np.float64)
            old_active_mae = float(np.mean(np.abs(old["fast_mean"][:3]-old["ref_mean"][:3])))
            old_all6_mae = float(np.mean(np.abs(old["fast_mean"]-old["ref_mean"])))
            assert np.isclose(old_active_mae, float(original_route["mean_mae_active"]), atol=1e-14, rtol=0)
            old_active_mae = float(original_route["mean_mae_active"])
            original_pooled = old["chains"].reshape(-1, 6)
            pooled = saved["chains"].reshape(-1, 6)
        mean = np.asarray(result["reference_4chain_mean"])
        quantiles = np.asarray(result["reference_4chain_quantiles"])
        old_quantiles = np.asarray(result["original_3chain_quantiles"])
        fast_q = np.quantile(fast, Q, axis=0)
        w1 = np.asarray([wasserstein_distance(fast[:, j], pooled[:, j]) for j in range(6)])
        old_w1 = np.asarray([wasserstein_distance(fast[:, j], original_pooled[:, j]) for j in range(6)])
        ratio = (fast_q[-1]-fast_q[0])/(quantiles[-1]-quantiles[0])
        errors, old_errors = np.abs(fast_mean-mean), np.abs(fast_mean-old_mean)
        row = dict(target=name, raw_mean=original_route["raw_mean"], psis_mean=original_route["psis_mean"],
            k_mean=original_route["k_mean"], route_raw=original_route["route_raw"], route_psis=original_route["route_psis"],
            fast_draws=int(fast.shape[0]), fast_samples_reused=True, fast_samples_filtered=0,
            mean_mae_active_3chain=old_active_mae, mean_mae_active_4chain=float(errors[:3].mean()),
            mean_mae_all6_3chain=old_all6_mae, mean_mae_all6_4chain=float(errors.mean()),
            active_mae_change=float(errors[:3].mean()-old_active_mae),
            w1_active_mean_3chain=float(old_w1[:3].mean()), w1_active_mean_4chain=float(w1[:3].mean()),
            q95_width_ratio_active_mean_4chain=float(ratio[:3].mean()),
            negative_proposal_draws=int((fast < 0).any(axis=1).sum()),
            original_reference_sha256=result["target_manifest"]["original_reference_sha256"], reference4_sha256=result["reference4_sha256"])
        for j in range(6):
            row.update({f"e{j+1}_fast_mean": float(fast_mean[j]), f"e{j+1}_reference_mean_3chain": float(old_mean[j]),
                f"e{j+1}_reference_mean_4chain": float(mean[j]), f"e{j+1}_wasserstein_3chain": float(old_w1[j]),
                f"e{j+1}_wasserstein_4chain": float(w1[j]), f"e{j+1}_q95_width_ratio": float(ratio[j])})
            rr = dict(target=name, parameter=f"e{j+1}", mean_3chain=float(old_mean[j]), mean_4chain=float(mean[j]), mean_change=float(mean[j]-old_mean[j]))
            for qi, label in enumerate(QL):
                row[f"e{j+1}_fast_{label}"] = float(fast_q[qi, j])
                row[f"e{j+1}_reference_3chain_{label}"] = float(old_quantiles[qi, j])
                row[f"e{j+1}_reference_4chain_{label}"] = float(quantiles[qi, j])
                rr[f"{label}_3chain"], rr[f"{label}_4chain"], rr[f"{label}_change"] = float(old_quantiles[qi, j]), float(quantiles[qi, j]), float(quantiles[qi, j]-old_quantiles[qi, j])
            reference_rows.append(rr)
        rows.append(row)
    fast_cases = [r for r in rows if r["target"].startswith("ablation_obs")]
    field = next(r for r in rows if r["target"] == "sailboat_field_frozen")
    boundary = next(r for r in rows if r["target"] == "sailboat_boundary_e1_075")
    fast_max3 = max(r["mean_mae_active_3chain"] for r in fast_cases)
    fast_max4 = max(r["mean_mae_active_4chain"] for r in fast_cases)
    field_scale_fast_misses = [r["target"] for r in fast_cases if r["route_psis"] == "FAST" and r["mean_mae_active_4chain"] >= field["mean_mae_active_4chain"]]
    unchanged = all(r["route_raw"] == r["route_psis"] == "FAST" for r in fast_cases) and field["route_psis"] == boundary["route_psis"] == "CORRECTION" and not field_scale_fast_misses and field["mean_mae_active_4chain"] > max(fast_max4, boundary["mean_mae_active_4chain"])
    summary = dict(five_FAST_cases_max_active_MAE_3chain=fast_max3, five_FAST_cases_max_active_MAE_4chain=fast_max4,
        field_active_gap_3chain=field["mean_mae_active_3chain"], field_active_gap_4chain=field["mean_mae_active_4chain"],
        boundary_active_gap_3chain=boundary["mean_mae_active_3chain"], boundary_active_gap_4chain=boundary["mean_mae_active_4chain"],
        high_ESS_FAST_case_with_field_scale_large_gap=field_scale_fast_misses, any_high_ESS_large_gap_missed_escalation=bool(field_scale_fast_misses),
        r2_1_main_qualitative_conclusion_changed=not unchanged,
        comparison_scope="Observed seven cases; field-scale gap comparison only; no added routing or gap threshold", routing_diagnostics_unchanged=True)
    csv_path, audit_path = OUT / "ROUTING_VS_GAP_4CHAIN.csv", OUT / "ROUTING_VS_GAP_4CHAIN_AUDIT.md"
    D.write_new_csv(csv_path, rows)
    D.write_new_csv(OUT / "REFERENCE_4CHAIN_SUMMARY.csv", reference_rows)
    text = [f"R2.1 main qualitative conclusion changed: {not unchanged}.",
        f"Five FAST cases maximum active mean MAE: {fast_max3:.12g} -> {fast_max4:.12g}.",
        f"Field active gap: {field['mean_mae_active_3chain']:.12g} -> {field['mean_mae_active_4chain']:.12g}.",
        f"Boundary active gap: {boundary['mean_mae_active_3chain']:.12g} -> {boundary['mean_mae_active_4chain']:.12g}.",
        f"Observed high-ESS FAST cases with a field-scale large gap and missed escalation: {field_scale_fast_misses}.", "",
        "The five ablation cases retain FAST routing; field and boundary retain CORRECTION routing. The seven-case comparison uses each existing frozen 40,000-draw fast array; no flow sampling or training occurs. Raw ESS, PSIS ESS, k and both routing decisions are copied unchanged from ROUTING_VS_GAP_TABLE.csv.",
        "The field-scale comparison above is descriptive for these seven observations and does not introduce a new routing or large-gap threshold. It does not establish universal coverage of future observations.",
        "All six parameter means, 1D Wasserstein distances (SciPy), 2.5/25/50/75/97.5 percent quantiles and fast/reference 95 percent width ratios are included. Active e1-e3 MAE is primary; all6 is supplementary. Existing negative fast draws remain unfiltered.",
        "Four-chain reference means accumulate the 2,400 retained draws in float64; linear quantiles use NumPy. Three-chain comparisons use the original saved means/quantiles. All seven references use the same four-chain policy after the complete quality gate passed; no favorable-reference selection occurs.",
        "Historical references, seed704 chains, routing diagnostics/thresholds, checkpoints, submitted baseline, manuscript/response and Git tracked files remain unchanged.", ""]
    D.write_new_bytes(audit_path, "\n".join(text).encode("utf-8"))
    D.write_new_json(CONTROL / "ROUTING_4CHAIN_EVALUATION.json", dict(summary=summary, rows=rows,
        original_routing_table_path=str(table_path), original_routing_table_sha256=D.sha(table_path),
        outputs={p.name: dict(path=str(p), sha256=D.sha(p)) for p in [csv_path, audit_path, OUT / "REFERENCE_4CHAIN_SUMMARY.csv"]}))
    print("R2_1_FIXED_SAMPLE_EVALUATION", json.dumps(summary), flush=True)


def finalize():
    calculation = json.loads(CALC.read_text(encoding="utf-8"))
    snapshot, history, _, _ = inputs()
    passed = calculation["summary"]["all_7_complete_quality_gates_pass"]
    routing_result = json.loads((CONTROL / "ROUTING_4CHAIN_EVALUATION.json").read_text(encoding="utf-8")) if passed else None
    ablation_result = json.loads((CONTROL / "ABLATION_4CHAIN_EVALUATION.json").read_text(encoding="utf-8")) if passed else None
    summary = calculation["summary"].copy()
    summary.update(r2_1_main_conclusion_changed=routing_result["summary"]["r2_1_main_qualitative_conclusion_changed"] if passed else "NOT_ASSESSED_QUALITY_GATE_FAILED",
        r2_3_main_conclusion_changed=ablation_result["summary"]["r2_3_main_qualitative_conclusion_changed"] if passed else "NOT_ASSESSED_QUALITY_GATE_FAILED",
        mcmc_stopped=True, flow_training_calls=0, awaiting_review=True)
    paths = [OUT / "REFERENCE_QUALITY_SUMMARY_RECOVERED.csv", OUT / "REFERENCE_QUALITY_AUDIT_RECOVERED.md", OUT / "REFERENCE_QUALITY_INTEGRITY_RECOVERED.json"]
    assert all(not p.exists() for p in paths)
    for row in calculation["target_rows"]:
        row.update(r2_1_main_conclusion_changed=summary["r2_1_main_conclusion_changed"], r2_3_main_conclusion_changed=summary["r2_3_main_conclusion_changed"])
    D.write_new_csv(paths[0], calculation["target_rows"])
    preservation = D.verify_preservation(snapshot)
    text = ["1. Historical exact reproduction: 21/21 PASS; frozen pilot included without rerun.",
        f"2. Historical 21-chain divergences: {summary['historical_divergences_total']}.",
        f"3. All 28-chain divergences: {summary['all_28_chains_divergences_total']}.",
        f"4. Complete four-chain quality gate: {summary['quality_gate_passed_targets']}/7 PASS; all seven={passed}.",
        f"5. Maximum rank-normalized split-Rhat: {summary['max_rank_normalized_split_rhat']:.15g}.",
        f"6. Minimum bulk ESS: {summary['min_bulk_ess']:.15g}.",
        f"7. Minimum tail ESS: {summary['min_tail_ess']:.15g}.",
        f"8. R2.1 main conclusion changed: {summary['r2_1_main_conclusion_changed']}.",
        f"9. R2.3 main conclusion changed: {summary['r2_3_main_conclusion_changed']}.", "",
        "Historical divergence evidence is recovered only from exact sample-by-sample replay under the original scientific run_chain function. Correct chain 0 keys recover divergence indices/counts and acceptance rates; original raw diagnostics are independently preserved in each replay NPZ/JSON.",
        "The old z-key fallback zeros were missing-key artifacts and remain unchanged as historical logs. The original 0/7 FAIL audit remains frozen; this recovered audit is a separate evidence update. No old reference, log or audit was overwritten.",
        "Every target is formed from its original (3,600,6) chains plus its already existing seed704 chain. No chain is added or resampled in this quality reevaluation. ArviZ 0.23.4 in the existing psis_env computes method=rank Rhat, method=bulk ESS and method=tail ESS with relative=False for each parameter.",
        "The unchanged gate requires at least 4 chains, maximum rank-Rhat <=1.01, minimum bulk ESS >=400, minimum tail ESS >=400 and all-four-chain divergences=0. Each component PASS/FAIL and all 42 diagnostic values are saved in the recovered summary.",
        "Per-target three-to-four mean displacements and five quantile changes are included; new numeric values are not constrained to match old values.",
        ("All seven complete gates passed, so R2.1 reuses original fast arrays and routing diagnostics; R2.3 reevaluates the frozen 24 scientific checkpoints with the five fixed observations, seeds42..46 and Nfast=2000. No training occurs. All seven evaluation references consistently use four chains. Final manuscript numeric adoption requires a separate authorized manuscript step; this task does not edit it." if passed else "At least one complete gate failed; no R2.1 or R2.3 reevaluation was executed. No tuning, chain deletion or additional chain was performed; stopped for review."),
        f"Integrity: {preservation['checked_files']} protected asset SHA256 values, frozen pilot, original 367 assets, seven original references, seven seed704 artifacts, 24 scientific checkpoints, submitted baseline and Git tracked-file hashes unchanged.",
        "No new seed, fifth chain, flow training, FE, damage grid, prior/noise or routing threshold adjustment, manuscript/response change or Git mutation occurred. All authorized work has stopped for review.", ""]
    if passed:
        text += ["R2.1 comparison:", json.dumps(routing_result["summary"], ensure_ascii=False, indent=2), "", "R2.3 comparison:", json.dumps(ablation_result["summary"], ensure_ascii=False, indent=2), ""]
    text += ["Per-target complete gate:", "",
        "| Target | max rank-Rhat | min bulk ESS | min tail ESS | historical divergences | seed704 divergences | complete gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for target in calculation["targets"]:
        dg = target["diagnostics_4chain"]
        text.append(f"| {target['target']} | {dg['max_rhat']:.12g} | {dg['min_bulk_ess']:.12g} | {dg['min_tail_ess']:.12g} | {target['historical_3chain_divergences']} | {target['seed704_divergence_count']} | {target['overall_quality_gate']} |")
    text += ["", "Recovered historical divergence locations:", ""]
    for target in calculation["targets"]:
        for record in target["historical_chain_records"]:
            count = record["recovered_historical_divergence_count"]
            if count:
                text.append(f"{target['target']}, historical chain index={record['manifest']['historical_chain_index']}, seed={record['manifest']['seed']}: {count} divergence(s), original Pyro retained-draw indices={record['replay_divergence_indices']}; exact reproduction={record['comparison']['np_array_equal']} and maximum sample difference={record['comparison']['max_abs_difference']}.")
    text.append("")
    D.write_new_bytes(paths[1], "\n".join(text).encode("utf-8"))
    outputs = {p.name: dict(path=str(p), sha256=D.sha(p)) for p in paths[:2]}
    if passed:
        outputs.update(routing_result["outputs"])
        outputs.update(ablation_result["outputs"])
    integrity = dict(schema_version=1, created_utc=D.utc_now(), summary=summary, targets=calculation["targets"],
        quality_gate=calculation["quality_gate"], arviz_version=calculation["arviz_version"], methods=calculation["methods"],
        historical_recovery_integrity=dict(path=calculation["historical_integrity_path"], sha256=calculation["historical_integrity_sha256"]),
        protected_files=snapshot["files"], git_tracked_file_hashes=snapshot["git_tracked_file_hashes"], frozen_pilot_files=snapshot["frozen_pilot_files"],
        preservation_after=preservation, routing_reevaluation=routing_result, ablation_reevaluation=ablation_result,
        outputs=outputs, execution_boundaries=dict(remaining_historical_chain_calls=20, pilot_reruns=0, additional_reference_chains=0,
            flow_training_calls=0, new_training_seeds=0, new_fe_runs=0, damage_grid_recomputations=0,
            paper_response_modifications=0, old_audit_overwrites=0, git_mutations=0))
    D.write_new_json(paths[2], integrity)
    print("RECOVERED_REFERENCE_QUALITY_FINAL", json.dumps(summary), flush=True)


if __name__ == "__main__":
    finalize() if "--finalize" in sys.argv else calculate()
