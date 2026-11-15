"""Independent stored-result and preservation checks; no sampler/flow execution."""
import json
import re
import statistics
from pathlib import Path

import numpy as np

import common as C


def close(a, b):
    return bool(np.isclose(float(a), float(b), atol=1e-14, rtol=1e-13))


def verify_results(routing, ablation, targets):
    assert routing["summary"]["qualified_r2_1_targets"] == 6
    route_rows = routing["rows"]
    assert [row["target"] for row in route_rows] == C.QUALIFIED
    original_routes = {row["target"]: row for row in C.read_csv(C.HERE / "ROUTING_VS_GAP_TABLE.csv")}
    references = []
    for row in route_rows:
        name = row["target"]
        for new, old in [("raw_ESS_IS_over_N", "raw_mean"), ("PSIS_ESS_over_N", "psis_mean"),
                         ("Pareto_k", "k_mean"), ("raw_route", "route_raw"), ("PSIS_route", "route_psis")]:
            assert row[new] == original_routes[name][old]
        assert row["routing_threshold_tau"] == .5 and not row["routing_threshold_recomputed"]
        old, _, mean, quantiles = C.qualified_reference(name, targets[name])
        fast_mean = old["fast_mean"].astype(np.float64)
        assert close(statistics.mean(abs(fast_mean[:3]-mean[:3])), row["MAE_reference_active_e1_e3"])
        assert close(statistics.mean(abs(fast_mean-mean)), row["MAE_reference_all6"])
        w1 = [row[f"e{j+1}_Wasserstein_1D"] for j in range(6)]
        assert all(np.isfinite(w1)) and min(w1) >= 0
        assert close(statistics.mean(w1[:3]), row["W1_active_e1_e3_mean"])
        assert max(w1[:3]) == row["W1_active_e1_e3_max"]
        widths = []
        for j in range(6):
            assert row[f"e{j+1}_reference4_mean"] == mean[j]
            for qi, label in enumerate(C.QL):
                assert row[f"e{j+1}_reference4_{label}"] == quantiles[qi,j]
                assert row[f"e{j+1}_fast_{label}"] == old["fast_quantiles"][qi,j]
            ratio = (row[f"e{j+1}_fast_q975"]-row[f"e{j+1}_fast_q025"])/(quantiles[-1,j]-quantiles[0,j])
            assert close(ratio, row[f"e{j+1}_width95_ratio_fast_over_reference"])
            widths.append(ratio)
        assert close(statistics.mean(widths[:3]), row["width95_fast_over_reference_active_mean"])
        assert row["fast_draws_filtered"] == 0 and row["fast_draws"] == 40000
        references.append(dict(target=name, qualified=True, reference_path=targets[name]["reference4_path"],
            reference_sha256=targets[name]["reference4_sha256"], original_reference_sha256=targets[name]["target_manifest"]["original_reference_sha256"],
            mean_float64=mean.tolist(), quantiles_float64=quantiles.tolist(), quality_gates=targets[name]["gates"],
            diagnostics=targets[name]["diagnostics_4chain"], all_four_chain_divergences=targets[name]["all_4chain_divergences"]))
    source_path = C.HERE / "ablation_multiseed/FULL_ABLATION_INTEGRITY_RECOVERED.json"
    source = C.read_json(source_path)
    original = {(row["config"],int(row["training_seed"])):row for row in source["scientific_results"]}
    original_checks = {(row["config"],int(row["training_seed"])):row for row in source["run_verifications"]}
    science = ablation["scientific_checkpoint_evaluations"]
    assert len(science) == 24 and len({(row["config"],row["training_seed"]) for row in science}) == 24
    for row in science:
        key = row["config"], row["training_seed"]
        assert row["original_training_metadata"] == original[key]
        assert C.sha(row["checkpoint"]) == row["checkpoint_sha256"] == original_checks[key]["checkpoint_sha256"]
        assert row["strict_load_pass"] and row["state_all_finite"]
        new = row["evaluation_4chain"]
        assert new["sample_calls"] == original_checks[key]["sample_observation"]["sample_calls"]
        assert new["proposals_filtered"] == 0 and all(call["nonfinite_elements"] == 0 for call in new["sample_calls"])
        fast = np.asarray(new["fast_seed_results"])
        assert fast[:,0].tolist() == [42,43,44,45,46]
        for index, metric, sd in [(1,"mae_ref_all6","fast_seed_sd_all6"),(2,"mae_ref_active","fast_seed_sd_active")]:
            assert close(statistics.mean(fast[:,index]),new[metric])
            assert close(statistics.stdev(fast[:,index]),new[sd])
    for name, group in ablation["groups"].items():
        assert group["training_seeds"] == [101,202,303]
        for metric in ["mae_ref_all6","mae_ref_active"]:
            comparison = group[metric]
            for tag in ["three_chain","four_chain"]:
                values = comparison[tag]["seed_values"]
                assert close(statistics.mean(values), comparison[tag]["mean"])
                assert close(statistics.stdev(values), comparison[tag]["sample_sd"])
                assert min(values)==comparison[tag]["min"] and max(values)==comparison[tag]["max"]
            assert comparison["absolute_mean_change"] == abs(comparison["four_chain"]["mean"]-comparison["three_chain"]["mean"])
    assert len(ablation["axis_rows"]) == 9
    shared = [row for row in ablation["axis_rows"] if row["config"] == "N50000_L8"]
    assert len(shared)==2 and {k:v for k,v in shared[0].items() if k!="axis"} == {k:v for k,v in shared[1].items() if k!="axis"}
    special = next(row for row in science if row["config"] == "N50000_L8" and row["training_seed"] == 202)
    assert Path(special["checkpoint"]).name == "N50000_L8_seed202_recovery.pt"
    assert special["original_training_metadata"]["prior_failed_attempt_preserved"]
    assert len(C.read_csv(C.OUT / "ROUTING_VS_GAP_4CHAIN_QUALIFIED.csv")) == 6
    assert len(C.read_csv(C.OUT / "FULL_ABLATION_4CHAIN_REFERENCE.csv")) == 9
    for result in [routing,ablation]:
        for output in result["outputs"].values():
            assert C.sha(output["path"]) == output["sha256"]
    for name in ["ROUTING_VS_GAP_4CHAIN_QUALIFIED_AUDIT.md","FULL_ABLATION_4CHAIN_REFERENCE_AUDIT.md","BOUNDARY_REFERENCE_EXCLUSION_NOTE.md"]:
        text = (C.OUT / name).read_text(encoding="utf-8")
        assert not re.search(r"best configuration|statistically equivalent|statistically comparable|\bsignificant\b", text, re.I)
    assert not ablation["summary"]["train_function_compiled"]
    return dict(verification_pass=True, qualified_reference_count=6, flow_checkpoint_count=24,
        unchanged_raw_PSIS_routing_values=True, fixed_tau=.5, boundary_absent_from_formal_quantitative_panel=True,
        sample_SD_independently_checked_with_statistics_stdev=True, all_metadata_and_checkpoint_hashes_unchanged=True,
        proposal_diagnostics_reproduced=True, shared_N50000_L8_identical=True, sources=references)


def main():
    audit_path = C.OUT / "QUALIFIED_4CHAIN_FINAL_AUDIT.md"
    integrity_path = C.OUT / "QUALIFIED_4CHAIN_FINAL_INTEGRITY.json"
    assert not audit_path.exists() and not integrity_path.exists()
    snapshot = C.read_json(C.FROZEN_PATH)
    before = C.verify(snapshot)
    quality, targets = C.frozen_quality()
    routing_path = C.CONTROL / "QUALIFIED_ROUTING_EVALUATION.json"
    ablation_path = C.CONTROL / "QUALIFIED_ABLATION_EVALUATION.json"
    routing, ablation = C.read_json(routing_path), C.read_json(ablation_path)
    checked = verify_results(routing,ablation,targets)
    rs, bs = routing["summary"], ablation["summary"]
    summary = dict(r2_1_formal_reference_targets_PASS="6/6",r2_3_formal_reference_targets_PASS="5/5",
        boundary="EXCLUDED_BY_PRESPECIFIED_REFERENCE_QUALITY_GATE",
        five_FAST_cases_max_active_MAE=rs["five_FAST_cases_max_active_MAE"], field_active_MAE=rs["field_active_MAE"],
        high_ESS_field_scale_large_gap_missed_escalation=rs["any_high_ESS_field_scale_large_gap_missed_escalation"],
        r2_1_main_conclusion_changed=rs["r2_1_main_qualitative_conclusion_changed"],
        r2_3_data_size_main_conclusion_changed=bs["r2_3_data_size_main_conclusion_changed"],
        r2_3_depth_main_conclusion_changed=bs["r2_3_depth_main_conclusion_changed"],
        flow_checkpoints_successfully_reevaluated="24/24", all_work_stopped=True,awaiting_review=True)
    preservation = C.verify(snapshot)
    lines = ["1. R2.1 formal reference targets: 6/6 complete quality gate PASS.",
        "2. R2.3 formal reference targets: 5/5 complete quality gate PASS.",
        "3. Boundary: EXCLUDED_BY_PRESPECIFIED_REFERENCE_QUALITY_GATE.",
        f"4. Five FAST cases maximum active posterior-mean MAE: {summary['five_FAST_cases_max_active_MAE']:.15g}.",
        f"5. Field active posterior-mean MAE: {summary['field_active_MAE']:.15g}.",
        f"6. Observed high-ESS + field-scale large-gap missed escalation: {summary['high_ESS_field_scale_large_gap_missed_escalation']}.",
        f"7. R2.1 main conclusion changed (3->4 chains): {summary['r2_1_main_conclusion_changed']}.",
        f"8. R2.3 data-size main conclusion changed: {summary['r2_3_data_size_main_conclusion_changed']}.",
        f"9. R2.3 depth main conclusion changed: {summary['r2_3_depth_main_conclusion_changed']}.",
        "10. Qualified four-chain reevaluation: 24/24 scientific flow checkpoints successful.","",
        "Reference qualification is read from the frozen REFERENCE_QUALITY_INTEGRITY_RECOVERED.json. The pre-specified gates are unchanged. Ablation_obs1-5 and sailboat_field_frozen pass every component. Boundary fails zero divergences because the exactly reproduced historical seed701 chain has one divergence at index375; all evidence remains frozen. This task explicitly authorizes a six-target formal panel rather than changing or repairing that gate.",
        "Boundary raw/PSIS ESS, Pareto-k and existing routing remain reference-independent and quotable. Neither its three-chain nor four-chain descriptive gap enters the formal quantitative panel, and this exclusion is not evidence for or against conservative false-positive escalation. No reviewer response or manuscript is edited.","",
        f"R2.1: five FAST cases maximum active mean 1D Wasserstein={rs['five_FAST_cases_max_active_mean_Wasserstein']:.12g}; maximum individual active-parameter Wasserstein={rs['five_FAST_cases_max_active_parameter_Wasserstein']:.12g}. Field raw/PSIS routes={rs['field_raw_route']}/{rs['field_PSIS_route']}; active mean/max Wasserstein={rs['field_active_mean_Wasserstein']:.12g}/{rs['field_active_max_Wasserstein']:.12g}.",
        f"Field active MAE is {rs['field_active_MAE_over_max_FAST']:.9g} times the largest observed FAST-case active MAE; its mean active Wasserstein is {rs['field_active_mean_Wasserstein_over_max_FAST']:.9g} times the largest FAST-case mean active Wasserstein. These are descriptive ratios without a formal statistical test. The six observed cases do not estimate universal false-negative rates, calibrated error probabilities or universal threshold optimality.",
        "The routing CSV retains the original raw IS-ESS/N, PSIS ESS/N, Pareto-k, routes and tau=0.5, and computes gaps using the existing fast samples and qualified four-chain references. All parameter quantiles and width ratios are included; active metrics are primary and all6 MAE is supplementary. Per-target three-to-four-chain MAE changes are in the routing audit.","",
        "R2.3: increasing training-set size generally reduces amortization error under the tested seeds. The three-seed mean errors decrease along the tested data axis for both metrics. Large low-data training-seed variability, especially 5k, persists. The 5k seed202 run remains retained as VALID_HIGH_VARIANCE_RUN; no seed is filtered or replaced."]
    for metric in ["mae_ref_all6","mae_ref_active"]:
        p = bs["per_metric_descriptive_comparison"][metric]
        lines.append(f"{metric}: 50k->100k mean reduction={p['mean_gain_50k_to_100k_4chain']:.12g}, {100*p['gain_50k_to_100k_over_full_data_axis_gain']:.6g}% of the full observed 5k->100k mean reduction; 5k training-seed sample SD={p['N5000_sample_SD_4chain']:.12g}.")
    lines += ["Depth has a weaker and non-monotonic influence over the tested range for the active metric; supplementary all6 means are monotone in this depth sweep. The observed descriptive patterns persist under the four-chain reference. Several adjacent depth differences are of the same order as the observed seed-to-seed variation, and all adjacent depth observed seed ranges overlap for both metrics. The repeated-seed results do not establish an eight-layer optimum and do not support fine-grained ranking. No formal statistical test is performed.",
        "Training-seed sample SD (ddof=1) remains the primary error bar; fast-seed sample SD is reported separately. Validation loss, best/last epoch, optimizer steps, training wall time, training seed, checkpoint and weights are unchanged. Reference means accumulate in float64 and are cast to float32 for the unchanged frozen evaluate() arithmetic; reference quantiles and routing reference means remain float64.",
        f"The fixed-seed checkpoint evaluation retains {bs['negative_fast_draws_retained']} negative proposal draws, with zero filtering and no NaN/Inf proposals. Negative proposals are not classified as I/O or numerical corruption.","",
        "Legacy Table8/9 provenance remains: old 50k+8-layer Table8=0.0210538022 and Table9=0.0181391388 come from two independent flow training calls. Neither is selected as the correct legacy value. The repeated-seed experiment has one scientific 50k+8-layer checkpoint per training seed, evaluated once per seed and shared identically across both axes. Seed202 uses the recovery checkpoint; its initial telemetry I/O FAILED attempt remains frozen.","",
        f"Integrity PASS: {preservation['checked_files']} protected pre-existing assets retain SHA256, including the original 367 frozen assets, all historical 3-chain references, existing seed704 chains, historical diagnostics recovery and pilot, all 24 scientific flow checkpoints including recovery, submitted manuscript/response and original training bank. All {len(snapshot['git_tracked_file_hashes'])} tracked-file hashes and Git tracked status remain unchanged.",
        "The former 0/7 FAIL audit and later 6/7 recovered audit are retained unchanged. New final files record the user-authorized qualified subset and its complete input/output provenance. No MCMC, flow training, extra seed, FE, damage grid, routing/prior/noise adjustment, paper/response edit, historical overwrite or Git mutation occurred. Evaluation has stopped for review.","",
        "Outputs:","", *[f"- {name}" for name in C.OUTPUT_NAMES], ""]
    text = "\n".join(lines)
    assert not re.search(r"best configuration|statistically equivalent|statistically comparable|\bsignificant\b",text,re.I)
    C.new_bytes(audit_path,text.encode("utf-8"))
    output_paths = [C.OUT/name for name in C.OUTPUT_NAMES if name != integrity_path.name]
    control_evidence = {str(path):C.sha(path) for path in C.CONTROL.glob("*.json")}
    integrity = dict(schema_version=1,created_utc=C.utc_now(),summary=summary,
        reference_qualification_source=dict(path=str(C.QUALITY_PATH),sha256=C.sha(C.QUALITY_PATH)),
        qualified_reference_details=checked["sources"],
        boundary_exclusion=dict(target=C.BOUNDARY,classification=summary["boundary"],
            historical_seed=701,historical_chain_index=0,divergence_count=1,divergence_indices=[375],
            excluded_from_formal_quantitative_panel=True,
            existing_reference_independent_routing={key:row for key,row in original_boundary().items()}),
        r2_1=routing,r2_3=ablation,independent_result_verification=checked,
        protected_files=snapshot["files"],git_tracked_file_hashes=snapshot["git_tracked_file_hashes"],
        original_367_files=snapshot["original_367_files"],preservation_before=before,preservation_after=preservation,
        control_evidence_sha256=control_evidence,
        controller_sources={str(path):C.sha(path) for path in C.CONTROL.glob("*.py")},
        outputs={path.name:dict(path=str(path),sha256=C.sha(path)) for path in output_paths},
        execution_boundaries=dict(new_MCMC_calls=0,flow_training_calls=0,new_training_seeds=0,
            additional_fast_experiment_seeds=0,r2_1_fast_samples_reused=True,r2_3_original_fixed_seed_evaluation_replayed=True,
            new_FE=0,damage_grid_recomputations=0,routing_threshold_changes=0,prior_noise_changes=0,
            historical_asset_overwrites=0,manuscript_response_modifications=0,Git_mutations=0),
        all_work_stopped=True,awaiting_review=True)
    C.new_json(integrity_path,integrity)
    for output in integrity["outputs"].values():
        assert C.sha(output["path"]) == output["sha256"]
    final_preservation=C.verify(snapshot)
    C.new_json(C.CONTROL/"FINAL_VERIFICATION.json",dict(created_utc=C.utc_now(),verification_pass=True,
        summary=summary,protected_files_checked=final_preservation["checked_files"],all_SHA256_unchanged=True,
        Git_tracked_files_unchanged=True,statistics_sample_SD_checks_pass=True,
        final_integrity_sha256=C.sha(integrity_path),outputs=integrity["outputs"]))
    print("QUALIFIED_FINAL_VERIFIED",json.dumps(summary),flush=True)


def original_boundary():
    row=next(row for row in C.read_csv(C.HERE/"ROUTING_VS_GAP_TABLE.csv") if row["target"]==C.BOUNDARY)
    return {key:row[key] for key in ["raw_mean","psis_mean","k_mean","route_raw","route_psis"]}


if __name__ == "__main__":
    main()
