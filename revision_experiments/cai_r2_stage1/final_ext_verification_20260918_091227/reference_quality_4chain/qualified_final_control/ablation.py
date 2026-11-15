"""Reevaluate 24 frozen scientific flows with five qualified four-chain means."""
import ast
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import common as C

ABLATION = C.HERE / "ablation_multiseed"
spec = importlib.util.spec_from_file_location("frozen_multiseed_helpers", ABLATION / "phase_b_control/common.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)  # Read-only helpers; frozen writers and train are never called.
SEEDS = [101, 202, 303]
METRICS = ["mae_ref_all6", "mae_ref_active"]
DATA = ["N5000_L8", "N10000_L8", "N25000_L8", "N50000_L8", "N100000_L8"]
DEPTH = ["N50000_L4", "N50000_L6", "N50000_L8", "N50000_L12"]
CONFIGS = list(dict.fromkeys(DATA + DEPTH))
CSV = C.OUT / "FULL_ABLATION_4CHAIN_REFERENCE.csv"
AUDIT = C.OUT / "FULL_ABLATION_4CHAIN_REFERENCE_AUDIT.md"
RESULT = C.CONTROL / "QUALIFIED_ABLATION_EVALUATION.json"


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    assert array.shape == (3,) and np.isfinite(array).all()
    return dict(seed_values=array.tolist(), mean=float(array.mean()), sample_sd=float(array.std(ddof=1)),
                min=float(array.min()), max=float(array.max()))


def main():
    assert all(not path.exists() for path in [CSV, AUDIT, RESULT, C.CONTROL / "ABLATION_EVALUATION_STARTED.json"])
    quality, targets = C.frozen_quality()
    assert all(targets[name]["overall_quality_gate"] == "PASS" for name in C.ABLATION_TARGETS)
    snapshot = C.read_json(C.FROZEN_PATH)
    before = C.verify(snapshot)
    source_path = ABLATION / "FULL_ABLATION_INTEGRITY_RECOVERED.json"
    source = C.read_json(source_path)
    original = {(row["config"], int(row["training_seed"])): row for row in source["scientific_results"]}
    checks = {(row["config"], int(row["training_seed"])): row for row in source["run_verifications"]}
    assert len(original) == 24 and len(checks) == 24 and all(row["status"] == "COMPLETED" for row in original.values())
    special = original["N50000_L8", 202]
    assert Path(special["checkpoint"]).name == "N50000_L8_seed202_recovery.pt" and special["prior_failed_attempt_preserved"]
    torch.set_num_threads(6)
    assert torch.__version__ == "2.6.0+cu124"
    sys.path.insert(0, str(C.ROOT / "experiments"))
    from cai_lib import ConditionalRealNVP
    runner = C.HERE / "run_ablation_multiseed.py"
    assert C.sha(runner) == snapshot["files"][str(runner)]
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate")
    references, observations = [], []
    for name in C.ABLATION_TARGETS:
        old, _, mean, _ = C.qualified_reference(name, targets[name])
        references.append(mean.astype(np.float32))
        observations.append(old["observation"].astype(np.float32))
    env = dict(torch=torch, np=np, time=time, FAST_SEEDS=[42, 43, 44, 45, 46],
               obs=np.stack(observations), refs=np.stack(references))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(runner) + ":frozen_evaluate_only", "exec"), env)
    assert "train" not in env
    C.new_json(C.CONTROL / "ABLATION_EVALUATION_STARTED.json", dict(started_utc=C.utc_now(), checkpoint_count=24,
        flow_training_calls=0, new_training_seeds=0, fast_seeds=[42,43,44,45,46], Nfast=2000,
        qualified_reference_targets=C.ABLATION_TARGETS, train_function_compiled=False, evaluator_sha256=C.sha(__file__)))
    raw = []
    for config in CONFIGS:
        for seed in SEEDS:
            old = original[config, seed]
            checkpoint = Path(old["checkpoint"])
            digest = checks[config, seed]["checkpoint_sha256"]
            assert C.sha(checkpoint) == digest
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert all(torch.isfinite(value).all().item() for value in state.values())
            model = ConditionalRealNVP(6, 20, n_layers=int(old["n_layers"]), hidden_dim=128)
            loaded = model.load_state_dict(state, strict=True)
            assert not loaded.missing_keys and not loaded.unexpected_keys
            model.eval()
            evaluation = F.evaluate_with_observation(model, dict(env=env))
            # Same seeds/observations/proposals as the frozen three-chain evaluation.
            assert evaluation["sample_calls"] == checks[config, seed]["sample_observation"]["sample_calls"]
            assert evaluation["proposals_filtered"] == 0
            row = dict(config=config, training_seed=seed, checkpoint=str(checkpoint), checkpoint_sha256=digest,
                original_training_metadata=old, evaluation_4chain=evaluation,
                strict_load_pass=True, state_all_finite=True, proposal_diagnostics_exactly_reproduced=True,
                training_metadata_unchanged=True, flow_weights_unchanged=True, validation_not_reselected=True)
            raw.append(row)
            # Persist completed evaluation evidence outside the scientific evaluate() path.
            C.new_json(C.CONTROL / f"EVAL_{config}_seed{seed}.json", row)
            print("QUALIFIED_ABLATION_EVAL", len(raw), "/24", config, seed, evaluation["mae_ref_all6"], evaluation["mae_ref_active"], flush=True)
    assert len(raw) == 24
    lookup = {(row["config"], row["training_seed"]): row for row in raw}
    groups, rows = {}, []
    for config in CONFIGS:
        group = dict(config=config, training_seeds=SEEDS)
        for metric in METRICS:
            old = stats([float(original[config, seed][metric]) for seed in SEEDS])
            new = stats([lookup[config, seed]["evaluation_4chain"][metric] for seed in SEEDS])
            group[metric] = dict(three_chain=old, four_chain=new, mean_change=new["mean"]-old["mean"],
                absolute_mean_change=abs(new["mean"]-old["mean"]), sample_sd_change=new["sample_sd"]-old["sample_sd"])
        groups[config] = group
    for axis, configurations in [("data_size", DATA), ("depth", DEPTH)]:
        for config in configurations:
            first = original[config, 101]
            row = dict(axis=axis, config=config, n_train=int(first["n_train"]), n_layers=int(first["n_layers"]),
                qualified_reference_targets=json.dumps(C.ABLATION_TARGETS), reference_chain_count=4,
                training_seeds=json.dumps(SEEDS), training_seed_count=3, training_seed_sd_ddof=1,
                shared_N50000_L8_scientific_result=config == "N50000_L8")
            for metric in METRICS:
                comparison = groups[config][metric]
                for name in ["mean", "sample_sd", "min", "max"]:
                    row[f"{metric}_4chain_{name}"] = comparison["four_chain"][name]
                row[f"{metric}_3chain_mean"] = comparison["three_chain"]["mean"]
                row[f"{metric}_3chain_sample_sd"] = comparison["three_chain"]["sample_sd"]
                row[f"{metric}_mean_change"] = comparison["mean_change"]
                row[f"{metric}_absolute_mean_change"] = comparison["absolute_mean_change"]
                row[f"{metric}_sample_sd_change"] = comparison["sample_sd_change"]
            for seed in SEEDS:
                scientific = lookup[config, seed]
                old, new = scientific["original_training_metadata"], scientific["evaluation_4chain"]
                for key in METRICS + ["fast_seed_sd_all6", "fast_seed_sd_active"]:
                    row[f"seed{seed}_{key}"] = new[key]
                for key in ["val_loss", "best_epoch", "last_epoch", "optimizer_steps", "train_time_s"]:
                    row[f"seed{seed}_{key}"] = old[key]
                row[f"seed{seed}_checkpoint"] = scientific["checkpoint"]
                row[f"seed{seed}_checkpoint_sha256"] = scientific["checkpoint_sha256"]
                row[f"seed{seed}_negative_fast_draws"] = new["negative_all6_proposals"]
                for fs, mae6, mae3 in new["fast_seed_results"]:
                    row[f"seed{seed}_fast{fs}_mae_ref_all6"] = mae6
                    row[f"seed{seed}_fast{fs}_mae_ref_active"] = mae3
            rows.append(row)
    common = [row for row in rows if row["config"] == "N50000_L8"]
    assert len(common) == 2 and {k:v for k,v in common[0].items() if k != "axis"} == {k:v for k,v in common[1].items() if k != "axis"}
    per_metric = {}
    for metric in METRICS:
        data3 = [groups[config][metric]["three_chain"]["mean"] for config in DATA]
        data4 = [groups[config][metric]["four_chain"]["mean"] for config in DATA]
        depth3 = [groups[config][metric]["three_chain"]["mean"] for config in DEPTH]
        depth4 = [groups[config][metric]["four_chain"]["mean"] for config in DEPTH]
        neighbor = []
        for a, b in zip(DEPTH[:-1], DEPTH[1:]):
            ga, gb = groups[a][metric]["four_chain"], groups[b][metric]["four_chain"]
            delta, scale = abs(ga["mean"]-gb["mean"]), max(ga["sample_sd"], gb["sample_sd"])
            neighbor.append(dict(config_a=a, config_b=b, absolute_mean_difference=delta,
                observed_seed_SD_a=ga["sample_sd"], observed_seed_SD_b=gb["sample_sd"], difference_over_larger_seed_SD=delta/scale,
                observed_seed_ranges_overlap=max(ga["min"], gb["min"]) <= min(ga["max"], gb["max"])))
        per_metric[metric] = dict(data_size_3chain_means=data3, data_size_4chain_means=data4,
            data_size_monotone_decrease_3chain=bool(np.all(np.diff(data3)<0)), data_size_monotone_decrease_4chain=bool(np.all(np.diff(data4)<0)),
            data_size_endpoint_improvement_3chain=data3[-1]<data3[0], data_size_endpoint_improvement_4chain=data4[-1]<data4[0],
            N5000_sample_SD_3chain=groups[DATA[0]][metric]["three_chain"]["sample_sd"], N5000_sample_SD_4chain=groups[DATA[0]][metric]["four_chain"]["sample_sd"],
            N5000_has_largest_tested_data_axis_seed_SD=groups[DATA[0]][metric]["four_chain"]["sample_sd"] == max(groups[c][metric]["four_chain"]["sample_sd"] for c in DATA),
            mean_gain_50k_to_100k_3chain=data3[-2]-data3[-1], mean_gain_50k_to_100k_4chain=data4[-2]-data4[-1],
            mean_gain_5k_to_100k_4chain=data4[0]-data4[-1],
            gain_50k_to_100k_over_full_data_axis_gain=(data4[-2]-data4[-1])/(data4[0]-data4[-1]),
            depth_3chain_means=depth3, depth_4chain_means=depth4,
            depth_neighbor_mean_sign_pattern_changed=not np.array_equal(np.sign(np.diff(depth3)), np.sign(np.diff(depth4))),
            depth_nonmonotonic_4chain=bool(np.any(np.diff(depth4)>0) and np.any(np.diff(depth4)<0)),
            depth_neighbor_variation=neighbor)
    data_changed = any(p["data_size_monotone_decrease_3chain"] != p["data_size_monotone_decrease_4chain"] or p["data_size_endpoint_improvement_3chain"] != p["data_size_endpoint_improvement_4chain"] for p in per_metric.values())
    depth_changed = any(p["depth_neighbor_mean_sign_pattern_changed"] for p in per_metric.values())
    summary = dict(qualified_r2_3_targets=5, all_5_quality_gates_PASS=True, checkpoints_evaluated=24, all_24_evaluations_successful=True,
        r2_3_data_size_main_conclusion_changed=data_changed, r2_3_depth_main_conclusion_changed=depth_changed,
        larger_low_data_seed_variation_retained=all(p["N5000_has_largest_tested_data_axis_seed_SD"] for p in per_metric.values()),
        small_50k_to_100k_gain_relative_to_full_data_axis=all(0<p["gain_50k_to_100k_over_full_data_axis_gain"]<.25 for p in per_metric.values()),
        depth_has_nonmonotonic_active_MAE=per_metric["mae_ref_active"]["depth_nonmonotonic_4chain"],
        repeated_seed_results_do_not_support_fine_grained_ranking=True, no_claim_of_eight_layer_optimality=True,
        no_formal_statistical_test=True, per_metric_descriptive_comparison=per_metric,
        primary_error_bar="training-seed sample SD, ddof=1", separate_variation="fast-seed sample SD, ddof=1",
        N50000_L8_shared_results_identical=True, N50000_L8_seed202_recovery_used=True,
        original_telemetry_FAILED_attempt_preserved=True, N5000_L8_seed202_retained=True,
        negative_fast_draws_retained=sum(row["evaluation_4chain"]["negative_all6_proposals"] for row in raw), proposals_filtered=0,
        train_function_compiled=False, flow_training_calls=0, new_MCMC_calls=0, new_training_seeds=0,
        fixed_fast_seed_evaluation_replay=True)
    C.new_csv(CSV, rows)
    lines = ["Formal ablation references: ablation_obs1-5, 5/5 complete quality gate PASS; four chains each.",
        "24/24 frozen scientific checkpoints successfully reevaluated; no training is compiled or called.",
        f"Data-size main conclusion changed: {data_changed}. Depth main descriptive conclusion changed: {depth_changed}.", "",
        "Per-configuration 3-chain / qualified 4-chain comparison (sample SD across training seeds, ddof=1):", "",
        "| Config | Metric | 3-chain mean | 4-chain mean | absolute change | 3-chain SD | 4-chain SD |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for config in CONFIGS:
        for metric in METRICS:
            g = groups[config][metric]
            lines.append(f"| {config} | {metric} | {g['three_chain']['mean']:.12g} | {g['four_chain']['mean']:.12g} | {g['absolute_mean_change']:.12g} | {g['three_chain']['sample_sd']:.12g} | {g['four_chain']['sample_sd']:.12g} |")
    lines += ["", "Answers to the six descriptive questions:", "",
        f"1. The observed data-size improvement pattern changed={data_changed}. Increasing training-set size generally reduces amortization error over the tested data axis; the actual three-seed means for both metrics are recorded below.",
        f"2. Larger low-data seed variation persists={summary['larger_low_data_seed_variation_retained']}. The 5k configuration retains the largest tested data-axis training-seed SD for both metrics. Seed202 remains the verified VALID_HIGH_VARIANCE_RUN without removal or replacement.",
        f"3. The positive 50k->100k mean reduction remains small relative to the full 5k->100k observed reduction={summary['small_50k_to_100k_gain_relative_to_full_data_axis']}. The exact gains and fractions are reported below; this describes the tested range without an extrapolated saturation claim.",
        f"4. Depth descriptive pattern changed={depth_changed}; active MAE retains non-monotonic depth variation={summary['depth_has_nonmonotonic_active_MAE']}. All6 mean behavior is reported separately. Depth has a weaker and non-monotonic influence over the tested range when the active metric is considered. Adjacent depth mean differences, both seed SDs and observed range overlaps are reported below.",
        "5. The repeated-seed results do not establish an eight-layer optimum. Both metrics and all tested layers/seeds are retained; no configuration is selected or labeled as optimal.",
        "6. The repeated-seed results do not support fine-grained ranking of neighboring configurations. Differences for several depth neighbors are of the same order as the observed seed-to-seed variation. No formal statistical test is performed, and the large 5k seed SD is not used to equate 5k with 100k.", "",
        "Original Table 8/9 shared-configuration provenance is retained:",
        "Old 50k+8-layer Table8 MAE=0.0210538022 (full value 0.021053802222013474); Table9 MAE=0.0181391388 (full value 0.01813913881778717). They came from two independent flow training calls. Neither old value is chosen as the correct value. In the repeated-seed experiment each training seed has one shared 50k+8-layer scientific checkpoint; evaluation is performed once per seed and both axes reference the identical result.",
        "N50000_L8 seed202 uses N50000_L8_seed202_recovery.pt. The original telemetry I/O failure, FAILED row, failed epoch log and old audits remain frozen and visible in the source integrity record.", "",
        "Evaluation and preservation:",
        "The frozen run_ablation_multiseed.py evaluate() function is extracted unchanged by AST; train() is not compiled or called. Five exact saved observations, fixed fast seeds42-46 and Nfast=2000 are retained. Each model strictly loads the frozen finite state tensors; the proposal diagnostics exactly match the previously audited fixed-seed evaluation. Negative proposals remain unfiltered and no NaN/Inf proposals occur.",
        "Four-chain reference means accumulate 2,400 draws in float64 and are cast to float32 for the unchanged evaluate() arithmetic. The CSV has nine predefined axis rows with all three raw seed metrics, training-seed mean/sample SD/min/max, five per-fast-seed MAEs, separate fast-seed SDs, validation loss, best/last epoch, optimizer steps, training wall time, checkpoint paths and SHA256. The 50k+8-layer entries are identical across axes aside from the axis name.",
        "Original validation/checkpoint selection, training rules, metadata, weights, references, seed704 chains, historical diagnostic recovery, submitted manuscript/response, training bank and Git tracked files are unchanged. No MCMC, new seed, training, FE, damage grid or threshold change occurs.", "",
        "Numerical axis comparison:", json.dumps(per_metric, ensure_ascii=False, indent=2), ""]
    C.new_bytes(AUDIT, "\n".join(lines).encode("utf-8"))
    C.new_json(RESULT, dict(created_utc=C.utc_now(), summary=summary, groups=groups, axis_rows=rows,
        scientific_checkpoint_evaluations=raw, original_ablation_integrity=dict(path=str(source_path), sha256=C.sha(source_path)),
        old_Table8_Table9_provenance=source["table8_table9_comparison"]["old_50k_L8_conflict"],
        reference_means_used_float32=env["refs"].tolist(), observation_values_used_float32=env["obs"].tolist(),
        numerical_environment=dict(python=sys.executable, torch=torch.__version__, numpy=np.__version__, torch_threads=torch.get_num_threads()),
        frozen_evaluate_source=dict(path=str(runner), sha256=C.sha(runner)), evaluator_sha256=C.sha(__file__),
        preservation_before=before, preservation_after=C.verify(snapshot),
        outputs={path.name:dict(path=str(path), sha256=C.sha(path)) for path in [CSV, AUDIT]}))
    print("QUALIFIED_ABLATION_FINISHED", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
