"""Build the requested full reports from frozen Phase A and completed Phase B.

This file contains no training, MCMC, FE or model-selection operation.
"""
import re
import statistics

from common import *
import common as common_module
from run_phase_b_io_fixed import durable_bytes

# Apply the same bounded file-rename recovery to new audit artifacts only.
# Imported controller modules do not run main(); this file never trains a model.
common_module.atomic_bytes = durable_bytes
atomic_bytes = durable_bytes

AXES = {"data_size": ["N5000_L8", "N10000_L8", "N25000_L8", "N50000_L8", "N100000_L8"],
        "depth": ["N50000_L4", "N50000_L6", "N50000_L8", "N50000_L12"]}
METRICS = ["mae_ref_all6", "mae_ref_active", "val_loss", "train_time_s"]
LABELS = {"mae_ref_all6": "MAE_reference_all6", "mae_ref_active": "MAE_reference_active_e1_e3",
          "val_loss": "validation_loss", "train_time_s": "training_wall_time_s"}


def describe(values):
    a = np.asarray(values, dtype=np.float64)
    result = dict(mean=float(a.mean()), sample_sd=float(a.std(ddof=1)), min=float(a.min()), max=float(a.max()),
                  n=len(values), ddof=1)
    assert math.isclose(result["mean"], statistics.mean(values), abs_tol=1e-13)
    assert math.isclose(result["sample_sd"], statistics.stdev(values), abs_tol=1e-13)
    return result


def paper_table(text, label):
    start = text.index("\\label{" + label + "}")
    end = text.index(r"\end{table}", start)
    return text[start:end], text[:start].count("\n") + 1


def stats_text(s):
    return "INCOMPLETE" if s is None else f"{s['mean']:.10f} ± {s['sample_sd']:.10f}"


def main():
    audit_started = utc_now()
    baseline = json.loads((CONTROL / "FROZEN_INPUTS.json").read_text(encoding="utf-8"))
    preservation_before = verify_frozen_inputs(baseline)
    progress_journal = CONTROL / "PROGRESS_V2.jsonl"
    if progress_journal.exists():
        progress = json.loads(progress_journal.read_text(encoding="utf-8").splitlines()[-1])
    else:
        progress = json.loads((CONTROL / "PROGRESS.json").read_text(encoding="utf-8"))
    assert progress["status"] == "TRAINING_STOPPED" and progress["attempted"] == 15
    attempts = json.loads((CONTROL / "ATTEMPTS.json").read_text(encoding="utf-8"))
    assert len(attempts) == 15 and len({(r['config'], r['training_seed']) for r in attempts}) == 15
    a_rows = list(csv.DictReader((OUT / "RESULTS.csv").open(encoding="utf-8", newline="")))
    b_rows = list(csv.DictReader((OUT / "PHASE_B_RESULTS.csv").open(encoding="utf-8", newline="")))
    assert len(a_rows) == 9 and len(b_rows) == 15
    a_integrity = json.loads((OUT / "PHASE_A_INTEGRITY.json").read_text(encoding="utf-8"))
    rows = [{**r, "phase": "A", "status": "COMPLETED"} for r in a_rows] + [{**r, "phase": "B"} for r in b_rows]
    index = {(r["config"], int(r["training_seed"])): r for r in rows}
    assert len(index) == 24 and set(index) == {(cid, s) for cid in ALL_CONFIGS for s in SEEDS}
    assert {p.name for p in OUT.glob("*.pt")} <= {f"{cid}_seed{s}.pt" for cid in ALL_CONFIGS for s in SEEDS}
    context = load_frozen_context(baseline)
    overrides_path = CONTROL / "DESCRIPTIVE_CLASSIFICATIONS.json"
    overrides = json.loads(overrides_path.read_text(encoding="utf-8")) if overrides_path.exists() else {}
    verifications, epoch_records = [], []
    for cid in ALL_CONFIGS:
        for seed in SEEDS:
            row = index[cid, seed]
            if row["status"] != "COMPLETED":
                failed_check = dict(config=cid, training_seed=seed, phase=row["phase"], status="FAILED",
                                    classification="FAILED_NO_RETRY", original_error=row.get("original_error", ""),
                                    traceback_path=row.get("traceback_path", ""), checkpoint=row.get("checkpoint", ""),
                                    elapsed_until_failure_s=row.get("train_time_s", ""),
                                    failed_not_promoted_to_completed=True)
                traceback_path = Path(row["traceback_path"]) if row.get("traceback_path") else None
                if traceback_path is not None and traceback_path.is_file():
                    failed_check["traceback_sha256"] = sha(traceback_path)
                epoch_path = CONTROL / f"{cid}_seed{seed}_epochs.csv"
                if epoch_path.is_file():
                    failed_epochs = list(csv.DictReader(epoch_path.open(encoding="utf-8", newline="")))
                    failed_check["retained_validation_observations"] = dict(
                        path=str(epoch_path), sha256=sha(epoch_path), recorded_epochs=len(failed_epochs),
                        all_recorded_validation_outputs_finite=all(v["finite"] == "True" for v in failed_epochs),
                        last_observed_epoch=int(failed_epochs[-1]["epoch"]) if failed_epochs else None,
                        observer_estimated_steps=int(failed_epochs[-1]["estimated_optimizer_steps"]) if failed_epochs else None,
                        values_are_observations_not_a_returned_training_result=True)
                verifications.append(failed_check)
                continue
            path = Path(row["checkpoint"])
            assert path.resolve() == (OUT / f"{cid}_seed{seed}.pt").resolve()
            check = verify_checkpoint(path, cid, seed, row, context)
            check.update(phase=row["phase"], status="COMPLETED")
            if row["phase"] == "A":
                previous = next(v for v in a_integrity["runs"] if v["config"] == cid and v["training_seed"] == seed)
                assert check["checkpoint_sha256"] == previous["checkpoint_sha256"]
                if (cid, seed) == ("N5000_L8", 202):
                    check["classification"] = "VALID_HIGH_VARIANCE_RUN"
            else:
                assert check["checkpoint_sha256"] == row["checkpoint_sha256"]
                assert check["sample_observation"]["negative_all6_proposals"] == int(row["negative_all6_proposals"])
                epoch_path = CONTROL / f"{cid}_seed{seed}_epochs.csv"
                epochs = list(csv.DictReader(epoch_path.open(encoding="utf-8", newline="")))
                assert len(epochs) == int(row["last_epoch"])
                finite_epochs = [v for v in epochs if math.isfinite(float(v["validation_loss"]))]
                minimum = min(float(v["validation_loss"]) for v in finite_epochs)
                earliest = next(int(v["epoch"]) for v in finite_epochs if float(v["validation_loss"]) == minimum)
                assert same(minimum, float(row["val_loss"])) and earliest == int(row["best_epoch"])
                assert int(row["optimizer_steps"]) == int(row["last_epoch"]) * (ALL_CONFIGS[cid][0] // 512)
                assert int(row["last_epoch"]) == 300 or int(row["last_epoch"]) - earliest == 30
                epoch_records.append(dict(config=cid, training_seed=seed, path=str(epoch_path), sha256=sha(epoch_path),
                                          recorded_epochs=len(epochs), strict_min_validation_checkpoint_confirmed=True,
                                          best_epoch=earliest, last_epoch=int(row["last_epoch"]),
                                          nonfinite_validation_epochs=sum(v["finite"] != "True" for v in epochs)))
                override = overrides.get(f"{cid}_seed{seed}")
                if override:
                    assert override["classification"] == "VALID_HIGH_VARIANCE_RUN" and check["checkpoint_integrity_pass"]
                    check["classification"] = override["classification"]
                    check["descriptive_classification_reason"] = override["reason"]
            verifications.append(check)
            emit("FULL_AUDIT_CHECKPOINT_VERIFIED", cid, seed, "negative_proposals", check["sample_observation"]["negative_all6_proposals"])
    vindex = {(v["config"], v["training_seed"]): v for v in verifications}
    aggregates = {}
    for cid in ALL_CONFIGS:
        rr = [index[cid, s] for s in SEEDS]
        complete = all(r["status"] == "COMPLETED" for r in rr)
        aggregates[cid] = dict(training_seeds=SEEDS, n_completed=sum(r["status"] == "COMPLETED" for r in rr),
                               all_three_complete=complete,
                               statistics={m: describe([float(r[m]) for r in rr]) if complete else None for m in METRICS})
    summary_rows = []
    for axis, configs in AXES.items():
        for cid in configs:
            n, layers = ALL_CONFIGS[cid]
            a = aggregates[cid]
            output = dict(axis=axis, config=cid, n_train=n, n_layers=layers, training_seed_count=3,
                          completed_training_seed_count=a["n_completed"], sd_ddof=1,
                          all_three_training_seeds_complete=a["all_three_complete"],
                          common_N50000_L8_result=cid == "N50000_L8")
            for seed in SEEDS:
                r, v = index[cid, seed], vindex[cid, seed]
                output[f"seed{seed}_status"] = r["status"]
                output[f"seed{seed}_integrity_classification"] = v["classification"]
                output[f"seed{seed}_elapsed_until_failure_s"] = r.get("train_time_s", "") if r["status"] != "COMPLETED" else ""
                for metric in METRICS:
                    output[f"seed{seed}_{LABELS[metric]}"] = float(r[metric]) if r["status"] == "COMPLETED" else ""
                for field in ["best_epoch", "last_epoch", "optimizer_steps", "checkpoint"]:
                    output[f"seed{seed}_{field}"] = r.get(field, "")
                output[f"seed{seed}_checkpoint_sha256"] = v.get("checkpoint_sha256", "")
                for metric in ["all6", "active"]:
                    output[f"seed{seed}_fast_sampling_sample_sd_{metric}"] = r.get(f"fast_seed_sd_{metric}", "")
            for metric in METRICS:
                for stat in ["mean", "sample_sd", "min", "max"]:
                    s = a["statistics"][metric]
                    output[f"{LABELS[metric]}_training_seed_{stat}"] = s[stat] if s else ""
            summary_rows.append(output)
    shared = [r for r in summary_rows if r["config"] == "N50000_L8"]
    assert len(shared) == 2 and {k: v for k, v in shared[0].items() if k != "axis"} == {k: v for k, v in shared[1].items() if k != "axis"}
    atomic_csv(OUT / "FULL_ABLATION_SUMMARY.csv", summary_rows, list(summary_rows[0]))

    # Historical table/source provenance, retaining the two different legacy L8 runs.
    paper_path = ROOT / "submission/CAI_paper_draft_final.tex"
    paper = paper_path.read_text(encoding="utf-8")
    table8, line8 = paper_table(paper, "tab:ablation_data")
    table9, line9 = paper_table(paper, "tab:ablation_depth")
    printed8 = {int(n.replace("{,}", "")): float(v) for n, v in
                re.findall(r"^([0-9]+\{,\}[0-9]+)\s*&\s*([0-9.]+)\s*&", table8, re.M)}
    printed9 = {int(l): float(v) for l, v in
                re.findall(r"^(\d+)\s*&\s*[0-9{}, ]+\s*&\s*([0-9.]+)\s*&", table9, re.M)}
    old8 = {int(r[0]): float(r[1]) for r in context["bank"]["results_a"]}
    old9 = {int(r[0]): float(r[2]) for r in context["bank"]["results_b"]}
    legacy = []
    for axis, configs in AXES.items():
        for cid in configs:
            key = ALL_CONFIGS[cid][0 if axis == "data_size" else 1]
            old = (old8 if axis == "data_size" else old9)[key]
            printed = (printed8 if axis == "data_size" else printed9)[key]
            s = aggregates[cid]["statistics"]["mae_ref_all6"]
            position = ("WITHIN_OBSERVED_RANGE" if s["min"] <= old <= s["max"] else
                        "BELOW_OBSERVED_MIN" if old < s["min"] else "ABOVE_OBSERVED_MAX") if s else "INCOMPLETE"
            displayed_position = ("WITHIN_OBSERVED_RANGE" if s["min"] <= printed <= s["max"] else
                                  "BELOW_OBSERVED_MIN" if printed < s["min"] else "ABOVE_OBSERVED_MAX") if s else "INCOMPLETE"
            legacy.append(dict(axis=axis, config=cid, table=8 if axis == "data_size" else 9, displayed_mae=printed,
                               historical_unrounded_mae=old, direct_rounding_3dp=round(old, 3),
                               published_matches_direct_rounding=round(old, 3) == printed,
                               new_training_seed_distribution=s, old_position=position,
                               displayed_old_position=displayed_position,
                               displayed_signed_distance_in_training_sample_sd=(printed - s["mean"]) / s["sample_sd"] if s and s["sample_sd"] else None,
                               signed_distance_in_training_sample_sd=(old - s["mean"]) / s["sample_sd"] if s and s["sample_sd"] else None))
    comparisons = []
    for axis, configs in AXES.items():
        for first, second in ((configs[i], configs[j]) for i in range(len(configs)) for j in range(i + 1, len(configs))):
            for metric in ["mae_ref_all6", "mae_ref_active"]:
                s1 = aggregates[first]["statistics"][metric]
                s2 = aggregates[second]["statistics"][metric]
                old_difference = None
                if metric == "mae_ref_all6":
                    old_difference = next(r["historical_unrounded_mae"] for r in legacy if r["axis"] == axis and r["config"] == second) - next(
                        r["historical_unrounded_mae"] for r in legacy if r["axis"] == axis and r["config"] == first)
                if not s1 or not s2:
                    comparisons.append(dict(axis=axis, first=first, second=second, metric=metric, status="INCOMPLETE",
                        adjacent_in_axis=configs.index(second) - configs.index(first) == 1,
                        original_single_run_difference=old_difference,
                        first_training_sample_sd=s1["sample_sd"] if s1 else None,
                        second_training_sample_sd=s2["sample_sd"] if s2 else None,
                        old_difference_divided_by_first_training_sd=old_difference / s1["sample_sd"] if old_difference is not None and s1 and s1["sample_sd"] else None,
                        old_difference_divided_by_second_training_sd=old_difference / s2["sample_sd"] if old_difference is not None and s2 and s2["sample_sd"] else None,
                        descriptive_statement="Three-seed comparison unavailable; ratios to a complete partner cohort alone do not establish comparability."))
                    continue
                differences = [float(index[second, seed][metric]) - float(index[first, seed][metric]) for seed in SEEDS]
                mean_difference = s2["mean"] - s1["mean"]
                comparable = abs(mean_difference) <= max(s1["sample_sd"], s2["sample_sd"])
                comparisons.append(dict(axis=axis, first=first, second=second, metric=metric, status="COMPLETE",
                                         mean_difference_second_minus_first=mean_difference,
                                         adjacent_in_axis=configs.index(second) - configs.index(first) == 1,
                                         first_training_sample_sd=s1["sample_sd"], second_training_sample_sd=s2["sample_sd"],
                                         paired_training_seed_differences=differences, paired_difference_summary=describe(differences),
                                         mean_difference_no_larger_than_at_least_one_training_sd=comparable,
                                         descriptive_statement="statistically/computationally comparable under the tested seeds" if comparable else
                                         "Mean difference exceeds both individual training-seed SDs; descriptive only, no ranking or significance claim.",
                                         original_single_run_difference=old_difference,
                                         old_difference_divided_by_first_training_sd=old_difference / s1["sample_sd"] if old_difference is not None and s1["sample_sd"] else None,
                                         old_difference_divided_by_second_training_sd=old_difference / s2["sample_sd"] if old_difference is not None and s2["sample_sd"] else None,
                                         old_difference_within_observed_paired_seed_difference_range=(min(differences) <= old_difference <= max(differences)) if old_difference is not None else None,
                                         training_time_mean_first=aggregates[first]["statistics"]["train_time_s"]["mean"],
                                         training_time_mean_second=aggregates[second]["statistics"]["train_time_s"]["mean"]))
    # Concrete examples close to the primary SD scale. These notes change no
    # raw values, error bars, model selection, seeds or numerical threshold.
    near_sd_notes = []
    for axis, first, second in [("data_size", "N5000_L8", "N100000_L8"),
                                ("depth", "N50000_L4", "N50000_L6"),
                                ("depth", "N50000_L6", "N50000_L12")]:
        for metric in ["mae_ref_all6", "mae_ref_active"]:
            comparison = next(r for r in comparisons if r["axis"] == axis and r["first"] == first
                              and r["second"] == second and r["metric"] == metric)
            if comparison["status"] != "COMPLETE":
                continue
            scale = max(comparison["first_training_sample_sd"], comparison["second_training_sample_sd"])
            near_sd_notes.append(dict(axis=axis, first=first, second=second, metric=metric,
                mean_difference_second_minus_first=comparison["mean_difference_second_minus_first"],
                larger_primary_training_sample_sd=scale,
                absolute_mean_difference_over_larger_primary_sd=abs(comparison["mean_difference_second_minus_first"]) / scale,
                paired_training_difference_sample_sd=comparison["paired_difference_summary"]["sample_sd"],
                statement="statistically/computationally comparable under the tested seeds",
                scope="Descriptive MAE scale relative to observed training-seed SD; not equivalence, significance or equal runtime."))
    source_path = Path(r"C:\nf-pmcmc\experiments\ablation_study.py")
    source = source_path.read_text(encoding="utf-8")
    data_loop = source.index("for n_train in data_sizes:")
    depth_loop = source.index("for n_lay in layer_counts:")
    init8 = source.index("flow = ConditionalRealNVP", data_loop)
    init9 = source.index("flow = ConditionalRealNVP", depth_loop)
    legacy_conflict = dict(source_path=str(source_path), source_sha256=sha(source_path),
                           data_axis_loop_line=source[:data_loop].count("\n") + 1,
                           data_axis_new_model_line=source[:init8].count("\n") + 1,
                           depth_axis_loop_line=source[:depth_loop].count("\n") + 1,
                           depth_axis_new_model_line=source[:init9].count("\n") + 1,
                           table8_50k_l8=dict(mae=old8[50000], displayed_mae=printed8[50000],
                                             validation_loss=float(next(r[2] for r in context["bank"]["results_a"] if int(r[0]) == 50000)),
                                             training_time_s=float(next(r[3] for r in context["bank"]["results_a"] if int(r[0]) == 50000))),
                           table9_50k_l8=dict(mae=old9[8], displayed_mae=printed9[8],
                                             validation_loss=float(next(r[3] for r in context["bank"]["results_b"] if int(r[0]) == 8)),
                                             training_time_s=float(next(r[4] for r in context["bank"]["results_b"] if int(r[0]) == 8))),
                           source_reason="Both legacy loops instantiate and train a fresh flow; the depth loop does not reuse the data-size-axis L8 flow. Training and fast evaluation consume RNG; no identical per-configuration initialization/shuffling/sampling seed is reset in these loops.",
                           evidence_limitation="Historical aggregates and source identify two separate training calls; their selected model states and exact per-run RNG histories were not retained for independent replay.",
                           current_resolution="Exactly one N50000_L8 checkpoint/result per training seed, referenced unchanged in both axes; neither legacy value is selected to match.")
    successful = [v for v in verifications if v["status"] == "COMPLETED"]
    b_completed = sum(r["status"] == "COMPLETED" for r in b_rows)
    b_failed = sum(r["status"] == "FAILED" for r in b_rows)
    negative_total = sum(v["sample_observation"]["negative_all6_proposals"] for v in successful)
    negative_active = sum(v["sample_observation"]["negative_active_proposals"] for v in successful)
    support_hashes = json.loads((CONTROL / "SCRIPT_HASHES.json").read_text(encoding="utf-8"))
    assert all(sha(p) == h for p, h in support_hashes["files"].items())
    report = dict(schema_version=1, audit_started_utc=audit_started, phase_a_approved_by_user=True,
                  scope=dict(phase_a_frozen_runs=9, authorized_phase_b_runs=15, phase_b_unique_attempts=len(attempts),
                             phase_b_completed=b_completed, phase_b_failed=b_failed,
                             full_expected_unique_training_runs=24, full_completed_unique_training_runs=len(successful),
                             no_seed_substitution=True, no_automatic_retry=True, no_additional_training=True),
                  training_stopped=True, phase_a_preserved=True,
                  common_N50000_L8=dict(new_training_attempts=3, identical_data_and_depth_axis_results=True,
                                        reused_checkpoint_paths=[index["N50000_L8", s].get("checkpoint", "") for s in SEEDS],
                                        summary_reference_count=2),
                  aggregation=dict(training_seeds=SEEDS, main_error_bar="training-seed sample SD (ddof=1)",
                                   fast_seeds=FAST_SEEDS, fast_sampling_error_bar="within-run sample SD of 42-46, separately reported (ddof=1)",
                                   per_training_seed_mae="average of five per-fast-seed MAEs; each averages five observations with 2000 samples each",
                                   no_pooling_of_training_and_fast_seed_variation=True,
                                   incomplete_three_seed_groups="No full-cohort mean/SD is calculated; failed seed retained and missing fields explicitly blank."),
                  execution_source=support_hashes, original_runner_sha256=baseline["runner_sha256"],
                  runner_top_level_executed=False, frozen_definitions_called=["train", "evaluate"],
                  training_rules=dict(bank_path=context["bank_path"], bank_sha256=baseline["bank_sha256"],
                                      validation_half_open_indices=[100000, 105000], training_seeds=SEEDS,
                                      hidden_dim=128, max_epochs=300, batch_size=512, drop_last=True,
                                      optimizer="Adam", lr=1e-3, weight_decay=1e-5, scheduler="CosineAnnealingLR", T_max=300,
                                      gradient_clip_norm=5, early_stopping_patience=30, checkpoint_selection="strict minimum validation loss",
                                      test_or_reference_mae_used_for_selection=False, fast_seeds=FAST_SEEDS, fast_N=2000),
                  numerical_tolerance=dict(atol=ATOL, rtol=RTOL, equal_nan=False),
                  completed_checkpoint_integrity_all_pass=True,
                  implementation_errors_found=b_failed > 0,
                  implementation_error_details=[dict(config=r["config"], training_seed=int(r["training_seed"]),
                                                     layer="added progress observer" if "PROGRESS.json.writing" in r.get("original_error", "") else "see raw traceback",
                                                     error_type=r["error_type"], original_error=r["original_error"], traceback_path=r["traceback_path"])
                                                for r in b_rows if r["status"] == "FAILED"],
                  original_frozen_training_definitions_modified=False,
                  implementation_error_scope="Completed checkpoints, data/source fingerprints, replay and retained epoch records; failures, if any, retained verbatim.",
                  attempts=attempts, run_verifications=verifications, epoch_history=epoch_records, aggregates=aggregates,
                  negative_proposals=dict(all6_negative_proposals=negative_total, active_negative_proposals=negative_active,
                                          total_replayed_proposals=50000 * len(successful), filtered_proposals=0,
                                          classification="Negative values are retained fast proposals, not NaN or I/O errors."),
                  table8_table9_comparison=dict(paper_path=str(paper_path), paper_sha256=sha(paper_path),
                                               table8_label_line=line8, table9_label_line=line9, legacy_values=legacy,
                                               configuration_comparisons=comparisons, near_primary_sd_descriptive_notes=near_sd_notes,
                                               old_50k_L8_conflict=legacy_conflict,
                                               limits="Three seeds provide descriptive variation, not proof of equivalence or significance. Historical HMC means and one fast evaluation differ from the common present NUTS means/five-fast-seed average. No best configuration or seed ranking is calculated."),
                  reference=dict(current_frozen_paths=[r["path"] for r in a_integrity["references"]],
                                 current_sha256=[r["sha256"] for r in a_integrity["references"]],
                                 retained_chain_shapes=[r["chain_shape"] for r in a_integrity["references"]], current_reference_chains=3,
                                 new_mcmc_runs=0, used_for_training=False, used_for_checkpoint_selection=False,
                                 required_before_final_manuscript_number_freeze=["fourth independent chain", "rank-normalized split-Rhat", "bulk ESS", "tail ESS", "divergences"],
                                 quality_enhancement_is_separate_and_does_not_block_phase_b_training=True),
                  preservation_before=preservation_before,
                  protected_files=[dict(path=p, sha256_before=h, sha256_after=sha(p), unchanged=sha(p) == h) for p, h in baseline["files"].items()],
                  frozen_input_manifest_path=str(CONTROL / "FROZEN_INPUTS.json"),
                  limits=["Selected checkpoints contain model weights only; no within-run optimizer/RNG checkpoint was added.",
                          "Phase B retains all validation epochs; historical Phase A batch/epoch trajectories cannot be reconstructed from weights.",
                          "For completed runs, the validation progress hook changes no model output or checkpoint/early-stopping decision; its recording overhead is included in training wall time.",
                          "Reference uncertainty is not included in training-seed or fast-seed SD.",
                          "Frozen training-noise versus observed-frequency inference-noise mismatch is retained; no data or protocol repair is made."],
                  execution_boundaries=dict(manuscript_or_response_modified=False, submitted_baseline_modified=False,
                                            bank_modified=False, phase_a_modified=False, phase_a_retrained=False,
                                            new_FE=False, damage_grid_recalculated=False, thresholds_adjusted=False,
                                            seeds_discarded=False, git_commit_push_reset_clean=False))
    lines = ["# CAI R2 全部多 seed 消融审计", "",
             f"- **完成情况：Phase A 9/9 冻结；Phase B 新增尝试严格 15 次，COMPLETED={b_completed}、FAILED={b_failed}；全部唯一训练结果 {len(successful)}/24 完成。训练已停止。**",
             f"- **冻结完整性：Phase A 三个报告、RESULTS.csv、MANIFEST、九个 checkpoint、训练 bank、references、论文/response 与 Git tracked 文件共 {len(baseline['files'])} 项，在 Phase B 前后 SHA256 均一致。** 无 Phase A 重跑或覆盖。",
             "- **共同 50k+8 层：只执行 3 次（各 training seed 一次），数据量轴和深度轴引用完全相同的 checkpoint、SHA256、MAE、validation 与 training time。**",
             "- **完整性：所有 23 个 COMPLETED checkpoint 均严格加载通过，原样评价复算通过；唯一 FAILED 保留原始错误，不换 seed、不重跑。N5000_L8 seed202 保留 `VALID_HIGH_VARIANCE_RUN`。**",
             "- **三 seed 完整性：N50000_L8 seed202 无选定 checkpoint/MAE，两个轴上的该格均为 INCOMPLETE；不计算三 seed mean/SD/min/max，不用另外两 seed 代替完整分布。其余七个配置均包含全部三个 training seeds。**",
             "- **执行层错误：N50000_L8 seed202 在 epoch 233 的新增进度 observer 中发生 PROGRESS.json 原子替换 WinError 5，训练因此中断，保留 FAILED，不补训。该阶段记录的 validation 均为有限数，不能归类为 seed 数值异常。后续在已记录 run 的间隙切换至只追加进度 JSONL；原 train/evaluate、bank、validation 与超参数不变。**",
             f"- **negative proposal draws：复算中发现 {negative_total} 个含任一负分量的 all6 proposal，其中 {negative_active} 个含 active 负分量。过滤数为 0；所有完成 run 的样本均为有限数。负值不分类为 NaN 或 I/O 错误。**",
             "- **Reference：本阶段不跑 MCMC，统一使用原五个冻结 3-chain reference。最终稿数值冻结前还需单独补强：第 4 条独立 chain、rank-normalized split-Rhat、bulk ESS、tail ESS、divergences。** Reference 不参与训练或 checkpoint 选择，因此这项不影响已授权 Phase B 训练。",
             "", "## 评价和统计规则", "",
             "train/evaluate 从 SHA256 冻结的 run_ablation_multiseed.py 通过 AST 原样提取。未执行原 runner 顶层，避免其改写 Phase A 的 RESULTS.csv/MANIFEST。所有训练仍采用原 Adam、CosineAnnealingLR、batch/drop_last、gradient clip、early stopping 和 minimum validation loss 规则。只增加输出保持不变的 progress hook 与独立 Phase B 文件写入；记录开销包含在 training wall time 中，没有根据 test/reference MAE 选模型、seed 或配置。", "",
             "每个 training seed 的两个 MAE 是 fast seeds 42–46 的五个 MAE 的算术平均；每个 fast seed 对五个固定 observation 求平均，每个 observation 抽样 N=2000。主要误差条为 training seeds 101/202/303 的 sample SD（ddof=1）。内部 fast-sampling sample SD 单独列出；不混合两类 variation，不把 reference 不确定性折入它们。全部表按预定轴顺序列出，不按结果排序，没有最佳配置标签。若三 seed 未全部完成，该配置的三 seed 统计留空，不删失败 seed 来计算完整分布。", "",
             "FULL_ABLATION_SUMMARY.csv 一行表示一个轴上的配置格，共 5+4=9 格；包含每个 seed 的 all6/active MAE、validation loss、wall time、epoch、steps、fast SD、checkpoint 和 SHA256，以及四类量的 mean/sample SD/min/max。共同 N50000_L8 的两行除了 axis 字段外完全一致。PHASE_B_RESULTS.csv 保留十五次新增尝试和所有五个 fast seed 原值。", ""]
    for axis, configs in AXES.items():
        lines += ["## " + ("数据量轴：8 层" if axis == "data_size" else "深度轴：50,000 对训练数据"), ""]
        for metric in ["mae_ref_all6", "mae_ref_active"]:
            lines += [f"{LABELS[metric]}，training-seed 原值及变异：", "",
                      "|配置|seed101 / seed202 / seed303|mean|sample SD (ddof=1)|min|max|",
                      "|---|---|---:|---:|---:|---:|"]
            for cid in configs:
                values = " / ".join(f"{float(index[cid, s][metric]):.10f}" if index[cid, s]["status"] == "COMPLETED" else "FAILED" for s in SEEDS)
                v = aggregates[cid]["statistics"][metric]
                stats = f"{v['mean']:.10f}|{v['sample_sd']:.10f}|{v['min']:.10f}|{v['max']:.10f}" if v else "INCOMPLETE|INCOMPLETE|INCOMPLETE|INCOMPLETE"
                lines.append(f"|{cid}|{values}|{stats}|")
            lines.append("")
    lines += ["## 每个唯一训练 run 的 validation 原值与预算", "",
              "|配置|seed|phase / status|validation loss 原值|best / last epoch|optimizer steps|training wall time (s)|",
              "|---|---:|---|---:|---:|---:|---:|"]
    for cid in ALL_CONFIGS:
        for seed in SEEDS:
            r = index[cid, seed]
            lines.append(f"|{cid}|{seed}|{r['phase']} / {r['status']}|{r.get('val_loss','')}|{r.get('best_epoch','')} / {r.get('last_epoch','')}|{r.get('optimizer_steps','')}|{r.get('train_time_s','')}|")
    lines += ["", "Phase B 保存了每个 validation epoch 的值，可确认完成 run 的严格 minimum loss 最早最佳 epoch、last epoch、patience=30 以及 steps=last_epoch×floor(N/512)。FAILED 行的时间是失败前已消耗的 wall time，不是正常返回的训练时间；其观测 epoch/estimated steps 不冒充选定 checkpoint 的训练元数据。Phase A 的过程数来自冻结 CSV/日志，保存的最佳权重本身不包含历史 epoch/optimizer/RNG 元数据。", "",
              "## 单独报告的 fast-sampling variation 和负 proposal", "",
              "|配置|training seed|fast SD all6|fast SD active|含 all6 负分量 proposal / 50,000|含 active 负分量 proposal / 50,000|classification|",
              "|---|---:|---:|---:|---:|---:|---|"]
    for cid in ALL_CONFIGS:
        for seed in SEEDS:
            r, v = index[cid, seed], vindex[cid, seed]
            neg = v.get("sample_observation", {})
            lines.append(f"|{cid}|{seed}|{r.get('fast_seed_sd_all6','')}|{r.get('fast_seed_sd_active','')}|{neg.get('negative_all6_proposals','')}|{neg.get('negative_active_proposals','')}|{v['classification']}|")
    lines += ["", "negative draws 原样返回给原 evaluate 函数并参与 sample mean，不 clamp、不删、不重采样。逐 fast seed/observation 的负 proposal 数、负元素数、最小值、最大值和有限性记录保存在 JSON 的 run_verifications/sample_observation/sample_calls 中。", "",
              "## 原 Table 8/9 单次 MAE 在新分布中的位置", "",
              "旧值按历史 ablation_checkpoint.pt 的 results_a/results_b 分别读取，旧表显示值单列保留，不选择旧值匹配新实验。历史 HMC reference mean 与一次 fast evaluation，和本次共同 NUTS mean/五个 fast-seed 平均有差别；位置比较是描述性的，不能把新旧差别全部归因于 training seed。旧 active MAE 没有保存，不补造。", "",
              "|旧表 / 轴|配置|旧表显示 MAE|历史未舍入 MAE|新 mean ± training SD|新 min–max|历史值位置|历史值距新 mean / SD|显示值位置|显示值距新 mean / SD|",
              "|---|---|---:|---:|---|---|---|---:|---|---:|"]
    positions = {"WITHIN_OBSERVED_RANGE": "在三个 seed 的 min–max 内", "BELOW_OBSERVED_MIN": "低于 min", "ABOVE_OBSERVED_MAX": "高于 max", "INCOMPLETE": "INCOMPLETE"}
    for r in legacy:
        s = r["new_training_seed_distribution"]
        interval = f"{s['min']:.10f}–{s['max']:.10f}" if s else "INCOMPLETE"
        distance = f"{r['signed_distance_in_training_sample_sd']:.4f}" if r["signed_distance_in_training_sample_sd"] is not None else ""
        displayed_distance = f"{r['displayed_signed_distance_in_training_sample_sd']:.4f}" if r["displayed_signed_distance_in_training_sample_sd"] is not None else ""
        lines.append(f"|Table {r['table']} / {r['axis']}|{r['config']}|{r['displayed_mae']:.3f}|{r['historical_unrounded_mae']:.10f}|{stats_text(s)}|{interval}|{positions[r['old_position']]}|{distance}|{positions[r['displayed_old_position']]}|{displayed_distance}|")
    rounding_differences = [r for r in legacy if not r["published_matches_direct_rounding"]]
    if rounding_differences:
        lines += ["", "显示精度核对：" + "；".join(f"Table {r['table']} {r['config']} 显示 {r['displayed_mae']:.3f}，历史值直接舍入三位小数为 {r['direct_rounding_3dp']:.3f}" for r in rounding_differences) + "。两个来源均原样保留，未修改论文或历史值。"]
    lines += ["", "## 原轴内差异与 training-seed variation 的量级", "",
              "差值始终定义为预定轴上后一配置减前一配置，以下描述相邻点的旧 all6 MAE 增减，不作排名。比值分母只使用对应配置的 training-seed sample SD，不使用内部 fast SD。", "",
              "|轴|比较|旧单次差值|旧差值 / 前配置 SD|旧差值 / 后配置 SD|新 mean 差值|新配对差值 sample SD|旧差值在新三组配对差值范围内？|",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in comparisons:
        if r["metric"] != "mae_ref_all6" or not r["adjacent_in_axis"]:
            continue
        if r["status"] != "COMPLETE":
            ratio1 = f"{r['old_difference_divided_by_first_training_sd']:.4f}" if r['old_difference_divided_by_first_training_sd'] is not None else "INCOMPLETE"
            ratio2 = f"{r['old_difference_divided_by_second_training_sd']:.4f}" if r['old_difference_divided_by_second_training_sd'] is not None else "INCOMPLETE"
            lines.append(f"|{r['axis']}|{r['first']} → {r['second']}|{r['original_single_run_difference']:.10f}|{ratio1}|{ratio2}|INCOMPLETE|INCOMPLETE|INCOMPLETE|")
            continue
        lines.append(f"|{r['axis']}|{r['first']} → {r['second']}|{r['original_single_run_difference']:.10f}|{r['old_difference_divided_by_first_training_sd']:.4f}|{r['old_difference_divided_by_second_training_sd']:.4f}|{r['mean_difference_second_minus_first']:.10f}|{r['paired_difference_summary']['sample_sd']:.10f}|{'是' if r['old_difference_within_observed_paired_seed_difference_range'] else '否'}|")
    lines += ["", "涉及 INCOMPLETE 配置时，只保留旧差值以及另一方完整三 seed SD 的可用比值；单方比值不能补足缺失的共同配置分布，也不能建立三 seed 配置可比性。", "",
              "仅按 MAE 差异与 training-seed SD 的描述量级，差异小于或接近 seed SD 的配置宜称为 **statistically/computationally comparable under the tested seeds**。这不是正式显著性或等效性检验，也不声称它们耗时相同；实际 wall time 和 optimizer steps 另列。下列预定顺序的配置对，其均值差不超过至少一方所观察到的 training-seed SD：", "",
              "|轴|指标|配置对|新 mean 差值|第一配置 SD|第二配置 SD|描述|", "|---|---|---|---:|---:|---:|---|"]
    for r in comparisons:
        if r.get("status") == "COMPLETE" and r["mean_difference_no_larger_than_at_least_one_training_sd"]:
            lines.append(f"|{r['axis']}|{LABELS[r['metric']]}|{r['first']} / {r['second']}|{r['mean_difference_second_minus_first']:.10f}|{r['first_training_sample_sd']:.10f}|{r['second_training_sample_sd']:.10f}|{r['descriptive_statement']}|")
    lines += ["", "接近主 SD 量级的具体例子（不另设模型选择或数值阈值）：", "",
              "|轴|指标|配置对|新 mean 差值|较大主 training SD|绝对差值 / 较大 SD|配对差值 SD|描述|",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for r in near_sd_notes:
        lines.append(f"|{r['axis']}|{LABELS[r['metric']]}|{r['first']} / {r['second']}|{r['mean_difference_second_minus_first']:.10f}|{r['larger_primary_training_sample_sd']:.10f}|{r['absolute_mean_difference_over_larger_primary_sd']:.3f}|{r['paired_training_difference_sample_sd']:.10f}|{r['statement']}|")
    lines += ["", "以上措辞仅描述 MAE 差异接近所观察到的 training-seed SD 量级；主误差条仍是每个配置的三 training seed SD，配对差值 SD 单列补充，不与 fast SD 混合。这不证明统计等效、显著性或耗时相同，不能据此排名。", "",
              "全部预定配置对的配对 seed 差值和 SD 均在 JSON 中，未筛选 training seeds。Phase A 的旧 10k、25k 值超过新三 seed 最大值，因此不能把旧完整非单调曲线宣称为完全位于新三 seed 分布内；部分局部增减可与 training variation 量级相容。不同 reference、不同单次 sampling 与三 seed 的小样本限制，使这些观察不足以判定旧趋势成因或认定任何配置最佳。", ""]
    lines += ["## 旧 Table 8 / Table 9 的 50k+8 层不一致来源", "",
              f"旧 Table 8 显示 {printed8[50000]:.3f}，历史 results_a 保存 {old8[50000]:.10f}；旧 Table 9 显示 {printed9[8]:.3f}，历史 results_b 保存 {old9[8]:.10f}。对应 validation loss 与 training time 也不同，详见 JSON 的 old_50k_L8_conflict。", "",
              f"原源码 `{source_path}` 第 {legacy_conflict['data_axis_loop_line']} 行的数据量循环在第 {legacy_conflict['data_axis_new_model_line']} 行新建 flow；第 {legacy_conflict['depth_axis_loop_line']} 行的深度循环又在第 {legacy_conflict['depth_axis_new_model_line']} 行新建 flow 并训练，未复用数据量轴的 8 层模型。两循环没有为同配置重置完全相同的初始化/shuffle/fast-evaluation RNG 状态，前面的训练和 sampling 会消耗 RNG。因此按原代码结构及历史聚合记录，两个旧数字来自两次不同训练，而非同一个已训练模型的共同结果。旧选定模型权重与逐 run RNG 历史未保留，不能独立复算旧训练。", "",
              "新实验只保存一个 N50000_L8/seed，三个 seed 的结果同时引用到两个轴；FULL_ABLATION_SUMMARY 两个共同配置行的全部数值和 checkpoint 引用完全一致。两个旧数均保留作来源说明，不选择其中一个来匹配新结果。", "",
              "## Checkpoint 路径与 SHA256", "",
              f"共同绝对目录：`{OUT}`；basename 与该目录组成完整路径。CSV/JSON 每个 seed 都另存完整绝对路径。", "",
              "|配置|training seed|checkpoint basename|SHA256|", "|---|---:|---|---|"]
    for cid in ALL_CONFIGS:
        for seed in SEEDS:
            v = vindex[cid, seed]
            name = Path(v["checkpoint"]).name if v.get("checkpoint") else "FAILED / no completed checkpoint"
            lines.append(f"|{cid}|{seed}|`{name}`|`{v.get('checkpoint_sha256','')}`|")
    lines += ["", "## FAILED 与数值证据限制", ""]
    if b_failed:
        for r in b_rows:
            if r["status"] == "FAILED":
                lines += [f"{r['config']} seed{r['training_seed']}：{r['error_type']}，原始错误：", "",
                          "```text", r["original_error"], "```", "",
                          f"完整 traceback 原样保留于 `{r['traceback_path']}`；未换 seed、未自动重跑，已有 checkpoint 或部分资产保留。", ""]
    else:
        lines += ["Phase B 没有 FAILED；所有完成 checkpoint 的保留指标均由其选定权重复算得到。", ""]
    lines += ["有限性检查覆盖保存权重、输入、reference、validation 输出和 42–46 下所有 replayed fast samples。Phase B 逐 validation epoch 的有限性也有记录；逐 minibatch loss/gradient 完整轨迹没有保留，因此不能把终态检查夸大为每个历史瞬间无数值问题的证明。高 MAE 本身不等于实现错误；先统一核验所有 checkpoint，再保留其有效性或记录原始错误，不过滤 seed。", "",
              "## Reference 后续补强与停止状态", "",
              "当前统一 3-chain reference 是本阶段暂定评价来源，文件与 Phase A 一致；未运行任何 MCMC。最终稿数值冻结前，另行完成第 4 条独立 chain、rank-normalized split-Rhat、bulk ESS、tail ESS 以及 divergences 检查。补强结果可能改变最终 MAE reference 数字，但与 optimizer、validation、early stopping、checkpoint 选择独立；本阶段不先行补跑、不换 reference。", "",
              "本次新增尝试严格为十五个指定 config/seed，N50000_L8 不重复训练；失败尝试如有也占原指定 run，不换 seed 或补训。保存选定 checkpoint 后进行 fast evaluation 和 checkpoint replay，新增 CSV 使用独立文件并即时原子落盘。Phase A 的 RESULTS.csv 保持九行和原始字节，未合并或重写；全数据由汇总报告引用。", "",
              "已有历史 checkpoint、submitted baseline、论文、response、训练 bank、references 和 Git tracked 内容通过前后 SHA256 核对。没有新 FE、damage grid 重算、调阈值、seed 丢弃或 Git commit/push/reset/clean。新增支持脚本、进度及错误日志位于 phase_b_control，仅用于本次指定运行的执行与审计。", "",
              "全部训练已停止，报告生成后等待审核，不自动启动下一阶段。", ""]
    audit_path = OUT / "FULL_ABLATION_AUDIT.md"
    atomic_bytes(audit_path, "\n".join(lines).encode("utf-8"))
    report["preservation_after"] = verify_frozen_inputs(baseline)
    report["audit_finished_utc"] = utc_now()
    names = ["PHASE_B_RESULTS.csv", "FULL_ABLATION_SUMMARY.csv", "FULL_ABLATION_AUDIT.md", "FULL_ABLATION_INTEGRITY.json"]
    report["outputs"] = {name: dict(path=str(OUT / name), sha256=sha(OUT / name)) for name in names[:-1]}
    report["outputs"][names[-1]] = dict(path=str(OUT / names[-1]), hash_note="Manifest itself; compute its SHA256 externally, no recursive self-hash.")
    report["audit_source"] = dict(path=str(Path(__file__).resolve()), sha256=sha(Path(__file__).resolve()))
    atomic_json(OUT / "FULL_ABLATION_INTEGRITY.json", report)
    written = list(csv.DictReader((OUT / "FULL_ABLATION_SUMMARY.csv").open(encoding="utf-8", newline="")))
    assert len(written) == 9
    shared_readback = [r for r in written if r["config"] == "N50000_L8"]
    assert {k: v for k, v in shared_readback[0].items() if k != "axis"} == {k: v for k, v in shared_readback[1].items() if k != "axis"}
    final = json.loads((OUT / "FULL_ABLATION_INTEGRITY.json").read_text(encoding="utf-8"))
    assert final["scope"]["phase_b_unique_attempts"] == 15 and final["common_N50000_L8"]["identical_data_and_depth_axis_results"]
    assert all(sha(v["path"]) == v["sha256"] for v in final["outputs"].values() if "sha256" in v)
    verify_frozen_inputs(baseline)
    progress.update(updated_utc=utc_now(), awaiting_full_readonly_audit=False, full_audit_complete=True,
                    full_audit_outputs=[str(OUT / name) for name in names])
    atomic_json(CONTROL / "FULL_AUDIT_FINAL_STATUS.json", progress)
    for name in names:
        emit("FINAL_ARTIFACT", str(OUT / name), "sha256", sha(OUT / name))
    emit("FULL_ABLATION_AUDIT_COMPLETE", "training_attempts=15", "training_stopped=True", "waiting_for_review=True")


if __name__ == "__main__":
    main()
