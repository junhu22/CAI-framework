"""Read-only scientific aggregation after the one authorized recovery succeeds.

Creates three new RECOVERED reports; never changes an original artifact and
contains no training, MCMC, FE, damage-grid or configuration-selection call.
"""
import csv
import io
import json
import math
import statistics
import sys
from pathlib import Path

from run_seed202_recovery import (C, OUT, CONTROL, RECOVERY_CONTROL, RESULT, LOG, CHECKPOINT,
                                  CLASSIFICATION, read_csv, verify_previous_assets,
                                  write_new_bytes, write_new_json)

AXES = {"data_size": ["N5000_L8", "N10000_L8", "N25000_L8", "N50000_L8", "N100000_L8"],
        "depth": ["N50000_L4", "N50000_L6", "N50000_L8", "N50000_L12"]}
METRICS = ["mae_ref_all6", "mae_ref_active", "val_loss", "train_time_s"]
LABELS = {"mae_ref_all6": "MAE_reference_all6", "mae_ref_active": "MAE_reference_active_e1_e3",
          "val_loss": "validation_loss", "train_time_s": "training_wall_time_s"}
SUMMARY = OUT / "FULL_ABLATION_SUMMARY_RECOVERED.csv"
AUDIT = OUT / "FULL_ABLATION_AUDIT_RECOVERED.md"
INTEGRITY = OUT / "FULL_ABLATION_INTEGRITY_RECOVERED.json"


def describe(values):
    values = [float(value) for value in values]
    assert len(values) == 3 and all(math.isfinite(value) for value in values)
    array = C.np.asarray(values, dtype=C.np.float64)
    result = dict(mean=float(array.mean()), sample_sd=float(array.std(ddof=1)),
                  min=min(values), max=max(values), n=3, ddof=1)
    assert math.isclose(result["mean"], statistics.mean(values), abs_tol=1e-13)
    assert math.isclose(result["sample_sd"], statistics.stdev(values), abs_tol=1e-13)
    return result


def position(value, distribution):
    if value < distribution["min"]:
        return "BELOW_OBSERVED_MIN"
    if value > distribution["max"]:
        return "ABOVE_OBSERVED_MAX"
    return "WITHIN_OBSERVED_RANGE"


def historical_analysis(old_integrity, aggregates, index):
    old_comparison = old_integrity["table8_table9_comparison"]
    legacy = []
    for retained in old_comparison["legacy_values"]:
        cid = retained["config"]
        current = aggregates[cid]["statistics"]["mae_ref_all6"]
        raw, printed = retained["historical_unrounded_mae"], retained["displayed_mae"]
        legacy.append(dict(axis=retained["axis"], config=cid, table=retained["table"],
                           displayed_mae=printed, historical_unrounded_mae=raw,
                           direct_rounding_3dp=round(raw, 3),
                           published_matches_direct_rounding=round(raw, 3) == printed,
                           new_training_seed_distribution=current,
                           old_position=position(raw, current), displayed_old_position=position(printed, current),
                           signed_distance_in_training_sample_sd=(raw-current["mean"])/current["sample_sd"],
                           displayed_signed_distance_in_training_sample_sd=(printed-current["mean"])/current["sample_sd"]))
    legacy_index = {(row["axis"], row["config"]): row for row in legacy}
    comparisons = []
    for axis, configurations in AXES.items():
        for i, first in enumerate(configurations):
            for j in range(i + 1, len(configurations)):
                second = configurations[j]
                for metric in ["mae_ref_all6", "mae_ref_active"]:
                    s1 = aggregates[first]["statistics"][metric]
                    s2 = aggregates[second]["statistics"][metric]
                    difference = s2["mean"] - s1["mean"]
                    paired = [float(index[second, seed][metric]) - float(index[first, seed][metric])
                              for seed in C.SEEDS]
                    old_difference = None if metric != "mae_ref_all6" else (
                        legacy_index[axis, second]["historical_unrounded_mae"] -
                        legacy_index[axis, first]["historical_unrounded_mae"])
                    overlap = max(s1["min"], s2["min"]) <= min(s1["max"], s2["max"])
                    comparisons.append(dict(axis=axis, first=first, second=second, metric=metric,
                        adjacent_in_axis=j-i == 1, mean_difference_second_minus_first=difference,
                        first_training_sample_sd=s1["sample_sd"], second_training_sample_sd=s2["sample_sd"],
                        absolute_mean_difference_over_first_sd=abs(difference)/s1["sample_sd"],
                        absolute_mean_difference_over_second_sd=abs(difference)/s2["sample_sd"],
                        paired_training_seed_differences=paired, paired_difference_summary=describe(paired),
                        observed_seed_ranges_overlap=overlap,
                        original_single_run_difference=old_difference,
                        old_difference_divided_by_first_training_sd=None if old_difference is None else old_difference/s1["sample_sd"],
                        old_difference_divided_by_second_training_sd=None if old_difference is None else old_difference/s2["sample_sd"],
                        old_difference_within_observed_paired_seed_difference_range=None if old_difference is None else min(paired) <= old_difference <= max(paired),
                        interpretation="Descriptive differences and observed variability only; no formal statistical test or ranking."))
    return dict(paper_path=old_comparison["paper_path"], paper_sha256=old_comparison["paper_sha256"],
                table8_label_line=old_comparison["table8_label_line"], table9_label_line=old_comparison["table9_label_line"],
                legacy_values=legacy, configuration_comparisons=comparisons,
                old_50k_L8_conflict=old_comparison["old_50k_L8_conflict"],
                limits="Historical HMC means and one fast evaluation differ from the common present NUTS means/five-fast-seed average. These differences cannot be assigned solely to training-seed variation. Historical active-e1-e3 MAEs were not retained. No formal statistical test is performed.")


def main():
    started = C.utc_now()
    for path in [SUMMARY, AUDIT, INTEGRITY]:
        assert not path.exists(), "Preserve existing recovered report: " + str(path)
    recovery = json.loads(RESULT.read_text(encoding="utf-8"))
    assert recovery["status"] == "COMPLETED" and recovery["integrity_pass"]
    assert recovery["classification"] == CLASSIFICATION and recovery["training_calls"] == 1
    assert recovery["training_stopped"] and recovery["automatic_retries"] == 0
    assert C.sha(LOG) == recovery["log_sha256"]
    baseline = json.loads((CONTROL / "FROZEN_INPUTS.json").read_text(encoding="utf-8"))
    snapshot_path = RECOVERY_CONTROL / "FROZEN_BEFORE_RECOVERY.json"
    assert C.sha(snapshot_path) == recovery["previous_assets_snapshot_sha256"]
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    before = C.verify_frozen_inputs(baseline)
    previous_before = verify_previous_assets(snapshot)
    old_integrity = json.loads((OUT / "FULL_ABLATION_INTEGRITY.json").read_text(encoding="utf-8"))
    old_checks = {(row["config"], int(row["training_seed"])): row for row in old_integrity["run_verifications"]}
    a = [{**row, "phase": "A", "status": "COMPLETED"} for row in read_csv(OUT / "RESULTS.csv")]
    b = [{**row, "phase": "B"} for row in read_csv(OUT / "PHASE_B_RESULTS.csv")]
    assert len(a) == 9 and len(b) == 15
    original_failed = next(row for row in b if row["config"] == "N50000_L8" and int(row["training_seed"]) == 202)
    assert {key: value for key, value in original_failed.items() if key != "phase"} == recovery["original_failure"]["row"]
    assert original_failed["status"] == "FAILED" and not original_failed["checkpoint"]
    scientific_rows = a + [row for row in b if row["status"] == "COMPLETED"] + [
        {**recovery["scientific_result"], "phase": "B_manual_recovery", "prior_failed_attempt_preserved": True}]
    index = {(row["config"], int(row["training_seed"])): row for row in scientific_rows}
    assert len(index) == 24 and set(index) == {(cid, seed) for cid in C.ALL_CONFIGS for seed in C.SEEDS}
    context = C.load_frozen_context(baseline)
    checks = []
    for cid in C.ALL_CONFIGS:
        for seed in C.SEEDS:
            row = index[cid, seed]
            path = Path(row["checkpoint"])
            expected_path = CHECKPOINT if (cid, seed) == ("N50000_L8", 202) else OUT/f"{cid}_seed{seed}.pt"
            assert path.resolve() == expected_path.resolve()
            check = C.verify_checkpoint(path, cid, seed, row, context)
            expected_sha = row.get("checkpoint_sha256") or old_checks[cid, seed]["checkpoint_sha256"]
            assert check["checkpoint_sha256"] == expected_sha
            assert check["comparisons"]["val_loss"]["abs_difference"] == 0.0
            assert check["fast_seed_max_abs_difference"] == 0.0
            assert int(row["optimizer_steps"]) == int(row["last_epoch"])*(C.ALL_CONFIGS[cid][0]//512)
            assert int(row["last_epoch"]) == 300 or int(row["last_epoch"])-int(row["best_epoch"]) == 30
            check.update(status="COMPLETED", phase=row["phase"],
                         classification=CLASSIFICATION if (cid, seed) == ("N50000_L8", 202)
                         else old_checks[cid, seed]["classification"])
            checks.append(check)
            print("CHECKPOINT_REPLAY_PASS", cid, seed, flush=True)
    check_index = {(row["config"], row["training_seed"]): row for row in checks}
    aggregates = {cid: dict(training_seeds=C.SEEDS, n_completed=3, all_three_complete=True,
                           statistics={metric: describe([index[cid, seed][metric] for seed in C.SEEDS])
                                       for metric in METRICS}) for cid in C.ALL_CONFIGS}
    summary_rows = []
    for axis, configurations in AXES.items():
        for cid in configurations:
            n, layers = C.ALL_CONFIGS[cid]
            output = dict(axis=axis, config=cid, n_train=n, n_layers=layers, training_seed_count=3,
                          completed_training_seed_count=3, sd_ddof=1, all_three_training_seeds_complete=True,
                          common_N50000_L8_result=cid == "N50000_L8")
            for seed in C.SEEDS:
                row, check = index[cid, seed], check_index[cid, seed]
                output[f"seed{seed}_status"] = row["status"]
                output[f"seed{seed}_integrity_classification"] = check["classification"]
                output[f"seed{seed}_prior_failed_attempt_preserved"] = (cid, seed) == ("N50000_L8", 202)
                for metric in METRICS:
                    output[f"seed{seed}_{LABELS[metric]}"] = float(row[metric])
                for field in ["best_epoch", "last_epoch", "optimizer_steps", "checkpoint"]:
                    output[f"seed{seed}_{field}"] = row[field]
                output[f"seed{seed}_checkpoint_sha256"] = check["checkpoint_sha256"]
                for metric in ["all6", "active"]:
                    output[f"seed{seed}_fast_sampling_sample_sd_{metric}"] = float(row[f"fast_seed_sd_{metric}"])
                for fs, mae6, mae3 in json.loads(row["fast_seed_results"]):
                    output[f"seed{seed}_fast_seed{fs}_MAE_reference_all6"] = mae6
                    output[f"seed{seed}_fast_seed{fs}_MAE_reference_active_e1_e3"] = mae3
            for metric in METRICS:
                for stat in ["mean", "sample_sd", "min", "max"]:
                    output[f"{LABELS[metric]}_training_seed_{stat}"] = aggregates[cid]["statistics"][metric][stat]
            summary_rows.append(output)
    shared = [row for row in summary_rows if row["config"] == "N50000_L8"]
    assert len(shared) == 2
    assert {k: v for k, v in shared[0].items() if k != "axis"} == {k: v for k, v in shared[1].items() if k != "axis"}
    history = historical_analysis(old_integrity, aggregates, index)
    negative = dict(all6_proposals=sum(v["sample_observation"]["negative_all6_proposals"] for v in checks),
                    active_proposals=sum(v["sample_observation"]["negative_active_proposals"] for v in checks),
                    all6_elements=sum(v["sample_observation"]["negative_all6_elements"] for v in checks),
                    total_proposals=sum(v["sample_observation"]["total_fast_proposals"] for v in checks),
                    nonfinite_elements=sum(call["nonfinite_elements"] for v in checks
                                           for call in v["sample_observation"]["sample_calls"]),
                    proposals_filtered=0)
    assert negative["nonfinite_elements"] == 0
    # Report construction and output creation follow in the same post-training audit.
    write_reports(started, recovery, baseline, snapshot, before, previous_before, old_integrity,
                  scientific_rows, checks, aggregates, summary_rows, history, negative)


def make_audit(recovery, scientific_rows, checks, aggregates, history, negative, previous_count):
    trajectory = recovery["trajectory_reproduction"]
    check_index = {(row["config"], row["training_seed"]): row for row in checks}
    row_index = {(row["config"], int(row["training_seed"])): row for row in scientific_rows}
    lines = [
        "全部 24/24 科学配置与 training seed 单元已完成；本轮仅人工授权恢复 N50000_L8 seed202 一次，训练已停止，等待审核。",
        "Phase B 首次 15 项尝试仍为 14 COMPLETED + 1 FAILED；第一次 seed202 在 epoch 233 因 telemetry observer 对 PROGRESS.json 原子替换触发 WinError 5 而失败的事实、FAILED 行、原始 traceback、233 条 epoch 记录和旧审计全部保留。",
        "恢复分类：RECOVERED_AFTER_TELEMETRY_IO_FAILURE。恢复训练调用冻结 train()，只挂内存 validation observer，无 progress 文件操作。未发现恢复运行的实现或数值错误；首次 observer I/O 实现错误仍保留在 provenance 中。",
        f"Selected checkpoint strict load、所有 state tensors finite、validation loss 和 fast seeds 42–46 的逐 seed MAE 复算全部通过，replay absolute difference 为 0。原审计容差 atol={C.ATOL}, rtol={C.RTOL}；无 NaN/Inf。",
        f"原 367 项冻结资产以及恢复前 {previous_count} 项既有资产均保持 SHA256 不变，Git tracked 状态保持原样。",
        f"前 233 个 epoch 轨迹核验：max absolute difference={trajectory['max_absolute_difference']!r}；mean absolute difference={trajectory['mean_absolute_difference']!r}；atol=1e-6, rtol=1e-7 全部通过={trajectory['all_233_within_tolerance']}；{trajectory['classification']}。逐 epoch 原值、恢复值和差异保存在 recovery JSON。",
        "科学统计采用恢复成功的 seed202，与原 seed101/303 共同组成唯一的 N50000_L8 三 seed 组。两个轴逐字段共享完全相同的 MAE、validation、时间、checkpoint 路径和 SHA256，没有重复训练。",
        "主要误差条为 training seeds 101/202/303 的 mean ± sample SD (ddof=1)。fast sampling seeds 42–46 的 SD 在每个 training seed 内单独列出，未混入 training-seed SD。",
        "", "**三个 training seeds 的 MAE 分布**", "",
        "|配置|all6 mean ± sample SD|all6 min–max|active e1–e3 mean ± sample SD|active min–max|",
        "|---|---|---|---|---|"
    ]
    for cid in C.ALL_CONFIGS:
        all6 = aggregates[cid]["statistics"]["mae_ref_all6"]
        active = aggregates[cid]["statistics"]["mae_ref_active"]
        lines.append(f"|{cid}|{all6['mean']:.10f} ± {all6['sample_sd']:.10f}|{all6['min']:.10f}–{all6['max']:.10f}|{active['mean']:.10f} ± {active['sample_sd']:.10f}|{active['min']:.10f}–{active['max']:.10f}|")
    lines += ["", "N5000_L8 seed202 继续保留 VALID_HIGH_VARIANCE_RUN 分类，无选择性剔除、替换或重训。",
              "", "**每个 training seed 的原值和训练元数据**", "",
              "|配置|training seed|all6 MAE|active MAE|validation loss|best epoch|last epoch|optimizer steps|wall time (s)|",
              "|---|---|---|---|---|---|---|---|---|"]
    for cid in C.ALL_CONFIGS:
        for seed in C.SEEDS:
            row = row_index[cid, seed]
            lines.append(f"|{cid}|{seed}|{float(row['mae_ref_all6']):.10f}|{float(row['mae_ref_active']):.10f}|{float(row['val_loss']):.15g}|{row['best_epoch']}|{row['last_epoch']}|{row['optimizer_steps']}|{float(row['train_time_s']):.6f}|")
    lines += ["", "训练 wall time 包含相应 observer 的开销；首次失败耗时单独保留，不加到恢复成功运行的 wall time。历次 CPU 调度与进度 I/O 开销不同，时间仅作描述，不据此排序。所有数值的完整精度保存在 CSV/JSON。",
              "", "**training seed 内的 fast-sampling variation**", "",
              "|配置|training seed|fast-seed sample SD all6|fast-seed sample SD active|negative all6 proposals|negative active proposals|",
              "|---|---|---|---|---|---|"]
    for cid in C.ALL_CONFIGS:
        for seed in C.SEEDS:
            row, check = row_index[cid, seed], check_index[cid, seed]
            observed = check["sample_observation"]
            lines.append(f"|{cid}|{seed}|{float(row['fast_seed_sd_all6']):.10f}|{float(row['fast_seed_sd_active']):.10f}|{observed['negative_all6_proposals']}|{observed['negative_active_proposals']}|")
    lines += ["", f"每次 fast evaluation N=2000，对五个 matched observations 使用 fast seeds 42–46。合计 {negative['total_proposals']} 个 proposal draws，其中 {negative['all6_proposals']} 个含 negative all6 分量、{negative['active_proposals']} 个含 negative active 分量。全部原样参与现有 fast-mode 评价，proposals_filtered=0；negative draws 不是 I/O 错误，也未被误判为 NaN/Inf。每个 fast seed 的两类 MAE 保存在新 summary CSV 和 integrity JSON。",
              "", "**旧 Table 8/9 的单次值在当前三 seed 分布中的位置**", "",
              "|轴/配置|旧表显示值|旧归档未舍入值|未舍入旧值的位置|距离当前 mean / training-seed SD|",
              "|---|---|---|---|---|"]
    for old in history["legacy_values"]:
        lines.append(f"|{old['axis']}/{old['config']}|{old['displayed_mae']:.3f}|{old['historical_unrounded_mae']:.12f}|{old['old_position']}|{old['signed_distance_in_training_sample_sd']:.4f}|")
    lines += ["", "上述位置为观测范围中的描述，不是置信区间或正式检验。Table 8 的 25k 显示 0.043，归档原值 0.04249884933233261 直接保留三位小数为 0.042；两者均保留，未据此修改历史值。旧 active-e1-e3 数字未保存，未补造。旧 HMC reference / 一次 fast evaluation 与当前 matched NUTS reference / 五个 fast seed 均值不同，因此新旧差异不能全部归因于 training-seed variation。",
              "", "**按预定轴顺序的相邻差异与 seed variation**", "",
              "|轴/相邻配置|指标|新 mean 差 (后−前)|前 SD|后 SD|paired seed 差的 SD|观测 seed 范围重叠|",
              "|---|---|---|---|---|---|---|"]
    for comparison in history["configuration_comparisons"]:
        if comparison["adjacent_in_axis"]:
            lines.append(f"|{comparison['axis']}/{comparison['first']}→{comparison['second']}|{LABELS[comparison['metric']]}|{comparison['mean_difference_second_minus_first']:+.10f}|{comparison['first_training_sample_sd']:.10f}|{comparison['second_training_sample_sd']:.10f}|{comparison['paired_difference_summary']['sample_sd']:.10f}|{comparison['observed_seed_ranges_overlap']}|")
    lines += ["", "**旧单次相邻差异的量级**", "",
              "|轴/相邻配置|旧 MAE 差 (后−前)|旧差 / 前 training SD|旧差 / 后 training SD|旧差落入当前 paired seed 差范围|",
              "|---|---|---|---|---|"]
    for comparison in history["configuration_comparisons"]:
        if comparison["adjacent_in_axis"] and comparison["metric"] == "mae_ref_all6":
            lines.append(f"|{comparison['axis']}/{comparison['first']}→{comparison['second']}|{comparison['original_single_run_difference']:+.10f}|{comparison['old_difference_divided_by_first_training_sd']:+.4f}|{comparison['old_difference_divided_by_second_training_sd']:+.4f}|{comparison['old_difference_within_observed_paired_seed_difference_range']}|")
    return lines


def write_reports(started, recovery, baseline, snapshot, before, previous_before, old_integrity,
                  scientific_rows, checks, aggregates, summary_rows, history, negative):
    lines = make_audit(recovery, scientific_rows, checks, aggregates, history, negative, len(snapshot["files"]))
    overlap_names = [f"{v['first']} / {v['second']} ({LABELS[v['metric']]})"
                     for v in history["configuration_comparisons"]
                     if v["adjacent_in_axis"] and v["observed_seed_ranges_overlap"]]
    lines += ["", "相邻配置的实际 seed 范围重叠项目：" + ("；".join(overlap_names) if overlap_names else "无") + "。范围是否重叠仅按三个已观测原值计算。",
              "旧 5k→10k→25k 的非单调上升幅度为 0.0018441118 和 0.0018913448，量级不超过相应已观测 training-seed SD；当前同 seed 的 paired 差范围位置另见上表，幅度相近不等于复现了旧趋势。旧深度轴相邻差异与 seed SD 的比例逐项给出，不能把所有旧差异概括为来自 seed variation。",
              "5k 和 100k 的 all6 已观测 seed 范围不重叠；三个对应 seed 的 5k MAE 均高于 100k。5k seed202 的较大 SD 不能据此建立两者的统计比较结论。",
              "所有上述比较都是三个固定 seeds 下的描述，没有正式统计检验。对于相邻配置的小幅差异，the three-seed results do not support fine-grained ranking；数值未按结果排序、未选择最接近旧稿的 seed。",
              "", "**旧 50k+8层的两个值**", "",
              "Table 8 显示 0.021，归档 0.021053802222013474；Table 9 显示 0.018，归档 0.01813913881778717。旧 ablation_study.py 数据量循环 (line 320，new model line 329) 与深度循环 (line 360，new model line 369) 分别创建、训练并评价新的 flow；深度循环未复用数据量轴的 L8 flow，且两个循环没有针对每个配置重置相同的初始化/shuffle/sampling RNG。因此旧值来自两个独立训练调用，而不是同一共享 checkpoint 的两种显示。两次 validation loss/time 也不同，原证据保存在 integrity JSON。历史 selected states 与确切逐 run RNG 轨迹未保留，无法独立 checkpoint replay 旧值。当前两个轴使用同一组三 seed，未选择任一个旧值来匹配。",
              "", "**冻结训练规则与执行边界**", "",
              "复用同一 ablation_checkpoint.pt nested training bank；train prefix [0,N)，validation 固定 [100000,105000)；hidden_dim=128，max_epochs=300，batch_size=512，drop_last=True；Adam lr=1e-3、weight_decay=1e-5；CosineAnnealingLR T_max=300；gradient clip=5；early stopping patience=30；按 strict minimum validation loss 选择 checkpoint。未使用 reference/test MAE 选择 checkpoint、seed 或配置。训练 seeds 101/202/303、fast seeds 42–46 和 Nfast=2000 保持原样。",
              "本轮训练调用严格为一次人工授权恢复 N50000_L8 seed202。原 run_ablation_multiseed.py 的顶层循环未执行，源文件未改动。内存 observer 不写文件、rename 文件、打印进度或访问网络；恢复 epoch log 在训练结束后一次写入。",
              "未运行额外 training seed、第4条 reference chain、新 MCMC、FE 或 damage grid；未调阈值、修改论文/response/submitted baseline/bank、修改旧 Phase A/B 结果，亦未执行 git commit/push/reset/clean。",
              "", "**Reference 质量后续独立工作**", "",
              "当前五个冻结 matched NUTS references 仍各为 3 chains，统一用于评价。最终稿数值冻结前还需单独完成第4条独立 chain、rank-normalized split-Rhat、bulk ESS、tail ESS、divergences。此质量补强未在本轮执行，reference 不参与训练或 checkpoint 选择。",
              "", "Checkpoint 路径及 SHA256、各 fast seed 原值、全部描述性比较、冻结资产清单与首次失败 provenance 见同目录新 summary CSV、integrity JSON 和 RECOVERY_N50000_L8_seed202.json。恢复完成后已停止训练，等待审核。", ""]
    # Guard the explicit wording boundary in new outputs; frozen historical text is not rewritten.
    forbidden = ["best configuration", "statistically equivalent", "statistically comparable", "significant improvement"]
    text = "\n".join(lines)
    assert not any(phrase in text.lower() for phrase in forbidden)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(summary_rows[0]))
    writer.writeheader()
    writer.writerows(summary_rows)
    after = C.verify_frozen_inputs(baseline)
    previous_after = verify_previous_assets(snapshot)
    integrity = dict(schema_version=2, audit_started_utc=started, audit_finished_utc=C.utc_now(),
        scope=dict(phase_a_frozen_completed=9, original_phase_b_attempts=15,
                   original_phase_b_completed=14, original_phase_b_failed=1,
                   manually_authorized_recovery_training_calls=1, recovery_completed=1,
                   scientific_unique_cells_expected=24, scientific_unique_cells_completed=24,
                   additional_training_seeds=0, seed_substitutions=0, automatic_retries=0),
        training_stopped=True, awaiting_review=True,
        original_failure_provenance=recovery["original_failure"],
        recovery=dict(result_path=str(RESULT), result_sha256=C.sha(RESULT), log_path=str(LOG),
                      log_sha256=C.sha(LOG), checkpoint=str(CHECKPOINT), checkpoint_sha256=C.sha(CHECKPOINT),
                      classification=CLASSIFICATION, trajectory_reproduction=recovery["trajectory_reproduction"],
                      frozen_execution=recovery["frozen_execution"], training_rules=recovery["training_rules"]),
        implementation_audit=dict(original_telemetry_io_error_retained=True,
                                  original_training_and_evaluation_definitions_modified=False,
                                  recovery_implementation_errors_found=False,
                                  recovery_nonfinite_validation_epochs=0,
                                  training_progress_file_operations=0),
        numerical_tolerance=dict(atol=C.ATOL, rtol=C.RTOL, equal_nan=False),
        completed_checkpoint_integrity_all_pass=True, completed_checkpoint_replay_exact=True,
        scientific_results=scientific_rows, run_verifications=checks, aggregates=aggregates,
        aggregation=dict(primary_error_bar="training-seed sample SD", ddof=1, training_seeds=C.SEEDS,
                         fast_sampling_seeds=C.FAST_SEEDS, fast_seed_variability_mixed_with_training=False,
                         formal_statistical_tests_performed=False, sorting_or_ranking_by_results=False),
        common_N50000_L8=dict(shared_between_both_axes=True, training_seeds=C.SEEDS,
                             summary_rows_equal_except_axis=True, unique_checkpoints=3),
        negative_proposals=negative, table8_table9_comparison=history, reference=old_integrity["reference"],
        training_rules=old_integrity["training_rules"], preservation_before=before, preservation_after=after,
        protected_files=baseline["files"], git_tracked_status_before=snapshot["git_tracked_status"],
        git_tracked_status_after=C.git_read("status", "--porcelain", "-uno"),
        previous_assets_snapshot=dict(path=str(RECOVERY_CONTROL/"FROZEN_BEFORE_RECOVERY.json"),
                                      sha256=C.sha(RECOVERY_CONTROL/"FROZEN_BEFORE_RECOVERY.json"),
                                      preservation_before=previous_before, preservation_after=previous_after,
                                      files=snapshot["files"]),
        execution_boundaries=dict(new_mcmc_runs=0, fourth_reference_chain_runs=0, new_fe_runs=0,
                                  damage_grid_recomputations=0, manuscript_edits=0,
                                  phase_a_retraining_calls=0, historical_result_edits=0,
                                  git_mutations=0, discarded_training_seeds=0),
        audit_source=dict(path=str(Path(__file__).resolve()), sha256=C.sha(__file__)))
    assert not any(phrase in json.dumps(integrity, ensure_ascii=False).lower() for phrase in forbidden)
    write_new_bytes(SUMMARY, buffer.getvalue().encode("utf-8"))
    write_new_bytes(AUDIT, text.encode("utf-8"))
    integrity["outputs"] = {path.name: dict(path=str(path), sha256=C.sha(path)) for path in [SUMMARY, AUDIT]}
    write_new_json(INTEGRITY, integrity)
    print("RECOVERED_REPORTS_CREATED", str(SUMMARY), str(AUDIT), str(INTEGRITY), flush=True)


if __name__ == "__main__":
    main()
