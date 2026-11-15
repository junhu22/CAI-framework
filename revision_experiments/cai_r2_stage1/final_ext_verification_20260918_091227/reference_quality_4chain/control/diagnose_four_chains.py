"""ArviZ 0.23.4 diagnostics and immutable reference-quality reports.

Run in psis_env. No Torch/Pyro imports, sampler, flow draws or training.
Historical divergence information that was not retained is never inferred from
posterior samples or substituted by the erroneous legacy logger's fallback 0.
"""
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import arviz as az
import numpy as np

ROOT = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
HERE = ROOT/"revision_experiments/cai_r2_stage1/final_ext_verification_20260918_091227"
OUT = HERE/"reference_quality_4chain"
CONTROL = OUT/"control"
PARAMETERS = [f"e{i}" for i in range(1, 7)]
QUANTILES = [.025, .25, .50, .75, .975]
QUANTILE_LABELS = ["q025", "q25", "q50", "q75", "q975"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state():
    return subprocess.check_output(["git", "status", "--porcelain", "-uno"], cwd=ROOT,
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).decode("utf-8")


def write_new_bytes(path, value):
    with Path(path).open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def write_new_json(path, value):
    write_new_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8"))


def write_new_csv(path, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
    writer.writeheader()
    writer.writerows(rows)
    write_new_bytes(path, buffer.getvalue().encode("utf-8"))


def verify_preservation(snapshot):
    changed = [path for path, digest in snapshot["files"].items()
               if not Path(path).is_file() or sha(path) != digest]
    assert not changed, changed
    assert git_state() == snapshot["git_tracked_before"]
    return dict(checked_utc=utc_now(), checked_files=len(snapshot["files"]),
                all_sha256_unchanged=True, changed_files=[], git_tracked_unchanged=True)


def arviz_diagnostics(chains):
    assert chains.ndim == 3 and chains.shape[1:] == (600, 6)
    data = az.from_dict(posterior={"theta": chains}, coords={"parameter": PARAMETERS},
                        dims={"theta": ["parameter"]})
    rhat = np.asarray(az.rhat(data, method="rank")["theta"].values, dtype=np.float64)
    bulk = np.asarray(az.ess(data, method="bulk", relative=False)["theta"].values, dtype=np.float64)
    tail = np.asarray(az.ess(data, method="tail", relative=False)["theta"].values, dtype=np.float64)
    assert rhat.shape == bulk.shape == tail.shape == (6,)
    clean = lambda array: [float(value) if math.isfinite(value) else None for value in array]
    return dict(rank_normalized_split_rhat=clean(rhat), bulk_ess=clean(bulk), tail_ess=clean(tail),
                max_rhat=float(rhat.max()) if np.isfinite(rhat).all() else None,
                min_bulk_ess=float(bulk.min()) if np.isfinite(bulk).all() else None,
                min_tail_ess=float(tail.min()) if np.isfinite(tail).all() else None,
                all_finite=bool(np.isfinite(rhat).all() and np.isfinite(bulk).all() and np.isfinite(tail).all()))


def main():
    assert az.__version__ == "0.23.4", "Use the existing pinned diagnostic implementation."
    for name in ["REFERENCE_QUALITY_SUMMARY.csv", "REFERENCE_QUALITY_AUDIT.md", "REFERENCE_QUALITY_INTEGRITY.json"]:
        assert not (OUT/name).exists(), "Never overwrite an existing audit."
    finished = json.loads((CONTROL/"FOURTH_CHAINS_FINISHED.json").read_text(encoding="utf-8"))
    assert finished["authorized_chain_attempts"] == 7 and finished["mcmc_stopped"]
    manifest = json.loads((CONTROL/"TARGET_MANIFEST.json").read_text(encoding="utf-8"))
    snapshot = json.loads((CONTROL/"FROZEN_BEFORE_REFERENCE_QUALITY.json").read_text(encoding="utf-8"))
    before = verify_preservation(snapshot)
    legacy_source = HERE/"run_reference_nuts.py"
    text = legacy_source.read_text(encoding="utf-8")
    assert "diag.get('divergences',{}).get('z',[])" in text
    logger_line = next(i for i, line in enumerate(text.splitlines(), 1)
                       if "diag.get('divergences',{}).get('z',[])" in line)
    original_settings = json.loads((HERE/"REFERENCE_RUN_SUMMARY.json").read_text(encoding="utf-8"))["settings"]
    assert original_settings["samples_per_chain"] == 600 and original_settings["warmup"] == 500
    assert original_settings["seeds"] == [701, 702, 703]
    evidence_issue = dict(status="UNVERIFIABLE_HISTORICAL_DIVERGENCES", source_path=str(legacy_source),
                         source_sha256=sha(legacy_source), incorrect_lookup_line=logger_line,
                         incorrect_key="z", actual_installed_pyro_chain_key="chain 0",
                         historical_npz_diagnostics_saved=False,
                         historical_log=str(HERE/"reference_nuts.log"), historical_log_sha256=sha(HERE/"reference_nuts.log"),
                         legacy_zero_is_missing_key_fallback_not_verified_count=True,
                         original_chain_divergence_counts=[None, None, None],
                         remedy_not_executed="Original chains were not rerun and no extra chain was added.")
    results, csv_rows = [], []
    for target in manifest["targets"]:
        name = target["target"]
        run_path = OUT/f"REFERENCE4_{name}_RUN.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        assert run["target_manifest"] == target
        assert sha(target["original_reference_path"]) == target["original_reference_sha256"]
        assert sha(target["source_surrogate_path"]) == target["source_surrogate_sha256"]
        with np.load(target["original_reference_path"], allow_pickle=False) as original:
            assert not any("diverg" in key.lower() or "diagnostic" in key.lower() for key in original.files)
            three = original["chains"].copy()
            old_mean = original["ref_mean"].astype(np.float64)
            old_quantiles = original["ref_quantiles"].astype(np.float64)
        row = dict(target=name, fourth_chain_status=run["status"], seed=704, chains=4 if run["status"] == "COMPLETED" else 3,
                   draws_per_chain=600, warmup_steps=500, target_accept_prob=0.8,
                   max_rank_normalized_split_rhat=None, min_bulk_ess=None, min_tail_ess=None,
                   fourth_chain_divergences=run.get("fourth_chain_divergence_count"),
                   original_3chain_divergences=None, all_4chain_divergences=None,
                   chains_gate="FAIL", rhat_gate="FAIL", bulk_ess_gate="FAIL", tail_ess_gate="FAIL",
                   divergences_gate="FAIL", overall_quality_gate="FAIL",
                   divergences_evidence_status="UNVERIFIABLE_HISTORICAL_DIVERGENCES",
                   three_to_four_mean_mae_all6=None, three_to_four_mean_mae_active_e1_e3=None,
                   three_to_four_quantile_max_absolute_change=None, runtime_s=run.get("runtime_s"),
                   reference4_path=run.get("artifact_path"), reference4_sha256=run.get("artifact_sha256"),
                   original_reference_sha256=target["original_reference_sha256"],
                   source_surrogate_sha256=target["source_surrogate_sha256"],
                   r2_1_conclusion_change="NOT_ASSESSED_QUALITY_GATE_FAILED",
                   r2_3_conclusion_change="NOT_ASSESSED_QUALITY_GATE_FAILED")
        result = dict(target=name, run_path=str(run_path), run_sha256=sha(run_path), target_manifest=target,
                      fourth_chain_status=run["status"], historical_divergence_evidence=evidence_issue,
                      all_four_chain_divergence_count=None, fourth_chain_divergence_count=run.get("fourth_chain_divergence_count"))
        if run["status"] == "COMPLETED":
            assert sha(run["artifact_path"]) == run["artifact_sha256"]
            with np.load(run["artifact_path"], allow_pickle=False) as saved:
                chains = saved["chains"].copy()
                assert chains.shape == (4, 600, 6)
                assert np.array_equal(chains[:3], three)
                assert np.array_equal(chains[3], saved["theta_samples"])
                assert np.array_equal(saved["theta_init"], np.exp(np.linspace(.20, -.20, 6)))
                assert saved["seed"].item() == 704
                assert len(saved["divergences"]) == run["fourth_chain_divergence_count"]
            diagnostics = arviz_diagnostics(chains)
            old_diagnostics = arviz_diagnostics(three)
            pooled = chains.reshape(-1, 6).astype(np.float64)
            new_mean = pooled.mean(axis=0)
            new_quantiles = np.quantile(pooled, QUANTILES, axis=0)
            mean_change = new_mean-old_mean
            quantile_change = new_quantiles-old_quantiles
            gates = dict(chains="PASS", rank_normalized_split_rhat="PASS" if diagnostics["max_rhat"] is not None and diagnostics["max_rhat"] <= 1.01 else "FAIL",
                         bulk_ess="PASS" if diagnostics["min_bulk_ess"] is not None and diagnostics["min_bulk_ess"] >= 400 else "FAIL",
                         tail_ess="PASS" if diagnostics["min_tail_ess"] is not None and diagnostics["min_tail_ess"] >= 400 else "FAIL",
                         divergences="FAIL")
            row.update(max_rank_normalized_split_rhat=diagnostics["max_rhat"], min_bulk_ess=diagnostics["min_bulk_ess"],
                       min_tail_ess=diagnostics["min_tail_ess"], chains_gate=gates["chains"],
                       rhat_gate=gates["rank_normalized_split_rhat"], bulk_ess_gate=gates["bulk_ess"],
                       tail_ess_gate=gates["tail_ess"],
                       three_to_four_mean_mae_all6=float(np.abs(mean_change).mean()),
                       three_to_four_mean_mae_active_e1_e3=float(np.abs(mean_change[:3]).mean()),
                       three_to_four_quantile_max_absolute_change=float(np.abs(quantile_change).max()))
            result.update(diagnostics_4chain=diagnostics, diagnostics_3chain_same_arviz_methods=old_diagnostics,
                          gates=gates, overall_quality_gate="FAIL",
                          fail_reasons=[key for key, value in gates.items() if value == "FAIL"],
                          three_to_four=dict(original_saved_3chain_mean=old_mean.tolist(),
                                             candidate_4chain_mean=new_mean.tolist(), signed_mean_change=mean_change.tolist(),
                                             mean_mae_all6=float(np.abs(mean_change).mean()),
                                             mean_mae_active_e1_e3=float(np.abs(mean_change[:3]).mean()),
                                             quantile_probabilities=QUANTILES, original_3chain_quantiles=old_quantiles.tolist(),
                                             candidate_4chain_quantiles=new_quantiles.tolist(), signed_quantile_change=quantile_change.tolist(),
                                             max_absolute_quantile_change=float(np.abs(quantile_change).max()),
                                             summary_for_diagnostic_stability_only_not_adopted_reference=True,
                                             candidate_4chain_mean_accumulation_dtype="float64"))
            for j, parameter in enumerate(PARAMETERS):
                row[f"{parameter}_rank_normalized_split_rhat"] = diagnostics["rank_normalized_split_rhat"][j]
                row[f"{parameter}_bulk_ess"] = diagnostics["bulk_ess"][j]
                row[f"{parameter}_tail_ess"] = diagnostics["tail_ess"][j]
                row[f"{parameter}_mean_3chain"] = float(old_mean[j])
                row[f"{parameter}_mean_4chain_candidate"] = float(new_mean[j])
                row[f"{parameter}_mean_change"] = float(mean_change[j])
                for q_index, label in enumerate(QUANTILE_LABELS):
                    row[f"{parameter}_{label}_3chain"] = float(old_quantiles[q_index, j])
                    row[f"{parameter}_{label}_4chain_candidate"] = float(new_quantiles[q_index, j])
                    row[f"{parameter}_{label}_change"] = float(quantile_change[q_index, j])
        results.append(result)
        csv_rows.append(row)
        print("ARVIZ_TARGET_DIAGNOSED", name, row["rhat_gate"], row["bulk_ess_gate"], row["tail_ess_gate"], "DIVERGENCE_EVIDENCE_FAIL", flush=True)
    write_quality_reports(results, csv_rows, manifest, snapshot, before, evidence_issue)


def write_quality_reports(results, rows, manifest, snapshot, before, evidence_issue):
    completed = sum(row["fourth_chain_status"] == "COMPLETED" for row in rows)
    numerical = [result["diagnostics_4chain"] for result in results if "diagnostics_4chain" in result]
    finite = all(result["all_finite"] for result in numerical) and len(numerical) == 7
    max_rhat = max(result["max_rhat"] for result in numerical if result["max_rhat"] is not None)
    min_bulk = min(result["min_bulk_ess"] for result in numerical if result["min_bulk_ess"] is not None)
    min_tail = min(result["min_tail_ess"] for result in numerical if result["min_tail_ess"] is not None)
    known_divergences = sum(row["fourth_chain_divergences"] or 0 for row in rows)
    passed = sum(row["overall_quality_gate"] == "PASS" for row in rows)
    all_pass = completed == 7 and passed == 7
    # The original chains' counts cannot be recovered from the saved theta draws.
    # Unknown historical counts mean this execution cannot establish the full gate.
    assert not all_pass
    candidate_summary = dict(fourth_chains_completed=completed, fourth_chains_expected=7,
                             all_7_fourth_chains_completed=completed == 7,
                             full_quality_gate_passed_targets=passed, all_7_quality_gates_pass=False,
                             max_rank_normalized_split_rhat=max_rhat, min_bulk_ess=min_bulk, min_tail_ess=min_tail,
                             all_rank_ess_diagnostics_finite=finite,
                             fourth_chain_divergences_total=known_divergences,
                             original_21_chains_divergences_total=None, all_28_chains_divergences_total=None,
                             r2_1_major_conclusion_change="NOT_ASSESSED_QUALITY_GATE_FAILED",
                             r2_3_major_conclusion_change="NOT_ASSESSED_QUALITY_GATE_FAILED",
                             conditional_reevaluations_executed=False, stopped_at_review=True)
    lines = [
        f"1. 7/7 第4链是否成功：{completed}/7；每个 target 仅新增 seed704 一条链。",
        f"2. 7/7 是否全部通过预定 quality gate：否；{passed}/7 完整 PASS。",
        f"3. 最大 rank-normalized split-Rhat：{max_rhat:.12f}。",
        f"4. 最小 bulk ESS：{min_bulk:.12f}。",
        f"5. 最小 tail ESS：{min_tail:.12f}。",
        f"6. Divergences 总数：新增7条第4链已记录 {known_divergences}；原21条链未保存可核验计数，因此全部28条链总数 UNKNOWN，不能报告为0。",
        "7. 3-chain→4-chain 是否改变 R2.1/R2.3 主要结论：NOT_ASSESSED_QUALITY_GATE_FAILED。完整 gate 未通过，两类只读重评价及统一采用4-chain reference 的步骤均未执行；不能据此声称结论已改变或未改变。",
        "", "训练调用数=0；MCMC 已停止，全部历史 reference、checkpoint、结果和审计保留，等待审核。",
        "", "**每个 target 的4-chain质量诊断与逐项 gate**", "",
        "|target|chains|rank split-Rhat max|bulk ESS min|tail ESS min|第4链 divergences|四链总 divergences|chains gate|Rhat gate|bulk gate|tail gate|divergences gate|overall|",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        numeric = lambda value: "UNAVAILABLE" if value is None else f"{value:.9f}"
        lines.append(f"|{row['target']}|{row['chains']}|{numeric(row['max_rank_normalized_split_rhat'])}|{numeric(row['min_bulk_ess'])}|{numeric(row['min_tail_ess'])}|{row['fourth_chain_divergences']}|UNKNOWN|{row['chains_gate']}|{row['rhat_gate']}|{row['bulk_ess_gate']}|{row['tail_ess_gate']}|{row['divergences_gate']}|{row['overall_quality_gate']}|")
    lines += ["", "门槛预先固定：chains≥4、max rank-normalized split-Rhat≤1.01、min bulk ESS≥400、min tail ESS≥400、divergences=0。所有六个参数均参与 gate，未用 active-only 诊断替代 all6。CSV 按 e1–e6 保存逐参数 rank split-Rhat、bulk ESS、tail ESS 原值。",
              "", "**历史 divergence 记录缺口**", "",
              f"冻结 run_reference_nuts.py line {evidence_issue['incorrect_lookup_line']} 以 diag.get('divergences',{{}}).get('z',[]) 取计数，acceptance rate 同样使用了 z 键。实际 Pyro 1.9.1 MCMC.diagnostics() 按 chain 0 返回 sampler diagnostics；z 键用于另一组参数 ESS/Rhat 诊断。旧日志的 div 0 是缺失键的默认空列表长度，且 acc=None，与该读取错误一致。七个旧 NPZ 均只保存 chains/fast/means/quantiles 等数组，没有保留原始 sampler diagnostics 或 divergence indices。原 summary 也未保存这些计数。",
              "因此每个 target 的 divergences gate 标记 FAIL，具体原因是无法核验历史三链的计数，不是断言历史链出现了 divergence。全部新增第4链的真实 diagnostics 和 divergence indices 已独立保存。未把未知记录当0，未重跑原三链或增加第5链来补证据。",
              "", "**3-chain 与4-chain候选均值/分位数的变化**", "",
              "|target|mean MAE shift all6|mean MAE shift active e1–e3|max absolute quantile shift|",
              "|---|---|---|---|"]
    for row in rows:
        numeric = lambda value: "UNAVAILABLE" if value is None else f"{value:.12f}"
        lines.append(f"|{row['target']}|{numeric(row['three_to_four_mean_mae_all6'])}|{numeric(row['three_to_four_mean_mae_active_e1_e3'])}|{numeric(row['three_to_four_quantile_max_absolute_change'])}|")
    lines += ["", "以上 MAE 是 reference mean 向量之间的平均绝对变化，并非 flow posterior mean gap。3-chain mean 使用原 NPZ 的保存原值；4-chain 候选 mean 对保留的2400 draws以float64累计。分位数固定为2.5/25/50/75/97.5%，采用 NumPy linear quantile；逐参数3-chain值、4-chain候选值及有符号差保存在 quality CSV/JSON。这些数值仅用于诊断 reference 稳定性，不作为通过 gate 后的统一新 reference。",
              "", "**冻结 target 和 sampler**", "",
              "ablation_obs1–5：original D6 surrogate_models/member_0.pt，K=20，exact saved observation，independent untruncated lognormal prior，sigma_log=0.30，kappa_sq=0.001。field：final ext surrogate_models_broad_ext/member_0.pt，K=6，exact frozen field observation，sigma_log=0.30，kappa_sq=0.001。boundary：同一 final ext surrogate，K=20，exact saved observation，sigma_log=0.55，kappa_sq=0.001。所有 target 采用 log-theta 参数化，冻结 run_chain() 的 likelihood/prior potential 未修改。",
              "所有新增链预先固定 seed=704、num_samples=600、warmup_steps=500、target_accept_prob=0.8、theta_init=exp(linspace(0.20,-0.20,6))。全部 init 在任何采样前写入 target manifest，未根据 posterior 或运行结果修改。原 runner 顶层未执行；只提取并调用冻结 run_chain()，没有 flow training、flow resampling、FE、damage grid、阈值/超参数调整、论文/response修改或 Git commit/push/reset/clean。",
              "成熟诊断实现：现有 psis_env 中 ArviZ 0.23.4；az.rhat(method='rank')、az.ess(method='bulk')、az.ess(method='tail', relative=False)，输入 theta chains shape=(4,600,6)，chain/draw/parameter 维度显式保留。未手写诊断近似，也未用旧 ordinary split-Rhat/per-chain ESS 代替要求的指标。",
              "", "**条件重评价的审核点**", "",
              "完整质量 gate 尚未全部通过，因此没有生成 REFERENCE_4CHAIN_SUMMARY.csv、ROUTING_VS_GAP_4CHAIN.csv/AUDIT 或 FULL_ABLATION_4CHAIN_REFERENCE.csv/AUDIT，没有为24个flow checkpoint重新抽样，也没有重计算任何 raw/PSIS routing 指标。R2.1/R2.3 的 mean/SD/trend/conclusion 变化未评价，历史文件继续冻结。新四链样本原样留待审核，未在三链/四链之间选择更有利的一套。", ""]
    after = verify_preservation(snapshot)
    lines += [f"完整性复核：{after['checked_files']} 项历史资产 SHA256 全部不变，其中包含原367项冻结资产、原七个3-chain NPZ、全部24个科学flow checkpoint、恢复产物和全部历史审计。Git tracked 状态与执行前完全一致。", ""]
    schema_path = CONTROL/"PYRO_DIAGNOSTIC_SCHEMA_EVIDENCE.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["pyro_version"] == "1.9.1" and "chain {}" in schema["MCMC_diagnostics_source"]
    assert sha(schema["source_path"]) == schema["source_sha256"]
    integrity = dict(schema_version=1, created_utc=utc_now(), summary=candidate_summary,
                     predeclared_quality_gate=dict(chains_min=4, max_rank_normalized_split_rhat=1.01,
                                                  min_bulk_ess=400, min_tail_ess=400, divergences=0),
                     diagnostics_implementation=dict(name="ArviZ", version=az.__version__,
                         python_executable=sys.executable, arviz_module_path=az.__file__,
                         arviz_module_sha256=sha(az.__file__), rhat_method="rank", bulk_ess_method="bulk",
                         tail_ess_method="tail", relative_ess=False, diagnostic_coordinate="theta",
                         input_dimensions=["chain", "draw", "parameter"], input_shape=[4,600,6],
                         hand_written_diagnostic_approximations=False),
                     target_manifest=dict(path=str(CONTROL/"TARGET_MANIFEST.json"),
                                          sha256=sha(CONTROL/"TARGET_MANIFEST.json"), content=manifest),
                     targets=results, historical_divergence_evidence_issue=evidence_issue,
                     pyro_diagnostics_schema_evidence=dict(path=str(schema_path), sha256=sha(schema_path)),
                     original_367_protected_files=snapshot["original_367_files"],
                     protected_historical_files=snapshot["files"], preservation_before=before, preservation_after=after,
                     historical_snapshot=dict(path=str(CONTROL/"FROZEN_BEFORE_REFERENCE_QUALITY.json"),
                                              sha256=sha(CONTROL/"FROZEN_BEFORE_REFERENCE_QUALITY.json")),
                     execution_boundaries=dict(fourth_chain_attempts=7, fourth_chain_seed=704,
                         old_chain_reruns=0, fifth_chain_runs=0, flow_training_calls=0, extra_training_seeds=0,
                         routing_flow_resampling_calls=0, ablation_checkpoint_reevaluation_calls=0,
                         new_fe_runs=0, damage_grid_recomputations=0, psis_threshold_changes=0,
                         prior_or_noise_changes=0, manuscript_or_response_edits=0, git_mutations=0,
                         historical_reference_overwrites=0, historical_audit_overwrites=0),
                     conditional_reevaluation=dict(allowed=False, reason="QUALITY_GATE_FAILED",
                         reference_4chain_summary_created=False, routing_vs_gap_reevaluated=False,
                         full_24_checkpoint_ablation_reevaluated=False,
                         r2_1_major_conclusion_change="NOT_ASSESSED_QUALITY_GATE_FAILED",
                         r2_3_major_conclusion_change="NOT_ASSESSED_QUALITY_GATE_FAILED",
                         selective_three_or_four_chain_reference_adoption=False),
                     mcmc_stopped=True, flow_training_started=False, stopped_at_review=True,
                     diagnostic_source=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)))
    summary_path = OUT/"REFERENCE_QUALITY_SUMMARY.csv"
    audit_path = OUT/"REFERENCE_QUALITY_AUDIT.md"
    write_new_csv(summary_path, rows)
    write_new_bytes(audit_path, ("\n".join(lines)).encode("utf-8"))
    integrity["outputs"] = {path.name: dict(path=str(path), sha256=sha(path)) for path in [summary_path, audit_path]}
    write_new_json(OUT/"REFERENCE_QUALITY_INTEGRITY.json", integrity)
    print(json.dumps(candidate_summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
