"""Read existing setup outputs and summarize them under author-confirmed grouping."""
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat, whosmat

RUN = Path(__file__).resolve().parent
assert RUN.parent.name == "cai_r2_stage1"
ROOT = RUN.parents[2]
MATERIAL = ROOT / "AC3 modal identification"
STAGE = RUN.parent / "run_20260917_094722_21d144f3db844a6b8a2e8167c5282dbe"
PREVIOUS = RUN.parent / "AC3_material_check_20260917_124310_a552fa014ed84784b83763588a08643d"
EXPECTED = np.array([0.91, 1.08, 1.47, 2.55, 3.16, 3.81])
AUTHOR_CONFIRMATIONS = [
    "作者确认：mode1–mode5文件夹分别对应论文第1–5阶模态。",
    "作者确认：顶层data_0001–data_0014对应论文第6阶的14个setup；论文3.81 Hz取自这14个setup频率的算术均值。",
    "作者确认：单setup coefv.f为无量纲相对不确定性；百分数=100×coefv.f；按频率CoV解释时σ_Hz=同setup同模态频率_Hz×coefv.f。",
    "作者确认：当前BayOMA仅提供单setup不确定性，没有现成的全局不确定性输出。",
    "作者本次未确认cmat坐标定义；本次不读取或解释cmat，不读取/解释/运行P-code，无需BayOMA可读源码。",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(path, large=False):
    st = path.stat()
    return {"path": str(path), "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns,
            "sha256": "NOT_HASHED_RAW_SIGNAL_METADATA_ONLY" if large else sha(path)}


groups = {}
folder_listing = {}
for mode in range(1, 6):
    directory = MATERIAL / f"mode{mode}"
    # Direct, nonrecursive listing of only the five author-specified result folders.
    listing = sorted(directory.iterdir(), key=lambda p: p.name)
    folder_listing[str(directory)] = [p.name for p in listing]
    outputs = [p for p in listing if p.is_file() and re.fullmatch(r"data_\d+_id\.mat", p.name)]
    groups[mode] = sorted(outputs, key=lambda p: int(re.search(r"data_(\d+)_id", p.name).group(1)))
groups[6] = [MATERIAL / f"data_{i:04d}_id.mat" for i in range(1, 15)]
raw_inputs = [MATERIAL / f"data_{i:04d}.mat" for i in range(1, 15)]
read_only_known = [STAGE / "TARGET_MANIFEST.json", STAGE / "STAGE1_REPORT.md", STAGE / "FE_OVERLAP_AUDIT.json",
                   PREVIOUS / "MODAL_MATERIAL_CHECK.md", PREVIOUS / "MODAL_IDENTIFICATION_EXTRACT.csv",
                   MATERIAL / "MnFunNoUnctnty.m", MATERIAL / "ObjL.m"]
protected = [(p, p in raw_inputs) for ps in groups.values() for p in ps if p.is_file()]
protected += [(p, True) for p in raw_inputs if p.is_file()]
protected += [(p, False) for p in read_only_known if p.is_file()]
before = [fingerprint(p, large) for p, large in protected]
manifest = json.loads((STAGE / "TARGET_MANIFEST.json").read_text(encoding="utf-8"))
field = next(o for o in manifest["observations"] if o["id"] == "sailboat_field_frozen")
frozen = np.array(field["observation_hz"], dtype=float)
assert np.allclose(frozen, EXPECTED, atol=1e-7, rtol=0)

raw_evidence = {}
for p in raw_inputs:
    if p.is_file():
        header = [{"name": name, "shape": list(shape), "MATLAB_class": cls} for name, shape, cls in whosmat(p)]
        # Load only the scalar fs. Never load the large tdata signal.
        scalar = loadmat(p, variable_names=["fs"])
        raw_evidence[str(p)] = {"status": "EXISTS_HEADER_AND_FS_READ_ONLY", "variables": header,
                                "sample_rate_hz": float(scalar["fs"].ravel()[0]) if "fs" in scalar else "NA_NOT_SAVED"}
    else:
        raw_evidence[str(p)] = {"status": "MISSING", "variables": [], "sample_rate_hz": "NA_INPUT_MISSING"}

rows = []
saved_schemas = []
for mode, paths in groups.items():
    seen_ids = set()
    seen_paths = set()
    for p in paths:
        setup = int(re.search(r"data_(\d+)_id", p.name).group(1))
        assert setup not in seen_ids and str(p.resolve()) not in seen_paths
        seen_ids.add(setup)
        seen_paths.add(str(p.resolve()))
        inp = MATERIAL / f"data_{setup:04d}.mat"
        row = {
            "paper_mode": mode, "setup_id": setup, "frequency_hz": "NA_OUTPUT_MISSING",
            "frequency_status": "OUTPUT_MISSING", "frequency_source_file": str(p),
            "frequency_field": "frq(1);OptVctr(1,1)", "frequency_is_final": "NA_OUTPUT_MISSING",
            "OptVctr_first_hz": "NA_OUTPUT_MISSING", "frequency_OptVctr_abs_difference_hz": "NA_OUTPUT_MISSING",
            "cov_raw": "NA_NOT_SAVED_IN_SETUP_OUTPUT", "cov_percent": "NA_NOT_SAVED_IN_SETUP_OUTPUT",
            "frequency_std_hz": "NA_COV_NOT_SAVED", "uncertainty_status": "MISSING_SAME_SETUP_COV",
            "uncertainty_source_file": "NA_NO_MATCHED_SAVED_COV", "uncertainty_field": "NA_NO_MATCHED_COEFV_F_FIELD",
            "cov_definition_status": "AUTHOR_CONFIRMED_DIMENSIONLESS_RELATIVE",
            "cov_percent_formula": "100*coefv.f", "frequency_std_formula": "same_setup_frequency_hz*coefv.f",
            "raw_input_file": str(inp), "raw_input_status": "EXISTS" if inp.is_file() else "MISSING",
            "input_output_link_basis": "AUTHOR_CONFIRMED_MODE6_SETUP_PLUS_SAME_BASENAME" if mode == 6 else "SAME_BASENAME_CANDIDATE_NOT_SAVED_PROVENANCE",
            "sample_rate_hz_raw": raw_evidence.get(str(inp), {}).get("sample_rate_hz", "NA_UNCHECKED_INPUT"),
            "mode_assignment_basis": "AUTHOR_CONFIRMED", "reason": "The saved setup MAT has no matched frequency CoV; values from other result records cannot be attached without provenance.",
        }
        if p.is_file():
            header = [{"name": name, "shape": list(shape), "MATLAB_class": cls} for name, shape, cls in whosmat(p)]
            data = loadmat(p, struct_as_record=False, squeeze_me=False)
            struct_fields = {}
            for name, value in data.items():
                if name.startswith("__"):
                    continue
                if value.size == 1:
                    entry = value.ravel()[0]
                    if hasattr(entry, "_fieldnames"):
                        struct_fields[name] = entry._fieldnames
            saved_schemas.append({"path": str(p), "variables": header, "struct_fields": struct_fields})
            if "frq" in data and np.asarray(data["frq"]).size == 1:
                value = float(np.asarray(data["frq"]).ravel()[0])
                opt = float(np.asarray(data["OptVctr"]).ravel(order="F")[0]) if "OptVctr" in data else float("nan")
                if np.isfinite(value):
                    row.update(frequency_hz=value, frequency_status="VALID_FINAL_SAVED_POINT_ESTIMATE",
                               frequency_is_final="CONFIRMED_OPTIMIZER_SAVE_CODE_AND_OPT_FIRST_VALUE",
                               OptVctr_first_hz=opt if np.isfinite(opt) else "NA_NOT_SAVED",
                               frequency_OptVctr_abs_difference_hz=abs(value-opt) if np.isfinite(opt) else "NA_NOT_SAVED")
                else:
                    row.update(frequency_status="NUMERIC_FAILURE_NONFINITE_SAVED_FREQUENCY", frequency_hz="NA_NONFINITE")
            # Only a scalar coefv.f saved in this SAME uniquely-labelled setup output
            # could be paired here. Multi-record result_AC3 is not an eligible source.
            if "coefv" in data and data["coefv"].size == 1:
                c = data["coefv"].ravel()[0]
                if hasattr(c, "f") and np.asarray(c.f).size == 1:
                    cov = float(np.asarray(c.f).ravel()[0])
                    if np.isfinite(cov) and cov >= 0 and row["frequency_status"].startswith("VALID"):
                        row.update(cov_raw=cov, cov_percent=100*cov, frequency_std_hz=row["frequency_hz"]*cov,
                                   uncertainty_status="VALID_AUTHOR_CONFIRMED_DEFINITION_SAME_FILE_SETUP",
                                   uncertainty_source_file=str(p), uncertainty_field="coefv.f(1)", reason="Same setup output and scalar frequency/CoV paired under author-confirmed definition.")
        rows.append(row)

summaries = []
for mode in range(1, 7):
    subset = [r for r in rows if r["paper_mode"] == mode]
    valid = [r for r in subset if r["frequency_status"].startswith("VALID")]
    values = np.array([r["frequency_hz"] for r in valid])
    available = [r for r in valid if r["uncertainty_status"].startswith("VALID")]
    covs = np.array([r["cov_raw"] for r in available])
    sigmas = np.array([r["frequency_std_hz"] for r in available])
    mean = float(values.mean()) if len(values) else "NA_NO_VALID_FREQUENCY"
    summary = {
        "paper_mode": mode, "setup_evidence_count": len(subset), "valid_frequency_count": len(valid),
        "frequency_mean_hz": mean, "frequency_min_hz": float(values.min()) if len(values) else "NA_NO_VALID_FREQUENCY",
        "frequency_max_hz": float(values.max()) if len(values) else "NA_NO_VALID_FREQUENCY",
        "setup_sample_std_hz_ddof1": float(values.std(ddof=1)) if len(values)>1 else "NA_FEWER_THAN_TWO_SETUPS",
        "setup_std_definition": "BETWEEN_SETUP_SAMPLE_DISPERSION_NOT_BAYOMA_POSTERIOR_STD",
        "frequency_mean_rounded_two_decimals_hz": float(np.round(mean, 2)) if len(values) else "NA_NO_VALID_FREQUENCY",
        "frozen_paper_observation_hz": float(EXPECTED[mode-1]), "manifest_actual_float32_hz": float(frozen[mode-1]),
        "mean_minus_frozen_hz": mean-EXPECTED[mode-1] if len(values) else "NA_NO_VALID_FREQUENCY",
        "mean_minus_frozen_percent": 100*(mean-EXPECTED[mode-1])/EXPECTED[mode-1] if len(values) else "NA_NO_VALID_FREQUENCY",
        "rounded_matches_frozen": bool(np.round(mean,2) == EXPECTED[mode-1]) if len(values) else False,
        "single_setup_cov_available_count": len(available),
        "single_setup_cov_min": float(covs.min()) if len(covs) else "NA_NO_MATCHED_SAVED_COV",
        "single_setup_cov_median": float(np.median(covs)) if len(covs) else "NA_NO_MATCHED_SAVED_COV",
        "single_setup_cov_max": float(covs.max()) if len(covs) else "NA_NO_MATCHED_SAVED_COV",
        "single_setup_cov_percent_min": 100*float(covs.min()) if len(covs) else "NA_NO_MATCHED_SAVED_COV",
        "single_setup_cov_percent_median": 100*float(np.median(covs)) if len(covs) else "NA_NO_MATCHED_SAVED_COV",
        "single_setup_cov_percent_max": 100*float(covs.max()) if len(covs) else "NA_NO_MATCHED_SAVED_COV",
        "single_setup_std_hz_available_count": len(available),
        "single_setup_std_hz_min": float(sigmas.min()) if len(sigmas) else "NA_COV_NOT_SAVED",
        "single_setup_std_hz_median": float(np.median(sigmas)) if len(sigmas) else "NA_COV_NOT_SAVED",
        "single_setup_std_hz_max": float(sigmas.max()) if len(sigmas) else "NA_COV_NOT_SAVED",
        "missing_cov_setup_ids": json.dumps([r["setup_id"] for r in subset if not r["uncertainty_status"].startswith("VALID")]),
        "global_uncertainty_status": "NOT_AVAILABLE_BY_AUTHOR_CONFIRMATION_NO_INDEPENDENCE_OR_SQRT_N_ASSUMED",
        "traceable_paper_prediction_hz": "NA_NOT_OBTAINED_IN_TARGETED_RESULTS",
        "traceable_prediction_residual_hz": "NA_PAPER_PREDICTION_SOURCE_NOT_OBTAINED",
        "traceable_prediction_residual_percent": "NA_PAPER_PREDICTION_SOURCE_NOT_OBTAINED",
        "reported_approx_residual_percent": -7.8 if mode == 6 else "NA_NOT_GIVEN",
        "reported_approx_residual_hz_scale_only": -0.078*EXPECTED[5] if mode == 6 else "NA_NOT_GIVEN",
        "reported_residual_status": "USER_SUPPLIED_APPROXIMATE_PERCENT_SCALE_NOT_REPRODUCED_PREDICTION" if mode == 6 else "NA_NOT_GIVEN",
        "status": "FREQUENCIES_COMPLETE_UNCERTAINTIES_MISSING" if not available else "SINGLE_SETUP_UNCERTAINTIES_AVAILABLE",
        "reason": "Author-confirmed mode grouping; each matched setup output contains point estimates only; no global BayOMA uncertainty exists in the supplied author confirmation.",
    }
    summaries.append(summary)

assert len(rows) == 84
assert len([r for r in rows if r["paper_mode"] == 6]) == 14
assert {r["setup_id"] for r in rows if r["paper_mode"] == 6} == set(range(1,15))
assert len({(r["paper_mode"], r["setup_id"]) for r in rows}) == len(rows)
assert all(r["frequency_OptVctr_abs_difference_hz"] == 0 for r in rows)
assert all(all(str(v) != "" for v in r.values()) for r in rows+summaries)
sixth = summaries[5]
assert sixth["frequency_mean_rounded_two_decimals_hz"] == 3.81
approx_delta = -0.078*EXPECTED[5]
scale_ratio = abs(approx_delta)/sixth["setup_sample_std_hz_ddof1"]

lines = [
    "# Sailboat 六阶 setup 结果重新提取",
    "",
    "本次以作者最新确认为模态/口径依据，不再要求BayOMA可读源码；仅读取已有输出并做统计后处理，未执行识别/FE/MCMC/训练/安装，未改原文件或此前报告。24次消融训练保持 **NOT RUN**。",
    "",
    f"资料目录：`{MATERIAL}`；本次输出目录：`{RUN}`。没有重新递归扫描全目录：只直接列出mode1–mode5结果文件，并检查顶层指定的14个输入/对应_id输出；另定向读取已知stage1报告/manifest/FE资产索引供残差来源核对。大型原始tdata数组未加载，只查MAT变量头和fs标量。",
    "",
    "## 作者确认与此次证据的区别",
    "",
]
lines += ["* "+s for s in AUTHOR_CONFIRMATIONS]
lines += [
    "",
    "作者确认已确定组别和coefv.f含义，**不等于每个setup数值已经保存/取得**。本次84个指定_id输出实际均仅保存OptVctr(4×1)、SNR(1×1)、MshpOpt(18×1)、frq(1×1)，没有coefv字段。result_AC3的15记录到当前setup/模态的来源对应尚未确认；因此没有从它关联CoV，也没有把其第6列CoV赋给约3.81 Hz。",
    "",
    "## 六阶频率与两类不确定性",
    "",
    "|论文模态|实际setup/有效频率数|均值 Hz|最小 Hz|最大 Hz|setup间样本SD Hz|冻结观测 Hz|均值两位小数|单setup CoV/σ可用数|",
    "|---|---|---|---|---|---|---|---|---|",
]
for s in summaries:
    lines.append(f"|{s['paper_mode']}|{s['setup_evidence_count']}/{s['valid_frequency_count']}|{s['frequency_mean_hz']:.12f}|{s['frequency_min_hz']:.9f}|{s['frequency_max_hz']:.9f}|{s['setup_sample_std_hz_ddof1']:.9f}|{s['frozen_paper_observation_hz']:.2f}|{s['frequency_mean_rounded_two_decimals_hz']:.2f}|{s['single_setup_cov_available_count']}/{s['valid_frequency_count']}；范围/中位数NA：未保存匹配CoV|")
lines += [
    "",
    "setup间样本SD公式为 sqrt(sum((f_i−mean(f))²)/(n−1))，ddof=1；它是setup间频率离散，**不是BayOMA后验标准差**。单setup CoV/σ范围及中位数在各模态均无法计算：可用数量0/14，缺失setup均为1–14。没有平均CoV作为均值频率CoV，没有除以sqrt(N)，没有假设setup独立，也没有构造全局标准误/全局z值。当前BayOMA全局不确定性不存在现成输出，这是作者确认，不是本次搜索其他位置的结论。",
    "",
    "1–4阶和第6阶均值两位舍入与冻结观测一致；第5阶不一致。论文冻结值仍保留 `[0.91,1.08,1.47,2.55,3.16,3.81] Hz`，未替换或交换模态。CSV另保留manifest float32原值，舍入核对使用论文两位小数而非float32噪声。",
    "",
    "## 第6阶：14个setup输入与输出清单",
    "",
    f"未舍入均值为 **{sixth['frequency_mean_hz']:.16f} Hz**（Python双精度完整表示 `{sixth['frequency_mean_hz']!r}`）；两位小数为 **3.81 Hz**。全部14行参与均值，未按接近3.81筛选。仅顶层data_0001–data_0014/_id计数，不将tdata别名、mode文件夹或result_AC3的15记录重复计入第6阶。",
    "",
    "|setup|频率 Hz（frq=OptVctr(1)）|CoV原值 / 百分数 / σ Hz|实际输入完整路径|识别输出完整路径|",
    "|---|---|---|---|---|",
]
for r in rows:
    if r["paper_mode"] == 6:
        lines.append(f"|{r['setup_id']:02d}|{r['frequency_hz']:.15f}|NA / NA / NA：对应输出未保存coefv.f|{r['raw_input_file']}|{r['frequency_source_file']}|")
lines += [
    "",
    "14个输入文件实际存在，MAT头均为tdata double 1228800×18、fs double 1×1，fs标量2048 Hz；每个对应_id存在且唯一，频率源均为该setup输出的frq(1)/OptVctr(1,1)。输入/输出按作者确认的第6阶setup及同名编号关联。",
    "",
    "## mode5差异复核：保存的是最终识别值",
    "",
    f"此前提取的是 `{MATERIAL / 'mode5'}` 下14个data_NNNN_id.mat的 **frq(1)**，各自均与 **OptVctr(1,1)** 完全一致（14/14，最大差0）。`{MATERIAL / 'MnFunNoUnctnty.m'}:68` 创建InlVctr初值，:69用fminsearch求得OptVctr，:70作abs，:83赋frq=OptVctr(1)，:110保存OptVctr/MshpOpt/frq/SNR；`ObjL.m:1`说明OptVctr第1项为natural frequency。因此这些已保存值是优化后的最终点估计，不是FrqIntl或InlVctr初值。",
    "",
    f"作者确认mode5仍为论文第5阶。本次最终值范围为{summaries[4]['frequency_min_hz']:.12f}–{summaries[4]['frequency_max_hz']:.12f} Hz，均值{summaries[4]['frequency_mean_hz']:.16f} Hz，舍入3.32 Hz；相对冻结3.16 Hz均值差 **+{summaries[4]['mean_minus_frozen_hz']:.12f} Hz（+{summaries[4]['mean_minus_frozen_percent']:.9f}%）**。差异没有被舍入解释掉，也没有通过重标/交换模式消除。为何论文使用3.16而此目录mode5最终输出约3.32，需要最终选用输出/版本或聚合记录；本次不推断具体原因，不把result_AC3第5列替换进来。",
    "",
    "|mode5 setup|最终frq Hz|OptVctr(1) Hz|两者差 Hz|",
    "|---|---|---|---|",
]
for r in rows:
    if r["paper_mode"] == 5:
        lines.append(f"|{r['setup_id']:02d}|{r['frequency_hz']:.12f}|{r['OptVctr_first_hz']:.12f}|{r['frequency_OptVctr_abs_difference_hz']:.1f}|")
lines += [
    "",
    "## 残差、单setup识别尺度和setup间离散分别呈现",
    "",
    "当前定向读取的作者分组_id结果只包含实测识别点估计；冻结manifest是观测/target定义；stage1报告引用约−7.8%但未提供可对应论文最终预测的六阶频率输出。已知FE资产索引中的nominal/训练锚点不能自行认定为论文最终更新预测。因此此次**没有取得可逐模态核实的论文预测频率/预测来源**，summary中prediction/residual字段明确NA，不以surrogate nominal输出、likelihood噪声或padding代替。这个结论仅限此次定向资产，不表示其他位置不存在预测表。",
    "",
    "|模态|论文预测残差|单setup识别σ Hz范围/中位数|setup间样本SD Hz|",
    "|---|---|---|---|",
]
for s in summaries:
    residual = f"用户提供约−7.8%；按3.81换算约{approx_delta:.5f} Hz，仅量级换算" if s["paper_mode"]==6 else "NA：未取得可追溯预测频率"
    lines.append(f"|{s['paper_mode']}|{residual}|NA：0/14匹配CoV，未保存|{s['setup_sample_std_hz_ddof1']:.9f}（setup间离散）|")
lines += [
    "",
    f"第6阶：按用户给出的约−7.8%（以论文3.81 Hz为分母）作量级换算，约Δf={approx_delta:.8f} Hz；setup间SD为{sixth['setup_sample_std_hz_ddof1']:.12f} Hz，按同一3.81分母约{100*sixth['setup_sample_std_hz_ddof1']/3.81:.9f}%。绝对残差量级约为setup间SD的 **{scale_ratio:.6f}倍**。这只是两种尺度的描述性比值，**不是全局z值、不是BayOMA posterior σ比较、不是显著性判断**。单setup识别σ仍缺，无法完成第二类尺度比较；也不从该比值或识别不确定性小推断具体FE建模误差来源。−7.8%未由此次可追溯预测重新算出，标记USER_SUPPLIED_APPROXIMATE_PERCENT_SCALE_NOT_REPRODUCED_PREDICTION。",
    "",
    "## 仍缺的具体字段与来源",
    "",
    "* 六阶均缺setup 1–14各自的**同setup同模态频率对应coefv.f数值**：共84项；84个指定_id文件没有coefv，已有result_AC3多记录到这些setup的来源映射未确认。定义已有作者确认，不再标记比例/百分数含义待确认；所缺是数值及同源关联。",
    "* 如果另有保存的BayOMA单setup输出，需其对应setup/模态频率与coefv.f，或可证明同源的文件/记录映射；不需要P-code解读，也不据目前缺失宣称其他位置不存在输出。",
    "* mode5最终约3.32 Hz与冻结3.16 Hz差异对应的论文采用输出/版本、选用/聚合记录；保持作者确认的mode5编号。",
    "* 论文最终逐模态预测频率及预测工况/模型版本/表格来源，供计算可追溯残差。",
    "* 全局不确定性没有现成输出（作者确认）；此次不提出未经支持的独立性估计，也不解释cmat。",
    "",
    "## 输出核验与执行边界",
    "",
    "MODAL_SETUP_RESULTS.csv为84行（六阶各14，实际文件证据计数），第6阶严格14行、setup1–14无重复；MODAL_SETUP_SUMMARY.csv为6行。全部频率逐值对应源frq/OptVctr(1)，均值、极值和ddof=1样本SD从原行重新核验。缺失项均有NA、状态和原因，没有空白默认成功。源小文件/前报告以SHA256核验，原始时程仅stat/头信息；本次所有写入仅限唯一运行目录。识别/FE/MCMC/训练/安装均未执行，24次消融NOT RUN。",
]

destinations = [RUN / "MODAL_SETUP_RESULTS.csv", RUN / "MODAL_SETUP_SUMMARY.csv", RUN / "MODAL_SETUP_REPORT.md",
                RUN / "AUTHOR_CONFIRMED_EXTRACTION_EVIDENCE.json", RUN / "OUTPUT_VERIFICATION.json"]
assert not any(p.exists() for p in destinations), "Refuse to overwrite an existing output."
for p, records in [(destinations[0], rows), (destinations[1], summaries)]:
    with p.open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
with destinations[2].open("x", encoding="utf-8") as f:
    f.write("\n".join(lines)+"\n")
with destinations[3].open("x", encoding="utf-8") as f:
    json.dump({"author_confirmations": AUTHOR_CONFIRMATIONS, "nonrecursive_targeted_folder_listings": folder_listing,
               "setup_output_metadata": saved_schemas, "raw_input_metadata_only": raw_evidence,
               "prior_report_unchanged": str(PREVIOUS / "MODAL_MATERIAL_CHECK.md"),
               "stage1_manifest": str(STAGE / "TARGET_MANIFEST.json"), "actual_frozen_observation_hz": frozen.tolist(),
               "cmat_definition": "NOT_CONFIRMED_NOT_INTERPRETED", "BayOMA_Pcode": "NOT_READ_NOT_EXECUTED",
               "result_AC3_link_to_84_setups": "NOT_CONFIRMED_NOT_USED_FOR_PAIRING",
               "approx_residual_source": "User instruction: about -7.8%; not a retrieved predicted-frequency output",
               "approx_residual_hz_scale": approx_delta, "descriptive_residual_over_between_setup_sd": scale_ratio,
               "global_z_computed": False, "single_setup_uncertainty_available_total": sum(s["single_setup_cov_available_count"] for s in summaries)},
              f, indent=2, ensure_ascii=False, allow_nan=False)

with destinations[0].open(encoding="utf-8-sig", newline="") as f:
    reread = list(csv.DictReader(f))
with destinations[1].open(encoding="utf-8-sig", newline="") as f:
    resummary = list(csv.DictReader(f))
assert len(reread) == 84 and len(resummary) == 6
assert all(all(v!="" for v in r.values()) for r in reread+resummary)
for s in resummary:
    mode = int(s["paper_mode"])
    values = np.array([float(r["frequency_hz"]) for r in reread if int(r["paper_mode"])==mode])
    assert values.mean() == float(s["frequency_mean_hz"])
    assert values.std(ddof=1) == float(s["setup_sample_std_hz_ddof1"])
    assert values.min() == float(s["frequency_min_hz"])
    assert values.max() == float(s["frequency_max_hz"])
    assert len(values) == int(s["valid_frequency_count"])
assert before == [fingerprint(p, large) for p, large in protected], "Source changed during read-only postprocessing."
with destinations[4].open("x", encoding="utf-8") as f:
    json.dump({"status": "PASSED", "results_rows": 84, "summary_rows": 6, "mode6_setup_ids": list(range(1,15)),
               "setup_pairs_unique": True, "frq_equals_OptVctr_first_all_84": True,
               "CSV_statistics_recomputed_exactly": True, "CSV_no_blank_values": True,
               "source_and_previous_reports_unchanged": True, "source_fingerprints": before,
               "global_std_or_z_computed": False, "modal_identification_executed": False, "FE_executed": False,
               "MCMC_executed": False, "training_executed": False, "dependencies_installed": False,
               "ablation_24_runs": "NOT_RUN",
               "outputs": [{"path":str(p),"sha256":sha(p),"size_bytes":p.stat().st_size} for p in destinations[:-1]]},
              f, indent=2, ensure_ascii=False, allow_nan=False)
print(json.dumps({"status":"PASSED","run":str(RUN),"setup_rows":len(rows),"summaries":summaries,
                  "mode6_details":[{"setup":r["setup_id"],"frequency_hz":r["frequency_hz"],"cov":r["cov_raw"]} for r in rows if r["paper_mode"]==6],
                  "approx_residual_over_between_setup_sd":scale_ratio,"training":"NOT_RUN"},ensure_ascii=True,indent=2))
