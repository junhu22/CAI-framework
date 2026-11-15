"""Read-only integration of existing results; never starts MATLAB or identification."""
from __future__ import annotations
import ast
import csv
import hashlib
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path
import h5py
import numpy as np
import scipy
from scipy.io import loadmat, whosmat

RUN = Path(__file__).resolve().parent
STAGE = RUN.parent
ROOT = STAGE.parent.parent
assert ROOT == Path(r"C:\neutra-hmc-surrogate\surrogate_training")
assert STAGE.name == "cai_r2_stage1" and RUN.name.startswith("field_modal_uncertainty_")
REEX = STAGE / "modal_setup_reextract_20260917_130854_6fbbb36c1ec642778b47717611c2da0a"
REC = STAGE / "field_modal_source_reconciliation_20260917_132624_5a0a800b89d44aa2b936571e11ba6cb5"
PRE = STAGE / "matlab_bayoma_preflight_20260917_154820_4df18c81885a4631bc62103552e5f6d3"
AUTHOR_RUN = PRE / "m6cmp_d1c30a5f5608/RUN_8cb7dbbc"
AUDIT = AUTHOR_RUN / "AUDIT_0382beec"
ARCHIVE = ROOT / "AC3 modal identification/result_AC3.mat"
PREDICTION_EXTRACT = REC / "FIELD_PREDICTION_EXTRACT.csv"
SOURCES = [ARCHIVE]
SOURCES += [REEX / n for n in ("MODAL_SETUP_RESULTS.csv", "MODAL_SETUP_SUMMARY.csv", "MODAL_SETUP_REPORT.md")]
SOURCES += [REC / n for n in ("FIELD_MODAL_SOURCE_RECONCILIATION.md", "MODAL_SOURCE_MAPPING.csv", "FIELD_PREDICTION_EXTRACT.csv", "SOURCE_RECONCILIATION_ARRAYS.npz", "SOURCE_EVIDENCE.json", "OUTPUT_VERIFICATION.json")]
SOURCES += [AUTHOR_RUN / n for n in ("MODE6_SETUP0001_POSTMORTEM.md", "RUN_STATUS.json")]
SOURCES += [AUDIT / n for n in ("NEIGHBOR_SUMMARY_CORRECTED.json", "NEIGHBOR_SUMMARY_CORRECTED.csv", "RAW_MODAL_RESULTS_AUDITED.csv", "CASE_STATUS_AUDITED.csv", "POSTMORTEM_EVIDENCE.json")]
SOURCES += [PRE / "m6cmp_d1c30a5f5608/RUN_PREPARATION.json", PRE / "AUTHOR_RUN_MODE6_SETUP0001_COMPARISON.m", PRE / "m6cmp_d1c30a5f5608/STATIC_VERIFICATION.json"]
for case in ("PRIMARY_ONE_MODE", "SENSITIVITY_TWO_MODE"):
    SOURCES += [AUTHOR_RUN / case / n for n in ("OUTPUT.mat", "INPUT.mat", "CASE_COMMAND_WINDOW.txt", "AU_CALL_STATUS.json")]

def fingerprint(path):
    st = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as s:
        for part in iter(lambda: s.read(1024 * 1024), b""):
            h.update(part)
    return dict(path=str(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns, sha256=h.hexdigest())

def write_text(name, text):
    target = RUN / name
    assert target.parent == RUN
    with target.open("x", encoding="utf-8", newline="") as s:
        s.write(text)

def write_json(name, data):
    write_text(name, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as s:
        return list(csv.DictReader(s))

def csv_text(rows):
    keys = list(rows[0])
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=keys, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        assert list(row) == keys
        values = {k: format(float(v), ".17g") if isinstance(v, (float, np.floating)) else str(v) for k, v in row.items()}
        assert all(v != "" for v in values.values())
        writer.writerow(values)
    return out.getvalue()

def h5num(group, name):
    # MATLAB v7.3/HDF5 dimensions are reversed, but component order is unchanged.
    return np.asarray(group[name], dtype=np.float64).T

def stats(x):
    assert np.all(np.isfinite(x)), "All included values must be finite"
    return dict(count=int(x.size), missing_count=0, min=float(np.min(x)), median=float(np.median(x)), max=float(np.max(x)))

def link(path, label):
    return f"[{label}](<{path.as_posix()}>)"

before = [fingerprint(p) for p in SOURCES]
write_json("SOURCE_MANIFEST_BEFORE.json", before)
source_metadata = dict(whosmat=whosmat(ARCHIVE), records=[])
r = loadmat(ARCHIVE, variable_names=["result"], struct_as_record=False, squeeze_me=False)["result"]
assert r.shape == (1, 15)
objects = [cell.flat[0] for cell in r.ravel(order="F")]
archive_f = np.vstack([o.f.ravel() for o in objects])
archive_cv = np.vstack([o.coefv.flat[0].f.ravel() for o in objects])
assert archive_f.shape == archive_cv.shape == (15, 9)
for i, o in enumerate(objects, 1):
    source_metadata["records"].append(dict(
        record_id=f"archived_record_{i:02d}", fields=o._fieldnames,
        f_shape=list(o.f.shape), coefv_fields=o.coefv.flat[0]._fieldnames,
        coefv_f_shape=list(o.coefv.flat[0].f.shape), phi_shape=list(o.phi.shape),
        identity="UNCONFIRMED_INPUT_SETUP_SEGMENT"))
with np.load(REC / "SOURCE_RECONCILIATION_ARRAYS.npz", allow_pickle=False) as old:
    assert np.array_equal(archive_f, old["result_f_hz"])
    assert np.array_equal(archive_cv, old["result_coefv_f_raw"])
    frozen = old["frozen_observation_hz"].copy()
    prediction = old["existing_stage1_prediction_hz"].copy()
    paper_prediction = old["paper_prediction_hz"].copy()
    theta = old["existing_frozen_theta_nuts"].copy()
assert np.array_equal(frozen, np.array([.91, 1.08, 1.47, 2.55, 3.16, 3.81]))
old_evidence = read_json(REC / "SOURCE_EVIDENCE.json")
old_verification = read_json(REC / "OUTPUT_VERIFICATION.json")
assert old_verification["all15_records_retained"]
assert np.allclose(archive_f.mean(axis=0), old_evidence["all15_column_means_hz"], rtol=0, atol=1e-14)
mapping = read_csv(REC / "MODAL_SOURCE_MAPPING.csv")
assert len(mapping) == old_verification["mapping_rows"] == 1488
old_setup_mapping = [row for row in mapping if row["row_kind"] == "SETUP_RECOVERY_STATUS"]
assert len(old_setup_mapping) == 84 and all(row["restored_cov_raw"].startswith("NA_") for row in old_setup_mapping)
old_reports = [(REEX / "MODAL_SETUP_REPORT.md").read_text(encoding="utf-8"),
               (REC / "FIELD_MODAL_SOURCE_RECONCILIATION.md").read_text(encoding="utf-8"),
               (AUTHOR_RUN / "MODE6_SETUP0001_POSTMORTEM.md").read_text(encoding="utf-8")]
legacy = read_csv(REEX / "MODAL_SETUP_RESULTS.csv")
assert len(legacy) == 84
legacy6 = sorted([row for row in legacy if row["paper_mode"] == "6"], key=lambda row: int(row["setup_id"]))
assert [int(row["setup_id"]) for row in legacy6] == list(range(1, 15))
legacy_frequency = np.array([float(row["frequency_hz"]) for row in legacy6])
legacy_mean = float(legacy_frequency.mean())
legacy_sd = float(legacy_frequency.std(ddof=1))
assert legacy_mean == 3.8107482214049893 and round(legacy_sd, 9) == 0.049021666
legacy_summaries = read_csv(REEX / "MODAL_SETUP_SUMMARY.csv")
assert len(legacy_summaries) == 6
legacy6_summary = next(row for row in legacy_summaries if row["paper_mode"] == "6")
assert any(v == str(legacy_mean) for v in legacy6_summary.values())
prediction_rows = read_csv(PREDICTION_EXTRACT)
saved_rows = sorted([row for row in prediction_rows if row["source_kind"] == "EXISTING_STAGE1_SAVED_UNROUNDED_SURROGATE"], key=lambda row: int(row["paper_mode"]))
paper_rows = sorted([row for row in prediction_rows if row["source_kind"] == "PAPER_REPORTED_ROUNDED_SURROGATE"], key=lambda row: int(row["paper_mode"]))
assert len(saved_rows) == len(paper_rows) == 6
assert np.array_equal(prediction, np.array([float(row["predicted_hz"]) for row in saved_rows]))
assert np.array_equal(paper_prediction, np.array([float(row["predicted_hz"]) for row in paper_rows]))
assert all(float(row["measured_hz"]) == frozen[i] for i, row in enumerate(saved_rows))
assert round(prediction[5], 9) == 3.514031649
assert np.array_equal(prediction, np.array(old_evidence["raw_prediction_six_hz"]))

au = {}
for case in ("PRIMARY_ONE_MODE", "SENSITIVITY_TWO_MODE"):
    p = AUTHOR_RUN / case
    with h5py.File(p / "OUTPUT.mat", "r") as h:
        o = h["o"]
        numeric = {k: h5num(o, k) for k in ("f", "z", "phi", "Sii", "Se", "sn", "rms")}
        numeric.update({"coefv_" + k: h5num(o["coefv"], k) for k in o["coefv"]})
        schema = dict(output_fields=list(o.keys()), coefv_fields=list(o["coefv"].keys()),
                      cmat_shape_matlab=list(o["cmat"].shape[::-1]), cmat_coordinates="NOT_INTERPRETED")
    with h5py.File(p / "INPUT.mat", "r") as h:
        x = h["in"]
        input_meta = dict(fields=list(x.keys()), tdata_shape_matlab=list(x["tdata"].shape[::-1]),
                          fs_hz=float(x["fs"][0, 0]), passed_band_hz=h5num(x, "f1f2").ravel().tolist(),
                          f0_hz=h[x["f0"][()].ravel()[0]][()].ravel().tolist(),
                          numeric_tdata_loaded_this_integration=False)
    log = (p / "CASE_COMMAND_WINDOW.txt").read_text(encoding="utf-8-sig")
    status = "AU_REPORTED_NOT_CONVERGED" if "Results have not converged" in log else "AU_REPORTED_CONVERGED" if "Results have converged" in log else "UNREPORTED"
    nh = int(re.search(r"No\. of non-positive eigenvalues of Hessian:\s*(\d+)", log).group(1))
    ne = int(re.search(r"No\. of principal directions excluded from calculating inverse:\s*(\d+)", log).group(1))
    log_lines = [dict(line_number=i + 1, text=line) for i, line in enumerate(log.splitlines())
                 if any(t in line for t in ("Results have", "non-positive eigenvalues", "principal directions excluded"))]
    call = read_json(p / "AU_CALL_STATUS.json")
    assert call["call_count"] == 1 and call["returned"] is True
    assert call["error_identifier"] == call["full_error_report"] == ""
    assert input_meta["fs_hz"] == 512 and input_meta["tdata_shape_matlab"] == [307200, 18]
    au[case] = dict(arrays=numeric, schema=schema, input=input_meta, convergence=status,
                    hessian_nonpositive=nh, directions_excluded=ne, diagnostic_log_lines=log_lines, call_status=call)
a, b = au["PRIMARY_ONE_MODE"], au["SENSITIVITY_TWO_MODE"]
af = float(a["arrays"]["f"].item())
acv = float(a["arrays"]["coefv_f"].item())
asigma = af * acv
assert af == 3.7930783457462747 and acv == 0.00178353693125842 and round(asigma, 12) == .006765095313
assert a["convergence"] == "AU_REPORTED_CONVERGED" and (a["hessian_nonpositive"], a["directions_excluded"]) == (1, 1)
assert all(np.all(np.isfinite(a["arrays"][k])) for k in ("f", "coefv_f", "z", "coefv_z", "phi"))
assert b["convergence"] == "AU_REPORTED_NOT_CONVERGED" and (b["hessian_nonpositive"], b["directions_excluded"]) == (2, 2)
assert b["arrays"]["f"].shape == (1, 2) and all(np.all(np.isnan(x)) for x in b["arrays"].values())
corrected = read_json(AUDIT / "NEIGHBOR_SUMMARY_CORRECTED.json")
assert corrected["primary_f_hz"] == af and corrected["primary_coefv_f_ratio"] == acv and corrected["primary_sigma_f_hz"] == asigma
assert corrected["status"] == "NOT_COMPARABLE_TWO_MODE_AU_NONCONVERGED_NAN"
assert len(read_csv(AUDIT / "NEIGHBOR_SUMMARY_CORRECTED.csv")) == 1
assert len(read_csv(AUDIT / "RAW_MODAL_RESULTS_AUDITED.csv")) == 3
assert len(read_csv(AUDIT / "CASE_STATUS_AUDITED.csv")) == 2
postmortem_evidence = read_json(AUDIT / "POSTMORTEM_EVIDENCE.json")
run_status = read_json(AUTHOR_RUN / "RUN_STATUS.json")
assert run_status["actual_AU_calls"] == 2
prep_path = PRE / "m6cmp_d1c30a5f5608/RUN_PREPARATION.json"
preparation = read_json(prep_path)
assert preparation["fixed_input"] == str(ROOT / "AC3 modal identification/data_0001.mat")
assert preparation["schemes"]["PRIMARY_ONE_MODE"]["f0_hz"] == a["input"]["f0_hz"]
assert preparation["schemes"]["SENSITIVITY_TWO_MODE"]["f0_hz"] == b["input"]["f0_hz"]
wrapper = PRE / "AUTHOR_RUN_MODE6_SETUP0001_COMPARISON.m"
wrapper_hash = fingerprint(wrapper)["sha256"]
assert wrapper_hash == "0719cd1d7653fd4dba4c2954e8aa96b6648028896f93a4657b460b39abdc6590"
static_preparation = read_json(PRE / "m6cmp_d1c30a5f5608/STATIC_VERIFICATION.json")
assert "data_0001.mat" in wrapper.read_text(encoding="utf-8")
assert prep_path.stat().st_mtime_ns < (AUTHOR_RUN / "PRIMARY_ONE_MODE/OUTPUT.mat").stat().st_mtime_ns

records = []
for i in range(15):
    for j in range(5):
        f, cv = float(archive_f[i, j]), float(archive_cv[i, j])
        assert np.isfinite(f) and np.isfinite(cv) and f > 0 and cv >= 0
        records.append(dict(
            paper_mode=j + 1, record_id=f"archived_record_{i+1:02d}",
            record_kind="ARCHIVED_RESULT_RECORD_NOT_CONFIRMED_INDEPENDENT_SETUP",
            setup_id="NA_RECORD_IDENTITY_UNCONFIRMED", frequency_hz=f, coefv_f_raw_ratio=cv,
            cov_percent=100*cv, sigma_f_hz=f*cv, source_file=str(ARCHIVE),
            frequency_field=f"result{{{i+1}}}.f({j+1})", cov_field=f"result{{{i+1}}}.coefv.f({j+1})",
            cov_definition="AUTHOR_CONFIRMED_DIMENSIONLESS_RELATIVE_COV",
            pairing_status="PAIRED_SAME_RECORD_SAME_COLUMN", value_status="ALL_REQUESTED_VALUES_FINITE",
            missing_reason="NONE_VALUES_PRESENT", record_identity_status="UNCONFIRMED_INPUT_SETUP_SEGMENT",
            global_mode_assignment_basis="AUTHOR_CONFIRMED_FINAL_GLOBAL_MODE5" if j == 4 else "ARCHIVE_COLUMNS1TO4_COMPATIBLE_WITH_FROZEN_OBSERVATION",
            upstream_export_chain_status="NOT_FULLY_TRACED", phi_channel_count=int(objects[i].phi.shape[0]),
            AU_scheme="NA_ARCHIVE_RUN_HISTORY_NOT_FULLY_TRACED",
            AU_log_convergence="NA_ARCHIVE_RUN_LOG_NOT_AVAILABLE_IN_THIS_EVIDENCE",
            hessian_nonpositive_count="NA_ARCHIVE_RUN_DIAGNOSTIC_NOT_AVAILABLE",
            excluded_direction_count="NA_ARCHIVE_RUN_DIAGNOSTIC_NOT_AVAILABLE",
            included_in_summary="YES", global_uncertainty_status="NOT_GLOBAL_AGGREGATE_UNCERTAINTY",
            reason="Preserved native same-record pair; unknown record identity prohibits independence claims or transfer to directory setup frequencies."))
records.append(dict(
    paper_mode=6, record_id="mode6_setup0001_PRIMARY_ONE_MODE",
    record_kind="NEW_AU_SAME_SETUP_PAIRED_IDENTIFICATION_DIAGNOSTIC",
    setup_id="0001", frequency_hz=af, coefv_f_raw_ratio=acv, cov_percent=100*acv, sigma_f_hz=asigma,
    source_file=str(AUTHOR_RUN / "PRIMARY_ONE_MODE/OUTPUT.mat"), frequency_field="o.f(1)", cov_field="o.coefv.f(1)",
    cov_definition="AUTHOR_CONFIRMED_DIMENSIONLESS_RELATIVE_COV",
    pairing_status="PAIRED_SAME_AU_OUTPUT_SAME_COMPONENT", value_status="FINITE_CONVERGED_HESSIAN_CAVEAT",
    missing_reason="NONE_FOR_THIS_SETUP_OTHER13_AU_UNCERTAINTIES_NOT_AVAILABLE",
    record_identity_status="FIXED_SETUP0001_BEFORE_RUN",
    global_mode_assignment_basis="AUTHOR_CONFIRMED_MODE6_DIAGNOSTIC_TARGET_NOT_NEW_GLOBAL_SHAPE_VERIFICATION",
    upstream_export_chain_status="EXPLICIT_FIXED_INPUT_WRAPPER_ORIGINAL_MAT_AND_LOG", phi_channel_count=18,
    AU_scheme="PRIMARY_ONE_MODE", AU_log_convergence=a["convergence"],
    hessian_nonpositive_count=1, excluded_direction_count=1, included_in_summary="YES",
    global_uncertainty_status="NOT_GLOBAL_AGGREGATE_UNCERTAINTY",
    reason="Prospectively fixed first setup; Hessian nonpositive/excluded1/1; do not attach CoV to historical mean or old point estimate."))
assert len(records) == 76
summary = []
for mode in range(1, 7):
    group = [x for x in records if x["paper_mode"] == mode]
    row = dict(paper_mode=mode, record_scope="15_ARCHIVED_RECORDS_IDENTITY_UNCONFIRMED" if mode < 6 else "ONLY_SETUP0001_NEW_AU_PRIMARY",
               record_count=len(group))
    for field, prefix, suffix in (("frequency_hz", "frequency", "_hz"), ("cov_percent", "cov_percent", ""), ("sigma_f_hz", "sigma_f", "_hz")):
        s = stats(np.array([x[field] for x in group]))
        row[prefix+"_count"] = s["count"]
        for label in ("min", "median", "max"):
            row[prefix+"_"+label+suffix] = s[label]
        row[prefix+"_missing_count"] = s["missing_count"]
    row.update(dict(
        frequency_arithmetic_mean_hz=float(np.mean([x["frequency_hz"] for x in group])),
        missing_count_scope="ONLY_INCLUDED_RECORDS_NOT_UNKNOWN_OR_NOT_RUN_RECORDS",
        legacy_mode6_setup_point_estimate_count=14 if mode == 6 else "NA_NOT_LEGACY_MODE6",
        legacy_mode6_frequency_mean_hz=legacy_mean if mode == 6 else "NA_NOT_LEGACY_MODE6",
        legacy_mode6_between_setup_sample_sd_hz=legacy_sd if mode == 6 else "NA_NOT_LEGACY_MODE6",
        mode6_AU_paired_setup_count=1 if mode == 6 else "NA_NOT_MODE6",
        mode6_other_AU_setup_uncertainty_missing_count=13 if mode == 6 else "NA_ARCHIVED_SETUP_IDENTITY_UNCONFIRMED",
        frozen_paper_observation_hz=float(frozen[mode-1]), existing_saved_surrogate_prediction_hz=float(prediction[mode-1]),
        paper_rounded_surrogate_prediction_hz=float(paper_prediction[mode-1]),
        prediction_residual_vs_frozen_hz=float(prediction[mode-1]-frozen[mode-1]),
        prediction_residual_vs_frozen_percent=float(100*(prediction[mode-1]-frozen[mode-1])/frozen[mode-1]),
        summary_status="DESCRIPTIVE_ARCHIVE_WITH_UNCONFIRMED_PROVENANCE" if mode < 6 else "SINGLE_SETUP_HESSIAN_CAVEAT_NO_GLOBAL_UNCERTAINTY",
        global_uncertainty_status="NOT_AVAILABLE",
        reason="No independence, pooled-mean CoV, global covariance, standard error, or significance claim; invalid two-mode outputs excluded."))
    summary.append(row)
residual = prediction - frozen
r6 = float(residual[5])
r6_mean = float(prediction[5] - legacy_mean)
ratios = dict(
    residual_definition="existing_saved_surrogate_prediction_minus_frozen_observation_3p81",
    surrogate_prediction_hz_unrounded_saved=float(prediction[5]), surrogate_prediction_hz_user_9dp=3.514031649,
    frozen_paper_observation_hz=3.81, residual_hz=r6, residual_percent=100*r6/3.81,
    setup0001_new_AU_sigma_f_hz=asigma, setup0001_sigma_user_12dp=.006765095313,
    legacy14_between_setup_sample_sd_hz=legacy_sd, legacy14_between_setup_sd_user_9dp=.049021666,
    abs_residual_over_setup0001_sigma=abs(r6)/asigma,
    abs_residual_over_legacy14_between_setup_sd=abs(r6)/legacy_sd,
    alternative_legacy14_mean_hz=legacy_mean, alternative_residual_vs_legacy14_mean_hz=r6_mean,
    alternative_residual_percent=100*r6_mean/legacy_mean,
    alternative_abs_residual_over_setup0001_sigma=abs(r6_mean)/asigma,
    alternative_abs_residual_over_between_setup_sd=abs(r6_mean)/legacy_sd,
    rounded_user_value_residual_hz=3.514031649-3.81,
    rounded_user_value_abs_residual_over_sigma=abs(3.514031649-3.81)/.006765095313,
    rounded_user_value_abs_residual_over_between_setup_sd=abs(3.514031649-3.81)/.049021666,
    interpretation="DESCRIPTIVE_ABSOLUTE_SCALE_RATIOS_NOT_GLOBAL_Z_SIGNIFICANCE_OR_ERROR_ATTRIBUTION")
evidence = dict(
    status="EXISTING_EVIDENCE_INTEGRATION_COMPLETED_NO_IDENTIFICATION_EXECUTED",
    run_directory=str(RUN), created_at_local=datetime.now().isoformat(),
    reader_versions=dict(Python=sys.version, numpy=np.__version__, scipy=scipy.__version__, h5py=h5py.__version__),
    archive_metadata=source_metadata, included_archived_record_ids=[f"archived_record_{i:02d}" for i in range(1,16)],
    included_archive_columns=[1,2,3,4,5], included_archive_pair_count=75, included_AU_setup_count=1,
    author_confirmations=["Dimensionless frequency CoV; percent=100*coefv.f; same-record sigma_f=f*coefv.f.",
                          "Final assembled global mode5 about3.16Hz and mode6 about3.81Hz.",
                          "Historical14setup mode6 mean3.8107482214049893Hz."],
    archive_column5_status="AUTHOR_CONFIRMED_FINAL_GLOBAL_MODE5_UPSTREAM_EXPORT_AND_IDENTITIES_UNRESOLVED",
    archive_column6_mean_hz_exclusion_context_only=float(archive_f[:,5].mean()),
    archive_column6_CoV_used_for_mode6=False, prior_restored_directory_setup_CoV_count=0,
    current_extraction_scope="75 native archived pairs; NOT75 newly matched directory setups.",
    legacy14_mode6=dict(count=14, mean_hz=legacy_mean, between_setup_sample_sd_hz_ddof1=legacy_sd,
                       source=str(REEX/"MODAL_SETUP_RESULTS.csv"), fields="paper_mode=6;setup_id=1..14;frequency_hz",
                       global_posterior_uncertainty="NOT_AVAILABLE"),
    PRIMARY_ONE_MODE={k:v for k,v in a.items() if k!="arrays"},
    PRIMARY_values=dict(f_hz=af,coefv_f_ratio=acv,CoV_percent=100*acv,sigma_f_hz=asigma,
                        z=float(a["arrays"]["z"].item()),coefv_z_ratio=float(a["arrays"]["coefv_z"].item()),
                        phi_finite_count=18,phi_total=18,limitation="Hessian1/1;NOT_fully_validated_global_uncertainty"),
    SENSITIVITY_TWO_MODE={k:v for k,v in b.items() if k!="arrays"},
    TWO_MODE_values=dict(f=["NaN","NaN"],coefv_f=["NaN","NaN"],z=["NaN","NaN"],coefv_z=["NaN","NaN"],
                         phi_finite_count=0,phi_total=36,all_coefv_fields="ALL_NAN",statistical_inclusion="EXCLUDED_FROM_ALL_AGGREGATES"),
    corrected_summary_status=corrected["status"], setup0001_fixed_before_run=True,
    selection_evidence=dict(preparation=str(prep_path),fixed_input=preparation["fixed_input"],
                            preparation_precedes_output=True,wrapper_hash=wrapper_hash,predeclared_A_and_B=True),
    prediction_source_rows=saved_rows,paper_prediction_source_rows=paper_rows,theta_frozen_NUTS_mean=theta.tolist(),
    mode6_residual_descriptive_comparisons=ratios,
    neighbor_peak_sensitivity="NOT_COMPARABLE_NONCONVERGED_RAW_NAN",
    cmat_coordinates="NOT_INTERPRETED",
    not_executed_this_task=["MATLAB","identification","P-code","FE","MCMC","training"],
    remaining13_mode6_AU_identifications="NOT_RUN_NO_DECISION_MADE",training_24_runs="NOT RUN")
write_text("FIELD_MODAL_UNCERTAINTY_RECORDS.csv",csv_text(records))
write_text("FIELD_MODAL_UNCERTAINTY_SUMMARY.csv",csv_text(summary))
write_json("INTEGRATED_EVIDENCE.json",evidence)
arrays = dict(archive_full_f_hz=archive_f,archive_full_coefv_f_raw=archive_cv,
              included_archive_first5_sigma_f_hz=archive_f[:,:5]*archive_cv[:,:5],
              legacy14_mode6_point_frequency_hz=legacy_frequency,frozen_observation_hz=frozen,
              existing_saved_surrogate_prediction_hz=prediction,paper_rounded_surrogate_prediction_hz=paper_prediction,
              prediction_residual_vs_frozen_hz=residual)
for case, values in au.items():
    arrays.update({case+"__"+k:v for k,v in values["arrays"].items()})
with (RUN/"INTEGRATED_RAW_ARRAYS.npz").open("xb") as s:
    np.savez_compressed(s,**arrays)
table = "\n".join(
    f"|{s['paper_mode']}|{s['record_count']}|{s['frequency_min_hz']:.12f} / {s['frequency_median_hz']:.12f} / {s['frequency_max_hz']:.12f}"
    f"|{s['cov_percent_min']:.9f} / {s['cov_percent_median']:.9f} / {s['cov_percent_max']:.9f}"
    f"|{s['sigma_f_min_hz']:.12f} / {s['sigma_f_median_hz']:.12f} / {s['sigma_f_max_hz']:.12f}|0 / 0 / 0|"
    for s in summary)
means_table = "\n".join(f"|{i+1}|{archive_f[:,i].mean():.15f}|{archive_f[:,i].mean():.2f}|{frozen[i]:.2f}|" for i in range(5))
pred_table = "\n".join(
    f"|{i+1}|{frozen[i]:.2f}|{paper_prediction[i]:.3f}|{prediction[i]:.15f}|{residual[i]:+.12f}|{100*residual[i]/frozen[i]:+.9f}|"
    for i in range(6))
report = f"""# 现场逐模态频率识别不确定性证据整合

本次完成指定已有资料的只读核对及统计后处理。第1–5列保留全部15条存档记录，共75项同记录、同列的频率/CoV配对，数值均有限。第6阶仅纳入运行前固定的setup0001单模态AU成套结果1项。双模态原始输出无效，全部排除统计。现有证据可以支持局部识别尺度与现场预测残差的描述性比较，**不能给出六阶全局识别不确定性或误差来源的确定性归因**。

唯一新目录：{RUN}。原文件、旧报告及论文均保持不变。本次没有启动MATLAB、识别、P-code、FE、MCMC或训练；剩余13个setup的AU识别没有执行，也没有作追加识别决定。24次消融训练保持 **NOT RUN**。

## 配对、口径及来源

作者确认：coefv.f为无量纲相对CoV；CoV百分数=100×coefv.f；sigma_f Hz=同一识别记录、同一模态的f Hz×coefv.f。每项CSV保存原值、字段、来源、身份状态及限制。频率CoV与阻尼CoV分开；未解释cmat坐标，未开任意矩阵对角线根号，未用推断likelihood整体噪声或kappa代替识别不确定性。

{link(ARCHIVE,"result_AC3.mat")}顶层result为1×15 cell，各记录f与coefv.f为1×9 double；本次仅对列1–5纳入统计。保存记录编号严格为archived_record_01至archived_record_15，记录编号不等同setup编号，也不称15个独立setup。记录1–14的phi为18×9，第15条为9×9。既有核对未发现频率向量完全相同的记录，不能据此证明数据独立或排除同一原始数据不同段/设置重复识别。

按作者最新确认，**第5列属于最终组装的全局模态5，约3.16 Hz**；不再质疑该最终选择。其上游导出链及15条记录的输入、测次/setup、时间段、通道/方向身份仍未完全追溯。旧mode5目录频率均值3.3151205173977383 Hz作为历史资料差异保留，没有用它替换最终全局模态5或改冻结观测。result_AC3第6列均值{archive_f[:,5].mean():.15f} Hz，不是论文第6阶3.81 Hz，**该列CoV从未移用给第6阶**。

此前来源核对的“恢复到目录setup的CoV为0项”仍成立：当时没有明确跨结果集setup对应。本次提取的是**存档内部75项原生f/coefv.f配对**，并非75个已匹配目录setup。作者确认最终模态归属不补齐记录身份或导出链。

## 六阶描述性汇总

每项依次为最小值 / 中位数 / 最大值。第1–5阶每种指标有15项，第6阶每种指标仅有setup0001的1项。缺失数仅针对本次纳入集合；第6阶表中0缺失不表示另外13个setup已有不确定性。

|论文模态|纳入记录数|频率Hz：最小 / 中位 / 最大|CoV%：最小 / 中位 / 最大|sigma_f Hz：最小 / 中位 / 最大|缺失数：f / CoV% / sigma|
|---|---:|---|---|---|---|
{table}

全部15条参与第1–5阶均值，没有剔除高CoV或偏离记录。两位均值匹配是数值相容证据，不替代尚缺的导出/聚合日志。

|存档列/最终模态|15条频率均值Hz|两位小数|冻结观测Hz|
|---|---:|---:|---:|
{means_table}

这些是逐记录量的描述性分布。没有把CoV均值作为平均频率CoV，没有假定记录或setup独立，没有除以sqrt(N)构造全局标准误、全局可信区间或显著性检验。较高CoV的第5阶记录也原样保留。

## 第6阶setup0001成套结果

直接读取{link(AUTHOR_RUN/"PRIMARY_ONE_MODE/OUTPUT.mat","PRIMARY_ONE_MODE/OUTPUT.mat")}原始o及同次o.coefv：

- o.f(1) = **{af:.17g} Hz**。
- o.coefv.f(1) = **{acv:.17g}**；CoV = **{100*acv:.15f}%**。
- sigma_f=f×coefv.f = **{asigma:.17g} Hz**，12位小数为作者给定的**0.006765095313 Hz**。
- o.z(1) = {float(a['arrays']['z'].item()):.17g}；o.coefv.z(1) = {float(a['arrays']['coefv_z'].item()):.17g}（阻尼CoV约26.964429709%，不作为频率不确定性）。
- o.phi为18×1，18/18项有限；cmat及其他完整诊断保留在原MAT，本次不解释坐标。

{link(AUTHOR_RUN/"PRIMARY_ONE_MODE/CASE_COMMAND_WINDOW.txt","单模态完整日志")}第242行报告在相对容差1.000e-03下收敛，迭代0–31。第244–245行报告**Hessian非正方向1个、计算逆矩阵排除方向1个**；打印非正特征值约−5.9e−12。正式异常字段为空。不能仅因负值小或CoV有限就称其统计有效性完全验证，排除方向的含义和影响仍未独立核实。

原INPUT.mat记录in.tdata为307200×18、fs=512 Hz、实际频带[3.72,3.825] Hz、本次prospective initialization为[3.802798184] Hz。本次仅检查维度和配置，未重新读取/处理时程。原流程固定18通道原序、1:4:end抽取，无额外滤波、去趋势、归一化或通道筛选。

setup0001是运行前固定的首个诊断对象：{link(prep_path,"运行前准备记录")}固定data_0001.mat及两个预定方案，时间早于输出；{link(wrapper,"手动比较脚本")}SHA256为{wrapper_hash}，与准备时相同。**不是看到有利CoV后才挑选setup；PRIMARY也是运行前声明的方案。**新AU CoV不能附到旧点估计3.8027981842875382 Hz或14个setup均值上。最终模态选择依据保留作者的振型组装确认；本次没有新增全局振型独立验证。

## 双模态失败与既有摘要更正

{link(AUTHOR_RUN/"SENSITIVITY_TWO_MODE/OUTPUT.mat","SENSITIVITY_TWO_MODE/OUTPUT.mat")}确实调用AU并返回struct，但原始o.f、o.coefv.f、o.z、o.coefv.z均为[NaN,NaN]，phi为18×2且36/36项NaN，所有6个coefv字段均NaN。coefv是存在的struct，频率字段确为o.coefv.f，不是包装脚本误读导致的空值。

{link(AUTHOR_RUN/"SENSITIVITY_TWO_MODE/CASE_COMMAND_WINDOW.txt","双模态完整日志")}第711行明确未收敛，迭代0–500，Hessian非正/排除方向2/2。第二频率在迭代313的有限精度打印中到3.89 Hz，越过实际上界3.8800000000000003 Hz，末次显示约3.93 Hz。这些不是有效最终返回值，不能据此恢复模态。

两个无效分量均保留在原MAT及本次支持NPZ，**不进入本次记录表及任何统计汇总**。单/双模态频率和CoV变化无有效配对差值，邻近峰敏感性检查仍未完成。没有调整频带、初值、容差、迭代上限或补算。

{link(AUTHOR_RUN/"MODE6_SETUP0001_POSTMORTEM.md","POSTMORTEM")}和{link(AUDIT/"NEIGHBOR_SUMMARY_CORRECTED.json","更正摘要")}已说明旧报告层遗漏：共享摘要因双模态候选失败连带将单模态显示为空，并遗漏否定收敛句；原单模态MAT没有被置空。本次原MAT与更正摘要核对一致，不把展示更正解释为双模态数值恢复。

## 历史14个setup、预测与残差

从{link(REEX/"MODAL_SETUP_RESULTS.csv","既有setup提取表")}读取paper_mode=6、setup_id=1..14的全部最终旧点估计，未按接近3.81程度筛选，也未重新追查均值来源：

- 原14个setup频率均值：**{legacy_mean:.17g} Hz**；两位小数3.81 Hz。
- setup间样本SD（ddof=1）：**{legacy_sd:.17g} Hz**；9位小数**0.049021666 Hz**。
- 此SD是setup间频率离散，不是BayOMA后验SD、均值标准误或全局不确定性。
- 新AU局部成套不确定性仅取得setup0001；**另外13个setup配对CoV仍缺失/NOT RUN**。旧14点估计与新AU1个结果属于不同输出集合，未拼接均值与任意CoV。

{link(PREDICTION_EXTRACT,"已有现场预测提取记录")}区分论文报告值和此前stage1保存的未舍入surrogate向量。本次只读这些记录，不调用surrogate或FE。参数为冻结tuned-NUTS后验均值{json.dumps(theta.tolist())}，观测为冻结六项{json.dumps(frozen.tolist())}。论文表来自submission/CAI_paper_marked.tex的tab:field_fit、逐模态行2214–2219；未舍入向量来自既有stage1 INACTIVE_DIRECTION_AUDIT.csv的model=ext、anchor=frozen_field_NUTS_summary_mean、anchor_frequency_hz。原field_freeze_results.json只保存整体RMS，没有逐模态预测向量。

|模态|冻结观测Hz|论文报告预测Hz|既有未舍入surrogate预测Hz|预测−冻结观测Hz|残差%|
|---|---:|---:|---:|---:|---:|
{pred_table}

第6阶已有预测为**{prediction[5]:.17g} Hz**，9位小数为作者指定**3.514031649 Hz**；这是surrogate计算记录，不是FE预测或本次验证。论文表显示3.514 Hz并报告−7.77%。相对于冻结3.81 Hz，未舍入残差为**{r6:.17g} Hz**、**{ratios['residual_percent']:.12f}%**。

以下量级比的分母是不同概念。主比较使用原保存数值保留精度；给定舍入值的重算另存INTEGRATED_EVIDENCE.json：

|参照尺度|尺度Hz|绝对残差/尺度|含义|
|---|---:|---:|---|
|setup0001新AU sigma_f|{asigma:.12f}|{ratios['abs_residual_over_setup0001_sigma']:.9f}|本次单setup局部识别尺度，Hessian1/1限制|
|原14个setup频率样本SD|{legacy_sd:.12f}|{ratios['abs_residual_over_legacy14_between_setup_sd']:.9f}|setup间离散|

仅作为补充描述，若基准为原14个setup未舍入均值{legacy_mean:.17g} Hz，则残差{r6_mean:.17g} Hz（{ratios['alternative_residual_percent']:.9f}%），两种尺度比分别{ratios['alternative_abs_residual_over_setup0001_sigma']:.9f}和{ratios['alternative_abs_residual_over_between_setup_sd']:.9f}。**未改论文冻结观测或推断target。**

这些只是量纲一致的描述性绝对尺度比，**不是全局z值、显著性检验、全局置信覆盖或误差归因**。新AU sigma对应新AU频率{af:.17g} Hz，残差观测基准来自旧14个setup整理值3.81 Hz，不能将分母当作该均值的SD。不能因残差大于局部识别尺度，就确定材料、构件、网格、边界或surrogate近似的具体影响；surrogate target上一致性也不代替FE物理模型验证。

## 可支持的审稿回复表述

本节只评估证据，不修改稿件或回复。可支持的表述及必须紧随的限定：

1. “存档结果保存了前五阶逐记录相对频率不确定性；按同一记录、同一模态配对提取并报告CoV和对应Hz标准差，保留全部15条记录。”须说明记录身份尚未完全追溯，不是15个已证实独立setup。全局模态5约3.16 Hz归属由作者最终振型组装确认，上游导出链仍缺。
2. “运行前固定的setup0001单模态AU诊断在日志容差下收敛，返回3.7930783457462747 Hz、CoV0.178353693125842%和sigma_f0.006765095313 Hz。”须同时列出Hessian非正/排除1/1，不称全局统计有效性完全验证，不把CoV附给历史3.81均值。
3. “第6阶surrogate预测对冻结3.81 Hz残差约−0.295968351 Hz（−7.7682%），绝对量级约为该setup局部sigma的{ratios['abs_residual_over_setup0001_sigma']:.2f}倍和旧14setup频率离散SD的{ratios['abs_residual_over_legacy14_between_setup_sd']:.2f}倍。”须紧接说明两种尺度含义不同，仅描述性比较，没有全局z/显著性或具体误差来源结论。
4. “双模态邻近峰方案未收敛且原始返回NaN，失败结果已保留，尚不能报告单/双模态CoV变化或声称邻近峰影响可忽略。”

## 不执行剩余13个setup时必须保留的限制

- 第6阶新AU局部不确定性覆盖只有1/14个运行前固定诊断对象，不能外推到14个setup、所有模态或论文平均频率。
- setup间离散可能涉及测次/布局/算法/输入差异；本次未分解来源，也没有确认setup独立或相关结构。
- 未取得第6阶跨setup/全局联合频率协方差、平均频率后验SD/可信区间或有定义的汇总传播规则，单setup输出不能填补。
- 第1–5阶15条存档记录的输入/setup/时间段、通道映射及导出链未完全追溯，只能做原生内部配对统计，不能转配另一个目录的点估计。
- 单模态Hessian排除方向的统计影响未独立验证；双模态失败使邻近峰敏感性缺有效配对证据。
- 预测是已有surrogate记录及论文报告值，没有本次FE验证；本证据不能作为六个有效物理参数识别性能、具体误差归因或普遍工程安全结论。

本次不决定追加识别，不执行剩余setup或其他计算。24次训练保持 **NOT RUN**。

## 交付与核验

FIELD_MODAL_UNCERTAINTY_RECORDS.csv共76行（15条存档×5列+1条AU）；FIELD_MODAL_UNCERTAINTY_SUMMARY.csv共6行。每项保留状态、来源字段和缺失原因，没有空白默认成功。SUMMARY缺失数仅针对纳入集合，另列第6阶剩余13项缺失覆盖。

INTEGRATED_RAW_ARRAYS.npz保留存档15×9原数组供核对（仅列1–5汇总）、AU单/双模态关键数组、历史14点、已有预测及残差；双模态IEEE NaN原样保留。INTEGRATED_EVIDENCE.json记录原始结构、日志行号、收敛及Hessian、预先固定setup、残差来源和舍入/比值依据。原cmat仍保留在原MAT，未解释或重算。

SOURCE_MANIFEST_BEFORE.json与SOURCE_PRESERVATION_CHECK.json核对定向读取来源文件的SHA256、大小、mtime前后不变；OUTPUT_VERIFICATION.json逐项核对CSV/NPZ与原结果、汇总重算、无效双模态排除和执行边界。INTEGRATE_EXISTING_EVIDENCE.py是完整只读/统计后处理脚本，新增写入仅在本目录。

核验结束后停止。本次MATLAB、识别、FE、MCMC及24次训练均未启动。
"""
write_text("FIELD_MODAL_UNCERTAINTY_EVIDENCE.md",report)
loaded = read_csv(RUN/"FIELD_MODAL_UNCERTAINTY_RECORDS.csv")
loaded_summary = read_csv(RUN/"FIELD_MODAL_UNCERTAINTY_SUMMARY.csv")
assert len(loaded)==76 and len(loaded_summary)==6
assert all(all(v!="" for v in row.values()) for row in loaded+loaded_summary)
assert {r["record_id"] for r in loaded if r["record_kind"].startswith("ARCHIVED")} == {f"archived_record_{i:02d}" for i in range(1,16)}
for row in loaded:
    mode=int(row["paper_mode"])
    if mode<6:
        i=int(row["record_id"].rsplit("_",1)[1])-1
        f,cv=archive_f[i,mode-1],archive_cv[i,mode-1]
    else:
        f,cv=af,acv
    assert float(row["frequency_hz"])==f and float(row["coefv_f_raw_ratio"])==cv
    assert float(row["cov_percent"])==100*cv and float(row["sigma_f_hz"])==f*cv
for row in loaded_summary:
    group=[r for r in loaded if r["paper_mode"]==row["paper_mode"]]
    for source,prefix,suffix in (("frequency_hz","frequency","_hz"),("cov_percent","cov_percent",""),("sigma_f_hz","sigma_f","_hz")):
        s=stats(np.array([float(r[source]) for r in group]))
        assert int(row[prefix+"_count"])==s["count"] and int(row[prefix+"_missing_count"])==s["missing_count"]
        for label in ("min","median","max"):
            assert float(row[prefix+"_"+label+suffix])==s[label]
assert len([r for r in loaded if r["paper_mode"]=="6"])==1
assert all(r["AU_scheme"]!="SENSITIVITY_TWO_MODE" for r in loaded)
with np.load(RUN/"INTEGRATED_RAW_ARRAYS.npz",allow_pickle=False) as a_saved:
    assert all(np.array_equal(a_saved[k],v,equal_nan=True) for k,v in arrays.items())
after=[fingerprint(p) for p in SOURCES]
assert before==after
write_json("SOURCE_PRESERVATION_CHECK.json",dict(status="PASSED",source_file_count=len(SOURCES),hash_size_mtime_unchanged=True,before=before,after=after))
tree=ast.parse(Path(__file__).read_text(encoding="utf-8"))
for node in ast.walk(tree):
    if isinstance(node,ast.Call):
        name=getattr(node.func,"attr","") or getattr(node.func,"id","")
        assert name not in ("system","Popen","run","exec","eval","bayoma","load_model")
verification=dict(status="PASSED",records_rows=76,summary_rows=6,archived_record_count=15,
                  archived_same_record_pairs=75,mode6_AU_paired_result_count=1,
                  all_archived_records_retained=True,CSV_no_blank_cells=True,
                  raw_MAT_prior_NPZ_records_exact=True,summary_recomputed_exact=True,
                  NPZ_round_trip_exact_including_two_mode_NaNs=True,two_mode_excluded_from_all_statistics=True,
                  mode6_no_archive_col6_CoV_transfer=True,remaining13_AU_uncertainties="MISSING_NOT_RUN",
                  source_files_unchanged=True,source_file_count=len(SOURCES),writes_limited_to_new_unique_run=True,
                  no_MATLAB_identification_forward_model_FE_MCMC_training_calls=True,training_24_runs="NOT RUN",
                  mode6_descriptive_scale_ratios=ratios,
                  outputs_before_verification_record=sorted(p.name for p in RUN.iterdir() if p.is_file()))
write_json("OUTPUT_VERIFICATION.json",verification)
print(json.dumps(dict(status="PASSED",run_directory=str(RUN),records=76,summary_modes=6,
                      mode6=dict(f_hz=af,CoV_percent=100*acv,sigma_f_hz=asigma),
                      residual_hz=r6,residual_percent=ratios["residual_percent"],
                      abs_residual_over_sigma=ratios["abs_residual_over_setup0001_sigma"],
                      abs_residual_over_between_setup_sd=ratios["abs_residual_over_legacy14_between_setup_sd"],
                      source_files_unchanged=len(SOURCES),remaining13_AU="NOT RUN",training24="NOT RUN"),
                 ensure_ascii=False))

