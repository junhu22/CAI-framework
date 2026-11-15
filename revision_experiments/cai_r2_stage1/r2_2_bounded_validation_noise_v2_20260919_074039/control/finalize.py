"""Verify saved V2 artifacts, compare both frozen designs, and seal the evidence package."""
import ast
import json

import numpy as np

import common as B

REQUIRED = ["DAMAGE_GRID_REPLAY_AUDIT_V2.json", "DAMAGE_GRID_ACTIVE_GROUPS_V2.csv",
    "DAMAGE_GRID_GROUP_RESOLVED_V2.csv", "DAMAGE_GRID_FIG20_DATA_V2.csv", "DAMAGE_GRID_TABLE18_DATA_V2.csv",
    "FE_HOLDOUT_MATCHED_INFERENCE_V2.csv", "FE_HOLDOUT_SUMMARY_V2.csv", "NOISE_SEMANTICS_V1_V2_COMPARISON.csv",
    "NOISE_SEMANTICS_V1_V2_AUDIT.md", "FE_HOLDOUT_MODAL_COMPARISON.csv", "FIELD_MODAL_UNCERTAINTY_TABLE.csv",
    "FIELD_MODE6_UNCERTAINTY_NOTE.md", "R2_2_RESPONSE_TEXT_CANDIDATE_V2.md",
    "R2_2_MANUSCRIPT_TEXT_CANDIDATE_V2.md", "R2_2_REPORT_V2.md", "INPUT_OUTPUT_INTEGRITY_V2.json"]


def check(value, expected):
    assert float(value) == float(expected), (value, expected)


def raw_ess(lw):
    assert not np.isnan(lw).any() and not np.isposinf(lw).any() and np.isfinite(lw).any()
    w = np.exp(lw-lw.max()); w /= w.sum()
    return float(1/np.sum(w*w)/len(w))


def frozen_metric_functions():
    path = B.OLD / "e3_ext_tpr.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ["auc", "tpr_at_fpr"]]
    assert len(nodes) == 2
    env = dict(np=np)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path)+":audit_metrics", "exec"), env)
    return env


def verify_artifacts():
    gate = B.read(B.OUT / "DAMAGE_GRID_REPLAY_AUDIT_V2.json")
    assert gate["exact_replay_pass"] and gate["sample_different_elements_total"] == 0
    assert gate["null_observations_exact"] == 360 and gate["original_damage_observations_exact"] == 324
    assert all(v == 0 for v in gate["metric_max_abs_differences"].values())
    assert len(gate["per_observation_comparisons"]) == 684 and len(gate["cells"]) == 27
    env = frozen_metric_functions()
    v1 = B.arrays(B.V1 / "control/ACTIVE_DAMAGE_GRID_ARRAYS.npz")
    trace = B.arrays(B.V1 / "control/HISTORICAL_NOISE_LATENT_REPLAY.npz")
    a = B.arrays(B.CONTROL / "ACTIVE_DAMAGE_GRID_ARRAYS_V2.npz")
    p = B.arrays(B.CONTROL / "FE_HOLDOUT_MATCHED_ARRAYS_V2.npz")
    repeats = B.rows(B.CONTROL / "DAMAGE_GRID_REPEAT_RESULTS_V2.csv")
    cells = B.rows(B.OUT / "DAMAGE_GRID_GROUP_RESOLVED_V2.csv")
    pairs = B.rows(B.OUT / "FE_HOLDOUT_MATCHED_INFERENCE_V2.csv")
    summaries = B.rows(B.OUT / "FE_HOLDOUT_SUMMARY_V2.csv")
    old_rows = B.rows(B.FINAL / "FINAL_GRID_ROUTING.csv")
    assert a["samples"].shape == (324, 1000, 6) and p["samples"].shape == (216, 2, 1000, 6)
    assert a["samples"].dtype == p["samples"].dtype == np.dtype("float32")
    assert len(repeats) == 324 and len(cells) == 27 and len(pairs) == 216 and len(summaries) == 18
    pm = a["samples"].mean(axis=1)
    assert np.array_equal(pm, a["posterior_means"])
    keys = lambda r: (r["parameter"], float(r["severity"]), float(r["kappa"]), int(r["rep"]))
    lookup = {keys(r): i for i, r in enumerate(repeats)}
    assert len(lookup) == 324
    assert [repeats[i]["parameter"] for i in [0,108,216]] == ["e1","e2","e3"]
    null = {k: trace["posterior_means"][[i for i,r in enumerate(old_rows) if r["group"] == "null" and float(r["kappa"]) == k]] for k in B.KAP}
    assert all(x.shape == (120, 6) for x in null.values())
    reused, changed_e2_observations = 0, 0
    for i, r in enumerate(repeats):
        old, hist = int(a["V1_active_array_indices"][i]), int(a["historical_array_indices"][i])
        coord, k = int(r["parameter"][1])-1, float(r["kappa"])
        assert np.array_equal(a["epsilon"][i], trace["epsilon"][hist])
        assert np.array_equal(a["torch_rng_state"][i], trace["torch_rng_state"][hist])
        assert np.array_equal(a["truths"][i], v1["truths"][old])
        assert np.array_equal(a["frequencies"][i], v1["frequencies"][old])
        expected_noise = k*np.abs(a["frequencies"][i])*a["epsilon"][i]
        assert np.array_equal(expected_noise, a["additive_noise"][i])
        assert np.array_equal(a["frequencies"][i]+expected_noise, a["observations"][i])
        if r["parameter"] in ["e1", "e3"]:
            for name in ["samples", "observations", "posterior_means", "additive_noise", "log_q", "log_target", "log_weights"]:
                assert np.array_equal(a[name][i], v1[name][old])
            reused += 1
        else:
            changed_e2_observations += int(not np.array_equal(a["observations"][i], v1["observations"][old]))
        check(r["affected_coordinate_mean"], pm[i, coord]); check(r["fast_posterior_damage_score"], -pm[i,coord])
        check(r["raw_IS_ESS_over_N"], raw_ess(a["log_weights"][i]))
        assert r["route_tau05"] == ("fast" if float(r["raw_IS_ESS_over_N"]) >= .5 else "correction")
        assert r["latent_sha256"] == gate["per_observation_comparisons"][hist]["latent_sha256"]
    assert reused == 216 and changed_e2_observations == 108
    for r in cells:
        group, severity, k = r["parameter"], float(r["severity"]), float(r["kappa"])
        coord = int(group[1])-1
        ids = [lookup[(group, severity, k, j)] for j in range(12)]
        scores, neg = -pm[ids, coord], -null[k][:, coord]
        check(r["AUC"], env["auc"](scores, neg)); check(r["TPR_at_5pct_FPR"], env["tpr_at_fpr"](scores, neg))
        loc = np.mean([abs(pm[j, coord]-a["truths"][j,coord])/a["truths"][j,coord] for j in ids])
        check(r["localization_parameter_recovery_relative_error"], loc)
        check(r["frozen_null_score_95percent_threshold"], np.quantile(neg, .95))
        check(r["raw_IS_ESS_over_N_mean"], np.mean([raw_ess(a["log_weights"][j]) for j in ids]))
    for name in ["samples", "observations", "epsilon", "additive_noise", "frequencies", "log_q"]:
        assert np.isfinite(a[name]).all() and np.isfinite(p[name]).all()
    for arr in [a,p]:
        valid = (arr["samples"] > 0).all(axis=-1)
        assert np.isfinite(arr["log_target"][valid]).all() and np.isneginf(arr["log_target"][~valid]).all()
        assert not np.isnan(arr["log_weights"]).any() and not np.isposinf(arr["log_weights"]).any()
    pair_pm = p["samples"].mean(axis=2)
    agreement, differences, differing_additive = 0, [], 0
    for i, r in enumerate(pairs):
        group, k, rep, severity = r["parameter"], float(r["kappa"]), int(r["rep"]), float(r["severity_percent"])/100
        coord, j = int(group[1])-1, lookup[(group,severity,k,rep)]
        hist = int(p["historical_array_indices"][i])
        assert hist == int(a["historical_array_indices"][j])
        assert np.array_equal(p["samples"][i,1], a["samples"][j])
        assert np.array_equal(p["epsilon"][i], a["epsilon"][j])
        assert np.array_equal(p["torch_rng_state"][i], a["torch_rng_state"][j])
        fe, nn = p["frequencies"][i,0], p["frequencies"][i,1].astype(np.float32)
        ef, en = k*np.abs(fe)*p["epsilon"][i], k*np.abs(nn)*p["epsilon"][i]
        assert np.array_equal(ef, p["additive_noise"][i,0]) and np.array_equal(en,p["additive_noise"][i,1])
        assert np.array_equal(fe+ef,p["observations"][i,0]) and np.array_equal(nn+en,p["observations"][i,1])
        assert np.array_equal(en, a["additive_noise"][j])
        differing_additive += int(not np.array_equal(ef,en))
        for side, source in enumerate(["FE", "NN"]):
            check(r[f"affected_mean_{source}"], pair_pm[i,side,coord])
            check(r[f"damage_score_{source}"], -pair_pm[i,side,coord])
            check(r[f"raw_IS_ESS_over_N_{source}"], raw_ess(p["log_weights"][i,side]))
            assert r[f"route_{source}_tau05"] == ("fast" if float(r[f"raw_IS_ESS_over_N_{source}"]) >= .5 else "correction")
        difference = abs(float(pair_pm[i,0,coord])-float(pair_pm[i,1,coord]))
        check(r["affected_mean_absolute_difference"],difference); differences.append(difference)
        same = r["route_FE_tau05"] == r["route_NN_tau05"]
        assert r["route_agreement"] == str(same) and r["route_changed"] == str(not same)
        agreement += int(same)
        assert r["latent_sha256"] == repeats[j]["latent_sha256"]
    for r in summaries:
        ids = [i for i, q in enumerate(pairs) if q["point"] == r["point"] and q["kappa"] == r["kappa"]]
        assert len(ids) == 12
        coord, k = int(r["parameter"][1])-1, float(r["kappa"])
        for side, source in enumerate(["FE", "NN"]):
            scores, neg = -pair_pm[ids,side,coord], -null[k][:,coord]
            check(r[f"AUC_{source}"], env["auc"](scores,neg)); check(r[f"TPR_at_5pct_FPR_{source}"],env["tpr_at_fpr"](scores,neg))
        check(r["AUC_FE_minus_NN"],float(r["AUC_FE"])-float(r["AUC_NN"]))
        check(r["TPR_FE_minus_NN"],float(r["TPR_at_5pct_FPR_FE"])-float(r["TPR_at_5pct_FPR_NN"]))
        check(r["frozen_null_score_95percent_threshold"],np.quantile(-null[k][:,coord],.95))
    figure = B.rows(B.OUT / "DAMAGE_GRID_FIG20_DATA_V2.csv")
    table = B.rows(B.OUT / "DAMAGE_GRID_TABLE18_DATA_V2.csv")
    assert len(figure) == 27 and set(r["curve_group"] for r in figure) == {"e1","e2","e3"}
    assert len(table) == 9 and set(r["parameter"] for r in table) == {"e1","e2","e3"}
    for r, f in zip(cells,figure):
        assert all(r[k] == f[k] for k in r)
    for r in table:
        cs = [c for c in cells if c["parameter"] == r["parameter"] and c["severity"] == r["severity"]]
        assert len(cs) == 3
        for metric in ["AUC","TPR_at_5pct_FPR","localization_parameter_recovery_error_percent"]:
            check(r[metric],np.mean([float(c[metric]) for c in cs]))
    direct_copies = B.read(B.CONTROL / "INFERENCE_AUDIT_V2.json")["direct_hash_verified_copies"]
    assert all(B.sha(r["source"]) == B.sha(r["output"]) == r["source_SHA256"] for r in direct_copies)
    assert len(B.rows(B.OUT / "FE_HOLDOUT_MODAL_COMPARISON.csv")) == 120
    assert len(B.rows(B.OUT / "FIELD_MODAL_UNCERTAINTY_TABLE.csv")) == 6
    overall = B.read(B.CONTROL / "INFERENCE_AUDIT_V2.json")["overall"]
    check(overall["mean_affected_coordinate_absolute_difference"], np.mean(differences))
    check(overall["maximum_affected_coordinate_absolute_difference"], max(differences))
    assert agreement == overall["route_agreement_count"]
    return dict(verification_pass=True, provenance_gate_360_null_324_damage_27_cells_exact=True,
        e1_e3_bitwise_reused_observations_and_samples=216, e2_new_relative_noise_observations=108,
        all_324_grid_noise_vectors_exact_source_relative_rule=True, all_standardized_epsilons_and_states_unchanged=True,
        matched_NN_samples_bitwise_reused=216, matched_pairs=216, matched_additive_vectors_differ=differing_additive,
        all_216_FE_NN_noise_vectors_exact_own_source_rule=True, latent_sequences_same=True,
        null_ensembles_unchanged=True, threshold_rule_and_tau_unchanged=True,
        raw_ESS_reconstruction_exact=True, negative_proposals_retained=True,
        active_nonpositive_support_draw_count=int((~(a["samples"] > 0).all(axis=2)).sum()),
        paired_FE_nonpositive_support_draw_count=int((~(p["samples"][:,0] > 0).all(axis=2)).sum()),
        paired_NN_nonpositive_support_draw_count=int((~(p["samples"][:,1] > 0).all(axis=2)).sum()),
        no_modal_discrepancy_recomputation=True, modal_and_field_files_byte_identical=True, routing_agreement_count=agreement)


def build_comparison():
    v1_cells = {(r["parameter"],float(r["severity"]),float(r["kappa"])): r for r in B.rows(B.V1 / "DAMAGE_GRID_GROUP_RESOLVED.csv")}
    v2_cells = B.rows(B.OUT / "DAMAGE_GRID_GROUP_RESOLVED_V2.csv")
    old_pairs = B.rows(B.V1 / "FE_HOLDOUT_MATCHED_INFERENCE.csv")
    new_pairs = B.rows(B.OUT / "FE_HOLDOUT_MATCHED_INFERENCE_V2.csv")
    old_blocks = {(r["point"],float(r["kappa"])): r for r in B.rows(B.V1 / "FE_HOLDOUT_SUMMARY.csv")}
    new_blocks = B.rows(B.OUT / "FE_HOLDOUT_SUMMARY_V2.csv")
    old_audit = B.read(B.V1 / "control/ACTIVE_MATCHED_AUDIT.json")
    overall1 = old_audit["matched_overall"]
    overall2 = B.read(B.CONTROL / "INFERENCE_AUDIT_V2.json")["overall"]
    comparison, cell_deltas = [], []
    for r in v2_cells:
        if r["parameter"] != "e2":
            continue
        s, k = float(r["severity"]),float(r["kappa"])
        old = v1_cells[("e2",s,k)]
        row = dict(row_type="E2_GRID_CELL", parameter="e2",severity=s,severity_percent=100*s,kappa=k,kappa_percent=100*k)
        for label, field in [("AUC","AUC"),("TPR_at_5pct_FPR","TPR_at_5pct_FPR"),
            ("recovery_error_percent","localization_parameter_recovery_error_percent"),("raw_ESS_over_N_mean","raw_IS_ESS_over_N_mean"),
            ("raw_ESS_over_N_min","raw_IS_ESS_over_N_min"),("raw_ESS_over_N_max","raw_IS_ESS_over_N_max"),
            ("fast_route_count","fast_route_count_tau05"),("damage_score_mean","fast_posterior_damage_score_mean")]:
            x, y = float(old[field]),float(r[field])
            row[f"{label}_V1"], row[f"{label}_V2"], row[f"{label}_V2_minus_V1"] = x,y,y-x
        comparison.append(row); cell_deltas.append(row)
    for r in new_blocks:
        old = old_blocks[(r["point"],float(r["kappa"]))]
        row = dict(row_type="MATCHED_POINT_KAPPA",point=r["point"],parameter=r["parameter"],severity_percent=r["severity_percent"],
            kappa=float(r["kappa"]),kappa_percent=float(r["kappa_percent"]),matched_pairs=12)
        for label, f1, f2 in [("posterior_abs_difference_mean","affected_mean_absolute_difference_mean","affected_mean_absolute_difference_mean"),
            ("posterior_abs_difference_max","affected_mean_absolute_difference_max","affected_mean_absolute_difference_max"),
            ("AUC_FE","AUC_FE","AUC_FE"),("AUC_NN","AUC_surrogate","AUC_NN"),
            ("AUC_FE_minus_NN","AUC_FE_minus_surrogate","AUC_FE_minus_NN"),
            ("TPR_FE","TPR_at_5pct_FPR_FE","TPR_at_5pct_FPR_FE"),("TPR_NN","TPR_at_5pct_FPR_surrogate","TPR_at_5pct_FPR_NN"),
            ("TPR_FE_minus_NN","TPR_FE_minus_surrogate","TPR_FE_minus_NN"),
            ("routing_agreement_count","routing_agreement_count","routing_agreement_count"),
            ("route_changes","route_changes","route_changes")]:
            x,y = float(old[f1]),float(r[f2]); row[f"{label}_V1"],row[f"{label}_V2"],row[f"{label}_V2_minus_V1"] = x,y,y-x
        comparison.append(row)
    overall_row = dict(row_type="MATCHED_OVERALL_216",matched_pairs=216)
    for label, x,y in [
        ("posterior_abs_difference_mean",overall1["affected_mean_absolute_difference_mean"],overall2["mean_affected_coordinate_absolute_difference"]),
        ("posterior_abs_difference_max",overall1["affected_mean_absolute_difference_max"],overall2["maximum_affected_coordinate_absolute_difference"]),
        ("AUC_FE_minus_NN_min",overall1["point_kappa_AUC_FE_minus_surrogate_min"],overall2["AUC_FE_minus_NN_min"]),
        ("AUC_FE_minus_NN_max",overall1["point_kappa_AUC_FE_minus_surrogate_max"],overall2["AUC_FE_minus_NN_max"]),
        ("AUC_FE_minus_NN_mean",overall1["point_kappa_AUC_FE_minus_surrogate_mean"],overall2["AUC_FE_minus_NN_mean"]),
        ("TPR_FE_minus_NN_min",overall1["point_kappa_TPR_FE_minus_surrogate_min"],overall2["TPR_FE_minus_NN_min"]),
        ("TPR_FE_minus_NN_max",overall1["point_kappa_TPR_FE_minus_surrogate_max"],overall2["TPR_FE_minus_NN_max"]),
        ("routing_agreement_count",overall1["routing_agreement_count"],overall2["route_agreement_count"]),
        ("route_changes",overall1["route_changes"],overall2["route_changes"])]:
        overall_row[f"{label}_V1"],overall_row[f"{label}_V2"],overall_row[f"{label}_V2_minus_V1"] = x,y,y-x
    comparison.append(overall_row)
    old_lookup = {(r["point"],float(r["kappa"]),int(r["rep"])): r for r in old_pairs}
    transitions = []
    for r in new_pairs:
        old = old_lookup[(r["point"],float(r["kappa"]),int(r["rep"]))]
        if old["route_FE_tau05"] != r["route_FE_tau05"] or old["route_surrogate_tau05"] != r["route_NN_tau05"]:
            transitions.append(dict(point=r["point"],kappa=float(r["kappa"]),rep_zero_based=int(r["rep"]),
                V1_FE_route=old["route_FE_tau05"],V1_NN_route=old["route_surrogate_tau05"],
                V2_FE_route=r["route_FE_tau05"],V2_NN_route=r["route_NN_tau05"],
                V1_FE_ESS=float(old["raw_IS_ESS_over_N_FE"]),V1_NN_ESS=float(old["raw_IS_ESS_over_N_surrogate"]),
                V2_FE_ESS=float(r["raw_IS_ESS_over_N_FE"]),V2_NN_ESS=float(r["raw_IS_ESS_over_N_NN"])))
    B.new_csv(B.OUT / "NOISE_SEMANTICS_V1_V2_COMPARISON.csv", comparison)
    return dict(e2_cell_comparisons=cell_deltas,V1_overall=overall1,V2_overall=overall2,
        e2_AUC_change_min=min(r["AUC_V2_minus_V1"] for r in cell_deltas),e2_AUC_change_max=max(r["AUC_V2_minus_V1"] for r in cell_deltas),
        e2_recovery_pct_change_min=min(r["recovery_error_percent_V2_minus_V1"] for r in cell_deltas),
        e2_recovery_pct_change_max=max(r["recovery_error_percent_V2_minus_V1"] for r in cell_deltas),
        e2_cells_with_TPR_change=[r for r in cell_deltas if r["TPR_at_5pct_FPR_V2_minus_V1"] != 0],
        source_routes_changed_V1_to_V2=transitions,
        within_V2_FE_NN_route_disagreements=[r for r in new_pairs if r["route_changed"] == "True"],
        comparison_row_count=len(comparison),V1_classification=B.CLASSIFICATION,
        V1_retained_not_invalidated=True,selection_basis="V2 implements the prespecified relative-noise semantics, not favorable performance")


def table(headers, data):
    return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join("---" for _ in headers)+" |"]
        + ["| "+" | ".join(str(x) for x in r)+" |" for r in data])


def build_texts(comparison):
    p1,p2 = comparison["V1_overall"],comparison["V2_overall"]
    grid = B.rows(B.OUT / "DAMAGE_GRID_TABLE18_DATA_V2.csv")
    old_grid = {(r["parameter"],float(r["severity"])):r for r in B.rows(B.V1 / "DAMAGE_GRID_TABLE18_DATA.csv")}
    monotonic_both = all(all(float(old_grid[(g,a)]["AUC"]) < float(old_grid[(g,b)]["AUC"]) for a,b in zip(B.SEV[:-1],B.SEV[1:]))
        and all(float(next(r for r in grid if r["parameter"]==g and float(r["severity"])==a)["AUC"])
        < float(next(r for r in grid if r["parameter"]==g and float(r["severity"])==b)["AUC"]) for a,b in zip(B.SEV[:-1],B.SEV[1:]))
        for g in ["e1","e2","e3"])
    assert monotonic_both
    mismatches = comparison["within_V2_FE_NN_route_disagreements"]
    assert len(mismatches)==1
    flip=mismatches[0]
    substantive = "The bounded substantive conclusions are retained: kappa-averaged AUC increases with severity for every active group; mild-damage detection is weak under the frozen null threshold; recovery is group dependent; and the tested FE/NN observation-source replacement produces small affected-mean differences. Two narrower quantitative statements change: the V1 e2 25%/3% cell has one threshold exceedance whereas V2 has none, and V1's 216/216 pairwise route agreement becomes 215/216. Thus V1's exact routing-invariance statement must not be retained. The FE modal-discrepancy and mode-6 uncertainty conclusions are unchanged because their evidence files are byte-identical. These conclusions remain limited to raw fast scores and six 10%/25% FE holdouts."
    grid_table = table(["Group","Reduction (%)","AUC","TPR@5%","Recovery error (%)","FAST/CORRECTION"],
        [[r["parameter"],f"{float(r['severity_percent']):g}",f"{float(r['AUC']):.6f}",f"{float(r['TPR_at_5pct_FPR']):.6f}",
          f"{float(r['localization_parameter_recovery_error_percent']):.4f}",f"{r['fast_route_count_tau05']}/{r['correction_route_count_tau05']}"] for r in grid])
    e2_table = table(["Severity (%)","kappa (%)","AUC V1/V2","TPR V1/V2","Recovery (%) V1/V2","Mean ESS/N V1/V2"],
        [[f"{r['severity_percent']:g}",f"{r['kappa_percent']:g}",f"{r['AUC_V1']:.6f}/{r['AUC_V2']:.6f}",
          f"{r['TPR_at_5pct_FPR_V1']:.6f}/{r['TPR_at_5pct_FPR_V2']:.6f}",
          f"{r['recovery_error_percent_V1']:.4f}/{r['recovery_error_percent_V2']:.4f}",
          f"{r['raw_ESS_over_N_mean_V1']:.6f}/{r['raw_ESS_over_N_mean_V2']:.6f}"] for r in comparison["e2_cell_comparisons"]])
    pair_table = table(["Metric (216 pairs / 18 blocks)","V1","V2"], [
        ["Mean affected-coordinate absolute difference",f"{p1['affected_mean_absolute_difference_mean']:.9f}",f"{p2['mean_affected_coordinate_absolute_difference']:.9f}"],
        ["Maximum difference",f"{p1['affected_mean_absolute_difference_max']:.9f}",f"{p2['maximum_affected_coordinate_absolute_difference']:.9f}"],
        ["AUC_FE-AUC_NN range",f"[{p1['point_kappa_AUC_FE_minus_surrogate_min']:+.6f}, {p1['point_kappa_AUC_FE_minus_surrogate_max']:+.6f}]",
            f"[{p2['AUC_FE_minus_NN_min']:+.6f}, {p2['AUC_FE_minus_NN_max']:+.6f}]"],
        ["FE-NN TPR differences","Zero in all 18 blocks","Zero in all 18 blocks"],
        ["FE/NN routing agreement",f"{p1['routing_agreement_count']}/216",f"{p2['route_agreement_count']}/216"],
        ["FE/NN route changes",p1['route_changes'],p2['route_changes']]])
    audit=f"""# V1/V2 noise-semantics comparison

**Does correcting the relative-noise scaling change the substantive R2.2 conclusions?**

{substantive}

## Frozen provenance and the sole scientific correction

V1 is retained intact and classified in this new V2 audit as `{B.CLASSIFICATION}`. It is not deleted or invalidated. V1 evaluates a common Hz additive perturbation inherited from the prior slot/source; those nominal kappa labels therefore are not exactly the replacement source's own relative noise. V2 retains the same standardized Gaussian epsilon and historical latent states, but defines every observation as `f_source+kappa*abs(f_source)*epsilon`. Choosing V2 is determined by the requested scientific noise semantics, not by which metric is favorable.

The original submitted e1/e3/e5 grid remains exactly reproducible: 360 null + 324 original damage observations/samples and all 27 metric cells pass with zero differences. This is a historical provenance verification, not a new corrected experiment. e1/e3 corrected observations and samples are reused from V1 without resampling; all null arrays are reused. Only 108 e2 grid flow inferences are new. In matched pairs, the NN samples are reused from that corrected grid; only the 216 FE-side flow inferences are computed. Epsilon, latent states, null ensemble, score definitions, rank rule, 95th-percentile rule, diagnostic target and tau=.5 remain fixed.

## Every e2 severity/kappa cell

{e2_table}

E2 cell AUC changes range from {comparison['e2_AUC_change_min']:+.6f} to {comparison['e2_AUC_change_max']:+.6f}; recovery error changes range from {comparison['e2_recovery_pct_change_min']:+.6f} to {comparison['e2_recovery_pct_change_max']:+.6f} percentage points. The 25%-reduction/3%-noise TPR changes from 1/12 to 0/12. At 50% severity, the average TPR across the three kappa cells is still 0.444444. E2 results must be replaced by V2 values wherever the noise label means source-relative kappa.

## 216 matched-pair summaries

{pair_table}

V2 pairs share the standardized epsilon and latent sequence, with separate `eta_FE=kappa*abs(f_FE)*epsilon` and `eta_NN=kappa*abs(f_NN)*epsilon`. The two Hz vectors may differ; no FE or surrogate observation is selectively replaced. All 18 point/kappa comparisons remain in `FE_HOLDOUT_SUMMARY_V2.csv`, and all 216 pair records are retained.

The sole within-V2 route mismatch is {flip['point']} ({flip['parameter']}, theta={flip['parameter_value']}, kappa={flip['kappa']}, rep={flip['rep']} zero-based): FE raw ESS/N={float(flip['raw_IS_ESS_over_N_FE']):.9f} routes CORRECTION; NN raw ESS/N={float(flip['raw_IS_ESS_over_N_NN']):.9f} routes FAST. Both lie close to the fixed .5 threshold. Affected means are {float(flip['affected_mean_FE']):.9f} FE and {float(flip['affected_mean_NN']):.9f} NN. This is a retained observation-source route sensitivity, not a reason to tune the threshold or repeat the draw. Full V1-to-V2 route transitions are recorded in `control/COMPARISON_AUDIT_V2.json`.

Modal discrepancies, the mapping and mode-6 uncertainty files are hash-verified copies of V1. First-six/all-20 maxima remain 0.048673%/0.253922%. Mode-6 residual remains approximately -0.295968 Hz (-7.768%), with scale ratios about 43.75 and 6.04. Modes 1–5 adopted-frequency uncertainty remains unavailable where reliable global pairing is absent. No new FE or identification work has occurred.

The old e5 136.4% recovery value remains in the V1/submitted/replay audit trail and does not re-enter the active physical-performance table. The null is the original heterogeneous clipped-prior-drawn ensemble, not a population of uniformly healthy structures. This is a fast-mode observation-generator sensitivity check at six independent 10%/25% holdouts, not complete FE ROC or end-to-end correction evidence. No population error probability, deployment readiness, unique discrepancy attribution or formal configuration ranking is asserted.
"""
    mapping="During the revision we re-audited the parameter-to-element mapping of the final simplified FE model. The previous e5 damage perturbation corresponded to a material label retained in the computational interface but not assigned to active elements in the final mesh. We therefore replaced that perturbation by the active e2 column group and now report the three active structural groups e1, e2 and e3 separately."
    prescribed="The same standardized Gaussian noise realization was retained, with the perturbation rescaled by the response of the replacement group/source so that the nominal 1%, 3%, and 5% relative-noise levels remained unchanged."
    matched="Matched pairs used the same standardized Gaussian realization and the same latent draws, while each observation was scaled according to its own FE or surrogate frequency vector under the unchanged relative-noise model."
    response=f"""# Reviewer response text candidate V2 — for review only

{mapping}

In the final mesh, e1 scales the specified material-1 main-building lower-storey/ground column groups together with the secondary/front column groups; e2 scales material-2 main-building higher-storey columns; and e3 scales material-3 beam groups across storeys. Coordinates e4–e6 are retained in the computational interface without active element assignments.

We preserve the system damage-sensitivity framework. The submitted grid was replayed exactly, and e1/e3 results and the complete null ensemble were retained. {prescribed} For e2, this gives `y=f_e2+kappa*abs(f_e2)*epsilon_old_e5_slot`, retaining the historical latent state, 1,000 fast draws, three severities and 12 repeats. Figure 20 data provide e1/e2/e3 curves separately, and Table 18 data retain group-specific recovery instead of averaging across locations. The damage score remains the negative raw fast posterior mean of the affected coordinate; IS-ESS is a routing diagnostic and is not a damage score.

The kappa-averaged AUCs at 10%, 25% and 50% reduction are e1: 0.5863, 0.6822, 0.9049; e2: 0.6579, 0.7354, 0.8968; e3: 0.5843, 0.7062, 0.9345. TPR is zero at 10% and 25% in this finite-repeat design; at 50%, the group averages are 0.4167, 0.4444 and 0.4167. The affected-parameter recovery errors at 50% are 10.72%, 21.48% and 4.22%, respectively. These metrics refer to the frozen heterogeneous clipped-prior null ensemble and raw fast scores, not a universal healthy-population detection rate or completed correction workflow.

Six already completed independent FE holdouts at 10% and 25% reduction were checked without running a new FE solve. The frozen modal comparisons give first-six/all-20 maximum relative discrepancies of 0.048673%/0.253922%. {matched} In 216 pairs, the affected-coordinate mean absolute difference averages {p2['mean_affected_coordinate_absolute_difference']:.6f}, with maximum {p2['maximum_affected_coordinate_absolute_difference']:.6f}. Routing agrees in 215/216 pairs; one e2=.90, kappa=3% repeat straddles tau=.5 (FE ESS/N 0.498591; NN 0.501603). AUC_FE-AUC_NN ranges from {p2['AUC_FE_minus_NN_min']:+.6f} to {p2['AUC_FE_minus_NN_max']:+.6f}, and FE/NN TPR values agree in all 18 point/noise blocks. These results support the bounded observation-generator sensitivity conclusion, while retaining the near-threshold routing difference. They do not establish full-system FE ROC or end-to-end exact CAI damage performance; the 50% FE training anchors are not independent holdouts.

The frozen mode-6 evidence is unchanged: residual approximately -0.295968 Hz (-7.768%), about 43.75 times the available setup0001 sigma_f and 6.04 times the 14-setup SD. The mode-6 discrepancy is substantially larger than the available random modal-identification uncertainty scale. Random identification uncertainty alone is unlikely to explain the full discrepancy; systematic identification bias and FE model-form discrepancy cannot be uniquely separated. The setup0001 Hessian limitation and missing reliably paired adopted-frequency uncertainty for modes 1–5 remain explicit. No new MATLAB/BayOMA identification was run.

The submitted manuscript/response is unchanged. This candidate narrows the claims to validated component evidence and the existing single field campaign; it does not assert deployable digital-twin readiness or damage attribution from low IS-ESS alone.
"""
    manuscript=f"""# Bounded manuscript text candidate V2 — for review only

The submitted manuscript is unchanged. These proposed local insertions/replacements preserve the system damage-sensitivity framework.

## Physical groups

The final FE mesh assigns e1 to material-1 specified column groups, including main lower-storey/ground and secondary/front column groups; e2 to material-2 main-building higher-storey columns; and e3 to material-3 beams spanning storeys. The retained interface coordinates e4–e6 have no active element assignments. The revision parameter-mapping audit replaces the former inactive e5 perturbation with active e2 and reports e1/e2/e3 separately.

## Noise semantics and group-resolved damage sensitivity

{prescribed} Every corrected-grid observation follows `y=f(theta)+kappa*abs(f(theta))*epsilon`. E1/e3 sample results and the submitted null ensemble are retained bitwise; e2 inherits the standardized epsilon and latent state of its historical e5 slot. The design is 10%, 25% and 50% reduction; kappa 1%, 3% and 5%; 12 repeats and 1,000 raw fast draws per observation. The affected-coordinate score is minus its raw fast mean, and recovery error is abs(mean-truth)/truth. This recovery quantity is not a blind spatial classification accuracy. AUC and strict exceedance of the null 95th percentile use their original definitions. The null is a heterogeneous clipped-prior-drawn ensemble, so the nominal 5% rule is conditional on that frozen ensemble.

Figure 20 shows separate curves/markers for e1/e2/e3 for each kappa, rather than a main curve averaged over locations. Table 18 reports each group/severity separately, with an explicitly labeled equal-weight average over the three kappa cells. At 10/25/50% severity the AUC averages are e1: 0.5863/0.6822/0.9049, e2: 0.6579/0.7354/0.8968 and e3: 0.5843/0.7062/0.9345. TPR is zero for all 10% and 25% cells; 50%-severity group means are 0.4167/0.4444/0.4167. Recovery errors at 50% are 10.72%/21.48%/4.22%. The old e5 136.4% value is retained in the revision evidence audit and is excluded from the active physical-performance table because its interface material label has no active mesh assignment.

IS-ESS/N diagnoses reliability and routing separately from the damage score. At the fixed tau=.5, 18/324 grid observations route FAST and 306/324 route CORRECTION. No correction was executed in this observation-generator check, so the scores describe fast-mode sensitivity rather than corrected damage-inference performance.

## Six independent FE holdouts and paired observation-source sensitivity

The completed independent FE points separately reduce e1/e2/e3 to .90 or .75 with all other parameters one. Their saved 20-mode observations and frozen surrogate predictions give maximum first-six/all-20 relative discrepancies 0.048673%/0.253922%. No discrepancy or FE solve is regenerated for V2. {matched} The 216 pairs retain the corresponding grid epsilon, latent state, CNF, sample count, score and null thresholds; the NN samples are reused from the corrected grid. The affected-coordinate mean absolute difference averages {p2['mean_affected_coordinate_absolute_difference']:.6f}, with maximum {p2['maximum_affected_coordinate_absolute_difference']:.6f}. AUC_FE-AUC_NN ranges from {p2['AUC_FE_minus_NN_min']:+.6f} to {p2['AUC_FE_minus_NN_max']:+.6f}; FE/NN TPRs agree in all 18 point/kappa blocks. Routes agree for 215/216 pairs. The single e2=.90, kappa 3% repeat has FE ESS/N=.498591 versus NN=.501603 and crosses the fixed .5 threshold. This route sensitivity is retained without tuning or replacement.

This is a bounded observation-generator sensitivity / independent FE holdout check covering 10% and 25% reduction. It does not validate every damage severity, a complete FE-based ROC design or the end-to-end exact CAI correction workflow. Existing 50% FE training anchors are historical anchors and not independent holdouts. The single field campaign does not establish deployable digital-twin readiness or distinguish all benign shifts from damage.

## Frozen field modal uncertainty

The adopted mode-6 frequency 3.81 Hz and saved model prediction 3.51403 Hz differ by approximately -0.295968 Hz (-7.768%). The saved setup0001 output has f6=3.7930783 Hz, CoV=.17835% and sigma_f=.0067651 Hz; the residual magnitude is about 43.75 times this available scale. Fourteen historical setup frequencies have mean 3.810748 Hz and sample SD .0490217 Hz, with residual magnitude about 6.04 times that SD.

The mode-6 discrepancy is substantially larger than the available random modal-identification uncertainty scale. Random identification uncertainty alone is unlikely to explain the full discrepancy; systematic identification bias and FE model-form discrepancy cannot be uniquely separated. The setup0001 Hessian limitation and setup-specific uncertainty scope remain. Adopted-frequency uncertainty for modes 1–5 is unavailable where no reliable archive/global pairing is established; native frequency/CoV pairs remain separately documented. These saved evidence files are unchanged.
"""
    report=f"""# CAI R2 noise-semantics bounded correction V2

1. **Submitted grid exact replay: PASS.** The 360 null and 324 original e1/e3/e5 observations/sample arrays and 27 metric cells reproduce exactly. Maximum differences and differing sample elements are zero.
2. **Sole V1→V2 scientific correction:** keep epsilon and latent states, but scale perturbations by the replacement/observation source's own frequencies: `y_source=f_source+kappa*abs(f_source)*epsilon`. V1 remains frozen as `{B.CLASSIFICATION}`. E1/e3/null are directly reused; only 108 e2 grid inferences and 216 FE-side paired inferences are new.
3. **Corrected group results:** e1 AUC10/25/50%=.5863/.6822/.9049; e2=.6579/.7354/.8968; e3=.5843/.7062/.9345. TPR at10%/25% is zero for all cells. At50%, kappa-averaged TPR is.4167/.4444/.4167 and recovery error is10.72%/21.48%/4.22%. All27 cells and nine group-specific summary rows are retained.
4. **Six FE modal discrepancy evidence:** copied and SHA256-verified, not recomputed. First-six maximum remains 0.048673%; all-20 maximum remains 0.253922%. No new FE solve.
5. **216 paired affected-coordinate mean differences:** mean absolute difference {p2['mean_affected_coordinate_absolute_difference']:.9f}; maximum {p2['maximum_affected_coordinate_absolute_difference']:.9f}.
6. **Routing agreement:** 215/216; one FE/NN route change at p03/e2=.90, kappa=.03, rep2 zero-based. FE ESS/N=.498590669 routes CORRECTION; NN=.501603113 routes FAST. The threshold remains .5.
7. **AUC/TPR and substantive conclusions:** {substantive}
8. **Mode-6 conclusion: unchanged.** Files are byte-identical to V1; residual approximately -.295968 Hz (-7.768%) is 43.75 times setup0001 sigma_f and 6.04 times the 14-setup SD. Random identification uncertainty alone is unlikely to explain the full discrepancy; systematic identification bias and FE model-form discrepancy cannot be uniquely separated.
9. **Reviewer 2.2 / 2.4 / 2.5 coverage:** 2.2 has bounded independent-FE observation-source and saved modal-uncertainty evidence with the corrected relative-noise model; single-campaign deployment and complete correction validation remain unsupported. For 2.4, group-resolved Figure 20 data and separate ESS/score/route fields are prepared; low ESS alone does not establish damage attribution. For 2.5, group-resolved Table 18 and transparent replacement of inactive e5 are prepared; this package does not add posterior-mode/prior adjudication or broader field validation. These are evidence and candidate-text deliverables, not a declaration that every part of the reviewer comments is resolved.
10. **Unsupported claims remain:** complete FE-based ROC validation across all severities, end-to-end exact CAI damage performance, correction performance for the many low-ESS cases, universal error rates or calibrated threshold optimality, field damage attribution, deployable digital-twin readiness, unique model-form attribution or verified global uncertainty for all adopted modal frequencies.

## Corrected group-resolved system results

{grid_table}

Rows average the three kappa cells within each group/severity only. Figure 20 data have 27 group-specific markers; Table 18 has nine group/severity rows. At 50% reduction, recovery is weaker for e2 than the beam group, and the difference remains visible instead of being hidden in a cross-group average. The previous e5 136.4% recovery result remains in the frozen V1/submitted/replay audits and does not re-enter the active physical-performance table.

`DAMAGE_GRID_ACTIVE_GROUPS_V2.csv` is a hash-identical copy of the V1 physical mapping. E1 includes main lower-storey/ground and secondary/front material-1 column groups, e2 is material-2 main higher-storey columns, e3 is material-3 cross-storey beams. E4–e6 remain interface coordinates without active element assignments. No APDL or stale config.py text was edited or used to redefine these groups.

## Relative-noise protocol and unchanged controls

For e2, the old e5 slot epsilon and pre-sample torch RNG state are unchanged. Only `kappa*abs(f_e2)` replaces the old slot's amplitude. Kappa labels are .01/.03/.05, severities .10/.25/.50, repeats 12 and Nfast 1000. E1/e3 already satisfy this rule and are reused bitwise. The complete submitted null ensemble is unchanged.

For the 216 matched pairs, `y_NN=f_NN+kappa*abs(f_NN)*epsilon` and `y_FE=f_FE+kappa*abs(f_FE)*epsilon`. Epsilon and latent draws are paired; additive Hz vectors can differ because the sources differ. The NN observations and sample arrays are reused from the corresponding corrected-grid cell, and FE sampling restores its same historical latent state. No fallback seeds, random design or repeated favorable run is used. The FE/modal comparison and mode-6 files are copied, not recalculated.

The damage score is the negative raw affected-coordinate fast mean. The original AUC two-argsort rank rule, strict threshold exceedance and 95th-percentile null-score rule are frozen. E2 uses e2-coordinate means of the same 120 null samples per kappa. This null consists of clipped-prior-drawn heterogeneous truths, not a uniformly healthy population, so the FPR label is conditional on that frozen ensemble.

Raw ESS uses the unchanged frozen diagnostic function with K=20, independent untruncated lognormal sigma_log=.55, kappa_sq=.001 and tau=.5. Observation-generator kappa and this diagnostic likelihood setting are distinct; neither has been retuned. All raw proposal draws remain in means/scores. Nonpositive-support draws receive zero target weight, and their expected log-target/log-weight -inf is not a NaN or I/O failure. The grid has 18 FAST and 306 CORRECTION routes; no correction MCMC is run.

## Direct V1/V2 comparison

{e2_table}

{pair_table}

`NOISE_SEMANTICS_V1_V2_COMPARISON.csv` records all nine e2 cells, all 18 matched point/kappa blocks and the 216-pair overall summary. `NOISE_SEMANTICS_V1_V2_AUDIT.md` answers the substantive-conclusion question with both the retained pattern and the changed finite-repeat statements. V2 replaces V1 for nominal source-relative noise reporting on scientific-definition grounds alone.

The FE-vs-NN source effect remains limited over the tested six independent points: mean absolute affected-coordinate difference is {p2['mean_affected_coordinate_absolute_difference']:.9f}, maximum {p2['maximum_affected_coordinate_absolute_difference']:.9f}; AUC differences range [{p2['AUC_FE_minus_NN_min']:+.6f}, {p2['AUC_FE_minus_NN_max']:+.6f}], and all 18 FE/NN TPR differences are zero. The single pairwise routing difference is explicitly retained. This check covers 10%/25% holdouts, not independent 50% holdouts; existing .50 FE training anchors remain historical training anchors.

## Frozen field uncertainty and source limitations

`FIELD_MODAL_UNCERTAINTY_TABLE.csv` and `FIELD_MODE6_UNCERTAINTY_NOTE.md` retain the exact V1 bytes and hashes. Setup0001 f6=3.7930783 Hz, CoV=.17835%, sigma_f=.0067651 Hz; historical 14-setup mean 3.810748 Hz and sample SD .0490217 Hz; adopted observation 3.81 Hz and saved model prediction 3.51403 Hz. Residual and descriptive scale ratios remain unchanged. The one non-positive Hessian eigenvalue and excluded inverse direction remain documented. Modes 1–5 adopted-frequency uncertainty stays unavailable where reliable archive/global pairing is absent. No uncertainty is copied onto an unconfirmed frequency record.

The mode-6 discrepancy is substantially larger than the available random modal-identification uncertainty scale. Random identification uncertainty alone is unlikely to explain the full discrepancy; systematic identification bias and FE model-form discrepancy cannot be uniquely separated. No unique source attribution or calibrated global error probability follows from these ratios.

## Integrity and stop point

All V1 files, its integrity manifest and sealing record remain unchanged. All earlier protected assets and 305 Git-tracked file hashes/status are verified unchanged. Model/reference/checkpoint/train-bank/submission/FE/APDL evidence is preserved. The corrected noise formulas, sample reuse, latent pairing, AUC/TPR/recovery statistics and raw ESS are independently checked from saved artifacts without new inference. The output manifest records source and output SHA256, copy equivalence, execution counts and the V1 classification. No training, MCMC, PSIS, new FE, MATLAB, new seed design, threshold tuning, null change, submitted manuscript/response edit or Git mutation occurred. Execution stops for review.
"""
    return audit,response,manuscript,report,substantive


def main():
    assert not (B.OUT / "INPUT_OUTPUT_INTEGRITY_V2.json").exists()
    snapshot=B.read(B.FROZEN)
    B.verify(snapshot)
    verification=verify_artifacts()
    comparison=build_comparison()
    audit,response,manuscript,report,substantive=build_texts(comparison)
    for text in [response,manuscript]:
        for phrase in ["inheriting each old e5 additive noise vector","without rescaling the noise","same actual additive noise"]:
            assert phrase not in text.lower()
    for text in [audit,response,manuscript,report]:
        for phrase in ["best configuration","statistically equivalent","statistically comparable","significant improvement","negative control","forgotten bug","coding error"]:
            assert phrase not in text.lower()
    for name,text in [("NOISE_SEMANTICS_V1_V2_AUDIT.md",audit),("R2_2_RESPONSE_TEXT_CANDIDATE_V2.md",response),
        ("R2_2_MANUSCRIPT_TEXT_CANDIDATE_V2.md",manuscript),("R2_2_REPORT_V2.md",report)]:
        B.new_bytes(B.OUT / name,text.encode("utf-8"))
    comparison["substantive_conclusion_assessment"]=substantive
    comparison["bounded_qualitative_conclusions_retained"]=True
    comparison["V1_exact_route_invariance_claim_retained"]=False
    B.new_json(B.CONTROL / "COMPARISON_AUDIT_V2.json",comparison)
    assert all((B.OUT / name).is_file() and (B.OUT / name).stat().st_size for name in REQUIRED[:-1])
    outputs={str(p.relative_to(B.OUT)):B.sha(p) for p in sorted(B.OUT.rglob("*")) if p.is_file()}
    preservation=B.verify(snapshot)
    inf=B.read(B.CONTROL / "INFERENCE_AUDIT_V2.json")
    manifest=dict(status="COMPLETE_NOISE_SEMANTICS_V2_STOPPED_FOR_REVIEW",created_utc=B.now(),output_directory=str(B.OUT),
        required_deliverables=REQUIRED,V1_directory=str(B.V1),V1_classification=B.CLASSIFICATION,V1_files=snapshot["V1_files"],
        V1_retained_not_invalidated=True,protected_inputs=snapshot["files"],original_367_frozen_assets=snapshot["original_367_files"],
        Git_tracked_file_hashes=snapshot["Git_tracked_file_hashes"],Git_tracked_status_before=snapshot["git_tracked_before"],
        Git_tracked_status_after=B.git("status","--porcelain","-uno"),preservation=preservation,
        output_SHA256=outputs,artifact_verification=verification,comparison=comparison,
        unchanged_hash_verified_copies=inf["direct_hash_verified_copies"],
        source_model_SHA256=dict(surrogate=B.sha(B.SUR),CNF=B.sha(B.CNF)),
        noise_design=dict(e2="f_e2+kappa*abs(f_e2)*epsilon_old_e5_slot",FE="f_FE+kappa*abs(f_FE)*epsilon",NN="f_NN+kappa*abs(f_NN)*epsilon",
            epsilon="Original slot's standardized Gaussian realization",latents="Original pre-sample torch RNG state",
            severities=B.SEV,kappa=B.KAP,repeats=12,Nfast=1000,null_per_kappa=120,tau=.5,
            no_new_random_seed_design=True,score="Negative raw affected-coordinate fast mean",threshold="Original np.quantile(null scores,.95), strict >",
            diagnostic_target=inf["diagnostic_target"]),
        execution_counts={key:inf[key] for key in ["historical_provenance_inverse_verification_calls","new_e2_grid_flow_sample_calls",
            "new_FE_matched_flow_sample_calls","new_e1_e3_or_NN_matched_flow_sample_calls","new_training_calls","new_MCMC_calls",
            "new_FE_solve_calls","new_PSIS_calls","new_MATLAB_calls","new_random_seed_designs"]},
        modal_discrepancy_recomputed=False,mode6_identification_recomputed=False,manuscript_response_modified=False,Git_mutation=False,
        excluded_from_embedded_output_hashes=["INPUT_OUTPUT_INTEGRITY_V2.json","control/FINAL_VERIFICATION_V2.json"],
        hash_exclusion_reason="Avoid self-hash cycle; final sealing record hashes the integrity manifest",stopped_for_review=True)
    B.new_json(B.OUT / "INPUT_OUTPUT_INTEGRITY_V2.json",manifest)
    assert all(B.sha(B.OUT / p)==h for p,h in outputs.items())
    final_preservation=B.verify(snapshot)
    B.new_json(B.CONTROL / "FINAL_VERIFICATION_V2.json",dict(verification_pass=True,finished_utc=B.now(),
        required_deliverables_present=len(REQUIRED),integrity_manifest_sha256=B.sha(B.OUT / "INPUT_OUTPUT_INTEGRITY_V2.json"),
        report_sha256=B.sha(B.OUT / "R2_2_REPORT_V2.md"),preservation=final_preservation,
        artifact_verification=verification,V1_retained=True,V1_classification=B.CLASSIFICATION,stopped_for_review=True))
    print("V2_FINAL_PACKAGE_PASS",json.dumps(dict(required_outputs=len(REQUIRED),preservation=final_preservation,
        matched_overall=comparison["V2_overall"],bounded_conclusions_retained=True,
        route_invariance_claim_changed=True,e2_25pct_3pct_TPR_changed=True)),flush=True)


if __name__=="__main__":
    main()
