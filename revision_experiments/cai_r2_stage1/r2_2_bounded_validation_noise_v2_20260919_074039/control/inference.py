"""V2 only: historical provenance gate, 108 e2 inferences, 216 paired FE inferences."""
import ast
import hashlib
import json
import sys
import time

import numpy as np
import torch

import common as B

sys.path.insert(0, str(B.ROOT / "experiments"))
import cai_lib as C


def functions(sur, cnf):
    env = dict(np=np, torch=torch, C=C, sur=sur, cnf=cnf)
    for path, names in [(B.OLD / "e3_ext_tpr.py", ["auc", "tpr_at_fpr"]),
                        (B.FINAL / "verify_final_ext.py", ["diagnostic"])]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        assert len(nodes) == len(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path)+":frozen_functions", "exec"), env)
    return env


def route(value):
    return "fast" if value >= .5 else "correction"


def historical_gate(sur, flow, env, historic, trace):
    path = B.OUT / "DAMAGE_GRID_REPLAY_AUDIT_V2.json"
    assert not path.exists()
    previous = B.read(B.V1 / "DAMAGE_GRID_REPLAY_AUDIT.json")
    assert previous["exact_replay_pass"]
    result = dict(exact_replay_pass=False, original_grid_groups=["e1", "e3", "e5"],
        historical_seed=42, frozen_V1_gate_sha256=B.sha(B.V1 / "DAMAGE_GRID_REPLAY_AUDIT.json"),
        original_script_sha256=B.sha(B.OLD / "e3_ext.py"), original_TPR_script_sha256=B.sha(B.OLD / "e3_ext_tpr.py"),
        original_arrays_sha256=B.sha(B.FINAL / "FINAL_GRID_REPLAY_ARRAYS.npz"),
        trace_sha256=B.sha(B.V1 / "control/HISTORICAL_NOISE_LATENT_REPLAY.npz"),
        purpose="Provenance only; corrected e1/e3/null arrays are subsequently reused, not resampled",
        method="Replay original numpy seed42 logic; reconstruct frozen historical latent sequence and deterministic CNF inverse",
        original_raw_storage="Original scripts stored aggregate cells; raw comparisons use frozen final-ext verified replay arrays")
    rng = np.random.RandomState(42)
    gen = torch.Generator(device="cpu")
    records, means, null, cells = [], [], {}, {}
    count = 0

    def one(theta, k):
        nonlocal count
        with torch.no_grad():
            frequency = sur(torch.tensor(theta, dtype=torch.float32).reshape(1, -1)).squeeze(0).numpy()
        epsilon = rng.randn(20)
        y = frequency+k*np.abs(frequency)*epsilon
        gen.set_state(torch.from_numpy(trace["torch_rng_state"][count].copy()))
        z = torch.randn(1000, 6, generator=gen, dtype=torch.float32)
        with torch.no_grad():
            tf, _ = flow.inverse(z, torch.tensor(y, dtype=torch.float32).unsqueeze(0).expand(1000, -1))
        tf = tf.numpy()
        row = dict(index=count, truth_exact=bool(np.array_equal(theta, historic["truths"][count])),
            epsilon_exact=bool(np.array_equal(epsilon, trace["epsilon"][count])),
            observation_exact=bool(np.array_equal(y, historic["observations"][count])),
            samples_exact=bool(np.array_equal(tf, historic["samples"][count])),
            sample_max_abs_difference=float(np.abs(tf.astype(float)-historic["samples"][count].astype(float)).max()),
            sample_different_elements=int(np.count_nonzero(tf != historic["samples"][count])),
            latent_sha256=hashlib.sha256(z.numpy().tobytes()).hexdigest())
        if count+1 < 684:
            row["next_historical_RNG_state_exact"] = bool(np.array_equal(gen.get_state().numpy(), trace["torch_rng_state"][count+1]))
        records.append(row)
        assert all(row[k] for k in ["truth_exact", "epsilon_exact", "observation_exact", "samples_exact"]), row
        assert row.get("next_historical_RNG_state_exact", True), row
        pm = tf.mean(0)
        assert np.array_equal(pm, trace["posterior_means"][count])
        means.append(pm); count += 1
        return pm

    try:
        for k in B.KAP:
            null[k] = np.array([one(np.clip(np.exp(.55*rng.randn(6)), .5, 3.5), k) for _ in range(120)])
            print("V2_PROVENANCE_EXACT_NULL", k, flush=True)
        for group, coord in {"e1": 0, "e3": 2, "e5": 4}.items():
            for severity in B.SEV:
                for k in B.KAP:
                    theta = np.ones(6); theta[coord] = 1-severity
                    block = np.array([one(theta, k) for _ in range(12)])
                    cells[f"{group}|{severity}|{k}"] = dict(
                        auc=float(env["auc"](-block[:, coord], -null[k][:, coord])),
                        tpr05=env["tpr_at_fpr"](-block[:, coord], -null[k][:, coord]),
                        loc_err=float(np.mean([abs(v[coord]-theta[coord])/theta[coord] for v in block])))
            print("V2_PROVENANCE_EXACT_DAMAGE", group, flush=True)
        original, tpr = B.read(B.OLD / "e3_ext_results.json"), B.read(B.OLD / "e3_ext_tpr_results.json")
        difference = {m: max(abs(cells[k][m]-tpr["cells"][k][m]) for k in cells) for m in ["auc", "tpr05", "loc_err"]}
        assert all(v == 0 for v in difference.values())
        assert all(cells[k][m] == original["cells"][k][m] for k in cells for m in ["auc", "loc_err"])
        assert count == 684
        result.update(exact_replay_pass=True, classification="EXACT_SUBMITTED_DAMAGE_GRID_REPLAY",
            null_observations_exact=360, original_damage_observations_exact=324,
            original_damage_per_group_exact=dict(e1=108, e3=108, e5=108), all_684_samples_bitwise_exact=True,
            sample_different_elements_total=0, sample_max_abs_difference=0., metric_max_abs_differences=difference,
            cells=cells, per_observation_comparisons=records, historical_provenance_inverse_verification_calls=684,
            new_corrected_e1_e3_sampling_calls=0, new_random_design=False)
    except Exception as error:
        result.update(classification="DAMAGE_GRID_NOT_EXACTLY_REPLAYED", original_error=str(error),
            error_type=type(error).__name__, per_observation_comparisons=records)
    result["finished_utc"] = B.now()
    B.new_json(path, result)
    assert result["exact_replay_pass"], "STOPPED_BEFORE_V2_INFERENCE: submitted exact replay gate failed"
    print("V2_SUBMITTED_EXACT_GATE_PASS", flush=True)


def cell_row(group, severity, k, block, null, env):
    coord = int(group[1])-1
    pm = np.stack([x["pm"] for x in block])
    scores, neg = -pm[:, coord], -null[k][:, coord]
    values = [x["diag"]["ess_ratio"] for x in block]
    loc = float(np.mean([abs(x["pm"][coord]-x["theta"][coord])/x["theta"][coord] for x in block]))
    return dict(parameter=group, severity=severity, severity_percent=100*severity, kappa=k, kappa_percent=100*k,
        repeats=12, null_repeats=120, Nfast=1000, AUC=float(env["auc"](scores, neg)),
        TPR_at_5pct_FPR=env["tpr_at_fpr"](scores, neg), localization_parameter_recovery_relative_error=loc,
        localization_parameter_recovery_error_percent=100*loc, affected_parameter_mean=float(pm[:, coord].mean()),
        fast_posterior_damage_score_mean=float(scores.mean()), fast_posterior_damage_score_sample_SD=float(scores.std(ddof=1)),
        fast_posterior_damage_scores=json.dumps([float(v) for v in scores]),
        frozen_null_score_95percent_threshold=float(np.quantile(neg, .95)),
        raw_IS_ESS_over_N_mean=float(np.mean(values)), raw_IS_ESS_over_N_min=min(values), raw_IS_ESS_over_N_max=max(values),
        raw_IS_ESS_over_N_by_repeat=json.dumps(values), routes_tau05_by_repeat=json.dumps([route(v) for v in values]),
        fast_route_count_tau05=sum(v >= .5 for v in values), correction_route_count_tau05=sum(v < .5 for v in values),
        original_noise_slot="e5" if group == "e2" else group,
        standardized_epsilon_and_latent_state_inherited=True, source_relative_noise_scaling=True,
        observation_rule="f_source+kappa*abs(f_source)*epsilon", score_definition="Negative raw affected-coordinate mean",
        null_kind="Frozen heterogeneous clipped-prior-drawn ensemble; not all-healthy theta=1")


def main():
    marker = B.CONTROL / "INFERENCE_STARTED_V2.json"
    assert not marker.exists()
    snapshot = B.read(B.FROZEN)
    B.verify(snapshot)
    torch.set_num_threads(6)
    assert torch.__version__ == "2.6.0+cu124" and np.__version__ == "2.4.4"
    start = time.perf_counter()
    B.new_json(marker, dict(started_utc=B.now(), allowed_new_grid_inferences=108, allowed_new_FE_inferences=216,
        e1_e3_and_NN_pair_samples_reused=True, standardized_epsilon_and_latent_states_frozen=True,
        relative_noise_rule="y_source=f_source+kappa*abs(f_source)*epsilon", new_seed_designs=0,
        controller_sha256=B.sha(__file__)))
    sur, flow = C.load_surrogate(B.SUR), C.load_cnf(B.CNF, dim_theta=6, dim_y=20, n_layers=8, hidden_dim=128)
    assert not sur.training and not flow.training
    assert all(torch.isfinite(t).all() for m in [sur, flow] for t in m.state_dict().values())
    env = functions(sur, flow)
    original = B.arrays(B.FINAL / "FINAL_GRID_REPLAY_ARRAYS.npz")
    trace = B.arrays(B.V1 / "control/HISTORICAL_NOISE_LATENT_REPLAY.npz")
    historical_gate(sur, flow, env, original, trace)
    copies = [B.copy_frozen(B.V1 / source, B.OUT / target) for source, target in [
        ("DAMAGE_GRID_ACTIVE_GROUPS.csv", "DAMAGE_GRID_ACTIVE_GROUPS_V2.csv"),
        ("FE_HOLDOUT_MODAL_COMPARISON.csv", "FE_HOLDOUT_MODAL_COMPARISON.csv"),
        ("FIELD_MODAL_UNCERTAINTY_TABLE.csv", "FIELD_MODAL_UNCERTAINTY_TABLE.csv"),
        ("FIELD_MODE6_UNCERTAINTY_NOTE.md", "FIELD_MODE6_UNCERTAINTY_NOTE.md")]]
    v1 = B.arrays(B.V1 / "control/ACTIVE_DAMAGE_GRID_ARRAYS.npz")
    old_repeats = B.rows(B.V1 / "control/DAMAGE_GRID_ACTIVE_REPEAT_RESULTS.csv")
    old_cells = {(r["parameter"], float(r["severity"]), float(r["kappa"])): r for r in B.rows(B.V1 / "DAMAGE_GRID_GROUP_RESOLVED.csv")}
    old_rows = B.rows(B.FINAL / "FINAL_GRID_ROUTING.csv")
    null = {k: trace["posterior_means"][[i for i, r in enumerate(old_rows) if r["group"] == "null" and float(r["kappa"]) == k]] for k in B.KAP}
    old_index = {(r["parameter"], float(r["severity"]), float(r["kappa"]), int(r["rep"])): i for i, r in enumerate(old_repeats)}
    grid, repeats, cells, t18, new_grid_calls, new_FE_calls = {}, [], [], [], 0, 0

    def draw(y, state):
        torch.set_rng_state(torch.from_numpy(state.copy()))
        samples = flow.sample(torch.tensor(y, dtype=torch.float32), 1000).numpy()
        assert samples.shape == (1000, 6) and np.isfinite(samples).all()
        d, lt, lq, lw = env["diagnostic"](samples, y)
        generator = torch.Generator(device="cpu")
        generator.set_state(torch.from_numpy(state.copy()))
        latent = torch.randn(1000, 6, generator=generator, dtype=torch.float32)
        assert np.array_equal(generator.get_state().numpy(), torch.get_rng_state().numpy())
        return samples, d, lt, lq, lw, hashlib.sha256(latent.numpy().tobytes()).hexdigest()

    frequency_cache = {}
    for group in ["e1", "e2", "e3"]:
        coord = int(group[1])-1
        for severity in B.SEV:
            for k in B.KAP:
                block = []
                for rep in range(12):
                    i = old_index[(group, severity, k, rep)]
                    source = int(v1["historical_array_indices"][i])
                    theta, freq = v1["truths"][i], v1["frequencies"][i]
                    epsilon, state = trace["epsilon"][source], trace["torch_rng_state"][source]
                    assert np.array_equal(epsilon, v1["epsilon"][i]) and np.array_equal(state, v1["torch_rng_state"][i])
                    noise = k*np.abs(freq)*epsilon
                    y = freq+noise
                    if group == "e2":
                        if severity not in frequency_cache:
                            with torch.no_grad():
                                frequency_cache[severity] = sur(torch.tensor(theta, dtype=torch.float32).reshape(1, -1)).squeeze(0).numpy()
                        assert np.array_equal(freq, frequency_cache[severity])
                        tf, d, lt, lq, lw, latent_hash = draw(y, state); new_grid_calls += 1
                        pm = tf.mean(0)
                    else:
                        assert np.array_equal(y, v1["observations"][i]) and np.array_equal(noise, v1["additive_noise"][i])
                        tf, pm = v1["samples"][i], v1["posterior_means"][i]
                        d = {"ess_ratio": float(old_repeats[i]["raw_IS_ESS_over_N"]),
                             "outside_target_fraction": float(old_repeats[i]["outside_target_fraction"])}
                        lt, lq, lw = v1["log_target"][i], v1["log_q"][i], v1["log_weights"][i]
                        latent_hash = B.read(B.OUT / "DAMAGE_GRID_REPLAY_AUDIT_V2.json")["per_observation_comparisons"][source]["latent_sha256"]
                    item = dict(source=source, old_V1_index=i, theta=theta, frequency=freq, epsilon=epsilon, noise=noise,
                        state=state, observations=y, samples=tf, pm=pm, diag=d, lt=lt, lq=lq, lw=lw, latent_hash=latent_hash)
                    grid[(group, severity, k, rep)] = item; block.append(item)
                    repeats.append(dict(parameter=group, severity=severity, severity_percent=100*severity, kappa=k,
                        kappa_percent=100*k, rep=rep, historical_array_index=source, V1_active_array_index=i,
                        affected_coordinate_mean=float(pm[coord]), fast_posterior_damage_score=float(-pm[coord]),
                        affected_parameter_relative_error=float(abs(pm[coord]-theta[coord])/theta[coord]),
                        raw_IS_ESS_over_N=d["ess_ratio"], route_tau05=route(d["ess_ratio"]),
                        outside_target_fraction=d["outside_target_fraction"], latent_sha256=latent_hash,
                        samples_bitwise_reused_from_V1=group != "e2", old_slot_parameter="e5" if group == "e2" else group,
                        standardized_epsilon_and_latent_state_retained=True, nominal_relative_noise_preserved=True,
                        observation_rule="f_source+kappa*abs(f_source)*epsilon"))
                r = cell_row(group, severity, k, block, null, env)
                cells.append(r)
                if group != "e2":
                    old = old_cells[(group, severity, k)]
                    for metric in ["AUC", "TPR_at_5pct_FPR", "localization_parameter_recovery_relative_error",
                                   "raw_IS_ESS_over_N_mean", "fast_posterior_damage_score_mean"]:
                        assert r[metric] == float(old[metric]), (group, metric)
            sr = [r for r in cells if r["parameter"] == group and r["severity"] == severity]
            t18.append(dict(parameter=group, severity=severity, severity_percent=100*severity, kappa_percent="1;3;5",
                repeats_total=36, averaging="Equal-weight across kappa only, never across structural groups",
                AUC=float(np.mean([r["AUC"] for r in sr])), TPR_at_5pct_FPR=float(np.mean([r["TPR_at_5pct_FPR"] for r in sr])),
                localization_parameter_recovery_error_percent=float(np.mean([r["localization_parameter_recovery_error_percent"] for r in sr])),
                fast_route_count_tau05=sum(r["fast_route_count_tau05"] for r in sr),
                correction_route_count_tau05=sum(r["correction_route_count_tau05"] for r in sr)))
        print("V2_GRID_COMPLETE", group, flush=True)
    assert new_grid_calls == 108
    B.new_csv(B.OUT / "DAMAGE_GRID_GROUP_RESOLVED_V2.csv", cells)
    B.new_csv(B.OUT / "DAMAGE_GRID_FIG20_DATA_V2.csv", [dict(curve_group=r["parameter"], curve_kappa_percent=r["kappa_percent"],
        marker_severity_percent=r["severity_percent"], **r) for r in cells])
    B.new_csv(B.OUT / "DAMAGE_GRID_TABLE18_DATA_V2.csv", t18)
    B.new_csv(B.CONTROL / "DAMAGE_GRID_REPEAT_RESULTS_V2.csv", repeats)
    items = list(grid.values())
    B.new_npz(B.CONTROL / "ACTIVE_DAMAGE_GRID_ARRAYS_V2.npz",
        samples=np.stack([x["samples"] for x in items]), observations=np.stack([x["observations"] for x in items]),
        truths=np.stack([x["theta"] for x in items]), frequencies=np.stack([x["frequency"] for x in items]),
        posterior_means=np.stack([x["pm"] for x in items]), epsilon=np.stack([x["epsilon"] for x in items]),
        additive_noise=np.stack([x["noise"] for x in items]), torch_rng_state=np.stack([x["state"] for x in items]),
        historical_array_indices=np.array([x["source"] for x in items]), V1_active_array_indices=np.array([x["old_V1_index"] for x in items]),
        log_target=np.stack([x["lt"] for x in items]), log_q=np.stack([x["lq"] for x in items]), log_weights=np.stack([x["lw"] for x in items]))
    modal = B.rows(B.OUT / "FE_HOLDOUT_MODAL_COMPARISON.csv")
    pairs, pair_arrays, summaries = [], [], []
    for pnum in range(1, 7):
        point, coord, severity = f"p{pnum:02d}", (pnum-1)//2, .1 if pnum % 2 else .25
        group = f"e{coord+1}"
        mr = sorted([r for r in modal if r["point"] == point], key=lambda r: int(r["mode"]))
        assert len(mr) == 20
        fe = np.loadtxt(mr[0]["FE_source_path"])
        nn = np.array([float(r["surrogate_frequency_hz"]) for r in mr], dtype=np.float32)
        assert np.array_equal(fe, np.array([float(r["FE_frequency_hz"]) for r in mr]))
        for k in B.KAP:
            rs = []
            for rep in range(12):
                item = grid[(group, severity, k, rep)]
                assert np.array_equal(nn, item["frequency"])
                epsilon, state = item["epsilon"], item["state"]
                noise_FE, noise_NN = k*np.abs(fe)*epsilon, k*np.abs(nn)*epsilon
                yf, yn = fe+noise_FE, nn+noise_NN
                assert np.array_equal(yn, item["observations"]) and np.array_equal(noise_NN, item["noise"])
                tf, df, ltf, lqf, lwf, latent_hash = draw(yf, state); new_FE_calls += 1
                assert latent_hash == item["latent_hash"]
                tn, dn, mf, mn = item["samples"], item["diag"], tf.mean(0), item["pm"]
                threshold = float(np.quantile(-null[k][:, coord], .95))
                row = dict(point=point, parameter=group, parameter_value=1-severity, severity_percent=100*severity,
                    kappa=k, kappa_percent=100*k, rep=rep, Nfast=1000,
                    affected_mean_FE=float(mf[coord]), affected_mean_NN=float(mn[coord]),
                    affected_mean_absolute_difference=abs(float(mf[coord])-float(mn[coord])),
                    damage_score_FE=float(-mf[coord]), damage_score_NN=float(-mn[coord]),
                    raw_IS_ESS_over_N_FE=df["ess_ratio"], raw_IS_ESS_over_N_NN=dn["ess_ratio"],
                    route_FE_tau05=route(df["ess_ratio"]), route_NN_tau05=route(dn["ess_ratio"]),
                    route_agreement=route(df["ess_ratio"]) == route(dn["ess_ratio"]),
                    route_changed=route(df["ess_ratio"]) != route(dn["ess_ratio"]),
                    outside_target_fraction_FE=df["outside_target_fraction"], outside_target_fraction_NN=dn["outside_target_fraction"],
                    frozen_null_score_95percent_threshold=threshold, exceeds_frozen_threshold_FE=float(-mf[coord]) > threshold,
                    exceeds_frozen_threshold_NN=float(-mn[coord]) > threshold, standardized_epsilon_identical=True,
                    same_latent_draws=True, source_relative_noise_model_identical=True,
                    additive_Hz_vectors_bitwise_identical=bool(np.array_equal(noise_FE, noise_NN)),
                    NN_samples_reused_from_corrected_grid=True, historical_array_index=item["source"], latent_sha256=latent_hash)
                for j in range(6):
                    row[f"posterior_mean_FE_e{j+1}"] = float(mf[j]); row[f"posterior_mean_NN_e{j+1}"] = float(mn[j])
                rs.append(row)
                pair_arrays.append(dict(samples=np.stack([tf, tn]), observations=np.stack([yf, yn]),
                    noise=np.stack([noise_FE, noise_NN]), frequencies=np.stack([fe, nn.astype(float)]),
                    epsilon=epsilon, state=state, source=item["source"],
                    lt=np.stack([ltf, item["lt"]]), lq=np.stack([lqf, item["lq"]]), lw=np.stack([lwf, item["lw"]])))
            neg = -null[k][:, coord]
            sf, sn = [np.array([r[name] for r in rs], dtype=np.float32) for name in ["damage_score_FE", "damage_score_NN"]]
            af, an = float(env["auc"](sf, neg)), float(env["auc"](sn, neg))
            tpf, tpn = env["tpr_at_fpr"](sf, neg), env["tpr_at_fpr"](sn, neg)
            block = dict(point=point, parameter=group, parameter_value=1-severity, severity_percent=100*severity,
                kappa=k, kappa_percent=100*k, matched_pairs=12, null_repeats=120,
                affected_mean_FE=float(np.mean([r["affected_mean_FE"] for r in rs])), affected_mean_NN=float(np.mean([r["affected_mean_NN"] for r in rs])),
                affected_mean_absolute_difference_mean=float(np.mean([r["affected_mean_absolute_difference"] for r in rs])),
                affected_mean_absolute_difference_max=max(r["affected_mean_absolute_difference"] for r in rs),
                damage_score_FE_mean=float(sf.mean()), damage_score_NN_mean=float(sn.mean()),
                AUC_FE=af, AUC_NN=an, AUC_FE_minus_NN=af-an, TPR_at_5pct_FPR_FE=tpf, TPR_at_5pct_FPR_NN=tpn,
                TPR_FE_minus_NN=tpf-tpn, frozen_null_score_95percent_threshold=float(np.quantile(neg, .95)),
                raw_IS_ESS_over_N_FE_mean=float(np.mean([r["raw_IS_ESS_over_N_FE"] for r in rs])),
                raw_IS_ESS_over_N_NN_mean=float(np.mean([r["raw_IS_ESS_over_N_NN"] for r in rs])),
                routing_agreement_count=sum(r["route_agreement"] for r in rs), route_changes=sum(r["route_changed"] for r in rs),
                fast_route_count_FE_tau05=sum(r["route_FE_tau05"] == "fast" for r in rs),
                fast_route_count_NN_tau05=sum(r["route_NN_tau05"] == "fast" for r in rs),
                routes_FE_by_repeat=json.dumps([r["route_FE_tau05"] for r in rs]), routes_NN_by_repeat=json.dumps([r["route_NN_tau05"] for r in rs]),
                standardized_epsilon_and_latent_draws_identical=True, relative_noise_scaled_by_own_frequency_vector=True)
            summaries.append(block)
            for r in rs:
                r.update(AUC_FE_point_kappa=af, AUC_NN_point_kappa=an, TPR_FE_point_kappa=tpf, TPR_NN_point_kappa=tpn)
            pairs.extend(rs)
        print("V2_MATCHED_FE_COMPLETE", point, flush=True)
    assert new_FE_calls == len(pairs) == 216 and len(summaries) == 18
    B.new_csv(B.OUT / "FE_HOLDOUT_MATCHED_INFERENCE_V2.csv", pairs)
    B.new_csv(B.OUT / "FE_HOLDOUT_SUMMARY_V2.csv", summaries)
    B.new_npz(B.CONTROL / "FE_HOLDOUT_MATCHED_ARRAYS_V2.npz",
        samples=np.stack([x["samples"] for x in pair_arrays]), observations=np.stack([x["observations"] for x in pair_arrays]),
        additive_noise=np.stack([x["noise"] for x in pair_arrays]), frequencies=np.stack([x["frequencies"] for x in pair_arrays]),
        epsilon=np.stack([x["epsilon"] for x in pair_arrays]), torch_rng_state=np.stack([x["state"] for x in pair_arrays]),
        historical_array_indices=np.array([x["source"] for x in pair_arrays]),
        log_target=np.stack([x["lt"] for x in pair_arrays]), log_q=np.stack([x["lq"] for x in pair_arrays]),
        log_weights=np.stack([x["lw"] for x in pair_arrays]))
    overall = dict(matched_pairs=216, mean_affected_coordinate_absolute_difference=float(np.mean([r["affected_mean_absolute_difference"] for r in pairs])),
        maximum_affected_coordinate_absolute_difference=max(r["affected_mean_absolute_difference"] for r in pairs),
        route_agreement_count=sum(r["route_agreement"] for r in pairs), route_changes=sum(r["route_changed"] for r in pairs),
        AUC_FE_minus_NN_min=min(r["AUC_FE_minus_NN"] for r in summaries), AUC_FE_minus_NN_max=max(r["AUC_FE_minus_NN"] for r in summaries),
        AUC_FE_minus_NN_mean=float(np.mean([r["AUC_FE_minus_NN"] for r in summaries])),
        TPR_FE_minus_NN_min=min(r["TPR_FE_minus_NN"] for r in summaries), TPR_FE_minus_NN_max=max(r["TPR_FE_minus_NN"] for r in summaries),
        point_kappa_blocks_with_TPR_difference=sum(r["TPR_FE_minus_NN"] != 0 for r in summaries))
    B.new_json(B.CONTROL / "INFERENCE_AUDIT_V2.json", dict(finished_utc=B.now(), runtime_seconds=time.perf_counter()-start,
        historical_provenance_inverse_verification_calls=684, new_e2_grid_flow_sample_calls=new_grid_calls,
        new_FE_matched_flow_sample_calls=new_FE_calls, new_e1_e3_or_NN_matched_flow_sample_calls=0,
        new_training_calls=0, new_MCMC_calls=0, new_FE_solve_calls=0, new_PSIS_calls=0, new_MATLAB_calls=0, new_random_seed_designs=0,
        V1_classification=B.CLASSIFICATION, V1_fully_preserved=True, direct_hash_verified_copies=copies,
        overall=overall, table18=t18, negative_proposals="Retained in raw fast means; zero target weight for nonpositive support",
        diagnostic_target=dict(K=20, sigma_log=.55, kappa_sq=.001, prior="Independent untruncated lognormal", tau=.5),
        preservation_after=B.verify(snapshot), controller_sha256=B.sha(__file__)))
    print("V2_MATCHED_OVERALL", json.dumps(overall), flush=True)


if __name__ == "__main__":
    main()
