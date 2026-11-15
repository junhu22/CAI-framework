"""Authorized historical replay only; exact gate precedes every next chain."""
import ast
import csv
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pyro
import torch
from pyro.infer import MCMC, NUTS

CONTROL = Path(__file__).resolve().parent
OUT = CONTROL.parent
QUALITY = OUT.parent
HERE = QUALITY.parent
ROOT = HERE.parents[2]
RUNNER = HERE / "run_reference_nuts.py"
spec = importlib.util.spec_from_file_location("frozen_pilot_helpers", OUT / "control/run_pilot.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)  # Definitions only: frozen pilot main is never called.
TARGETS = [f"ablation_obs{i}" for i in range(1, 6)] + ["sailboat_field_frozen", "sailboat_boundary_e1_075"]
SEEDS = [701, 702, 703]
CSV = OUT / "HISTORICAL_DIAGNOSTICS_RECOVERY.csv"
AUDIT = OUT / "HISTORICAL_DIAGNOSTICS_RECOVERY_AUDIT.md"
INTEGRITY = OUT / "HISTORICAL_DIAGNOSTICS_RECOVERY_INTEGRITY.json"


def write_csv(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
    writer.writeheader()
    writer.writerows(rows)
    P.write_new_bytes(path, stream.getvalue().encode("utf-8"))


def pilot_record():
    path = OUT / "PILOT_ablation_obs1_seed701.json"
    pilot = json.loads(path.read_text(encoding="utf-8"))
    assert pilot["classification"] == "EXACT_HISTORICAL_CHAIN_REPRODUCTION"
    assert pilot["comparison"]["np_array_equal"] and pilot["comparison"]["max_abs_difference"] == 0
    assert P.sha(pilot["artifact_path"]) == pilot["artifact_sha256"]
    with np.load(pilot["artifact_path"], allow_pickle=False) as saved, np.load(HERE / "REFERENCE_ablation_obs1.npz", allow_pickle=False) as old:
        assert np.array_equal(saved["replay_theta"], old["chains"][0])
    return pilot


def build_manifest():
    known = json.loads((QUALITY / "control/TARGET_MANIFEST.json").read_text(encoding="utf-8"))
    old_environment = json.loads((HERE / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert torch.__version__ == old_environment["torch"] == known["torch_version"] == "2.6.0+cu124"
    assert pyro.__version__ == known["pyro_version"] == "1.9.1"
    assert np.__version__ == old_environment["numpy"] == "2.4.4"
    assert P.sha(RUNNER) == known["sampler_source_sha256"]
    assert P.sha(ROOT / "experiments/cai_lib.py") == known["library_sha256"]
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    stage1 = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "S1" for t in n.targets))
    routing = Path(ast.literal_eval(stage1.args[0])) / "routing_arrays"
    historical_log = P.log_text(HERE / "reference_nuts.log").splitlines()
    manifests, inputs = [], []
    for name in TARGETS:
        frozen = next(t for t in known["targets"] if t["target"] == name)
        assert P.sha(frozen["original_reference_path"]) == frozen["original_reference_sha256"]
        assert P.sha(frozen["source_surrogate_path"]) == frozen["source_surrogate_sha256"]
        with np.load(frozen["original_reference_path"], allow_pickle=False) as saved:
            observation, old_fast = saved["observation"].copy(), saved["fast"].copy()
            assert saved["chains"].shape == (3, 600, 6)
        assert np.array_equal(observation, np.asarray(frozen["observation"]))
        chunks, fast_inputs = [], []
        for fs in [42, 43, 44, 45, 46]:
            path = routing / f"{name}_seed{fs}_N8000.npz"
            with np.load(path, allow_pickle=False) as data:
                theta = data["theta"].copy()
                assert theta.shape == (8000, 6) and np.array_equal(data["observation"].astype(float), observation)
            chunks.append(theta)
            entry = dict(path=str(path), sha256=P.sha(path), fast_seed=fs, theta_dtype=str(theta.dtype), theta_shape=list(theta.shape))
            fast_inputs.append(entry)
            inputs.append(path)
        fast = np.concatenate(chunks, axis=0)
        init_fast = np.clip(np.nanmean(fast, axis=0), .15, 5.)
        assert np.array_equal(fast, old_fast) and np.isfinite(init_fast).all()
        header = next(line for line in historical_log if line.startswith(f"TARGET {name} K ") and " fast_mean " in line)
        assert np.array_equal(init_fast, np.asarray(ast.literal_eval(header.split("fast_mean ", 1)[1])))
        for index, seed in enumerate(SEEDS):
            if name == "ablation_obs1" and index == 0:
                continue
            init = [init_fast, np.ones(6), np.exp(np.linspace(-.20, .20, 6))][index]
            manifest = dict(target=name, historical_chain_index=index, seed=seed,
                surrogate_path=frozen["source_surrogate_path"], surrogate_sha256=frozen["source_surrogate_sha256"],
                reference_path=frozen["original_reference_path"], reference_sha256=frozen["original_reference_sha256"],
                runner_path=str(RUNNER), runner_sha256=P.sha(RUNNER), library_sha256=known["library_sha256"],
                K=frozen["K"], observation=observation.tolist(), observation_dtype=str(observation.dtype),
                sigma_log=frozen["sigma_log"], kappa_sq=.001, prior="independent untruncated lognormal",
                prior_log_mean=0., parameterization="log-theta", sampler="Pyro NUTS", target_accept_prob=.8,
                num_samples=600, warmup_steps=500, CPU=True, torch_threads=6,
                environment=dict(python=sys.executable, torch_version=torch.__version__, pyro_version=pyro.__version__, numpy_version=np.__version__),
                init_rule=["clip(nanmean(concatenate(seed42..46 theta),axis=0),0.15,5.0)", "np.ones(6)", "np.exp(np.linspace(-0.20,0.20,6))"][index],
                rebuilt_init=init.tolist(), init_dtype=str(init.dtype), init_log_float32=torch.log(torch.tensor(init, dtype=torch.float32)).tolist(),
                init_inputs=fast_inputs if index == 0 else [], reconstructed_fast_exactly_matches_saved_reference_fast=True,
                reconstructed_init_fast_exactly_matches_historical_logged_init=True,
                RNG_rule=f"Original run_chain torch.manual_seed({seed}); no other RNG reset",
                diagnostic_chain_key="chain 0", exact_rule="np.array_equal == True OR max_abs_difference == 0")
            manifests.append(manifest)
    assert len(manifests) == 20 and len({(m["target"], m["seed"]) for m in manifests}) == 20
    return manifests, inputs, tree


def freeze(inputs):
    previous = json.loads((OUT / "control/FROZEN_BEFORE_PILOT.json").read_text(encoding="utf-8"))
    P.verify_preservation(previous)
    paths = [p for p in HERE.rglob("*") if p.is_file() and CONTROL not in p.parents and HERE / "psis_env" not in p.parents]
    paths += [Path(p) for p in previous["files"]] + inputs
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=ROOT).decode().strip())
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo).decode("utf-8").split("\0")
    tracked_files = [repo / path for path in tracked if path and (repo / path).is_file()]
    paths += tracked_files
    snapshot = dict(created_utc=P.utc_now(), files={str(p): P.sha(p) for p in sorted(set(paths))},
                    original_367_files=previous["original_367_files"], git_tracked_before=P.git_state(),
                    git_tracked_file_hashes={str(p): P.sha(p) for p in tracked_files},
                    frozen_pilot_files={str(p): P.sha(p) for p in OUT.rglob("*") if p.is_file() and CONTROL not in p.parents})
    P.write_new_json(CONTROL / "FROZEN_BEFORE_FULL_RECOVERY.json", snapshot)
    return snapshot


def save_chain(result, samples, old, init, observation, runtime, raw):
    manifest = result["manifest"]
    comparison = P.compare(samples, old)
    exact = comparison["exact_reproduction_pass"]
    indices = list(raw["divergences"]["chain 0"])
    acceptance = raw["acceptance rate"]["chain 0"]
    diagnostics = P.plain(raw)
    buffer = io.BytesIO()
    torch.save(raw, buffer)
    raw_bytes = buffer.getvalue()
    path = OUT / f"RECOVER_{manifest['target']}_seed{manifest['seed']}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, replay_theta=samples, historical_theta=old, theta_init=init, observation=observation,
            seed=np.asarray(manifest["seed"]), historical_chain_index=np.asarray(manifest["historical_chain_index"]),
            runtime_s=np.asarray(runtime), divergences=np.asarray(indices, dtype=np.int64), acceptance_rate=np.asarray(acceptance),
            raw_mc_diagnostics_torch_bytes=np.frombuffer(raw_bytes, dtype=np.uint8), raw_mc_diagnostics_json=np.asarray(json.dumps(diagnostics)),
            comparison_json=np.asarray(json.dumps(comparison)), manifest_json=np.asarray(json.dumps(manifest)),
            source_reference_sha256=np.asarray(manifest["reference_sha256"]), source_runner_sha256=np.asarray(manifest["runner_sha256"]),
            source_surrogate_sha256=np.asarray(manifest["surrogate_sha256"]))
        stream.flush()
        os.fsync(stream.fileno())
    result.update(status="COMPLETED", classification=comparison["classification"], comparison=comparison,
        runtime_s=runtime, replay_theta_all_finite=bool(np.isfinite(samples).all()), exact_reproduction_pass=exact,
        raw_mc_diagnostics=diagnostics, raw_diagnostics_serialized_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        replay_divergence_count=len(indices), replay_divergence_indices=indices, replay_acceptance_rate=acceptance,
        recovered_historical_divergence_count=len(indices) if exact else None,
        recovered_historical_divergence_indices=indices if exact else None,
        recovered_historical_acceptance_rate=acceptance if exact else None,
        replay_diagnostics_attributed_to_historical_chain=exact, artifact_path=str(path), artifact_sha256=P.sha(path))


def write_reports(pilot, results, snapshot, failure):
    records = [pilot] + results
    lookup = {(r["manifest"]["target"], r["manifest"]["seed"]): r for r in records}
    exact_count = sum(r["exact_reproduction_pass"] for r in records)
    all_exact = exact_count == 21
    known_total = sum(r["recovered_historical_divergence_count"] for r in records if r["recovered_historical_divergence_count"] is not None)
    accepts = [r["recovered_historical_acceptance_rate"] for r in records if r["recovered_historical_acceptance_rate"] is not None]
    rows = []
    for target in TARGETS:
        row = dict(target=target)
        target_records = [lookup.get((target, seed)) for seed in SEEDS]
        valid = all(r is not None and r["exact_reproduction_pass"] for r in target_records)
        for seed, record in zip(SEEDS, target_records):
            row[f"seed{seed}_status"] = record["classification"] if record else "NOT_RUN"
            row[f"seed{seed}_divergence_count"] = record["recovered_historical_divergence_count"] if record else None
            row[f"seed{seed}_divergence_indices"] = json.dumps(record.get("recovered_historical_divergence_indices", record.get("replay_divergence_indices"))) if record and record["exact_reproduction_pass"] else None
            row[f"seed{seed}_acceptance_rate"] = record["recovered_historical_acceptance_rate"] if record else None
            for key in ["shape_identical", "dtype_identical", "np_array_equal", "max_abs_difference", "mean_abs_difference", "number_of_different_elements", "allclose_atol0_rtol0", "allclose_atol1e_minus8_rtol1e_minus7"]:
                row[f"seed{seed}_{key}"] = record.get("comparison", {}).get(key) if record else None
        row.update(historical_3chain_total_divergences=sum(r["recovered_historical_divergence_count"] for r in target_records) if valid else None,
                   all_three_exact_reproduced=valid)
        rows.append(row)
    write_csv(CSV, rows)
    preservation = P.verify_preservation(snapshot)
    summary = dict(exact_reproduced_chains=exact_count, expected_historical_chains=21, all_21_exact_reproduced=all_exact,
        remaining_chain_calls=len(results), authorized_remaining_chain_calls=20, historical_divergences_total=known_total if all_exact else None,
        exact_reproduced_subset_divergences=known_total, min_acceptance_rate=min(accepts), max_acceptance_rate=max(accepts),
        any_exact_replay_failure=bool(failure), failure_position=failure, mcmc_stopped=True,
        eligible_for_four_chain_quality_recovery=all_exact, automatic_retries=0, pilot_chain_reruns=0,
        flow_training_calls=0, flow_resampling_calls=0, new_seed_calls=0, fifth_chain_calls=0)
    lines = [f"1. Exact historical reproduction: {exact_count}/21; all exact={all_exact}.",
        f"2. Historical 21-chain divergences: {summary['historical_divergences_total']}; attributed exact subset={known_total}.",
        f"3. Historical acceptance rate min/max: {min(accepts)} / {max(accepts)}.",
        f"4. Any exact replay failure: {bool(failure)}; position={json.dumps(failure)}.", "",
        "The frozen pilot was read, verified and retained; it was never rerun. This execution makes at most the authorized remaining 20 original-chain calls.",
        "Each chain directly calls the AST-extracted, unmodified run_reference_nuts.py run_chain. The original target, CPU/6 threads, Torch 2.6.0+cu124, Pyro 1.9.1, 600 draws/500 warmup and target_accept_prob=0.8 are retained.",
        "Seed701 init is rebuilt from the five seed42..46 N8000 routing files in ascending order and matches the historical saved fast array and logged init exactly; seed702 uses np.ones(6); seed703 uses exp(linspace(-0.20,0.20,6)). Original init dtype is retained before the original float32 conversion.",
        "Every replay records shape/dtype, array_equal, maximum/mean absolute difference, changed-element count and both requested allclose checks. Only array_equal=True or zero maximum difference attributes diagnostics to history. A failure stops all subsequent MCMC without retry or parameter changes.",
        'Raw mc.diagnostics() is preserved both as original torch.save bytes in each NPZ and as the original hierarchy/value JSON. Divergences and acceptance rate are read from "chain 0", never the former erroneous "z" keys.',
        "The old zero-divergence fallback logs and old 0/7 FAIL audit remain unchanged. The recovered evidence is a separate exact-replay provenance record, not a rewrite of the historical log.",
        f"Integrity: {preservation['protected_files']} protected assets and all tracked-file hashes unchanged; original 367 assets, seven historical references, seven seed704 chains, 24 scientific flow checkpoints, submitted baseline and all pilot files retained.",
        "No flow training, new experiment flow draws, new seeds, fifth chains, FE, damage grid, manuscript/response modifications or Git mutations were performed.", "",
        "Four-chain gate reevaluation is permitted only after 21/21 exact reproductions; R2.1/R2.3 reevaluation is additionally conditional on 7/7 complete gate PASS.", ""]
    P.write_new_bytes(AUDIT, "\n".join(lines).encode("utf-8"))
    integrity = dict(schema_version=1, created_utc=P.utc_now(), summary=summary, target_rows=rows,
        chain_results=records, protected_files=snapshot["files"], frozen_pilot_files=snapshot["frozen_pilot_files"],
        git_tracked_file_hashes=snapshot["git_tracked_file_hashes"], original_367_files=snapshot["original_367_files"],
        preservation_after=preservation, controller_path=str(Path(__file__)), controller_sha256=P.sha(__file__),
        outputs={p.name: dict(path=str(p), sha256=P.sha(p)) for p in [CSV, AUDIT]}, scope=summary)
    P.write_new_json(INTEGRITY, integrity)
    P.write_new_json(CONTROL / "REMAINING20_FINISHED.json", summary)
    print("HISTORICAL_RECOVERY_FINISHED", json.dumps(summary), flush=True)


def main():
    for path in [CSV, AUDIT, INTEGRITY, CONTROL / "REMAINING20_STARTED.json", CONTROL / "FROZEN_BEFORE_FULL_RECOVERY.json", CONTROL / "REMAINING20_FINISHED.json"]:
        assert not path.exists(), "Existing attempt must be preserved; no automatic replay: " + str(path)
    assert not list(OUT.glob("RECOVER_*")), "Preserve all previous recovery attempts; do not retry."
    torch.set_num_threads(6)
    pilot = pilot_record()
    manifests, inputs, tree = build_manifest()
    snapshot = freeze(inputs)
    P.write_new_json(CONTROL / "REMAINING20_MANIFEST.json", dict(created_utc=P.utc_now(), remaining_chain_count=20, pilot_frozen=True, chains=manifests))
    P.write_new_json(CONTROL / "REMAINING20_STARTED.json", dict(started_utc=P.utc_now(), pid=os.getpid(), authorized_calls=20, controller_sha256=P.sha(__file__)))
    sys.path.insert(0, str(ROOT / "experiments"))
    from cai_lib import load_surrogate
    models = {path: load_surrogate(path) for path in dict.fromkeys(m["surrogate_path"] for m in manifests)}
    assert all(torch.isfinite(t).all().item() for model in models.values() for t in model.state_dict().values())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_chain")
    env = dict(torch=torch, MCMC=MCMC, NUTS=NUTS, time=time, NSAMP=600, NWARM=500)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(RUNNER) + ":frozen_historical_run_chain", "exec"), env)
    results, failure = [], None
    for manifest in manifests:
        name, seed, index = manifest["target"], manifest["seed"], manifest["historical_chain_index"]
        with np.load(manifest["reference_path"], allow_pickle=False) as original:
            observation, old = original["observation"].copy(), original["chains"][index].copy()
        init = np.asarray(manifest["rebuilt_init"], dtype=manifest["init_dtype"])
        attempt = dict(started_utc=P.utc_now(), target=name, historical_chain_index=index, seed=seed, call_number=len(results) + 1, pid=os.getpid())
        P.write_new_json(CONTROL / f"ATTEMPT_{name}_seed{seed}.json", attempt)
        print("HISTORICAL_CHAIN_START", len(results) + 1, "/20", name, index, seed, flush=True)
        result = dict(manifest=manifest, attempt=attempt, status="FAILED", exact_reproduction_pass=False,
            classification="HISTORICAL_CHAIN_NOT_EXACTLY_REPRODUCED", recovered_historical_divergence_count=None,
            recovered_historical_acceptance_rate=None, replay_diagnostics_attributed_to_historical_chain=False)
        try:
            # No sampler hook, observer or file operation occurs inside this frozen scientific function.
            samples, runtime, raw = env["run_chain"](models[manifest["surrogate_path"]], observation, manifest["sigma_log"], init, seed)
            save_chain(result, samples, old, init, observation, runtime, raw)
        except Exception as error:
            result.update(error_type=type(error).__name__, original_error=str(error), traceback=traceback.format_exc())
        result.update(finished_utc=P.utc_now(), preservation_after=P.verify_preservation(snapshot))
        base = OUT / f"RECOVER_{name}_seed{seed}"
        P.write_new_bytes(base.with_suffix(".log"), (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        result.update(log_path=str(base.with_suffix(".log")), log_sha256=P.sha(base.with_suffix(".log")))
        P.write_new_json(base.with_suffix(".json"), result)
        results.append(result)
        print("HISTORICAL_CHAIN_RESULT", name, seed, result["classification"], result.get("comparison", {}).get("max_abs_difference"), result["recovered_historical_divergence_count"], result.get("replay_acceptance_rate"), flush=True)
        if not result["exact_reproduction_pass"]:
            failure = dict(target=name, historical_chain_index=index, seed=seed, call_number=len(results), classification=result["classification"], comparison=result.get("comparison"), original_error=result.get("original_error"))
            break
    write_reports(pilot, results, snapshot, failure)


if __name__ == "__main__":
    main()
