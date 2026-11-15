"""Exactly seven authorized fourth chains; no flow training or resampling."""
import ast
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import pyro
from pyro.infer import MCMC, NUTS

ROOT = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
HERE = ROOT / "revision_experiments/cai_r2_stage1/final_ext_verification_20260918_091227"
OUT = HERE / "reference_quality_4chain"
CONTROL = OUT / "control"
TARGETS = [f"ablation_obs{i}" for i in range(1, 6)] + ["sailboat_field_frozen", "sailboat_boundary_e1_075"]
SEED, NSAMP, NWARM = 704, 600, 500
THETA_INIT = np.exp(np.linspace(0.20, -0.20, 6))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state():
    return subprocess.check_output(["git", "status", "--porcelain", "-uno"], cwd=ROOT,
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).decode("utf-8")


def write_new_bytes(path, data):
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def write_new_json(path, value):
    write_new_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8"))


def plain(value):
    if isinstance(value, torch.Tensor):
        return plain(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_original_value": repr(value)}
    return value


def verify_preservation(snapshot):
    changed = [path for path, digest in snapshot["files"].items()
               if not Path(path).is_file() or sha(path) != digest]
    assert not changed, "Historical asset changed: " + repr(changed)
    assert git_state() == snapshot["git_tracked_before"]
    return dict(checked_utc=utc_now(), checked_files=len(snapshot["files"]),
                all_sha256_unchanged=True, changed_files=[], git_tracked_unchanged=True)


def build_manifest():
    entries = []
    for name in TARGETS:
        source = HERE / f"REFERENCE_{name}.npz"
        surrogate = ROOT / ("surrogate_models/member_0.pt" if name.startswith("ablation")
                            else "surrogate_models_broad_ext/member_0.pt")
        sigma = 0.55 if name == "sailboat_boundary_e1_075" else 0.30
        with np.load(source, allow_pickle=False) as stored:
            observation = stored["observation"].copy()
            assert stored["chains"].shape == (3, 600, 6)
            assert np.isfinite(stored["chains"]).all() and (stored["chains"] > 0).all()
        k = 6 if name == "sailboat_field_frozen" else 20
        assert observation.shape == (k,) and np.isfinite(observation).all() and (observation > 0).all()
        if name == "sailboat_field_frozen":
            assert np.array_equal(observation.astype(np.float32), np.asarray([.91,1.08,1.47,2.55,3.16,3.81], np.float32))
        entries.append(dict(target=name, original_reference_path=str(source), original_reference_sha256=sha(source),
                            source_surrogate_path=str(surrogate), source_surrogate_sha256=sha(surrogate),
                            K=k, observation=observation.tolist(), observation_dtype=str(observation.dtype),
                            observation_used_float32=observation.astype(np.float32).tolist(),
                            prior="independent untruncated lognormal", prior_log_mean=0.0,
                            sigma_log=sigma, kappa_sq=0.001, parameterization="log-theta",
                            likelihood="0.5 * sum(((surrogate(exp(z))[:K]-observation)/observation)**2) / kappa_sq",
                            negative_log_prior_in_z="0.5 * sum(z**2) / sigma_log**2",
                            lognormal_jacobian_cancelled_in_log_theta=True,
                            seed=SEED, num_samples=NSAMP, warmup_steps=NWARM, target_accept_prob=0.8,
                            theta_init=THETA_INIT.tolist(), theta_init_used_float32=THETA_INIT.astype(np.float32).tolist(),
                            log_theta_init_used_float32=torch.log(torch.tensor(THETA_INIT, dtype=torch.float32)).tolist(),
                            original_chain_seeds=[701, 702, 703], original_chain_shape=[3, 600, 6]))
    return dict(created_utc=utc_now(), authorized_fourth_chains=7, additional_chains_beyond_fourth=0,
                initialization_predeclared=True, targets=entries,
                sampler_source_path=str(HERE/"run_reference_nuts.py"),
                sampler_source_sha256=sha(HERE/"run_reference_nuts.py"),
                library_path=str(ROOT/"experiments/cai_lib.py"), library_sha256=sha(ROOT/"experiments/cai_lib.py"),
                torch_version=torch.__version__, pyro_version=pyro.__version__, numpy_version=np.__version__,
                torch_threads=6, device="cpu", flow_training_calls=0)


def main():
    for name in TARGETS:
        for suffix in [".npz", "_RUN.json", ".log"]:
            assert not (OUT/f"REFERENCE4_{name}{suffix}").exists(), "Preserve existing fourth-chain artifact; no automatic rerun."
        assert not (CONTROL/f"ATTEMPT_{name}.json").exists(), "Preserve existing attempt; no retry."
    assert not (CONTROL/"TARGET_MANIFEST.json").exists()
    assert not (CONTROL/"FROZEN_BEFORE_REFERENCE_QUALITY.json").exists()
    torch.set_num_threads(6)
    manifest = build_manifest()
    baseline = json.loads((HERE/"ablation_multiseed/phase_b_control/FROZEN_INPUTS.json").read_text(encoding="utf-8"))
    historical = [path for path in HERE.rglob("*") if path.is_file()
                  and OUT not in path.parents and HERE/"psis_env" not in path.parents]
    historical += [Path(path) for path in baseline["files"]]
    historical += [Path(entry["source_surrogate_path"]) for entry in manifest["targets"]]
    historical += [ROOT/name for name in subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT,
                   env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).decode("utf-8").split("\0") if name and (ROOT/name).is_file()]
    snapshot = dict(created_utc=utc_now(), files={str(path): sha(path) for path in sorted(set(historical))},
                    original_367_files=baseline["files"], original_367_count=len(baseline["files"]),
                    git_tracked_before=git_state(), environment_directory_excluded=str(HERE/"psis_env"))
    assert snapshot["original_367_count"] == 367
    assert all(snapshot["files"][path] == digest for path, digest in baseline["files"].items())
    write_new_json(CONTROL/"FROZEN_BEFORE_REFERENCE_QUALITY.json", snapshot)
    write_new_json(CONTROL/"TARGET_MANIFEST.json", manifest)
    verify_preservation(snapshot)
    sys.path.insert(0, str(ROOT/"experiments"))
    from cai_lib import load_surrogate
    models = {path: load_surrogate(path) for path in sorted({entry["source_surrogate_path"]
                                                          for entry in manifest["targets"]})}
    assert all(torch.isfinite(value).all().item() for model in models.values() for value in model.state_dict().values())
    tree = ast.parse((HERE/"run_reference_nuts.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_chain")
    environment = dict(torch=torch, MCMC=MCMC, NUTS=NUTS, time=time, NSAMP=NSAMP, NWARM=NWARM)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(HERE/"run_reference_nuts.py")+":frozen_run_chain", "exec"), environment)
    statuses = []
    print("AUTHORIZED_FOURTH_CHAINS_STARTED", os.getpid(), flush=True)
    for entry in manifest["targets"]:
        name = entry["target"]
        attempt = dict(target=name, seed=SEED, pid=os.getpid(), started_utc=utc_now(), manual_authorization=True,
                       training_calls=0, automatic_retries=0, initialization=THETA_INIT.tolist())
        write_new_json(CONTROL/f"ATTEMPT_{name}.json", attempt)
        result = dict(target_manifest=entry, attempt=attempt, status="FAILED", sampler_source=manifest,
                      controller_source_sha256=sha(__file__), original_reference_preserved=True)
        print("FOURTH_CHAIN_START", name, flush=True)
        try:
            pyro.set_rng_seed(SEED)
            samples, runtime, diagnostics = environment["run_chain"](
                models[entry["source_surrogate_path"]], np.asarray(entry["observation"]),
                entry["sigma_log"], THETA_INIT.copy(), SEED)
            diagnostics = plain(diagnostics)
            divergences = diagnostics["divergences"]["chain 0"]
            with np.load(entry["original_reference_path"], allow_pickle=False) as original:
                combined = np.concatenate([original["chains"], samples[None, :, :]], axis=0)
            sample_finite = bool(np.isfinite(samples).all() and (samples > 0).all())
            assert samples.shape == (600, 6) and combined.shape == (4, 600, 6)
            artifact = OUT/f"REFERENCE4_{name}.npz"
            with artifact.open("xb") as stream:
                np.savez_compressed(stream, theta_samples=samples, chains=combined,
                                    chain_seeds=np.asarray([701,702,703,704]), seed=np.asarray(SEED),
                                    theta_init=THETA_INIT, theta_init_used_float32=THETA_INIT.astype(np.float32),
                                    observation=np.asarray(entry["observation"]), runtime_s=np.asarray(runtime),
                                    divergences=np.asarray(divergences, dtype=np.int64),
                                    target_manifest_json=np.asarray(json.dumps(entry, ensure_ascii=False)),
                                    sampler_diagnostics_json=np.asarray(json.dumps(diagnostics, ensure_ascii=False)),
                                    original_reference_sha256=np.asarray(entry["original_reference_sha256"]),
                                    source_surrogate_sha256=np.asarray(entry["source_surrogate_sha256"]))
                stream.flush()
                os.fsync(stream.fileno())
            result.update(status="COMPLETED" if sample_finite else "FAILED_NONFINITE_SAMPLES",
                          runtime_s=runtime, samples_shape=list(samples.shape), combined_chains_shape=list(combined.shape),
                          samples_finite_and_positive=sample_finite, sampler_diagnostics=diagnostics,
                          fourth_chain_divergence_count=len(divergences), fourth_chain_divergence_indices=divergences,
                          acceptance_rate=diagnostics["acceptance rate"]["chain 0"],
                          artifact_path=str(artifact), artifact_sha256=sha(artifact),
                          scientific_chain_calls=1, frozen_run_chain_function_called=True,
                          original_reference_runner_top_level_executed=False, progress_hook_used=False)
        except Exception as error:
            result.update(error_type=type(error).__name__, original_error=str(error), traceback=traceback.format_exc())
        result.update(finished_utc=utc_now(), preservation_after=verify_preservation(snapshot))
        log = OUT/f"REFERENCE4_{name}.log"
        write_new_bytes(log, (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8"))
        result.update(log_path=str(log), log_sha256=sha(log))
        write_new_json(OUT/f"REFERENCE4_{name}_RUN.json", result)
        statuses.append(dict(target=name, status=result["status"], run_json=str(OUT/f"REFERENCE4_{name}_RUN.json")))
        print("FOURTH_CHAIN_FINISH", name, result["status"], result.get("fourth_chain_divergence_count"), flush=True)
    write_new_json(CONTROL/"FOURTH_CHAINS_FINISHED.json", dict(finished_utc=utc_now(), statuses=statuses,
                   authorized_chain_attempts=7, training_calls=0, additional_chains=0,
                   preservation_after=verify_preservation(snapshot), mcmc_stopped=True))
    print("ALL_SEVEN_AUTHORIZED_ATTEMPTS_FINISHED_MCMC_STOPPED", flush=True)


if __name__ == "__main__":
    main()
