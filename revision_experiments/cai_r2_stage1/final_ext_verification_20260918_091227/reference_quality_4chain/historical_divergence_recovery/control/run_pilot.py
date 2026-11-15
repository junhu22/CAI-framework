"""Only ablation_obs1 chain0/seed701; stop after one exact-reproduction pilot."""
import ast
import hashlib
import io
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
HERE = ROOT/"revision_experiments/cai_r2_stage1/final_ext_verification_20260918_091227"
QUALITY = HERE/"reference_quality_4chain"
OUT = QUALITY/"historical_divergence_recovery"
CONTROL = OUT/"control"
REFERENCE = HERE/"REFERENCE_ablation_obs1.npz"
RUNNER = HERE/"run_reference_nuts.py"
SURROGATE = ROOT/"surrogate_models/member_0.pt"
SEED = 701
NPZ = OUT/"PILOT_ablation_obs1_seed701.npz"
RESULT = OUT/"PILOT_ablation_obs1_seed701.json"
LOG = OUT/"PILOT_ablation_obs1_seed701.log"
AUDIT = OUT/"PILOT_REPRODUCTION_AUDIT.md"
INTEGRITY = OUT/"PILOT_REPRODUCTION_INTEGRITY.json"


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


def write_new_bytes(path, data):
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def write_new_json(path, data):
    write_new_bytes(path, (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8"))


def plain(value):
    if isinstance(value, torch.Tensor):
        return plain(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite_original_value": repr(value)}
    return value


def verify_preservation(snapshot):
    changed = [path for path, digest in snapshot["files"].items()
               if not Path(path).is_file() or sha(path) != digest]
    assert not changed, changed
    assert git_state() == snapshot["git_tracked_before"]
    return dict(checked_utc=utc_now(), protected_files=len(snapshot["files"]),
                all_sha256_unchanged=True, changed_files=[], git_tracked_unchanged=True)


def log_text(path):
    data = Path(path).read_bytes()
    encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    return data.decode(encoding)


def compare(samples, original):
    shapes_match = samples.shape == original.shape
    arrays_equal = bool(np.array_equal(samples, original))
    if shapes_match:
        difference = np.abs(samples.astype(np.float64)-original.astype(np.float64))
        max_difference = float(difference.max())
        mean_difference = float(difference.mean())
        different = int(np.count_nonzero(samples != original))
    else:
        max_difference = mean_difference = different = None
    exact = shapes_match and (arrays_equal or max_difference == 0)
    return dict(shape_identical=shapes_match, original_shape=list(original.shape), replay_shape=list(samples.shape),
                original_dtype=str(original.dtype), replay_dtype=str(samples.dtype), dtype_identical=samples.dtype == original.dtype,
                np_array_equal=arrays_equal, max_abs_difference=max_difference, mean_abs_difference=mean_difference,
                number_of_different_elements=different, total_elements=int(original.size),
                allclose_atol0_rtol0=bool(shapes_match and np.allclose(samples, original, atol=0, rtol=0)),
                allclose_atol1e_minus8_rtol1e_minus7=bool(shapes_match and np.allclose(samples, original, atol=1e-8, rtol=1e-7)),
                exact_reproduction_pass=exact,
                classification="EXACT_HISTORICAL_CHAIN_REPRODUCTION" if exact
                else "HISTORICAL_CHAIN_NOT_EXACTLY_REPRODUCED")


def main():
    for path in [NPZ, RESULT, LOG, AUDIT, INTEGRITY, CONTROL/"ATTEMPT_STARTED.json",
                 CONTROL/"PILOT_MANIFEST.json", CONTROL/"FROZEN_BEFORE_PILOT.json"]:
        assert not path.exists(), "Preserve existing pilot attempt; no automatic rerun: "+str(path)
    torch.set_num_threads(6)
    known_environment = json.loads((QUALITY/"control/TARGET_MANIFEST.json").read_text(encoding="utf-8"))
    historical_environment = json.loads((HERE/"RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert torch.__version__ == historical_environment["torch"] == known_environment["torch_version"] == "2.6.0+cu124"
    assert pyro.__version__ == known_environment["pyro_version"] == "1.9.1"
    assert np.__version__ == historical_environment["numpy"] == "2.4.4"
    assert sha(RUNNER) == known_environment["sampler_source_sha256"]
    assert sha(ROOT/"experiments/cai_lib.py") == known_environment["library_sha256"]
    target = next(target for target in known_environment["targets"] if target["target"] == "ablation_obs1")
    assert sha(SURROGATE) == target["source_surrogate_sha256"]
    assert sha(REFERENCE) == target["original_reference_sha256"]
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    stage1 = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(name, ast.Name) and name.id == "S1" for name in node.targets))
    routing = Path(ast.literal_eval(stage1.args[0]))/"routing_arrays"
    chunks, input_files = [], []
    with np.load(REFERENCE, allow_pickle=False) as saved:
        observation = saved["observation"].copy()
        original = saved["chains"][0].copy()
        saved_fast = saved["fast"].copy()
    assert observation.shape == (20,) and original.shape == (600,6)
    for fast_seed in [42,43,44,45,46]:
        path = routing/f"ablation_obs1_seed{fast_seed}_N8000.npz"
        with np.load(path, allow_pickle=False) as data:
            samples = data["theta"].copy()
            obs_matches = bool(np.array_equal(data["observation"].astype(float), observation))
        assert samples.shape == (8000,6) and obs_matches
        chunks.append(samples)
        input_files.append(dict(path=str(path), sha256=sha(path), fast_seed=fast_seed,
                                theta_shape=list(samples.shape), theta_dtype=str(samples.dtype),
                                observation_exact_match=obs_matches))
    fast = np.concatenate(chunks, axis=0)
    init_fast = np.clip(np.nanmean(fast, axis=0), 0.15, 5.0)
    assert np.array_equal(fast, saved_fast) and np.isfinite(init_fast).all()
    header = next(line for line in log_text(HERE/"reference_nuts.log").splitlines()
                  if line.startswith("TARGET ablation_obs1 K 20 sigma 0.3 fast_mean "))
    logged_init = ast.literal_eval(header.split("fast_mean ",1)[1])
    assert np.array_equal(init_fast, np.asarray(logged_init))
    manifest = dict(created_utc=utc_now(), target="ablation_obs1", historical_chain_index=0, seed=701,
                    authorized_scientific_chain_calls=1, pilot_only=True, other_target_calls=0, other_seed_calls=0,
                    surrogate_path=str(SURROGATE), surrogate_sha256=sha(SURROGATE), K=20,
                    observation=observation.tolist(), observation_dtype=str(observation.dtype),
                    sigma_log=.30, kappa_sq=.001, prior="independent untruncated lognormal(0,0.30)",
                    parameterization="log-theta", sampler="Pyro NUTS", target_accept_prob=.8,
                    num_samples=600, warmup_steps=500, CPU=True, torch_threads=torch.get_num_threads(),
                    environment=dict(python=sys.executable, torch_version=torch.__version__,
                                     pyro_version=pyro.__version__, numpy_version=np.__version__,
                                     torch_matches_original_manifest=True, pyro_matches_frozen_environment_snapshot=True,
                                     original_reference_runner_did_not_record_pyro_version=True),
                    init_rule="clip(nanmean(concatenate(seed42..46 theta, axis=0), axis=0),0.15,5.0)",
                    rebuilt_init=init_fast.tolist(), init_dtype=str(init_fast.dtype),
                    init_log_float32=torch.log(torch.tensor(init_fast,dtype=torch.float32)).tolist(),
                    inputs=input_files, reconstructed_fast_exactly_matches_saved_reference_fast=True,
                    reconstructed_init_exactly_matches_original_logged_init=True,
                    runner_path=str(RUNNER), runner_sha256=sha(RUNNER), library_sha256=sha(ROOT/"experiments/cai_lib.py"),
                    reference_path=str(REFERENCE), reference_sha256=sha(REFERENCE),
                    original_log_sha256=sha(HERE/"reference_nuts.log"),
                    RNG_rule="Original run_chain() torch.manual_seed(701); no added Python/NumPy/Pyro RNG reset",
                    acceptance_and_divergence_chain_key="chain 0",
                    exact_rule="np.array_equal == True OR max_abs_difference == 0")
    prior_snapshot = json.loads((QUALITY/"control/FROZEN_BEFORE_REFERENCE_QUALITY.json").read_text(encoding="utf-8"))
    paths = [path for path in HERE.rglob("*") if path.is_file() and OUT not in path.parents
             and HERE/"psis_env" not in path.parents]
    paths += [Path(path) for path in prior_snapshot["files"]]
    paths += [Path(entry["path"]) for entry in input_files]
    snapshot = dict(created_utc=utc_now(), files={str(path):sha(path) for path in sorted(set(paths))},
                    original_367_files=prior_snapshot["original_367_files"], git_tracked_before=git_state())
    write_new_json(CONTROL/"FROZEN_BEFORE_PILOT.json", snapshot)
    write_new_json(CONTROL/"PILOT_MANIFEST.json", manifest)
    before = verify_preservation(snapshot)
    sys.path.insert(0,str(ROOT/"experiments"))
    from cai_lib import load_surrogate
    surrogate = load_surrogate(SURROGATE)
    assert all(torch.isfinite(value).all().item() for value in surrogate.state_dict().values())
    function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name == "run_chain")
    environment = dict(torch=torch, MCMC=MCMC, NUTS=NUTS, time=time, NSAMP=600, NWARM=500)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(RUNNER)+":frozen_pilot_run_chain","exec"), environment)
    attempt = dict(started_utc=utc_now(),pid=os.getpid(),target="ablation_obs1",historical_chain_index=0,
                   seed=701,authorized_calls=1,automatic_retries=0,controller_sha256=sha(__file__))
    write_new_json(CONTROL/"ATTEMPT_STARTED.json", attempt)
    print("PILOT_ONLY_START ablation_obs1 chain0 seed701",os.getpid(),flush=True)
    result = dict(manifest=manifest, attempt=attempt, preservation_before=before, status="FAILED",
                  exact_reproduction_pass=False, qualification_for_remaining_20_diagnostics=False,
                  recovered_historical_divergence_count=None, recovered_historical_acceptance_rate=None)
    try:
        # Exactly one scientific sampler call with the original RNG/target/init.
        samples,runtime,raw_diagnostics = environment["run_chain"](surrogate,observation,.30,init_fast,701)
        record_completed_pilot(result,samples,original,init_fast,observation,runtime,raw_diagnostics)
    except Exception as error:
        result.update(error_type=type(error).__name__,original_error=str(error),traceback=traceback.format_exc(),
                      classification="HISTORICAL_CHAIN_NOT_EXACTLY_REPRODUCED")
    result.update(finished_utc=utc_now(),preservation_after=verify_preservation(snapshot),
                  scientific_chain_calls=1,other_target_calls=0,other_seed_calls=0,
                  remaining_20_chain_calls=0,flow_training_calls=0,flow_resampling_calls=0,
                  original_runner_top_level_executed=False,mcmc_stopped=True,awaiting_review=True)
    write_final_reports(result,snapshot)
    print(json.dumps({key:result[key] for key in ["status","classification","exact_reproduction_pass",
                                                "recovered_historical_divergence_count","qualification_for_remaining_20_diagnostics"]}),flush=True)


def record_completed_pilot(result,samples,original,init_fast,observation,runtime,raw_diagnostics):
    comparison = compare(samples,original)
    divergences = raw_diagnostics["divergences"]["chain 0"]
    acceptance = raw_diagnostics["acceptance rate"]["chain 0"]
    diagnostics = plain(raw_diagnostics)
    exact = comparison["exact_reproduction_pass"]
    raw_buffer = io.BytesIO()
    torch.save(raw_diagnostics,raw_buffer)
    raw_bytes = raw_buffer.getvalue()
    with NPZ.open("xb") as stream:
        np.savez_compressed(stream,replay_theta=samples,historical_theta=original,
                            theta_init=init_fast,observation=observation,seed=np.asarray(701),
                            historical_chain_index=np.asarray(0),runtime_s=np.asarray(runtime),
                            divergences=np.asarray(divergences,dtype=np.int64),acceptance_rate=np.asarray(acceptance),
                            raw_mc_diagnostics_torch_bytes=np.frombuffer(raw_bytes,dtype=np.uint8),
                            raw_mc_diagnostics_json=np.asarray(json.dumps(diagnostics,ensure_ascii=False)),
                            comparison_json=np.asarray(json.dumps(comparison)),
                            manifest_json=np.asarray(json.dumps(result["manifest"],ensure_ascii=False)),
                            source_reference_sha256=np.asarray(result["manifest"]["reference_sha256"]),
                            source_runner_sha256=np.asarray(result["manifest"]["runner_sha256"]),
                            source_surrogate_sha256=np.asarray(result["manifest"]["surrogate_sha256"]))
        stream.flush()
        os.fsync(stream.fileno())
    result.update(status="COMPLETED",classification=comparison["classification"],comparison=comparison,
                  runtime_s=runtime,replay_theta_all_finite=bool(np.isfinite(samples).all()),
                  raw_mc_diagnostics=diagnostics,raw_diagnostics_serialization="torch.save dict preserved as uint8 NPZ array",
                  raw_diagnostics_serialized_sha256=hashlib.sha256(raw_bytes).hexdigest(),
                  replay_divergence_count=len(divergences),replay_divergence_indices=divergences,
                  replay_acceptance_rate=acceptance,exact_reproduction_pass=exact,
                  recovered_historical_divergence_count=len(divergences) if exact else None,
                  recovered_historical_acceptance_rate=acceptance if exact else None,
                  replay_diagnostics_attributed_to_historical_chain=exact,
                  qualification_for_remaining_20_diagnostics=exact,
                  qualification_does_not_authorize_automatic_batch=True,
                  artifact_path=str(NPZ),artifact_sha256=sha(NPZ))


def write_final_reports(result,snapshot):
    comparison = result.get("comparison",{})
    recovered = result["recovered_historical_divergence_count"]
    acceptance = result.get("replay_acceptance_rate")
    lines = [
        "1. Exact reproduction："+("PASS" if result["exact_reproduction_pass"] else "FAIL")+"；"+result["classification"]+"。",
        f"2. Max abs sample difference：{comparison.get('max_abs_difference','UNAVAILABLE')}。",
        f"3. Number of different elements：{comparison.get('number_of_different_elements','UNAVAILABLE')}。",
        f"4. Recovered historical divergence count：{recovered if recovered is not None else 'NOT_ATTRIBUTABLE_TO_HISTORY'}；replay divergence count={result.get('replay_divergence_count','UNAVAILABLE')}。",
        f"5. Acceptance rate：{acceptance if acceptance is not None else 'UNAVAILABLE'}；归属于历史chain0={result['exact_reproduction_pass']}。",
        f"6. 是否具备批量恢复剩余20条历史链diagnostics的资格：{result['qualification_for_remaining_20_diagnostics']}。本轮只授权pilot，批处理未执行。",
        "", "仅执行 ablation_obs1 historical chain index=0 / seed701 一次；MCMC 已停止，等待审核。",
        "", "**逐样本比较**", "",
        f"shape完全一致={comparison.get('shape_identical')}；historical shape={comparison.get('original_shape')}；replay shape={comparison.get('replay_shape')}。",
        f"dtype：historical={comparison.get('original_dtype')}，replay={comparison.get('replay_dtype')}；dtype一致={comparison.get('dtype_identical')}。",
        f"np.array_equal={comparison.get('np_array_equal')}；max_abs_difference={comparison.get('max_abs_difference')}；mean_abs_difference={comparison.get('mean_abs_difference')}；different elements={comparison.get('number_of_different_elements')}/{comparison.get('total_elements')}。",
        f"allclose(atol=0,rtol=0)={comparison.get('allclose_atol0_rtol0')}；allclose(atol=1e-8,rtol=1e-7)={comparison.get('allclose_atol1e_minus8_rtol1e_minus7')}。",
        "判定只按预先固定的 np.array_equal==True 或 max_abs_difference==0；近似allclose不作为历史diagnostics归属证据。",
        "", "**init与科学路径**", "",
        "routing输入按42/43/44/45/46原顺序concatenate，然后计算clip(nanmean(fast,axis=0),0.15,5.0)。没有使用第4链init。拼接fast与原reference保存的fast逐元素一致，重建init与原日志完整数值一致。",
        "Rebuilt init（完整精度）："+json.dumps(result["manifest"]["rebuilt_init"])+"；dtype="+result["manifest"]["init_dtype"]+"。",
        "Original D6 surrogate_models/member_0.pt；K=20；exact saved observation；sigma_log=0.30；kappa_sq=0.001；log-theta Pyro NUTS；target_accept_prob=0.8；600 retained draws+500 warmup；CPU、torch threads=6。",
        "冻结run_reference_nuts.py的run_chain()直接通过AST提取执行，未改科学函数；原runner顶层未执行。只使用原函数torch.manual_seed(701)，未新增其他RNG重置或sampler参数。",
        "Torch 2.6.0+cu124与原RUN_MANIFEST一致；Pyro 1.9.1与采样前冻结环境记录一致。原reference runner没有单独保存历史Pyro版本；本次记录这一provenance限制，逐样本exact判定仍独立执行。",
        "Inputs、runner、surrogate与reference路径/SHA256均保存在pilot JSON和NPZ manifest。原始mc.diagnostics()在NPZ中以torch.save字节完整保存，并在JSON中保留原层级和值；divergences与acceptance rate均读取chain 0。",
        "", f"完整性：{result['preservation_after']['protected_files']}项历史资产SHA256保持不变，包括原367项冻结资产、七个原reference、新第4链、全部flow checkpoints、历史结果和审计；Git tracked状态保持原样。",
        "其他target/seed调用=0，剩余20条历史chain调用=0，flow训练/重抽样=0，FE/damage grid/阈值调整/论文修改/Git mutations=0。原quality gate审计未覆盖或改写；本pilot不把七target quality gate升级为PASS。", ""]
    if "traceback" in result:
        lines += ["原始pilot错误（未重跑）：",result["traceback"],""]
    write_new_bytes(LOG,(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode("utf-8"))
    result.update(log_path=str(LOG),log_sha256=sha(LOG))
    write_new_json(RESULT,result)
    write_new_bytes(AUDIT,("\n".join(lines)).encode("utf-8"))
    integrity = dict(schema_version=1,created_utc=utc_now(),exact_reproduction_pass=result["exact_reproduction_pass"],
                     classification=result["classification"],comparison=comparison,
                     recovered_historical_divergence_count=recovered,
                     replay_divergence_count=result.get("replay_divergence_count"),
                     replay_acceptance_rate=acceptance,replay_diagnostics_attributed_to_history=result["exact_reproduction_pass"],
                     qualification_for_remaining_20_diagnostics=result["qualification_for_remaining_20_diagnostics"],
                     automatic_batch_execution_authorized=False,remaining_20_chain_calls=0,
                     pilot_manifest=result["manifest"],attempt=result["attempt"],
                     frozen_historical_files=snapshot["files"],original_367_protected_files=snapshot["original_367_files"],
                     preservation_before=result["preservation_before"],preservation_after=result["preservation_after"],
                     scientific_chain_calls=1,other_target_calls=0,other_seed_calls=0,flow_training_calls=0,
                     flow_resampling_calls=0,new_fe_runs=0,damage_grid_recomputations=0,git_mutations=0,
                     previous_quality_audit_overwritten=False,mcmc_stopped=True,awaiting_review=True,
                     outputs={path.name:dict(path=str(path),sha256=sha(path)) for path in [NPZ,RESULT,LOG,AUDIT] if path.exists()})
    write_new_json(INTEGRITY,integrity)


if __name__ == "__main__":
    main()
