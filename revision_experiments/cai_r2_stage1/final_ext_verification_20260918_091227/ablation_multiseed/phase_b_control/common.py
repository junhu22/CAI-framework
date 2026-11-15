"""Read frozen experiment definitions without executing the runner's top level."""
import ast
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
HERE = ROOT / "revision_experiments/cai_r2_stage1/final_ext_verification_20260918_091227"
OUT = HERE / "ablation_multiseed"
CONTROL = OUT / "phase_b_control"
PHASE_A_CONFIGS = {"N5000_L8": (5000, 8), "N10000_L8": (10000, 8), "N25000_L8": (25000, 8)}
PHASE_B_CONFIGS = {"N50000_L8": (50000, 8), "N100000_L8": (100000, 8),
                   "N50000_L4": (50000, 4), "N50000_L6": (50000, 6), "N50000_L12": (50000, 12)}
ALL_CONFIGS = {**PHASE_A_CONFIGS, **PHASE_B_CONFIGS}
SEEDS = [101, 202, 303]
FAST_SEEDS = [42, 43, 44, 45, 46]
ATOL, RTOL = 1e-8, 1e-7


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def git_read(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).decode("utf-8")


def atomic_bytes(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def atomic_json(path, data):
    atomic_bytes(path, (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def atomic_csv(path, rows, columns):
    buffer = io.StringIO(newline="")
    w = csv.DictWriter(buffer, fieldnames=columns)
    w.writeheader()
    w.writerows(rows)
    atomic_bytes(path, buffer.getvalue().encode("utf-8"))


def same(a, b):
    return bool(np.allclose(a, b, atol=ATOL, rtol=RTOL, equal_nan=False))


def emit(*values):
    message = utc_now() + " " + " ".join(str(x) for x in values)
    print(message, flush=True)
    with (CONTROL / "execution.log").open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def create_or_read_frozen_inputs():
    path = CONTROL / "FROZEN_INPUTS.json"
    if path.exists():
        baseline = json.loads(path.read_text(encoding="utf-8"))
        verify_frozen_inputs(baseline)
        return baseline
    a_path = OUT / "PHASE_A_INTEGRITY.json"
    a = json.loads(a_path.read_text(encoding="utf-8"))
    assert a["completion"]["all_9_complete"] and a["seed202_integrity_pass"]
    paths = {Path(r["path"]) for r in a["preservation"]["files"]}
    paths.update(OUT / n for n in ["PHASE_A_SUMMARY.csv", "PHASE_A_AUDIT.md", "PHASE_A_INTEGRITY.json"])
    paths.update(Path(r["checkpoint"]) for r in a["runs"])
    paths.update(ROOT / n for n in git_read("ls-files", "-z").split("\0") if n and (ROOT / n).is_file())
    paths.update([Path(r"C:\nf-pmcmc\experiments\ablation_study.py"),
                  Path(r"C:\nf-pmcmc\experiments\results\ablation_study.pt")])
    baseline = dict(created_utc=utc_now(), phase_a_approved_by_user=True,
                    authorized_new_configs=PHASE_B_CONFIGS, training_seeds=SEEDS, fast_seeds=FAST_SEEDS,
                    new_training_attempt_limit=15, runner_sha256=a["source"]["runner_sha256"],
                    library_sha256=a["source"]["cai_lib_sha256"], bank_sha256=a["bank"]["sha256"],
                    git_tracked_status_before=git_read("status", "--porcelain", "-uno"),
                    files={str(p): sha(p) for p in sorted(paths)})
    assert sha(HERE / "run_ablation_multiseed.py") == baseline["runner_sha256"]
    assert sha(ROOT / "experiments/cai_lib.py") == baseline["library_sha256"]
    assert all(sha(r["checkpoint"]) == r["checkpoint_sha256"] for r in a["runs"])
    for name in ["PHASE_A_SUMMARY.csv", "PHASE_A_AUDIT.md"]:
        assert sha(OUT / name) == a["outputs"][name]["sha256"]
    atomic_json(path, baseline)
    return baseline


def verify_frozen_inputs(baseline):
    changed = [p for p, h in baseline["files"].items() if not Path(p).is_file() or sha(p) != h]
    assert not changed, "Frozen asset changed: " + repr(changed)
    assert git_read("status", "--porcelain", "-uno") == baseline["git_tracked_status_before"]
    return dict(checked_utc=utc_now(), checked_files=len(baseline["files"]), changed_files=[], all_unchanged=True)


def load_frozen_context(baseline):
    torch.set_num_threads(6)
    sys.path.insert(0, str(ROOT / "experiments"))
    from cai_lib import ConditionalRealNVP
    a = json.loads((OUT / "PHASE_A_INTEGRITY.json").read_text(encoding="utf-8"))
    bank_path = Path(a["bank"]["path"])
    assert sha(bank_path) == baseline["bank_sha256"]
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    t_all, y_all = bank["t_all"].float(), bank["y_all"].float()
    assert tuple(t_all.shape) == (105000, 6) and tuple(y_all.shape) == (105000, 20)
    assert torch.isfinite(t_all).all() and torch.isfinite(y_all).all()
    references, observations = [], []
    for entry in a["references"]:
        assert sha(entry["path"]) == entry["sha256"]
        with np.load(entry["path"], allow_pickle=False) as v:
            references.append(v["ref_mean"].astype(np.float32))
            observations.append(v["observation"].astype(np.float32))
    tree = ast.parse((HERE / "run_ablation_multiseed.py").read_text(encoding="utf-8"))
    configs_node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "CONFIGS" for t in n.targets))
    source_configs = ast.literal_eval(configs_node)
    assert source_configs == ALL_CONFIGS
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ["train", "evaluate"]]
    assert {n.name for n in fns} == {"train", "evaluate"}
    env = dict(np=np, torch=torch, time=time, t_val=t_all[100000:105000], y_val=y_all[100000:105000],
               refs=np.stack(references), obs=np.stack(observations), FAST_SEEDS=FAST_SEEDS)
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(HERE / "run_ablation_multiseed.py") + ":frozen_functions", "exec"), env)
    return dict(model_class=ConditionalRealNVP, bank=bank, t_all=t_all, y_all=y_all, env=env,
                runner_sha256=baseline["runner_sha256"], bank_path=str(bank_path))


class SampleObserver:
    """Return the original samples unchanged; count negative and nonfinite draws."""
    def __init__(self, model):
        self.model, self.calls = model, []

    def sample(self, y, n_samples):
        samples = self.model.sample(y, n_samples)
        index = len(self.calls)
        negative = samples < 0
        call = dict(fast_seed=FAST_SEEDS[index // 5], observation_index=index % 5 + 1,
                    n_proposals=n_samples, negative_all6_elements=int(negative.sum().item()),
                    negative_all6_proposals=int(negative.any(dim=1).sum().item()),
                    negative_active_proposals=int(negative[:, :3].any(dim=1).sum().item()),
                    nonfinite_elements=int((~torch.isfinite(samples)).sum().item()),
                    min_value=float(samples.min().item()), max_value=float(samples.max().item()),
                    proposals_filtered=0)
        self.calls.append(call)
        return samples


def evaluate_with_observation(model, context):
    observer = SampleObserver(model)
    fast, mae6, mae3, sd6, sd3 = context["env"]["evaluate"](observer)
    assert len(observer.calls) == 25
    assert all(c["nonfinite_elements"] == 0 for c in observer.calls), "Nonfinite fast proposals"
    assert all(math.isfinite(v) for v in [mae6, mae3, sd6, sd3])
    return dict(fast_seed_results=fast, mae_ref_all6=mae6, mae_ref_active=mae3,
                fast_seed_sd_all6=sd6, fast_seed_sd_active=sd3, sample_calls=observer.calls,
                negative_all6_proposals=sum(c["negative_all6_proposals"] for c in observer.calls),
                negative_active_proposals=sum(c["negative_active_proposals"] for c in observer.calls),
                negative_all6_elements=sum(c["negative_all6_elements"] for c in observer.calls),
                total_fast_proposals=50000, proposals_filtered=0)


def verify_checkpoint(path, cid, seed, expected, context):
    n, layers = ALL_CONFIGS[cid]
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert all(torch.isfinite(v).all().item() for v in state.values()), "Nonfinite saved weights"
    model = context["model_class"](6, 20, n_layers=layers, hidden_dim=128)
    loaded = model.load_state_dict(state, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    model.eval()
    with torch.no_grad():
        zv, ldv = model(context["env"]["t_val"], context["env"]["y_val"])
        assert torch.isfinite(zv).all() and torch.isfinite(ldv).all()
        vl = -((-0.5 * (zv ** 2).sum(-1)) + ldv).mean().item()
    replay = evaluate_with_observation(model, context)
    comparisons = {}
    for key in ["val_loss", "mae_ref_all6", "mae_ref_active", "fast_seed_sd_all6", "fast_seed_sd_active"]:
        retained = float(expected[key])
        recomputed = vl if key == "val_loss" else replay[key]
        comparisons[key] = dict(retained=retained, recomputed=recomputed,
                                abs_difference=abs(retained - recomputed), match=same(retained, recomputed))
    retained_fast = expected["fast_seed_results"]
    if isinstance(retained_fast, str):
        retained_fast = json.loads(retained_fast)
    fast_match = same(retained_fast, replay["fast_seed_results"])
    assert all(c["match"] for c in comparisons.values()) and fast_match, "Checkpoint replay mismatch"
    return dict(config=cid, training_seed=seed, checkpoint=str(path), checkpoint_sha256=sha(path),
                strict_load_pass=True, state_all_finite=True,
                checkpoint_integrity_pass=True, comparisons=comparisons,
                fast_seed_results_recomputed=replay["fast_seed_results"],
                fast_seed_max_abs_difference=float(np.abs(np.asarray(retained_fast)[:, 1:] -
                                                          np.asarray(replay["fast_seed_results"])[:, 1:]).max()),
                sample_observation=replay, proposals_filtered=0,
                classification="VALID_RUN_CHECKPOINT_VERIFIED")
