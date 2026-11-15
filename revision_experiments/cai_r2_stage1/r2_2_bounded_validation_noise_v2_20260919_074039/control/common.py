"""Exclusive V2 output I/O and preservation checks; no scientific execution here."""
import csv
import hashlib
import io
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent
CONTROL = OUT / "control"
STAGE = OUT.parent
ROOT = STAGE.parents[1]
V1 = STAGE / "r2_2_bounded_validation_20260918_225643"
FINAL = STAGE / "final_ext_verification_20260918_091227"
OLD = ROOT / "revision_experiments/task_r3_domain"
FE = STAGE / "local_fe_rerun_20260918_085458"
SUR = ROOT / "surrogate_models_broad_ext/member_0.pt"
CNF = ROOT / "experiments/results/conditional_flow_broad_ext.pt"
FROZEN = CONTROL / "FROZEN_INPUTS_V2.json"
KAP, SEV = [.01, .03, .05], [.10, .25, .50]
NFAST, NREP, NNULL = 1000, 12, 120
CLASSIFICATION = "SUPERSEDED_FOR_NOISE_SCALING_SEMANTICS_ONLY"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def arrays(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def new_bytes(path, data):
    with Path(path).open("xb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())


def new_json(path, data):
    new_bytes(path, (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8"))


def new_csv(path, data):
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(dict.fromkeys(k for r in data for k in r)))
    writer.writeheader(); writer.writerows(data)
    new_bytes(path, buf.getvalue().encode("utf-8"))


def new_npz(path, **data):
    with Path(path).open("xb") as f:
        np.savez_compressed(f, **data); f.flush(); os.fsync(f.fileno())


def copy_frozen(source, target):
    new_bytes(target, Path(source).read_bytes())
    assert sha(source) == sha(target)
    return dict(source=str(source), output=str(target), source_SHA256=sha(source), output_SHA256=sha(target), bitwise_identical=True)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).decode("utf-8")


def verify(snapshot):
    changed = [p for p, h in snapshot["files"].items() if not Path(p).is_file() or sha(p) != h]
    assert not changed, changed
    tracked = [p for p, h in snapshot["Git_tracked_file_hashes"].items() if not Path(p).is_file() or sha(p) != h]
    assert not tracked, tracked
    assert git("status", "--porcelain", "-uno") == snapshot["git_tracked_before"]
    return dict(checked_utc=now(), checked_protected_files=len(snapshot["files"]),
        checked_V1_files=len(snapshot["V1_files"]), all_protected_SHA256_unchanged=True,
        checked_Git_tracked_files=len(snapshot["Git_tracked_file_hashes"]), all_tracked_SHA256_unchanged=True,
        Git_tracked_status_unchanged=True)


def initialize():
    assert not FROZEN.exists()
    old = read(V1 / "INPUT_OUTPUT_INTEGRITY.json")
    seal = read(V1 / "control/FINAL_VERIFICATION.json")
    assert seal["verification_pass"]
    assert sha(V1 / "INPUT_OUTPUT_INTEGRITY.json") == seal["integrity_manifest_sha256"]
    assert sha(V1 / "R2_2_REPORT.md") == seal["report_sha256"]
    assert all(sha(V1 / p) == h for p, h in old["output_SHA256"].items())
    assert all(sha(p) == h for p, h in old["protected_inputs"].items())
    assert all(sha(p) == h for p, h in old["Git_tracked_file_hashes"].items())
    v1_files = {str(p): sha(p) for p in sorted(V1.rglob("*")) if p.is_file()}
    files = dict(old["protected_inputs"], **v1_files)
    principles = STAGE / "REVISION_WORKING_PRINCIPLES.md"
    files[str(principles)] = sha(principles)
    snapshot = dict(created_utc=now(), files=files, V1_files=v1_files,
        V1_directory=str(V1), V1_classification=CLASSIFICATION,
        V1_not_deleted_not_invalidated=True, V1_integrity_manifest_sha256=sha(V1 / "INPUT_OUTPUT_INTEGRITY.json"),
        original_367_files=old["original_367_frozen_assets"],
        Git_tracked_file_hashes=old["Git_tracked_file_hashes"], git_tracked_before=git("status", "--porcelain", "-uno"),
        authorized_scope=dict(relative_noise_scaling_only=True, e1_e3_null_reused=True,
            new_e2_grid_inference_calls=108, matched_FE_inference_calls=216, matched_NN_samples_reused=True,
            no_training=True, no_MCMC=True, no_PSIS=True, no_new_FE=True, no_MATLAB=True,
            no_new_seed_design=True, no_threshold_tuning=True, no_null_changes=True,
            no_submitted_text_edit=True, no_Git_mutations=True))
    verification = verify(snapshot)
    new_json(FROZEN, snapshot)
    print("V2_ALL_INPUTS_FROZEN", json.dumps(verification), flush=True)


if __name__ == "__main__":
    initialize()
