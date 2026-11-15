"""Qualified-reference manifests and exclusive output I/O; no scientific runner."""
import csv
import hashlib
import io
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CONTROL = Path(__file__).resolve().parent
OUT = CONTROL.parent
HERE = OUT.parent
ROOT = HERE.parents[2]
HISTORY = OUT / "historical_divergence_recovery"
QUALIFIED = [f"ablation_obs{i}" for i in range(1, 6)] + ["sailboat_field_frozen"]
ABLATION_TARGETS = QUALIFIED[:5]
BOUNDARY = "sailboat_boundary_e1_075"
QUALITY_PATH = OUT / "REFERENCE_QUALITY_INTEGRITY_RECOVERED.json"
FROZEN_PATH = CONTROL / "FROZEN_BEFORE_QUALIFIED_FINAL.json"
Q = [.025, .25, .50, .75, .975]
QL = ["q025", "q25", "q50", "q75", "q975"]
OUTPUT_NAMES = ["ROUTING_VS_GAP_4CHAIN_QUALIFIED.csv", "ROUTING_VS_GAP_4CHAIN_QUALIFIED_AUDIT.md",
    "BOUNDARY_REFERENCE_EXCLUSION_NOTE.md", "FULL_ABLATION_4CHAIN_REFERENCE.csv",
    "FULL_ABLATION_4CHAIN_REFERENCE_AUDIT.md", "QUALIFIED_4CHAIN_FINAL_AUDIT.md", "QUALIFIED_4CHAIN_FINAL_INTEGRITY.json"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_read(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).decode("utf-8")


def new_bytes(path, data):
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def new_json(path, data):
    new_bytes(path, (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def new_csv(path, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
    writer.writeheader()
    writer.writerows(rows)
    new_bytes(path, buffer.getvalue().encode("utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify(snapshot):
    changed = [path for path, digest in snapshot["files"].items() if not Path(path).is_file() or sha(path) != digest]
    assert not changed, changed
    assert git_read("status", "--porcelain", "-uno") == snapshot["git_tracked_before"]
    return dict(checked_utc=utc_now(), checked_files=len(snapshot["files"]), all_sha256_unchanged=True,
                changed_files=[], git_tracked_unchanged=True)


def frozen_quality():
    quality = read_json(QUALITY_PATH)
    targets = {target["target"]: target for target in quality["targets"]}
    assert len(targets) == 7
    assert [name for name, target in targets.items() if target["overall_quality_gate"] == "PASS"] == QUALIFIED
    assert all(all(value == "PASS" for value in targets[name]["gates"].values()) for name in QUALIFIED)
    excluded = targets[BOUNDARY]
    assert excluded["overall_quality_gate"] == "FAIL" and excluded["gates"]["divergences"] == "FAIL"
    assert excluded["historical_divergence_counts"] == [1, 0, 0] and excluded["seed704_divergence_count"] == 0
    historical701 = next(record for record in excluded["historical_chain_records"] if record["manifest"]["seed"] == 701)
    assert historical701["comparison"]["np_array_equal"] and historical701["replay_divergence_indices"] == [375]
    return quality, targets


def initialize():
    assert not FROZEN_PATH.exists(), "Existing qualified final run must be preserved."
    for name in OUTPUT_NAMES:
        assert not (OUT / name).exists(), "Never overwrite any existing result: " + name
    quality, targets = frozen_quality()
    previous = read_json(HISTORY / "full_recovery_control/FROZEN_BEFORE_FULL_RECOVERY.json")
    verify(previous)
    assert read_json(HISTORY / "full_recovery_control/FINAL_VERIFICATION.json")["verification_pass"]
    paths = [p for p in HERE.rglob("*") if p.is_file() and CONTROL not in p.parents and HERE / "psis_env" not in p.parents]
    paths += [Path(path) for path in previous["files"]]
    snapshot = dict(created_utc=utc_now(), files={str(p): sha(p) for p in sorted(set(paths))},
        original_367_files=previous["original_367_files"], git_tracked_before=git_read("status", "--porcelain", "-uno"),
        git_tracked_file_hashes=previous["git_tracked_file_hashes"],
        reference_qualification_source=dict(path=str(QUALITY_PATH), sha256=sha(QUALITY_PATH)))
    new_json(FROZEN_PATH, snapshot)
    new_json(CONTROL / "QUALIFIED_REFERENCE_MANIFEST.json", dict(created_utc=utc_now(),
        qualified_r2_1_targets=QUALIFIED, qualified_r2_3_targets=ABLATION_TARGETS,
        excluded_target=BOUNDARY, exclusion_classification="EXCLUDED_BY_PRESPECIFIED_REFERENCE_QUALITY_GATE",
        exclusion_reason="historical seed701 divergence count=1, retained-draw index=375",
        original_quality_gate=quality["quality_gate"], qualification_source=snapshot["reference_qualification_source"],
        targets=quality["targets"], routing_threshold_tau=.5, fast_seeds=[42, 43, 44, 45, 46],
        training_seeds=[101, 202, 303], ablation_Nfast=2000, new_mcmc_calls=0, flow_training_calls=0,
        original_all7_FAIL_audit_retained=True, qualification_subset_explicitly_authorized_by_user=True))
    print("QUALIFIED_FINAL_INPUTS_FROZEN", len(snapshot["files"]), flush=True)


def qualified_reference(name, target):
    assert name in QUALIFIED and target["overall_quality_gate"] == "PASS"
    manifest = target["target_manifest"]
    assert sha(manifest["original_reference_path"]) == manifest["original_reference_sha256"]
    assert sha(target["reference4_path"]) == target["reference4_sha256"]
    with np.load(manifest["original_reference_path"], allow_pickle=False) as old, np.load(target["reference4_path"], allow_pickle=False) as four:
        chains = four["chains"].copy()
        assert chains.shape == (4, 600, 6) and np.array_equal(chains[:3], old["chains"])
        assert np.array_equal(chains[3], four["theta_samples"]) and four["seed"].item() == 704
        original = {key: old[key].copy() for key in old.files}
    assert np.isfinite(chains).all()
    pooled = chains.reshape(-1, 6).astype(np.float64)
    mean, quantiles = pooled.mean(0), np.quantile(pooled, Q, axis=0)
    assert np.array_equal(mean, target["reference_4chain_mean"])
    assert np.array_equal(quantiles, target["reference_4chain_quantiles"])
    return original, pooled, mean, quantiles


if __name__ == "__main__":
    initialize()
