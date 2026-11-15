"""Independent, file-only verification after training and the full audit stop."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

CONTROL = Path(__file__).resolve().parent
OUT = CONTROL.parent
SEEDS = [101, 202, 303]
AXES = {"data_size": ["N5000_L8", "N10000_L8", "N25000_L8", "N50000_L8", "N100000_L8"],
        "depth": ["N50000_L4", "N50000_L6", "N50000_L8", "N50000_L12"]}
METRICS = {"mae_ref_all6": "MAE_reference_all6", "mae_ref_active": "MAE_reference_active_e1_e3",
           "val_loss": "validation_loss", "train_time_s": "training_wall_time_s"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def equal(actual, expected):
    assert math.isfinite(float(actual)) and math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-12), (actual, expected)


def main():
    frozen = json.loads((CONTROL / "FROZEN_INPUTS.json").read_text(encoding="utf-8"))
    integrity = json.loads((OUT / "FULL_ABLATION_INTEGRITY.json").read_text(encoding="utf-8"))
    assert integrity["training_stopped"] and integrity["phase_a_preserved"]
    assert integrity["scope"]["phase_b_unique_attempts"] == 15
    for path, digest in frozen["files"].items():
        assert sha(path) == digest, path
    for output in integrity["outputs"].values():
        if "sha256" in output:
            assert sha(output["path"]) == output["sha256"], output["path"]
    a = read_csv(OUT / "RESULTS.csv")
    b = read_csv(OUT / "PHASE_B_RESULTS.csv")
    summary = read_csv(OUT / "FULL_ABLATION_SUMMARY.csv")
    attempts = json.loads((CONTROL / "ATTEMPTS.json").read_text(encoding="utf-8"))
    assert len(a) == 9 and len(b) == 15 and len(attempts) == 15 and len(summary) == 9
    all_rows = [{**r, "status": "COMPLETED"} for r in a] + b
    index = {(r["config"], int(r["training_seed"])): r for r in all_rows}
    configs = set(AXES["data_size"] + AXES["depth"])
    assert len(index) == 24 and set(index) == {(cid, seed) for cid in configs for seed in SEEDS}
    assert {(r["config"], int(r["training_seed"])) for r in b} == {(r["config"], int(r["training_seed"])) for r in attempts}
    statistic_checks = 0
    for row in summary:
        cid = row["config"]
        assert cid in AXES[row["axis"]]
        raw = [index[cid, seed] for seed in SEEDS]
        complete = all(r["status"] == "COMPLETED" for r in raw)
        assert int(row["completed_training_seed_count"]) == sum(r["status"] == "COMPLETED" for r in raw)
        assert row["all_three_training_seeds_complete"] == str(complete) and int(row["sd_ddof"]) == 1
        for seed, retained in zip(SEEDS, raw):
            assert row[f"seed{seed}_status"] == retained["status"]
            for original, label in METRICS.items():
                actual = row[f"seed{seed}_{label}"]
                if retained["status"] == "COMPLETED":
                    equal(actual, retained[original])
                else:
                    assert actual == ""
        for original, label in METRICS.items():
            values = [float(r[original]) for r in raw] if complete else None
            expected = {"mean": statistics.fmean(values), "sample_sd": statistics.stdev(values),
                        "min": min(values), "max": max(values)} if complete else None
            for name in ["mean", "sample_sd", "min", "max"]:
                actual = row[f"{label}_training_seed_{name}"]
                if complete:
                    equal(actual, expected[name])
                else:
                    assert actual == "", (cid, label, name)
                statistic_checks += 1
    shared = [r for r in summary if r["config"] == "N50000_L8"]
    assert len(shared) == 2 and {k: v for k, v in shared[0].items() if k != "axis"} == {k: v for k, v in shared[1].items() if k != "axis"}
    verified = {(r["config"], r["training_seed"]): r for r in integrity["run_verifications"]}
    completed = [r for r in all_rows if r["status"] == "COMPLETED"]
    for row in completed:
        cid, seed = row["config"], int(row["training_seed"])
        check = verified[cid, seed]
        assert check["checkpoint_integrity_pass"] and sha(row["checkpoint"]) == check["checkpoint_sha256"]
        fast = json.loads(row["fast_seed_results"])
        assert [int(r[0]) for r in fast] == [42, 43, 44, 45, 46]
        for i, suffix in [(1, "all6"), (2, "active")]:
            values = [float(r[i]) for r in fast]
            equal(row[f"mae_ref_{suffix}"], statistics.fmean(values))
            equal(row[f"fast_seed_sd_{suffix}"], statistics.stdev(values))
        assert check["sample_observation"]["proposals_filtered"] == 0
        assert check["sample_observation"]["total_fast_proposals"] == 50000
    assert integrity["reference"]["new_mcmc_runs"] == 0 and integrity["reference"]["current_reference_chains"] == 3
    final = dict(status="PASS", training_stopped=True, phase_b_unique_attempts=15,
                 phase_b_completed=sum(r["status"] == "COMPLETED" for r in b),
                 phase_b_failed=sum(r["status"] == "FAILED" for r in b), full_completed=len(completed),
                 protected_files_unchanged=len(frozen["files"]), summary_cells=9,
                 aggregate_field_checks=statistic_checks, common_L8_references_identical=True,
                 fast_variation_separate=True,
                 artifacts={name: sha(OUT / name) for name in ["PHASE_B_RESULTS.csv", "FULL_ABLATION_SUMMARY.csv",
                            "FULL_ABLATION_AUDIT.md", "FULL_ABLATION_INTEGRITY.json"]})
    (CONTROL / "ROOT_FINAL_VERIFICATION.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final))


if __name__ == "__main__":
    main()
