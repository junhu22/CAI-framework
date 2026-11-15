"""Independent arithmetic/preservation verification; no scientific execution."""
import csv
import json
import math
import statistics
from pathlib import Path

from run_seed202_recovery import (C, OUT, CONTROL, RECOVERY_CONTROL, RESULT, LOG, CHECKPOINT,
                                  CLASSIFICATION, verify_previous_assets, write_new_json)


def close(actual, expected):
    assert math.isfinite(actual) and math.isfinite(expected)
    assert math.isclose(actual, expected, rel_tol=1e-13, abs_tol=1e-13), (actual, expected)


def main():
    recovery = json.loads(RESULT.read_text(encoding="utf-8"))
    integrity_path = OUT/"FULL_ABLATION_INTEGRITY_RECOVERED.json"
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    with (OUT/"FULL_ABLATION_SUMMARY_RECOVERED.csv").open(encoding="utf-8", newline="") as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == 9
    expected_cells = {("data_size", f"N{n}_L8") for n in [5000, 10000, 25000, 50000, 100000]}
    expected_cells |= {("depth", f"N50000_L{layers}") for layers in [4, 6, 8, 12]}
    assert {(row["axis"], row["config"]) for row in summary} == expected_cells
    rows = integrity["scientific_results"]
    index = {(row["config"], int(row["training_seed"])): row for row in rows}
    assert len(index) == 24 and all(row["status"] == "COMPLETED" for row in rows)
    assert set(index) == {(cid, seed) for cid in C.ALL_CONFIGS for seed in C.SEEDS}
    original_completed = []
    for name in ["RESULTS.csv", "PHASE_B_RESULTS.csv"]:
        with (OUT/name).open(encoding="utf-8", newline="") as stream:
            original_completed += [row for row in csv.DictReader(stream)
                                   if row.get("status", "COMPLETED") == "COMPLETED"]
    assert len(original_completed) == 23
    for original in original_completed:
        current = index[original["config"], int(original["training_seed"])]
        assert all(current[key] == value for key, value in original.items())
    recovered_row = index["N50000_L8", 202]
    assert all(recovered_row[key] == value for key, value in recovery["scientific_result"].items())
    epochs = recovery["validation_epochs"]
    assert len(epochs) == int(recovered_row["last_epoch"])
    assert all(record["finite"] and math.isfinite(record["validation_loss"]) for record in epochs)
    minimum = min(record["validation_loss"] for record in epochs)
    earliest = next(record["epoch"] for record in epochs if record["validation_loss"] == minimum)
    assert minimum == recovered_row["val_loss"] and earliest == recovered_row["best_epoch"]
    assert recovered_row["optimizer_steps"] == len(epochs)*(50000//512)
    with (CONTROL/"N50000_L8_seed202_epochs.csv").open(encoding="utf-8", newline="") as stream:
        old_epochs = list(csv.DictReader(stream))
    trajectory = recovery["trajectory_reproduction"]
    pairs = list(zip(old_epochs, epochs[:233]))
    differences = [abs(float(old["validation_loss"])-new["validation_loss"]) for old, new in pairs]
    assert trajectory["compared_epochs"] == len(pairs)
    close(trajectory["max_absolute_difference"], max(differences))
    close(trajectory["mean_absolute_difference"], statistics.mean(differences))
    matches = len(pairs) == 233 and all(abs(float(old["validation_loss"])-new["validation_loss"])
                                      <= 1e-6+1e-7*abs(float(old["validation_loss"])) for old, new in pairs)
    assert trajectory["all_233_within_tolerance"] == matches
    assert len(trajectory["epochwise_comparisons"]) == len(pairs)
    labels = {"mae_ref_all6": "MAE_reference_all6", "mae_ref_active": "MAE_reference_active_e1_e3",
              "val_loss": "validation_loss", "train_time_s": "training_wall_time_s"}
    aggregate_fields = 0
    fast_fields = 0
    checkpoints = set()
    for output in summary:
        cid = output["config"]
        assert output["sd_ddof"] == "1" and output["completed_training_seed_count"] == "3"
        for metric, label in labels.items():
            values = [float(index[cid, seed][metric]) for seed in C.SEEDS]
            expected = {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values),
                        "min": min(values), "max": max(values)}
            for stat, value in expected.items():
                close(float(output[f"{label}_training_seed_{stat}"]), value)
                aggregate_fields += 1
            for seed in C.SEEDS:
                close(float(output[f"seed{seed}_{label}"]), float(index[cid, seed][metric]))
        for seed in C.SEEDS:
            row = index[cid, seed]
            path = Path(output[f"seed{seed}_checkpoint"])
            assert path == Path(row["checkpoint"])
            assert C.sha(path) == output[f"seed{seed}_checkpoint_sha256"]
            checkpoints.add(str(path))
            fast = json.loads(row["fast_seed_results"])
            assert [values[0] for values in fast] == C.FAST_SEEDS
            for fs, mae_all6, mae_active in fast:
                close(float(output[f"seed{seed}_fast_seed{fs}_MAE_reference_all6"]), mae_all6)
                close(float(output[f"seed{seed}_fast_seed{fs}_MAE_reference_active_e1_e3"]), mae_active)
                fast_fields += 2
            for k, suffix in [(1, "all6"), (2, "active")]:
                values = [values[k] for values in fast]
                close(statistics.stdev(values), float(row[f"fast_seed_sd_{suffix}"]))
                close(float(output[f"seed{seed}_fast_sampling_sample_sd_{suffix}"]), statistics.stdev(values))
                close(statistics.mean(values), float(row[f"mae_ref_{suffix}"]))
    assert aggregate_fields == 144 and fast_fields == 270 and len(checkpoints) == 24
    shared = [row for row in summary if row["config"] == "N50000_L8"]
    assert len(shared) == 2
    assert {k: v for k, v in shared[0].items() if k != "axis"} == {k: v for k, v in shared[1].items() if k != "axis"}
    assert shared[0]["seed202_checkpoint"] == str(CHECKPOINT)
    assert shared[0]["seed202_integrity_classification"] == CLASSIFICATION
    baseline = json.loads((CONTROL/"FROZEN_INPUTS.json").read_text(encoding="utf-8"))
    preservation = C.verify_frozen_inputs(baseline)
    snapshot = json.loads((RECOVERY_CONTROL/"FROZEN_BEFORE_RECOVERY.json").read_text(encoding="utf-8"))
    previous = verify_previous_assets(snapshot)
    with (OUT/"PHASE_B_RESULTS.csv").open(encoding="utf-8", newline="") as stream:
        failed = next(row for row in csv.DictReader(stream)
                      if row["config"] == "N50000_L8" and int(row["training_seed"]) == 202)
    assert failed == snapshot["original_failed_row"] == recovery["original_failure"]["row"]
    assert failed["status"] == "FAILED" and not failed["checkpoint"]
    assert C.sha(LOG) == recovery["log_sha256"]
    for output in integrity["outputs"].values():
        assert C.sha(output["path"]) == output["sha256"]
    for check in integrity["run_verifications"]:
        assert check["strict_load_pass"] and check["state_all_finite"] and check["checkpoint_integrity_pass"]
        assert check["fast_seed_max_abs_difference"] == 0.0
        assert all(comparison["abs_difference"] == 0.0 for comparison in check["comparisons"].values())
    forbidden = ["best configuration", "statistically equivalent", "statistically comparable", "significant improvement"]
    for path in [OUT/"FULL_ABLATION_AUDIT_RECOVERED.md", integrity_path, RESULT, LOG]:
        assert not any(phrase in path.read_text(encoding="utf-8").lower() for phrase in forbidden)
    verification = dict(status="PASS", checked_utc=C.utc_now(), scientific_cells=24, summary_axis_cells=9,
                        independently_checked_aggregate_fields=aggregate_fields,
                        independently_checked_fast_seed_fields=fast_fields,
                        distinct_checkpoints=24, shared_N50000_L8_rows_equal_except_axis=True,
                        original_failed_row_preserved=True, frozen_367=preservation,
                        all_previous_assets=previous, tracked_git_files_unchanged=True,
                        training_stopped=True, new_training_calls=1, new_mcmc_runs=0,
                        outputs={path.name: C.sha(path) for path in [RESULT, LOG, CHECKPOINT,
                                 OUT/"FULL_ABLATION_SUMMARY_RECOVERED.csv",
                                 OUT/"FULL_ABLATION_AUDIT_RECOVERED.md", integrity_path]})
    write_new_json(RECOVERY_CONTROL/"FINAL_VERIFICATION.json", verification)
    print(json.dumps(verification, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
