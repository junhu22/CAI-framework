"""One explicitly authorized recovery; frozen training with memory-only telemetry.

The original runner's top level is never executed. No progress file or stream
operation is performed by the validation hook or within the training call.
Existing assets and the original FAILED result are read and preserved.
"""
import csv
import json
import math
import os
import sys
import traceback
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent
CONTROL = OUT / "phase_b_control"
RECOVERY_CONTROL = Path(__file__).resolve().parent
sys.path.insert(0, str(CONTROL))
import common as C

CONFIG, SEED, N_TRAIN, N_LAYERS = "N50000_L8", 202, 50000, 8
CLASSIFICATION = "RECOVERED_AFTER_TELEMETRY_IO_FAILURE"
CHECKPOINT = OUT / "N50000_L8_seed202_recovery.pt"
RESULT = OUT / "RECOVERY_N50000_L8_seed202.json"
LOG = OUT / "RECOVERY_N50000_L8_seed202.log"
TRAJECTORY_ATOL, TRAJECTORY_RTOL = 1e-6, 1e-7


def write_new_bytes(path, data):
    """Exclusive creation, used only before or after training, never in a hook."""
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def write_new_json(path, value):
    write_new_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n").encode("utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_previous_assets(snapshot):
    changed = [path for path, digest in snapshot["files"].items()
               if not Path(path).is_file() or C.sha(path) != digest]
    assert not changed, "Previously existing asset changed: " + repr(changed)
    assert C.git_read("status", "--porcelain", "-uno") == snapshot["git_tracked_status"]
    return dict(checked_utc=C.utc_now(), checked_files=len(snapshot["files"]),
                changed_files=[], all_unchanged=True, git_tracked_status_unchanged=True)


class MemoryValidationObserver:
    """Read validation outputs and append Python values; no I/O or model changes."""
    def __init__(self, context):
        self.t_val = context["env"]["t_val"]
        self.y_val = context["env"]["y_val"]
        self.records = []
        self.best = float("inf")
        self.best_epoch = 0

    def hook(self, model, inputs, output):
        if len(inputs) != 2 or inputs[0] is not self.t_val or inputs[1] is not self.y_val:
            return None
        z, ld = output
        loss = -((-0.5 * (z ** 2).sum(-1)) + ld).mean().item()
        finite = bool(C.torch.isfinite(z).all().item()
                      and C.torch.isfinite(ld).all().item() and math.isfinite(loss))
        epoch = len(self.records) + 1
        if loss < self.best:
            self.best, self.best_epoch = loss, epoch
        self.records.append(dict(epoch=epoch, validation_loss=loss,
                                 best_validation_loss=self.best, best_epoch=self.best_epoch,
                                 no_improvement_epochs=epoch - self.best_epoch,
                                 estimated_optimizer_steps=epoch * (N_TRAIN // 512), finite=finite))
        return None


def compare_trajectory(records, original_epochs):
    comparisons = []
    for old, new in zip(original_epochs, records[:233]):
        assert int(old["epoch"]) == new["epoch"]
        old_value, new_value = float(old["validation_loss"]), new["validation_loss"]
        difference = abs(new_value - old_value)
        comparisons.append(dict(epoch=new["epoch"], original_validation_loss=old_value,
                                recovery_validation_loss=new_value, absolute_difference=difference,
                                within_tolerance=bool(C.np.isclose(
                                    new_value, old_value, atol=TRAJECTORY_ATOL,
                                    rtol=TRAJECTORY_RTOL, equal_nan=False))))
    complete = len(comparisons) == 233
    matches = complete and all(v["within_tolerance"] for v in comparisons)
    differences = [v["absolute_difference"] for v in comparisons]
    return dict(original_recorded_epochs=233, compared_epochs=len(comparisons),
                atol=TRAJECTORY_ATOL, rtol=TRAJECTORY_RTOL, equal_nan=False,
                max_absolute_difference=max(differences) if differences else None,
                mean_absolute_difference=float(C.np.mean(differences)) if differences else None,
                all_233_within_tolerance=matches,
                classification="TRAINING_TRAJECTORY_REPRODUCED_THROUGH_EPOCH233" if matches
                else "TRAINING_TRAJECTORY_DIFFERENCES_RETAINED_WITHOUT_RETRAINING",
                epochwise_comparisons=comparisons)


def main():
    for path in [CHECKPOINT, RESULT, LOG, RECOVERY_CONTROL / "ATTEMPT_STARTED.json",
                 RECOVERY_CONTROL / "FROZEN_BEFORE_RECOVERY.json"]:
        assert not path.exists(), "Preserve existing recovery attempt; do not retrain: " + str(path)
    baseline = json.loads((CONTROL / "FROZEN_INPUTS.json").read_text(encoding="utf-8"))
    assert len(baseline["files"]) == 367
    preservation_before = C.verify_frozen_inputs(baseline)
    assert C.sha(C.HERE / "run_ablation_multiseed.py") == baseline["runner_sha256"]
    assert C.sha(C.ROOT / "experiments/cai_lib.py") == baseline["library_sha256"]
    original_b_rows = read_csv(OUT / "PHASE_B_RESULTS.csv")
    original_failed = next(row for row in original_b_rows
                           if row["config"] == CONFIG and int(row["training_seed"]) == SEED)
    assert len(original_b_rows) == 15 and original_failed["status"] == "FAILED"
    assert original_failed["error_type"] == "PermissionError" and not original_failed["checkpoint"]
    failed_path = CONTROL / "N50000_L8_seed202_FAILED.txt"
    epochs_path = CONTROL / "N50000_L8_seed202_epochs.csv"
    original_epochs = read_csv(epochs_path)
    assert len(original_epochs) == 233
    assert [int(row["epoch"]) for row in original_epochs] == list(range(1, 234))
    assert all(row["finite"] == "True" and math.isfinite(float(row["validation_loss"]))
               for row in original_epochs)
    assert float(original_epochs[-1]["validation_loss"]) == -8.378612518310547
    assert int(original_epochs[-1]["best_epoch"]) == 233
    # Freeze every existing output/control asset as well as the original 367 assets.
    previous_paths = [path for path in OUT.rglob("*") if path.is_file()
                      and RECOVERY_CONTROL not in path.parents]
    previous_paths += [Path(path) for path in baseline["files"]]
    previous_paths = sorted(set(previous_paths))
    snapshot = dict(created_utc=C.utc_now(), files={str(path): C.sha(path) for path in previous_paths},
                    original_367_manifest_path=str(CONTROL / "FROZEN_INPUTS.json"),
                    original_367_manifest_sha256=C.sha(CONTROL / "FROZEN_INPUTS.json"),
                    git_tracked_status=C.git_read("status", "--porcelain", "-uno"),
                    original_failed_row=original_failed)
    write_new_json(RECOVERY_CONTROL / "FROZEN_BEFORE_RECOVERY.json", snapshot)
    context = C.load_frozen_context(baseline)
    attempt = dict(started_utc=C.utc_now(), pid=os.getpid(), config=CONFIG, training_seed=SEED,
                   manual_recovery_authorized_by_user=True, authorized_training_calls=1,
                   automatic_retry=False, seed_replacement=False,
                   source_path=str(Path(__file__).resolve()), source_sha256=C.sha(__file__))
    write_new_json(RECOVERY_CONTROL / "ATTEMPT_STARTED.json", attempt)
    C.torch.manual_seed(SEED)
    C.np.random.seed(SEED)
    flow = context["model_class"](6, 20, n_layers=N_LAYERS, hidden_dim=128)
    observer = MemoryValidationObserver(context)
    handle = flow.register_forward_hook(observer.hook)
    result = dict(schema_version=1, config=CONFIG, training_seed=SEED,
                  attempt=attempt, status="FAILED", classification="RECOVERY_FAILED_NO_RETRY",
                  original_failure=dict(row=original_failed,
                                        results_csv=str(OUT / "PHASE_B_RESULTS.csv"),
                                        results_csv_sha256=C.sha(OUT / "PHASE_B_RESULTS.csv"),
                                        traceback_path=str(failed_path), traceback_sha256=C.sha(failed_path),
                                        epochs_path=str(epochs_path), epochs_sha256=C.sha(epochs_path),
                                        recorded_epochs=233, all_validation_records_finite=True,
                                        last_observed_validation_loss=-8.378612518310547,
                                        last_observed_best_epoch=233, selected_checkpoint_produced=False,
                                        cause="Observer PROGRESS.json atomic replace PermissionError WinError 5",
                                        original_failed_record_preserved=True),
                  preservation_before=preservation_before)
    error_text = None
    try:
        # Scientific training call: unchanged function and inputs; observer is memory-only.
        # There are no prints, open(), write(), rename(), progress or network operations here.
        flow, best, best_epoch, last_epoch, steps, duration = context["env"]["train"](
            flow, context["t_all"][:N_TRAIN], context["y_all"][:N_TRAIN], SEED)
        handle.remove()
        handle = None
        result["trajectory_reproduction"] = compare_trajectory(observer.records, original_epochs)
        assert len(observer.records) == last_epoch
        assert observer.best == best and observer.best_epoch == best_epoch
        assert steps == last_epoch * (N_TRAIN // 512)
        assert last_epoch == 300 or last_epoch - best_epoch == 30
        assert all(row["finite"] for row in observer.records)
        assert all(C.torch.isfinite(value).all().item() for value in flow.state_dict().values())
        row = dict(config=CONFIG, n_train=N_TRAIN, n_layers=N_LAYERS, training_seed=SEED,
                   val_loss=best, best_epoch=best_epoch, last_epoch=last_epoch,
                   optimizer_steps=steps, train_time_s=duration, checkpoint=str(CHECKPOINT),
                   status="COMPLETED", integrity_classification=CLASSIFICATION,
                   started_utc=attempt["started_utc"], nonfinite_validation_epochs=0, proposals_filtered=0)
        # Persist only the selected minimum-validation checkpoint before reference evaluation.
        with CHECKPOINT.open("xb") as stream:
            C.torch.save(flow.state_dict(), stream)
            stream.flush()
            os.fsync(stream.fileno())
        row["checkpoint_sha256"] = C.sha(CHECKPOINT)
        evaluation = C.evaluate_with_observation(flow, context)
        for key in ["mae_ref_all6", "mae_ref_active", "fast_seed_sd_all6", "fast_seed_sd_active",
                    "negative_all6_proposals", "negative_active_proposals", "negative_all6_elements",
                    "total_fast_proposals"]:
            row[key] = evaluation[key]
        row["fast_seed_results"] = json.dumps(evaluation["fast_seed_results"])
        for fast_seed, mae_all6, mae_active in evaluation["fast_seed_results"]:
            row[f"fast_seed{fast_seed}_mae_ref_all6"] = mae_all6
            row[f"fast_seed{fast_seed}_mae_ref_active"] = mae_active
        verification = C.verify_checkpoint(CHECKPOINT, CONFIG, SEED, row, context)
        assert evaluation["sample_calls"] == verification["sample_observation"]["sample_calls"]
        assert verification["comparisons"]["val_loss"]["abs_difference"] == 0.0
        assert verification["fast_seed_max_abs_difference"] == 0.0
        verification["classification"] = CLASSIFICATION
        row.update(checkpoint_replay_pass=True,
                   replay_max_abs_mae_difference=verification["fast_seed_max_abs_difference"],
                   validation_replay_abs_difference=verification["comparisons"]["val_loss"]["abs_difference"],
                   finished_utc=C.utc_now())
        preservation_after = C.verify_frozen_inputs(baseline)
        previous_preservation = verify_previous_assets(snapshot)
        result.update(status="COMPLETED", classification=CLASSIFICATION, scientific_result=row,
                      checkpoint_verification=verification, fast_evaluation=evaluation,
                      numerical_tolerance=dict(atol=C.ATOL, rtol=C.RTOL, equal_nan=False),
                      validation_epochs=observer.records,
                      training_rules=dict(train_prefix=[0, 50000], validation_indices=[100000, 105000],
                                          n_layers=8, hidden_dim=128, max_epochs=300, batch_size=512,
                                          optimizer="Adam", lr=1e-3, weight_decay=1e-5,
                                          scheduler="CosineAnnealingLR", T_max=300, gradient_clip=5,
                                          patience=30, checkpoint_rule="strict minimum validation loss",
                                          fast_seeds=C.FAST_SEEDS, n_fast=2000),
                      frozen_execution=dict(runner_path=str(C.HERE / "run_ablation_multiseed.py"),
                                            runner_sha256=baseline["runner_sha256"],
                                            model_source_sha256=baseline["library_sha256"],
                                            bank_path=context["bank_path"], bank_sha256=baseline["bank_sha256"],
                                            functions_called=["train", "evaluate"],
                                            original_runner_top_level_executed=False,
                                            validation_observer="memory-only, returns None, no I/O",
                                            training_progress_file_operations=0,
                                            numpy_version=C.np.__version__, torch_version=C.torch.__version__,
                                            torch_threads=C.torch.get_num_threads(), device="cpu"),
                      preservation_after=preservation_after,
                      previous_assets_preservation_after=previous_preservation,
                      previous_assets_snapshot_path=str(RECOVERY_CONTROL / "FROZEN_BEFORE_RECOVERY.json"),
                      previous_assets_snapshot_sha256=C.sha(RECOVERY_CONTROL / "FROZEN_BEFORE_RECOVERY.json"),
                      integrity_pass=True, training_calls=1, automatic_retries=0,
                      new_mcmc_runs=0, new_fe_runs=0, damage_grid_recomputations=0,
                      training_stopped=True, finished_utc=C.utc_now())
    except Exception as error:
        error_text = traceback.format_exc()
        result.update(error_type=type(error).__name__, original_error=str(error),
                      traceback=error_text, validation_epochs=observer.records,
                      trajectory_reproduction=compare_trajectory(observer.records, original_epochs),
                      training_stopped=True, automatic_retries=0, finished_utc=C.utc_now())
    finally:
        if handle is not None:
            handle.remove()
    lines = ["Manual recovery N50000_L8 training seed 202", "Started UTC: " + attempt["started_utc"],
             "Status: " + result["status"], "Classification: " + result["classification"],
             "Original FAILED attempt and all previous assets are preserved.",
             "Scientific training used the frozen train() with a memory-only validation observer.",
             "No progress file operations occurred during training. This log was written after training.",
             "Trajectory comparison: " + json.dumps({k: v for k, v in result["trajectory_reproduction"].items()
                                                     if k != "epochwise_comparisons"}),
             "epoch,validation_loss,best_validation_loss,best_epoch,no_improvement_epochs,estimated_optimizer_steps,finite"]
    for epoch in observer.records:
        lines.append(",".join(str(epoch[key]) for key in ["epoch", "validation_loss", "best_validation_loss",
                                                          "best_epoch", "no_improvement_epochs",
                                                          "estimated_optimizer_steps", "finite"]))
    if result["status"] == "COMPLETED":
        lines += ["Scientific result: " + json.dumps(result["scientific_result"]),
                  "Checkpoint replay passed; validation and all five fast-seed MAEs reproduced exactly.",
                  "Original 367 frozen assets and all pre-existing output assets retained their SHA256."]
    if error_text:
        lines += ["Original recovery error; no retry:", error_text]
    write_new_bytes(LOG, ("\n".join(lines) + "\n").encode("utf-8"))
    result["log_path"], result["log_sha256"] = str(LOG), C.sha(LOG)
    write_new_json(RESULT, result)
    print(json.dumps({key: result[key] for key in ["status", "classification", "training_stopped"]}), flush=True)
    if result["status"] != "COMPLETED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
