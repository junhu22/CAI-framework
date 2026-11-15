"""Execute exactly the authorized remaining 15 runs using unchanged definitions."""
import csv
import json
import math
import msvcrt
import os
import time
import traceback

from common import *

RESULTS = OUT / "PHASE_B_RESULTS.csv"
PROGRESS = CONTROL / "PROGRESS.json"
ATTEMPTS = CONTROL / "ATTEMPTS.json"
CORE = ["config", "n_train", "n_layers", "training_seed", "status", "integrity_classification",
        "val_loss", "best_epoch", "last_epoch", "optimizer_steps", "train_time_s",
        "mae_ref_all6", "mae_ref_active", "fast_seed_sd_all6", "fast_seed_sd_active",
        "checkpoint", "checkpoint_sha256", "fast_seed_results"]
FAST_COLUMNS = [f"fast_seed{s}_{metric}" for s in FAST_SEEDS for metric in ["mae_ref_all6", "mae_ref_active"]]
COLUMNS = CORE + FAST_COLUMNS + ["negative_all6_proposals", "negative_active_proposals", "negative_all6_elements",
          "total_fast_proposals", "proposals_filtered", "checkpoint_replay_pass", "replay_max_abs_mae_difference",
          "validation_replay_abs_difference", "started_utc", "finished_utc", "attempt_number",
          "nonfinite_validation_epochs", "error_type", "original_error", "traceback_path"]


class EpochObserver:
    """An output-preserving validation hook for progress and epoch history only."""
    def __init__(self, cid, seed, n, context, completed, failed):
        self.cid, self.seed, self.n, self.context = cid, seed, n, context
        self.completed, self.failed = completed, failed
        self.epoch, self.best, self.best_epoch, self.nonfinite = 0, float("inf"), 0, 0
        self.started = time.time()
        self.path = CONTROL / f"{cid}_seed{seed}_epochs.csv"
        assert not self.path.exists(), "An attempted run already has epoch history; no automatic retry."
        self.history = self.path.open("x", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.history, fieldnames=["epoch", "validation_loss", "best_validation_loss",
                                     "best_epoch", "no_improvement_epochs", "estimated_optimizer_steps", "elapsed_s", "finite"])
        self.writer.writeheader()

    def hook(self, model, inputs, output):
        if not (len(inputs) == 2 and inputs[0] is self.context["env"]["t_val"] and inputs[1] is self.context["env"]["y_val"]):
            return None
        self.epoch += 1
        z, ld = output
        vl = -((-0.5 * (z ** 2).sum(-1)) + ld).mean().item()
        finite = bool(torch.isfinite(z).all() and torch.isfinite(ld).all() and math.isfinite(vl))
        self.nonfinite += int(not finite)
        if vl < self.best:
            self.best, self.best_epoch = vl, self.epoch
        elapsed = time.time() - self.started
        self.writer.writerow(dict(epoch=self.epoch, validation_loss=repr(vl), best_validation_loss=repr(self.best),
                                   best_epoch=self.best_epoch, no_improvement_epochs=self.epoch - self.best_epoch,
                                   estimated_optimizer_steps=(self.n // 512) * self.epoch, elapsed_s=elapsed, finite=finite))
        self.history.flush()
        atomic_json(PROGRESS, dict(updated_utc=utc_now(), pid=os.getpid(), status="TRAINING", config=self.cid,
                                   training_seed=self.seed, epoch=self.epoch, max_epochs=300,
                                   validation_loss=vl if math.isfinite(vl) else None,
                                   best_validation_loss=self.best if math.isfinite(self.best) else None,
                                   best_epoch=self.best_epoch, no_improvement_epochs=self.epoch - self.best_epoch,
                                   estimated_optimizer_steps=(self.n // 512) * self.epoch,
                                   run_elapsed_s=elapsed, phase_b_completed=self.completed, phase_b_failed=self.failed,
                                   nonfinite_validation_epochs=self.nonfinite, phase_a_unchanged=True))
        if self.epoch % 10 == 0:
            emit("EPOCH", self.cid, self.seed, self.epoch, "val", vl, "best", self.best, "best_ep", self.best_epoch)
        return None

    def close(self):
        self.history.close()


def main():
    CONTROL.mkdir(exist_ok=True)
    lock_file = (CONTROL / "runner.lock").open("a+b")
    lock_file.seek(0, 2)
    if lock_file.tell() == 0:
        lock_file.write(b"0")
        lock_file.flush()
    lock_file.seek(0)
    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    baseline = create_or_read_frozen_inputs()
    context = load_frozen_context(baseline)
    existing = list(csv.DictReader(RESULTS.open(encoding="utf-8", newline=""))) if RESULTS.exists() else []
    attempts = json.loads(ATTEMPTS.read_text(encoding="utf-8")) if ATTEMPTS.exists() else []
    assert len(attempts) <= 15 and len({(a["config"], a["training_seed"]) for a in attempts}) == len(attempts)
    requested = [(cid, seed) for cid in PHASE_B_CONFIGS for seed in SEEDS]
    assert all((r["config"], int(r["training_seed"])) in requested for r in existing)
    assert len({(r["config"], int(r["training_seed"])) for r in existing}) == len(existing)
    if not RESULTS.exists():
        atomic_csv(RESULTS, [], COLUMNS)
    for cid, seed in requested:
        if any(r["config"] == cid and int(r["training_seed"]) == seed for r in existing):
            emit("SKIP_RECORDED_ATTEMPT", cid, seed)
            continue
        n, layers = PHASE_B_CONFIGS[cid]
        previous = next((a for a in attempts if a["config"] == cid and a["training_seed"] == seed), None)
        if previous is not None:
            # A prior process stopped during this run: retain assets and never retrain it automatically.
            failed = {k: "" for k in COLUMNS}
            failed.update(config=cid, training_seed=seed, n_train=n, n_layers=layers, status="FAILED",
                           integrity_classification="FAILED_INTERRUPTED_NO_RETRY", error_type="InterruptedPriorAttempt",
                           original_error="The previously recorded attempt has no completed result; no automatic retry.",
                           started_utc=previous["started_utc"], finished_utc=utc_now(), attempt_number=previous["attempt_number"])
            existing.append(failed)
            previous["status"] = "FAILED_INTERRUPTED"
            atomic_csv(RESULTS, existing, COLUMNS)
            atomic_json(ATTEMPTS, attempts)
            emit("FAILED_INTERRUPTED_NO_RETRY", cid, seed)
            continue
        preservation = verify_frozen_inputs(baseline)
        checkpoint = OUT / f"{cid}_seed{seed}.pt"
        assert not checkpoint.exists(), "Unrecorded checkpoint exists; preserve it and do not overwrite or retry."
        attempt = dict(config=cid, training_seed=seed, status="RUNNING", started_utc=utc_now(),
                       pid=os.getpid(), attempt_number=len(attempts) + 1)
        attempts.append(attempt)
        assert len(attempts) <= 15
        atomic_json(ATTEMPTS, attempts)
        row = {k: "" for k in COLUMNS}
        row.update(config=cid, n_train=n, n_layers=layers, training_seed=seed,
                   started_utc=attempt["started_utc"], attempt_number=attempt["attempt_number"], proposals_filtered=0)
        observer, handle = None, None
        run_start = time.time()
        emit("RUN", cid, "seed", seed, "attempt", len(attempts), "of 15")
        try:
            torch.manual_seed(seed)
            np.random.seed(seed)
            flow = context["model_class"](6, 20, n_layers=layers, hidden_dim=128)
            observer = EpochObserver(cid, seed, n, context,
                                      sum(r["status"] == "COMPLETED" for r in existing),
                                      sum(r["status"] == "FAILED" for r in existing))
            handle = flow.register_forward_hook(observer.hook)
            flow, best, best_ep, last_ep, steps, duration = context["env"]["train"](
                flow, context["t_all"][:n], context["y_all"][:n], seed)
            handle.remove()
            handle = None
            observer.close()
            assert observer.epoch == last_ep and observer.best_epoch == best_ep and same(observer.best, best)
            assert steps == last_ep * (n // 512) and (last_ep == 300 or last_ep - best_ep == 30)
            assert all(torch.isfinite(v).all().item() for v in flow.state_dict().values())
            row.update(val_loss=best, best_epoch=best_ep, last_epoch=last_ep, optimizer_steps=steps,
                       train_time_s=duration, nonfinite_validation_epochs=observer.nonfinite,
                       checkpoint=str(checkpoint))
            # Persist the same selected state immediately after training; evaluation cannot choose it.
            with checkpoint.open("xb") as f:
                torch.save(flow.state_dict(), f)
                f.flush()
                os.fsync(f.fileno())
            row["checkpoint_sha256"] = sha(checkpoint)
            evaluation = evaluate_with_observation(flow, context)
            for key in ["mae_ref_all6", "mae_ref_active", "fast_seed_sd_all6", "fast_seed_sd_active",
                        "negative_all6_proposals", "negative_active_proposals", "negative_all6_elements", "total_fast_proposals"]:
                row[key] = evaluation[key]
            row["fast_seed_results"] = json.dumps(evaluation["fast_seed_results"])
            for fs, mae6, mae3 in evaluation["fast_seed_results"]:
                row[f"fast_seed{fs}_mae_ref_all6"] = mae6
                row[f"fast_seed{fs}_mae_ref_active"] = mae3
            verification = verify_checkpoint(checkpoint, cid, seed, row, context)
            assert evaluation["sample_calls"] == verification["sample_observation"]["sample_calls"]
            row.update(status="COMPLETED", integrity_classification="VALID_RUN_CHECKPOINT_VERIFIED",
                       checkpoint_replay_pass=True, replay_max_abs_mae_difference=verification["fast_seed_max_abs_difference"],
                       validation_replay_abs_difference=verification["comparisons"]["val_loss"]["abs_difference"])
            atomic_json(CONTROL / f"{cid}_seed{seed}_verification.json", verification)
        except Exception as error:
            original = traceback.format_exc()
            error_path = CONTROL / f"{cid}_seed{seed}_FAILED.txt"
            error_path.write_text(original, encoding="utf-8")
            row.update(status="FAILED", integrity_classification="FAILED_NO_RETRY", error_type=type(error).__name__,
                       original_error=str(error), traceback_path=str(error_path))
            if not row["train_time_s"]:
                row["train_time_s"] = time.time() - run_start
            if observer is not None:
                row["last_epoch"] = observer.epoch
                row["nonfinite_validation_epochs"] = observer.nonfinite
            emit("FAILED", cid, seed, type(error).__name__, str(error), "NO_RETRY")
        finally:
            if handle is not None:
                handle.remove()
            if observer is not None and not observer.history.closed:
                observer.close()
        row["finished_utc"] = utc_now()
        attempt["status"], attempt["finished_utc"] = row["status"], row["finished_utc"]
        existing.append(row)
        atomic_csv(RESULTS, existing, COLUMNS)
        atomic_json(ATTEMPTS, attempts)
        completed = sum(r["status"] == "COMPLETED" for r in existing)
        failed = sum(r["status"] == "FAILED" for r in existing)
        atomic_json(PROGRESS, dict(updated_utc=utc_now(), pid=os.getpid(), status="BETWEEN_RUNS", config=cid,
                                   training_seed=seed, phase_b_completed=completed, phase_b_failed=failed,
                                   attempted=len(existing), phase_a_unchanged=verify_frozen_inputs(baseline)["all_unchanged"]))
        emit("DONE" if row["status"] == "COMPLETED" else "FAILED_RECORDED", cid, seed,
             "val", row["val_loss"], "mae6", row["mae_ref_all6"], "mae3", row["mae_ref_active"],
             "time", row["train_time_s"], "negative_proposals", row["negative_all6_proposals"],
             "completed", completed, "failed", failed)
    assert len(attempts) == 15 and len(existing) == 15
    preservation = verify_frozen_inputs(baseline)
    atomic_json(PROGRESS, dict(updated_utc=utc_now(), pid=os.getpid(), status="TRAINING_STOPPED",
                               phase_b_completed=sum(r["status"] == "COMPLETED" for r in existing),
                               phase_b_failed=sum(r["status"] == "FAILED" for r in existing),
                               attempted=15, phase_b_training_done=True, new_training_attempts=15,
                               phase_a_unchanged=preservation["all_unchanged"], awaiting_full_readonly_audit=True))
    emit("ALL_PHASE_B_ATTEMPTS_DONE", 15, "TRAINING_STOPPED")
    lock_file.seek(0)
    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    lock_file.close()


if __name__ == "__main__":
    main()
