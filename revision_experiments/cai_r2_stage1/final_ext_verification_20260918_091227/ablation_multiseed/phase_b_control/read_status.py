"""Read live outputs with Windows delete sharing; never load/train a model."""
import csv
import ctypes
import io
import json
import msvcrt
import os
from ctypes import wintypes
from pathlib import Path

CONTROL = Path(__file__).resolve().parent
OUT = CONTROL.parent
k = ctypes.WinDLL("kernel32", use_last_error=True)
k.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                         wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
k.CreateFileW.restype = wintypes.HANDLE


def shared_text(path):
    handle = k.CreateFileW(str(path), 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    descriptor = msvcrt.open_osfhandle(int(handle), os.O_RDONLY | os.O_BINARY)
    with os.fdopen(descriptor, "rb") as f:
        return f.read().decode("utf-8")


def main():
    journal = CONTROL / "PROGRESS_V2.jsonl"
    if journal.exists():
        lines = shared_text(journal).splitlines()
        progress = None
        for line in reversed(lines):
            try:
                progress = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        assert progress is not None
    else:
        progress = json.loads(shared_text(CONTROL / "PROGRESS.json"))
    selected = ["updated_utc", "pid", "status", "config", "training_seed", "epoch", "best_epoch",
                "validation_loss", "best_validation_loss", "no_improvement_epochs", "run_elapsed_s",
                "nonfinite_validation_epochs", "phase_b_completed", "phase_b_failed", "attempted"]
    rows = list(csv.DictReader(io.StringIO(shared_text(OUT / "PHASE_B_RESULTS.csv"))))
    print(json.dumps(dict(progress={name: progress[name] for name in selected if name in progress},
                          recorded_attempts=len(rows), completed=sum(r["status"] == "COMPLETED" for r in rows),
                          failed=sum(r["status"] == "FAILED" for r in rows),
                          latest_record={name: rows[-1].get(name) for name in ["config", "training_seed", "status",
                          "mae_ref_all6", "mae_ref_active", "negative_all6_proposals", "checkpoint_replay_pass"]} if rows else None),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
