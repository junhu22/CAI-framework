"""Continue unattempted slots; fix telemetry I/O without changing train/evaluate.

Completed and FAILED slots are skipped by the original controller. Progress is
append-only JSONL, so a progress-file rename cannot interrupt validation.
"""
import os
import time
import uuid
from pathlib import Path

import common as C
import run_phase_b_frozen as R

PROGRESS_JOURNAL = C.CONTROL / "PROGRESS_V2.jsonl"
ORIGINAL_ATOMIC_JSON = C.atomic_json


def durable_bytes(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing." + uuid.uuid4().hex)
    with temporary.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(100):
        try:
            os.replace(temporary, path)
            return
        except PermissionError as error:
            with (C.CONTROL / "IO_RENAME_WARNINGS.log").open("a", encoding="utf-8") as log:
                log.write(C.utc_now() + " " + repr(error) + " rename_attempt=" + str(attempt + 1) + "\n")
            if attempt == 99:
                raise
            time.sleep(0.05)


def progress_or_atomic_json(path, data):
    if Path(path).resolve() == R.PROGRESS.resolve():
        encoded = (C.json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        with PROGRESS_JOURNAL.open("ab") as f:
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())
        return
    ORIGINAL_ATOMIC_JSON(path, data)


def main():
    C.atomic_bytes = durable_bytes
    R.atomic_json = progress_or_atomic_json
    C.emit("IO_FIXED_CONTROLLER_STARTED", "progress=append-only JSONL", "no_training_or_evaluation_rule_change")
    R.main()


if __name__ == "__main__":
    main()
