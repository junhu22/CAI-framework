"""Switch controllers only after a recorded run ends, never during training."""
import csv
import ctypes
import io
import json
import msvcrt
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

import common as C

PID = 23156
k = ctypes.WinDLL("kernel32", use_last_error=True)
n = ctypes.WinDLL("ntdll")
k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
k.OpenProcess.restype = wintypes.HANDLE
k.CloseHandle.argtypes = [wintypes.HANDLE]
k.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
k.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                          wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
k.CreateFileW.restype = wintypes.HANDLE
n.NtSuspendProcess.argtypes = [wintypes.HANDLE]
n.NtResumeProcess.argtypes = [wintypes.HANDLE]


def shared_text(path):
    handle = k.CreateFileW(str(path), 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    descriptor = msvcrt.open_osfhandle(int(handle), os.O_RDONLY | os.O_BINARY)
    with os.fdopen(descriptor, "rb") as f:
        return f.read().decode("utf-8")


def main():
    handle = k.OpenProcess(0x1801, False, PID)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    C.emit("SAFE_CONTROLLER_SWITCH_ARMED", "old_pid", PID, "wait_until_recorded_attempts>=3_and_no_unrecorded_attempt")
    while True:
        if k.WaitForSingleObject(handle, 0) == 0:
            raise RuntimeError("Old controller ended before a verified safe switch; inspect its state manually.")
        rows = list(csv.DictReader(io.StringIO(shared_text(C.OUT / "PHASE_B_RESULTS.csv"))))
        if len(rows) < 3:
            time.sleep(0.025)
            continue
        if n.NtSuspendProcess(handle) != 0:
            raise RuntimeError("Cannot suspend controller for safe-boundary verification.")
        try:
            rows = list(csv.DictReader(io.StringIO(shared_text(C.OUT / "PHASE_B_RESULTS.csv"))))
            attempts = json.loads(shared_text(C.CONTROL / "ATTEMPTS.json"))
            safe = len(rows) == len(attempts) and len(rows) >= 3 and len(rows) < 15
            safe = safe and {(r['config'], int(r['training_seed'])) for r in rows} == {(a['config'], a['training_seed']) for a in attempts}
            if not safe:
                n.NtResumeProcess(handle)
                time.sleep(0.025)
                continue
            evidence = dict(utc=C.utc_now(), old_pid=PID, recorded_results=len(rows), registered_attempts=len(attempts),
                            no_registered_unfinished_attempt=True, latest_result=rows[-1],
                            reason="Fix observer progress-file replacement; preserve every completed/FAILED attempt and original frozen training/evaluation definitions.")
            if not k.TerminateProcess(handle, 204):
                raise ctypes.WinError(ctypes.get_last_error())
            k.WaitForSingleObject(handle, 5000)
            C.atomic_json(C.CONTROL / "CONTROLLER_SWITCH_EVIDENCE.json", evidence)
            break
        except BaseException:
            n.NtResumeProcess(handle)
            raise
    k.CloseHandle(handle)
    by_key = {(r['config'], int(r['training_seed'])): r for r in rows}
    for attempt in attempts:
        result = by_key[attempt['config'], attempt['training_seed']]
        attempt.update(status=result['status'], finished_utc=result['finished_utc'])
    C.atomic_json(C.CONTROL / "ATTEMPTS.json", attempts)
    hashes = json.loads((C.CONTROL / "SCRIPT_HASHES.json").read_text(encoding="utf-8"))
    hashes.update(io_fix="Epoch progress changed from atomic replacement to append-only JSONL; critical outputs use unique temporary files and bounded rename retries. No seed is retrained.",
                  controller_switch_evidence=str(C.CONTROL / "CONTROLLER_SWITCH_EVIDENCE.json"))
    for name in ["run_phase_b_io_fixed.py", "takeover_after_run.py"]:
        path = C.CONTROL / name
        hashes['files'][str(path)] = C.sha(path)
    C.atomic_json(C.CONTROL / "SCRIPT_HASHES.json", hashes)
    with (C.CONTROL / "controller_v2_stdout.log").open("ab") as log:
        process = subprocess.Popen([sys.executable, "-B", "-u", str(C.CONTROL / "run_phase_b_io_fixed.py")],
                                   cwd=C.ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=0x08000000)
        C.atomic_json(C.CONTROL / "CONTROLLER_V2_PROCESS.json", dict(pid=process.pid, started_utc=C.utc_now()))
        C.emit("SAFE_CONTROLLER_SWITCH_COMPLETE", "new_pid", process.pid, "recorded_attempts_preserved", len(rows))
        code = process.wait()
    C.emit("IO_FIXED_CONTROLLER_EXIT", code)
    if code == 0:
        subprocess.run([sys.executable, "-B", "-u", str(C.CONTROL / "audit_full.py")], cwd=C.ROOT, check=True)


if __name__ == "__main__":
    main()
