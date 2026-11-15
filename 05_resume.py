"""Collect completed ANSYS results and prepare remaining samples for resume."""
import csv
from pathlib import Path
from config import *

# 1. Collect completed results from batch_runs/
print("Scanning batch_runs/...", flush=True)
results = []
corrupt = 0
for d in sorted(Path("batch_runs").glob("s*")):
    ff = d / "modal_freqs.txt"
    if not ff.exists():
        continue
    sid = int(d.name[1:])
    try:
        raw = ff.read_bytes()
        if b"\x00" in raw:
            corrupt += 1
            continue
        freqs = [float(x) for x in raw.decode("utf-8").split() if x.strip()]
        if len(freqs) < N_MODES:
            continue
        mac = (d / "run.mac").read_text(encoding="utf-8")
        params = []
        for line in mac.splitlines():
            s = line.strip()
            if s.startswith("*SET,e") and "e10" not in s.split(",")[0]:
                val = float(s.split(",")[2]) / 2e10
                params.append(val)
        if len(params) == N_PARAMS:
            row = {"id": sid}
            for j in range(N_PARAMS):
                row[PARAM_NAMES[j]] = params[j]
            for j in range(N_MODES):
                row[FREQ_NAMES[j]] = freqs[j]
            results.append(row)
    except Exception as e:
        corrupt += 1
        continue

results.sort(key=lambda r: r["id"])
print(f"Collected {len(results)} successful, {corrupt} corrupt (skipped)", flush=True)

# Save partial dataset
with open("partial_train.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id"] + PARAM_NAMES + FREQ_NAMES)
    w.writeheader()
    w.writerows(results)
print("Saved partial_train.csv", flush=True)

# 2. Find how many were attempted
max_id = max(r["id"] for r in results) if results else -1
print(f"Highest completed sample ID: {max_id}", flush=True)

# 3. Generate remaining samples CSV
all_rows = list(csv.DictReader(open("samples_train.csv", encoding="utf-8")))
skip = max_id + 1
remaining = all_rows[skip:]
print(f"Total samples: {len(all_rows)}, Skip: {skip}, Remaining: {len(remaining)}", flush=True)

with open("samples_remaining.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
    w.writeheader()
    w.writerows(remaining)
print("Saved samples_remaining.csv", flush=True)
print(f"\nNext step: run 02_run_batch.py --samples samples_remaining.csv --parallel 2", flush=True)
