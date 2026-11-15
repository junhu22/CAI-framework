"""Merge partial_train.csv + dataset_train_part2.csv into dataset_train.csv"""
import csv
from config import *

KEEP = PARAM_NAMES + FREQ_NAMES

rows = []
for fname in ["partial_train.csv", "dataset_train_part2.csv"]:
    try:
        count = 0
        for r in csv.DictReader(open(fname, encoding="utf-8")):
            rows.append({k: r[k] for k in KEEP})
            count += 1
        print(f"  {fname}: {count} rows", flush=True)
    except FileNotFoundError:
        print(f"  {fname}: NOT FOUND", flush=True)

print(f"Total: {len(rows)} rows", flush=True)
with open("dataset_train.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=KEEP)
    w.writeheader()
    w.writerows(rows)
print("Saved dataset_train.csv", flush=True)
