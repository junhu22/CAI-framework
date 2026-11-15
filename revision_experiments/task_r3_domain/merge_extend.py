"""
R3-T1.2: merge the extension FE data (300 pts, up to e=3.5) into the clean dataset,
producing dataset_train_broad_ext.csv (domain [0.5,3.5]). ANSYS output is 28-col
(id,e1..e6,f1..f20,time_s) -> take the 26 (e1..e6,f1..f20). Validate + dedup.
"""
import csv, json
from pathlib import Path
import numpy as np

SURR = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
HERE = Path(__file__).parent
CLEAN = SURR / "dataset_train_broad_clean.csv"
EXT = HERE / "dataset_extend_new.csv"
OUT = SURR / "dataset_train_broad_ext.csv"
COLS = [f"e{i}" for i in range(1, 7)] + [f"f{i}" for i in range(1, 21)]

def load_named(path):
    rows = []
    for r in csv.DictReader(open(path)):
        try:
            v = [float(r[c]) for c in COLS]
        except (KeyError, ValueError):
            continue
        params, freqs = np.array(v[:6]), np.array(v[6:])
        if (params < 0.3).any() or (params > 4.0).any(): continue
        if (freqs <= 0).any() or (np.diff(freqs) < -1e-6).any(): continue
        rows.append(tuple(round(x, 8) for x in v))
    return rows

clean = load_named(CLEAN)
ext = load_named(EXT)
before = len(clean) + len(ext)
merged = sorted(set(clean) | set(ext))
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(COLS); w.writerows(merged)
arr = np.array(merged)
report = {"clean_rows": len(clean), "ext_rows": len(ext),
          "merged_unique": len(merged), "dupes_removed": before - len(merged),
          "theta_max_per_dim": arr[:, :6].max(0).round(3).tolist(),
          "n_with_coord_gt_3": int((arr[:, :6].max(1) > 3.0).sum())}
json.dump(report, open(HERE / "merge_report.json", "w"), indent=2)
print(json.dumps(report, indent=2))
