"""
R3-T1.1: targeted LHS to extend the surrogate training domain from [0.5,3.0] to cover the
field posterior support + 20% outward (which pushes e1 to ~3.5, the sole source of the
Case-B max delta_post=0.052). 250 points in the field-posterior box (e1 extended to 3.5)
+ 50 general extended-shell points (>=1 coord in [3.0,3.5]) = 300.
"""
import csv
from pathlib import Path
import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).parent
np.random.seed(42)

# field posterior box: support (mean +-2std) expanded ~20% outward, e1 -> 3.5, clamp [0.5,3.5]
lo = np.array([2.00, 0.50, 0.50, 0.50, 0.50, 0.50])
hi = np.array([3.50, 2.60, 2.90, 2.60, 2.60, 2.40])
p1 = lo + qmc.LatinHypercube(d=6, seed=42).random(250) * (hi - lo)

# general extended-shell coverage: LHS over [0.5,3.5]^6, force >=1 coord into [3.0,3.5]
p2 = 0.5 + qmc.LatinHypercube(d=6, seed=7).random(50) * (3.5 - 0.5)
for i in range(50):
    d = np.random.randint(6)
    p2[i, d] = 3.0 + np.random.rand() * 0.5

pts = np.clip(np.vstack([p1, p2]), 0.5, 3.5)
with open(HERE / "samples_extend.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow([f"e{i}" for i in range(1, 7)])
    for p in pts:
        w.writerow([f"{v:.6f}" for v in p])
n_ext = int((pts.max(1) > 3.0).sum())
print(f"wrote {len(pts)} LHS points -> samples_extend.csv")
print(f"  points with a coord > 3.0 (new region): {n_ext}")
print(f"  e1 range [{pts[:,0].min():.2f}, {pts[:,0].max():.2f}]  overall max {pts.max():.2f}")
