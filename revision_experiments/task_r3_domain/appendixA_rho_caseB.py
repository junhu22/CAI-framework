"""Appendix-A table support: residual rho and residual-aware bound for Case B (field,
FROZEN narrow-prior config).  Pure post-processing of frozen artifacts -- no new sampling,
no new FE runs.  Reads the 50 ANSYS runs already stored in fe_caseB_narrow.csv.

rho = max_r |f_r^FE - y_r| / y_r  over the posterior draws and the M=6 measured modes.
Standard bound      : M*delta*(2   + delta)/(2 kappa^2)     [Eq. 10 exponent]
Residual-aware bound: M*delta*(2*rho + delta)/(2 kappa^2)   [Appendix A, Eq. A.5]
"""
import csv, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
Y = np.array([0.91, 1.08, 1.47, 2.55, 3.16, 3.81])   # revlib.FIELD_MEASURED
M, KAPPA_SQ = 6, 1e-3
DELTA_POST = 0.009425324496418655                     # caseB_narrow_stats.json
MEASURED_DLL_MAX = 0.2883409616634163                 # caseB_narrow_stats.json

rows = list(csv.DictReader(open(HERE / "fe_caseB_narrow.csv")))
F = np.array([[float(r[f"f{i}"]) for i in range(1, M + 1)] for r in rows])
res = np.abs(F - Y) / Y
rho = float(res.max())

std = M * DELTA_POST * (2 + DELTA_POST) / (2 * KAPPA_SQ)
ra = M * DELTA_POST * (2 * rho + DELTA_POST) / (2 * KAPPA_SQ)

out = {
    "n_fe_points": len(rows),
    "rho_max": rho,
    "rho_per_point_max_median": float(np.median(res.max(1))),
    "delta_post": DELTA_POST,
    "bound_standard": std,
    "bound_residual_aware": ra,
    "tightening_factor": std / ra,
    "measured_dll_max": MEASURED_DLL_MAX,
    "residual_aware_holds": bool(ra >= MEASURED_DLL_MAX),
    "note": "Case B row of Table A.1; A and C rho/bounds come from task1_retrain/appendixA_numbers.json",
}
json.dump(out, open(HERE / "appendixA_rho_caseB.json", "w"), indent=2)
for k, v in out.items():
    print(f"{k}: {v}")
