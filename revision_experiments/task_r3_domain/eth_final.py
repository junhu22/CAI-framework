"""
R3-T4: final E-th field/damage rows with ext models -> the sailboat-damage reclassification
(round-1 ESS 0.04 with corrupted CNF -> ~0.5 with the clean/ext CNF => routes to FAST).
Coupled rows unchanged (coupled models untouched).
"""
import sys, json
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(6)
sys.path.insert(0, r"C:\neutra-hmc-surrogate\surrogate_training\revision_experiments")
import revlib as R
HERE = Path(__file__).parent
SEEDS = [42, 43, 44, 45, 46]

surE = R.load_broad_ext(); cnfE = R.load_broad_ext_cnf()
def ess(y_cnf, y_like):
    lp = R.make_log_post(surE, y_like, "broad"); yt = torch.as_tensor(y_cnf, dtype=torch.float32)
    vals = []
    for s in SEEDS:
        torch.manual_seed(s); tf = cnfE.sample(yt, 2000)
        _, r = R.importance_weight_ess(tf, lp.forward, cnfE, yt); vals.append(r)
    return float(np.mean(vals)), float(np.std(vals))

rows = []
# field data (real, off-manifold)
yf = R.field_yfull(surE)
m, sd = ess(yf, R.FIELD_MEASURED); rows.append(("Field data (Lam 2019)", "real/OOD", m, sd, 0.0005))
# sailboat damage (synthetic on-manifold) -> reclassified
with torch.no_grad():
    yd = surE(torch.tensor([0.75,1,1,1,1,1], dtype=torch.float32).reshape(1,-1)).squeeze(0).numpy()
m, sd = ess(yd, list(yd)); rows.append(("Sailboat damage (e1=0.75)", "synthetic-damage", m, sd, 0.0204))

TAUS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
out = {"tau_grid": TAUS, "rows": [], "coupled_note": "coupled models untouched; round-1 values stand (normal 0.032, damage 0.128)"}
print("Final E-th field/damage rows (ext models):")
for name, cat, mu, sd, old in rows:
    dec = ["fast" if mu >= t else "exact" for t in TAUS]
    out["rows"].append(dict(name=name, category=cat, ess=mu, ess_std=sd, old_ess=old, route=dec))
    print(f"  {name:26s} ESS/N={mu:.4f}+-{sd:.4f} (round-1 {old}) route@0.5={'fast' if mu>=0.5 else 'exact'}", flush=True)
json.dump(out, open(HERE / "eth_final_results.json", "w"), indent=2)
print("Wrote eth_final_results.json", flush=True)
