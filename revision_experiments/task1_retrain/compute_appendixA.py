"""
Task 3: numbers for the Appendix A correction memo.
- Corrected Eq.10 global bound (M=20, delta=1.2e-4, kappa=0.03) vs the paper's 2.7e-3 slip.
- Per-case delta_post -> standard d_TV bound and log-lik bound.
- Measured |Delta ell| (max/mean) from E1.
- Residual-aware tightened bound: |Delta ell| <= M*delta*(2*rho + delta)/(2 kappa^2),
  rho = max relative residual |(f_FE - y)/y| in the posterior region. Verify it upper-bounds
  the measured |Delta ell|.
"""
import csv, json, math
from pathlib import Path
import numpy as np

E1 = Path(r"C:\neutra-hmc-surrogate\surrogate_training\revision_experiments\E1_posterior_region_FE")
HERE = Path(__file__).parent
stats = json.load(open(E1 / "E1_case_stats.json"))
meta = json.load(open(E1 / "points_metadata.json"))

def dtv(delta, M, ksq):  # Eq.10
    return 1.0 - math.exp(-M * delta * (2 + delta) / (2 * ksq))
def ll_bound_std(delta, M, ksq):
    return M * delta * (2 + delta) / (2 * ksq)
def ll_bound_resaware(delta, rho, M, ksq):
    return M * delta * (2 * rho + delta) / (2 * ksq)

# ── global corrected bound ──────────────────────────────────────────────
M, dg, kappa = 20, 1.2e-4, 0.03
ksq = kappa ** 2
B = M * dg * (2 + dg) / (2 * ksq)
paper_wrong = M * dg * (2 + dg) / 1.8      # the /1.8 (instead of /0.0018) slip
glob = {"M": M, "delta": dg, "kappa": kappa, "two_ksq": 2 * ksq,
        "B_correct": B, "d_TV_le": dtv(dg, M, ksq),
        "paper_reported": 2.7e-3, "paper_recompute_with_1.8": paper_wrong,
        "slip_factor": B / paper_wrong}

# ── per-case residual rho from FE ───────────────────────────────────────
def read_fe(tag):
    out = {}
    for row in csv.DictReader(open(E1 / f"fe_case{tag}.csv")):
        out[int(row["id"])] = [float(row[f"f{j}"]) for j in range(1, 21)]
    return out

KSQ = {"A": 1e-3, "B": 1e-3, "C": 9e-4}
cases = {}
for tag in ("A", "B", "C"):
    fe = read_fe(tag)
    pts = [m for m in meta if m["case"] == tag]
    K = pts[0]["K"]; ksq_c = KSQ[tag]
    rho = 0.0
    for m in pts:
        li = m["local_idx"]
        if li not in fe: continue
        ffe = np.array(fe[li]); y = np.array(m["y_obs"])
        r = np.abs((ffe[:K] - y[:K]) / y[:K])
        rho = max(rho, float(r.max()))
    dpost = stats[tag]["delta_post_K"]
    cases[tag] = {
        "K": K, "kappa_sq": ksq_c, "delta_post": dpost, "rho_maxres": rho,
        "dtv_bound_post": dtv(dpost, K, ksq_c),
        "llbound_standard": ll_bound_std(dpost, K, ksq_c),
        "llbound_resaware": ll_bound_resaware(dpost, rho, K, ksq_c),
        "measured_dll_max": stats[tag]["dloglik_max"],
        "measured_dll_mean": stats[tag]["dloglik_mean"],
    }
    c = cases[tag]
    c["resaware_holds"] = c["llbound_resaware"] >= c["measured_dll_max"]
    c["tightening_factor"] = c["llbound_standard"] / max(c["llbound_resaware"], 1e-30)

out = {"global": glob, "cases": cases}
json.dump(out, open(HERE / "appendixA_numbers.json", "w"), indent=2)
print(json.dumps(out, indent=2))
