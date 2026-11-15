"""
E1 Step 8 (harvest): compare surrogate f_hat vs true FE f at the 151 posterior-region
theta points. Computes per-mode relative error, delta_post, Delta-log-likelihood, and
the recomputed d_TV bound (paper Eq. 10) per case; compares to the global test-set
delta = 1.2e-4.

Uses the inference surrogate (member_0) for each case, matching the paper's likelihood.
"""
import sys, csv, json, math
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, r"C:\neutra-hmc-surrogate\surrogate_training\revision_experiments")
import revlib as R

HERE = Path(__file__).resolve().parent
DELTA_GLOBAL = 1.2e-4
KAPPA_SQ = {"A": 1e-3, "B": 1e-3, "C": 9e-4}

meta = json.load(open(HERE / "points_metadata.json"))
# inference surrogates (member_0) per case
sur = {"A": R.load_d6(), "B": R.load_broad(), "C": R.load_coupled()}

def read_fe(tag):
    """Map local_idx -> FE 20 freqs from fe_case{tag}.csv."""
    path = HERE / f"fe_case{tag}.csv"
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[int(row["id"])] = [float(row[f"f{j}"]) for j in range(1, 21)]
    return out

def dtv_bound(delta, M, ksq):
    return 1.0 - math.exp(-M * delta * (2 + delta) / (2 * ksq))

per_point = []
case_stat = {}
for tag in ("A", "B", "C"):
    fe = read_fe(tag)
    pts = [m for m in meta if m["case"] == tag]
    ksq = KAPPA_SQ[tag]
    dpost_K = dpost_all = 0.0
    dll_list = []
    n_valid = 0
    for m in pts:
        li = m["local_idx"]
        if li not in fe:
            per_point.append(dict(case=tag, idx=li, kind=m["kind"], valid=0))
            continue
        n_valid += 1
        th = torch.tensor(m["theta"], dtype=torch.float32).reshape(1, -1)
        with torch.no_grad():
            fhat = sur[tag](th).squeeze(0).numpy()      # member_0, 20 modes
        ffe = np.array(fe[li])                            # true FE, 20 modes
        y = np.array(m["y_obs"]); K = m["K"]
        rel_ffe = np.abs(fhat - ffe) / ffe                # matches paper delta (/f)
        rel_y = np.abs(fhat - ffe)[:K] / y[:K]            # manual variant (/y)
        pd_K = float(rel_ffe[:K].max())
        pd_all = float(rel_ffe.max())
        dpost_K = max(dpost_K, pd_K); dpost_all = max(dpost_all, pd_all)
        # Delta log-likelihood over K modes (relative-error Gaussian)
        ll_fe = -0.5 / ksq * (((ffe[:K] - y[:K]) / y[:K]) ** 2).sum()
        ll_hat = -0.5 / ksq * (((fhat[:K] - y[:K]) / y[:K]) ** 2).sum()
        dll = abs(ll_hat - ll_fe)
        dll_list.append(dll)
        per_point.append(dict(case=tag, idx=li, kind=m["kind"], valid=1,
            delta_point_K=pd_K, delta_point_all=pd_all, rel_y_max=float(rel_y.max()),
            dloglik=float(dll)))
    K = pts[0]["K"]
    case_stat[tag] = dict(
        n_total=len(pts), n_valid=n_valid, K=K, kappa_sq=ksq,
        delta_post_K=dpost_K, delta_post_all=dpost_all,
        dloglik_max=float(np.max(dll_list)), dloglik_mean=float(np.mean(dll_list)),
        dtv_bound_post=dtv_bound(dpost_K, K, ksq),
        dtv_bound_global=dtv_bound(DELTA_GLOBAL, K, ksq),
        dloglik_bound_post=K * dpost_K * (2 + dpost_K) / (2 * ksq))
    print(f"[Case {tag}] valid={n_valid}/{len(pts)}  delta_post(K)={dpost_K:.3e} "
          f"(global {DELTA_GLOBAL:.1e})  dLL max={case_stat[tag]['dloglik_max']:.3e}  "
          f"d_TV<= {case_stat[tag]['dtv_bound_post']:.3e}")

# ── CSV ─────────────────────────────────────────────────────────────────
with open(HERE / "E1_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["case", "idx", "kind", "valid", "delta_point_K", "delta_point_all",
                "rel_y_max", "dloglik"])
    for p in per_point:
        w.writerow([p["case"], p["idx"], p["kind"], p["valid"],
                    f"{p.get('delta_point_K', float('nan')):.4e}",
                    f"{p.get('delta_point_all', float('nan')):.4e}",
                    f"{p.get('rel_y_max', float('nan')):.4e}",
                    f"{p.get('dloglik', float('nan')):.4e}"])

# ── summary md ──────────────────────────────────────────────────────────
names = {"A": "Case A — D=6 Obs1 (K=20)", "B": "Case B — field data (K=6)",
         "C": "Case C — coupled building (K=6)"}
L = ["# E1 — Surrogate accuracy in the HMC posterior region (R1#3)\n",
     "FE modal analysis (ANSYS 16.0) at posterior-region theta points vs the inference "
     "surrogate f_hat (member_0). delta = max relative error ||(f_hat - f_FE)/f_FE||_inf; "
     f"global test-set delta = {DELTA_GLOBAL:.1e}. d_TV via Eq. 10.\n",
     "| case | valid pts | delta_post (K-mode) | delta_post (20-mode) | vs global | "
     "max Dloglik | d_TV bound (post) | d_TV bound (global) |",
     "|---|---|---|---|---|---|---|---|"]
for tag in ("A", "B", "C"):
    s = case_stat[tag]
    ratio = s["delta_post_K"] / DELTA_GLOBAL
    L.append(f"| {names[tag]} | {s['n_valid']}/{s['n_total']} | {s['delta_post_K']:.3e} | "
             f"{s['delta_post_all']:.3e} | {ratio:.1f}x | {s['dloglik_max']:.3e} | "
             f"{s['dtv_bound_post']:.3e} | {s['dtv_bound_global']:.3e} |")
L += ["",
      "**Reading:** delta_post is the surrogate's worst-case relative error evaluated *inside* "
      "the region the HMC sampler actually explores (posterior draws + mode centres + MAP), "
      "rather than on the LHS prior grid. If delta_post is the same order as the global "
      "delta, the surrogate-error and d_TV guarantees established on the prior domain carry "
      "over to the posterior region, closing the gap R1#3 raised.",
      "",
      "Per-point detail in `E1_results.csv`."]
(HERE / "E1_results_summary.md").write_text("\n".join(L), encoding="utf-8")
json.dump(case_stat, open(HERE / "E1_case_stats.json", "w"), indent=1)
print("\nWrote E1_results.csv, E1_results_summary.md, E1_case_stats.json")
