"""
E1 Step 1: build 150 posterior-region theta points across 3 cases and write
ANSYS-ready sample CSVs + metadata for harvest.

Case A (D=6 Obs1): 47 random exact-mode posterior + mean + MAP + 2 near-MAP.
Case B (field, broad): 41 random exact-mode + 3 centers x (center, +1sigma, -1sigma).
Case C (coupled D=3): 47 random exact-mode + mean + theta_MAP + damage mean.
"""
import sys, csv, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, r"C:\neutra-hmc-surrogate\surrogate_training\revision_experiments")
import revlib as R

torch.manual_seed(42); np.random.seed(42)
HERE = Path(__file__).resolve().parent
N_EXACT = 500                       # exact-mode samples to draw per case
CLAMP = (0.3, 4.5)

FIELD_STD = {
    "narrow":     [0.4240, 0.4746, 0.2109, 0.2515, 0.2872, 0.5163],
    "broad":      [0.5154, 0.4285, 0.2481, 0.2983, 0.3569, 0.2990],
    "broad_mass": [0.7066, 0.4098, 0.2505, 0.3433, 0.3130, 0.3977],
}

def clamp(a): return np.clip(a, *CLAMP)

def logpost_np(log_post, thetas):
    t = torch.tensor(np.asarray(thetas), dtype=torch.float32)
    with torch.no_grad():
        return log_post.forward(t).numpy()

# ─────────────────────────────────────────────────────────────────────────
# Case A — D=6 Obs1, standard NeuTra-HMC (theta-space CNF)
# ─────────────────────────────────────────────────────────────────────────
def case_A():
    sur = R.load_d6(); cnf = R.load_d6_cnf()
    theta_true, y = R.load_obs1()
    lp = R.make_log_post(sur, y, "d6")
    y_t = torch.tensor(y, dtype=torch.float32)
    sampler = R.NeuTraHMC_Conditional(lp.forward, cnf, y_t, step_size=0.05, n_leapfrog=20)
    torch.manual_seed(42)
    chain, ar = sampler.run(torch.ones(6), n_samples=N_EXACT, n_burnin=200)
    chain = chain.numpy()
    print(f"[A] NeuTra-HMC AR={ar:.2f} mean={chain.mean(0).round(3)}")
    idx = np.random.choice(len(chain), 47, replace=False)
    pts, meta = [], []
    for i in idx:
        pts.append(clamp(chain[i])); meta.append("random")
    mean = clamp(chain.mean(0)); pts.append(mean); meta.append("post_mean")
    lps = logpost_np(lp, chain); mapi = int(lps.argmax()); themap = chain[mapi]
    pts.append(clamp(themap)); meta.append("MAP")
    for k in range(2):
        pts.append(clamp(themap + 0.03*np.random.randn(6))); meta.append(f"near_MAP_{k+1}")
    return sur_ens_pred(R.load_d6_ensemble(), pts), pts, meta, y, 20

# ─────────────────────────────────────────────────────────────────────────
# Case B — field, mass-matrix NeuTra-HMC (broad_mass config)
# ─────────────────────────────────────────────────────────────────────────
def case_B():
    sur = R.load_broad(); cnf = R.load_broad_cnf()
    y20 = torch.tensor(R.field_yfull(sur))
    lp = R.make_log_post(sur, R.FIELD_MEASURED, "broad")
    sampler = R.NeuTraHMC_MassDiag(lp.forward, cnf, y20, step_size=0.04,
                                   n_leapfrog=40, mass_diag=R.FIELD_MASS)
    chain, ar = sampler.run(torch.tensor(R.FIELD_CENTERS["broad_mass"], dtype=torch.float32),
                            n_samples=N_EXACT, n_burnin=100, seed=42)
    chain = chain.numpy()
    print(f"[B] mass NeuTra-HMC AR={ar:.2f} mean={chain.mean(0).round(3)}")
    idx = np.random.choice(len(chain), 41, replace=False)
    pts, meta = [], []
    for i in idx:
        pts.append(clamp(chain[i])); meta.append("random")
    for name in ("narrow", "broad", "broad_mass"):
        c = np.array(R.FIELD_CENTERS[name]); s = np.array(FIELD_STD[name])
        pts.append(clamp(c));     meta.append(f"{name}_center")
        pts.append(clamp(c + s)); meta.append(f"{name}_+1sig")
        pts.append(clamp(c - s)); meta.append(f"{name}_-1sig")
    return sur_ens_pred(R.load_broad_ensemble(), pts), pts, meta, R.FIELD_MEASURED, 6

# ─────────────────────────────────────────────────────────────────────────
# Case C — coupled D=3, mass-matrix NeuTra-HMC
# ─────────────────────────────────────────────────────────────────────────
def case_C():
    sur = R.load_coupled(); cnf = R.load_coupled_cnf()
    y20 = torch.tensor(R.COUP_NORMAL_YFULL, dtype=torch.float32)
    lp = R.make_log_post(sur, R.COUP_MEASURED, "coupled")
    sampler = R.NeuTraHMC_MassDiag(lp.forward, cnf, y20, step_size=0.04,
                                   n_leapfrog=40, mass_diag=R.COUP_MASS)
    chain, ar = sampler.run(torch.tensor(R.COUP_NORMAL_MEAN, dtype=torch.float32),
                            n_samples=N_EXACT, n_burnin=100, seed=42)
    chain = chain.numpy()
    print(f"[C] coupled mass NeuTra-HMC AR={ar:.2f} mean={chain.mean(0).round(3)}")
    idx = np.random.choice(len(chain), 47, replace=False)
    pts, meta = [], []
    for i in idx:
        pts.append(clamp(chain[i])); meta.append("random")
    pts.append(clamp(chain.mean(0)));                          meta.append("post_mean")
    pts.append(clamp(np.array(R.COUP_THETA_MAP)));             meta.append("theta_MAP")
    pts.append(clamp(np.array(R.COUP_DAMAGE_MEAN)));           meta.append("damage_mean")
    return sur_ens_pred([sur], pts), pts, meta, R.COUP_MEASURED, 6

def sur_ens_pred(members, pts):
    """Ensemble-mean surrogate prediction (20 modes) at each theta point."""
    t = torch.tensor(np.asarray(pts), dtype=torch.float32)
    preds = []
    for m in members:
        with torch.no_grad():
            preds.append(m(t).numpy())
    return np.mean(preds, axis=0)   # (n_pts, 20)

def write_case(tag, cols, fhat, pts, meta, y_obs, K, records):
    csv_path = HERE / f"samples_case{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for p in pts:
            w.writerow([f"{v:.10f}" for v in p])
    for i, (p, mt, fp) in enumerate(zip(pts, meta, fhat)):
        records.append(dict(case=tag, local_idx=i, kind=mt,
                            theta=[float(v) for v in p],
                            fhat=[float(v) for v in fp],
                            y_obs=[float(v) for v in y_obs], K=int(K)))
    print(f"  wrote {csv_path.name}: {len(pts)} points")

def main():
    records = []
    fA, pA, mA, yA, kA = case_A()
    write_case("A", ["e1","e2","e3","e4","e5","e6"], fA, pA, mA, yA, kA, records)
    fB, pB, mB, yB, kB = case_B()
    write_case("B", ["e1","e2","e3","e4","e5","e6"], fB, pB, mB, yB, kB, records)
    fC, pC, mC, yC, kC = case_C()
    write_case("C", ["e1_column","e2_roof","e3_beam"], fC, pC, mC, yC, kC, records)
    (HERE / "points_metadata.json").write_text(json.dumps(records, indent=1))
    print(f"\nTotal points: {len(records)}  (A={len(pA)}, B={len(pB)}, C={len(pC)})")
    print(f"metadata -> {HERE/'points_metadata.json'}")

if __name__ == "__main__":
    main()
