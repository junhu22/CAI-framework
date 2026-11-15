"""
Task 6 (R2#6): D=12 identifiability analysis.
Regenerate D=12 exact-mode posterior samples (NeuTra-HMC, log-theta space) for the
Near-nominal in-distribution observation, then report per-parameter posterior std and the
12x12 posterior correlation matrix (heatmap). Relate the weakly-constrained groups and
correlation blocks to the surrogate sensitivity (Jacobian) diagonal-dominant structure.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(6)
sys.path.insert(0, r"C:\neutra-hmc-surrogate\surrogate_training\revision_experiments")
import revlib as R
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).parent
torch.manual_seed(42); np.random.seed(42)
nn12, flow12, blob = R.load_d12()
KAPPA, SIGMA = 0.03, 0.5

# Near-nominal observation (theta=1)
th_true = torch.ones(12, dtype=torch.float64)
with torch.no_grad():
    yc = nn12(th_true.unsqueeze(0)).squeeze(0)
g = torch.Generator().manual_seed(500)
y = yc + torch.randn(yc.shape, generator=g, dtype=torch.float64) * KAPPA * yc.abs()

# ── D=12 NeuTra-HMC exact mode (log-theta, z-space leapfrog, mirrors d12_validation.py) ──
def U_z(z):
    z = z.detach().requires_grad_(True)
    yb = y.unsqueeze(0)
    lt = flow12.inverse(z.unsqueeze(0), yb).squeeze(0)
    lp = R.d12_log_posterior(lt, y, nn12, KAPPA, SIGMA)
    _, log_det = flow12.forward(lt.unsqueeze(0), yb)
    U = -(lp - log_det.squeeze(0))
    U.backward()
    return U.detach(), z.grad.detach()

N, WARM, L, EPS = 2000, 500, 20, 0.15
with torch.no_grad():
    z_cur = torch.randn(12, dtype=torch.float64)
samples, acc = [], 0
for i in range(N + WARM):
    p = torch.randn(12, dtype=torch.float64)
    z = z_cur.clone(); U0, gU = U_z(z); H0 = U0 + 0.5 * (p ** 2).sum()
    p = p - 0.5 * EPS * gU
    for _ in range(L - 1):
        z = z + EPS * p; _, gU = U_z(z); p = p - EPS * gU
    z = z + EPS * p; U1, gU = U_z(z); p = p - 0.5 * EPS * gU
    H1 = U1 + 0.5 * (p ** 2).sum()
    if torch.log(torch.rand(1, dtype=torch.float64)) < (H0 - H1):
        z_cur = z
        if i >= WARM: acc += 1
    if i >= WARM:
        with torch.no_grad():
            lt = flow12.inverse(z_cur.unsqueeze(0), y.unsqueeze(0))
            samples.append(torch.exp(lt.squeeze(0)))
theta = torch.stack(samples).numpy()   # (N, 12)
ar = acc / N
print(f"D=12 exact-mode: N={len(theta)} AR={ar:.2f} mean[:6]={theta.mean(0)[:6].round(3)}", flush=True)

post_std = theta.std(0)
corr = np.corrcoef(theta.T)   # 12x12

# surrogate sensitivity (Jacobian |df/dtheta| at theta=1), column-normalised, to show diag-dominance
th1 = torch.ones(1, 12, dtype=torch.float64, requires_grad=True)
f1 = nn12(th1)
J = np.zeros((20, 12))
for m in range(20):
    gr = torch.autograd.grad(f1[0, m], th1, retain_graph=True)[0][0].numpy()
    J[m] = gr
# per-parameter total sensitivity (column L2 norm) -> weak params have low sensitivity
col_sens = np.linalg.norm(J, axis=0)

# ── figure: correlation heatmap + sensitivity ───────────────────────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
im = a1.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
a1.set_xticks(range(12)); a1.set_yticks(range(12))
a1.set_xticklabels([f"θ{i+1}" for i in range(12)], fontsize=7); a1.set_yticklabels([f"θ{i+1}" for i in range(12)], fontsize=7)
a1.set_title("D=12 posterior correlation matrix"); fig.colorbar(im, ax=a1, fraction=0.046)
x = np.arange(12)
a2.bar(x - 0.2, post_std, 0.4, label="posterior std (identifiability)", color="#c44")
a2b = a2.twinx(); a2b.bar(x + 0.2, col_sens / col_sens.max(), 0.4, label="norm. sensitivity |∂f/∂θ|", color="#48a")
a2.set_xticks(x); a2.set_xticklabels([f"θ{i+1}" for i in range(12)], fontsize=7)
a2.set_ylabel("posterior std"); a2b.set_ylabel("normalised column sensitivity")
a2.set_title("Per-parameter posterior std vs surrogate sensitivity")
a2.legend(loc="upper left", fontsize=7); a2b.legend(loc="upper right", fontsize=7)
fig.tight_layout(); fig.savefig(HERE / "E7_d12_identifiability.png", dpi=150)

# ── annotate: weakest params, strongest off-diagonal correlations ───────
weak = list(np.argsort(-post_std)[:4] + 1)        # 4 highest-std (least constrained)
strong = list(np.argsort(col_sens)[:4] + 1)       # 4 lowest sensitivity
oc = corr.copy(); np.fill_diagonal(oc, 0)
pairs = []
idx = np.dstack(np.unravel_index(np.argsort(-np.abs(oc).ravel()), oc.shape))[0]
seen = set()
for i, j in idx:
    if i < j and (i, j) not in seen:
        pairs.append((int(i+1), int(j+1), float(corr[i, j]))); seen.add((i, j))
    if len(pairs) >= 5: break

out = {"ar": ar, "n_samples": len(theta), "post_std": post_std.tolist(),
       "least_constrained_params_by_std": [int(w) for w in weak],
       "lowest_sensitivity_params": [int(s) for s in strong],
       "top_correlated_pairs": pairs,
       "corr_matrix": corr.tolist(), "col_sensitivity": col_sens.tolist()}
json.dump(out, open(HERE / "d12_identifiability.json", "w"), indent=2)
print("weakest (highest std):", weak, " lowest-sensitivity:", strong, flush=True)
print("top correlated pairs:", pairs, flush=True)
