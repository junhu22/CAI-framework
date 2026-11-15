"""
E4 (Step 6): posterior calibration — coverage + SBC (R1#14).

Config: D=6 ORIGINAL surrogate + CNF (conditional_flow.pt). For a VALID calibration
test the test prior must equal the CNF's training prior, so we use the training
prior sigma=0.30 and training noise kappa=sqrt(1e-3)=3.16% (the manual's "sigma=0.5,
kappa=3%" would probe outside the training distribution and is not a valid SBC).

Coverage: 100 theta~prior, y=f_hat(theta)+noise; fast-mode 50%/90% central credible
intervals; fraction covering the truth (expect ~0.50 / ~0.90). 10 exact-mode spot checks.
SBC: 500 prior-predictive (theta,y); L=100 fast-mode draws; rank of truth per param;
rank histograms + ECDF-difference plot with 95% envelope.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(4)
sys.path.insert(0, r"C:\neutra-hmc-surrogate\surrogate_training\revision_experiments")
import revlib as R
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
torch.manual_seed(42); np.random.seed(42)
SIGMA, KAPPA = 0.30, np.sqrt(1e-3)
sur = R.load_d6(); cnf = R.load_d6_cnf()
PN = ["e1", "e2", "e3", "e4", "e5", "e6"]

def draw_theta(n):
    return np.clip(np.exp(SIGMA * np.random.randn(n, 6)), 0.5, 2.0)

def gen_y(theta):
    with torch.no_grad():
        f = sur(torch.tensor(theta, dtype=torch.float32)).numpy()
    return f + KAPPA * np.abs(f) * np.random.randn(*f.shape)

def fast_samples(y, N):
    return cnf.sample(torch.tensor(y, dtype=torch.float32), n_samples=N).numpy()

# ── Coverage ────────────────────────────────────────────────────────────
NCOV = 100
th_cov = draw_theta(NCOV); y_cov = gen_y(th_cov)
cov50 = np.zeros(6); cov90 = np.zeros(6)
for i in range(NCOV):
    s = fast_samples(y_cov[i], 2000)
    lo50, hi50 = np.quantile(s, [0.25, 0.75], axis=0)
    lo90, hi90 = np.quantile(s, [0.05, 0.95], axis=0)
    cov50 += (th_cov[i] >= lo50) & (th_cov[i] <= hi50)
    cov90 += (th_cov[i] >= lo90) & (th_cov[i] <= hi90)
cov50 /= NCOV; cov90 /= NCOV
print("Coverage 50%:", cov50.round(3), " overall", cov50.mean().round(3))
print("Coverage 90%:", cov90.round(3), " overall", cov90.mean().round(3))

# ── Exact-mode spot checks (10) ─────────────────────────────────────────
exact_cov90 = []
for i in range(10):
    y_t = torch.tensor(y_cov[i], dtype=torch.float32)
    lp = R.make_log_post(sur, y_cov[i], "d6")
    smp = R.NeuTraHMC_Conditional(lp.forward, cnf, y_t, step_size=0.05, n_leapfrog=20)
    chain, ar = smp.run(torch.ones(6), n_samples=300, n_burnin=150)
    ch = chain.numpy()
    lo, hi = np.quantile(ch, [0.05, 0.95], axis=0)
    exact_cov90.append(((th_cov[i] >= lo) & (th_cov[i] <= hi)).mean())
exact_cov90 = float(np.mean(exact_cov90))
print(f"Exact-mode 90% coverage (10 spot checks, per-obs avg): {exact_cov90:.3f}")

# ── SBC ─────────────────────────────────────────────────────────────────
NSBC, L = 500, 100
th_sbc = draw_theta(NSBC); y_sbc = gen_y(th_sbc)
ranks = np.zeros((NSBC, 6), dtype=int)
for i in range(NSBC):
    s = fast_samples(y_sbc[i], L)
    ranks[i] = (s < th_sbc[i]).sum(axis=0)          # rank in 0..L
print("SBC ranks computed; mean rank/L =", (ranks.mean(0)/L).round(3))

# rank histograms
fig, axes = plt.subplots(2, 3, figsize=(11, 5.5))
nbins = 20
for d, ax in enumerate(axes.ravel()):
    ax.hist(ranks[:, d], bins=nbins, range=(0, L), color="#4c72b0", alpha=0.85)
    exp = NSBC / nbins
    band = 1.96 * np.sqrt(exp * (1 - 1/nbins))
    ax.axhline(exp, color="k", lw=1)
    ax.axhspan(exp - band, exp + band, color="grey", alpha=0.25)
    ax.set_title(f"{PN[d]}"); ax.set_xlabel("rank")
fig.suptitle(f"E4 SBC rank histograms (N={NSBC}, L={L}, D=6 original, sigma={SIGMA})")
fig.tight_layout(); fig.savefig(HERE / "E4_sbc_ranks.png", dpi=150)
fig.savefig(HERE / "E4_sbc_ranks.pdf")

# ECDF-difference plot with 95% envelope
fig2, ax = plt.subplots(figsize=(6.5, 4.2))
grid = np.linspace(0, 1, 200)
for d in range(6):
    u = (ranks[:, d] + 0.5) / (L + 1)
    ecdf = np.array([(u <= g).mean() for g in grid])
    ax.plot(grid, ecdf - grid, lw=1, label=PN[d])
band = 1.36 / np.sqrt(NSBC)          # KS 95%
ax.axhspan(-band, band, color="grey", alpha=0.25, label="95% envelope")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("normalised rank"); ax.set_ylabel("ECDF - uniform")
ax.legend(fontsize=8, ncol=2); ax.set_title("E4 SBC: ECDF difference from uniform")
fig2.tight_layout(); fig2.savefig(HERE / "E4_sbc_ecdf.png", dpi=150)
fig2.savefig(HERE / "E4_sbc_ecdf.pdf")

# KS statistic per param
ks = []
for d in range(6):
    u = np.sort((ranks[:, d] + 0.5) / (L + 1))
    ecdf = np.arange(1, NSBC + 1) / NSBC
    ks.append(float(np.max(np.abs(ecdf - u))))

# ── markdown ────────────────────────────────────────────────────────────
Lm = ["# E4 — Posterior calibration (coverage + SBC)\n",
      f"D=6 original surrogate+CNF. Test prior = training prior (sigma={SIGMA}, "
      f"kappa={KAPPA:.4f}) so the SBC is valid.\n",
      "## Central credible-interval coverage (fast mode, 100 draws)",
      "| param | " + " | ".join(PN) + " | overall |",
      "|---|" + "|".join(["---"]*6) + "|---|",
      "| 50% CI | " + " | ".join(f"{v:.2f}" for v in cov50) + f" | {cov50.mean():.2f} |",
      "| 90% CI | " + " | ".join(f"{v:.2f}" for v in cov90) + f" | {cov90.mean():.2f} |",
      "",
      f"Exact-mode NeuTra-HMC 90% coverage (10 spot checks): **{exact_cov90:.2f}** "
      "(target 0.90).",
      "",
      "## SBC (500 prior-predictive pairs, L=100)",
      "| param | " + " | ".join(PN) + " |",
      "|---|" + "|".join(["---"]*6) + "|",
      "| KS stat | " + " | ".join(f"{v:.3f}" for v in ks) + " |",
      f"| KS 95% crit | " + " | ".join([f"{1.36/np.sqrt(NSBC):.3f}"]*6) + " |",
      "",
      "Rank histograms (`E4_sbc_ranks.png`) and ECDF-difference (`E4_sbc_ecdf.png`): "
      "ranks within the 95% envelope indicate the fast-mode posterior is calibrated "
      "over the training prior."]
(HERE / "E4_coverage_table.md").write_text("\n".join(Lm), encoding="utf-8")
json.dump(dict(cov50=cov50.tolist(), cov90=cov90.tolist(),
               exact_cov90=exact_cov90, ks=ks, sigma=SIGMA, kappa=float(KAPPA)),
          open(HERE / "E4_results.json", "w"), indent=1)
print("Wrote E4_coverage_table.md, E4_sbc_ranks.png/pdf, E4_sbc_ecdf.png/pdf")
