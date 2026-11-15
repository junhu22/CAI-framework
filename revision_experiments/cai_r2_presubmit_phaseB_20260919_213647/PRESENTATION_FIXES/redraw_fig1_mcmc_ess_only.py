from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "candidate/baseline_assets"
plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})
C_GREEN, C_RED, C_BLUE = "#2ca02c", "#d62728", "#1f77b4"
dims = [2, 6, 10]
fast_mae = [0.0313, 0.0236, 0.0218]
neutra_mae = [0.0294, 0.0230, 0.0215]
hmc_mae = [0.7906, 0.0627, 0.1568]
neutra_ess = [500, 500, 500]
hmc_ess = [191, 17, 68]
neutra_ar = [99.9, 99.9, 99.7]
hmc_ar = [85.7, 98.5, 87.4]


def main():
    x = np.arange(3)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    w = 0.25
    ax.bar(x - w, fast_mae, w, label="Amortized", color=C_GREEN, alpha=.85)
    ax.bar(x, neutra_mae, w, label="NeuTra-HMC", color=C_RED, alpha=.85)
    ax.bar(x + w, hmc_mae, w, label="Standard HMC", color=C_BLUE, alpha=.85)
    ax.set_yscale("log")
    ax.set_ylabel("Posterior mean error (MAE)")
    ax.set_xticks(x, [f"D={d}" for d in dims])
    ax.legend(loc="upper right", framealpha=.9, fontsize=8)
    ax.set_title("(a) Mean error", fontweight="bold")
    ax.grid(True, alpha=.2, axis="y")
    ax.annotate("27$\\times$ worse", xy=(w + .04, .79), xytext=(w + .18, .79),
                fontsize=8, ha="left", va="center", color="black", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=1, shrinkA=2, shrinkB=4, mutation_scale=10))

    # Existing MCMC ESS values only. Fixed fast proposal counts are intentionally absent.
    ax = axes[1]
    w2 = 0.30
    ax.bar(x - w2 / 2, neutra_ess, w2, label="NeuTra-HMC", color=C_RED, alpha=.85)
    ax.bar(x + w2 / 2, hmc_ess, w2, label="Standard HMC", color=C_BLUE, alpha=.85)
    ax.set_yscale("log")
    ax.set_ylabel(r"ESS$_{\min}$")
    ax.set_xticks(x, [f"D={d}" for d in dims])
    ax.legend(loc="lower left", framealpha=.9, fontsize=8)
    ax.set_title("(b) MCMC effective sample size", fontweight="bold")
    ax.grid(True, alpha=.2, axis="y")
    for xi, value in zip(x - w2 / 2, neutra_ess):
        ax.text(xi, value * 1.20, str(value), ha="center", va="bottom", fontsize=7, color=C_RED, fontweight="bold")
    for xi, value in zip(x + w2 / 2, hmc_ess):
        ax.text(xi, value * 1.20, str(value), ha="center", va="bottom", fontsize=7, color="black", fontweight="bold")

    ax = axes[2]
    ax.bar(x - .15, neutra_ar, .3, label="NeuTra-HMC", color=C_RED, alpha=.85)
    ax.bar(x + .15, hmc_ar, .3, label="Standard HMC", color=C_BLUE, alpha=.85)
    ax.axhline(65, color="gray", linestyle="--", linewidth=1, alpha=.5)
    ax.text(2.5, 66, "HMC optimal (0.65)", fontsize=7, color="black")
    ax.set_ylabel("Acceptance rate (%)")
    ax.set_xticks(x, [f"D={d}" for d in dims])
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left", framealpha=.9, fontsize=8)
    ax.set_title("(c) Acceptance rate", fontweight="bold")
    ax.grid(True, alpha=.2, axis="y")

    fig.suptitle("Conditional Banana benchmark ($b = 0.1$, $\\sigma_1 = 3$)", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_banana_benchmark.pdf")
    fig.savefig(OUT / "fig_banana_benchmark.png")
    print(OUT / "fig_banana_benchmark.pdf")


if __name__ == "__main__":
    main()

