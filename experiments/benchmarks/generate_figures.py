"""
Generate publication-quality figures for CAI paper analytical benchmarks.
Style matches MSSP paper (matplotlib with LaTeX rendering).

Run after all 3 benchmarks complete:
    python generate_figures.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "text.usetex": False,  # Set True if LaTeX available
})

COLORS = {
    "amortized": "#2ca02c",
    "neutra": "#d62728",
    "hmc": "#1f77b4",
    "kde": "#ff7f0e",
}

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)


def load_results(path):
    with open(path) as f:
        return json.load(f)


def fig_cross_target_benchmark():
    """
    Cross-target benchmark figure (like MSSP Figure 5).
    Three panels: ESS, Mean Error, Acceptance Rate.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    targets = ["Banana\nD=10", "Funnel\nD=10", "Scaling\nD=10"]
    # Placeholder data — replace with actual benchmark results
    methods = ["Std HMC", "NeuTra-HMC", "Amortized"]
    ess_data = {
        "Std HMC": [210, 50, 300],
        "NeuTra-HMC": [1120, 800, 1500],
        "Amortized": [2000, 2000, 2000],
    }
    error_data = {
        "Std HMC": [0.089, 0.3, 0.05],
        "NeuTra-HMC": [0.038, 0.08, 0.02],
        "Amortized": [0.043, 0.15, 0.03],
    }
    acc_data = {
        "Std HMC": [0.45, 0.10, 0.60],
        "NeuTra-HMC": [0.88, 0.85, 0.92],
        "Amortized": [None, None, None],
    }

    x = np.arange(len(targets))
    width = 0.25

    # Panel (a): ESS
    ax = axes[0]
    for i, method in enumerate(methods):
        vals = ess_data[method]
        ax.bar(x + i * width, vals, width, label=method,
               color=list(COLORS.values())[i + 1], alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylabel("ESS at T=1")
    ax.set_xticks(x + width)
    ax.set_xticklabels(targets)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("(a) Effective sample size")

    # Panel (b): Mean error
    ax = axes[1]
    for i, method in enumerate(methods):
        vals = error_data[method]
        ax.bar(x + i * width, vals, width,
               color=list(COLORS.values())[i + 1], alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylabel(r"Mean error $\|\hat\mu - \mu_{\rm true}\|_2 / D$")
    ax.set_xticks(x + width)
    ax.set_xticklabels(targets)
    ax.set_title("(b) Mean error (log scale)")

    # Panel (c): Acceptance rate
    ax = axes[2]
    for i, method in enumerate(["Std HMC", "NeuTra-HMC"]):
        vals = acc_data[method]
        ax.bar(x[:len(vals)] + i * width, vals, width, label=method,
               color=list(COLORS.values())[i + 1], alpha=0.85)
    ax.axhline(y=0.234, color="gray", linestyle="--", linewidth=1,
               label="RWMH optimum (0.234)")
    ax.set_ylabel("Acceptance rate")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(targets)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("(c) Acceptance at T=1")

    fig.suptitle(
        "Cross-target benchmark on three conditional analytical targets",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_cross_target_benchmark.pdf")
    print(f"Saved: {FIG_DIR / 'fig_cross_target_benchmark.pdf'}")


def fig_dimension_scaling():
    """
    Dimension scaling figure (like MSSP Figure 11b,c).
    Two panels: MAE vs D and Time vs D.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Placeholder — replace with actual results
    dims = [2, 4, 6, 8, 12, 20]
    mae_fast = [0.005, 0.012, 0.020, 0.035, 0.060, 0.110]
    mae_exact = [0.004, 0.010, 0.018, 0.030, 0.055, None]
    mae_hmc = [0.006, 0.015, 0.030, 0.060, 0.146, None]
    time_fast = [1.5, 2.5, 3.5, 5.0, 7.0, 10.0]

    # Panel (a): MAE vs D
    ax = axes[0]
    ax.plot(dims, mae_fast, "o-", color=COLORS["amortized"],
            label="Amortized (fast)", linewidth=2, markersize=6)
    ax.plot(dims[:5], mae_exact[:5], "s-", color=COLORS["neutra"],
            label="NeuTra-HMC (exact)", linewidth=2, markersize=6)
    ax.plot(dims[:5], mae_hmc[:5], "^-", color=COLORS["hmc"],
            label="Standard HMC", linewidth=2, markersize=6)
    ax.set_xlabel("Parameter dimension D")
    ax.set_ylabel("Posterior mean error")
    ax.set_yscale("log")
    ax.legend(framealpha=0.9)
    ax.set_title("(a) Mean error scaling")
    ax.grid(True, alpha=0.3)

    # Annotate crossover
    ax.annotate("CAI advantage\ngrows with D",
                xy=(12, 0.06), fontsize=8, color=COLORS["amortized"],
                ha="center",
                arrowprops=dict(arrowstyle="->", color=COLORS["amortized"]))

    # Panel (b): Time vs D
    ax = axes[1]
    ax.plot(dims, time_fast, "o-", color=COLORS["amortized"],
            label="Fast mode", linewidth=2, markersize=6)
    ax.axhline(y=10, color="gray", linestyle="--", alpha=0.5)
    ax.text(15, 11, "10 ms threshold", fontsize=8, color="gray")
    ax.fill_between(dims, 0, time_fast, alpha=0.15, color=COLORS["amortized"])
    ax.set_xlabel("Parameter dimension D")
    ax.set_ylabel("Inference time (ms)")
    ax.set_title("(b) Fast-mode inference time")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_dimension_scaling.pdf")
    print(f"Saved: {FIG_DIR / 'fig_dimension_scaling.pdf'}")


def fig_fast_vs_exact_diagnostic():
    """
    New figure: importance weight ESS diagnostic for fast-vs-exact mode switching.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Placeholder data — replace with actual
    obs_labels = [f"Obs {i+1}" for i in range(10)]
    fast_mae = [0.013, 0.014, 0.015, 0.009, 0.019, 0.015, 0.014, 0.039, 0.014, 0.009]
    is_ess_ratio = [0.92, 0.88, 0.90, 0.95, 0.78, 0.87, 0.93, 0.62, 0.89, 0.96]

    scatter = ax.scatter(fast_mae, is_ess_ratio, c=is_ess_ratio,
                         cmap="RdYlGn", s=80, edgecolors="black", linewidth=0.5,
                         vmin=0.5, vmax=1.0, zorder=3)
    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1.5,
               label="Exact mode trigger (ESS_IS/N = 0.5)")
    ax.axhline(y=0.8, color="orange", linestyle=":", linewidth=1,
               label="Warning threshold (0.8)")

    for i, label in enumerate(obs_labels):
        ax.annotate(label, (fast_mae[i], is_ess_ratio[i]),
                    fontsize=7, ha="center", va="bottom", xytext=(0, 5),
                    textcoords="offset points")

    ax.set_xlabel("Fast-mode MAE (amortized vs. HMC)")
    ax.set_ylabel("Importance weight ESS / N")
    ax.set_title("Fast-vs-exact mode diagnostic (D=6)")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label="ESS_IS / N")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_fast_vs_exact_diagnostic.pdf")
    print(f"Saved: {FIG_DIR / 'fig_fast_vs_exact_diagnostic.pdf'}")


if __name__ == "__main__":
    print("Generating CAI paper benchmark figures...")
    fig_cross_target_benchmark()
    fig_dimension_scaling()
    fig_fast_vs_exact_diagnostic()
    print("\nAll figures generated in figures/")
    print("Replace placeholder data with actual benchmark results before submission!")
