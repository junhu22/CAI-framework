"""
Generate publication-quality figures for CAI paper analytical benchmarks.
Uses REAL benchmark data from the completed runs.

Run: python generate_figures_final.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

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
})

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ============================================================
# REAL DATA from benchmark runs
# ============================================================

# --- Banana ---
banana = {
    "dims": [2, 6, 10, 20],
    "fast_mae": [0.0313, 0.0236, 0.0218, 0.0222],
    "fast_time": [7.6, 8.5, 7.4, 3.1],
    "neutra_mae": [0.0294, 0.0230, 0.0215, None],
    "neutra_ess": [500, 500, 500, None],
    "neutra_ar": [0.999, 0.999, 0.997, None],
    "hmc_mae": [0.7906, 0.0627, 0.1568, None],
    "hmc_ess": [191, 17, 68, None],
    "hmc_ar": [0.857, 0.985, 0.874, None],
}

# --- Funnel ---
funnel = {
    "dims": [2, 6, 10],
    "fast_mae": [0.041, 0.042, 0.041],
    "fast_mae_std": [0.012, 0.015, 0.010],
    "fast_time": [9.9, 8.7, 7.2],
    # The unconditional ML-KDE reference point (and the 580x improvement it
    # implied) was withdrawn from Figure 2: it was a single number from a
    # different setup, not a like-for-like baseline. The qualitative statement
    # about unconditional flow-/KDE-based samplers on heteroscedastic funnels
    # stays in Section 4.4, where it rests on published results.
}

# --- Scaling ---
scaling = {
    "dims": [2, 4, 6, 8, 12, 20],
    "mae_mean": [0.0065, 0.0078, 0.0072, 0.0089, 0.0112, 0.0132],
    "mae_std": [0.003, 0.004, 0.003, 0.003, 0.004, 0.006],
    "cov_err": [0.075, 0.078, 0.082, 0.086, 0.119, 0.163],
    "time_ms": [9.1, 7.8, 8.1, 8.6, 13.9, 28.2],
}

C_GREEN = "#2ca02c"
C_RED = "#d62728"
C_BLUE = "#1f77b4"
C_ORANGE = "#ff7f0e"


def fig1_banana_comparison():
    """Figure: Banana benchmark — 3-panel comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    dims_hmc = [2, 6, 10]
    x = np.arange(len(dims_hmc))
    w = 0.25

    # (a) MAE
    ax = axes[0]
    ax.bar(x - w, [banana["fast_mae"][i] for i in range(3)], w,
           label="Amortized", color=C_GREEN, alpha=0.85)
    ax.bar(x, [banana["neutra_mae"][i] for i in range(3)], w,
           label="NeuTra-HMC", color=C_RED, alpha=0.85)
    ax.bar(x + w, [banana["hmc_mae"][i] for i in range(3)], w,
           label="Standard HMC", color=C_BLUE, alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylabel("Posterior mean error (MAE)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}" for d in dims_hmc])
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    ax.set_title("(a) Mean error", fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    # Annotate HMC failure at D=2: text just right of the bar, very short
    # arrow with visible head, in black for legibility against the bars.
    ax.annotate("27$\\times$ worse",
                xy=(0 + w + 0.04, 0.79),
                xytext=(0 + w + 0.18, 0.79),
                fontsize=8, ha="left", va="center",
                color="black", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black",
                                lw=1.0, shrinkA=2, shrinkB=4,
                                mutation_scale=10))

    # (b) ESS_min
    ax = axes[1]
    amort_vals  = [2000, 2000, 2000]
    neutra_vals = [500, 500, 500]
    hmc_vals    = [banana["hmc_ess"][i] for i in range(3)]
    ax.bar(x - w, amort_vals, w,
           label="Amortized (i.i.d.)", color=C_GREEN, alpha=0.85)
    ax.bar(x, neutra_vals, w,
           label="NeuTra-HMC", color=C_RED, alpha=0.85)
    ax.bar(x + w, hmc_vals, w,
           label="Standard HMC", color=C_BLUE, alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylabel(r"ESS$_{\min}$")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}" for d in dims_hmc])
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8)
    ax.set_title("(b) Effective sample size", fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    # Per-bar value labels — every bar carries its own ESS number.
    # Amortized labels sit *inside* the green bar near its top (white
    # text) so they remain inside the panel even when matplotlib's
    # auto y-limit hugs the top of the bar.
    for xi, v in zip(x - w, amort_vals):
        ax.text(xi, v * 0.78, f"{v}", ha="center", va="top",
                fontsize=7, color="white", fontweight="bold")
    for xi, v in zip(x, neutra_vals):
        ax.text(xi, v * 1.20, f"{v}", ha="center", va="bottom",
                fontsize=7, color=C_RED, fontweight="bold")
    for xi, v in zip(x + w, hmc_vals):
        ax.text(xi, v * 1.20, f"{v}", ha="center", va="bottom",
                fontsize=7, color="black", fontweight="bold")

    # (c) Acceptance rate
    ax = axes[2]
    ax.bar(x - 0.15, [banana["neutra_ar"][i] * 100 for i in range(3)], 0.3,
           label="NeuTra-HMC", color=C_RED, alpha=0.85)
    ax.bar(x + 0.15, [banana["hmc_ar"][i] * 100 for i in range(3)], 0.3,
           label="Standard HMC", color=C_BLUE, alpha=0.85)
    ax.axhline(y=65, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(2.5, 66, "HMC optimal (0.65)", fontsize=7, color="black")
    ax.set_ylabel("Acceptance rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D={d}" for d in dims_hmc])
    ax.set_ylim([0, 105])
    ax.legend(loc="lower left", framealpha=0.9, fontsize=8)
    ax.set_title("(c) Acceptance rate", fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Conditional Banana benchmark ($b = 0.1$, $\\sigma_1 = 3$)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_banana_benchmark.pdf")
    fig.savefig(FIG_DIR / "fig_banana_benchmark.png")
    print(f"Saved: fig_banana_benchmark.pdf/png")


def fig2_funnel_comparison():
    """Figure: Funnel benchmark — fast-mode MAE stability + inference time."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    dims = funnel["dims"]

    # (a) MAE vs D
    ax = axes[0]
    ax.errorbar(dims, funnel["fast_mae"], yerr=funnel["fast_mae_std"],
                fmt="o-", color=C_GREEN, linewidth=2, markersize=8,
                capsize=4, label="CAI fast mode")
    ax.set_xlabel("Parameter dimension D")
    ax.set_ylabel("Posterior mean error (MAE)")
    ax.set_yscale("log")
    # upper right: the curve sits low in the panel, and a default-placed legend
    # lands on top of it
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("(a) Fast-mode accuracy", fontweight="bold")
    ax.grid(True, alpha=0.2)

    # (b) Inference time
    ax = axes[1]
    bars = ax.bar(range(len(dims)), funnel["fast_time"],
                  color=C_GREEN, alpha=0.8)
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=1)
    # Reference label above & to the right of the dashed line, not on it.
    # It is a reference line, not a threshold: no source defines 100 ms as the
    # real-time requirement, and the paper no longer makes a generic real-time
    # claim.
    ax.text(len(dims) - 1, 100, "  100 ms reference",
            fontsize=8, color="gray",
            ha="right", va="bottom")
    # Value labels on each bar
    for b, t in zip(bars, funnel["fast_time"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                f"{t:.1f} ms", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=C_GREEN)
    ax.set_xlabel("Parameter dimension D")
    ax.set_ylabel("Inference time (ms)")
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([f"D={d}" for d in dims])
    ax.set_ylim([0, 130])
    ax.set_title("(b) Fast-mode inference time", fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Conditional Funnel benchmark ($\\sigma_v = 3$, $\\kappa = 0.05$)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_funnel_benchmark.pdf")
    fig.savefig(FIG_DIR / "fig_funnel_benchmark.png")
    print(f"Saved: fig_funnel_benchmark.pdf/png")


def fig3_dimension_scaling():
    """Figure: Dimension scaling — MAE + Cov error + Time."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    dims = scaling["dims"]

    # (a) MAE vs D
    ax = axes[0]
    ax.errorbar(dims, scaling["mae_mean"], yerr=scaling["mae_std"],
                fmt="o-", color=C_GREEN, linewidth=2, markersize=7,
                capsize=4, label="CAI fast mode")
    # Least-squares power-law fit to the plotted points, drawn so the D^0.3
    # annotation refers to something visible rather than to a bare number.
    log_d = np.log(dims)
    log_mae = np.log(scaling["mae_mean"])
    slope, intercept = np.polyfit(log_d, log_mae, 1)
    d_fit = np.linspace(min(dims), max(dims), 100)
    # no legend entry: the panel already carries the "propto D^0.3" annotation, and
    # a second entry makes the tight bounding box taller, which shifts the float
    ax.plot(d_fit, np.exp(intercept) * d_fit ** slope, "--",
            color="gray", linewidth=1.2, alpha=0.8)
    ax.text(12, 0.005, f"$\\propto D^{{{slope:.1f}}}$", fontsize=10,
            color=C_GREEN, fontweight="bold")
    ax.set_xlabel("Parameter dimension D")
    ax.set_ylabel("MAE vs analytical posterior")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(dims)
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_title("(a) Posterior mean error", fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.legend(framealpha=0.9)

    # (b) Covariance error vs D
    ax = axes[1]
    ax.plot(dims, scaling["cov_err"], "s-", color=C_RED, linewidth=2,
            markersize=7, label="Relative Frobenius error")
    ax.axhline(y=0.2, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    # "20% threshold" below-left of dashed line; legend lower-right (separated)
    ax.text(3, 0.17, "20% threshold", fontsize=8, color="gray",
            ha="left", va="top")
    ax.fill_between(dims, 0, scaling["cov_err"], alpha=0.1, color=C_RED)
    ax.set_xlabel("Parameter dimension D")
    ax.set_ylabel("Covariance error (relative)")
    ax.set_title("(b) Covariance recovery", fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower right", framealpha=0.9)

    # (c) Inference time vs D
    ax = axes[2]
    ax.plot(dims, scaling["time_ms"], "o-", color=C_GREEN, linewidth=2,
            markersize=7, label="Fast mode")
    ax.fill_between(dims, 0, scaling["time_ms"], alpha=0.1, color=C_GREEN)
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    # Threshold label below the dashed line so it doesn't fly off-axis;
    # also raise the y-limit so the dashed line + label have headroom.
    ax.set_ylim(top=120)
    ax.text(3, 92, "100 ms reference", fontsize=8, color="gray",
            ha="left", va="top")
    ax.set_xlabel("Parameter dimension D")
    ax.set_ylabel("Inference time (ms)")
    ax.set_title("(c) Wall-clock time", fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", framealpha=0.9)

    fig.suptitle("Dimension scaling on linear-Gaussian inverse problem ($\\rho = 0.9$)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_dimension_scaling.pdf")
    fig.savefig(FIG_DIR / "fig_dimension_scaling.png")
    print(f"Saved: fig_dimension_scaling.pdf/png")


def fig4_cross_target_summary():
    """Figure: Cross-target comparison: amortized + (where available) HMC variants."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))

    targets = ["Banana\n(D=10)", "Funnel\n(D=10)", "Linear-Gaussian\n(D=12)"]
    # Numbers are mean MAE-of-posterior-mean over 3 held-out observations.
    # Banana from earlier benchmark runs; funnel/lin-Gauss backfilled by
    # experiments/backfill_funnel_gaussian_hmc.py (3 obs each, N=500+100).
    fast_mae   = [0.022, 0.038, 0.004]
    neutra_mae = [0.022, 0.037, 0.001]
    hmc_mae    = [0.157, 0.050, 0.015]

    x = np.arange(len(targets))
    w = 0.26

    # Bars: place all three methods at every target (None → drawn at 0 with N/A label)
    fast_vals   = [v if v is not None else 0 for v in fast_mae]
    neutra_vals = [v if v is not None else 0 for v in neutra_mae]
    hmc_vals    = [v if v is not None else 0 for v in hmc_mae]

    ax.bar(x - w, fast_vals,   w, label="Amortized (fast)",  color=C_GREEN, alpha=0.85)
    ax.bar(x,     neutra_vals, w, label="NeuTra-HMC (exact)", color=C_RED,   alpha=0.85)
    ax.bar(x + w, hmc_vals,    w, label="Standard HMC",       color=C_BLUE,  alpha=0.85)

    def annotate(xs, vals, raw, color):
        for xi, vi, ri in zip(xs, vals, raw):
            if ri is None:
                ax.text(xi, 0.004, "n/a", ha="center", va="bottom",
                        fontsize=8, color="gray", style="italic")
            else:
                ax.text(xi, ri + 0.004, f"{ri:.3f}", ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color=color)

    annotate(x - w, fast_vals,   fast_mae,   C_GREEN)
    annotate(x,     neutra_vals, neutra_mae, C_RED)
    annotate(x + w, hmc_vals,    hmc_mae,    C_BLUE)

    ax.set_ylabel("Posterior mean error (MAE)")
    ax.set_xticks(x)
    ax.set_xticklabels(targets)
    ax.set_ylim([0, max(filter(None, fast_mae + neutra_mae + hmc_mae)) * 1.25])
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("Cross-target benchmark: CAI fast mode handles diverse geometries",
                 fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig_cross_target_summary.pdf")
    fig.savefig(FIG_DIR / "fig_cross_target_summary.png")
    print(f"Saved: fig_cross_target_summary.pdf/png")


if __name__ == "__main__":
    print("Generating CAI paper figures with REAL benchmark data...\n")
    fig1_banana_comparison()
    fig2_funnel_comparison()
    fig3_dimension_scaling()
    fig4_cross_target_summary()
    print(f"\nAll figures saved to {FIG_DIR}/")
