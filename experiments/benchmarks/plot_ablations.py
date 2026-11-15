"""
Regenerate the two ablation figures (data-size, flow-depth) from the
saved ablation_study.pt checkpoint, with annotation positions moved out
of the bar-chart clutter.

Run:
  python plot_ablations.py
"""
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PT_PATH = Path(r"C:\nf-pmcmc\experiments\results\ablation_study.pt")
OUT_DIR = Path(r"C:\neutra-hmc-surrogate\surrogate_training\experiments\benchmarks\figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

C_GREEN = "#2ca02c"
C_RED = "#d62728"
C_BLUE = "#1f77b4"

d = torch.load(PT_PATH, weights_only=False)
A = d["ablation_a"]   # list of (N, MAE, val_loss, time_s)
B = d["ablation_b"]   # list of (n_layers, n_params, MAE, val_loss, time_s)


# ── Ablation A: data size ─────────────────────────────────────────
N_a    = [r[0] for r in A]
mae_a  = [float(r[1]) for r in A]
vl_a   = [r[2] for r in A]
time_a = [r[3] for r in A]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# (a) MAE vs N
ax = axes[0]
ax.plot(N_a, mae_a, "o-", color=C_GREEN, linewidth=2, markersize=8,
        label="Held-out MAE vs HMC reference posterior")
# The "sweet spot" is the elbow at N=50k — MAE collapses from 0.042
# down to 0.021 between N=25k and N=50k, with only marginal further
# gain at N=100k.  We mark the elbow (not the strict argmin).
elbow_i = 3                       # corresponds to N=50,000
ax.scatter([N_a[elbow_i]], [mae_a[elbow_i]], s=200,
           facecolors="none", edgecolors=C_RED, linewidth=2, zorder=5)
ax.set_xscale("log")
ax.set_xlabel("Training pairs $N$")
ax.set_ylabel("Posterior-mean MAE")
ax.set_title("(a) Effect of training data size", fontweight="bold")
ax.grid(True, alpha=0.2)
ax.legend(loc="lower left", framealpha=0.9)
# "Sweet spot" label centred *above* the red-circled elbow, with a
# strictly vertical arrow dropping from the text down to the data
# point.  Anchored to N=50k itself so the entire annotation stays
# inside the panel bounds.
ax.set_ylim(top=max(mae_a) * 1.28)
ax.annotate("Sweet spot:\n$N=50{,}000$",
            xy=(N_a[elbow_i], mae_a[elbow_i] + 0.001),
            xytext=(N_a[elbow_i], mae_a[elbow_i] + 0.020),
            fontsize=9, fontweight="bold", color=C_RED,
            ha="center", va="bottom",
            arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.2,
                            shrinkA=2, shrinkB=2))

# (b) Training time vs N
ax = axes[1]
ax.plot(N_a, time_a, "s-", color=C_BLUE, linewidth=2, markersize=8,
        label="Training wall-clock")
ax.set_xscale("log")
ax.set_xlabel("Training pairs $N$")
ax.set_ylabel("Training time (s)")
ax.set_title("(b) Training cost", fontweight="bold")
ax.grid(True, alpha=0.2)
ax.legend(loc="upper left", framealpha=0.9)

fig.suptitle("Ablation A: training data size",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig10_ablation_data_size.pdf")
fig.savefig(OUT_DIR / "fig10_ablation_data_size.png")
print(f"Saved: {OUT_DIR / 'fig10_ablation_data_size.pdf'}")


# ── Ablation B: flow depth ───────────────────────────────────────
L_b   = [r[0] for r in B]
np_b  = [r[1] for r in B]
mae_b = [float(r[2]) for r in B]
vl_b  = [r[3] for r in B]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# (a) MAE vs depth
ax = axes[0]
bars = ax.bar(range(len(L_b)), mae_b, color=C_GREEN, alpha=0.85, width=0.55)
best_i = int(np.argmin(mae_b))
bars[best_i].set_color(C_RED)
bars[best_i].set_alpha(0.85)
ax.set_xticks(range(len(L_b)))
ax.set_xticklabels([f"L={l}" for l in L_b])
ax.set_xlabel("Number of coupling layers")
ax.set_ylabel("Posterior-mean MAE")
ax.set_title("(a) Effect of flow depth", fontweight="bold")
ax.grid(True, alpha=0.2, axis="y")
# Per-bar value labels
for b, v in zip(bars, mae_b):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0008,
            f"{v:.4f}", ha="center", va="bottom",
            fontsize=9, fontweight="bold",
            color=C_RED if b.get_facecolor()[0] > 0.5 else C_GREEN)
# "Best" label sits directly above the per-bar value on the L=8 bar.
# No arrow — the red bar + its numeric label already direct the eye;
# the word "Best" just makes the verdict explicit.
ymax = max(mae_b) * 1.35
ax.set_ylim([0, ymax])
ax.text(best_i, mae_b[best_i] + 0.005,
        "Best",
        ha="center", va="bottom",
        fontsize=9, fontweight="bold", color=C_RED)

# (b) Parameter count vs depth
ax = axes[1]
ax.plot(L_b, np_b, "o-", color=C_BLUE, linewidth=2, markersize=8,
        label="Trainable parameters")
ax.set_xticks(L_b)
ax.set_xlabel("Number of coupling layers")
ax.set_ylabel("Parameter count")
ax.set_title("(b) Model size", fontweight="bold")
ax.grid(True, alpha=0.2)
ax.legend(loc="lower right", framealpha=0.9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))

fig.suptitle("Ablation B: flow depth",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig11_ablation_flow_depth.pdf")
fig.savefig(OUT_DIR / "fig11_ablation_flow_depth.png")
print(f"Saved: {OUT_DIR / 'fig11_ablation_flow_depth.pdf'}")
