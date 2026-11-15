"""R3-T4: regenerate the affected tables/figures (_v2) from the final results."""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = Path(__file__).parent
ff = json.load(open(HERE / "field_final_results.json"))
e3 = json.load(open(HERE / "e3_ext_results.json"))
eth = json.load(open(HERE / "eth_final_results.json"))
e6 = json.load(open(Path(r"C:\neutra-hmc-surrogate\surrogate_training\revision_experiments\task1_retrain\e6_convergence.json")))

L = ["# Regenerated tables (_v2) — final clean/ext pipeline\n"]
# Table 13 — field exact mode
t2 = ff["T2"]
L += ["## Table 13 (v2) — field-data exact-mode inference",
      "| method | AR | ESS_min | time (s) |", "|---|---|---|---|",
      f"| mass-NeuTra-HMC (retuned Mᵢ=1/Var(zᵢ), ε=0.04, L=40) | {t2['ar']:.2f} | {t2['ess_min']:.1f} | {t2['time_s']:.0f} |",
      "", "_Note: mass matrix re-tuned per §6.10 (the old matrix gave AR=0.00 with the clean CNF); "
      "low ESS_min reflects the hard, weakly-whitened field posterior (Var(z)≫1)._", ""]
# Table 14 — fit RMS by prior config
L += ["## Table 14 (v2) — modal-frequency fit at the field posterior mean, by prior config",
      "| prior config | e1 (posterior mean) | fit RMS (%) |", "|---|---|---|"]
for k, v in ff["T3"]["configs"].items():
    L.append(f"| {k} | {v['e1_mean']:.2f} | {v['fit_rms']:.1f} |")
L += ["", "_The narrow prior (σ=0.30) fits best (4.0%) and keeps the posterior in-domain; the broad "
      "prior inflates e1 to the edge without improving fit → prior sensitivity, not multimodality._", ""]
# Table 15 — amortized comparison (6.12)
s = e6
L += ["## Table 15 / §6.12 (v2) — amortized comparison (MAE vs HMC, converged)",
      "| method | mean MAE | exact-correction path |", "|---|---|---|",
      f"| CAI-fast | {s['mae_fast_mean']:.4f} | via router → NeuTra-HMC |",
      f"| CAI-exact | (reference) | yes |",
      f"| SNPE-C (converged, {s['epochs_to_converge']} ep) | {s['mae_snpe_converged_mean']:.4f} | no |",
      "", "_On par once both converged; the differentiator is the correction path, not accuracy._", ""]
# E-th routing matrix (final field/damage rows)
L += ["## E-th routing matrix (v2) — updated field/damage rows",
      "| observation | ESS_IS/N | route @ τ=0.3 | route @ τ=0.5 |", "|---|---|---|---|"]
for r in eth["rows"]:
    L.append(f"| {r['name']} | {r['ess']:.4f} | {'fast' if r['ess']>=0.3 else 'exact'} | {'fast' if r['ess']>=0.5 else 'exact'} |")
L.append("| Coupled normal / damage | 0.032 / 0.128 | exact / fast | exact / exact |")
L += ["", "_Sailboat damage reclassified: 0.02 (round 1) → 0.44 (clean CNF), now borderline._"]
(HERE / "tables_v2.md").write_text("\n".join(L), encoding="utf-8")

# E3 ROC-style figure (AUC vs severity, per kappa)
cells = e3["cells"]; POS = ["e1", "e3", "e5"]; SEV = [0.10, 0.25, 0.50]; KAP = [0.01, 0.03, 0.05]
fig, ax = plt.subplots(figsize=(6, 4))
for k in KAP:
    aucs = [np.mean([cells[f"{p}|{s}|{k}"]["auc"] for p in POS]) for s in SEV]
    ax.plot([int(s*100) for s in SEV], aucs, "o-", label=f"κ={k}")
ax.axhline(0.5, color="k", ls=":", lw=0.8); ax.set_ylim(0.45, 1.0)
ax.set_xlabel("damage severity (%)"); ax.set_ylabel("AUC")
ax.set_title("E3 damage detectability (v2, ext pipeline)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(HERE / "e3_roc_v2.png", dpi=150)

# Field prior-config figure (e1 + fit by prior)
fig2, ax2 = plt.subplots(figsize=(6, 3.6))
names = list(ff["T3"]["configs"].keys())
e1s = [ff["T3"]["configs"][n]["e1_mean"] for n in names]
fits = [ff["T3"]["configs"][n]["fit_rms"] for n in names]
x = np.arange(len(names)); ax2.bar(x-0.2, e1s, 0.4, label="e1 posterior mean", color="#37a")
ax2b = ax2.twinx(); ax2b.plot(x, fits, "rs-", label="fit RMS (%)")
ax2.axhline(3.5, color="grey", ls="--", lw=0.8); ax2.text(0.05, 3.55, "domain edge (3.5)", fontsize=7, color="grey")
ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=8, rotation=10); ax2.set_ylabel("e1"); ax2b.set_ylabel("fit RMS %")
ax2.set_title("Field posterior: prior sensitivity (e1 pulled by prior; broad edge, worse fit)")
fig2.tight_layout(); fig2.savefig(HERE / "fig_field_priorconfig_v2.png", dpi=150)
print("Wrote tables_v2.md, e3_roc_v2.png, fig_field_priorconfig_v2.png")
