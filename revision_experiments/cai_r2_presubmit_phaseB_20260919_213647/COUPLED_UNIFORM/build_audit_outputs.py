import csv
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
OUT = ROOT / "revision_experiments/cai_r2_presubmit_phaseB_20260919_213647/COUPLED_UNIFORM"
SOURCE = ROOT / "experiments/coupled_building/04_train_cnf_coupled.py"
SURROGATE_PATH = ROOT / "surrogate_models_coupled/member_0.pt"
Y_MEASURED = np.asarray([2.06, 2.30, 2.62, 6.14, 6.53, 6.76], dtype=np.float64)
LO, HI = 0.5, 4.0


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_surrogate():
    sys.path.insert(0, str(ROOT / "experiments/coupled_building"))
    spec = importlib.util.spec_from_file_location("coupled_audit_source", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Surrogate().double().cpu()
    model.load_state_dict(torch.load(SURROGATE_PATH, map_location="cpu", weights_only=True))
    model.eval()
    return model


def in_support(theta):
    return ((theta >= LO) & (theta <= HI)).all(axis=1)


def save_figure(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    expected = [
        OUT / "COUPLED_UNIFORM_TABLE13_AUDIT.csv",
        OUT / "COUPLED_UNIFORM_TABLE14_AUDIT.csv",
        OUT / "FIG15_COUPLED_UNIFORM_AUDIT.png",
        OUT / "FIG15_COUPLED_UNIFORM_AUDIT.pdf",
        OUT / "FIG16_COUPLED_UNIFORM_DAMAGE_AUDIT.png",
        OUT / "FIG16_COUPLED_UNIFORM_DAMAGE_AUDIT.pdf",
        OUT / "COUPLED_UNIFORM_AUDIT_OUTPUTS.json",
    ]
    existing = [str(path) for path in expected if path.exists()]
    if existing:
        raise RuntimeError(f"Refusing to overwrite audit outputs: {existing}")

    summary = json.loads((OUT / "COUPLED_UNIFORM_FAST_ROUTING_SUMMARY.json").read_text(encoding="utf-8"))
    gate = json.loads((OUT / "COUPLED_UNIFORM_NUTS_GATE.json").read_text(encoding="utf-8"))
    raw_nuts = json.loads((OUT / "COUPLED_UNIFORM_NUTS_RAW.json").read_text(encoding="utf-8"))
    with np.load(OUT / "COUPLED_UNIFORM_FAST_ROUTING_RAW.npz", allow_pickle=False) as data:
        theta = data["theta"].astype(np.float64)
        latent_seeds = data["latent_seeds"].astype(int)
        training_seeds = data["flow_training_seeds"].astype(int)
    with np.load(OUT / "COUPLED_UNIFORM_NUTS_REFERENCE.npz", allow_pickle=False) as data:
        reference = data["pooled"].astype(np.float64)

    assert training_seeds.tolist() == [42, 202, 303]
    assert latent_seeds.tolist() == [42, 43, 44, 45, 46]
    assert gate["overall_gate"] == "FAIL"
    assert in_support(reference).all()

    primary_field_raw = theta[0, 0, 0]
    primary_damage_raw = theta[0, 1, 0]
    field_valid = primary_field_raw[in_support(primary_field_raw)]
    damage_valid = primary_damage_raw[in_support(primary_damage_raw)]

    total_nuts_runtime = float(sum(record["runtime_s"] for record in raw_nuts["records"]))
    primary_field = summary["42"]["field"]
    table13 = [
        {
            "method": "Uniform-CNF fast proposal (primary training seed 42)",
            "status": "AUDIT ONLY",
            "sample_definition": "N=2000 latent-seed-42 proposals; bounded summaries reject outside-prior proposals",
            "time": f"{primary_field['mean_inference_time_ms']:.6f} ms (mean seeds 42-46)",
            "mcmc_ess_min": "",
            "acceptance_rate": "",
            "raw_ess_is_over_n_mean": primary_field["raw_ess_ratio_mean"],
            "psis_ess_over_n_mean": primary_field["psis_ess_ratio_mean"],
            "pareto_k_mean": primary_field["pareto_k_mean"],
            "route_tau_0_5": "CORRECTION (5/5 raw; 5/5 PSIS)",
            "qualification": "proposal/routing result complete; exact comparison unqualified",
        },
        {
            "method": "Bounded-Uniform transformed NUTS (4 chains)",
            "status": "GATE FAIL",
            "sample_definition": "4 x 600 retained; seeds 811/812/813/814",
            "time": f"{total_nuts_runtime:.6f} s total serial runtime",
            "mcmc_ess_min": gate["min_bulk_ess"],
            "acceptance_rate": float(np.mean(gate["acceptance_rates"])),
            "raw_ess_is_over_n_mean": "",
            "psis_ess_over_n_mean": "",
            "pareto_k_mean": "",
            "route_tau_0_5": "",
            "qualification": f"FAIL: max rank-Rhat={gate['max_rhat']:.9f} > 1.01; no chain changed",
        },
    ]
    write_csv(OUT / "COUPLED_UNIFORM_TABLE13_AUDIT.csv", table13)

    model = load_surrogate()
    posterior_mean = np.asarray(gate["posterior_mean"], dtype=np.float64)
    with torch.no_grad():
        predicted = model(torch.as_tensor(posterior_mean).unsqueeze(0)).squeeze(0).numpy()[:6]
    rel_error_pct = 100.0 * (predicted - Y_MEASURED) / Y_MEASURED
    table14 = []
    for index, (measured, pred, error) in enumerate(zip(Y_MEASURED, predicted, rel_error_pct), 1):
        table14.append({
            "mode": index,
            "measured_hz": measured,
            "predicted_at_unqualified_nuts_candidate_mean_hz": pred,
            "relative_error_pct": error,
            "qualification": "AUDIT ONLY: four-chain NUTS gate FAIL",
        })
    write_csv(OUT / "COUPLED_UNIFORM_TABLE14_AUDIT.csv", table14)

    colors = {"fast": "#2878B5", "nuts": "#E07A1F", "field": "#2878B5", "damage": "#D84A3A"}
    labels = [r"$e_1$", r"$e_2$", r"$e_3$"]
    bins = np.linspace(LO, HI, 46)

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.25), sharey=False)
    for j, ax in enumerate(axes):
        ax.hist(field_valid[:, j], bins=bins, density=True, histtype="stepfilled", alpha=0.30,
                color=colors["fast"], label="Uniform-CNF fast (in-support)")
        ax.hist(reference[:, j], bins=bins, density=True, histtype="step", linewidth=2.0,
                color=colors["nuts"], label="Bounded NUTS candidate")
        ax.set_xlim(LO, HI)
        ax.set_xlabel(labels[j])
        ax.set_ylabel("Density")
        ax.grid(alpha=0.18)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Coupled-building field posterior audit — NUTS convergence gate FAIL", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "FIG15_COUPLED_UNIFORM_AUDIT")

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.25), sharey=False)
    for j, ax in enumerate(axes):
        ax.hist(field_valid[:, j], bins=bins, density=True, histtype="step", linewidth=1.8,
                color=colors["field"], label="Field (in-support fast)")
        ax.hist(damage_valid[:, j], bins=bins, density=True, histtype="stepfilled", alpha=0.28,
                color=colors["damage"], label="Damage (in-support fast)")
        ax.axvline([0.5, 1.0, 1.0][j], color="black", linestyle="--", linewidth=1.2,
                   label="Damage truth" if j == 0 else None)
        ax.set_xlim(LO, HI)
        ax.set_xlabel(labels[j])
        ax.set_ylabel("Density")
        ax.grid(alpha=0.18)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Coupled-building damage routing audit — bounded support displayed", fontsize=11)
    fig.tight_layout()
    save_figure(fig, "FIG16_COUPLED_UNIFORM_DAMAGE_AUDIT")

    plotted = np.vstack([field_valid, damage_valid, reference])
    output = {
        "status": "AUDIT_ONLY_NOT_SUBMISSION_QUALIFIED",
        "reason": "four-chain bounded-Uniform NUTS gate failed rank-Rhat criterion",
        "table13": {
            "standard_hmc_omitted": True,
            "neutra_crosscheck_performed": False,
            "nuts_total_serial_runtime_s": total_nuts_runtime,
        },
        "table14": {
            "posterior_mean_source": "unqualified bounded-Uniform NUTS candidate",
            "posterior_mean": posterior_mean.tolist(),
            "rms_relative_error_pct": float(np.sqrt(np.mean((rel_error_pct / 100.0) ** 2)) * 100.0),
        },
        "figure15": {
            "fast_source": "primary training seed 42, latent seed 42, in-support proposals only",
            "fast_raw_count": int(len(primary_field_raw)),
            "fast_in_support_count": int(len(field_valid)),
            "fast_raw_above_4_count": int((primary_field_raw > HI).any(axis=1).sum()),
            "fast_raw_below_0_5_count": int((primary_field_raw < LO).any(axis=1).sum()),
            "plotted_theta_above_4_count": int((np.vstack([field_valid, reference]) > HI).any(axis=1).sum()),
            "plotted_theta_below_0_5_count": int((np.vstack([field_valid, reference]) < LO).any(axis=1).sum()),
            "exact_candidate_theta_above_4_count": int((reference > HI).any(axis=1).sum()),
            "exact_candidate_theta_below_0_5_count": int((reference < LO).any(axis=1).sum()),
        },
        "figure16": {
            "damage_fast_raw_count": int(len(primary_damage_raw)),
            "damage_fast_in_support_count": int(len(damage_valid)),
            "damage_fast_raw_above_4_count": int((primary_damage_raw > HI).any(axis=1).sum()),
            "damage_fast_raw_below_0_5_count": int((primary_damage_raw < LO).any(axis=1).sum()),
            "plotted_theta_above_4_count": int((np.vstack([field_valid, damage_valid]) > HI).any(axis=1).sum()),
            "plotted_theta_below_0_5_count": int((np.vstack([field_valid, damage_valid]) < LO).any(axis=1).sum()),
        },
        "all_plotted_samples_theta_above_4_count": int((plotted > HI).any(axis=1).sum()),
        "all_plotted_samples_theta_below_0_5_count": int((plotted < LO).any(axis=1).sum()),
        "support_note": (
            "The unchanged RealNVP has R^3 proposal support. Out-of-box draws are proposals, not bounded-target "
            "posterior mass; the target assigns them zero density. Audit figures explicitly display only in-support "
            "fast proposals. This truncation must be disclosed if a later qualified reference is landed."
        ),
    }
    (OUT / "COUPLED_UNIFORM_AUDIT_OUTPUTS.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
