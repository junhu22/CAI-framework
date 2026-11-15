import csv
import hashlib
import json
from pathlib import Path

import arviz as az
import numpy as np

ROOT = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
OUT = ROOT / "revision_experiments/cai_r2_presubmit_phaseB_20260919_213647/COUPLED_UNIFORM"
PARAMETERS = ["e1", "e2", "e3"]
SEEDS = [811, 812, 813, 814]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    assert az.__version__ == "0.23.4"
    with np.load(OUT / "COUPLED_UNIFORM_NUTS_REFERENCE.npz", allow_pickle=False) as a:
        chains = a["chains"].astype(np.float64)
        acceptance = a["acceptance_rates"].astype(float)
        divergence_counts = a["divergence_counts"].astype(int)
    assert chains.shape == (4, 600, 3)
    data = az.from_dict(posterior={"theta": chains}, coords={"parameter": PARAMETERS}, dims={"theta": ["parameter"]})
    rhat = np.asarray(az.rhat(data, method="rank")["theta"].values, dtype=float)
    bulk = np.asarray(az.ess(data, method="bulk", relative=False)["theta"].values, dtype=float)
    tail = np.asarray(az.ess(data, method="tail", relative=False)["theta"].values, dtype=float)
    divergences = int(divergence_counts.sum())
    gates = {
        "rank_rhat": bool(np.isfinite(rhat).all() and rhat.max() <= 1.01),
        "bulk_ess": bool(np.isfinite(bulk).all() and bulk.min() >= 400),
        "tail_ess": bool(np.isfinite(tail).all() and tail.min() >= 400),
        "divergences": divergences == 0,
        "bounded_support": bool((chains >= 0.5).all() and (chains <= 4.0).all()),
    }
    passed = all(gates.values())
    pooled = chains.reshape(-1, 3)
    quantiles = np.quantile(pooled, [0.025, 0.5, 0.975], axis=0)
    rows = []
    for j, p in enumerate(PARAMETERS):
        rows.append({
            "parameter": p,
            "rank_normalized_split_rhat": rhat[j],
            "bulk_ess": bulk[j],
            "tail_ess": tail[j],
            "posterior_mean": pooled[:, j].mean(),
            "posterior_std": pooled[:, j].std(ddof=1),
            "q025": quantiles[0, j],
            "median": quantiles[1, j],
            "q975": quantiles[2, j],
            "min_sample": pooled[:, j].min(),
            "max_sample": pooled[:, j].max(),
            "theta_above_4_count_all": int((chains > 4).sum()),
            "theta_below_0_5_count_all": int((chains < 0.5).sum()),
            "overall_gate": "PASS" if passed else "FAIL",
        })
    with (OUT / "COUPLED_UNIFORM_NUTS.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "overall_gate": "PASS" if passed else "FAIL",
        "gates": gates,
        "rank_rhat": rhat.tolist(),
        "bulk_ess": bulk.tolist(),
        "tail_ess": tail.tolist(),
        "max_rhat": float(rhat.max()),
        "min_bulk_ess": float(bulk.min()),
        "min_tail_ess": float(tail.min()),
        "divergences": divergences,
        "acceptance_rates": acceptance.tolist(),
        "posterior_mean": pooled.mean(0).tolist(),
        "posterior_std": pooled.std(0, ddof=1).tolist(),
        "theta_above_4_count": int((chains > 4).sum()),
        "theta_below_0_5_count": int((chains < 0.5).sum()),
        "reference_sha256": sha(OUT / "COUPLED_UNIFORM_NUTS_REFERENCE.npz"),
    }
    (OUT / "COUPLED_UNIFORM_NUTS_GATE.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Coupled bounded-Uniform four-chain NUTS audit",
        "",
        f"Overall gate: **{result['overall_gate']}**",
        "",
        "- target: independent Uniform(0.5,4.0)^3 with first-six-mode likelihood;",
        "- transform: theta=0.5+3.5 sigmoid(z), including the exact transformation Jacobian;",
        "- chains: seeds 811/812/813/814, 500 warmup, 600 retained, target acceptance 0.90;",
        f"- maximum rank-normalized split R-hat: `{result['max_rhat']:.10f}`;",
        f"- minimum bulk/tail ESS: `{result['min_bulk_ess']:.6f}` / `{result['min_tail_ess']:.6f}`;",
        f"- divergences: `{divergences}`;",
        f"- acceptance rates: `{acceptance.tolist()}`;",
        f"- samples above 4 or below 0.5: `{result['theta_above_4_count']}` / `{result['theta_below_0_5_count']}`.",
        "",
        "No chain was dropped or replaced; no retry was performed.",
        "",
    ]
    (OUT / "COUPLED_UNIFORM_NUTS_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print("COUPLED_UNIFORM_NUTS_GATE", result["overall_gate"], result["max_rhat"], result["min_bulk_ess"], result["min_tail_ess"], divergences, flush=True)


if __name__ == "__main__":
    main()

