import csv
import hashlib
import json
from pathlib import Path

import arviz as az
import numpy as np
from scipy.stats import wasserstein_distance

ROOT = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
OUT = ROOT / "revision_experiments/cai_r2_presubmit_phaseB_20260919_213647/BOUNDARY"
STAGE1 = ROOT / "revision_experiments/cai_r2_stage1/final_ext_verification_20260918_091227"
ROUTING = ROOT / "revision_experiments/cai_r2_stage1/run_20260917_094722_21d144f3db844a6b8a2e8167c5282dbe/routing_arrays"
PARAMETERS = [f"e{i}" for i in range(1, 7)]
SEEDS = [711, 712, 713, 714]
FAST_SEEDS = [42, 43, 44, 45, 46]
Q = [0.025, 0.975]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    assert az.__version__ == "0.23.4"
    with np.load(OUT / "BOUNDARY_REFERENCE_RERUN.npz", allow_pickle=False) as a:
        chains = a["chains"].astype(np.float64)
        acceptances = a["acceptance_rates"].astype(np.float64)
        divergence_counts = a["divergence_counts"].astype(int)
    assert chains.shape == (4, 600, 6)
    data = az.from_dict(posterior={"theta": chains}, coords={"parameter": PARAMETERS}, dims={"theta": ["parameter"]})
    rhat = np.asarray(az.rhat(data, method="rank")["theta"].values, dtype=float)
    bulk = np.asarray(az.ess(data, method="bulk", relative=False)["theta"].values, dtype=float)
    tail = np.asarray(az.ess(data, method="tail", relative=False)["theta"].values, dtype=float)
    pooled = chains.reshape(-1, 6)
    total_divergences = int(divergence_counts.sum())
    gates = {
        "rank_rhat": bool(np.isfinite(rhat).all() and rhat.max() <= 1.01),
        "bulk_ess": bool(np.isfinite(bulk).all() and bulk.min() >= 400),
        "tail_ess": bool(np.isfinite(tail).all() and tail.min() >= 400),
        "divergences": total_divergences == 0,
    }
    overall_pass = all(gates.values())
    rows = []
    ref_q = np.quantile(pooled, Q, axis=0)
    for j, p in enumerate(PARAMETERS):
        rows.append({
            "parameter": p,
            "rank_normalized_split_rhat": rhat[j],
            "bulk_ess": bulk[j],
            "tail_ess": tail[j],
            "posterior_mean": pooled[:, j].mean(),
            "posterior_std": pooled[:, j].std(ddof=1),
            "q025": ref_q[0, j],
            "q975": ref_q[1, j],
            "max_rhat_all6": rhat.max(),
            "min_bulk_ess_all6": bulk.min(),
            "min_tail_ess_all6": tail.min(),
            "total_divergences": total_divergences,
            "overall_gate": "PASS" if overall_pass else "FAIL",
            "chain_seeds": json.dumps(SEEDS),
            "chain_acceptance_rates": json.dumps(acceptances.tolist()),
            "chain_divergence_counts": json.dumps(divergence_counts.tolist()),
        })
    write_csv(OUT / "BOUNDARY_REFERENCE_RERUN.csv", rows)

    gap_rows = []
    if overall_pass:
        routing_rows = list(csv.DictReader((STAGE1 / "PSIS_STAGE1_DETAIL.csv").open(encoding="utf-8")))
        routing_by_seed = {
            int(r["seed"]): r for r in routing_rows
            if r["observation"] == "sailboat_boundary_e1_075" and int(r["N"]) == 2000
        }
        fast_parts = []
        for seed in FAST_SEEDS:
            with np.load(ROUTING / f"sailboat_boundary_e1_075_seed{seed}_N2000.npz", allow_pickle=False) as a:
                fast = a["theta"].astype(np.float64)
            fast_parts.append(fast)
            fq = np.quantile(fast, Q, axis=0)
            w1 = np.asarray([wasserstein_distance(fast[:, j], pooled[:, j]) for j in range(6)])
            rr = routing_by_seed[seed]
            gap_rows.append({
                "scope": f"seed{seed}",
                "fast_seed": seed,
                "N_fast": len(fast),
                "raw_ess_ratio": float(rr["raw_ess_ratio"]),
                "psis_ess_ratio": float(rr["psis_ess_ratio"]),
                "pareto_k": float(rr["pareto_k"]),
                "raw_route_tau05": rr["route_raw_tau05"],
                "psis_route_tau05": rr["route_psis_tau05"],
                "active_mean_mae": np.abs(fast.mean(0)[:3] - pooled.mean(0)[:3]).mean(),
                "active_w1_mean": w1[:3].mean(),
                "active_w1_max": w1[:3].max(),
                "active_width_ratio_mean": ((fq[1] - fq[0])[:3] / (ref_q[1] - ref_q[0])[:3]).mean(),
                "active_width_ratio_min": ((fq[1] - fq[0])[:3] / (ref_q[1] - ref_q[0])[:3]).min(),
                "active_width_ratio_max": ((fq[1] - fq[0])[:3] / (ref_q[1] - ref_q[0])[:3]).max(),
            })
        fast = np.concatenate(fast_parts, axis=0)
        fq = np.quantile(fast, Q, axis=0)
        w1 = np.asarray([wasserstein_distance(fast[:, j], pooled[:, j]) for j in range(6)])
        gap_rows.append({
            "scope": "pooled_seeds42_46",
            "fast_seed": "42-46",
            "N_fast": len(fast),
            "raw_ess_ratio": "",
            "psis_ess_ratio": "",
            "pareto_k": "",
            "raw_route_tau05": "4 CORRECTION / 1 FAST",
            "psis_route_tau05": "4 CORRECTION / 1 FAST",
            "active_mean_mae": np.abs(fast.mean(0)[:3] - pooled.mean(0)[:3]).mean(),
            "active_w1_mean": w1[:3].mean(),
            "active_w1_max": w1[:3].max(),
            "active_width_ratio_mean": ((fq[1] - fq[0])[:3] / (ref_q[1] - ref_q[0])[:3]).mean(),
            "active_width_ratio_min": ((fq[1] - fq[0])[:3] / (ref_q[1] - ref_q[0])[:3]).min(),
            "active_width_ratio_max": ((fq[1] - fq[0])[:3] / (ref_q[1] - ref_q[0])[:3]).max(),
        })
        write_csv(OUT / "BOUNDARY_ROUTING_GAP_UPDATED.csv", gap_rows)

    gap_text = "Not computed because the reference gate failed."
    if overall_pass:
        pooled_row = gap_rows[-1]
        gap_text = (
            f"Pooled fixed fast draws (seeds 42–46, N=10,000) versus the new reference: "
            f"active mean MAE `{pooled_row['active_mean_mae']:.8f}`, active mean/max W1 "
            f"`{pooled_row['active_w1_mean']:.8f}` / `{pooled_row['active_w1_max']:.8f}`, and mean "
            f"95% interval-width ratio `{pooled_row['active_width_ratio_mean']:.8f}`."
        )
    lines = [
        "# Boundary robust-reference rerun audit",
        "",
        f"Overall gate: **{'PASS' if overall_pass else 'FAIL'}**",
        "",
        "The four predeclared chains used seeds 711/712/713/714, 500 warmup draws, 600 retained draws, target acceptance 0.95, and the frozen common initialization. No chain was dropped or replaced and no retry was performed.",
        "",
        "## Diagnostics",
        "",
        f"- maximum rank-normalized split R-hat: `{rhat.max():.10f}` ({'PASS' if gates['rank_rhat'] else 'FAIL'});",
        f"- minimum bulk ESS: `{bulk.min():.6f}` ({'PASS' if gates['bulk_ess'] else 'FAIL'});",
        f"- minimum tail ESS: `{tail.min():.6f}` ({'PASS' if gates['tail_ess'] else 'FAIL'});",
        f"- divergences: `{total_divergences}` ({'PASS' if gates['divergences'] else 'FAIL'});",
        f"- acceptance rates by seed: `{acceptances.tolist()}`.",
        "",
        "## Posterior gap",
        "",
        gap_text,
        "",
        "Routing values in the updated gap table are copied from the original seeds 42–46, N=2,000 artifacts. The robust-reference rerun did not generate or replace routing samples.",
        "",
        "## Integrity",
        "",
        f"- combined reference SHA-256: `{sha(OUT / 'BOUNDARY_REFERENCE_RERUN.npz')}`;",
        "- same 20-mode observation, broad-extended surrogate, likelihood, sigma=0.55 prior, and log-theta target as the frozen protocol;",
        "- new FE solves, flow training, and routing draws: zero.",
        "",
    ]
    (OUT / "BOUNDARY_REFERENCE_RERUN_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "BOUNDARY_REFERENCE_GATE.json").write_text(json.dumps({
        "overall_gate": "PASS" if overall_pass else "FAIL",
        "gates": gates,
        "max_rhat": float(rhat.max()),
        "min_bulk_ess": float(bulk.min()),
        "min_tail_ess": float(tail.min()),
        "total_divergences": total_divergences,
        "gap_computed": overall_pass,
    }, indent=2) + "\n", encoding="utf-8")
    print("BOUNDARY_GATE", "PASS" if overall_pass else "FAIL", rhat.max(), bulk.min(), tail.min(), total_divergences, flush=True)


if __name__ == "__main__":
    main()

