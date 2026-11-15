"""Independent file, threshold and sample-preservation verification; no MCMC."""
import csv
import json
import math
from pathlib import Path

import numpy as np

from diagnose_four_chains import OUT, CONTROL, sha, verify_preservation, write_new_json, utc_now


def close(actual, expected):
    assert math.isclose(float(actual), float(expected), rel_tol=1e-13, abs_tol=1e-13)


def main():
    integrity_path = OUT/"REFERENCE_QUALITY_INTEGRITY.json"
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    manifest = json.loads((CONTROL/"TARGET_MANIFEST.json").read_text(encoding="utf-8"))
    with (OUT/"REFERENCE_QUALITY_SUMMARY.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 7 and [row["target"] for row in rows] == [row["target"] for row in manifest["targets"]]
    results = {result["target"]: result for result in integrity["targets"]}
    observed_divergences = 0
    init = np.exp(np.linspace(.20, -.20, 6))
    for row, target in zip(rows, manifest["targets"]):
        name = row["target"]
        result = results[name]
        run_path = OUT/f"REFERENCE4_{name}_RUN.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        assert run["status"] == "COMPLETED" and run["scientific_chain_calls"] == 1
        assert run["target_manifest"] == target and target["seed"] == 704
        assert target["num_samples"] == 600 and target["warmup_steps"] == 500 and target["target_accept_prob"] == .8
        assert np.array_equal(np.asarray(target["theta_init"]), init)
        assert sha(run["artifact_path"]) == run["artifact_sha256"] == row["reference4_sha256"]
        assert sha(run["log_path"]) == run["log_sha256"]
        with np.load(run["artifact_path"], allow_pickle=False) as fourth, np.load(target["original_reference_path"], allow_pickle=False) as old:
            assert fourth["chains"].shape == (4,600,6) and fourth["theta_samples"].shape == (600,6)
            assert np.array_equal(fourth["chains"][:3], old["chains"])
            assert np.array_equal(fourth["chains"][3], fourth["theta_samples"])
            assert np.array_equal(fourth["observation"], old["observation"])
            assert np.array_equal(fourth["theta_init"], init)
            assert np.array_equal(fourth["chain_seeds"], [701,702,703,704])
            assert np.isfinite(fourth["chains"]).all() and (fourth["chains"] > 0).all()
            diag = json.loads(fourth["sampler_diagnostics_json"].item())
            indices = diag["divergences"]["chain 0"]
            assert indices == fourth["divergences"].tolist() == run["fourth_chain_divergence_indices"]
            assert len(indices) == run["fourth_chain_divergence_count"] == int(row["fourth_chain_divergences"])
            observed_divergences += len(indices)
            current_mean = fourth["chains"].reshape(-1,6).astype(np.float64).mean(axis=0)
            original_mean = old["ref_mean"].astype(np.float64)
            close(float(np.abs(current_mean-original_mean).mean()), row["three_to_four_mean_mae_all6"])
            close(float(np.abs(current_mean[:3]-original_mean[:3]).mean()), row["three_to_four_mean_mae_active_e1_e3"])
        diagnostics = result["diagnostics_4chain"]
        close(row["max_rank_normalized_split_rhat"], max(diagnostics["rank_normalized_split_rhat"]))
        close(row["min_bulk_ess"], min(diagnostics["bulk_ess"]))
        close(row["min_tail_ess"], min(diagnostics["tail_ess"]))
        assert row["rhat_gate"] == ("PASS" if float(row["max_rank_normalized_split_rhat"]) <= 1.01 else "FAIL")
        assert row["bulk_ess_gate"] == ("PASS" if float(row["min_bulk_ess"]) >= 400 else "FAIL")
        assert row["tail_ess_gate"] == ("PASS" if float(row["min_tail_ess"]) >= 400 else "FAIL")
        assert row["divergences_gate"] == row["overall_quality_gate"] == "FAIL"
        assert row["all_4chain_divergences"] == "" and result["all_four_chain_divergence_count"] is None
    summary = integrity["summary"]
    assert summary["fourth_chains_completed"] == 7
    assert summary["fourth_chain_divergences_total"] == observed_divergences
    assert summary["all_28_chains_divergences_total"] is None and not summary["all_7_quality_gates_pass"]
    for name in ["REFERENCE_4CHAIN_SUMMARY.csv", "ROUTING_VS_GAP_4CHAIN.csv", "ROUTING_VS_GAP_4CHAIN_AUDIT.md",
                 "FULL_ABLATION_4CHAIN_REFERENCE.csv", "FULL_ABLATION_4CHAIN_REFERENCE_AUDIT.md"]:
        assert not (OUT/name).exists(), "Conditional artifact must not be created after a quality failure."
    snapshot = json.loads((CONTROL/"FROZEN_BEFORE_REFERENCE_QUALITY.json").read_text(encoding="utf-8"))
    preservation = verify_preservation(snapshot)
    assert len(snapshot["original_367_files"]) == 367
    for output in integrity["outputs"].values():
        assert sha(output["path"]) == output["sha256"]
    verification = dict(integrity_verification_status="PASS", full_quality_gate_pass=False, checked_utc=utc_now(),
                        fourth_chains_completed=7, fourth_chain_divergences_total=observed_divergences,
                        original_three_chains_preserved_exactly=True, historical_preservation=preservation,
                        original_367_unchanged=True, flow_training_calls=0, conditional_reevaluations_executed=False,
                        review_stop_confirmed=True, outputs={path.name: sha(path) for path in [
                         OUT/"REFERENCE_QUALITY_SUMMARY.csv", OUT/"REFERENCE_QUALITY_AUDIT.md", integrity_path]})
    write_new_json(CONTROL/"FINAL_VERIFICATION.json", verification)
    print(json.dumps(verification, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
