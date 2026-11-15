"""Independent stored-evidence checks, without sampler or flow evaluation calls."""
import csv
import hashlib
import io
import json
import os
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

CONTROL = Path(__file__).resolve().parent
HOUT = CONTROL.parent
OUT = HOUT.parent
HERE = OUT.parent
ROOT = HERE.parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(a, b):
    return bool(np.isclose(float(a), float(b), atol=1e-14, rtol=1e-13))


def main():
    result_path = CONTROL / "FINAL_VERIFICATION.json"
    assert not result_path.exists()
    snapshot = read(CONTROL / "FROZEN_BEFORE_FULL_RECOVERY.json")
    changed = [p for p, digest in snapshot["files"].items() if not Path(p).is_file() or sha(p) != digest]
    assert not changed, changed
    git = subprocess.check_output(["git", "status", "--porcelain", "-uno"], cwd=ROOT,
        env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).decode("utf-8")
    assert git == snapshot["git_tracked_before"]
    assert all(sha(p) == digest for p, digest in snapshot["git_tracked_file_hashes"].items())
    assert all(sha(p) == digest for p, digest in snapshot["frozen_pilot_files"].items())
    history = read(HOUT / "HISTORICAL_DIAGNOSTICS_RECOVERY_INTEGRITY.json")
    manifests = read(CONTROL / "REMAINING20_MANIFEST.json")["chains"]
    calls = list(CONTROL.glob("ATTEMPT_*_seed*.json"))
    assert len(calls) == history["summary"]["remaining_chain_calls"]
    assert len(calls) <= len(manifests) == 20 and history["summary"]["pilot_chain_reruns"] == 0
    planned = {(m["target"], m["seed"]) for m in manifests}
    assert ("ablation_obs1", 701) not in planned
    verified_chains, exact_count, total = [], 0, 0
    for record in history["chain_results"]:
        manifest = record["manifest"]
        if record["status"] != "COMPLETED":
            continue
        with np.load(record["artifact_path"], allow_pickle=False) as saved, np.load(manifest["reference_path"], allow_pickle=False) as old:
            replay = saved["replay_theta"].copy()
            historical = old["chains"][manifest["historical_chain_index"]]
            assert np.array_equal(saved["historical_theta"], historical)
            assert replay.shape == historical.shape == (600, 6)
            assert replay.dtype == historical.dtype and np.isfinite(replay).all()
            exact = bool(np.array_equal(replay, historical))
            difference = np.abs(replay.astype(np.float64)-historical.astype(np.float64))
            comparison = record["comparison"]
            assert exact == comparison["np_array_equal"]
            assert float(difference.max()) == comparison["max_abs_difference"]
            assert float(difference.mean()) == comparison["mean_abs_difference"]
            assert int(np.count_nonzero(replay != historical)) == comparison["number_of_different_elements"]
            raw_bytes = saved["raw_mc_diagnostics_torch_bytes"].tobytes()
            assert hashlib.sha256(raw_bytes).hexdigest() == record["raw_diagnostics_serialized_sha256"]
            raw = torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=True)
            indices = raw["divergences"]["chain 0"]
            acceptance = raw["acceptance rate"]["chain 0"]
            assert indices == saved["divergences"].tolist() == record["replay_divergence_indices"]
            assert len(indices) == record["replay_divergence_count"]
            assert acceptance == saved["acceptance_rate"].item() == record["replay_acceptance_rate"]
            if exact:
                assert record["recovered_historical_divergence_count"] == len(indices)
                exact_count += 1
                total += len(indices)
            else:
                assert record["recovered_historical_divergence_count"] is None
                assert not record["replay_diagnostics_attributed_to_historical_chain"]
        assert sha(record["artifact_path"]) == record["artifact_sha256"]
        assert sha(record["log_path"]) == record["log_sha256"]
        verified_chains.append(dict(target=manifest["target"], seed=manifest["seed"], exact=exact, divergence_count=len(indices)))
    assert exact_count == history["summary"]["exact_reproduced_chains"]
    quality_path = OUT / "REFERENCE_QUALITY_INTEGRITY_RECOVERED.json"
    quality = read(quality_path) if exact_count == 21 else None
    if quality is not None:
        assert total == quality["summary"]["historical_divergences_total"]
        old_quality = read(OUT / "REFERENCE_QUALITY_INTEGRITY.json")
        old_lookup = {r["target"]: r for r in old_quality["targets"]}
        four_total = 0
        for target in quality["targets"]:
            name = target["target"]
            run704 = read(OUT / f"REFERENCE4_{name}_RUN.json")
            with np.load(target["reference4_path"], allow_pickle=False) as saved, np.load(target["target_manifest"]["original_reference_path"], allow_pickle=False) as old:
                chains = saved["chains"]
                assert chains.shape == (4, 600, 6) and np.array_equal(chains[:3], old["chains"])
                assert np.array_equal(chains[3], saved["theta_samples"])
                count704 = len(saved["divergences"])
                assert count704 == run704["fourth_chain_divergence_count"] == target["seed704_divergence_count"]
                pooled = chains.reshape(-1, 6).astype(np.float64)
                assert np.array_equal(pooled.mean(0), target["reference_4chain_mean"])
                assert np.array_equal(np.quantile(pooled, [.025, .25, .5, .75, .975], axis=0), target["reference_4chain_quantiles"])
            for key in ["rank_normalized_split_rhat", "bulk_ess", "tail_ess"]:
                assert np.array_equal(target["diagnostics_4chain"][key], old_lookup[name]["diagnostics_4chain"][key])
            expected_gate = target["diagnostics_4chain"]["max_rhat"] <= 1.01 and target["diagnostics_4chain"]["min_bulk_ess"] >= 400 and target["diagnostics_4chain"]["min_tail_ess"] >= 400 and sum(target["historical_divergence_counts"]) + count704 == 0
            assert (target["overall_quality_gate"] == "PASS") == expected_gate
            four_total += target["all_4chain_divergences"]
        assert four_total == quality["summary"]["all_28_chains_divergences_total"]
        assert quality["arviz_version"] == "0.23.4"
        passed = quality["summary"]["all_7_complete_quality_gates_pass"]
        if passed:
            routing = quality["routing_reevaluation"]
            route_original = {r["target"]:r for r in csv_rows(HERE / "ROUTING_VS_GAP_TABLE.csv")}
            assert len(routing["rows"]) == 7
            for row in routing["rows"]:
                for key in ["raw_mean", "psis_mean", "k_mean", "route_raw", "route_psis"]:
                    assert row[key] == route_original[row["target"]][key]
                assert row["fast_samples_filtered"] == 0
            ablation = quality["ablation_reevaluation"]
            assert len(ablation["scientific_checkpoint_evaluations"]) == 24
            original = read(HERE / "ablation_multiseed/FULL_ABLATION_INTEGRITY_RECOVERED.json")
            old_lookup = {(r["config"], int(r["training_seed"])):r for r in original["scientific_results"]}
            for row in ablation["scientific_checkpoint_evaluations"]:
                assert row["original_training_metadata"] == old_lookup[row["config"], row["training_seed"]]
                assert sha(row["checkpoint"]) == row["checkpoint_sha256"]
                metrics = row["evaluation_4chain"]
                fast = np.asarray(metrics["fast_seed_results"])
                assert fast[:, 0].tolist() == [42, 43, 44, 45, 46]
                for index, metric, sd in [(1, "mae_ref_all6", "fast_seed_sd_all6"), (2, "mae_ref_active", "fast_seed_sd_active")]:
                    assert close(statistics.mean(fast[:, index]), metrics[metric])
                    assert close(statistics.stdev(fast[:, index]), metrics[sd])
                assert metrics["proposals_filtered"] == 0
            for group in ablation["groups"].values():
                for metric in ["mae_ref_all6", "mae_ref_active"]:
                    values = group[metric]["four_chain"]["seed_values"]
                    assert close(statistics.mean(values), group[metric]["four_chain"]["mean"])
                    assert close(statistics.stdev(values), group[metric]["four_chain"]["sample_sd"])
                    assert min(values) == group[metric]["four_chain"]["min"] and max(values) == group[metric]["four_chain"]["max"]
            common = [r for r in ablation["axis_rows"] if r["config"] == "N50000_L8"]
            assert len(common) == 2 and {k:v for k,v in common[0].items() if k != "axis"} == {k:v for k,v in common[1].items() if k != "axis"}
            assert len(csv_rows(OUT / "FULL_ABLATION_4CHAIN_REFERENCE.csv")) == 9
            assert len(csv_rows(OUT / "REFERENCE_4CHAIN_SUMMARY.csv")) == 42
        else:
            assert not (OUT / "ROUTING_VS_GAP_4CHAIN.csv").exists()
            assert not (OUT / "FULL_ABLATION_4CHAIN_REFERENCE.csv").exists()
        for output in quality["outputs"].values():
            assert sha(output["path"]) == output["sha256"]
    for output in history["outputs"].values():
        assert sha(output["path"]) == output["sha256"]
    result = dict(checked_utc=datetime.now(timezone.utc).isoformat(), verification_pass=True,
        protected_files_checked=len(snapshot["files"]), all_frozen_SHA256_unchanged=True,
        git_tracked_hashes_unchanged=True, git_tracked_status_unchanged=True, pilot_unchanged=True,
        historical_chain_replays_checked=len(verified_chains), exact_reproduced_chains=exact_count,
        recovered_historical_divergences=total, remaining_scientific_calls=len(calls), pilot_reruns=0,
        raw_diagnostics_original_torch_serialization_verified=True,
        quality_summary=quality["summary"] if quality else None,
        mcmc_stopped=True, flow_training_calls=0, awaiting_review=True,
        outputs={str(p): sha(p) for p in [HOUT / "HISTORICAL_DIAGNOSTICS_RECOVERY_INTEGRITY.json", quality_path] if p.exists()},
        verifier_sha256=sha(__file__))
    with result_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
