"""Conditional fixed-seed checkpoint evaluation; no train function is compiled."""
import ast
import csv
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

CONTROL = Path(__file__).resolve().parent
HOUT = CONTROL.parent
OUT = HOUT.parent
HERE = OUT.parent
ROOT = HERE.parents[2]
ABLATION = HERE / "ablation_multiseed"
spec = importlib.util.spec_from_file_location("frozen_ablation_helpers", ABLATION / "phase_b_control/common.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)  # Helpers only; no frozen result writers are called.
METRICS = ["mae_ref_all6", "mae_ref_active"]
SEEDS = [101, 202, 303]
DATA = ["N5000_L8", "N10000_L8", "N25000_L8", "N50000_L8", "N100000_L8"]
DEPTH = ["N50000_L4", "N50000_L6", "N50000_L8", "N50000_L12"]
RESULT = CONTROL / "ABLATION_4CHAIN_EVALUATION.json"
CSV = OUT / "FULL_ABLATION_4CHAIN_REFERENCE.csv"
AUDIT = OUT / "FULL_ABLATION_4CHAIN_REFERENCE_AUDIT.md"


def stats(values):
    array = np.asarray(values, dtype=np.float64)
    assert array.shape == (3,) and np.isfinite(array).all()
    return dict(seed_values=array.tolist(), mean=float(array.mean()), sample_sd=float(array.std(ddof=1)),
                min=float(array.min()), max=float(array.max()))


def main():
    for path in [RESULT, CSV, AUDIT, CONTROL / "ABLATION_EVALUATION_STARTED.json"]:
        assert not path.exists(), "Do not overwrite or repeat a previous evaluation: " + str(path)
    calculation = json.loads((CONTROL / "RECOVERED_QUALITY_CALCULATION.json").read_text(encoding="utf-8"))
    assert calculation["summary"]["all_7_complete_quality_gates_pass"]
    history = json.loads((HOUT / "HISTORICAL_DIAGNOSTICS_RECOVERY_INTEGRITY.json").read_text(encoding="utf-8"))
    assert history["summary"]["all_21_exact_reproduced"]
    snapshot = json.loads((CONTROL / "FROZEN_BEFORE_FULL_RECOVERY.json").read_text(encoding="utf-8"))
    old_integrity_path = ABLATION / "FULL_ABLATION_INTEGRITY_RECOVERED.json"
    source = json.loads(old_integrity_path.read_text(encoding="utf-8"))
    old_rows = source["scientific_results"]
    assert len(old_rows) == 24 and all(row["status"] == "COMPLETED" for row in old_rows)
    checks = {(r["config"], int(r["training_seed"])): r for r in source["run_verifications"]}
    old_lookup = {(r["config"], int(r["training_seed"])): r for r in old_rows}
    assert len(old_lookup) == 24
    special = old_lookup["N50000_L8", 202]
    assert Path(special["checkpoint"]).name == "N50000_L8_seed202_recovery.pt"
    assert special["prior_failed_attempt_preserved"]
    torch.set_num_threads(6)
    assert torch.__version__ == "2.6.0+cu124"
    sys.path.insert(0, str(ROOT / "experiments"))
    from cai_lib import ConditionalRealNVP
    runner = HERE / "run_ablation_multiseed.py"
    frozen_runner_sha = source["audit_source"]["frozen_runner_sha256"] if "frozen_runner_sha256" in source["audit_source"] else snapshot["files"][str(runner)]
    assert C.sha(runner) == frozen_runner_sha
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    evaluation = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
    reference_targets = {r["target"]: r for r in calculation["targets"]}
    references, observations = [], []
    for i in range(1, 6):
        target = reference_targets[f"ablation_obs{i}"]
        references.append(np.asarray(target["reference_4chain_mean"], dtype=np.float32))
        with np.load(target["target_manifest"]["original_reference_path"], allow_pickle=False) as saved:
            observations.append(saved["observation"].astype(np.float32))
    env = dict(torch=torch, np=np, time=time, FAST_SEEDS=[42, 43, 44, 45, 46],
               obs=np.stack(observations), refs=np.stack(references))
    exec(compile(ast.Module(body=[evaluation], type_ignores=[]), str(runner) + ":frozen_evaluate_only", "exec"), env)
    assert "train" not in env
    context = dict(env=env)
    before = verify(snapshot)
    write_json(CONTROL / "ABLATION_EVALUATION_STARTED.json", dict(started_utc=C.utc_now(),
        checkpoint_count=24, training_calls=0, evaluation_seed_list=[42, 43, 44, 45, 46],
        Nfast=2000, train_function_compiled=False, runner_sha256=C.sha(runner)))
    raw = []
    for config in list(dict.fromkeys(DATA + DEPTH)):
        for seed in SEEDS:
            old = old_lookup[config, seed]
            checkpoint = Path(old["checkpoint"])
            digest = checks[config, seed]["checkpoint_sha256"]
            assert C.sha(checkpoint) == digest
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            assert all(torch.isfinite(value).all().item() for value in state.values())
            model = ConditionalRealNVP(6, 20, n_layers=int(old["n_layers"]), hidden_dim=128)
            loaded = model.load_state_dict(state, strict=True)
            assert not loaded.missing_keys and not loaded.unexpected_keys
            model.eval()
            metrics = C.evaluate_with_observation(model, context)
            old_calls = checks[config, seed]["sample_observation"]["sample_calls"]
            assert metrics["sample_calls"] == old_calls, "Deterministic fast proposal replay changed."
            for label in ["negative_all6_proposals", "negative_active_proposals", "negative_all6_elements"]:
                assert metrics[label] == checks[config, seed]["sample_observation"][label]
            row = dict(config=config, training_seed=seed, checkpoint=str(checkpoint), checkpoint_sha256=digest,
                original_training_metadata=old, evaluation_4chain=metrics, strict_load_pass=True, state_all_finite=True,
                original_fixed_seed_proposal_diagnostics_exactly_reproduced=True,
                training_metadata_unchanged=True, validation_not_reselected=True, flow_weights_unchanged=True)
            raw.append(row)
            print("FOUR_CHAIN_FIXED_CHECKPOINT_EVAL", len(raw), "/24", config, seed, metrics["mae_ref_all6"], metrics["mae_ref_active"], flush=True)
    assert len(raw) == 24
    lookup = {(r["config"], r["training_seed"]): r for r in raw}
    groups, axis_rows = {}, []
    for config in list(dict.fromkeys(DATA + DEPTH)):
        group = dict(config=config, training_seeds=SEEDS)
        for metric in METRICS:
            new = stats([lookup[config, seed]["evaluation_4chain"][metric] for seed in SEEDS])
            old = stats([float(old_lookup[config, seed][metric]) for seed in SEEDS])
            group[metric] = dict(three_chain=old, four_chain=new,
                mean_change=new["mean"]-old["mean"], sample_sd_change=new["sample_sd"]-old["sample_sd"])
        groups[config] = group
    for axis, configs in [("data_size", DATA), ("depth", DEPTH)]:
        for config in configs:
            first = old_lookup[config, 101]
            row = dict(axis=axis, config=config, n_train=int(first["n_train"]), n_layers=int(first["n_layers"]),
                training_seeds=json.dumps(SEEDS), training_seed_count=3, training_seed_sd_ddof=1,
                shared_N50000_L8_scientific_result=config == "N50000_L8")
            for metric in METRICS:
                comparison = groups[config][metric]
                for suffix in ["mean", "sample_sd", "min", "max"]:
                    row[f"{metric}_4chain_{suffix}"] = comparison["four_chain"][suffix]
                row[f"{metric}_3chain_mean"] = comparison["three_chain"]["mean"]
                row[f"{metric}_3chain_sample_sd"] = comparison["three_chain"]["sample_sd"]
                row[f"{metric}_mean_change"] = comparison["mean_change"]
                row[f"{metric}_sample_sd_change"] = comparison["sample_sd_change"]
            for seed in SEEDS:
                scientific = lookup[config, seed]
                old, metrics = scientific["original_training_metadata"], scientific["evaluation_4chain"]
                for key in METRICS + ["fast_seed_sd_all6", "fast_seed_sd_active"]:
                    row[f"seed{seed}_{key}"] = metrics[key]
                for key in ["val_loss", "best_epoch", "last_epoch", "optimizer_steps", "train_time_s"]:
                    row[f"seed{seed}_{key}"] = old[key]
                row[f"seed{seed}_checkpoint"] = scientific["checkpoint"]
                row[f"seed{seed}_checkpoint_sha256"] = scientific["checkpoint_sha256"]
                row[f"seed{seed}_negative_proposal_draws"] = metrics["negative_all6_proposals"]
                for fs, mae6, mae3 in metrics["fast_seed_results"]:
                    row[f"seed{seed}_fast{fs}_mae_ref_all6"] = mae6
                    row[f"seed{seed}_fast{fs}_mae_ref_active"] = mae3
            axis_rows.append(row)
    common = [r for r in axis_rows if r["config"] == "N50000_L8"]
    assert len(common) == 2 and {k:v for k,v in common[0].items() if k != "axis"} == {k:v for k,v in common[1].items() if k != "axis"}
    conclusions = {}
    for metric in METRICS:
        old_data = [groups[c][metric]["three_chain"]["mean"] for c in DATA]
        new_data = [groups[c][metric]["four_chain"]["mean"] for c in DATA]
        old_depth = [groups[c][metric]["three_chain"]["mean"] for c in DEPTH]
        new_depth = [groups[c][metric]["four_chain"]["mean"] for c in DEPTH]
        neighbor = []
        for a, b in zip(DEPTH[:-1], DEPTH[1:]):
            ga, gb = groups[a][metric]["four_chain"], groups[b][metric]["four_chain"]
            delta = abs(ga["mean"]-gb["mean"])
            scale = max(ga["sample_sd"], gb["sample_sd"])
            neighbor.append(dict(configuration_a=a, configuration_b=b, absolute_mean_difference=delta,
                larger_observed_seed_sd=scale, difference_over_larger_seed_sd=delta/scale,
                observed_seed_ranges_overlap=max(ga["min"], gb["min"]) <= min(ga["max"], gb["max"])))
        conclusions[metric] = dict(data_size_3chain_means=old_data, data_size_4chain_means=new_data,
            data_size_monotone_decrease_3chain=bool(np.all(np.diff(old_data) < 0)),
            data_size_monotone_decrease_4chain=bool(np.all(np.diff(new_data) < 0)),
            data_size_endpoint_decrease_3chain=old_data[-1] < old_data[0], data_size_endpoint_decrease_4chain=new_data[-1] < new_data[0],
            depth_3chain_means=old_depth, depth_4chain_means=new_depth,
            depth_neighbor_mean_difference_signs_changed=not np.array_equal(np.sign(np.diff(old_depth)), np.sign(np.diff(new_depth))),
            depth_neighbor_variation=neighbor)
    data_changed = any(c["data_size_monotone_decrease_3chain"] != c["data_size_monotone_decrease_4chain"] or c["data_size_endpoint_decrease_3chain"] != c["data_size_endpoint_decrease_4chain"] for c in conclusions.values())
    depth_changed = any(c["depth_neighbor_mean_difference_signs_changed"] for c in conclusions.values())
    summary = dict(checkpoints_evaluated=24, flow_training_calls=0, train_function_compiled=False,
        training_seed_SD_primary=True, fast_seed_SD_reported_separately=True, checkpoints_unchanged=True,
        shared_N50000_L8_results_identical_between_axes=True,
        N50000_L8_seed202_recovery_used=True, original_FAILED_attempt_preserved=True,
        r2_3_data_size_observed_trend_changed=data_changed, r2_3_depth_descriptive_pattern_changed=depth_changed,
        r2_3_main_qualitative_conclusion_changed=data_changed or depth_changed,
        per_metric_descriptive_comparison=conclusions,
        no_formal_statistical_test=True, no_configuration_ranking=True,
        negative_fast_draws_retained=sum(r["evaluation_4chain"]["negative_all6_proposals"] for r in raw), proposals_filtered=0)
    write_csv(CSV, axis_rows)
    lines = [f"R2.3 main qualitative conclusion changed: {summary['r2_3_main_qualitative_conclusion_changed']}.",
        f"Data-size observed trend changed: {data_changed}; depth descriptive pattern changed: {depth_changed}.",
        "24/24 frozen scientific checkpoints evaluated without compiling or calling train(); state tensors are finite and strict load passes. The original seeds42..46, five fixed observations and Nfast=2000 reproduce the previously audited proposal diagnostics exactly. Only the reference means change, in memory.",
        "Data-size configurations are shown as 5k/10k/25k/50k/100k at eight layers; depth configurations as 4/6/8/12 at 50k. Configurations are in their predefined axis order. The shared 50k+8-layer three-seed result is evaluated once and is identical in both axis rows.",
        "The 50k+8-layer seed202 recovery checkpoint is used. Its initial telemetry I/O failure, FAILED row, epoch log and audit remain frozen. N5000_L8 seed202 is retained as the previously verified VALID_HIGH_VARIANCE_RUN; no seed is discarded, replaced or selected to match old numbers.",
        "Each row contains training seeds101/202/303 raw MAEs, mean, sample SD (ddof=1), min/max, all five per-fast-seed MAEs and separate fast-seed SD. Original validation loss, best/last epoch, optimizer steps, training wall time, checkpoint paths and SHA256 values are copied unchanged.",
        "Reference means accumulate four-chain draws in float64, then are cast to float32 to retain the frozen evaluate() arithmetic. This retains model/proposal evaluation precision; reference quantiles and the reference summary remain float64.",
        "For each configuration, the three-to-four-chain mean and training-seed SD changes are reported numerically. The observed means and adjacent depth differences are descriptive under three tested training seeds. Neighbor range overlaps and mean-difference/seed-SD ratios are reported without a formal statistical test; the three-seed results do not support fine-grained ranking. The large 5k seed SD is not used to equate 5k and 100k.",
        "All negative proposal draws are retained under the frozen fast-mode rules; no NaN/Inf proposals occur. Reference/flow weights, validation/checkpoint selection, training time, training seeds and all prior audits remain unchanged. No prior/noise/routing threshold change, manuscript/response edit or Git mutation occurs.", "",
        "Per-configuration reference-induced changes (all6 / active e1-e3):", ""]
    for config, group in groups.items():
        descriptions = []
        for metric in METRICS:
            c = group[metric]
            descriptions.append(f"{metric}: mean {c['three_chain']['mean']:.12g} -> {c['four_chain']['mean']:.12g} (delta {c['mean_change']:.12g}); SD {c['three_chain']['sample_sd']:.12g} -> {c['four_chain']['sample_sd']:.12g} (delta {c['sample_sd_change']:.12g})")
        lines.append(config + ": " + "; ".join(descriptions))
    lines += ["", "Descriptive axis comparison:", json.dumps(conclusions, ensure_ascii=False, indent=2), ""]
    write_bytes(AUDIT, "\n".join(lines).encode("utf-8"))
    result = dict(created_utc=C.utc_now(), summary=summary, scientific_checkpoint_evaluations=raw, groups=groups,
        axis_rows=axis_rows, original_integrity_path=str(old_integrity_path), original_integrity_sha256=C.sha(old_integrity_path),
        reference_means_used_float32=env["refs"].tolist(), reference_mean_dtype="float32",
        frozen_evaluate_source_sha256=C.sha(runner), evaluator_sha256=C.sha(__file__),
        preservation_before=before, preservation_after=verify(snapshot),
        outputs={p.name: dict(path=str(p), sha256=C.sha(p)) for p in [CSV, AUDIT]})
    write_json(RESULT, result)
    print("R2_3_FIXED_CHECKPOINT_EVALUATION_FINISHED", json.dumps(summary), flush=True)


def verify(snapshot):
    changed = [p for p, digest in snapshot["files"].items() if not Path(p).is_file() or C.sha(p) != digest]
    assert not changed, changed
    assert C.git_read("status", "--porcelain", "-uno") == snapshot["git_tracked_before"]
    return dict(checked_files=len(snapshot["files"]), all_sha256_unchanged=True, git_tracked_unchanged=True)


def write_bytes(path, content):
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        import os
        os.fsync(stream.fileno())


def write_json(path, value):
    write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def write_csv(path, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
    writer.writeheader()
    writer.writerows(rows)
    write_bytes(path, stream.getvalue().encode("utf-8"))


if __name__ == "__main__":
    main()
