import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
OUT = ROOT / "revision_experiments/cai_r2_presubmit_phaseB_20260919_213647/COUPLED_UNIFORM"
SOURCE = ROOT / "experiments/coupled_building/04_train_cnf_coupled.py"
SURROGATE_PATH = ROOT / "surrogate_models_coupled/member_0.pt"
SEED_MAP = {
    42: ("primary", "conditional_flow_coupled_uniform_primary.pt"),
    202: ("secondary", "conditional_flow_coupled_uniform_secondary_seed202.pt"),
    303: ("secondary", "conditional_flow_coupled_uniform_secondary_seed303.pt"),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_source():
    sys.path.insert(0, str(ROOT / "experiments/coupled_building"))
    spec = importlib.util.spec_from_file_location("old_coupled_cnf_source", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_uniform(module, surrogate, seed):
    torch.manual_seed(seed)
    total = 55000
    all_theta, all_y = [], []
    with torch.no_grad():
        for start in range(0, total, 5000):
            batch = min(5000, total - start)
            theta = 0.5 + 3.5 * torch.rand((batch, 3), device=module.DEV)
            clean = surrogate(theta)
            y = clean + 0.03 * clean * torch.randn_like(clean)
            all_theta.append(theta)
            all_y.append(y)
    theta = torch.cat(all_theta)
    y = torch.cat(all_y)
    return theta[:50000], y[:50000], theta[50000:], y[50000:]


def train(module, flow, theta_tr, y_tr, theta_val, y_val):
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 500)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(theta_tr, y_tr),
        batch_size=512, shuffle=True, drop_last=True,
    )
    best = float("inf")
    best_state = None
    no_improvement = 0
    history = []
    t0 = time.time()
    stop_epoch = 500
    for epoch in range(1, 501):
        flow.train()
        total_loss, n = 0.0, 0
        for theta_b, y_b in loader:
            z, log_det = flow(theta_b, y_b)
            loss = -(-0.5 * (z ** 2).sum(dim=-1) + log_det).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            opt.step()
            total_loss += loss.item() * len(theta_b)
            n += len(theta_b)
        scheduler.step()
        train_loss = total_loss / n
        flow.eval()
        with torch.no_grad():
            z, log_det = flow(theta_val, y_val)
            val_loss = float(-(-0.5 * (z ** 2).sum(dim=-1) + log_det).mean())
        improved = val_loss < best
        if improved:
            best = val_loss
            best_state = {k: v.detach().clone() for k, v in flow.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "best_validation_loss": best,
            "learning_rate": scheduler.get_last_lr()[0],
            "improved": improved,
        })
        if epoch == 1 or epoch % 25 == 0:
            print(f"seed training epoch={epoch} train={train_loss:.6f} val={val_loss:.6f} best={best:.6f}", flush=True)
        if no_improvement >= 50:
            stop_epoch = epoch
            break
    flow.load_state_dict(best_state)
    flow.eval()
    return flow, best, stop_epoch, time.time() - t0, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=sorted(SEED_MAP))
    args = parser.parse_args()
    seed = args.seed
    role, checkpoint_name = SEED_MAP[seed]
    checkpoint = OUT / checkpoint_name
    result_path = OUT / f"COUPLED_UNIFORM_CNF_TRAINING_seed{seed}.json"
    history_path = OUT / f"COUPLED_UNIFORM_CNF_TRAINING_seed{seed}.csv"
    if checkpoint.exists() or result_path.exists() or history_path.exists():
        raise RuntimeError(f"Refusing to overwrite seed {seed}")
    module = load_source()
    assert module.D == 3 and module.M == 20
    assert sha(SURROGATE_PATH).lower() == "6a1e4f530d4e3628e0b7fdbf08661b4dc32a71a0c2b538b26f5af9d7c07b9986"
    surrogate = module.load_surrogate()
    theta_tr, y_tr, theta_val, y_val = generate_uniform(module, surrogate, seed)
    theta_all = torch.cat([theta_tr, theta_val])
    assert bool(((theta_all >= 0.5) & (theta_all < 4.0)).all())
    # The RNG state intentionally continues from data generation, as in the old script.
    flow = module.ConditionalRealNVP(dim_theta=3, dim_y=20, n_layers=8, hidden_dim=128).to(module.DEV)
    parameter_count = sum(p.numel() for p in flow.parameters())
    flow, best, stop_epoch, runtime, history = train(module, flow, theta_tr, y_tr, theta_val, y_val)
    with torch.no_grad():
        z, _ = flow(theta_val[:200], y_val[:200])
        back, _ = flow.inverse(z, y_val[:200])
        recon_error = float((back - theta_val[:200]).abs().max())
    torch.save(flow.state_dict(), checkpoint)
    with history_path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    result = {
        "seed": seed,
        "role": role,
        "selection_rule": "seed 42 remains primary; no seed selection",
        "theta_distribution": "independent Uniform(0.5,4.0)^3",
        "theta_min_per_coordinate": theta_all.min(0).values.cpu().tolist(),
        "theta_max_per_coordinate": theta_all.max(0).values.cpu().tolist(),
        "theta_mean_per_coordinate": theta_all.mean(0).cpu().tolist(),
        "training_pairs": 50000,
        "validation_pairs": 5000,
        "noise": "multiplicative independent Gaussian, kappa=0.03",
        "architecture": {"coupling_layers": 8, "hidden": 128, "y_embed": [20, 64, 20], "parameters": parameter_count},
        "optimizer": {"name": "Adam", "lr": 0.001, "weight_decay": 1e-5, "batch_size": 512, "gradient_clip": 5.0, "scheduler": "CosineAnnealingLR(500)", "patience": 50, "max_epochs": 500},
        "normalization": "none for direct theta/y flow inputs (unchanged from old script)",
        "best_validation_nll": best,
        "stop_epoch": stop_epoch,
        "runtime_s": runtime,
        "max_roundtrip_reconstruction_error": recon_error,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "history": str(history_path),
        "surrogate": str(SURROGATE_PATH),
        "surrogate_sha256": sha(SURROGATE_PATH),
        "source_old_training_script": str(SOURCE),
        "source_old_training_script_sha256": sha(SOURCE),
        "new_fe_calls": 0,
        "surrogate_training_calls": 0,
    }
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("COUPLED_UNIFORM_TRAINING_COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

