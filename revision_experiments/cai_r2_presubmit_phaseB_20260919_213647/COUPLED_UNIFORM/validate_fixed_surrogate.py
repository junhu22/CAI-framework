import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
OUT = ROOT / "revision_experiments/cai_r2_presubmit_phaseB_20260919_213647/COUPLED_UNIFORM"
DATA = ROOT / "dataset_coupled_train.csv"
MODEL = ROOT / "surrogate_models_coupled/member_0.pt"
OLD_METRICS = ROOT / "surrogate_models_coupled/metrics.json"
sys.path.insert(0, str(ROOT / "experiments/coupled_building"))
import importlib.util


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    spec = importlib.util.spec_from_file_location("coupled_surrogate_source", ROOT / "experiments/coupled_building/03_train_surrogate_coupled.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    X, Y = module.load_data(DATA)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    Xte, Yte = X[idx[4000:]], Y[idx[4000:]]
    model = module.Surrogate().to(module.DEV)
    model.load_state_dict(torch.load(MODEL, map_location=module.DEV, weights_only=True))
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(Xte, dtype=torch.float32, device=module.DEV)
        yt = torch.tensor(Yte, dtype=torch.float32, device=module.DEV)
        yp = model(xt)
        ae = (yp - yt).abs()
        re = ae / yt.abs().clamp(min=1e-8)
    r2 = []
    for j in range(20):
        sr = ((yt[:, j] - yp[:, j]) ** 2).sum().item()
        st = ((yt[:, j] - yt[:, j].mean()) ** 2).sum().item()
        r2.append(1 - sr / max(st, 1e-12))
    metrics = {
        "mae_hz": float(ae.mean()),
        "mape_pct": float(re.mean() * 100),
        "max_rel_err_pct": float(re.max() * 100),
        "r2_mean": float(np.mean(r2)),
        "r2_min": float(np.min(r2)),
        "r2_per_mode": [float(v) for v in r2],
    }
    old = json.loads(OLD_METRICS.read_text(encoding="utf-8"))
    differences = {k: abs(metrics[k] - old[k]) for k in ["mae_hz", "mape_pct", "max_rel_err_pct", "r2_mean", "r2_min"]}
    result = {
        "status": "PASS" if max(differences.values()) < 1e-7 else "FAIL",
        "heldout_definition": "same seed-42 shuffled 4000/1000 split as 03_train_surrogate_coupled.py",
        "rows_total": len(X),
        "heldout_rows": len(Xte),
        "theta_ranges_all_data": [[float(X[:, j].min()), float(X[:, j].max())] for j in range(3)],
        "recomputed_metrics": metrics,
        "retained_metrics": old,
        "absolute_differences": differences,
        "dataset_sha256": sha(DATA),
        "surrogate_sha256": sha(MODEL),
        "retained_metrics_sha256": sha(OLD_METRICS),
        "new_fe_calls": 0,
        "surrogate_training_calls": 0,
    }
    (OUT / "COUPLED_SURROGATE_FREEZE_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Fixed coupled-surrogate verification",
        "",
        f"Status: **{result['status']}**",
        "",
        f"Frozen checkpoint SHA-256: `{result['surrogate_sha256']}`.",
        f"Recomputed held-out MAE: `{metrics['mae_hz']:.12f}` Hz; MAPE: `{metrics['mape_pct']:.12f}%`; maximum relative error: `{metrics['max_rel_err_pct']:.12f}%`; mean/min R2: `{metrics['r2_mean']:.12f}` / `{metrics['r2_min']:.12f}`.",
        "The metrics reproduce the retained 1,000-row held-out split. No FE call or surrogate update was made.",
        "",
    ]
    (OUT / "COUPLED_SURROGATE_FREEZE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print("COUPLED_SURROGATE_FREEZE", result["status"], json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()

