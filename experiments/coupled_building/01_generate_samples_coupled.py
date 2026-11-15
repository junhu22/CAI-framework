#!/usr/bin/env python3
"""Generate LHS training/test samples for the coupled (cic) building, D=3.

Outputs:
  samples_coupled_train.csv  (5000 rows)
  samples_coupled_test.csv   (500 rows)
"""
import sys, csv, argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_coupled import PARAM_NAMES, THETA_LO, THETA_HI, N_PARAMS


def lhs(n, d, rng):
    """Latin Hypercube sample on [0,1]^d, n points."""
    u = (rng.permuted(np.tile(np.arange(n), (d, 1)), axis=1)
         + rng.random((d, n))) / n
    return u.T


def to_csv(path, samples):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id"] + PARAM_NAMES)
        for i, s in enumerate(samples):
            w.writerow([i] + [f"{v:.10f}" for v in s])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=5000)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20240516)
    ap.add_argument("--out_dir", default=".",
                    help="Directory for output CSVs (default: cwd)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    train = THETA_LO + (THETA_HI - THETA_LO) * lhs(args.n_train, N_PARAMS, rng)
    test  = THETA_LO + (THETA_HI - THETA_LO) * lhs(args.n_test,  N_PARAMS, rng)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    train_path = out_dir / "samples_coupled_train.csv"
    test_path  = out_dir / "samples_coupled_test.csv"
    to_csv(train_path, train)
    to_csv(test_path,  test)

    print(f"Wrote {train_path}  ({args.n_train} rows)")
    print(f"Wrote {test_path}   ({args.n_test} rows)")
    print(f"Range per param: [{THETA_LO}, {THETA_HI}]")


if __name__ == "__main__":
    main()
