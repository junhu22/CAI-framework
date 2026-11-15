#!/usr/bin/env python3
"""Step 1: Generate LHS samples for D=6 sailboat building."""
import numpy as np, argparse
from config import *

def latin_hypercube(n, d, seed=42):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, d))
    for j in range(d):
        p = rng.permutation(n)
        out[:, j] = (p + rng.uniform(size=n)) / n
    return THETA_LO + out * (THETA_HI - THETA_LO)

def add_anchors(samples):
    d = samples.shape[1]
    pts = [np.ones(d)]  # nominal
    pts.append(np.full(d, THETA_LO))
    pts.append(np.full(d, THETA_HI))
    for j in range(d):
        lo = np.ones(d); lo[j] = THETA_LO
        hi = np.ones(d); hi[j] = THETA_HI
        pts.extend([lo, hi])
    return np.vstack([np.array(pts), samples])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=8000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train = add_anchors(latin_hypercube(args.n, N_PARAMS, args.seed))
    test = latin_hypercube(args.n_test, N_PARAMS, args.seed + 1000)

    hdr = ",".join(PARAM_NAMES)
    np.savetxt("samples_train.csv", train, delimiter=",", header=hdr, comments="")
    np.savetxt("samples_test.csv", test, delimiter=",", header=hdr, comments="")
    print(f"Train: {train.shape[0]} samples  |  Test: {test.shape[0]} samples")
    print(f"Total FE runs: {train.shape[0] + test.shape[0]}")
    print(f"Est. time (2 min/run, 8‖): {(train.shape[0]+test.shape[0])*2/60/8:.0f} h")
