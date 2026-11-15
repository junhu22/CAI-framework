import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(r"C:\neutra-hmc-surrogate\surrogate_training")
OUT = ROOT / "revision_experiments/cai_r2_provenance_closure_20260920_083412"
REFERENCE = ROOT / "revision_experiments/cai_r2_presubmit_phaseB_20260919_213647/COUPLED_UNIFORM/COUPLED_UNIFORM_NUTS_REFERENCE_2000.npz"
EXPECTED_REFERENCE_SHA256 = "c3174b9f49aed3cf0cd2ef68afddd7290ad10f47f5c20744ea60491cc1819d60"
CHAIN_ORDER = [811, 812, 813, 814]
COUNTS = [13, 13, 12, 12]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    indices_path = OUT / "CASE_C_SELECTED_INDICES.json"
    theta_path = OUT / "CASE_C_SELECTED_THETA.csv"
    hashes_path = OUT / "CASE_C_SELECTION_HASHES.json"
    for path in (indices_path, theta_path, hashes_path):
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite immutable selection file: {path}")

    reference_hash = sha256(REFERENCE)
    if reference_hash != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("Qualified reference hash mismatch")

    with np.load(REFERENCE, allow_pickle=False) as archive:
        chains = np.asarray(archive["chains"])
        seeds = np.asarray(archive["seeds"], dtype=np.int64)
        retained = int(np.asarray(archive["retained_per_chain"]).item())
    if chains.shape != (4, 2000, 3) or retained != 2000:
        raise RuntimeError(f"Unexpected reference shape/retained count: {chains.shape}, {retained}")
    if seeds.tolist() != CHAIN_ORDER:
        raise RuntimeError(f"Unexpected chain order: {seeds.tolist()}")

    generator = np.random.Generator(np.random.PCG64(20260920))
    selections = []
    rows = []
    selection_order = 0
    for chain_position, (seed, count) in enumerate(zip(CHAIN_ORDER, COUNTS)):
        indices = generator.choice(2000, size=count, replace=False)
        selections.append({
            "chain_position": chain_position,
            "chain_seed": seed,
            "count": count,
            "retained_indices_zero_based_generator_order": [int(value) for value in indices],
        })
        for retained_index in indices:
            theta = chains[chain_position, int(retained_index)]
            rows.append({
                "selection_order": selection_order,
                "chain_position": chain_position,
                "chain_seed": seed,
                "retained_index_zero_based": int(retained_index),
                "e1": repr(float(theta[0])),
                "e2": repr(float(theta[1])),
                "e3": repr(float(theta[2])),
            })
            selection_order += 1

    payload = {
        "schema": "bounded-uniform-case-c-selection-v1",
        "reference_path": str(REFERENCE),
        "reference_sha256": reference_hash,
        "reference_shape": list(chains.shape),
        "chain_order": CHAIN_ORDER,
        "counts": COUNTS,
        "total": len(rows),
        "generator": "numpy.random.Generator(numpy.random.PCG64(20260920))",
        "replacement": False,
        "index_base": 0,
        "within_chain_order": "generator-return order; not sorted",
        "selections": selections,
    }
    indices_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with theta_path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    hashes = {
        "schema": "bounded-uniform-case-c-selection-hashes-v1",
        "hash_algorithm": "SHA-256",
        "immutable_after_creation": True,
        "files": [
            {"path": indices_path.name, "bytes": indices_path.stat().st_size, "sha256": sha256(indices_path)},
            {"path": theta_path.name, "bytes": theta_path.stat().st_size, "sha256": sha256(theta_path)},
        ],
    }
    hashes_path.write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
