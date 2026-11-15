"""Export the already-completed MAP timing runs from their immutable raw JSON.

This utility performs no optimization, surrogate evaluation, or timing.  It exists
only because the original benchmark completed all runs and wrote the raw JSON,
then hit a CSV field-list error while exporting its summary rows.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RAW_JSON = HERE / "MAP_COMPLETION_TIMING_RAW.json"
TIMING_CSV = HERE / "MAP_COMPLETION_TIMING.csv"


def main() -> None:
    if TIMING_CSV.exists():
        raise RuntimeError(f"Refusing to overwrite {TIMING_CSV}")
    payload = json.loads(RAW_JSON.read_text(encoding="utf-8"))
    fields = [
        "case",
        "row_type",
        "run_index",
        "wall_time_s",
        "theta_map",
        "max_relative_difference_from_retained",
        "internal_optimization_gate",
        "n_warmup",
        "n_timed",
        "mean_s",
        "median_s",
        "min_s",
        "max_s",
        "q1_s",
        "q3_s",
        "iqr_s",
        "max_relative_difference_across_all_runs",
        "gate",
    ]
    with TIMING_CSV.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in payload["cases"]:
            for run in case["runs"]:
                passed = (
                    run["max_relative_difference_from_retained"] <= 1e-6
                    and run["internal_optimization_gate"]
                )
                writer.writerow(
                    {
                        "case": case["case"],
                        "row_type": run["run_type"],
                        "run_index": run["run_index"],
                        "wall_time_s": repr(float(run["wall_time_s"])),
                        "theta_map": json.dumps(run["theta_map"], separators=(",", ":")),
                        "max_relative_difference_from_retained": repr(
                            float(run["max_relative_difference_from_retained"])
                        ),
                        "internal_optimization_gate": run["internal_optimization_gate"],
                        "gate": "PASS" if passed else "FAIL",
                    }
                )
            writer.writerow(
                {
                    "case": case["case"],
                    "row_type": "summary",
                    **case["summary"],
                }
            )


if __name__ == "__main__":
    main()
