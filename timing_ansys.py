"""
Step A: Accurate timing of ANSYS modal analysis calls.
Runs N_TIMING_CALLS evaluations at different parameter values,
records per-call wall-clock time, and reports statistics.

Usage:
    python -u timing_ansys.py "C:\Program Files\ANSYS Inc\v160\ANSYS\bin\winx64\ANSYS160.exe"

Output: timing_results.json with per-call times and summary statistics.
"""
import sys, time, json, csv, statistics
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────
N_TIMING_CALLS = 200          # enough for reliable statistics
SAMPLES_CSV    = "samples_train.csv"
MAC_FILE       = "update.mac"
# ────────────────────────────────────────────────────────

def main():
    exe = sys.argv[1] if len(sys.argv) > 1 else \
        r"C:\Program Files\ANSYS Inc\v160\ANSYS\bin\winx64\ANSYS160.exe"

    # ── import batch runner ──
    from importlib import import_module
    batch = import_module("02_run_batch")
    from config import PARAM_NAMES

    # ── load samples ──
    samples = []
    with open(SAMPLES_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            samples.append([float(row[p]) for p in PARAM_NAMES])
    # use first N_TIMING_CALLS samples (or cycle if needed)
    if len(samples) < N_TIMING_CALLS:
        import itertools
        samples = list(itertools.islice(itertools.cycle(samples), N_TIMING_CALLS))
    else:
        samples = samples[:N_TIMING_CALLS]

    # ── prepare APDL template and geometry ──
    geo = batch.extract_geometry_body(Path(MAC_FILE))
    tmpl = Path("apdl_template/run_sample_template.mac").read_text(encoding="utf-8")

    print(f"Timing {N_TIMING_CALLS} ANSYS calls...", flush=True)
    print(f"ANSYS exe: {exe}", flush=True)

    times = []
    freqs_all = []
    n_ok = 0
    n_fail = 0

    for i, params in enumerate(samples):
        import subprocess, shutil

        wd = Path(f"timing_run_{i:04d}")
        wd.mkdir(exist_ok=True)

        apdl = batch.build_apdl(tmpl, geo, params)
        (wd / "run.mac").write_text(apdl, encoding="utf-8")

        t0 = time.perf_counter()
        result = subprocess.run(
            [exe, "-b", "-i", "run.mac", "-o", "ans.log", "-j", "ac3", "-np", "1"],
            cwd=str(wd.resolve()), timeout=600, capture_output=True, text=True
        )
        elapsed = time.perf_counter() - t0

        ff = wd / "modal_freqs.txt"
        if ff.exists() and result.returncode == 0:
            freqs = [float(x.strip()) for x in ff.read_text().strip().split("\n") if x.strip()]
            freqs_all.append(freqs[:6])
            times.append(elapsed)
            n_ok += 1
        else:
            n_fail += 1

        # clean up to save disk
        shutil.rmtree(wd, ignore_errors=True)

        if (i + 1) % 20 == 0 or i == 0:
            avg_so_far = statistics.mean(times) if times else 0
            print(f"  {i+1}/{N_TIMING_CALLS}  ok={n_ok} fail={n_fail}  "
                  f"avg={avg_so_far:.3f}s  last={elapsed:.3f}s", flush=True)

    # ── statistics ──
    if not times:
        print("ERROR: no successful calls!", flush=True)
        return

    result = {
        "n_calls": N_TIMING_CALLS,
        "n_ok": n_ok,
        "n_fail": n_fail,
        "times_seconds": times,
        "mean_s": statistics.mean(times),
        "median_s": statistics.median(times),
        "stdev_s": statistics.stdev(times) if len(times) > 1 else 0,
        "min_s": min(times),
        "max_s": max(times),
        "p05_s": sorted(times)[int(0.05 * len(times))],
        "p95_s": sorted(times)[int(0.95 * len(times))],
    }

    print("\n" + "=" * 50, flush=True)
    print(f"ANSYS Timing Results ({n_ok} successful calls)", flush=True)
    print(f"  Mean:   {result['mean_s']:.3f} s", flush=True)
    print(f"  Median: {result['median_s']:.3f} s", flush=True)
    print(f"  Stdev:  {result['stdev_s']:.3f} s", flush=True)
    print(f"  P05:    {result['p05_s']:.3f} s", flush=True)
    print(f"  P95:    {result['p95_s']:.3f} s", flush=True)
    print(f"  Min:    {result['min_s']:.3f} s", flush=True)
    print(f"  Max:    {result['max_s']:.3f} s", flush=True)
    print("=" * 50, flush=True)

    # ── projected wall-clock comparison ──
    t_eval = result["mean_s"]
    print("\nProjected wall-clock comparison (serial, single core):", flush=True)
    methods = [
        ("Raw RWMH",           12000),
        ("Cov-precond. RWMH",  12000),
        ("Transport RWMH",     12000),
        ("Independent MH",     12000),
        ("TMCMC (3k particles)", 105000),
    ]
    # ESS values from surrogate experiments (Table 5 in paper)
    ess_values = {
        "Raw RWMH": 87,
        "Cov-precond. RWMH": 245,
        "Transport RWMH": 384,       # best training size
        "Independent MH": 3148,       # best training size
        "TMCMC (3k particles)": 1804,
    }

    print(f"\n{'Method':<25} {'Evals':>8} {'Wall-clock':>12} {'ESS':>6} {'ESS/hour':>10}", flush=True)
    print("-" * 65, flush=True)
    for name, evals in methods:
        wc = evals * t_eval
        ess = ess_values[name]
        ess_h = ess / (wc / 3600)
        hrs = wc / 3600
        print(f"{name:<25} {evals:>8} {hrs:>10.1f} h {ess:>6} {ess_h:>10.1f}", flush=True)

    # ── save ──
    with open("timing_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to timing_results.json", flush=True)


if __name__ == "__main__":
    main()
