#!/usr/bin/env python3
"""Run ANSYS batch for the coupled (cic) building surrogate.

Injects *SET commands for e1,e2,e3 (each as scaling*1e10) into a self-contained
template, runs ANSYS in batch mode per sample, parses modal_freqs.txt.

Run from the surrogate_training/ directory:
  python experiments/coupled_building/02_run_batch_coupled.py \
      --samples samples_coupled_pilot.csv \
      --ansys "D:\\Program Files\\ANSYS Inc\\v160\\ansys\\bin\\winx64\\ANSYS160.exe" \
      --parallel 1 --work_dir batch_runs_coupled_pilot
"""
import sys, csv, time, argparse, subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Make sibling config_coupled.py importable regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config_coupled import (
    N_PARAMS, N_MODES, PARAM_NAMES, APDL_VAR, E_BASE, FREQ_NAMES,
)

PARAMS_MARKER = "! >>> PARAMS_PLACEHOLDER <<<"


def make_param_block(params):
    """Generate *SET commands. params are scaling factors; multiply by E_BASE."""
    lines = []
    for j, val in enumerate(params):
        e_val = val * E_BASE
        lines.append(f"*SET,{APDL_VAR[j]},{e_val:.10e}")
    return "\n".join(lines)


def build_apdl(template, params):
    return template.replace(PARAMS_MARKER, make_param_block(params))


def run_one(sid, params, template, base, exe, timeout):
    wd = base / f"s{sid:06d}"
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "run.mac").write_text(build_apdl(template, params), encoding="utf-8")
    t0 = time.time()
    try:
        subprocess.run(
            [exe, "-b", "-i", "run.mac", "-o", "ans.log",
             "-j", "coupled", "-np", "1"],
            cwd=str(wd.resolve()), timeout=timeout, capture_output=True)
        dt = time.time() - t0
        ff = wd / "modal_freqs.txt"
        if ff.exists():
            freqs = [float(x) for x in ff.read_text(encoding="utf-8").split() if x.strip()]
            if len(freqs) >= N_MODES:
                return {"id": sid,
                        **{PARAM_NAMES[j]: params[j] for j in range(N_PARAMS)},
                        **{FREQ_NAMES[j]: freqs[j] for j in range(N_MODES)},
                        "time_s": dt}
        print(f"  [FAIL] s{sid:06d}: no freqs (check {wd/'ans.log'})", flush=True)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] s{sid:06d}", flush=True)
    except Exception as e:
        print(f"  [ERR] s{sid:06d}: {e}", flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True,
                    help="CSV with columns matching PARAM_NAMES")
    ap.add_argument("--template",
                    default="apdl_template/run_coupled_template.mac")
    ap.add_argument("--ansys", required=True,
                    help="Full path to ANSYS executable (e.g. ANSYS160.exe)")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--output", default=None)
    ap.add_argument("--work_dir", default="batch_runs_coupled")
    ap.add_argument("--max", type=int, default=None)
    args = ap.parse_args()

    tmpl_path = Path(args.template)
    if not tmpl_path.exists():
        print(f"ERROR: template not found: {tmpl_path}"); sys.exit(1)
    template = tmpl_path.read_text(encoding="utf-8")
    if PARAMS_MARKER not in template:
        print(f"ERROR: template missing PARAMS_PLACEHOLDER"); sys.exit(1)

    samples = []
    with open(args.samples, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            samples.append([float(row[p]) for p in PARAM_NAMES])
    if args.max:
        samples = samples[:args.max]
    print(f"Loaded {len(samples)} samples ({N_PARAMS} params each)", flush=True)

    test_apdl = build_apdl(template, [1.0] * N_PARAMS)
    print(f"Full APDL: {len(test_apdl.splitlines())} lines", flush=True)

    base = Path(args.work_dir); base.mkdir(exist_ok=True)

    t0 = time.time()
    results = []
    if args.parallel <= 1:
        for i, p in enumerate(samples):
            r = run_one(i, p, template, base, args.ansys, args.timeout)
            if r: results.append(r)
            if (i + 1) % 50 == 0 or len(samples) <= 20:
                print(f"  {i+1}/{len(samples)} done, {len(results)} ok", flush=True)
    else:
        with ProcessPoolExecutor(args.parallel) as ex:
            futs = {ex.submit(run_one, i, p, template, base, args.ansys,
                              args.timeout): i for i, p in enumerate(samples)}
            done = 0
            for fut in as_completed(futs):
                done += 1
                r = fut.result()
                if r: results.append(r)
                if done % 50 == 0 or len(samples) <= 20:
                    print(f"  {done}/{len(samples)} done, {len(results)} ok",
                          flush=True)

    dt = time.time() - t0
    out = Path(args.output or args.samples.replace("samples_", "dataset_"))
    results.sort(key=lambda r: r["id"])
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id"] + PARAM_NAMES + FREQ_NAMES
                                          + ["time_s"])
        w.writeheader(); w.writerows(results)
    print(f"\nDone: {len(results)}/{len(samples)} ok in {dt/3600:.2f}h -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
