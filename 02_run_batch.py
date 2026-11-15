#!/usr/bin/env python3
"""Step 2: Run ANSYS batch — injects params as *SET, extracts geometry from update.mac."""
import subprocess, csv, time, argparse, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from config import *

GEOMETRY_MARKER = "! >>> GEOMETRY_BODY_PLACEHOLDER <<<"
PARAMS_MARKER = "! >>> PARAMS_PLACEHOLDER <<<"

# Mapping: PARAM_NAMES[i] -> APDL variable name
APDL_VAR = ["e1", "e2", "e3", "e4", "e5", "e6"]
E_BASE = 2e10  # Nominal elastic modulus


def extract_geometry_body(mac: Path) -> str:
    lines = mac.read_text(encoding="utf-8", errors="replace").splitlines()
    start = end = None
    found_loop = False
    for i, L in enumerate(lines):
        s = L.strip().lower()
        if "*do,ii,0,7" in s and not found_loop:
            found_loop = True
        if found_loop and s == "*enddo" and start is None:
            start = i + 1; break
    for i, L in enumerate(lines):
        if L.strip().upper().startswith("/SOL"):
            end = i; break
    if start is None or end is None:
        raise RuntimeError(f"Cannot locate geometry in {mac}")
    return "\n".join(lines[start:end])


def make_param_block(params):
    """Generate *SET commands for the 6 parameters."""
    lines = []
    for j, val in enumerate(params):
        # e.g. *SET,e1,1.5000000000e+10
        e_val = val * E_BASE
        lines.append(f"*SET,{APDL_VAR[j]},{e_val:.10e}")
    return "\n".join(lines)


def build_apdl(template, geometry, params):
    """Build complete APDL script with injected params and geometry."""
    apdl = template.replace(PARAMS_MARKER, make_param_block(params))
    apdl = apdl.replace(GEOMETRY_MARKER, geometry)
    return apdl


def run_one(sid, params, template, geometry, base, exe, timeout=600):
    wd = base / f"s{sid:06d}"
    wd.mkdir(parents=True, exist_ok=True)
    apdl = build_apdl(template, geometry, params)
    (wd / "run.mac").write_text(apdl, encoding="utf-8")
    t0 = time.time()
    try:
        subprocess.run(
            [exe, "-b", "-i", "run.mac", "-o", "ans.log",
             "-j", "ac3", "-np", "1"],
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
        print(f"  [FAIL] s{sid}: no freqs (check {wd/'ans.log'})", flush=True)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] s{sid}", flush=True)
    except Exception as e:
        print(f"  [ERR] s{sid}: {e}", flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--mac", default="update.mac")
    ap.add_argument("--template", default="apdl_template/run_sample_template.mac")
    ap.add_argument("--ansys", default="ansys242")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--output", default=None)
    ap.add_argument("--work_dir", default="batch_runs")
    ap.add_argument("--max", type=int, default=None)
    args = ap.parse_args()

    mac, tmpl_path = Path(args.mac), Path(args.template)
    if not mac.exists():
        print(f"ERROR: {mac} not found."); sys.exit(1)
    if not tmpl_path.exists():
        print(f"ERROR: {tmpl_path} not found."); sys.exit(1)

    samples = []
    with open(args.samples, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            samples.append([float(row[p]) for p in PARAM_NAMES])
    if args.max:
        samples = samples[:args.max]
    print(f"Loaded {len(samples)} samples", flush=True)

    geometry = extract_geometry_body(mac)
    template = tmpl_path.read_text(encoding="utf-8")
    # Quick check
    test_apdl = build_apdl(template, geometry, [1.0]*N_PARAMS)
    print(f"Full APDL: {len(test_apdl.splitlines())} lines", flush=True)

    base = Path(args.work_dir); base.mkdir(exist_ok=True)

    t0 = time.time()
    results = []
    if args.parallel <= 1:
        for i, p in enumerate(samples):
            r = run_one(i, p, template, geometry, base, args.ansys, args.timeout)
            if r: results.append(r)
            if (i+1) % 100 == 0 or len(samples) <= 20:
                print(f"  {i+1}/{len(samples)} done, {len(results)} ok", flush=True)
    else:
        with ProcessPoolExecutor(args.parallel) as ex:
            futs = {ex.submit(run_one, i, p, template, geometry, base, args.ansys, args.timeout): i
                    for i, p in enumerate(samples)}
            done = 0
            for fut in as_completed(futs):
                done += 1
                r = fut.result()
                if r: results.append(r)
                if done % 50 == 0 or len(samples) <= 20:
                    print(f"  {done}/{len(samples)} done, {len(results)} ok", flush=True)

    dt = time.time() - t0
    out = Path(args.output or args.samples.replace("samples_", "dataset_"))
    results.sort(key=lambda r: r["id"])
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id"] + PARAM_NAMES + FREQ_NAMES + ["time_s"])
        w.writeheader(); w.writerows(results)
    print(f"\nDone: {len(results)}/{len(samples)} ok in {dt/3600:.1f}h -> {out}", flush=True)


if __name__ == "__main__":
    main()
