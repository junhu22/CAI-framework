"""Recover the realised acceptance rate of the frozen field NUTS chain.

field_freeze.py reads it as mc.diagnostics()["acceptance rate"]["z"], but in Pyro
1.9.1 that key holds a float, not a per-site dict, so the lookup raises and the
script falls back to the target 0.8 -- which is what Table 15 prints.  The chain
is reproducible from its seed, so the realised value can be recovered: this reruns
chain 0 exactly (same init, same seed, same warmup) and reads the counters off the
kernel.

Gated on the chain reproducing the frozen ESS_min 296 and e1 2.22 before its
acceptance rate is reported.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(6)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import revlib as R  # noqa: E402
from pyro.infer import MCMC, NUTS  # noqa: E402

SIG_N = 0.30
WARMUP, NSAMP = 500, 500
sur = R.load_broad_ext()
y6 = R.FIELD_MEASURED
yt = torch.tensor(y6)


def U(p):
    z = p["z"]
    f = sur(torch.exp(z).unsqueeze(0)).squeeze(0)[:6]
    return -(-0.5 * (((f - yt) / yt) ** 2).sum() / 1e-3 - 0.5 * (z ** 2).sum() / SIG_N ** 2)


torch.manual_seed(42)
kernel = NUTS(potential_fn=U, target_accept_prob=0.8)

# MCMC calls kernel.cleanup() when the run ends, which zeroes the counters, so
# snapshot them first.  This reads a diagnostic the frozen script discarded; it
# does not change the chain.
snap = {}
_sample = kernel.sample


def _snap_sample(params):
    out = _sample(params)
    snap.update(mean_accept_prob=kernel._mean_accept_prob,
                accept_cnt=kernel._accept_cnt, t=kernel._t,
                divergences=len(kernel._divergences),
                step_size=float(kernel.step_size))
    return out


kernel.sample = _snap_sample
mc = MCMC(kernel, num_samples=NSAMP, warmup_steps=WARMUP,
          initial_params={"z": torch.log(torch.tensor(R.FIELD_FASTMEAN, dtype=torch.float32))},
          disable_progbar=True)
t0 = time.time()
mc.run()
elapsed = time.time() - t0
th = torch.exp(mc.get_samples()["z"])

ess_min = float(R.effective_sample_size(th).min())
e1 = float(th.numpy().mean(0)[0])
gate = abs(ess_min - 296) / 296 <= 0.10 and abs(e1 - 2.22) <= 0.05
print(f"gate: ESS_min {ess_min:.1f} (frozen 296), e1 {e1:.3f} (frozen 2.22) -> "
      f"{'PASS' if gate else 'FAIL'}")
if not gate:
    sys.exit("chain did not reproduce; not reporting an acceptance rate")

diag = mc.diagnostics()
raw = diag.get("acceptance rate")
post_warmup = snap["t"] - WARMUP
counted = snap["accept_cnt"] / post_warmup if post_warmup else float("nan")

print(f"\nwhat field_freeze tried to read: diagnostics()['acceptance rate'] = {raw!r} "
      f"-- it is keyed by chain, not by site, so ['z'] raises and 0.8 is used")
print(f"  kernel._accept_cnt / post-warmup    = {counted:.4f}   "
      f"({snap['accept_cnt']}/{post_warmup})  <- NUTS always moves to a point on the "
      "trajectory, so this is 1 by construction and is not a Metropolis accept rate")
print(f"  kernel._mean_accept_prob            = {snap['mean_accept_prob']:.4f}   "
      "<- mean Metropolis accept probability over the trajectory: the quantity dual "
      "averaging drives to the target")
print(f"  divergences                         = {snap['divergences']}")
print(f"  step size after adaptation          = {snap['step_size']:.5f}")
print(f"  wall clock                          = {elapsed:.1f}s")

out = {"gate": {"ess_min": ess_min, "e1": e1, "passed": bool(gate)},
       "warmup": WARMUP, "num_samples": NSAMP,
       "diagnostics_acceptance_rate": raw if isinstance(raw, (int, float)) else str(raw),
       "mean_accept_prob": float(snap["mean_accept_prob"]),
       "accept_cnt_over_post_warmup": float(counted),
       "divergences": snap["divergences"],
       "step_size": snap["step_size"], "time_s": elapsed,
       "target_accept_prob": 0.8}
(HERE / "nuts_ar_recover_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("\nwrote nuts_ar_recover_results.json")
