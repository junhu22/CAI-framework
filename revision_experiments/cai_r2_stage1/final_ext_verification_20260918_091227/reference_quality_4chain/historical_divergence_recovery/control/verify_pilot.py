"""Verify one saved pilot without invoking any sampler or training function."""
import hashlib
import io
import json
import math

import numpy as np
import torch

from run_pilot import (OUT,CONTROL,NPZ,RESULT,LOG,AUDIT,INTEGRITY,REFERENCE,
                       sha,plain,verify_preservation,write_new_json,utc_now)


def main():
    result=json.loads(RESULT.read_text(encoding="utf-8"))
    integrity=json.loads(INTEGRITY.read_text(encoding="utf-8"))
    assert result["status"]=="COMPLETED" and result["scientific_chain_calls"]==1
    with np.load(NPZ,allow_pickle=False) as artifact,np.load(REFERENCE,allow_pickle=False) as historical:
        replay=artifact["replay_theta"]
        original=historical["chains"][0]
        assert np.array_equal(artifact["historical_theta"],original)
        assert replay.shape==original.shape==(600,6)
        assert np.isfinite(replay).all()
        delta=np.abs(replay.astype(np.float64)-original.astype(np.float64))
        comparison=result["comparison"]
        assert comparison["np_array_equal"]==bool(np.array_equal(replay,original))
        assert comparison["max_abs_difference"]==float(delta.max())
        assert comparison["mean_abs_difference"]==float(delta.mean())
        assert comparison["number_of_different_elements"]==int(np.count_nonzero(replay!=original))
        assert comparison["allclose_atol0_rtol0"]==bool(np.allclose(replay,original,atol=0,rtol=0))
        assert comparison["allclose_atol1e_minus8_rtol1e_minus7"]==bool(np.allclose(replay,original,atol=1e-8,rtol=1e-7))
        chunks=[]
        for source in result["manifest"]["inputs"]:
            assert sha(source["path"])==source["sha256"]
            with np.load(source["path"],allow_pickle=False) as data:
                chunks.append(data["theta"])
        init=np.clip(np.nanmean(np.concatenate(chunks,axis=0),axis=0),.15,5.)
        assert np.array_equal(init,artifact["theta_init"])
        assert np.array_equal(init,np.asarray(result["manifest"]["rebuilt_init"]))
        raw_bytes=artifact["raw_mc_diagnostics_torch_bytes"].tobytes()
        assert hashlib.sha256(raw_bytes).hexdigest()==result["raw_diagnostics_serialized_sha256"]
        raw=torch.load(io.BytesIO(raw_bytes),map_location="cpu",weights_only=True)
        assert plain(raw)==result["raw_mc_diagnostics"]
        divergences=raw["divergences"]["chain 0"]
        acceptance=raw["acceptance rate"]["chain 0"]
        assert artifact["divergences"].tolist()==divergences==result["replay_divergence_indices"]
        assert len(divergences)==result["replay_divergence_count"]
        assert artifact["acceptance_rate"].item()==acceptance==result["replay_acceptance_rate"]
    exact=bool(np.array_equal(replay,original) or float(delta.max())==0)
    assert exact==result["exact_reproduction_pass"]==integrity["exact_reproduction_pass"]
    assert result["qualification_for_remaining_20_diagnostics"]==exact
    assert result["replay_diagnostics_attributed_to_historical_chain"]==exact
    assert result["recovered_historical_divergence_count"]==(len(divergences) if exact else None)
    assert result["recovered_historical_acceptance_rate"]==(acceptance if exact else None)
    assert result["remaining_20_chain_calls"]==result["other_target_calls"]==result["other_seed_calls"]==0
    assert result["mcmc_stopped"] and result["awaiting_review"]
    snapshot=json.loads((CONTROL/"FROZEN_BEFORE_PILOT.json").read_text(encoding="utf-8"))
    preservation=verify_preservation(snapshot)
    assert len(snapshot["original_367_files"])==367
    for output in integrity["outputs"].values():
        assert sha(output["path"])==output["sha256"]
    final=dict(integrity_verification_status="PASS",checked_utc=utc_now(),exact_reproduction_pass=exact,
               comparison=comparison,recovered_divergence_count=result["recovered_historical_divergence_count"],
               acceptance_rate=acceptance,historical_preservation=preservation,original_367_unchanged=True,
               scientific_sampler_calls=1,remaining_20_sampler_calls=0,flow_training_calls=0,
               stopped_at_pilot_review=True,outputs={path.name:sha(path) for path in [NPZ,RESULT,LOG,AUDIT,INTEGRITY]})
    write_new_json(CONTROL/"FINAL_VERIFICATION.json",final)
    print(json.dumps(final,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
