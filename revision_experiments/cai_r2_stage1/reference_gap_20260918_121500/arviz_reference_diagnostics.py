import json
from pathlib import Path
import numpy as np, arviz as az
ROOT=Path(r"C:\neutra-hmc-surrogate\surrogate_training")
HERE=Path(r"C:\neutra-hmc-surrogate\surrogate_training\revision_experiments\cai_r2_stage1\reference_gap_20260918_121500")
names=[f"ablation_obs{i}" for i in range(1,6)]+["sailboat_boundary_e1_075","sailboat_field_frozen"]
out={}
for n in names:
    if n=="sailboat_field_frozen":
        a=np.load(ROOT/"revision_experiments"/"fig17_replay_samples.npz"); ch=a["nuts_all"].reshape(3,500,6)
    else:
        ch=np.load(HERE/f"{n}_reference_chains.npz")["chains"]
    idata=az.from_dict(posterior={"theta":ch})
    rh=np.asarray(az.rhat(idata,var_names=["theta"],method="rank")["theta"])
    eb=np.asarray(az.ess(idata,var_names=["theta"],method="bulk")["theta"])
    et=np.asarray(az.ess(idata,var_names=["theta"],method="tail")["theta"])
    out[n]={"rank_rhat":rh.tolist(),"rank_rhat_max":float(np.nanmax(rh)),
            "bulk_ess":eb.tolist(),"bulk_ess_min":float(np.nanmin(eb)),
            "tail_ess":et.tolist(),"tail_ess_min":float(np.nanmin(et))}
    print(n,"rhat",out[n]["rank_rhat_max"],"bulk",out[n]["bulk_ess_min"],"tail",out[n]["tail_ess_min"])
(HERE/"ARVIZ_REFERENCE_DIAGNOSTICS.json").write_text(json.dumps({"arviz":az.__version__,"targets":out},indent=2),encoding="utf-8")
