"""Authorized stage one: frozen manifest, forward audits and raw/available PSIS diagnostics.
No training, MCMC, repairs, clamping of proposals, installation or external writes.
"""
import os, sys, csv, json, math, hashlib, warnings, traceback, importlib
from pathlib import Path
from datetime import datetime, timezone
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[2]
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key,folder in [('MPLCONFIGDIR','mpl'),('XDG_CACHE_HOME','cache'),('TEMP','tmp'),('TMP','tmp')]:
    p=RUN/'environment'/folder;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
import numpy as np
import torch
from scipy import stats
from scipy.special import logsumexp
torch.set_num_threads(6)
INTAKE=json.loads((RUN/'ASSET_INTAKE.json').read_text(encoding='utf-8'))
ASSETS={Path(r['path']).as_posix():r for r in INTAKE['assets']}
LOG=(RUN/'analysis.log').open('x',encoding='utf-8')
def log(s):
    print(s,flush=True);LOG.write(s+'\n');LOG.flush()
def write_json(path,obj):
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,ensure_ascii=False,allow_nan=False)
def write_csv(path,rows):
    names=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=names,restval='NA_NOT_APPLICABLE');w.writeheader();w.writerows(rows)
def save_npz(path,**arrays):
    with path.open('xb') as f:np.savez_compressed(f,**arrays)
def asset(p):return ASSETS[Path(p).as_posix()]
def snap(p):return RUN/asset(p)['snapshot']
sys.path.insert(0,str(ROOT/'experiments'))
from cai_lib import load_surrogate,load_cnf,KModeLogPosterior
NARROW_SUR=ROOT/'surrogate_models/member_0.pt'
NARROW_CNF=Path('C:/nf-pmcmc/experiments/results/conditional_flow.pt')
EXT_SUR=ROOT/'surrogate_models_broad_ext/member_0.pt'
EXT_CNF=ROOT/'experiments/results/conditional_flow_broad_ext.pt'
models={};flows={};model_errors={}
for key,s,c in [('d6',NARROW_SUR,NARROW_CNF),('ext',EXT_SUR,EXT_CNF)]:
    try:models[key]=load_surrogate(snap(s));flows[key]=load_cnf(snap(c))
    except Exception as e:model_errors[key]=repr(e);log(f'MODEL BLOCKED {key}: {e!r}')
ckpt=torch.load(snap(Path('C:/nf-pmcmc/experiments/results/ablation_checkpoint.pt')),map_location='cpu',weights_only=False)
freeze=json.loads(snap(ROOT/'revision_experiments/task_r3_domain/field_freeze_results.json').read_text())
TEST_THETAS=[[.7,1.2,1.8,1,.8,1.1],[1.3,.8,1.5,.6,1.2,.9],[1]*6,[.6,1.5,.9,1.3,.7,1.4],[.85,1.1,1.7,.75,1.05,.95]]
FIELD_Y=[.91,1.08,1.47,2.55,3.16,3.81]
SEEDS=[42,43,44,45,46];NS=[2000,8000];TAUS=[.3,.5,.7]
OBS=[]
for i,truth in enumerate(TEST_THETAS):
    if 'hmc_means' in ckpt and i<len(ckpt['hmc_means']):
        mean,y=ckpt['hmc_means'][i]
        OBS.append(dict(id=f'ablation_obs{i+1}',model='d6',y=np.asarray(y,dtype=np.float32),
            y_proposal=np.asarray(y,dtype=np.float32),sigma=.30,truth=truth,
            observation_source='ablation_checkpoint.pt:hmc_means[%d][1]'%i,
            reference=[dict(method='standard_HMC',mean=np.asarray(mean).tolist(),
                source='C:/nf-pmcmc/experiments/results/ablation_checkpoint.pt:hmc_means[%d][0]'%i,
                target_match='MATCH_BY_SOURCE_RECIPE_CURRENT_CHECKPOINT_HASH_NOT_EMBEDDED_HISTORICALLY',
                evidence_status='INSUFFICIENT_MEAN_ONLY',chain_status='NOT_SAVED',
                diagnostics='No chain, per-parameter ESS or Rhat saved for these observations.')],
            category='fixed_ablation_synthetic',status='READY'))
    else:
        OBS.append(dict(id=f'ablation_obs{i+1}',model='d6',sigma=.30,truth=truth,
                        status='NOT_EXECUTED',reason='Fixed observation missing in ablation checkpoint.'))
if 'ext' in models:
    with torch.no_grad():f_nom=models['ext'](torch.ones(1,6)).squeeze(0).numpy()
    hybrid=np.concatenate([np.asarray(FIELD_Y,dtype=np.float32),f_nom[6:]]).astype(np.float32)
    fieldrefs=[]
    for method,key in [('NUTS','nuts'),('mass_NeuTra_HMC','mass_neutra')]:
        r=freeze['table13'][key]
        fieldrefs.append(dict(method=method,mean=r['mean'],
            source=f'revision_experiments/task_r3_domain/field_freeze_results.json:table13.{key}',
            target_match='MATCH_BY_FROZEN_SOURCE_RECIPE_CURRENT_CHECKPOINT_HASH_NOT_EMBEDDED_HISTORICALLY',
            evidence_status='INSUFFICIENT_CHAIN_NOT_SAVED',chain_status='NOT_SAVED',
            reported_ESS_min=r['ESS_min'],reported_split_Rhat=freeze['table13']['split_rhat_3chain'],
            diagnostics='Summary only. Three NUTS chains are local variables in field_freeze.py and are not persisted.'))
    OBS.append(dict(id='sailboat_field_frozen',model='ext',y=np.asarray(FIELD_Y,dtype=np.float32),
        y_proposal=hybrid,sigma=.30,truth=None,category='field',status='READY',
        observation_source='revlib.FIELD_MEASURED; field_freeze.py frozen six-mode target',reference=fieldrefs))
    theta_b=np.array([.75,1,1,1,1,1],dtype=np.float32)
    with torch.no_grad():yb=models['ext'](torch.from_numpy(theta_b).reshape(1,-1)).squeeze(0).numpy()
    OBS.append(dict(id='sailboat_boundary_e1_075',model='ext',y=yb,y_proposal=yb,sigma=.55,
        truth=theta_b.tolist(),category='boundary_synthetic',status='READY',
        observation_source='eth_final.py:surE([0.75,1,1,1,1,1]); all 20 modes, no added noise',
        reference=[dict(method='NONE',source='No matching reference saved by eth_final.py',
                        target_match='MISSING',evidence_status='INSUFFICIENT_NO_REFERENCE',chain_status='MISSING')]))
else:
    for oid in ['sailboat_field_frozen','sailboat_boundary_e1_075']:
        OBS.append(dict(id=oid,model='ext',status='NOT_EXECUTED',reason=model_errors.get('ext','Asset missing')))

CORRECTED=torch.load(snap(Path('C:/nf-pmcmc/experiments/results/corrected_amortized.pt')),map_location='cpu',weights_only=False)
ROBUST=torch.load(snap(Path('C:/nf-pmcmc/experiments/results/robustness_test.pt')),map_location='cpu',weights_only=False)
existing_targets=[dict(id='existing_corrected_obs1',sigma=.30,K=20,
    observation=np.asarray(CORRECTED['f_obs']).tolist(),observation_status='SAVED',
    source='corrected_amortized.pt:f_obs',surrogate=asset(NARROW_SUR),
    checkpoint_history='Current checkpoint fingerprint; original result did not embed a checkpoint hash.'),
    dict(id='existing_corrected_obs2',sigma=.30,K=20,
    observation=np.asarray(CORRECTED['generalization']['y_new']).tolist(),observation_status='SAVED',
    source='corrected_amortized.pt:generalization.y_new',surrogate=asset(NARROW_SUR),
    checkpoint_history='Current checkpoint fingerprint; original result did not embed a checkpoint hash.')]
for i,r in enumerate(ROBUST):
    existing_targets.append(dict(id=f'existing_robustness_obs{i+1}',sigma=.30,K=20,
        observation='NOT_RECOVERABLE_WITHOUT_REPLAYING_FORBIDDEN_MCMC_RNG_HISTORY',
        observation_status='MISSING',source=f'robustness_test.pt[{i}]; robustness_test.py',
        reason='Source uses one evolving global RNG, including prior observations\' HMC draws. Observations were not saved; no MCMC replay is permitted.',
        surrogate=asset(NARROW_SUR),checkpoint_history='Target prior/likelihood recipe known; actual observation not saved.'))

PSIS=dict(status='NOT_EXECUTED',package='arviz',version='NOT_INSTALLED',function='arviz.psislw',
    key_parameters={'reff':1.0,'normalize':True},
    reason='No installed mature PSIS implementation found: arviz, arviz_stats, psis, pymc and loo absent. No installation or handwritten replacement.',
    input_policy='Use entire N log-weight array, including negative infinity for true zero target density.')
try:
    az=importlib.import_module('arviz')
    if hasattr(az,'psislw'):
        PSIS.update(status='AVAILABLE',version=az.__version__,reason='Installed implementation available.')
except ModuleNotFoundError:az=None
except Exception as e:PSIS['reason']='Existing PSIS package import failure: '+repr(e);az=None

manifest_obs=[]
for o in OBS:
    if o['status']!='READY':manifest_obs.append(o);continue
    key=o['model'];s=NARROW_SUR if key=='d6' else EXT_SUR;c=NARROW_CNF if key=='d6' else EXT_CNF
    K=len(o['y']);sigma=o['sigma']
    manifest_obs.append(dict(id=o['id'],status='READY',category=o['category'],
        surrogate=asset(s),CNF=asset(c),observation_hz=o['y'].astype(float).tolist(),
        proposal_condition_hz=o['y_proposal'].astype(float).tolist(),observation_source=o['observation_source'],
        preprocessing={'dtype':'float32','observation_normalization':'none','parameter_input':'dimensionless e1..e6',
            'surrogate_normalization':'(theta-xm)/xs; output*ys+ym; buffers stored in checkpoint',
            'padding':'measured first six + extended-surrogate nominal modes 7..20' if o['category']=='field' else 'none (20 observed synthetic modes)'},
        modes=list(range(1,K+1)),proposal_modes=list(range(1,21)),
        prior={'family':'independent lognormal','log_mean':[0.0]*6,'log_std':[sigma]*6,
            'support':'theta_i > 0 for all i, no upper bound, no truncation, no correlations or other constraints',
            'parameter_mean':math.exp(sigma*sigma/2)},
        likelihood={'log_density_up_to_common_constant':'-0.5/0.001 * sum_{j=1..K} ((f_j(theta)-y_j)/y_j)^2',
            'kappa_sq':.001,'kappa':math.sqrt(.001),'noise_sigma_hz':(math.sqrt(.001)*o['y']).astype(float).tolist(),
            'covariance':'diagonal(0.001*y_j^2); independent heteroscedastic in Hz, fixed by observed y',
            'not_identification_uncertainty':True},
        transforms={'coordinate':'theta (dimensionless)','prior_log_coordinate':'z=log(theta)',
            'prior_Jacobian_in_theta':'-sum(log(theta)) included in log target',
            'flow':'theta-space affine RealNVP; Gaussian z_base in R^6',
            'proposal_log_density':'-0.5*sum(z_base^2)-3*log(2*pi)+log_abs_det_dz_dtheta',
            'surrogate_normalization_Jacobian':'not a parameter density transform; no additional density Jacobian',
            'E_i_APDL_Pa':'theta_i * 2e10 exactly once'},
        surrogate_applicable_domain=[.5,2.] if key=='d6' else [.5,3.5],
        training_distribution={'sigma':.30 if key=='d6' else .55,
            'prior_sampling':'exp(sigma*randn), no clamp' if key=='d6' else 'exp(0.55*randn).clamp(0.5,3.5); has endpoint atoms',
            'observation':'20 modes: y=f(theta)+sqrt(0.001)*f(theta)*epsilon',
            'noise_covariance':'diagonal(0.001*f_j(theta)^2), depends on clean training frequency',
            'inference_difference':'Inference variance uses observed y_j^2, training uses clean f_j(theta)^2.',
            'field_extra_difference':'old proposal trained with sigma=0.55 and clamp using 20 genuine modes; field target sigma=0.30 untruncated, six measured modes and hybrid padding' if o['category']=='field' else 'not applicable'},
        proposal_support={'mathematical_support':'R^6; affine bijections with finite nonzero scales and Gaussian base; no output clamp',
            'target_support_contained':True,
            'hard_boundary_from_training_clamp':False,
            'limitation':'Finite diagnostics cannot exclude practically negligible proposal mass in positive-target regions, or extrapolation outside surrogate training domain.'},
        reference=o['reference'],reference_reliable=False,
        target_match_status='SOURCE_RECIPE_MATCH_BUT_REFERENCE_EVIDENCE_INSUFFICIENT' if o['category']!='boundary_synthetic' else 'REFERENCE_MISSING',
        truth=o['truth']))
manifest=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),
    baseline={'branch':'revision-es','commit':'17f07923f38f199db2987f009111a27d4fd8dd8f'},
    execution={'run':str(RUN),'seeds':SEEDS,'N':NS,'nested_samples':True,'max_draws_per_observation_seed':8000,
        'tau':TAUS,'training':'FORBIDDEN','new_MCMC':'FORBIDDEN'},
    physical_mapping={'e1':'Selected main-building G/F-8/F columns and annex front ground/support/lobby columns',
        'e2':'Selected main-building 9/F-R/F columns',
        'e3':'Selected external/internal beams on multiple floors in main building and annex',
        'e4_e5_e6':'Inactive in the final simplified FE model; author-confirmed, no verification FE runs'},
    target_density_support_policy='Replace original implementation\'s -1e30 sentinel on nonpositive theta with exact -inf for mathematical zero target density. Original code remains unchanged.',
    PSIS=PSIS,observations=manifest_obs,existing_posterior_targets=existing_targets,
    observed_reference_mismatch_evidence='E5_rhat.json pools different priors/models and has max Rhat=1.181; it is not a matched frozen-target convergence certificate.',
    identification_uncertainty={'status':'NOT_OBTAINED','reason':'No per-mode fast Bayesian FFT standard deviations or covariance in the required local observation assets.'})
write_json(RUN/'TARGET_MANIFEST.json',manifest)
log('TARGET_MANIFEST completed before forward/weight/error calculations.')

for folder in ['inactive_arrays','routing_arrays','posterior_arrays']:(RUN/folder).mkdir(exist_ok=False)
def ll_and_target(sur,theta,y,sigma):
    """Exact frozen mathematical density in theta coordinates; retain all samples."""
    x=torch.as_tensor(theta,dtype=torch.float32);yt=torch.as_tensor(y,dtype=torch.float32)
    with torch.no_grad():
        fp=sur(x)[:,:len(y)]
        ll=-.5/.001*(((fp-yt)/yt)**2).sum(-1)
        positive=(x>0).all(-1)
        lp=torch.full_like(ll,-float('inf'))
        lx=torch.log(x[positive])
        lp[positive]=ll[positive]+(-.5*(lx/sigma)**2-lx).sum(-1)
        nonfinite=~torch.isfinite(x).all(-1)
        lp[nonfinite]=float('nan')
    return ll.numpy().astype(float),lp.numpy().astype(float)

# First substantive computation: inactive directions at fixed active parameters.
audit=[]
for key in ['d6','ext']:
    if key not in models:continue
    sur=models[key]
    o=next(o for o in OBS if o['id']==('ablation_obs3' if key=='d6' else 'sailboat_field_frozen'))
    anchors=[('nominal',np.ones(6,dtype=np.float32),'theta=(1,1,1,1,1,1)')]
    if key=='ext':anchors.append(('frozen_field_NUTS_summary_mean',np.array(freeze['table13']['nuts']['mean'],dtype=np.float32),
        'field_freeze_results.json:table13.nuts.mean; representative summary mean, raw chain unavailable'))
    grid=[.5,.75,1,1.25,1.5,1.75,2] if key=='d6' else [.5,.75,1,1.25,1.5,2,2.5,3,3.5]
    for anchor_id,anchor,source in anchors:
        with torch.no_grad():fa=sur(torch.from_numpy(anchor).reshape(1,-1)).squeeze(0).numpy().astype(float)
        la,_=ll_and_target(sur,anchor.reshape(1,-1),o['y'],o['sigma'])
        for d in range(3,6):
            values=sorted(set(grid+[float(anchor[d])]))
            theta=np.tile(anchor,(len(values),1));theta[:,d]=values
            with torch.no_grad():freq=sur(torch.from_numpy(theta)).numpy().astype(float)
            ll,_=ll_and_target(sur,theta,o['y'],o['sigma'])
            sigma_hz=math.sqrt(.001)*o['y'].astype(float)
            scaled=(freq[:,:len(o['y'])]-fa[:len(o['y'])])/sigma_hz
            mahal=np.sqrt((scaled**2).sum(1))
            save_npz(RUN/'inactive_arrays'/f'{key}_{anchor_id}_e{d+1}.npz',theta=theta,
                anchor_theta=anchor,anchor_frequency=fa,frequencies=freq,log_likelihood=ll,
                anchor_log_likelihood=la,noise_sigma_hz=sigma_hz,standardized_delta=scaled)
            for i,x in enumerate(theta):
                for j in range(20):
                    delta=freq[i,j]-fa[j]
                    audit.append(dict(status='COMPLETED',reason='OK',model=key,anchor=anchor_id,
                        anchor_source=source,direction=f'e{d+1}',point=i,mode=j+1,
                        **{f'e{k+1}':float(x[k]) for k in range(6)},anchor_frequency_hz=fa[j],
                        surrogate_frequency_hz=freq[i,j],delta_hz=delta,relative_delta=delta/fa[j],
                        in_likelihood=j<len(o['y']),likelihood_sigma_hz=float(sigma_hz[j]) if j<len(o['y']) else 'NA',
                        standardized_delta=float(scaled[i,j]) if j<len(o['y']) else 'NA',
                        noise_status='DEFINED_BY_TARGET' if j<len(o['y']) else 'NOT_IN_LIKELIHOOD',
                        standardized_joint_change=float(mahal[i]),log_likelihood=float(ll[i]),
                        delta_log_likelihood=float(ll[i]-la[0]),observation_id=o['id'],
                        checkpoint_sha256=asset(NARROW_SUR if key=='d6' else EXT_SUR)['sha256']))
write_csv(RUN/'INACTIVE_DIRECTION_AUDIT.csv',audit)
log(f'Inactive-direction audit completed: {len(audit)} modal rows.')

POST=[];QUANTS=[.025,.25,.5,.75,.975]
def posterior_audit(theta,target_id,kind,sigma,source,evidence,observation_status,seed='EXISTING',N=None):
    theta=np.asarray(theta,dtype=float)
    assert theta.ndim==2 and theta.shape[1]==6
    for d in range(3,6):
        x=theta[:,d];q=np.quantile(x,QUANTS);prior=stats.lognorm(s=sigma,scale=1)
        pq=prior.ppf(QUANTS);width=q[-1]-q[0];pw=pq[-1]-pq[0]
        ordered=np.sort(x);cdf=prior.cdf(ordered)
        ks=float(max(np.max(np.arange(1,len(x)+1)/len(x)-cdf),np.max(cdf-np.arange(len(x))/len(x))))
        # CDF distance and quantile-W1 are descriptive, never a significance test.
        pp=(np.arange(2048)+.5)/2048
        w1=float(np.mean(np.abs(np.quantile(x,pp)-prior.ppf(pp))))
        pear=[];spear=[]
        for a in range(3):
            pear.append(float(np.corrcoef(x,theta[:,a])[0,1]))
            spear.append(float(stats.spearmanr(x,theta[:,a]).statistic))
        POST.append(dict(status='COMPLETED_DESCRIPTIVE',reason='No iid MCMC significance claim',
            target=target_id,sample_kind=kind,seed=seed,N=len(x) if N is None else N,parameter=f'e{d+1}',
            source=source,reference_evidence=evidence,observation_status=observation_status,
            prior_sigma_log=sigma,prior_structure='independent untruncated lognormal, no cross-parameter constraint',
            **{f'q{p*100:g}':float(q[k]) for k,p in enumerate(QUANTS)},
            **{f'prior_q{p*100:g}':float(pq[k]) for k,p in enumerate(QUANTS)},
            width95=float(width),prior_width95=float(pw),width_ratio=float(width/pw),
            descriptive_KS=ks,descriptive_quantile_W1=w1,prior_support_outside_fraction=float((x<=0).mean()),
            **{f'pearson_e{k+1}':pear[k] for k in range(3)},
            **{f'spearman_e{k+1}':spear[k] for k in range(3)}))

existing=[]
for kind,key in [('existing_fast','fast_mode'),('existing_reference_NeuTra','exact_mode'),('existing_reference_HMC','hmc_baseline')]:
    existing.append(('existing_corrected_obs1',kind,np.asarray(CORRECTED[key]['chain']),f'corrected_amortized.pt:{key}.chain','SAVED'))
for kind,key in [('existing_fast','amortized'),('existing_reference_NeuTra','corrected')]:
    existing.append(('existing_corrected_obs2',kind,np.asarray(CORRECTED['generalization'][key]['chain']),f'corrected_amortized.pt:generalization.{key}.chain','SAVED'))
for i,r in enumerate(ROBUST):
    for kind,key in [('existing_fast','chain_amort'),('existing_reference_HMC','chain_hmc')]:
        existing.append((f'existing_robustness_obs{i+1}',kind,np.asarray(r[key]),f'robustness_test.pt[{i}].{key}','MISSING'))
for ti,kind,x,src,ostatus in existing:
    save_npz(RUN/'posterior_arrays'/f'{ti}_{kind}.npz',theta=x)
    posterior_audit(x,ti,kind,.30,src,'DESCRIPTIVE_SINGLE_CHAIN_NOT_A_MATCHED_ROUTING_REFERENCE',ostatus)
log('Existing fast/reference posterior marginal and active/inactive correlation audits completed.')

ROUTES=[];ERRORS=[];PAIR_ROWS=[];SUPPORT=[]
def normalize_raw(logw):
    if np.isnan(logw).any():return None,'INVALID','NaN log weight retained; no filtering'
    if np.isposinf(logw).any():return None,'INVALID','Positive infinite log weight retained; no filtering'
    if np.isneginf(logw).all():return None,'INVALID','All weights zero'
    if (~(np.isfinite(logw)|np.isneginf(logw))).any():return None,'INVALID','Unsupported nonfinite log weight'
    total=logsumexp(logw)
    w=np.exp(logw-total)
    return w,'VALID','OK'
def error_metrics(x,ref_mean,weights=None):
    mean=x.mean(0) if weights is None else np.sum(weights[:,None]*x,axis=0)
    d=np.abs(mean-ref_mean)
    return mean,d,float(d[:3].mean()),float(d[3:].mean()),float(d.mean())

for o in OBS:
    if o['status']!='READY' or o['model'] not in models:
        for seed in SEEDS:
            for N in NS:ROUTES.append(dict(observation=o['id'],seed=seed,N=N,status='NOT_EXECUTED',reason=o.get('reason','Model assets unavailable'),raw_ESS_ratio='NA',smoothed_ESS_ratio='NA',Pareto_k='NA'))
        continue
    sur=models[o['model']];flow=flows[o['model']]
    # Positive target tails retain proposal density: training clamp is not a hard support limit.
    probes=np.ones((5,6),dtype=np.float32);probes[1,:]=.3;probes[2,:]=4.;probes[3,3:]=.3;probes[4,3:]=4.
    with torch.no_grad():probe_q=flow.log_prob(torch.from_numpy(probes),torch.from_numpy(o['y_proposal'])).numpy()
    _,probe_t=ll_and_target(sur,probes,o['y'],o['sigma'])
    for i,p in enumerate(probes):SUPPORT.append(dict(observation=o['id'],theta=p.tolist(),log_q=float(probe_q[i]),log_target=float(probe_t[i]),both_finite=bool(np.isfinite(probe_q[i]) and np.isfinite(probe_t[i])),
        extrapolation=bool((p<.5).any() or (p>(2 if o['model']=='d6' else 3.5)).any())))
    for seed in SEEDS:
        try:
            torch.manual_seed(seed)
            yt=torch.from_numpy(o['y_proposal'])
            with torch.no_grad():
                theta=flow.sample(yt,8000).numpy()
                # Batch surrogate to limit memory; no resampling or removal.
                logq=flow.log_prob(torch.from_numpy(theta),yt).numpy().astype(float)
            ll_parts=[];lt_parts=[]
            for start in range(0,8000,1000):
                l,t=ll_and_target(sur,theta[start:start+1000],o['y'],o['sigma']);ll_parts.append(l);lt_parts.append(t)
            ll=np.concatenate(ll_parts);lt=np.concatenate(lt_parts);lw=lt-logq
            # Record agreement with the original frozen implementation on its valid support.
            pos=(theta>0).all(1)&np.isfinite(theta).all(1)
            original_lp=KModeLogPosterior(sur,o['y'].tolist(),kappa_sq=.001,prior_sigma=o['sigma'])
            idx=np.flatnonzero(pos)[:100]
            with torch.no_grad():orig=original_lp(torch.from_numpy(theta[idx])).numpy().astype(float)
            implementation_diff=float(np.max(np.abs(orig-lt[idx]))) if len(idx) else 'NA_NO_POSITIVE_SAMPLE'
            for N in NS:
                x=theta[:N];q=logq[:N];t=lt[:N];l=ll[:N];wlog=lw[:N]
                weights,status,reason=normalize_raw(wlog)
                outside=~(x>0).all(1);numx=~np.isfinite(x).all(1)
                if numx.any() or (~np.isfinite(q)).any() or np.isnan(t).any() or np.isposinf(t).any():
                    weights=None;status='INVALID';reason='Nonfinite theta/log_q or NaN/+inf target retained; no filtering'
                smw=None;smess='NA';k='NA';smmax='NA';smstatus='NOT_EXECUTED';smreason=PSIS['reason']
                psis_logw=np.full(N,np.nan)
                if status=='VALID' and PSIS['status']=='AVAILABLE':
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            psis_logw,pk=az.psislw(wlog,reff=1.0)
                        k=float(pk);smw,smstatus,smreason=normalize_raw(np.asarray(psis_logw))
                        if smw is not None:
                            smess=float(1/(smw**2).sum()/N);smmax=float(smw.max())
                        if caught:smreason+='; '+ '; '.join(str(c.message) for c in caught)
                    except Exception as e:smstatus='NUMERICAL_FAILURE';smreason=repr(e)
                elif status!='VALID':smstatus='NOT_EXECUTED_INVALID_RAW';smreason=reason
                ess=float(1/(weights**2).sum()/N) if weights is not None else 'NA'
                maxw=float(weights.max()) if weights is not None else 'NA'
                r=dict(observation=o['id'],seed=seed,N=N,status=status,reason=reason,
                    raw_ESS_ratio=ess,smoothed_ESS_ratio=smess,Pareto_k=k,raw_max_weight=maxw,smoothed_max_weight=smmax,
                    PSIS_status=smstatus,PSIS_reason=smreason,target_outside_fraction=float(outside.mean()),
                    surrogate_domain_outside_fraction=float(((x<.5)|(x>(2 if o['model']=='d6' else 3.5))).any(1).mean()),
                    zero_target_count=int(np.isneginf(t).sum()),NaN_theta_count=int(numx.sum()),
                    NaN_logq_count=int(np.isnan(q).sum()),NaN_logtarget_count=int(np.isnan(t).sum()),
                    positive_inf_logweights_count=int(np.isposinf(wlog).sum()),
                    reference_evidence='INSUFFICIENT',reference_target_match=manifest_obs[next(i for i,v in enumerate(manifest_obs) if v['id']==o['id'])]['target_match_status'],
                    raw_fast_error_status='DESCRIPTIVE_MEAN_ONLY' if any('mean' in z for z in o['reference']) else 'NO_REFERENCE',
                    raw_fast_MAE_reference_e1_e3='NA',raw_fast_MAE_reference_e4_e6='NA',raw_fast_MAE_reference_all6='NA',
                    PSIS_MAE_reference_e1_e3='NA',original_target_implementation_max_abs_diff=implementation_diff)
                for tau in TAUS:
                    r[f'raw_route_tau_{tau}']='INVALID' if weights is None else ('FAST' if ess>=tau else 'EXACT_REQUIRED')
                    r[f'smoothed_route_tau_{tau}']='NOT_EXECUTED' if smw is None else ('FAST' if smess>=tau else 'EXACT_REQUIRED')
                reference_with_mean=[z for z in o['reference'] if 'mean' in z]
                for ri,ref in enumerate(reference_with_mean):
                    mean,err,ma,mi,mall=error_metrics(x,np.array(ref['mean']))
                    if ri==0:r.update(raw_fast_MAE_reference_e1_e3=ma,raw_fast_MAE_reference_e4_e6=mi,raw_fast_MAE_reference_all6=mall)
                    for d in range(6):
                        quant=np.quantile(x[:,d],QUANTS)
                        PAIR_ROWS.append(dict(observation=o['id'],seed=seed,N=N,method='RAW_UNWEIGHTED_FAST',
                            parameter=f'e{d+1}',physical_group='effective' if d<3 else 'inactive',
                            reference_method=ref['method'],status='DESCRIPTIVE_MEAN_ONLY',reason='Reference chain missing; no reliable paired distribution or convergence evidence',
                            fast_mean=float(mean[d]),reference_mean=float(ref['mean'][d]),mean_absolute_error=float(err[d]),
                            **{f'fast_q{p*100:g}':float(quant[k]) for k,p in enumerate(QUANTS)},fast_width95=float(quant[-1]-quant[0]),
                            reference_quantiles='NA_MISSING_CHAIN',reference_width95='NA_MISSING_CHAIN',
                            width_difference='NA_MISSING_CHAIN',distribution_distance='NA_MISSING_CHAIN'))
                    if smw is not None:
                        _,_,sma,_,_=error_metrics(x,np.array(ref['mean']),smw)
                        if ri==0:r['PSIS_MAE_reference_e1_e3']=sma
                if o['truth'] is not None:
                    mean,err,ma,mi,mall=error_metrics(x,np.asarray(o['truth']))
                    r.update(raw_fast_MAE_truth_e1_e3=ma,raw_fast_MAE_truth_e4_e6=mi,raw_fast_MAE_truth_all6=mall)
                else:r.update(raw_fast_MAE_truth_e1_e3='NA_NO_FIELD_TRUTH',raw_fast_MAE_truth_e4_e6='NA_NO_FIELD_TRUTH',raw_fast_MAE_truth_all6='NA_NO_FIELD_TRUTH')
                path=RUN/'routing_arrays'/f'{o["id"]}_seed{seed}_N{N}.npz'
                save_npz(path,theta=x,log_q=q,log_target=t,log_likelihood=l,log_weights=wlog,
                    raw_normalized_weights=weights if weights is not None else np.full(N,np.nan),
                    PSIS_log_weights=psis_logw,PSIS_normalized_weights=smw if smw is not None else np.full(N,np.nan),
                    target_support_outside=outside,numerical_theta_failure=numx,
                    observation=o['y'],proposal_condition=o['y_proposal'])
                r['arrays']=str(path.relative_to(RUN));ROUTES.append(r)
                posterior_audit(x,o['id'],'NEW_RAW_FAST',o['sigma'],r['arrays'],'INSUFFICIENT_REFERENCE', 'SAVED',seed=seed,N=N)
                log(f'{o["id"]} seed={seed} N={N} raw_ESS/N={ess} max_w={maxw} status={status}')
        except Exception as e:
            msg=traceback.format_exc();log(msg);ERRORS.append(dict(observation=o['id'],seed=seed,reason=msg))
            for N in NS:ROUTES.append(dict(observation=o['id'],seed=seed,N=N,status='NUMERICAL_FAILURE',reason=repr(e),raw_ESS_ratio='NA',smoothed_ESS_ratio='NA',Pareto_k='NA'))
write_csv(RUN/'ROUTING_DIAGNOSTICS.csv',ROUTES)
write_csv(RUN/'POSTERIOR_INACTIVE_AUDIT.csv',POST)
write_csv(RUN/'FAST_REFERENCE_PAIRING.csv',PAIR_ROWS)
write_json(RUN/'PROPOSAL_SUPPORT_AUDIT.json',SUPPORT)
write_json(RUN/'NUMERICAL_FAILURES.json',ERRORS)
SUMMARY=[]
for o in OBS:
    for N in NS:
        rows=[r for r in ROUTES if r['observation']==o['id'] and r['N']==N]
        good=[r for r in rows if r['status']=='VALID'];v=[r['raw_ESS_ratio'] for r in good]
        s=dict(observation=o['id'],N=N,valid_seeds=len(good),invalid_or_failed_seeds=len(rows)-len(good),
            raw_ESS_ratio_mean=float(np.mean(v)) if v else 'NA',raw_ESS_ratio_SD=float(np.std(v,ddof=1)) if len(v)>1 else 'NA',
            raw_ESS_ratio_min=float(np.min(v)) if v else 'NA',raw_ESS_ratio_max=float(np.max(v)) if v else 'NA')
        for tau in TAUS:
            routes=[r.get(f'raw_route_tau_{tau}','INVALID') for r in rows]
            s[f'raw_tau_{tau}_FAST_count']=routes.count('FAST')
            s[f'raw_tau_{tau}_EXACT_count']=routes.count('EXACT_REQUIRED')
            s[f'raw_tau_{tau}_INVALID_count']=routes.count('INVALID')
            s[f'raw_tau_{tau}_cross_seed_stable']=len(set(routes))==1
        SUMMARY.append(s)
write_csv(RUN/'ROUTING_STABILITY.csv',SUMMARY)
NESTED=[]
for o in OBS:
    for seed in SEEDS:
        rr=[r for r in ROUTES if r['observation']==o['id'] and r['seed']==seed]
        a=next(r for r in rr if r['N']==2000);b=next(r for r in rr if r['N']==8000)
        r=dict(observation=o['id'],seed=seed,status='COMPLETED' if a['status']=='VALID' and b['status']=='VALID' else 'INVALID',
            ratio_2000=a['raw_ESS_ratio'],ratio_8000=b['raw_ESS_ratio'],
            ratio_delta=b['raw_ESS_ratio']-a['raw_ESS_ratio'] if a['status']=='VALID' and b['status']=='VALID' else 'NA',
            PSIS_stability='NOT_EXECUTED' if PSIS['status']!='AVAILABLE' else 'SEE_DIAGNOSTICS')
        for tau in TAUS:r[f'raw_tau_{tau}_same_route']=a.get(f'raw_route_tau_{tau}','INVALID')==b.get(f'raw_route_tau_{tau}','INVALID')
        NESTED.append(r)
write_csv(RUN/'NESTED_N_STABILITY.csv',NESTED)
log('Analysis completed. No training or MCMC has run. PSIS status: '+PSIS['status'])
LOG.close()
