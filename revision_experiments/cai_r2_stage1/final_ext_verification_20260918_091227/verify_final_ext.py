"""Bounded R2 verification: final ext replay and paired FE/NN observations only."""
import csv, datetime, hashlib, json, math, subprocess, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
import torch
ROOT = Path(r'C:\neutra-hmc-surrogate\surrogate_training')
HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'local_fe_rerun_20260918_085458'
SRC = ROOT / 'revision_experiments' / 'task_r3_domain'
SUR_PATH = ROOT / 'surrogate_models_broad_ext' / 'member_0.pt'
CNF_PATH = ROOT / 'experiments' / 'results' / 'conditional_flow_broad_ext.pt'
sys.path.insert(0, str(ROOT / 'experiments'))
import cai_lib as C
assert Path(C.__file__).resolve() == (ROOT / 'experiments' / 'cai_lib.py').resolve()
torch.set_num_threads(6)
KAP = [0.01, 0.03, 0.05]
SEV = [0.10, 0.25, 0.50]
POS = {'e1': 0, 'e3': 2, 'e5': 4}
NREP, NNULL, NFAST = 12, 120, 1000

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name, data):
    with (HERE / name).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
def table(name, rows):
    with (HERE / name).open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def git_state():
    r = subprocess.run(['git', '--no-optional-locks', '-C', str(ROOT), 'status', '--short', '--untracked-files=no'], capture_output=True, text=True, check=True)
    return r.stdout
sources = [SUR_PATH, CNF_PATH, ROOT/'experiments'/'cai_lib.py', SRC/'e3_ext.py', SRC/'e3_ext_tpr.py', SRC/'e3_ext_results.json', SRC/'e3_ext_tpr_results.json']
for i in range(1, 7):
    p = f'p{i:02d}'
    sources += [OLD/p/'run.mac', OLD/p/'solver.log', OLD/p/f'cai1_{p}_c5282dbe_modal.txt', OLD/p/f'cai1_{p}_c5282dbe_parameters.txt']
before = {str(p): sha(p) for p in sources}
initial_git = git_state()
manifest = {'created': datetime.datetime.now().isoformat(), 'python': sys.executable, 'torch': torch.__version__, 'numpy': np.__version__, 'source_sha256': before, 'script_sha256': sha(__file__), 'git_tracked_before': initial_git,
    'model_version': 'FINAL_EXT_EXPLICIT_PATHS', 'grid_replay': {'damage_n': 324, 'null_n': 360, 'seed': 42, 'fast_n': 1000},
    'diagnostic_target': {'K': 20, 'prior': 'independent untruncated lognormal(0,0.55)', 'kappa_squared': 0.001, 'coordinate': 'theta', 'positive_support': True},
    'paired_design': {'points': 6, 'noise_levels': KAP, 'replicates': 12, 'fast_n': 1000, 'common_noise_and_latent_draws': True},
    'boundaries': ['No training', 'No MCMC', 'No new FE', 'No package installation', 'No manuscript edit', 'No AUC inference from the FE paired subset']}
dump('RUN_MANIFEST.json', manifest)
torch.manual_seed(42); np.random.seed(42)
sur = C.load_surrogate(SUR_PATH)
cnf = C.load_cnf(CNF_PATH, dim_theta=6, dim_y=20, n_layers=8, hidden_dim=128)
def surro(t):
    with torch.no_grad(): return sur(torch.tensor(t, dtype=torch.float32).reshape(1,-1)).squeeze(0).numpy()
def draw(y): return cnf.sample(torch.tensor(y, dtype=torch.float32), NFAST).numpy()
def auc(p,n):
    a=np.concatenate([p,n]); r=a.argsort().argsort().astype(float)+1
    return float((r[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n)))

def diagnostic(samples, y):
    th = torch.as_tensor(samples, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32)
    post = C.KModeLogPosterior(sur, list(y), kappa_sq=0.001, prior_sigma=0.55)
    with torch.no_grad():
        lq = cnf.log_prob(th, yt).double().numpy()
        lt = post(th).double().numpy()
    valid = np.all(samples > 0, axis=1)
    lt[~valid] = -np.inf
    lw = lt-lq
    if np.isnan(lw).any() or np.isposinf(lw).any() or not np.isfinite(lw).any():
        raise RuntimeError('Invalid importance weights; no filtering or replacement allowed')
    w=np.exp(lw-np.max(lw)); w/=w.sum()
    return {'ess_ratio': float(1/np.sum(w*w)/len(w)), 'max_weight': float(w.max()), 'outside_target_fraction': float((~valid).mean())}, lt, lq, lw

records=[]; arrays=[]; ys=[]; truths=[]; null={}; cells={}
for k in KAP:
    null[k]=[]
    for rep in range(NNULL):
        th=np.clip(np.exp(0.55*np.random.randn(6)),0.5,3.5)
        y=surro(th); y=y+k*np.abs(y)*np.random.randn(20)
        tf=draw(y); pm=tf.mean(0); null[k].append(pm)
        records.append({'group':'null','parameter':'', 'severity':0.0,'kappa':k,'rep':rep})
        arrays.append(tf); ys.append(y); truths.append(th)
    null[k]=np.array(null[k])
    print(f'REPLAY null kappa={k} complete', flush=True)
for pn,pi in POS.items():
    for s in SEV:
        for k in KAP:
            dm=[]; loc=[]
            for rep in range(NREP):
                th=np.ones(6); th[pi]=1-s
                y=surro(th); y=y+k*np.abs(y)*np.random.randn(20)
                tf=draw(y); pm=tf.mean(0); dm.append(pm); loc.append(abs(pm[pi]-th[pi])/th[pi])
                records.append({'group':'damage','parameter':pn,'severity':s,'kappa':k,'rep':rep})
                arrays.append(tf); ys.append(y); truths.append(th)
            dm=np.array(dm); pos=-dm[:,pi]; neg=-null[k][:,pi]
            cells[f'{pn}|{s}|{k}']={'auc':auc(pos,neg),'tpr05':float((pos>np.quantile(neg,0.95)).mean()),'loc_err':float(np.mean(loc))}
    print(f'REPLAY {pn} complete', flush=True)
frozen=json.loads((SRC/'e3_ext_tpr_results.json').read_text())
maxdiff={key:max(abs(cells[c][key]-frozen['cells'][c][key]) for c in cells) for key in ['auc','tpr05','loc_err']}
passed=maxdiff['auc']<1e-12 and maxdiff['tpr05']<1e-12 and maxdiff['loc_err']<1e-6
np.savez_compressed(HERE/'FINAL_GRID_REPLAY_ARRAYS.npz', samples=np.stack(arrays), observations=np.stack(ys), truths=np.stack(truths))
dump('FROZEN_REPLAY_CHECK.json', {'passed':passed,'max_abs_differences':maxdiff,'cells':cells,'scope':'replay of submitted 324-grid including historical inactive e5; not a new physical interpretation'})
print('FROZEN_REPLAY',passed,maxdiff,flush=True)
if not passed:
    raise RuntimeError('Frozen version gate failed; retained attempted replay, no further experiment')
weights=[]; ltargets=[]; lqs=[]
for row,tf,y in zip(records,arrays,ys):
    d,lt,lq,lw=diagnostic(tf,y); row.update(d)
    row.update({'route_tau03':'fast' if d['ess_ratio']>=0.3 else 'correction','route_tau05':'fast' if d['ess_ratio']>=0.5 else 'correction','route_tau07':'fast' if d['ess_ratio']>=0.7 else 'correction'})
    weights.append(lw); ltargets.append(lt); lqs.append(lq)
table('FINAL_GRID_ROUTING.csv', records)
np.savez_compressed(HERE/'FINAL_GRID_WEIGHTS.npz',log_weights=np.stack(weights),log_target=np.stack(ltargets),log_q=np.stack(lqs))
route_summary=[]
for group,par in [('null','')]+[('damage',p) for p in POS]:
    rs=[r for r in records if r['group']==group and r['parameter']==par]
    ratios=np.array([r['ess_ratio'] for r in rs])
    route_summary.append({'group':group,'parameter':par,'n':len(rs),'raw_ess_ratio_min':float(ratios.min()),'raw_ess_ratio_median':float(np.median(ratios)),'raw_ess_ratio_max':float(ratios.max()),'correction_n_tau05':int((ratios<0.5).sum())})
table('FINAL_GRID_ROUTING_SUMMARY.csv',route_summary)
print('FINAL_GRID_ROUTES',json.dumps(route_summary),flush=True)

paired=[]; pair_samples=[]; pair_observations=[]; pair_eps=[]; pair_lw=[]; pair_lt=[]; pair_lq=[]; modal=[]
for pi in range(6):
    p=f'p{pi+1:02d}'; idx=pi//2
    theta=np.ones(6); theta[idx]=0.9 if pi%2==0 else 0.75
    fe=np.loadtxt(OLD/p/f'cai1_{p}_c5282dbe_modal.txt')
    echo=np.loadtxt(OLD/p/f'cai1_{p}_c5282dbe_parameters.txt')
    assert fe.shape==(20,) and np.isfinite(fe).all() and (fe>0).all() and (np.diff(fe)>=0).all()
    assert echo.shape==(6,2) and np.allclose(echo[:,0],theta,rtol=0,atol=1e-8)
    assert np.allclose(echo[:,1],theta*2e10,rtol=1e-12,atol=0)
    su=surro(theta); rel=(su-fe)/fe
    modal.append({'point':p,'parameter':f'e{idx+1}','value':float(theta[idx]),'max_abs_rel_pct_20':float(100*np.abs(rel).max()),'max_abs_rel_pct_6':float(100*np.abs(rel[:6]).max())})
    for k in KAP:
        for rep in range(NREP):
            nseed=20260918+1000*pi+100*int(k*100)+rep
            fseed=50000+pi*1000+int(k*100)*100+rep
            eps=np.random.default_rng(nseed).standard_normal(20)
            yf=fe+k*np.abs(fe)*eps; ysurr=su+k*np.abs(su)*eps
            torch.manual_seed(fseed); tf=draw(yf)
            torch.manual_seed(fseed); ts=draw(ysurr)
            df,ltf,lqf,lwf=diagnostic(tf,yf); ds,lts,lqs_,lws=diagnostic(ts,ysurr)
            mf=tf.mean(0); ms=ts.mean(0); delta=mf-ms
            row={'point':p,'parameter':f'e{idx+1}','value':float(theta[idx]),'kappa':k,'rep':rep,'noise_seed':nseed,'fast_seed':fseed,'affected_mean_fe':float(mf[idx]),'affected_mean_sur':float(ms[idx]),'abs_delta_affected':float(abs(delta[idx])),'mean_abs_delta_active':float(np.abs(delta[:3]).mean()),'mean_abs_delta_inactive':float(np.abs(delta[3:]).mean()),'ess_fe':df['ess_ratio'],'ess_sur':ds['ess_ratio'],'route_changed_tau05':int((df['ess_ratio']<0.5)!=(ds['ess_ratio']<0.5))}
            for j in range(6): row.update({f'fe_mean_e{j+1}':float(mf[j]),f'sur_mean_e{j+1}':float(ms[j])})
            paired.append(row); pair_samples.append(np.stack([tf,ts])); pair_observations.append(np.stack([yf,ysurr])); pair_eps.append(eps)
            pair_lw.append(np.stack([lwf,lws])); pair_lt.append(np.stack([ltf,lts])); pair_lq.append(np.stack([lqf,lqs_]))
    print('PAIRED',p,'complete',flush=True)
table('FINAL_EXT_PAIRED_RESULTS.csv',paired)
table('FINAL_EXT_FE_MODAL_CHECK.csv',modal)
np.savez_compressed(HERE/'FINAL_EXT_PAIRED_ARRAYS.npz',samples=np.stack(pair_samples),observations=np.stack(pair_observations),noise=np.stack(pair_eps),log_weights=np.stack(pair_lw),log_target=np.stack(pair_lt),log_q=np.stack(pair_lq))
paired_summary=[]
for p in [f'p{i:02d}' for i in range(1,7)]:
    rs=[r for r in paired if r['point']==p]
    paired_summary.append({'point':p,'n_pairs':len(rs),'mean_abs_delta_affected':float(np.mean([r['abs_delta_affected'] for r in rs])),'max_abs_delta_affected':float(max(r['abs_delta_affected'] for r in rs)),'mean_abs_delta_active':float(np.mean([r['mean_abs_delta_active'] for r in rs])),'median_ess_fe':float(np.median([r['ess_fe'] for r in rs])),'median_ess_sur':float(np.median([r['ess_sur'] for r in rs])),'correction_fe_n':sum(r['ess_fe']<0.5 for r in rs),'correction_sur_n':sum(r['ess_sur']<0.5 for r in rs),'route_changes':sum(r['route_changed_tau05'] for r in rs)})
table('FINAL_EXT_PAIRED_SUMMARY.csv',paired_summary)
after={str(p):sha(p) for p in sources}
assert before==after, 'Source changed during verification'
assert git_state()==initial_git, 'Tracked files changed during verification'
summary={'frozen_replay_passed':passed,'frozen_replay_maxdiff':maxdiff,'grid_routes':route_summary,'paired':paired_summary,'paired_overall':{'n_pairs':len(paired),'mean_abs_delta_affected':float(np.mean([r['abs_delta_affected'] for r in paired])),'max_abs_delta_affected':float(max(r['abs_delta_affected'] for r in paired)),'route_changes_tau05':sum(r['route_changed_tau05'] for r in paired)},'sources_unchanged':True,'tracked_files_unchanged':True,'psis':'NOT_RUN_MISSING_DEPENDENCY','training':'NOT_RUN','MCMC':'NOT_RUN','FE':'NO_NEW_SOLVES'}
dump('VERIFICATION_SUMMARY.json',summary)
print('VERIFICATION_SUMMARY',json.dumps(summary),flush=True)
