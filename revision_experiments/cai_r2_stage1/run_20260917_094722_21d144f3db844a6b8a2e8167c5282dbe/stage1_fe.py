"""At most six authorized effective FE triples, isolated jobs, unchanged model body."""
import os,sys,json,csv,math,hashlib,subprocess,time,traceback,re
from pathlib import Path
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[2]
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ['TEMP','TMP']:
    p=RUN/'environment'/'tmp';p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
import numpy as np
import torch
torch.set_num_threads(6)
sys.path.insert(0,str(ROOT/'experiments'))
from cai_lib import load_surrogate
INTAKE=json.loads((RUN/'ASSET_INTAKE.json').read_text())
ASSETS={Path(r['path']).as_posix():r for r in INTAKE['assets']}
def snap(p):return RUN/ASSETS[Path(p).as_posix()]['snapshot']
def dump(path,obj):
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def csvout(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,restval='NA_NOT_APPLICABLE');w.writeheader();w.writerows(rows)
LOG=(RUN/'FE_DRIVER.log').open('x',encoding='utf-8')
def log(s):print(s,flush=True);LOG.write(s+'\n');LOG.flush()
CANDIDATES=[(.9,1,1),(.75,1,1),(1,.9,1),(1,.75,1),(1,1,.9),(1,1,.75)]
ANCHORS=[(1,1,1),(.5,1,1),(1,.5,1),(1,1,.5)]
ATOL=1e-8;FREQ_ATOL=1e-6
assert len(set(CANDIDATES))==6
SOURCE_PATHS=[
    ('dataset_train.csv','original_surrogate_training_input'),
    ('dataset_train_broad_clean.csv','clean_broad_surrogate_training_input'),
    ('dataset_train_broad_ext.csv','extended_surrogate_training_and_validation_input'),
    ('dataset_pilot.csv','FE_pilot'),('dataset_test.csv','original_surrogate_test_input'),
    ('dataset_extended.csv','FE_extension_raw'),('dataset_train_part2.csv','FE_training_part2_raw'),
    ('revision_experiments/E1_posterior_region_FE/fe_caseA.csv','existing_FE_A'),
    ('revision_experiments/E1_posterior_region_FE/fe_caseB.csv','existing_FE_B'),
    ('revision_experiments/E1_posterior_region_FE/fe_caseC.csv','existing_FE_C'),
    ('revision_experiments/task1_retrain/fe_caseB_clean.csv','existing_FE_clean_field'),
    ('revision_experiments/task1_retrain/fe_caseB_nuts.csv','existing_FE_clean_NUTS'),
    ('revision_experiments/task1_retrain/fe_theta_one.csv','existing_FE_nominal'),
    ('revision_experiments/task_r3_domain/fe_caseB_ext.csv','existing_FE_extended_field'),
    ('revision_experiments/task_r3_domain/fe_caseB_narrow.csv','existing_FE_extended_narrow'),
    ('revision_experiments/task_r3_domain/dataset_extend_new.csv','existing_FE_extended_training')]
MATCHES=[];DATA_ASSETS=[];data_errors=[]
for rel,kind in SOURCE_PATHS:
    p=ROOT/rel
    meta=dict(path=str(p),kind=kind,status='READ',sha256=digest(p) if p.is_file() else 'MISSING')
    if not p.is_file():meta.update(status='MISSING',reason='Required named asset does not exist');DATA_ASSETS.append(meta);continue
    parsed=[];bad=0;rows=0
    with p.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        for index,r in enumerate(reader):
            rows+=1
            try:
                if None in r or any(v is None for v in r.values()):raise ValueError('Malformed shifted CSV field count')
                theta=np.array([float(r[f'e{k}']) for k in range(1,7)])
                freq=np.array([float(r[f'f{k}']) for k in range(1,21)])
                if not np.isfinite(theta).all() or not np.isfinite(freq).all() or (freq<=0).any() or (np.diff(freq)<-1e-6).any():raise ValueError('Nonfinite/nonpositive/nonmonotone values')
                parsed.append((index,theta,freq))
            except Exception as e:
                bad+=1
                if len(data_errors)<30:data_errors.append(dict(path=str(p),row_index=index,reason=str(e)))
    ext_train=None
    if rel=='dataset_train_broad_ext.csv':
        idx=np.random.RandomState(42).permutation(rows);nv=rows//10;ext_train=set(idx[nv:].tolist())
    for index,theta,freq in parsed:
        for role,triples in [('candidate',CANDIDATES),('known_training_anchor',ANCHORS)]:
            for k,triple in enumerate(triples):
                if np.allclose(theta[:3],triple,rtol=0,atol=ATOL):
                    split='TRAIN' if 'training_input' in kind else 'FE_OR_TEST_ASSET'
                    if ext_train is not None:split='TRAIN' if index in ext_train else 'VALIDATION'
                    MATCHES.append(dict(role=role,point_index=k+1,effective_theta=list(triple),theta_stored=theta.tolist(),
                        frequencies_hz=freq.tolist(),path=str(p),kind=kind,row_index=index,split=split,
                        matching_basis='e1/e2/e3 only; e4/e5/e6 author-confirmed inactive',abs_tolerance=ATOL,
                        relative_tolerance=0,aligned_valid_20_modes=True))
    meta.update(total_rows=rows,valid_rows=len(parsed),invalid_rows=bad)
    DATA_ASSETS.append(meta)
dump(RUN/'FE_OVERLAP_AUDIT.json',dict(candidate_triples=CANDIDATES,known_anchors=ANCHORS,
    matches=MATCHES,assets=DATA_ASSETS,invalid_input_rows=data_errors,
    matching={'abs_tolerance':ATOL,'rtol':0,'dimensions':['e1','e2','e3'],'frequency_consistency_atol_hz':FREQ_ATOL},
    anchor_relation='Nominal and the three effective 0.5 axis points are author-confirmed training anchors regardless of re-solving.'))
models={}
for name,path in [('d6',ROOT/'surrogate_models/member_0.pt'),('ext',ROOT/'surrogate_models_broad_ext/member_0.pt')]:
    try:models[name]=load_surrogate(snap(path))
    except Exception as e:log(f'SURROGATE BLOCKED {name}: {e!r}')
original=snap(ROOT/'update.mac').read_text(encoding='utf-8',errors='replace')
lines=original.splitlines()
start=next(i for i,l in enumerate(lines) if l.strip().upper()=='/PREP7')
solve=next(i for i,l in enumerate(lines) if i>=start and l.strip().upper()=='SOLVE')
end=next(i for i,l in enumerate(lines) if i>solve and l.strip().upper()=='FINISH')
MODEL_BODY='\n'.join(lines[start:end+1])+'\n'
BODY_SHA=hashlib.sha256(MODEL_BODY.encode('utf-8')).hexdigest()
EXE=Path('D:/Program Files/ANSYS Inc/v160/ansys/bin/winx64/ANSYS160.exe')
(RUN/'FE_JOBS').mkdir(exist_ok=False)
rows=[];paired=[];NEW_JOBS=0
for k,triple in enumerate(CANDIDATES):
    theta=np.array(list(triple)+[1,1,1],dtype=float)
    matches=[m for m in MATCHES if m['role']=='candidate' and m['point_index']==k+1]
    training=[m for m in matches if m['split']=='TRAIN']
    any_training=[m for m in matches if 'surrogate_training' in m['kind']]
    freq=None;status='NOT_EXECUTED';reason='NOT_STARTED';source='NA';job='NA';wd='NA';command=[]
    conflicts=False
    if matches:
        bank=np.array([m['frequencies_hz'] for m in matches])
        conflicts=float(np.max(np.ptp(bank,axis=0)))>FREQ_ATOL
        if not conflicts:
            # Prefer a directly harvested FE CSV or the clean extended dataset.
            preferred=sorted(matches,key=lambda m:('existing_FE' not in m['kind'],'extended' not in m['kind']))[0]
            freq=np.array(preferred['frequencies_hz']);status='REUSED_EXISTING_VALID_FE';reason='Aligned positive ordered 20-mode CSV; all same-effective-point outputs agree within frequency tolerance'
            source=f'{preferred["path"]}:row_index={preferred["row_index"]}'
    if freq is None:
        job=f'cai1_p{k+1:02d}_{RUN.name[-8:]}'
        wd=RUN/'FE_JOBS'/job;wd.mkdir(exist_ok=False)
        head='/uis,msgpop,3\n/nerr,5,10000,\n/FILNAME,'+job+',0\n/TITLE,opti-analysis\n'
        for d,x in enumerate(theta):head+=f'*SET,stage_e{d+1},{x:.12g}\n*SET,e{d+1},stage_e{d+1}*2e10\n'
        head+=f'*CFOPEN,{job}_parameters,txt\n'
        for d in range(6):head+=f'*VWRITE,stage_e{d+1},e{d+1}\n(F14.8,1X,E24.14)\n'
        head+='*CFCLOS\n'
        tail='\n/POST1\nSET,LIST\n*DIM,stage_freqs,ARRAY,20,1\n*DO,stage_im,1,20\nSET,1,stage_im\n*GET,stage_freqs(stage_im,1),ACTIVE,0,SET,FREQ\n*ENDDO\n'
        tail+=f'*CFOPEN,{job}_modal,txt\n*VWRITE,stage_freqs(1,1)\n(F20.10)\n*CFCLOS\nFINISH\n/EXIT,NOSAVE\n'
        actual=head+MODEL_BODY+tail
        with (wd/'run.mac').open('x',encoding='utf-8') as f:f.write(actual)
        params=dict(theta_dimensionless=theta.tolist(),E_Pa=(theta*2e10).tolist(),job=job,
            original_APDL=str(ROOT/'update.mac'),original_sha256=ASSETS[(ROOT/'update.mac').as_posix()]['sha256'],
            unchanged_PREP7_through_SOLVE_FINISH_sha256=BODY_SHA,
            wrapper_changes='Inject parameter values, job/title and parameter echo; remove external parameter/measurement macro IO; append frequency extraction. Original model/mesh/material assignment/solution body verbatim.',
            training_matches=training,input_dataset_matches=matches,existing_frequency_conflict=conflicts)
        dump(wd/'PARAMETERS.json',params)
        modal=wd/(job+'_modal.txt');echo=wd/(job+'_parameters.txt')
        assert not modal.exists() and not echo.exists()
        command=[str(EXE),'-b','-i','run.mac','-o','solver.log','-j',job,'-np','1','-dir',str(wd)]
        dump(wd/'LAUNCH.json',dict(command=command,cwd=str(wd),status='PREPARED',timeout_seconds=600,
            actual_APDL_sha256=digest(wd/'run.mac'),expected_output=str(modal),outputs_absent_before_launch=True))
        t0=time.time();launch=dict(command=command,cwd=str(wd),started_epoch=t0)
        if EXE.is_file():
            NEW_JOBS+=1;assert NEW_JOBS<=6
            log(f'FE LAUNCH {k+1}: {command!r}')
            env=os.environ.copy();env['TEMP']=str(wd);env['TMP']=str(wd)
            try:
                proc=subprocess.run(command,cwd=wd,env=env,capture_output=True,timeout=600)
                with (wd/'process_stdout.log').open('xb') as f:f.write(proc.stdout)
                with (wd/'process_stderr.log').open('xb') as f:f.write(proc.stderr)
                launch.update(returncode=proc.returncode,elapsed_seconds=time.time()-t0)
                solver=(wd/'solver.log').read_text(errors='replace') if (wd/'solver.log').exists() else ''
                error_lines=[l for l in solver.splitlines() if '*** ERROR ***' in l or 'FATAL' in l.upper()]
                launch['solver_error_lines']=error_lines
                if proc.returncode!=0:raise RuntimeError(f'ANSYS returned {proc.returncode}; '+repr(error_lines[:10]))
                if error_lines:raise RuntimeError('ANSYS solver error messages: '+repr(error_lines[:10]))
                if not modal.is_file():raise RuntimeError('This job did not produce its unique frequency file')
                if modal.stat().st_mtime<t0-2:raise RuntimeError('Frequency timestamp predates this launch')
                vals=np.array([float(v) for v in modal.read_text().split()])
                if len(vals)!=20 or not np.isfinite(vals).all() or (vals<=0).any() or (np.diff(vals)<-1e-6).any():raise RuntimeError('Invalid 20-mode frequency output')
                if not echo.is_file():raise RuntimeError('Job parameter echo missing')
                ev=np.array([[float(v) for v in l.split()] for l in echo.read_text().splitlines() if l.strip()])
                if ev.shape!=(6,2) or not np.allclose(ev[:,0],theta,rtol=0,atol=1e-8) or not np.allclose(ev[:,1],theta*2e10,rtol=1e-10,atol=1):raise RuntimeError('Actual APDL parameter/modulus echo mismatch')
                completed=bool(re.search(r'RUN\s+COMPLETED',solver,re.I))
                if not completed:raise RuntimeError('ANSYS completion marker missing; output evidence retained')
                freq=vals;status='COMPLETED_NEW_FE';reason='Unique output created after launch, echoed parameters validated, no solver errors, successful return and completion marker'
                source=str(modal);launch.update(status=status,reason=reason,frequencies_hz=freq.tolist(),parameter_echo=ev.tolist(),output_sha256=digest(modal),run_completed_marker=completed)
            except subprocess.TimeoutExpired as e:
                status='FAILED_TIMEOUT';reason=repr(e);launch.update(status=status,reason=reason,elapsed_seconds=time.time()-t0)
                for filename,blob in [('process_stdout.log',e.stdout or b''),('process_stderr.log',e.stderr or b'')]:
                    with (wd/filename).open('xb') as f:f.write(blob)
            except Exception as e:
                status='FAILED_NEW_FE';reason=repr(e);launch.update(status=status,reason=reason,traceback=traceback.format_exc(),elapsed_seconds=time.time()-t0)
        else:status='NOT_EXECUTED_MISSING_SOLVER';reason='Named ANSYS executable unavailable';launch.update(status=status,reason=reason)
        dump(wd/'COMPLETION.json',launch)
    r=dict(candidate=f'p{k+1:02d}',**{f'e{d+1}':float(theta[d]) for d in range(6)},
        **{f'E{d+1}_Pa':float(theta[d]*2e10) for d in range(6)},status=status,reason=reason,
        FE_source=source,job=job,working_directory=str(wd),
        effective_match_atol=ATOL,matching_dimensions='e1,e2,e3',
        training_overlap=bool(training),training_or_validation_input_overlap=bool(any_training),
        training_match_records=json.dumps(training),all_match_records=json.dumps(matches),
        existing_frequency_conflict=conflicts,independent_physical_holdout=not bool(any_training),
        new_solve=status=='COMPLETED_NEW_FE',identification_sigma_status='NOT_OBTAINED',
        max_absolute_difference_hz_ext='NA_FE_OR_SURROGATE_UNAVAILABLE',max_relative_difference_ext='NA_FE_OR_SURROGATE_UNAVAILABLE')
    for j in range(20):r[f'FE_f{j+1}_Hz']=float(freq[j]) if freq is not None else 'NA_FAILED_OR_NOT_EXECUTED'
    for name,sur in models.items():
        with torch.no_grad():pred=sur(torch.tensor(theta,dtype=torch.float32).reshape(1,-1)).squeeze(0).numpy().astype(float)
        if freq is not None:
            if name=='ext':r.update(max_absolute_difference_hz_ext=float(np.max(np.abs(pred-freq))),max_relative_difference_ext=float(np.max(np.abs(pred-freq)/freq)))
            for j in range(20):
                scale=math.sqrt(.001)*freq[j]
                paired.append(dict(candidate=r['candidate'],model=name,mode=j+1,status='COMPLETED_PAIRING',reason='OK',
                    training_overlap=r['training_overlap'],training_or_validation_input_overlap=r['training_or_validation_input_overlap'],
                    FE_status=status,FE_source=source,FE_frequency_hz=float(freq[j]),surrogate_frequency_hz=float(pred[j]),
                    delta_surrogate_minus_FE_hz=float(pred[j]-freq[j]),relative_delta=float((pred[j]-freq[j])/freq[j]),
                    comparison_likelihood_sigma_hz=float(scale),standardized_delta=float((pred[j]-freq[j])/scale),
                    comparison_noise_definition='Inference-form kappa_sq=0.001 with this FE frequency as y; comparison scale only, not FFT identification uncertainty',
                    field_likelihood_sigma_hz=float(math.sqrt(.001)*[.91,1.08,1.47,2.55,3.16,3.81][j]) if j<6 else 'NA_NOT_IN_FIELD_LIKELIHOOD',
                    identification_sigma_status='NOT_OBTAINED'))
        else:
            for j in range(20):paired.append(dict(candidate=r['candidate'],model=name,mode=j+1,status='NOT_EXECUTED_PAIRING',reason=reason,
                surrogate_frequency_hz=float(pred[j]),FE_frequency_hz='NA_FE_FAILED',relative_delta='NA_FE_FAILED',identification_sigma_status='NOT_OBTAINED'))
    rows.append(r);log(f'FE RESULT p{k+1:02d}: {status}; training overlap={bool(training)}; reason={reason}')
csvout(RUN/'FE_VALIDATION_POINTS.csv',rows)
csvout(RUN/'FE_SURROGATE_MODAL_PAIRING.csv',paired)
dump(RUN/'FE_EXECUTION_SUMMARY.json',dict(launched_jobs=NEW_JOBS,completed_new_jobs=sum(r['status']=='COMPLETED_NEW_FE' for r in rows),
    reused_points=sum(r['status']=='REUSED_EXISTING_VALID_FE' for r in rows),failed_or_unexecuted=sum(r['status'] not in ['COMPLETED_NEW_FE','REUSED_EXISTING_VALID_FE'] for r in rows),
    points=rows,unchanged_model_body_sha256=BODY_SHA,
    null_design={'fixed_nominal_plus_noise':{'FE_requirement':'One trustworthy nominal FE output plus six effective damage triples; same FE model and extraction source for both null and damage',
        'nominal_training_anchor':True,'new_nominal_solve_in_stage1':False,
        'existing_nominal_asset':'revision_experiments/task1_retrain/fe_theta_one.csv',
        'cost':'Cheap reuse across noise replicates; only tests a conditional nominal null, not prior population variability.'},
        'original_prior_drawn_null':{'status':'NOT_EXECUTED_REQUIRES_SEPARATE_AUTHORIZATION',
            'source':'task_r3_domain/e3_ext_tpr.py: NNULL=120 for each kappa in [0.01,0.03,0.05]',
            'parameter_distribution':'Six independent lognormal(0,0.55), clamp [0.5,3.5]; physically deduplicate by e1/e2/e3',
            'additional_FE_requirement':'Exact original design: up to 3*120=360 distinct effective FE triples (minus verified existing overlaps). No such jobs executed.',
            'alternative':'A newly specified common 120-effective-triple null bank could reuse FE across noise levels, but changes the original random null protocol and must be authorized separately.',
            'cost':'Broad population null requires substantially more FE work; posterior-score changes in inactive e4-e6 cannot be treated as physical damage.'}}))
LOG.close()
