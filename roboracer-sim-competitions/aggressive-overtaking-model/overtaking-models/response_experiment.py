"""Paired, offline Gym experiments for conditional inside challenges. No HTML."""
import argparse,json,hashlib,shutil
from dataclasses import replace
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parent

def run(job,out):
    import run_experiment as ex
    from world import make_stage2_opponent
    ex.make_stage2_opponent=make_stage2_opponent
    from expanded_planner import ExpandedStage4
    from response_stage4 import ResponseStage4
    from benchmark import offline_only
    records=[];counts={'decisions':0,'candidates':0,'accepted':0,'selected':0,'max_considered':0}
    cls=ResponseStage4 if job['policy']=='response' else ExpandedStage4
    class Observed(cls):
        def plan(self,obs,*a,**kw):
            self.planner.capture_paths=True
            command=super().plan(obs,*a,**kw)
            counts['decisions']+=1
            data=self.telemetry.get('response',{})
            counts['candidates']+=data.get('considered',0);counts['accepted']+=data.get('accepted',0)
            counts['max_considered']=max(counts['max_considered'],data.get('considered',0))
            counts['selected']+=bool(data.get('selected',False))
            if data.get('selected') and counts['decisions']%5==1:
                records.append({'time':job['scenario'].get('warmup_seconds',0.)+(counts['decisions']-1)*.05,
                    'response':data.copy(),'ego':[float(obs[k]) for k in ('pose_x','pose_y','pose_theta','linear_vel_x')],
                    'command':list(command)})
            return command
    ex.MODELS[3]=Observed
    if job['mode']=='delayed':
        factory=ex.make_stage2_opponent
        def delayed(*args,**kwargs):
            obj=factory(*args,**kwargs);old=obj.plan;queue=[]
            def plan(obs,*a,**kw):
                command=old(obs,*a,**kw);queue.append(command)
                return queue.pop(0) if len(queue)>5 else (obj.steering,float(obs['linear_vel_x']))
            obj.plan=plan;return obj
        ex.make_stage2_opponent=delayed
    with offline_only():r,frames,_,_=ex.episode(ex.Scenario(**job['scenario']),control_dt=.05)
    errors=[]
    if frames:
        times=np.array([f['t'] for f in frames]);truth=np.array([f['opponent'][:2] for f in frames])
        for record in records:
            pred=record['response']['predictions'][0]
            for horizon in (.5,1.):
                when=record['time']+horizon
                if when>times[-1] or horizon>pred['times'][-1]:continue
                actual=np.array([np.interp(when,times,truth[:,j]) for j in (0,1)])
                predicted=np.array([np.interp(horizon,pred['times'],np.array(pred['path'])[:,j]) for j in (0,1)])
                constant=np.array([np.interp(horizon,pred['times'],np.array(record['response']['no_response_path'])[:,j]) for j in (0,1)])
                errors.append(dict(horizon=horizon,response_error_m=float(np.linalg.norm(actual-predicted)),constant_error_m=float(np.linalg.norm(actual-constant))))
    payload=dict(job=job,result=r,response_counts=counts,prediction_errors=errors,predictions=records,frames=frames)
    Path(out,job['id']+'.json').write_text(json.dumps(payload,separators=(',',':')))
    print(json.dumps(dict(id=job['id'],outcome=r['outcome'],pass_time=r['passed_at'],selected=counts['selected'],accepted=counts['accepted'],p95_ms=r['controller_latency']['p95_ms'])),flush=True)
    return {k:payload[k] for k in ('job','result','response_counts','prediction_errors')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--local-validation',action='store_true');p.add_argument('--validation',action='store_true');p.add_argument('--targeted',action='store_true');p.add_argument('--development',action='store_true');p.add_argument('--stress',action='store_true');a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for i,s in enumerate((4.,19.,34.,49.)):
        for j,cap in enumerate((2.1,2.2)):
            if a.development and j==0:continue
            scenario=dict(stage=3,seed=41000+i*10+j*2,s=s,gap=1.25,lateral=0.,ego_speed=2.4,opponent_speed=cap,
                          seconds=30. if a.development else 45.,blocked=False,opponent='reactive',warmup_seconds=.4)
            for policy in ('baseline','response'):
                jobs.append(dict(id=f'{i}-{j}-{policy}',scenario=scenario.copy(),policy=policy,mode='reactive'))
            if a.stress and j==1:
                for mode in ('nonreactive','delayed'):
                    case=scenario.copy()
                    if mode=='nonreactive':case['opponent']='line'
                    jobs.append(dict(id=f'{i}-{j}-response-{mode}',scenario=case,policy='response',mode=mode))
    if a.targeted:
        jobs=[]
        for i,(s,gap) in enumerate(((44.2,.85),(45.,.85),(49.3,.85),(50.2,1.05))):
            case=dict(stage=3,seed=92000+i,s=s,gap=gap,lateral=0.,ego_speed=2.4,opponent_speed=2.2,
                      seconds=25.,blocked=False,opponent='reactive',warmup_seconds=.4)
            for policy in ('baseline','response'):
                jobs.append(dict(id=f'target-{i}-{policy}',scenario=case.copy(),policy=policy,mode='reactive'))
            if a.stress:
                for mode in ('nonreactive','delayed'):
                    scenario=case.copy()
                    if mode=='nonreactive':scenario['opponent']='line'
                    jobs.append(dict(id=f'target-{i}-response-{mode}',scenario=scenario,policy='response',mode=mode))
    if a.validation:
        jobs=[];rng=np.random.default_rng(73463)
        for i in range(4):
            scenario=dict(stage=3,seed=973463+i,s=float(rng.uniform(0.,59.)),gap=float(rng.uniform(.85,1.25)),
                lateral=0.,ego_speed=2.4,opponent_speed=float(rng.choice([2.1,2.2])),seconds=45.,blocked=False,
                opponent='reactive',warmup_seconds=.4)
            for policy in ('baseline','response'):
                jobs.append(dict(id=f'validation-{i}-{policy}',scenario=scenario.copy(),policy=policy,mode='reactive'))
    if a.local_validation:
        jobs=[];rng=np.random.default_rng(884201)
        for i in range(4):
            scenario=dict(stage=3,seed=988420+i,s=float(rng.uniform(49.7,50.7)),gap=float(rng.uniform(.95,1.15)),
                lateral=0.,ego_speed=2.4,opponent_speed=2.2,seconds=25.,blocked=False,
                opponent='reactive',warmup_seconds=.4)
            for policy in ('baseline','response'):
                jobs.append(dict(id=f'local-{i}-{policy}',scenario=scenario.copy(),policy=policy,mode='reactive'))
    frozen=out/'source';frozen.mkdir()
    for name in ('response_stage4.py','response_experiment.py','maneuver_core.py','expanded_planner.py'):
        shutil.copy2(ROOT/name,frozen/name)
    (out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,api_calls=0,networks_retrained=False,
        source_sha256={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in frozen.iterdir()}),indent=2))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results=[f.result() for f in as_completed([pool.submit(run,j,str(out)) for j in jobs])]
    summary={}
    for mode in ('reactive','nonreactive','delayed'):
        for policy in ('baseline','response'):
            group=[r for r in results if r['job']['mode']==mode and r['job']['policy']==policy]
            if not group:continue
            errors=[e for r in group for e in r['prediction_errors']]
            summary[mode+'-'+policy]=dict(trials=len(group),passes=sum(r['result']['success'] for r in group),
                contacts=sum(r['result']['collision'] is not None for r in group),
                selected_challenges=sum(r['response_counts']['selected'] for r in group),
                prediction_samples=len(errors),response_rmse_m=float(np.sqrt(np.mean([e['response_error_m']**2 for e in errors]))) if errors else None,
                unchanged_motion_rmse_m=float(np.sqrt(np.mean([e['constant_error_m']**2 for e in errors]))) if errors else None)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
