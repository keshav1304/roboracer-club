"""Paired trials against the unchanged reactive Stage 2 rival, exclusively."""
import argparse,hashlib,json,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parent

def run(job,out):
    assert job['scenario']['opponent']=='reactive' and job['mode']=='reactive'
    import run_experiment as ex
    from response_stage4 import ResponseStage4
    from learned_stage4 import LearnedStage4
    from world import make_env
    from track_geometry import TrackGeometry
    from benchmark import offline_only
    chosen=LearnedStage4 if job['policy']=='response' else ResponseStage4
    records=[];counts=dict(decisions=0,selected=0,candidates=0,accepted=0,aborts=0)
    class Observed(chosen):
        def plan(self,obs,*a,**kw):
            self.planner.capture_paths=True;command=super().plan(obs,*a,**kw)
            counts['decisions']+=1;data=self.telemetry['response']
            counts['selected']+=bool(data.get('selected'));counts['candidates']+=data.get('considered',0);counts['accepted']+=data.get('accepted',0)
            counts['aborts']=data.get('abort_count',0)
            if data.get('selected'):
                records.append(dict(time=job['scenario']['warmup_seconds']+(counts['decisions']-1)*.05,response=data.copy()))
            return command
    ex.MODELS[3]=Observed
    with offline_only():
        env=make_env(seed=job['scenario']['seed']);g=TrackGeometry(env.unwrapped.track)
        result,frames,_,_=ex.episode(ex.Scenario(**job['scenario']),env=env,geometry=g,control_dt=.05)
        t=np.array([f['t'] for f in frames]);ego=np.array([f['ego'] for f in frames]);opp=np.array([f['opponent'] for f in frames])
        from audit_pressure_results import metrics as observed_yield_metrics
        metrics=observed_yield_metrics(g,frames,result)
        errors=[]
        for r in records:
            p=r['response']['predictions'][0];times=p['times'];xy=np.array(p['path']);constant=np.array(r['response']['no_response_path'])
            for h in (.25,.5,1.):
                when=r['time']+h
                if when>t[-1] or h>times[-1]:continue
                truth=np.array([np.interp(when,t,opp[:,j]) for j in (0,1)])
                pred=np.array([np.interp(h,times,xy[:,j]) for j in (0,1)])
                base=np.array([np.interp(h,times,constant[:,j]) for j in (0,1)])
                errors.append(dict(horizon=h,error_m=float(np.linalg.norm(pred-truth)),constant_error_m=float(np.linalg.norm(base-truth))))
        env.close()
    payload=dict(job=job,result=result,response_counts=counts,yield_metrics=metrics,prediction_errors=errors,predictions=records,frames=frames)
    (Path(out)/(job['id']+'.json')).write_text(json.dumps(payload,separators=(',',':')))
    print(json.dumps(dict(id=job['id'],outcome=result['outcome'],pass_time=result['passed_at'],lateral_yield=metrics['lateral_yield'],outward_m=round(metrics['outward_displacement_m'],3),selected=counts['selected'])),flush=True)
    return {k:payload[k] for k in ('job','result','response_counts','yield_metrics','prediction_errors')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=8)
    p.add_argument('--seed',type=int,default=71103);p.add_argument('--seconds',type=float,default=35.)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--baseline-from',type=Path);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(a.seed);jobs=[]
    for i in range(a.count):
        s=(i+rng.uniform(.1,.9))*59.029902/a.count
        scenario=dict(stage=3,seed=a.seed*100+i,s=float(s),gap=float(rng.uniform(.85,1.4)),lateral=0.,ego_speed=2.4,
            opponent_speed=float(rng.choice([2.1,2.2])),seconds=a.seconds,blocked=False,opponent='reactive',warmup_seconds=.4)
        for policy in ('baseline','response'):jobs.append(dict(id=f'case-{i:03d}-{policy}',policy=policy,mode='reactive',scenario=scenario.copy()))
    source=out/'source';source.mkdir()
    for name in ('learned_opponent.py','learned_stage4.py','response_stage4.py','reactive_pressure_experiment.py','maneuver_core.py','expanded_planner.py','world.py','run_experiment.py','audit_pressure_results.py'):
        shutil.copy2(ROOT/name,source/name)
    shutil.copy2(ROOT/'artifacts/learned_reaction_v2/model.npz',source/'opponent_forecast.npz')
    (out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,controller="LearnedStage4",all_opponents_reactive=True,seed=a.seed,baseline_reused_from=str(a.baseline_from) if a.baseline_from else None,
        source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir()},api_calls=0),indent=2))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        pending=[];results=[]
        for job in jobs:
            if a.baseline_from and job['policy']=='baseline':
                original=a.baseline_from/(job['id']+'.json')
                data=json.loads(original.read_text())
                if data['job']!=job:raise ValueError('Baseline reuse requires an identical seeded scenario')
                shutil.copy2(original,out/original.name)
                results.append({k:data[k] for k in ('job','result','response_counts','yield_metrics','prediction_errors')})
            else:pending.append(pool.submit(run,job,str(out)))
        results.extend(f.result() for f in as_completed(pending))
    summary={}
    for policy in ('baseline','response'):
        group=[r for r in results if r['job']['policy']==policy];errors=[e for r in group for e in r['prediction_errors']]
        summary[policy]=dict(trials=len(group),passes=sum(r['result']['success'] for r in group),contacts=sum(r['result']['collision'] is not None for r in group),
            lateral_yields=sum(r['yield_metrics']['lateral_yield'] for r in group),lateral_yield_passes=sum(r['yield_metrics']['lateral_yield_pass'] for r in group),
            forecast_samples=len(errors),forecast_rmse_m=float(np.sqrt(np.mean([e['error_m']**2 for e in errors]))) if errors else None,
            constant_motion_rmse_m=float(np.sqrt(np.mean([e['constant_error_m']**2 for e in errors]))) if errors else None)
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
