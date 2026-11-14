"""Paired real-Gym evaluation: demonstrated Stage 4 vs joint geometry/speed rescue."""
import argparse,hashlib,json,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parent

def run(job,out):
 import run_experiment as ex
 from joint_path_stage4 import JointStage4,BaselineStage4
 from world import make_env
 from track_geometry import TrackGeometry
 from benchmark import offline_only
 from audit_pressure_results import metrics
 chosen=JointStage4 if job['policy']=='response' else BaselineStage4
 counts=dict(decisions=0,selected=0,candidates=0,accepted=0,aborts=0,geometries=0,timed_candidates=0,joint_feasible=0,rescued_sides=0,joint_selected=0)
 reasons={};joint_selections=[]
 class Observed(chosen):
  def plan(self,*a,**kw):
   command=super().plan(*a,**kw);counts['decisions']+=1
   r=self.telemetry.get('response',{});counts['selected']+=bool(r.get('selected'));counts['candidates']+=r.get('considered',0);counts['accepted']+=r.get('accepted',0);counts['aborts']=r.get('abort_count',0)
   j=self.telemetry.get('joint_search',{})
   for k in ('geometries','timed_candidates','rescued_sides'):counts[k]+=j.get(k,0)
   counts['joint_feasible']+=j.get('feasible',0);counts['joint_selected']+=bool(j.get('selected'))
   for k,v in j.get('reasons',{}).items():reasons[k]=reasons.get(k,0)+v
   if j.get('selected'):
    joint_selections.append(dict(decision=counts['decisions'],action=self.action,command=list(command),target=self.telemetry.get('target'),profile=j['selected_profile']))
   return command
 ex.MODELS[3]=Observed
 assert job['scenario']['opponent']=='reactive'
 with offline_only():
  env=make_env(seed=job['scenario']['seed']);g=TrackGeometry(env.unwrapped.track)
  result,frames,_,_=ex.episode(ex.Scenario(**job['scenario']),env=env,geometry=g,control_dt=.05)
  ym=metrics(g,frames,result);env.close()
 payload=dict(job=job,result=result,response_counts=counts,yield_metrics=ym,joint_rejections=reasons,joint_selections=joint_selections,frames=frames)
 with (Path(out)/(job['id']+'.json')).open('x') as f:json.dump(payload,f,separators=(',',':'))
 summary=dict(id=job['id'],policy=job['policy'],outcome=result['outcome'],pass_time=result['passed_at'],collision=result['collision'],yield_pass=ym['lateral_yield_pass'],joint_feasible=counts['joint_feasible'],rescued_sides=counts['rescued_sides'],p95_ms=result['controller_latency']['p95_ms'])
 print(json.dumps(summary),flush=True);return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=6);p.add_argument('--seed',type=int,default=161113);p.add_argument('--seconds',type=float,default=35.);p.add_argument('--workers',type=int,default=2);p.add_argument('--baseline-from',type=Path);a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False)
 rng=np.random.default_rng(a.seed);jobs=[]
 for i in range(a.count):
  scene=dict(stage=3,seed=a.seed*100+i,s=float((i+rng.uniform(.1,.9))*59.029902/a.count),gap=float(rng.uniform(.85,1.4)),lateral=0.,ego_speed=2.4,opponent_speed=float(rng.choice([2.1,2.2])),seconds=a.seconds,blocked=False,opponent='reactive',warmup_seconds=.4)
  for policy in ('baseline','response'):jobs.append(dict(id=f'case-{i:03d}-{policy}',policy=policy,mode='reactive',scenario=scene))
 source=out/'source';source.mkdir()
 for name in ('joint_path_stage4.py','joint_path_experiment.py','learned_opponent.py','expanded_planner.py','response_stage4.py','maneuver_core.py','world.py','run_experiment.py','track_geometry.py','perception.py','test_joint_path_stage4.py'):shutil.copy2(ROOT/name,source/name)
 for name in ('stage2.npz','stage2.value.npz','stage3.npz','stage3.value.npz'):shutil.copy2(ROOT/'artifacts/models'/name,source/name)
 shutil.copy2(ROOT/'artifacts/learned_pressure_random_r1/source/learned_stage4.py',source/'baseline_learned_stage4.py')
 shutil.copy2(ROOT/'artifacts/learned_reaction_v2/model.npz',source/'opponent_forecast.npz')
 (out/'manifest.json').write_text(json.dumps(dict(controller='JointStage4',sampling_seed=a.seed,purpose='development' if 'dev' in out.name else 'held-out evaluation',jobs=jobs,all_opponents_reactive=True,api_calls=0,source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir()}),indent=2))
 results=[];pending=[]
 for job in jobs:
  if a.baseline_from and job['policy']=='baseline':
   previous=a.baseline_from/(job['id']+'.json');data=json.loads(previous.read_text())
   if data['job']!=job:raise ValueError('Reused baseline scenario differs')
   old=json.loads((a.baseline_from/'manifest.json').read_text())
   for name in ('baseline_learned_stage4.py','opponent_forecast.npz','stage2.npz','stage2.value.npz','stage3.npz','stage3.value.npz','expanded_planner.py','response_stage4.py','maneuver_core.py','world.py','run_experiment.py','track_geometry.py','perception.py','learned_opponent.py'):
    if old['source_sha256'][name]!=hashlib.sha256((source/name).read_bytes()).hexdigest():raise ValueError('Reused baseline implementation differs: '+name)
   shutil.copy2(previous,out/previous.name);r=data['result'];c=data['response_counts']
   results.append(dict(id=job['id'],policy='baseline',outcome=r['outcome'],pass_time=r['passed_at'],collision=r['collision'],yield_pass=data['yield_metrics']['lateral_yield_pass'],joint_feasible=0,rescued_sides=0,p95_ms=r['controller_latency']['p95_ms']))
  else:pending.append(job)
 if a.baseline_from:(out/'baseline_reuse.json').write_text(json.dumps(dict(source=str(a.baseline_from.resolve()),note='Identical scenarios and baseline source/checkpoint hashes checked; timing is historical.'),indent=2))
 with ProcessPoolExecutor(max_workers=a.workers) as pool:results.extend(f.result() for f in as_completed([pool.submit(run,j,str(out)) for j in pending]))
 summary={}
 for policy in ('baseline','response'):
  group=[r for r in results if r['policy']==policy];summary[policy]=dict(trials=len(group),passes=sum(r['outcome']=='pass' for r in group),contacts=sum(r['collision'] is not None for r in group),yield_passes=sum(r['yield_pass'] for r in group),joint_feasible=sum(r['joint_feasible'] for r in group),rescued_sides=sum(r['rescued_sides'] for r in group))
 (out/'summary.json').write_text(json.dumps(dict(summary=summary,trials=results),indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
