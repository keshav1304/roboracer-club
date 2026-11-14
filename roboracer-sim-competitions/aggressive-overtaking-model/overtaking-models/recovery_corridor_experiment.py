"""Paired Gym evaluation against the previous wider CorridorStage4."""
import argparse,hashlib,json,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parent

def run(job,out):
 import run_experiment as ex
 from recovery_corridor_stage4 import RecoveryCorridorStage4
 from corridor_stage4 import CorridorStage4
 from world import make_env
 from track_geometry import TrackGeometry
 from benchmark import offline_only
 from audit_pressure_results import metrics
 chosen=RecoveryCorridorStage4 if job['policy']=='response' else CorridorStage4
 counts=dict(decisions=0,active=0,overrides=0,recovery=0,brake=0,released_beams=0,joint_selected=0)
 observations=[]
 class Observed(chosen):
  def plan(self,obs,*a,**kw):
   command=super().plan(obs,*a,**kw);counts['decisions']+=1
   d=self.telemetry.get('interaction',{});counts['active']+=bool(d.get('active'));counts['overrides']+=bool(d.get('override'))
   counts['recovery']+=d.get('mode') in ('checked lane recovery','checked wall corridor');counts['brake']+=d.get('mode')=='brake in current lane'
   counts['released_beams']+=d.get('padding_beams_released',0);counts['joint_selected']+=bool(self.telemetry.get('joint_search',{}).get('selected'))
   if counts['decisions']%5==1 or d.get('override'):
    target=self.sensor.target
    observations.append(dict(decision=counts['decisions'],time=job['scenario']['warmup_seconds']+(counts['decisions']-1)*.05,ego=[float(obs[k]) for k in ('pose_x','pose_y','pose_theta','linear_vel_x')],target=None if target is None else vars(target).copy(),interaction=d,command=list(command),action=self.action))
   return command
 ex.MODELS[3]=Observed
 assert job['scenario']['opponent']=='reactive'
 with offline_only():
  env=make_env(seed=job['scenario']['seed']);g=TrackGeometry(env.unwrapped.track)
  result,frames,_,_=ex.episode(ex.Scenario(**job['scenario']),env=env,geometry=g,control_dt=.05)
  ym=metrics(g,frames,result);env.close()
 data=dict(job=job,result=result,interaction_counts=counts,observations=observations,frames=frames,yield_metrics=ym)
 with (Path(out)/(job['id']+'.json')).open('x') as f:json.dump(data,f,separators=(',',':'),default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())
 row=dict(id=job['id'],policy=job['policy'],outcome=result['outcome'],pass_time=result['passed_at'],collision=result['collision'],counts=counts,p95_ms=result['controller_latency']['p95_ms'])
 print(json.dumps(row),flush=True);return row

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--cases',type=Path);p.add_argument('--indices',default='4,5,1,7');p.add_argument('--count',type=int,default=24);p.add_argument('--seed',type=int,default=891114);p.add_argument('--seconds',type=float,default=40.);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
 out=a.output.resolve();out.mkdir(exist_ok=False);jobs=[]
 if a.cases:
  for i in map(int,a.indices.split(',')):
   d=json.loads((a.cases/f'case-{i:03d}-response.json').read_text());scene=d['job']['scenario']
   job=dict(id=f'case-{i:03d}-baseline',policy='baseline',mode='reactive',scenario=scene)
   d['baseline_origin']=str((a.cases/f'case-{i:03d}-response.json').resolve());d['job']=job
   (out/(job['id']+'.json')).write_text(json.dumps(d,separators=(',',':')))
   jobs.extend([job,dict(id=f'case-{i:03d}-response',policy='response',mode='reactive',scenario=scene)])
 else:
  rng=np.random.default_rng(a.seed)
  for i in range(a.count):
   scene=dict(stage=3,seed=a.seed*100+i,s=float((i+rng.uniform(.1,.9))*59.029902/a.count),gap=float(rng.uniform(.85,1.4)),lateral=0.,ego_speed=2.4,opponent_speed=float(rng.choice([2.1,2.2])),seconds=a.seconds,blocked=False,opponent='reactive',warmup_seconds=.4)
   for policy in ('baseline','response'):jobs.append(dict(id=f'case-{i:03d}-{policy}',policy=policy,mode='reactive',scenario=scene))
 source=out/'source';source.mkdir()
 names=('recovery_corridor_stage4.py','recovery_corridor_experiment.py','robust_corridor_stage4.py','robust_corridor_experiment.py','corridor_stage4.py','corridor_experiment.py','interaction_stage4.py','interaction_experiment.py','joint_path_stage4.py','learned_opponent.py','expanded_planner.py','response_stage4.py','maneuver_core.py','world.py','run_experiment.py','track_geometry.py','perception.py','test_interaction_stage4.py','test_corridor_stage4.py','test_robust_corridor_stage4.py')
 for name in names:shutil.copy2(ROOT/name,source/name)
 for name in ('stage2.npz','stage2.value.npz','stage3.npz','stage3.value.npz'):shutil.copy2(ROOT/'artifacts/models'/name,source/name)
 shutil.copy2(ROOT/'artifacts/learned_pressure_random_r1/source/learned_stage4.py',source/'baseline_learned_stage4.py')
 shutil.copy2(ROOT/'artifacts/learned_reaction_v2/model.npz',source/'opponent_forecast.npz')
 hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir()}
 if a.cases:
  old=json.loads((a.cases/'manifest.json').read_text())['source_sha256']
  for name,digest in hashes.items():
   if name in old:assert old[name]==digest, 'Reused baseline code or checkpoint differs: '+name
 manifest=dict(controller='RecoveryCorridorStage4',baseline='RobustCorridorStage4' if a.cases else 'CorridorStage4',jobs=jobs,source_sha256=hashes,api_calls=0,all_opponents_reactive=True,purpose='development: previously observed cases' if a.cases else 'fresh paired validation',baseline_reuse=str(a.cases) if a.cases else None)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 pending=[j for j in jobs if not a.cases or j['policy']=='response']
 with ProcessPoolExecutor(max_workers=a.workers) as pool:results=[f.result() for f in as_completed([pool.submit(run,j,str(out)) for j in pending])]
 summary={}
 for policy in ('baseline','response'):
  data=[json.loads((out/(j['id']+'.json')).read_text()) for j in jobs if j['policy']==policy]
  summary[policy]=dict(trials=len(data),passes=sum(d['result']['success'] for d in data),contacts=sum(d['result']['collision'] is not None for d in data),timeouts=sum(d['result']['outcome']=='no_pass' for d in data))
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
