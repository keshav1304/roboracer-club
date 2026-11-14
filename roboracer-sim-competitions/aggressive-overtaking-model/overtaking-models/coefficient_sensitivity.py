"""Controlled score-weight experiment; frozen demonstrated planner and reactive rival."""
import argparse,hashlib,importlib.util,json,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parent
VARIANTS={'original':(.8,.8),'no_D':(0.,.8),'no_U':(.8,0.),'double_D':(1.6,.8),'double_U':(.8,1.6)}
FROZEN=ROOT/'artifacts/learned_pressure_random_r1/source/learned_stage4.py'

def run(job,out):
 import run_experiment as ex
 from benchmark import offline_only
 from world import make_env
 from track_geometry import TrackGeometry
 from audit_pressure_results import metrics
 from learned_opponent import OpponentForecast
 from functools import partial
 source=FROZEN.read_text();old='score=cost-.8*outward+.8*uncertainty'
 assert source.count(old)==1
 source=source.replace(old,'score=cost-self.displacement_weight*outward+self.uncertainty_weight*uncertainty')
 source=source.replace("self.response_debug['accepted']+=1;choices.append((score,candidate))", "self.response_debug.setdefault('ranking_candidates',[]).append(dict(cost=cost,D=outward,U=uncertainty,score=score,entry=entry,padding=padding,merge=merge))\n            self.response_debug['accepted']+=1;choices.append((score,candidate))")
 ns={};exec(compile(source,str(FROZEN),'exec'),ns)
 ns['OpponentForecast']=partial(OpponentForecast,path=ROOT/'artifacts/learned_pressure_random_r1/source/opponent_forecast.npz')
 base=ns['LearnedStage4'];decisions=[]
 class Observed(base):
  def __init__(self,*a,**kw):
   super().__init__(*a,**kw);self.planner.displacement_weight,self.planner.uncertainty_weight=VARIANTS[job['variant']]
  def plan(self,*a,**kw):
   command=super().plan(*a,**kw);r=self.telemetry['response']
   if r.get('ranking_candidates'):
    choices=r['ranking_candidates'];picks={k:int(np.argmin([c['cost']-w[0]*c['D']+w[1]*c['U'] for c in choices])) for k,w in VARIANTS.items()}
    decisions.append(dict(time=self.response_time,selected=r.get('selected',False),candidates=choices,picks_on_same_state=picks))
   return command
 ex.MODELS[3]=Observed
 assert job['scenario']['opponent']=='reactive'
 with offline_only():
  env=make_env(seed=job['scenario']['seed']);g=TrackGeometry(env.unwrapped.track)
  result,frames,_,_=ex.episode(ex.Scenario(**job['scenario']),env=env,geometry=g,control_dt=.05)
  yielding=metrics(g,frames,result);env.close()
 payload=dict(job=job,result=result,yield_metrics=yielding,ranking_decisions=decisions,frames=frames)
 dest=Path(out)/(job['id']+'.json')
 with dest.open('x') as f:json.dump(payload,f,separators=(',',':'))
 summary=dict(id=job['id'],variant=job['variant'],kind=job['kind'],outcome=result['outcome'],passed_at=result['passed_at'],collision=result['collision'],yield_pass=yielding['lateral_yield_pass'],ranking_decisions=len(decisions),same_state_weight_sensitive_decisions=sum(len(set(d['picks_on_same_state'].values()))>1 for d in decisions))
 print(json.dumps(summary),flush=True);return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('artifacts/coefficient_sensitivity_20261113'));a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False)
 scenarios=[]
 # Known demonstrations deliberately provide cases where the extension activates.
 for i in (16,21):
  data=json.loads((ROOT/f'artifacts/learned_pressure_random_r1/case-{i:03d}-response.json').read_text())
  scenarios.append(('demonstration',data['job']['scenario']))
 rng=np.random.default_rng(13112026)
 for i in range(4):
  scenarios.append(('fresh',dict(stage=3,seed=131120260+i,s=float((i+rng.uniform(.1,.9))*59.029902/4),gap=float(rng.uniform(.85,1.4)),lateral=0.,ego_speed=2.4,opponent_speed=float(rng.choice([2.1,2.2])),seconds=40.,blocked=False,opponent='reactive',warmup_seconds=.4)))
 jobs=[dict(id=f'case-{i:02d}-{v}',variant=v,kind=kind,scenario=s) for i,(kind,s) in enumerate(scenarios) for v in VARIANTS]
 source=out/'source';source.mkdir()
 for name in ('coefficient_sensitivity.py','learned_opponent.py','expanded_planner.py','maneuver_core.py','response_stage4.py','world.py','run_experiment.py','audit_pressure_results.py','track_geometry.py','perception.py'):shutil.copy2(ROOT/name,source/name)
 shutil.copy2(FROZEN,source/'demonstrated_learned_stage4.py')
 for name in ('stage2.npz','stage2.value.npz','stage3.npz','stage3.value.npz'):shutil.copy2(ROOT/'artifacts/models'/name,source/name)
 shutil.copy2(ROOT/'artifacts/learned_pressure_random_r1/source/opponent_forecast.npz',source/'opponent_forecast.npz')
 (out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,weights=VARIANTS,all_opponents_reactive=True,no_training=True,api_calls=0,source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir()},note='Two known demonstrations plus four newly sampled initial states. Five weights share each initial condition. Only the D/U ranking coefficients vary; all feasibility gates remain fixed.'),indent=2))
 with ProcessPoolExecutor(max_workers=2) as pool:results=[f.result() for f in as_completed([pool.submit(run,j,str(out)) for j in jobs])]
 summary={}
 for v in VARIANTS:
  group=[r for r in results if r['variant']==v]
  summary[v]=dict(weights=VARIANTS[v],trials=len(group),passes=sum(r['outcome']=='pass' for r in group),contacts=sum(r['collision'] is not None for r in group),yield_passes=sum(r['yield_pass'] for r in group),ranking_decisions=sum(r['ranking_decisions'] for r in group),weight_sensitive_decisions=sum(r['same_state_weight_sensitive_decisions'] for r in group))
 (out/'summary.json').write_text(json.dumps(dict(summary=summary,trials=results),indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
