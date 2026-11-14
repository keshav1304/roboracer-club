"""Serial whole-controller timings on identical recorded ego/LiDAR streams.

This isolates compute; replayed commands do not drive physics in the timing
portion. Closed-loop success is evaluated separately in real Gym episodes.
"""
import argparse,json,time,gc,hashlib,platform
from pathlib import Path
import numpy as np
import run_experiment as ex
from world import make_env
from track_geometry import TrackGeometry
from corridor_stage4 import CorridorStage4
from hybrid_corridor_stage4 import HybridCorridorStage4
from benchmark import offline_only
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
streams=[]
with offline_only():
 for case in (0,7,14):
  job=json.loads(Path(f'artifacts/corridor_validation_20261114/case-{case:03d}-response.json').read_text())['job'];scene=dict(job['scenario']);scene['seconds']=6.
  inputs=[]
  class Capture(HybridCorridorStage4):
   def plan(self,obs,*args,**kw):
    inputs.append({k:np.asarray(v).copy() for k,v in obs.items()});return super().plan(obs,*args,**kw)
  ex.MODELS[3]=Capture;env=make_env(seed=scene['seed']);g=TrackGeometry(env.unwrapped.track)
  try:ex.episode(ex.Scenario(**scene),env=env,geometry=g,control_dt=.05)
  finally:env.close()
  streams.append((case,inputs))
  np.savez_compressed(a.output/f'ego_stream_{case:03d}.npz',**{k:np.array([x[k] for x in inputs]) for k in inputs[0]})
 env=make_env(seed=99);g=TrackGeometry(env.unwrapped.track);angles=env.unwrapped.sim.agents[0].scan_angles
 rows=[]
 try:
  # Alternate order between cases to reduce systematic temperature/order bias.
  for case,inputs in streams:
   for name,driver_type in ([('baseline',CorridorStage4),('updated',HybridCorridorStage4)] if case!=7 else [('updated',HybridCorridorStage4),('baseline',CorridorStage4)]):
    driver=driver_type(g,angles);values=[]
    for obs in inputs:
     before=time.perf_counter_ns();driver.plan(obs,dt=.05);values.append((time.perf_counter_ns()-before)/1e6)
    rows.append(dict(case=case,policy=name,count=len(values),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),p99_ms=float(np.percentile(values,99)),samples_ms=values));print(json.dumps({k:v for k,v in rows[-1].items() if k!='samples_ms'}),flush=True)
 finally:env.close()
summary={}
for name in ('baseline','updated'):
 values=np.concatenate([r['samples_ms'] for r in rows if r['policy']==name]);summary[name]=dict(count=len(values),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),p99_ms=float(np.percentile(values,99)),max_ms=float(np.max(values)),fraction_over_50ms=float(np.mean(values>50.)))
result=dict(method='Serial whole-controller timing; same ego and LiDAR streams for both policies; commands not used to drive physics during timing',summary=summary,rows=rows,controllers=dict(baseline='CorridorStage4',updated='HybridCorridorStage4'),architecture=platform.machine(),python_version=platform.python_version(),source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in ('hybrid_corridor_stage4.py','recovery_corridor_stage4.py','robust_corridor_stage4.py','corridor_stage4.py','interaction_stage4.py','joint_path_stage4.py','maneuver_core.py','perception.py','track_geometry.py')},stream_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.glob('ego_stream_*.npz')})
(a.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2))
