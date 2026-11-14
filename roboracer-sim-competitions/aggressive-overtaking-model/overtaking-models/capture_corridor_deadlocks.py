"""Capture ego-only inputs from the two previous deadlocks for diagnosis."""
import json
from pathlib import Path
import numpy as np
import run_experiment as ex
from interaction_stage4 import InteractionStage4
from world import make_env
from track_geometry import TrackGeometry
out=Path('artifacts/corridor_deadlock_inputs');out.mkdir(exist_ok=True)
for number in (4,5):
 data=json.loads(Path(f'artifacts/interaction_validation_20261114/case-{number:03d}-response.json').read_text())
 recorded=[]
 class Capture(InteractionStage4):
  def plan(self,obs,*a,**kw):
   before=self.steering;result=super().plan(obs,*a,**kw)
   if self.counter%20==0:
    recorded.append(dict(decision=self.counter,observation={k:np.asarray(v).tolist() for k,v in obs.items()},target=None if self.sensor.target is None else vars(self.sensor.target).copy(),initial_steer=before,interaction=self.interaction_debug))
   return result
 ex.MODELS[3]=Capture;scene=dict(data['job']['scenario']);scene['seconds']=10.
 env=make_env(seed=scene['seed']);g=TrackGeometry(env.unwrapped.track)
 try:r,f,*_=ex.episode(ex.Scenario(**scene),env=env,geometry=g,control_dt=.05)
 finally:env.close()
 (out/f'case-{number:03d}.json').write_text(json.dumps(dict(snapshots=recorded,result=r,frames=f),default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()))
 print('Captured',number,flush=True)
