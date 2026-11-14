"""Read-only diagnosis of fresh failures; this turns these cases into development inputs."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def capture(number):
 import numpy as np
 import run_experiment as ex
 from robust_corridor_stage4 import RobustCorridorStage4
 from world import make_env
 from track_geometry import TrackGeometry
 d=json.loads(Path(f'artifacts/robust_corridor_validation_20261114/case-{number:03d}-response.json').read_text());recorded=[]
 class Capture(RobustCorridorStage4):
  def plan(self,obs,*a,**kw):
   before=self.steering;result=super().plan(obs,*a,**kw)
   if self.counter%10==0:
    recorded.append(dict(decision=self.counter,observation={k:np.asarray(v).tolist() for k,v in obs.items()},target=None if self.sensor.target is None else vars(self.sensor.target).copy(),initial_steer=before,interaction=self.interaction_debug,telemetry={k:v for k,v in self.telemetry.items() if k in ('planned_speed','lidar_speed_cap','command')}))
   return result
 ex.MODELS[3]=Capture;scene=dict(d['job']['scenario']);scene['seconds']=16.
 env=make_env(seed=scene['seed']);g=TrackGeometry(env.unwrapped.track)
 try:r,f,*_=ex.episode(ex.Scenario(**scene),env=env,geometry=g,control_dt=.05)
 finally:env.close()
 out=Path('artifacts/robust_corridor_deadlock_inputs');out.mkdir(exist_ok=True)
 (out/f'case-{number:03d}.json').write_text(json.dumps(dict(snapshots=recorded,result=r,frames=f),default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()))
 print('Captured',number,flush=True)
if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=2) as p:list(p.map(capture,[5,8]))
