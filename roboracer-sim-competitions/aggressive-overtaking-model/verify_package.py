"""Verify the bundled final controller, frozen demos, Gym and saved map offline."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODEL=ROOT/'overtaking-models'
sys.path.insert(0,str(ROOT/'gym-two-car-reactive/f1tenth_gym'))
sys.path.insert(0,str(MODEL))
from benchmark import offline_only
from world import make_env
from track_geometry import TrackGeometry
from learned_opponent import OpponentForecast
from hybrid_corridor_stage4 import HybridCorridorStage4
import f1tenth_gym
assert Path(f1tenth_gym.__file__).resolve().is_relative_to(ROOT)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for folder in ('hybrid_corridor_demo_20261114','hybrid_corridor_validation_20261114','hybrid_corridor_dev_r1'):
 d=MODEL/'artifacts'/folder;m=json.loads((d/'manifest.json').read_text())
 assert m['controller']=='HybridCorridorStage4' and m['all_opponents_reactive'] and m['api_calls']==0
 for name,h in m['source_sha256'].items():
  assert digest(d/'source'/name)==h,(folder,name)
  if (MODEL/name).exists():assert digest(MODEL/name)==h,('runtime',name)
 for name in ('stage2.npz','stage2.value.npz','stage3.npz','stage3.value.npz'):
  assert digest(MODEL/'artifacts/models'/name)==m['source_sha256'][name],name
 assert digest(MODEL/'artifacts/learned_reaction_v2/model.npz')==m['source_sha256']['opponent_forecast.npz']
 assert digest(MODEL/'artifacts/learned_pressure_random_r1/source/learned_stage4.py')==m['source_sha256']['baseline_learned_stage4.py']
with offline_only():
 env=make_env(seed=12701);g=TrackGeometry(env.unwrapped.track)
 assert g.reference_digest==OpponentForecast().meta['reference_digest']
 env.close()
print('PASS: packaged Gym, saved map, reference identity, final controller and all frozen demo/evaluation sources verified offline.')
