"""Check manifest identity and run the included Gym/map without Unity assets."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'gym-two-car-reactive/f1tenth_gym'))
sys.path.insert(0,str(ROOT/'overtaking-models'))
from benchmark import offline_only
from world import make_env
from track_geometry import TrackGeometry
from learned_opponent import OpponentForecast
import f1tenth_gym
assert Path(f1tenth_gym.__file__).resolve().is_relative_to(ROOT)
manifest=json.loads((ROOT/'overtaking-models/artifacts/learned_pressure_random_r1/manifest.json').read_text())
for name in ('learned_opponent.py','expanded_planner.py','maneuver_core.py','world.py','run_experiment.py'):
 assert hashlib.sha256((ROOT/'overtaking-models'/name).read_bytes()).hexdigest()==manifest['source_sha256'][name],name
with offline_only():
 env=make_env(seed=12701);g=TrackGeometry(env.unwrapped.track)
 assert g.reference_digest==OpponentForecast().meta['reference_digest']
 env.close()
print('PASS: packaged Gym, saved map, reference identity, and frozen demo source verified offline.')
