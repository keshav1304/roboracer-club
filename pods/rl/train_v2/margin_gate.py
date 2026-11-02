"""Does a checkpoint keep room to the walls? ICRA, power rule on (8.0) and a 0.08 m wall margin around the collision box
(the margin that keeps pure pursuit and P5 clean but crashes both end to end policies that touched in AutoDRIVE):
twin + empirical nominal 3 laps, 10 randomised twin episodes x 3 laps. Run from pods/rl:  python3 margin_gate.py CKPT.zip"""
import os, sys
import numpy as np
sys.path.insert(0, 'env_v2')
import vehicle as V
from race_env_v2 import make_env
from stable_baselines3 import PPO
ck = sys.argv[1]
m = PPO.load(ck, device='cpu', custom_objects={'learning_rate': 0.0, 'lr_schedule': lambda _: 0.0, 'clip_range': lambda _: 0.0})
def run(model, randomize, seed):
    env = make_env(tracks=['icra'], randomize=randomize, random_start=False, seed=seed, model=model, timing=True,
                   power_lat=8.0, wall_margin=0.08)
    o, _ = env.reset()
    while True:
        o, _, term, trunc, info = env.step(m.predict(o, deterministic=True)[0])
        if term or trunc or info['laps'] >= 3:
            return info
nom = [run(V.MODEL_TWIN, False, 7), run(V.MODEL_EMPIRICAL, False, 7)]
rnd = [run(V.MODEL_TWIN, True, 6000 + k) for k in range(10)]
fly = [np.mean(i['lap_times'][1:]) for i in nom if not i['crash'] and len(i['lap_times']) > 1]
print(f'MARGINGATE {os.path.basename(os.path.dirname(ck))} {os.path.basename(ck)} nominal_crash {sum(i["crash"] for i in nom)} '
      f'nominal_wall {sum(i["wall"] for i in nom)} nominal_power {sum(i["power"] for i in nom)} random_crash {sum(i["crash"] for i in rnd)} '
      f'random_wall {sum(i["wall"] for i in rnd)} lap {np.mean(fly) if fly else float("nan"):.3f}')
