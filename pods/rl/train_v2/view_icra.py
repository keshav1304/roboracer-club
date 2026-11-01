"""Watch an env_v2 policy drive ICRA in the f1tenth gym, live, both physics models side by side (OpenCV window).

  python3 train_v2/view_icra.py policies/rl_policy_v2_s0_8M.npz   # from pods/rl; nominal physics, real time, 5 laps
  python3 train_v2/view_icra.py POLICY.npz --laps 10 --speed 2      # twice real time
  python3 train_v2/view_icra.py POLICY.npz --randomize --seed 3     # one randomised physics draw
  python3 train_v2/view_icra.py POLICY.npz --video out.mp4          # also record it (--headless for no window)

Left panel twin model, right panel empirical model, same policy, timing (delays, late steps) on.
Trail colour is speed (blue slow, red fast), the grey line is the teacher's minimum curvature line.
Space pauses, q quits.
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

RL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(RL, 'env_v2'))
from race_env_v2 import make_env              # noqa: E402
import contract as C                          # noqa: E402
import vehicle as V                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('policy')
ap.add_argument('--laps', type=int, default=5)
ap.add_argument('--speed', type=float, default=1.0, help='playback rate, 1 = real time')
ap.add_argument('--randomize', action='store_true')
ap.add_argument('--seed', type=int, default=7)
ap.add_argument('--no-timing', action='store_true')
ap.add_argument('--video', default='')
ap.add_argument('--headless', action='store_true')
a = ap.parse_args()

P = dict(np.load(a.policy))


def act(o):
    h = np.tanh(P['w1'] @ o + P['b1'])
    h = np.tanh(P['w2'] @ h + P['b2'])
    return np.clip(P['w3'] @ h + P['b3'], -1, 1).astype(np.float32)


MAP_H, HUD = 820, 150
runs = []
for label, model in (('twin', V.MODEL_TWIN), ('empirical', V.MODEL_EMPIRICAL)):
    env = make_env(tracks=['icra'], randomize=a.randomize, random_start=False, seed=a.seed, model=model,
                   timing=not a.no_timing)
    obs, _ = env.reset()
    runs.append(dict(env=env, obs=obs, label=label, pts=[], info=None, done=False, slides=0))

tr = runs[0]['env'].track
teach = tr.teacher.xy
pad = float(max(tr.wl.max(), tr.wr.max())) + 0.6
lo = np.array([tr.cx.min() - pad, tr.cy.min() - pad])
hi = np.array([tr.cx.max() + pad, tr.cy.max() + pad])
scale = (MAP_H - 20) / (hi[1] - lo[1])
W = max(420, int((hi[0] - lo[0]) * scale) + 20)
H = MAP_H + HUD
lo[0] -= (W - 20 - (hi[0] - lo[0]) * scale) / 2 / scale       # centre the map in the panel


def px(p):
    p = np.atleast_2d(p)
    x = 10 + (p[:, 0] - lo[0]) * scale
    y = H - 10 - (p[:, 1] - lo[1]) * scale
    return np.stack([x, y], 1).astype(np.int32)


# the real occupancy map the LiDAR and the collision check use: free white, wall grey, edges black
occ = np.asarray(tr.gym_track.occupancy_map)
yy, xx = np.mgrid[0:H, 0:W]
wx = lo[0] + (xx - 10) / scale
wy = lo[1] + (H - 10 - yy) / scale
col = ((wx - tr.ox) / tr.res).astype(int)
row = ((wy - tr.oy) / tr.res).astype(int)
inside = (row >= 0) & (row < occ.shape[0]) & (col >= 0) & (col < occ.shape[1])
free = np.zeros((H, W), bool)
free[inside] = occ[row[inside], col[inside]] > 128
base = np.full((H, W, 3), 215, np.uint8)
base[free] = 255
edge = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
base[edge] = 0
base[:HUD] = 255
cv2.polylines(base, [px(teach)], True, (170, 170, 170), 1, cv2.LINE_AA)


def colour(v):
    c = cv2.applyColorMap(np.uint8([[min(255, max(0, v / 8 * 255))]]), cv2.COLORMAP_TURBO)[0, 0]
    return tuple(int(x) for x in c)


for r in runs:
    r['img'] = base.copy()

writer = None
if a.video:
    writer = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*'mp4v'), 20, (2 * W, H + 40))
title = f'env_v2 on ICRA  {os.path.basename(a.policy)}'
dt = C.DT_CONTROL
t0, k, paused = time.time(), 0, False
while not all(r['done'] for r in runs):
    if not paused:
        for r in runs:
            if r['done']:
                continue
            r['obs'], _, term, trunc, info = r['env'].step(act(r['obs']))
            r['info'] = info
            r['slides'] += info['slide']
            c = r['env'].car
            p = px((c.x, c.y))[0]
            if r['pts']:
                cv2.line(r['img'], tuple(r['pts'][-1]), tuple(p), colour(c.speed), 3, cv2.LINE_AA)
            r['pts'].append(p)
            if term or trunc or info['laps'] >= a.laps:
                r['done'] = True
        k += 1
    panels = []
    for r in runs:
        img, info, c = r['img'].copy(), r['info'], r['env'].car
        hx, hy = c.x + 0.3 * math.cos(c.yaw), c.y + 0.3 * math.sin(c.yaw)
        cv2.arrowedLine(img, tuple(px((c.x, c.y))[0]), tuple(px((hx, hy))[0]), (40, 40, 220), 3, tipLength=0.5)
        if info['crash']:
            why = 'wall' if info['wall'] else 'slide' if info['slide'] else 'spin' if info['spin'] else 'stall'
            status, col = f'CRASH ({why})', (0, 0, 220)
            cv2.circle(img, tuple(px((c.x, c.y))[0]), 14, (0, 0, 220), 3)
        else:
            status, col = ('done' if r['done'] else 'driving'), (0, 130, 0)
        laps, fly = info['lap_times'], info['lap_times'][1:]
        lines = [(f'{r["label"]} model   {status}', col),
                 (f'lap {info["laps"] + (0 if r["done"] else 1)}   sim time {info["sim_time"]:6.2f} s', (0, 0, 0)),
                 (f'speed {c.speed:4.1f} m/s   steer {math.degrees(info["steer_cmd"] * C.MAX_STEER):+5.1f} deg', (0, 0, 0)),
                 (f'grip used {info["demand_ratio"]:4.2f}   slide steps {r["slides"]}', (0, 0, 0)),
                 ('laps ' + ' '.join(f'{x:.2f}' for x in laps[-6:]), (0, 0, 0)),
                 (f'flying lap mean {np.mean(fly):.3f} s' if fly else 'flying lap mean -', (0, 0, 0))]
        for j, (s, cc) in enumerate(lines):
            cv2.putText(img, s, (14, 26 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.55, cc, 1, cv2.LINE_AA)
        panels.append(img)
    frame = np.hstack(panels)
    bar = np.full((40, 2 * W, 3), 245, np.uint8)
    cv2.putText(bar, f'{os.path.basename(a.policy)}   physics {"randomised" if a.randomize else "nominal"}   '
                f'timing {"off" if a.no_timing else "on"}   teacher lap {tr.teacher.lap_time:.2f} s   '
                f'playback x{a.speed:g}', (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
    frame = np.vstack([bar, frame])
    if writer is not None and not paused:
        writer.write(frame)
    if not a.headless:
        cv2.imshow(title, frame)
        wait = t0 + k * dt / a.speed - time.time()
        key = cv2.waitKey(max(1, int(wait * 1000))) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            paused = not paused
            t0 = time.time() - k * dt / a.speed

for r in runs:
    i = r['info']
    fly = i['lap_times'][1:]
    print(f'{r["label"]:9s} laps {i["laps"]}  crash {i["crash"]}  lap times {i["lap_times"]}  '
          f'flying mean {np.mean(fly):.3f}' if fly else f'{r["label"]:9s} laps {i["laps"]}  crash {i["crash"]}')
if writer is not None:
    writer.release()
if not a.headless:
    cv2.waitKey(0)
    cv2.destroyAllWindows()
