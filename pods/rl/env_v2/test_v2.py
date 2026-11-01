"""Checks env_v2 against AutoDRIVE. Run from pods/rl/env_v2:  python test_v2.py   (about 1 minute)

If you change vehicle.py and this fails, the env no longer matches what we measured in AutoDRIVE.
"""
import math
import sys

import numpy as np

import calibrate as K
import contract as C
import vehicle as V
from race_env_v2 import make_env
from validate_v2 import drive


def close(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f'  {"ok " if ok else "BAD"} {name}: {got:.3f} (AutoDRIVE {want}, tolerance {tol})')
    return ok


def main():
    ok = []
    tw = V.MODEL_TWIN
    print('longitudinal (twin)')
    ok.append(close('settled speed per unit throttle', K.settle_speed(tw, 0.2) / 0.2, 24.6, 0.4))
    ok.append(close('ramp from rest, throttle 0.20', K.ramp(tw, 0.20), 4.7, 0.3))
    t, d = K.decel(tw, 8.0, 4.0, 0.0)
    ok.append(close('throttle 0 (brakes), 8 -> 4 m/s mean decel', 4 / t, 5.7, 0.3))
    ok.append(close('throttle 0, 8 -> 4 m/s distance', d, 4.2, 0.3))
    print('lateral (twin)')
    ok.append(close('yaw rate / kinematic at 4 deg', K.steady_turn(tw, 4.0, 0.16)[3], 0.955, 0.012))
    held16, slide16, axle16 = K.grip_limit(tw, 16)
    ok.append(close('sliding lateral accel', slide16, 4.9, 0.8))
    ok.append(axle16 == 'front')
    print(f'  {"ok " if axle16 == "front" else "BAD"} slides at the front (every AutoDRIVE slide was)')
    print('steering (twin, VehicleController.Steer)')
    un, gr = K.steering_response(tw)
    ok.append(close('unwind 15 -> 3 deg incl. 30 ms delay (s)', un, 0.031, 0.005))
    ok.append(close('grow 3 -> 15 deg incl. 30 ms delay (s)', gr, 0.091, 0.008))
    print('closed loop on ICRA, nominal (validate_v2.py)')
    want = {1.10: (False, 13.21), 1.15: (False, 12.67), 1.20: (True, None)}
    for scale, (crash, lap) in want.items():
        info, _ = drive(tw, scale, False, laps=3)
        good = info['crash'] == crash
        if lap and not crash:
            good = good and abs(np.mean(info['lap_times'][1:]) - lap) < 0.15
        print(f'  {"ok " if good else "BAD"} pure pursuit x{scale:.2f}: crash {info["crash"]}, laps {info["lap_times"]}')
        ok.append(good)
    info, _ = drive(tw, 1.15, True, laps=3)
    good = not info['crash'] and abs(np.mean(info['lap_times'][1:]) - 11.52) < 0.15
    print(f'  {"ok " if good else "BAD"} P5 + grip shield: crash {info["crash"]}, laps {info["lap_times"]}')
    ok.append(good)
    print('contract')
    for v in (1.0, 4.0, 7.0):
        for a in ((0.3, 0.2), (-1.0, -1.0), (0.9, 0.95)):
            s, t = C.action_to_commands(a, v)
            back = C.commands_to_action(s, t, v)
            assert abs(back[0] - a[0]) < 1e-6 and (t == 0.0 or abs(back[1] - a[1]) < 1e-6)
    assert C.action_to_commands((0, -0.95), 3.0)[1] == 0.0 and C.action_to_commands((0, -0.85), 3.0)[1] > 0.0
    w7 = math.degrees(C.envelope(7.0))
    assert 2.0 < w7 < 6.0, w7
    print(f'  ok action mapping round trip, brake zone, envelope at 7 m/s {w7:.1f} deg')
    env = make_env(tracks=['practice'], seed=1)
    obs, _ = env.reset()
    assert obs.shape == (C.OBS_DIM,) and env.observation_space.contains(obs)
    for _ in range(20):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs), obs
        if term or trunc:
            break
    print('  ok env reset / step / observation space')
    n_bad = ok.count(False)
    print('all checks passed' if n_bad == 0 else f'{n_bad} checks FAILED')
    sys.exit(1 if n_bad else 0)


if __name__ == '__main__':
    main()
