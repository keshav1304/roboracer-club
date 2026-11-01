"""Turn probe logs into env_v2 parameters.  python3 analyze_probe.py probe_*.npz   (numpy only)

Prints, per test, the measured number and the env_v2 value it should replace (vehicle.py NOMINAL /
DR_RANGES). Speeds and yaw rates are AutoDRIVE's odometry, times its simulator clock.
"""
import math
import sys

import numpy as np

L, MAXS = 0.33, math.radians(30.0)


def load(path):
    d = np.load(path)
    r = d['rows']
    c = {k: r[:, i] for i, k in enumerate(d['cols'])}
    return str(d['test']), c


def slope(t, v):
    if len(t) < 3:
        return float('nan')
    return float(np.polyfit(t, v, 1)[0])


def main(paths):
    for p in paths:
        name, c = load(p)
        t, v, ph = c['sim_t'], c['v'], c['phase']
        if name == 'accel':
            m = (ph == 1) & (v > 0.3) & (v < 0.8 * 24.6 * 0.20)
            print(f'{p}: acceleration at throttle 0.20 {slope(t[m], v[m]):.2f} m/s^2 '
                  f'(env_v2 total ~4.7 = a_drive 5.4 - drag 0.273 v). Settled speed {v[ph == 1][-5:].mean():.2f} m/s')
        elif name in ('brake', 'coast'):
            m = ph == 0
            tt, vv = t[m], v[m]
            k = np.argmax(vv < vv[0] - 0.05) if np.any(vv < vv[0] - 0.05) else 0     # first effect = delay
            dec = -slope(tt[k:], vv[k:])
            ref = 'a_brake 4.1' if name == 'brake' else 'a_eb 2.9'
            print(f'{p}: {name} deceleration {dec:.2f} m/s^2 at ~{vv[k:].mean():.1f} m/s, '
                  f'after {1000 * (tt[k] - tt[0]):.0f} ms (env_v2: {ref} + 0.273 v)')
        elif name == 'steer':
            m = ph >= 0
            tt, wz, st, vv = t[m], c['yaw_rate'][m], c['steer_cmd'][m], v[m]
            kin = vv * np.tan(st * MAXS) / L
            t0 = tt[0]
            k1 = np.argmax(np.abs(wz) > 0.3 * abs(kin[0])) if np.any(np.abs(wz) > 0.3 * abs(kin[0])) else 0
            j = np.argmax(ph[m] == 1)
            after = wz[j:j + 6]
            print(f'{p}: yaw rate starts {1000 * (tt[k1] - t0):.0f} ms after the step (env_v2 30 ms + 183 deg/s). '
                  f'After the 0.15 -> 0.03 step the yaw rate goes {wz[j - 1]:.2f} -> {after.round(2).tolist()} rad/s; '
                  f'a drop within one sample means the unwind is instant (Steer() reading)')
        elif name == 'combined':
            wz, st, vv = c['yaw_rate'], c['steer_cmd'], v
            kin = vv * np.tan(st * MAXS) / L
            ratio = np.where(np.abs(kin) > 0.3, wz / np.where(np.abs(kin) > 0.3, kin, 1), np.nan)
            base = np.nanmean(ratio[ph == 1])
            for k, lab in ((0, 'full throttle'), (2, 'throttle 0 (brakes)')):
                r = np.nanmean(ratio[ph == k])
                print(f'{p}: yaw / kinematic {r:.3f} under {lab} vs {base:.3f} on PP throttle. '
                      f'A drop of more than ~3 % means k_couple > 0 (env_v2 randomises 0 to 0.2)')


if __name__ == '__main__':
    main(sys.argv[1:])
