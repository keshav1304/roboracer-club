"""Open-loop manoeuvres that reproduce what we measured in AutoDRIVE, for both vehicle models.

Run from pods/rl/env_v2:  python calibrate.py            (prints a table, no map needed)
test_vehicle.py asserts the same numbers within tolerances. The AutoDRIVE column is
../env/AUTODRIVE_PHYSICS.md (ICRA 2026 compete build, simulator clock).
"""
from __future__ import annotations

import math

import vehicle as V


def car(model=V.MODEL_TWIN, **over):
    p = V.nominal_params(model)
    for k, v in over.items():
        p[V.IDX[k]] = v
    c = V.Car(p)
    return c


def settle_speed(model, thr, seconds=8.0):
    c = car(model)
    c.command(0.0, thr)
    c.run(seconds)
    return c.speed


def ramp(model, thr):
    """Mean acceleration from 0.3 m/s to 80 % of the settled speed."""
    c = car(model)
    c.command(0.0, thr)
    v_end = 24.6 * thr * 0.8
    t0 = None
    t = 0.0
    while c.speed < v_end and t < 6:
        c.run(0.005)
        t += 0.005
        if t0 is None and c.speed > 0.3:
            t0 = t
    return (v_end - 0.3) / max(1e-6, t - t0)


def decel(model, v0, v1, thr):
    """Time and distance to slow from v0 to v1 at a fixed throttle, on a straight."""
    c = car(model)
    c.reset(0.0, 0.0, 0.0, v0)
    c.command(0.0, v0 / 24.6)
    c.run(0.2)                                   # let the throttle settle at v0
    c.st[V.S_VX] = v0
    c.command(0.0, thr)
    c.run(c.p[V.IDX['delay_t']] + 0.001)         # time from when the new throttle reaches the car
    x0, t, v0 = c.x, 0.0, c.speed
    while c.speed > v1 and t < 10:
        c.run(0.002)
        t += 0.002
    return t, c.x - x0


def steady_turn(model, wheel_deg, thr):
    """Settle on a circle at a fixed wheel angle and throttle; returns speed, yaw rate, lateral accel,
    and the ratio of the yaw rate to the kinematic one."""
    c = car(model)
    c.reset(0.0, 0.0, 0.0, 24.6 * thr * 0.9)
    c.command(wheel_deg / 30.0, thr)
    c.run(4.0)
    v, r = c.speed, c.yaw_rate
    w = math.radians(wheel_deg)
    kin = v * math.tan(w) / V.WHEELBASE
    return v, r, v * abs(r), r / kin if kin else float('nan')


def grip_limit(model, wheel_deg):
    """Raise the speed slowly on a fixed wheel angle until the car slides. Returns the highest lateral
    demand (v^2 tan(wheel) / L) held, and the lateral acceleration once sliding."""
    c = car(model)
    w = math.radians(wheel_deg)
    v = 2.0
    c.reset(0.0, 0.0, 0.0, v)
    c.command(wheel_deg / 30.0, v / 24.6)
    c.run(1.0)
    held = 0.0
    while v < 9.0:
        v += 0.05
        c.command(wheel_deg / 30.0, v / 24.6)
        out = c.run(0.15)
        dem = c.speed ** 2 * math.tan(w) / V.WHEELBASE
        if out[V.O_TSF] > 0.02 or out[V.O_TSR] > 0.02:
            c.run(0.4)
            return held, c.speed * abs(c.yaw_rate), ('front' if out[V.O_TSF] >= out[V.O_TSR] else 'rear')
        held = max(held, dem)
    return held, float('nan'), 'none'


def steering_response(model):
    """Unwind test: hold +0.5, then command +0.1. How long until the wheel is within 1 deg of 3 deg."""
    c = car(model)
    c.reset(0.0, 0.0, 0.0, 2.0)
    c.command(0.5, 0.08)
    c.run(0.5)
    c.command(0.1, 0.08)
    t = 0.0
    while abs(math.degrees(c.wheel) - 3.0) > 1.0 and t < 1.0:
        c.run(0.001)
        t += 0.001
    unwind = t
    c.command(0.5, 0.08)                          # and back out: grows at the steering rate
    t = 0.0
    while abs(math.degrees(c.wheel) - 15.0) > 1.0 and t < 1.0:
        c.run(0.001)
        t += 0.001
    return unwind, t


def main():
    for model, name in ((V.MODEL_TWIN, 'twin'), (V.MODEL_EMPIRICAL, 'empirical')):
        print(f'==== {name}')
        print(f'  settle speed per throttle   0.12 -> {settle_speed(model, 0.12) / 0.12:.1f}, '
              f'0.25 -> {settle_speed(model, 0.25) / 0.25:.1f} m/s per unit       AutoDRIVE 24.6')
        print(f'  ramp from rest               thr 0.20 {ramp(model, 0.20):.2f}, thr 0.26 {ramp(model, 0.26):.2f} m/s^2'
              f'         AutoDRIVE ~4.7, same for both')
        t, d = decel(model, 8.0, 4.0, 0.0)
        print(f'  throttle 0, 8 -> 4 m/s       {4 / t:.2f} m/s^2 mean, {d:.2f} m            AutoDRIVE 5.7, 4.2 m')
        t, d = decel(model, 7.0, 4.0, 0.03)
        print(f'  throttle 0.03, 7 -> 4 m/s    {3 / t:.2f} m/s^2 mean                    AutoDRIVE 4.3 to 4.5 (laps)')
        for wd in (4.0, 8.0):
            v, r, alat, ratio = steady_turn(model, wd, 0.16)
            print(f'  steady turn {wd:.0f} deg, thr .16 v {v:.2f} m/s, a_lat {alat:.2f}, yaw / kinematic {ratio:.3f}'
                  f'   AutoDRIVE 0.955')
        for wd in (6, 10, 14, 16, 18, 22):
            held, slide_a, axle = grip_limit(model, wd)
            print(f'  grip limit at {wd:2d} deg wheel     held {held:5.2f} m/s^2, then slides ({axle}) at '
                  f'{slide_a:4.2f}   AutoDRIVE <=14: ~10.4 brief, 16-18: 8-9.6, sliding 4.2-5.6')
        un, gr = steering_response(model)
        print(f'  steering 15 -> 3 deg {un * 1000:.0f} ms, 3 -> 15 deg {gr * 1000:.0f} ms'
              f'   AutoDRIVE source: unwinding instant, growing 183 deg/s (66 ms)')


if __name__ == '__main__':
    main()
