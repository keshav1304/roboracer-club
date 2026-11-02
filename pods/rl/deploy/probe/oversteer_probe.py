"""Does the AutoDRIVE car oversteer, and is it the throttle or the brake? ICRA 2026 compete build.

  python3 probe/oversteer_probe.py --tag os1                      (all modes, 2 repeats, about 10 minutes)
  python3 probe/oversteer_probe.py --modes full full_brake --repeats 3
  python3 probe/oversteer_probe.py --analyze probe_oversteer_*.npz  (numpy only, any machine)

Why: rl_policy_v2_s0_8M spun three times in the right-hander before the start straight (club line s 34 to 40).
Each time it entered at 5.5 to 6.3 m/s on full throttle, the car rotated faster than its steering asked for,
the policy lifted to throttle 0 (which brakes all four wheels) and the car went round. env_v2's twin has rear
grip x1.25, fitted on logs where only the front ever slid, so in the gym the rear almost never lets go.

Each trial resets the car, drives one lap on the club line with pure pursuit at 0.9 x its plan, then from
s = 33.0 (the straight before the corner) runs one throttle pattern with pure pursuit steering:
  hold        pure pursuit at 1.0 x plan through the corner (baseline, should not slide)
  mid         throttle 0.20 to the straight
  full        throttle 0.32 (the maximum) to the straight, what the policy did
  full_brake  0.32 until s = 36.5, then throttle 0 (brakes) for 0.4 s, what the policy did next
  full_coast  0.32 until s = 36.5, then throttle 0.01 (coasts, no brake) for 0.4 s
If the car spins (body slip over 35 deg), throttle goes to 0.03 and steering to 0 for 1 s, and the trial ends.
Logged at every odometry message on the simulator clock. Keep AutoDRIVE in the background.
"""
import argparse
import math
import os
import time

import numpy as np

L, MAXS, SPT = 0.3302, 0.5236, 24.6
S_GO, S_LIFT, S_END = 33.0, 36.5, 41.0
MODES = ('hold', 'mid', 'full', 'full_brake', 'full_coast')
COLS = ['sim_t', 'x', 'y', 'yaw', 'v', 'yaw_rate', 'steer_cmd', 'thr_cmd', 'phase', 's', 'col']


def analyze(paths):
    print(f'{"file":34s} {"mode":10s} {"v@34.5":>6s} {"vmax":>5s} {"ratio p90":>9s} {"slip max":>8s} '
          f'{"lat acc max":>11s}  {"slip > 10 deg first at":24s}  result')
    for p in paths:
        d = np.load(p)
        c = {k: d['rows'][:, i] for i, k in enumerate(d['cols'])}
        t, x, y, yaw, v, wz, st, ph, s = (c[k] for k in ('sim_t', 'x', 'y', 'yaw', 'v', 'yaw_rate', 'steer_cmd',
                                                           'phase', 's'))
        m = ph >= 1
        if not m.any():
            print(f'{os.path.basename(p):34s} {str(d["mode"]):10s} never reached the corner')
            continue
        # body slip: heading vs direction of travel from positions (3 samples apart)
        k = 3
        vx, vy = x[k:] - x[:-k], y[k:] - y[:-k]
        moving = np.hypot(vx, vy) > 0.05
        beta = np.full(len(x), np.nan)
        b = np.degrees(np.arctan2(np.sin(np.arctan2(vy, vx) - yaw[k:]), np.cos(np.arctan2(vy, vx) - yaw[k:])))
        beta[k:] = np.where(moving, b, np.nan)
        kin = v * np.tan(st * MAXS) / L
        ok = m & (np.abs(kin) > 0.4) & (v > 2.0)
        ratio = np.where(ok, wz / np.where(ok, kin, 1.0), np.nan)
        p90 = np.nanpercentile(ratio, 90) if np.any(np.isfinite(ratio)) else float('nan')
        at = np.argmin(np.abs(s - 34.5) + 100 * (~m))
        bm = np.where(m, np.abs(beta), np.nan)
        spun = bool(np.nanmax(bm) > 35) if np.any(np.isfinite(bm)) else False
        hit = bool(c['col'][m].max() > c['col'][m].min())
        first = np.flatnonzero(bm > 10)
        onset = (f'v {v[first[0]]:.1f} thr {c["thr_cmd"][first[0]]:.2f} s {s[first[0]]:.1f}' if len(first) else '-')
        res = ('SPUN' if spun else 'clean') + (' + wall' if hit else '')
        print(f'{os.path.basename(p):34s} {str(d["mode"]):10s} {v[at]:6.2f} {v[m].max():5.2f} {p90:9.2f} '
              f'{np.nanmax(bm):8.1f} {np.nanmax(np.abs(v[m] * wz[m])):11.2f}  {onset:24s}  {res}')
    print('ratio = measured yaw rate / v tan(steer) / L. About 0.95 is normal driving (env_v2 understeer gain),\n'
          'clearly above 1 means the rear is stepping out. slip = heading vs direction of travel, deg.')


def drive(a):
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import Bool, Float32, Int32

    P = dict(np.load(a.policy_file))
    ref_xy, ref_s, ref_len = P['ref_xy'], P['ref_s'], float(P['ref_len'])
    ref_v = P['ref_v'] / float(P['ref_speed_scale'])
    if a.offset:                                  # shift the line sideways (+ left) in the corner, ramped in and out
        tx, ty = np.gradient(ref_xy[:, 0]), np.gradient(ref_xy[:, 1])
        nrm = np.hypot(tx, ty)
        w = np.clip(np.minimum(ref_s - (S_GO - 2.0), (S_END + 1.0) - ref_s) / 2.0, 0.0, 1.0)
        ref_xy = ref_xy + a.offset * w[:, None] * np.stack([-ty / nrm, tx / nrm], 1)
    T = '/autodrive/roboracer_1/'
    rclpy.init()
    node = rclpy.create_node('oversteer_probe')
    be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
    rel = QoSProfile(durability=DurabilityPolicy.VOLATILE, reliability=ReliabilityPolicy.RELIABLE,
                     history=HistoryPolicy.KEEP_LAST, depth=1)
    S = {'odom': None, 'lap_time': None, 'sim': 0.0, 'new': False, 'col': 0}

    def on_lap_time(m):
        if S['lap_time'] is not None:
            S['sim'] += m.data - S['lap_time'] if m.data >= S['lap_time'] else m.data
        S['lap_time'] = m.data

    node.create_subscription(Odometry, T + 'odom', lambda m: S.update(odom=m, new=True), be)
    node.create_subscription(Float32, T + 'lap_time', on_lap_time, 10)
    node.create_subscription(Int32, T + 'collision_count', lambda m: S.update(col=m.data), 10)
    pub_t = node.create_publisher(Float32, T + 'throttle_command', rel)
    pub_s = node.create_publisher(Float32, T + 'steering_command', rel)
    pub_r = node.create_publisher(Bool, '/autodrive/reset_command', rel)
    ref_i = [None]

    def send(thr, steer):
        pub_t.publish(Float32(data=float(thr)))
        pub_s.publish(Float32(data=float(steer)))

    def reset():
        for val, k in ((False, 25), (True, 25), (False, 50)):
            for _ in range(k):
                send(0.0, 0.0)
                pub_r.publish(Bool(data=val))
                rclpy.spin_once(node, timeout_sec=0.02)
        ref_i[0] = None

    def pose():
        o = S['odom']
        p, q = o.pose.pose.position, o.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return p.x, p.y, yaw, o.twist.twist.linear.x, o.twist.twist.angular.z

    def pure_pursuit(x, y, yaw, v, scale):
        n = len(ref_xy)
        idx = np.arange(n) if ref_i[0] is None else np.arange(ref_i[0] - 40, ref_i[0] + 41) % n
        i = int(idx[np.argmin((ref_xy[idx, 0] - x) ** 2 + (ref_xy[idx, 1] - y) ** 2)])
        ref_i[0] = i
        goal = (ref_s[i] + 0.5 + 0.25 * max(0.0, v)) % ref_len
        tx, ty = ref_xy[int(np.searchsorted(ref_s, goal) % n)]
        dx, dy = tx - x, ty - y
        xc, yc = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
        steer = float(np.clip(math.atan(2.0 * yc / max(1e-6, xc * xc + yc * yc) * L) / MAXS, -1, 1))
        return steer, min(0.32, scale * ref_v[i] / SPT), float(ref_s[i])

    def trial(mode, out):
        reset()
        t_wait = time.time() + 5
        while time.time() < t_wait:                                  # let the reset land
            rclpy.spin_once(node, timeout_sec=0.01)
        rows, state, t_go, t_lift, t_spin, lapped = [], 'approach', None, None, None, False
        last = (0.0, None)
        t_end = time.time() + a.timeout
        while rclpy.ok() and time.time() < t_end:
            rclpy.spin_once(node, timeout_sec=0.002)
            if not S['new'] or S['lap_time'] is None:
                continue
            S['new'] = False
            x, y, yaw, v, wz = pose()
            sim = S['sim']
            steer, thr_pp, s = pure_pursuit(x, y, yaw, v, 1.0 if (mode == 'hold' and state == 'test') else 0.9)
            if s < 20.0:
                lapped = True                                         # past the line once, so s 33 is the next lap
            if state == 'approach' and lapped and S_GO <= s < S_GO + 1.0:
                state, t_go = 'test', sim
            phase = 0
            thr = thr_pp
            if state == 'test':
                phase = 1
                if mode == 'mid':
                    thr = 0.20
                elif mode in ('full', 'full_brake', 'full_coast'):
                    thr = 0.32
                    if mode != 'full' and (t_lift is not None or s >= S_LIFT):
                        t_lift = sim if t_lift is None else t_lift
                        phase = 2
                        thr = (0.0 if mode == 'full_brake' else 0.01) if sim - t_lift < 0.4 else 0.05
                # spin guard: heading vs direction of travel
                if last[1] is not None:
                    dx, dy = x - last[1][0], y - last[1][1]
                    if math.hypot(dx, dy) > 0.05:
                        beta = math.degrees(math.atan2(math.sin(math.atan2(dy, dx) - yaw), math.cos(math.atan2(dy, dx) - yaw)))
                        if abs(beta) > 35 and t_spin is None:
                            t_spin = sim
                        last = (sim, (x, y))
                else:
                    last = (sim, (x, y))
                if t_spin is not None:
                    phase, thr, steer = 3, 0.03, 0.0
                    if sim - t_spin > 1.0:
                        break
                if (S_END <= s < S_END + 3.0 and t_spin is None) or sim - t_go > 4.0:
                    break
            elif last[1] is None or math.hypot(x - last[1][0], y - last[1][1]) > 0.05:
                last = (sim, (x, y))
            send(thr, steer)
            rows.append((sim, x, y, yaw, v, wz, steer, thr, phase, s, S['col']))
        for _ in range(10):
            send(0.0, 0.0)
            rclpy.spin_once(node, timeout_sec=0.02)
        np.savez(out, rows=np.array(rows), cols=np.array(COLS), mode=mode,
                 offset=a.offset, note='oversteer_probe.py, club line from ' + os.path.basename(a.policy_file))
        print(f'{mode}: {len(rows)} samples, {"reached" if state == "test" else "did NOT reach"} the corner'
              f'{", spun" if t_spin is not None else ""} -> {out}', flush=True)

    outs = []
    try:
        for r in range(a.repeats):
            for mode in a.modes:
                out = os.path.join(a.log_dir, f'probe_oversteer_{a.tag}_{mode}_{r}.npz')
                trial(mode, out)
                outs.append(out)
    finally:
        reset()
        node.destroy_node()
        rclpy.shutdown()
    analyze(outs)


if __name__ == '__main__':
    HERE = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--tag', default='os')
    ap.add_argument('--timeout', type=float, default=150.0, help='real seconds per trial')
    ap.add_argument('--log-dir', default='.')
    ap.add_argument('--policy-file', default=os.path.join(HERE, '..', 'rl_policy_P5.npz'), help='only its club line is used')
    ap.add_argument('--offset', type=float, default=0.0, help='shift the line sideways in the corner, m, + left')
    ap.add_argument('--analyze', nargs='+', metavar='NPZ')
    a = ap.parse_args()
    if a.analyze:
        analyze(a.analyze)
    else:
        drive(a)
