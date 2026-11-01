"""Measure the AutoDRIVE numbers env_v2 still guesses. About 30 to 60 minutes, ICRA 2026 compete build.

  python3 probe/autodrive_probe_node.py --test brake --tag brake1          (run each test 2 to 3 times)
  python3 probe/autodrive_probe_node.py --test all
  python3 probe/analyze_probe.py probe_*.npz                               (on any machine, numpy only)

Between tests the car drives the club line with pure pursuit at 0.9 x its plan (the P5 file's line), so
nothing here needs a new map. Each test is a short scripted command pattern started at a fixed place on
the line, logged at every odometry message on the simulator clock.

Tests (what env_v2 parameter each one pins down):
  accel     straight after the spawn: throttle 0.20 from standstill for 1.2 s       a_drive, k_servo
  brake     main straight at ~6 m/s: throttle 0 for 0.5 s                          a_brake (+ drag)
  coast     main straight at ~6 m/s: throttle 0.01 for 0.5 s                       a_eb, the brake/coast gap
  steer     main straight at ~3 m/s: steering +0.15 (0.15 s), then +0.03, then 0   steering delay, the instant
                                                                                   unwind in Steer()
  combined  in the long left corner: hold PP steering, throttle 0.32 for 0.3 s,    k_couple (does driving or
            then later throttle 0 for 0.3 s; yaw rate vs v tan(wheel)/L             braking cost sideways grip?)
Keep AutoDRIVE in the background, start only the bridge (deploy/README.md).
"""
import argparse
import math
import os
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32

T = '/autodrive/roboracer_1/'
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--test', choices=['accel', 'brake', 'coast', 'steer', 'combined', 'all'], default='all')
ap.add_argument('--tag', default='probe')
ap.add_argument('--policy-file', default=os.path.join(HERE, '..', 'rl_policy_P5.npz'), help='only its club line is used')
a = ap.parse_args()
P = dict(np.load(a.policy_file))
REF_XY, REF_S, REF_LEN = P['ref_xy'], P['ref_s'], float(P['ref_len'])
REF_V = P['ref_v'] / float(P['ref_speed_scale']) * 0.9
SPT, MAXS, L = 24.6, float(P['max_steer']), float(P['wheelbase'])

# (test, start at s on the club line [m], pattern [(duration s, steer or None = PP, throttle or None = PP, target v)])
PLANS = {
    'accel': (None, [(1.0, 0.0, 0.0, None), (1.2, None, 0.20, None)]),             # from standstill at the spawn
    'brake': (3.0, [(0.5, None, 0.0, None)]),
    'coast': (3.0, [(0.5, None, 0.01, None)]),
    'steer': (1.0, [(0.15, 0.15, None, None), (0.15, 0.03, None, None), (0.3, 0.0, None, None)]),
    'combined': (35.0, [(0.3, None, 0.32, None), (0.4, None, None, None), (0.3, None, 0.0, None)]),
}
SPEED_AT_START = {'brake': 6.0, 'coast': 6.0, 'steer': 3.0, 'combined': None}

rclpy.init()
node = rclpy.create_node('autodrive_probe')
be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
S = {'odom': None, 'lap_time': None, 'sim': 0.0, 'new': False}


def on_lap_time(m):
    if S['lap_time'] is not None:
        S['sim'] += m.data - S['lap_time'] if m.data >= S['lap_time'] else m.data
    S['lap_time'] = m.data


node.create_subscription(Odometry, T + 'odom', lambda m: S.update(odom=m, new=True), be)
node.create_subscription(Float32, T + 'lap_time', on_lap_time, 10)
pub_t = node.create_publisher(Float32, T + 'throttle_command', 1)
pub_s = node.create_publisher(Float32, T + 'steering_command', 1)
ref_i = None


def pose():
    o = S['odom']
    p, q = o.pose.pose.position, o.pose.pose.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return p.x, p.y, yaw, o.twist.twist.linear.x, o.twist.twist.angular.z


def pure_pursuit(x, y, yaw, v, v_target=None):
    global ref_i
    n = len(REF_XY)
    idx = np.arange(n) if ref_i is None else np.arange(ref_i - 40, ref_i + 41) % n
    i = int(idx[np.argmin((REF_XY[idx, 0] - x) ** 2 + (REF_XY[idx, 1] - y) ** 2)])
    ref_i = i
    goal = (REF_S[i] + 0.5 + 0.25 * max(0.0, v)) % REF_LEN
    tx, ty = REF_XY[int(np.searchsorted(REF_S, goal) % n)]
    dx, dy = tx - x, ty - y
    xc, yc = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
    steer = float(np.clip(math.atan(2.0 * yc / max(1e-6, xc * xc + yc * yc) * L) / MAXS, -1, 1))
    vt = REF_V[i] if v_target is None else v_target
    return steer, min(0.32, vt / SPT), REF_S[i]


def send(thr, steer):
    pub_t.publish(Float32(data=float(thr)))
    pub_s.publish(Float32(data=float(steer)))


def run_test(name):
    s_start, pattern = PLANS[name]
    rows = []
    phase, phase_t0, state = -1, None, 'approach'
    t_end = time.time() + 120
    while rclpy.ok() and time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=0.002)
        if not S['new'] or S['lap_time'] is None:
            continue
        S['new'] = False
        x, y, yaw, v, wz = pose()
        sim = S['sim']
        steer_pp, thr_pp, s = pure_pursuit(x, y, yaw, v, SPEED_AT_START.get(name))
        if state == 'approach':
            steer, thr = steer_pp, thr_pp
            v_ok = SPEED_AT_START.get(name) is None or abs(v - SPEED_AT_START[name]) < 0.4
            if s_start is None or (s_start <= s < s_start + 1.0 and v_ok):
                state, phase, phase_t0 = 'test', 0, sim
        if state == 'test':
            dur, st, th, _ = pattern[phase]
            if sim - phase_t0 >= dur:
                phase += 1
                phase_t0 = sim
                if phase >= len(pattern):
                    break
                dur, st, th, _ = pattern[phase]
            steer = steer_pp if st is None else st
            thr = thr_pp if th is None else th
        send(thr, steer)
        rows.append((sim, x, y, yaw, v, wz, steer, thr, phase if state == 'test' else -1, s))
    send(0.0, 0.0)
    out = f'probe_{name}_{a.tag}.npz'
    np.savez(out, rows=np.array(rows), test=name,
             cols=np.array(['sim_t', 'x', 'y', 'yaw', 'v', 'yaw_rate', 'steer_cmd', 'thr_cmd', 'phase', 's']))
    print(f'{name}: {len(rows)} samples -> {out}', flush=True)


tests = ['accel', 'brake', 'coast', 'steer', 'combined'] if a.test == 'all' else [a.test]
try:
    for t in tests:
        print(f'test {t}: driving to the start point', flush=True)
        run_test(t)
        if t == 'accel':
            send(0.0, 0.0)
finally:
    for _ in range(10):
        send(0.0, 0.0)
        time.sleep(0.02)
    node.destroy_node()
    rclpy.shutdown()
