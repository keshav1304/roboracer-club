"""Drive the AutoDRIVE car with an exported policy (train/export_policy.py). Needs numpy and ROS 2 only.

Run it where the AutoDRIVE bridge runs (ROS 2 Humble, the devkit's autodrive_bridge connected):
  python3 rl_deploy_node.py rl_policy_P5.npz --laps 3 --tag first_try
  --base-only        drive the base driver alone (pure pursuit on the club line), no policy. A good
                     first check that topics, units and timing are right
  --no-grip-shield   turn the grip shield off (P5 was trained and tested with it on)
Every 50 ms of SIMULATOR time (20 Hz, as in training; AutoDRIVE can run at 0.4 to 1.0 x real time, so
wall-clock timing would not act at the rate the policy was trained for): build the same observation as
race_env (108 LiDAR bins, speed / 8, last steering command, and in residual mode the base action), run
the actor, send steering_command and throttle_command. The simulator clock is the lap_time topic.
Stops itself after --laps laps, after 0.5 s without LiDAR or odometry (sends zeros), after
--max-contacts wall contacts, and on Ctrl-C or SIGTERM (sends zeros first). Prints laps and wall
contacts, and saves every control step to LOG_DIR/rl_run_TAG.npz.
"""
import argparse
import math
import os
import signal
import sys
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Int32

T = '/autodrive/roboracer_1/'
ap = argparse.ArgumentParser()
ap.add_argument('policy')
ap.add_argument('--laps', type=int, default=3)
ap.add_argument('--tag', default='run')
ap.add_argument('--base-only', action='store_true')
ap.add_argument('--speed-scale', type=float, default=1.0, help='multiply the base driver planned speeds')
ap.add_argument('--grip-shield', action=argparse.BooleanOptionalAction, default=True,
                help='coast while the corner asks for more grip than AutoDRIVE holds for long: lateral demand '
                'v^2 tan(steer)/L above 9.2 m/s^2 for 2 steps, until it drops under 8.8 (default on)')
ap.add_argument('--max-contacts', type=int, default=2, help='stop after this many wall contacts')
ap.add_argument('--log-dir', default='.', help='where rl_run_TAG.npz goes')
a = ap.parse_args()
P = dict(np.load(a.policy))
RES = float(P['residual'])
RES_STEER = float(P['steer_residual']) if 'steer_residual' in P else RES   # 0 = the policy only sets throttle
MAXT, MAXS, SPT = float(P['max_throttle']), float(P['max_steer']), float(P['speed_per_throttle'])
REF_XY, REF_V, REF_S, REF_LEN = P['ref_xy'], P['ref_v'] * a.speed_scale, P['ref_s'], float(P['ref_len'])


def preprocess_scan(ranges):
    scan = np.asarray(ranges, dtype=np.float32)
    mr = float(P['max_range'])
    scan = np.clip(np.nan_to_num(scan, nan=mr, posinf=mr, neginf=0.0), 0.0, mr)
    return np.array([c.mean() for c in np.array_split(scan, int(P['lidar_bins']))], dtype=np.float32) / mr


def actor(obs):
    h = np.tanh(P['w1'] @ obs + P['b1'])
    h = np.tanh(P['w2'] @ h + P['b2'])
    return np.clip(P['w3'] @ h + P['b3'], -1.0, 1.0)


ref_i = None


def base_action(x, y, yaw, v):
    """Same as RaceEnv.reference_action + expert_action: pure pursuit on the club line, planned speed."""
    global ref_i
    n = len(REF_XY)
    if ref_i is None:
        tang = np.roll(REF_XY, -1, 0) - REF_XY
        ok = np.where(tang[:, 0] * math.cos(yaw) + tang[:, 1] * math.sin(yaw) > 0)[0]
        idx = ok if len(ok) else np.arange(n)
    else:
        idx = np.arange(ref_i - 40, ref_i + 41) % n
    i = int(idx[np.argmin((REF_XY[idx, 0] - x) ** 2 + (REF_XY[idx, 1] - y) ** 2)])
    ref_i = i
    lead = float(P['ref_lead']) * v * float(P['steer_delay']) / float(P['control_hz'])
    goal = (REF_S[i] + P['ref_la'][0] + P['ref_la'][1] * max(0.0, v) + lead) % REF_LEN
    tx, ty = REF_XY[int(np.searchsorted(REF_S, goal) % n)]
    dx, dy = tx - x, ty - y
    xc, yc = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
    steer = math.atan(2.0 * yc / max(1e-6, xc * xc + yc * yc) * float(P['wheelbase'])) / MAXS
    thr = min(MAXT, REF_V[i] / SPT)
    return np.array([np.clip(steer, -1, 1), thr / MAXT * 2.0 - 1.0])


rclpy.init(args=None, signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
node = rclpy.create_node('rl_deploy')
be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
S = {'scan': None, 'scan_t': 0.0, 'odom': None, 'odom_t': 0.0, 'laps': None, 'col': None, 'last_lap': None,
     'lap_time': None, 'sim': 0.0}


def on_lap_time(m):
    """Simulator clock: sum of lap_time increases (lap_time restarts at 0 on every new lap)."""
    if S['lap_time'] is not None:
        S['sim'] += m.data - S['lap_time'] if m.data >= S['lap_time'] else m.data
    S['lap_time'] = m.data


node.create_subscription(LaserScan, T + 'lidar', lambda m: S.update(scan=m.ranges, scan_t=time.time()), be)
node.create_subscription(Odometry, T + 'odom', lambda m: S.update(odom=m, odom_t=time.time()), be)
node.create_subscription(Int32, T + 'lap_count', lambda m: S.update(laps=m.data), 10)
node.create_subscription(Int32, T + 'collision_count', lambda m: S.update(col=m.data), 10)
node.create_subscription(Float32, T + 'last_lap_time', lambda m: S.update(last_lap=m.data), 10)
node.create_subscription(Float32, T + 'lap_time', on_lap_time, 10)
pub_t = node.create_publisher(Float32, T + 'throttle_command', 1)
pub_s = node.create_publisher(Float32, T + 'steering_command', 1)
rows, laps0, col0, lap_seen, hits_seen, last_steer = [], None, None, 0, 0, 0.0
over, coasting, shield_steps = 0, False, 0                          # grip shield state
running = True


def send(thr, steer):
    pub_t.publish(Float32(data=float(thr)))
    pub_s.publish(Float32(data=float(steer)))


def stop(*_):
    global running
    running = False


signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
mode = (f'base only x{a.speed_scale}' if a.base_only else None) or (f'residual steer {RES_STEER} throttle {RES}' if RES else 'end-to-end')
print(f'policy {P["source"]} ({mode}), {a.laps} laps', flush=True)
t0, next_sim = time.time(), None
try:
    while running and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.002)
        now = time.time()
        if S['lap_time'] is None:
            continue
        if next_sim is None:
            next_sim = S['sim']
        if S['sim'] < next_sim:                                       # wait for 50 ms of simulator time
            if now - max(S['scan_t'], S['odom_t']) > 0.5:
                send(0.0, 0.0)
            continue
        next_sim += 1.0 / float(P['control_hz'])
        if S['sim'] > next_sim:                                       # fell behind: skip, don't burst
            next_sim = S['sim'] + 1.0 / float(P['control_hz'])
        if S['scan'] is None or S['odom'] is None or now - S['scan_t'] > 0.5 or now - S['odom_t'] > 0.5:
            send(0.0, 0.0)                                           # watchdog: no fresh data, no driving
            continue
        o = S['odom']
        p, q, v = o.pose.pose.position, o.pose.pose.orientation, o.twist.twist.linear.x
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        base = base_action(p.x, p.y, yaw, v)
        obs = np.concatenate([preprocess_scan(S['scan']), [np.clip(v / float(P['speed_norm']), 0, 1), last_steer]]
                             + ([np.clip(base, -1, 1)] if RES else []))
        act = np.zeros(2) if a.base_only else actor(obs)
        sent = np.clip(base + np.array([RES_STEER, RES]) * act, -1, 1) if RES else act
        thr_cmd, steer_cmd = (sent[1] + 1.0) * 0.5 * MAXT, float(sent[0])
        if 'steer_lat_cap' in P:                                      # trained with the old env's grip cap: apply it here too
            lim = math.atan(float(P['steer_lat_cap']) * float(P['wheelbase']) / v ** 2) / MAXS if v > 0.5 else 1.0
            steer_cmd = float(np.clip(steer_cmd, max(-lim, float(P['steer_min'])), min(lim, float(P['steer_max']))))
        if a.grip_shield:                                             # AutoDRIVE logs: 9.0-9.4 held for 0.3 s stayed on,
            demand = v * v * math.tan(abs(steer_cmd) * MAXS) / float(P['wheelbase'])   # 9.4-9.9 for 0.5 s slid off
            over = over + 1 if demand > 9.2 else 0
            coasting = (coasting and demand > 8.8) or over >= 2
            if coasting:
                thr_cmd = 0.0
                shield_steps += 1
        send(thr_cmd, steer_cmd)
        last_steer = steer_cmd
        rows.append((now - t0, S['sim'], p.x, p.y, yaw, v, *base, *act, steer_cmd, thr_cmd,
                     -1 if S['col'] is None else S['col']))
        if laps0 is None and S['laps'] is not None:
            laps0, col0 = S['laps'], S['col'] or 0
        if laps0 is not None and S['laps'] is not None and S['laps'] - laps0 > lap_seen:
            lap_seen = S['laps'] - laps0
            hits = (S['col'] or 0) - col0
            print(f'LAP {lap_seen}: {S["last_lap"]:.3f} s, wall contacts so far {hits}', flush=True)
            if lap_seen >= a.laps:
                break
        if S['col'] is not None and col0 is not None and S['col'] - col0 > hits_seen:
            hits_seen = S['col'] - col0
            print(f'WALL CONTACT {hits_seen} at ({p.x:.2f}, {p.y:.2f}), {v:.1f} m/s', flush=True)
            if hits_seen >= a.max_contacts:
                print('too many wall contacts, stopping', flush=True)
                break
finally:
    for _ in range(10):
        send(0.0, 0.0)
        time.sleep(0.02)
    log = os.path.join(a.log_dir, f'rl_run_{a.tag}.npz')
    np.savez(log, rows=np.array(rows),
             cols=np.array(['t', 'sim_t', 'x', 'y', 'yaw', 'v', 'base_steer', 'base_thr', 'act_steer', 'act_thr',
                            'steer_cmd', 'thr_cmd', 'col']))
    hits = (S['col'] or 0) - (col0 or 0)
    print(f'stopped: {lap_seen} laps, wall contacts {hits}, {len(rows)} control steps saved to {log}'
          + (f', grip shield coasted {shield_steps} steps' if a.grip_shield else ''), flush=True)
    node.destroy_node()
    rclpy.shutdown()
