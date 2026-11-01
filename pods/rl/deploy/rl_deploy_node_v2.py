"""Drive the AutoDRIVE car with an env_v2 policy (train_v2/export_policy.py). Needs numpy and ROS 2 only.

  python3 rl_deploy_node_v2.py rl_policy_v2.npz --laps 3 --tag first_try
  --grip-shield    add P5's grip shield (off by default: v2 policies are trained without it)
  --no-envelope    send the raw action as steering (only for policies trained with use_envelope False)

Same loop as rl_deploy_node.py (every 50 ms of SIMULATOR time, watchdog, zeros on exit, log), with the
v2 contract from the .npz:
  observation  108 LiDAR bins, speed / 8, last steering command, last throttle action, yaw rate / 4 (from the
               change of odometry yaw over the last control step), two accelerations / 10 (from odometry speed)
  action       action[0] x the steering envelope at the current speed -> steering_command
               action[1] < -0.9 -> throttle_command 0 (brakes all four wheels), else 0.005 .. 0.32
Pitfalls from deploy/README.md all apply: keep AutoDRIVE in the background (late steps are what crash
policies), start only the bridge, check the compete build is open.
"""
import argparse
import math
import os
import signal
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
ap.add_argument('--tag', default='v2')
ap.add_argument('--grip-shield', action='store_true')
ap.add_argument('--no-envelope', action='store_true')
ap.add_argument('--max-contacts', type=int, default=2)
ap.add_argument('--log-dir', default='.')
a = ap.parse_args()
P = dict(np.load(a.policy))
assert int(P.get('obs_version', 1)) == 2, 'this node runs env_v2 policies; use rl_deploy_node.py for P5'
HZ, DT = float(P['control_hz']), 1.0 / float(P['control_hz'])
MAXT, MAXS, L = float(P['max_throttle']), float(P['max_steer']), float(P['wheelbase'])
BRAKE, MIN_T = float(P['brake_zone']), float(P['min_drive_throttle'])
ENV_V, ENV_W = P['env_speeds'], P['env_wheel']


def preprocess_scan(ranges):
    scan = np.asarray(ranges, dtype=np.float32)
    mr = float(P['max_range'])
    scan = np.clip(np.nan_to_num(scan, nan=mr, posinf=mr, neginf=0.0), 0.0, mr)
    return np.array([c.mean() for c in np.array_split(scan, int(P['lidar_bins']))], dtype=np.float32) / mr


def actor(obs):
    h = np.tanh(P['w1'] @ obs + P['b1'])
    h = np.tanh(P['w2'] @ h + P['b2'])
    return np.clip(P['w3'] @ h + P['b3'], -1.0, 1.0)


def to_commands(act, v):
    limit = 1.0 if a.no_envelope else float(np.interp(abs(v), ENV_V, ENV_W)) / MAXS
    steer = float(act[0] * limit)
    thr = 0.0 if act[1] < BRAKE else max(MIN_T, MAXT * (act[1] - BRAKE) / (1.0 - BRAKE))
    return steer, thr


rclpy.init(args=None, signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
node = rclpy.create_node('rl_deploy_v2')
be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
S = {'scan': None, 'scan_t': 0.0, 'odom': None, 'odom_t': 0.0, 'laps': None, 'col': None, 'last_lap': None,
     'lap_time': None, 'sim': 0.0}


def on_lap_time(m):
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
rows, laps0, col0, lap_seen, hits_seen = [], None, None, 0, 0
last_steer, last_thr_action = 0.0, -1.0
v_hist, prev_yaw, prev_sim = None, None, None
over, coasting = 0, False
running = True


def send(thr, steer):
    pub_t.publish(Float32(data=float(thr)))
    pub_s.publish(Float32(data=float(steer)))


def stop(*_):
    global running
    running = False


signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
print(f'policy {P["source"]}, {a.laps} laps, envelope {"off" if a.no_envelope else "on"}, '
      f'grip shield {"on" if a.grip_shield else "off"}', flush=True)
t0, next_sim = time.time(), None
try:
    while running and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.002)
        now = time.time()
        if S['lap_time'] is None:
            continue
        if next_sim is None:
            next_sim = S['sim']
        if S['sim'] < next_sim:
            if now - max(S['scan_t'], S['odom_t']) > 0.5:
                send(0.0, 0.0)
            continue
        next_sim += DT
        if S['sim'] > next_sim:
            next_sim = S['sim'] + DT
        if S['scan'] is None or S['odom'] is None or now - S['scan_t'] > 0.5 or now - S['odom_t'] > 0.5:
            send(0.0, 0.0)
            continue
        o = S['odom']
        p, q, v = o.pose.pose.position, o.pose.pose.orientation, o.twist.twist.linear.x
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        sim = S['sim']
        if prev_yaw is None:
            yaw_rate, v_hist = 0.0, [v, v, v]
        else:
            dts = max(1e-3, sim - prev_sim)
            yaw_rate = math.atan2(math.sin(yaw - prev_yaw), math.cos(yaw - prev_yaw)) / dts
            v_hist = [v] + v_hist[:2]
        prev_yaw, prev_sim = yaw, sim
        acc_now = (v_hist[0] - v_hist[1]) / DT / float(P['accel_norm'])
        acc_prev = (v_hist[1] - v_hist[2]) / DT / float(P['accel_norm'])
        tail = [np.clip(v / float(P['speed_norm']), 0, 1), last_steer, last_thr_action,
                np.clip(yaw_rate / float(P['yaw_rate_norm']), -1, 1), np.clip(acc_now, -1, 1), np.clip(acc_prev, -1, 1)]
        obs = np.concatenate([preprocess_scan(S['scan']), np.asarray(tail, np.float32)]).astype(np.float32)
        act = actor(obs)
        steer_cmd, thr_cmd = to_commands(act, v)
        if a.grip_shield:
            dem = v * v * math.tan(abs(steer_cmd) * MAXS) / L
            over = over + 1 if dem > 9.2 else 0
            coasting = (coasting and dem > 8.8) or over >= 2
            if coasting:
                thr_cmd = 0.0
        send(thr_cmd, steer_cmd)
        last_steer, last_thr_action = steer_cmd, float(act[1])
        rows.append((now - t0, sim, p.x, p.y, yaw, v, yaw_rate, *act, steer_cmd, thr_cmd,
                     -1 if S['col'] is None else S['col']))
        if laps0 is None and S['laps'] is not None:
            laps0, col0 = S['laps'], S['col'] or 0
        if laps0 is not None and S['laps'] is not None and S['laps'] - laps0 > lap_seen:
            lap_seen = S['laps'] - laps0
            print(f'LAP {lap_seen}: {S["last_lap"]:.3f} s, wall contacts so far {(S["col"] or 0) - col0}', flush=True)
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
    r = np.array(rows)
    late = int(np.sum(np.diff(r[:, 1]) > 0.1)) if len(r) > 1 else 0
    np.savez(log, rows=r, cols=np.array(['t', 'sim_t', 'x', 'y', 'yaw', 'v', 'yaw_rate', 'act_steer', 'act_thr',
                                         'steer_cmd', 'thr_cmd', 'col']))
    print(f'stopped: {lap_seen} laps, wall contacts {(S["col"] or 0) - (col0 or 0)}, {len(rows)} steps, '
          f'{late} late steps (>0.1 s), log {log}', flush=True)
    node.destroy_node()
    rclpy.shutdown()
