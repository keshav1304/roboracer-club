#!/usr/bin/env python3
"""
Kinematic MPC for the AutoDRIVE RoboRacer (playground edition).

Follows a raceline CSV (s, x, y, theta, velocity) in the map frame.
Pose for control comes from /amcl_pose; odom is used only for speed.
"""
import csv
import math
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Float32MultiArray

from racing_algo.kinematic_mpc import KMPCPlanner, MpcConfig

DEBUG_KEYS = (
    "s_curr",
    "target_x",
    "target_y",
    "steering_norm",
    "target_velocity",
    "speed",
    "speed_error",
    "throttle",
    "solve_ms",
)

DEFAULT_WAYPOINT_FILE = os.path.join(
    os.path.dirname(__file__), '..', 'comp_waypoints.csv')

AMCL_STALE_S = 2.0


class MpcNode(Node):
    """Kinematic MPC driven by particle-filter pose (map frame)."""

    def __init__(self):
        super().__init__('mpc_node')

        self.declare_parameter('waypoint_file', DEFAULT_WAYPOINT_FILE)
        self.declare_parameter('max_steering_rad', 0.58)
        self.declare_parameter('throttle_gain', 0.043)
        self.declare_parameter('corner_throttle_gain', 0.025)
        self.declare_parameter('corner_steering_threshold', 0.6)
        self.declare_parameter('kp_speed', 0.20)
        self.declare_parameter('ki_speed', 0.05)
        self.declare_parameter('integral_limit', 2.0)
        self.declare_parameter('velocity_scale', 1.0)
        self.declare_parameter('steering_direction', -1.0)
        self.declare_parameter('path_direction', -1.0)
        self.declare_parameter('max_speed', 6.0)
        self.declare_parameter('q_pos', 13.5)
        self.declare_parameter('q_yaw', 13.0)
        self.declare_parameter('q_vel', 5.5)
        self.declare_parameter('r_accel', 0.01)
        self.declare_parameter('r_steer', 100.0)
        self.declare_parameter('amcl_stale_s', AMCL_STALE_S)

        self._load_params()
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.raw_waypoints = self._load_waypoints(
            self.get_parameter('waypoint_file').value)
        self.last_closest_idx = None
        self.speed = 0.0
        self.speed_integral = 0.0
        self._last_control_time = None
        self._amcl_pose = None
        self._amcl_stamp_mono = 0.0
        self._warned_stale = False
        self._last_steer = 0.0

        self.planner = self._make_planner()

        self.amcl_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose',
            self.amcl_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/autodrive/roboracer_1/odom',
            self.odom_callback, 10)

        self.throttle_pub = self.create_publisher(
            Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.steering_pub = self.create_publisher(
            Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.debug_pub = self.create_publisher(
            Float32MultiArray, '/playground/mpc/debug', 10)

        self.get_logger().info(
            f"MPC initialized with {len(self.raw_waypoints)} waypoints "
            f"(pose=/amcl_pose, path_direction={self.path_direction})")

    def _load_params(self):
        self.max_steering_rad = float(
            self.get_parameter('max_steering_rad').value)
        self.throttle_gain = float(self.get_parameter('throttle_gain').value)
        self.corner_throttle_gain = float(
            self.get_parameter('corner_throttle_gain').value)
        self.corner_steering_threshold = float(
            self.get_parameter('corner_steering_threshold').value)
        self.kp_speed = float(self.get_parameter('kp_speed').value)
        self.ki_speed = float(self.get_parameter('ki_speed').value)
        self.integral_limit = float(
            self.get_parameter('integral_limit').value)
        self.velocity_scale = float(
            self.get_parameter('velocity_scale').value)
        self.steering_direction = float(
            self.get_parameter('steering_direction').value)
        self.path_direction = float(
            self.get_parameter('path_direction').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self.q_pos = float(self.get_parameter('q_pos').value)
        self.q_yaw = float(self.get_parameter('q_yaw').value)
        self.q_vel = float(self.get_parameter('q_vel').value)
        self.r_accel = float(self.get_parameter('r_accel').value)
        self.r_steer = float(self.get_parameter('r_steer').value)
        self.amcl_stale_s = float(self.get_parameter('amcl_stale_s').value)

    def _cost_names(self):
        return (
            'q_pos', 'q_yaw', 'q_vel', 'r_accel', 'r_steer',
            'max_speed', 'max_steering_rad',
        )

    def _make_planner(self) -> KMPCPlanner:
        steer_lim = max(0.05, abs(self.max_steering_rad))
        cfg = MpcConfig(
            Rk=np.diag([self.r_accel, self.r_steer]),
            Rdk=np.diag([self.r_accel, self.r_steer]),
            Qk=np.diag([self.q_pos, self.q_pos, self.q_vel, self.q_yaw]),
            Qfk=np.diag([self.q_pos, self.q_pos, self.q_vel, self.q_yaw]),
            MAX_SPEED=max(0.5, self.max_speed),
            MAX_STEER=steer_lim,
            MIN_STEER=-steer_lim,
        )
        planner = KMPCPlanner(waypoints=self._mpc_waypoints(), config=cfg)
        return planner

    def _mpc_waypoints(self):
        """Build [cx, cy, cyaw, speed] for the planner from the CSV rows."""
        if len(self.raw_waypoints) == 0:
            return [
                np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1),
            ]
        wps = self.raw_waypoints
        if self.path_direction < 0:
            # Reverse travel sense along the stored raceline.
            x = wps[::-1, 1].copy()
            y = wps[::-1, 2].copy()
            yaw = wps[::-1, 3].copy() + math.pi
            v = wps[::-1, 4].copy() * self.velocity_scale
            for i in range(len(yaw)):
                while yaw[i] > math.pi:
                    yaw[i] -= 2.0 * math.pi
                while yaw[i] < -math.pi:
                    yaw[i] += 2.0 * math.pi
        else:
            x = wps[:, 1].copy()
            y = wps[:, 2].copy()
            yaw = wps[:, 3].copy()
            v = wps[:, 4].copy() * self.velocity_scale
        v = np.clip(v, 0.0, self.max_speed)
        return [x, y, yaw, v]

    def _refresh_planner_waypoints(self):
        self.planner.waypoints = self._mpc_waypoints()

    def _on_set_parameters(self, params):
        for p in params:
            if p.name == 'max_steering_rad' and float(p.value) <= 0.0:
                return SetParametersResult(
                    successful=False, reason='max_steering_rad must be > 0')
            if p.name in (
                    'throttle_gain', 'corner_throttle_gain', 'velocity_scale',
                    'kp_speed', 'ki_speed', 'integral_limit', 'amcl_stale_s',
                    'max_speed', 'q_pos', 'q_yaw', 'q_vel',
                    'r_accel', 'r_steer') and float(p.value) < 0.0:
                return SetParametersResult(
                    successful=False, reason=f'{p.name} must be >= 0')
            if p.name in ('steering_direction', 'path_direction'):
                if float(p.value) not in (-1.0, 1.0):
                    return SetParametersResult(
                        successful=False,
                        reason=f'{p.name} must be -1 or +1')

        rebuild = False
        refresh_wps = False
        for p in params:
            if p.name == 'waypoint_file':
                wps = self._load_waypoints(str(p.value))
                if len(wps) == 0:
                    return SetParametersResult(
                        successful=False,
                        reason=f'Could not load waypoints from {p.value}')
                self.raw_waypoints = wps
                self.last_closest_idx = None
                refresh_wps = True
            elif hasattr(self, p.name):
                setattr(self, p.name, float(p.value))
                if p.name in ('ki_speed', 'integral_limit'):
                    self.speed_integral = 0.0
                if p.name in self._cost_names():
                    rebuild = True
                if p.name in (
                        'velocity_scale', 'path_direction', 'max_speed'):
                    refresh_wps = True

        if rebuild:
            self.planner = self._make_planner()
        elif refresh_wps:
            self._refresh_planner_waypoints()
        return SetParametersResult(successful=True)

    def _load_waypoints(self, path: str) -> np.ndarray:
        rows = []
        try:
            with open(path, 'r') as f:
                for row in csv.reader(f):
                    if not row:
                        continue
                    try:
                        rows.append([float(x) for x in row[:5]])
                    except ValueError:
                        continue
        except OSError as e:
            self.get_logger().error(f"Failed to load waypoints: {e}")
            return np.zeros((0, 5))
        wps = np.array(rows, dtype=float)
        if wps.ndim != 2 or wps.shape[0] == 0:
            return np.zeros((0, 5))
        if wps.shape[1] < 5:
            pad = np.zeros((wps.shape[0], 5 - wps.shape[1]))
            wps = np.hstack([wps, pad])
        self.get_logger().info(f"Loaded {len(wps)} waypoints from {path}")
        return wps

    @staticmethod
    def euler_from_quaternion(x, y, z, w):
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def get_frenet_s(self, x, y):
        num_waypoints = len(self.raw_waypoints)
        if num_waypoints == 0:
            return 0.0, x, y
        if self.last_closest_idx is None:
            self.last_closest_idx = int(np.argmin(np.linalg.norm(
                self.raw_waypoints[:, 1:3] - np.array([x, y]), axis=1)))

        search_back = 20
        search_fwd = 100
        best_idx = self.last_closest_idx
        min_dist_sq = float('inf')
        for i in range(-search_back, search_fwd):
            idx = (self.last_closest_idx + i) % num_waypoints
            wp = self.raw_waypoints[idx]
            dist_sq = (wp[1] - x) ** 2 + (wp[2] - y) ** 2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_idx = idx
        self.last_closest_idx = best_idx

        next_idx = (best_idx + 1) % num_waypoints
        p1 = self.raw_waypoints[best_idx, 1:3]
        p2 = self.raw_waypoints[next_idx, 1:3]
        s1 = self.raw_waypoints[best_idx, 0]
        s2 = self.raw_waypoints[next_idx, 0]
        if s2 < s1:
            s2 += self.raw_waypoints[-1, 0]

        v = p2 - p1
        w = np.array([x, y]) - p1
        v_norm_sq = np.dot(v, v)
        if v_norm_sq < 1e-6:
            return s1 % self.raw_waypoints[-1, 0], p1[0], p1[1]

        t = np.dot(w, v) / v_norm_sq
        t = max(0.0, min(1.0, t))
        refined_s = s1 + t * (s2 - s1)
        p_path = p1 + t * (p2 - p1)
        return refined_s % self.raw_waypoints[-1, 0], p_path[0], p_path[1]

    def _control_dt(self):
        now = self.get_clock().now()
        if self._last_control_time is None:
            self._last_control_time = now
            return 0.02
        dt = (now - self._last_control_time).nanoseconds * 1e-9
        self._last_control_time = now
        return max(1e-3, min(dt, 0.1))

    def compute_throttle(self, target_v, speed, norm_steering, dt):
        if abs(norm_steering) > self.corner_steering_threshold:
            u_ff = target_v * self.corner_throttle_gain
        else:
            u_ff = target_v * self.throttle_gain

        error = target_v - speed
        self.speed_integral += error * dt
        self.speed_integral = max(
            -self.integral_limit, min(self.integral_limit, self.speed_integral))

        u = u_ff + self.kp_speed * error + self.ki_speed * self.speed_integral
        u_sat = max(-1.0, min(1.0, u))

        if (u_sat >= 1.0 and error > 0.0) or (u_sat <= -1.0 and error < 0.0):
            self.speed_integral -= error * dt
            self.speed_integral = max(
                -self.integral_limit,
                min(self.integral_limit, self.speed_integral))

        return u_sat

    def _publish_hold(self):
        self.throttle_pub.publish(Float32(data=0.0))
        self.steering_pub.publish(Float32(data=0.0))

    def odom_callback(self, msg: Odometry):
        tw = msg.twist.twist.linear
        self.speed = math.sqrt(tw.x ** 2 + tw.y ** 2)

    def amcl_callback(self, msg: PoseWithCovarianceStamped):
        self._amcl_pose = msg
        self._amcl_stamp_mono = time.monotonic()
        self._warned_stale = False
        self._run_control()

    def _run_control(self):
        if len(self.raw_waypoints) == 0:
            return

        age = time.monotonic() - self._amcl_stamp_mono
        if self._amcl_pose is None or age > self.amcl_stale_s:
            if not self._warned_stale:
                self.get_logger().warn(
                    "Particle filter pose stale/missing; holding zero "
                    "throttle/steer (no odom fallback for control)")
                self._warned_stale = True
            self._publish_hold()
            return

        pose = self._amcl_pose.pose.pose
        curr_x = pose.position.x
        curr_y = pose.position.y
        q = pose.orientation
        curr_yaw = self.euler_from_quaternion(q.x, q.y, q.z, q.w)
        dt = self._control_dt()

        s_curr, _, _ = self.get_frenet_s(curr_x, curr_y)

        state = np.array([
            curr_x, curr_y, self._last_steer, self.speed,
            curr_yaw, 0.0, 0.0,
        ], dtype=float)

        t0 = time.perf_counter()
        steer, target_v = self.planner.plan(state)
        solve_ms = (time.perf_counter() - t0) * 1000.0

        if steer is None or target_v is None:
            self._publish_hold()
            return

        steer = max(-self.max_steering_rad, min(self.max_steering_rad, steer))
        self._last_steer = float(steer)

        norm_steering = self.steering_direction * (
            steer / self.max_steering_rad)
        self.steering_pub.publish(Float32(data=float(-norm_steering)))

        target_v = float(target_v)
        throttle = self.compute_throttle(
            target_v, self.speed, norm_steering, dt)
        self.throttle_pub.publish(Float32(data=float(throttle)))

        ref_x = self.planner.last_ref_x
        ref_y = self.planner.last_ref_y
        tx = float(ref_x[0]) if len(ref_x) else curr_x
        ty = float(ref_y[0]) if len(ref_y) else curr_y

        pred_x = self.planner.last_pred_x
        pred_y = self.planner.last_pred_y
        n_pred = int(min(len(pred_x), len(pred_y)))

        data = [
            float(s_curr),
            float(tx),
            float(ty),
            float(-norm_steering),
            float(target_v),
            float(self.speed),
            float(target_v - self.speed),
            float(throttle),
            float(solve_ms),
            float(n_pred),
        ]
        for i in range(n_pred):
            data.append(float(pred_x[i]))
            data.append(float(pred_y[i]))

        debug = Float32MultiArray()
        debug.data = data
        self.debug_pub.publish(debug)


def main(args=None):
    rclpy.init(args=args)
    node = MpcNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
