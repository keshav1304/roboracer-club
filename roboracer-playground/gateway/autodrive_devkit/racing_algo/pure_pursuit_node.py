#!/usr/bin/env python3
"""
Pure Pursuit for the AutoDRIVE RoboRacer (playground edition).

Follows a raceline CSV (s, x, y, theta, velocity) in the **map** frame.
Pose for control comes from the particle filter (/amcl_pose); ground-truth odom is used
only for speed (twist) and is not used for path tracking.
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
from std_msgs.msg import ColorRGBA, Float32, Float32MultiArray
from visualization_msgs.msg import Marker

# Layout for /playground/pure_pursuit/debug (Float32MultiArray)
DEBUG_KEYS = (
    "s_curr",
    "target_x",
    "target_y",
    "y_local",
    "steering_norm",
    "target_velocity",
    "speed",
    "speed_error",
    "lookahead",
    "throttle",
)

DEFAULT_WAYPOINT_FILE = os.path.join(
    os.path.dirname(__file__), '..', 'comp_waypoints.csv')

# If the particle-filter pose is older than this, hold zero commands (no odom fallback).
# Keep generous: the filter may only update on scan cycles (~20–40 Hz max).
AMCL_STALE_S = 2.0


class PurePursuit(Node):
    """Frenet-frame pure pursuit driven by particle-filter pose (map frame)."""

    def __init__(self):
        super().__init__('pure_pursuit_node')

        self.declare_parameter('waypoint_file', DEFAULT_WAYPOINT_FILE)
        self.declare_parameter('lookahead', 1.0)
        self.declare_parameter('wheelbase', 0.33)
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
        self.declare_parameter('amcl_stale_s', AMCL_STALE_S)

        self._load_params()
        self.add_on_set_parameters_callback(self._on_set_parameters)

        self.waypoints = self._load_waypoints(
            self.get_parameter('waypoint_file').value)
        self.last_closest_idx = None
        self.speed = 0.0
        self.speed_integral = 0.0
        self._last_control_time = None
        self._amcl_pose = None
        self._amcl_stamp_mono = 0.0
        self._warned_stale = False

        # Control pose: particle filter (map frame). Odom: speed only.
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
        self.lookahead_pub = self.create_publisher(
            Marker, '/autodrive/roboracer_1/lookahead_marker', 10)
        self.debug_pub = self.create_publisher(
            Float32MultiArray, '/playground/pure_pursuit/debug', 10)

        self.get_logger().info(
            f"Pure Pursuit initialized with {len(self.waypoints)} waypoints "
            f"(pose=/amcl_pose, path_direction={self.path_direction})")

    # --- parameters ---------------------------------------------------------

    def _load_params(self):
        self.lookahead = float(self.get_parameter('lookahead').value)
        self.wheelbase = float(self.get_parameter('wheelbase').value)
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
        self.amcl_stale_s = float(self.get_parameter('amcl_stale_s').value)

    def _on_set_parameters(self, params):
        for p in params:
            if p.name == 'lookahead' and not (0.1 <= float(p.value) <= 5.0):
                return SetParametersResult(
                    successful=False, reason='lookahead must be in [0.1, 5]')
            if p.name == 'max_steering_rad' and float(p.value) <= 0.0:
                return SetParametersResult(
                    successful=False, reason='max_steering_rad must be > 0')
            if p.name in ('throttle_gain', 'corner_throttle_gain',
                          'velocity_scale', 'kp_speed', 'ki_speed',
                          'integral_limit', 'amcl_stale_s') and float(p.value) < 0.0:
                return SetParametersResult(
                    successful=False, reason=f'{p.name} must be >= 0')
            if p.name in ('steering_direction', 'path_direction'):
                v = float(p.value)
                if v not in (-1.0, 1.0):
                    return SetParametersResult(
                        successful=False,
                        reason=f'{p.name} must be -1 or +1')

        for p in params:
            if p.name == 'waypoint_file':
                wps = self._load_waypoints(str(p.value))
                if len(wps) == 0:
                    return SetParametersResult(
                        successful=False,
                        reason=f'Could not load waypoints from {p.value}')
                self.waypoints = wps
                self.last_closest_idx = None
            elif hasattr(self, p.name):
                setattr(self, p.name, float(p.value))
                if p.name in ('ki_speed', 'integral_limit'):
                    self.speed_integral = 0.0
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

    # --- geometry -----------------------------------------------------------

    @staticmethod
    def euler_from_quaternion(x, y, z, w):
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def get_frenet_s(self, x, y):
        """Map-frame (x, y) → Frenet s via localized search + projection."""
        num_waypoints = len(self.waypoints)
        if self.last_closest_idx is None:
            self.last_closest_idx = int(np.argmin(np.linalg.norm(
                self.waypoints[:, 1:3] - np.array([x, y]), axis=1)))

        search_back = 20
        search_fwd = 100
        best_idx = self.last_closest_idx
        min_dist_sq = float('inf')
        for i in range(-search_back, search_fwd):
            idx = (self.last_closest_idx + i) % num_waypoints
            wp = self.waypoints[idx]
            dist_sq = (wp[1] - x) ** 2 + (wp[2] - y) ** 2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_idx = idx
        self.last_closest_idx = best_idx

        next_idx = (best_idx + 1) % num_waypoints
        p1 = self.waypoints[best_idx, 1:3]
        p2 = self.waypoints[next_idx, 1:3]
        s1 = self.waypoints[best_idx, 0]
        s2 = self.waypoints[next_idx, 0]
        if s2 < s1:
            s2 += self.waypoints[-1, 0]

        v = p2 - p1
        w = np.array([x, y]) - p1
        v_norm_sq = np.dot(v, v)
        if v_norm_sq < 1e-6:
            return s1 % self.waypoints[-1, 0], p1[0], p1[1]

        t = np.dot(w, v) / v_norm_sq
        t = max(0.0, min(1.0, t))
        refined_s = s1 + t * (s2 - s1)
        p_path = p1 + t * (p2 - p1)
        return refined_s % self.waypoints[-1, 0], p_path[0], p_path[1]

    def get_point_at_s(self, s_target):
        """Interpolate (x, y, v) at arc-length s_target."""
        max_s = self.waypoints[-1, 0]
        s_target = s_target % max_s

        idx = np.searchsorted(self.waypoints[:, 0], s_target) - 1
        idx = max(0, min(len(self.waypoints) - 2, idx))

        s0 = self.waypoints[idx, 0]
        s1 = self.waypoints[idx + 1, 0]
        p0 = self.waypoints[idx, 1:3]
        p1 = self.waypoints[idx + 1, 1:3]
        v0 = self.waypoints[idx, 4]
        v1 = self.waypoints[idx + 1, 4]

        ratio = (s_target - s0) / max(s1 - s0, 1e-9)
        p_interp = p0 + ratio * (p1 - p0)
        v_interp = v0 + ratio * (v1 - v0)
        return p_interp[0], p_interp[1], v_interp

    def publish_markers(self, tx, ty, cx, cy):
        """Target = red, current path pose = blue (map frame)."""
        t_marker = Marker()
        t_marker.header.frame_id = "map"
        t_marker.header.stamp = self.get_clock().now().to_msg()
        t_marker.ns = "lookahead"
        t_marker.id = 1
        t_marker.type = Marker.SPHERE
        t_marker.action = Marker.ADD
        t_marker.pose.position.x = float(tx)
        t_marker.pose.position.y = float(ty)
        t_marker.pose.position.z = 0.2
        t_marker.scale.x = t_marker.scale.y = t_marker.scale.z = 0.3
        t_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
        self.lookahead_pub.publish(t_marker)

        c_marker = Marker()
        c_marker.header.frame_id = "map"
        c_marker.header.stamp = self.get_clock().now().to_msg()
        c_marker.ns = "curr_s"
        c_marker.id = 2
        c_marker.type = Marker.SPHERE
        c_marker.action = Marker.ADD
        c_marker.pose.position.x = float(cx)
        c_marker.pose.position.y = float(cy)
        c_marker.pose.position.z = 0.1
        c_marker.scale.x = c_marker.scale.y = c_marker.scale.z = 0.2
        c_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
        self.lookahead_pub.publish(c_marker)

    # --- control ------------------------------------------------------------

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
        """Zero commands when the particle-filter pose is missing/stale."""
        self.throttle_pub.publish(Float32(data=0.0))
        self.steering_pub.publish(Float32(data=0.0))

    def odom_callback(self, msg: Odometry):
        """Speed only — never used for path pose."""
        tw = msg.twist.twist.linear
        self.speed = math.sqrt(tw.x ** 2 + tw.y ** 2)

    def amcl_callback(self, msg: PoseWithCovarianceStamped):
        self._amcl_pose = msg
        self._amcl_stamp_mono = time.monotonic()
        self._warned_stale = False
        self._run_control()

    def _run_control(self):
        if len(self.waypoints) == 0:
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

        s_curr, path_x, path_y = self.get_frenet_s(curr_x, curr_y)
        s_target = s_curr + (self.path_direction * self.lookahead)
        tx, ty, tv = self.get_point_at_s(s_target)
        self.publish_markers(tx, ty, path_x, path_y)

        dx = tx - curr_x
        dy = ty - curr_y
        y_local = -dx * math.sin(curr_yaw) + dy * math.cos(curr_yaw)

        L = self.lookahead
        gamma = 2.0 * y_local / (L ** 2)
        steering_angle = math.atan(gamma * self.wheelbase)
        steering_angle = max(-self.max_steering_rad,
                             min(self.max_steering_rad, steering_angle))

        norm_steering = self.steering_direction * (
            steering_angle / self.max_steering_rad)

        self.steering_pub.publish(Float32(data=float(-norm_steering)))

        target_v = float(tv) * self.velocity_scale
        throttle = self.compute_throttle(
            target_v, self.speed, norm_steering, dt)
        self.throttle_pub.publish(Float32(data=float(throttle)))

        debug = Float32MultiArray()
        debug.data = [
            float(s_curr),
            float(tx),
            float(ty),
            float(y_local),
            float(-norm_steering),
            float(target_v),
            float(self.speed),
            float(target_v - self.speed),
            float(self.lookahead),
            float(throttle),
        ]
        self.debug_pub.publish(debug)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
