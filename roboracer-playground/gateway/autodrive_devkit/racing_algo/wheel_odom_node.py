#!/usr/bin/env python3
"""Dead-reckoning odom from legal race-time sensors only.

Subscribes to wheel encoders and IMU gyro. Publishes `/wheel_odom` and an
isolated TF tree (`odom` → `base_link` → `lidar_link`) so AMCL never looks
up the simulator's ground-truth `world` → `roboracer_1` transform.

IMU orientation is ignored (that would be treating AHRS as map pose).
LiDAR is republished on `/scan` with `frame_id=lidar_link`.
"""
import math
import threading

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState, LaserScan
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


def _yaw_to_quat(yaw: float):
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class WheelOdom(Node):
    def __init__(self):
        super().__init__('wheel_odom_node')
        self.declare_parameter('wheel_radius', 0.059)
        self.declare_parameter('track_width', 0.236)
        # AWD + hard throttle makes rear encoders report wheel-spin spikes
        # (~20–30 m/s). Clip translation to a real vehicle speed so odom
        # stays a usable AMCL motion prior; lidar only corrects residual.
        self.declare_parameter('max_speed', 14.0)
        self.declare_parameter('lidar_x', 0.2733)
        self.declare_parameter('lidar_y', 0.0)
        self.declare_parameter('lidar_z', 0.096)

        self.wheel_radius = float(self.get_parameter('wheel_radius').value)
        self.track_width = float(self.get_parameter('track_width').value)
        self.max_speed = float(self.get_parameter('max_speed').value)
        self._lidar_x = float(self.get_parameter('lidar_x').value)
        self._lidar_y = float(self.get_parameter('lidar_y').value)
        self._lidar_z = float(self.get_parameter('lidar_z').value)

        self._left = None
        self._right = None
        self._gyro_z = 0.0
        self._have_imu = False
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._v = 0.0
        self._w = 0.0
        self._last_t = None
        self._last_enc_t = None
        self._prev_left = None
        self._prev_right = None
        self._lock = threading.Lock()
        self._cg_fast = MutuallyExclusiveCallbackGroup()
        self._cg_scan = MutuallyExclusiveCallbackGroup()

        self._tf = TransformBroadcaster(self)
        self._static_tf = StaticTransformBroadcaster(self)
        self._publish_static_lidar_tf()

        self.odom_pub = self.create_publisher(Odometry, '/wheel_odom', 10)
        self.scan_pub = self.create_publisher(LaserScan, '/scan', 10)

        self.create_subscription(
            JointState, '/autodrive/roboracer_1/left_encoder',
            self._left_cb, 10, callback_group=self._cg_fast)
        self.create_subscription(
            JointState, '/autodrive/roboracer_1/right_encoder',
            self._right_cb, 10, callback_group=self._cg_fast)
        self.create_subscription(
            Imu, '/autodrive/roboracer_1/imu', self._imu_cb, 10,
            callback_group=self._cg_fast)
        self.create_subscription(
            LaserScan, '/autodrive/roboracer_1/lidar', self._scan_cb, 10,
            callback_group=self._cg_scan)

        self.create_timer(0.02, self._tick, callback_group=self._cg_fast)
        with self._lock:
            self._publish_locked(self.get_clock().now())
        self.get_logger().info(
            'Wheel odom from encoders + IMU gyro (no IPS / ground-truth odom)')

    def _publish_static_lidar_tf(self):
        tf = TransformStamped()
        tf.header.stamp = Header().stamp
        tf.header.frame_id = 'base_link'
        tf.child_frame_id = 'lidar_link'
        tf.transform.translation.x = self._lidar_x
        tf.transform.translation.y = self._lidar_y
        tf.transform.translation.z = self._lidar_z
        tf.transform.rotation.w = 1.0
        self._static_tf.sendTransform(tf)

    def _joint_angle(self, msg: JointState):
        if msg.position:
            return float(msg.position[0])
        return None

    def _left_cb(self, msg: JointState):
        ang = self._joint_angle(msg)
        if ang is not None:
            with self._lock:
                self._left = ang

    def _right_cb(self, msg: JointState):
        ang = self._joint_angle(msg)
        if ang is not None:
            with self._lock:
                self._right = ang

    def _imu_cb(self, msg: Imu):
        with self._lock:
            self._gyro_z = float(msg.angular_velocity.z)
            self._have_imu = True

    def _tf_stamp(self, now):
        # Stamp slightly in the past so AMCL lookups at "now" are in the
        # TF cache instead of extrapolating into the future.
        return (now - Duration(seconds=0.05)).to_msg()

    def _scan_cb(self, msg: LaserScan):
        now = self.get_clock().now()
        with self._lock:
            self._publish_locked(now)
            stamp = self._tf_stamp(now)
        # Reuse the incoming message (no 1080-range copy) so encoder
        # callbacks are not starved on the executor.
        msg.header.stamp = stamp
        msg.header.frame_id = 'lidar_link'
        self.scan_pub.publish(msg)

    def _tick(self):
        now = self.get_clock().now()
        with self._lock:
            self._tick_locked(now)

    def _tick_locked(self, now):
        if self._last_t is None:
            self._last_t = now
            self._prev_left = self._left
            self._prev_right = self._right
            self._publish_locked(now)
            return
        raw_dt = (now - self._last_t).nanoseconds * 1e-9
        self._last_t = now
        if raw_dt <= 0.0:
            return
        # Clamp only the gyro step so a stalled executor cannot spin yaw;
        # encoder distance is applied in full below.
        dt = min(raw_dt, 0.15)

        if self._have_imu:
            self._w = self._gyro_z
        self._yaw = _wrap(self._yaw + self._w * dt)

        if self._left is None or self._right is None:
            self._publish_locked(now)
            return
        if self._prev_left is None or self._prev_right is None:
            self._prev_left = self._left
            self._prev_right = self._right
            self._last_enc_t = now
            self._publish_locked(now)
            return

        # AutoDRIVE publishes unbounded cumulative wheel angle (radians),
        # e.g. 2500+. Do not wrap to [-pi, pi]: at racing speed a single
        # tick already exceeds pi and wrapping aliases that to ~-9 m/s.
        dl = self._left - self._prev_left
        dr = self._right - self._prev_right
        if abs(dl) < 1e-12 and abs(dr) < 1e-12:
            self._publish_locked(now)
            return
        self._prev_left = self._left
        self._prev_right = self._right

        ds_l = self.wheel_radius * dl
        ds_r = self.wheel_radius * dr
        ds = 0.5 * (ds_l + ds_r)
        if self._last_enc_t is None:
            dt_enc = max(dt, 1e-3)
        else:
            dt_enc = max((now - self._last_enc_t).nanoseconds * 1e-9, 1e-3)
        self._last_enc_t = now
        # Do not drop large Δθ (encoders are ~23 Hz; 7 m/s is already
        # 5–13 rad/tick). Clip wheel-spin spikes to max_speed instead.
        ds_cap = self.max_speed * dt_enc
        if ds > ds_cap:
            ds = ds_cap
        elif ds < -ds_cap:
            ds = -ds_cap
        self._v = ds / dt_enc
        if not self._have_imu:
            self._w = (ds_r - ds_l) / max(self.track_width, 1e-6) / dt_enc
        self._x += ds * math.cos(self._yaw)
        self._y += ds * math.sin(self._yaw)
        self._publish_locked(now)

    def _publish(self, now):
        with self._lock:
            self._publish_locked(now)

    def _publish_locked(self, now):
        qx, qy, qz, qw = _yaw_to_quat(self._yaw)
        stamp = self._tf_stamp(now)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'
        odom.pose.pose.position.x = self._x
        odom.pose.pose.position.y = self._y
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = self._v
        odom.twist.twist.angular.z = self._w
        self.odom_pub.publish(odom)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = 'odom'
        tf.child_frame_id = 'base_link'
        tf.transform.translation.x = self._x
        tf.transform.translation.y = self._y
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = qw

        lidar = TransformStamped()
        lidar.header.stamp = stamp
        lidar.header.frame_id = 'base_link'
        lidar.child_frame_id = 'lidar_link'
        lidar.transform.translation.x = self._lidar_x
        lidar.transform.translation.y = self._lidar_y
        lidar.transform.translation.z = self._lidar_z
        lidar.transform.rotation.w = 1.0
        self._tf.sendTransform([tf, lidar])


def main(args=None):
    rclpy.init(args=args)
    node = WheelOdom()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
