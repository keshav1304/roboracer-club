#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import csv
import os
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, ColorRGBA
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
import math

class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')
        
        # Parameters
        self.lookahead_distance = 1.0  # meters (Reduced from 1.0 for tighter tracking)
        self.wheelbase = 0.33          # meters (from bridge tf)
        self.max_steering_angle = 0.58 # rad
        
        # Load waypoints
        self.waypoints = []
        waypoint_file = os.path.join(os.path.dirname(__file__), '..', 'comp_waypoints.csv')
        # Absolute path fallback
        if not os.path.exists(waypoint_file):
            waypoint_file = '/home/autodrive_devkit/src/autodrive_devkit/comp_waypoints.csv'
        
        try:
            with open(waypoint_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    try:
                        self.waypoints.append([float(x) for x in row])
                    except ValueError:
                        continue  # skip header / comments
            self.waypoints = np.array(self.waypoints)
            self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints.")
        except Exception as e:
            self.get_logger().error(f"Failed to load waypoints: {e}")

        # Subscriptions
        self.odom_sub = self.create_subscription(
            Odometry,
            '/autodrive/roboracer_1/odom',
            self.pose_callback,
            10)

        # Publishers
        self.throttle_pub = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.steering_pub = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.lookahead_pub = self.create_publisher(Marker, '/autodrive/roboracer_1/lookahead_marker', 10)

    def euler_from_quaternion(self, x, y, z, w):
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def publish_markers(self, tx, ty, cx, cy):
        """
        Publish visualization markers for debugging.
        tx, ty: Target Point (Red)
        cx, cy: Current Position on Path (Blue)
        """
        # 1. Target Marker (Red)
        t_marker = Marker()
        t_marker.header.frame_id = "world"
        t_marker.header.stamp = self.get_clock().now().to_msg()
        t_marker.ns = "lookahead"
        t_marker.id = 1
        t_marker.type = Marker.SPHERE
        t_marker.action = Marker.ADD
        t_marker.pose.position.x = float(tx)
        t_marker.pose.position.y = float(ty)
        t_marker.pose.position.z = 0.2
        t_marker.scale.x = 0.3
        t_marker.scale.y = 0.3
        t_marker.scale.z = 0.3
        t_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0) # Red
        self.lookahead_pub.publish(t_marker)

        # 2. Current Position on Path Marker (Blue)
        c_marker = Marker()
        c_marker.header.frame_id = "world"
        c_marker.header.stamp = self.get_clock().now().to_msg()
        c_marker.ns = "curr_s"
        c_marker.id = 2
        c_marker.type = Marker.SPHERE
        c_marker.action = Marker.ADD
        c_marker.pose.position.x = float(cx)
        c_marker.pose.position.y = float(cy)
        c_marker.pose.position.z = 0.1
        c_marker.scale.x = 0.2
        c_marker.scale.y = 0.2
        c_marker.scale.z = 0.2
        c_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0) # Blue
        self.lookahead_pub.publish(c_marker)

    def get_frenet_s(self, x, y):
        """
        Convert world (x, y) to Frenet s using localized search and projection.
        """
        num_waypoints = len(self.waypoints)
        
        # Initialize last_closest_idx globally on first run
        if not hasattr(self, 'last_closest_idx'):
            self.last_closest_idx = np.argmin(np.linalg.norm(self.waypoints[:, 1:3] - np.array([x, y]), axis=1))
        
        # Local search window: generous enough to handle speed and skips
        search_back = 20 
        search_fwd = 100
        best_idx = self.last_closest_idx
        min_dist_sq = float('inf')
        
        for i in range(-search_back, search_fwd):
            idx = (self.last_closest_idx + i) % num_waypoints
            wp = self.waypoints[idx]
            dist_sq = (wp[1] - x)**2 + (wp[2] - y)**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                best_idx = idx
        
        self.last_closest_idx = best_idx
        
        # Projection for refined s
        next_idx = (best_idx + 1) % num_waypoints
        p1 = self.waypoints[best_idx, 1:3]
        p2 = self.waypoints[next_idx, 1:3]
        s1 = self.waypoints[best_idx, 0]
        s2 = self.waypoints[next_idx, 0]
        
        if s2 < s1: s2 += self.waypoints[-1, 0]
        
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
        """
        Interpolate (x, y, v) at the given arc-length s_target.
        """
        max_s = self.waypoints[-1, 0]
        s_target = s_target % max_s
        
        idx = np.searchsorted(self.waypoints[:, 0], s_target) - 1
        idx = max(0, min(len(self.waypoints) - 2, idx))
        
        s0 = self.waypoints[idx, 0]
        s1 = self.waypoints[idx+1, 0]
        p0 = self.waypoints[idx, 1:3]
        p1 = self.waypoints[idx+1, 1:3]
        v0 = self.waypoints[idx, 4]
        v1 = self.waypoints[idx+1, 4]
        
        ratio = (s_target - s0) / (s1 - s0)
        p_interp = p0 + ratio * (p1 - p0)
        v_interp = v0 + ratio * (v1 - v0)
        
        return p_interp[0], p_interp[1], v_interp

    def pose_callback(self, pose_msg):
        if len(self.waypoints) == 0:
            return

        # Current pose
        curr_x = pose_msg.pose.pose.position.x
        curr_y = pose_msg.pose.pose.position.y
        qx = pose_msg.pose.pose.orientation.x
        qy = pose_msg.pose.pose.orientation.y
        qz = pose_msg.pose.pose.orientation.z
        qw = pose_msg.pose.pose.orientation.w
        curr_yaw = self.euler_from_quaternion(qx, qy, qz, qw)

        # 1. Convert Current Position to Frenet Frame
        s_curr, path_x, path_y = self.get_frenet_s(curr_x, curr_y)
        
        # 2. Find Lookahead Target in Frenet Frame
        # NOTE: If the Red marker is physically BEHIND the Blue marker, set path_direction to -1.0
        self.path_direction = 1.0 
        s_target = s_curr + (self.path_direction * self.lookahead_distance)
        
        # 3. Interpolate Target Point in World Frame
        tx, ty, tv = self.get_point_at_s(s_target)
        
        # Publish markers for visualization (Target=Red, CurrentPathPos=Blue)
        self.publish_markers(tx, ty, path_x, path_y)

        # 4. Transform Target to Vehicle Local Frame
        dx = tx - curr_x
        dy = ty - curr_y
        x_local = dx * math.cos(curr_yaw) + dy * math.sin(curr_yaw)
        y_local = -dx * math.sin(curr_yaw) + dy * math.cos(curr_yaw)

        # 5. Calculate Curvature and Steering
        # Curvature gamma = 2*dy / L^2
        # Use lookahead_distance for L to match Pure Pursuit theory precisely
        L = self.lookahead_distance
        gamma = 2.0 * y_local / (L**2)
        steering_angle = math.atan(gamma * self.wheelbase)

        # 6. Limit and Publish Commands
        steering_angle = max(-self.max_steering_angle, min(self.max_steering_angle, steering_angle))
        
        # Inversion check (User manual inversion included)
        self.steering_direction = -1.0 
        norm_steering = self.steering_direction * (steering_angle / self.max_steering_angle)
        
        steering_msg = Float32()
        steering_msg.data = float(-norm_steering)
        self.steering_pub.publish(steering_msg)
        
        throttle_msg = Float32()
        if norm_steering > 0.6:
            throttle_msg.data = float(tv * 0.025)
        else:
            throttle_msg.data = float(tv * 0.043)
        self.throttle_pub.publish(throttle_msg)
        
        # Throttled Debug Logging
        if self.get_clock().now().nanoseconds % 1000000000 < 50000000:
            actual_lookahead = math.sqrt((tx-path_x)**2 + (ty-path_y)**2)
            self.get_logger().info(f"Frenet: s={s_curr:.2f}, target_s={s_target:.2f}")
            self.get_logger().info(f"Local: x={x_local:.3f}, y={y_local:.3f}, Dist={actual_lookahead:.3f}m")
            self.get_logger().info(f"Steer: {norm_steering:.2f} (Yaw: {curr_yaw:.2f})")

def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)
    pure_pursuit_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
