import math
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

import numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Float32MultiArray


class ReactiveFollowGap(Node):
    """
    Follow the Gap adapted for AutoDRIVE Simulator.

    All tunables are ROS 2 parameters so they can be adjusted live
    (e.g. from the web playground) without restarting the node.
    """

    def __init__(self):
        super().__init__('follow_gap_node')

        # Tunable parameters (defaults match the previous hardcoded values)
        self.declare_parameter('vehicle_half_width', 0.55)   # m, safety bubble
        self.declare_parameter('disparity_threshold', 2.5)   # m, depth jump
        self.declare_parameter('smoothing_window', 25.0)     # beams (used as int)
        self.declare_parameter('free_space_threshold', 0.9)  # m, "free" cutoff
        self.declare_parameter('best_point_threshold', 2.0)  # m, sub-gap cutoff
        self.declare_parameter('max_steering_rad', 0.523)    # ~30 deg
        self.declare_parameter('fov_deg', 100.0)             # closest-point FOV
        self.declare_parameter('heading_weight', 0.7)        # 0..1 straight bias
        self.declare_parameter('throttle', 0.04)             # normalized

        self._load_params()
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # AutoDRIVE Topics
        lidarscan_topic = '/autodrive/roboracer_1/lidar'
        throttle_topic = '/autodrive/roboracer_1/throttle_command'
        steering_topic = '/autodrive/roboracer_1/steering_command'

        # Subscribe to LIDAR
        self.subscription = self.create_subscription(
            LaserScan,
            lidarscan_topic,
            self.lidar_callback,
            10)

        # Publish to throttle and steering directly via Float32 messages
        self.throttle_pub = self.create_publisher(Float32, throttle_topic, 10)
        self.steering_pub = self.create_publisher(Float32, steering_topic, 10)
        # Debug: best_point_angle, best_point_range, closest_range,
        #        gap_start_angle, gap_end_angle, fov_deg
        self.debug_pub = self.create_publisher(
            Float32MultiArray, '/playground/follow_gap/debug', 10)

        self.get_logger().info("Follow The Gap Node for AutoDRIVE Built!")

    def _load_params(self):
        self.vehicle_half_width = self.get_parameter('vehicle_half_width').value
        self.disparity_threshold = self.get_parameter('disparity_threshold').value
        self.smoothing_window = int(self.get_parameter('smoothing_window').value)
        self.free_space_threshold = self.get_parameter('free_space_threshold').value
        self.best_point_threshold = self.get_parameter('best_point_threshold').value
        self.max_steering_rad = self.get_parameter('max_steering_rad').value
        self.fov_deg = self.get_parameter('fov_deg').value
        self.heading_weight = self.get_parameter('heading_weight').value
        self.throttle = self.get_parameter('throttle').value

    def _on_set_parameters(self, params):
        for p in params:
            v = float(p.value)
            if p.name == 'throttle' and not (-1.0 <= v <= 1.0):
                return SetParametersResult(
                    successful=False, reason='throttle must be in [-1, 1]')
            if p.name == 'heading_weight' and not (0.0 <= v <= 1.0):
                return SetParametersResult(
                    successful=False, reason='heading_weight must be in [0, 1]')
            if p.name == 'smoothing_window' and v < 1:
                return SetParametersResult(
                    successful=False, reason='smoothing_window must be >= 1')
            if p.name == 'max_steering_rad' and v <= 0.0:
                return SetParametersResult(
                    successful=False, reason='max_steering_rad must be > 0')
            if p.name in ('vehicle_half_width', 'disparity_threshold',
                          'free_space_threshold', 'best_point_threshold',
                          'fov_deg') and v <= 0.0:
                return SetParametersResult(
                    successful=False, reason=f'{p.name} must be > 0')

        for p in params:
            if p.name == 'smoothing_window':
                self.smoothing_window = int(float(p.value))
            elif hasattr(self, p.name):
                setattr(self, p.name, float(p.value))
        return SetParametersResult(successful=True)

    def preprocess_lidar(self, ranges, range_max):
        # Clamp NaN/inf (no-return beams) to the sensor max: inf would smear
        # through the smoothing means and produce non-JSON-safe debug values.
        proc_ranges = np.nan_to_num(
            np.asarray(ranges, dtype=np.float64),
            nan=range_max, posinf=range_max, neginf=0.0)
        window_size = self.smoothing_window
        for i in range(len(proc_ranges) - window_size):
            mean_val = np.mean(proc_ranges[i:i + window_size])
            proc_ranges[i:i + window_size] = np.array([mean_val] * window_size)

        return proc_ranges

    def find_max_gap(self, free_space_ranges, angle_min, angle_increment):
        # Collect all contiguous gaps
        gaps = []  # list of (start, end) tuples
        current_start = None
        for i in range(len(free_space_ranges)):
            if free_space_ranges[i] >= self.free_space_threshold:
                if current_start is None:
                    current_start = i
            else:
                if current_start is not None:
                    gaps.append((current_start, i))
                    current_start = None
        if current_start is not None:
            gaps.append((current_start, len(free_space_ranges)))

        if not gaps:
            return 0, 0

        # Score each gap: width * avg_depth * heading_weight
        n_beams = len(free_space_ranges)
        center_index = n_beams / 2.0
        best_score = -1
        best_gap = gaps[0]
        for start, end in gaps:
            width = end - start
            avg_depth = np.mean(free_space_ranges[start:end])
            gap_center = (start + end) / 2.0
            # Angle of gap center relative to forward (index at center_index)
            angle = (gap_center - center_index) * angle_increment
            heading_w = (1.0 - self.heading_weight) + self.heading_weight * math.cos(angle)
            score = width * avg_depth * heading_w
            if score > best_score:
                best_score = score
                best_gap = (start, end)

        return best_gap[0], best_gap[1]

    def find_best_point(self, start_i, end_i, ranges):
        max_length = 0
        current_length = 0
        max_length_start_index = start_i
        for i in range(start_i, end_i):
            if ranges[i] >= self.best_point_threshold:
                current_length += 1
                if (current_length) > max_length:
                    max_length = current_length
                    max_length_start_index = i - current_length + 1
            else:
                current_length = 0
        if max_length == 0:
            # No sub-gap found — fall back to center of the gap
            return (start_i + end_i) / 2.0
        return (max_length_start_index + max_length_start_index + max_length) / 2

    def lidar_callback(self, data):
        n_beams = len(data.ranges)
        ranges = np.array(data.ranges, dtype=np.float32)
        proc_ranges = self.preprocess_lidar(ranges, data.range_max)

        # Disparity extension – derive extension width from angle_increment & vehicle width
        ext_beams = max(1, int(math.atan2(self.vehicle_half_width, self.disparity_threshold) / data.angle_increment))
        for i in range(1, n_beams - ext_beams):
            if abs(proc_ranges[i] - proc_ranges[i - 1]) >= self.disparity_threshold:
                fill_val = min(proc_ranges[i - 1], proc_ranges[i])
                proc_ranges[i - 1:i + ext_beams - 1] = fill_val

        # Restrict closest-point search to ±fov_deg/2 around front
        fov_half_rad = math.radians(self.fov_deg / 2.0)
        fov_start = max(0, int((-fov_half_rad - data.angle_min) / data.angle_increment))
        fov_end = min(n_beams, int((fov_half_rad - data.angle_min) / data.angle_increment))

        # Find closest point to LiDAR within FOV
        closest_point_index = fov_start + np.argmin(proc_ranges[fov_start:fov_end])
        closest_range = float(proc_ranges[closest_point_index])

        # Eliminate all points inside 'bubble' (set them to zero)
        if proc_ranges[closest_point_index] > 0.0:
            bubble_size = int(math.atan2(self.vehicle_half_width, proc_ranges[closest_point_index]) / data.angle_increment)
            left = max(0, closest_point_index - bubble_size)
            right = min(n_beams, closest_point_index + bubble_size)
            proc_ranges[left:right] = 0.0

        # Find max length gap
        max_length_start_index, max_length_end_index = self.find_max_gap(proc_ranges, data.angle_min, data.angle_increment)

        # Find the best point in the gap
        best_point_index = self.find_best_point(max_length_start_index, max_length_end_index, proc_ranges)
        best_idx = int(best_point_index)
        best_idx = max(0, min(best_idx, n_beams - 1))
        best_point_range = float(proc_ranges[best_idx]) if proc_ranges[best_idx] > 0 else 0.0

        # Calculate raw steering angle from beam index using actual scan geometry
        steering_angle = 1 * (data.angle_min + best_point_index * data.angle_increment)
        gap_start_angle = data.angle_min + max_length_start_index * data.angle_increment
        gap_end_angle = data.angle_min + max_length_end_index * data.angle_increment

        # Normalize steering angle to [-1.0, 1.0]
        norm_steering = max(-1.0, min(1.0, steering_angle / self.max_steering_rad))

        # Publish Float32 commands to AutoDRIVE
        throttle_msg = Float32()
        throttle_msg.data = float(self.throttle)
        self.throttle_pub.publish(throttle_msg)

        steering_msg = Float32()
        steering_msg.data = float(norm_steering)
        self.steering_pub.publish(steering_msg)

        debug = Float32MultiArray()
        debug.data = [
            float(steering_angle),
            float(best_point_range),
            float(closest_range),
            float(gap_start_angle),
            float(gap_end_angle),
            float(self.fov_deg),
        ]
        self.debug_pub.publish(debug)


def main(args=None):
    rclpy.init(args=args)
    reactive_node = ReactiveFollowGap()
    rclpy.spin(reactive_node)
    reactive_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
