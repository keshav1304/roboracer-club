import math
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Float32MultiArray


# Layout for /playground/wall_follow/debug (Float32MultiArray)
DEBUG_KEYS = (
    "ray_a",
    "ray_b",
    "alpha",
    "car_dist",
    "car_dist_future",
    "error",
    "p_term",
    "i_term",
    "d_term",
    "desired_dist",
    "theta_rad",
    "lookahead",
)


class WallFollow(Node):
    """
    Wall Following node adapted for AutoDRIVE Simulator.
    Follows the left wall using a PID controller.

    All gains and setpoints are ROS 2 parameters, so they can be tuned live
    (e.g. from the web playground) via `ros2 param set` / SetParameters
    without restarting the node.
    """

    def __init__(self):
        super().__init__('wall_follow_node')

        # Tunable parameters (defaults match the previous hardcoded values)
        self.declare_parameter('kp', 6.8)
        self.declare_parameter('kd', 2.0)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('desired_dist', 0.7)      # meters to left wall
        self.declare_parameter('lookahead', 0.45)        # meters projected ahead
        self.declare_parameter('throttle', 0.05)         # normalized [-1, 1]
        self.declare_parameter('max_steering_rad', 0.6)  # for normalization
        self.declare_parameter('theta_deg', 65.0)        # second-ray offset

        self._load_params()
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # AutoDRIVE topics
        lidarscan_topic = '/autodrive/roboracer_1/lidar'
        throttle_topic = '/autodrive/roboracer_1/throttle_command'
        steering_topic = '/autodrive/roboracer_1/steering_command'

        # Subscriber
        self.subscription = self.create_subscription(
            LaserScan,
            lidarscan_topic,
            self.scan_callback,
            10)

        # Publishers
        self.throttle_pub = self.create_publisher(Float32, throttle_topic, 10)
        self.steering_pub = self.create_publisher(Float32, steering_topic, 10)
        self.error_pub = self.create_publisher(
            Float32, '/playground/wall_follow/error', 10)
        self.debug_pub = self.create_publisher(
            Float32MultiArray, '/playground/wall_follow/debug', 10)

        # PID state
        self.integral = 0.0
        self.prev_error = 0.0

        self.get_logger().info("Wall Follow Node for AutoDRIVE Initialized!")

    def _load_params(self):
        self.kp = self.get_parameter('kp').value
        self.kd = self.get_parameter('kd').value
        self.ki = self.get_parameter('ki').value
        self.desired_dist = self.get_parameter('desired_dist').value
        self.lookahead = self.get_parameter('lookahead').value
        self.throttle = self.get_parameter('throttle').value
        self.max_steering_rad = self.get_parameter('max_steering_rad').value
        self.theta = math.radians(self.get_parameter('theta_deg').value)

    def _on_set_parameters(self, params):
        # Validate then re-cache all values so the next scan uses them.
        for p in params:
            if p.name in ('kp', 'kd', 'ki') and float(p.value) < 0.0:
                return SetParametersResult(
                    successful=False, reason=f'{p.name} must be >= 0')
            if p.name == 'desired_dist' and not (0.1 <= float(p.value) <= 3.0):
                return SetParametersResult(
                    successful=False, reason='desired_dist must be in [0.1, 3.0] m')
            if p.name == 'throttle' and not (-1.0 <= float(p.value) <= 1.0):
                return SetParametersResult(
                    successful=False, reason='throttle must be in [-1, 1]')
            if p.name == 'max_steering_rad' and float(p.value) <= 0.0:
                return SetParametersResult(
                    successful=False, reason='max_steering_rad must be > 0')

        for p in params:
            if p.name == 'theta_deg':
                self.theta = math.radians(float(p.value))
            elif hasattr(self, p.name):
                setattr(self, p.name, float(p.value))
        return SetParametersResult(successful=True)

    def get_range(self, range_data, angle, angle_min, angle_increment):
        """Return the range measurement at a given angle (radians)."""
        index = int((angle - angle_min) / angle_increment)
        index = max(0, min(index, len(range_data) - 1))
        r = range_data[index]
        if math.isinf(r) or math.isnan(r):
            return 10.0  # fallback to max range
        return r

    def get_error(self, range_data, dist, angle_min, angle_increment):
        """
        Calculate the error to the left wall using two rays.
        Returns (error, ray_a, ray_b, alpha, car_dist, car_dist_future).
        """
        ray_b_angle = math.radians(90)   # 90° left = perpendicular to left wall
        ray_a_angle = ray_b_angle - self.theta  # angled forward

        b = self.get_range(range_data, ray_b_angle, angle_min, angle_increment)
        a = self.get_range(range_data, ray_a_angle, angle_min, angle_increment)

        alpha = math.atan2(a * math.cos(self.theta) - b, a * math.sin(self.theta))
        car_dist = b * math.cos(alpha)
        car_dist_future = car_dist + self.lookahead * math.sin(alpha)

        return dist - car_dist_future, a, b, alpha, car_dist, car_dist_future

    def scan_callback(self, msg):
        """Callback for LaserScan messages."""
        error, ray_a, ray_b, alpha, car_dist, car_dist_future = self.get_error(
            msg.ranges, self.desired_dist, msg.angle_min, msg.angle_increment)

        # PID control
        self.integral += error
        derivative = error - self.prev_error
        p_term = self.kp * error
        i_term = self.ki * self.integral
        d_term = self.kd * derivative
        correction = p_term + i_term + d_term
        self.prev_error = error

        # Normalize steering to [-1, 1]
        norm_steering = max(-1.0, min(1.0, -correction / self.max_steering_rad))

        throttle_msg = Float32()
        throttle_msg.data = float(self.throttle)
        self.throttle_pub.publish(throttle_msg)

        steering_msg = Float32()
        steering_msg.data = float(norm_steering)
        self.steering_pub.publish(steering_msg)

        error_msg = Float32()
        error_msg.data = float(error)
        self.error_pub.publish(error_msg)

        debug = Float32MultiArray()
        debug.data = [
            float(ray_a),
            float(ray_b),
            float(alpha),
            float(car_dist),
            float(car_dist_future),
            float(error),
            float(p_term),
            float(i_term),
            float(d_term),
            float(self.desired_dist),
            float(self.theta),
            float(self.lookahead),
        ]
        self.debug_pub.publish(debug)


def main(args=None):
    rclpy.init(args=args)
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)
    wall_follow_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
