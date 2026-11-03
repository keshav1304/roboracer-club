"""Follow a circle drawn in the open gap of the saved room map.

The steering law is the one in
roboracer-sim-competitions/icra-2026/autodrive_devkit/racing_algo/pure_pursuit_node.py:
a lookahead point ahead of the nearest path point, then

    gamma = 2 * y_local / lookahead^2
    steering = atan(gamma * wheelbase)

That sim publishes the result on its own throttle topics. This car takes the
same curvature and sends it on /drive. On /drive, positive steering is left.

The raceline is a rounded rectangle inside the white gap of room.pgm,
rotated 20 degrees to follow the gap. It is 1.6 m along its length and
1.1 m across. Corner radius is 0.55 m, just above the car's 0.49 m
minimum turn.

Foxglove, fixed frame map:
    /map                 the saved grid
    /pursuit/path        the circle
    /pursuit/markers     green circle, blue nearest point, red lookahead
    /scan                lidar, once AMCL has a pose
"""

import math
from pathlib import Path

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path as NavPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

MAP_YAML = Path.home() / "osracer_ws/src/osracer/osracer_slam/maps/room.yaml"

# Rounded rectangle inside the wall loop of room.pgm. Measured from the
# gap center: walls are 1.60 m to -x, 1.20 m to -y, 1.35 m to +y.
# Corner radius stays above the 0.49 m minimum turn.
CENTER_X = 0.525
CENTER_Y = -3.55
HALF_X = 0.90
HALF_Y = 0.9
CORNER_RADIUS = 0.85
# Long axis of the free gap, measured from the walls around the center.
YAW = math.radians(-50.0)
POINTS_PER_CORNER = 16
POINTS_PER_STRAIGHT = 12

# Same geometry constants the sim node uses, scaled to this chassis.
LOOKAHEAD = 0.50
WHEELBASE = 0.28
MAX_STEER_RAD = math.radians(30.0)
SPEED = 0.08


def _arc(cx, cy, radius, start, end, count):
    points = []
    for i in range(count):
        theta = start + (end - start) * i / count
        points.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))
    return points


def circle_points():
    """Counterclockwise rounded rectangle. Straights plus 0.60 m corners."""
    cx, cy = CENTER_X, CENTER_Y
    hx, hy, radius = HALF_X, HALF_Y, CORNER_RADIUS
    straight = POINTS_PER_STRAIGHT
    corner = POINTS_PER_CORNER
    points = []
    # Right side, bottom to top.
    x = cx + hx
    y0, y1 = cy - (hy - radius), cy + (hy - radius)
    for i in range(straight):
        points.append((x, y0 + (y1 - y0) * i / straight))
    # Top-right corner.
    points.extend(_arc(cx + hx - radius, cy + hy - radius, radius, 0.0, math.pi / 2, corner))
    # Top side, right to left.
    y = cy + hy
    x0, x1 = cx + (hx - radius), cx - (hx - radius)
    for i in range(straight):
        points.append((x0 + (x1 - x0) * i / straight, y))
    # Top-left corner.
    points.extend(_arc(cx - hx + radius, cy + hy - radius, radius, math.pi / 2, math.pi, corner))
    # Left side, top to bottom.
    x = cx - hx
    y0, y1 = cy + (hy - radius), cy - (hy - radius)
    for i in range(straight):
        points.append((x, y0 + (y1 - y0) * i / straight))
    # Bottom-left corner.
    points.extend(_arc(cx - hx + radius, cy - hy + radius, radius, math.pi, 3 * math.pi / 2, corner))
    # Bottom side, left to right.
    y = cy - hy
    x0, x1 = cx - (hx - radius), cx + (hx - radius)
    for i in range(straight):
        points.append((x0 + (x1 - x0) * i / straight, y))
    # Bottom-right corner.
    points.extend(_arc(cx + hx - radius, cy - hy + radius, radius, -math.pi / 2, 0.0, corner))
    cosine, sine = math.cos(YAW), math.sin(YAW)
    rotated = []
    for x, y in points:
        dx, dy = x - CENTER_X, y - CENTER_Y
        rotated.append((CENTER_X + cosine * dx - sine * dy, CENTER_Y + sine * dx + cosine * dy))
    return rotated


def load_occupancy(yaml_path):
    """Read a map_server yaml+pgm into an OccupancyGrid. Row 0 is the bottom."""
    meta = {}
    for line in Path(yaml_path).read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    image = Path(yaml_path).parent / meta["image"]
    resolution = float(meta["resolution"])
    origin = [float(v) for v in meta["origin"].strip("[]").split(",")]
    occupied_thresh = float(meta.get("occupied_thresh", 0.65))
    free_thresh = float(meta.get("free_thresh", 0.25))
    negate = int(meta.get("negate", 0))

    raw = image.read_bytes()
    header, rest = raw.split(b"\n", 1)
    if header != b"P5":
        raise ValueError(f"{image} is not a binary PGM")
    while rest.startswith(b"#"):
        rest = rest.split(b"\n", 1)[1]
    size, rest = rest.split(b"\n", 1)
    width, height = (int(v) for v in size.split())
    _maxv, pixels = rest.split(b"\n", 1)
    pixels = pixels[: width * height]

    data = []
    for row in range(height - 1, -1, -1):
        for col in range(width):
            shade = pixels[row * width + col] / 255.0
            occ = shade if negate else 1.0 - shade
            if occ > occupied_thresh:
                data.append(100)
            elif occ < free_thresh:
                data.append(0)
            else:
                data.append(-1)

    grid = OccupancyGrid()
    grid.header.frame_id = "map"
    grid.info.resolution = resolution
    grid.info.width = width
    grid.info.height = height
    grid.info.origin.position.x = origin[0]
    grid.info.origin.position.y = origin[1]
    grid.info.origin.orientation.w = 1.0
    grid.data = data
    return grid


def yaw_from_quaternion(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class CirclePursuit(Node):
    def __init__(self):
        super().__init__("circle_pursuit")
        self.points = circle_points()
        self.grid = load_occupancy(MAP_YAML)

        map_qos = QoSProfile(depth=1)
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        map_qos.history = HistoryPolicy.KEEP_LAST
        drive_qos = QoSProfile(depth=1)
        drive_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        drive_qos.durability = DurabilityPolicy.VOLATILE

        self.map_pub = self.create_publisher(OccupancyGrid, "/map", map_qos)
        self.path_pub = self.create_publisher(NavPath, "/pursuit/path", map_qos)
        self.marker_pub = self.create_publisher(MarkerArray, "/pursuit/markers", 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", drive_qos)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(0.05, self._tick)
        self._publish_map_and_path()
        self.get_logger().info(
            f"rounded rect center ({CENTER_X:.2f}, {CENTER_Y:.2f}) "
            f"{2*HALF_X:.2f} x {2*HALF_Y:.2f} m, yaw={math.degrees(YAW):.0f} deg. "
            "Set the initial pose in Foxglove, then flip SWB down."
        )

    def _publish_map_and_path(self):
        stamp = self.get_clock().now().to_msg()
        self.grid.header.stamp = stamp
        self.map_pub.publish(self.grid)

        path = NavPath()
        path.header.frame_id = "map"
        path.header.stamp = stamp
        for x, y in self.points:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        path.poses.append(path.poses[0])
        self.path_pub.publish(path)

        ring = Marker()
        ring.header.frame_id = "map"
        ring.header.stamp = stamp
        ring.ns = "circle"
        ring.id = 0
        ring.type = Marker.LINE_STRIP
        ring.action = Marker.ADD
        ring.scale.x = 0.04
        ring.color.r = 0.1
        ring.color.g = 0.85
        ring.color.b = 0.2
        ring.color.a = 1.0
        ring.pose.orientation.w = 1.0
        for pose in path.poses:
            ring.points.append(pose.pose.position)
        array = MarkerArray()
        array.markers.append(ring)
        self.marker_pub.publish(array)

    def _nearest(self, x, y):
        best_i = 0
        best_d = float("inf")
        for i, (px, py) in enumerate(self.points):
            d = (px - x) ** 2 + (py - y) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def _lookahead(self, x, y):
        """Walk forward from the nearest point, as the sim node walks along s."""
        start = self._nearest(x, y)
        count = len(self.points)
        for step in range(1, count):
            i = (start + step) % count
            px, py = self.points[i]
            if math.hypot(px - x, py - y) >= LOOKAHEAD:
                return px, py, start
        px, py = self.points[start]
        return px, py, start

    def _drive(self, speed, steering):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_footprint"
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self.drive_pub.publish(msg)

    def _sphere(self, marker_id, x, y, red, green, blue):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "pursuit"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.15
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.18
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = 1.0
        return marker

    def _tick(self):
        self._publish_map_and_path()
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "base_footprint", rclpy.time.Time()
            )
        except Exception:
            self._drive(0.0, 0.0)
            return

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        curr_x = translation.x
        curr_y = translation.y
        curr_yaw = yaw_from_quaternion(rotation.x, rotation.y, rotation.z, rotation.w)

        tx, ty, nearest_i = self._lookahead(curr_x, curr_y)
        nx, ny = self.points[nearest_i]
        dx = tx - curr_x
        dy = ty - curr_y
        # Vehicle frame: +x forward, +y left. Same transform as the sim node.
        x_local = dx * math.cos(curr_yaw) + dy * math.sin(curr_yaw)
        y_local = -dx * math.sin(curr_yaw) + dy * math.cos(curr_yaw)

        gamma = 2.0 * y_local / (LOOKAHEAD ** 2)
        steering_angle = math.atan(gamma * WHEELBASE)
        steering_angle = max(-MAX_STEER_RAD, min(MAX_STEER_RAD, steering_angle))
        # Positive /drive steering is left, which is positive y_local.
        norm_steering = steering_angle / MAX_STEER_RAD

        markers = MarkerArray()
        markers.markers.append(self._sphere(1, tx, ty, 1.0, 0.1, 0.1))
        markers.markers.append(self._sphere(2, nx, ny, 0.1, 0.3, 1.0))
        self.marker_pub.publish(markers)
        self._drive(SPEED, norm_steering)


def main():
    rclpy.init()
    node = CirclePursuit()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._drive(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
