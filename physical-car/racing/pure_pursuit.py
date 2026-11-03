"""Pure pursuit on a saved map, following a raceline drawn in the editor.

Generalizes ../circle_pursuit.py: the raceline comes from
<maps>/<name>_raceline.csv (written by raceline_editor.py) instead of a
hardcoded rounded rectangle, and the map is chosen with --map.

Steering law (same as the ICRA 2026 sim node):
    lookahead point = first raceline point >= L ahead of the nearest
    gamma           = 2 * y_local / L^2
    steering        = atan(gamma * wheelbase), clipped, normalized to [-1, 1]
On /drive, positive steering is left. /drive speed is normalized throttle
(the throttle node clamps it to [-1, 1] and scales by its cap).

Speed is inversely proportional to the upcoming curvature:
    kappa = max |curvature| over the next (L + preview) meters of raceline
    v     = clip(max_speed * (1/kappa) / full_speed_radius, min_speed, max_speed)
so any radius >= full_speed_radius runs at max_speed, and tighter turns slow
down in proportion. The command is rate limited (accel / decel per second).

Lookahead grows with the speed command:
    L = L_min + (L_max - L_min) * (v - min_speed) / (max_speed - min_speed)

Foxglove, fixed frame map:
    /map                 the saved grid (AMCL reads it from here)
    /pursuit/path        the raceline
    /pursuit/markers     green line, blue nearest point, red lookahead
    /scan                lidar, once AMCL has a pose

Usage:
    python3 pure_pursuit.py --map track1 [--max-speed 0.36] [--min-speed 0.13]
        [--full-speed-radius 1.93] [--lookahead-min 0.45] [--lookahead-max 1.45]
"""

import argparse
import math
import os
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

MAPS_DIR = Path(
    os.environ.get(
        "RACING_MAPS_DIR",
        Path.home() / "osracer_ws/src/osracer/osracer_slam/maps",
    )
)


def load_raceline(csv_path):
    """Read x,y rows. Lines starting with # are comments."""
    points = []
    for line in Path(csv_path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        x, y = line.split(",")[:2]
        points.append((float(x), float(y)))
    if len(points) < 3:
        raise ValueError(f"{csv_path} has fewer than 3 points")
    return points


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


def closed_loop_geometry(points, span_m=0.15):
    """Per-point |curvature| (1/m) and mean spacing for a closed raceline.

    Curvature is 1 / circumradius of (p[i-k], p[i], p[i+k]) with k chosen so
    the three points span about +-span_m, which filters waypoint jitter.
    """
    n = len(points)
    seg = [math.hypot(points[(i + 1) % n][0] - points[i][0],
                      points[(i + 1) % n][1] - points[i][1]) for i in range(n)]
    ds = max(sum(seg) / n, 1e-3)
    k = max(1, int(round(span_m / ds)))
    kappa = []
    for i in range(n):
        ax, ay = points[(i - k) % n]
        px, py = points[i]
        bx, by = points[(i + k) % n]
        a = math.hypot(px - ax, py - ay)
        b = math.hypot(bx - px, by - py)
        c = math.hypot(bx - ax, by - ay)
        cross = abs((px - ax) * (by - ay) - (py - ay) * (bx - ax))
        kappa.append(0.0 if a * b * c < 1e-12 else 2.0 * cross / (a * b * c))
    return kappa, ds


class PurePursuit(Node):
    def __init__(self, args):
        super().__init__("pure_pursuit")
        self.max_speed = args.max_speed
        self.min_speed = min(args.min_speed, args.max_speed)
        self.full_speed_radius = args.full_speed_radius
        self.lookahead_min = args.lookahead_min
        self.lookahead_max = max(args.lookahead_max, args.lookahead_min)
        self.preview = args.preview
        self.accel = args.accel
        self.decel = args.decel
        self.wheelbase = args.wheelbase
        self.max_steer = math.radians(args.max_steer_deg)

        map_yaml = MAPS_DIR / f"{args.map}.yaml"
        raceline_csv = Path(args.raceline) if args.raceline else MAPS_DIR / f"{args.map}_raceline.csv"
        self.points = load_raceline(raceline_csv)
        self.kappa, self.ds = closed_loop_geometry(self.points)
        self.grid = load_occupancy(map_yaml)

        # State: commanded speed (rate limited), lookahead, last nearest index.
        self.v_cmd = 0.0
        self.lookahead = self.lookahead_min
        self.last_nearest = None
        self.last_tick = None

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
        self.create_timer(2.0, self._publish_map_and_path)
        self._publish_map_and_path()
        r_min = 1.0 / max(max(self.kappa), 1e-6)
        self.get_logger().info(
            f"map={map_yaml.name} raceline={raceline_csv.name} ({len(self.points)} pts, "
            f"tightest radius {r_min:.2f} m). speed {self.min_speed}..{self.max_speed} "
            f"(full at radius >= {self.full_speed_radius} m), "
            f"lookahead {self.lookahead_min}..{self.lookahead_max} m. "
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
        ring.ns = "raceline"
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
        """Nearest raceline index. Searches a window around the last match so
        the car can't jump to another part of the track where it passes close
        by; falls back to a global search when it is lost (> 1 m off)."""
        n = len(self.points)

        def best_in(indices):
            best_i, best_d = 0, float("inf")
            for i in indices:
                px, py = self.points[i]
                d = (px - x) ** 2 + (py - y) ** 2
                if d < best_d:
                    best_i, best_d = i, d
            return best_i, best_d

        if self.last_nearest is not None:
            back = int(0.3 / self.ds) + 1
            ahead = int(1.5 / self.ds) + 1
            window = [(self.last_nearest + j) % n for j in range(-back, ahead + 1)]
            i, d = best_in(window)
            if d < 1.0:
                self.last_nearest = i
                return i
        i, _ = best_in(range(n))
        self.last_nearest = i
        return i

    def _target_speed(self, nearest_i):
        """Inverse-curvature speed from the sharpest bend in the preview window."""
        n = len(self.points)
        count = max(1, int((self.lookahead + self.preview) / self.ds))
        kappa = max(self.kappa[(nearest_i + j) % n] for j in range(count + 1))
        if kappa < 1e-6:
            return self.max_speed
        v = self.max_speed * (1.0 / kappa) / self.full_speed_radius
        return max(self.min_speed, min(self.max_speed, v))

    def _lookahead_for(self, v):
        span = self.max_speed - self.min_speed
        frac = 0.0 if span <= 1e-9 else (v - self.min_speed) / span
        frac = max(0.0, min(1.0, frac))
        return self.lookahead_min + (self.lookahead_max - self.lookahead_min) * frac

    def _lookahead(self, x, y):
        """Walk forward from the nearest point until one is lookahead away."""
        start = self._nearest(x, y)
        count = len(self.points)
        for step in range(1, count):
            i = (start + step) % count
            px, py = self.points[i]
            if math.hypot(px - x, py - y) >= self.lookahead:
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
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "base_footprint", rclpy.time.Time()
            )
        except Exception:
            # No localization yet: hold still and restart the speed ramp.
            self.v_cmd = 0.0
            self.last_tick = None
            self._drive(0.0, 0.0)
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.05 if self.last_tick is None else max(0.0, min(0.2, now - self.last_tick))
        self.last_tick = now

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        curr_x = translation.x
        curr_y = translation.y
        curr_yaw = yaw_from_quaternion(rotation.x, rotation.y, rotation.z, rotation.w)

        # Lookahead from the current speed command, then the target point.
        self.lookahead = self._lookahead_for(self.v_cmd)
        tx, ty, nearest_i = self._lookahead(curr_x, curr_y)
        nx, ny = self.points[nearest_i]

        # Speed: inverse to upcoming curvature, rate limited. Starts from
        # min_speed so the car gets rolling right away.
        v_target = self._target_speed(nearest_i)
        v_prev = max(self.v_cmd, self.min_speed)
        if v_target > v_prev:
            self.v_cmd = min(v_target, v_prev + self.accel * dt)
        else:
            self.v_cmd = max(v_target, v_prev - self.decel * dt)
        self.get_logger().info(
            f"v={self.v_cmd:.3f} (target {v_target:.3f})  L={self.lookahead:.2f} m  "
            f"R_ahead={1.0 / max(self.kappa[nearest_i], 1e-6):.2f} m",
            throttle_duration_sec=1.0,
        )
        dx = tx - curr_x
        dy = ty - curr_y
        # Vehicle frame: +x forward, +y left.
        y_local = -dx * math.sin(curr_yaw) + dy * math.cos(curr_yaw)

        gamma = 2.0 * y_local / (self.lookahead ** 2)
        steering_angle = math.atan(gamma * self.wheelbase)
        steering_angle = max(-self.max_steer, min(self.max_steer, steering_angle))
        norm_steering = steering_angle / self.max_steer

        markers = MarkerArray()
        markers.markers.append(self._sphere(1, tx, ty, 1.0, 0.1, 0.1))
        markers.markers.append(self._sphere(2, nx, ny, 0.1, 0.3, 1.0))
        self.marker_pub.publish(markers)
        self._drive(self.v_cmd, norm_steering)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--map", required=True, help="map name in the maps dir (no extension)")
    p.add_argument("--raceline", help="raceline csv (default <maps>/<map>_raceline.csv)")
    # Turn speed is set by max_speed / full_speed_radius (0.187 throttle per
    # meter of radius) and lookahead by its slope vs speed (4.33 m per unit
    # throttle). Raise max_speed, full_speed_radius and lookahead_max together
    # to make straights faster without changing the turns.
    p.add_argument("--max-speed", type=float, default=0.36,
                   help="/drive speed on straights, normalized throttle (default 0.36)")
    p.add_argument("--speed", dest="max_speed", type=float, help="alias for --max-speed")
    p.add_argument("--min-speed", type=float, default=0.13,
                   help="slowest speed, in the tightest turns (default 0.13)")
    p.add_argument("--full-speed-radius", type=float, default=1.93,
                   help="turn radius (m) at and above which max speed is used (default 1.93)")
    p.add_argument("--lookahead-min", type=float, default=0.45, help="lookahead at min speed, m (default 0.45)")
    p.add_argument("--lookahead-max", type=float, default=1.45, help="lookahead at max speed, m (default 1.45)")
    p.add_argument("--lookahead", type=float, help="fixed lookahead (sets both min and max)")
    p.add_argument("--preview", type=float, default=0.8,
                   help="extra meters past the lookahead scanned for curvature (default 0.8)")
    p.add_argument("--accel", type=float, default=0.25, help="max speed increase per second (default 0.25)")
    p.add_argument("--decel", type=float, default=0.60, help="max speed decrease per second (default 0.60)")
    p.add_argument("--wheelbase", type=float, default=0.28, help="meters (default 0.28)")
    p.add_argument("--max-steer-deg", type=float, default=30.0, help="degrees (default 30)")
    args, _ros = p.parse_known_args()
    if args.lookahead is not None:
        args.lookahead_min = args.lookahead_max = args.lookahead
    return args


def main():
    args = parse_args()
    rclpy.init()
    node = PurePursuit(args)
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
