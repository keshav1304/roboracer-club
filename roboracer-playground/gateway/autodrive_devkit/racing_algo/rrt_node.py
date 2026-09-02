"""RRT / RRT* local planner for AutoDRIVE.

Builds a local occupancy grid from LiDAR, plans in the vehicle frame, and
tracks the path with Pure Pursuit. Publishes throttle/steer and debug data.
"""

from __future__ import annotations

import math
import time

import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Float32MultiArray


class TreeNode:
    __slots__ = ("x", "y", "parent", "cost", "is_root")

    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0
        self.is_root = False


# Scalar keys must match gateway DEBUG_LAYOUTS for rrt / rrt_star.
DEBUG_KEYS = (
    "plan_ms",
    "tree_size",
    "path_len",
    "reached_goal",
    "steering",
    "throttle",
    "lookahead_dist",
    "goal_x",
    "target_x",
    "target_y",
    "goal_tolerance",
    "grid_resolution",
)

# Cap streamed geometry so WS stays light. Tree cap ≥ max_rrt_iters so the
# full tree usually ships; occupancy is the inflate ring + raw hits.
_MAX_PATH_PTS = 40
_MAX_TREE_EDGES = 800
_MAX_OCC_CELLS = 600


class LocalRRTNode(Node):
    """RRT or RRT* with Pure Pursuit path tracking."""

    def __init__(self, use_rrt_star: bool = False):
        name = "rrt_star_node" if use_rrt_star else "rrt_node"
        super().__init__(name)
        self.use_rrt_star = use_rrt_star
        self.debug_topic = (
            "/playground/rrt_star/debug"
            if use_rrt_star
            else "/playground/rrt/debug"
        )

        self.declare_parameter("max_rrt_iters", 220.0)
        self.declare_parameter("max_planning_ms", 40.0)
        self.declare_parameter("plan_hz", 5.0)
        self.declare_parameter("expand_dist", 0.2)
        self.declare_parameter("goal_x", 3.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("goal_tolerance", 0.35)
        self.declare_parameter("goal_sample_rate", 0.2)
        self.declare_parameter("inflate_radius", 0.1)
        self.declare_parameter("grid_resolution", 0.05)
        self.declare_parameter("grid_length", 6.0)
        self.declare_parameter("grid_width", 4.0)
        self.declare_parameter("near_radius", 0.75)
        self.declare_parameter("lookahead_dist", 1.4)
        self.declare_parameter("throttle", 0.06)
        self.declare_parameter("max_steering_rad", 0.4189)
        self.declare_parameter("wheelbase", 0.33)
        self.declare_parameter("scan_angle_min_deg", -90.0)
        self.declare_parameter("scan_angle_max_deg", 90.0)

        self._load_params()
        self.add_on_set_parameters_callback(self._on_set_parameters)
        self._rebuild_grid()

        self.occupancy = np.zeros((self.grid_nx, self.grid_ny), dtype=np.int8)
        self.occupancy_hits = np.zeros((self.grid_nx, self.grid_ny), dtype=np.int8)
        self._scan_ready = False
        self._last_plan_mono = 0.0

        ns = "/autodrive/roboracer_1"
        self.create_subscription(LaserScan, f"{ns}/lidar", self._on_scan, 10)
        self.throttle_pub = self.create_publisher(Float32, f"{ns}/throttle_command", 10)
        self.steering_pub = self.create_publisher(Float32, f"{ns}/steering_command", 10)
        self.debug_pub = self.create_publisher(Float32MultiArray, self.debug_topic, 10)

        period = 1.0 / max(self.plan_hz, 0.5)
        self._plan_timer = self.create_timer(period, self._plan_tick)

        mode = "RRT*" if use_rrt_star else "RRT"
        self.get_logger().info(f"Local {mode} node ready (plan_hz={self.plan_hz})")

    def _load_params(self):
        self.max_rrt_iters = int(self.get_parameter("max_rrt_iters").value)
        self.max_planning_ms = float(self.get_parameter("max_planning_ms").value)
        self.plan_hz = float(self.get_parameter("plan_hz").value)
        self.expand_dist = float(self.get_parameter("expand_dist").value)
        self.goal_x = float(self.get_parameter("goal_x").value)
        self.goal_y = float(self.get_parameter("goal_y").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.goal_sample_rate = float(self.get_parameter("goal_sample_rate").value)
        self.inflate_radius = float(self.get_parameter("inflate_radius").value)
        self.grid_resolution = float(self.get_parameter("grid_resolution").value)
        self.grid_length = float(self.get_parameter("grid_length").value)
        self.grid_width = float(self.get_parameter("grid_width").value)
        self.near_radius = float(self.get_parameter("near_radius").value)
        self.lookahead_dist = float(self.get_parameter("lookahead_dist").value)
        self.throttle = float(self.get_parameter("throttle").value)
        self.max_steering_rad = float(self.get_parameter("max_steering_rad").value)
        self.wheelbase = float(self.get_parameter("wheelbase").value)
        self.scan_angle_min = math.radians(
            float(self.get_parameter("scan_angle_min_deg").value)
        )
        self.scan_angle_max = math.radians(
            float(self.get_parameter("scan_angle_max_deg").value)
        )

    def _rebuild_grid(self):
        self.grid_nx = max(1, int(self.grid_length / self.grid_resolution))
        self.grid_ny = max(1, int(self.grid_width / self.grid_resolution))

    def _on_set_parameters(self, params):
        for p in params:
            v = float(p.value)
            if p.name == "throttle" and not (-1.0 <= v <= 1.0):
                return SetParametersResult(
                    successful=False, reason="throttle must be in [-1, 1]"
                )
            if p.name in (
                "max_rrt_iters",
                "max_planning_ms",
                "plan_hz",
                "expand_dist",
                "goal_tolerance",
                "inflate_radius",
                "grid_resolution",
                "grid_length",
                "grid_width",
                "near_radius",
                "lookahead_dist",
                "max_steering_rad",
                "wheelbase",
            ) and v <= 0.0:
                return SetParametersResult(
                    successful=False, reason=f"{p.name} must be > 0"
                )
            if p.name == "goal_sample_rate" and not (0.0 <= v <= 1.0):
                return SetParametersResult(
                    successful=False, reason="goal_sample_rate must be in [0, 1]"
                )

        rebuild = False
        retimer = False
        for p in params:
            name = p.name
            v = float(p.value)
            if name == "max_rrt_iters":
                self.max_rrt_iters = int(v)
            elif name == "plan_hz":
                self.plan_hz = v
                retimer = True
            elif name in ("grid_resolution", "grid_length", "grid_width"):
                setattr(self, name, v)
                rebuild = True
            elif name == "scan_angle_min_deg":
                self.scan_angle_min = math.radians(v)
            elif name == "scan_angle_max_deg":
                self.scan_angle_max = math.radians(v)
            elif hasattr(self, name):
                setattr(self, name, v)

        if rebuild:
            self._rebuild_grid()
            self.occupancy = np.zeros((self.grid_nx, self.grid_ny), dtype=np.int8)
            self.occupancy_hits = np.zeros(
                (self.grid_nx, self.grid_ny), dtype=np.int8
            )
        if retimer:
            self._plan_timer.cancel()
            self._plan_timer = self.create_timer(
                1.0 / max(self.plan_hz, 0.5), self._plan_tick
            )
        return SetParametersResult(successful=True)

    def _on_scan(self, scan: LaserScan):
        self.occupancy.fill(0)
        angle = scan.angle_min
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        for r in ranges:
            if self.scan_angle_min <= angle <= self.scan_angle_max:
                if (
                    not np.isinf(r)
                    and not np.isnan(r)
                    and scan.range_min <= r <= scan.range_max
                ):
                    x = float(r * math.cos(angle))
                    y = float(r * math.sin(angle))
                    if 0.0 <= x <= self.grid_length and abs(y) <= self.grid_width / 2.0:
                        gx, gy = self._xy_to_grid(x, y)
                        if self._in_bounds(gx, gy):
                            self.occupancy[gx, gy] = 1
            angle += scan.angle_increment
        # Keep raw hits for viz; inflate a copy used for planning.
        self.occupancy_hits = self.occupancy.copy()
        self._inflate_obstacles()
        self._scan_ready = True

    def _plan_tick(self):
        if not self._scan_ready:
            return

        t0 = time.perf_counter()
        root = TreeNode(0.0, 0.0)
        root.is_root = True
        tree = [root]
        latest = root
        reached = False

        direct = TreeNode(self.goal_x, self.goal_y)
        if self._edge_free(root, direct):
            path = [root, direct]
            plan_ms = (time.perf_counter() - t0) * 1000.0
            steer, thr, tx, ty = self._follow_path(path)
            self._publish_debug(plan_ms, tree, path, True, steer, thr, tx, ty)
            return

        deadline = t0 + self.max_planning_ms / 1000.0
        for _ in range(self.max_rrt_iters):
            if time.perf_counter() >= deadline:
                break
            sample = self._sample()
            nearest_idx = self._nearest(tree, sample)
            new_node = self._steer(tree[nearest_idx], sample)
            if new_node is None:
                continue
            if not self._edge_free(tree[nearest_idx], new_node):
                continue

            if self.use_rrt_star:
                neighbors = self._near(tree, new_node)
                best_parent = tree[nearest_idx]
                best_cost = tree[nearest_idx].cost + self._line_cost(
                    tree[nearest_idx], new_node
                )
                for nb in neighbors:
                    c = nb.cost + self._line_cost(nb, new_node)
                    if c < best_cost and self._edge_free(nb, new_node):
                        best_parent = nb
                        best_cost = c
                new_node.parent = best_parent
                new_node.cost = best_cost
                tree.append(new_node)
                latest = new_node
                for nb in neighbors:
                    c = new_node.cost + self._line_cost(new_node, nb)
                    if c < nb.cost and self._edge_free(new_node, nb):
                        nb.parent = new_node
                        nb.cost = c
            else:
                new_node.parent = tree[nearest_idx]
                new_node.cost = tree[nearest_idx].cost + self._line_cost(
                    tree[nearest_idx], new_node
                )
                tree.append(new_node)
                latest = new_node

            if self._is_goal(new_node):
                reached = True
                break

        if reached:
            goal = TreeNode(self.goal_x, self.goal_y)
            goal.parent = latest
            path = self._find_path(goal)
        else:
            path = self._find_path(latest)

        plan_ms = (time.perf_counter() - t0) * 1000.0
        steer, thr, tx, ty = self._follow_path(path)
        self._publish_debug(plan_ms, tree, path, reached, steer, thr, tx, ty)

    def _sample(self):
        if np.random.rand() < self.goal_sample_rate:
            return (self.goal_x, self.goal_y)
        for _ in range(20):
            x = float(np.random.uniform(0.5, self.grid_length))
            y = float(np.random.uniform(-self.grid_width / 2.0, self.grid_width / 2.0))
            if self._point_is_free(x, y):
                return (x, y)
        return (self.goal_x, self.goal_y)

    def _nearest(self, tree, sample):
        sx, sy = sample
        best_i = 0
        best_d = float("inf")
        for i, node in enumerate(tree):
            d = (node.x - sx) ** 2 + (node.y - sy) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def _steer(self, nearest, sample):
        sx, sy = sample
        dx = sx - nearest.x
        dy = sy - nearest.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return None
        step = min(self.expand_dist, dist)
        theta = math.atan2(dy, dx)
        nx = nearest.x + step * math.cos(theta)
        ny = nearest.y + step * math.sin(theta)
        if not self._point_in_window(nx, ny):
            return None
        return TreeNode(nx, ny)

    def _edge_free(self, a, b) -> bool:
        """True if the segment is collision-free."""
        dx = b.x - a.x
        dy = b.y - a.y
        dist = math.hypot(dx, dy)
        n_checks = max(2, int(dist / self.grid_resolution))
        for i in range(1, n_checks + 1):
            t = i / n_checks
            if not self._point_is_free(a.x + t * dx, a.y + t * dy):
                return False
        return True

    def _is_goal(self, node) -> bool:
        return math.hypot(node.x - self.goal_x, node.y - self.goal_y) <= self.goal_tolerance

    def _find_path(self, node):
        path = []
        cur = node
        while cur is not None:
            path.append(cur)
            cur = cur.parent
        path.reverse()
        return path

    def _line_cost(self, n1, n2):
        return math.hypot(n2.x - n1.x, n2.y - n1.y)

    def _near(self, tree, node):
        r = self.near_radius
        return [
            other
            for other in tree
            if math.hypot(other.x - node.x, other.y - node.y) <= r
        ]

    def _follow_path(self, path):
        if len(path) < 2:
            self._publish_cmd(0.0, 0.0)
            return 0.0, 0.0, 0.0, 0.0

        target = None
        for node in path:
            if math.hypot(node.x, node.y) >= self.lookahead_dist:
                target = node
                break
        if target is None:
            target = path[-1]

        x, y = target.x, target.y
        if x < 1e-3:
            steering = 0.0
        else:
            curvature = 2.0 * y / (x * x + y * y)
            steering = math.atan(self.wheelbase * curvature)
        steering = float(np.clip(steering, -self.max_steering_rad, self.max_steering_rad))

        thr = self.throttle
        if abs(steering) > 0.30:
            thr = min(thr, self.throttle * 0.85)
        if abs(steering) > 0.40:
            thr = min(thr, self.throttle * 0.7)

        # Normalize steer to [-1, 1] for AutoDRIVE.
        steer_cmd = float(np.clip(steering / self.max_steering_rad, -1.0, 1.0))
        self._publish_cmd(steer_cmd, thr)
        return steering, thr, x, y

    def _publish_cmd(self, steering_norm: float, throttle: float):
        self.steering_pub.publish(Float32(data=float(steering_norm)))
        self.throttle_pub.publish(Float32(data=float(throttle)))

    def _publish_debug(
        self, plan_ms, tree, path, reached, steering, throttle, tx, ty
    ):
        data = [
            float(plan_ms),
            float(len(tree)),
            float(len(path)),
            1.0 if reached else 0.0,
            float(steering),
            float(throttle),
            float(self.lookahead_dist),
            float(self.goal_x),
            float(tx),
            float(ty),
            float(self.goal_tolerance),
            float(self.grid_resolution),
        ]

        # Packed geometry: n_path, xy...; n_edges, x1 y1 x2 y2 cost...;
        # n_occ_hit, xy...; n_occ_inf, xy...
        if len(path) > _MAX_PATH_PTS:
            idx = np.linspace(0, len(path) - 1, _MAX_PATH_PTS).astype(int)
            path_pts = [path[i] for i in idx]
        else:
            path_pts = path
        data.append(float(len(path_pts)))
        for n in path_pts:
            data.append(float(n.x))
            data.append(float(n.y))

        edges = []
        for n in tree:
            if n.parent is not None:
                edges.append((n.parent.x, n.parent.y, n.x, n.y, n.cost))
        if len(edges) > _MAX_TREE_EDGES:
            step = max(1, len(edges) // _MAX_TREE_EDGES)
            edges = edges[::step][:_MAX_TREE_EDGES]
        data.append(float(len(edges)))
        for x1, y1, x2, y2, cost in edges:
            data.extend([float(x1), float(y1), float(x2), float(y2), float(cost)])

        hits = self._occupied_cell_centers(self.occupancy_hits)
        # Stream only the inflate *ring* (inflated \ hits) so the UI can show
        # clearance cells without them being covered by hit cells.
        ring = np.zeros_like(self.occupancy)
        ring[(self.occupancy == 1) & (self.occupancy_hits == 0)] = 1
        inflated = self._occupied_cell_centers(ring)
        data.append(float(len(hits)))
        for x, y in hits:
            data.extend([float(x), float(y)])
        data.append(float(len(inflated)))
        for x, y in inflated:
            data.extend([float(x), float(y)])

        msg = Float32MultiArray()
        msg.data = data
        self.debug_pub.publish(msg)

    def _occupied_cell_centers(self, grid: np.ndarray):
        """Return up to _MAX_OCC_CELLS (x,y) centers of occupied cells."""
        cells = np.argwhere(grid == 1)
        if cells.size == 0:
            return []
        if len(cells) > _MAX_OCC_CELLS:
            step = max(1, len(cells) // _MAX_OCC_CELLS)
            cells = cells[::step][:_MAX_OCC_CELLS]
        out = []
        res = self.grid_resolution
        half_w = self.grid_width / 2.0
        for gx, gy in cells:
            x = (gx + 0.5) * res
            y = (gy + 0.5) * res - half_w
            out.append((x, y))
        return out

    def _inflate_obstacles(self):
        cells = int(self.inflate_radius / self.grid_resolution)
        if cells <= 0:
            return
        inflated = self.occupancy.copy()
        occ = np.argwhere(self.occupancy == 1)
        for gx, gy in occ:
            x0 = max(0, gx - cells)
            x1 = min(self.grid_nx, gx + cells + 1)
            y0 = max(0, gy - cells)
            y1 = min(self.grid_ny, gy + cells + 1)
            inflated[x0:x1, y0:y1] = 1
        self.occupancy = inflated

    def _xy_to_grid(self, x, y):
        gx = int(x / self.grid_resolution)
        gy = int((y + self.grid_width / 2.0) / self.grid_resolution)
        return gx, gy

    def _in_bounds(self, gx, gy):
        return 0 <= gx < self.grid_nx and 0 <= gy < self.grid_ny

    def _point_in_window(self, x, y):
        return 0.0 <= x <= self.grid_length and abs(y) <= self.grid_width / 2.0

    def _point_is_free(self, x, y):
        if math.hypot(x, y) < 0.15:
            return True
        if not self._point_in_window(x, y):
            return False
        gx, gy = self._xy_to_grid(x, y)
        if not self._in_bounds(gx, gy):
            return False
        return self.occupancy[gx, gy] == 0


def main(args=None):
    rclpy.init(args=args)
    node = LocalRRTNode(use_rrt_star=False)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main_rrt_star(args=None):
    rclpy.init(args=args)
    node = LocalRRTNode(use_rrt_star=True)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
