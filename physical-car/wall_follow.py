"""Left-wall follower for the physical NeoRacer.

Copy to the car and run it there:

    ~/jupyter_ws/neoracer-os/labs/tests/wall_follow.py
    python3 wall_follow.py

The LiDAR array starts straight ahead and increases clockwise, in centimeters.
0 means no return. The car creeps until Ctrl+Z. Flip the FlySky SWB switch
down so these commands reach the motor.

The tracked rays are published on /wall_follow/rays in the laser frame.
In Foxglove, add that topic to the same 3D panel as /scan. The green arrow
is LEFT_DEG. The yellow arrow is the second ray, angled toward the front.
"""

import math

import racecar_core
import rclpy
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from visualization_msgs.msg import Marker, MarkerArray

# Meters from the left wall, measured a short distance ahead.
DESIRED_DIST = 0.50
LOOKAHEAD = 0.40
# Angle between the perpendicular ray and the forward ray.
THETA = math.radians(50.0)

KP = 1.2
KD = 0.4
KI = 0.0
MAX_STEER = 0.6

# rc.drive speed is a fraction of this cap, not of full throttle.
MAX_SPEED = 0.25
SPEED = 0.20
# Stop if something is this close straight ahead.
FRONT_STOP = 0.35

# 0 deg is forward. Samples increase clockwise, so left is 270 deg.
# The second ray has to move toward the nose, which is 270 + THETA, not minus.
LEFT_DEG = 270.0


rc = racecar_core.create_racecar()

if not rclpy.ok():
    rclpy.init()
_node = rclpy.create_node("wall_follow_rays")
_ray_pub = _node.create_publisher(MarkerArray, "/wall_follow/rays", 10)

integral = 0.0
prev_error = 0.0
print_timer = 0.0


def ray_meters(scan, clockwise_deg, window_deg=8.0):
    """Average the non-zero returns in a window. None if the window is empty."""
    n = len(scan)
    if n == 0:
        return None
    center = int(round((clockwise_deg % 360.0) / 360.0 * n)) % n
    half = max(1, int(round(window_deg / 2.0 / 360.0 * n)))
    values = []
    for offset in range(-half, half + 1):
        reading = float(scan[(center + offset) % n])
        if reading > 1.0:
            values.append(reading / 100.0)
    if not values:
        return None
    return sum(values) / len(values)


def wall_error(scan):
    """Return (error, distance_ahead) using two rays on the left wall."""
    b = ray_meters(scan, LEFT_DEG)
    a = ray_meters(scan, LEFT_DEG + math.degrees(THETA))
    if a is None or b is None:
        return None, None
    alpha = math.atan2(a * math.cos(THETA) - b, a * math.sin(THETA))
    distance_now = b * math.cos(alpha)
    distance_ahead = distance_now + LOOKAHEAD * math.sin(alpha)
    return DESIRED_DIST - distance_ahead, distance_ahead


def ray_xy(scan, clockwise_deg):
    """Hit point in the laser frame. +x is forward, +y is left."""
    distance = ray_meters(scan, clockwise_deg)
    if distance is None:
        return None
    theta = math.radians(clockwise_deg)
    return distance * math.cos(theta), -distance * math.sin(theta)


def _arrow(marker_id, x, y, red, green, blue):
    marker = Marker()
    marker.header.frame_id = "laser"
    marker.header.stamp = _node.get_clock().now().to_msg()
    marker.ns = "wall_rays"
    marker.id = marker_id
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.scale.x = 0.02
    marker.scale.y = 0.05
    marker.scale.z = 0.05
    marker.color.r = red
    marker.color.g = green
    marker.color.b = blue
    marker.color.a = 1.0
    marker.lifetime = Duration(seconds=0.3).to_msg()
    marker.points = [
        Point(x=0.0, y=0.0, z=0.0),
        Point(x=float(x), y=float(y), z=0.0),
    ]
    return marker


def direction_xy(clockwise_deg, distance):
    """A point at a fixed range. +x is forward, +y is left."""
    theta = math.radians(clockwise_deg)
    return distance * math.cos(theta), -distance * math.sin(theta)


def publish_rays(scan):
    """Green arrow is LEFT_DEG and ends on the return. Red means that ray missed."""
    arrows = MarkerArray()
    left = ray_xy(scan, LEFT_DEG)
    ahead = ray_xy(scan, LEFT_DEG + math.degrees(THETA))
    if left is not None:
        arrows.markers.append(_arrow(0, left[0], left[1], 0.1, 0.9, 0.2))
    else:
        miss = direction_xy(LEFT_DEG, 1.0)
        arrows.markers.append(_arrow(0, miss[0], miss[1], 0.9, 0.15, 0.1))
    if ahead is not None:
        arrows.markers.append(_arrow(1, ahead[0], ahead[1], 0.95, 0.8, 0.1))
    _ray_pub.publish(arrows)


def start():
    global integral, prev_error, print_timer
    integral = 0.0
    prev_error = 0.0
    print_timer = 0.0
    rc.drive.set_max_speed(MAX_SPEED)
    rc.drive.stop()
    print("wall follow: left wall, wheels up for the first try")


def update():
    global integral, prev_error, print_timer
    scan = rc.lidar.get_samples()
    publish_rays(scan)
    front = ray_meters(scan, 0.0)
    error, distance_ahead = wall_error(scan)

    if front is not None and front < FRONT_STOP:
        rc.drive.stop()
        label = "stop, front wall"
    elif error is None:
        rc.drive.stop()
        label = "stop, no left wall"
    else:
        integral += error
        derivative = error - prev_error
        correction = KP * error + KI * integral + KD * derivative
        prev_error = error
        # Positive angle is right. Too close to the left wall means steer right.
        steer = max(-1.0, min(1.0, correction / MAX_STEER))
        rc.drive.set_speed_angle(SPEED, steer)
        label = f"dist {distance_ahead:.2f} m  steer {steer:+.2f}"

    print_timer += rc.get_delta_time()
    if print_timer > 0.5:
        print_timer = 0.0
        def show(angle):
            reading = ray_meters(scan, angle)
            return "none" if reading is None else f"{reading:.2f} m"

        print(
            f"{label}  front {show(0)}  right {show(90)}  "
            f"back {show(180)}  left {show(270)}"
        )


rc.set_start_update(start, update)
rc.go()
