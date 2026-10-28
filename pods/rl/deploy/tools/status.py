import math, time, rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Int32
T = '/autodrive/roboracer_1/'
be = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
rclpy.init(); n = rclpy.create_node('status'); S = {}
def keep(k):
    def f(m): S[k] = m
    return f
n.create_subscription(Odometry, T + 'odom', keep('odom'), be)
for k in ('lap_time', 'last_lap_time', 'best_lap_time', 'throttle', 'steering'):
    n.create_subscription(Float32, T + k, keep(k), be)
for k in ('lap_count', 'collision_count'):
    n.create_subscription(Int32, T + k, keep(k), be)
t0 = time.time()
while time.time() - t0 < 4.0 and not all(k in S for k in ("odom", "lap_count", "collision_count", "throttle", "steering", "lap_time", "last_lap_time", "best_lap_time")): rclpy.spin_once(n, timeout_sec=0.02)
o = S['odom']; v = o.twist.twist.linear; p = o.pose.pose.position
print(f"pos=({p.x:.2f},{p.y:.2f}) speed={math.hypot(v.x, v.y):.2f} m/s thr={S['throttle'].data:.3f} "
      f"steer={S['steering'].data:+.3f} laps={S['lap_count'].data} lap_time={S['lap_time'].data:.1f} "
      f"last={S['last_lap_time'].data:.2f} best={S['best_lap_time'].data:.2f} collisions={S['collision_count'].data}")
n.destroy_node(); rclpy.shutdown()
