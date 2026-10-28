"""Zero the commands and reset the car. Waits for the bridge to match before publishing."""
import sys, time, rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Bool, Float32
rel = QoSProfile(durability=DurabilityPolicy.VOLATILE, reliability=ReliabilityPolicy.RELIABLE,
                 history=HistoryPolicy.KEEP_LAST, depth=1)
do_reset = '--stop-only' not in sys.argv
rclpy.init(); n = rclpy.create_node('resetter')
p = n.create_publisher(Bool, '/autodrive/reset_command', rel)
pt = n.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', rel)
ps = n.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', rel)
t0 = time.time()
while min(p.get_subscription_count(), pt.get_subscription_count(), ps.get_subscription_count()) < 1:
    rclpy.spin_once(n, timeout_sec=0.05)
    if time.time() - t0 > 10: print('bridge not matched after 10 s'); sys.exit(1)
print(f'matched bridge after {time.time() - t0:.2f} s')
def burst(reset_val, k):
    for _ in range(k):
        pt.publish(Float32(data=0.0)); ps.publish(Float32(data=0.0))
        if do_reset: p.publish(Bool(data=reset_val))
        rclpy.spin_once(n, timeout_sec=0.02)
burst(False, 25)                 # stop first
if do_reset:
    burst(True, 25); burst(False, 50)
print('stopped' + (' and reset' if do_reset else ''))
n.destroy_node(); rclpy.shutdown()
