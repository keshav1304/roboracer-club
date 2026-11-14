"""Verify live RViz receives the conditional maneuver and both Gym outcomes."""
import json,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
rclpy.init();node=Node('response_demo_verifier')
state={'conditional_move_seen':False,'reactive_pass_seen':False,'counterfactual_contact_seen':False,'messages':0}
def receive(msg):
    try:d=json.loads(msg.data)
    except json.JSONDecodeError:return
    state['messages']+=1
    state['conditional_move_seen'] |= bool(d.get('response_selected') and d.get('assumes_yield') and d.get('no_response_gap',1.)<0 and d.get('response_gap',-1.)>=.035)
    state['reactive_pass_seen'] |= d.get('status')=='pass' and not d.get('command_replay')
    state['counterfactual_contact_seen'] |= d.get('status')=='collision' and bool(d.get('command_replay'))
sub=node.create_subscription(String,'/planner/status',receive,10)
deadline=time.monotonic()+110
while time.monotonic()<deadline:
    rclpy.spin_once(node,timeout_sec=.2)
    if all(state[k] for k in ('conditional_move_seen','reactive_pass_seen','counterfactual_contact_seen')):break
state['rviz_connected']=any(x.node_name in ('rviz','rviz2') for x in node.get_subscriptions_info_by_topic('/planner/candidates'))
state['passed']=all(state[k] for k in ('conditional_move_seen','reactive_pass_seen','counterfactual_contact_seen','rviz_connected'))
Path(__file__).with_name('response_ros_verification.json').write_text(json.dumps(state,indent=2));print(json.dumps(state,indent=2))
node.destroy_node();rclpy.try_shutdown();raise SystemExit(0 if state['passed'] else 1)
