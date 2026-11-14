"""Verify actual RViz subscriptions and a live reactive-only Gym pass."""
import json,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
rclpy.init();node=Node('learned_response_verifier')
state=dict(messages=0,learned_forecast_seen=False,reactive_pass_seen=False,command_replay_seen=False,scenes=[],maximum_observed_offset_m=0.)
def receive(msg):
    try:d=json.loads(msg.data)
    except json.JSONDecodeError:return
    state['messages']+=1
    state['learned_forecast_seen'] |= bool(d.get('response_selected') and d.get('learned_forecast'))
    state['reactive_pass_seen'] |= d.get('status')=='pass' and not d.get('command_replay')
    state['command_replay_seen'] |= bool(d.get('command_replay'))
    state['maximum_observed_offset_m']=max(state['maximum_observed_offset_m'],abs(d.get('opponent_offset_m') or 0.))
    if d.get('scene') not in state['scenes']:state['scenes'].append(d.get('scene'))
sub=node.create_subscription(String,'/planner/status',receive,10)
end=time.monotonic()+180
while time.monotonic()<end:
    rclpy.spin_once(node,timeout_sec=.2)
    if state['learned_forecast_seen'] and state['reactive_pass_seen']:break
state['rviz_connected']=any(x.node_name in ('rviz','rviz2') for x in node.get_subscriptions_info_by_topic('/planner/candidates'))
state['passed']=state['messages']>0 and state['reactive_pass_seen'] and state['rviz_connected'] and not state['command_replay_seen']
state['note']='Offset alone is not proof of yielding; strict avoidance metrics are in the experiment yield_audit.json.'
Path(__file__).with_name('learned_ros_verification.json').write_text(json.dumps(state,indent=2));print(json.dumps(state,indent=2))
node.destroy_node();rclpy.try_shutdown();raise SystemExit(0 if state['passed'] else 1)
