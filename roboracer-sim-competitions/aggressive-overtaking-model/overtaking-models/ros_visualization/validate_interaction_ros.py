"""Check actual RViz connection and two live Gym passes with interaction prediction/fallback."""
import argparse,json,time
from pathlib import Path
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=240.);p.add_argument('--passes',type=int,default=2);args=p.parse_args()
rclpy.init();node=Node('interaction_verifier')
state=dict(messages=0,interaction_seen=False,fallback_seen=False,pass_scenes=[],scenes=[],command_replay_seen=False)
def receive(msg):
 try:d=json.loads(msg.data)
 except json.JSONDecodeError:return
 state['messages']+=1
 state['interaction_seen'] |= bool(d.get('interaction_active'))
 state['fallback_seen'] |= bool(d.get('interaction_override'))
 state['command_replay_seen'] |= bool(d.get('command_replay'))
 if d.get('scene') not in state['scenes']:state['scenes'].append(d.get('scene'))
 if d.get('status')=='pass' and d['scene'] not in state['pass_scenes']:state['pass_scenes'].append(d['scene'])
sub=node.create_subscription(String,'/planner/status',receive,10)
end=time.monotonic()+args.seconds
while time.monotonic()<end:
 rclpy.spin_once(node,timeout_sec=.2)
 if len(state['pass_scenes'])>=args.passes and state['interaction_seen'] and state['fallback_seen']:break
state['rviz_connected']=any(x.node_name in ('rviz','rviz2') for x in node.get_subscriptions_info_by_topic('/planner/candidates'))
state['passed']=len(state['pass_scenes'])>=args.passes and state['fallback_seen'] and state['rviz_connected'] and not state['command_replay_seen']
args.output.write_text(json.dumps(state,indent=2));print(json.dumps(state,indent=2))
node.destroy_node();rclpy.try_shutdown();raise SystemExit(0 if state['passed'] else 1)
