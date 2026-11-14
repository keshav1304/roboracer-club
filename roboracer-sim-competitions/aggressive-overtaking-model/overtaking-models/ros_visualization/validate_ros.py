"""End-to-end live-topic check: run alongside the Gym, bridge, and RViz."""
import json
from pathlib import Path
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy
from std_msgs.msg import String
from nav_msgs.msg import Path as RosPath, OccupancyGrid
from visualization_msgs.msg import MarkerArray
from rosgraph_msgs.msg import Clock

rclpy.init();node=Node('gym_rviz_verification')
state={'stages':{},'map_cells':0,'wall_points':0,'selected_points':0,'clock_backwards':False,'last_clock':-1.,'messages':0}
latched=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
def status(msg):
    try:d=json.loads(msg.data)
    except json.JSONDecodeError:return
    item=state['stages'].setdefault(str(d['stage']),{'max_candidates':0,'actions':[]})
    item['max_candidates']=max(item['max_candidates'],d['evaluated'])
    if d['selected'] not in item['actions']:item['actions'].append(d['selected'])
    state['messages']+=1

def grid(msg):state['map_cells']=len(msg.data)
def walls(msg):state['wall_points']=sum(len(m.points) for m in msg.markers)
def selected(msg):state['selected_points']=len(msg.poses)
def clock(msg):
    t=msg.clock.sec+msg.clock.nanosec/1e9
    if t<state['last_clock']:state['clock_backwards']=True
    state['last_clock']=t
subs=[node.create_subscription(String,'/planner/status',status,10),node.create_subscription(OccupancyGrid,'/map',grid,latched),
      node.create_subscription(MarkerArray,'/gym/walls',walls,latched),node.create_subscription(RosPath,'/planner/selected_path',selected,10),
      node.create_subscription(Clock,'/clock',clock,10)]
deadline=time.monotonic()+85
while time.monotonic()<deadline:
    rclpy.spin_once(node,timeout_sec=.2)
    if (len(state['stages'])==2 and all(v['max_candidates']>4 for v in state['stages'].values())
        and state['map_cells'] and state['wall_points'] and state['selected_points']):break
state['rviz_subscriptions']=[dict(node=x.node_name,topic='/planner/candidates') for x in node.get_subscriptions_info_by_topic('/planner/candidates')]
state['passed']=bool(len(state['stages'])==2 and all(v['max_candidates']>4 for v in state['stages'].values()) and state['map_cells'] and state['wall_points'] and state['selected_points'] and not state['clock_backwards'] and any(x['node'] in ('rviz','rviz2') for x in state['rviz_subscriptions']))
Path(__file__).with_name('verification.json').write_text(json.dumps(state,indent=2))
print(json.dumps(state,indent=2),flush=True)
node.destroy_node();rclpy.shutdown()
raise SystemExit(0 if state['passed'] else 1)
