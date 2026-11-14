"""ROS 2 visualization bridge. No physics, planning, or commands sent back to Gym."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path as RosPath, Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from rosgraph_msgs.msg import Clock
from tf2_ros import TransformBroadcaster


def stamp(seconds):
    ns=round(seconds*1e9)
    return Time(sec=ns//1000000000,nanosec=ns%1000000000)


def path_message(points,t):
    msg=RosPath();msg.header.frame_id='map';msg.header.stamp=t
    for x,y in points:
        pose=PoseStamped();pose.header=msg.header
        pose.pose.position.x=float(x);pose.pose.position.y=float(y);pose.pose.orientation.w=1.
        msg.poses.append(pose)
    return msg


class Display(Node):
    def __init__(self,directory):
        super().__init__('f1tenth_planner_display')
        self.directory=directory;self.sequence=None;self.map_loaded=False
        latched=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub=self.create_publisher(OccupancyGrid,'/map',latched)
        self.wall_pub=self.create_publisher(MarkerArray,'/gym/walls',latched)
        self.ref_pub=self.create_publisher(RosPath,'/planner/reference',latched)
        self.path_pub=self.create_publisher(RosPath,'/planner/selected_path',10)
        self.mark_pub=self.create_publisher(MarkerArray,'/planner/candidates',10)
        self.scan_pub=self.create_publisher(LaserScan,'/ego/scan',10)
        self.status_pub=self.create_publisher(String,'/planner/status',10)
        self.clock_pub=self.create_publisher(Clock,'/clock',10)
        self.odom=[self.create_publisher(Odometry,f'/{name}/odom',10) for name in ('ego','opponent')]
        self.tf=TransformBroadcaster(self)
        self.timer=self.create_timer(.05,self.update)
        self.published=0

    def marker(self,ns,ident,kind,t,color):
        m=Marker();m.header.frame_id='map';m.header.stamp=t;m.ns=ns;m.id=ident;m.type=kind
        m.pose.orientation.w=1.;m.color.r,m.color.g,m.color.b,m.color.a=map(float,color)
        return m

    def update(self):
        try:
            if not self.map_loaded:
                data=json.loads((self.directory/'map.json').read_text())
                m=OccupancyGrid();m.header.frame_id='map';m.info.resolution=float(data['resolution'])
                m.info.width=data['width'];m.info.height=data['height']
                m.info.origin.position.x=float(data['origin'][0]);m.info.origin.position.y=float(data['origin'][1])
                yaw=float(data['origin'][2]);m.info.origin.orientation.z=math.sin(yaw/2);m.info.origin.orientation.w=math.cos(yaw/2)
                m.data=data['data'];self.map_pub.publish(m)
                occupied=np.asarray(data['data']).reshape(data['height'],data['width'])>50
                free=~occupied
                neighbors=np.zeros_like(free)
                neighbors[1:]|=free[:-1];neighbors[:-1]|=free[1:]
                neighbors[:,1:]|=free[:,:-1];neighbors[:,:-1]|=free[:,1:]
                rows,cols=np.where(occupied & neighbors)
                wall=self.marker('gym_map_walls',0,Marker.CUBE_LIST,stamp(0.),(.75,.78,.82,1.))
                wall.scale.x=wall.scale.y=float(data['resolution']);wall.scale.z=.12
                wall.points=[Point(x=float(data['origin'][0]+(c+.5)*data['resolution']),
                    y=float(data['origin'][1]+(r+.5)*data['resolution']),z=0.) for r,c in zip(rows,cols)]
                self.wall_pub.publish(MarkerArray(markers=[wall]))
                self.ref_pub.publish(path_message(data['reference'],stamp(0.)))
                self.reference=data['reference'];self.map_loaded=True
            frame=json.loads((self.directory/'frame.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):return
        if time.time()-frame['wall_time']>3 and frame['status']=='running':
            # Do not silently display a disconnected run as live.
            self.status_pub.publish(String(data='STALE: Gym is not publishing live data'))
            stale=self.marker('connection',0,Marker.TEXT_VIEW_FACING,stamp(frame.get('clock_time',frame['simulation_time'])),(1.,.1,.1,1.))
            stale.pose.position.z=1.;stale.scale.z=.5;stale.text='DISCONNECTED: no fresh Gym data'
            self.mark_pub.publish(MarkerArray(markers=[stale]))
            return
        key=(frame['episode'],frame['sequence'],frame['status'])
        if key==self.sequence:return
        self.sequence=key
        t=stamp(frame.get('clock_time',frame['simulation_time']))
        self.clock_pub.publish(Clock(clock=t))
        markers=MarkerArray()
        clear=Marker();clear.action=Marker.DELETEALL;markers.markers.append(clear)
        te=frame['telemetry'];paths=frame['paths']
        for i,p in enumerate(paths):
            color=(.15,.55,1.,.65) if p['feasible'] else (1.,.25,.25,.4)
            m=self.marker('candidate_paths',i,Marker.LINE_STRIP,t,color);m.scale.x=.025
            m.points=[Point(x=float(x),y=float(y),z=.04) for x,y in p['path']]
            markers.markers.append(m)
        reps=frame['representatives']
        selected=next(p for p in reps if p['action']==te['action'])
        self.path_pub.publish(path_message(selected['path'],t))
        m=self.marker('selected',0,Marker.LINE_STRIP,t,(.15,1.,.15,1.));m.scale.x=.075
        m.points=[Point(x=float(x),y=float(y),z=.09) for x,y in selected['path']]
        markers.markers.append(m)
        for i,(name,car,color) in enumerate(zip(('ego','opponent'),frame['cars'],((.15,.9,.25,1.),(1.,.55,.1,1.)))):
            x,y,yaw,speed=car
            transform=TransformStamped();transform.header.frame_id='map';transform.header.stamp=t
            transform.child_frame_id=name+'/base_link';transform.transform.translation.x=x;transform.transform.translation.y=y
            transform.transform.rotation.z=math.sin(yaw/2);transform.transform.rotation.w=math.cos(yaw/2)
            self.tf.sendTransform(transform)
            odom=Odometry();odom.header=transform.header;odom.child_frame_id=transform.child_frame_id
            odom.pose.pose.position.x=x;odom.pose.pose.position.y=y;odom.pose.pose.orientation=transform.transform.rotation
            odom.twist.twist.linear.x=speed;self.odom[i].publish(odom)
            m=self.marker('cars',i,Marker.CUBE,t,color);m.pose=odom.pose.pose;m.pose.position.z=.13
            m.scale.x=.58;m.scale.y=.31;m.scale.z=.20;markers.markers.append(m)
            label=self.marker('car_labels',i,Marker.TEXT_VIEW_FACING,t,color)
            label.pose.position.x=x;label.pose.position.y=y+.5;label.pose.position.z=.4;label.scale.z=.22
            label.text=f'{name}: {speed:.2f} m/s';markers.markers.append(label)
        scan=LaserScan();scan.header.frame_id='ego/base_link';scan.header.stamp=t
        scan.angle_min=frame['angle_min'];scan.angle_max=frame['angle_max']
        scan.angle_increment=(scan.angle_max-scan.angle_min)/(len(frame['scan'])-1)
        scan.range_min=0.;scan.range_max=30.;scan.scan_time=.05;scan.ranges=frame['scan'];self.scan_pub.publish(scan)
        actions=['LEFT','RIGHT','WAIT','PROBE'][:len(te['probabilities'])]
        probs='  '.join(f'{a} {p:.0%}' for a,p in zip(actions,te['probabilities']))
        reasons=Counter(p['reason'] for p in paths if not p['feasible'])
        feasible=sum(p['feasible'] for p in paths)
        lines=[f"STAGE {frame['stage']} | {frame['simulation_time']:.1f}s | {frame['status']}",
            'GREEN chosen | BLUE feasible | RED rejected | WHITE reference',
            f"{len(paths)} checked / {feasible} feasible | selected {te['action']} | network {te['raw_action']}",
            probs,
            f"Safety override {bool(te['safety_mask_override'])} | commitment {bool(te['commitment_override'])}",
            f"Command: steer {te['command'][0]:+.2f} rad, speed {te['command'][1]:.2f} m/s",
            'Rejected: '+', '.join(f'{k} ({v})' for k,v in reasons.items())]
        for rep in reps[:len(actions)]:
            lines.append(f"{rep['action']}: {'feasible' if rep['feasible'] else 'blocked'}; cost {rep['estimated_time']:.2f}; clearance {rep['clearance']:.2f} m")
        parameters=selected.get('parameters')
        if parameters:
            lines.append(f"Chosen entry {parameters['entry_m']:.1f} m / padding {parameters['padding_m']:.2f} m / merge x{parameters['merge_factor']:.2f}")
        lines.append('Lowest-cost path per side -> network/value ranking -> safety + commitment')
        text='\n'.join(lines)
        label=self.marker('decision_summary',0,Marker.TEXT_VIEW_FACING,t,(1.,1.,1.,1.))
        label.pose.position.x=min(p[0] for p in self.reference)+4.
        label.pose.position.y=max(p[1] for p in self.reference)+2.8
        label.pose.position.z=.5;label.scale.z=.26;label.text=text;markers.markers.append(label)
        response=te.get('response',{})
        if te.get('command_replay'):
            label.text=f"STAGE 4 COUNTERFACTUAL | {frame['simulation_time']:.2f}s | {frame['status']}\nIdentical ego commands replayed in live Gym\nOpponent follows reference WITHOUT reactive avoidance\nGREEN = original ego trajectory; candidate overlays disabled\nCommand: steer {te['command'][0]:+.2f} rad, speed {te['command'][1]:.2f} m/s"
        label.text+=f"\nReactive rival offset from racing line: {frame.get('opponent_offset_m',0.):+.2f}m"
        if response.get('selected'):
            learned=response.get('learned_forecast',False)
            overlays=[(response['predictions'][0]['path'],(.1,1.,1.,.9),'learned reactive prediction')]
            if not learned:overlays.append((response['no_response_path'],(1.,.15,.8,.8),'constant-motion prediction'))
            for ident,(points,color,name) in enumerate(overlays):
                line=self.marker('opponent_predictions',ident,Marker.LINE_STRIP,t,color);line.scale.x=.065
                line.points=[Point(x=float(x),y=float(y),z=.18) for x,y in points];markers.markers.append(line)
            if learned:
                label.text+='\nLEARNED REACTIVE FORECAST | CYAN predicted rival path'
                label.text+=f"\nPredicted extra outward move {response['predicted_outward_m']:.3f}m; ensemble spread {response['ensemble_lateral_spread_m']:.3f}m"
                label.text+=f"\nPredicted body gap {response['minimum_response_gap']:.3f}m | replans every 0.05s"
            else:
                label.text+='\nRESPONSE PLANNER: analytical yielding hypothesis'
                label.text+=f"\nPredicted yielding gap: {response['minimum_response_gap']:.3f}m"
        self.mark_pub.publish(markers)
        self.status_pub.publish(String(data=json.dumps(dict(stage=frame['stage'],time=frame['simulation_time'],
            selected=te['action'],network=te['raw_action'],status=frame['status'],scene=frame['scene'],evaluated=len(paths),feasible=feasible,rejected=dict(reasons),
            command=te['command'],response_selected=bool(response.get('selected')),
            assumes_yield=bool(response.get('assumes_yield')),command_replay=bool(te.get('command_replay')),response_gap=response.get('minimum_response_gap'),
            learned_forecast=bool(response.get('learned_forecast')),opponent_offset_m=frame.get('opponent_offset_m'),
            predicted_outward_m=response.get('predicted_outward_m'),no_response_gap=response.get('no_response_gap'),text=label.text))))
        self.published+=1
        if self.published%100==1:self.get_logger().info(f"Gym stage {frame['stage']}: {len(paths)} paths, selected {te['action']}")


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=Path(__file__).parent/'live')
    args,rosargs=p.parse_known_args();rclpy.init(args=rosargs);node=Display(args.directory)
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:node.destroy_node();rclpy.try_shutdown()

if __name__=='__main__':main()
