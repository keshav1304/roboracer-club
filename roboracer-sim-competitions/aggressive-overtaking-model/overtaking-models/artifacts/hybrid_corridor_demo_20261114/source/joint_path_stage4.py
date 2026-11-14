"""Stage 4 rescue search over lane-change geometry and timed speed profiles.

Frozen demonstrated controller is the baseline. Extra paths retain physical
body/wall margins and are checked against measured and map-limited opponent
motion, without relying on a yield or reading opponent simulator state.
"""
from pathlib import Path
import importlib.util
import numpy as np
from maneuver_core import Candidate,delta_s,smoothstep
from response_stage4 import separation
from track_geometry import MAX_STEER,WHEELBASE

ROOT=Path(__file__).resolve().parent
baseline_path=ROOT/'baseline_learned_stage4.py'
if not baseline_path.exists():baseline_path=ROOT/'artifacts/learned_pressure_random_r1/source/learned_stage4.py'
spec=importlib.util.spec_from_file_location('joint_baseline',baseline_path)
frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
if (ROOT/'opponent_forecast.npz').exists():
    from functools import partial
    frozen.OpponentForecast=partial(frozen.OpponentForecast,path=ROOT/'opponent_forecast.npz')

class JointPlanner(frozen.LearnedPlanner):
    joint_entries=(.9,1.3,1.8,2.6,3.5,4.5)
    speed_caps=(2.4,2.0,1.6)
    def planning_profile(self):
        return {**super().planning_profile(),'joint_path_speed_search':True,'joint_entries':list(self.joint_entries),'joint_speed_caps':list(self.speed_caps),'activation':'blocked side while losing ground to the observed rival','minimum_forward_progress_gain_m':.58,'network_retrained':False}

    @staticmethod
    def speed_profile(arc,curvature,initial,cap,lateral=3.2):
        ds=np.diff(arc);curve_limit=np.sqrt(lateral/np.maximum(curvature,.03))
        v=np.maximum(np.minimum(cap,curve_limit),np.sqrt(np.maximum(0.,initial**2-6*arc)))
        if np.any(v>curve_limit+.05):return None
        for i in range(len(v)-2,-1,-1):v[i]=min(v[i],np.sqrt(v[i+1]**2+6*ds[i]))
        if initial>v[0]+.15:return None
        v[0]=max(0.,initial)
        for i in range(1,len(v)):v[i]=min(v[i],np.sqrt(v[i-1]**2+4*ds[i-1]))
        if np.any((v[:-1]**2-v[1:]**2)>6*ds+.02):return None
        times=np.r_[0.,np.cumsum(2*ds/np.maximum(.1,v[:-1]+v[1:]))]
        return v,times

    def opponent_path(self,target,times):
        if self._opponent_cache is not None:
            grid,xy,yaw=self._opponent_cache
            return np.column_stack([np.interp(times,grid,xy[:,i]) for i in (0,1)]),np.interp(times,grid,yaw)
        requested=times;times=np.linspace(0.,12.,121)
        g=self.track;s=[target.s];speed=max(0.,target.speed)
        for dt in np.diff(times):
            allowed=float(g.speed_at(s[-1],min(self.vmax,target.speed+.2),2.4,2.,3.))
            new=float(np.clip(allowed,max(0.,speed-3*dt),speed+2*dt))
            s.append(s[-1]+(speed+new)*.5*dt);speed=new
        d=target.d+target.lateral_speed*.35*(1-np.exp(-times/.35))
        xy,yaw,_=g.sample(np.array(s),d)
        self._opponent_cache=(times,xy,np.unwrap(yaw))
        return self.opponent_path(target,requested)

    def candidates(self,obs,target,committed=None):
        result=super().candidates(obs,target,committed)
        self.last_candidates=result
        self._opponent_cache=None
        self.joint_debug=dict(geometries=0,timed_candidates=0,feasible=0,rescued_sides=0,reasons={})
        if target is None or target.confidence<.6 or target.time_since_seen>.1 or target.observations*self.control_dt<.25:return result
        g=self.track;s,d=map(float,g.project(float(obs['pose_x']),float(obs['pose_y'])))
        rel=float(delta_s(s,target.s,g.length))
        if not -1.8<rel<4.5:return result
        _,refyaw,k=g.sample(s);yaw=float(obs['pose_theta']);v0=max(0.,float(obs['linear_vel_x']))
        # Keep a currently closing approach intact. Rescue paths are useful
        # when our measured speed has fallen below the tracked rival's speed.
        if rel>0. and v0>=target.speed:return result
        error=np.arctan2(np.sin(yaw-refyaw),np.cos(yaw-refyaw));slope=float(np.clip(np.tan(error)*(1-float(k)*d),-.8,.8))
        ahead=np.linspace(0.,6.,61);ss=s+ahead
        left,right=g.widths(ss)
        def reject(reason,xy,action,details):
            self.joint_debug['reasons'][reason]=self.joint_debug['reasons'].get(reason,0)+1
            self._record_path(xy,action,False,reason,source='joint',**details)
        for index,side in ((0,1),(1,-1)):
            old=result[index]
            if old.feasible and old.speed>=.2:continue
            if committed and committed[0]!=old.action:continue
            choices=[]
            merges=(0.,.8,2.) if rel<-.3 else (2.5,4.,99.)
            for entry in self.joint_entries:
              for pad in (.12,.20):
               desired=float(target.d+side*(.155+target.width/2+pad))
               for merge in merges:
                def offsets(x):
                    distance=delta_s(s,x,g.length);u=np.clip(distance/entry,0.,1.)
                    lw,rw=g.widths(x);lane=np.clip(desired,-np.asarray(rw)+.24+self.wall_margin,np.asarray(lw)-.24-self.wall_margin)
                    value=d+(lane-d)*smoothstep(u)+slope*entry*(u-6*u**3+8*u**4-3*u**5)
                    return value*(1-smoothstep((distance-merge)/2.))
                xy,_,_=g.sample(ss,offsets(ss));tangent=np.gradient(xy,axis=0);angles=np.arctan2(tangent[:,1],tangent[:,0]);angles[0]=yaw
                self.joint_debug['geometries']+=1
                details=dict(entry_m=entry,padding_m=pad,merge_factor=1.,merge_at_m=merge)
                wall=float(g.body_clearance(xy,angles).min());curve=self.path_curvature(xy)
                if wall<self.wall_margin:reject('joint wall clearance',xy,old.action,details);continue
                if curve.max()>.95*np.tan(MAX_STEER)/WHEELBASE:reject('joint steering curvature',xy,old.action,details);continue
                arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
                self.tracker.lateral_acceleration=self.passing_lateral_acceleration
                for cap in self.speed_caps:
                    cap=min(self.vmax,cap);self.joint_debug['timed_candidates']+=1;self.search_stats['evaluated']+=1
                    meta={**details,'speed_cap_m_s':cap}
                    profile=self.speed_profile(arc,curve,v0,cap,self.passing_lateral_acceleration)
                    if profile is None:reject('joint braking feasibility',xy,old.action,meta);continue
                    vel,times=profile
                    if times[-1]>12.:continue
                    # Both the currently measured motion and reference-curvature
                    # braking scenario must fit. Margins are unchanged.
                    car=self.motion_clearance(xy,angles,target,times)
                    other,other_yaw=self.opponent_path(target,times)
                    car=min(car,float(separation(xy,angles,other,other_yaw,target.width,target.length).min()))
                    other_end=float(g.project(*other[-1])[0])
                    gain=6.-float(delta_s(target.s,other_end,g.length))
                    required_gain=.58 if rel>0. else .25
                    gain=min(gain,6.-target.speed*times[-1])
                    if gain<required_gain:reject('joint insufficient passing progress',xy,old.action,meta);continue
                    if car<self.car_margin:reject('joint timed body clearance',xy,old.action,meta);continue
                    steer,tracking_speed=self.tracker.control(obs,offset=offsets,vmax=self.vmax,smooth=False)
                    command=min(tracking_speed,float(np.interp(self.control_dt,times,vel)))
                    if command<.15:continue
                    cost=float(times[-1])+.15*abs(desired)+.1/max(.1,wall)
                    c=Candidate(old.action,True,cost,min(wall,car),float(steer),command,desired,xy)
                    c.path_parameters=meta
                    c.joint=dict(**meta,minimum_predicted_body_gap_m=car,minimum_wall_gap_m=wall,predicted_progress_gain_m=gain,required_progress_gain_m=required_gain,planned_times=times.tolist(),planned_speeds=vel.tolist())
                    self.joint_debug['feasible']+=1;self.search_stats['additional_feasible']+=1
                    self._record_path(xy,c.action,True,'joint feasible',cost=cost,source='joint',**meta)
                    choices.append(c)
            if choices:
                chosen=min(choices,key=lambda c:c.cost+(.05*abs(c.offset-committed[1]) if committed else 0.))
                result[index]=chosen;self.joint_debug['rescued_sides']+=1
                self.search_stats['replaced'].append(dict(action=chosen.action,**chosen.path_parameters))
        return self._finish_paths(result)

class JointStage4(frozen.LearnedStage4):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.planner=JointPlanner(self.planner.track,self.vmax,self.stage)
    def plan(self,*args,**kwargs):
        result=super().plan(*args,**kwargs)
        selected=next(c for c in self.planner.last_candidates if c.action==self.action)
        self.telemetry['joint_search']={**self.planner.joint_debug,'selected':hasattr(selected,'joint'),'selected_profile':getattr(selected,'joint',None)}
        return result

BaselineStage4=frozen.LearnedStage4
