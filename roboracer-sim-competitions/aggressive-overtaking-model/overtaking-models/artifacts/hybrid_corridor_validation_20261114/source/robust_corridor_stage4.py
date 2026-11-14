"""Experimental Stage 4: occlusion-aware checks and short recovery motions."""
from dataclasses import replace
import numpy as np
from corridor_stage4 import CorridorPlanner,CorridorStage4
from interaction_stage4 import InteractionPlanner,InteractionForecast
from joint_path_stage4 import JointStage4
from maneuver_core import Candidate,delta_s,smoothstep
from track_geometry import WHEELBASE,MAX_STEER

class FastInteractionForecast(InteractionForecast):
    """Evaluate the same three motion hypotheses in one array per time step."""
    def trajectories(self,target,times):
        g=self.track;v0=self.physical_speed(target)
        if v0<.05:v0=0.
        lateral=float(np.clip(self.lateral_rate,-.6,.6))
        ss=np.full(3,target.s);ds=np.full(3,target.d);v=np.full(3,v0);positions=[ss.copy()];offsets=[ds.copy()]
        for t,dt in zip(times[1:],np.diff(times)):
            u=min(1.,t/1.4)
            d=np.full(3,target.d+lateral*.7*(1-np.exp(-t/.7)))
            d[2]=target.d*(1-smoothstep(u))+lateral*1.4*(u-6*u**3+8*u**4-3*u**5)
            d=ds+np.clip(d-ds,-.5*v*dt,.5*v*dt)
            k=g.sample(ss)[2];factor=np.maximum(.3,1-k*d)
            limit=np.minimum(2.4,np.minimum(g.speed_at(ss,2.4,2.4,2.,3.),np.sqrt(2.4/np.maximum(.03,np.abs(k)/factor))))
            accel=np.full(3,self.acceleration*np.exp(-t/.5))
            if t<=.45:accel[1]=min(self.acceleration,-1.2)
            if v0==0.:accel[:]=0.
            nv=np.maximum(0.,np.maximum(v-3*dt,np.minimum(v+accel*dt,limit)))
            ss=ss+.5*(v+nv)*dt/factor;ds=d;v=nv;positions.append(ss.copy());offsets.append(ds.copy())
        xy,refyaw,_=g.sample(np.stack(positions,axis=1),np.stack(offsets,axis=1))
        tangent=np.gradient(xy,axis=1);yaw=np.arctan2(tangent[:,:,1],tangent[:,:,0]);stopped=np.linalg.norm(tangent,axis=2)<1e-7;yaw[stopped]=refyaw[stopped]
        return [dict(name=name,path=xy[i],yaw=yaw[i]) for i,name in enumerate(('trend','braking','return'))]

class CachedGeometry:
    """Exact per-decision reuse of map work; never cache moving opponents."""
    def __init__(self,geometry):
        self.geometry=geometry;self.clear();self._speeds={}
    def __getattr__(self,name):return getattr(self.geometry,name)
    def clear(self):self._samples={};self._projects={};self._widths={}
    @staticmethod
    def key(value):
        a=np.asarray(value,dtype=float);return a.shape,a.tobytes()
    def sample(self,s,d=0.):
        ss,dd=np.broadcast_arrays(np.asarray(s,float),np.asarray(d,float));key=self.key(ss)
        if key not in self._samples:
            base,yaw,k=self.geometry.sample(ss)
            normal=np.stack([-np.sin(yaw),np.cos(yaw)],axis=-1)
            self._samples[key]=(base,yaw,k,normal)
        base,yaw,k,normal=self._samples[key]
        return base+dd[...,None]*normal,np.array(yaw,copy=True),np.array(k,copy=True)
    def project(self,x,y):
        key=self.key(x),self.key(y)
        if key not in self._projects:self._projects[key]=self.geometry.project(x,y)
        return self._projects[key]
    def widths(self,s):
        key=self.key(s)
        if key not in self._widths:self._widths[key]=self.geometry.widths(s)
        return self._widths[key]
    def speed_at(self,s,vmax=2.4,lateral_acceleration=2.,acceleration=2.,braking=3.):
        key=tuple(float(v) for v in (vmax,lateral_acceleration,acceleration,braking))
        if key not in self._speeds:
            self.geometry.speed_at(0.,*key)
            values=self.geometry._speed_profiles[key][0];ss=self.geometry.s;length=self.geometry.length
            self._speeds[key]=(np.r_[ss[-1]-length,ss,ss[0]+length],np.r_[values[-1],values,values[0]])
        positions,values=self._speeds[key]
        return np.interp(np.asarray(s)%self.length,positions,values)

class RobustCorridorPlanner(CorridorPlanner):
    def __init__(self,track,*a,**kw):
        super().__init__(CachedGeometry(track),*a,**kw)
        self.interaction=FastInteractionForecast(self.track)
    def planning_profile(self):
        return {**super().planning_profile(),'occluded_opponent_checks_seconds':.65,
                'local_recovery_arc_speeds_m_s':[.25,.5,.8],
                'search_trigger':'unsafe or low executable speed; preserve progressing paths',
                'planning_optimization':'batched map corridor, unique LiDAR arcs, exact map cache, batched opponent hypotheses'}
    def candidates(self,*a,**kw):
        self.track.clear()
        return super().candidates(*a,**kw)

    def lidar_caps(self,obs,target,candidates):
        if not candidates:return np.array([])
        # Speed variants share a steering arc. Compute its scan intersection once.
        _,indices,inverse=np.unique([c.steering for c in candidates],return_index=True,return_inverse=True)
        return super().lidar_caps(obs,target,[candidates[i] for i in indices])[inverse]

    def fallback_candidates(self,obs,target):
        if getattr(self,"legacy",False):
            yield from InteractionPlanner.fallback_candidates(self,obs,target)
            return
        g=self.track;self._width_cache={}
        s,d=map(float,g.project(obs['pose_x'],obs['pose_y']));_,ry,k=g.sample(s)
        error=np.arctan2(np.sin(obs['pose_theta']-ry),np.cos(obs['pose_theta']-ry));slope=np.clip(np.tan(error)*(1-k*d),-.8,.8)
        ahead=np.linspace(0.,4.5,46);ss=s+ahead
        left,right=self.wall_widths(ss);inset=.24+self.wall_margin
        low,high=float(-max(right)+inset),float(max(left)-inset)
        goals=np.unique(np.r_[np.linspace(low,high,max(2,int(np.ceil((high-low)/.25))+1)),0,d,d-.15,d+.15,target.d-.45,target.d+.45])
        goals=goals[(goals>=low)&(goals<=high)]
        entries=np.tile([.65,1.2,2.4,3.6],len(goals));goals=np.repeat(goals,4)
        u=np.clip(ahead[None,:]/entries[:,None],0,1)
        lanes=np.clip(goals[:,None],-right+inset,left-inset)
        offsets=d+(lanes-d)*smoothstep(u)+slope*entries[:,None]*(u-6*u**3+8*u**4-3*u**5)
        base,yaws,_=g.sample(ss);normal=np.stack([-np.sin(yaws),np.cos(yaws)],axis=-1)
        paths=base[None,:,:]+offsets[:,:,None]*normal[None,:,:]
        tangent=np.gradient(paths,axis=1);angles=np.arctan2(tangent[:,:,1],tangent[:,:,0]);angles[:,0]=obs['pose_theta']
        good=g.body_clearance(paths,angles).min(axis=1)>=.025
        curves=np.stack([self.path_curvature(p) for p in paths]);good&=curves.max(axis=1)<=.95*np.tan(MAX_STEER)/WHEELBASE
        ids=np.flatnonzero(good)
        if len(ids):
            projected,_=g.project(paths[ids,:,0].ravel(),paths[ids,:,1].ravel());ref=g.sample(projected)[1].reshape(len(ids),-1)
            good[ids]&=np.all(np.cos(angles[ids]-ref)>=0,axis=1)
        choices=[]
        for i in np.flatnonzero(good):
            goal,entry=float(goals[i]),float(entries[i])
            def offset(x):
                v=np.clip(delta_s(s,x,g.length)/entry,0,1);l,r=self.wall_widths(x)
                return d+(np.clip(goal,-r+inset,l-inset)-d)*smoothstep(v)+slope*entry*(v-6*v**3+8*v**4-3*v**5)
            steer,limit=self.tracker.control(obs,offset=offset,vmax=self.vmax,smooth=False)
            for cap in (self.vmax,1.2,.35):
                action='WAIT' if abs(goal)<.1 else 'LEFT' if goal>target.d else 'RIGHT'
                c=Candidate(action,True,4.5/max(.3,cap),0.,steer,min(cap,limit),goal,paths[i])
                c.path_parameters=dict(entry_m=entry,padding_m=abs(goal-target.d)-.31,merge_factor=1.,lateral_goal_m=goal)
                choices.append(c)
        # Local recovery arcs are certified over their executed horizon. Unlike
        # a full lane-change curve, they need not fit a distant, unexecuted bend.
        if obs['linear_vel_x']<.8:
            arc=np.linspace(0.,1.5,46);yaw=float(obs['pose_theta']);origin=np.array([obs['pose_x'],obs['pose_y']])
            for steer in np.unique(np.r_[np.linspace(-MAX_STEER,MAX_STEER,15),0.]):
                curv=np.tan(steer)/WHEELBASE
                a=arc*curv;px=arc if abs(curv)<1e-8 else np.sin(a)/curv;py=np.zeros_like(arc) if abs(curv)<1e-8 else (1-np.cos(a))/curv
                path=origin+np.stack([np.cos(yaw)*px-np.sin(yaw)*py,np.sin(yaw)*px+np.cos(yaw)*py],axis=-1)
                _,goal=g.project(*path[15]);goal=float(goal)
                for cap in (.25,.5,.8):
                    c=Candidate('LEFT' if goal>target.d else 'RIGHT',True,4.,0.,float(steer),cap,goal,path)
                    c.path_parameters=dict(entry_m=0.,padding_m=abs(goal-target.d)-.31,merge_factor=1.,lateral_goal_m=goal,local_arc=True)
                    choices.append(c)
        caps=self.lidar_caps(obs,target,choices)
        self.corridor_debug=dict(lower_offset_m=low,upper_offset_m=high,goals=len(goals)//4,generated=len(choices),lidar_blocked=int(sum(caps<.05)))
        for c,cap in zip(choices,caps):
            c.speed=min(c.speed,float(cap));yield c

class RobustCorridorStage4(CorridorStage4):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        self.planner=RobustCorridorPlanner(self.planner.track,self.vmax,self.stage);self.planner.angles=self.angles
    def select_response_maneuver(self,candidates,index,obs,target):
        if target is None:
            self._low_corner_mode=None
        else:
            s,_=self.planner.track.project(obs['pose_x'],obs['pose_y'])
            rel=float(delta_s(s,target.s,self.planner.track.length))
            if not -2.2<rel<4.:self._low_corner_mode=None
            if getattr(self,'_low_corner_mode',None) is None and .3<rel<2.5:
                self._low_corner_mode=obs['linear_vel_x']<.75*self.vmax
            if getattr(self,'_low_corner_mode',False) and target.confidence>=.6 and target.time_since_seen<=.1:
                return self._blocked_wide_select(candidates,index,obs,target)
        index=JointStage4.select_response_maneuver(self,candidates,index,obs,target)
        self.interaction_debug=dict(active=False,override=False,checked=0,mode='baseline')
        self.slow_seconds=self.slow_seconds+self.planner.control_dt if obs['linear_vel_x']<.2 else 0.
        if target is None:return index
        # A temporarily hidden rival still occupies space. Keep its existing
        # measured-motion forecast active rather than authorizing a blind merge.
        if target.observed_seconds<.2 or target.time_since_seen>.65:return index
        g=self.planner.track;s,d=map(float,g.project(obs['pose_x'],obs['pose_y']));rel=float(delta_s(s,target.s,g.length))
        if not -2.2<rel<3.:return index
        predictions=self.planner.interaction.trajectories(target,np.arange(25)*.05)
        chosen=candidates[index];checked=self.planner.check(obs,chosen,self.steering,predictions)
        self.interaction_debug.update(active=True,checked=1,selected_gap=checked['gap'],predictions=[dict(name=p['name'],path=p['path'].tolist()) for p in predictions],occluded=target.time_since_seen>.15)
        # Avoid rebuilding a large lattice while a valid selected route is
        # progressing. Stalls still trigger a fresh full-width search.
        if checked['safe'] and min(chosen.speed,checked['lidar_cap'])>.2 and self.slow_seconds<.5:return index
        alternatives=list(self.planner.fallback_candidates(obs,target));checks=self.planner.check_many(obs,alternatives,self.steering,predictions) if alternatives else []
        options=[]
        for c,test in zip(alternatives,checks):
            self.interaction_debug['checked']+=1
            self.planner._record_path(c.path,c.action,test['safe'],'checked corridor' if test['safe'] else 'forecast conflict',source='corridor',cost=c.cost,**c.path_parameters)
            if test['safe'] and c.speed>.05:
                c.interaction_certificate=test
                options.append((test['score']-.08*abs(c.offset-d),c,test))
        if options:
            _,c,test=max(options,key=lambda x:x[0]);index={'LEFT':0,'RIGHT':1,'WAIT':2}[c.action];candidates[index]=c
            self.interaction_debug.update(override=True,mode='checked wall corridor',selected_gap=test['gap'],rollout=test['path'].tolist(),selected_lateral_goal_m=c.offset)
        elif checked['safe']:return index
        else:
            steer,_=self.planner.tracker.control(obs,offset=d,vmax=self.vmax,smooth=False)
            candidates[2]=Candidate('WAIT',True,30.,checked['gap'],steer,0.,d,g.sample(s+np.linspace(0.,3.,31),d)[0]);index=2
            self.interaction_debug.update(override=True,mode='brake in current lane')
        self.planner.response_choice=None;self.planner.response_debug.update(mode='robust corridor fallback',selected=False);self.planner._finish_paths(candidates)
        return index
