"""Stage 4 interaction forecast and lane-preserving fallback, for Gym research.

Inputs are the ego observation and LiDAR-derived Target only. Opponent truth is
never read. The old neural checkpoints and reactive opponent are unchanged.
"""
from dataclasses import replace
import time
import numpy as np
from joint_path_stage4 import JointPlanner,JointStage4
from maneuver_core import Candidate,delta_s,smoothstep
from response_stage4 import separation
from track_geometry import MAX_STEER,WHEELBASE


class InteractionForecast:
    def __init__(self,track):
        self.track=track;self.previous=None;self.acceleration=0.;self.lateral_rate=0.

    def physical_speed(self,target):
        k=float(self.track.sample(target.s)[2])
        return float(np.hypot(max(.25,1-k*target.d)*max(0.,target.speed),target.lateral_speed))

    def update(self,target,dt):
        if target is None:
            self.previous=None;self.acceleration=0.;self.lateral_rate=0.;return
        speed=self.physical_speed(target)
        if self.previous is not None and target.time_since_seen<.1:
            previous=self.previous
            self.acceleration=.7*self.acceleration+.3*float(np.clip((speed-previous[0])/dt,-3.,2.))
            rate=float(np.clip((target.d-previous[1])/dt,-.7,.7))
            self.lateral_rate=.5*self.lateral_rate+.5*rate
        else:self.lateral_rate=float(target.lateral_speed)
        self.previous=(speed,target.d)

    def trajectories(self,target,times):
        g=self.track;v0=self.physical_speed(target)
        if v0<.05:v0=0.
        lateral=float(np.clip(self.lateral_rate,-.6,.6))
        # Different plausible responses: current trend, earlier braking, and a
        # return toward the reference. None assumes that the rival yields.
        modes=[]
        modes_to_try=[('trend',False,False),('braking',True,False),('return',False,True)]
        for name,brake,returning in modes_to_try:
            ss=[target.s];ds=[target.d];v=v0
            for t,dt in zip(times[1:],np.diff(times)):
                if returning:
                    u=min(1.,t/1.4);d=target.d*(1-smoothstep(u))+lateral*1.4*(u-6*u**3+8*u**4-3*u**5)
                else:d=target.d+lateral*.7*(1-np.exp(-t/.7))
                d=ds[-1]+float(np.clip(d-ds[-1],-.5*v*dt,.5*v*dt))
                k=float(g.sample(ss[-1])[2]);factor=max(.3,1-k*d)
                limit=min(2.4,float(g.speed_at(ss[-1],2.4,2.4,2.,3.)),np.sqrt(2.4/max(.03,abs(k)/factor)))
                accel=min(self.acceleration,-1.2) if brake and t<=.45 else self.acceleration*np.exp(-t/.5)
                if v0==0.:accel=0.
                nv=max(0.,v-3*dt,min(v+accel*dt,limit))
                # Physical speed -> progress along reference (Frenet s rate).
                ss.append(ss[-1]+.5*(v+nv)*dt/factor);ds.append(d);v=nv
            xy,refyaw,_=g.sample(np.asarray(ss),np.asarray(ds));tangent=np.gradient(xy,axis=0)
            yaw=np.arctan2(tangent[:,1],tangent[:,0]);stopped=np.linalg.norm(tangent,axis=1)<1e-7;yaw[stopped]=refyaw[stopped]
            modes.append(dict(name=name,path=xy,yaw=yaw))
        return modes


class InteractionPlanner(JointPlanner):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.interaction=InteractionForecast(self.track)
    def planning_profile(self):
        return {**super().planning_profile(),'interaction_forecast':'physical-speed trend / braking / reference return',
                'fallback':'short-horizon tracked rollout, lane hold, brake or certified creep','network_retrained':False}

    def rollout(self,obs,candidate,initial_steer,horizon=1.2):
        dt=.05;times=np.arange(round(horizon/dt)+1)*dt;path=np.asarray(candidate.path)
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
        curves=self.path_curvature(path)
        x,y,yaw,v=[float(obs[k]) for k in ('pose_x','pose_y','pose_theta','linear_vel_x')]
        steer=float(initial_steer);xy=[];angles=[];speeds=[]
        for step,t in enumerate(times):
            xy.append([x,y]);angles.append(yaw);speeds.append(v)
            index=int(np.argmin(np.sum((path-[x,y])**2,axis=1)))
            lookahead=float(np.clip(.65+.14*max(v,0.),.65,1.1))
            goal=np.array([np.interp(arc[index]+lookahead,arc,path[:,j]) for j in (0,1)])
            diff=goal-[x,y];lateral=-np.sin(yaw)*diff[0]+np.cos(yaw)*diff[1]
            desired=float(np.clip(np.arctan2(2*WHEELBASE*lateral,max(.05,diff@diff)),-MAX_STEER,MAX_STEER))
            if step==0:desired=candidate.steering
            steer+=float(np.clip(desired-steer,-2.5*dt,2.5*dt))
            curve_limit=float(np.sqrt(self.passing_lateral_acceleration/max(.03,curves[index])))
            requested=min(candidate.speed,curve_limit)
            v=float(np.clip(requested,max(0.,v-3*dt),v+2*dt))
            beta=np.arctan(.5193*np.tan(steer))
            x+=dt*v*np.cos(yaw+beta);y+=dt*v*np.sin(yaw+beta)
            yaw+=dt*v*np.cos(beta)*np.tan(steer)/WHEELBASE
        return np.asarray(xy),np.asarray(angles),times,np.asarray(speeds)

    def check(self,obs,candidate,initial_steer,predictions):
        return self.check_many(obs,[candidate],initial_steer,predictions)[0]

    def check_many(self,obs,candidates,initial_steer,predictions):
        """Batch bicycle rollouts; identical dynamics for every candidate."""
        n=len(candidates);lengths={len(c.path) for c in candidates}
        if len(lengths)>1:return [self.check(obs,c,initial_steer,predictions) for c in candidates]
        paths=np.stack([c.path for c in candidates]);arcs=np.column_stack([np.zeros(n),np.cumsum(np.linalg.norm(np.diff(paths,axis=1),axis=2),axis=1)])
        curves=np.stack([self.path_curvature(p) for p in paths]);row=np.arange(n)
        x,y,yaw,v=[np.full(n,float(obs[k])) for k in ('pose_x','pose_y','pose_theta','linear_vel_x')]
        steer=np.full(n,float(initial_steer));xy=[];angles=[];dt=.05;tt=np.arange(25)*dt
        cap=np.array([c.speed for c in candidates]);first_steer=np.array([c.steering for c in candidates])
        for step in range(25):
            xy.append(np.column_stack([x,y]));angles.append(yaw.copy())
            nearest=np.argmin(np.sum((paths-np.column_stack([x,y])[:,None,:])**2,axis=2),axis=1)
            look=np.clip(.65+.14*np.maximum(v,0.),.65,1.1);aim=np.minimum(arcs[:,-1],arcs[row,nearest]+look)
            hi=np.argmax(arcs>=aim[:,None],axis=1);lo=np.maximum(0,hi-1)
            fraction=(aim-arcs[row,lo])/np.maximum(1e-10,arcs[row,hi]-arcs[row,lo])
            goal=paths[row,lo]+fraction[:,None]*(paths[row,hi]-paths[row,lo]);dx=goal[:,0]-x;dy=goal[:,1]-y
            lateral=-np.sin(yaw)*dx+np.cos(yaw)*dy
            desired=np.clip(np.arctan2(2*WHEELBASE*lateral,np.maximum(.05,dx*dx+dy*dy)),-MAX_STEER,MAX_STEER)
            if step==0:desired=first_steer
            steer+=np.clip(desired-steer,-2.5*dt,2.5*dt)
            requested=np.minimum(cap,np.sqrt(self.passing_lateral_acceleration/np.maximum(.03,curves[row,nearest])))
            v=np.clip(requested,np.maximum(0.,v-3*dt),v+2*dt);beta=np.arctan(.5193*np.tan(steer))
            x=x+dt*v*np.cos(yaw+beta);y=y+dt*v*np.sin(yaw+beta);yaw=yaw+dt*v*np.cos(beta)*np.tan(steer)/WHEELBASE
        xy=np.stack(xy,axis=1);angles=np.stack(angles,axis=1)
        walls=self.track.body_clearance(xy,angles)
        gaps=np.min([separation(xy.reshape(-1,2),angles.ravel(),np.tile(p['path'],(n,1)),np.tile(p['yaw'],n),self.target.width,self.target.length).reshape(n,-1) for p in predictions],axis=0)
        # The first 0.8 seconds covers a 2.4 m/s stop at 3 m/s^2. Longer
        # predictions rank options but cannot be treated as certain opponent motion.
        near=tt<=.8;end_s,end_d=self.track.project(xy[:,-1,0],xy[:,-1,1]);start_s,_=self.track.project(*xy[0,0])
        result=[]
        for i in range(n):
            initial=float(gaps[i,0]);floor=.06 if initial>=.06 else max(.025,initial-.005)
            wall=float(walls[i,near].min());gap=float(gaps[i,near].min())
            required_wall=min(.025,max(.01,float(walls[i,0])-.005))
            safe=bool(wall>=required_wall and gap>=floor)
            if initial<.06:safe=safe and gaps[i,-1]>=.08
            score=float(delta_s(start_s,end_s[i],self.track.length))-.20*abs(float(end_d[i]))-.2*max(0.,.06-float(gaps[i].min()))
            result.append(dict(safe=safe,gap=gap,wall=wall,score=score,path=xy[i],times=tt,initial_gap=initial,recovery=bool(safe and initial<.08 and gaps[i,-1]>=.08)))
        return result

    def fallback_candidates(self,obs,target):
        g=self.track;s,d=map(float,g.project(float(obs['pose_x']),float(obs['pose_y'])))
        _,ry,k=g.sample(s);error=np.arctan2(np.sin(obs['pose_theta']-ry),np.cos(obs['pose_theta']-ry))
        slope=float(np.clip(np.tan(error)*(1-float(k)*d),-.8,.8));ss=s+np.linspace(0.,4.5,46)
        offsets_to_try=sorted(set([0.,round(d,3),round(d-.15,3),round(d+.15,3),round(target.d-.45,3),round(target.d+.45,3)]))
        for goal in offsets_to_try:
            for entry in (1.2,2.4):
                def offsets(x):
                    u=np.clip(delta_s(s,x,g.length)/entry,0.,1.);l,r=g.widths(x)
                    lane=np.clip(goal,-np.asarray(r)+.24+self.wall_margin,np.asarray(l)-.24-self.wall_margin)
                    return d+(lane-d)*smoothstep(u)+slope*entry*(u-6*u**3+8*u**4-3*u**5)
                path,_,_=g.sample(ss,offsets(ss));tangent=np.gradient(path,axis=0);yaw=np.arctan2(tangent[:,1],tangent[:,0]);yaw[0]=obs['pose_theta']
                if self.track.body_clearance(path[:25],yaw[:25]).min()<.025:continue
                steer,limit=self.tracker.control(obs,offset=offsets,vmax=self.vmax,smooth=False)
                for cap in (self.vmax,min(1.3,self.interaction.physical_speed(target)),.5,0.):
                    action='WAIT' if abs(goal)<.1 else 'LEFT' if goal>target.d else 'RIGHT'
                    c=Candidate(action,True,4.5/max(.3,cap),0.,float(steer),min(cap,limit),goal,path)
                    c.path_parameters=dict(entry_m=entry,padding_m=abs(goal-target.d)-.31,merge_factor=1.)
                    yield c


class InteractionStage4(JointStage4):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.planner=InteractionPlanner(self.planner.track,self.vmax,self.stage)
        self.interaction_debug={};self.slow_seconds=0.
    def update_pass_completion_context(self,obs,target):
        super().update_pass_completion_context(obs,target)
        self.planner.interaction.update(target,self.planner.control_dt)
        self.planner.target=target
    def select_response_maneuver(self,candidates,index,obs,target):
        index=super().select_response_maneuver(candidates,index,obs,target)
        self.interaction_debug=dict(active=False,override=False,checked=0,mode='baseline')
        self.slow_seconds=self.slow_seconds+self.planner.control_dt if obs['linear_vel_x']<.2 else 0.
        if target is None or target.confidence<.6 or target.time_since_seen>.15:return index
        g=self.planner.track;s,d=map(float,g.project(obs['pose_x'],obs['pose_y']));rel=float(delta_s(s,target.s,g.length))
        if not -1.5<rel<2.5:return index
        predictions=self.planner.interaction.trajectories(target,np.arange(25)*.05)
        chosen=candidates[index]
        checked=self.planner.check(obs,chosen,self.steering,predictions)
        self.interaction_debug.update(active=True,checked=1,physical_opponent_speed=self.planner.interaction.physical_speed(target),
            tracked_reference_speed=target.speed,selected_gap=checked['gap'],
            predictions=[dict(name=p['name'],path=p['path'].tolist()) for p in predictions])
        # Retain the chosen neural maneuver when its short rollout remains clear.
        if checked['safe'] and not (self.slow_seconds>.5 or chosen.speed<.2):return index
        options=[]
        alternatives=list(self.planner.fallback_candidates(obs,target))
        checks=self.planner.check_many(obs,alternatives,self.steering,predictions) if alternatives else []
        for c,test in zip(alternatives,checks):
            self.interaction_debug['checked']+=1
            self.planner._record_path(c.path,c.action,test['safe'],'interaction clear' if test['safe'] else 'interaction forecast conflict',source='interaction',cost=c.cost,**c.path_parameters)
            if test['safe']:
                c.clearance=min(test['gap'],test['wall']);c.interaction_certificate=test
                options.append((test['score']-.08*abs(c.offset-d),c,test))
        if options:
            _,replacement,test=max(options,key=lambda p:p[0]);index={'LEFT':0,'RIGHT':1,'WAIT':2}[replacement.action]
            candidates[index]=replacement
            self.interaction_debug.update(override=True,mode='checked lane recovery',selected_gap=test['gap'],rollout=test['path'].tolist())
            if replacement.action=='WAIT' and rel>1.:self.commit=None
        else:
            # No clear motion found: brake while holding the current lane. Never
            # command an unverified return across an overlapping opponent.
            steer,_=self.planner.tracker.control(obs,offset=d,vmax=self.vmax,smooth=False)
            path=g.sample(s+np.linspace(0.,3.,31),d)[0]
            replacement=Candidate('WAIT',True,30.,checked['gap'],steer,0.,d,path)
            candidates[2]=replacement;index=2
            self.interaction_debug.update(override=True,mode='brake in current lane',selected_gap=checked['gap'])
        self.planner.response_choice=None
        self.planner.response_debug.update(mode='interaction fallback',selected=False)
        self.planner._finish_paths(candidates)
        return index
    def response_speed_caps(self,caps,tracked,selected,obs,target):
        # Only a checked recovery may release close side-padding beams. All wall,
        # unknown and front-axle returns keep the original independent brake.
        certificate=getattr(selected,'interaction_certificate',None)
        if certificate is None:return super().response_speed_caps(caps,tracked,selected,obs,target)
        ranges=np.nan_to_num(np.asarray(obs['scan']),nan=0.,posinf=20.,neginf=0.)
        x,y=ranges*np.cos(self.angles),ranges*np.sin(self.angles);curv=np.tan(selected.steering)/WHEELBASE
        if abs(curv)<1e-3:lateral,along=np.abs(y),x
        else:
            radius=1/curv;lateral=np.abs(np.hypot(x,y-radius)-abs(radius));along=abs(radius)*np.arctan2(x,np.sign(radius)*(radius-y))
        forward=(x>0)&(along>0)&(along<2)&(lateral<.195)
        padding=tracked&(x[forward]<.29)&(lateral[forward]>=.17)
        caps=caps.copy();caps[padding]=self.vmax
        self.interaction_debug['padding_beams_released']=int(padding.sum())
        return caps
    def plan(self,*a,**kw):
        begin=time.perf_counter_ns();result=super().plan(*a,**kw)
        self.telemetry['interaction']=self.interaction_debug
        self.control_times[-1]=(time.perf_counter_ns()-begin)/1e6
        return result
