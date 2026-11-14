"""Wall-bounded Stage 4 lateral search with executable LiDAR-limited controls.

No fixed lateral offset budget. First map walls bound the sampled corridor;
body, prediction, steering, and independent LiDAR checks still apply.
"""
from dataclasses import replace
import numpy as np
from interaction_stage4 import InteractionPlanner,InteractionStage4
from maneuver_core import Candidate,delta_s,smoothstep
from track_geometry import WHEELBASE,MAX_STEER
from joint_path_stage4 import JointStage4

class CorridorPlanner(InteractionPlanner):
    def planning_profile(self):
        return {**super().planning_profile(),'lateral_search':'first map wall on either side; no fixed offset cap',
                'corridor_grid_spacing_m':.25,'corridor_entry_distances_m':[.65,1.2,2.4,3.6],
                'fallback_scores_lidar_limited_speed':True}

    def wall_widths(self,positions):
        g=self.track;original=np.asarray(positions);ss=np.atleast_1d(original).astype(float)
        key=(ss.shape,ss.tobytes())
        cache=getattr(self,'_width_cache',{})
        if key in cache:left,right=cache[key]
        else:
            left,right=[np.array(v,copy=True) for v in g.widths(ss)]
            # The shared geometry's cached width rays stop at 3 m. Extend only
            # rays that reach that range, until the first actual wall/map edge.
            xy,yaw,_=g.sample(ss);normal=np.stack([-np.sin(yaw),np.cos(yaw)],axis=-1)
            distance=np.arange(0.,np.hypot(*g.occupancy.shape)*g.resolution+.03,.025)
            for widths,sign in ((left,1.),(right,-1.)):
                mask=widths>=2.96
                if mask.any():
                    free=g.clearance(xy[mask,None,:]+sign*normal[mask,None,:]*distance[None,:,None])>.015
                    first=np.argmax(~free,axis=1);first[np.all(free,axis=1)]=len(distance)-1
                    widths[mask]=np.maximum(0.,distance[first]-g.resolution)
            cache[key]=(left,right);self._width_cache=cache
        if original.ndim==0:return float(left[0]),float(right[0])
        return left,right

    def lidar_caps(self,obs,target,candidates):
        """Mirror the final controller's normal brake before ranking movement.

        No beams are released here. A path that will receive a zero command
        cannot win by claiming forward progress at its uncapped target speed.
        """
        if not candidates:return np.array([])
        ranges=np.nan_to_num(np.asarray(obs['scan'],float),nan=0.,posinf=20.,neginf=0.)
        x,y=ranges*np.cos(self.angles),ranges*np.sin(self.angles)
        curvature=np.tan(np.array([c.steering for c in candidates]))/WHEELBASE
        safe_curvature=np.where(np.abs(curvature)<1e-3,1.,curvature);radius=1/safe_curvature
        lateral=np.abs(np.hypot(x[None,:],y[None,:]-radius[:,None])-np.abs(radius[:,None]))
        along=np.abs(radius[:,None])*np.arctan2(x[None,:],np.sign(radius[:,None])*(radius[:,None]-y[None,:]))
        straight=np.abs(curvature)<1e-3;lateral[straight]=np.abs(y);along[straight]=x
        forward=(x[None,:]>0)&(along>0)&(along<2.)&(lateral<.195)
        obstacle_speed=np.zeros(len(x))
        if target is not None and target.confidence>.6 and target.time_since_seen<.15:
            txy,tyaw,_=self.track.sample(target.s,target.d);eyaw=float(obs['pose_theta']);co,sn=np.cos(eyaw),np.sin(eyaw)
            hits=np.column_stack([float(obs['pose_x'])+co*x-sn*y,float(obs['pose_y'])+sn*x+co*y]);relative=hits-txy
            longitudinal=relative@np.array([np.cos(tyaw),np.sin(tyaw)]);transverse=relative@np.array([-np.sin(tyaw),np.cos(tyaw)])
            tracked=(np.abs(longitudinal)<target.length/2+.12)&(np.abs(transverse)<target.width/2+.08)&(self.track.clearance(hits)>.12)
            obstacle_speed[tracked]=max(0.,target.speed-.3-3*target.time_since_seen)*max(0.,float(np.cos(tyaw-eyaw)))
        limits=np.sqrt(obstacle_speed[None,:]**2+6*np.maximum(0.,along-.40))
        return np.min(np.where(forward,limits,np.inf),axis=1)

    def check(self,obs,candidate,initial_steer,predictions):
        if getattr(self,'legacy',False):return InteractionPlanner.check(self,obs,candidate,initial_steer,predictions)
        cap=float(self.lidar_caps(obs,self.target,[candidate])[0])
        executable=replace(candidate,speed=min(candidate.speed,cap))
        result=super().check(obs,executable,initial_steer,predictions)
        result['lidar_cap']=cap
        if candidate.speed>.2 and cap<.1:result['safe']=False
        return result

    def fallback_candidates(self,obs,target):
        if getattr(self,'legacy',False):
            yield from InteractionPlanner.fallback_candidates(self,obs,target)
            return
        g=self.track;self._width_cache={}
        s,d=map(float,g.project(float(obs['pose_x']),float(obs['pose_y'])))
        _,ry,k=g.sample(s);error=np.arctan2(np.sin(obs['pose_theta']-ry),np.cos(obs['pose_theta']-ry))
        slope=float(np.clip(np.tan(error)*(1-float(k)*d),-.8,.8));ss=s+np.linspace(0.,4.5,46)
        left,right=self.wall_widths(ss);inset=.24+self.wall_margin
        low=float(-max(right)+inset);high=float(max(left)-inset)
        goals=np.unique(np.r_[np.linspace(low,high,max(2,int(np.ceil((high-low)/.25))+1)),0.,d,d-.15,d+.15,target.d-.45,target.d+.45])
        goals=goals[(goals>=low)&(goals<=high)]
        self.corridor_debug=dict(lower_offset_m=low,upper_offset_m=high,goals=len(goals),generated=0,lidar_blocked=0)
        choices=[]
        for goal in goals:
            for entry in (.65,1.2,2.4,3.6):
                def offsets(x):
                    u=np.clip(delta_s(s,x,g.length)/entry,0.,1.);l,r=self.wall_widths(x)
                    lane=np.clip(goal,-r+inset,l-inset)
                    return d+(lane-d)*smoothstep(u)+slope*entry*(u-6*u**3+8*u**4-3*u**5)
                path,_,_=g.sample(ss,offsets(ss));tangent=np.gradient(path,axis=0);yaw=np.arctan2(tangent[:,1],tangent[:,0]);yaw[0]=obs['pose_theta']
                if g.body_clearance(path,yaw).min()<.025:continue
                # Large normal offsets can fold a Frenet path inside a bend.
                # Reject unsteerable/reversing curves, not their lateral extent.
                if self.path_curvature(path).max()>.95*np.tan(MAX_STEER)/WHEELBASE:continue
                projected,_=g.project(path[:,0],path[:,1]);reference_yaw=g.sample(projected)[1]
                if np.any(np.cos(yaw-reference_yaw)<0.):continue
                steer,limit=self.tracker.control(obs,offset=offsets,vmax=self.vmax,smooth=False)
                for cap in (self.vmax,1.2,.35):
                    action='WAIT' if abs(goal)<.1 else 'LEFT' if goal>target.d else 'RIGHT'
                    c=Candidate(action,True,4.5/max(.3,cap),0.,float(steer),min(cap,limit),float(goal),path)
                    c.path_parameters=dict(entry_m=entry,padding_m=abs(float(goal)-target.d)-.31,merge_factor=1.,lateral_goal_m=float(goal))
                    choices.append(c)
        caps=self.lidar_caps(obs,target,choices)
        for c,cap in zip(choices,caps):
            c.speed=min(c.speed,float(cap));self.corridor_debug['generated']+=1
            if c.speed<.05:self.corridor_debug['lidar_blocked']+=1
            yield c

class CorridorStage4(InteractionStage4):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.planner=CorridorPlanner(self.planner.track,self.vmax,self.stage);self.planner.angles=self.angles
    def _blocked_wide_select(self,candidates,index,obs,target):
        # First retain the established controller's complete choice, including
        # its narrow recovery. Wider paths supplement it rather than replacing
        # a maneuver that is already making progress.
        self.planner.legacy=True
        try:index=InteractionStage4.select_response_maneuver(self,candidates,index,obs,target)
        finally:self.planner.legacy=False
        if target is None or target.confidence<.6 or target.time_since_seen>.15:
            self.progress_history=[];return index
        g=self.planner.track;s,d=map(float,g.project(obs['pose_x'],obs['pose_y']));rel=float(delta_s(s,target.s,g.length))
        if not -1.5<rel<2.5:return index
        now=self.counter*self.planner.control_dt
        history=getattr(self,'progress_history',[]);history.append((now,rel))
        history=[v for v in history if now-v[0]<=3.1];self.progress_history=history
        stalled=bool(history and now-history[0][0]>=3. and history[0][1]-rel<.15 and rel>.3)
        chosen=candidates[index];cap=float(self.planner.lidar_caps(obs,target,[chosen])[0])
        low_motion=min(chosen.speed,cap)<.7
        proactive=stalled and self.counter%6==0
        if not (low_motion or proactive):return index
        predictions=self.planner.interaction.trajectories(target,np.arange(25)*.05)
        checked=self.planner.check(obs,chosen,self.steering,predictions)
        self.interaction_debug.update(active=True,physical_opponent_speed=self.planner.interaction.physical_speed(target),tracked_reference_speed=target.speed,
            selected_gap=checked['gap'],predictions=[dict(name=p['name'],path=p['path'].tolist()) for p in predictions],proactive_search=proactive)
        alternatives=list(self.planner.fallback_candidates(obs,target));checks=self.planner.check_many(obs,alternatives,self.steering,predictions) if alternatives else []
        options=[]
        for c,test in zip(alternatives,checks):
            self.interaction_debug['checked']+=1
            self.planner._record_path(c.path,c.action,test['safe'],'wall corridor clear' if test['safe'] else 'wall corridor forecast conflict',source='corridor',cost=c.cost,**c.path_parameters)
            if test['safe']:
                c.clearance=min(test['gap'],test['wall']);c.interaction_certificate=test
                options.append((test['score']-.08*abs(c.offset-d),c,test))
        if options:
            score,replacement,test=max(options,key=lambda p:p[0])
            # Keep a clear existing trajectory unless the wider route offers
            # useful extra progress. This limits needless lane switching.
            if checked['safe'] and chosen.speed>.2 and score<checked['score']+.10:return index
            index={'LEFT':0,'RIGHT':1,'WAIT':2}[replacement.action];candidates[index]=replacement
            self.interaction_debug.update(override=True,mode='checked wall corridor',corridor_override=True,selected_gap=test['gap'],rollout=test['path'].tolist(),selected_lateral_goal_m=replacement.offset)
            if replacement.action=='WAIT' and rel>1.:self.commit=None
        elif checked['safe']:return index
        else:
            steer,_=self.planner.tracker.control(obs,offset=d,vmax=self.vmax,smooth=False);path=g.sample(s+np.linspace(0.,3.,31),d)[0]
            candidates[2]=Candidate('WAIT',True,30.,checked['gap'],steer,0.,d,path);index=2
            self.interaction_debug.update(override=True,mode='brake in current lane')
        self.planner.response_choice=None;self.planner.response_debug.update(mode='wall corridor fallback',selected=False)
        self.planner._finish_paths(candidates)
        return index
    def select_response_maneuver(self,candidates,index,obs,target):
        # Establish broad maneuvers before braking into a bend. Once established,
        # keep that planning mode through the encounter. A car already deep in
        # a slow bend keeps the proven local recovery unless its motion blocks.
        if target is None:
            self.wide_encounter=False
        else:
            s,_=self.planner.track.project(obs['pose_x'],obs['pose_y'])
            rel=float(delta_s(s,target.s,self.planner.track.length))
            if rel< -1.5 or rel>4.:self.wide_encounter=False
            if .3<rel<2.5 and obs['linear_vel_x']>=.75*self.vmax:self.wide_encounter=True
        if getattr(self,'wide_encounter',False):
            return self._early_wide_select(candidates,index,obs,target)
        return self._blocked_wide_select(candidates,index,obs,target)
    def _early_wide_select(self,candidates,index,obs,target):
        index=JointStage4.select_response_maneuver(self,candidates,index,obs,target)
        self.interaction_debug=dict(active=False,override=False,checked=0,mode='baseline')
        self.slow_seconds=self.slow_seconds+self.planner.control_dt if obs['linear_vel_x']<.2 else 0.
        if target is None or target.confidence<.6 or target.time_since_seen>.15:return index
        g=self.planner.track;s,d=map(float,g.project(obs['pose_x'],obs['pose_y']));rel=float(delta_s(s,target.s,g.length))
        if not -1.5<rel<2.5:return index
        predictions=self.planner.interaction.trajectories(target,np.arange(25)*.05)
        chosen=candidates[index];checked=self.planner.check(obs,chosen,self.steering,predictions)
        proactive=self.counter%6==0 and -.5<rel<2.5
        self.interaction_debug.update(active=True,checked=1,physical_opponent_speed=self.planner.interaction.physical_speed(target),tracked_reference_speed=target.speed,
            selected_gap=checked['gap'],predictions=[dict(name=p['name'],path=p['path'].tolist()) for p in predictions],proactive_search=proactive)
        if checked['safe'] and not (proactive or self.slow_seconds>.5 or chosen.speed<.2):return index
        alternatives=list(self.planner.fallback_candidates(obs,target));checks=self.planner.check_many(obs,alternatives,self.steering,predictions) if alternatives else []
        options=[]
        for c,test in zip(alternatives,checks):
            self.interaction_debug['checked']+=1
            self.planner._record_path(c.path,c.action,test['safe'],'wall corridor clear' if test['safe'] else 'wall corridor forecast conflict',source='corridor',cost=c.cost,**c.path_parameters)
            if test['safe']:
                c.clearance=min(test['gap'],test['wall']);c.interaction_certificate=test
                options.append((test['score']-.08*abs(c.offset-d),c,test))
        if options:
            score,replacement,test=max(options,key=lambda p:p[0])
            # Keep a clear existing trajectory unless the wider route offers
            # useful extra progress. This limits needless lane switching.
            if checked['safe'] and chosen.speed>.2 and score<checked['score']+.10:return index
            index={'LEFT':0,'RIGHT':1,'WAIT':2}[replacement.action];candidates[index]=replacement
            self.interaction_debug.update(override=True,mode='checked wall corridor',selected_gap=test['gap'],rollout=test['path'].tolist(),selected_lateral_goal_m=replacement.offset)
            if replacement.action=='WAIT' and rel>1.:self.commit=None
        elif checked['safe']:return index
        else:
            steer,_=self.planner.tracker.control(obs,offset=d,vmax=self.vmax,smooth=False);path=g.sample(s+np.linspace(0.,3.,31),d)[0]
            candidates[2]=Candidate('WAIT',True,30.,checked['gap'],steer,0.,d,path);index=2
            self.interaction_debug.update(override=True,mode='brake in current lane')
        self.planner.response_choice=None;self.planner.response_debug.update(mode='wall corridor fallback',selected=False)
        self.planner._finish_paths(candidates)
        return index
    def plan(self,*a,**kw):
        self.planner.corridor_debug={};result=super().plan(*a,**kw)
        self.telemetry['interaction']['corridor']=self.planner.corridor_debug
        return result
