"""Experimental response-aware Stage 4, for Gym only.

Predicts a bounded brake/outward response to a visible inside challenge. This is
an explicit behavioral hypothesis, not the opponent's future state or a learned
probability. Frozen neural Stage 4 remains the fallback. No opponent commands,
truth positions, or future observations enter this policy.
"""
from dataclasses import replace
import numpy as np
from expanded_planner import ExpandedPlanner,ExpandedStage4
from maneuver_core import Candidate,delta_s,smoothstep
from track_geometry import MAX_STEER,WHEELBASE


def separation(xy,yaw,other,oyaw,width=.31,length=.58):
    forward=np.column_stack((np.cos(yaw),np.sin(yaw)))
    normal=np.column_stack((-np.sin(yaw),np.cos(yaw)))
    of=np.column_stack((np.cos(oyaw),np.sin(oyaw)))
    on=np.column_stack((-np.sin(oyaw),np.cos(oyaw)))
    axes=np.stack((forward,normal,of,on),axis=1)
    gap=np.abs(np.einsum('nij,nj->ni',axes,other-xy))
    for direction,half in ((forward,.29),(normal,.155),(of,length/2),(on,width/2)):
        gap-=half*np.abs(np.einsum('nij,nj->ni',axes,direction))
    return np.max(gap,axis=1)


class ResponsePlanner(ExpandedPlanner):
    response_margin=.035
    response_enabled=True
    def candidates(self,obs,target,committed=None):
        result=super().candidates(obs,target,committed)
        self.response_choice=None;self.response_debug={'considered':0,'accepted':0,'mode':'baseline'}
        if not self.response_enabled:return result
        if target is None or target.confidence<.6 or target.time_since_seen>.10 or target.speed<.5:return result
        g=self.track;s,d=map(float,g.project(float(obs['pose_x']),float(obs['pose_y'])))
        rel=float(delta_s(s,target.s,g.length))
        bends=g.sample(s+np.linspace(.5,4.,8))[2]
        bend=float(bends[np.argmax(np.abs(bends))])
        if abs(bend)<.18 or not -.55<rel<.25 or abs(d-target.d)<.25 or target.observations*self.control_dt<.25:return result
        side=1 if bend>0 else -1
        if (d-target.d)*side<.25:return result
        action='LEFT' if side==1 else 'RIGHT';index=0 if side==1 else 1
        # Do not reverse an already-established maneuver across the rival.
        if committed and committed[0]!=action:return result
        ahead=np.linspace(0.,6.,61);ss=s+ahead
        _,refyaw,curv=g.sample(s)
        error=float(np.arctan2(np.sin(obs['pose_theta']-refyaw),np.cos(obs['pose_theta']-refyaw)))
        slope=float(np.clip(np.tan(error)*(1-float(curv)*d),-.8,.8))
        options=[]
        for entry in (1.8,):
            for offset,merge_at in ((abs(d-target.d),0.),(abs(d-target.d),.4),(abs(d-target.d),.8)):
                desired=float(target.d+side*offset)
                def offsets(x):
                    distance=delta_s(s,x,g.length);u=np.clip(distance/entry,0.,1.)
                    left,right=g.widths(x)
                    lane=np.clip(desired,-np.asarray(right)+.24+self.wall_margin,np.asarray(left)-.24-self.wall_margin)
                    departure=d+(lane-d)*smoothstep(u)+slope*entry*(u-6*u**3+8*u**4-3*u**5)
                    return departure*(1-smoothstep((distance-merge_at)/1.8))
                xy,_,_=g.sample(ss,offsets(ss));tangent=np.gradient(xy,axis=0);yaw=np.arctan2(tangent[:,1],tangent[:,0])
                self.response_debug['considered']+=1;self.search_stats['evaluated']+=1
                details=dict(source='response',entry_m=entry,inside_offset_m=offset,merge_at_m=merge_at)
                wall=float(min(g.body_clearance(xy,yaw).min(),g.body_clearance(xy[0],float(obs['pose_theta']))))
                curve=self.path_curvature(xy)
                if wall<self.wall_margin or curve.max()>.95*np.tan(MAX_STEER)/WHEELBASE:
                    self._record_path(xy,action,False,'response path wall/curvature',**details);continue
                self.tracker.lateral_acceleration=self.passing_lateral_acceleration
                steer,speed=self.tracker.control(obs,offset=offsets,vmax=self.vmax,smooth=False)
                arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
                limits=np.minimum(self.vmax,np.sqrt(self.passing_lateral_acceleration/np.maximum(curve,.05)))
                speed=min(speed,float(np.sqrt(limits**2+6*arc).min()))
                if speed<.6:
                    self._record_path(xy,action,False,'too slow',**details);continue
                travel=arc/max(.6,min(speed,max(.6,float(obs['linear_vel_x']))))
                other,oyaw,_=g.sample(target.s+target.speed*travel,target.d+target.lateral_speed*.35*(1-np.exp(-travel/.35)))
                gaps=separation(xy,yaw,other,oyaw,target.width,target.length)
                conflict=np.where(gaps<0.)[0]
                # A challenge is specifically a predicted physical conflict if
                # the opponent keeps its observed motion. Existing-gap passes
                # remain the responsibility of the baseline planner.
                if not len(conflict):
                    self._record_path(xy,action,False,'no predicted physical conflict',**details);continue
                relative=xy-other
                of=np.column_stack((np.cos(oyaw),np.sin(oyaw)))
                on=np.column_stack((-np.sin(oyaw),np.cos(oyaw)))
                longitudinal=np.sum(relative*of,axis=1);lateral=np.sum(relative*on,axis=1)
                ef=np.column_stack((np.cos(yaw),np.sin(yaw)))
                en=np.column_stack((-np.sin(yaw),np.cos(yaw)))
                vertices=xy[:,None,:]+np.array([.29,.29,-.29,-.29])[None,:,None]*ef[:,None,:]+np.array([.155,-.155,-.155,.155])[None,:,None]*en[:,None,:]
                rv=vertices-other[:,None,:]
                forward_vertex=np.einsum('nvi,ni->nv',rv,of)
                lateral_vertex=np.einsum('nvi,ni->nv',rv,on)
                visible=(np.abs(np.arctan2(lateral_vertex,forward_vertex))<2.35).any(axis=1)&(np.abs(longitudinal)<1.2)&(np.abs(lateral)<.8)
                trigger=np.where(visible & (travel<travel[conflict[0]]))[0]
                if not len(trigger):
                    self._record_path(xy,action,False,'opponent cannot see challenge early enough',**details);continue
                trigger_time=float(travel[trigger[0]])
                predicted=[];minimum=100.;wall_other=100.
                for delay in (.10,.25):
                    elapsed=np.maximum(0.,travel-trigger_time-delay)
                    # Bounded Stage-2-like retreat: braking and an outside move.
                    # Vary reaction delay; neither response is commanded to the rival.
                    braking=2.5;stop_time=max(0.,target.speed-.7)/braking
                    active=np.minimum(elapsed,stop_time)
                    os=target.s+target.speed*travel-.5*braking*active**2-(target.speed-.7)*np.maximum(0.,elapsed-stop_time)
                    left,right=g.widths(os)
                    desired_other=desired-side*(.155+target.width/2+.20)
                    od_goal=np.clip(desired_other,-np.asarray(right)+.24+.07,np.asarray(left)-.24-.07)
                    od=target.d+(od_goal-target.d)*smoothstep(elapsed/1.1)
                    o,_,_=g.sample(os,od);ot=np.gradient(o,axis=0);angle=np.arctan2(ot[:,1],ot[:,0])
                    stopped=np.linalg.norm(ot,axis=1)<1e-6
                    angle[stopped]=g.sample(os[stopped])[1]
                    min_gap=float(separation(xy,yaw,o,angle,target.width,target.length).min())
                    opponent_curve=self.path_curvature(o)
                    opponent_speed=np.linalg.norm(np.gradient(o,axis=0),axis=1)/np.maximum(1e-5,np.gradient(travel))
                    physical=(opponent_curve.max()<=np.tan(MAX_STEER)/WHEELBASE and np.max(opponent_speed**2*opponent_curve)<=2.6)
                    if not physical:min_gap=-100.
                    minimum=min(minimum,min_gap);wall_other=min(wall_other,float(g.body_clearance(o,angle).min()))
                    predicted.append(dict(delay=delay,path=o.tolist(),times=travel.tolist(),minimum_gap=min_gap))
                if minimum<self.response_margin or wall_other<.07:
                    self._record_path(xy,action,False,'predicted yielding insufficient',response_gap=minimum,opponent_wall=wall_other,**details);continue
                cost=float(np.sum(np.diff(arc)/np.maximum(.4,limits[:-1])))+.15*abs(desired)+.1/max(.1,wall)
                candidate=Candidate(action,True,cost,min(minimum,wall),float(steer),float(speed),desired,xy)
                candidate.path_parameters=dict(entry_m=entry,padding_m=offset-.155-target.width/2,merge_factor=1.)
                candidate.response=dict(mode='conditional inside challenge',assumes_yield=True,
                    no_response_gap=float(gaps.min()),no_response_contact_time=float(travel[conflict[0]]),
                    no_response_path=other.tolist(),predictions=predicted,minimum_response_gap=minimum,
                    trigger_time=trigger_time,entry_m=entry,inside_offset_m=offset,merge_at_m=merge_at)
                self.response_debug['accepted']+=1
                self._record_path(xy,action,True,'conditional on opponent yielding',cost=cost,**details)
                options.append(candidate)
        if options:
            chosen=min(options,key=lambda c:c.cost+.15/max(.02,c.clearance))
            self.response_choice=chosen
            self.response_debug.update(chosen.response)
            result[index]=chosen
        return self._finish_paths(result)


class ResponseStage4(ExpandedStage4):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.planner=ResponsePlanner(self.planner.track,self.vmax,self.stage)
        self.response_selections=0
        self.response_time=0.;self.response_cooldown=0.;self.pending_response=None;self.response_aborts=0

    def select_response_maneuver(self,candidates,index,obs,target):
        if self.pending_response is not None and self.response_time>=self.pending_response['due']:
            if target is None:error=1.
            else:
                observed=self.planner.track.sample(target.s,target.d)[0]
                error=float(np.linalg.norm(observed-self.pending_response['position']))
            self.pending_response=None
            if error>.25:
                self.response_cooldown=self.response_time+2.;self.response_aborts+=1
                self.planner.response_choice=None
                self.planner.response_debug.update(mode='abort: response prediction mismatch',forecast_error_m=error)
                return 2
        choice=self.planner.response_choice
        if choice is not None:
            self.response_selections+=1
            if self.pending_response is None:
                forecast=choice.response['predictions'][-1]
                horizon=min(forecast['times'][-1],choice.response['trigger_time']+.6)
                points=np.asarray(forecast['path'])
                position=np.array([np.interp(horizon,forecast['times'],points[:,j]) for j in (0,1)])
                self.pending_response=dict(due=self.response_time+horizon,position=position)
            return 0 if choice.action=='LEFT' else 1
        return index

    def response_speed_caps(self,caps,tracked,selected,obs,target):
        # Simulator-only conditional risk: release only observed rival returns,
        # never walls/unknown objects, and stop betting near predicted contact.
        choice=self.planner.response_choice
        if choice is not None and selected is choice and choice.response['no_response_contact_time']>.4:
            caps=caps.copy();caps[tracked]=self.vmax
            self.planner.response_debug['released_rival_beams']=int(tracked.sum())
        return caps

    def plan(self,*args,**kwargs):
        self.response_time+=float(kwargs.get('dt',.01))
        self.planner.response_enabled=self.response_time>=self.response_cooldown
        command=super().plan(*args,**kwargs)
        self.telemetry['response']=self.planner.response_debug
        self.telemetry['response']['selected']=bool(self.planner.response_choice is not None and self.action==self.planner.response_choice.action)
        self.telemetry['response']['selection_count']=self.response_selections
        self.telemetry['response']['abort_count']=self.response_aborts
        return command
