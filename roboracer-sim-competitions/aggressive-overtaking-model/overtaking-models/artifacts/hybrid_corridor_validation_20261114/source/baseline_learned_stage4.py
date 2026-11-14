"""Stage 4 inside pressure ranked by a learned reactive-opponent forecast."""
import numpy as np
from expanded_planner import ExpandedPlanner
from response_stage4 import ResponseStage4,separation
from learned_opponent import OpponentForecast
from maneuver_core import Candidate,delta_s,smoothstep
from track_geometry import MAX_STEER,WHEELBASE,PathTracker

class LearnedPlanner(ExpandedPlanner):
    response_enabled=True
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.forecast=OpponentForecast()
    def ego_rollout(self,obs,offsets):
        tracker=PathTracker(self.track,lateral_acceleration=self.passing_lateral_acceleration)
        state={k:float(obs[k]) for k in ('pose_x','pose_y','pose_theta','linear_vel_x')}
        steer=float(getattr(self,'initial_steer',0.));points=[];headings=[]
        for i in range(31):
            points.append([state['pose_x'],state['pose_y']]);headings.append(state['pose_theta'])
            requested,speed=tracker.control(state,offset=offsets,vmax=self.vmax,smooth=False)
            steer+=float(np.clip(requested-steer,-.125,.125))
            v=float(np.clip(speed,max(0.,state['linear_vel_x']-.15),state['linear_vel_x']+.10))
            beta=np.arctan(.5193*np.tan(steer));yaw=state['pose_theta']
            state['pose_x']+=.05*v*np.cos(yaw+beta);state['pose_y']+=.05*v*np.sin(yaw+beta)
            state['pose_theta']+=.05*v*np.cos(beta)*np.tan(steer)/WHEELBASE;state['linear_vel_x']=v
        return np.asarray(points),np.asarray(headings),np.arange(31)*.05

    def candidates(self,obs,target,committed=None):
        result=super().candidates(obs,target,committed)
        self.response_choice=None;self.response_debug=dict(considered=0,accepted=0,mode='learned forecast / baseline')
        if not self.response_enabled or target is None or target.confidence<.6 or target.time_since_seen>.10 or target.speed<.3:return result
        g=self.track;s,d=map(float,g.project(float(obs['pose_x']),float(obs['pose_y'])))
        rel=float(delta_s(s,target.s,g.length))
        if not -1.<rel<4. or target.observations*self.control_dt<.25:return result
        bends=g.sample(s+np.linspace(0.,3.,7))[2];bend=float(bends[np.argmax(np.abs(bends))])
        if abs(bend)<.15:return result
        side=1 if bend>0 else -1;action='LEFT' if side>0 else 'RIGHT';index=0 if side>0 else 1
        if committed and committed[0]!=action:return result
        ahead=np.linspace(0.,6.,61);ss=s+ahead
        _,refyaw,curv=g.sample(s)
        error=float(np.arctan2(np.sin(obs['pose_theta']-refyaw),np.cos(obs['pose_theta']-refyaw)))
        slope=float(np.clip(np.tan(error)*(1-float(curv)*d),-.8,.8))
        parameters=[(entry,pad,99.) for entry in (1.2,1.8,2.6) for pad in (.06,.10)]
        if rel<.5 and (d-target.d)*side>.25:
            parameters += [(1.8,max(.03,abs(d-target.d)-.31),merge) for merge in (0.,.4,.8)]
        choices=[]
        for entry,padding,merge in parameters:
            desired=float(target.d+side*(.155+target.width/2+padding))
            def offsets(x):
                distance=delta_s(s,x,g.length);u=np.clip(distance/entry,0.,1.)
                left,right=g.widths(x);lane=np.clip(desired,-np.asarray(right)+.24+self.wall_margin,np.asarray(left)-.24-self.wall_margin)
                value=d+(lane-d)*smoothstep(u)+slope*entry*(u-6*u**3+8*u**4-3*u**5)
                return value*(1-smoothstep((distance-merge)/2.))
            xy,_,_=g.sample(ss,offsets(ss));v=np.gradient(xy,axis=0);yaw=np.arctan2(v[:,1],v[:,0])
            self.search_stats['evaluated']+=1;self.response_debug['considered']+=1
            info=dict(source='learned_response',entry_m=entry,padding_m=padding,merge_at_m=merge)
            wall=float(min(g.body_clearance(xy,yaw).min(),g.body_clearance(xy[0],float(obs['pose_theta']))))
            curve=self.path_curvature(xy)
            if wall<self.wall_margin or curve.max()>.95*np.tan(MAX_STEER)/WHEELBASE:
                self._record_path(xy,action,False,'wall or steering curvature',**info);continue
            self.tracker.lateral_acceleration=self.passing_lateral_acceleration
            steer,speed=self.tracker.control(obs,offset=offsets,vmax=self.vmax,smooth=False)
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
            limits=np.minimum(self.vmax,np.sqrt(self.passing_lateral_acceleration/np.maximum(curve,.05)))
            speed=min(speed,float(np.sqrt(limits**2+6*arc).min()))
            if speed<.5:continue
            times=arc/max(.6,min(speed,max(.6,float(obs['linear_vel_x']))))
            ego,egoyaw,tt=self.ego_rollout(obs,offsets);horizon=float(tt[-1])
            if g.body_clearance(ego,egoyaw).min()<self.wall_margin:continue
            predicted,oyaw,spread,od=self.forecast.trajectory(g,target,obs,ego,tt,tt)
            unchanged,uyaw,_=g.sample(target.s+target.speed*tt,target.d+target.lateral_speed*.35*(1-np.exp(-tt/.35)))
            learned_gap=separation(ego,egoyaw,predicted,oyaw,target.width,target.length)
            constant_gap=separation(ego,egoyaw,unchanged,uyaw,target.width,target.length)
            # Uncertainty is reported and penalized; this is not a calibrated tube.
            uncertainty=float(np.max(spread[:,1]));clear=float(learned_gap.min())
            immediate=float(constant_gap[tt<=.6].min())
            other_wall=float(g.body_clearance(predicted,oyaw).min())
            if clear<.06 or immediate<.03 or other_wall<.07 or uncertainty>.14:
                self._record_path(xy,action,False,'learned clearance or uncertainty',**info);continue
            # Compare proposals under the SAME reactive-response predictor.
            # This is a planning comparison, never a nonreactive rival experiment.
            reference=result[2].path
            refarc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(reference,axis=0),axis=1))]
            reftimes=refarc/max(.25,min(max(.25,float(obs['linear_vel_x'])),result[2].speed))
            _,_,_,refd=self.forecast.trajectory(g,target,obs,reference,reftimes,tt)
            outward=float(np.max(-side*(od-refd)))
            if outward<max(.015,uncertainty):continue
            noresponse_min=float(constant_gap.min());conflict=np.where(constant_gap<0.)[0]
            ttc=float(tt[conflict[0]]) if len(conflict) else 99.
            # A pressured approach must be physically clear even without yielding;
            # an actual conditional cut needs a positive predicted displacement.
            conditional=noresponse_min<.03
            if conditional and (outward<.05 or ttc<.6):continue
            cost=float(np.sum(np.diff(arc)/np.maximum(.4,limits[:-1])))+.15*abs(desired)+.1/max(.1,wall)
            score=cost-.8*outward+.8*uncertainty
            # Keep a viable lower-cost nominal route unless a response benefit
            # or a stalled original maneuver justifies the alternative.
            existing=result[index]
            if existing.feasible and existing.speed>.5 and cost>existing.cost+.35 and outward<.06:continue
            candidate=Candidate(action,True,cost,min(clear,wall),float(steer),float(speed),desired,xy)
            candidate.path_parameters=dict(entry_m=entry,padding_m=padding,merge_factor=1.)
            candidate.response=dict(mode='learned inside pressure',assumes_yield=conditional,learned_forecast=True,
                no_response_gap=noresponse_min,no_response_contact_time=ttc,no_response_path=unchanged.tolist(),
                predictions=[dict(delay=0.,path=predicted.tolist(),times=tt.tolist(),minimum_gap=clear)],
                minimum_response_gap=clear,trigger_time=0.,entry_m=entry,inside_offset_m=abs(desired-target.d),
                merge_at_m=merge,predicted_outward_m=outward,ensemble_lateral_spread_m=uncertainty,
                ego_prediction_path=ego.tolist(),forecast_horizon_s=horizon,forecast_name='reactive_stage2_conditional_mlp')
            self._record_path(xy,action,True,'learned response clearance',cost=cost,**info)
            self.response_debug['accepted']+=1;choices.append((score,candidate))
        if choices:
            _,chosen=min(choices,key=lambda x:x[0]);self.response_choice=chosen
            self.response_debug.update(chosen.response);result[index]=chosen
        return self._finish_paths(result)

class LearnedStage4(ResponseStage4):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw);self.planner=LearnedPlanner(self.planner.track,self.vmax,self.stage)
    def plan(self,*a,**kw):
        self.planner.initial_steer=self.steering
        return super().plan(*a,**kw)
    def response_speed_caps(self,caps,tracked,selected,obs,target):
        choice=self.planner.response_choice
        if choice is not None and selected is choice and choice.response['assumes_yield'] and choice.response['no_response_contact_time']>.7:
            caps=caps.copy();caps[tracked]=self.vmax
            self.planner.response_debug['released_rival_beams']=int(tracked.sum())
        return caps
