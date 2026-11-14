"""Wider planner with certified escape from a padding-only standstill."""
from dataclasses import replace
import numpy as np
from robust_corridor_stage4 import RobustCorridorPlanner,RobustCorridorStage4
from interaction_stage4 import InteractionPlanner
from response_stage4 import separation
from maneuver_core import delta_s
from track_geometry import WHEELBASE

class RecoveryCorridorPlanner(RobustCorridorPlanner):
    def planning_profile(self):
        return {**super().planning_profile(),'padding_escape':'fresh tracked side returns only; executed only with body rollout certificate',
                'depleted_gap_escape':'positive initial separation must not decrease and must restore 0.08m by horizon'}
    def candidates(self,*a,**kw):
        self._decision_tick=getattr(self,'_decision_tick',0)+1
        return super().candidates(*a,**kw)
    def stopped_key(self,obs,target):
        if (target is None or obs['linear_vel_x']>.02 or self.interaction.physical_speed(target)>.05
            or target.confidence<.6 or target.time_since_seen>.1):return None
        return tuple(round(float(v),2) for v in (obs['pose_x'],obs['pose_y'],obs['pose_theta'],target.s,target.d,getattr(self,'initial_steer',0.)))
    def fallback_candidates(self,obs,target):
        key=self.stopped_key(obs,target);cached=getattr(self,'_failed_search',None)
        if key is not None and cached is not None and cached[0]==key and getattr(self,'_decision_tick',0)<cached[1]:
            self.corridor_debug=dict(failed_search_reused=True,command='stop',maximum_reuse_seconds=.25)
            return
        self._padding_proposal=True;seen=set()
        try:
            for c in super().fallback_candidates(obs,target):
                # Clamped speed variants can become the exact same motion.
                identity=(c.steering,c.speed,c.path.tobytes())
                if identity in seen:continue
                seen.add(identity);yield c
        finally:self._padding_proposal=False
    def lidar_caps(self,obs,target,candidates):
        normal=super().lidar_caps(obs,target,candidates)
        if (not getattr(self,'_padding_proposal',False) or not candidates or obs['linear_vel_x']>.8 or target is None
            or target.confidence<=.6 or target.time_since_seen>=.1):return normal
        ranges=np.nan_to_num(np.asarray(obs['scan'],float),nan=0.,posinf=20.,neginf=0.)
        x,y=ranges*np.cos(self.angles),ranges*np.sin(self.angles)
        steering,indices,inverse=np.unique([c.steering for c in candidates],return_index=True,return_inverse=True)
        curvature=np.tan(steering)/WHEELBASE;radius=1/np.where(np.abs(curvature)<1e-3,1.,curvature)
        lateral=np.abs(np.hypot(x[None,:],y[None,:]-radius[:,None])-np.abs(radius[:,None]))
        along=np.abs(radius[:,None])*np.arctan2(x[None,:],np.sign(radius[:,None])*(radius[:,None]-y[None,:]))
        straight=np.abs(curvature)<1e-3;lateral[straight]=np.abs(y);along[straight]=x
        txy,tyaw,_=self.track.sample(target.s,target.d);yaw=float(obs['pose_theta']);co,sn=np.cos(yaw),np.sin(yaw)
        hits=np.column_stack([obs['pose_x']+co*x-sn*y,obs['pose_y']+sn*x+co*y]);relative=hits-txy
        longitudinal=relative@np.array([np.cos(tyaw),np.sin(tyaw)]);transverse=relative@np.array([-np.sin(tyaw),np.cos(tyaw)])
        tracked=(np.abs(longitudinal)<target.length/2+.12)&(np.abs(transverse)<target.width/2+.08)&(self.track.clearance(hits)>.12)
        forward=(x[None,:]>0)&(along>0)&(along<2.)&(lateral<.195)
        # This proposes a command only. Selection still needs a complete body
        # certificate; the final filter independently recognizes the same beams.
        padding=tracked[None,:]&(x[None,:]<.29)&(lateral>=.16)
        obstacle=np.where(tracked,max(0.,target.speed-.3-3*target.time_since_seen)*max(0.,float(np.cos(tyaw-yaw))),0.)
        caps=np.min(np.where(forward&~padding,np.sqrt(obstacle[None,:]**2+6*np.maximum(0.,along-.40)),np.inf),axis=1)[inverse]
        for c,strict,relaxed in zip(candidates,normal,caps):
            if relaxed>strict:
                c.padding_escape_proposal=True
                if self.interaction.physical_speed(target)<.1:c.speed=min(c.speed,.25)
        return caps
    def measured_escape(self,obs,candidate,steer):
        # At a standstill the measured obstacle cloud provides an additional
        # check on face-fit/yaw error. All nearby returns, including walls,
        # must retain positive, nondecreasing clearance during the slow move.
        ranges=np.asarray(obs['scan']);valid=np.isfinite(ranges)&(ranges>.05)&(ranges<2.)
        angles=self.angles[valid]+obs['pose_theta']
        points=np.column_stack([obs['pose_x']+ranges[valid]*np.cos(angles),obs['pose_y']+ranges[valid]*np.sin(angles)])
        if not len(points):return False
        xy,yaw,_,_=self.rollout(obs,candidate,steer)
        diff=points[None,:,:]-xy[:,None,:]
        longitudinal=diff[:,:,0]*np.cos(yaw[:,None])+diff[:,:,1]*np.sin(yaw[:,None])
        lateral=-diff[:,:,0]*np.sin(yaw[:,None])+diff[:,:,1]*np.cos(yaw[:,None])
        clearance=np.maximum(np.abs(longitudinal)-.29,np.abs(lateral)-.155).min(axis=1)
        return bool(clearance.min()>=max(.005,clearance[0]-1e-4))
    def restore_depleted(self,obs,c,r,steer,predictions):
        initial=r['initial_gap']
        if (not r['safe'] and .005<initial<.025 and r['gap']>=initial-1e-6 and r['wall']>=.025 and c.speed>0.):
            xy,yaw,_,_=self.rollout(obs,c,steer)
            end=min(float(separation(xy[-1:],yaw[-1:],p['path'][-1:],p['yaw'][-1:],self.target.width,self.target.length)[0]) for p in predictions)
            if end>=.08:r.update(safe=True,recovery=True,nonworsening_depleted_escape=True)
        return r
    def check_many(self,obs,candidates,initial_steer,predictions):
        results=super().check_many(obs,candidates,initial_steer,predictions)
        stationary=self.target is not None and self.interaction.physical_speed(self.target)<.1
        for c,r in zip(candidates,results):
            self.restore_depleted(obs,c,r,initial_steer,predictions)
            if getattr(c,'padding_escape_proposal',False):
                if r['gap']<r['initial_gap']-1e-6:r['safe']=False
                if r['safe'] and stationary:
                    r['safe']=self.measured_escape(obs,c,initial_steer)
                    c.measured_padding_escape=bool(r['safe'])
            if (not r['safe'] and stationary and obs['linear_vel_x']<.05 and getattr(c,'path_parameters',{}).get('local_arc',False)
                and .05<c.speed<=.25):
                # Steer at rest before accelerating; turning the wheel need not
                # move the body. Never assume an instantaneous steering change
                # for a vehicle that is already moving.
                test=InteractionPlanner.check_many(self,obs,[c],c.steering,predictions)[0]
                self.restore_depleted(obs,c,test,c.steering,predictions)
                if test['safe'] and test['gap']>=test['initial_gap']-1e-6 and self.measured_escape(obs,c,c.steering):
                    r.update(test);c.presteer_escape=True;c.measured_padding_escape=True
        if len(candidates)>4 and not getattr(self,'legacy',False) and not any(r['safe'] and c.speed>.05 for c,r in zip(candidates,results)):
            key=self.stopped_key(obs,self.target)
            if key is not None:self._failed_search=(key,getattr(self,'_decision_tick',0)+5)
        return results

class RecoveryCorridorStage4(RobustCorridorStage4):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        geometry=self.planner.track.geometry
        self.planner=RecoveryCorridorPlanner(geometry,self.vmax,self.stage);self.planner.angles=self.angles
    def select_response_maneuver(self,candidates,index,obs,target):
        index=super().select_response_maneuver(candidates,index,obs,target)
        return self.prepare_command(candidates,index)
    def prepare_command(self,candidates,index):
        chosen=candidates[index]
        if getattr(chosen,'presteer_escape',False) and abs(self.steering-chosen.steering)>.04:
            chosen.speed=0.
            self.interaction_debug['presteer_at_rest']=True
        return index
    def response_speed_caps(self,caps,tracked,selected,obs,target):
        caps=super().response_speed_caps(caps,tracked,selected,obs,target)
        if (not getattr(selected,'measured_padding_escape',False) or target is None
            or target.time_since_seen>=.1 or target.confidence<=.6):return caps
        ranges=np.asarray(obs['scan']);x,y=ranges*np.cos(self.angles),ranges*np.sin(self.angles);curv=np.tan(selected.steering)/WHEELBASE
        if abs(curv)<1e-3:lateral,along=np.abs(y),x
        else:
            radius=1/curv;lateral=np.abs(np.hypot(x,y-radius)-abs(radius));along=abs(radius)*np.arctan2(x,np.sign(radius)*(radius-y))
        forward=(x>0)&(along>0)&(along<2)&(lateral<.195)
        padding=tracked&(x[forward]<.29)&(lateral[forward]>=.16)
        caps=caps.copy();caps[padding]=.25
        self.interaction_debug['measured_padding_beams_released']=int(padding.sum())
        return caps
