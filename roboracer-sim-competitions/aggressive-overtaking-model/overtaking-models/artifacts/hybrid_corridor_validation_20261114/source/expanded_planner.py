"""Extra ego path candidates; frozen networks and opponent remain unchanged.

One representative per maneuver preserves the 50-feature neural interface.
The old candidates remain in the set. All new paths use the existing body,
wall, motion-clearance, and departure-certificate checks.
"""
import numpy as np
from dataclasses import replace
from maneuver_core import CandidatePlanner, Candidate, delta_s, smoothstep
from moving_model import MovingModel
from interactive_model import InteractiveModel
from track_geometry import MAX_STEER, WHEELBASE

class ExpandedPlanner(CandidatePlanner):
    entries=(1.1,1.7,2.5,3.5)
    capture_paths=False

    def _record_path(self, path, action, feasible, reason, **details):
        # Observation only: never changes scoring, features, or controls.
        if self.capture_paths:
            self.debug_paths.append(dict(path=np.asarray(path).tolist(), action=action,
                                         feasible=bool(feasible), reason=reason, **details))

    def _finish_paths(self, candidates):
        if self.capture_paths:
            self.debug_representatives=[dict(path=c.path.tolist(),parameters=getattr(c,'path_parameters',None), **c.record()) for c in candidates]
        return candidates

    @staticmethod
    def path_curvature(xy):
        segments=np.linalg.norm(np.diff(xy,axis=0),axis=1)
        if np.min(segments)<1e-5:return np.full(len(xy),np.inf)
        tangent=np.gradient(xy,axis=0)
        yaw=np.unwrap(np.arctan2(tangent[:,1],tangent[:,0]))
        return np.abs(np.gradient(yaw,np.r_[0.,np.cumsum(segments)]))

    def planning_profile(self):
        return {**super().planning_profile(),'expanded_paths':True,
                'entry_distances_m':list(self.entries),'merge_distance_factors':[.8,1.,1.25],
                'candidate_reduction':'retain a viable nominal path; search alternatives for blocked or stalled maneuvers',
                'network_retrained':False}

    def candidates(self,obs,target,committed=None):
        self.debug_paths=[]
        original=super().candidates(obs,target,committed)
        if self.stage==3:
            # Even a nominal path can become untrackable after a new entry.
            # Use distance along the DETOUR, not along the reference line:
            # inside offsets shorten arcs and increase their true curvature.
            original=[replace(c,feasible=False,cost=30.)
                if c.action!='WAIT' and c.feasible and
                self.path_curvature(c.path).max()>.95*np.tan(MAX_STEER)/WHEELBASE else c
                for c in original]
        for c in original:
            self._record_path(c.path,c.action,c.feasible,'nominal feasible' if c.feasible else 'nominal safety or maneuver gate',cost=float(c.cost),source='nominal')
        self.search_stats={'evaluated':len(original),'additional_feasible':0,'replaced':[]}
        if self.stage<2 or target is None or target.speed<.4 or target.observations*self.control_dt<.18:
            return self._finish_paths(original)
        g=self.track
        s,d=map(float,g.project(float(obs['pose_x']),float(obs['pose_y'])))
        rel=float(delta_s(s,target.s,g.length))
        if not -2.2<rel<7.:return self._finish_paths(original)
        _,heading,bends=g.sample(s+np.array([0.,1.,2.,3.,4.,5.]))
        corner=np.max(np.abs(bends[1:]))>=.18
        if self.stage==3 and abs(d)<.16 and not committed and not corner and self.vmax-target.speed<.25:
            return self._finish_paths(original)
        yaw=float(obs['pose_theta'])
        error=np.arctan2(np.sin(yaw-heading[0]),np.cos(yaw-heading[0]))
        slope=float(np.clip(np.tan(error)*(1-bends[0]*d),-.8,.8))
        distances=self.distances
        path_s=s+distances
        closing=max(.05,self.vmax-target.speed)
        merge=max(0.,(rel+.90)*self.vmax/closing)
        pads=(max(self.car_margin+.03,self.detour_clearance-.06),self.detour_clearance,self.detour_clearance+.08)
        results=list(original)
        for index,side in ((0,1),(1,-1)):
            # Extra options should rescue a blocked/stalled maneuver rather
            # than continually disturb an already viable passing trajectory.
            if original[index].feasible and original[index].speed>=.2:
                continue
            variants=[original[index]]
            parameters=[(entry,pad,1.) for entry in self.entries for pad in pads]
            parameters.extend((1.7,self.detour_clearance,factor) for factor in (.8,1.25))
            for entry,padding,merge_factor in parameters:
                desired=float(target.d+side*(.155+target.width/2+padding))
                # Commitment preserves the side, but a better corridor may use
                # a different offset. Ranking gently discourages offset chatter.
                def offsets(ss):
                    ahead=delta_s(s,ss,g.length)
                    left,right=g.widths(ss)
                    lane=np.clip(desired,-np.asarray(right)+.24+self.wall_margin,
                                 np.asarray(left)-.24-self.wall_margin)
                    t=np.clip(ahead/entry,0.,1.)
                    value=d+(lane-d)*smoothstep(t)
                    value+=slope*entry*(t-6*t**3+8*t**4-3*t**5)
                    return value*(1-smoothstep((ahead-merge*merge_factor)/2.))
                xy,_,_=g.sample(path_s,offsets(path_s))
                tangent=np.gradient(xy,axis=0)
                yaws=np.arctan2(tangent[:,1],tangent[:,0])
                # Check the measured starting orientation as well as path yaw.
                wall=float(np.min(g.body_clearance(xy,yaws)))
                wall=min(wall,float(g.body_clearance(xy[0],yaw)))
                self.search_stats['evaluated']+=1
                details=dict(entry_m=float(entry),padding_m=float(padding),merge_factor=float(merge_factor),source='expanded')
                if wall<self.wall_margin:
                    self._record_path(xy,original[index].action,False,'wall clearance',**details)
                    continue
                allowance=self.passing_lateral_acceleration
                curvature=self.path_curvature(xy)
                # Slowing down cannot make an Ackermann vehicle follow an
                # arbitrarily tight curve. Keep some tracking headroom too.
                if curvature.max()>.95*np.tan(MAX_STEER)/WHEELBASE:
                    self._record_path(xy,original[index].action,False,'steering curvature limit',**details)
                    continue
                limits=np.minimum(self.vmax,np.sqrt(allowance/np.maximum(curvature,.05)))
                self.tracker.lateral_acceleration=allowance
                steer,speed=self.tracker.control(obs,offset=offsets,vmax=self.vmax,smooth=False)
                segments=np.linalg.norm(np.diff(xy,axis=0),axis=1)
                speed=min(speed,float(np.min(np.sqrt(limits**2+6*np.r_[0.,np.cumsum(segments)]))))
                times=np.r_[0.,np.cumsum(segments)]/max(.7,min(self.vmax,max(float(obs['linear_vel_x']),speed)))
                car=self.motion_clearance(xy,yaws,target,times)
                certified=self.certify_car_departure(xy,yaws,target,times,obs)
                if self.stage==3:
                    # A reactive rival may close the gap if our car brakes
                    # through the detour. Check a slower traversal as well;
                    # extra paths cannot rely on depleted-padding recovery.
                    slow_times=np.r_[0.,np.cumsum(segments)]/max(.4,min(float(obs['linear_vel_x']),speed))
                    car=min(car,self.motion_clearance(xy,yaws,target,slow_times))
                    # The front car can also brake for the approaching bend.
                    # Bound this using the shared map and its observed speed,
                    # without reading the rival's state or control commands.
                    bend_speed=float(np.min(g.speed_at(target.s+np.linspace(0.,6.,25),
                        self.vmax,2.,2.,3.)))
                    slower=replace(target,speed=max(0.,min(target.speed-.3,bend_speed)))
                    car=min(car,self.motion_clearance(xy,yaws,slower,times),
                            self.motion_clearance(xy,yaws,slower,slow_times))
                    certified=False
                if car<self.car_margin and not certified:
                    self._record_path(xy,original[index].action,False,'predicted vehicle clearance',**details)
                    continue
                cost=float(np.sum(segments/np.maximum(.4,limits[:-1])))+.15*abs(desired)+.1/max(.1,wall)
                candidate=Candidate(original[index].action,True,cost,min(wall,car),float(steer),
                                    float(max(0.,speed)),desired,xy,certified)
                candidate.path_parameters={'entry_m':entry,'padding_m':padding,'merge_factor':merge_factor}
                self._record_path(xy,candidate.action,True,'feasible',cost=cost,**details)
                variants.append(candidate)
                self.search_stats['additional_feasible']+=1
            feasible=[c for c in variants if c.feasible]
            if feasible:
                def score(c):
                    continuity=.05*abs(c.offset-committed[1]) if committed and committed[0]==c.action else 0.
                    return c.cost+continuity
                chosen=min(feasible,key=score)
                results[index]=chosen
                if chosen is not original[index]:
                    self.search_stats['replaced'].append({'action':chosen.action,**chosen.path_parameters})
        return self._finish_paths(results)

class _ExpandedMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.planner=ExpandedPlanner(self.planner.track,self.vmax,self.stage)

    def plan(self,*args,**kwargs):
        result=super().plan(*args,**kwargs)
        self.telemetry['path_search']=self.planner.search_stats
        return result

class ExpandedStage2(_ExpandedMixin,MovingModel):pass
class ExpandedStage4(_ExpandedMixin,InteractiveModel):pass
