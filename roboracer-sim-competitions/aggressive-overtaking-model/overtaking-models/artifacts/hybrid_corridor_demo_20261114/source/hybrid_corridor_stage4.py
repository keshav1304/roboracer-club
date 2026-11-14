"""Keep broad approach choices; use guarded recovery in close encounters."""
from recovery_corridor_stage4 import RecoveryCorridorStage4
from corridor_stage4 import CorridorStage4
from maneuver_core import delta_s

class HybridCorridorStage4(RecoveryCorridorStage4):
    def select_response_maneuver(self,candidates,index,obs,target):
        close=getattr(self,'_close_encounter',False)
        if target is None:
            close=False
        else:
            s,_=self.planner.track.project(obs['pose_x'],obs['pose_y'])
            rel=float(delta_s(s,target.s,self.planner.track.length))
            if -.9<rel<.7:close=True
            if rel>1.5 or rel< -2.2:close=False
        self._close_encounter=close
        fresh=target is not None and target.confidence>=.6 and target.time_since_seen<=.1
        if fresh and not close:
            index=CorridorStage4.select_response_maneuver(self,candidates,index,obs,target)
            index=self.prepare_command(candidates,index)
            self.interaction_debug['encounter_phase']='wide approach'
            return index
        index=super().select_response_maneuver(candidates,index,obs,target)
        self.interaction_debug['encounter_phase']='guarded close encounter'
        return index
