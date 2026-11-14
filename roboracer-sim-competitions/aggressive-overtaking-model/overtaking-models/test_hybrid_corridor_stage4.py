import unittest
from unittest.mock import patch
from hybrid_corridor_stage4 import HybridCorridorStage4
from recovery_corridor_stage4 import RecoveryCorridorStage4
from corridor_stage4 import CorridorStage4
from maneuver_core import Candidate
from perception import Target
from test_corridor_stage4 import RayMap
import numpy as np

class HybridTests(unittest.TestCase):
 def test_close_mode_latches_and_recovers_approach_only_after_separation(self):
  from types import SimpleNamespace
  class Straight(RayMap):
   length=100.
   def project(self,x,y):return x,y
  d=HybridCorridorStage4.__new__(HybridCorridorStage4);d.planner=SimpleNamespace(track=Straight());d.interaction_debug={};d.steering=0.
  obs=dict(pose_x=0.,pose_y=0.,pose_theta=0.,linear_vel_x=2.)
  cs=[Candidate('LEFT',True,1.,1.,0.,1.,0.,np.zeros((2,2)))]
  with patch.object(CorridorStage4,'select_response_maneuver',return_value=0) as approach,patch.object(RecoveryCorridorStage4,'select_response_maneuver',return_value=0) as near:
   for gap in (1.2,.6,.9,1.6):d.select_response_maneuver(cs,0,obs,Target(gap,0.,confidence=.98))
   self.assertEqual(approach.call_count,2);self.assertEqual(near.call_count,2)
   d.select_response_maneuver(cs,0,obs,Target(1.6,0.,confidence=.4,time_since_seen=.3));self.assertEqual(near.call_count,3)
if __name__=='__main__':unittest.main()
