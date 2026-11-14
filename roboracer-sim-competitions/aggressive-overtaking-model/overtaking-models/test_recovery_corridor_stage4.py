import unittest
from unittest.mock import patch
import numpy as np
from recovery_corridor_stage4 import RecoveryCorridorPlanner
from robust_corridor_stage4 import RobustCorridorPlanner
from maneuver_core import Candidate
from perception import Target
from test_corridor_stage4 import RayMap

class RecoveryTests(unittest.TestCase):
 def planner(self,x=.025,y=.193):
  p=RecoveryCorridorPlanner.__new__(RecoveryCorridorPlanner);p.track=RayMap();p.angles=np.array([np.arctan2(y,x)]);p._padding_proposal=True
  from types import SimpleNamespace
  p.interaction=SimpleNamespace(physical_speed=lambda target: target.speed)
  obs=dict(scan=np.array([np.hypot(x,y)]),pose_x=0.,pose_y=0.,pose_theta=0.,linear_vel_x=0.)
  c=Candidate('LEFT',True,1.,0.,0.,.5,0.,np.zeros((2,2)))
  return p,obs,c
 def test_fresh_known_side_padding_can_be_proposed(self):
  p,o,c=self.planner();t=Target(-.17,.35,confidence=.98)
  self.assertTrue(np.isinf(p.lidar_caps(o,t,[c])[0]));self.assertTrue(c.padding_escape_proposal)
  p._padding_proposal=False;self.assertEqual(p.lidar_caps(o,t,[c])[0],0.)
 def test_unknown_stale_and_body_returns_keep_braking(self):
  p,o,c=self.planner();self.assertEqual(p.lidar_caps(o,None,[c])[0],0.)
  self.assertEqual(p.lidar_caps(o,Target(-.17,.35,confidence=.98,time_since_seen=.2),[c])[0],0.)
  p,o,c=self.planner(.025,.159)
  self.assertEqual(p.lidar_caps(o,Target(-.17,.31,confidence=.98),[c])[0],0.)
 def test_front_return_is_never_released(self):
  p,o,c=self.planner(.35,0.)
  self.assertEqual(p.lidar_caps(o,Target(.55,0.,confidence=.98),[c])[0],0.)
 def test_presteer_keeps_vehicle_stopped_until_aligned(self):
  from recovery_corridor_stage4 import RecoveryCorridorStage4
  driver=RecoveryCorridorStage4.__new__(RecoveryCorridorStage4);driver.steering=.2;driver.interaction_debug={}
  c=Candidate('RIGHT',True,1.,.1,-.2,.25,0.,np.zeros((2,2)));c.presteer_escape=True
  driver.prepare_command([c],0);self.assertEqual(c.speed,0.)
  driver.steering=-.2;c.speed=.25;driver.prepare_command([c],0);self.assertEqual(c.speed,.25)
 def test_measured_cloud_rejects_an_escape_that_approaches_returns(self):
  p,o,c=self.planner(.5,0.)
  yaw=np.zeros(2);tt=np.array([0.,1.2]);v=np.ones(2)
  with patch.object(p,'rollout',return_value=(np.array([[0.,0.],[.3,0.]]),yaw,tt,v)):
   self.assertFalse(p.measured_escape(o,c,0.))
  with patch.object(p,'rollout',return_value=(np.array([[0.,0.],[0.,-.3]]),yaw,tt,v)):
   self.assertTrue(p.measured_escape(o,c,0.))
 def test_failed_search_reuse_only_stops_at_unchanged_standstill(self):
  p,o,c=self.planner();t=Target(0.,0.,confidence=.98);p._decision_tick=10
  key=p.stopped_key(o,t);p._failed_search=(key,15)
  with patch.object(RobustCorridorPlanner,'fallback_candidates',return_value=iter([c])) as generate:
   self.assertEqual(list(p.fallback_candidates(o,t)),[]);generate.assert_not_called()
  o['linear_vel_x']=.03
  with patch.object(RobustCorridorPlanner,'fallback_candidates',return_value=iter([c])) as generate:
   self.assertEqual(len(list(p.fallback_candidates(o,t))),1);generate.assert_called_once()
  o['linear_vel_x']=0.;p._decision_tick=15
  with patch.object(RobustCorridorPlanner,'fallback_candidates',return_value=iter([c])) as generate:
   self.assertEqual(len(list(p.fallback_candidates(o,t))),1);generate.assert_called_once()
 def test_identical_capped_motions_are_checked_once(self):
  from dataclasses import replace
  p,o,c=self.planner()
  with patch.object(RobustCorridorPlanner,'fallback_candidates',return_value=iter([c,replace(c)])):
   self.assertEqual(len(list(p.fallback_candidates(o,None))),1)
 def test_depleted_gap_requires_nonworsening_and_restoration(self):
  p,o,c=self.planner();p.target=Target(0.,0.)
  xy=np.array([[0.,0.],[1.,0.]]);yaw=np.zeros(2);pred=[dict(path=np.zeros((2,2)),yaw=yaw)]
  for initial,gap,expected in [(.02,.02,True),(.02,.019,False),(.004,.004,False)]:
   result=dict(safe=False,initial_gap=initial,gap=gap,wall=.1)
   with patch.object(RobustCorridorPlanner,'check_many',return_value=[result]),patch.object(p,'rollout',return_value=(xy,yaw,np.array([0.,1.2]),np.ones(2))):
    self.assertEqual(p.check_many(o,[c],0.,pred)[0]['safe'],expected)
  result=dict(safe=False,initial_gap=.02,gap=.02,wall=.1)
  with patch.object(RobustCorridorPlanner,'check_many',return_value=[result]),patch.object(p,'rollout',return_value=(xy*.1,yaw,np.array([0.,1.2]),np.ones(2))):
   self.assertFalse(p.check_many(o,[c],0.,pred)[0]['safe'])
if __name__=='__main__':unittest.main()
