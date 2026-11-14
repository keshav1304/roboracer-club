import unittest
import numpy as np
from joint_path_stage4 import JointPlanner,frozen
from world import make_env
from track_geometry import TrackGeometry
from perception import Target

class JointTests(unittest.TestCase):
 def test_speed_profile_brakes_gradually_and_obeys_acceleration(self):
  arc=np.linspace(0.,6.,61);curve=np.full(61,.1)
  speeds,times=JointPlanner.speed_profile(arc,curve,2.4,1.6)
  self.assertAlmostEqual(speeds[0],2.4);self.assertAlmostEqual(speeds[-1],1.6)
  self.assertTrue(np.all(np.diff(times)>0))
  accel=np.diff(speeds**2)/(2*np.diff(arc))
  self.assertLessEqual(accel.max(),2.+1e-8);self.assertGreaterEqual(accel.min(),-3.-1e-8)
 def test_straight_constant_speed_arrival_time(self):
  speeds,times=JointPlanner.speed_profile(np.linspace(0,6,61),np.full(61,.01),2.4,2.4)
  np.testing.assert_allclose(speeds,2.4);self.assertAlmostEqual(times[-1],2.5)
 def test_inherited_lane_change_choices_preserved(self):
  self.assertEqual(JointPlanner.entries,frozen.LearnedPlanner.entries)
 def test_impossible_immediate_turn_rejected(self):
  self.assertIsNone(JointPlanner.speed_profile(np.linspace(0,6,61),np.full(61,1.2),2.4,2.4))
 def test_clear_track_preserves_baseline(self):
  env=make_env(seed=1971)
  try:
   g=TrackGeometry(env.unwrapped.track);a=frozen.LearnedPlanner(g,stage=3);b=JointPlanner(g,stage=3)
   for s in (4.,19.,34.,49.):
    xy,yaw,_=g.sample(s);obs=dict(pose_x=xy[0],pose_y=xy[1],pose_theta=yaw,linear_vel_x=2.1,scan=np.full(1080,20.))
    for before,after in zip(a.candidates(obs,None),b.candidates(obs,None)):
     np.testing.assert_array_equal(before.path,after.path)
     self.assertEqual((before.speed,before.steering,before.feasible),(after.speed,after.steering,after.feasible))
  finally:env.close()
 def test_closing_approach_preserves_all_existing_paths(self):
  env=make_env(seed=1973)
  try:
   g=TrackGeometry(env.unwrapped.track);a=frozen.LearnedPlanner(g,stage=3);b=JointPlanner(g,stage=3)
   a.control_dt=b.control_dt=.05
   xy,yaw,_=g.sample(4.);obs=dict(pose_x=xy[0],pose_y=xy[1],pose_theta=yaw,linear_vel_x=2.3,scan=np.full(1080,20.))
   target=Target(5.3,0.,speed=2.,width=.31,confidence=.99,observations=100)
   for before,after in zip(a.candidates(obs,target),b.candidates(obs,target)):
    np.testing.assert_array_equal(before.path,after.path)
    self.assertEqual(before.record(),after.record())
   self.assertEqual(b.joint_debug['geometries'],0)
  finally:env.close()
 def test_full_width_blocker_cannot_be_rescued(self):
  env=make_env(seed=1972)
  try:
   g=TrackGeometry(env.unwrapped.track);p=JointPlanner(g,stage=3);p.control_dt=.05
   xy,yaw,_=g.sample(4.);obs=dict(pose_x=xy[0],pose_y=xy[1],pose_theta=yaw,linear_vel_x=.2,scan=np.full(1080,20.))
   candidates=p.candidates(obs,Target(5.,0.,speed=.5,width=5.,confidence=.99,observations=100))
   self.assertFalse(candidates[0].feasible);self.assertFalse(candidates[1].feasible)
   self.assertGreater(p.joint_debug['geometries'],0)
  finally:env.close()
if __name__=='__main__':unittest.main()
