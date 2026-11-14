import unittest
import numpy as np
from corridor_stage4 import CorridorPlanner
from maneuver_core import Candidate
from perception import Target
from world import make_env
from track_geometry import TrackGeometry

class RayMap:
 resolution=.1
 occupancy=np.zeros((200,200))
 def __init__(self,intervening_wall=False):self.intervening_wall=intervening_wall
 def sample(self,s,d=0.):
  s,d=np.broadcast_arrays(s,d);return np.stack([s,d],axis=-1),np.zeros_like(s),np.zeros_like(s)
 def widths(self,s):return np.full_like(s,2.98,dtype=float),np.full_like(s,2.98,dtype=float)
 def clearance(self,points):
  y=np.abs(np.asarray(points)[...,1]);free=5.-y
  return np.where((y>=3)&(y<=3.2),-.01,free) if self.intervening_wall else free

class CorridorTests(unittest.TestCase):
 def planner(self,map):
  p=CorridorPlanner.__new__(CorridorPlanner);p.track=map;p._width_cache={};return p
 def test_width_search_extends_beyond_old_three_metre_limit(self):
  p=self.planner(RayMap());l,r=p.wall_widths(np.array([0.,1.,2.]))
  self.assertTrue(np.all(l>4.8));np.testing.assert_allclose(l,r)
 def test_ray_stops_at_first_wall_not_a_neighboring_open_lane(self):
  p=self.planner(RayMap(True));l,r=p.wall_widths(0.)
  self.assertLess(l,3.1);self.assertGreater(l,2.8);self.assertEqual(l,r)
 def test_front_lidar_brake_is_included_in_planned_speed(self):
  p=self.planner(RayMap());p.angles=np.array([0.]);c=Candidate('WAIT',True,1.,1.,0.,2.4,0.,np.zeros((2,2)))
  cap=p.lidar_caps(dict(scan=np.array([.5]),pose_x=0.,pose_y=0.,pose_theta=0.),None,[c])[0]
  self.assertAlmostEqual(cap,np.sqrt(.6))
 def test_wide_straight_has_executable_large_lane_changes(self):
  from track_geometry import PathTracker
  class WideStraight(RayMap):
   length=100.
   def project(self,x,y):return x,y
   def body_clearance(self,xy,yaw):return 5.-np.abs(np.asarray(xy)[...,1])-.34
   def speed_at(self,*a):return 2.4
  g=WideStraight();p=self.planner(g);p.vmax=2.4;p.wall_margin=.04;p.tracker=PathTracker(g);p.angles=np.linspace(-2.35,2.35,1080)
  obs=dict(pose_x=0.,pose_y=0.,pose_theta=0.,linear_vel_x=.5,scan=np.full(1080,20.))
  choices=list(p.fallback_candidates(obs,Target(1.,0.,speed=.5)))
  self.assertGreater(max(abs(c.offset) for c in choices),2.)
 def test_clear_track_keeps_previous_controller_command(self):
  from corridor_stage4 import CorridorStage4
  from interaction_stage4 import InteractionStage4
  env=make_env(seed=285)
  try:
   g=TrackGeometry(env.unwrapped.track);angles=env.unwrapped.sim.agents[0].scan_angles
   old=InteractionStage4(g,angles);new=CorridorStage4(g,angles)
   xy,yaw,_=g.sample(4.);obs=dict(pose_x=xy[0],pose_y=xy[1],pose_theta=yaw,linear_vel_x=2.,scan=np.full(1080,20.))
   np.testing.assert_allclose(old.plan(obs,dt=.05),new.plan(obs,dt=.05),atol=1e-12)
   self.assertEqual(old.action,new.action)
  finally:env.close()
 def test_wall_bounded_search_has_options_beyond_fixed_lane_offsets(self):
  env=make_env(seed=281)
  try:
   g=TrackGeometry(env.unwrapped.track);p=CorridorPlanner(g,stage=3);p.angles=env.unwrapped.sim.agents[0].scan_angles
   xy,yaw,_=g.sample(32.2,-.2);obs=dict(pose_x=xy[0],pose_y=xy[1],pose_theta=yaw,linear_vel_x=0.,scan=np.full(1080,20.))
   target=Target(32.4,.2,speed=0.,confidence=.99,observations=100)
   choices=list(p.fallback_candidates(obs,target));self.assertTrue(choices)
   self.assertLess(p.corridor_debug['lower_offset_m'],-1.)
   self.assertGreater(p.corridor_debug['upper_offset_m'],1.)
   for c,cap in zip(choices,p.lidar_caps(obs,target,choices)):
    self.assertLessEqual(c.speed,cap+1e-10)
    self.assertGreaterEqual(c.offset,p.corridor_debug['lower_offset_m'])
    self.assertLessEqual(c.offset,p.corridor_debug['upper_offset_m'])
  finally:env.close()
if __name__=='__main__':unittest.main()
