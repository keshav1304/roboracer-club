import unittest
import numpy as np
from perception import Target
from interaction_stage4 import InteractionForecast,InteractionStage4
from maneuver_core import Candidate
from world import make_env
from track_geometry import TrackGeometry

class StraightTrack:
 def sample(self,s,d=0.):
  s,d=np.broadcast_arrays(s,d);return np.stack([s,d],axis=-1),np.zeros_like(s),np.zeros_like(s)
 def speed_at(self,*a):return 2.4

class InteractionTests(unittest.TestCase):
 def test_frenet_speed_is_not_physical_speed_on_inside_line(self):
  class Curve(StraightTrack):
   def sample(self,s,d=0.):
    p,y,k=super().sample(s,d);return p,y,np.full_like(k,.8,dtype=float)
  f=InteractionForecast(Curve())
  self.assertAlmostEqual(f.physical_speed(Target(0.,.3,speed=2.)),1.52)
  self.assertAlmostEqual(f.physical_speed(Target(0.,-.3,speed=2.)),2.48)
 def test_stationary_opponent_does_not_slide_sideways(self):
  f=InteractionForecast(StraightTrack());t=Target(1.,.3,speed=0.)
  f.update(t,.05);pred=f.trajectories(t,np.arange(25)*.05)
  for p in pred[:3]:np.testing.assert_allclose(p['path'],np.tile([1.,.3],(25,1)))
  self.assertEqual(len(pred),3)
 def test_braking_hypothesis_is_slower_than_trend(self):
  f=InteractionForecast(StraightTrack());t=Target(1.,0.,speed=2.)
  f.update(t,.05);pred=f.trajectories(t,np.arange(25)*.05)
  self.assertLess(pred[1]['path'][-1,0],pred[0]['path'][-1,0])
 def test_batch_rollout_matches_scalar_dynamics(self):
  from interaction_stage4 import InteractionPlanner
  env=make_env(seed=723)
  try:
   g=TrackGeometry(env.unwrapped.track);p=InteractionPlanner(g,stage=3)
   xy,yaw,_=g.sample(4.);obs=dict(pose_x=xy[0],pose_y=xy[1],pose_theta=yaw,linear_vel_x=1.4)
   target=Target(6.,.2,speed=1.5,confidence=.99,observations=100);p.target=target;p.interaction.update(target,.05)
   candidates=list(p.fallback_candidates(obs,target))[:8];pred=p.interaction.trajectories(target,np.arange(25)*.05)
   checks=p.check_many(obs,candidates,.03,pred)
   for c,check in zip(candidates,checks):
    expected=p.rollout(obs,c,.03)[0]
    np.testing.assert_allclose(check['path'],expected,atol=1e-10)
  finally:env.close()
 def test_blocked_reference_rollout_fails_and_fallback_brakes(self):
  env=make_env(seed=721)
  try:
   g=TrackGeometry(env.unwrapped.track);driver=InteractionStage4(g,env.unwrapped.sim.agents[0].scan_angles)
   xy,yaw,_=g.sample(4.);obs=dict(pose_x=xy[0],pose_y=xy[1],pose_theta=yaw,linear_vel_x=1.,scan=np.full(1080,20.))
   t=Target(5.2,0.,speed=0.,width=6.,confidence=.99,observations=100)
   driver.planner.target=t;driver.planner.control_dt=.05;driver.planner.interaction.update(t,.05)
   path=g.sample(4.+np.linspace(0,6,61))[0]
   cs=[Candidate(a,True,3.,1.,0.,2.4,0.,path) for a in ('LEFT','RIGHT','WAIT','PROBE')]
   driver.planner.response_choice=None;driver.planner.response_debug={};driver.planner.debug_paths=[]
   predictions=driver.planner.interaction.trajectories(t,np.arange(25)*.05)
   self.assertFalse(driver.planner.check(obs,cs[2],0.,predictions)['safe'])
   index=driver.select_response_maneuver(cs,2,obs,t)
   self.assertLessEqual(cs[index].speed,.5)
   self.assertTrue(driver.interaction_debug['override'])
   driver.angles=np.array([0.]);cs[index].interaction_certificate={}
   np.testing.assert_array_equal(driver.response_speed_caps(np.array([.1]),np.array([True]),cs[index],{**obs,'scan':np.array([.5])},t),[.1])
  finally:env.close()
if __name__=='__main__':unittest.main()
