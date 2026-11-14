import unittest
from unittest.mock import patch
import numpy as np
from robust_corridor_stage4 import RobustCorridorPlanner,RobustCorridorStage4
from corridor_stage4 import CorridorPlanner,CorridorStage4
from joint_path_stage4 import JointStage4
from maneuver_core import Candidate
from perception import Target
from world import make_env
from track_geometry import TrackGeometry

class RobustCorridorTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.env=make_env(seed=294);cls.g=TrackGeometry(cls.env.unwrapped.track);cls.angles=cls.env.unwrapped.sim.agents[0].scan_angles
 @classmethod
 def tearDownClass(cls):cls.env.close()
 def obs(self,s=32.2,d=-.2,v=2.):
  xy,yaw,_=self.g.sample(s,d)
  return dict(pose_x=xy[0],pose_y=xy[1],pose_theta=yaw,linear_vel_x=v,scan=np.full(len(self.angles),20.))
 def test_batched_forecast_matches_all_original_hypotheses(self):
  from robust_corridor_stage4 import FastInteractionForecast,CachedGeometry
  from interaction_stage4 import InteractionForecast
  a=InteractionForecast(self.g);b=FastInteractionForecast(CachedGeometry(self.g))
  for speed in [0.,.03,1.2,2.4]:
   for position in [0.,14.7,59.]:
    target=Target(position,.35,speed=speed,lateral_speed=-.1)
    a.acceleration=b.acceleration=.7;a.lateral_rate=b.lateral_rate=.14
    for x,y in zip(a.trajectories(target,np.arange(25)*.05),b.trajectories(target,np.arange(25)*.05)):
     np.testing.assert_allclose(x['path'],y['path'],atol=1e-11);np.testing.assert_allclose(x['yaw'],y['yaw'],atol=1e-11)
 def test_cached_geometry_and_periodic_speed_are_equivalent(self):
  from robust_corridor_stage4 import CachedGeometry
  c=CachedGeometry(self.g);ss=np.linspace(-2.,self.g.length+2.,100)
  for d in [0.,.35,-.4]:
   for x,y in zip(self.g.sample(ss,d),c.sample(ss,d)):np.testing.assert_allclose(x,y,atol=1e-12)
  np.testing.assert_allclose(self.g.speed_at(ss),c.speed_at(ss),atol=1e-12)
  c.clear();np.testing.assert_allclose(self.g.widths(ss),c.widths(ss),atol=1e-12)
 def test_batched_geometry_matches_original_wall_lattice(self):
  old=CorridorPlanner(self.g,stage=3);new=RobustCorridorPlanner(self.g,stage=3)
  old.angles=new.angles=self.angles;obs=self.obs();target=Target(33.,.1,speed=1.5)
  a=list(old.fallback_candidates(obs,target));b=list(new.fallback_candidates(obs,target))
  self.assertGreater(len(a),0);self.assertEqual(len(a),len(b))
  for x,y in zip(a,b):
   np.testing.assert_allclose(x.path,y.path,atol=1e-10)
   np.testing.assert_allclose([x.steering,x.speed,x.offset],[y.steering,y.speed,y.offset],atol=1e-10)
 def test_unique_arc_lidar_limits_match_original(self):
  old=CorridorPlanner(self.g,stage=3);new=RobustCorridorPlanner(self.g,stage=3);old.angles=new.angles=self.angles
  obs=self.obs();obs['scan']=np.random.default_rng(7).uniform(.25,9.,len(self.angles))
  choices=[Candidate('LEFT',True,1.,0.,steer,speed,0.,np.zeros((2,2))) for steer in [-.3,0.,.2] for speed in [.3,1.,2.4]]
  np.testing.assert_allclose(old.lidar_caps(obs,None,choices),new.lidar_caps(obs,None,choices))
 def test_recently_occluded_rival_still_checked(self):
  driver=RobustCorridorStage4(self.g,self.angles);obs=self.obs(s=4.,d=.4)
  target=Target(3.4,0.,speed=2.,confidence=.25,time_since_seen=.4,observed_seconds=2.)
  driver.planner.target=target
  chosen=Candidate('LEFT',True,1.,1.,0.,1.,.4,self.g.sample(4+np.linspace(0,3,31),.4)[0])
  with patch.object(JointStage4,'select_response_maneuver',return_value=0),patch.object(driver.planner,'check',return_value=dict(safe=True,lidar_cap=2.,gap=.1)) as checker:
   self.assertEqual(driver.select_response_maneuver([chosen],0,obs,target),0)
   checker.assert_called_once();self.assertTrue(driver.interaction_debug['active']);self.assertTrue(driver.interaction_debug['occluded'])
 def test_no_target_command_unchanged(self):
  old=CorridorStage4(self.g,self.angles);new=RobustCorridorStage4(self.g,self.angles)
  np.testing.assert_allclose(old.plan(self.obs(s=4.,d=0.),dt=.05),new.plan(self.obs(s=4.,d=0.),dt=.05),atol=1e-12)
 def test_recovery_arcs_still_obey_front_lidar_limit(self):
  p=RobustCorridorPlanner(self.g,stage=3);p.angles=self.angles;obs=self.obs(v=0.)
  obs['scan']=np.full(len(self.angles),.3)
  choices=list(p.fallback_candidates(obs,Target(32.4,.2,speed=0.)))
  arcs=[c for c in choices if c.path_parameters.get('local_arc')];self.assertTrue(arcs)
  for c in arcs:self.assertLessEqual(c.speed,float(p.lidar_caps(obs,None,[c])[0])+1e-10)
if __name__=='__main__':unittest.main()
