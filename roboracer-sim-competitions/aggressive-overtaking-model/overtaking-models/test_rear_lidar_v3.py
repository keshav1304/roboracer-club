"""Actual ray-cast scans for cropped rear returns and 20Hz side acquisition."""
import unittest
import numpy as np

from world import make_env
from track_geometry import TrackGeometry, PathTracker
from perception import LidarTracker


class RearLidarRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=make_env(seed=9911)
        cls.g=TrackGeometry(cls.env.unwrapped.track)
        cls.angles=cls.env.unwrapped.sim.agents[1].scan_angles

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def pose(self,s,d=0.):
        xy,yaw,_=self.g.sample(s,d)
        return [float(xy[0]),float(xy[1]),float(yaw)]

    def test_cropped_rear_side_stays_behind_after_leaving_curve(self):
        sensor=LidarTracker(self.g,self.angles)
        errors=[]
        endpoint_frames=0
        for index in range(110):
            t=index*.05
            observer_s=55.+2.4*t
            relative=-.55+.15*min(t,2.)
            poses=np.array([self.pose(observer_s+relative,-.52),self.pose(observer_s)])
            obs,_=self.env.reset(options={'poses':poses})
            target=sensor.update(obs['agent_1'],dt=.05)
            self.assertIsNotNone(target)
            observed_s=float(self.g.project(obs['agent_1']['pose_x'],obs['agent_1']['pose_y'])[0])
            estimate=float(self.g.delta_s(observed_s,target.s))
            errors.append(abs(estimate-relative))
            self.assertLess(estimate,0.,'A following car must not drift into a fictitious lead')
            if target.endpoint_only:
                endpoint_frames+=1
                self.assertLessEqual(target.confidence,.75)
        self.assertGreater(endpoint_frames,30)
        self.assertLess(max(errors),.09)
        self.assertAlmostEqual(target.speed,2.4,delta=.15)

    def drive_fixture(self,relative,lateral,observer_speed,target_speed):
        obs,_=self.env.reset(options={'poses':np.array([self.pose(5.+relative,lateral),self.pose(5.)])})
        for agent,speed in zip(self.env.unwrapped.sim.agents,(target_speed,observer_speed)):
            agent.state[3]=speed
        obs,*_=self.env.step(np.array([[0.,target_speed],[0.,observer_speed]]))
        sensor=LidarTracker(self.g,self.angles)
        controllers=[PathTracker(self.g),PathTracker(self.g)]
        errors=[]
        measured=[]
        for tick in range(60):
            if tick%5==0:
                actual_s=[float(self.g.project(obs[f'agent_{i}']['pose_x'],obs[f'agent_{i}']['pose_y'])[0])for i in(0,1)]
                truth=float(self.g.delta_s(actual_s[1],actual_s[0]))
                target=sensor.update(obs['agent_1'],dt=.05)
                if target is not None:
                    errors.append(abs(float(self.g.delta_s(actual_s[1],target.s))-truth))
                    measured.append(target.speed)
                commands=[]
                for i,speed in enumerate((target_speed,observer_speed)):
                    commands.append((0.,0.) if speed==0 else controllers[i].control(
                        obs[f'agent_{i}'],offset=lateral if i==0 else 0.,vmax=speed,dt=.05))
            obs,*_=self.env.step(np.array(commands))
        self.assertGreater(len(errors),8)
        return errors,measured

    def test_moving_side_midpoint_initializes_at_20hz(self):
        for side in(-.55,.55):
            with self.subTest(side=side):
                errors,speeds=self.drive_fixture(.1,side,2.4,2.4)
                self.assertLess(max(errors),.09)
                self.assertAlmostEqual(speeds[-1],2.4,delta=.15)

    def test_stationary_side_target_does_not_inherit_observer_velocity(self):
        for side in(-.55,.55):
            with self.subTest(side=side):
                errors,speeds=self.drive_fixture(.8,side,2.4,0.)
                self.assertLess(max(errors),.10)
                # Observer-dependent visibility can cause a short small bias;
                # it must settle without inheriting the observer's2.4m/s.
                self.assertLess(max(speeds),.4)
                self.assertLess(speeds[-1],.15)


if __name__=='__main__':
    unittest.main()
