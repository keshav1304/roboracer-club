"""Shared racing-reference feasibility, speed envelopes and measured-lap model."""
import unittest
import numpy as np

from world import make_env
from track_geometry import TrackGeometry, periodic_speed_profile


class SpeedEnvelopeTests(unittest.TestCase):
    def test_closed_circle_respects_lateral_acceleration(self):
        theta=np.linspace(0,2*np.pi,200,endpoint=False)
        xy=2*np.column_stack([np.cos(theta),np.sin(theta)])
        speed,lap=periodic_speed_profile(xy,np.full(len(xy),.5),vmax=4.,lateral_acceleration=2.)
        np.testing.assert_allclose(speed,2.,atol=1e-8)
        self.assertAlmostEqual(lap,2*np.pi,places=3)

    def test_periodic_braking_constraint_propagates_across_start(self):
        theta=np.linspace(0,2*np.pi,100,endpoint=False)
        xy=5*np.column_stack([np.cos(theta),np.sin(theta)])
        curvature=np.full(len(xy),.05);curvature[0]=3.
        speed,_=periodic_speed_profile(xy,curvature,vmax=3.,lateral_acceleration=2.,acceleration=1.5,braking=2.5)
        ds=np.linalg.norm(np.roll(xy,-1,axis=0)-xy,axis=1)
        v2=speed**2
        self.assertTrue(np.all(np.roll(v2,-1)-v2<=3.*ds+1e-9))
        self.assertTrue(np.all(v2-np.roll(v2,-1)<=5.*ds+1e-9))
        self.assertLess(speed[-1],3.)


class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=make_env(seed=889)
        cls.reference=TrackGeometry(cls.env.unwrapped.track)
        cls.legacy=TrackGeometry(cls.env.unwrapped.track,reference_mode='legacy')

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_refined_line_is_feasible_and_improves_predicted_time(self):
        g=self.reference
        self.assertGreaterEqual(float(g.body_clearance(g.xy,g.yaw).min()),.18)
        self.assertLess(g.predicted_lap_time(),self.legacy.predicted_lap_time())
        self.assertNotEqual(g.reference_hash,self.legacy.reference_hash)
        self.assertEqual(g.reference_mode,'lap_time')

    def test_both_cars_obtain_identical_periodic_speed_reference(self):
        g=self.reference
        ss=np.linspace(0,g.length,101)
        a=g.speed_at(ss,2.4,2.,2.,3.)
        b=g.speed_at(ss+g.length,2.4,2.,2.,3.)
        np.testing.assert_allclose(a,b,rtol=1e-7,atol=1e-7)
        self.assertTrue(np.all(a<=2.4+1e-8))
        self.assertTrue(np.all(g.speed_at(ss,1.6)<=1.6+1e-8))


if __name__=='__main__':
    unittest.main()
