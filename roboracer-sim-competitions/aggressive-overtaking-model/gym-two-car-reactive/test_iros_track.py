"""Check source-map fidelity and rectangular-map rendering, without a display."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import unittest
import numpy as np
import yaml
from PIL import Image

from import_iros_track import NAME, SOURCE, ensure_iros_track, configure_iros_renderer


class IROSTrackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dest = ensure_iros_track()

    def test_map_pixels_and_frame_preserved(self):
        source = SOURCE / "maps" / NAME
        np.testing.assert_array_equal(np.asarray(Image.open(source / "map.pgm")),
                                      np.asarray(Image.open(self.dest / f"{NAME}.png")))
        original = yaml.safe_load((source / "map.yaml").read_text())
        imported = yaml.safe_load((self.dest / f"{NAME}_map.yaml").read_text())
        for key in ("resolution", "origin", "negate", "occupied_thresh", "free_thresh"):
            self.assertEqual(original[key], imported[key])
        self.assertEqual(imported["resolution"], 0.05)

    def test_raceline_positions_and_speeds_preserved(self):
        source = np.genfromtxt(SOURCE / "racelines" / f"{NAME}-line" / "waypoints.csv",
                               delimiter=",", names=True)
        imported = np.loadtxt(self.dest / f"{NAME}_raceline.csv", delimiter=";")
        np.testing.assert_allclose(imported[:, 1], source["x"], atol=1e-8)
        np.testing.assert_allclose(imported[:, 2], source["y"], atol=1e-8)
        np.testing.assert_allclose(imported[:, 5], source["velocity"], atol=1e-8)
        self.assertTrue(np.isfinite(imported).all())
        self.assertTrue((np.diff(imported[:, 0]) > 0).all())
        self.assertLess(np.hypot(*(imported[0, 1:3] - imported[-1, 1:3])), 0.1)
        self.assertTrue((np.loadtxt(self.dest / f"{NAME}_centerline.csv", delimiter=",")[:, 2:] > 0.205).all())

    def test_render_matches_collision_grid_without_stretching(self):
        import gymnasium as gym
        import pygame
        import run_two_cars as baseline
        env = gym.make("f1tenth_gym:f1tenth-v0", config={"map": NAME, "num_agents": 2,
                       "observation_config": {"type": "features", "features": baseline.FEATURES}},
                       render_mode="rgb_array")
        try:
            configure_iros_renderer(env)
            u = env.unwrapped
            obs, _ = env.reset(options={"poses": baseline.start_poses(u.track, 4)})
            self.assertTrue(all(o["collision"] == 0 for o in obs.values()))
            self.assertEqual(env.render().shape, (800, 800, 3))
            renderer = u.renderer
            height, width = u.track.occupancy_map.shape
            for key, obj in renderer.map_renderers.items():
                sw, sh = obj.map_surface.get_size()
                self.assertAlmostEqual(sw / sh, width / height, delta=0.005)
                # Actual surface colours at world pixel centres must match the
                # LiDAR/collision grid. Asymmetric coordinates catch rotations.
                pixels = pygame.surfarray.array3d(obj.map_surface)
                for row, col in ((40, 30), (90, 55), (180, 42), (240, 80)):
                    x, y = int((col + 0.5) * sw / width), int((row + 0.5) * sh / height)
                    self.assertEqual(pixels[x, y, 0], u.track.occupancy_map[row, col])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
