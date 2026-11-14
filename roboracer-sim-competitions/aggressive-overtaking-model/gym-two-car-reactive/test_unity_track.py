"""Geometry checks for the read-only AutoDRIVE default-track importer."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import json
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from import_unity_track import APP, EXPORT, NAME, ensure_unity_track, rasterize, world_matrix


class GeometryTests(unittest.TestCase):
    def test_parent_rotation_scale_and_translation(self):
        def transform(position, rotation, scale, parent):
            return {"m_LocalPosition": dict(zip("xyz", position)), "m_LocalRotation": dict(zip("xyzw", rotation)),
                    "m_LocalScale": dict(zip("xyz", scale)), "m_Father": {"m_PathID": parent}}
        transforms = {1: transform([10, 0, 0], Rotation.from_euler("y", 90, degrees=True).as_quat(), [2, 2, 2], 0),
                      2: transform([1, 0, 0], [0, 0, 0, 1], [1, 1, 1], 1)}
        np.testing.assert_allclose(world_matrix(2, transforms)[:3, 3], [10, 0, -2], atol=1e-10)

    def test_raster_uses_enclosed_spawn_area_and_retains_island(self):
        vertices, triangles = [], []
        for x0, y0, x1, y1 in [(0, 0, 10, .2), (0, 9.8, 10, 10), (0, 0, .2, 10),
                               (9.8, 0, 10, 10), (4, 4, 6, 6)]:
            base = len(vertices)
            vertices.extend([[x0, .3, y0], [x1, .3, y0], [x1, .3, y1], [x0, .3, y1]])
            triangles.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
        free, origin = rasterize(np.array(vertices), np.array(triangles), [2, 2, 0], resolution=.05)
        def cell(x, y):
            c, r = np.floor((np.array([x, y]) - origin) / .05).astype(int)
            return free[len(free) - 1 - r, c]
        self.assertTrue(cell(2, 2))
        self.assertFalse(cell(5, 5))
        self.assertFalse(cell(-.5, 2))
        self.assertFalse(free[0].any())


@unittest.skipUnless(APP.exists(), "Local AutoDRIVE app is needed for the integration check")
class DefaultTrackTests(unittest.TestCase):
    def test_default_scene_map_and_rectangular_render(self):
        import gymnasium as gym
        import run_two_cars as baseline
        from import_iros_track import configure_iros_renderer
        ensure_unity_track()
        metadata = json.loads((EXPORT / "source.json").read_text())
        self.assertEqual(metadata["track_name"], "SRL 2026 ICRA Track")
        self.assertEqual(len(metadata["checkpoint_positions"]), 21)
        self.assertEqual(metadata["resolution"], .02)
        self.assertGreater(metadata["reference_length"], 60)
        env = gym.make("f1tenth_gym:f1tenth-v0", config={"map": NAME, "num_agents": 2,
                       "observation_config": {"type": "features", "features": baseline.FEATURES}},
                       render_mode="rgb_array")
        try:
            configure_iros_renderer(env)
            u = env.unwrapped
            obs, _ = env.reset(options={"poses": baseline.start_poses(u.track, 5)})
            self.assertTrue(all(o["collision"] == 0 for o in obs.values()))
            self.assertEqual(env.render().shape, (800, 800, 3))
            height, width = u.track.occupancy_map.shape
            for obj in u.renderer.map_renderers.values():
                sw, sh = obj.map_surface.get_size()
                self.assertAlmostEqual(sw / sh, width / height, delta=.005)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
