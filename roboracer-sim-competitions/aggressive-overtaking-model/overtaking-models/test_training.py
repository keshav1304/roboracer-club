"""Dataset split integrity and explicit bootstrap-label checks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from train_models import (bootstrap_probabilities, append_rollout_samples,
                          sector_at, split_for_sector, FEATURE_NAMES, mirror_examples, reactive_opponent)


class TrainingDataTests(unittest.TestCase):
    def test_moving_opponent_does_not_brake_from_zero_command_initialization(self):
        from unittest.mock import patch
        angles = np.linspace(-.75 * np.pi, .75 * np.pi, 1080)
        observation = {"linear_vel_x": 2.4, "scan": np.full(1080, 20.)}
        geometry = object()
        # The physics/state initialization is exercised in test_world. This
        # guards training/testing against silently using different opponents.
        with patch("train_models.make_stage2_opponent") as factory:
            self.assertIs(reactive_opponent(geometry, angles, 2.4, observation, steering=.1), factory.return_value)
            factory.assert_called_once_with(geometry, angles, 2.4, observation, .1)

    def test_reflection_is_involutive_and_swaps_action_labels(self):
        X = np.arange(100, dtype=np.float32).reshape(2, 50)
        y = np.array([[.8, .1, .1, 0.], [.2, .3, .1, .4]], dtype=np.float32)
        reflected, labels = mirror_examples(X, y)
        restored, restored_labels = mirror_examples(reflected, labels)
        np.testing.assert_array_equal(restored, X)
        np.testing.assert_array_equal(restored_labels, y)
        np.testing.assert_array_equal(labels, y[:, [1, 0, 2, 3]])

    def test_blocked_candidates_and_cost_preferences(self):
        candidates = [{"action": "LEFT", "feasible": False, "estimated_time": 0.},
                      {"action": "RIGHT", "feasible": True, "estimated_time": 3.},
                      {"action": "WAIT", "feasible": True, "estimated_time": 8.}]
        p = bootstrap_probabilities(candidates)
        self.assertEqual(p[0], 0.)
        self.assertGreater(p[1], .99)
        self.assertAlmostEqual(p.sum(), 1.)
        candidates[1]["feasible"] = False
        np.testing.assert_equal(bootstrap_probabilities(candidates), [0., 0., 1.])

    def test_reserved_sector_boundaries(self):
        self.assertEqual(sector_at(120., 120.), 0)
        self.assertEqual(split_for_sector(sector_at(15., 120.)), "holdout")
        self.assertEqual(split_for_sector(sector_at(25., 120.)), "validation")
        self.assertEqual(split_for_sector(sector_at(35., 120.)), "train")

    def test_rollouts_require_current_and_initial_training_location(self):
        candidates = [{"action": a, "feasible": True, "estimated_time": c}
                      for a, c in zip(("LEFT", "RIGHT", "WAIT"), (3., 4., 5.))]
        base = {"stage": 1, "features": [0.] * len(FEATURE_NAMES), "candidates": candidates,
                "scenario": {"s": 5., "stage": 1}}
        incoming = [dict(base), {**base, "ego_s": 15.}, {**base, "ego_s": 25.},
                    {**base, "ego_s": 35., "scenario": {"s": 15., "stage": 1}},
                    {**base, "ego_s": 35.}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.json"
            path.write_text(json.dumps(incoming))
            rows = []
            result = append_rollout_samples(rows, [path], 1, 120.)
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["rejections"]["missing_observation_location"], 1)
        self.assertEqual(result["rejections"]["not_training_sector"], 3)
        self.assertEqual(rows[0]["label_source"], "geometric_bootstrap")
        self.assertFalse(rows[0]["rollout_verified"])


if __name__ == "__main__":
    unittest.main()
