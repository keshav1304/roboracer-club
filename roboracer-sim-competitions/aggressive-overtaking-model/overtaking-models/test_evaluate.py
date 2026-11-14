"""Checks for disjoint sweep seeds and honest overtaking completion metrics."""
import unittest

from evaluate import randomized_scenarios
from evaluation_metrics import passing_metrics, aggregate_metrics, TRAIN_START_SECTORS, RESERVED_START_SECTORS


class FakeGeometry:
    length = 20.

    def project(self, x, y):
        return x % self.length, y


def frame(t, x, relative, speed=2.):
    return {"t": t, "ego": [x, 0., 0., speed], "opponent": [x-relative, 0., 0., 2.],
            "relative_progress": relative, "action": "LEFT"}


class EvaluationTests(unittest.TestCase):
    def test_development_and_reserved_starts_are_disjoint_and_reproducible(self):
        tune = randomized_scenarios(2, 120., 18, suite="tuning")
        test = randomized_scenarios(2, 120., 18, suite="heldout")
        self.assertEqual(tune, randomized_scenarios(2, 120., 18, suite="tuning"))
        self.assertEqual({int(s.s / 10) for s in tune}, set(TRAIN_START_SECTORS))
        self.assertEqual({int(s.s / 10) for s in test}, set(RESERVED_START_SECTORS))
        self.assertTrue(set(s.seed for s in tune).isdisjoint(s.seed for s in test))
        self.assertTrue(all(s.opponent_speed < s.ego_speed for s in test))
        self.assertGreater(len({round(s.opponent_speed, 2) for s in test}), 3)

    def test_interactive_trials_enforce_equal_caps(self):
        scenes = randomized_scenarios(3, 120., 12)
        self.assertTrue(all(s.ego_speed == s.opponent_speed and s.opponent == "reactive" for s in scenes))

    def test_almost_pass_does_not_receive_completion_time(self):
        result = {"scenario": {"s": 0}, "success": False, "passed_at": None}
        m = passing_metrics(result, [frame(.05, .1, -.5), frame(.10, .2, .1)], FakeGeometry())
        self.assertEqual(m["time_first_alongside_seconds"], .10)
        self.assertIsNone(m["time_to_pass_seconds"])
        self.assertFalse(m["early_pass"])
        stats = aggregate_metrics([{**result, "passing_metrics": m}])
        self.assertIsNone(stats["median_time_to_pass_seconds_successes_only"])
        self.assertEqual(stats["failed_or_censored_trials"], 1)

    def test_wraparound_progress_and_stopped_following(self):
        result = {"scenario": {"s": 19.9}, "success": True, "passed_at": .15}
        frames = [frame(.05, 19.95, -.5, 0.), frame(.10, .05, .1), frame(.15, .15, 1.1)]
        m = passing_metrics(result, frames, FakeGeometry())
        self.assertAlmostEqual(m["ego_progress_at_pass_m"], .25)
        self.assertAlmostEqual(m["track_fraction_to_pass"], .0125)
        self.assertAlmostEqual(m["stopped_behind_seconds"], .05)
        self.assertAlmostEqual(m["distance_travelled_while_behind_m"], .05)
        self.assertTrue(m["early_pass"])

    def test_collision_after_lead_is_not_success(self):
        result = {"scenario": {"s": 0}, "success": False, "passed_at": 1.}
        m = passing_metrics(result, [frame(1., 2., 1.5)], FakeGeometry())
        self.assertIsNone(m["time_to_pass_seconds"])


if __name__ == "__main__":
    unittest.main()
