"""Learning and API-provenance checks, with all HTTP calls mocked."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np

from tiny_network import TinyNetwork, fit_network
from llm_teacher import LLMTeacher, TeacherError, geometric_teacher, load_local_env


class NetworkTests(unittest.TestCase):
    def test_learning_serialization_and_schema(self):
        rng = np.random.default_rng(14)
        X = rng.normal(size=(450, 4)).astype(np.float32)
        y = np.where(X[:, 0] > .3, 0, np.where(X[:, 0] < -.3, 1, 2))
        names = ["gap_left", "gap_right", "speed", "heading"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            result = fit_network(X[:360], y[:360], path, validation=(X[360:], y[360:]),
                                 feature_names=names, epochs=100, hidden=(16, 12), seed=5)
            model = TinyNetwork.load(path, expected_feature_names=names)
            probs = model.predict_proba(X[360:])
            self.assertGreater(np.mean(probs.argmax(axis=1) == y[360:]), .92)
            np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6)
            np.testing.assert_allclose(model.proba(X[360]), probs[0], atol=1e-6)
            self.assertTrue(all(Path(p).exists() for p in result["checkpoints"].values()))
            with self.assertRaisesRegex(ValueError, "schema"):
                TinyNetwork.load(path, names[::-1])
            with self.assertRaises(ValueError):
                model.proba([np.nan] * 4)
            with np.load(path, allow_pickle=False) as checkpoint:
                self.assertEqual(checkpoint["w0"].dtype, np.float32)

    def test_soft_labels_and_transfer_to_four_actions(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(180, 3)).astype(np.float32)
        y = np.tile([.15, .7, .15], (len(X), 1))
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp) / "source.npz", Path(tmp) / "dest.npz"
            fit_network(X, y, source, epochs=30, hidden=(12, 8))
            model = TinyNetwork.load(source)
            np.testing.assert_allclose(model.predict_proba(X).mean(axis=0), y[0], atol=.06)
            targets = np.where(X[:, 0] > 0, 3, 1)
            fit_network(X, targets, dest, classes=("LEFT", "RIGHT", "WAIT", "PROBE"),
                        warmstart=source, epochs=150, hidden=(12, 8))
            extended = TinyNetwork.load(dest)
            self.assertEqual(extended.proba(X[0]).shape, (4,))
            self.assertGreater(np.mean(extended.predict_proba(X).argmax(axis=1) == targets), .90)

    def test_reject_bad_probability_targets(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            fit_network(np.zeros((4, 2)), np.ones((4, 3)), Path(tmp) / "bad.npz")

    def test_transfer_handles_newly_moving_target_feature(self):
        rng = np.random.default_rng(6)
        static = rng.normal(size=(300, 3)).astype(np.float32)
        static[:, 2] = 0.
        labels = np.where(static[:, 0] > 0, 0, 1)
        moving = static.copy()
        moving[:, 2] = rng.uniform(1., 2.4, len(moving))
        new_labels = np.where(moving[:, 2] > 1.7, 2, labels)
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp) / "static.npz", Path(tmp) / "moving.npz"
            fit_network(static, labels, source, epochs=50, hidden=(16, 12))
            result = fit_network(moving, new_labels, dest, warmstart=source,
                                 epochs=100, hidden=(16, 12))
            model = TinyNetwork.load(dest)
            self.assertGreater(model.scale[2], .2)
            self.assertGreater(result["train"]["accuracy"], .95)


class TeacherTests(unittest.TestCase):
    def setUp(self):
        self.scenario = {"max_speed": 3., "state": {"gap": 2.}, "candidates": [
            {"action": "LEFT", "feasible": True, "estimated_time": 3.1,
             "clearance": .3, "steering": .1, "target_speed": 2.},
            {"action": "RIGHT", "feasible": False, "estimated_time": 2.9,
             "clearance": -.1, "steering": -.1, "target_speed": 2.},
            {"action": "WAIT", "feasible": True, "estimated_time": 12.,
             "clearance": .5, "steering": 0., "target_speed": .5}]}

    def response(self, action="LEFT"):
        label = {"action": action, "confidence": .9, "reason": "Left has adequate room.",
                 "steering": .1, "target_speed": 2.}
        body = {"id": "test-response", "model": "test-model", "status": "completed",
                "usage": {"input_tokens": 20, "output_tokens": 30},
                "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(label)}]}]}
        return io.BytesIO(json.dumps(body).encode())

    def test_real_response_cache_and_bounded_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "labels.jsonl"
            teacher = LLMTeacher("test-model", cache, max_calls=1, api_key="test-only")
            with patch("urllib.request.urlopen", return_value=self.response()) as request:
                first = teacher.label(self.scenario)
                second = teacher.label(self.scenario)
                self.assertEqual(request.call_count, 1)
            self.assertEqual(first["source"], "openai_responses")
            self.assertTrue(first["llm_used"])
            self.assertFalse(first["rollout_verified"])
            self.assertTrue(second["cache_hit"])
            self.assertNotIn("test-only", cache.read_text())
            altered = {**self.scenario, "state": {"gap": 3.}}
            with self.assertRaisesRegex(TeacherError, "limit"):
                teacher.label(altered)
            reloaded = LLMTeacher("test-model", cache, max_calls=0, api_key="")
            self.assertTrue(reloaded.label(self.scenario)["cache_hit"])

    def test_cost_guard_blocks_before_network_and_tracks_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "labels.jsonl"
            tiny = LLMTeacher("gpt-4.1-mini", cache, max_calls=2, api_key="test-only", max_cost_usd=.00001)
            with patch("urllib.request.urlopen") as request:
                with self.assertRaisesRegex(TeacherError, "cost limit"):
                    tiny.label(self.scenario)
                request.assert_not_called()
            teacher = LLMTeacher("gpt-4.1-mini", cache, max_calls=2, api_key="test-only", max_cost_usd=1.)
            with patch("urllib.request.urlopen", return_value=self.response()):
                teacher.label(self.scenario)
            self.assertGreater(teacher.reserved_cost_usd, teacher.usage_cost_usd)
            self.assertAlmostEqual(teacher.usage_cost_usd, .000056)
            restored = LLMTeacher("gpt-4.1-mini", cache, max_calls=2, api_key="test-only", max_cost_usd=1.)
            self.assertEqual(restored.reserved_cost_usd, teacher.reserved_cost_usd)
            self.assertEqual(restored.usage_cost_usd, teacher.usage_cost_usd)

    def test_infeasible_llm_advice_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "labels.jsonl"
            teacher = LLMTeacher("test-model", cache, max_calls=1, api_key="test-only")
            with patch("urllib.request.urlopen", return_value=self.response("RIGHT")):
                with self.assertRaisesRegex(TeacherError, "infeasible"):
                    teacher.label(self.scenario)
            self.assertEqual(json.loads(cache.read_text())["status"], "rejected")
            self.assertFalse(teacher.cache)

    def test_missing_key_and_geometric_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            teacher = LLMTeacher("test-model", Path(tmp) / "labels.jsonl", max_calls=1, api_key="")
            with self.assertRaisesRegex(TeacherError, "not configured"):
                teacher.label(self.scenario)
            self.assertEqual(teacher.calls, 0)
            local = geometric_teacher(self.scenario)
            self.assertEqual(local["action"], "LEFT")
            self.assertEqual(local["source"], "geometric_bootstrap")
            self.assertFalse(local["llm_used"])

    def test_refusal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            teacher = LLMTeacher("test-model", Path(tmp) / "labels.jsonl", max_calls=1, api_key="test-only")
            body = {"output": [{"content": [{"type": "refusal", "refusal": "No response"}]}]}
            with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(body).encode())):
                with self.assertRaisesRegex(TeacherError, "declined"):
                    teacher.label(self.scenario)

    def test_quota_status_is_logged_without_raw_error_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "labels.jsonl"
            teacher = LLMTeacher("test-model", cache, max_calls=1, api_key="test-only")
            error = {"error": {"code": "credit_balance_exhausted", "type": "insufficient_quota",
                               "message": "do-not-log-this-raw-service-body"}}
            http_error = HTTPError("https://api.openai.com/v1/responses", 429, "quota", {},
                                   io.BytesIO(json.dumps(error).encode()))
            with patch("urllib.request.urlopen", side_effect=http_error):
                with self.assertRaisesRegex(TeacherError, "credit_balance_exhausted"):
                    teacher.label(self.scenario)
            self.assertNotIn("do-not-log-this", cache.read_text())
            self.assertEqual(json.loads(cache.read_text())["service_error"]["type"], "insufficient_quota")

    def test_env_loader_never_evaluates_shell(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            env = Path(tmp) / ".env"
            env.write_text("OPENAI_API_KEY='test-key'\nOPENAI_MODEL=gpt-test\nHOME=/wrong\nUNKNOWN=$(touch bad)\n")
            self.assertEqual(load_local_env(env), ["OPENAI_API_KEY", "OPENAI_MODEL"])
            self.assertEqual(os.environ["OPENAI_API_KEY"], "test-key")
            self.assertNotIn("HOME", os.environ)
            os.environ["OPENAI_MODEL"] = "existing"
            load_local_env(env)
            self.assertEqual(os.environ["OPENAI_MODEL"], "existing")
            del os.environ["OPENAI_API_KEY"]
            env.write_text("OPENAI_API_KEY=replace_with_your_key\n")
            self.assertEqual(load_local_env(env), [])
            self.assertNotIn("OPENAI_API_KEY", os.environ)


if __name__ == "__main__":
    unittest.main()
