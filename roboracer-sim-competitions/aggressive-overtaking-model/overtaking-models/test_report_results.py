"""Report provenance and credit accounting are checked before publication."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from report_results import api_usage, condition_row, validate_evidence


class ReportTests(unittest.TestCase):
    def test_rejected_billed_response_is_included_in_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reviews.jsonl'
            records = [dict(status='accepted', usage=dict(input_tokens=100, output_tokens=20), estimated_usage_cost_usd=.01),
                       dict(status='rejected', usage=dict(input_tokens=50, output_tokens=10), estimated_usage_cost_usd=.005)]
            path.write_text('\n'.join(json.dumps(r) for r in records))
            result = api_usage([path])
            self.assertEqual(result['accepted_responses'], 1)
            self.assertEqual(result['recorded_requests'], 2)
            self.assertEqual(result['input_tokens'], 150)
            self.assertAlmostEqual(result['estimated_usage_cost_usd'], .015)

    def test_incomplete_evaluation_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'incomplete'):
            validate_evidence({1: dict(results=[{}], requested_trials=12, evaluation_complete=False)}, {}, {})

    def test_stale_model_refused(self):
        report = dict(results=[dict(model_sha256='old', sensor_seeds=[1,2])], requested_trials=1, evaluation_complete=True)
        with self.assertRaisesRegex(RuntimeError, 'mismatch'):
            validate_evidence({1: report}, {}, {1: 'new'})

    def test_empty_condition_does_not_invent_clearance(self):
        row = condition_row('absent condition', [])
        self.assertEqual(row['trials'], 0)
        self.assertIsNone(row['min_body_gap'])
        self.assertIsNone(row['median_time_to_pass_seconds_successes_only'])


if __name__ == '__main__':
    unittest.main()
