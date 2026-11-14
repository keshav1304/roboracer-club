"""Pure offline analysis checks; no simulator or final-test records are read."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analyze_round_metrics import (decision_audit,manifest_audit,summarize_trials,
                                   test_trial_metrics,validation_comparison,wilson_interval)
from maneuver_core import ACTIONS,FEATURE_NAMES
from rollout_training import digest,make_manifest,sources


class RoundMetricsTests(unittest.TestCase):
    def test_manifest_counts_and_resealed_different_paired_encounter(self):
        manifest=make_manifest(120.,'reference',train_cases=102)
        audit=manifest_audit(manifest)
        self.assertEqual(audit['unique_encounters'],284)
        self.assertEqual(audit['unique_car_sensor_seeds'],568)
        altered=copy.deepcopy(manifest)
        job=altered['jobs'][1];job['scenario']=dict(job['scenario'])
        job['scenario']['gap']+=.01;job['scenario_digest']=digest(job['scenario'])
        altered['manifest_digest']=digest({k:v for k,v in altered.items() if k!='manifest_digest'})
        with self.assertRaisesRegex(ValueError,'Paired variants'):
            manifest_audit(altered)

    def trial(self,attempt=True):
        job=dict(trial_id='synthetic',case_id='synthetic',stage=2,start_sector=1,scenario_group=0,
                 scenario=dict(ego_speed=2.4,opponent_speed=1.6,gap=1.8))
        result=dict(warmup_seconds=.4,sim_seconds=.56,success=False,collision=None,passed_at=None,
                    minimum_body_gap=.04,minimum_body_wall_clearances=[.1,.08],action_seconds={'WAIT':.16},
                    physics_dt_seconds=.01,safety_mask_override_ticks=10,commitment_override_ticks=0)
        frames=[]
        for t,speed in ((.41,0.),(.46,0.),(.51,1.)):
            action='LEFT' if attempt and t>.5 else 'WAIT'
            frames.append(dict(t=t,ego=[0.,0.,0.,speed],relative_progress=-.5,target={'distance':.5},
                               action=action,probabilities=[.8,.1,.1],command=[.1,1.]))
        return dict(job=job,result=result),frames

    def test_attempt_and_stopping_use_actual_sample_times_and_keep_failures(self):
        record,frames=self.trial()
        row=test_trial_metrics(record,frames)
        self.assertAlmostEqual(row['first_attempt_seconds'],.51)
        self.assertAlmostEqual(row['first_attempt_after_policy_start_seconds'],.11)
        self.assertAlmostEqual(row['estimated_stopped_behind_seconds'],.1)
        self.assertAlmostEqual(row['measured_policy_coverage_seconds'],.15)
        self.assertIsNone(row['passed_at_seconds'])
        summary=summarize_trials([row])
        self.assertEqual(summary['passes'],0)
        self.assertIsNone(summary['successful_pass_seconds']['median'])
        self.assertEqual(summary['contact_free_trials_below_body_gap_m']['0.05'],1)
        self.assertEqual(summary['contact_free_trials_below_body_gap_m']['0.025'],0)
        record,frames=self.trial(False)
        self.assertEqual(summarize_trials([test_trial_metrics(record,frames)])['trials_without_recorded_attempt'],1)

    def test_binomial_uncertainty_is_per_encounter(self):
        low,high=wilson_interval(0,30)
        self.assertAlmostEqual(low,0.)
        self.assertGreater(high,.11)
        self.assertAlmostEqual(1-.05**(1/30),.0950338529)
        self.assertIsNone(wilson_interval(0,0))

    def test_decision_audit_matches_masked_probability_fallback(self):
        manifest=make_manifest(120.,'reference')
        row=dict(features=[0.]*len(FEATURE_NAMES),candidates=[
            dict(action='LEFT',feasible=True,estimated_time=1.,steering=.1,target_speed=2.),
            dict(action='RIGHT',feasible=False,estimated_time=30.,steering=-.1,target_speed=2.),
            dict(action='WAIT',feasible=True,estimated_time=2.,steering=0.,target_speed=.4)])
        class FakeModel:
            metadata=dict(reference_digest='reference',stage=2)
            classes=ACTIONS[:3]
            def predict_proba(self,_):
                return np.array([[0.,.9,.1],[2e-9,1.-2.001e-9,1e-12]])
        with tempfile.TemporaryDirectory() as tmp:
            model=Path(tmp)/'stage2.npz';model.write_bytes(b'analysis-only mock')
            with patch('analyze_round_metrics.load_collection',return_value={'trials':[{}]}), \
                 patch('analyze_round_metrics.eligible_samples',return_value=[row,row]), \
                 patch('analyze_round_metrics.TinyNetwork.load',return_value=FakeModel()):
                report=decision_audit(tmp,manifest,2,'validation',model)
        self.assertEqual(report['multiple_feasible_action_rows'],2)
        self.assertEqual(report['final_recommendation_vs_geometric_disagreement_on_choice_rows'],1.)
        self.assertEqual(report['different_candidate_control_hint_fraction_on_choice_rows'],1.)

    def test_paired_validation_comparison_counts_wins_and_joint_success_time(self):
        manifest=make_manifest(120.,'reference',validation_cases=2)
        def record(case,variant,passed):
            return dict(job=dict(case_id=case,variant=variant,scenario_digest=case),
                        result=dict(success=passed is not None,passed_at=passed,collision=None))
        teachers=dict(model_artifacts={},trials=[record('a','neutral',None),record('b','neutral',8.),
                                                record('a','left',5.),record('b','left',None)])
        learned=dict(completed=True,policy='neural',manifest_digest=manifest['manifest_digest'],
                     source_sha256=sources(),model_sha256='trained',model_artifacts={},
                     trials=[record('a','learned',4.),record('b','learned',None)])
        selection=dict(validation_only=True,qualified=True,selected='final',reports={'final':{'model_sha256':'trained'}})
        with patch('analyze_round_metrics.load_collection',return_value=teachers), \
             patch('analyze_round_metrics.read_json',side_effect=lambda path:selection if str(path).endswith('.selection.json') else learned):
            result=validation_comparison('/unused',manifest,2)
        neutral=result['comparisons']['neutral'];left=result['comparisons']['left']
        self.assertEqual(neutral['neural_only_pass'],1)
        self.assertEqual(neutral['teacher_only_pass'],1)
        self.assertEqual(left['both_pass'],1)
        self.assertEqual(left['neural_minus_teacher_pass_seconds_on_joint_successes']['mean'],-1.)
        self.assertTrue(result['no_final_test_data_used'])


if __name__=='__main__':unittest.main()
