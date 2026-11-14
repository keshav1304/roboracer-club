"""Partition integrity and outcome-label semantics for complete-rollout learning."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from maneuver_core import ACTIONS, FEATURE_NAMES
from rollout_training import (SECTORS, VALUE_FEATURE_NAMES, build_arrays, digest,
                               make_manifest, policy_rank, probability_metrics, select_checkpoint, select_jobs,
                               successful_winners, validate_manifest)


class RolloutTrainingTests(unittest.TestCase):
    def setUp(self):
        self.manifest = make_manifest(120., "reference-test", seed=81,
                                      train_cases=36, validation_cases=10, test_cases=30)

    def test_108_real_training_jobs_and_disjoint_sealed_encounters(self):
        for stage in (2, 3):
            jobs = select_jobs(self.manifest,stage,"train")
            self.assertEqual(len(jobs),108)
            self.assertEqual(len({j['case_id'] for j in jobs}),36)
            self.assertEqual(len(select_jobs(self.manifest,stage,'validation')),30)
            self.assertEqual(len(select_jobs(self.manifest,stage,'test',policy='neural')),30)
            for split in SECTORS:
                scenes = [j for j in self.manifest['jobs'] if j['stage']==stage and j['split']==split]
                self.assertTrue(all(j['start_sector'] in SECTORS[split] for j in scenes))
        scene_splits = {}
        for j in self.manifest['jobs']:
            scene_splits.setdefault(j['scenario_digest'],set()).add(j['split'])
        self.assertTrue(all(len(v)==1 for v in scene_splits.values()))
        changed = copy.deepcopy(self.manifest)
        changed['jobs'][0]['scenario']['gap'] += .1
        with self.assertRaisesRegex(ValueError,'sealed'):
            validate_manifest(changed)

    def test_even_resealed_sensor_leak_or_wrong_sector_rejected(self):
        changed = copy.deepcopy(self.manifest)
        training = next(j for j in changed['jobs'] if j['split']=='train')
        test = next(j for j in changed['jobs'] if j['split']=='test')
        test['scenario']['seed'] = training['scenario']['seed']
        test['scenario_digest'] = digest(test['scenario'])
        changed['manifest_digest'] = digest({k:v for k,v in changed.items() if k!='manifest_digest'})
        with self.assertRaisesRegex(ValueError,'overlap'):
            validate_manifest(changed)
        with self.assertRaisesRegex(ValueError,'frozen neural'):
            select_jobs(self.manifest,2,'test',policy='teacher')

    def test_speed_and_gap_strata_vary_within_every_training_sector(self):
        self.assertEqual(self.manifest['warmup_seconds'],.4)
        self.assertTrue(all(j['scenario']['warmup_seconds']==.4 for j in self.manifest['jobs']))
        for split in SECTORS:
            for sector in SECTORS[split]:
                stage2 = [j for j in self.manifest['jobs'] if j['stage']==2 and j['split']==split
                          and j['start_sector']==sector]
                groups = {0 if j['scenario']['opponent_speed']<1.8 else
                          1 if j['scenario']['opponent_speed']<2.1 else 2 for j in stage2}
                self.assertEqual(groups,{0,1,2})
                stage3 = [j for j in self.manifest['jobs'] if j['stage']==3 and j['split']==split
                          and j['start_sector']==sector]
                self.assertEqual({j['scenario']['gap']<1.3 for j in stage3},{True,False})
                self.assertEqual({j['scenario_group'] for j in stage3},{0,1,2})

    def test_expanded_round_has_102_distinct_encounters_and_306_rollouts_per_stage(self):
        manifest=make_manifest(120.,'reference-test',seed=20261122,train_cases=102)
        self.assertEqual(manifest['train_cases_per_stage'],102)
        self.assertEqual(manifest['training_rollouts_per_stage'],306)
        self.assertEqual(manifest['stage3_gap_ranges_m'],{'close':[.70,1.05],'approach':[1.3,2.]})
        self.assertGreater(manifest['stage3_gap_ranges_m']['close'][0],.63)
        for stage in (2,3):
            training=select_jobs(manifest,stage,'train')
            self.assertEqual(len(training),306)
            self.assertEqual(len({j['case_id'] for j in training}),102)
            self.assertEqual(len({j['scenario']['seed'] for j in training}),102)
            self.assertEqual(len({j['scenario']['s'] for j in training}),102)
            self.assertEqual(len(select_jobs(manifest,stage,'validation')),30)
            self.assertEqual(len(select_jobs(manifest,stage,'test',policy='neural')),30)
            for sector in SECTORS['train']:
                self.assertEqual({j['scenario_group'] for j in training if j['start_sector']==sector},{0,1,2})
        for job in manifest['jobs']:
            if job['stage']==3:
                low,high=manifest['stage3_gap_ranges_m']['close' if job['scenario_group'] else 'approach']
                self.assertTrue(low<=job['scenario']['gap']<=high)
        validate_manifest(manifest)

    def test_resealed_overlap_or_out_of_range_gap_is_rejected(self):
        manifest=copy.deepcopy(self.manifest)
        manifest['stage3_gap_ranges_m']['close'][0]=.63
        manifest['manifest_digest']=digest({k:v for k,v in manifest.items() if k!='manifest_digest'})
        with self.assertRaisesRegex(ValueError,'starting separation'):
            validate_manifest(manifest)
        manifest=copy.deepcopy(self.manifest)
        job=next(j for j in manifest['jobs'] if j['stage']==3 and j['scenario_group'])
        job['scenario']['gap']=1.2
        job['scenario_digest']=digest(job['scenario'])
        manifest['manifest_digest']=digest({k:v for k,v in manifest.items() if k!='manifest_digest'})
        with self.assertRaisesRegex(ValueError,'gap stratum'):
            validate_manifest(manifest)

    def test_warmup_is_sealed_bounded_and_a_whole_control_period(self):
        for duration in (-.05,.45,.025,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                make_manifest(120.,'reference-test',warmup_seconds=duration)
        with self.assertRaisesRegex(ValueError,'deadline'):
            make_manifest(120.,'reference-test',seconds=.4,warmup_seconds=.4)
        for duration in (0.,.2):
            manifest=make_manifest(120.,'reference-test',warmup_seconds=duration)
            self.assertTrue(all(j['scenario']['warmup_seconds']==duration for j in manifest['jobs']))
        changed=copy.deepcopy(self.manifest)
        job=changed['jobs'][0]
        job['scenario']['warmup_seconds']=.2
        job['scenario_digest']=digest(job['scenario'])
        changed['manifest_digest']=digest({k:v for k,v in changed.items() if k!='manifest_digest'})
        with self.assertRaisesRegex(ValueError,'Scenario warmup'):
            validate_manifest(changed)

    def record(self, directory, *, trial_id, case_id, action, success, passed_at=None, split='train', samples=None):
        x=np.zeros(len(FEATURE_NAMES),np.float32)
        x[FEATURE_NAMES.index('target_present')]=1.
        x[FEATURE_NAMES.index('target_distance')]=.3
        sample=dict(features=x.tolist(),action=action,t=.3,ego_s=1.)
        path=Path(directory)/(trial_id+'.samples.jsonl')
        path.write_text(''.join(json.dumps(r)+'\n' for r in ([sample] if samples is None else samples)))
        return dict(job=dict(stage=2,split=split,trial_id=trial_id,case_id=case_id),
                    result=dict(success=success,collision=None,passed_at=passed_at,
                                outcome='pass' if success else 'no_pass'),samples=str(path),
                    manifest_digest=self.manifest['manifest_digest'],reference_digest='reference-test')

    def test_failed_actions_only_label_failure_not_a_fictitious_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            fast=self.record(tmp,trial_id='a-left',case_id='a',action='LEFT',success=True,passed_at=5.)
            slow=self.record(tmp,trial_id='a-right',case_id='a',action='RIGHT',success=True,passed_at=9.)
            failed=self.record(tmp,trial_id='b-right',case_id='b',action='RIGHT',success=False)
            arrays=build_arrays([fast,slow,failed],self.manifest,2,'train')
            np.testing.assert_array_equal(arrays['imitation_y'],[ACTIONS.index('LEFT')])
            np.testing.assert_array_equal(arrays['value_y'],[1,1,0])
            np.testing.assert_array_equal(arrays['value_X'][-1,-4:],[0,1,0,0])
            self.assertEqual(arrays['successful_winning_encounters'],1)
            self.assertEqual(VALUE_FEATURE_NAMES[-4:],['action_left','action_right','action_wait','action_probe'])

    def test_postpass_no_target_and_wrong_sector_frames_never_enter_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            good=self.record(tmp,trial_id='safe',case_id='one',action='LEFT',success=True,passed_at=5.)
            row=json.loads(Path(good['samples']).read_text())
            holdout={**row,'ego_s':15.}
            postpass={**row,'t':5.1}
            gone=copy.deepcopy(row);gone['features'][FEATURE_NAMES.index('target_present')]=0.
            passed=copy.deepcopy(row);passed['features'][FEATURE_NAMES.index('target_distance')]=-.1
            Path(good['samples']).write_text(''.join(json.dumps(r)+'\n' for r in (row,holdout,postpass,gone,passed)))
            arrays=build_arrays([good],self.manifest,2,'train')
            self.assertEqual(len(arrays['value_y']),1)
            self.assertEqual(len(arrays['imitation_y']),1)
            with self.assertRaisesRegex(ValueError,'Test data'):
                build_arrays([good],self.manifest,2,'test')
            bad=copy.deepcopy(good);bad['job']['split']='test'
            with self.assertRaisesRegex(ValueError,'partitions'):
                build_arrays([bad],self.manifest,2,'train')

    def test_prelude_contact_excludes_only_unattributable_supervision(self):
        with tempfile.TemporaryDirectory() as tmp:
            before=self.record(tmp,trial_id='before-policy',case_id='one',action='LEFT',success=False)
            after=self.record(tmp,trial_id='after-policy',case_id='two',action='RIGHT',success=False)
            for record,when in ((before,.39),(after,.44)):
                row=json.loads(Path(record['samples']).read_text());row['t']=.4
                Path(record['samples']).write_text(json.dumps(row)+'\n')
                record['result'].update(warmup_seconds=.4,collision={'time':when,'type':'body_contact'},outcome='collision')
            raw=copy.deepcopy([before,after])
            arrays=build_arrays([before,after],self.manifest,2,'train')
            np.testing.assert_array_equal(arrays['value_y'],[0])
            self.assertEqual(arrays['provenance'][0]['trial_id'],'after-policy')
            self.assertEqual(len(arrays['excluded_supervision_rows']),1)
            self.assertEqual(arrays['excluded_supervision_rows'][0]['trial_id'],'before-policy')
            self.assertEqual([before,after],raw)
            self.assertEqual(sum(r['result']['collision'] is not None for r in raw),2)

    def test_success_primary_then_contacts_and_time_for_selection(self):
        def report(passes, collisions, seconds):
            return dict(outcomes=dict(passes=passes,collisions=collisions,mean_success_time_s=seconds))
        self.assertGreater(policy_rank(report(8,0,12.)),policy_rank(report(7,0,3.)))
        self.assertGreater(policy_rank(report(8,0,12.)),policy_rank(report(8,1,3.)))
        self.assertGreater(policy_rank(report(8,0,7.)),policy_rank(report(8,0,12.)))
        with tempfile.TemporaryDirectory() as tmp:
            record=self.record(tmp,trial_id='collision',case_id='x',action='LEFT',success=True,passed_at=2.)
            record['result']['collision']={'time':3.}
            self.assertEqual(successful_winners([record]),[])

    def test_binary_outcome_calibration_reports_all_labels(self):
        perfect=probability_metrics([0.,1.,1.,0.],[0,1,1,0])
        self.assertEqual(perfect['brier_score'],0.)
        self.assertEqual(perfect['expected_calibration_error_10_bins'],0.)
        self.assertEqual(perfect['pass_samples'],2)
        uncertain=probability_metrics([.5,.5],[0,1])
        self.assertEqual(uncertain['brier_score'],.25)

    def test_checkpoint_selection_rejects_contacts_before_ranking_success(self):
        def fake_collect(root, manifest, stage, split, **kw):
            name=kw['tag'].removeprefix('selection_')
            passes, contacts, seconds={'early':(9,1,4.),'mid':(6,0,7.),'final':(7,0,10.)}[name]
            return dict(model_file=str(kw['model']),outcomes=dict(passes=passes,collisions=contacts,mean_success_time_s=seconds))
        with tempfile.TemporaryDirectory() as tmp, patch('rollout_training.collect',side_effect=fake_collect):
            selected=select_checkpoint(tmp,self.manifest,2,include_imitation_only=False)
            self.assertEqual(selected['selected'],'final')
            self.assertTrue(selected['qualified'])

    def test_no_checkpoint_deployed_when_all_make_contact(self):
        def fake_collect(*args, **kw):
            return dict(model_file=str(kw['model']),outcomes=dict(passes=9,collisions=1,mean_success_time_s=4.))
        with tempfile.TemporaryDirectory() as tmp, patch('rollout_training.collect',side_effect=fake_collect):
            destination=Path(tmp)/'never.npz'
            with self.assertRaisesRegex(RuntimeError,'no zero-contact'):
                select_checkpoint(tmp,self.manifest,2,include_imitation_only=False,deploy=destination)
            self.assertFalse(destination.exists())
            self.assertFalse(json.loads((Path(tmp)/'stage2.selection.json').read_text())['qualified'])


if __name__=='__main__':
    unittest.main()
