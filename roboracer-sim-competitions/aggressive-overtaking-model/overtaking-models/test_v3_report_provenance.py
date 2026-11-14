"""Predecessor reporting cannot weaken current test provenance or double bill advice."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import v3_report as report


class PredecessorReportTests(unittest.TestCase):
    def test_common_preview_setup_is_explicitly_stage3_only(self):
        text=report.setup_description(True)
        self.assertTrue(text.startswith('In Stage 3 only,'))
        self.assertIn('Stage 2 retains its original physical setup.',text)
        self.assertNotIn('1.2 m',report.setup_description(False))

    def test_serial_timing_is_omitted_without_matching_deployed_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'benchmark_v3.json'
            manifest={'reference_digest':'ref','control_dt':.05}
            self.assertIsNone(report.verified_serial_benchmark(path,manifest))
            self.assertEqual(report.serial_benchmark_section({}),'')
            models=root/'models';models.mkdir()
            results={}
            for stage in ('2','3'):
                model=models/f'stage{stage}.npz';model.write_bytes(stage.encode())
                value=models/f'stage{stage}.value.npz';value.write_bytes(('value'+stage).encode())
                results[stage]={'model_sha256':report.file_hash(model),'value_model_sha256':report.file_hash(value),
                    'deployed_network':{'count':240,'p95_ms':.021},'full_controller':{'count':240,'p95_ms':3.5}}
            sources={'run_experiment.py':'current'}
            data={'source_sha256':sources,'reference_digest':'ref',
                  'benchmark_sha256':report.file_hash(Path(report.__file__).with_name('benchmark_v3.py')),
                  'network_access':'blocked','control_period_ms':50,'physics_period_ms':10,
                  'scenes':[{'samples':700,'warmup_samples':100}]*2,'results':results}
            with patch.object(report,'MODELS',models),patch.object(report,'sources',return_value=sources):
                path.write_text(json.dumps(data));verified=report.verified_serial_benchmark(path,manifest)
                self.assertIsNotNone(verified)
                section=report.serial_benchmark_section({'serial_benchmark':verified})
                self.assertIn('0.021 ms',section);self.assertIn('3.500 ms',section)
                self.assertIn('two fixed recorded scenes',section);self.assertIn('every 50 ms',section)
                for change in ('source','reference','model','value','count','timing','network','script'):
                    bad=copy.deepcopy(data)
                    if change=='source':bad['source_sha256']={'run_experiment.py':'old'}
                    elif change=='reference':bad['reference_digest']='other'
                    elif change=='model':bad['results']['2']['model_sha256']='old'
                    elif change=='value':bad['results']['3']['value_model_sha256']='old'
                    elif change=='count':bad['results']['3']['full_controller']['count']=239
                    elif change=='timing':bad['results']['2']['deployed_network']['p95_ms']=-1
                    elif change=='network':bad['network_access']='allowed'
                    else:bad['benchmark_sha256']='old'
                    path.write_text(json.dumps(bad))
                    self.assertIsNone(report.verified_serial_benchmark(path,manifest),change)

    def test_zero_test_passes_explicitly_fail_overtaking_objective(self):
        status=report.passing_status(3,{'passes':0,'trials':10,'accuracy':1.,'qualified':True},
                                    {'passes':0,'trials':30,'collisions':0})
        self.assertIs(status['passing_objective_met'],False)
        self.assertEqual(status['headline'],'Stage 3 passing objective not met')
        self.assertEqual(status['statement'],'Overtaking not demonstrated in this evaluation.')

    def test_test_pass_does_not_rewrite_zero_validation_result(self):
        status=report.passing_status(3,{'passes':0,'trials':10},
                                    {'passes':1,'trials':30,'collisions':0})
        self.assertIs(status['passing_objective_met'],False)
        self.assertIn('not demonstrated during validation',status['statement'])
        self.assertIn('1/30 test passes',status['detail'])

    def test_sparse_passes_are_counts_not_a_reliability_claim(self):
        status=report.passing_status(3,{'passes':1,'trials':10},
                                    {'passes':1,'trials':30,'collisions':2})
        self.assertIsNone(status['passing_objective_met'])
        self.assertEqual(status['headline'],'Stage 3: 1/30 test passes observed')
        self.assertIn('2 test contacts',status['statement'])
        section=report.outcome_status_section({'stages':{'3':{
            'passing_status':status,'training':{'passes':3,'trials':306},
            'training_unique_winning_encounters':1,'training_unique_encounters':102}}})
        self.assertIn('3/306 rollouts',section)
        self.assertIn('1/102 independent encounters',section)
        self.assertIn('separate from',section)

    def collection(self, stage=2, split='train'):
        job={'trial_id':f'{stage}-{split}', 'stage':stage, 'split':split}
        manifest={'manifest_digest':'old-manifest','reference_digest':'reference','jobs':[job]}
        record={'job':job,'manifest_digest':'old-manifest','reference_digest':'reference',
                'source_sha256':{'run_experiment.py':'old'}}
        collection={'completed':True,'completed_trials':1,'trials':[record],
                    'manifest_digest':'old-manifest','reference_digest':'reference',
                    'source_sha256':{'run_experiment.py':'old'}}
        return manifest,collection

    def test_predecessor_is_limited_to_stage2_training(self):
        old={'run_experiment.py':'old'}
        with patch.object(report,'sources',return_value={'run_experiment.py':'current'}):
            manifest,data=self.collection()
            report.verify_collection(data,manifest,2,'train',1,predecessor_sources=old)
            with self.assertRaisesRegex(RuntimeError,'source changed'):
                report.verify_collection(data,manifest,2,'train',1)
            for stage,split in ((2,'test'),(3,'train'),(3,'validation')):
                manifest,data=self.collection(stage,split)
                with self.assertRaisesRegex(RuntimeError,'allowed only'):
                    report.verify_collection(data,manifest,stage,split,1,predecessor_sources=old)

    def test_predecessor_does_not_allow_mixed_collection_versions(self):
        manifest,data=self.collection()
        data['trials'][0]['source_sha256']={'run_experiment.py':'different'}
        with self.assertRaisesRegex(RuntimeError,'Mixed controller'):
            report.verify_collection(data,manifest,2,'train',1,
                                     predecessor_sources={'run_experiment.py':'old'})

    def selection_fixture(self):
        path=Path('/private/tmp/frozen-stage2.early.npz')
        value=str(path.with_name(path.stem+'.value.npz').resolve())
        old={'completed':True,'policy':'neural','split':'validation','completed_trials':10,
             'manifest_digest':'old-manifest','reference_digest':'ref',
             'source_sha256':{'run_experiment.py':'old'},'model_sha256':'model',
             'model_file':str(path),'model_artifacts':{value:'head'}}
        current={**old,'manifest_digest':'new-manifest','source_sha256':{'run_experiment.py':'current'}}
        selection={'selected':'early','reports':{'early':old},'deployed_value_head':True}
        manifest={'manifest_digest':'new-manifest','reference_digest':'ref','validation_cases_per_stage':10}
        test={'stage':2,'source_sha256':current['source_sha256'],'model_sha256':'model',
              'trials':[{'result':{'value_model_sha256':'head'}}]}
        predecessor={'selected_checkpoint':'early','predecessor_manifest':{
            'manifest_digest':'old-manifest','reference_digest':'ref'},
            'current_validation_reports':{'early':current},'selected_model_sha256':'model',
            'selected_value_sha256':'head'}
        return selection,test,manifest,predecessor

    def test_current_equivalent_validation_is_displayed_without_reselection(self):
        selection,test,manifest,predecessor=self.selection_fixture()
        chosen=report.verify_selected_checkpoint(selection,test,manifest,predecessor=predecessor)
        self.assertEqual(chosen['source_sha256'],test['source_sha256'])
        self.assertEqual(selection['reports']['early']['manifest_digest'],'old-manifest')
        with self.assertRaisesRegex(RuntimeError,'matching validation'):
            report.verify_selected_checkpoint(selection,test,manifest)

    def test_cross_version_selection_refuses_other_stage_model_or_head(self):
        for mutation in ('stage','model','head','manifest','current_source'):
            selection,test,manifest,predecessor=self.selection_fixture()
            if mutation=='stage':test['stage']=3
            elif mutation=='model':predecessor['current_validation_reports']['early']['model_sha256']='changed'
            elif mutation=='head':predecessor['selected_value_sha256']='changed'
            elif mutation=='manifest':selection['reports']['early']['manifest_digest']='changed'
            else:predecessor['current_validation_reports']['early']['source_sha256']={'run_experiment.py':'old'}
            with self.assertRaises(RuntimeError):
                report.verify_selected_checkpoint(selection,test,manifest,predecessor=predecessor)

    def test_imported_reviews_are_not_new_calls_or_new_cost(self):
        def ledger(imported,cost):
            return {'advice_consumed':{'rows_consumed':10},'new_api_requests_this_round':0 if imported else 10,
                    'new_advice_rows_this_round':0 if imported else 10,'imported_advice_rows':10 if imported else 0,
                    'new_estimated_usage_cost_usd':0. if imported else cost,
                    'imported_estimated_usage_cost_usd':cost if imported else 0.}
        result=report.llm_accounting({'2':{'llm_training':ledger(True,.04)},
                                     '3':{'llm_training':ledger(False,.06)}})
        self.assertEqual(result['new_api_calls'],10)
        self.assertEqual(result['total_llm_advice_rows_consumed'],20)
        self.assertEqual(result['fresh_llm_advice_rows_consumed'],10)
        self.assertEqual(result['imported_llm_advice_rows_consumed'],10)
        self.assertAlmostEqual(result['estimated_api_usage_cost_usd'],.06)
        self.assertAlmostEqual(result['imported_api_usage_cost_usd'],.04)

    def test_page_explains_reused_training_without_claiming_reselection(self):
        data={'stages':{'2':{'data_provenance':{'training_imported_unchanged':True,
                                              'current_runtime_validation_trials':40}}},
              'total_llm_advice_rows_consumed':20,'fresh_llm_advice_rows_consumed':10,
              'imported_llm_advice_rows_consumed':10,'new_api_calls':10,
              'estimated_api_usage_cost_usd':.02,'imported_api_usage_cost_usd':.01,
              'stage3_setup_semantics':'Shared speed only during setup.'}
        text=report.provenance_section(data)
        self.assertIn('40 validation replays matched',text)
        self.assertIn('not reselected or retrained',text)
        self.assertIn('10 preserved with Stage 2',text)
        self.assertIn('This round made 10 review requests',text)
        self.assertNotIn('This round made 20',text)

    def test_request_accounting_uses_exact_questions_and_keeps_failed_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'reviews').mkdir()
            request={'review_type':'closed_loop_training_v1','stage':2,'manifest_digest':'m','split':'train','case':'selected'}
            other={**request,'case':'unrelated'}
            cache=root/'ledger.jsonl'
            entries=[{'scenario':request,'cache_key':'selected','wall_timestamp':1,'status':'rejected','estimated_usage_cost_usd':.01},
                     {'scenario':request,'cache_key':'selected','wall_timestamp':2,'status':'accepted','estimated_usage_cost_usd':.02},
                     {'scenario':other,'cache_key':'other','wall_timestamp':3,'status':'accepted','estimated_usage_cost_usd':100.}]
            cache.write_text('\n'.join(json.dumps(r)for r in entries))
            sources={'run_experiment.py':'old'}
            row={'stage':2,'split':'train','manifest_digest':'m','reference_digest':'ref',
                 'source_sha256':sources,'request':request,'cache_path':str(cache)}
            advice=root/'reviews/stage2.llm.jsonl';advice.write_text(json.dumps(row)+'\n')
            (root/'reviews/stage2.questions.json').write_text(json.dumps([{'request':request}]))
            fit={'metadata':{'source_sha256':sources},'fresh_llm_advice':{
                'rows_consumed':1,'file':str(advice),'sha256':report.file_hash(advice)}}
            with patch.object(report,'ARTIFACTS',root):
                result=report.llm_provenance(root,{'manifest_digest':'m','reference_digest':'ref'},2,fit,imported=True)
            self.assertEqual(result['api_requests_recorded'],2)
            self.assertAlmostEqual(result['estimated_usage_cost_usd'],.03)
            self.assertEqual(result['new_api_requests_this_round'],0)
            self.assertAlmostEqual(result['imported_estimated_usage_cost_usd'],.03)
            altered=copy.deepcopy(fit);altered['metadata']['source_sha256']={'run_experiment.py':'current'}
            with self.assertRaisesRegex(RuntimeError,'training provenance'):
                report.llm_provenance(root,{'manifest_digest':'m','reference_digest':'ref'},2,altered,imported=True)


if __name__=='__main__':
    unittest.main()
