"""The optional old/new report comparison needs matched frozen evidence."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import v3_report as report


class BaselineReportTests(unittest.TestCase):
    def fixture(self, root):
        directory=root/'baseline_stage2_evaluation';directory.mkdir()
        models=root/'models';models.mkdir();archive=root/'archive';archive.mkdir()
        def artifact(path,data):
            path.write_text(data);return report.file_hash(path)
        source={name:artifact(root/name,'current'+name) for name in
                ('run_experiment.py','maneuver_core.py','perception.py','track_geometry.py','world.py')}
        core=archive/'maneuver_core.py';oldmodel=archive/'stage2.npz';oldhead=archive/'stage2.value.npz'
        artifacts={str(p):artifact(p,'old'+p.name)for p in (core,oldmodel,oldhead)}
        model=models/'stage2.npz';head=models/'stage2.value.npz'
        modelhash=artifact(model,'current model');headhash=artifact(head,'current head')
        wrapper=artifact(directory/'evaluation_runner.py','wrapper')
        docs={};jobs=[];records=[];pairs=[];items=[];old_results=[]
        for i in range(30):
            key=f'trial-{i:02d}';scenario={'stage':2,'seed':i,'opponent':'line'}
            job={'trial_id':key,'stage':2,'split':'test','scenario':scenario,'scenario_digest':f's{i}'};jobs.append(job)
            success=i<8;t=9.5+i/8 if success else None
            result={'scenario':scenario,'outcome':'pass'if success else'no_pass','success':success,'passed_at':t,
                    'collision':None,'minimum_body_gap':.1,'minimum_wall_clearance':.1,'pass_followthrough_complete':success,
                    'source_sha256':source,'model_sha256':modelhash,'value_model_sha256':headhash,
                    'teacher':False,'llm_calls_in_test':0,'network_access':'blocked','sensor_seeds':[i,i+1],
                    'control_dt_seconds':.05,'physics_dt_seconds':.01,'warmup_seconds':.4,'reference_sha256':'reference-xy'}
            finalpath=root/f'{key}.json';finaltrace=root/f'{key}.trace.json'
            rec={'job':job,'result':result,'trace':str(finaltrace),'model_artifacts':{str(model):modelhash,str(head):headhash},
                 'manifest_digest':'manifest','reference_digest':'ref','source_sha256':source}
            docs[finalpath]=rec;docs[finaltrace]={'result':result,'frames':[{}]};records.append(rec)
            old=copy.deepcopy(result);old_results.append(old)
            old.update(passed_at=t+.5 if success else None,source_sha256={**source,'maneuver_core.py':artifacts[str(core)]},
                       model_sha256=artifacts[str(oldmodel)],value_model_sha256=artifacts[str(oldhead)],used_for_tuning=False,
                       post_freeze_evaluation=True,held_out_test=True,evaluation_role='predeclared_archived_baseline',
                       paired_final_trial_id=key,scenario_digest=f's{i}',baseline_provenance={
                           'controller_core_path':str(core),'controller_core_sha256':artifacts[str(core)],
                           'model_path':str(oldmodel),'model_sha256':artifacts[str(oldmodel)],
                           'value_model_path':str(oldhead),'value_model_sha256':artifacts[str(oldhead)],
                           'episode_harness_sha256':source['run_experiment.py'],'wrapper_sha256':wrapper,
                           'harness_imported_current_core_sha256':source['maneuver_core.py'],
                           'shared_sources':{k:v for k,v in source.items()if k not in ('run_experiment.py','maneuver_core.py')},
                           'reference_digest':'ref'})
            oldpath=directory/f'{key}.json';oldtrace=directory/f'{key}.trace.json'
            docs[oldpath]={'original_job':job,'result':old,'trace':str(oldtrace),'paired_final_record':str(finalpath),'completed':True}
            docs[oldtrace]={'result':old,'frames':[{}]}
            fields=('outcome','success','passed_at','collision','minimum_body_gap','minimum_wall_clearance','pass_followthrough_complete')
            pair={'trial_id':key,'baseline':{k:old[k]for k in fields},'final':{k:result[k]for k in fields},
                  'baseline_record':str(oldpath),'baseline_trace':str(oldtrace)};pairs.append(pair)
            items.append({'trial_id':key,'scenario':scenario,'scenario_digest':f's{i}','job':job,'record_path':str(finalpath),'final_result':pair['final']})
        manifest={'jobs':jobs,'manifest_digest':'manifest','reference_digest':'ref','control_dt':.05}
        current={'completed':True,'completed_trials':30,'trials':records,'manifest_digest':'manifest','reference_digest':'ref','source_sha256':source}
        plan={'baseline_artifacts':artifacts,'harness_sha256':source['run_experiment.py'],'wrapper_sha256':wrapper,'control_dt_seconds':.05,'items':items}
        comparison={'provenance':plan,'paired_outcomes':pairs,'new_only_passes':[],'baseline_only_passes':[],
                    'paired_success_time_change':{'same_scenarios_passed':8,'individual_seconds_saved':{f'trial-{i:02d}':.5 for i in range(8)},
                                                 'mean_seconds_saved':.5,'all_common_passes_faster_in_final':True}}
        for key,results in (('baseline',old_results),('final',[r['result']for r in records])):
            comparison[key]={'runs':30,'passes':8,'contacts':0,'median_successful_pass_seconds':float(np.median([r['passed_at']for r in results if r['success']]))}
        docs[directory/'comparison.json']=comparison;docs[directory/'predeclared_manifest.json']=plan
        (directory/'comparison.json').write_text('comparison fixture')
        return models,source,docs,manifest,current,old_results

    def test_matched_comparison_is_reported_without_success_rate_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);models,source,docs,manifest,current,old=self.fixture(root)
            with patch.object(report,'MODELS',models),patch.object(report,'sources',return_value=source),patch.object(report,'read_json',side_effect=lambda p:docs[Path(p)]):
                result=report.verified_stage2_baseline(root,manifest,current)
            self.assertEqual(result['baseline']['passes'],8);self.assertEqual(result['final']['passes'],8)
            self.assertEqual(result['common_successes_faster'],8)
            text=report.baseline_comparison_section({'fixed_stage2_baseline_comparison':result})
            self.assertIn('No unseen success-rate improvement demonstrated',text)
            self.assertIn('cannot be attributed solely to neural learning',text)
            self.assertIn('exactly the same 8 scenarios',text)

    def test_stale_source_model_scenario_or_summary_is_omitted(self):
        for change in ('source','model','scenario','missing','summary','wrapper','tuning','trace'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);models,source,docs,manifest,current,old=self.fixture(root)
                if change=='source':old[0]['source_sha256']['perception.py']='old'
                elif change=='model':old[0]['model_sha256']='wrong'
                elif change=='scenario':old[0]['scenario']['seed']=-1
                elif change=='missing':docs[root/'baseline_stage2_evaluation/comparison.json']['paired_outcomes'].pop()
                elif change=='summary':docs[root/'baseline_stage2_evaluation/comparison.json']['final']['passes']=9
                elif change=='wrapper':(root/'baseline_stage2_evaluation/evaluation_runner.py').write_text('changed')
                elif change=='tuning':old[0]['used_for_tuning']=True
                else:docs[root/'baseline_stage2_evaluation/trial-00.trace.json']={'result':{}}
                with patch.object(report,'MODELS',models),patch.object(report,'sources',return_value=source),patch.object(report,'read_json',side_effect=lambda p:docs[Path(p)]):
                    self.assertIsNone(report.verified_stage2_baseline(root,manifest,current))
        self.assertEqual(report.baseline_comparison_section({}),'')


if __name__=='__main__':
    unittest.main()
