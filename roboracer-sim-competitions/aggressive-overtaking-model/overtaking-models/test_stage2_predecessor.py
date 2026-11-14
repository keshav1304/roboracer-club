"""Compatibility must preserve exact behavior and immutable artifact identity."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rollout_training import file_hash,make_manifest,select_jobs
from stage2_predecessor import (STABLE_RESULT_FIELDS,behavior_digest,compare_validation_reports,
                                runtime_change,verify_deployed_pair,verify_import_inventory)


class Stage2PredecessorTests(unittest.TestCase):
    def trace(self,scenario=None):
        result={name:None for name in STABLE_RESULT_FIELDS}
        result.update(scenario=scenario or {},success=False,outcome='no_pass',model_sha256='unchanged-model',
                      value_model_sha256=None,teacher=False,network_access='blocked',collision=None,
                      source_sha256={'run_experiment.py':'old'},wall_seconds=1.,model_file='/old/path.npz')
        return dict(result=result,frames=[dict(t=.41,ego=[0.,0.,0.,1.],action='WAIT',probabilities=[.1,.1,.8])])

    def test_behavior_ignores_execution_metadata_but_not_controls_or_probabilities(self):
        old=self.trace();new=copy.deepcopy(old)
        new['result'].update(source_sha256={'run_experiment.py':'new'},wall_seconds=99.,model_file='/new/path.npz')
        self.assertEqual(behavior_digest(old),behavior_digest(new))
        new['frames'][0]['probabilities']=[.2,.1,.7]
        self.assertNotEqual(behavior_digest(old),behavior_digest(new))
        del new['result']['collision']
        with self.assertRaisesRegex(ValueError,'Missing stable'):
            behavior_digest(new)

    def test_only_declared_runtime_file_may_change(self):
        old={'run_experiment.py':'a','maneuver_core.py':'b'}
        self.assertEqual(runtime_change(old,{**old,'run_experiment.py':'c'}),['run_experiment.py'])
        with self.assertRaises(ValueError):runtime_change(old,{**old,'maneuver_core.py':'c'})
        with self.assertRaises(ValueError):runtime_change(old,old)

    def test_both_original_and_imported_bytes_must_remain_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            predecessor=Path(tmp)/'previous';imported=Path(tmp)/'imported'
            predecessor.mkdir();imported.mkdir()
            original=predecessor/'stage2.json';copied=imported/'stage2.json'
            original.write_text('original');copied.write_bytes(original.read_bytes())
            preparation=dict(predecessor_root=str(predecessor),imported_root=str(imported),
                             imported_files=[dict(relative_path='stage2.json',original_path=str(original.resolve()),sha256=file_hash(original))])
            verify_import_inventory(preparation)
            copied.write_text('rewritten')
            with self.assertRaisesRegex(ValueError,'artifact changed'):verify_import_inventory(preparation)
            copied.write_text('original');original.write_text('rewritten')
            with self.assertRaisesRegex(ValueError,'artifact changed'):verify_import_inventory(preparation)

    def test_optional_outcome_head_is_part_of_frozen_deployment(self):
        with tempfile.TemporaryDirectory() as tmp,patch('stage2_predecessor.MODELS',Path(tmp)):
            model=Path(tmp)/'stage2.npz';model.write_text('fixed')
            checksum=file_hash(model);verify_deployed_pair(checksum,None)
            value=Path(tmp)/'stage2.value.npz';value.write_text('new-head')
            with self.assertRaisesRegex(ValueError,'frozen deployed'):verify_deployed_pair(checksum,None)
            verify_deployed_pair(checksum,file_hash(value))

    def test_all_ten_paired_cases_must_match_before_certification(self):
        manifest=make_manifest(120.,'reference');current_sources={'run_experiment.py':'new'}
        previous=dict(trials=[]);current=dict(trials=[],completed=True,policy='neural',
            source_sha256=current_sources,manifest_digest=manifest['manifest_digest'])
        with tempfile.TemporaryDirectory() as tmp:
            for index,job in enumerate(select_jobs(manifest,2,'validation',policy='neural')):
                old=self.trace(job['scenario']);new=copy.deepcopy(old)
                new['result']['source_sha256']=current_sources
                for collection,trace,name in ((previous,old,'old'),(current,new,'new')):
                    path=Path(tmp)/f'{index}-{name}.json';path.write_text(json.dumps(trace))
                    collection['trials'].append(dict(job=job,trace=str(path),result=trace['result'],
                        source_sha256=current_sources,manifest_digest=manifest['manifest_digest']))
            matches=compare_validation_reports(previous,current,manifest,current_sources)
            self.assertEqual(len(matches),10)
            self.assertTrue(all(r['exact_behavior_match'] for r in matches))
            current['trials'].pop()
            with self.assertRaisesRegex(ValueError,'all ten'):
                compare_validation_reports(previous,current,manifest,current_sources)


if __name__=='__main__':unittest.main()
