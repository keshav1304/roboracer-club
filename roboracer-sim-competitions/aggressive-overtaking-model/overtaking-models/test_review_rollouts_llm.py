"""Training-only API advice provenance, isolation and actual loss consumption."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from maneuver_core import FEATURE_NAMES
from review_rollouts_llm import load_advisory_rows, review_examples, run_reviews
from rollout_training import fit_stage, make_manifest, sources
from tiny_network import TinyNetwork


class ReviewRolloutsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = make_manifest(120.,"new-reference",seed=7,train_cases=1,validation_cases=1,test_cases=1)
        (self.root/"manifest.json").write_text(json.dumps(self.manifest))
        self.training = self.records("train")
        self.validation = self.records("validation")

    def records(self, split):
        records = []
        jobs = [j for j in self.manifest["jobs"] if j["stage"]==2 and j["split"]==split]
        for i,job in enumerate(jobs):
            x=np.zeros(len(FEATURE_NAMES),np.float32)
            for key,value in dict(target_present=1.,target_distance=.3,target_speed=.5,target_confidence=.9).items():
                x[FEATURE_NAMES.index(key)]=value
            x[0]=i*.1
            action={"neutral":"WAIT","left":"LEFT","right":"RIGHT"}[job["variant"]]
            candidates=[dict(action=a,feasible=True,estimated_time=4.+j,clearance=.25,
                             steering=.1,target_speed=2.4) for j,a in enumerate(("LEFT","RIGHT","WAIT"))]
            samples=[dict(features=x.tolist(),action=action,candidates=candidates,stage=2,
                          seed=job["scenario"]["seed"],t=t,ego_s=job["scenario"]["s"],max_speed=2.4,
                          state=dict(distance=1.5,speed=1.5,confidence=.9)) for t in (.5,.6)]
            sample_path=self.root/(job["trial_id"]+".samples.jsonl")
            sample_path.write_text("".join(json.dumps(r)+"\n" for r in samples))
            success=action=="LEFT"
            records.append(dict(job=job,result=dict(success=success,collision=None,passed_at=4. if success else None,
                                outcome="pass" if success else "no_pass",sim_seconds=6. if success else 18.,
                                minimum_body_gap=.2,minimum_wall_clearance=.15),samples=str(sample_path),
                                completed=True,source_sha256=sources(),reference_digest="new-reference",
                                manifest_digest=self.manifest["manifest_digest"]))
        return records

    def advice(self):
        row=review_examples(self.training,self.manifest,2,limit=1)[0]
        label=dict(action="LEFT",confidence=.8,reason="The feasible inside candidate can complete earlier.",
                   steering=.1,target_speed=2.4,source="openai_responses",llm_used=True,rollout_verified=False,
                   response_id="mock-response",model="mock-model",cache_key="mock-key",cache_hit=False)
        cache=self.root/"mock-api-cache.jsonl"
        cache.write_text(json.dumps(dict(status="accepted",cache_key="mock-key",scenario=row["request"],label=label))+"\n")
        row.update(label=label,cache_path=str(cache),source="openai_responses",sample_weight=1.)
        path=self.root/"advice.jsonl"
        path.write_text(json.dumps(row)+"\n")
        return path,row

    def test_reviews_use_current_training_snapshots_and_actual_mixed_outcomes(self):
        examples=review_examples(self.training,self.manifest,2,limit=10)
        self.assertEqual(len(examples),1)
        row=examples[0]
        self.assertEqual(row["anchor_trial_id"],self.training[1]["job"]["trial_id"])
        outcomes=row["request"]["actual_counterfactual_policy_outcomes"]
        self.assertEqual(sum(r["success"] for r in outcomes),1)
        self.assertEqual(len(outcomes),3)
        self.assertEqual(row["reference_digest"],"new-reference")
        self.assertFalse(row["value_label_eligible"])
        with self.assertRaisesRegex(ValueError,"training partition"):
            review_examples(self.validation,self.manifest,2)

    def test_wait_only_or_unreliable_motion_states_not_sent(self):
        for record in self.training:
            path=Path(record["samples"])
            rows=[json.loads(r) for r in path.read_text().splitlines()]
            for row in rows:
                for candidate in row["candidates"]:
                    candidate["feasible"]=candidate["action"]=="WAIT"
            path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        self.assertEqual(review_examples(self.training,self.manifest,2),[])

    def test_dry_run_never_constructs_teacher_or_calls_api(self):
        with patch("review_rollouts_llm.load_collection",return_value=dict(trials=self.training)), \
             patch("review_rollouts_llm.LLMTeacher") as teacher:
            report=run_reviews(self.root,2)
        teacher.assert_not_called()
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["requests_sent"],0)

    def test_validation_test_old_reference_and_altered_labels_rejected(self):
        path,row=self.advice()
        for split in ("validation","test"):
            invalid={**row,"split":split}
            path.write_text(json.dumps(invalid)+"\n")
            with self.assertRaisesRegex(ValueError,"forbidden"):
                load_advisory_rows(path,self.manifest,2,self.training)
        invalid={**row,"reference_digest":"old-reference"}
        path.write_text(json.dumps(invalid)+"\n")
        with self.assertRaisesRegex(ValueError,"stale"):
            load_advisory_rows(path,self.manifest,2,self.training)
        invalid=copy.deepcopy(row);invalid["label"]["action"]="RIGHT"
        path.write_text(json.dumps(invalid)+"\n")
        with self.assertRaisesRegex(ValueError,"accepted API"):
            load_advisory_rows(path,self.manifest,2,self.training)

    def test_fabricated_snapshot_and_duplicate_case_rejected(self):
        path,row=self.advice()
        invalid=copy.deepcopy(row);invalid["sample"]["features"][0]+=.01
        path.write_text(json.dumps(invalid)+"\n")
        with self.assertRaisesRegex(ValueError,"actual eligible"):
            load_advisory_rows(path,self.manifest,2,self.training)
        path.write_text((json.dumps(row)+"\n")*2)
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            load_advisory_rows(path,self.manifest,2,self.training)

    def test_actual_fit_consumes_advice_but_never_changes_binary_outcomes(self):
        path,_=self.advice()
        def collection(_root,_stage,split,_manifest):
            return dict(trials=self.training if split=="train" else self.validation,
                        completed_trials=3,unique_encounters=1,outcomes=dict(passes=1,collisions=0))
        with patch("rollout_training.load_collection",side_effect=collection):
            report=fit_stage(self.root,self.manifest,2,epochs=2,allow_small=True,reviews=path)
        self.assertEqual(report["fresh_llm_advice"]["rows_consumed"],1)
        self.assertEqual(report["successful_rollout_imitation_rows"],2)
        self.assertEqual(report["imitation_rows"],3)
        arrays=np.load(self.root/"datasets/stage2.train.npz")
        np.testing.assert_array_equal(arrays["value_y"],[0,0,1,1,0,0])
        self.assertEqual(len(arrays["imitation_y"]),3)
        model=TinyNetwork.load(self.root/"models/stage2.npz")
        self.assertEqual(model.metadata["label_sources"],dict(successful_rollout_action=2,llm_advisory=1))
        value=TinyNetwork.load(self.root/"models/stage2.value.npz")
        self.assertEqual(value.metadata["label_source"],"observed complete episode outcome")
        self.assertNotIn("fresh_llm_advice",value.metadata)


if __name__=="__main__":
    unittest.main()
