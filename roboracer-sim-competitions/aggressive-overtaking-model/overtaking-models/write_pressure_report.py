"""Write the experiment report from measured artifacts, including failures."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    folder=ROOT/'artifacts/learned_pressure_random_r2'
    summary=json.loads((folder/'summary.json').read_text())
    early=json.loads((ROOT/'artifacts/learned_pressure_random_r1/summary.json').read_text())['response']
    training=json.loads((ROOT/'artifacts/learned_reaction_v2/report.json').read_text())
    timing=json.loads((ROOT/'artifacts/learned_reaction_v2/inference_timing.json').read_text())
    rows=[json.loads(p.read_text()) for p in folder.glob('case-*.json')]
    ours=[r for r in rows if r['job']['policy']=='response']
    previous={r['job']['id'].replace('-baseline',''):r for r in rows if r['job']['policy']=='baseline'}
    changed=[]
    for r in ours:
        b=previous[r['job']['id'].replace('-response','')]
        if (r['result']['success'],r['result']['passed_at'])!=(b['result']['success'],b['result']['passed_at']):
            changed.append(dict(case=r['job']['id'],previous=b['result']['passed_at'],new=r['result']['passed_at'],contact=r['result']['collision']))
    best=sorted((r for r in ours if r['result']['success'] and r['result']['collision'] is None),
        key=lambda r:(0 if r['yield_metrics']['lateral_yield_pass'] else 1 if not previous[r['job']['id'].replace('-response','')]['result']['success'] else 2 if r['response_counts']['selected'] else 3,r['result']['passed_at']))[:2]
    a,b=summary['baseline'],summary['response']
    text=f'''STAGE 4: LEARNED REACTIVE-OPPONENT FORECAST EXPERIMENT

Status: NOT ACCEPTED as a consistent forced-yielding improvement. Experimental simulation work only. A better regression score is not proof of reliable forced yielding. Original Stage 2 and Stage 4 maneuver/value weights are unchanged. No hardware package was replaced.

RESULTS ON 24 RANDOM TRACK POSITIONS
Previous response planner: {a['passes']}/{a['trials']} completed passes; {a['contacts']} contacts; {a['lateral_yield_passes']} completed passes with strict sideways avoidance.
Earlier learned pressure variant: {early['passes']}/{early['trials']} completed passes; {early['contacts']} contacts; {early['lateral_yield_passes']} completed passes with strict sideways avoidance.
Overlap-gated learned response planner: {b['passes']}/{b['trials']} completed passes; {b['contacts']} contacts; {b['lateral_yield_passes']} completed passes with strict sideways avoidance.
Opportunity window: 40 seconds, plus two seconds of collision-free follow-through after a confirmed pass. Pass confirmation requires >1m lead, near the reference line, sustained for one second.
Neither new variant established consistent forced sideways yielding. The early-pressure variant trades fewer contacts in this sample for fewer passes; the overlap-gated variant did not remove the baseline contact. This is not a verified no-crash policy.
All 24 starting positions are stratified random samples around the complete 59.03m track. Each paired test uses identical initial conditions and sensor seeds. Ego cap is 2.4m/s; opponent cap is 2.1 or 2.2m/s. Both use the same locally lap-time-refined reference (not a certified global optimum).
All race opponents are the original reactive MovingModel(stage=2), with their own LiDAR and unchanged weights/safety settings. The 0.4s flying-start setup uses coordinated reference driving to initialize measured history; all avoidance evaluation follows that setup. Setup contacts count. No nonreactive race opponent or fixed-command collision demonstration was run in this work.

WHAT CHANGED
A three-member 16-48-32-2 neural ensemble predicts opponent progress and lateral offset, conditioned on a proposed ego trajectory. Inputs are ego localization/velocity, map geometry, and the LiDAR-derived opponent estimate. Opponent true future poses are training labels or evaluation measurements only.
Inside candidate paths vary lane-change entry, side spacing and return distance. Each candidate is rolled forward using ego steering/acceleration limits. Predictions are checked against rectangular car footprints and track boundaries. The planner compares predicted response to a waiting candidate and replans every 0.05s. It requires close overlap before relying on the forecast: earlier weak predicted reactions made passing worse in development.
The original geometric/neural planner remains the fallback. Forecast error above 0.25m at a monitored 0.6s horizon triggers a two-second pressure cooldown. Ensemble spread is a heuristic, not a calibrated collision guarantee. Decisions remain simulation-only.

TRAINING AND VALIDATION
Initial regression training: 306 recorded reactive-rival episodes from 102 encounters; validation: 30 episodes from 10 different encounters.
New exploration: 160 randomized initial-state proposals, 325 executed short ego maneuvers against fully reactive opponents. 99 initial states produced usable labels: 82 training and 17 validation. There are 903 new training rows and 173 validation rows, split by entire starting state, not neighboring frames.
Exploration included 91 contacts: 18 during moving warmup and 73 during maneuvers. Those are training exploration failures, not contact-free race tests. Only pre-contact labels were used. The abandoned reactive_branches_v1 collection had a sensor-update bug and is excluded; reactive_branches_v2 is the valid collection.
The 24 race-test positions and seeds are different from training and validation seeds. The same evaluation positions compare the early-pressure and overlap-gated variants; they are not an untouched second replication. The gate was selected from the separate eight-case development set. Validation was used to select regression checkpoints, so its error is not a final held-out test score.
On branch validation, longitudinal RMSE improved from {training['prior_model_rmse_m'][0]:.3f}m to {training['validation_rmse_m'][0]:.3f}m; lateral RMSE changed from {training['prior_model_rmse_m'][1]:.4f}m to {training['validation_rmse_m'][1]:.4f}m. These forecasts condition on executed ego paths; predictions of unexecuted proposals can be worse.

YIELD MEASUREMENT
A sideways avoidance event requires the opponent to have an observed target, choose the side away from ego, stay >0.12m off the racing line opposite ego for >=0.25s, be within 2m, and occur while ego relative progress is between -0.9m and +0.8m. Braking, corner tracking error, and an opponent attempting a later counter-pass do not qualify. This is a conservative behavioral indicator, not proof that the ego caused every deviation.
A separate diagnostic tested 40 prescribed pressure maneuvers from overlapping positions against the same reactive rival. Fourteen contacted a wall or car, and four additional starts contacted during warmup; none of the contact-free probes produced more than 0.12m outward displacement. These exploratory probes were not used to train the final checkpoint or counted in its race-test score. They support the limitation that this opponent often brakes or holds its corridor instead of offering a large sideways yield.
The original loose offset-only audit overcounted corner drift; yield_audit.json and summary.json contain the corrected definition. All trial traces, including failures, are retained. RViz displays representative successes, not the full success rate.

TIMING
Forecast ensemble only, batch of 31 horizon queries: median {timing['median_ms']:.3f}ms, p95 {timing['p95_ms']:.3f}ms across 1000 timed calls. This excludes LiDAR processing, geometric candidate search and control. End-to-end timings are stored per trial; the full planner is not guaranteed to meet its 50ms update interval. No comparison to an unidentified JEV implementation is claimed.
No LLM/API calls were made for this training or testing; API cost was zero.

PAIRED CHANGES
{json.dumps(changed,indent=2)}

OVERLAP-GATED EXAMPLES (RViz launcher may instead show the earlier pressure experiment)
{json.dumps([dict(case=r['job']['id'],passed_at=r['result']['passed_at'],learned_decisions=r['response_counts']['selected'],yield_metrics=r['yield_metrics']) for r in best],indent=2)}

FILES
learned_opponent.py: inference and feature schema.
learned_stage4.py: experimental planning integration.
artifacts/learned_reaction_v2/model.npz: trained forecast ensemble.
artifacts/reactive_branches_v2/: new valid exploration data.
artifacts/learned_pressure_dev_v1 through v5/: development history including worse variants.
artifacts/learned_pressure_random_r1/: early-pressure evaluation.
artifacts/learned_pressure_random_r2/: overlap-gated evaluation and exact source/checkpoint snapshots.
ros_visualization/gym_source.py: live actual Gym; ros_display.py: ROS/RViz publishing.
The viewer loads the experiment's frozen planner/checkpoint and refuses changed shared dependencies. Nonreactive counterfactual demos are disabled.
'''
    (ROOT/'STAGE4_LEARNED_REACTION_RESULTS.txt').write_text(text)
    (folder/'report.txt').write_text(text)
    print(text[:900])
if __name__=='__main__':main()
