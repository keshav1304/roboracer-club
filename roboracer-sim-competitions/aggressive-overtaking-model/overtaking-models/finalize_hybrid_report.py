"""Summarize frozen Gym validation, serial timing, and actual ROS verification."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
v=root/'artifacts/hybrid_corridor_validation_20261114'
d=root/'artifacts/hybrid_corridor_dev_r1'
b=root/'artifacts/hybrid_corridor_speed_20261114'
demo=root/'artifacts/hybrid_corridor_demo_20261114'
a=json.loads((v/'analysis.json').read_text());development=json.loads((d/'summary.json').read_text())
bench=json.loads((b/'results.json').read_text());rv=json.loads((demo/'ros_verification.json').read_text())
assert rv['passed']
summary=a['summary'];old,new=summary['baseline'],summary['response'];bt=bench['summary']
lines=['STAGE 4: WIDER APPROACH WITH GUARDED CLOSE ENCOUNTERS','',
'The final controller is HybridCorridorStage4. All results use actual F1TENTH Gym on the same imported AutoDRIVE default track, with the unchanged reactive Stage 2 opponent. The neural checkpoints are unchanged. There were no API calls or neural-network training in this work.','',
'FINAL INDEPENDENT TEST',
'24 new starting positions, seed 981114; 48 paired runs. Same starting state, sensor seeds, 40-second passing deadline, and opponent for each pair. Ego speed cap 2.4 m/s; opponent cap 2.1 or 2.2 m/s; initial gaps sampled from 0.85 to 1.4 m. Starting positions were stratified around the track. Passing also requires returning near the reference and completing follow-through.',
f"Previous wider planner: {old['passes']}/24 passes, {old['contacts']} contacts, {old['timeouts']} timeouts.",
f"Final planner:          {new['passes']}/24 passes, {new['contacts']} contacts, {new['timeouts']} timeouts.",
'The sole remaining held-out timeout (014) keeps moving and follows without finishing a pass within 40 seconds. It has no mutual standstill; both planners time out there. See remaining_timeout.json.',
'New-only passes: '+str(a['new_only_passes']), 'Previous-only passes: '+str(a['baseline_only_passes']), '',
'ORIGINAL FAILURES AND DEVELOPMENT CHECKS',
'The original three timeouts now pass at 7.23 s (000), 10.53 s (007), and 38.75 s (017). The original body-contact case (014) now passes at 18.44 s. All eight original/regression cases pass without contact.',
'Across the broader 17-case development suite: previous wider planner 11 passes, one contact, five timeouts; final planner 15 passes, zero contacts, two timeouts. It retains every previous success in that suite. Development cases 212 and 308 still time out with both versions; those are additional challenges, not the original four failures.',
'Earlier exploratory rounds covered 32 and 24 random positions. Their failures were used during development, so they are not held-out evidence for this final controller. The final 24-position set is disjoint in seeds and starting positions and was not used to tune the final controller. In total, 80 new random positions were explored across the three rounds. All failed and interrupted experiments remain archived in development_history.json and their original folders.','',
'WHAT CHANGED',
'1. Preserve the existing wider approach while the opponent is ahead. Switch to guarded close-encounter checks when relative longitudinal position is between -0.9 and +0.7 m. Keep that mode until the opponent is over 1.5 m ahead or more than 2.2 m behind. This hysteresis avoids rapid switching. A temporarily hidden opponent always uses the guarded mode.',
'2. Keep a recently occluded tracked opponent in the interaction checks for up to 0.65 seconds rather than dropping the check as soon as detection confidence falls. The planner evaluates observed-motion, braking and return-to-reference hypotheses; it does not assume the opponent will yield.',
'3. Add short, checked steering arcs for low-speed recovery. Known side-padding scan returns can be relaxed only with a motion certificate; front and unknown returns remain subject to braking. For a stopped opponent, proposed padding escapes are capped at 0.25 m/s and checked against the measured nearby scan points. Measured clearance must remain positive and nondecreasing. Steering can be positioned while stopped before creeping. An already-depleted positive predicted gap must not worsen and must restore 0.08 m by the horizon. This is an experimental simulator recovery rule, not a physical-vehicle safety certification.',
'4. Keep lateral search bounded by the first actual walls, with no fixed lane-offset cap. Body clearance, steering feasibility, forward motion and opponent collision checks still apply. The requested wall-bounded lateral extent is preserved.',
'5. Batch path geometry and opponent forecasts, share LiDAR calculations between speed variants, reuse exact static-map calculations within each decision, remove identical capped motions, and briefly reuse failed searches only at an unchanged standstill while commanding a stop.',
'44 tests passed. Tests cover forecast/geometry equivalence, scan limits, occlusion handling, measured-clearance recovery, steering-at-rest commands, failed-search reuse, and encounter-mode hysteresis.','',
'SERIAL SPEED COMPARISON',
'Both controllers were run serially on the same 336 ego/LiDAR observations recorded from three Gym scenarios. No other test or demo ran during this benchmark. Commands were not fed back to physics in the timing replay; the closed-loop driving results above come from separate real Gym episodes. The measurement includes perception and the whole planner, not just neural inference.',
f"Previous: median {bt['baseline']['median_ms']:.2f} ms; p95 {bt['baseline']['p95_ms']:.2f} ms; p99 {bt['baseline']['p99_ms']:.2f} ms.",
f"Final:    median {bt['updated']['median_ms']:.2f} ms; p95 {bt['updated']['p95_ms']:.2f} ms; p99 {bt['updated']['p99_ms']:.2f} ms.",
f"Median reduction: {100*(1-bt['updated']['median_ms']/bt['baseline']['median_ms']):.1f}%; p95 reduction: {100*(1-bt['updated']['p95_ms']/bt['baseline']['p95_ms']):.1f}%.",
'A 20 Hz wall-clock loop has a 50 ms budget. Tail latency above that remains a limitation. This small fixed-input benchmark does not establish a worst-case execution bound. Parallel trial timing is also retained in analysis.json, but is affected by concurrent work.','',
'RVIZ DEMONSTRATION',
'Four selected successful scenes were recomputed live in actual Gym. RViz was verified subscribed to the candidate-path topic; prediction and fallback messages were observed, all four scenes passed with no contact and completed follow-through, and command replay was false. These are selected highlights, not aggregate evidence.',
'Launcher: '+'bash run-demo.sh (from package root)',
'Demo folder: '+str(demo), '', 'ALL FINAL PAIRED OUTCOMES']
for row in a['paired']:
 lines.append(f"{row['case']}: previous {row['baseline']['outcome']} at {row['baseline']['passed_at']}; final {row['response']['outcome']} at {row['response']['passed_at']}; final contact {row['response']['collision']}")
lines+=['','LIMITS','Collision-free results on a finite sample do not guarantee collision-free behavior elsewhere. Remaining timeouts are retained and counted. This uses the same imported default track as the preceding experiments, not a newly verified competition map. The shared racing reference is locally optimized rather than certified globally optimal.']
(v/'REPORT.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(validation=summary,timing=bt,rviz_verified=rv['passed']),indent=2))
