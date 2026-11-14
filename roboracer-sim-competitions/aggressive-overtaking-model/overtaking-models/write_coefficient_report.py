from pathlib import Path
import json,hashlib,shutil
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/coefficient_sensitivity_20261113'

def main():
 study=json.loads((OUT/'summary.json').read_text());math=json.loads((OUT/'polynomial_calculations.json').read_text())
 detail=[];parity=[]
 for i in range(6):
  base=json.loads((OUT/f'case-{i:02d}-original.json').read_text())
  for v in study['summary']:
   data=json.loads((OUT/f'case-{i:02d}-{v}.json').read_text())
   a=np.array([f['command'] for f in base['frames']]);b=np.array([f['command'] for f in data['frames']])
   parity.append(dict(case=i,variant=v,commands_identical=bool(a.shape==b.shape and np.array_equal(a,b))))
  dec=base['ranking_decisions']
  detail.append(dict(case=i,kind=base['job']['kind'],outcome=base['result']['outcome'],passed_at=base['result']['passed_at'],response_ranking_steps=len(dec),multiple_candidates_steps=sum(len(r['candidates'])>1 for r in dec),choices=[r['candidates'] for r in dec]))
 report=dict(aggregate=study['summary'],baseline_details=detail,command_parity=parity)
 (OUT/'analysis.json').write_text(json.dumps(report,indent=2))
 text='''POLYNOMIAL DERIVATION AND STAGE 4 COEFFICIENT SENSITIVITY

The coefficients 10, -15, 6 are derived from six endpoint conditions.
Let q(u)=a0+a1*u+a2*u^2+a3*u^3+a4*u^4+a5*u^5, u in [0,1].
Require q(0)=0, q(1)=1, q'(0)=q'(1)=0, q''(0)=q''(1)=0.
The first three conditions at zero give a0=a1=a2=0.
The remaining equations are:
  a3+a4+a5=1
  3*a3+4*a4+5*a5=0
  6*a3+12*a4+20*a5=0
Solving gives a3=10, a4=-15, a5=6. Degree five is the lowest degree
that can satisfy these six independent endpoint constraints.
This is a chosen interpolation family, not a globally optimal racing path.
Zero lateral slope does not mean zero forward speed. Zero endpoint offset
second derivative does not mean zero absolute curvature on a curved track.

For transition length ell, u=(s-s0)/ell and d=d0+(d1-d0)*q(u).
The first derivative scales as 1/ell and the second as 1/ell^2.
Thus the same coefficients can create a gentle or sharp maneuver depending
on the transition length. The spatial polynomial does not by itself enforce
physical acceleration, wall clearance or opponent avoidance.

The implemented heading correction is h(u)=u-6u^3+8u^4-3u^5.
It satisfies h(0)=h(1)=0, h'(0)=1, h'(1)=0 and h''(0)=h''(1)=0.
The combined offset before return taper is:
  d0+(lane-d0)*q(u)+m*ell*h(u)
where m=clip(tan(heading_error)*(1-reference_curvature*d0),-0.8,0.8).
This starts approximately aligned with the car rather than demanding an
instantaneous change in heading. The slope clip +/-0.8 is a separate parameter,
not the D/U score weight. The return taper is another smoothstep, and lane
width clipping depends on track position; the complete implemented path is
therefore not simply one global polynomial. Joins at active clipping boundaries
must still be checked by the geometric/vehicle constraints.

The straight-reference calculation uses a 0.5m shift, zero initial lateral
slope, no return taper and constant speed 2.4m/s. Exact curvature for the graph
(x,d(x)) is |d''|/(1+d'^2)^(3/2), and lateral acceleration is v^2*curvature.
It is a geometric example, not a full-track simulation or a guarantee of safety.
With steering limit 0.4189 rad and wheelbase 0.3302m, maximum kinematic curvature
is about 1.35/m. The 1.2m example reaches about 1.77/m, so slowing down alone
would not make that exact path steerable. The 1.8m example is steerable but
needs speed reduction to meet the 3.2m/s^2 passing acceleration allowance.
'''
 for e in math['examples']:
  text+=f"  Entry {e['entry_m']:.1f}m: peak curvature {e['max_straight_reference_curvature_per_m']:.3f}/m, required lateral acceleration {e['max_lateral_acceleration_at_constant_speed_m_s2']:.2f}m/s^2.\n"
 text+='''
WHY THE SCORE HAS 0.8
J=C-w_D*D+w_U*U. The two default 0.8 coefficients were selected as heuristics.
They were not derived from physics, fitted by the neural network, selected by
GPT, or established as optimal by a parameter search. The neural ensemble
learns motion residuals; it does not learn these scoring weights.
C includes estimated traversal time and heuristic offset/wall penalties.
D is peak predicted extra outward opponent displacement compared with the WAIT
proposal, using the same reactive predictor. It is not measured yielding.
U is maximum ensemble standard deviation in lateral prediction over the horizon.
It is not a calibrated crash probability or a guaranteed uncertainty bound.
D and U use metres. Their weights convert metres into the cost's scale (roughly
seconds-like cost per metre). Raw .8 values have no universal meaning if units
or normalization change. At w_D=.8, an extra .1m predicted move earns .08 cost
units; at w_U=.8, .02m uncertainty costs .016 units.

The feasibility/response gate already requires D>=max(.015,U). Therefore
with equal .8 weights, J=C-.8*(D-U)<=C for accepted candidates. U reduces the
D bonus; it cannot make J exceed C in this accepted subset. The code also
compares cost with nominal candidates and checks clearance, time to collision,
visibility-related overlap settings and confidence before ranking. Changing
these two weights only ranks accepted response paths; it does not remove the
D/U gates or change safety margins.

The coefficient comparison keeps the frozen early-pressure planner shown in
RViz, both maneuver/value networks, opponent network, forecast checkpoint,
reference, sensor seeds, physics, gates and limits fixed. It varies only
(w_D,w_U): (.8,.8), (0,.8), (.8,0), (1.6,.8), (.8,1.6).
There are six paired initial encounters: the two known demonstrations and four
new stratified random starts around the track. Five configurations give 30
closed-loop Gym runs, not 30 independent starting conditions. All racing
opponents are reactive Stage 2; the inherited 0.4s flying-start setup is retained.
The new starts use seed 13112026 for selection; per-scenario seeds and initial
conditions are in manifest.json. A 40s passing window and the existing success,
contact and strict sideways-yield definitions are unchanged. No LLM calls,
network retraining, published-checkpoint changes or GitHub pushes occurred.
This is a small sensitivity study, not global optimization or a blind claim
of general performance improvement.

RESULTS
'''
 for v,r in study['summary'].items():
  text+=f"{v}: weights={r['weights']}; passes={r['passes']}/{r['trials']}; contacts={r['contacts']}; sideways-yield passes={r['yield_passes']}; ranking steps={r['ranking_decisions']}; same-state weight-sensitive steps={r['weight_sensitive_decisions']}.\n"
 text+='\nPer-encounter baseline and command parity:\n'+json.dumps(dict(encounters=[{k:v for k,v in d.items() if k!='choices'} for d in detail],command_parity=parity),indent=2)
 text+='''

INTERPRETATION
Identical controls across weight settings mean that this sampled range did not
change the closed-loop decisions in those encounters. This does not validate
0.8 as optimal or show that opponent prediction is unnecessary. If only one
candidate survives, no ranking coefficient can change the choice. If several
survive, the time/geometry cost may dominate their small D/U differences.
The appropriate next optimization target is selected from actual diagnostics:
accepted-path diversity, prediction calibration and valid trajectory coverage,
not a claim that raising the aggressiveness weight automatically creates a gap.
'''
 (OUT/'REPORT.txt').write_text(text)
 for name in ('polynomial_calculations.py','plot_polynomial_analysis.py','write_coefficient_report.py'):
  shutil.copy2(ROOT/name,OUT/'source'/name)
 (OUT/'calculation_source_hashes.json').write_text(json.dumps({name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('polynomial_calculations.py','plot_polynomial_analysis.py','write_coefficient_report.py')},indent=2))
 print(json.dumps(report['aggregate'],indent=2))
if __name__=='__main__':main()
