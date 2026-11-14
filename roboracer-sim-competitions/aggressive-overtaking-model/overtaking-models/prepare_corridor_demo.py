"""Show both fixed diagnostic cases plus two successful fresh-test highlights."""
import argparse,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--validation',type=Path,required=True);p.add_argument('--development',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert (a.validation/'summary.json').exists(),'Complete all validation runs first'
m=json.loads((a.validation/'manifest.json').read_text());dm=json.loads((a.development/'manifest.json').read_text())
for name in ('corridor_stage4.py','interaction_stage4.py','joint_path_stage4.py','world.py','maneuver_core.py','stage2.npz','stage3.npz'):
 assert m['source_sha256'][name]==dm['source_sha256'][name],name
selected=[]
for i in (4,5):
 path=a.development/f'case-{i:03d}-response.json';d=json.loads(path.read_text());b=json.loads(path.with_name(path.name.replace('-response','-baseline')).read_text())
 assert d['result']['success'] and d['result']['collision'] is None
 selected.append(('fixed',path,d,b))
options=[]
for path in a.validation.glob('case-*-response.json'):
 d=json.loads(path.read_text());b=json.loads(path.with_name(path.name.replace('-response','-baseline')).read_text())
 wide=any(o['interaction'].get('mode')=='checked wall corridor' for o in d.get('observations',[]))
 if d['result']['success'] and d['result']['collision'] is None and wide:
  priority=0 if not b['result']['success'] else 1
  improvement=40. if not b['result']['success'] else b['result']['passed_at']-d['result']['passed_at']
  options.append((priority,-improvement,d['result']['passed_at'],path,d,b))
for _,_,_,path,d,b in sorted(options,key=lambda x:x[:3])[:2]:selected.append(('random',path,d,b))
assert len(selected)==4,'Need two fresh successes that used the wider planner'
a.output.mkdir(exist_ok=False);shutil.copytree(a.validation/'source',a.output/'source');jobs=[];notes=[]
for label,path,d,b in selected:
 for data in (d,b):
  data['demo_origin']=dict(original_file=str(path if data is d else path.with_name(path.name.replace('-response','-baseline'))),dataset=label,original_job_id=data['job']['id'])
  data['job']['id']=label+'-'+data['job']['id'];jobs.append(data['job']);(a.output/(data['job']['id']+'.json')).write_text(json.dumps(data,separators=(',',':')))
 notes.append(dict(scene=d['job']['id'],dataset=label,new_seconds=d['result']['passed_at'],baseline_outcome=b['result']['outcome'],baseline_seconds=b['result']['passed_at'],original_file=str(path.resolve())))
m.update(jobs=jobs,purpose='Selected live demonstrations: both original fixes and two fresh highlights',demonstrations=notes)
(a.output/'manifest.json').write_text(json.dumps(m,indent=2));(a.output/'DEMO_NOTES.txt').write_text('Selected successes, not aggregate results. Fixed-prefix scenes are development cases; random-prefix scenes are fresh held-out cases. All four are recomputed live in actual Gym. Full outcomes: '+str((a.validation/'REPORT.txt').resolve())+'\n'+json.dumps(notes,indent=2)+'\n');print(json.dumps(notes,indent=2))
