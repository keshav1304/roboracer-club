"""Pick fixed collision/deadlock plus fresh successes, retaining full provenance."""
import argparse,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--validation',type=Path,required=True);p.add_argument('--development',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=json.loads((a.validation/'manifest.json').read_text());dm=json.loads((a.development/'manifest.json').read_text())
assert m['source_sha256']==dm['source_sha256'],'Demo and test must use identical frozen code'
selected=[]
for i in (14,7):
 path=a.development/f'case-{i:03d}-response.json';d=json.loads(path.read_text());b=json.loads(path.with_name(path.name.replace('-response','-baseline')).read_text());assert d['result']['success'] and d['result']['collision'] is None
 selected.append(('fixed',path,d,b))
options=[]
for path in a.validation.glob('case-*-response.json'):
 d=json.loads(path.read_text());b=json.loads(path.with_name(path.name.replace('-response','-baseline')).read_text())
 if d['result']['success'] and d['result']['collision'] is None and d['interaction_counts']['overrides']:
  options.append((0 if not b['result']['success'] else 1,d['result']['passed_at'],path,d,b))
for _,_,path,d,b in sorted(options,key=lambda x:x[:2])[:2]:selected.append(('random',path,d,b))
assert len(selected)==4
a.output.mkdir(exist_ok=False);shutil.copytree(a.validation/'source',a.output/'source');jobs=[];notes=[]
for label,path,d,b in selected:
 for obj in (d,b):
  obj['demo_origin']=dict(dataset=label,source_file=str(path.resolve()),original_job_id=obj['job']['id'])
  obj['job']['id']=label+'-'+obj['job']['id'];jobs.append(obj['job']);(a.output/(obj['job']['id']+'.json')).write_text(json.dumps(obj,separators=(',',':')))
 notes.append(dict(scene=d['job']['id'],dataset=label,pass_seconds=d['result']['passed_at'],baseline_outcome=b['result']['outcome'],source=str(path.resolve())))
m.update(jobs=jobs,purpose='Selected successes recomputed live in Gym',demonstrations=notes)
(a.output/'manifest.json').write_text(json.dumps(m,indent=2));(a.output/'DEMO_NOTES.txt').write_text('Selected successes. Full results: '+str((a.validation/'REPORT.txt').resolve())+'\n'+json.dumps(notes,indent=2)+'\n');print(json.dumps(notes,indent=2))
