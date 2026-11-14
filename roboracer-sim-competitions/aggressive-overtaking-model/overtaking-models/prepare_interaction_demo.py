"""Pick two successful held-out runs; all aggregate outcomes stay in validation."""
import argparse,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('validation',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
assert (a.validation/'summary.json').exists(),'Complete the paired evaluation first'
manifest=json.loads((a.validation/'manifest.json').read_text());options=[]
for path in a.validation.glob('case-*-response.json'):
 d=json.loads(path.read_text());b=json.loads(path.with_name(path.name.replace('-response','-baseline')).read_text())
 if d['result']['success'] and d['result']['collision'] is None:
  priority=0 if not b['result']['success'] else 1
  options.append((priority,d['result']['passed_at'],path,d,b))
chosen=sorted(options,key=lambda x:x[:2])[:2]
assert len(chosen)==2,'Two successful cases required'
a.output.mkdir(exist_ok=False);shutil.copytree(a.validation/'source',a.output/'source');jobs=[];notes=[]
for _,_,path,d,b in chosen:
 for data in (d,b):
  jobs.append(data['job']);(a.output/(data['job']['id']+'.json')).write_text(json.dumps(data,separators=(',',':')))
 notes.append(dict(scene=d['job']['id'],source=str(path.resolve()),baseline=b['result']['outcome'],baseline_seconds=b['result']['passed_at'],new_seconds=d['result']['passed_at'],fallback_decisions=d['interaction_counts']['overrides']))
manifest.update(jobs=jobs,purpose='Selected held-out successes, recomputed live in Gym',demonstrations=notes)
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));(a.output/'DEMO_NOTES.txt').write_text('Selected successes from the full paired evaluation. These will be recomputed in live Gym, not command-replayed. Full results: '+str((a.validation/'REPORT.txt').resolve())+'\n'+json.dumps(notes,indent=2)+'\n');print(json.dumps(notes,indent=2))
