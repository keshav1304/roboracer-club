"""Select labeled development/held-out successes for fresh live Gym reruns."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--validation',type=Path,required=True);p.add_argument('--development',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert (a.validation/'summary.json').exists(), 'Finish evaluation before selecting demonstrations'
final_manifest=json.loads((a.validation/'manifest.json').read_text())
selections=[]
for label,folder in [('development',a.development),('heldout',a.validation)]:
 manifest=json.loads((folder/'manifest.json').read_text())
 for name,digest in final_manifest['source_sha256'].items():
  if name not in ('joint_path_experiment.py','test_joint_path_stage4.py'):
   assert manifest['source_sha256'][name]==digest, 'Controllers differ: '+name
 candidates=[]
 for path in folder.glob('case-*-response.json'):
  d=json.loads(path.read_text());r=d['result'];count=d['response_counts'].get('joint_selected',0)
  if r['success'] and r['collision'] is None:
   baseline=json.loads(path.with_name(path.name.replace('-response','-baseline')).read_text())
   candidates.append((0 if not baseline['result']['success'] else 1,0 if count else 1,r['passed_at'],path,d,baseline))
 assert candidates,'No successful run to display'
 _,_,_,path,d,b=min(candidates,key=lambda x:x[:3]);selections.append((label,path,d,b))
a.output.mkdir(exist_ok=False);shutil.copytree(a.validation/'source',a.output/'source')
jobs=[];provenance=[]
for label,path,d,b in selections:
 for data in (d,b):
  data['demo_origin']=dict(dataset=label,file=str(path if data is d else path.with_name(path.name.replace('-response','-baseline'))),original_job_id=data['job']['id'])
  data['job']['id']=label+'-'+data['job']['id'];jobs.append(data['job'])
  (a.output/(data['job']['id']+'.json')).write_text(json.dumps(data,separators=(',',':')))
 provenance.append(dict(dataset=label,file=str(path),scene=d['job']['id'],pass_time=d['result']['passed_at'],baseline_outcome=b['result']['outcome'],baseline_pass_time=b['result']['passed_at'],joint_selected_decisions=d['response_counts'].get('joint_selected',0)))
final_manifest.update(jobs=jobs,purpose='Selected demonstrations; not aggregate evaluation',demonstrations=provenance)
(a.output/'manifest.json').write_text(json.dumps(final_manifest,indent=2))
(a.output/'DEMO_NOTES.txt').write_text('These are selected successful examples, not overall performance. The development and heldout prefixes identify their origins. Gym will recompute both cars live; it will not replay saved commands. Full test outcomes are in '+str(a.validation.resolve())+'\n'+json.dumps(provenance,indent=2)+'\n')
print(json.dumps(provenance,indent=2))
