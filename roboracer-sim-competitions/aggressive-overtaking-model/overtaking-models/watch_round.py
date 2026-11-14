"""Drive representative encounters from the latest completed moving-model round."""
import argparse
import json
from pathlib import Path

from benchmark import offline_only
from bootstrap import ARTIFACTS, MODELS
from rollout_training import file_hash, sources
from run_experiment import Scenario, episode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('2','3','all'),default='all')
    parser.add_argument('--no-render',action='store_true')
    args=parser.parse_args()
    report_path=ARTIFACTS/'v3_report.json'
    if not report_path.exists():
        raise SystemExit('The new round has not finished validation and testing yet. Use Watch Overtaking Models to view the previous recordings.')
    report=json.loads(report_path.read_text())
    examples=json.loads((Path(report['round_directory'])/'representative_tests.json').read_text())
    for stage in ((2,3) if args.stage=='all' else (int(args.stage),)):
        choices=[]
        for record in examples:
            trace=json.loads(Path(record['trace']).read_text())
            if trace['result']['scenario']['stage']==stage:
                choices.append(trace['result'])
        successes=sorted((r for r in choices if r['success']),key=lambda r:r['passed_at'])
        saved=(successes[len(successes)//2] if successes else choices[0])
        model=MODELS/f'stage{stage}.npz'
        value=model.with_name(model.stem+'.value.npz')
        if (saved['source_sha256']!=sources() or saved['model_sha256']!=file_hash(model)
                or saved.get('value_model_sha256')!=(file_hash(value) if value.exists() else None)):
            raise SystemExit('The controller changed after these tests. Complete a new validation round before reproducing its demonstrations.')
        if stage == 3:
            rival=MODELS/'stage2.npz'
            rival_value=MODELS/'stage2.value.npz'
            if (saved.get('opponent_model_sha256')!=file_hash(rival)
                    or saved.get('opponent_value_model_sha256')!=(file_hash(rival_value) if rival_value.exists() else None)):
                raise SystemExit('The rival changed after these tests. Complete a new validation round before reproducing its demonstrations.')
        with offline_only():
            result,frames,_,_=episode(Scenario(**saved['scenario']),model_path=model,
                render=not args.no_render,control_dt=saved['control_dt_seconds'])
        result['demonstration_note']='Representative previously tested encounter; not an independent success-rate estimate'
        (ARTIFACTS/f'live_stage{stage}_latest.trace.json').write_text(json.dumps(dict(result=result,frames=frames))+'\n')
        print(f"Stage {stage}: {result['outcome']}; completed pass at {result['passed_at']}",flush=True)


if __name__=='__main__':
    main()
