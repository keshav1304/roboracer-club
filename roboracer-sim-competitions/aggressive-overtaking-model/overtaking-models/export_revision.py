"""Export the completed revision, keeping every qualification trial visible."""
import json
from pathlib import Path
import subprocess
import sys

from bootstrap import ARTIFACTS


def main():
    selected, all_paths = [], []
    for stage in (1, 2, 3):
        report = json.loads((ARTIFACTS / f'revised_stage{stage}.json').read_text())
        if not report.get('evaluation_complete'):
            raise RuntimeError(f'Stage {stage} evaluation is not complete')
        runs = report['results']
        all_paths.extend(ARTIFACTS / r['trace'] for r in runs)
        passed = sorted((r for r in runs if r['success']), key=lambda r:r['passed_at'])
        # Show an early stage-2 pass; all late passes and failures remain in HTML.
        chosen = passed[0] if passed else runs[len(runs)//2]
        data = json.loads((ARTIFACTS / chosen['trace']).read_text())
        data['result']['demonstration_note'] = ('Illustrative completed pass selected from the full evaluation; consult all trials for success rates.'
                                                if passed else 'No pass succeeded in this evaluation; this is an actual unsuccessful run.')
        output = ARTIFACTS / f'revision_demo_stage{stage}.trace.json'
        output.write_text(json.dumps(data) + '\n')
        selected.append(output)
    subprocess.run([sys.executable, 'replay.py', *map(str, all_paths), '--output', str(ARTIFACTS/'replay.html'),
                    '--poster', str(ARTIFACTS/'overview.png')], check=True)
    subprocess.run([sys.executable, 'replay.py', *map(str, selected), '--output', str(ARTIFACTS/'demonstrations.html')], check=True)
    subprocess.run([sys.executable, 'render_preview.py', *map(str, selected), '--output', str(ARTIFACTS/'three_models.gif')], check=True)


if __name__ == '__main__':
    main()
