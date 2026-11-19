"""Independent probe: which check rejects each historical case at a given checkout."""
import copy, json, shutil, sys, traceback
from pathlib import Path
root = Path(sys.argv[1]).resolve()
work = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(root / 'src')); sys.path.insert(0, str(root / 'scripts'))
import contrastive_prepare as P
BASE = root / 'results/infra/contrastive_comparison_20261116'
CAL = BASE / 'calibration_completed_20261117'
def outcome(fn):
    try:
        fn()
        return 'ACCEPTED'
    except Exception as e:  # report type + message + raising frame
        tb = traceback.extract_tb(e.__traceback__)[-1]
        return f'{type(e).__name__}: {e} [{Path(tb.filename).name}:{tb.lineno} {tb.name}]'
base = json.loads((CAL / 'calibration_report.json').read_text())
cases = {}
for field in ('none', 'weight', 'selection', 'reader', 'source'):
    r = copy.deepcopy(base)
    if field == 'weight': r['weights']['nce'] += 1e-9
    if field == 'selection': r['selected_steps'][0] += 1
    if field == 'reader': r['reader_provenance']['source_manifest_sha256'] = '0' * 64
    if field == 'source': r['source_sha256'] = '0' * 64
    cases['receipt_' + field] = outcome(lambda r=r: P.authenticate_calibration(r, CAL / 'run'))
for label, tamper in (('none', False), ('metrics_newline', True)):
    folder = work / ('run_' + label)
    if folder.exists(): shutil.rmtree(folder)
    shutil.copytree(CAL / 'run', folder)
    if tamper:
        with (folder / 'metrics.jsonl').open('a') as f: f.write('\n')
    cases['calibration_report_' + label] = outcome(lambda folder=folder: P.calibration_report(folder))
    shutil.rmtree(folder)
print(json.dumps({'source_digest': P.source_digest(), 'cases': cases}, indent=1))
