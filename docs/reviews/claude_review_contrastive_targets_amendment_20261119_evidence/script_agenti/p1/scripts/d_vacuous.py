import json, sys, shutil, copy
from pathlib import Path
ROOT = Path(sys.argv[1]); TMP = Path(sys.argv[2])
import contrastive_prepare as P
CAL = ROOT / 'results/infra/contrastive_comparison_20261116/calibration_completed_20261117'
base = json.loads((CAL / 'calibration_report.json').read_text())
for field in ('none', 'weight', 'selection', 'reader', 'source'):
    receipt = copy.deepcopy(base)
    if field == 'weight': receipt['weights']['nce'] += 1e-9
    if field == 'selection': receipt['selected_steps'][0] += 1
    if field == 'reader': receipt['reader_provenance']['source_manifest_sha256'] = '0' * 64
    if field == 'source': receipt['source_sha256'] = '0' * 64
    try:
        P.authenticate_calibration(receipt, CAL / 'run'); print(field, 'ACCEPTED')
    except ValueError as e:
        print(f'{field:10s} rejected by: {str(e)[:90]}')
folder = TMP / 'run'
if folder.exists(): shutil.rmtree(folder)
shutil.copytree(CAL / 'run', folder)
with (folder / 'metrics.jsonl').open('a') as f: f.write('\n')
try:
    P.calibration_report(folder); print('byte-exact ACCEPTED')
except ValueError as e:
    print('byte-exact metrics tamper rejected by:', str(e)[:90])
shutil.rmtree(folder)
