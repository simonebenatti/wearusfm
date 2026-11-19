"""Print the error raised by the CURRENT tree for the historical receipt and each parametrized modification."""
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(root / 'src'), str(root / 'scripts')]
import contrastive_prepare as P
import contrastive_source_compat as S
assert Path(P.ROOT).resolve() == root
CAL = root / 'results/infra/contrastive_comparison_20261116/calibration_completed_20261117'
out = {'source_digest': P.source_digest()}
for field in ['none', 'weight', 'selection', 'reader', 'source']:
    receipt = json.loads((CAL / 'calibration_report.json').read_text())
    if field == 'weight': receipt['weights']['nce'] += 1e-9
    if field == 'selection': receipt['selected_steps'][0] += 1
    if field == 'reader': receipt['reader_provenance']['source_manifest_sha256'] = '0' * 64
    if field == 'source': receipt['source_sha256'] = '0' * 64
    calls = []
    orig = S.validate_historical_evidence
    def guard(kind, payload):
        calls.append(kind)
        return orig(kind, payload)
    S.validate_historical_evidence = guard
    try:
        P.authenticate_calibration(receipt, CAL / 'run')
        out[field] = {'raised': None, 'guard_calls': calls}
    except ValueError as exc:
        out[field] = {'raised': str(exc), 'guard_calls': calls}
    finally:
        S.validate_historical_evidence = orig
# The compat guard itself on the current tree, for each kind, with the exact historical evidence.
receipt = json.loads((CAL / 'calibration_report.json').read_text())
for kind, payload in (('calibration_receipt', receipt), ('calibration_run', CAL / 'run')):
    try:
        S.validate_historical_evidence(kind, payload)
        out['guard_' + kind] = 'ACCEPTED'
    except ValueError as exc:
        out['guard_' + kind] = str(exc)
print(json.dumps(out, indent=1))
