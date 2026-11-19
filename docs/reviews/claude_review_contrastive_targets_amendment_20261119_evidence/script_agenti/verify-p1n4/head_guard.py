import gzip, json, sys
from pathlib import Path
root = Path(sys.argv[1]); tmp = Path(sys.argv[2])
sys.path[:0] = [str(root/'src'), str(root/'scripts')]
import contrastive_source_compat as S
import contrastive_prepare as P
base = root/'results/infra/contrastive_comparison_20261116'
audit = tmp/'audit.json'
audit.write_bytes(gzip.decompress((base/'audit_completed_20261117/audit.json.gz').read_bytes()))
receipt = json.loads((base/'calibration_completed_20261117/calibration_report.json').read_text())
for kind, payload in (('audit', audit), ('calibration_receipt', receipt), ('calibration_run', base/'calibration_completed_20261117/run')):
    try:
        S.validate_historical_evidence(kind, payload); print(kind, 'ACCEPTED')
    except ValueError as e:
        print(kind, 'REJECTED:', e)
try:
    P.authenticate_calibration(receipt, base/'calibration_completed_20261117/run'); print('authenticate ACCEPTED')
except ValueError as e:
    print('authenticate REJECTED:', e)
print('source_digest HEAD', P.source_digest())
