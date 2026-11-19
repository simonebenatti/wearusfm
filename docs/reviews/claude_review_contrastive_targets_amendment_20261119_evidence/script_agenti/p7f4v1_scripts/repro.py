import gzip, sys, traceback
from pathlib import Path
root = Path(sys.argv[1])
out = Path(sys.argv[2])
sys.path.insert(0, str(root / 'scripts'))
from contrastive_source_compat import validate_historical_evidence
import contrastive_source_compat as S
print('helper file:', S.__file__)
base = root / 'results/infra/contrastive_comparison_20261116'
path = out / 'audit.json'
path.write_bytes(gzip.decompress((base / 'audit_completed_20261117/audit.json.gz').read_bytes()))
try:
    validate_historical_evidence('audit', path)
except ValueError as exc:
    print('ValueError type:', type(exc).__name__)
    print('message:', exc)
else:
    print('ACCEPTED (no ValueError)')
