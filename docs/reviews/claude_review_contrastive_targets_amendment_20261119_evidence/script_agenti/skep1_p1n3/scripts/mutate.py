import sys
from pathlib import Path
root = Path(sys.argv[1]); level = sys.argv[2]
def sub(rel, old, new):
    p = root / rel; s = p.read_text()
    assert s.count(old) == 1, (rel, old, s.count(old))
    p.write_text(s.replace(old, new))
# M1: remove every tamper-detection branch the five tests are named after
sub('scripts/contrastive_prepare.py',
    '        raise ValueError("calibration receipt differs from authenticated recalculation")',
    '        pass  # MUTANT: receipt/recalculation comparison removed')
sub('scripts/contrastive_source_compat.py',
    '_require(artifact_sha == receipt_pin, "historical calibration receipt changed")',
    'pass  # MUTANT')
sub('scripts/contrastive_source_compat.py',
    '_require(artifact_sha == run_pins, "historical calibration run bytes changed")',
    'pass  # MUTANT')
sub('scripts/contrastive_prepare.py',
    '        raise ValueError("calibration metrics are missing, duplicated or out of order")',
    '        pass  # MUTANT')
if level == 'M2':
    # M2: also remove the 100/125 config gate that rejects first
    sub('scripts/contrastive_prepare.py',
        '        raise ValueError("calibration run config differs from the frozen 512-batch recipe")',
        '        pass  # MUTANT')
print('mutated', level)
