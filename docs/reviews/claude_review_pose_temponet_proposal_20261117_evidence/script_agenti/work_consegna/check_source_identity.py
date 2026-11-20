"""Reviewer check (read-only): does the long-evaluation source guard cover scripts/*.py?"""
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path[:0] = [str(root / 'src'), str(root / 'scripts')]
import consolidation_long_common as C
assert Path(C.__file__).resolve().parent == (root / 'scripts').resolve(), C.__file__
plan = json.loads((root / 'results/infra/consolidation_long_20261105/plan.json').read_text())
actual = C.source_identity()
paths = sorted((C.ROOT / 'src').rglob('*.py')) + sorted((C.ROOT / 'scripts').glob('*.py'))
print('ROOT', C.ROOT)
print('n src py', len(sorted((C.ROOT / 'src').rglob('*.py'))), 'n scripts/*.py', len(sorted((C.ROOT / 'scripts').glob('*.py'))))
print('plan source_sha256  ', plan['source_sha256'])
print('source_identity()   ', actual)
print('EQUAL', plan['source_sha256'] == actual)
