"""Apply each mutation alone in the temp clone, run probe.py, restore; print the probe keys that change."""
import json, os, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mut import MUT, PY, TMP

clone = Path(sys.argv[1])
ids = sys.argv[2].split(',')
env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1',
       'PYTHONPATH': f'{clone}/src:{clone}/tests/cpu:{clone}/scripts'}


def probe(tag):
    proc = subprocess.run([PY, '-B', str(Path(__file__).parent / 'probe.py'), str(TMP / f'sk2r54_work_{tag}')],
                          cwd=str(Path(__file__).parent), env=env, capture_output=True, text=True)
    text = proc.stdout
    return json.loads(text[text.index('{\n'):])


base = probe('base')
for mid in ids:
    rel, old, new = MUT[mid]
    p = clone / rel
    text = p.read_text()
    assert text.count(old) == 1, mid
    p.write_text(text.replace(old, new))
    try:
        res = probe(mid)
    finally:
        p.write_text(text)
    clean = subprocess.run(['git', '-C', str(clone), 'status', '--porcelain'], capture_output=True, text=True).stdout == ''
    changed = {k: {'orig': base[k], 'mut': res[k]} for k in base if k != 'module' and base[k] != res[k]}
    print(json.dumps({'mutation': mid, 'clean_after': clean, 'changed': changed}, indent=1), flush=True)
