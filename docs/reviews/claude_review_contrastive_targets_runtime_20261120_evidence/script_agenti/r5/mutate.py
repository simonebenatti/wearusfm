"""Mutation driver: apply one textual mutation in a temp clone, run tests, revert."""
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path

PY = '/Users/simonebenatti/.venvs/wearusfm-tok/bin/python'
TMP = Path('/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp/r5')

def receipt_files(clone):
    r = json.loads((clone/'results/infra/contrastive_targets_20261118/amendment_20261119/validation_final/local_validation.json').read_text())
    return [x['path'] for x in r['test_files']]

def run(clone, mid, edits, tests, results):
    clone = Path(clone)
    if tests == 'ALL':
        tests = receipt_files(clone)
    touched = []
    for (rel, old, new) in edits:
        path = clone/rel
        text = path.read_text()
        n = text.count(old)
        if n != 1:
            results.write(json.dumps({'id': mid, 'error': f'{rel}: occurrences {n}'})+'\n'); results.flush()
            subprocess.run(['git', '-C', str(clone), 'checkout', '--', *touched], check=False)
            return
        path.write_text(text.replace(old, new)); touched.append(rel)
    bt = TMP/('bt_'+mid)
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': f'{clone}/src:{clone}/tests/cpu:{clone}/scripts'}
    t0 = time.time()
    proc = subprocess.run([PY, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-o', 'tmp_path_retention_policy=none',
                           '-rfE', '--basetemp='+str(bt), *tests], cwd=clone, env=env, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    failed = sorted(set(re.findall(r'^(?:FAILED|ERROR) (\S+)', out, re.M)))
    summary = [l for l in out.splitlines() if re.search(r'\d+ (passed|failed)', l)]
    reasons = {}
    for line in out.splitlines():
        m = re.match(r'^(?:FAILED|ERROR) (\S+) - (.*)$', line)
        if m: reasons[m.group(1)] = m.group(2)[:300]
    subprocess.run(['git', '-C', str(clone), 'checkout', '--', *touched], check=True)
    status = subprocess.run(['git', '-C', str(clone), 'status', '--porcelain'], capture_output=True, text=True).stdout
    shutil.rmtree(bt, ignore_errors=True)
    results.write(json.dumps({'id': mid, 'tests': len(tests), 'rc': proc.returncode, 'seconds': round(time.time()-t0, 1),
                              'summary': summary[-1] if summary else out[-500:], 'failed': failed, 'reasons': reasons,
                              'clean_after': status == ''})+'\n')
    results.flush()

if __name__ == '__main__':
    clone, specfile, out = sys.argv[1], sys.argv[2], sys.argv[3]
    only = sys.argv[4].split(',') if len(sys.argv) > 4 else None
    ns = {}
    exec(Path(specfile).read_text(), ns)
    with open(out, 'a') as results:
        for spec in ns['SPECS']:
            if only and spec['id'] not in only: continue
            run(clone, spec['id'], spec['edits'], spec['tests'], results)
