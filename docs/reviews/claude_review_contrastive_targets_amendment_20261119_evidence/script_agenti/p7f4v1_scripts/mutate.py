"""Mutation driver for P7-F4 (skeptic 1). Runs only in the temp clone."""
import json, os, re, subprocess, sys, shutil
from pathlib import Path
T = Path('/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_targets/tmp')
C = T / 'p7f4v1_clone'
assert C.is_dir() and (C / '.git').exists()
PY = '/Users/simonebenatti/.venvs/wearusfm-tok/bin/python'
MAN = C / 'results/infra/contrastive_comparison_20261116/compatibility_20261118/manifest.json'
HELP = C / 'scripts/contrastive_source_compat.py'
TEST = C / 'tests/cpu/test_contrastive_historical_calibration.py'
FILES = ['tests/cpu/test_contrastive_historical_calibration.py', 'tests/cpu/test_contrastive_source_compat.py',
         'tests/cpu/test_contrastive_historical_callsites.py']
env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
       'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': f'{C}/src:{C}/tests/cpu:{C}/scripts'}

def restore():
    subprocess.run(['git', '-C', str(C), 'checkout', '-q', '--', '.'], check=True)
    subprocess.run(['git', '-C', str(C), 'clean', '-qfd', '-e', '!.'], check=True)

def run(tag):
    bt = T / ('p7f4v1_bt_' + tag)
    r = subprocess.run([PY, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-rf', '--basetemp=' + str(bt), *FILES],
                       cwd=C, env=env, capture_output=True, text=True, timeout=600)
    shutil.rmtree(bt, ignore_errors=True)
    tail = r.stdout.strip().splitlines()
    failed = [l for l in tail if l.startswith('FAILED')]
    return {'summary': tail[-1] if tail else r.stderr[-500:], 'failed': failed}

def reason():
    code = ("import gzip,sys;from pathlib import Path;root=Path(sys.argv[1]);sys.path.insert(0,str(root/'scripts'));"
            "from contrastive_source_compat import validate_historical_evidence as v;p=Path(sys.argv[2]);"
            "p.write_bytes(gzip.decompress((root/'results/infra/contrastive_comparison_20261116/audit_completed_20261117/audit.json.gz').read_bytes()))\n"
            "try:\n v('audit',p);print('ACCEPTED')\nexcept ValueError as e:print(type(e).__name__+': '+str(e)[:140])")
    out = T / 'p7f4v1_reason_audit.json'
    r = subprocess.run([PY, '-B', '-c', code, str(C), str(out)], cwd=T, env=env, capture_output=True, text=True)
    out.unlink(missing_ok=True)
    return (r.stdout + r.stderr).strip()[-300:]

def match_fix():
    s = TEST.read_text()
    old = "    with pytest.raises(ValueError):\n        validate_historical_evidence('audit',path)"
    assert s.count(old) == 1
    TEST.write_text(s.replace(old, "    with pytest.raises(ValueError, match='current execution source inventory'):\n        validate_historical_evidence('audit',path)"))

def m_protocol():
    d = json.loads(MAN.read_text()); d['protocol'] = 'another-protocol'
    MAN.write_text(json.dumps(d, sort_keys=True, separators=(',', ':')))

def m_truncate():
    MAN.write_bytes(MAN.read_bytes()[:100])

def m_delete():
    MAN.unlink()

def m_drop162():
    s = HELP.read_text()
    old = '    _require(actual == manifest["current_sources"], "current execution source inventory/pins differ")\n'
    assert s.count(old) == 1; HELP.write_text(s.replace(old, ''))

def m_drop162_164():
    m_drop162()
    s = HELP.read_text()
    old = '    _require(actual_digest == manifest["current_source_sha256"], "current source digest differs from manifest")\n'
    assert s.count(old) == 1; HELP.write_text(s.replace(old, ''))

results = {}
for tag, muts in [('M0', []), ('Ma1_protocol', [m_protocol]), ('Ma2_truncate', [m_truncate]), ('Ma3_delete', [m_delete]),
                  ('Mc_drop162', [m_drop162]), ('Mb_drop162_164', [m_drop162_164]),
                  ('F0_matchfix', [match_fix]), ('F1_matchfix_protocol', [match_fix, m_protocol]),
                  ('F2_matchfix_truncate', [match_fix, m_truncate]), ('F3_matchfix_drop162', [match_fix, m_drop162])]:
    restore()
    for m in muts: m()
    results[tag] = {'reason': reason(), **run(tag)}
    print(tag, json.dumps(results[tag]), flush=True)
restore()
st = subprocess.run(['git', '-C', str(C), 'status', '--short'], capture_output=True, text=True).stdout
print('clean after restore:', st == '', repr(st[:200]))
