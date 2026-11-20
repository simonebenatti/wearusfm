import json, hashlib, sys
from pathlib import Path
C = Path(sys.argv[1])
A = C/'results/infra/contrastive_targets_20261118/amendment_20261119'
rec = json.loads((A/'validation_final/local_validation.json').read_text())
cmd = rec['runs'][0]['command']
cmd_tests = [x for x in cmd if x.startswith('tests/')]
pin_tests = [x['path'] for x in rec['test_files']]
print('n cmd tests', len(cmd_tests), 'n pinned', len(pin_tests), 'same list', cmd_tests == pin_tests)
glob = sorted(str(p.relative_to(C)) for p in (C/'tests/cpu').glob('test_contrastive*.py'))
extra = ['tests/cpu/test_target_windows_fix.py','tests/cpu/test_rms_target_window.py','tests/cpu/test_ms_fast_target_window.py']
expect = sorted(set(glob + [e for e in extra if (C/e).is_file()]))
print('glob at HEAD == receipt list', expect == sorted(pin_tests), 'missing', set(expect)-set(pin_tests), 'extra', set(pin_tests)-set(expect))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
dp = json.loads((A/'delivery_pin.json').read_text())
for item in dp['plans'] + [dp['validation']]:
    p = C/item['path']; print(item['path'], 'sha ok', sha(p)==item['sha256'], 'bytes ok', p.stat().st_size==item['bytes'])
print('delivery source == receipt source', dp['source_sha256']==rec['source_sha256'])
print('science plan sha', dp['science_plan_sha256']==sha(C/'results/infra/contrastive_targets_20261118/plan.json'))
print('preflight plan sha', dp['preflight_plan_sha256']==sha(C/'results/infra/contrastive_targets_20261118/preflight_plan.json'))
v = json.loads((A/'verification.json').read_text())
print('verification receipt sha', v['receipt']['sha256']==sha(C/v['receipt']['path']))
print('env', rec['environment'])
print('cwd', rec['cwd'])
