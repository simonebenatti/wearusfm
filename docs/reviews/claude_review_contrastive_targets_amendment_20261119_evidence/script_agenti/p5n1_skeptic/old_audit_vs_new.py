import gzip, hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / 'scripts')); sys.path.insert(0, str(root / 'src'))
from contrastive_prepare import build_review_configs, _read, BASELINE, source_digest
old = json.loads(gzip.decompress((root / 'results/infra/contrastive_comparison_20261116/audit_completed_20261117/audit.json.gz').read_bytes()))
new_cal = build_review_configs(_read(BASELINE))['calibration']
new_sha = hashlib.sha256(json.dumps(new_cal, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
pinned = _read(root / 'results/infra/contrastive_targets_20261118/review_configs/calibration.config.json')
print('old report config_sha256 :', old['config_sha256'])
print('new expected config_sha256:', new_sha)
print('new review_configs/calibration == derived:', pinned == new_cal)
print('old source_sha256 :', old['source_sha256'])
print('current source_digest :', source_digest())
print('new loader windows:', new_cal['loader'].get('rms_window_ms'), new_cal['loader'].get('ms_fast_window_ms'), new_cal.get('loader', {}).get('target_version', None))
