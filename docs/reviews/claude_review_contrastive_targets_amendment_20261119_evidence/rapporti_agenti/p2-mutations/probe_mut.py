"""Does the (possibly mutated) checkout accept the historical audit/calibration for the new recipe?

Usage: python -B probe_mut.py <checkout> <tmpdir>
Only the physical data-file hashing in validate_audit is neutralised (the
historical audit's input pins are returned for placeholder paths): data are the
same by construction, so the probe isolates the source and recipe barriers.
"""
import gzip, json, sys
from pathlib import Path

ROOT, TMP = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
TMP.mkdir(parents=True, exist_ok=True)
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import contrastive_prepare as P
import contrastive_run as RUN

H = ROOT / 'results/infra/contrastive_comparison_20261116'
CAL = H / 'calibration_completed_20261117'
out = {'source_digest': P.source_digest()}


def attempt(name, fn):
    try:
        out[name] = {'accepted': True, 'result': fn()}
    except Exception as e:  # noqa: BLE001
        out[name] = {'accepted': False, 'error': type(e).__name__ + ': ' + str(e)[:200]}


receipt = P._read(CAL / 'calibration_report.json')
new_cal = P.build_review_configs(P._read(P.BASELINE))['calibration']
out['expected_calibration_windows'] = [new_cal['loader'].get('rms_window_ms'), new_cal['loader'].get('ms_fast_window_ms'), new_cal.get('target_version')]
attempt('calibration_report(hist run)', lambda: [P.calibration_report(CAL / 'run')[k] for k in ('status', 'weights')])
def cohort():
    cfg = P.calibrated_configs(receipt, CAL / 'run')['CN']
    return {'CN_windows': [cfg['loader']['rms_window_ms'], cfg['loader']['ms_fast_window_ms'], cfg['target_version']],
            'CN_weights': [cfg['contrastive']['nce_weight'], cfg['contrastive']['rnc_weight']]}
attempt('calibrated_configs(hist receipt) -> new arms', cohort)
audit = TMP / 'hist_audit.json'
audit.write_bytes(gzip.decompress((H / 'audit_completed_20261117/audit.json.gz').read_bytes()))
report = json.loads(audit.read_text())
paths = {}
for name, digest in report['inputs'].items():
    p = TMP / ('placeholder_' + name); p.write_text(name); paths[name] = p
pins = {str(paths[n]): d for n, d in report['inputs'].items()}
real_sha = RUN.sha
RUN.sha = lambda p: pins.get(str(p)) or real_sha(p)
attempt('validate_audit(hist audit, new calibration recipe)', lambda: bool(RUN.validate_audit(audit, paths, new_cal)))
RUN.sha = real_sha
print(json.dumps(out, indent=1, default=str))
