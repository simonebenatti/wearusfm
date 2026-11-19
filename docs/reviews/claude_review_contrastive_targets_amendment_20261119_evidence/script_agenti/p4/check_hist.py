import json, sys, copy, shutil, tempfile
from pathlib import Path
import contrastive_prepare as P
H = P.ROOT / 'results/infra/contrastive_comparison_20261116/calibration_completed_20261117'
receipt = json.loads((H/'calibration_report.json').read_text())
run = H/'run'
def attempt(label, fn):
    try:
        fn(); print(label, '-> ACCEPTED (!)')
    except Exception as e:
        print(label, '-> rejected:', type(e).__name__, str(e)[:160])
attempt('calibration_report(historical run)', lambda: P.calibration_report(run))
attempt('authenticate_calibration(hist receipt, hist run)', lambda: P.authenticate_calibration(receipt, run))
attempt('calibrated_configs(hist receipt, hist run, 576)', lambda: P.calibrated_configs(receipt, run, 576))
# Relabelled historical run: config rewritten to the new calibration recipe
tmp = Path(sys.argv[1]); tmp.mkdir(parents=True, exist_ok=True)
fake = tmp/'relabel'; 
if fake.exists(): shutil.rmtree(fake)
shutil.copytree(run, fake)
newcal = json.loads((P.ROOT/'results/infra/contrastive_targets_20261118/review_configs/calibration.config.json').read_text())
(fake/'config.json').write_text(json.dumps(newcal))
s = json.loads((fake/'summary.json').read_text()); s['contrastive']['config'] = newcal['contrastive']
(fake/'summary.json').write_text(json.dumps(s))
attempt('calibration_report(relabelled config, old source digest)', lambda: P.calibration_report(fake))
# Same plus provenance source digest rewritten to current
s['contrastive']['provenance']['source_sha256'] = P.source_digest(); (fake/'summary.json').write_text(json.dumps(s))
def full():
    r = P.calibration_report(fake)
    print('   forged report status', r['status'], 'weights', r['weights'], 'multiplier', r['historical_gradient_multiplier'])
attempt('calibration_report(relabelled config + current digest)', full)
print('build_review_configs historical == new calibration?', P.build_review_configs(json.loads(P.BASELINE.read_text()))['calibration'] == newcal)
