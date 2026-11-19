"""Apply one named rewrite to the working-tree copy of a historical artifact (temp clone only)."""
import gzip, json, sys
from pathlib import Path
clone = Path(sys.argv[1]); name = sys.argv[2]
B = clone / 'results/infra/contrastive_comparison_20261116'
def append_nl(p):
    p.write_bytes(p.read_bytes() + b'\n')
if name == 'M1_receipt_weight':
    p = B / 'calibration_completed_20261117/calibration_report.json'
    raw = p.read_text(); old = '0.11942686122572264'
    assert raw.count(old) >= 1
    p.write_text(raw.replace(old, '0.11942686122572265', 1))
elif name == 'M1b_receipt_weight_value':
    p = B / 'calibration_completed_20261117/calibration_report.json'
    raw = p.read_text(); old = '"nce": 0.11942686122572264'
    assert raw.count(old) == 1
    p.write_text(raw.replace(old, '"nce": 0.21942686122572264', 1))
elif name == 'M3b_cohort_weight_value':
    for arm in ('CN', 'CR'):
        p = B / ('calibrated_cohort_20261117/' + arm + '.config.json')
        raw = p.read_text(); old = '"nce_weight": 0.11942686122572264'
        assert raw.count(old) == 1, (arm, raw.count(old))
        p.write_text(raw.replace(old, '"nce_weight": 0.21942686122572264', 1))
elif name == 'M2_audit_bytes':
    p = B / 'audit_completed_20261117/audit.json.gz'
    p.write_bytes(gzip.compress(gzip.decompress(p.read_bytes()) + b'\n'))
elif name == 'M3_cohort_config':
    p = B / 'calibrated_cohort_20261117/CN.config.json'
    raw = p.read_text(); old = '0.11942686122572264'
    assert raw.count(old) >= 1
    p.write_text(raw.replace(old, '0.11942686122572265', 1))
elif name == 'M4_t100g_config_bytes':
    append_nl(clone / 'results/infra/tokenizer_execution_20261103/19618/T100-G/config.json')
elif name == 'M5_gradient_reference':
    append_nl(B / 'review_response/gradient_reference.json')
elif name == 'M6_initial_reference':
    append_nl(B / 'review_response/initial_historical_reference.json')
elif name == 'M7_compat_manifest':
    append_nl(B / 'compatibility_20261118/manifest.json')
elif name == 'M8_calibration_run_summary':
    append_nl(B / 'calibration_completed_20261117/run/summary.json')
elif name == 'M9_historical_plan':
    append_nl(B / 'plan.json')
else:
    raise SystemExit('unknown mutation ' + name)
print('mutated', name)
