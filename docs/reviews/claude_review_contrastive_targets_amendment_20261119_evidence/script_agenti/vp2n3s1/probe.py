"""Independent reproduction for P2-N3 (skeptic 1). Writes only under WORK.

Usage: python -B probe.py <clone> <work>
"""
import copy, hashlib, json, shutil, sys
from pathlib import Path

CLONE, WORK = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
WORK.mkdir(parents=True, exist_ok=False)
sys.path[:0] = [str(CLONE / 'src'), str(CLONE / 'scripts')]
import torch
import contrastive_prepare as P
import contrastive_run as RUN
import contrastive_benchmark_readout as B
from wearusfm.training.pairing import state_sha256

H = CLONE / 'results/infra/contrastive_comparison_20261116'
RUNDIR = H / 'calibration_completed_20261117/run'
out = {}


def attempt(name, fn):
    try:
        out[name] = {'accepted': True, 'result': fn()}
    except Exception as e:  # record rejection reason
        out[name] = {'accepted': False, 'error': type(e).__name__ + ': ' + str(e)[:200]}


current = P.source_digest()
new_cal = P.build_review_configs(P._read(P.BASELINE))['calibration']
hist_summary = P._read(RUNDIR / 'summary.json')
out['hist_code'] = hist_summary.get('code')
out['hist_provenance_source'] = hist_summary['contrastive']['provenance']['source_sha256']
out['current_source'] = current

# Variant builder: copy the archived historical run and edit selected fields.
def variant(name, config=False, source=False, code_sha=None):
    d = WORK / name
    shutil.copytree(RUNDIR, d)
    s = P._read(d / 'summary.json')
    if config:
        (d / 'config.json').write_text(json.dumps(new_cal, indent=2))
        s['contrastive']['config'] = new_cal['contrastive']
    if source:
        s['contrastive']['provenance']['source_sha256'] = current
    if code_sha is not None:
        s['code']['sha'] = code_sha
    (d / 'summary.json').write_text(json.dumps(s))
    return d

def status(d):
    r = P.calibration_report(d)
    return {'status': r['status'], 'weights': r['weights'], 'run_code_sha': r['run_code'].get('sha'),
            'initial_state_sha256': r['initial_state_sha256']}

attempt('A0 unmodified historical run', lambda: status(RUNDIR))
attempt('A1 config relabelled only', lambda: status(variant('cfg_only', config=True)))
attempt('A2 source relabelled only', lambda: status(variant('src_only', source=True)))
forged = variant('forged', config=True, source=True)
attempt('A3 config+source relabelled (2g)', lambda: status(forged))
attempt('A4 config+source relabelled, code.sha = 40 x "0"', lambda: status(variant('forged_code0', config=True, source=True, code_sha='0' * 40)))

# Downstream chain from the forged run: receipt, authentication, calibrated configs, cohort.
receipt = P.calibration_report(forged)
rp = WORK / 'forged_receipt.json'
rp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
attempt('C1 authenticate_calibration(forged receipt, forged run)',
        lambda: P.authenticate_calibration(P._read(rp), forged)['status'])
attempt('C2 calibrated_configs CN/CR weights',
        lambda: {a: [c['contrastive']['nce_weight'], c['contrastive']['rnc_weight'], c['loader']['rms_window_ms'],
                     c['loader']['ms_fast_window_ms'], c['target_version']]
                 for a, c in P.calibrated_configs(P._read(rp), forged).items()})
cohort_dir = WORK / 'cohort'
argv = sys.argv
sys.argv = ['contrastive_prepare.py', 'calibrated-configs', '--receipt', str(rp), '--run-dir', str(forged), '--out', str(cohort_dir)]
try:
    P.main()
finally:
    sys.argv = argv
attempt('C3 validate_cohort(cohort from forged receipt, CN member)',
        lambda: RUN.validate_cohort(cohort_dir / 'cohort.json', rp, forged, cohort_dir / 'CN.config.json')['status'])

# Readout barrier on the calibration checkpoint (logic only, tiny tensors).
student = {'w': torch.zeros(3)}
fake_receipt = dict(receipt, initial_state_sha256=state_sha256(student))
hist_cfg = P._read(RUNDIR / 'config.json')
def readout_initial(embedded_config):
    d = WORK / ('ckpt_' + hashlib.sha256(json.dumps(embedded_config, sort_keys=True).encode()).hexdigest()[:8])
    shutil.copytree(forged, d)
    torch.save({'step': 512, 'optimizer': {'state': {}}, 'config': embedded_config,
                'student': student, 'teacher': {'w': torch.zeros(3)}}, d / 'checkpoint.pt')
    return sorted(B._initial_from_calibration(d, fake_receipt))
attempt('D1 _initial_from_calibration: checkpoint embeds historical 25/50 config, config.json forged 100/125',
        lambda: readout_initial(hist_cfg))
attempt('D2 CONTROL _initial_from_calibration: checkpoint embeds the same 100/125 config',
        lambda: readout_initial(P._read(forged / 'config.json')))

# Model section identity between historical and new calibration recipes (initialization inputs).
out['E1 model section identical'] = hist_cfg['model'] == new_cal['model']
out['E2 seed identical'] = (hist_cfg['seed'], new_cal['seed'])
out['E3 top-level/loader keys that differ'] = sorted(
    [k for k in set(hist_cfg) | set(new_cal) if k not in ('loader',) and hist_cfg.get(k) != new_cal.get(k)] +
    ['loader.' + k for k in set(hist_cfg['loader']) | set(new_cal['loader']) if hist_cfg['loader'].get(k) != new_cal['loader'].get(k)])
print(json.dumps(out, indent=1, default=str))
