"""Positive control: synthetic 100/125 run with the CURRENT source, receipt with a foreign source.
Reports which error authenticate_calibration raises (guard vs comparison)."""
import copy, json, sys, tempfile
from pathlib import Path
root, tmp = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.path[:0] = [str(root / 'src'), str(root / 'tests/cpu'), str(root / 'scripts')]
import test_contrastive_prepare as T
P = T.P
assert Path(P.ROOT).resolve() == root
run = tmp / 'run'
cfg, _, _ = T.fixture_run(run)
out = {'recipe': [cfg['loader']['rms_window_ms'], cfg['loader']['ms_fast_window_ms'], cfg['target_version']]}
receipt = P.calibration_report(run)
out['honest_receipt_accepted'] = P.authenticate_calibration(receipt, run) == receipt
for label, change in (('foreign_source', lambda r: r.update(source_sha256='0' * 64)),
                      ('foreign_source_and_reader', lambda r: (r.update(source_sha256='0' * 64),
                                                               r['reader_provenance'].update(source_manifest_sha256='0' * 64)))):
    bad = copy.deepcopy(receipt); change(bad)
    try:
        P.authenticate_calibration(bad, run); out[label] = 'ACCEPTED'
    except ValueError as exc:
        out[label] = str(exc)
print(json.dumps(out, indent=1))
