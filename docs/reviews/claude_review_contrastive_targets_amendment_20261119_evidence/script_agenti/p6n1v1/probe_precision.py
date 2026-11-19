"""Probe: a config differing only in amp_bf16 is rejected by validate_recipe; dispatcher allocations are fixed."""
import json, copy, sys
from pathlib import Path
CLONE = Path(sys.argv[1])
import torch  # trainer import order
import contrastive_run as CR
import contrastive_job as CJ
raw = json.loads((CLONE / 'results/infra/contrastive_targets_20261118/review_configs/calibration.config.json').read_text())
print('stored calibration amp_bf16 =', raw.get('amp_bf16'))
CR.validate_recipe(copy.deepcopy(raw), 'calibration')
print('stored calibration config: validate_recipe OK')
mut = copy.deepcopy(raw); mut['amp_bf16'] = False
try:
    CR.validate_recipe(mut, 'calibration'); print('MUTATED ACCEPTED (unexpected)')
except ValueError as e:
    print('amp_bf16=False rejected:', e)
print('ALLOCATED_SECONDS =', CJ.ALLOCATED_SECONDS)
job = {'status': 'APPROVED_FOR_PREFLIGHT', 'mode': 'full', 'source_sha256': CJ.source_digest()}
for secs in (1800, 3600):
    try:
        CJ.build_command(dict(job), 'gpu', secs)
    except ValueError as e:
        print(f'full mode allocated {secs}s ->', e)
    except Exception as e:
        print(f'full mode allocated {secs}s -> passed allocation guard, stopped later at', type(e).__name__, str(e)[:80])
print('source_digest =', CJ.source_digest())
