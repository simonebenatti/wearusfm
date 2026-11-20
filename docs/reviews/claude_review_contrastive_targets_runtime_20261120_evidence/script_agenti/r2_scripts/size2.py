import json, sys
from pathlib import Path
C = Path(sys.argv[1])
sys.path[:0] = [str(C/'src'), str(C/'scripts')]
from contrastive_prepare import build_review_configs, BASELINE, _read
cfgs = build_review_configs(_read(BASELINE))
raw = cfgs['CN']
print(json.dumps(raw['contrastive'], indent=0)[:3000])
print('datasets', raw.get('datasets'))
print({k: raw.get(k) for k in ('pose_panel_users','pose_panel_recordings','pose_panel_windows','fixed_eval_batches','amp_bf16','accum_steps','seed','lr','warmup_steps','grad_clip','prefetch_audit')})
print('model dim', raw['model'].get('dim'), 'muscle_dropout', raw['model'].get('muscle_dropout'))
