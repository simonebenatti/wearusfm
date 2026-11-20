"""Functional check of contrastive_run.validate_target_fork (untested in the suite) on a synthetic 544/576 fork."""
import copy, hashlib, json, shutil, sys
from pathlib import Path
C, EXP, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path[:0] = [str(C / 'src'), str(C / 'scripts')]
import torch
import contrastive_run as RUN
from contrastive_prepare import config_from_dict, source_digest
from wearusfm.training import run as R

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
fork = torch.load(str(EXP / 'continuous/checkpoint_step0000004.pt'), map_location='cpu', weights_only=False)
raw = copy.deepcopy(fork['config'])
raw['max_steps'] = 576
raw['schedule'] = {**raw['schedule'], 'total_steps': 49046, 'keep_checkpoints': [544]}
cfg = config_from_dict(raw)
raw = R.config_to_dict(cfg)
inputs = {'manifest': 'm', 'scales': 's', 'window_seconds': 'w', 'pose_stats': 'p'}
cohort = {'initial_state_sha256': fork['contrastive_initial_sha256'], 'calibration_sha256': 'c' * 64}
prov = {'source_sha256': source_digest(), 'calibration_sha256': cohort['calibration_sha256'], 'cohort_sha256': 'h' * 64, 'input_sha256': inputs}
fork.update(step=544, config=raw, contrastive_provenance=prov, world_size=4)
fork['contrastive_state']['queue']['last_step'] = 543
fork['rank_rng'] = [{'torch_rng': r['torch_rng'], 'cuda_rng': torch.zeros(16, dtype=torch.uint8)} for r in fork['rank_rng']]
def build(name, mutate=None):
    d = OUT / name; d.mkdir(parents=True)
    f = d / 'checkpoint_step0000544.pt'; torch.save(fork, f)
    summary = {'steps': 576, 'stopped': 'max_steps', 'world_size': 4,
               'contrastive': {'config': raw['contrastive'], 'initial_sha256': cohort['initial_state_sha256'], 'provenance': prov}}
    release = {'arm': raw['contrastive']['arm'], 'cohort_sha256': 'h' * 64, 'inputs': inputs}
    if mutate: mutate(d, f, summary, release)
    s = d / 'summary.json'; s.write_text(json.dumps(summary))
    release.setdefault('fork_checkpoint', {'path': str(f), 'sha256': sha(f)})
    release['continuous_summary'] = {'path': str(s), 'sha256': sha(s)}
    return release
cases = {
    'ok': None,
    'stop_file': lambda d, f, s, r: (d / 'STOP').write_text('x'),
    'summary_575': lambda d, f, s, r: s.update(steps=575),
    'summary_ws1': lambda d, f, s, r: s.update(world_size=1),
    'arm_differs': lambda d, f, s, r: r.update(arm='CN' if r['arm'] == 'CR' else 'CR'),
    'provenance_cohort_differs': lambda d, f, s, r: r.update(cohort_sha256='x' * 64),
    'wrong_name': lambda d, f, s, r: (shutil.copyfile(f, d / 'checkpoint_step0000543.pt'),
                                      r.update(fork_checkpoint={'path': str(d / 'checkpoint_step0000543.pt'), 'sha256': sha(f)})),
    'other_dir': lambda d, f, s, r: ((OUT / 'elsewhere').mkdir(exist_ok=True), shutil.copyfile(f, OUT / 'elsewhere/checkpoint_step0000544.pt'),
                                     r.update(fork_checkpoint={'path': str(OUT / 'elsewhere/checkpoint_step0000544.pt'), 'sha256': sha(f)})),
    'pin_sha_wrong': lambda d, f, s, r: r.update(fork_checkpoint={'path': str(f), 'sha256': '0' * 64}),
}
for name, mutate in cases.items():
    release = build(name, mutate)
    try:
        out = RUN.validate_target_fork(release, cfg, cohort)
        print(name, 'ACCEPTED', out.name)
    except (ValueError, FileNotFoundError) as e:
        print(name, 'REJECTED', type(e).__name__, str(e)[:90])
