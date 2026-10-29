#!/bin/bash
#SBATCH -A IscrB_WearUsFM
#SBATCH -p boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH -J pose_b19618
#SBATCH -o /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/pose_b19618_%j.out
#SBATCH -e /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/pose_b19618_%j.out

# One authorized continuation: 4 GPUs * 4 h = 16 GPUh; 128 coreh.
# Submit this versioned wrapper on stdin; the scientific checkout stays 24139ab.
# Required exports: CODE_DIR, EXPECTED_SHA, WRAPPER_SHA256, B19618_SOURCE_RUN,
# B19618_RUN_DIR. No fallback, existing destination, warm-start or automatic retry.
# The source B5000 is read only; ordinary trainer resume uses a byte-identical
# independent checkpoint copy in the fresh destination (never a hardlink).
set -euo pipefail
: "${CODE_DIR:?}" "${EXPECTED_SHA:?}" "${WRAPPER_SHA256:?}" "${B19618_SOURCE_RUN:?}" "${B19618_RUN_DIR:?}"
[ "$EXPECTED_SHA" = 24139ab901c052357a46a65bc20466f2eb1e8358 ] || exit 2
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
module list 2>&1
python3 - <<'PY_B19618'
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

CODE_SHA = '24139ab901c052357a46a65bc20466f2eb1e8358'
STATS_SHA = '757f2aeebf688063747d50e55d983dd6b3c4cea268ce43d2bae06f3e99524a7e'
MANIFEST_SHA = '6d6d8167b882eb4079fcc566748028dc6de81692859f220a0246291c198574ec'
SOURCE_HASHES = {
    'checkpoint.pt': '815956b98d27cd2ff71f1255d0e220e2b0e08e467e7bedf93a54d0984b10783e',
    'config.json': 'f55fa6a5f117ad8898c20bbaf8b20f288c169a9a9a12e7bf3bfdad26fef1d157',
    'summary.json': 'd5ffc17b878d38af57ea7f54b0c76ae78b5016a99a4b099dfd679bf0324b2a10',
    'pose_panel_set.json': '7b21e1c079111a11685dcc46a4606cca5110619fbab854a2f69f51153741c018',
    'fixed_eval_set.json': 'af2f7cb1f54449af22a8bac8566c3736117ab15699474cc8b9609babfa028af5',
    'pip_freeze_62523672.txt': 'f97a0225bfecbb5d25ddc1056e9be6519b801dc4fed2186e7109134b1d2c8f36',
}
END_STEP, INTERNAL_SECONDS = 19618, 13800


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fresh_destination(out, source, roots):
    require(out.is_absolute() and not out.exists() and not out.is_symlink(), 'destination must be absolute and new')
    resolved, original = out.resolve(), source.resolve()
    require(original != resolved and original not in resolved.parents and resolved not in original.parents, 'destination overlaps source B')
    require(not any(resolved == r.resolve() or r.resolve() in resolved.parents for r in roots), 'destination inside code')


def trainer_args(work, scratch, out):
    base = work/'wearusfm_runs/results/passo6'
    return ['--preset','sanity', '--manifest',str(base/'derived_v1_1/manifest_v1_1_nested_1.json.gz'),
            '--root',str(work/'data/processed'), '--root',str(scratch/'data/processed'),
            '--scales',str(base/'loader_59623700/session_scales.json'), '--out-dir',str(out),
            '--num-workers','6', '--distributed', '--max-steps',str(END_STEP), '--time-limit-s',str(INTERNAL_SECONDS),
            '--window1-rules', '--datasets','all', '--prefetch', '--fixed-eval-every','500',
            '--manifest-sha256',MANIFEST_SHA, '--erank-min','38.4',
            '--window-seconds',str(base/'window_seconds_59425316.json'), '--batch-size','32', '--accum-steps','1',
            '--seed','0', '--schedule','wsd', '--total-steps','49046', '--steps-at-d','98094',
            '--keep-checkpoints','19618', '--lr','0.0001', '--nested','1', '--jepa-weight','0',
            '--pose-root',str(scratch/'data/processed/emg2pose_pose'), '--pose-weight','1.0', '--pose-unprompted','on',
            '--pose-grad-every','500', '--pose-panel-every','500', '--pose-panel-users','32',
            '--pose-panel-recordings','2', '--pose-panel-windows','4']


def check_recipe(old, new):
    require(old.get('max_steps') == 5000 and new.get('max_steps') == END_STEP, 'wrong continuation endpoints')
    require({k:v for k,v in old.items() if k != 'max_steps'} == {k:v for k,v in new.items() if k != 'max_steps'},
            'continuation recipe differs beyond max_steps')
    require(new['schedule'] == {'kind':'wsd','total_steps':49046,'decay_fraction':.2,'keep_checkpoints':[19618]}, 'schedule changed')
    require(new['warmup_steps'] == 1962, 'warmup changed')


def check_checkpoint(state, config, step, torch):
    require(state.get('step') == step and state.get('config') == config, 'checkpoint step/config mismatch')
    require(state.get('pose_stats_sha256') == STATS_SHA, 'checkpoint pose stats mismatch')
    require(state.get('stream') == {'num_workers':6,'segment':1200,'world_size':4}, 'stream identity mismatch')
    require(state.get('world_size') == 4 and len(state.get('rank_rng', [])) == 4, 'DDP rank state missing')
    for rank in state['rank_rng']:
        for name in ('torch_rng','cuda_rng'):
            value = rank.get(name)
            require(isinstance(value, torch.Tensor) and value.dtype == torch.uint8 and value.ndim == 1 and value.numel() > 0,
                    'missing/malformed rank RNG: '+name)
    for name in ('student','teacher'):
        weights = state.get(name)
        require(isinstance(weights, dict) and bool(weights), 'missing model state: '+name)
        require(all(isinstance(t,torch.Tensor) and bool(torch.isfinite(t).all()) for t in weights.values()), 'invalid model tensors')
    optimizer = state.get('optimizer', {})
    require(bool(optimizer.get('state')) and bool(optimizer.get('param_groups')), 'missing optimizer moments')
    require(isinstance(state.get('data_signature'),dict) and bool(state['data_signature']), 'missing data signature')
    require(len(state.get('alarm_counts',())) == 2, 'missing alarm counters')


def seed_destination(source, out, hashes):
    # Directory creation is exclusive; a failed attempt is not silently resumed.
    out.mkdir(parents=True, exist_ok=False)
    for name in ('checkpoint.pt','pose_panel_set.json','fixed_eval_set.json'):
        with (source/name).open('rb') as src, (out/name).open('xb') as dst:
            shutil.copyfileobj(src, dst, length=8*1024*1024)
        require(sha(out/name) == hashes[name], 'copied input SHA mismatch: '+name)
        require((source/name).stat().st_ino != (out/name).stat().st_ino or
                (source/name).stat().st_dev != (out/name).stat().st_dev, 'copy unexpectedly aliases source')


def summary_complete(summary):
    return (summary.get('steps') == END_STEP and summary.get('stopped') == 'max_steps'
            and summary.get('world_size') == 4 and summary.get('code',{}).get('sha') == CODE_SHA
            and summary.get('pose_stats_sha256') == STATS_SHA)


def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    import torch
    from wearusfm.training import run as R
    source, out = Path(os.environ['B19618_SOURCE_RUN']), Path(os.environ['B19618_RUN_DIR'])
    work, scratch, code = (Path(os.environ[k]) for k in ('WORK','SCRATCH','CODE_DIR'))
    require(os.environ.get('WEARUSFM_CODE_SHA') == CODE_SHA, 'wrong frozen code')
    wrapper = os.environ['WRAPPER_SHA256']
    require(len(wrapper) == 64 and all(c in '0123456789abcdef' for c in wrapper), 'invalid wrapper SHA')
    require(torch.cuda.device_count() == 4 and int(os.environ['SLURM_CPUS_PER_TASK']) == 32
            and int(os.environ['SLURM_JOB_NUM_NODES']) == 1, 'requires one node, 4 visible GPUs, 32 CPUs')
    fresh_destination(out, source, [code,work/'wearusfm',work/'wearusfm_campaigns'])
    require(not (source/'STOP').exists(), 'source B has STOP')
    actual = {name:sha(source/name) for name in SOURCE_HASHES}
    require(actual == SOURCE_HASHES, 'source B immutable input SHA mismatch')
    summary, old = [json.loads((source/name).read_text()) for name in ('summary.json','config.json')]
    require(summary.get('steps') == 5000 and summary.get('stopped') == 'max_steps' and summary.get('world_size') == 4,
            'B source is not completed at step5000/DDP4')
    spec = importlib.util.spec_from_file_location('frozen_train_jepa',code/'scripts/train_jepa.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    argv = trainer_args(work,scratch,out)
    parsed = cli.build_parser().parse_args(argv)
    cfg = cli.build_config(parsed)
    new = R.config_to_dict(cfg)
    check_recipe(old,new)
    require(sha(Path(cfg.loader.pose_root)/'pose_stats.json') == STATS_SHA, 'pose statistics changed')
    R.check_input_files(parsed.manifest,parsed.window_seconds,parsed.nested,MANIFEST_SHA)
    scales = json.loads(parsed.scales.read_text())['scales']
    windows = json.loads(parsed.window_seconds.read_text())['window_s']
    state = torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=False)
    check_checkpoint(state,old,5000,torch)
    R._check_resume_recipe(cfg,state)
    signature = json.loads(json.dumps(R.calibration_signature(cfg,parsed.manifest,scales,windows)))
    require(state['data_signature'] == signature, 'manifest/scales/window/loader signature changed')
    del state
    environment = {'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda}
    require(all(environment[k] == summary['environment'][k] for k in environment), 'Python/torch/CUDA environment changed')
    freeze = subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    require(sorted(freeze.splitlines()) == sorted((source/'pip_freeze_62523672.txt').read_text().splitlines()), 'pip environment changed')
    freeze = '\n'.join(sorted(freeze.splitlines()))+'\n'
    environment.update(pip_freeze_sha256=hashlib.sha256(freeze.encode()).hexdigest(), modules_loaded=os.environ.get('LOADEDMODULES'))
    command = ['torchrun','--standalone','--nproc_per_node=4',str(code/'scripts/train_jepa.py'),*argv]
    provenance = {'schema_version':1,'source_run':str(source),'source_hashes':actual,'source_step':5000,'expected_step':END_STEP,
                  'code_sha':CODE_SHA,'wrapper_sha256':wrapper,'environment':environment,'config':new,'data_signature':signature,
                  'command':command,'slurm_job_id':os.environ.get('SLURM_JOB_ID'), 'copied_logs':False,
                  'note':'Exact-state resume from independent checkpoint copy; new metrics begin at5001. Source B untouched; no retry.'}
    seed_destination(source,out,actual)
    (out/f"pip_freeze_{os.environ.get('SLURM_JOB_ID','local')}.txt").write_text(freeze)
    write_json(out/'continuation_provenance.json',provenance)
    print(json.dumps({'preflight':'passed','command':command}),flush=True)
    receipt = {'schema_version':1,'expected_step':END_STEP,'complete':False,'source_run':str(source),
               'code_sha':CODE_SHA,'wrapper_sha256':wrapper,'environment':environment,'source_unchanged':False,
               'provenance_sha256':sha(out/'continuation_provenance.json')}
    try:
        rc = subprocess.run(command,check=False).returncode
        receipt['trainer_exit_code'] = rc
        require(rc == 0, f'trainer failed with {rc}')
        require(not (out/'STOP').exists(), 'continuation has STOP')
        final_summary = json.loads((out/'summary.json').read_text())
        receipt['summary'] = final_summary
        require(summary_complete(final_summary), 'continuation incomplete (including internal time limit)')
        require(json.loads((out/'config.json').read_text()) == new, 'final config differs')
        for name in ('checkpoint.pt',f'checkpoint_step{END_STEP:07d}.pt'):
            final_state = torch.load(out/name,map_location='cpu',weights_only=False)
            check_checkpoint(final_state,new,END_STEP,torch)
            require(final_state['data_signature'] == signature, 'final data signature differs')
            del final_state
        require(all(sha(out/name) == actual[name] for name in ('pose_panel_set.json','fixed_eval_set.json')), 'diagnostic panel identities changed')
        receipt['outputs'] = {name:{'sha256':sha(out/name),'bytes':(out/name).stat().st_size}
                              for name in ('checkpoint.pt',f'checkpoint_step{END_STEP:07d}.pt','config.json','summary.json')}
        receipt['complete'] = True
    except Exception as exc:
        receipt['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        try:
            receipt['source_unchanged'] = all(sha(source/name) == digest for name,digest in actual.items())
        except Exception as exc:
            receipt['source_check_error'] = f'{type(exc).__name__}: {exc}'
            receipt['source_unchanged'] = False
        receipt['complete'] = receipt['complete'] and receipt['source_unchanged']
        write_json(out/'continuation_receipt.json',receipt)
    print(json.dumps(receipt),flush=True)
    return 0 if receipt['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
PY_B19618
