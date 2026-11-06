"""Provenance and execution gates for the reviewed long-dose evaluation only."""
from __future__ import annotations

from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import sys
import resource
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
import torch
import tokenizer_train as T
import tokenizer_evaluation as E
import train_jepa as J
from wearusfm.harness import fm_features as FF
from wearusfm.training.pairing import state_sha256
import consolidation_long_amendment as A

PROTOCOL = 'consolidation-long-evaluation-20261105-v1'
DOSES = (19618, 39237, 49046)
OUTPUT_ROOT = '/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/consolidation_long_20261105'
require = T.require
READING_RULE = {
    'dose_contrasts': [[49046, 19618], [39237, 19618], [49046, 39237]],
    'primary_dose_contrast': [49046, 19618], 'bootstrap_replicates': 2000, 'bootstrap_seed': 20261102,
    'classification_primary_view': 'pool', 'depth_contrast': 'post_minus_pre_same_D_reader',
    'classification_practical_pp': 1.0, 'classification_other_index_lower_margin_pp': -1.0,
    'pose_practical_relative_percent': 2.0,
    'pose_primary_panels': ['post_ridge', 'trained_head'], 'global_pose_reference': None,
    'pose_support': 'crop6, original full-window dynamic selection; E2 historical pairing then crop6 common mask',
    'scope': 'development, one pretraining seed; no multiplicity correction; no official test or universal FM claim',
}
RESOURCE_POLICY = {
    'extract': {'gpus': 1, 'cpus': 8, 'memory_gib': 100, 'seconds': 2700},
    'classification_benchmark': {'gpus': 0, 'cpus': 8, 'memory_gib': 30, 'seconds': 1800},
    'pose_benchmark': {'gpus': 0, 'cpus': 8, 'memory_gib': 30, 'seconds': 900},
    'classification_fit': {'gpus': 0, 'cpus': 8, 'memory_gib': 30, 'seconds': 14400},
    'classification_fit_db2': {'gpus': 0, 'cpus': 8, 'memory_gib': 30, 'seconds': 14400},
    'pose_fit': {'gpus': 0, 'cpus': 8, 'memory_gib': 30, 'seconds': 3600},
    'classification_compare': {'gpus': 0, 'cpus': 8, 'memory_gib': 30, 'seconds': 3600},
    'pose_compare': {'gpus': 0, 'cpus': 8, 'memory_gib': 30, 'seconds': 3600},
    'summary': {'gpus': 0, 'cpus': 2, 'memory_gib': 2, 'seconds': 600},
}


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_pin(path):
    value = T.P.file_identity(path)
    return {k: value[k] for k in ('path', 'sha256', 'bytes')}


def pinned(pin):
    value = T.P.pinned_file(pin)
    require('bytes' not in pin or pin['bytes'] == value['bytes'], 'pinned size differs')
    return Path(value['path'])


def read_pin(pin):
    return json.loads(pinned(pin).read_text())


def write_json(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def source_identity():
    paths = sorted((ROOT / 'src').rglob('*.py')) + sorted((ROOT / 'scripts').glob('*.py'))
    paths += [ROOT / 'scripts/slurm/consolidation_long.sbatch', ROOT / 'scripts/slurm/lib_code_dir.sh']
    return digest({str(p.relative_to(ROOT)): T.R.sha256_file(p) for p in paths})


def recipe_identity(plan):
    return digest({k: plan[k] for k in ('protocol', 'doses', 'views', 'teacher', 'classification_recipe',
                   'pose_recipe', 'reading_rule', 'classification_baseline_policy', 'extraction_batch')})


def validate_plan(plan, *, require_review=True):
    import consolidation_long_classification as C
    import consolidation_long_pose as P
    require(plan['protocol'] == PROTOCOL and plan['doses'] == list(DOSES), 'unregistered protocol/dose')
    require(plan['views'] == ['pre', 'post', 'pool'] and plan['teacher'] == 'teacher', 'wrong readout/encoder')
    require(set(plan['endpoints']) == set(map(str, DOSES)), 'missing endpoint')
    require(plan['classification_recipe'] == {v: C.probe_recipe(v, 384) for v in C.VIEWS}, 'classification recipe changed')
    require(plan['pose_recipe'] == P.recipe(), 'pose recipe changed')
    require(plan['reading_rule'] == READING_RULE and plan['resources'] == RESOURCE_POLICY, 'reading/resource policy changed')
    require(plan['classification_baseline_policy'] in ('exact', 'declared_historical_cache'), 'unregistered baseline policy')
    require(plan['extraction_batch'] == {'classification': 64, 'pose': 32}, 'extraction batch changed')
    import consolidation_long_classification_runtime as CR
    import consolidation_long_pose_runtime as PR
    require(plan['classification_benchmark_policy'] == CR.BENCHMARK_POLICY
            and plan['pose_benchmark_policy'] == PR.BENCHMARK_POLICY, 'benchmark policy changed')
    # Amendment A1: registered fit shards, frozen walltimes, closed input list, recorded decision.
    A.validate_block(plan['amendment'], {CR.shard_key(*shard): shard for shard in
                                         A.registered_shards(C.DOSES, C.VIEWS, C.DATASETS)},
                     require_decision=require_review)
    require(plan['recipe_sha256'] == recipe_identity(plan), 'evaluation recipe hash mismatch')
    require(plan['output_root'] == OUTPUT_ROOT and plan['background_monitor'] is False, 'output/monitor policy changed')
    if require_review:
        review = plan['review']
        require(review['status'] == 'claude_review_converged' and isinstance(review['commit'], str)
                and len(review['commit']) == 40 and all(c in '0123456789abcdef' for c in review['commit']),
                'external Claude review has not converged')
    return plan


def load_context(plan_path, expected_code):
    code = T.P.code_identity(expected_code)
    plan_pin = T.P.file_identity(plan_path)
    plan = validate_plan(json.loads(Path(plan_path).read_text()))
    files = A.source_files(ROOT, T.R.sha256_file)
    require(plan['source_sha256'] == source_identity() == digest(files), 'evaluation sources differ from reviewed plan')
    # Amendment A1: the new source may read the 28 original receipts only if it differs
    # from the original source inside the closed list; new outputs carry the new source.
    inputs = A.validate_inputs(read_pin(plan['amendment']['input_compatibility']), files,
                               {path: (ROOT / path).read_text() for path in A.CHANGED_PATHS}, digest)
    require(inputs['protocol'] == PROTOCOL and inputs['recipe_sha256'] == plan['recipe_sha256']
            and inputs['source_sha256'] != plan['source_sha256']
            and plan['cache_manifest'] == inputs['cache_manifest'], 'amendment inputs differ from this evaluation')
    legacy = read_pin(plan['historical_extraction_plan'])
    for key, pin in legacy['inputs'].items():
        legacy['inputs'][key] = {**pin, 'path': str(pinned(pin))}
    ctx = {'plan': plan, 'plan_pin': plan_pin, 'code_sha': code, 'source_sha256': plan['source_sha256'],
           'inputs': inputs, 'amendment': A.record(),
           'legacy': legacy, 'pose_reader': read_pin(plan['historical_pose_reader_plan'])}
    ctx['caches'] = read_pin(plan['cache_manifest'])
    require(ctx['caches']['recipe_sha256'] == plan['recipe_sha256']
            and ctx['caches']['source_sha256'] == inputs['source_sha256'], 'cache manifest recipe/source differs')
    return ctx


def verify_training_receipt(entry, step):
    receipt = read_pin(entry['receipt'])
    require(receipt['complete'] is True and receipt['code_sha'] == entry['training_code_sha'], 'training receipt incomplete/code differs')
    if step == 19618:
        require(receipt['protocol'] == T.PROTOCOL and receipt['checkpoint_step'] == step
                and receipt['arm'] == 'T100-G' and receipt['checkpoint_sha256'] == entry['checkpoint']['sha256'],
                'reference training receipt differs')
        archive_receipt = read_pin(entry['archive_receipt'])
        require(archive_receipt['complete'] is True and archive_receipt['step'] == 30000
                and all(archive_receipt['persistent_source'][k] == entry['checkpoint'][k]
                        for k in ('path', 'sha256', 'bytes')), 'reference archive not bound to continuation receipt')
    else:
        require(receipt['protocol'] == 'consolidation-t100g-20261104-v1' and receipt['step'] == step
                and receipt['summary']['steps'] == step and receipt['summary']['stopped'] == 'max_steps'
                and receipt['plan_sha256'] == entry['training_plan_sha256'], 'long training dose/plan differs')
        archived = receipt['persistent_checkpoint']
        require(all(archived[k] == entry['checkpoint'][k] for k in ('path', 'sha256', 'bytes'))
                and receipt['checkpoint_sha256'] == archived['sha256'], 'durable checkpoint differs from receipt')
        require(all(receipt['persistent_source'][k] == entry['reference_checkpoint'][k]
                    for k in ('path', 'sha256', 'bytes')), 'continuation source archive differs')
    return receipt


def load_teacher(ctx, step, device='cpu'):
    require(step in DOSES, 'unregistered endpoint')
    legacy, plan = ctx['legacy'], ctx['plan']
    entry = plan['endpoints'][str(step)]
    receipt = verify_training_receipt(entry, step)
    comparison = read_pin(legacy['comparison'])
    argv = comparison['arm_configurations']['T100-G']['validated_trainer_argv']
    require(argv[0] == 'scripts/train_jepa.py', 'unexpected historical trainer')
    cfg = replace(J.build_config(J.build_parser().parse_args(argv[1:])), max_steps=step)
    require(cfg.model.dim == 384 and cfg.model.k_latents == 64 and cfg.model.window_ms == 100
            and cfg.model.tokenizer_activation == 'gelu' and cfg.model.pose_head,
            'not the registered T100-G teacher')
    require(cfg.schedule.total_steps == 49046 and cfg.schedule.decay_start == 39237, 'schedule differs')
    scales = json.loads(pinned(legacy['inputs']['scales']).read_text())['scales']
    windows = json.loads(pinned(legacy['inputs']['windows']).read_text())['window_s']
    signature = canonical(T.R.calibration_signature(cfg, legacy['inputs']['manifest']['path'], scales, windows))
    path = pinned(entry['checkpoint'])
    require(not (path.parent / 'STOP').exists(), 'STOP present')
    state_guard = T.checkpoint_identity(path, cfg, step, signature)
    model = FF.load_model(str(path), device=device, which='teacher')
    state = torch.load(path, map_location='cpu', weights_only=False)
    teacher = state['teacher']
    actual = model.state_dict()
    require(actual.keys() == teacher.keys() and all(v.dtype == teacher[k].dtype and
            torch.equal(v.detach().cpu(), teacher[k]) for k, v in actual.items()), 'teacher tensors/gain not restored exactly')
    del state, teacher, actual
    gc.collect()
    require(canonical(asdict(model.cfg)) == T.R.config_to_dict(cfg)['model'], 'loaded teacher config differs')
    require(model.tokenizer.tokenizer_activation == 'gelu' and model.tokenizer.window_s == .1,
            'loaded tokenizer differs')
    require(np.isfinite(float(model.tokenizer.fe_gain)) and float(model.tokenizer.fe_gain) > 0, 'invalid restored gain')
    model.eval()
    ctx.setdefault('_checkpoint_guards', []).append(state_guard)
    ctx.setdefault('_models', []).append((model, state_sha256(model.state_dict())))
    return model, {'checkpoint': str(path), 'checkpoint_sha256': state_guard['sha256'], 'checkpoint_step': step,
                   'which': 'teacher', 'config': T.R.config_to_dict(cfg), 'training_code_sha': receipt['code_sha'],
                   'checkpoint_pose_stats_sha256': legacy['inputs']['pose_stats']['sha256'],
                   'restored_state_verified': True}


def begin_output(ctx, out, kind, **fields):
    out = Path(out)
    root = Path(ctx['plan']['output_root'])
    require(out.is_absolute() and out != root and out.resolve().is_relative_to(root)
            and not out.exists() and not out.is_symlink()
            and not any(p.is_symlink() for p in out.parents), 'fresh external output required')
    out.mkdir(parents=True)
    ctx['_started'] = time.perf_counter()
    if ctx.get('_models') and torch.cuda.is_available():
        # The model is already loaded; include allocations retained at entry.
        torch.cuda.reset_peak_memory_stats()
    return {'protocol': PROTOCOL, 'complete': False, 'kind': kind, 'code_sha': ctx['code_sha'],
            'plan_sha256': ctx['plan_pin']['sha256'], 'source_sha256': ctx['source_sha256'],
            'recipe_sha256': ctx['plan']['recipe_sha256'], 'amendment': ctx['amendment'], **fields}


def finish_output(ctx, out, report):
    require(T.P.file_identity(ctx['plan_pin']['path']) == ctx['plan_pin'], 'plan changed during execution')
    T.P.code_identity(ctx['code_sha'])
    require(source_identity() == ctx['source_sha256'], 'sources changed during execution')
    for guard in ctx.get('_checkpoint_guards', []):
        require(file_pin(guard['path']) == {k: guard[k] for k in ('path', 'sha256', 'bytes')}
                and not (Path(guard['path']).parent / 'STOP').exists(), 'checkpoint changed or STOP appeared')
    for model, before in ctx.get('_models', []):
        require(state_sha256(model.state_dict()) == before, 'frozen teacher changed during extraction')
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    measured = {'peak_rss_bytes': int(rss), 'elapsed_seconds': time.perf_counter() - ctx.get('_started', time.perf_counter())}
    if ctx.get('_models'):
        reserved = torch.cuda.max_memory_reserved()
        total = torch.cuda.get_device_properties(0).total_memory
        require(reserved <= .85 * total and rss <= .85 * 100 * 1024**3, 'extraction memory margin failed')
        measured.update(peak_reserved_bytes=reserved, device_memory_bytes=total)
    value = {**report, 'complete': True, 'checkpoint_unchanged': bool(ctx.get('_checkpoint_guards')),
             'measured_resources': measured,
             'frozen_teacher_unchanged': bool(ctx.get('_models'))}
    write_json(Path(out) / 'report.json', value)
    return value


def verified_input(ctx, pin, kind):
    """Amendment A1: read one of the 28 listed original receipts; nothing else is an input."""
    inputs = ctx['inputs']
    listed = [r for r in inputs['receipts'] if pin == {k: r[k] for k in ('path', 'sha256', 'bytes')}]
    require(len(listed) == 1 and listed[0]['kind'] == kind, 'receipt is not a listed amendment input of this kind')
    report, phase = read_pin(pin), inputs['phases'][listed[0]['phase']]
    require(report.get('protocol') == PROTOCOL and report.get('complete') is True and report.get('kind') == kind
            and report.get('source_sha256') == inputs['source_sha256'] != ctx['source_sha256']
            and report.get('recipe_sha256') == inputs['recipe_sha256'] == ctx['plan']['recipe_sha256']
            and report.get('code_sha') == phase['code_sha'] and report.get('plan_sha256') == phase['plan_sha256']
            and all(report.get(k) == v for k, v in listed[0]['fields'].items()),
            'amendment input receipt provenance differs')
    return report


def verified_extraction(ctx, entry, step, kind):
    report = verified_input(ctx, entry['receipt'], kind)
    require(report['complete'] is True and report['source_sha256'] == ctx['inputs']['source_sha256']
            and report['recipe_sha256'] == ctx['plan']['recipe_sha256']
            and report['checkpoint_step'] == step and report['kind'] == kind
            and report['checkpoint_sha256'] == ctx['plan']['endpoints'][str(step)]['checkpoint']['sha256'],
            'extraction receipt differs')
    require(report.get('which') == 'teacher' and report.get('checkpoint_unchanged') is True
            and report.get('frozen_teacher_unchanged') is True, 'extraction immutability attestation missing')
    pinned(entry['cache'])
    require(entry['cache'] in report['artifacts'].values(), 'cache not covered by extraction receipt')
    return report
