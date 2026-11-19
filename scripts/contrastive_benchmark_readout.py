"""Read the frozen contrastive preflight; never submit or retune a candidate."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import torch  # preserve production import order
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from contrastive_prepare import _read, _loads, config_from_dict, source_digest, authenticate_calibration, calibrated_configs
from wearusfm.model.fm import WearUsFM
from wearusfm.training.contrastive_objectives import attach_contrastive_head, ContrastiveConfig, ContrastiveRuntime
from wearusfm.training.pairing import state_sha256

ARMS = ('C0', 'CN', 'CR')
ANCHORS = ('log_rms', 'band_shape', 'log_env', 'ms_fast', 'ms_slow')
RATIO_RULE = {'median_min': .005, 'median_max': .5, 'p95_max': 1., 'between_min': .25, 'between_max': 4.}
RESUME_RELATIVE_DISPLACEMENT = .01
HISTORICAL_INITIAL_REFERENCE = ROOT/'results/infra/contrastive_comparison_20261116/review_response/initial_historical_reference.json'
HISTORICAL_INITIAL_SHA256 = '367afd99165240014c20ec42a0ecd66e1d9afffa3e3a319f2068f200d3aebb52'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quantile(values, q):
    values = sorted(values)
    at = (len(values)-1)*q
    lo, hi = math.floor(at), math.ceil(at)
    return values[lo] + (values[hi]-values[lo])*(at-lo)


def component_ratios(records, cfg):
    """All 64 dense measurements; actual scheduled weights, global shared norms."""
    panel = [r for r in records if 513 <= r['step'] <= 576]
    if [r['step'] for r in panel] != list(range(513, 577)):
        raise ValueError('missing full-weight gradient panel513..576')
    out = {}
    policy = ContrastiveConfig(**cfg['contrastive'])
    for record in panel:
        if tuple(record.get('contrast_'+term+'_weight') for term in ('nce', 'rnc')) != policy.weights(record['step'] - 1):
            raise ValueError('dense panel scheduled contrastive weights differ')
    for term in ('nce', 'rnc'):
        weight = cfg['contrastive'][term+'_weight']
        values = []
        for r in panel:
            base, norm = r.get('contrast_global_grad_base'), r.get('contrast_global_grad_'+term)
            if any(type(x) not in (int, float) or not math.isfinite(x) for x in (base, norm)) or base <= 0 or norm < 0:
                raise ValueError('missing/nonfinite global gradient norm')
            values.append(weight*norm/base)
        median, p95 = quantile(values, .5), quantile(values, .95)
        active = cfg['contrastive']['arm'] == ('CN' if term == 'nce' else 'CR')
        out[term] = {'active': active, 'hypothetical_full_weight_ratio': values,
                     'median': median, 'p95': p95,
                     'within_screening_bounds': RATIO_RULE['median_min'] <= median <= RATIO_RULE['median_max'] and p95 <= RATIO_RULE['p95_max']}
    return out


def measured_allocation(summary, records):
    """Conservative full-run projection, explicitly including diagnostic overhead.

    Step time already includes loader time. Take the larger of
    full end-to-end mean and nondiagnostic mean plus measured diagnostic premium
    at the future frequency; reserve50% and300s per phase; round up5minutes.
    """
    elapsed = summary.get('elapsed_s')
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('invalid measured elapsed time')
    normal, diagnostic = [], []
    for r in records[8:]:
        v = r.get('t_step_s')
        if type(v) not in (float, int) or not math.isfinite(v) or v <= 0:
            raise ValueError('invalid measured update time')
        (diagnostic if 'contrast_global_grad_base' in r else normal).append(v)
    if not normal or not diagnostic:
        raise ValueError('benchmark lacks normal or diagnostic updates')
    mean = lambda x: sum(x)/len(x)
    per_update = max(elapsed/576, mean(normal) + max(0., mean(diagnostic)-mean(normal))*.02)
    # Two independent allocations, keeping the planned5k checkpoint boundary.
    phase_seconds = [int(math.ceil((1.5*per_update*n + 300)/300)*300) for n in (5000, 14618)]
    return {'observed_end_to_end_s': elapsed, 'normal_update_mean_s': mean(normal),
            'diagnostic_update_mean_s': mean(diagnostic), 'conservative_update_s': per_update,
            'projected_gpu_hours': 4*per_update*19618/3600,
            'phase_allocation_seconds': phase_seconds,
            'allocated_gpu_hours': 4*sum(phase_seconds)/3600,
            'allocated_core_hours': 32*sum(phase_seconds)/3600}


def initial_calibration_description(run_dir):
    """Predeclared descriptive comparison; never changes weights or pass/fail."""
    if sha(HISTORICAL_INITIAL_REFERENCE) != HISTORICAL_INITIAL_SHA256:
        raise ValueError('historical first-eight reference changed')
    reference = _read(HISTORICAL_INITIAL_REFERENCE)
    rows = [_loads(line) for line in (Path(run_dir)/'metrics.jsonl').read_text().splitlines() if line.strip()][:8]
    if [r.get('step') for r in rows] != list(range(1, 9)):
        raise ValueError('calibration lacks initial eight updates')
    values = {key: [row.get(key) for row in rows] for key in reference['historical_first8']}
    if any(type(v) not in (float, int) or not math.isfinite(v) for series in values.values() for v in series):
        raise ValueError('calibration initial diagnostics missing/nonfinite')
    means = {key: sum(series)/8 for key, series in values.items()}
    return {'descriptive_only': True, 'historical_reference_sha256': HISTORICAL_INITIAL_SHA256,
            'historical_first8': reference['historical_first8'], 'historical_means': reference['historical_means'],
            'calibration_first8': values, 'calibration_means': means,
            'mean_ratios': {key: means[key]/reference['historical_means'][key] for key in means},
            'interpretation': 'Descriptive historical reference only: target supports changed from RMS25/fast50 to RMS100/fast125, so even update 1 need not match. Subsequent historical weights trained while calibration weights stay fixed.'}


def _tensor_distance(a, b, reference=None):
    if set(a) != set(b) or (reference is not None and set(a) != set(reference)):
        raise ValueError('checkpoint tensor keys differ')
    error2 = scale2 = 0.
    for key in a:
        if a[key].dtype != b[key].dtype or (reference is not None and a[key].dtype != reference[key].dtype):
            raise ValueError('checkpoint tensor dtypes differ')
        x, y = a[key].double(), b[key].double()
        if x.shape != y.shape or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('nonfinite/mismatched checkpoint tensor')
        error2 += float((x-y).square().sum())
        scale = x if reference is None else x-reference[key].double()
        scale2 += float(scale.square().sum())
    return math.sqrt(error2)/max(math.sqrt(scale2), 1e-12)


def _nested_equal(a, b, path='state', floating_tolerance=None, floating_errors=None):
    """Exact structure/metadata/RNG; optionally collect finite floating violations."""
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(path+' tensor schema differs')
        if a.is_floating_point() and floating_tolerance is not None and not path.endswith('.step'):
            error = _tensor_distance({'x': a}, {'x': b})
            if not math.isfinite(error):
                raise ValueError(path+' nonfinite numeric distance')
            if error > floating_tolerance:
                if floating_errors is None:
                    raise ValueError(path+' differs beyond1% numeric bound')
                floating_errors[path] = {'relative_error': error, 'bound': floating_tolerance}
        elif not torch.equal(a, b):
            raise ValueError(path+' differs exactly')
    elif isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
            raise ValueError(path+' ndarray differs exactly')
    elif isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise ValueError(path+' dictionary differs')
        for k in a: _nested_equal(a[k], b[k], path+'.'+str(k), floating_tolerance, floating_errors)
    elif isinstance(a, (tuple, list)):
        if type(a) != type(b) or len(a) != len(b): raise ValueError(path+' sequence differs')
        for i, (x, y) in enumerate(zip(a, b)): _nested_equal(x, y, path+'.'+str(i), floating_tolerance, floating_errors)
    elif type(a) != type(b) or a != b:
        raise ValueError(path+' metadata differs')


def compare_resume(full_path, resumed_path, cfg, initial_sha, initial):
    a = torch.load(str(full_path), map_location='cpu', weights_only=False, mmap=True)
    b = torch.load(str(resumed_path), map_location='cpu', weights_only=False, mmap=True)
    if a['step'] != 576 or b['step'] != 576:
        raise ValueError('resume comparison needs576-update complete checkpoints')
    if state_sha256(initial) != initial_sha:
        raise ValueError('saved calibration initialization differs from calibrated cohort; no reconstruction tolerance')
    # Instantiate only to classify parameters/buffers; values come exclusively
    # from the authenticated fixed-weight calibration checkpoint.
    with torch.random.fork_rng(devices=[]):
        model = attach_contrastive_head(WearUsFM(cfg.model))
    parameters = {name for name, _ in model.named_parameters()}
    if set(model.state_dict()) != set(initial):
        raise ValueError('initial checkpoint schema differs from matched architecture')
    del model
    relative = {}
    for part in ('student', 'teacher'):
        if set(a[part]) != set(initial) or set(b[part]) != set(initial):
            raise ValueError('resumed model state schema differs from initial state')
        _nested_equal({k: v for k, v in a[part].items() if k not in parameters},
                      {k: v for k, v in b[part].items() if k not in parameters}, part+'.buffers')
        relative[part] = _tensor_distance({k: v for k, v in a[part].items() if k in parameters},
                                         {k: v for k, v in b[part].items() if k in parameters},
                                         {k: v for k, v in initial.items() if k in parameters})
    floating_errors = {}
    for part in ('optimizer', 'contrastive_state'):
        if part not in a or part not in b: raise ValueError('checkpoint missing '+part)
        _nested_equal(a[part], b[part], part, .01, floating_errors)
    for part in ('rank_rng', 'stream', 'config', 'data_signature', 'pose_stats_sha256', 'contrastive_provenance',
                 'contrastive_initial_sha256', 'alarm_counts'):
        if part not in a or part not in b: raise ValueError('checkpoint missing '+part)
        _nested_equal(a[part], b[part], part)
    failure_reasons = [f'{part}: relative displacement {value:.17g} exceeds {RESUME_RELATIVE_DISPLACEMENT:.17g}'
                       for part, value in relative.items() if value > RESUME_RELATIVE_DISPLACEMENT]
    failure_reasons.extend(f'{path}: relative error {row["relative_error"]:.17g} exceeds {row["bound"]:.17g}'
                           for path, row in floating_errors.items())
    return {'passed': all(v <= RESUME_RELATIVE_DISPLACEMENT for v in relative.values()) and not floating_errors,
            'relative_to_full_displacement': relative, 'bound': RESUME_RELATIVE_DISPLACEMENT,
            'floating_state_errors': floating_errors, 'failure_reasons': failure_reasons,
            'full_sha256': sha(full_path), 'resumed_sha256': sha(resumed_path)}


def _run_evidence(folder, arm, cohort, expected_cfg, provenance, first_update=1, expected_phase="full"):
    folder = Path(folder)
    summary, cfg = _read(folder/'summary.json'), _read(folder/'config.json')
    records = [_loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines() if line.strip()]
    if (cfg != expected_cfg or any(type(summary.get(k)) is not int or summary[k] != value for k, value in
            (("steps", 576), ("world_size", 4), ("workers_per_rank", 6), ("batch_per_process", 8), ("accum_steps", 1))) or
            summary.get('stopped') != 'max_steps' or (folder/'STOP').exists() or
            any(type(r.get('step')) is not int for r in records) or [r['step'] for r in records] != list(range(first_update, 577))):
        raise ValueError('incomplete/unmatched benchmark '+arm)
    from contrastive_numeric_profile import assert_numeric_flags
    numeric = []
    for rank in range(4):
        file = folder/f'numeric_profile.rank{rank}.json'
        row = _read(file)
        if (row.get('rank') != rank or row.get('world_size') != 4 or
                row.get('source_sha256') != cohort['source_sha256'] or
                row.get('config_canonical_sha256') != hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',', ':')).encode()).hexdigest() or
                row.get('phase') != expected_phase):
            raise ValueError('production numeric profile rank/source/config mismatch')
        assert_numeric_flags(row.get('flags', {}))
        numeric.append(row)
    if any(row['flags'] != numeric[0]['flags'] for row in numeric[1:]):
        raise ValueError('production numeric flags differ across ranks')
    if type(summary.get('gpu_max_memory_gib')) not in (int,float) or not math.isfinite(summary['gpu_max_memory_gib']) or summary['gpu_max_memory_gib'] <= 0:
        raise ValueError('missing full-batch peak GPU memory measurement')
    c = summary.get('contrastive', {})
    if (c.get('initial_sha256') != cohort['initial_state_sha256'] or c.get('config') != cfg['contrastive'] or
            c.get('provenance') != provenance or c.get('optimizer_updates') != 576 or c.get('queue_last_step') != 575):
        raise ValueError('benchmark provenance/config/counts differ from common calibration')
    policy = ContrastiveConfig(**cfg['contrastive'])
    for r in records:
        scalar_keys = ('total', 'pose', 'grad_norm', 'contrast_nce', 'contrast_rnc', 'contrast_projector_grad_norm', *ANCHORS)
        if (any(type(r.get(k)) not in (int, float) or not math.isfinite(r[k]) for k in scalar_keys) or
                r.get('contrastive_calibration_only') is not False or r.get('windows') != 32 or
                not isinstance(r.get('origin'), list) or len(r['origin']) != 32):
            raise ValueError('missing/nonfinite benchmark metrics or origin evidence')
        wn, wr = policy.weights(r['step'] - 1)
        if (r.get('contrast_nce_weight'), r.get('contrast_rnc_weight')) != (wn, wr):
            raise ValueError('benchmark scheduled weights differ')
        measured = all(type(r.get('contrast_global_grad_'+term)) in (int, float) and
                       math.isfinite(r['contrast_global_grad_'+term]) and r['contrast_global_grad_'+term] >= 0
                       for term in ('base', 'nce', 'rnc'))
        if measured != policy.wants_gradients(r['step'] - 1):
            raise ValueError('benchmark gradient diagnostic schedule differs')
        if measured and (type(r.get('contrast_rnc_cross_window_rank_valid')) not in (float, int) or
                         r.get('contrast_rnc_cross_window_rank_valid') not in (0., 1.) or
                         type(r.get('contrast_rnc_cross_window_rank_correlation')) not in (float, int) or
                         not -1 <= r['contrast_rnc_cross_window_rank_correlation'] <= 1 or
                         type(r.get('contrast_rnc_cross_window_rank_pairs')) not in (float, int) or
                         r['contrast_rnc_cross_window_rank_pairs'] < 0):
            raise ValueError('benchmark cross-window alignment diagnostic missing/invalid')
        expected = .2*sum(r[k] for k in ANCHORS)+r['pose']+wn*r['contrast_nce']+wr*r['contrast_rnc']
        if not math.isclose(r['total'], expected, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError('weighted objective composition differs')
    checkpoint = torch.load(str(folder/'checkpoint.pt'), map_location='cpu', weights_only=False, mmap=True)
    if (type(checkpoint.get('step')) is not int or checkpoint['step'] != 576 or checkpoint.get('world_size') != 4 or
            checkpoint.get('config') != cfg or checkpoint.get('contrastive_initial_sha256') != cohort['initial_state_sha256'] or
            checkpoint.get('contrastive_provenance') != provenance or len(checkpoint.get('rank_rng', [])) != 4 or
            checkpoint.get('stream') != {'num_workers': 6, 'segment': cfg['stream_segment'], 'world_size': 4} or
            checkpoint.get('pose_stats_sha256') != provenance['input_sha256']['pose_stats'] or
            checkpoint.get('contrastive_state', {}).get('queue', {}).get('last_step') != 575):
        raise ValueError('checkpoint metadata differs from complete benchmark')
    for part in ('student', 'teacher'):
        if state_sha256(checkpoint[part]) != c.get('final_'+part+'_sha256'):
            raise ValueError('checkpoint tensor state differs from benchmark summary')
    engine = ContrastiveRuntime(cfg['contrastive'])
    engine.expected_datasets = tuple(c.get('expected_datasets', []))
    engine.load_state_dict(checkpoint['contrastive_state'], 576)
    if engine.coverage_reports != c.get('coverage_completed_blocks') or engine.coverage_report() != c.get('coverage_partial_block'):
        raise ValueError('benchmark coverage checkpoint differs from summary')
    pins = {name: sha(folder/name) for name in ('config.json', 'summary.json', 'metrics.jsonl', 'checkpoint.pt', 'origin_sessions.json')}
    pins.update({f'numeric_profile.rank{rank}.json': sha(folder/f'numeric_profile.rank{rank}.json') for rank in range(4)})
    return summary, cfg, records, pins


def _initial_from_calibration(run_dir, receipt):
    saved = torch.load(str(Path(run_dir)/'checkpoint.pt'), map_location='cpu', weights_only=False, mmap=True)
    if (saved.get('step') != 512 or saved.get('optimizer', {}).get('state') or
            saved.get('config') != _read(Path(run_dir)/'config.json') or
            any(state_sha256(saved[part]) != receipt['initial_state_sha256'] for part in ('student', 'teacher'))):
        raise ValueError('calibration checkpoint is not the authenticated fixed initial state')
    return saved['student']


def _cuda_evidence(path, cohort, input_pins, calibration_config):
    from contrastive_cuda_check import PROTOCOL, CRITERIA, TECHNICAL_OVERRIDES
    cuda = _read(path)
    if (cuda.get('passed') is not True or cuda.get('source_sha256') != cohort['source_sha256'] or
            cuda.get('protocol') != PROTOCOL or cuda.get('criteria') != CRITERIA or cuda.get('technical_overrides') != TECHNICAL_OVERRIDES or
            cuda.get('world_size') != 4 or cuda.get('inputs_unchanged') is not True or
            cuda.get('config_canonical_sha256') != hashlib.sha256(json.dumps(calibration_config, sort_keys=True, separators=(',', ':')).encode()).hexdigest() or
            any(cuda.get('inputs', {}).get(k) != input_pins[k] for k in ('manifest', 'scales', 'window_seconds')) or
            cuda.get('pose_stats_sha256') != input_pins['pose_stats'] or
            cuda.get('initial_sha256') != cohort['initial_state_sha256']):
        raise ValueError('CUDA/NCCL check missing, failed or from different source/cohort/inputs')
    ranks = cuda.get('ranks', [])
    if len(ranks) != 4 or {r.get('rank') for r in ranks} != set(range(4)):
        raise ValueError('CUDA report missing four-rank evidence')
    for rank in ranks:
        from contrastive_numeric_profile import assert_numeric_flags
        assert_numeric_flags(rank.get('numeric_profile', {}))
        if rank.get('passed') is not True or rank.get('stop_checks') != {'coverage': True, 'numeric': True, 'passed': True}:
            raise ValueError('CUDA rank/STOP checks failed')
        if set(rank.get('arms', {})) != {'CN', 'CR'}:
            raise ValueError('CUDA report missing active arm')
        error_keys = {'total', 'pose', 'contrast_nce', 'contrast_rnc', 'contrast_global_grad_base',
                      'contrast_global_grad_nce', 'contrast_global_grad_rnc', 'gradient', 'student', 'teacher'}
        for arm in rank['arms'].values():
            if set(arm) != {'fp32', 'bf16'}:
                raise ValueError('CUDA report missing precision')
            for name, precision in arm.items():
                rows = precision.get('updates', [])
                if (precision.get('passed') is not True or [r.get('update') for r in rows] != [1, 2] or
                        any(r.get('passed') is not True or set(r.get('errors', {})) != error_keys or set(r.get('limits', {})) != error_keys or
                            any(type(value) not in (float, int) or type(r['limits'][key]) not in (float, int) or
                                not 0 <= value <= r['limits'][key] for key, value in r['errors'].items()) for r in rows)):
                    raise ValueError('CUDA numerical comparisons failed/incomplete')
                for row in rows:
                    for key in error_keys:
                        fixed = (.01 if key in ('student', 'teacher') else 1e-5) if name == 'fp32' else (.01 if key in ('total', 'pose', 'contrast_nce', 'contrast_rnc') else None)
                        if fixed is not None and row['limits'][key] != fixed:
                            raise ValueError('CUDA fixed numerical criterion changed')
    return cuda


def trajectory_drift(control, candidate):
    """Keep the original 256 scalar comparisons as descriptive evidence only."""
    if any([r.get('step') for r in rows] != list(range(1, 257)) for rows in (control, candidate)):
        raise ValueError('trajectory drift requires exactly updates 1..256')
    fields = {}
    for key in ('total', 'grad_norm'):
        rows = []
        for a, b in zip(control, candidate):
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in (a.get(key), b.get(key))):
                raise ValueError('trajectory drift requires finite loss/norm')
            rows.append({'step': a['step'], 'control': a[key], 'candidate': b[key],
                         'absolute_difference': abs(a[key]-b[key]),
                         'relative_to_control': abs(a[key]-b[key])/max(abs(a[key]), 1e-30),
                         'within_historical_tolerance': math.isclose(a[key], b[key], rel_tol=1e-4, abs_tol=1e-6)})
        fields[key] = {'updates': rows, 'outside_count': sum(not r['within_historical_tolerance'] for r in rows)}
    return {'descriptive_only': True, 'rel_tol': 1e-4, 'abs_tol': 1e-6, 'fields': fields,
            'interpretation': 'Longitudinal drift of independent trajectories, not the decisive zero-weight neutrality gate.'}


def numeric_context_across_runs(full_dirs, resumed_dirs):
    """Require one actual numerical context across seven independently launched runs.

    This includes recorded reduction and runtime flags, not only the minimum
    math/TF32 contract. Missing evidence cannot qualify any comparison. The
    subsequent per-run reader still authenticates source, config and phase.
    """
    from contrastive_numeric_profile import assert_numeric_flags, NumericProfileUnavailable
    runs = {**{arm: Path(path) for arm, path in full_dirs.items()},
            **{name: Path(path) for name, path in resumed_dirs.items()}}
    if set(runs) != {'C0', 'CN', 'CR', 'CN_A', 'CN_B', 'CR_A', 'CR_B'}:
        raise ValueError('numeric context requires all seven full/reload runs')
    contexts, evidence, missing, differences = {}, {}, [], []
    for name in sorted(runs):
        evidence[name] = {}
        for rank in range(4):
            key = name + '/rank' + str(rank)
            path = runs[name]/f'numeric_profile.rank{rank}.json'
            if not path.is_file():
                missing.append(key + ': missing profile receipt')
                continue
            row = _read(path)
            evidence[name][path.name] = sha(path)
            flags = row.get('flags')
            if not isinstance(flags, dict) or not flags or flags.get('profile') is None:
                missing.append(key + ': missing effective profile flags')
                continue
            contexts[key] = flags
            try:
                assert_numeric_flags(flags)
            except NumericProfileUnavailable as exc:
                missing.append(key + ': required profile unavailable: ' + str(exc))
    reference_key = next(iter(contexts), None)
    reference = contexts.get(reference_key)
    if reference is not None:
        for key, flags in contexts.items():
            changed = sorted(field for field in reference.keys() | flags.keys()
                             if reference.get(field) != flags.get(field))
            if changed:
                differences.append({'reference': reference_key, 'compared': key, 'fields': changed})
    status = 'FAILED' if differences else 'INCONCLUSIVE' if missing else 'PASSED'
    return {'status': status, 'required_runs': sorted(runs), 'required_rank_receipts': 28,
            'profile_receipts_sha256': evidence, 'reference_context': reference_key,
            'effective_flags': reference, 'differences': differences, 'missing': missing,
            'training_approved': False}


def read_preflight(cohort_path, full_dirs, resumed_dirs, cuda_path, *, cpu_audit_path, calibration_path, calibration_run_dir, rank_exit_path=None, cpu_audit_sacct_path=None):
    if set(full_dirs) != set(ARMS) or set(resumed_dirs) != {'CN_A', 'CN_B', 'CR_A', 'CR_B'}:
        raise ValueError('require full C0/CN/CR and two fresh reloads for each CN/CR')
    from contrastive_run import validate_cohort
    from contrastive_prepare import BASELINE, build_review_configs
    cohort, receipt = _read(cohort_path), _read(calibration_path)
    authenticate_calibration(receipt, calibration_run_dir)
    expected = calibrated_configs(receipt, calibration_run_dir, 576)
    if cohort.get('endpoint') != 576:
        raise ValueError('preflight needs the common 576-update cohort')
    for arm in ARMS:
        validate_cohort(cohort_path, calibration_path, calibration_run_dir, Path(cohort_path).parent/cohort['configs'][arm]['path'])
    input_pins = receipt['input_sha256']
    audit = _read(cpu_audit_path)
    calibration_cfg = build_review_configs(_read(BASELINE))['calibration']
    # validate_audit normally hashes physical inputs. Here its exact same gate is
    # reconstructed from the authenticated calibration input pins and config.
    from contrastive_prepare import _coverage
    panel = _coverage({'expected_datasets': audit.get('expected_datasets'), 'coverage_completed_blocks': [],
                       'coverage_partial_block': audit.get('calibration_panel'), 'config': calibration_cfg['contrastive']})
    canonical_calibration = hashlib.sha256(json.dumps(calibration_cfg, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if audit.get('source_sha256') != cohort['source_sha256']:
        from contrastive_source_compat import validate_historical_evidence
        validate_historical_evidence('audit', cpu_audit_path)
    if (audit.get('passed') is not True or not panel['passed'] or
            audit.get('inputs') != input_pins or audit.get('steps') != 1024 or audit.get('world_size') != 4 or
            audit.get('workers_per_rank') != 6 or audit.get('config_sha256') != canonical_calibration):
        raise ValueError('CPU audit missing, failed or from different inputs/config/source')
    from contrastive_release import validate_audit_accounting
    accounting = validate_audit_accounting(cpu_audit_path, cpu_audit_sacct_path, expected_source=cohort['source_sha256'])
    cuda = _cuda_evidence(cuda_path, cohort, input_pins, calibration_cfg)
    from contrastive_rank_exit import validate_receipt
    reported_rank_exit = cuda.get('rank_exit_report_path')
    if not isinstance(reported_rank_exit, str) or not reported_rank_exit:
        raise ValueError('CUDA report missing abrupt rank-exit evidence path')
    rank_exit_path = Path(rank_exit_path or reported_rank_exit)
    if rank_exit_path.resolve() != Path(reported_rank_exit).resolve() or sha(rank_exit_path) != cuda.get('rank_exit_report_sha256'):
        raise ValueError('abrupt rank-exit receipt pin/path differs')
    validate_receipt(rank_exit_path, expected_source_sha256=cohort['source_sha256'])
    if cuda['inputs'].get('calibration_checkpoint') != sha(Path(calibration_run_dir)/'checkpoint.pt'):
        raise ValueError('CUDA initialization checkpoint differs from fixed calibration')
    initial = _initial_from_calibration(calibration_run_dir, receipt)
    provenance = {'source_sha256': cohort['source_sha256'], 'calibration_sha256': cohort['calibration_sha256'],
                  'cohort_sha256': sha(cohort_path), 'input_sha256': input_pins}
    evidence_paths = {'cohort_path': str(Path(cohort_path).resolve()), 'full_dirs': {k: str(Path(v).resolve()) for k, v in full_dirs.items()},
                      'resumed_dirs': {k: str(Path(v).resolve()) for k, v in resumed_dirs.items()}, 'cuda_path': str(Path(cuda_path).resolve()),
                      'cpu_audit_path': str(Path(cpu_audit_path).resolve()), 'calibration_path': str(Path(calibration_path).resolve()),
                      'calibration_run_dir': str(Path(calibration_run_dir).resolve()), 'rank_exit_path': str(rank_exit_path.resolve()),
                      'cpu_audit_sacct_path': str(Path(cpu_audit_sacct_path).resolve())}
    result = {'schema': 1, 'status': 'PASSED_FOR_REVIEW', 'failure_reasons': [], 'training_approved': False,
              'source_sha256': cohort['source_sha256'], 'calibration_sha256': cohort['calibration_sha256'],
              'initial_state_sha256': cohort['initial_state_sha256'], 'cohort_sha256': sha(cohort_path),
              'cuda_report_sha256': sha(cuda_path), 'cpu_audit_sha256': sha(cpu_audit_path),
              'rank_exit_report_sha256': sha(rank_exit_path),
              'calibration_checkpoint_sha256': sha(Path(calibration_run_dir)/'checkpoint.pt'),
              'benchmark_summaries': {}, 'arms': {}, 'evidence_paths': evidence_paths, 'audit_accounting': accounting,
              'trajectory_drift_descriptive': {}, 'inconclusive_reasons': [],
              'initial_calibration_description': initial_calibration_description(calibration_run_dir),
              'training_allocations_gpu_hours': {}, 'training_allocations_gpu_hours_by_phase': {}, 'ratio_rule': RATIO_RULE}
    numerical = numeric_context_across_runs(full_dirs, resumed_dirs)
    result['numeric_context_across_runs'] = numerical
    if numerical['status'] != 'PASSED':
        if numerical['status'] == 'FAILED':
            result['failure_reasons'].append('effective numerical flags differ between full/reload runs or ranks')
            result['status'] = 'FAILED_REQUIRES_REVIEW'
        else:
            result['inconclusive_reasons'].append('required full/reload numerical profile is missing or unavailable')
            result['status'] = 'INCONCLUSIVE_REQUIRES_REVIEW'
        # No metric comparison or training resource projection from an
        # unqualified or mixed numerical context.
        return result
    streams = {}
    for arm, folder in full_dirs.items():
        summary, cfg, records, pins = _run_evidence(folder, arm, cohort, expected[arm], provenance)
        ratios = component_ratios(records, cfg)
        active = 'nce' if arm == 'CN' else 'rnc' if arm == 'CR' else None
        if active and not ratios[active]['within_screening_bounds']:
            result['failure_reasons'].append(arm+': measured intervention outside preregistered bounds')
        allocation = measured_allocation(summary, records)
        result['arms'][arm] = {'ratios': ratios, 'resources': allocation, 'evidence_sha256': pins,
                               'full_batch_gpu_peak_gib': summary['gpu_max_memory_gib'], 'numeric_profile': 'bf16_math_tf32_off'}
        result['training_allocations_gpu_hours'][arm] = allocation['allocated_gpu_hours']
        result['training_allocations_gpu_hours_by_phase'][arm] = dict(zip(('5000', '19618'), (4*s/3600 for s in allocation['phase_allocation_seconds'])))
        result['benchmark_summaries'][arm] = pins['summary.json']
        streams[arm] = (records, pins['origin_sessions.json'])
        if arm in ('CN', 'CR'):
            from contrastive_target_resume import compare_three_paths
            paths = {'continuous': Path(folder), 'reload_a': Path(resumed_dirs[arm+'_A']),
                     'reload_b': Path(resumed_dirs[arm+'_B'])}
            reload_pins = {}
            for phase in ('reload_a', 'reload_b'):
                _, _, reload_records, rp = _run_evidence(paths[phase], arm, cohort, expected[arm], provenance, first_update=545, expected_phase=phase.replace('reload_', 'resume_'))
                if [r['origin'] for r in reload_records] != [r['origin'] for r in records[544:]]:
                    raise ValueError('reloaded stream differs from continuous suffix')
                reload_pins[phase] = rp
            check = compare_three_paths(paths, config_from_dict(cfg), cohort['initial_state_sha256'], initial,
                                        sha(Path(folder)/'checkpoint_step0000544.pt'), ranks=4)
            check['evidence_sha256'] = reload_pins
            result['arms'][arm]['resume'] = check
            if check.get('status') == 'INCONCLUSIVE':
                result['inconclusive_reasons'].append(arm+': common-checkpoint comparison inconclusive')
            elif not check.get('passed'):
                result['failure_reasons'].append(arm+': common-checkpoint historical bounds or invariants failed')
    ratio = result['arms']['CN']['ratios']['nce']['median']/max(result['arms']['CR']['ratios']['rnc']['median'], 1e-30)
    result['ratio_CN_over_CR'] = ratio
    if not RATIO_RULE['between_min'] <= ratio <= RATIO_RULE['between_max']:
        result['failure_reasons'].append('contrastive interventions differ by more than4x')
    for arm in ('CN', 'CR'):
        if streams['C0'][1] != streams[arm][1] or [r['origin'] for r in streams['C0'][0]] != [r['origin'] for r in streams[arm][0]]:
            raise ValueError('matched-arm window stream or session mapping differs')
        result['trajectory_drift_descriptive'][arm] = trajectory_drift(streams['C0'][0][:256], streams[arm][0][:256])
    parity_path = Path(full_dirs['C0'])/'zero_weight_parity.json'
    if not parity_path.is_file():
        result['inconclusive_reasons'].append('missing required production zero-weight profile')
    else:
        from contrastive_target_parity import validate_parity_report
        parity = _read(parity_path)
        cfg = _read(Path(full_dirs['C0'])/'config.json')
        canonical_cfg = hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
        if (parity.get('source_sha256') != cohort['source_sha256'] or parity.get('config_sha256') != canonical_cfg or
                parity.get('summary_sha256') != sha(Path(full_dirs['C0'])/'summary.json') or
                parity.get('initial_state_sha256') != cohort['initial_state_sha256'] or parity.get('provenance') != provenance):
            raise ValueError('zero-weight parity receipt is not bound to the full control run')
        assessment = validate_parity_report(parity)
        result['zero_weight_parity'] = {'sha256': sha(parity_path), 'assessment': assessment, 'report': parity}
        status = assessment if isinstance(assessment, str) else assessment.get('status')
        if status == 'INCONCLUSIVE':
            result['inconclusive_reasons'].append('production zero-weight comparison lacks required profile/support')
        elif status != 'PASSED':
            result['failure_reasons'].append('production zero-weight gradients/invariants not byte-identical')
    if result['failure_reasons']:
        result['status'] = 'FAILED_REQUIRES_REVIEW'
    elif result['inconclusive_reasons']:
        result['status'] = 'INCONCLUSIVE_REQUIRES_REVIEW'
    return result


def authenticate_preflight(receipt_path):
    receipt = _read(receipt_path)
    if receipt.get('status') != 'PASSED_FOR_REVIEW' or receipt.get('failure_reasons') != [] or receipt.get('training_approved') is not False:
        raise ValueError('preflight receipt did not pass every frozen gate')
    paths = receipt.get('evidence_paths')
    required = {'cohort_path', 'full_dirs', 'resumed_dirs', 'cuda_path', 'cpu_audit_path', 'calibration_path', 'calibration_run_dir', 'rank_exit_path', 'cpu_audit_sacct_path'}
    if not isinstance(paths, dict) or set(paths) != required:
        raise ValueError('preflight receipt missing complete evidence paths')
    actual = read_preflight(**paths)
    if actual != receipt:
        raise ValueError('preflight receipt differs from authenticated recomputation')
    return actual

def _read_json(text):
    return _loads(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cohort', type=Path, required=True)
    ap.add_argument('--full', action='append', required=True, help='ARM=run_directory')
    ap.add_argument('--resumed', action='append', required=True, help='ARM=run_directory')
    ap.add_argument('--cuda-report', type=Path, required=True)
    ap.add_argument('--cpu-audit', type=Path, required=True)
    ap.add_argument('--cpu-audit-sacct', type=Path, required=True, help='authenticated audit accounting JSON with pinned sacct TSV')
    ap.add_argument('--calibration-report', type=Path, required=True)
    ap.add_argument('--calibration-run-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    def mapping(values):
        rows = [v.split('=', 1) for v in values]
        if any(len(row) != 2 for row in rows) or len({r[0] for r in rows}) != len(rows):
            raise ValueError('unique ARM=path entries required')
        return dict(rows)
    report = read_preflight(a.cohort, mapping(a.full), mapping(a.resumed), a.cuda_report,
                            cpu_audit_path=a.cpu_audit, calibration_path=a.calibration_report, calibration_run_dir=a.calibration_run_dir, cpu_audit_sacct_path=a.cpu_audit_sacct)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f: json.dump(report, f, indent=2, allow_nan=False)
    if report['status'] != 'PASSED_FOR_REVIEW': raise SystemExit(2)


if __name__ == '__main__': main()
