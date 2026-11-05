"""Read-only model-construction diagnostic; never invokes training or loads weights.

The parent deliberately imports ONLY the standard library. Each import context
runs in a fresh subprocess. Historical-hash mismatches are observations, not
permission to replace the historical gate or release another training run.

Completed on 05/11/2026 with the dry run of the proposed correction
(scripts/logsquare_initial_state.py, scripts/logsquare_train_rank.py):
every T100-LQ model built here also receives the verdict of the reviewed guard,
and one more context starts the production fresh-phase command under torchrun
with --verify-only: four real ranks build and verify their student and stop
inside the constructor, so no model reaches the trainer. The reading of the
report is fixed in the plan before the run; it never authorizes training.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import traceback

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = 'logsquare-initialization-diagnostic-20261105-v1'
RANK_VARIANT = 'rank_entry_torchrun'
CONSTRUCTION_VARIANTS = ('controller', 'preflight', 'trainer', 'trainer_rank0', 'controller_threads1')
VARIANTS = CONSTRUCTION_VARIANTS + (RANK_VARIANT,)
MODULES = {'controller': 'scripts/logsquare_train.py',
           'controller_threads1': 'scripts/logsquare_train.py',
           'preflight': 'scripts/logsquare_preflight.py', 'trainer': 'scripts/train_jepa.py',
           'trainer_rank0': 'scripts/train_jepa.py'}
CONSTRUCTION_ORDER = {variant: ('T100-L', 'T100-LQ') if variant == 'preflight' else ('T100-LQ', 'T100-L')
                      for variant in CONSTRUCTION_VARIANTS}
NEW_WEIGHT = 'tokenizer.logsquare_weight'
GAIN = 'tokenizer.fe_gain'


def require(value, message):
    if not value:
        raise ValueError(message)


def read_pin(pin, root=REPO):
    path = Path(pin['path'])
    path = path if path.is_absolute() else root / path
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == pin['sha256'], 'SHA mismatch: ' + str(path))
    require('bytes' not in pin or len(data) == pin['bytes'], 'size mismatch: ' + str(path))
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def same_pin(first, second):
    return first['path'] == second['path'] and first['sha256'] == second['sha256']


def import_file(relative, name):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Standard library only at import time, like this parent: the guard under dry run.
G = import_file('scripts/logsquare_initial_state.py', 'logsquare_diagnostic_initial_state')


def validate_plan(path, root=REPO):
    """Validate local pinned provenance; do not open datasets or checkpoints."""
    plan = json.loads(Path(path).read_text())
    require(plan.get('protocol') == PROTOCOL and plan.get('status') == 'APPROVED_FOR_DIAGNOSTIC',
            'diagnostic plan not approved')
    require(plan.get('variants') == list(VARIANTS), 'wrong import contexts')
    require(type(plan.get('cuda_required')) is bool, 'explicit CUDA policy required')
    require(type(plan.get('child_timeout_seconds')) is int and 1 <= plan['child_timeout_seconds'] <= 90,
            'unbounded child timeout')
    require(type(plan.get('rank_entry_timeout_seconds')) is int and 60 <= plan['rank_entry_timeout_seconds'] <= 300,
            'unbounded rank-entry timeout')
    require(plan.get('expected_scientific_sources') == 102, 'wrong source inventory size')
    training = read_pin(plan['training_plan'], root)
    require(training.get('protocol') == 'tokenizer-logsquare-training-20261104-v1' and
            training.get('arm') == 'T100-LQ', 'wrong historical training plan')
    sources = training['scientific_source_sha256']
    require(len(sources) == 102 and set(sources) == set(training['required_scientific_sources']),
            'incomplete scientific inventory')
    for relative, digest in sources.items():
        source = (root / relative).resolve()
        require(source.is_relative_to(root.resolve()) and hashlib.sha256(source.read_bytes()).hexdigest() == digest,
                'scientific source changed: ' + relative)
    require(same_pin(plan['preflight_report'], training['preflight_report']), 'report provenance changed')
    preflight = read_pin(training['preflight_plan'], root)
    require(same_pin(plan['candidate_config'], preflight['candidate_config']), 'candidate provenance changed')
    reference = read_pin(preflight['reference_plan'], root)
    require(same_pin(plan['comparison'], reference['comparison']), 'comparison provenance changed')
    candidate = read_pin(plan['candidate_config'], root)
    require(candidate['seed'] == 0 and candidate['model']['tokenizer_logsquare'] is True and
            candidate['model']['tokenizer_logsquare_eps'] == .001, 'candidate recipe changed')
    comparison = read_pin(plan['comparison'], root)
    report = read_pin(plan['preflight_report'], root)
    require(report.get('complete') is True and report.get('protocol') == 'tokenizer-logsquare-preflight-20261104-v1' and
            report.get('code_sha') == training['preflight_code_sha'] and
            report.get('initial_shared_weights_rng_and_function_exact') is True,
            'invalid original preflight report')
    expected = training['shared_initial_sha256']
    initials = [report['arms'][arm]['initial'] for arm in ('T100-L', 'T100-LQ')]
    gain = initials[0]['fe_gain']
    require(type(gain) in (float, int) and math.isfinite(gain) and gain > 0 and
            all(row['shared_sha256'] == expected and row['fe_gain'] == gain and row['new_weight_zero'] is True
                for row in initials), 'historical initialization evidence inconsistent')
    # The guard under dry run: pinned, bound to the same reference, re-read from its historical records.
    require(isinstance(plan.get('initial_state_guard'), dict), 'diagnostic plan without the guard under dry run')
    pin = plan['initial_state_guard']
    guard_path = (root / pin['path']).resolve()
    require(guard_path.is_relative_to(root.resolve()) and
            ('bytes' not in pin or guard_path.stat().st_size == pin['bytes']), 'guard outside the code or of another size')
    guard = G.load_guard(guard_path, pin['sha256'])
    require(guard['reference']['shared_sha256'] == expected and G.same_pin(guard['reference']['source'], plan['preflight_report']) and
            G.bits(float(gain)) == G.bits(guard['reference']['fe_gain']), 'guard reference is not the diagnostic reference')
    G.validate_provenance(guard, root)
    return plan, candidate, comparison, {'shared_sha256': expected, 'fe_gain': gain}, guard


def code_identity(expected, root=REPO):
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    require(len(expected) == 40 and actual == expected, 'wrong frozen code SHA')
    require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=root, text=True).strip(),
            'dirty diagnostic checkout')
    return actual


def classify_match(original_hash, substituted_hash, expected_hash):
    if original_hash == expected_hash:
        return 'exact_original_match'
    if substituted_hash == expected_hash:
        return 'only_gain_substitution_recovers_exact_hash'
    return 'difference_not_explained_by_gain_alone'


def inspect_state(model, expected):
    """Hash originals and one hypothetical copy; never mutate the model."""
    import torch
    from wearusfm.training.pairing import state_sha256

    state = model.state_dict()
    original = state_sha256(state)
    shared = {key: value for key, value in state.items() if key != NEW_WEIGHT}
    gain = shared[GAIN]
    require(gain.shape == torch.Size([]) and gain.dtype == torch.float64 and
            bool(torch.isfinite(gain)) and gain.item() > 0, 'invalid calibrated gain')
    substituted = dict(shared)
    substituted[GAIN] = gain.detach().clone().fill_(expected['fe_gain'])
    shared_hash = state_sha256(shared)
    substituted_hash = state_sha256(substituted)
    tensor_rows = {key: {'sha256': state_sha256(value), 'shape': list(value.shape), 'dtype': str(value.dtype)}
                   for key, value in shared.items()}
    branch = state.get(NEW_WEIGHT)
    after = state_sha256(model.state_dict())
    require(after == original, 'diagnostic modified original model')
    return {'full_sha256_before': original, 'full_sha256_after': after, 'shared_sha256': shared_hash,
            'fe_gain': gain.item(), 'per_tensor': tensor_rows,
            'original_matches_historical': shared_hash == expected['shared_sha256'],
            'hypothetical_pinned_gain_shared_sha256': substituted_hash,
            'hypothetical_pinned_gain_matches_historical': substituted_hash == expected['shared_sha256'],
            'diagnostic_classification': classify_match(shared_hash, substituted_hash, expected['shared_sha256']),
            'logsquare_weight_present': branch is not None,
            'logsquare_weight_zero': branch is None or not bool(branch.count_nonzero()),
            'logsquare_shape': list(branch.shape) if branch is not None else None}


def cheap_environment():
    """No numeric-runtime calls before construction: even a query can warm it."""
    return {'host': socket.gethostname(), 'python': sys.version,
            'loaded_versions': {name: getattr(sys.modules.get(name), '__version__', None)
                                for name in ('torch', 'numpy', 'scipy')},
            'environment': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                            'OPENBLAS_NUM_THREADS', 'MKL_THREADING_LAYER')}}


def environment(torch):
    cpu = None
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        cpu = next((line.partition(':')[2].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith('model name')), None)
    libraries = []
    maps = Path('/proc/self/maps')
    if maps.exists():
        libraries = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                            if any(word in line for word in ('libgomp', 'libiomp', 'libmkl', 'libopenblas'))})
    return {'host': socket.gethostname(), 'cpu': cpu, 'python': sys.version, 'torch': torch.__version__,
            'numpy': getattr(sys.modules.get('numpy'), '__version__', None),
            'scipy': getattr(sys.modules.get('scipy'), '__version__', None),
            'cuda': torch.version.cuda, 'torch_threads': torch.get_num_threads(),
            'torch_interop_threads': torch.get_num_interop_threads(), 'parallel_info': torch.__config__.parallel_info(),
            'environment': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                            'OPENBLAS_NUM_THREADS', 'MKL_THREADING_LAYER')}, 'loaded_numeric_libraries': libraries}


COHERENT = {'exact_original_match': (G.PREFLIGHT_STATE,),
            'only_gain_substitution_recovers_exact_hash': (G.PASSED, G.UNREGISTERED_GAIN),
            'difference_not_explained_by_gain_alone': (G.OTHER_TENSORS,)}


def guard_observations(model, row, arm, guard):
    """Exact gain distances for both arms; for T100-LQ the verdict the corrected launch would give this model."""
    row['fe_gain_hex'] = row['fe_gain'].hex()
    row['gain_ulps'] = {'from_preflight_reference': G.bits(row['fe_gain']) - G.bits(guard['reference']['fe_gain']),
                        'from_historical_training': G.bits(row['fe_gain']) - G.bits(guard['training_gain']['fe_gain'])}
    if arm != 'T100-LQ':
        return
    verdict = G.inspect(model.state_dict(), guard)
    # Two independent tools read the same state: they must tell the same story.
    require(verdict['classification'] in COHERENT[row['diagnostic_classification']] and
            verdict['shared_sha256'] == row['shared_sha256'], 'diagnostic and guard disagree on one state')
    row['guard'] = verdict


def child(plan_path, variant, expected_code):
    code_identity(expected_code)
    plan, candidate, comparison, expected, guard = validate_plan(plan_path)
    # Do not move numpy/torch imports above this point: import order is the experiment.
    actual = import_file(MODULES[variant], 'logsquare_diagnostic_actual_' + variant)
    import torch
    from wearusfm.model.fm import WearUsFM
    from wearusfm.training.pairing import state_sha256

    cli = actual if variant in ('trainer', 'trainer_rank0') else getattr(actual, 'CLI', None)
    if cli is None:
        cli = import_file('scripts/train_jepa.py', 'logsquare_diagnostic_cli')
    argv = list(comparison['arm_configurations']['T100-L']['validated_trainer_argv'])
    cfg = cli.build_config(cli.build_parser().parse_args(argv[1:] +
                           ['--tokenizer-logsquare', '--tokenizer-logsquare-eps', '.001']))
    require(cli.R.config_to_dict(cfg) == candidate, 'CLI candidate differs from pinned recipe')
    require(not plan['cuda_required'] or torch.cuda.is_available(), 'required CUDA unavailable')
    if variant == 'preflight' and plan['cuda_required']:
        torch.cuda.set_device(0)
    if variant in ('preflight', 'controller_threads1'):
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    result = {'variant': variant, 'imported_entrypoint': MODULES[variant], 'environment_before': cheap_environment(),
              'construction_scope': 'seeded CPU construction and optional CUDA transfer; no DDP, dataset, update or trainer main',
              'trainer_scope': ('rank-zero parameter-count warmup replicated; DDP startup omitted' if variant == 'trainer_rank0'
                                else 'non-main-rank construction approximation; DDP startup omitted' if variant == 'trainer'
                                else None),
              'expected': expected, 'arms': {}, 'construction_order': list(CONSTRUCTION_ORDER[variant]),
              'unseeded_parameter_count_warmups': 0}
    if variant == 'trainer_rank0':
        # train_jepa.main prints R.count_parameters(cfg.model) on rank zero
        # before R.train seeds/builds the actual student. Reproduce that CPU
        # construction, without reading training data or entering the trainer.
        result['parameter_count_warmup_result'] = cli.R.count_parameters(cfg.model)
        result['unseeded_parameter_count_warmups'] = 1
    for arm in CONSTRUCTION_ORDER[variant]:
        model_cfg = replace(cfg.model, tokenizer_logsquare=arm == 'T100-LQ')
        # Match the controller's fork_rng context too; preflight/trainer seed
        # directly. No construction context is silently normalized to another.
        context = torch.random.fork_rng(devices=[]) if variant.startswith('controller') else nullcontext()
        with context:
            torch.manual_seed(cfg.seed)
            model = WearUsFM(model_cfg)
            row = inspect_state(model, expected)
            row['cpu_rng_after_construction_sha256'] = state_sha256(torch.get_rng_state())
            guard_observations(model, row, arm, guard)
        require(row['logsquare_weight_zero'] and (arm == 'T100-L' or row['logsquare_shape'] == [cfg.model.dim, 64]),
                'invalid fresh log-square branch')
        if plan['cuda_required']:
            cpu_hash = row['full_sha256_after']
            model.to('cuda:0')
            row['cuda_state'] = inspect_state(model, expected)
            row['cuda_transfer_preserves_exact_state'] = row['cuda_state']['full_sha256_before'] == cpu_hash
            row['cuda_device'] = torch.cuda.get_device_name(0)
            row['cuda_rng_after_transfer_sha256'] = state_sha256(torch.cuda.get_rng_state(0))
        result['arms'][arm] = row
        del model
    linear, nonlinear = (result['arms'][arm] for arm in ('T100-L', 'T100-LQ'))
    result['paired_shared_state_exact'] = linear['shared_sha256'] == nonlinear['shared_sha256']
    result['paired_cpu_rng_exact'] = linear['cpu_rng_after_construction_sha256'] == nonlinear['cpu_rng_after_construction_sha256']
    result['paired_cuda_rng_exact'] = (linear['cuda_rng_after_transfer_sha256'] == nonlinear['cuda_rng_after_transfer_sha256']
                                       if plan['cuda_required'] else None)
    result['environment_after'] = environment(torch)
    validate_plan(plan_path)
    code_identity(expected_code)
    return result


def run_bounded(command, seconds, stdout_path, stderr_path):
    """Run a process tree under a hard bound: a timeout kills the whole session, not only its leader."""
    with open(stdout_path, 'wb') as stdout, open(stderr_path, 'wb') as stderr:
        proc = subprocess.Popen(command, cwd=REPO, stdout=stdout, stderr=stderr, start_new_session=True,
                                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        try:
            return proc.wait(timeout=seconds), False
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            return proc.wait(), True


def rank_entry_child(plan_path, expected_code, workspace):
    """The production fresh-phase command with --verify-only: four real ranks, nothing trained.

    The command is the controller's own (command_for), with the production
    allocation, guard and trainer arguments; only the directories are the
    diagnostic's and --verify-only is added. Every rank imports the production
    entry, joins the process group, reads the same inputs, builds its student
    through the trainer's own code path and verifies it; the constructor then
    raises, so the trainer never receives a model. The controller's own reader
    judges the four records.
    """
    code_identity(expected_code)
    plan, candidate, comparison, expected, guard = validate_plan(plan_path)
    training = read_pin(plan['training_plan'])
    controller = import_file('scripts/logsquare_train.py', 'logsquare_diagnostic_rank_controller')
    cli = controller.CLI
    argv = list(comparison['arm_configurations']['T100-L']['validated_trainer_argv'])
    cfg = cli.build_config(cli.build_parser().parse_args(argv[1:] +
                           ['--tokenizer-logsquare', '--tokenizer-logsquare-eps', '.001']))
    require(cli.R.config_to_dict(cfg) == candidate, 'CLI candidate differs from pinned recipe')
    run_dir, records = workspace / (RANK_VARIANT + '_run'), workspace / (RANK_VARIANT + '_initial_state')
    require(workspace.is_dir() and not run_dir.exists() and not records.exists(), 'new rank-entry workspace required')
    records.mkdir()
    pin = {'path': str((REPO / plan['initial_state_guard']['path']).resolve()), 'sha256': plan['initial_state_guard']['sha256']}
    command = controller.command_for(comparison, cfg, run_dir, 5000, training['allocations']['5000']['walltime_seconds'],
                                     pin, records)
    position = command.index(controller.RANK_ENTRY) + 1
    command = command[:position] + ['--verify-only'] + command[position:]
    outputs = [workspace / (RANK_VARIANT + '.torchrun.' + suffix) for suffix in ('stdout', 'stderr')]
    returncode, timed_out = run_bounded(command, plan['rank_entry_timeout_seconds'] - 30, *outputs)
    result = {'variant': RANK_VARIANT, 'imported_entrypoint': controller.RANK_ENTRY,
              'scope': ('production fresh-phase command with --verify-only under torchrun: four ranks, process group, trainer '
                        'inputs and model construction; the constructor raises after verification, so no model reaches the '
                        'trainer: no teacher, optimizer, DDP wrapper, data forward, update or checkpoint'),
              'expected': expected, 'command': command, 'torchrun_returncode': returncode, 'torchrun_timed_out': timed_out,
              'torchrun_stderr_tail': outputs[1].read_text(errors='replace')[-4000:], 'ranks': {}}
    for rank in range(4):
        path = records / f'rank{rank}.json'
        result['ranks'][str(rank)] = json.loads(path.read_text()) if path.is_file() else None
    try:
        result['attestation'] = {'passed': True, **controller.rank_attestation(records, pin['sha256'], verify_only=True)}
    except Exception as error:  # a rejected student is an observation, like a hash mismatch in the other contexts
        result['attestation'] = {'passed': False, 'reason': str(error)}
    result['run_dir_entries'] = sorted(path.name for path in run_dir.iterdir()) if run_dir.is_dir() else None
    judged = all(row is not None and [c.get('role') for c in row.get('constructions', ())][-1:] == ['student']
                 for row in result['ranks'].values())
    if timed_out:
        result['error'] = 'torchrun timed out'
    elif not judged:
        result['error'] = 'not every rank judged a student; see the torchrun output'
    validate_plan(plan_path)
    code_identity(expected_code)
    return result


def reading(variants):
    """The reading fixed in the plan before the run, computed on the report. Descriptive: it authorizes nothing."""
    def rows(variant):
        return [variants[variant]['arms'][arm] for arm in ('T100-L', 'T100-LQ')]

    def training_state(variant):  # reference tensors, and the gain the historical T100-L training saved
        return all(row['hypothetical_pinned_gain_matches_historical'] and not row['original_matches_historical'] and
                   row['gain_ulps']['from_historical_training'] == 0 for row in rows(variant))

    try:
        entry = variants[RANK_VARIANT]
        facts = {
            'preflight_context_reproduces_the_reference': all(row['original_matches_historical'] and
                                                              row['gain_ulps']['from_preflight_reference'] == 0
                                                              for row in rows('preflight')),
            'controller_context_builds_reference_tensors_with_training_gain': training_state('controller'),
            'trainer_contexts_build_reference_tensors_with_training_gain': training_state('trainer') and training_state('trainer_rank0'),
            'guard_accepts_controller_and_trainer_models': all(variants[variant]['arms']['T100-LQ']['guard']['passed'] is True
                                                               for variant in ('controller', 'trainer', 'trainer_rank0')),
            'guard_refuses_the_preflight_context': variants['preflight']['arms']['T100-LQ']['guard']['classification'] ==
                                                   G.PREFLIGHT_STATE,
            'four_real_ranks_verified_one_state': entry['attestation']['passed'] is True and entry['torchrun_returncode'] == 0 and
                                                  entry['run_dir_entries'] == [],
        }
        facts['real_ranks_state_is_the_trainer_context_state'] = (
            facts['four_real_ranks_verified_one_state'] and
            entry['attestation']['shared_sha256'] == variants['trainer_rank0']['arms']['T100-LQ']['shared_sha256'])
        threads = variants['controller_threads1']['arms']['T100-LQ']
        return {'computable': True, **facts,
                # Why job 63532335 stopped: a controller-context model against a preflight-context hash.
                'cause_demonstrated': facts['preflight_context_reproduces_the_reference'] and
                                      facts['controller_context_builds_reference_tensors_with_training_gain'],
                # What the unchanged trainer builds, for T100-L as for T100-LQ: the state the correction requires.
                'trainer_builds_the_registered_training_state': facts['trainer_contexts_build_reference_tensors_with_training_gain'],
                'fix_dry_run_passed': facts['guard_accepts_controller_and_trainer_models'] and
                                      facts['real_ranks_state_is_the_trainer_context_state'],
                'informative_controller_one_thread': {'diagnostic_classification': threads['diagnostic_classification'],
                                                      'gain_ulps': threads['gain_ulps']},
                'training_authorized_by_reading': False}
    except Exception as error:  # a missing or failed context: no reading
        return {'computable': False, 'reason': f'{type(error).__name__}: {error}', 'cause_demonstrated': False,
                'trainer_builds_the_registered_training_state': False, 'fix_dry_run_passed': False,
                'training_authorized_by_reading': False}


def execute(plan_path, expected_code, out):
    code = code_identity(expected_code)
    plan, _, _, expected, _ = validate_plan(plan_path)
    require(out.is_absolute() and not out.exists() and not out.is_symlink(), 'new absolute output required')
    forbidden = (REPO, REPO.parent, Path('/leonardo_work/IscrB_WearUsFM/wearusfm'),
                 Path('/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns'))
    require(not any(out.resolve().is_relative_to(root.resolve()) for root in forbidden), 'output inside code workspace')
    out.mkdir(parents=True, exist_ok=False)
    report = {'protocol': PROTOCOL, 'code_sha': code, 'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              'expected': expected, 'variants': {}, 'diagnostic_only': True,
              'training_authorized_by_report': False, 'complete': False}
    for variant in VARIANTS:
        destination = out / (variant + '.json')
        command = [sys.executable, str(Path(__file__).resolve()), '--plan', str(plan_path),
                   '--expected-code-sha', expected_code, '--child', variant, '--child-output', str(destination)]
        seconds = plan['rank_entry_timeout_seconds' if variant == RANK_VARIANT else 'child_timeout_seconds']
        try:
            proc = subprocess.run(command, cwd=REPO, capture_output=True, text=True,
                                  timeout=seconds, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            (out / (variant + '.stdout')).write_text(proc.stdout)
            (out / (variant + '.stderr')).write_text(proc.stderr)
            row = json.loads(destination.read_text()) if destination.exists() else {'error': 'child produced no report'}
            row['process_returncode'] = proc.returncode
        except subprocess.TimeoutExpired as error:
            row = {'error': 'child timed out', 'timeout_seconds': seconds}
            for suffix, captured in (('stdout', error.stdout), ('stderr', error.stderr)):
                (out / (variant + '.' + suffix)).write_bytes(captured.encode() if isinstance(captured, str) else captured or b'')
        except Exception as error:
            row = {'error': str(error), 'traceback': traceback.format_exc()}
        report['variants'][variant] = row
    try:
        validate_plan(plan_path)
        code_identity(expected_code)
        require(hashlib.sha256(plan_path.read_bytes()).hexdigest() == report['plan_sha256'], 'diagnostic plan changed')
    except Exception as error:
        report['final_provenance_error'] = str(error)
    report['complete'] = all(row.get('process_returncode') == 0 and 'error' not in row
                             for row in report['variants'].values()) and 'final_provenance_error' not in report
    report['reading'] = reading(report['variants'])
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--expected-code-sha', required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--child', choices=VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument('--child-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        if args.child_output is None or args.child_output.exists():
            parser.error('new child output required')
        try:
            report = (rank_entry_child(args.plan, args.expected_code_sha, args.child_output.parent)
                      if args.child == RANK_VARIANT else child(args.plan, args.child, args.expected_code_sha))
        except Exception as error:
            report = {'variant': args.child, 'error': str(error), 'traceback': traceback.format_exc()}
        args.child_output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        return int('error' in report)
    if args.out is None:
        parser.error('--out required')
    return int(not execute(args.plan, args.expected_code_sha, args.out)['complete'])


if __name__ == '__main__':
    raise SystemExit(main())
