"""Read-only model-construction diagnostic; never invokes training or loads weights.

The parent deliberately imports ONLY the standard library. Each import context
runs in a fresh subprocess. Historical-hash mismatches are observations, not
permission to replace the historical gate or release another training run.

Completed on 05/11/2026 with the dry run of the proposed correction
(scripts/logsquare_initial_state.py, scripts/logsquare_train_rank.py):
every T100-LQ model built here also receives the verdict of the reviewed guard,
and one more context starts the production fresh-phase command under torchrun
with --verify-only: four real ranks build and verify their student and stop
inside the constructor, so no model reaches the trainer. The ranks read the
manifest, the index and the scales as the trainer does; no training batch is
drawn and no forward runs on the signals. The reading of the report is fixed
in the plan before the run; it never authorizes training.

Every context is one bounded command (run_bounded): when its bound expires,
or when it returns, none of its processes is left running, including the
workers torchrun starts in sessions of their own. The cleanup has one
absolute deadline, declared in the plan, and names only processes of that
invocation.
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
import time
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
    # Each timeout is when waiting stops; the cleanup bound is added to it. Together they stay inside the declared
    # limits of a context: 90 s for a construction child, 300 s for the torchrun dry run.
    require(type(plan.get('cleanup_bound_seconds')) is int and plan['cleanup_bound_seconds'] == CLEANUP_BOUND_SECONDS,
            'cleanup bound differs from the supervisor')
    require(type(plan.get('child_timeout_seconds')) is int and
            1 <= plan['child_timeout_seconds'] <= 90 - CLEANUP_BOUND_SECONDS, 'unbounded child timeout')
    require(type(plan.get('rank_entry_timeout_seconds')) is int and
            60 <= plan['rank_entry_timeout_seconds'] <= 300 - CLEANUP_BOUND_SECONDS, 'unbounded rank-entry timeout')
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


# One bounded command and all of its processes ------------------------------------------------------------------------
# The command starts in a session of its own. Its processes are the members of the process groups this invocation
# created: the root's group, and the group of every child of a member. Group membership survives the death of a
# parent, so a worker that its launcher started in another session (as torchrun does) stays tracked when the launcher
# is gone. Signals go only to processes that the current process table lists as members: never to a group, a stored
# identifier or anything outside the invocation.
# The cleanup asks before it kills, and asks launchers first: SIGTERM goes to the topmost members only (those whose
# parent is not a running member), so that each launcher closes its own processes in order, as torchrun does with its
# workers; a member left without its parent becomes topmost and is asked in its turn. After orderly_seconds every
# member still running is asked; after term_grace_seconds more, whoever is still running gets SIGKILL.
SUPERVISION = {'poll_seconds': .5, 'cleanup_poll_seconds': .1, 'settle_seconds': 2., 'orderly_seconds': 5.,
               'term_grace_seconds': 3., 'kill_wait_seconds': 5., 'process_table_seconds': 2.}
CLEANUP_BOUND_SECONDS = 15  # one absolute deadline for the cleanup of a command; the plan declares it
REPORT_SECONDS = 20         # kept by the rank-entry child for its own start, the records and the report after torchrun
MIN_LAUNCH_SECONDS = 30     # torchrun is not started with less time than this
PS = ('/bin/ps', '-A', '-o', 'pid=', '-o', 'ppid=', '-o', 'pgid=', '-o', 'stat=', '-o', 'comm=')


def process_table(seconds=SUPERVISION['process_table_seconds']):
    """pid -> (ppid, pgid, state, name) of every visible process; None when the table cannot be read."""
    try:
        listing = subprocess.run(PS, capture_output=True, text=True, errors='replace', timeout=max(seconds, .1),
                                 stdin=subprocess.DEVNULL, env={'LC_ALL': 'C'})
    except (OSError, subprocess.SubprocessError):
        return None
    table = {}
    for line in listing.stdout.splitlines():
        fields = line.split(None, 4)
        if len(fields) >= 4 and all(field.isdigit() for field in fields[:3]):
            table[int(fields[0])] = (int(fields[1]), int(fields[2]), fields[3], fields[4] if len(fields) == 5 else '')
    return table if listing.returncode == 0 and os.getpid() in table else None


class Invocation:
    """The processes of one bounded command: the members of the process groups it created."""

    def __init__(self, root):
        try:
            own_session = os.getpgid(root) == root
        except ProcessLookupError:  # it has already exited: only its unreaped entry is left
            own_session = True
        self.never = {0, 1, os.getpgid(0)}  # no group of the kernel, of init or of the supervisor is ever adopted
        require(root > 1 and own_session and root not in self.never, 'bounded command without a session of its own')
        self.root, self.groups, self.names = root, {root}, {}

    def alive(self, table, root_reaped):
        """Take one process table: adopt the new groups, forget the empty ones, return the members still running."""
        while True:
            members = {pid for pid, row in table.items() if row[1] in self.groups}
            created = {row[1] for row in table.values() if row[0] in members} - self.groups - self.never
            if not created:
                break
            self.groups |= created
        # An empty group is forgotten at once, so its number can never name a later, unrelated group. The root's own
        # number cannot be reused while the root is not reaped.
        self.groups = {table[pid][1] for pid in members} | (set() if root_reaped else {self.root})
        for pid in members:  # the name it had while it ran, not the placeholder of an entry that is being removed
            if pid not in self.names or not (table[pid][2].startswith('Z') or table[pid][3].startswith(('<', '('))):
                self.names[pid] = table[pid][3]
        return sorted(pid for pid in members if not table[pid][2].startswith('Z'))


def clean_up(proc, invocation, forced, timing):
    """End what the invocation left running, inside one absolute deadline, and say what had to be done.

    forced: the bound expired, or the supervisor was interrupted. SIGTERM goes at once to the topmost members, so
    that a launcher with a handler closes its own workers in order; then to every member still running; whoever is
    still running after that gets SIGKILL. Not forced: the command returned by itself. Members still running get a
    short time to end by themselves, then the same steps. Nothing waits past the deadline: a survivor is reported,
    never awaited.
    """
    begun = time.monotonic()
    term_at = begun + (0 if forced else timing['settle_seconds'])
    all_at = term_at + timing['orderly_seconds']
    kill_at = all_at + timing['term_grace_seconds']
    end_at = min(kill_at + timing['kill_wait_seconds'], begun + CLEANUP_BOUND_SECONDS)
    done = {'forced': forced, 'terminated': [], 'killed': [], 'survivors': []}
    while True:
        table = process_table(min(timing['process_table_seconds'], max(end_at - time.monotonic(), .1)))
        if table is None:
            # Without a table no descendant can be named safely. What remains is the root's own group, and only
            # while the root is not reaped, because until then its number cannot have been reused. The table is
            # read again until the deadline: one failed read neither ends the cleanup nor marks it unverified.
            proc.poll()
            alive = [] if proc.returncode is not None else [proc.pid]
        else:
            alive = invocation.alive(table, proc.returncode is not None)
            proc.poll()
        now = time.monotonic()
        if now >= end_at or (not alive and table is not None):
            done['survivors'] = alive
            break
        signum = signal.SIGKILL if now >= kill_at else signal.SIGTERM if now >= term_at else None
        if signum is not None:
            sent = done['killed' if signum == signal.SIGKILL else 'terminated']
            # Until all_at only the topmost members are asked: the others still have a parent that may close them.
            running = set(alive) if table is not None and now < all_at else ()
            for pid in alive:
                if running and table[pid][0] in running:
                    continue
                if signum == signal.SIGKILL or pid not in sent:
                    try:
                        (os.killpg if table is None else os.kill)(pid, signum)
                    except OSError:  # already gone, or not ours to signal: the next table says which
                        pass
                    if pid not in sent:
                        sent.append(pid)
        time.sleep(max(min(timing['cleanup_poll_seconds'], end_at - time.monotonic()), 0))
    # The last table read is the one the cleanup ended on: without it, the end of the descendants was not verified.
    done['process_table_unavailable'] = table is None
    proc.poll()
    if proc.returncode is None and proc.pid not in done['survivors']:
        done['survivors'].append(proc.pid)
    named = sorted({*done['terminated'], *done['killed'], *done['survivors']})
    done.update(observed_processes=len(invocation.names), cleanup_seconds=round(time.monotonic() - begun, 3),
                names={str(pid): invocation.names.get(pid, '') for pid in named})
    return done


def supervision_problem(record):
    """What the supervision could not guarantee, or had to end after a command that had returned by itself."""
    def named(pids):
        return ', '.join(f"{pid} ({record.get('names', {}).get(str(pid), '')})" for pid in pids)
    if record.get('survivors'):
        return 'processes survived the cleanup: ' + named(record['survivors'])
    if record.get('process_table_unavailable'):
        return 'process table unavailable during the cleanup: descendants were not verified'
    if record.get('terminated') and not record.get('forced'):
        return 'processes outlived the command and were ended: ' + named(record['terminated'])
    return None


def output_tails(out, variant, characters=4000):
    """The end of every output file of a failed context, kept inside the report next to its error."""
    names = [variant + suffix for suffix in ('.stdout', '.stderr', '.torchrun.stdout', '.torchrun.stderr')]
    return {name: (out / name).read_text(errors='replace')[-characters:] for name in names if (out / name).is_file()}


def run_bounded(command, seconds, stdout_path, stderr_path, record=None, **timing):
    """Run one command under a bound and leave none of its processes running: (status, timed_out).

    The bound is when waiting stops. Ending the processes then takes at most CLEANUP_BOUND_SECONDS more, whether the
    command timed out, returned or failed. `record` receives what the supervision saw and did. The status is None
    only if the command itself survived SIGKILL. Output goes to the two files, which outlive every process.
    """
    deadline = time.monotonic() + seconds
    timing = {**SUPERVISION, **timing}
    record = {} if record is None else record
    require(process_table(timing['process_table_seconds']) is not None,
            'process table unavailable: no command is started unsupervised')
    with open(stdout_path, 'wb') as stdout, open(stderr_path, 'wb') as stderr:
        proc = subprocess.Popen(command, cwd=REPO, stdout=stdout, stderr=stderr, start_new_session=True,
                                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        try:
            invocation = Invocation(proc.pid)
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        timed_out, interrupted = False, True
        try:
            while proc.returncode is None and not timed_out:
                try:
                    proc.wait(timeout=max(min(timing['poll_seconds'], deadline - time.monotonic()), 0))
                except subprocess.TimeoutExpired:
                    timed_out = time.monotonic() >= deadline
                    # While the command runs, each table adopts the groups its descendants create. At the bound the
                    # cleanup takes its own table before any signal. Reading a table never outlasts the bound.
                    table = None if timed_out else process_table(
                        min(timing['process_table_seconds'], max(deadline - time.monotonic(), .1)))
                    if table is not None:
                        invocation.alive(table, False)
            interrupted = False
        finally:
            record.update(clean_up(proc, invocation, timed_out or interrupted, timing), timed_out=timed_out,
                          bound_seconds=seconds)
    return proc.returncode, timed_out


def rank_entry_child(plan_path, expected_code, workspace, started=None):
    """The corrected launch up to the student's construction: nothing is trained.

    As the controller does: its pre-check on a temporary CPU model, then its own
    fresh-phase command (command_for) with the production allocation, guard and
    trainer arguments; only the directories are the diagnostic's and
    --verify-only is added. Every rank imports the production entry, joins the
    process group, reads the same inputs as the trainer (manifest, index and
    scales: no training batch is drawn, no forward runs on the signals), builds
    its student through the trainer's own code path and verifies it; the
    constructor then raises, so the trainer never receives a model. The
    controller's own reader judges the four records. Unlike the controller, a
    refused pre-check does not stop the ranks: both observations are recorded.

    `started` is when this child began. torchrun gets what is left of the
    context's timeout after the preparation, less the cleanup bound and the
    time kept for the report: its own bound therefore always expires before
    the parent's, whatever the preparation took.
    """
    started = time.monotonic() if started is None else started
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
    require(guard['new_parameter']['shape'] == [cfg.model.dim, 64], 'guard written for another model width')
    try:
        precheck = {'passed': True, **controller.verify_initialization(cfg, guard)}
    except controller.G.InitialStateError as error:
        precheck = {'passed': False, 'message': str(error), **error.record}
    records.mkdir()
    pin = {'path': str((REPO / plan['initial_state_guard']['path']).resolve()), 'sha256': plan['initial_state_guard']['sha256']}
    command = controller.command_for(comparison, cfg, run_dir, 5000, training['allocations']['5000']['walltime_seconds'],
                                     pin, records)
    position = command.index(controller.RANK_ENTRY) + 1
    command = command[:position] + ['--verify-only'] + command[position:]
    outputs = [workspace / (RANK_VARIANT + '.torchrun.' + suffix) for suffix in ('stdout', 'stderr')]
    result = {'variant': RANK_VARIANT, 'imported_entrypoint': controller.RANK_ENTRY,
              'scope': ('production fresh-phase command with --verify-only under torchrun: four ranks, process group, the '
                        'trainer inputs read as the trainer reads them (manifest, index, scales) and model construction; '
                        'the constructor raises after verification, so no model reaches the trainer: no teacher, optimizer, '
                        'DDP wrapper, training batch, forward on signals, update or checkpoint'),
              'expected': expected, 'controller_precheck': precheck, 'command': command, 'torchrun_returncode': None,
              'torchrun_timed_out': False, 'torchrun_stderr_tail': '', 'ranks': {}}
    supervision, timed_out = {}, False
    seconds = plan['rank_entry_timeout_seconds'] - CLEANUP_BOUND_SECONDS - REPORT_SECONDS - (time.monotonic() - started)
    result['torchrun_bound_seconds'] = round(seconds, 3)
    if seconds < MIN_LAUNCH_SECONDS:
        result['error'] = (f'nothing launched: the preparation left {seconds:.1f}s for torchrun, '
                           f'fewer than {MIN_LAUNCH_SECONDS}s')
    else:
        returncode, timed_out = run_bounded(command, seconds, *outputs, record=supervision)
        result.update(torchrun_returncode=returncode, torchrun_timed_out=timed_out, torchrun_supervision=supervision,
                      torchrun_stderr_tail=outputs[1].read_text(errors='replace')[-4000:])
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
    problem = supervision_problem(supervision)
    if problem:
        result['cleanup_error'] = problem
    if 'error' in result:
        pass
    elif timed_out:
        result['error'] = 'torchrun timed out'
    elif not judged:
        result['error'] = 'not every rank judged a student; see the torchrun output'
    elif problem:
        result['error'] = problem
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

    failed = [name for name in VARIANTS if 'error' in variants.get(name, {'error': 'missing'})]
    try:
        require(not failed, 'failed contexts: ' + ', '.join(failed))
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
            'controller_precheck_passes': entry['controller_precheck']['passed'] is True,
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
                                      facts['controller_precheck_passes'] and
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
    require(process_table() is not None, 'process table unavailable: the diagnostic cannot supervise its processes')
    out.mkdir(parents=True, exist_ok=False)
    report = {'protocol': PROTOCOL, 'code_sha': code, 'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              'expected': expected, 'variants': {}, 'diagnostic_only': True,
              'training_authorized_by_report': False, 'complete': False}
    for variant in VARIANTS:
        destination = out / (variant + '.json')
        command = [sys.executable, str(Path(__file__).resolve()), '--plan', str(plan_path),
                   '--expected-code-sha', expected_code, '--child', variant, '--child-output', str(destination)]
        seconds = plan['rank_entry_timeout_seconds' if variant == RANK_VARIANT else 'child_timeout_seconds']
        supervision = {}
        try:
            # The child and whatever it started end here: a child that is stopped at the bound cannot clean up
            # after itself, so this supervisor ends its launcher and its workers too.
            returncode, timed_out = run_bounded(command, seconds, out / (variant + '.stdout'), out / (variant + '.stderr'),
                                                supervision)
            if timed_out:
                row = {'error': 'child timed out', 'timeout_seconds': seconds}
            else:
                row = json.loads(destination.read_text()) if destination.exists() else {'error': 'child produced no report'}
            row['process_returncode'] = returncode
        except Exception as error:
            row = {'error': str(error), 'traceback': traceback.format_exc()}
        problem = supervision_problem(supervision)
        if problem:
            row['cleanup_error'] = problem
            row.setdefault('error', problem)
        row['supervision'] = supervision
        if 'error' in row:
            row['output_tails'] = output_tails(out, variant)
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
    started = time.monotonic()
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
            report = (rank_entry_child(args.plan, args.expected_code_sha, args.child_output.parent, started)
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
