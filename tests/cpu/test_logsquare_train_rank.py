"""The guarded training-rank entry of T100-LQ, exercised through the real, unchanged trainer on CPU."""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest
import torch
from wearusfm.model.fm import WearUsFM
from wearusfm.model.patch_tokens import PatchTokenizer
from wearusfm.training import run as R
from wearusfm.training.pairing import state_sha256

from test_logsquare_initial_state import G, local_guard, neighbour, small_model
from test_pretraining_loader import _tree

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('logsquare_rank_entry_test', ROOT / 'scripts/logsquare_train_rank.py')
K = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(K)
SPEC = importlib.util.spec_from_file_location('logsquare_rank_controller_test', ROOT / 'scripts/logsquare_train.py')
T = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(T)
# Child processes import the package of THIS checkout, not an editable install of another one.
WORKTREE = {**os.environ, 'PYTHONPATH': str(ROOT / 'src'), 'PYTHONDONTWRITEBYTECODE': '1'}


@pytest.fixture
def run(tmp_path):
    """A one-step small T100-LQ-like run on synthetic data, with a guard for this machine's seed-zero model."""
    root, manifest = _tree(tmp_path)
    guard = local_guard(small_model())
    guard_path = tmp_path / 'guard.json'
    guard_path.write_text(json.dumps(guard))
    records = tmp_path / 'initial_state'
    records.mkdir()

    def trainer(out, *extra):
        return ['--preset', 'small', '--manifest', str(manifest), '--root', str(root), '--out-dir', str(tmp_path / out),
                '--device', 'cpu', '--max-steps', '1', '--datasets', 'all', '--tokenizer-logsquare',
                '--tokenizer-logsquare-eps', '0.001', *extra]

    def entry(out, *extra, sha=None, directory=records, script=K.TRAINER, verify=False):
        return ['--initial-state-guard', str(guard_path), '--initial-state-guard-sha256', sha or R.sha256_file(guard_path),
                '--initial-state-records', str(directory), *(['--verify-only'] if verify else []), script, *trainer(out, *extra)]

    return {'tmp': tmp_path, 'guard': guard, 'guard_sha256': R.sha256_file(guard_path), 'records': records,
            'trainer': trainer, 'entry': entry}


def four_ranks(arguments):
    """Four real training processes in one process group, as torchrun starts them (its environment, without its launcher)."""
    with socket.socket() as free:
        free.bind(('127.0.0.1', 0))
        port = free.getsockname()[1]
    procs = [subprocess.Popen([sys.executable, *arguments, '--distributed', '--batch-size', '4'], cwd=ROOT,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                              env={**WORKTREE, 'RANK': str(rank), 'LOCAL_RANK': str(rank), 'WORLD_SIZE': '4',
                                   'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port)}) for rank in range(4)]
    results = []
    for proc in procs:
        try:
            _, stderr = proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            for other in procs:
                other.kill()
            raise
        results.append((proc.returncode, stderr))
    return results


def metrics(path):
    """Logged losses and diagnostics of every step, without the wall-clock fields."""
    rows = [json.loads(line) for line in (path / 'metrics.jsonl').read_text().splitlines()]
    return [{key: value for key, value in row.items() if not key.startswith('t_')} for row in rows]


def trained(path):
    state = torch.load(path / 'checkpoint.pt', weights_only=False)
    return state, state_sha256({key: state[key] for key in ('student', 'teacher', 'optimizer', 'torch_rng')})


ORDER = """
import importlib.util, sys
started = []
class Recorder:
    @staticmethod
    def find_spec(name, path=None, target=None):
        if name in ('torch', 'numpy') and name not in started:
            started.append(name)
        return None
sys.meta_path.insert(0, Recorder)
spec = importlib.util.spec_from_file_location('m', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(','.join(started))
"""


def test_entry_keeps_the_trainer_numeric_import_order():
    """The gain depends on which numeric library a process starts importing first. The rank entry must be a
    trainer context (Torch first), unlike the preflights (NumPy first) whose state the old check demanded."""
    order = {script: subprocess.run([sys.executable, '-c', ORDER, str(ROOT / script)], capture_output=True, text=True,
                                    check=True, env=WORKTREE).stdout.strip()
             for script in ('scripts/train_jepa.py', 'scripts/logsquare_train_rank.py', 'scripts/logsquare_train.py',
                            'scripts/tokenizer_train.py', 'scripts/logsquare_preflight.py', 'scripts/tokenizer_preflight.py')}
    assert order == {'scripts/train_jepa.py': 'torch,numpy', 'scripts/logsquare_train_rank.py': 'torch,numpy',
                     'scripts/logsquare_train.py': 'torch,numpy', 'scripts/tokenizer_train.py': 'torch,numpy',
                     'scripts/logsquare_preflight.py': 'numpy,torch', 'scripts/tokenizer_preflight.py': 'numpy,torch'}


def test_guarded_entry_verifies_the_trained_student_and_changes_nothing(run):
    assert K.CLI.main(run['trainer']('plain')) == 0
    assert K.main(run['entry']('guarded')) == 0
    assert R.WearUsFM is WearUsFM and R.count_parameters.__name__ == 'count_parameters'
    plain, plain_sha = trained(run['tmp'] / 'plain')
    guarded, guarded_sha = trained(run['tmp'] / 'guarded')
    # Same weights, teacher, optimizer moments and generator after one update: the guard only reads.
    assert guarded_sha == plain_sha and guarded['step'] == plain['step'] == 1
    assert metrics(run['tmp'] / 'guarded') == metrics(run['tmp'] / 'plain') and len(metrics(run['tmp'] / 'plain')) == 1
    assert json.loads((run['tmp'] / 'guarded' / 'config.json').read_text()) == json.loads((run['tmp'] / 'plain' / 'config.json').read_text())
    report = json.loads((run['records'] / 'rank0.json').read_text())
    assert report['passed'] is True and report['trainer_returned'] is True and (report['rank'], report['world_size']) == (0, 1)
    # Rank zero: the unseeded parameter-count model is recorded and not judged; the student is the verified one.
    assert [row['role'] for row in report['constructions']] == ['parameter_count', 'student']
    counted, student = report['constructions']
    assert counted['judged'] is False and student['classification'] == G.PASSED and student['passed'] is True
    assert student['reference_gain_shared_sha256'] == run['guard']['reference']['shared_sha256'] != student['shared_sha256']
    assert student['reference_gain']['observed_minus_registered_ulps'] == -1
    # The verified gain is the buffer the checkpoint saves, in student and teacher: the state actually trained.
    for kind in ('student', 'teacher'):
        assert guarded[kind][G.GAIN].item() == student['fe_gain'] == run['guard']['training_gain']['fe_gain']
    assert not bool(torch.equal(guarded['student'][G.NEW_WEIGHT], torch.zeros_like(guarded['student'][G.NEW_WEIGHT])))


@pytest.mark.parametrize('case', ['seed', 'preflight_gain', 'other_gain'])
def test_rejected_student_stops_the_rank_before_any_update(run, monkeypatch, case):
    registered = run['guard']['training_gain']['fe_gain']
    if case != 'seed':  # another numeric context: the constructor calibrates one or two ULP away
        gain = run['guard']['reference']['fe_gain'] if case == 'preflight_gain' else neighbour(registered, -2)
        monkeypatch.setattr(PatchTokenizer, '_calibrate_gain', lambda self, *args, **kwargs: gain)
    with pytest.raises(K.G.InitialStateError) as caught:
        K.main(run['entry']('stopped', *(('--seed', '1') if case == 'seed' else ())))
    assert R.WearUsFM is WearUsFM and R.count_parameters.__name__ == 'count_parameters'
    out = run['tmp'] / 'stopped'
    assert not (out / 'checkpoint.pt').exists() and not (out / 'metrics.jsonl').exists() and not (out / 'config.json').exists()
    report = json.loads((run['records'] / 'rank0.json').read_text())
    assert report['passed'] is False and report['trainer_returned'] is False
    student = report['constructions'][-1]
    expected = {'seed': G.OTHER_TENSORS, 'preflight_gain': G.PREFLIGHT_STATE, 'other_gain': G.UNREGISTERED_GAIN}[case]
    assert student['role'] == 'student' and student['classification'] == expected == caught.value.record['classification']
    assert 'training rank 0 of 1' in str(caught.value) and 'the student' in str(caught.value)
    assert len(student['localization']['per_tensor']) == student['tensors']


def test_verify_only_stops_inside_the_constructor_and_trains_nothing(run, monkeypatch):
    """The diagnostic dry run: the student is built by the unchanged trainer and verified; the trainer never receives it."""
    assert K.main(run['entry']('dry', verify=True)) == 0
    assert R.WearUsFM is WearUsFM and R.count_parameters.__name__ == 'count_parameters'
    assert (run['tmp'] / 'dry').is_dir() and not list((run['tmp'] / 'dry').iterdir())  # the trainer only made its directory
    report = json.loads((run['records'] / 'rank0.json').read_text())
    assert report['verify_only'] is True and report['passed'] is True and report['trainer_returned'] is False
    assert [row['role'] for row in report['constructions']] == ['parameter_count', 'student']
    assert report['constructions'][-1]['classification'] == G.PASSED
    # The same switch on a rejected student still fails, with the full record.
    with pytest.raises(K.G.InitialStateError):
        K.main(run['entry']('dry_rejected', '--seed', '1', verify=True))
    assert json.loads((run['records'] / 'rank0.json').read_text())['passed'] is False
    # A trainer that returns without building a student (here: a STOP file) is not a verification.
    (run['tmp'] / 'stopped').mkdir()
    (run['tmp'] / 'stopped' / 'STOP').write_text('earlier alarm')
    with pytest.raises(ValueError, match='returned without building a student'):
        K.main(run['entry']('stopped', verify=True))
    with pytest.raises(ValueError, match='without a verified student'):
        K.main(run['entry']('stopped'))
    assert R.WearUsFM is WearUsFM


def test_four_real_ranks_train_the_verified_state_and_the_dry_run_trains_nothing(run):
    """Real process group, DDP wrapper and buffer broadcast on CPU: behind the entry the four ranks train exactly the
    state the plain trainer trains, every rank attests the student it built, and the dry run stops before training."""
    tmp, records = run['tmp'], run['records']
    guarded = ['scripts/logsquare_train_rank.py', *run['entry']('guarded', '--max-steps', '2')]
    plain = ['scripts/train_jepa.py', *run['trainer']('plain', '--max-steps', '2')]
    for arguments in (plain, guarded):
        assert [code for code, _ in four_ranks(arguments)] == [0, 0, 0, 0]
    states = {name: torch.load(tmp / name / 'checkpoint.pt', weights_only=False) for name in ('plain', 'guarded')}
    kinds = ('student', 'teacher', 'optimizer', 'torch_rng', 'rank_rng')
    assert states['guarded']['step'] == states['plain']['step'] == 2 and states['guarded']['world_size'] == 4
    assert state_sha256({k: states['guarded'][k] for k in kinds}) == state_sha256({k: states['plain'][k] for k in kinds})
    assert metrics(tmp / 'guarded') == metrics(tmp / 'plain') and len(metrics(tmp / 'plain')) == 2
    attested = T.rank_attestation(records, run['guard_sha256'])
    assert attested['classification'] == G.PASSED and attested['models_built_per_rank'] == [2, 1, 1, 1]
    for kind in ('student', 'teacher'):  # the buffer rank zero saved is the one every rank verified
        assert states['guarded'][kind][G.GAIN].item().hex() == attested['fe_gain_hex']
    assert attested['fe_gain'] == run['guard']['training_gain']['fe_gain']

    dry = tmp / 'dry_records'
    dry.mkdir()
    command = ['scripts/logsquare_train_rank.py', *run['entry']('dry', directory=dry, verify=True)]
    assert [code for code, _ in four_ranks(command)] == [0, 0, 0, 0]
    assert not list((tmp / 'dry').iterdir())
    verified = T.rank_attestation(dry, run['guard_sha256'], verify_only=True)
    assert (verified['shared_sha256'], verified['fe_gain_hex']) == (attested['shared_sha256'], attested['fe_gain_hex'])
    with pytest.raises(ValueError, match='unverified initial state'):
        T.rank_attestation(dry, run['guard_sha256'])

    rejected = tmp / 'rejected_records'
    rejected.mkdir()
    command = ['scripts/logsquare_train_rank.py', *run['entry']('rejected', '--seed', '1', directory=rejected)]
    results = four_ranks(command)
    assert all(code != 0 for code, _ in results) and all('tensor_difference_beyond_gain' in stderr for _, stderr in results)
    assert not (tmp / 'rejected' / 'checkpoint.pt').exists() and not (tmp / 'rejected' / 'metrics.jsonl').exists()
    for rank in range(4):
        row = json.loads((rejected / f'rank{rank}.json').read_text())
        assert row['passed'] is False and row['constructions'][-1]['classification'] == G.OTHER_TENSORS
    with pytest.raises(ValueError, match='unverified initial state'):
        T.rank_attestation(rejected, run['guard_sha256'])


def test_entry_is_only_for_a_fresh_guarded_production_run(run, tmp_path):
    with pytest.raises(ValueError, match='differs from its pin'):
        K.main(run['entry']('a', sha='0' * 64))
    with pytest.raises(ValueError, match='only the production trainer'):
        K.main(run['entry']('a', script='scripts/tokenizer_train.py'))
    with pytest.raises(ValueError, match='record directory'):
        K.main(run['entry']('a', directory=tmp_path / 'absent'))
    with pytest.raises(ValueError, match='no warm start'):
        K.main(run['entry']('a', '--init-from', str(tmp_path / 'other.pt')))
    without_branch = [x for x in run['entry']('a') if x not in ('--tokenizer-logsquare', '--tokenizer-logsquare-eps', '0.001')]
    with pytest.raises(ValueError, match='would modify the student'):
        K.main(without_branch)
    assert K.main(run['entry']('resumed')) == 0  # a finished run: its checkpoint makes a second start a resume
    with pytest.raises(ValueError, match='fresh run only'):
        K.main(run['entry']('resumed'))
    assert R.WearUsFM is WearUsFM and not (tmp_path / 'a').exists()


def test_install_refuses_a_replaced_constructor_and_judges_every_other_model(run):
    report, write, originals = K.install(run['guard'], 'f' * 64, run['records'], 2, 4)
    try:
        with pytest.raises(ValueError, match='already replaced'):
            K.install(run['guard'], 'f' * 64, run['records'], 2, 4)
        model_cfg = replace(R.small_config().model, tokenizer_logsquare=True)
        torch.manual_seed(0)
        assert isinstance(R.WearUsFM(model_cfg), WearUsFM)  # seeded as the trainer seeds it: the registered state
        assert R.count_parameters(model_cfg) > 0             # counted, not judged, whatever the generator state
        torch.manual_seed(5)
        with pytest.raises(K.G.InitialStateError):               # any model outside the count is a student
            R.WearUsFM(model_cfg)
    finally:
        R.WearUsFM, R.count_parameters = originals
    assert [row['role'] for row in report['constructions']] == ['student', 'parameter_count', 'student']
    assert report['passed'] is False and report['constructions'][0]['passed'] is True
    saved = json.loads((run['records'] / 'rank2.json').read_text())
    assert saved == json.loads(json.dumps(report)) and (saved['rank'], saved['world_size']) == (2, 4)
