"""One emulated rank running the REAL contrastive_run.main control flow (fbfde797/0efa4d45).

Only the expensive validations, CUDA and the trainer are replaced. torch.distributed.init_process_group runs the
real torch code path (env:// rendezvous, TCPStore, TORCH_DIST_INIT_BARRIER flag); the NCCL backend is replaced by a
lazily-initialised backend whose constructor, like ProcessGroupNCCL in torch 2.2.1, performs no store operation.
"""
import os, sys, time, json, traceback
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

CLONE = Path(os.environ['V2_CLONE'])
sys.path[:0] = [str(CLONE / 'src'), str(CLONE / 'scripts')]
import torch
import torch.distributed as dist

rank = int(os.environ['RANK'])
DELAY_RANK = int(os.environ.get('V2_DELAY_RANK', '-1'))
DELAY = float(os.environ.get('V2_DELAY', '0'))
CUDA_INIT = float(os.environ.get('V2_CUDA_INIT', '0'))
BACKEND = os.environ.get('V2_BACKEND', 'lazy')
TIMEOUT = float(os.environ.get('V2_TIMEOUT', '20'))
OUT = Path(os.environ['V2_OUT'])


def stamp(event, **kw):
    print(json.dumps({'rank': rank, 'event': event, 't': time.time(), **kw}), flush=True)


import contrastive_run as runner
from contrastive_prepare import config_from_dict
import contrastive_numeric_profile as NP
import contrastive_target_resume as TR


def validate_recipe(raw, stage, calibration=None, run_dir=None):
    return config_from_dict(raw)


def validate_release(release, stage, config_path, input_paths, *args, **kwargs):
    if rank == DELAY_RANK:
        time.sleep(DELAY)  # emulated skew: this rank is still validating
    return {'initial_state_sha256': '0' * 64, 'calibration_sha256': '0' * 64}


runner.validate_recipe = validate_recipe
runner.validate_release = validate_release
runner.validate_target_fork = lambda *a, **k: None
runner.R.check_input_files = lambda *a, **k: None

_exists, _mkdir = Path.exists, Path.mkdir


def exists(self, *a, **k):
    result = _exists(self, *a, **k)
    if self == OUT:
        stamp('exists_check', result=result)
    return result


def mkdir(self, *a, **k):
    if self == OUT:
        stamp('mkdir_out_dir', already=_exists(self))
    return _mkdir(self, *a, **k)


Path.exists, Path.mkdir = exists, mkdir

torch.cuda.is_available = lambda: True
torch.cuda.set_device = lambda device: time.sleep(CUDA_INIT)


class LazyProcessGroup(dist.ProcessGroup):
    def __init__(self, rank, size):
        super().__init__(rank, size)

    def getBackendName(self):
        return 'lazy'


def create_lazy(store, group_rank, group_size, timeout):
    return LazyProcessGroup(group_rank, group_size)


dist.Backend.register_backend('lazy', create_lazy, devices=['cpu'])
_init = dist.init_process_group


def init_process_group(backend=None, **kw):
    stamp('init_enter')
    _init(BACKEND, timeout=timedelta(seconds=TIMEOUT))
    stamp('init_return')


def store_barrier(*a, **k):
    store = dist.distributed_c10d._get_default_store()
    stamp('barrier_enter')
    store.add('v2_barrier', 1)
    deadline = time.time() + TIMEOUT
    while int(store.get('v2_barrier')) < int(os.environ['WORLD_SIZE']):
        if time.time() > deadline:
            raise RuntimeError('emulated barrier timeout')
        time.sleep(0.005)
    stamp('barrier_return')


torch.distributed.init_process_group = init_process_group
if BACKEND == 'lazy':
    torch.distributed.barrier = store_barrier


@contextmanager
def numeric_profile(*, require_cuda=True):
    yield {'profile': 'emulated-no-cuda'}


NP.numeric_profile = numeric_profile


@contextmanager
def trace(path, phase, **kw):
    Path(path).open('x').close()
    yield


TR.trace_resume_updates = trace


def prepare_reload(source, sha, out_dir, cfg, provenance, initial, *, phase):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)  # as contrastive_target_resume.py:100
    (out / 'checkpoint.pt').write_bytes(Path(source).read_bytes())


TR.prepare_reload = prepare_reload


def train(cfg, manifest, roots, out_dir, **kw):
    stamp('train_called')
    # Emulates the trainer's collectives: nobody leaves before all ranks arrived; rank 0 (store host) leaves last.
    store = dist.distributed_c10d._get_default_store()
    store.add('v2_done', 1)
    deadline = time.time() + TIMEOUT
    while int(store.get('v2_done')) < int(os.environ['WORLD_SIZE']):
        if time.time() > deadline:
            raise RuntimeError('emulated trainer collective timeout')
        time.sleep(0.005)
    store.add('v2_left', 1)
    if rank == 0:
        while int(store.get('v2_left')) < int(os.environ['WORLD_SIZE']):
            time.sleep(0.005)
        time.sleep(0.2)
    return {'steps': cfg.max_steps, 'stopped': 'max_steps'}


runner.R.train = train

sys.argv = ['contrastive_run.py'] + json.loads(os.environ['V2_ARGV'])
stamp('start')
try:
    runner.main()
    stamp('done')
except BaseException as exc:
    stamp('error', type=type(exc).__name__, message=str(exc)[:300])
    sys.exit(1)
