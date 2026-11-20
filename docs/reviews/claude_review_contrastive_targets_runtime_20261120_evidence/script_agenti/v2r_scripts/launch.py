"""Launch 4 emulated ranks with the rendezvous semantics of torch 2.2.1 `torchrun --standalone`
(TORCHELASTIC_USE_AGENT_STORE=False: rank 0 hosts the TCPStore) or, with --agent-store, those of a shared agent store."""
import argparse, hashlib, json, os, socket, subprocess, sys, time
from datetime import timedelta
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--clone', required=True)
ap.add_argument('--base', required=True)
ap.add_argument('--mode', choices=('full', 'calibration', 'resume_a'), required=True)
ap.add_argument('--delay-rank', type=int, default=-1)
ap.add_argument('--delay', type=float, default=0.0)
ap.add_argument('--cuda-init', type=float, default=0.0)
ap.add_argument('--backend', default='lazy')
ap.add_argument('--agent-store', action='store_true')
ap.add_argument('--init-barrier', action='store_true')
ap.add_argument('--python', default=sys.executable)
a = ap.parse_args()

clone, base = Path(a.clone), Path(a.base)
base.mkdir(parents=True, exist_ok=False)
sys.path[:0] = [str(clone / 'src'), str(clone / 'scripts')]
from contrastive_prepare import source_digest
src = source_digest()
out = base / 'out'
for name, payload in (('manifest.json', {}), ('scales.json', {'scales': {}}), ('window.json', {'window_s': {}})):
    (base / name).write_text(json.dumps(payload))
(base / 'root').mkdir()
release = {'stage': 'calibration' if a.mode == 'calibration' else 'benchmark', 'inputs': {'manifest': '0' * 64},
           'source_sha256': src}
if a.mode != 'calibration':
    release['phase'] = a.mode
if a.mode == 'resume_a':
    fork = base / 'fork' / 'checkpoint_step0000544.pt'
    fork.parent.mkdir()
    fork.write_bytes(b'emulated fork checkpoint')
    release['fork_checkpoint'] = {'path': str(fork), 'sha256': hashlib.sha256(fork.read_bytes()).hexdigest()}
(base / 'release.json').write_text(json.dumps(release))
configs = clone / 'results/infra/contrastive_targets_20261118/review_configs'
config = configs / ('calibration.config.json' if a.mode == 'calibration' else 'CN.config.json')
argv = ['--config', str(config), '--stage', release['stage'], '--release', str(base / 'release.json'),
        '--manifest', str(base / 'manifest.json'), '--root', str(base / 'root'), '--scales', str(base / 'scales.json'),
        '--window-seconds', str(base / 'window.json'), '--out-dir', str(out), '--time-limit-s', '1770']

with socket.socket() as s:
    s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
agent_store = None
env_common = {'WORLD_SIZE': '4', 'MASTER_ADDR': '127.0.0.1', 'MASTER_PORT': str(port), 'V2_CLONE': str(clone),
              'V2_OUT': str(out), 'V2_ARGV': json.dumps(argv), 'V2_DELAY_RANK': str(a.delay_rank), 'V2_DELAY': str(a.delay),
              'V2_CUDA_INIT': str(a.cuda_init), 'V2_BACKEND': a.backend, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
              'MKL_NUM_THREADS': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'TORCHELASTIC_RESTART_COUNT': '0'}
if a.agent_store:
    from torch.distributed import TCPStore
    agent_store = TCPStore('127.0.0.1', port, 4, True, timedelta(seconds=60), wait_for_workers=False)
    env_common['TORCHELASTIC_USE_AGENT_STORE'] = 'True'
else:
    env_common['TORCHELASTIC_USE_AGENT_STORE'] = 'False'
if a.init_barrier:
    env_common['TORCH_DIST_INIT_BARRIER'] = '1'
procs = []
worker = Path(__file__).resolve().parent / 'worker.py'
for r in range(4):
    env = dict(os.environ, **env_common, RANK=str(r), LOCAL_RANK=str(r))
    env.pop('TORCH_DIST_INIT_BARRIER', None) if not a.init_barrier else None
    f = (base / f'rank{r}.log').open('w')
    procs.append((subprocess.Popen([a.python, '-B', str(worker)], env=env, stdout=f, stderr=subprocess.STDOUT), f))
t0 = time.time(); first_fail = None
while True:
    codes = [p.poll() for p, _ in procs]
    if first_fail is None and any(c not in (None, 0) for c in codes):
        first_fail = time.time()
    if all(c is not None for c in codes):
        break
    if first_fail is not None and time.time() - first_fail > 3:  # torchrun-like: tear down the others
        for p, _ in procs:
            if p.poll() is None: p.kill()
    if time.time() - t0 > 120:
        for p, _ in procs:
            if p.poll() is None: p.kill()
    time.sleep(0.05)
for _, f in procs: f.close()
events = []
for r in range(4):
    for line in (base / f'rank{r}.log').read_text().splitlines():
        if line.startswith('{"rank"'):
            events.append(json.loads(line))
events.sort(key=lambda e: e['t'])
start = min(e['t'] for e in events)
for e in events:
    if e['event'] in ('start',):
        continue
    extra = {k: v for k, v in e.items() if k not in ('rank', 'event', 't')}
    print(f"{e['t'] - start:7.3f}s rank{e['rank']} {e['event']} {json.dumps(extra) if extra else ''}")
print('exit codes', [p.returncode for p, _ in procs])
