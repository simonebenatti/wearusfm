import json, re, sys
from pathlib import Path
from fractions import Fraction as F
root = Path(sys.argv[1])
sys.path.insert(0, str(root/'scripts'))
import contrastive_job as J
plan = json.loads((root/'results/infra/contrastive_targets_20261118/preflight_plan.json').read_text())
def sb(path):
    d = {}
    for line in (root/path).read_text().splitlines():
        m = re.match(r'#SBATCH\s+(--?[\w-]+)(?:[= ](\S+))?', line)
        if m: d[m.group(1)] = m.group(2)
    return d
gpu, cpu, chk = sb('scripts/slurm/contrastive_preflight_gpu.sbatch'), sb('scripts/slurm/contrastive_preflight_cpu.sbatch'), sb('scripts/slurm/campaign_checkout.sbatch')
def secs(t):
    p = [int(x) for x in t.split(':')]; return p[0]*3600+p[1]*60+p[2]
print('gpu launcher', gpu); print('cpu launcher', cpu); print('checkout launcher', chk)
G = C = F(0); bad = []
for c in plan['cells']:
    m = c['mode']; kind = 'gpu' if m in J.GPU_MODES else 'cpu'
    L = gpu if kind == 'gpu' else cpu
    default = secs(L['--time'])
    need_override = default != c['seconds']
    ok = (J.ALLOCATED_SECONDS[m] == c['seconds'] and (c['gpus'] == 4) == (kind == 'gpu') and
          c['cpus'] == (32 if kind == 'gpu' else 8) and int(L['--cpus-per-task']) == c['cpus'] and
          L['--mem'] == f"{c['memory_gib']}G")
    G += F(c['gpus']*c['seconds'], 3600); C += F(c['cpus']*c['seconds'], 3600)
    if F(c['gpus']*c['seconds'],3600) != F(c['gpu_hours']).limit_denominator(10**6) or F(c['cpus']*c['seconds'],3600) != F(c['core_hours']).limit_denominator(10**6): bad.append(c['id'])
    print(f"{c['id']:45s} {m:11s} {kind} sec={c['seconds']} dispatcher={J.ALLOCATED_SECONDS[m]} launcher_default={default} override_needed={need_override} consistent={ok}")
k = plan['checkouts']; Ck = F(k['count']*k['cpus']*k['seconds_each'], 3600)
print('GPU-h', G, float(G), 'core-h cells', C, float(C), 'checkouts', Ck, float(Ck), 'total core', float(C+Ck), 'mismatched rows', bad)
print('checkout default cost if overrides forgotten (1 CPU x 600 s x 6):', 6*1*600/3600, '; worst (4 CPU x 600 s x 6):', 6*4*600/3600)
print('ALLOCATED_SECONDS', J.ALLOCATED_SECONDS)
