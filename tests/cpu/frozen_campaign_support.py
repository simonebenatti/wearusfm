"""Separate archived campaign evidence from synthetic tests on the merged tree.

No published plan, source contract, controller or execution guard is rewritten.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def archive(tmp_path, sha, *paths):
    target = tmp_path / ('frozen-' + sha[:10])
    target.mkdir()
    raw = subprocess.check_output(['git', 'archive', sha, *paths], cwd=ROOT)
    subprocess.run(['tar', '-xf', '-', '-C', str(target)], input=raw, check=True)
    return target


def run_frozen(tmp_path, sha, paths, program, *args):
    target = archive(tmp_path, sha, *paths)
    bootstrap = """import sys\nfrom pathlib import Path\nroot=Path(sys.argv[1]).resolve()\nsys.path[:0]=[str(root/'src'),str(root/'scripts')]\n"""
    env = {**os.environ, 'PYTHONPATH': str(target / 'src'), 'PYTHONDONTWRITEBYTECODE': '1',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    result = subprocess.run([sys.executable, '-c', bootstrap + program, str(target), *map(str, args)],
                            cwd=target, env=env, text=True, capture_output=True, check=True, timeout=90)
    return json.loads(result.stdout)


def git_source_identity(sha):
    """The long evaluation hashes source bytes, not Git object identifiers."""
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', sha, 'src', 'scripts'],
                                    cwd=ROOT, text=True).splitlines()
    selected = [p for p in paths if (p.startswith('src/') and p.endswith('.py')) or
                (p.startswith('scripts/') and p.count('/') == 1 and p.endswith('.py')) or
                p in ('scripts/slurm/consolidation_long.sbatch', 'scripts/slurm/lib_code_dir.sh')]
    hashes = {p: hashlib.sha256(subprocess.check_output(['git', 'show', sha + ':' + p], cwd=ROOT)).hexdigest()
              for p in sorted(selected)}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, allow_nan=False).encode()).hexdigest()


def synthetic_scaling_contract(module, tmp_path):
    """A test-only contract for the current tree; historical evidence is tested separately.

    Inputs/recipe remain the registered ones, while current source inventory and
    neutral configuration fields are generated into temporary synthetic pins.
    This never makes the archived production contract valid on different bytes.
    """
    original = ROOT / 'results/infra/scaling_300m_20261104/training_source_contract.json'
    contract = json.loads(original.read_text())
    comparison = json.loads((ROOT / contract['comparison']['path']).read_text())
    args = comparison['arm_configurations']['T100-G']['validated_trainer_argv']
    config = module.R.config_to_dict(module.CLI.build_config(module.CLI.build_parser().parse_args(args[1:])))
    paths = {str(p.relative_to(ROOT)) for p in (ROOT / 'src/wearusfm').rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo')}
    paths |= module.SCRIPT_PATHS
    contract['scientific_source_sha256'] = {p: module.R.sha256_file(ROOT / p) for p in sorted(paths)}
    def save(name, data):
        path = tmp_path / name
        path.write_text(json.dumps(data))
        return {'path': str(path), 'sha256': module.R.sha256_file(path)}
    contract['config_30m'] = save('synthetic_control_config.json', config)
    comparison['arm_configurations']['T100-G']['sha256'] = contract['config_30m']['sha256']
    contract['comparison'] = save('synthetic_comparison.json', comparison)
    return Path(save('synthetic_source_contract.json', contract)['path'])
