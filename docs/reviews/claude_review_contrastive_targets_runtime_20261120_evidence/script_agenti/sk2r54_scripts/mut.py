"""Independent mutation driver (skeptic 2, R5-4). Usage: mut.py CLONE OUT.jsonl GROUP ID[,ID...]
GROUP: 'targeted' (parity + targets_readout test files) or 'receipt' (all 29 receipt files).
Several IDs joined by '+' are applied together in one run."""
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path

PY = '/Users/simonebenatti/.venvs/wearusfm-tok/bin/python'
TMP = Path('/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp')
PAR = 'scripts/contrastive_target_parity.py'
MUT = {
    'C1c': (PAR, '    if replicated_inputs and len(set(replicated_inputs)) != 1:\n', '    if False:\n'),
    'C1d': (PAR, '            row["live_state_restored"] = before == after and state_sha256(saved_grads) == state_sha256(_gradient_state(student))\n',
                 '            row["live_state_restored"] = True\n'),
    'C1t': (PAR, '        if not all(r["live_state_restored"] for r in point["ranks"]):\n', '        if False:\n'),
    'C1f': (PAR, '        if batch.index != runtime.step:\n', '        if False:\n'),
    'C1l': (PAR, '        if cfg.batch_size != 32 or cfg.accum_steps != 1 or cfg.amp_bf16 is not True or ctx.world_size != 4:\n', '        if False:\n'),
    'C1m': (PAR, '        if self.config_sha256 not in (None, cfg_digest):\n', '        if False:\n'),
    'C1n': (PAR, '        if update in [p["update"] for p in self.points]:\n', '        if False:\n'),
    'C1q': (PAR, '    if report.get("schema") != 1 or report.get("profile") != PROFILE:\n', '    if False:\n'),
    # control: must be KILLED (byte identity of gradients)
    'CTRL_bytes': (PAR, '    if hashes and len(set(hashes)) != 1 and not missing:\n', '    if False:\n'),
    # control: live_state_restored check inside the gate (assess_point:171)
    'CTRL_gate_restored': (PAR, '        if row.get("live_state_restored") is not True:\n', '        if False:\n'),
}
TARGETED = ['tests/cpu/test_contrastive_target_parity.py', 'tests/cpu/test_contrastive_targets_readout.py']


def receipt(clone):
    r = json.loads((clone / 'results/infra/contrastive_targets_20261118/amendment_20261119/validation_final/local_validation.json').read_text())
    return [x['path'] for x in r['test_files']]


def main():
    clone, out, group, ids = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4].split(',')
    tests = TARGETED if group == 'targeted' else receipt(clone)
    with open(out, 'a') as res:
        for combo in ids:
            names = [n for n in combo.split('+') if n != 'NONE']
            originals = {}
            err = None
            for n in names:
                rel, old, new = MUT[n]
                p = clone / rel
                text = p.read_text()
                originals.setdefault(rel, text)
                if text.count(old) != 1:
                    err = f'{n}: occurrences {text.count(old)}'
                    break
                p.write_text(text.replace(old, new))
            if err is None:
                diff = subprocess.run(['git', '-C', str(clone), 'diff', '--stat'], capture_output=True, text=True).stdout.strip()
                bt = TMP / ('sk2r54_bt_' + combo.replace('+', '_') + '_' + group)
                env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': f'{clone}/src:{clone}/tests/cpu:{clone}/scripts'}
                t0 = time.time()
                proc = subprocess.run([PY, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-rfE',
                                       '--basetemp=' + str(bt), *tests], cwd=clone, env=env, capture_output=True, text=True)
                output = proc.stdout + proc.stderr
                summ = [l for l in output.splitlines() if re.search(r'\d+ (passed|failed|error)', l)]
                failed = sorted(set(re.findall(r'^(?:FAILED|ERROR) (\S+)', output, re.M)))
                shutil.rmtree(bt, ignore_errors=True)
            for rel, text in originals.items():
                (clone / rel).write_text(text)
            clean = subprocess.run(['git', '-C', str(clone), 'status', '--porcelain'], capture_output=True, text=True).stdout == ''
            row = {'combo': combo, 'group': group, 'n_test_files': len(tests), 'clean_after': clean}
            if err:
                row['error'] = err
            else:
                row.update(rc=proc.returncode, seconds=round(time.time() - t0, 1), summary=summ[-1] if summ else output[-400:],
                           failed=failed, diffstat=diff.splitlines()[-1] if diff else '')
            res.write(json.dumps(row) + '\n'); res.flush()
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
