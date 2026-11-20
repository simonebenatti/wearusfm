"""Skeptic R5-5: apply one exact textual mutation in a throwaway clone, run pytest, restore.

usage: mut.py CLONE OUT.jsonl SCOPE ID[,ID...]
SCOPE: 'targeted' (files observing the mutated module) or 'ALL' (the 29 files of the receipt).
"""
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path

PY = '/Users/simonebenatti/.venvs/wearusfm-tok/bin/python'
TMP = Path('/private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp')
REL = 'scripts/contrastive_release.py'
READ = 'scripts/contrastive_benchmark_readout.py'
T_REL = ['tests/cpu/test_contrastive_release.py', 'tests/cpu/test_contrastive_job.py', 'tests/cpu/test_contrastive_run.py',
         'tests/cpu/test_contrastive_historical_callsites.py', 'tests/cpu/test_contrastive_targets_readout.py']
T_READ = ['tests/cpu/test_contrastive_benchmark_readout.py', 'tests/cpu/test_contrastive_targets_readout.py',
          'tests/cpu/test_contrastive_historical_callsites.py', 'tests/cpu/test_contrastive_numeric_profile.py',
          'tests/cpu/test_contrastive_release.py', 'tests/cpu/test_contrastive_run.py', 'tests/cpu/test_contrastive_job.py']

M = {
    # rilievo
    'C4d_audit_source': (REL, [('            or accounting.get("source_sha256") != source or audit.get("source_sha256") != source):\n',
                                '            or accounting.get("source_sha256") != source):\n')], T_REL),
    'C4g_required_columns': (REL, [('    if not records or not required <= set(records[0]):\n', '    if not records:\n')], T_REL),
    'C4h_job_id_digits': (REL, [('    if not isinstance(job_id, str) or not job_id.isdigit() or (expected_job_id is not None and job_id != str(expected_job_id)):\n',
                                 '    if (expected_job_id is not None and job_id != str(expected_job_id)):\n')], T_REL),
    'C5c_context_assert': (READ, [('                assert_numeric_flags(flags)\n            except NumericProfileUnavailable as exc:\n',
                                   '                pass\n            except NumericProfileUnavailable as exc:\n')], T_READ),
    'C5f_rank_flags_equal': (READ, [("    if any(row['flags'] != numeric[0]['flags'] for row in numeric[1:]):\n", '    if False:\n')], T_READ),
    'C5m_cuda_profile_assert': (READ, [("        assert_numeric_flags(rank.get('numeric_profile', {}))\n        if rank.get('passed') is not True",
                                        "        pass\n        if rank.get('passed') is not True")], T_READ),
    # extra mio: assert del profilo dentro _run_evidence (difesa in profondita' di C5c)
    'X220_run_evidence_assert': (READ, [("        assert_numeric_flags(row.get('flags', {}))\n        numeric.append(row)\n",
                                         "        pass\n        numeric.append(row)\n")], T_READ),
    # controlli positivi (devono morire)
    'K1_state_completed': (REL, [('    if (row["State"] != "COMPLETED" or row["ExitCode"] != "0:0" or not 0 <= elapsed <= 7200\n',
                                  '    if (False or row["ExitCode"] != "0:0" or not 0 <= elapsed <= 7200\n')], T_REL),
    'K2_context_differences': (READ, [('            if changed:\n                differences.append', '            if False:\n                differences.append')], T_READ),
    'K3_cuda_fixed_criterion': (READ, [("                        if fixed is not None and row['limits'][key] != fixed:\n",
                                        '                        if False:\n')], T_READ),
}


def receipt_files(clone):
    r = json.loads((clone / 'results/infra/contrastive_targets_20261118/amendment_20261119/validation_final/local_validation.json').read_text())
    return [x['path'] for x in r['test_files']]


def run(clone, mids, scope, out):
    clone = Path(clone).resolve(strict=True)
    touched, edits_all = [], []
    for mid in mids:
        rel, edits, tests = M[mid]
        for old, new in edits:
            edits_all.append((rel, old, new))
    tests = receipt_files(clone) if scope == 'ALL' else sorted({t for mid in mids for t in M[mid][2]})
    for rel, old, new in edits_all:
        path = clone / rel
        text = path.read_text()
        n = text.count(old)
        if n != 1:
            subprocess.run(['git', '-C', str(clone), 'checkout', '--', *touched], check=False)
            out.write(json.dumps({'ids': mids, 'error': f'{rel}: occurrences {n}'}) + '\n'); out.flush()
            return
        path.write_text(text.replace(old, new))
        touched.append(rel)
    diff = subprocess.run(['git', '-C', str(clone), 'diff', '--stat'], capture_output=True, text=True).stdout.strip().splitlines()
    bt = TMP / ('sk1r55_bt_' + '+'.join(mids) + '_' + scope)
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': f'{clone}/src:{clone}/tests/cpu:{clone}/scripts'}
    t0 = time.time()
    proc = subprocess.run([PY, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', '-o', 'tmp_path_retention_policy=none',
                           '-rfE', '--basetemp=' + str(bt), *tests], cwd=clone, env=env, capture_output=True, text=True)
    text = proc.stdout + proc.stderr
    failed = sorted(set(re.findall(r'^(?:FAILED|ERROR) (\S+)', text, re.M)))
    summary = [l for l in text.splitlines() if re.search(r'\d+ (passed|failed)', l)]
    subprocess.run(['git', '-C', str(clone), 'checkout', '--', *touched], check=True)
    clean = subprocess.run(['git', '-C', str(clone), 'status', '--porcelain'], capture_output=True, text=True).stdout == ''
    shutil.rmtree(bt, ignore_errors=True)
    out.write(json.dumps({'ids': mids, 'scope': scope, 'n_test_files': len(tests), 'rc': proc.returncode,
                          'seconds': round(time.time() - t0, 1), 'summary': summary[-1] if summary else text[-400:],
                          'failed': failed, 'tail': (text[-1500:] if proc.returncode not in (0, 1) else ''), 'diff_stat': diff[-1:] if diff else [], 'clean_after': clean}) + '\n')
    out.flush()


if __name__ == '__main__':
    clone, outpath, scope = sys.argv[1], sys.argv[2], sys.argv[3]
    groups = sys.argv[4:]
    with open(outpath, 'a') as out:
        for group in groups:
            run(clone, group.split(','), scope, out)
