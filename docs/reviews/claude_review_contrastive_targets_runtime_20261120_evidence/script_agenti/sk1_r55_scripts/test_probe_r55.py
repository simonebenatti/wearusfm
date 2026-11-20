"""Skeptic R5-5 probes: behaviour of the original guards and of in-memory mutants.

Runs against the probe clone (PYTHONPATH clone/src:clone/tests/cpu:clone/scripts). The clone files are
never modified: mutants are compiled from the source text into separate module objects.
"""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest

import contrastive_release as C
import contrastive_benchmark_readout as B
from contrastive_numeric_profile import NumericProfileUnavailable, numeric_profile, numeric_flags

LOG = []


def note(*parts):
    LOG.append(' '.join(str(p) for p in parts))
    print('PROBE', *parts)


def mutant(module, old, new, name):
    path = Path(module.__file__)
    text = path.read_text()
    assert text.count(old) == 1, (name, text.count(old))
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(text.replace(old, new), str(path), 'exec'), mod.__dict__)
    return mod


C4D = ('            or accounting.get("source_sha256") != source or audit.get("source_sha256") != source):\n',
       '            or accounting.get("source_sha256") != source):\n')
C4G = ('    if not records or not required <= set(records[0]):\n', '    if not records:\n')
C4H = ('    if not isinstance(job_id, str) or not job_id.isdigit() or (expected_job_id is not None and job_id != str(expected_job_id)):\n',
       '    if (expected_job_id is not None and job_id != str(expected_job_id)):\n')
C5C = ('                assert_numeric_flags(flags)\n            except NumericProfileUnavailable as exc:\n',
       '                pass\n            except NumericProfileUnavailable as exc:\n')
X220 = ("        assert_numeric_flags(row.get('flags', {}))\n        numeric.append(row)\n", "        pass\n        numeric.append(row)\n")
C5M = ("        assert_numeric_flags(rank.get('numeric_profile', {}))\n        if rank.get('passed') is not True",
       "        pass\n        if rank.get('passed') is not True")

S = 'a' * 64


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')
    return Path(path)


def accounting_case(tmp_path, *, header='JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw',
                    rows=('123|COMPLETED|0:0|7000|8|120', '123.batch|COMPLETED|0:0|6999|8|120'),
                    job_id='123', audit_source=S, accounting_source=S):
    preflight = dump(tmp_path / 'preflight.json', {'cells': [{'id': 'new-audit', 'mode': 'audit'}]})
    audit = dump(tmp_path / 'audit.json', {'source_sha256': audit_source, 'passed': True})
    tsv = tmp_path / 'sacct.tsv'
    tsv.write_text(header + '\n' + ''.join(r + '\n' for r in rows))
    receipt = dump(tmp_path / 'accounting.json', {'schema': C.AUDIT_SCHEMA, 'source_sha256': accounting_source,
                   'audit_sha256': C.file_sha(audit), 'logical_id': 'new-audit', 'job_id': job_id, 'sacct': C.pin(tsv)})
    return preflight, audit, receipt


def outcome(fn, *args, **kwargs):
    try:
        value = fn(*args, **kwargs)
        return 'ACCEPTED ' + json.dumps({k: value[k] for k in ('job_id', 'elapsed_seconds', 'state')} if isinstance(value, dict) and 'job_id' in value else str(type(value).__name__))
    except Exception as exc:  # report, never hide
        return f'{type(exc).__name__}: {exc}'


def run_accounting(module, monkeypatch, preflight, audit, receipt):
    monkeypatch.setattr(module, 'PREFLIGHT', preflight)
    return outcome(module.validate_audit_accounting, audit, receipt, expected_source=S)


def test_A0_valid_baseline(tmp_path, monkeypatch):
    pre, audit, receipt = accounting_case(tmp_path)
    res = run_accounting(C, monkeypatch, pre, audit, receipt)
    note('A0 original valid case:', res)
    assert res.startswith('ACCEPTED')


def test_A1_required_columns(tmp_path, monkeypatch):
    m = mutant(C, *C4G, 'cr_c4g')
    for missing in ('TimelimitRaw', 'State', 'JobIDRaw'):
        cols = [c for c in 'JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|TimelimitRaw'.split('|') if c != missing]
        values = {'JobIDRaw': '123', 'State': 'COMPLETED', 'ExitCode': '0:0', 'ElapsedRaw': '7000', 'AllocCPUS': '8', 'TimelimitRaw': '120'}
        d = tmp_path / missing
        d.mkdir()
        pre, audit, receipt = accounting_case(d, header='|'.join(cols), rows=('|'.join(values[c] for c in cols),))
        orig = run_accounting(C, monkeypatch, pre, audit, receipt)
        mut = run_accounting(m, monkeypatch, pre, audit, receipt)
        note(f'A1 header without {missing}: original ->', orig, '| C4g ->', mut)
        assert orig.startswith('ValueError') and not mut.startswith('ACCEPTED')


def test_A2_job_id(tmp_path, monkeypatch):
    m = mutant(C, *C4H, 'cr_c4h')
    cases = {
        'fixture_format_step_120': dict(job_id='123.batch'),
        'leonardo_format_step_blank': dict(job_id='123.batch', header='JobIDRaw|State|ExitCode|ElapsedRaw|AllocCPUS|ReqMem|MaxRSS|TimelimitRaw',
                                           rows=('123|COMPLETED|0:0|7000|8|30G||120', '123.batch|COMPLETED|0:0|7000|8||825572K|',
                                                 '123.extern|COMPLETED|0:0|7000|8|||')),
        'int_job_id': dict(job_id=123),
        'none_job_id': dict(job_id=None),
    }
    for name, kw in cases.items():
        d = tmp_path / name
        d.mkdir()
        pre, audit, receipt = accounting_case(d, **kw)
        orig = run_accounting(C, monkeypatch, pre, audit, receipt)
        mut = run_accounting(m, monkeypatch, pre, audit, receipt)
        note(f'A2 {name}: original ->', orig, '| C4h ->', mut)
        assert orig.startswith('ValueError')


def test_A3_audit_source(tmp_path, monkeypatch):
    m = mutant(C, *C4D, 'cr_c4d')
    pre, audit, receipt = accounting_case(tmp_path, audit_source='h' * 64, accounting_source=S)
    orig = run_accounting(C, monkeypatch, pre, audit, receipt)
    mut = run_accounting(m, monkeypatch, pre, audit, receipt)
    note('A3 audit source != accounting/expected source: original ->', orig, '| C4d ->', mut)
    assert orig.startswith('ValueError') and mut.startswith('ACCEPTED')
    # Other gates in production callers: run.validate_audit (run.py:97-99) and readout 413-415 both call
    # validate_historical_evidence('audit', path) when the audit source differs.
    from contrastive_source_compat import validate_historical_evidence
    barrier = outcome(validate_historical_evidence, 'audit', audit)
    note('A3 historical identity barrier on a different-source audit (current clone sources):', barrier)
    assert not barrier.startswith('ACCEPTED')


# ---------------------------------------------------------------- numeric context
from test_contrastive_targets_readout import readout_fixture, numeric_folder  # noqa: E402


def tf32_on(flags):
    flags = dict(flags)
    flags['matmul_tf32'] = True
    return flags


def rewrite_all(kwargs, transform):
    n = 0
    for folder in [*kwargs['full_dirs'].values(), *kwargs['resumed_dirs'].values()]:
        for rank in range(4):
            path = Path(folder) / f'numeric_profile.rank{rank}.json'
            row = json.loads(path.read_text())
            row['flags'] = transform(row['flags'])
            dump(path, row)
            n += 1
    return n


def test_B1_coherent_nonconforming_28(tmp_path, monkeypatch):
    kwargs, *_ = readout_fixture(tmp_path, monkeypatch)
    assert rewrite_all(kwargs, tf32_on) == 28
    report = B.read_preflight(**kwargs)
    ctx = report['numeric_context_across_runs']
    note('B1 original, 28 coherent TF32-on receipts (readout fixture):', report['status'], ctx['status'],
         'missing', len(ctx['missing']), 'differences', len(ctx['differences']), 'arms', sorted(report['arms']))
    assert report['status'] == 'INCONCLUSIVE_REQUIRES_REVIEW' and len(ctx['missing']) == 28
    m = mutant(B, *C5C, 'rb_c5c')
    monkeypatch.setattr(B, 'numeric_context_across_runs', m.numeric_context_across_runs)
    report = B.read_preflight(**kwargs)
    note('B1 C5c mutant, same 28 receipts (fixture, _run_evidence patched):', report['status'],
         report['numeric_context_across_runs']['status'], 'arms', sorted(report['arms']))


def test_B2_real_run_evidence_is_second_barrier(tmp_path):
    cfg, provenance, cohort = numeric_folder(tmp_path)
    for rank in range(4):
        path = tmp_path / f'numeric_profile.rank{rank}.json'
        row = json.loads(path.read_text())
        row['flags'] = tf32_on(row['flags'])
        dump(path, row)
    orig = outcome(B._run_evidence, tmp_path, 'CN', cohort, cfg, provenance)
    note('B2 real _run_evidence with TF32-on receipts: original ->', orig[:160])
    assert orig.startswith('NumericProfileUnavailable')
    m = mutant(B, *X220, 'rb_x220')
    mut = outcome(m._run_evidence, tmp_path, 'CN', cohort, cfg, provenance)
    note('B2 X220 mutant (_run_evidence assert removed) ->', mut[:160])


def test_B3_rank_consistency_reachability(tmp_path):
    cfg, provenance, cohort = numeric_folder(tmp_path)
    keys = sorted(numeric_flags())
    note('B3 producer numeric_flags() keys:', len(keys), 'cudnn_sdp_enabled' in keys)
    for rank in range(4):
        path = tmp_path / f'numeric_profile.rank{rank}.json'
        row = json.loads(path.read_text())
        if rank == 0:
            row['flags']['cudnn_sdp_enabled'] = None
        else:
            row['flags'].pop('cudnn_sdp_enabled')
        dump(path, row)
    names = ('C0', 'CN', 'CR', 'CN_A', 'CN_B', 'CR_A', 'CR_B')
    ctx = B.numeric_context_across_runs({n: tmp_path for n in names[:3]}, {n: tmp_path for n in names[3:]})
    note('B3 crafted receipts (rank0 key=None, ranks1-3 key absent): numeric context ->', ctx['status'],
         'differences', ctx['differences'], 'missing', ctx['missing'])
    res = outcome(B._run_evidence, tmp_path, 'CN', cohort, cfg, provenance)
    note('B3 real _run_evidence on the same receipts ->', res[:160])


def cuda_report(flags_transform):
    from contrastive_cuda_check import PROTOCOL, CRITERIA, TECHNICAL_OVERRIDES
    calibration_cfg = {'contrastive': {'calibration_only': True, 'nce_weight': 1., 'rnc_weight': 1.}}
    canonical = hashlib.sha256(json.dumps(calibration_cfg, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    cohort = {'source_sha256': 's' * 64, 'initial_state_sha256': 'i' * 64}
    inputs = {k: k + '-pin' for k in ('manifest', 'scales', 'window_seconds', 'pose_stats')}
    keys = ('total', 'pose', 'contrast_nce', 'contrast_rnc', 'contrast_global_grad_base', 'contrast_global_grad_nce',
            'contrast_global_grad_rnc', 'gradient', 'student', 'teacher')
    precisions = {}
    for name in ('fp32', 'bf16'):
        limits = {k: (.01 if name == 'bf16' or k in ('student', 'teacher') else 1e-5) for k in keys}
        precisions[name] = {'passed': True, 'updates': [{'update': step, 'passed': True, 'errors': dict.fromkeys(keys, 0.), 'limits': limits}
                                                       for step in (1, 2)]}
    report = {'passed': True, 'source_sha256': cohort['source_sha256'], 'initial_sha256': cohort['initial_state_sha256'],
              'protocol': PROTOCOL, 'criteria': CRITERIA, 'technical_overrides': TECHNICAL_OVERRIDES, 'world_size': 4,
              'inputs_unchanged': True, 'inputs': {**inputs, 'config': 'x'},
              'pose_stats_sha256': inputs['pose_stats'], 'config_canonical_sha256': canonical,
              'ranks': [{'rank': rank, 'passed': True, 'stop_checks': {'coverage': True, 'numeric': True, 'passed': True},
                         'arms': {'CN': copy.deepcopy(precisions), 'CR': copy.deepcopy(precisions)}} for rank in range(4)]}
    with numeric_profile(require_cuda=False) as flags:
        for row in report['ranks']:
            row['numeric_profile'] = flags_transform(dict(flags))
    return report, cohort, inputs, calibration_cfg


def test_B4_cuda_profile(tmp_path):
    report, cohort, inputs, cfg = cuda_report(lambda f: f)
    path = dump(tmp_path / 'cuda.json', report)
    assert outcome(B._cuda_evidence, path, cohort, inputs, cfg).startswith('ACCEPTED')
    report, *_ = cuda_report(tf32_on)
    dump(path, report)
    orig = outcome(B._cuda_evidence, path, cohort, inputs, cfg)
    m = mutant(B, *C5M, 'rb_c5m')
    mut = outcome(m._cuda_evidence, path, cohort, inputs, cfg)
    note('B4 CUDA report, TF32-on numeric_profile on 4 ranks: original ->', orig[:140], '| C5m ->', mut)
    assert orig.startswith('NumericProfileUnavailable')
