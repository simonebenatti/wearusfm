"""Independent recomputation of the diagnostic reading from the archived cluster files.

It reads raw fields and per-tensor hashes only. It does not import the project and does not call reading().
"""
import gzip
import hashlib
import json
import math
import os
import sys

root = sys.argv[1]  # checkout of GPT's branch
base = os.path.join(root, 'results/infra/logsquare_initialization_20261105')
done = os.path.join(base, 'completed')


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def gz(path):
    return json.load(gzip.open(path, 'rt'))


report = gz(os.path.join(done, 'report.json.gz'))
plan = json.load(open(os.path.join(base, 'plan.json')))
guard_file = json.load(open(os.path.join(base, 'initial_state_guard.json')))
variants = report['variants']
construction = ['preflight', 'controller', 'controller_threads1', 'trainer', 'trainer_rank0']
expected_hash = report['expected']['shared_sha256']
expected_gain = report['expected']['fe_gain']
out = {}

# 0. provenance
out['report_code_sha'] = report['code_sha']
out['report_plan_sha256_matches_plan_file'] = report['plan_sha256'] == sha(os.path.join(base, 'plan.json'))
out['plan_status'] = plan['status']
out['guard_file_sha256_matches_plan_pin'] = sha(os.path.join(base, 'initial_state_guard.json')) == plan['initial_state_guard']['sha256']
out['report_gz_sha256'] = sha(os.path.join(done, 'report.json.gz'))
out['report_original_sha256'] = hashlib.sha256(gzip.open(os.path.join(done, 'report.json.gz'), 'rb').read()).hexdigest()

# 1. every context returned zero and carries no error
errors = {}
for name, variant in variants.items():
    bad = [key for key in ('error', 'cleanup_error') if variant.get(key)]
    sup = [variant.get('supervision') or {}, variant.get('torchrun_supervision') or {}]
    signalled = any(s.get('timed_out') or s.get('terminated') or s.get('killed') or s.get('survivors')
                    or s.get('process_table_unavailable') for s in sup)
    errors[name] = {'returncode': variant.get('process_returncode'), 'error_keys': bad, 'signalled': signalled}
out['contexts'] = errors
out['all_contexts_clean'] = all(v['returncode'] == 0 and not v['error_keys'] and not v['signalled'] for v in errors.values())

# 2. tensors: ten constructions against the preflight T100-L arm
reference = variants['preflight']['arms']['T100-L']['per_tensor']
names = set(reference)
table = {}
differing = set()
for name in construction:
    for arm, data in variants[name]['arms'].items():
        tensors = data['per_tensor']
        shared = {key: value for key, value in tensors.items() if key in names}
        extra = sorted(set(tensors) - names)
        missing = sorted(names - set(tensors))
        diff = sorted(key for key in shared if shared[key] != reference[key])
        differing.update(diff)
        table[f'{name}/{arm}'] = {
            'tensors': len(tensors), 'shared_with_reference': len(shared), 'extra': extra, 'missing': missing,
            'different_from_preflight_T100-L': diff, 'gain_hex': data['fe_gain_hex'],
            'shared_sha256': data['shared_sha256'][:8],
            'pinned_gain_copy_sha256_is_reference': data['hypothetical_pinned_gain_shared_sha256'] == expected_hash,
            'cuda_preserved': data.get('cuda_transfer_preserves_exact_state'),
            'cpu_rng': data['cpu_rng_after_construction_sha256'][:8],
            'cuda_rng': (data.get('cuda_rng_after_transfer_sha256') or '')[:8],
            'logsquare_zero': data['logsquare_weight_zero'], 'logsquare_shape': data['logsquare_shape'],
            'full_before_equals_after': data['full_sha256_before'] == data['full_sha256_after'],
        }
out['constructions'] = table
out['tensor_names_in_reference'] = len(names)
out['all_tensors_that_ever_differ'] = sorted(differing)

# 3. the gains, from their hexadecimal form
pre = float.fromhex(variants['preflight']['arms']['T100-L']['fe_gain_hex'])
ctl = float.fromhex(variants['controller']['arms']['T100-L']['fe_gain_hex'])
out['gain_preflight'] = repr(pre)
out['gain_controller'] = repr(ctl)
out['preflight_gain_is_expected'] = pre == expected_gain
out['controller_is_next_double_below_preflight'] = math.nextafter(pre, -math.inf) == ctl
out['difference'] = repr(ctl - pre)
out['guard_file_gains'] = {key: value for key, value in guard_file.items() if 'gain' in key.lower()}

# 4. the three statements, from the raw fields
def arm_ok(name, arm, gain):
    data = variants[name]['arms'][arm]
    return (float.fromhex(data['fe_gain_hex']) == gain
            and data['hypothetical_pinned_gain_shared_sha256'] == expected_hash
            and table[f'{name}/{arm}']['different_from_preflight_T100-L'] in ([], ['tokenizer.fe_gain'])
            and not table[f'{name}/{arm}']['missing'])

preflight_ok = all(variants['preflight']['arms'][arm]['shared_sha256'] == expected_hash and arm_ok('preflight', arm, pre)
                   for arm in ('T100-L', 'T100-LQ'))
controller_ok = all(arm_ok('controller', arm, ctl) and variants['controller']['arms'][arm]['shared_sha256'] != expected_hash
                    for arm in ('T100-L', 'T100-LQ'))
trainer_ok = all(arm_ok(name, arm, ctl) for name in ('trainer', 'trainer_rank0') for arm in ('T100-L', 'T100-LQ'))
same_state = len({variants[name]['arms'][arm]['shared_sha256'] for name in ('controller', 'trainer', 'trainer_rank0')
                  for arm in ('T100-L', 'T100-LQ')}) == 1
guards = {name: (variants[name]['arms']['T100-LQ'].get('guard') or {}) for name in construction}
out['guard'] = {name: {'passed': g.get('passed'), 'classification': g.get('classification'), 'error': g.get('error')}
                for name, g in guards.items()}
guard_ok = all(guards[name].get('passed') is True for name in ('controller', 'trainer', 'trainer_rank0'))

rank = variants['rank_entry_torchrun']
out['rank_entry_keys'] = sorted(rank)
records = []
folder = os.path.join(done, 'rank_entry_torchrun_initial_state')
for index in range(4):
    record = json.load(open(os.path.join(folder, f'rank{index}.json')))
    records.append(record)
out['rank_record_keys'] = sorted(records[0])
out['rank_records'] = [{key: (value[:8] if isinstance(value, str) and len(value) == 64 else value)
                        for key, value in record.items() if not isinstance(value, (dict, list))} for record in records]
trainer_hash = variants['trainer_rank0']['arms']['T100-LQ']['shared_sha256']
out['trainer_rank0_shared_sha256'] = trainer_hash
out['statements'] = {
    'cause_demonstrated': preflight_ok and controller_ok,
    'trainer_builds_the_registered_training_state': trainer_ok and same_state,
    'guard_accepts_controller_and_trainers': guard_ok,
}
out['paired'] = {name: {key: variants[name].get(key) for key in ('paired_shared_state_exact', 'paired_cpu_rng_exact', 'paired_cuda_rng_exact')}
                 for name in construction}
json.dump(out, sys.stdout, indent=1)
