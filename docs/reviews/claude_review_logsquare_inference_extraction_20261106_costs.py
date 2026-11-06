"""Independent recomputation of the T100-LQ inference benchmark reading (rule D1, frozen on 04/11/2026).

Input: the archived receipt of job 63614514. Nothing is imported from the project: the rule is rewritten here from
`docs/decisioni.md` and compared with what the receipt declares.

usage, from a checkout of GPT's branch at 63eba9a:
    python -I claude_review_logsquare_inference_extraction_20261106_costs.py \
        results/infra/logsquare_inference_20261105/completed/receipt.json
"""
import json
import math
import sys

receipt = json.load(open(sys.argv[1]))
ARMS = ('T100-L', 'T100-LQ')
WARMUP, ROUNDS = 2, 8
CELLS = {'pose_5000': 2700, 'pose_19618': 2700, 'p1_19618': 1800, 'p2_19618': 2700}
problems = []


def close(a, b, tol=1e-12):
    return math.isclose(a, b, rel_tol=tol, abs_tol=0.)


def check(condition, message):
    if not condition:
        problems.append(message)


# 1. Raw calls of every record: alternating order, two warm-up rounds, six measured rounds per arm.
per_group = {}
print('record  group         precision            calls  mean L (s)   mean LQ (s)  rho         min-max of per-round LQ/L')
for index, record in enumerate(receipt['records']):
    calls = record['calls']
    check(len(calls) == ROUNDS * 2, f'record {index}: {len(calls)} calls')
    measured = {arm: [] for arm in ARMS}
    by_round = {}
    for position, call in enumerate(calls):
        round_index = position // 2
        order = ARMS if round_index % 2 == 0 else ARMS[::-1]
        check(call['round'] == round_index, f'record {index}: round field at call {position}')
        check(call['arm'] == order[position % 2], f'record {index}: arm order at call {position}')
        check(call['warmup'] is (round_index < WARMUP), f'record {index}: warm-up flag at call {position}')
        check(math.isfinite(call['seconds']) and call['seconds'] > 0, f'record {index}: duration at call {position}')
        by_round.setdefault(round_index, {})[call['arm']] = call['seconds']
        if round_index >= WARMUP:
            measured[call['arm']].append(call['seconds'])
    means = {arm: sum(values) / len(values) for arm, values in measured.items()}
    check(all(len(values) == ROUNDS - WARMUP for values in measured.values()), f'record {index}: measured rounds')
    rho = means['T100-LQ'] / means['T100-L']
    for arm in ARMS:
        check(close(means[arm], record['mean_seconds'][arm]), f'record {index}: mean of {arm}')
    check(close(rho, record['rho_infer']), f'record {index}: rho')
    ratios = [by_round[r]['T100-LQ'] / by_round[r]['T100-L'] for r in range(WARMUP, ROUNDS)]
    per_group.setdefault(record['group'], []).append(means)
    print(f"{index:<7} {record['group']:<13} {record['precision']:<20} {len(calls):<6} {means['T100-L']:<12.6f} "
          f"{means['T100-LQ']:<12.6f} {rho:<11.6f} {min(ratios):.4f}-{max(ratios):.4f}")

# 2. Group ratios: sum of candidate means over sum of control means.
groups = {}
print('\ngroup         batches  rho recomputed   rho in receipt')
for group, rows in per_group.items():
    base = sum(row['T100-L'] for row in rows)
    candidate = sum(row['T100-LQ'] for row in rows)
    groups[group] = candidate / base
    declared = receipt['groups'][group]
    check(close(groups[group], declared['rho_infer']), f'group {group}: rho')
    check(declared['n_representative_batches'] == len(rows), f'group {group}: batches')
    check(close(base / len(rows), declared['control_mean_seconds']), f'group {group}: control mean')
    check(close(candidate / len(rows), declared['candidate_mean_seconds']), f'group {group}: candidate mean')
    print(f"{group:<13} {len(rows):<8} {groups[group]:<16.9f} {declared['rho_infer']:.9f}")
check(set(groups) == set(receipt['groups']), 'group set')

# 3. Cells: maximum ratio for the cap, window-weighted ratio for the estimate, cap rounded up to 300 s.
print('\ncell        cap rho     point rho   proxy (s)  cap (s)  GPU-h cap  core-h cap  historical elapsed (s)')
total_cap = total_proxy = 0.
for cell, walltime in CELLS.items():
    old = receipt['historical_jobs'][cell]
    counts = old['historical_window_counts']
    check(old['historical_walltime_seconds'] == walltime, f'{cell}: historical walltime')
    check(set(counts) <= set(groups), f'{cell}: groups without a measurement')
    cap_rho = max(groups[g] for g in counts)
    point_rho = sum(groups[g] * n for g, n in counts.items()) / sum(counts.values())
    cap = math.ceil(walltime * max(1., cap_rho) / 300) * 300
    proxy = old['historical_elapsed_seconds'] * point_rho
    declared = receipt['resource_proposal']['jobs'][cell]
    check(declared['walltime_seconds'] == cap, f'{cell}: cap')
    check(close(declared['rho_infer_cap'], cap_rho), f'{cell}: cap rho')
    check(close(declared['rho_infer_point'], point_rho), f'{cell}: point rho')
    check(close(declared['expected_seconds_proxy'], proxy), f'{cell}: proxy')
    check(close(declared['gpu_hours_cap'], cap / 3600), f'{cell}: GPU-h cap')
    check(close(declared['allocated_core_hours_cap'], 8 * cap / 3600), f'{cell}: core-h cap')
    check((declared['nodes'], declared['gpus'], declared['cpus'], declared['memory'], declared['partition'])
          == (1, 1, 8, '100G', 'boost_usr_prod'), f'{cell}: allocation')
    check(declared['historical_window_counts'] == counts, f'{cell}: weights')
    total_cap += cap
    total_proxy += proxy
    print(f"{cell:<11} {cap_rho:<11.6f} {point_rho:<11.6f} {proxy:<10.2f} {cap:<8} {cap / 3600:<10.6f} {8 * cap / 3600:<11.6f} "
          f"{old['historical_elapsed_seconds']:.2f}")
proposal = receipt['resource_proposal']
check(close(proposal['gpu_hours_cap'], total_cap / 3600), 'total GPU-h cap')
check(close(proposal['allocated_core_hours_cap'], 8 * total_cap / 3600), 'total core-h cap')
check(close(proposal['gpu_hours_point_proxy'], total_proxy / 3600), 'total proxy')
print(f"\ntotal cap: {total_cap:.0f} s = {total_cap / 3600:.6f} GPU-h, {8 * total_cap / 3600:.6f} core-h; "
      f"proxy {total_proxy / 3600:.6f} GPU-h")

# 4. Sensitivity of each cap: how far the ratio is from the next 300 s step, and what a ratio of exactly 1 would give.
print('\ncell        cap at rho=1  ratio that would add 300 s more')
for cell, walltime in CELLS.items():
    cap = receipt['resource_proposal']['jobs'][cell]['walltime_seconds']
    print(f"{cell:<11} {math.ceil(walltime / 300) * 300:<13} {cap / walltime:.6f}")

print('\nPROBLEMS:', problems if problems else 'none')
sys.exit(1 if problems else 0)
