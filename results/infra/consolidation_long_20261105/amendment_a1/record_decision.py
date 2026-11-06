#!/usr/bin/env python3
"""Record the decision of Simone on amendment A1 in plan.json and launch_plan.json.

The runtime refuses every stage until the plan carries this record. The words
are quoted literally from the chat; nothing else changes and an existing record
is never replaced. No cluster access, no submission. From the repository root:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:scripts \
      python results/infra/consolidation_long_20261105/amendment_a1/record_decision.py \
      --date 2026-11-06 --question '<question asked>' --quote '<literal answer>'
"""
import argparse
import json

import consolidation_long_common as C

A = C.A
CAMPAIGN = C.ROOT / 'results/infra/consolidation_long_20261105'
APPROVED = 'approved_by_simone'
SCOPE = ('explicit exception to R-2(i) limited to the 28 original receipts of the pinned input-compatibility '
         'manifest; amended chain of 69 classification fits, 3 pose fits, 2 comparisons, 1 summary and 2 checkouts '
         'within the unchanged caps of 9 GPU-hours and 1711.666667 core-hours; no other experiment, input or retry')


def need(ok, message):
    if not ok:
        raise SystemExit('record_decision: ' + message)


def record(plan, launch, date, question, quote):
    """Return the two plans with the decision recorded; refuses to replace an existing record."""
    need(isinstance(date, str) and len(date) == 10 and date[4] == date[7] == '-' and date.replace('-', '').isdigit()
         and isinstance(quote, str) and quote.strip() and isinstance(question, str) and question.strip(),
         'date as YYYY-MM-DD, non-empty question and literal quote required')
    plan, launch = json.loads(json.dumps(plan)), json.loads(json.dumps(launch))
    blocks = [plan['amendment'], plan['phase_release']['source_change_policy']['amendment_A1'],
              launch['phase_source_policy']['amendment_A1']]
    need(all(block['decision'] is None for block in blocks), 'a decision is already recorded')
    need(plan['source_sha256'] == C.source_identity() == launch['source_sha256']
         and plan['recipe_sha256'] == C.recipe_identity(plan) == launch['recipe_sha256'] == A.ORIGINAL_RECIPE,
         'the plans do not describe this checkout')
    decision = {'by': 'Simone', 'date': date, 'question': question, 'quote': quote, 'scope': SCOPE}
    for block in blocks:
        block['decision'] = decision
    plan['amendment']['status'] = plan['authorization']['amendment_A1'] = launch['amendment_A1']['status'] = APPROVED
    need(plan['phase_release']['source_change_policy'] == launch['phase_source_policy'], 'the plans disagree')
    C.validate_plan(plan)
    return plan, launch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', required=True, help='YYYY-MM-DD')
    parser.add_argument('--question', required=True)
    parser.add_argument('--quote', required=True)
    args = parser.parse_args()
    paths = [CAMPAIGN / 'plan.json', CAMPAIGN / 'launch_plan.json']
    values = record(*(json.loads(path.read_text()) for path in paths), args.date, args.question, args.quote)
    for path, value in zip(paths, values):
        path.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(values[0]['amendment']['decision'], indent=1, ensure_ascii=False))
