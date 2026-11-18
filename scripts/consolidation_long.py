"""Single entry point for the externally reviewed long-evaluation stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import consolidation_long_common as C

STAGES = ('extract-classification', 'extract-pose-train', 'extract-pose-val',
          'benchmark-classification', 'benchmark-pose', 'fit-classification', 'fit-pose',
          'compare-classification', 'compare-pose', 'summary')


def execute(args):
    ctx = C.load_context(args.plan, args.expected_code)
    if args.stage == 'summary':
        result = C.begin_output(ctx, args.out, 'summary')
        reports = {k: C.read_pin(pin) for k, pin in ctx['plan']['final_reports'].items()}
        C.require(set(reports) == {'classification', 'pose'}, 'both task reports required')
        expected_kind = {'classification': 'compare-classification', 'pose': 'compare-pose'}
        for name, value in reports.items():
            C.require(value['complete'] is True and value['source_sha256'] == ctx['source_sha256']
                      and value['recipe_sha256'] == ctx['plan']['recipe_sha256']
                      and value['protocol'] == C.PROTOCOL and value['kind'] == expected_kind[name], 'mixed final reports')
            for pin in value['artifacts'].values():
                C.pinned(pin)
        classification_result = C.read_pin(reports['classification']['artifacts']['comparison'])
        C.require(classification_result['complete'] is True and classification_result['kind'] == 'comparison',
                  'classification final artifact is incomplete')
        result.update(reports=reports, inputs=ctx['plan']['final_reports'],
                      classification_result=classification_result,
                      historical_replica=C.read_pin(ctx['plan']['historical_context']['replica']),
                      historical_replica_pin=ctx['plan']['historical_context']['replica'],
                      interpretation='development, one pretraining seed, all prespecified contrasts; separate task references')
        return C.finish_output(ctx, args.out, result)
    if 'classification' in args.stage:
        import consolidation_long_classification_runtime as R
        return R.execute(args.stage, ctx, args.out, step=args.step, task=args.task,
                         dataset=args.dataset, view=args.view, folds=args.folds)
    import consolidation_long_pose_runtime as R
    return R.execute(args.stage, ctx, args.out, step=args.step)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--expected-code', required=True)
    p.add_argument('--stage', choices=STAGES, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--step', type=int, choices=C.DOSES)
    p.add_argument('--task', choices=('p1', 'p2'))
    p.add_argument('--dataset', choices=('ninapro_db2', 'ninapro_db3', 'ninapro_db6', 'epn612', 'uci_emg'))
    p.add_argument('--view', choices=('pre', 'post', 'pool'))
    p.add_argument('--folds', type=int, nargs='+', choices=range(4))
    return p


if __name__ == '__main__':
    execute(parser().parse_args())
