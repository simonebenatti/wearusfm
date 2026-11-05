"""Training-rank entry for the fresh T100-LQ phase: the unchanged production trainer,
with the student it constructs verified before the trainer receives it.

torchrun starts this file instead of scripts/train_jepa.py. The trainer builds
its student with the module-level name run.WearUsFM (run.py, pinned by the
plan). Here that name verifies the new model against the reviewed guard and
returns the same object, unmodified. A rejected state raises before the trainer
receives the model, so before any teacher, optimizer, DDP wrapper or update
exists. One accepted state means that all ranks start bit-identical, and that
the buffers DDP later broadcasts from rank zero change nothing.

Rank zero first builds one more model, in run.count_parameters, before the
trainer sets the seed: unseeded weights, never trained, discarded. It is
recorded with its gain and not judged. Any other model is judged as a student.

--verify-only (diagnostic, never part of a training command): the constructor
raises once the student is verified, so no model reaches the trainer and nothing
is trained. Its record says so, and the controller refuses it as a training
attestation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAINER = 'scripts/train_jepa.py'


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


# The first non-standard import of the process is the production trainer, so the
# numeric libraries load in its order (Torch, then the package and NumPy). The
# calibrated gain depends on that order. Keep it here, at import time: a data
# worker that re-imports this file as its main module repeats the same order.
CLI = module('logsquare_rank_trainer', TRAINER)
G = module('logsquare_rank_initial_state', 'scripts/logsquare_initial_state.py')
R = CLI.R


class VerifiedOnly(Exception):
    """Verification-only mode: raised inside the constructor, after the student passed."""


def check_fresh(argv):
    """Only a fresh run whose student nothing touches between construction and the first update."""
    args = CLI.build_parser().parse_args(argv)
    cfg = CLI.build_config(args)
    G.require(args.init_from is None and args.branch_from is None and getattr(args, 'branch_state_from', None) is None,
              'guarded entry: no warm start or branch')
    G.require(not (args.out_dir / 'checkpoint.pt').exists(), 'guarded entry: fresh run only, checkpoint present')
    G.require(cfg.model.tokenizer_logsquare and not cfg.freeze_encoder and not cfg.model.keep_readout and
              not cfg.model.spectral_branches, 'guarded entry: the trainer would modify the student after construction')


def install(guard, guard_sha256, directory, rank, world_size, verify_only=False):
    """Verify every student built through run.WearUsFM; write this rank's record after each model."""
    original, original_count = R.WearUsFM, R.count_parameters
    G.require(isinstance(original, type), 'model constructor already replaced')
    G.require('WearUsFM' in R.train.__code__.co_names and 'WearUsFM' in original_count.__code__.co_names and
              'count_parameters' in CLI.main.__code__.co_names,
              'trainer no longer constructs its models through run.WearUsFM and run.count_parameters')
    G.require(Path(directory).is_dir(), 'record directory must exist')
    report = {'rank': rank, 'world_size': world_size, 'guard_sha256': guard_sha256, 'passed': True,
              'verify_only': verify_only, 'trainer_returned': False, 'constructions': []}
    path = Path(directory) / f'rank{rank}.json'
    counting = []

    def write():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)

    def count(cfg):
        counting.append(True)
        try:
            return original_count(cfg)
        finally:
            counting.pop()

    def construct(cfg):
        model = original(cfg)
        number = len(report['constructions']) + 1
        if counting:
            gain = model.state_dict()[G.GAIN].item()
            report['constructions'].append({'role': 'parameter_count', 'judged': False, 'fe_gain': gain,
                                            'fe_gain_hex': gain.hex()})
        else:
            context = f'training rank {rank} of {world_size}, model construction {number}, the student'
            try:
                report['constructions'].append({'role': 'student', **G.verify(model.state_dict(), guard, context)})
            except G.InitialStateError as error:
                report['passed'] = False
                report['constructions'].append({'role': 'student', **error.record})
                write()
                raise
        write()
        if verify_only and not counting:
            raise VerifiedOnly('student verified; verification-only mode stops before the trainer receives it')
        return model

    R.WearUsFM, R.count_parameters = construct, count
    return report, write, (original, original_count)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--initial-state-guard', type=Path, required=True)
    parser.add_argument('--initial-state-guard-sha256', required=True)
    parser.add_argument('--initial-state-records', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true',
                        help='diagnostic: stop inside the constructor once the student is verified; nothing is trained')
    parser.add_argument('trainer', nargs=argparse.REMAINDER, help=TRAINER + ' followed by its unchanged arguments')
    args = parser.parse_args(argv)
    G.require(args.trainer[:1] == [TRAINER], 'only the production trainer runs behind this entry')
    trainer_argv = args.trainer[1:]
    guard = G.load_guard(args.initial_state_guard, args.initial_state_guard_sha256)
    check_fresh(trainer_argv)
    ctx = R.dist_from_env()
    report, write, originals = install(guard, args.initial_state_guard_sha256, args.initial_state_records,
                                       ctx.rank, ctx.world_size, args.verify_only)
    try:
        status = CLI.main(trainer_argv)
    except VerifiedOnly:
        G.require(args.verify_only and report['passed'] and
                  [row['role'] for row in report['constructions']][-1:] == ['student'],
                  'verification-only stop without a verified student')
        return 0
    finally:
        R.WearUsFM, R.count_parameters = originals
    G.require(not args.verify_only, 'verification-only mode: the trainer returned without building a student')
    G.require(status == 0 and report['passed'] and
              [row['role'] for row in report['constructions']][-1:] == ['student'], 'trainer returned without a verified student')
    report['trainer_returned'] = True
    write()
    return status


if __name__ == '__main__':
    raise SystemExit(main())
