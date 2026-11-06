"""Exact initial-state guard for T100-LQ; shared by controller, training ranks and diagnostic.

The reviewed reference hash describes the model built by the preflight. Its
calibrated gain buffer depends on the order in which the numeric libraries were
imported: the preflights import NumPy before Torch, every trainer and controller
Torch before NumPy, and the two contexts differ by one ULP of that one buffer.
The historical T100-L training therefore never started from the preflight state.

The guard keeps the reviewed hash as the exact reference for every other tensor
(the observed state hashed with only the gain replaced on a copy) and requires
the gain bit-equal to the buffer saved by the historical T100-L training. One
state passes. Nothing is restored, tolerated, or adopted from the observed
process; no new reference hash exists.

Only the standard library is imported here: the caller's numeric import order
is part of what is being verified.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
import struct

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = 'tokenizer-logsquare-initial-state-guard-20261105-v1'
GAIN = 'tokenizer.fe_gain'
NEW_WEIGHT = 'tokenizer.logsquare_weight'
FRONTEND = 'tokenizer.frontend.'
PASSED = 'historical_training_state'
PREFLIGHT_STATE = 'preflight_reference_state'
UNREGISTERED_GAIN = 'reference_tensors_with_unregistered_gain'
OTHER_TENSORS = 'tensor_difference_beyond_gain'


class InitialStateError(ValueError):
    """Rejected initial state; `record` carries every observation."""

    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


def require(ok, message):
    if not ok:
        raise ValueError(message)


def bits(value):
    """The float64 as its integer bit pattern: equality and ULP distance are exact."""
    require(type(value) is float and math.isfinite(value) and value > 0, 'finite positive float64 gain required')
    return struct.unpack('<q', struct.pack('<d', value))[0]


def read_pin(pin, root=REPO):
    path = Path(pin['path'])
    path = path if path.is_absolute() else root / path
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == pin['sha256'], 'SHA mismatch: ' + str(path))
    require('bytes' not in pin or len(data) == pin['bytes'], 'size mismatch: ' + str(path))
    return json.loads(gzip.decompress(data) if path.suffix == '.gz' else data)


def same_pin(first, second):
    return first['path'] == second['path'] and first['sha256'] == second['sha256']


def validate_guard(guard):
    """Structure and internal consistency only; `validate_provenance` reads the sources."""
    require(isinstance(guard, dict) and guard.get('protocol') == PROTOCOL and guard.get('arm') == 'T100-LQ',
            'wrong initial-state guard')
    new = guard.get('new_parameter')
    require(guard.get('gain_buffer') == GAIN and isinstance(new, dict) and set(new) == {'name', 'shape'} and
            new['name'] == NEW_WEIGHT and isinstance(new['shape'], list) and len(new['shape']) == 2 and
            type(new['shape'][0]) is int and new['shape'][0] > 0 and new['shape'][1] == 64,
            'guard names another buffer or branch')
    reference, training, groups = (guard.get(key) for key in ('reference', 'training_gain', 'group_references'))
    require(all(isinstance(value, dict) for value in (reference, training, groups)), 'incomplete guard')
    for name, row in (('reference', reference), ('training', training)):
        require(type(row.get('fe_gain')) is float and float.fromhex(row.get('fe_gain_hex', '')) == row['fe_gain'] and
                bits(row['fe_gain']) > 0, 'decimal and hexadecimal gain differ: ' + name)
    for digest in (reference.get('shared_sha256'), groups.get('outside_frontend_and_gain_sha256'),
                   groups.get('frontend_and_gain_sha256')):
        require(isinstance(digest, str) and len(digest) == 64 and set(digest) <= set('0123456789abcdef'), 'invalid reference hash')
    require(training.get('arm') == 'T100-L' and isinstance(training.get('sources'), list) and
            [source.get('checkpoint_step') for source in training['sources']] == [5000, 19618],
            'training gain needs both historical T100-L endpoints')
    return guard


def load_guard(path, sha256):
    """The pinned guard file; a file with another content never guards a run."""
    data = Path(path).read_bytes()
    require(hashlib.sha256(data).hexdigest() == sha256, 'initial-state guard differs from its pin')
    return validate_guard(json.loads(data))


def validate_provenance(guard, root=REPO):
    """Re-read every registered value from its pinned historical record."""
    reference, training, groups = guard['reference'], guard['training_gain'], guard['group_references']
    report = read_pin(reference['source'], root)
    initials = [report['arms'][arm]['initial'] for arm in ('T100-L', 'T100-LQ')]
    require(report.get('complete') is True and
            all(row['shared_sha256'] == reference['shared_sha256'] and row['new_weight_zero'] is True and
                bits(row['fe_gain']) == bits(reference['fe_gain']) for row in initials),
            'guard reference differs from the pinned preflight report')
    for source in training['sources']:
        summary = read_pin(source, root)['receipt_summary']
        saved = summary['checkpoint_guard']
        require(summary.get('arm') == 'T100-L' and summary.get('complete') is True and
                summary.get('checkpoint_step') == saved.get('checkpoint_step') == source['checkpoint_step'] and
                summary.get('checkpoint_sha256') == saved.get('checkpoint_sha256') == source['checkpoint_sha256'] and
                saved.get('training_code_sha') == source['training_code_sha'] and
                bits(saved['fe_gain']) == bits(training['fe_gain']),
                'training gain differs from the pinned historical T100-L checkpoint record: ' + source['path'])
    historical = read_pin(groups['source'], root)['arms']['T100-L']['initial']
    require(historical['all_sha256'] == reference['shared_sha256'] and
            bits(historical['fe_gain']) == bits(reference['fe_gain']) and
            historical['shared_sha256'] == groups['outside_frontend_and_gain_sha256'] and
            historical['frontend_and_gain_sha256'] == groups['frontend_and_gain_sha256'],
            'group references differ from the pinned historical preflight report')
    return guard


def gain_row(observed, registered):
    return {'fe_gain': registered, 'fe_gain_hex': registered.hex(), 'observed_minus_registered_ulps': bits(observed) - bits(registered)}


def inspect(state, guard):
    """Hash the state and one hypothetical copy with the reference gain; never mutate it."""
    import torch  # before the hashing module, which imports NumPy first
    from wearusfm.training.pairing import state_sha256

    require(NEW_WEIGHT in state and GAIN in state, 'state without log-square weight or calibrated gain')
    new, gain = state[NEW_WEIGHT], state[GAIN]
    require(list(new.shape) == guard['new_parameter']['shape'], 'wrong log-square weight shape')
    require(gain.shape == torch.Size([]) and gain.dtype == torch.float64 and bool(torch.isfinite(gain)) and gain.item() > 0,
            'invalid calibrated gain')
    reference, training = guard['reference'], guard['training_gain']
    observed = gain.item()
    shared = {key: value for key, value in state.items() if key != NEW_WEIGHT}
    substituted = dict(shared)
    substituted[GAIN] = torch.tensor(reference['fe_gain'], dtype=torch.float64)
    record = {'shared_sha256': state_sha256(shared), 'reference_gain_shared_sha256': state_sha256(substituted),
              'expected_shared_sha256': reference['shared_sha256'], 'fe_gain': observed, 'fe_gain_hex': observed.hex(),
              'reference_gain': gain_row(observed, reference['fe_gain']),
              'historical_training_gain': gain_row(observed, training['fe_gain']),
              'new_parameters': new.numel(), 'logsquare_weight_zero': not bool(new.count_nonzero()), 'tensors': len(shared)}
    others = record['reference_gain_shared_sha256'] == reference['shared_sha256']
    record['other_tensors_match_reference'] = others
    record['classification'] = (OTHER_TENSORS if not others else
                                PASSED if bits(observed) == bits(training['fe_gain']) else
                                PREFLIGHT_STATE if bits(observed) == bits(reference['fe_gain']) else UNREGISTERED_GAIN)
    record['passed'] = record['classification'] == PASSED and record['logsquare_weight_zero']
    if record['classification'] == PREFLIGHT_STATE:  # the two hashes of one state must agree
        require(record['shared_sha256'] == reference['shared_sha256'], 'inconsistent hashing of the reference state')
    return record


def localize(state, guard):
    """Failure path only: which registered group differs, and every observed tensor hash."""
    import torch
    from wearusfm.training.pairing import state_sha256

    groups = guard['group_references']
    shared = {key: value for key, value in state.items() if key != NEW_WEIGHT}
    outside = state_sha256({key: value for key, value in shared.items() if not key.startswith(FRONTEND) and key != GAIN})
    frontend = {key: value for key, value in shared.items() if key.startswith(FRONTEND)}
    frontend[GAIN] = torch.tensor(guard['reference']['fe_gain'], dtype=torch.float64)
    frontend = state_sha256(frontend)
    return {'outside_frontend_and_gain': {'sha256': outside, 'matches_reference': outside == groups['outside_frontend_and_gain_sha256']},
            'frontend_with_reference_gain': {'sha256': frontend, 'matches_reference': frontend == groups['frontend_and_gain_sha256']},
            'per_tensor': {key: {'sha256': state_sha256(value), 'shape': list(value.shape), 'dtype': str(value.dtype)}
                           for key, value in shared.items()}}


def failure_message(record, guard):
    reference, training = guard['reference'], guard['training_gain']
    match = lambda ok: 'MATCH' if ok else 'DIFFER'  # noqa: E731
    lines = [
        'fresh shared initialization differs from historical T100-L: ' + record['classification'] + ' (' + record['context'] + ')',
        '  expected SHA-256, state without the new weight, preflight gain: ' + reference['shared_sha256'],
        '  observed SHA-256: ' + record['shared_sha256'],
        '  observed SHA-256 with ONLY the gain replaced by the preflight value: ' + record['reference_gain_shared_sha256'] +
        ' -> every other tensor ' + match(record['other_tensors_match_reference']),
        f"  observed gain {record['fe_gain']!r} ({record['fe_gain_hex']})",
        f"  historical T100-L training gain {training['fe_gain']!r} ({training['fe_gain_hex']}): observed is "
        f"{record['historical_training_gain']['observed_minus_registered_ulps']:+d} ULP",
        f"  preflight reference gain {reference['fe_gain']!r} ({reference['fe_gain_hex']}): observed is "
        f"{record['reference_gain']['observed_minus_registered_ulps']:+d} ULP",
        '  new log-square weight zero: ' + str(record['logsquare_weight_zero']),
    ]
    if 'localization' in record:
        local = record['localization']
        lines += ['  tensors outside front-end and gain: ' + match(local['outside_frontend_and_gain']['matches_reference']) +
                  ' (observed ' + local['outside_frontend_and_gain']['sha256'] + ')',
                  '  front-end tensors, gain set to the preflight value: ' +
                  match(local['frontend_with_reference_gain']['matches_reference']) +
                  ' (observed ' + local['frontend_with_reference_gain']['sha256'] + ')',
                  f"  per-tensor SHA-256 of the {len(local['per_tensor'])} observed tensors: in the JSON record"]
    lines.append({
        OTHER_TENSORS: '  reading: a tensor other than the gain differs from the reviewed state. Not the import-order effect; stop.',
        PREFLIGHT_STATE: '  reading: exactly the preflight state. The historical T100-L training used the other gain; this '
                         'process is not in the trainer numeric context. Stop.',
        UNREGISTERED_GAIN: '  reading: only the gain differs, and it is neither registered value. New numeric context; stop.',
        PASSED: '  reading: registered state, but the new log-square weight is not zero. Stop.'}[record['classification']])
    return '\n'.join(lines)


def verify(state, guard, context):
    """Return the record of the single registered state, or raise with every observation."""
    record = {'context': context, **inspect(state, guard)}
    if not record['passed']:
        record['localization'] = localize(state, guard)
        raise InitialStateError(failure_message(record, guard), record)
    return record
