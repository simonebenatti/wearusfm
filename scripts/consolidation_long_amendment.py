"""Resource amendment A1 of the long evaluation: fit packaging, walltimes, input bridge.

Post hoc with respect to the sixteen train-only benchmark timings; fixed before
any fit, score or prediction of the long comparison. It changes job packaging,
resource admission, input provenance and phase release only. The exception to
R-2(i) is limited to the 28 original receipts of the pinned input-compatibility
manifest: no other source, receipt or input is admitted, and nothing is rewritten.
"""
from __future__ import annotations

import ast
import hashlib
import math

AMENDMENT = 'A1'
ORIGINAL_SOURCE = '1ca7aead5cd48a1fd9ff398338d80f2471a8d4041ccc0ac7fcdb3c3c03d0b1e1'
ORIGINAL_RECIPE = 'b2356a8e09fd7bc560a2a677c2c93f8446afcff96254c91454452c937f865422'
ORIGINAL_PHASES = {
    'extract': {'code_sha': '7d276c98fc54d66dab1226226eb6c5e447422c69',
                'plan_sha256': 'e24d3fae8642695dcf0dd1ff45347d985eed0c8e47f7e85c1c1e138db11d058a'},
    'benchmark': {'code_sha': '166fb8653b53439507183d76e3e36a79dd28d166',
                  'plan_sha256': '22975a40ba100cf8df49b18a442c5b0a755d7c8dfd4eeb5e088651b5d3c6615f'},
}
# The closed list of inputs is part of the frozen source: another list is another source.
INPUT_MANIFEST = {'path': 'results/infra/consolidation_long_20261105/amendment_a1/input_compatibility.json',
                  'sha256': '983df48465531014275b6f5f677e6907a1219f26fbeef529b6489ea85c473c95', 'bytes': 65919}
INPUT_KINDS = {'extract-classification': 6, 'extract-pose-train': 3, 'extract-pose-val': 3,
               'benchmark-classification': 15, 'benchmark-pose': 1}
CLOSED_STAGES = ('extract-classification', 'extract-pose-train', 'extract-pose-val',
                 'benchmark-classification', 'benchmark-pose')

COMMON = 'scripts/consolidation_long_common.py'
CLASSIFICATION = 'scripts/consolidation_long_classification_runtime.py'
POSE = 'scripts/consolidation_long_pose_runtime.py'
LAUNCHER = 'scripts/slurm/consolidation_long.sbatch'
CHANGED_PATHS = (COMMON, CLASSIFICATION, POSE, LAUNCHER)
ADDED_PATHS = ('scripts/consolidation_long_amendment.py',)
AMENDED_DEFINITIONS = {
    COMMON: ('validate_plan', 'load_context', 'begin_output', 'verified_extraction'),
    CLASSIFICATION: ('_benchmarks', 'execute'),
    POSE: ('_manifest', '_read', '_benchmark_gate', 'execute'),
}
ADDED_DEFINITIONS = {
    COMMON: ('import consolidation_long_amendment as A', 'verified_input'),
    CLASSIFICATION: ('amended_gate',),
    POSE: (),
}
# The order of the top-level statements is frozen too: each added one directly follows this original one.
ADDED_AFTER = {
    COMMON: {'import consolidation_long_amendment as A': 'from wearusfm.training.pairing import state_sha256',
             'verified_input': 'finish_output'},
    CLASSIFICATION: {'amended_gate': 'resource_gate'},
    POSE: {},
}
END_OF_FILE = '<end of file>'
# Closed stages keep their original text inside the amended dispatcher.
FROZEN_BLOCKS = {CLASSIFICATION: {'execute': ("stage == 'extract-classification'",
                                              "stage == 'benchmark-classification'")}}
LAUNCHER_REPLACED_LINES = (52, 53)

SINGLES, PAIRS, WHOLE = [[0], [1], [2], [3]], [[0, 1], [2, 3]], [[0, 1, 2, 3]]
FIT_POLICY = {
    'protocol': 'consolidation-classification-resources-20261106-a1',
    'amends': 'consolidation-classification-resources-20261105-v1',
    'status': 'post hoc resource amendment: fixed after the sixteen train-only benchmark timings, '
              'before any fit, score or prediction of the long comparison; not a pre-registered rule',
    'projection': 'unchanged per-fold projection of the original policy on the same fifteen receipts, '
                  'plus the original fixed seconds and twice load and scale for every job',
    'fold_shards': {
        'ninapro_db2': {'pre': SINGLES, 'post': PAIRS, 'pool': SINGLES},
        'ninapro_db3': {'pre': SINGLES, 'post': WHOLE, 'pool': WHOLE},
        'ninapro_db6': {'pre': WHOLE, 'post': WHOLE, 'pool': PAIRS},
        'epn612': {'pre': WHOLE, 'post': WHOLE, 'pool': WHOLE},
        'uci_emg': {'pre': WHOLE, 'post': WHOLE, 'pool': WHOLE},
    },
    'walltime_rule': 'W = 900 * ceil((P + 600) / 900); admitted when P <= W - 600 and W <= 14400',
    'walltime_quantum_seconds': 900, 'walltime_reserve_seconds': 600.0, 'max_walltime_seconds': 14400,
    'memory': 'unchanged original admission: three times the measured peak within 85% of 30 GiB',
    'cpus': 8, 'memory_gib': 30, 'new_fit_cells': 40, 'physical_fit_jobs': 69,
    'limitation': 'empirical planning estimate, not a runtime guarantee; a timeout stops the phase; '
                  'no automatic shard, retry, walltime or budget extension',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def fit_shapes(dataset, view):
    """Registered fold shards of one dataset and view, as a fresh list."""
    return [list(folds) for folds in FIT_POLICY['fold_shards'][dataset][view]]


def registered_shards(doses, views, datasets, reference=(19618, 'pool')):
    """Every physical fit: each new logical cell, each of its four folds exactly once."""
    return [(step, view, dataset, folds) for step in doses for view in views for dataset in datasets
            if (step, view) != reference for folds in fit_shapes(dataset, view)]


def walltime_seconds(projection):
    """Requested limit of one shard: its projection plus the reserve, rounded up to the quantum."""
    quantum = FIT_POLICY['walltime_quantum_seconds']
    return quantum * math.ceil((projection + FIT_POLICY['walltime_reserve_seconds']) / quantum)


def record():
    """What every report written under the amendment states about its inputs."""
    return {'id': AMENDMENT, 'input_compatibility': dict(INPUT_MANIFEST), 'input_source_sha256': ORIGINAL_SOURCE}


def validate_block(block, shards, *, require_decision=True):
    """Check the amendment section of a plan; ``shards`` maps each registered key to its shard."""
    require(block['id'] == AMENDMENT and block['fit_policy'] == FIT_POLICY
            and block['closed_stages'] == list(CLOSED_STAGES)
            and block['input_compatibility'] == INPUT_MANIFEST, 'amendment A1 policy or inputs differ')
    table = block['fit_shards']
    require(set(table) == set(shards) and len(shards) == FIT_POLICY['physical_fit_jobs'],
            'amended fit shard matrix differs')
    for key, (step, view, dataset, folds) in shards.items():
        row = table[key]
        require(row['step'] == step and row['view'] == view and row['dataset'] == dataset
                and row['folds'] == folds and type(row['walltime_seconds']) is int
                and row['projected_seconds'] > 0
                and row['walltime_seconds'] == walltime_seconds(row['projected_seconds'])
                and row['walltime_seconds'] <= FIT_POLICY['max_walltime_seconds'],
                'amended fit shard walltime differs: ' + key)
    if require_decision:
        decision = block['decision']
        require(isinstance(decision, dict) and decision.get('by') == 'Simone'
                and isinstance(decision.get('date'), str) and len(decision['date']) == 10
                and isinstance(decision.get('quote'), str) and bool(decision['quote'].strip()),
                'amendment A1 lacks the recorded decision of Simone')
    return block


def source_files(root, sha256_file):
    """The files covered by the source identity, with the same enumeration."""
    paths = sorted((root / 'src').rglob('*.py')) + sorted((root / 'scripts').glob('*.py'))
    paths += [root / LAUNCHER, root / 'scripts/slurm/lib_code_dir.sh']
    return {str(p.relative_to(root)): sha256_file(p) for p in paths}


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def regions(text):
    """Top-level statements of a module in file order, keyed by what they define, with their text.

    A statement owns its own lines and every line since the previous statement
    (decorators, comments, blank lines); what follows the last statement is
    keyed END_OF_FILE. The regions partition the file: joined by newlines, with
    the last one appended, they give back the text byte for byte.
    """
    lines, found, previous = text.split('\n'), {}, 0
    for index, node in enumerate(ast.parse(text).body):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            key = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            key = ast.unparse(node.targets[0])
        elif index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            key = 'docstring'
        else:
            key = ast.unparse(node)
        first = min([node.lineno] + [decorator.lineno for decorator in getattr(node, 'decorator_list', [])])
        require(key not in found and key != END_OF_FILE, 'duplicate top-level definition: ' + key)
        require(first > previous, 'top-level statements share a line: ' + key)
        found[key] = '\n'.join(lines[previous:node.end_lineno])
        previous = node.end_lineno
    found[END_OF_FILE] = '\n'.join(['', *lines[previous:]]) if previous else text
    return found


def definitions(text):
    """SHA-256 of every region of a module, in file order."""
    return {key: _sha(region) for key, region in regions(text).items()}


def blocks(text, spec):
    """SHA-256 of the named ``if`` blocks directly inside top-level functions."""
    tree, found = ast.parse(text), {}
    for name, tests in spec.items():
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
        require(len(functions) == 1, 'frozen block function missing: ' + name)
        found[name] = {}
        for test in tests:
            matches = [n for n in functions[0].body if isinstance(n, ast.If) and ast.unparse(n.test) == test]
            require(len(matches) == 1 and not matches[0].orelse, 'frozen block missing or ambiguous: ' + test)
            found[name][test] = _sha(ast.get_source_segment(text, matches[0]))
    return found


def launcher_frame(text, prefix_lines, suffix_lines):
    """SHA-256 of the launcher lines before and after its replaced allocation lines."""
    lines = text.split('\n')
    require(len(lines) >= prefix_lines + suffix_lines, 'launcher is shorter than its frozen frame')
    return {'prefix_sha256': _sha('\n'.join(lines[:prefix_lines])),
            'suffix_sha256': _sha('\n'.join(lines[-suffix_lines:]))}


def validate_inputs(inputs, files, texts, digest):
    """Fail closed unless this checkout differs from the original source only inside the closed list.

    ``files`` is the current source map, ``texts`` the current text of every
    changed path, ``digest`` the identity function of the evaluation. In the
    changed Python files every region outside the amended and added definitions
    keeps its original bytes and the statements keep their original order; in
    the launcher only the replaced allocation lines differ.
    """
    require(inputs.get('amendment') == AMENDMENT and inputs.get('kind') == 'input-compatibility'
            and inputs['source_sha256'] == ORIGINAL_SOURCE == digest(inputs['source_files'])
            and inputs['recipe_sha256'] == ORIGINAL_RECIPE and inputs['phases'] == ORIGINAL_PHASES,
            'input-compatibility manifest identity differs')
    original = inputs['source_files']
    changed = {path for path in files if path in original and files[path] != original[path]}
    require(changed == set(CHANGED_PATHS) and set(files) - set(original) == set(ADDED_PATHS)
            and not set(original) - set(files) and digest(files) != ORIGINAL_SOURCE,
            'covered sources differ from the original outside the closed amendment list')
    for path, amended in AMENDED_DEFINITIONS.items():
        old, order, new = inputs['definitions'][path], inputs['order'][path], definitions(texts[path])
        added, after = ADDED_DEFINITIONS[path], ADDED_AFTER[path]
        require(len(order) == len(old) == len(set(order)) and set(order) == set(old) and order[-1] == END_OF_FILE
                and set(after) == set(added) and len(added) == len(set(added)) and not set(old) & set(added)
                and set(after.values()) <= set(order[:-1]) and set(amended) <= set(order[:-1]),
                'closed list of definitions is inconsistent: ' + path)
        expected = [key for name in order for key in (name, *(a for a in added if after[a] == name))]
        require(list(new) == expected, 'top-level statements differ from the closed list or its order: ' + path)
        require(all(new[key] == old[key] for key in order if key not in amended),
                'frozen definition changed: ' + path)
    for path, spec in FROZEN_BLOCKS.items():
        require(blocks(texts[path], spec) == inputs['blocks'][path], 'frozen stage block changed: ' + path)
    frame = inputs['launcher']
    require(frame['path'] == LAUNCHER and launcher_frame(texts[LAUNCHER], frame['prefix_lines'], frame['suffix_lines'])
            == {key: frame[key] for key in ('prefix_sha256', 'suffix_sha256')},
            'launcher changed outside its allocation lines')
    receipts, kinds = inputs['receipts'], {}
    for receipt in receipts:
        kinds[receipt['kind']] = kinds.get(receipt['kind'], 0) + 1
    require(kinds == INPUT_KINDS and len(receipts) == sum(INPUT_KINDS.values())
            and len({r['id'] for r in receipts}) == len({r['path'] for r in receipts}) == len(receipts)
            and all(r['phase'] in ORIGINAL_PHASES and r['kind'].startswith(r['phase'] + '-') for r in receipts),
            'amendment input list differs')
    return inputs
