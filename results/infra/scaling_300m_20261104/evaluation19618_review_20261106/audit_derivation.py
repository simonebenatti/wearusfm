"""Independent read-only audit of the delivered 19,618 derivation and scientific policy."""
import ast
import copy
import hashlib
import importlib.util
import itertools
import json
import subprocess
from pathlib import Path

ROOT = Path('/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-scaling-exec')
RELEASE = 'f6dc8ec0fb33ae1bcf6d34018194c2e9c71f7cb5'
FROZEN = '97125bd8bb8b14590958501ba91986c8456037f9'
BASE = 'results/infra/scaling_300m_20261104/'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

def body_functions(data):
    tree = ast.parse(data)
    # Ignore only the first docstring of each function for algorithm comparisons.
    result = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node = copy.deepcopy(node)
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
            result[node.name] = ast.dump(node, include_attributes=False)
    return result

assert git('rev-parse', 'HEAD').decode().strip() == RELEASE
assert git('status', '--porcelain') == b''
D = module('scaling_derive19618')
rows = []
for target, (source, expected, edits) in D.DERIVATIONS.items():
    raw = git('show', FROZEN+':'+source)
    assert raw == (ROOT/source).read_bytes()
    assert sha(raw) == expected
    text = raw.decode()
    for edit in edits:
        old, new, count = edit if len(edit) == 3 else (*edit, 1)
        assert text.count(old) == count and old and old != new
        text = text.replace(old, new)
    output = (ROOT/target).read_bytes()
    assert text.encode() == output == git('show', RELEASE+':'+target)
    item = {'source': source, 'target': target, 'source_sha256': expected,
            'target_sha256': sha(output), 'replacement_instructions': len(edits),
            'replacement_occurrences': sum(edit[2] if len(edit) == 3 else 1 for edit in edits)}
    if target.startswith('scripts/') and target.endswith('.py'):
        a, b = body_functions(raw), body_functions(output)
        item['identical_function_ast'] = sorted(k for k in a.keys() & b.keys() if a[k] == b[k])
        item['changed_function_ast'] = sorted(k for k in a.keys() & b.keys() if a[k] != b[k])
        item['removed_functions'] = sorted(a.keys()-b.keys())
        item['added_functions'] = sorted(b.keys()-a.keys())
    rows.append(item)
assert len(rows) == 15
assert len([r for r in rows if r['target'].startswith('scripts/')]) == 7
assert git('diff', '73524b6', RELEASE, '--', 'src') == b''
assert git('diff', '73524b6', RELEASE, '--', 'scripts/scaling_continue.py', 'scripts/slurm/scaling_continue.sbatch') == b''

old, new = module('scaling_reading'), module('scaling_reading19618')
policy_path = BASE+'continuation19618_20261106/scientific_reading_policy_19618.json'
assert json.loads((ROOT/policy_path).read_text()) == new.POLICY
shared = ('candidate','control','bootstrap','epn_margin_pp','pose_practical_point_reduction',
          'p1_p2_noninferiority_margin_pp','historical_pair_is_noise_floor')
assert all(old.POLICY[k] == new.POLICY[k] for k in shared)
endpoint = BASE+'continuation19618_20261106/endpoint19618_reading.md'
assert git('show','73524b6:'+endpoint) == (ROOT/endpoint).read_bytes()
freeze = json.loads((ROOT/(BASE+'continuation19618_budget_freeze_20261105_v2.json')).read_text())
tree = ast.parse((ROOT/'scripts/scaling_evaluation19618.py').read_text())
constants = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and
             len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='CONTROL_CHECKPOINT_SHA256'}
assert constants['CONTROL_CHECKPOINT_SHA256'] == freeze['control19618']['checkpoint_sha256']
assert (freeze['start_step'],freeze['end_step'],freeze['additional_steps']) == (5000,19618,14618)

# Independent formulas, including interval equality at 0 and -1, rather than only
# comparing two implementations that might share the same mistake.
intervals = [(-2,-1.1),(-2,-1),(-1.1,-.1),(-1,0),(-.1,0),(-.1,.1),(0,.1),(.1,.2)]
ratios = [.5,.979,.98,.981,1,1.05]
cases = 0
for epn, p1, p2, pose, ratio in itertools.product(intervals, intervals, intervals, intervals, ratios):
    row = {'protocol':'scaling-comparison-19618-20261106-v1','checkpoint_step':19618,
           'candidate':'T100-G-300M','control':'T100-G-30M','bootstrap':dict(new.POLICY['bootstrap']),
           'quality':{'artifact_integrity':True,'paired_support':True,'solver_converged':True},
           'epn':{'delta_pp':.1,'ci95':list(epn)},'p1':{'delta_pp':0.,'ci95':list(p1)},
           'p2':{'delta_pp':0.,'ci95':list(p2)},
           'pose':{'control_E_dyn':1.,'candidate_E_dyn':ratio,'delta_E_dyn':ratio-1.,'ci95':list(pose)}}
    got = new.read_comparison(row)
    expected_epn = 'better' if epn[0]>0 else 'worse' if epn[1]<0 else 'unresolved'
    expected_pose = 'better' if pose[1]<0 else 'worse' if pose[0]>0 else 'unresolved'
    expected_ni = {k:'supported' if v[0]>-1 else 'loss_beyond_margin' if v[1]<-1 else 'unresolved' for k,v in [('p1',p1),('p2',p2)]}
    assert got['primary_reading'] == {'epn':expected_epn,'pose':expected_pose}
    assert got['classification_noninferiority'] == expected_ni
    assert got['pose_practical_signal'] == ((1.-ratio)>=.02 and pose[1]<0)
    assert 'recommendation' not in got and got['further_training_authorized'] is False
    cases += 1

out = {'release':RELEASE,'source_commit':FROZEN,'status':'PASS','files':rows,
       'independent_policy_cases':cases,'shared_policy_fields':list(shared),
       'endpoint_reading_sha256':sha((ROOT/endpoint).read_bytes()),'endpoint_unchanged_since':'73524b6',
       'control_checkpoint_sha256':constants['CONTROL_CHECKPOINT_SHA256'],
       'src_and_training_wrapper_unchanged':True,
       'notes':['S-2/S-4 are outcome acceptance checks, not submission gates; numeric bootstrap defaults remain frozen.',
                'No live cache/checkpoint at 19,618 exists yet; this proves release code/policy, not future data outcomes.']}
Path('/private/tmp/eval19618-derivation-review.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':'PASS','derived_files':len(rows),'independent_policy_cases':cases}))
