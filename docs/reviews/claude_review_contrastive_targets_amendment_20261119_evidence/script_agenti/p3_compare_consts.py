import json, math
import contrastive_compare as K
plan = json.load(open('results/infra/contrastive_targets_20261118/plan.json'))
old = json.load(open('results/infra/contrastive_comparison_20261116/plan.json'))
e = plan['evaluation']
checks = {
 'protocol': (K.PROTOCOL == plan['protocol'] == old['protocol']),
 'pairs': ([list(p) for p in K.PAIRS] == e['pairs']),
 'primary': (list(K.PRIMARY_ENDPOINTS) == e['primary_endpoints']),
 'descriptive': (list(K.DESCRIPTIVE_ENDPOINTS) == e['descriptive_endpoints']),
 'replicates': (K.N_BOOT == e['bootstrap']['replicates']),
 'seed': (K.SEED == e['bootstrap']['seed']),
 'family': (K.FAMILY_SIZE == e['bootstrap']['family_size']),
 'alpha': (K.ALPHA == e['bootstrap']['family_alpha']),
 'simultaneous': math.isclose(K.SIMULTANEOUS_LEVEL, e['bootstrap']['nominal_simultaneous_confidence'], rel_tol=0, abs_tol=1e-15),
 'p1_weights': all(math.isclose(w, e['weights']['p1'][d], abs_tol=1e-15) for w, d in zip(K.WEIGHTS['p1'], K.TASKS['p1'])),
 'p2_weights': all(math.isclose(w, e['weights']['p2'][d], abs_tol=1e-15) for w, d in zip(K.WEIGHTS['p2'], K.TASKS['p2'])),
 'E_dyn_margins': (K.MARGINS['E_dyn'] == {'gain': e['margins']['E_dyn']['useful_reduction_percent'], 'ni': e['margins']['E_dyn']['noninferiority_increase_percent']}),
 'p1_margins': (K.MARGINS['p1'] == {'gain': e['margins']['p1']['useful_gain_pp'], 'ni': e['margins']['p1']['noninferiority_loss_pp']}),
 'epn_margins': (K.MARGINS['epn612'] == {'gain': e['margins']['epn612']['useful_gain_pp'], 'ni': e['margins']['epn612']['noninferiority_loss_pp']}),
 'mc_3se': (K.MC_BOUNDARY_SE == 3.0 and 'within3SE' in e['monte_carlo_boundary_rule']),
 'comparator_path': (e['comparator'] == 'scripts/contrastive_compare.py'),
 'evaluation_identical_old_new': (e == old['evaluation']),
 'fits_420': (3*5*4*7 == e['expected_classification_fits']),
}
for k, v in checks.items(): print(f'{k:30s} {v}')
