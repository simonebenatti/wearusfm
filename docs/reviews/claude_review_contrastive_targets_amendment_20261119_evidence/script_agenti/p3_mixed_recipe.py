"""Show that the final comparator accepts pose cells annotated with different target recipes per arm
and that its output carries no amendment/recipe identity (read-only demonstration on synthetic data)."""
import sys, json
sys.path.insert(0, sys.argv[1])
import test_contrastive_compare as T
import contrastive_compare as C
cl, po = T.inputs()
# Annotate cells as a future extractor might: C0 historical 25/50/v1, CN/CR new 100/125/v3.
po['C0']['recipe'] = {'rms_window_ms': 25., 'ms_fast_window_ms': 50., 'target_version': 'step1_psd_power_shape_energy_v1'}
for arm in ('CN', 'CR'):
    po[arm]['recipe'] = {'rms_window_ms': 100., 'ms_fast_window_ms': 125., 'target_version': 'step1_psd_power_shape_energy_fast125_v3'}
r = C.compare(cl, po, n_boot=100, audit_records=T.audits())
print('accepted mixed-recipe grid; decision_valid only false because n_boot!=100000:', r['decision_valid'])
print('output protocol:', r['protocol'])
keys = set(r)
print('output mentions amendment/recipe/target:', sorted(k for k in keys if any(s in k for s in ('amend', 'recipe', 'target', 'cohort', 'config'))))
print('pose report keys:', sorted(r['pose']))
