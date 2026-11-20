"""Probes that isolate each guard of R5-4. Usage: probe.py WORKDIR (PYTHONPATH selects the code under test)."""
import json, sys, copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import contrastive_target_parity as P
from test_contrastive_target_parity import point, case
from test_contrastive_runtime import _config
from test_pose_training import _pose_setup
from wearusfm.data.pretraining_loader import PretrainLoader

work = Path(sys.argv[1]); work.mkdir(parents=True, exist_ok=True)
out = {'module': P.__file__}


def gate(rec):
    r = P.assess_point(rec)
    return r['status'] + ' ' + ';'.join(r['errors'])


rec = point()
for c in rec['ranks'][1]['cases'].values():
    c['before']['student_sha256'] = 'x' * 64
rec['ranks'][1]['live_before'] = dict(rec['ranks'][1]['live_before'], student_sha256='x' * 64)
out['C1c_rank1_student_consistent'] = gate(rec)
rec = point(); rec['ranks'][1]['global_ids_sha256'] = 'e'
out['C1c_rank1_global_ids'] = gate(rec)
good = {'schema': 1, 'profile': P.PROFILE, 'points': [point(128), point(256)]}
out['C1q_baseline'] = P.validate_parity_report(good)['status']
out['C1q_profile_other'] = P.validate_parity_report({**good, 'profile': 'other'})['status']
out['C1q_schema_2'] = P.validate_parity_report({**good, 'schema': 2})['status']

(work / 'pose').mkdir(exist_ok=True)
root, manifest, pose = _pose_setup(work / 'pose')
base_cfg = replace(_config(pose), batch_size=32, contrastive={**_config(pose).contrastive, 'fill_steps': 256})
loader = PretrainLoader(P.trainer.load_index(manifest, [root], base_cfg.datasets, base_cfg.nested), base_cfg.loader)
P.assert_numeric_profile = lambda **kw: {}


def observer_run(tag, *, index=127, world=4, cfg=None, modify=False, calls=((127, None),)):
    """Each call: (step, cfg override); captured batch is set before each call like observed_prepare."""
    cfg0 = cfg or base_cfg
    student = P.attach_contrastive_head(P.WearUsFM(cfg0.model))
    teacher = P.make_teacher(student)
    runtime = P.ContrastiveRuntime(cfg0.contrastive)
    runtime.expected_datasets = ('emg2pose',)
    observer = P.ControlParityObserver(work / ('out_' + tag))
    ctx = SimpleNamespace(world_size=world, active=False, rank=0)

    def mock_case(name, *a, **k):
        if modify and name == 'C0a':
            with torch.no_grad():
                next(student.parameters()).add_(1.)
        return case()
    P._measure_case = mock_case
    res = []
    for step, cfg_i in calls:
        cfg_i = cfg_i or cfg0
        batch = loader.batch(32, np.random.default_rng(42))
        batch.index = index if step == 127 else step
        micros = [P.trainer._prepare(batch, 'cpu')]
        try:
            runtime.queue.last_step = step - 1
            runtime.prepare(batch, step)
            observer.captured[id(runtime)] = batch
            observer.measure(None, student, teacher, micros, cfg_i, ctx, 'cpu', None,
                             P.trainer._accumulate_terms, P.ContrastiveRuntime.prepare, contrastive=runtime)
            pt = observer.points[-1] if observer.points else None
            res.append('OK points=%d restored=%s gate=%s' % (len(observer.points), pt and pt['ranks'][0]['live_state_restored'],
                                                              pt and pt['gate']['status']))
        except Exception as exc:
            pt = observer.points[-1] if observer.points else None
            res.append('%s: %s | points=%d restored=%s' % (type(exc).__name__, exc, len(observer.points),
                                                            pt and pt['ranks'][0]['live_state_restored']))
    return res


out['OBS_baseline'] = observer_run('base')
out['C1d_C1t_live_modified'] = observer_run('mod', modify=True)
out['C1f_batch_index_126'] = observer_run('idx', index=126)
out['C1l_world_size_2'] = observer_run('ws', world=2)
out['C1l_accum_2'] = observer_run('acc', cfg=replace(base_cfg, accum_steps=2))
out['C1m_cfg_changed'] = observer_run('cfgch', calls=((127, None), (255, replace(base_cfg, lr=2e-4))))
out['C1n_duplicate_127'] = observer_run('dup', calls=((127, None), (127, None)))
print(json.dumps(out, indent=1))
