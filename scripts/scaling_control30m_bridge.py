"""Pure early-control plan and mixed-producer join for the frozen 49k readers.

No submission, remote reads, cache writes or numerical changes. The controller
supplies collected evidence and the actual, separately attested producer SHA.
"""
from copy import deepcopy
import json
from pathlib import Path


READER_CODE = '87da975bc0fd4824684e166659bb996e118ff107'
PROTOCOL = 'scaling-control30m-join-20261121-v1'
OLD = 'results/infra/scaling_300m_20261104/continuation19618_20261106'
LAUNCH = 'results/infra/scaling_49046_20261118/launcher'
CONTROL_RECEIPT = LAUNCH + '/control_line2_evidence/continuation/49046/receipt.json'


def build_control_plan(S, root):
    """Actual preserved 30M pin; no candidate placeholder and no candidate read."""
    plan = S._load(OLD + '/evaluation/extraction_plan.json', root)
    receipt = S._load(CONTROL_RECEIPT, root)
    checkpoint = receipt['persistent_checkpoint']
    S.require(receipt.get('protocol') == 'consolidation-t100g-20261104-v1' and
              receipt.get('complete') is True and receipt.get('step') == 49046 and
              receipt.get('summary', {}).get('steps') == 49046 and
              receipt['summary'].get('stopped') == 'max_steps' and
              receipt.get('checkpoint_sha256') == checkpoint.get('sha256') == S.E.CONTROL_CHECKPOINT_SHA256 and
              checkpoint.get('bytes') == 454381728 and
              Path(checkpoint.get('path', '')).is_absolute() and
              Path(checkpoint['path']).name == 'checkpoint_postdecay.pt',
              'the preserved 30M endpoint receipt is incomplete or differs')
    control = {k: checkpoint[k] for k in ('path', 'sha256', 'bytes')}
    control.update(receipt_path=S.checkout('code') + '/' + CONTROL_RECEIPT,
                   receipt_sha256=S.digest((Path(root) / CONTROL_RECEIPT).read_bytes()))
    plan.update(protocol=S.E.PROTOCOL, step=49046, arms=[S.ARMS[0]], control_only=True,
                control_checkpoint_sha256=S.E.CONTROL_CHECKPOINT_SHA256,
                checkpoints={S.ARMS[0]: control})
    plan['training_source_contract'] = S.shared('training_source_contract', plan['training_source_contract'])
    return plan


def _frozen_readers(S, code_sha, root):
    S.require(code_sha == READER_CODE, 'mixed caches must feed the original frozen 49k readers')
    sources = {**S.P.source_hashes(), **S.C.source_hashes()}
    S.require(all(S._blob_sha256(READER_CODE, name, root) == sha for name, sha in sources.items()),
              'local numerical reader sources differ from the frozen 49k readers')
    return sources


def _compatible_plans(S, control, control_sha, final, root):
    S.require(S.ST.digest(control_sha) and S.digest(S.dumps(control)) == control_sha,
              'early-control plan bytes differ from their released hash')
    S.require(control == build_control_plan(S, root), 'early-control plan differs from the registered control release')
    S.require(final.get('arms') == list(S.ARMS) and set(final.get('checkpoints', {})) == set(S.ARMS) and
              'control_only' not in final, 'final extraction plan must retain both registered arms')
    S.require(control['checkpoints'][S.ARMS[0]] == final['checkpoints'][S.ARMS[0]],
              'control checkpoint/receipt pins changed between the extraction plans')
    common_control = {k: v for k, v in control.items() if k not in ('arms', 'checkpoints', 'control_only')}
    common_final = {k: v for k, v in final.items() if k not in ('arms', 'checkpoints')}
    S.require(common_control == common_final,
              'control and candidate extraction plans differ in their common inputs or recipe')


def mixed_reader_plans(S, code_sha, root, control_plan, control_plan_sha256, control_code_sha):
    """Join all six accepted extractions while retaining each producer's real pin.

    The original collection still has all six logical jobs and accounting rows.
    Only the expected code/plan pair is arm-specific. No cache, receipt or
    provenance is rewritten to masquerade as the other producer.
    """
    sources = _frozen_readers(S, code_sha, root)
    S.require(S.ST.digest(control_code_sha, 40) and control_code_sha != READER_CODE,
              'the early-control producer must be its actual separate released commit')
    final = S._load(S.PACKAGE + '/' + S.DOCUMENTS['extraction_plan'][1], root)
    _compatible_plans(S, control_plan, control_plan_sha256, final, root)
    final_sha = S.document('extraction_plan', root)['sha256']
    files, manifest = S.evidence('extractions', root)
    S.require(set(manifest['jobs']) == {j['id'] for j in S.jobs('extractions')},
              'all six original extraction jobs must be collected once')
    producers = {
        S.ARMS[0]: {'checkpoint': control_plan['checkpoints'][S.ARMS[0]],
                    'code_sha': control_code_sha, 'plan_sha256': control_plan_sha256},
        S.ARMS[1]: {'checkpoint': final['checkpoints'][S.ARMS[1]],
                    'code_sha': READER_CODE, 'plan_sha256': final_sha},
    }
    pose_caches, probe_caches, receipts = {}, {}, {}
    tasks = {dataset: task for task, datasets in S.TASKS.items() for dataset in datasets}
    for arm in S.ARMS:
        prefix = arm + '/'
        expected = producers[arm]
        pose_caches[arm] = S.candidate_pose_caches(
            json.loads(files[prefix + 'pose/receipt.json'][0]), files[prefix + 'pose/receipt.json'][1],
            {split: files[prefix + 'pose/' + name + '.npz.provenance.json'][0]
             for split, name in (('train', 'train_subset'), ('val', 'val'))}, expected, arm=arm)
        probe_caches[arm] = S.candidate_probe_caches(
            {task: json.loads(files[prefix + task + '/receipt.json'][0]) for task in S.TASKS},
            {task: files[prefix + task + '/receipt.json'][1] for task in S.TASKS},
            {dataset: files[prefix + tasks[dataset] + '/' + dataset + '.npz.provenance.json'][0]
             for dataset in S.DATASETS}, expected, arm=arm)
        receipts[arm] = {task: files[prefix + task + '/receipt.json'][1] for task in ('pose', *S.TASKS)}
    old_pose = S._load(OLD + '/readers/pose_plan.json', root)
    old_probe = S._load(OLD + '/readers/probe_plan.json', root)
    shared = {k: S.shared(k) for k in S.SHARED}
    pose = S.P.make_plan(pose_caches, old_pose['references'], old_pose['fold_users'],
                         shared['projection'], shared['training_source_contract'])
    probe = {'protocol': S.C.PROTOCOL, 'step': 49046, 'recipe': S.C.recipe(), 'sources': S.C.source_hashes(),
             'caches': probe_caches, 'datasets': old_probe['datasets'], 'references': old_probe['references'], **shared}
    local = {**probe, **{k: {**pin, 'path': str(Path(root) / S.SHARED[k])} for k, pin in shared.items()}}
    S.C.validate_plan(local)
    lineage = {'protocol': PROTOCOL, 'step': 49046, 'reader_code_sha': READER_CODE,
               'numerical_sources': sources, 'producers': deepcopy(producers), 'receipts': receipts,
               'collection_sha256': S.digest(S.dumps(manifest)),
               'common_extraction_inputs_identical': True, 'caches_rewritten': False,
               'control_extracted_again': False}
    return pose, probe, lineage
