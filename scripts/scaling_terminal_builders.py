"""Terminal-dose document builders with explicit per-arm provenance."""
from pathlib import Path
import json

OLD='results/infra/scaling_300m_20261104/continuation19618_20261106'
LAUNCH='results/infra/scaling_49046_20261118/launcher'


def training_job(S,end,root):
    name='training_predecay' if end==39237 else 'training_final'
    relative=S.PACKAGE+'/'+S.DOCUMENTS[name][1]
    primary=(Path(root)/relative).read_bytes()
    S.require(primary==(Path(root)/LAUNCH/f'training_{end}.json').read_bytes(),'training plan mirror differs')
    plan=json.loads(primary)
    stage='train-predecay' if end==39237 else 'train-final'
    out=plan['receipt_dir']
    files={n:out+'/'+n for n in ('receipt.json','launch.json','config.json','summary.json','training.log',*(f'rank_{r}.json' for r in range(4)))}
    return S._job(stage,f'SCALING-49046-train-{end}-20261118',32,plan['allocation']['walltime_seconds'],out,'',
                  files=files,gpus=4,memory='480G',log='scaling_continue49046',exact=list(files))


def extraction_plan_from_receipts(S,root):
    files,collected=S.evidence('train-final',root)
    receipt=json.loads(files['receipt.json'][0])
    summary=json.loads(files['summary.json'][0])
    submitted=S.ledger_job('SCALING-49046-train-49046-20261118')
    S.require(str(receipt['slurm_job_id'])==submitted and receipt['complete'] is True and
              receipt['checkpoint_step']==49046 and receipt['start_step']==39237 and
              summary['stopped']=='max_steps' and summary['steps']==49046 and
              receipt.get('protocol')=='scaling-continuation49046-20261118-v1' and
              receipt.get('scientific_endpoint') is True and receipt.get('previous_run_unchanged') is True and
              receipt.get('additional_steps')==9809 and receipt.get('code_sha')==S.checkout_commit('code'),
              'terminal training is incomplete')
    S.require(receipt['plan_sha256']==S.document('training_final',root)['sha256'],'training plan differs')
    plan=S._load(OLD+'/evaluation/extraction_plan.json',root)
    plan.update(protocol=S.E.PROTOCOL,step=49046,control_checkpoint_sha256=S.E.CONTROL_CHECKPOINT_SHA256)
    native=receipt['run_inventory'][Path(receipt['checkpoint']).name]
    S.require(native['sha256']==receipt['checkpoint_sha256'] and type(native['bytes']) is int and
              0<native['bytes']<=6*2**30,'terminal native pin missing or exceeds storage envelope')
    candidate={'path':receipt['checkpoint'],'sha256':native['sha256'],'bytes':native['bytes']}
    for name in ('config','receipt','summary'):
        candidate[name+'_path']=files[name+'.json'][1]['path']
        candidate[name+'_sha256']=files[name+'.json'][1]['sha256']
    control_receipt=S._load(LAUNCH+'/control_line2_evidence/continuation/49046/receipt.json',root)
    control=control_receipt['persistent_checkpoint']
    control={k:control[k] for k in ('path','sha256','bytes')}
    control_pin=Path(root)/LAUNCH/'control_line2_evidence/continuation/49046/receipt.json'
    control.update(receipt_path=S.checkout('code')+'/'+LAUNCH+'/control_line2_evidence/continuation/49046/receipt.json',
                   receipt_sha256=S.digest(control_pin.read_bytes()))
    plan['checkpoints']={S.ARMS[0]:control,S.ARMS[1]:candidate}
    # Input and reference pins retain the accepted historical selection.
    plan['training_source_contract']=S.shared('training_source_contract',plan['training_source_contract'])
    return plan


def reader_plans_from_receipts(S,code_sha,root):
    files,manifest=S.evidence('extractions',root)
    S.require(set(manifest['jobs'])=={j['id'] for j in S.jobs('extractions')},'incomplete extraction collection')
    plan=S._load(S.PACKAGE+'/'+S.DOCUMENTS['extraction_plan'][1],root)
    plan_sha=S.document('extraction_plan',root)['sha256']
    pose_caches,probe_caches={},{}
    tasks={ds:t for t,datasets in S.TASKS.items() for ds in datasets}
    for arm in S.ARMS:
        prefix=arm+'/'
        ext={'checkpoint':plan['checkpoints'][arm],'code_sha':code_sha,'plan_sha256':plan_sha}
        pose_caches[arm]=S.candidate_pose_caches(json.loads(files[prefix+'pose/receipt.json'][0]),
            files[prefix+'pose/receipt.json'][1],
            {'train':files[prefix+'pose/train_subset.npz.provenance.json'][0],
             'val':files[prefix+'pose/val.npz.provenance.json'][0]},ext,arm=arm)
        probe_caches[arm]=S.candidate_probe_caches(
            {t:json.loads(files[prefix+t+'/receipt.json'][0]) for t in S.TASKS},
            {t:files[prefix+t+'/receipt.json'][1] for t in S.TASKS},
            {ds:files[prefix+tasks[ds]+'/'+ds+'.npz.provenance.json'][0] for ds in S.DATASETS},ext,arm=arm)
    old_pose=S._load(OLD+'/readers/pose_plan.json',root)
    old_probe=S._load(OLD+'/readers/probe_plan.json',root)
    shared={k:S.shared(k) for k in S.SHARED}
    pose=S.P.make_plan(pose_caches,old_pose['references'],old_pose['fold_users'],shared['projection'],shared['training_source_contract'])
    probe={'protocol':S.C.PROTOCOL,'step':49046,'recipe':S.C.recipe(),'sources':S.C.source_hashes(),
           'caches':probe_caches,'datasets':old_probe['datasets'],'references':old_probe['references'],**shared}
    local={**probe,**{k:{**pin,'path':str(Path(root)/S.SHARED[k])} for k,pin in shared.items()}}
    S.C.validate_plan(local)
    return pose,probe
