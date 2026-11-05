import copy
import importlib.util
import json
from pathlib import Path
import sys
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import consolidation_long_common as C


@pytest.mark.parametrize('stage,expected', [
    ('extract-classification', 1), ('extract-pose-train', 1), ('extract-pose-val', 1),
    ('benchmark-classification', 8), ('benchmark-pose', 8), ('fit-classification', 8),
    ('fit-pose', 8), ('compare-classification', 8), ('compare-pose', 8), ('summary', 1),
])
def test_launcher_configures_historical_threads_before_scientific_imports(stage, expected):
    text = (ROOT/'scripts/slurm/consolidation_long.sbatch').read_text()
    policy = text.split("<<'THREAD_POLICY'\n", 1)[1].split('\nTHREAD_POLICY\n', 1)[0]
    result = subprocess.run([sys.executable, '-c', policy, '--stage', stage, '--out', '/unused'],
                            check=True, capture_output=True, text=True)
    assert result.stdout.strip() == str(expected)
    assert text.index('export OMP_NUM_THREADS="$NUMERIC_THREADS"') < text.index('from consolidation_long import parser')
    key = ('summary' if stage == 'summary' else 'extract' if stage.startswith('extract')
           else '_'.join(reversed(stage.split('-'))))
    allocation = C.RESOURCE_POLICY[key]
    assert expected <= allocation['cpus']


def test_launcher_rejects_unregistered_thread_stage_before_imports():
    text = (ROOT/'scripts/slurm/consolidation_long.sbatch').read_text()
    policy = text.split("<<'THREAD_POLICY'\n", 1)[1].split('\nTHREAD_POLICY\n', 1)[0]
    result = subprocess.run([sys.executable, '-c', policy, '--stage', 'unexpected'], capture_output=True, text=True)
    assert result.returncode != 0 and not result.stdout.strip()


def test_receipt_binds_durable_reference(monkeypatch):
    ck = {'path':'/work/ref.pt','sha256':'a'*64,'bytes':100}
    rec = {'complete':True,'code_sha':'b'*40,'protocol':C.T.PROTOCOL,
           'checkpoint_step':19618,'arm':'T100-G','checkpoint_sha256':'a'*64}
    archive = {'complete':True,'step':30000,'persistent_source':ck.copy()}
    monkeypatch.setattr(C,'read_pin',lambda p: rec if p['path']=='receipt' else archive)
    entry={'checkpoint':ck,'receipt':{'path':'receipt'},'archive_receipt':{'path':'archive'},'training_code_sha':'b'*40}
    C.verify_training_receipt(entry,19618)
    archive['persistent_source']['path']='/wrong/copy.pt'
    with pytest.raises(ValueError,match='archive'):C.verify_training_receipt(entry,19618)


def test_long_receipt_requires_dose_summary_and_reference(monkeypatch):
    ck={'path':'/work/new.pt','sha256':'c'*64,'bytes':100}
    ref={'path':'/work/ref.pt','sha256':'a'*64,'bytes':100}
    rec={'complete':True,'code_sha':'b'*40,'protocol':'consolidation-t100g-20261104-v1',
         'step':39237,'summary':{'steps':39237,'stopped':'max_steps'},'plan_sha256':'d'*64,
         'persistent_checkpoint':ck.copy(),'checkpoint_sha256':'c'*64,'persistent_source':ref.copy()}
    monkeypatch.setattr(C,'read_pin',lambda pin:rec)
    entry={'checkpoint':ck,'reference_checkpoint':ref,'receipt':{},'training_code_sha':'b'*40,'training_plan_sha256':'d'*64}
    C.verify_training_receipt(entry,39237)
    rec['summary']['stopped']='time_limit'
    with pytest.raises(ValueError,match='dose'):C.verify_training_receipt(entry,39237)


def test_fresh_output_rejects_overwrite_outside_and_symlink(tmp_path):
    root=tmp_path/'results';root.mkdir()
    ctx={'plan':{'output_root':str(root),'recipe_sha256':'r'},'plan_pin':{'sha256':'p'},'code_sha':'c','source_sha256':'s'}
    out=root/'new';C.begin_output(ctx,out,'test')
    with pytest.raises(ValueError):C.begin_output(ctx,out,'test')
    with pytest.raises(ValueError):C.begin_output(ctx,tmp_path/'outside','test')
    (root/'link').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError):C.begin_output(ctx,root/'link'/'child','test')


def test_finish_rejects_changed_frozen_teacher(monkeypatch,tmp_path):
    import torch
    model=torch.nn.Linear(2,1)
    ctx={'plan_pin':{'path':'plan','sha256':'p'},'code_sha':'c','source_sha256':'s',
         '_models':[(model,C.state_sha256(model.state_dict()))]}
    monkeypatch.setattr(C.T.P,'file_identity',lambda p:ctx['plan_pin'])
    monkeypatch.setattr(C.T.P,'code_identity',lambda c:c)
    monkeypatch.setattr(C,'source_identity',lambda:'s')
    with torch.no_grad():model.weight.add_(1)
    with pytest.raises(ValueError,match='teacher changed'):C.finish_output(ctx,tmp_path,{})


def test_finish_rejects_checkpoint_changed_or_stop(monkeypatch,tmp_path):
    path=tmp_path/'checkpoint.pt';path.write_bytes(b'old')
    guard=C.file_pin(path)
    ctx={'plan_pin':{'path':'plan'},'code_sha':'c','source_sha256':'s','_checkpoint_guards':[guard]}
    monkeypatch.setattr(C.T.P,'file_identity',lambda p:ctx['plan_pin'] if p=='plan' else guard)
    monkeypatch.setattr(C.T.P,'code_identity',lambda c:c)
    monkeypatch.setattr(C,'source_identity',lambda:'s')
    (tmp_path/'STOP').write_text('stop')
    with pytest.raises(ValueError,match='STOP'):C.finish_output(ctx,tmp_path,{})


def test_json_writer_refuses_nan_and_overwrite(tmp_path):
    with pytest.raises(ValueError):C.write_json(tmp_path/'bad.json',{'x':float('nan')})
    p=tmp_path/'ok.json';C.write_json(p,{'x':1})
    with pytest.raises(FileExistsError):C.write_json(p,{'x':2})


def test_extraction_cannot_be_reused_after_source_amendment_with_same_recipe_and_checkpoint(monkeypatch):
    ctx = {'source_sha256': 'old-source', 'plan': {'recipe_sha256': 'recipe',
            'endpoints': {'19618': {'checkpoint': {'sha256': 'checkpoint'}}}}}
    entry = {'receipt': {'path': 'receipt'}, 'cache': {'path': 'cache', 'sha256': 'cache-sha'}}
    report = {'complete': True, 'source_sha256': 'old-source', 'recipe_sha256': 'recipe',
              'checkpoint_step': 19618, 'kind': 'extract-pose-train', 'checkpoint_sha256': 'checkpoint',
              'which': 'teacher', 'checkpoint_unchanged': True, 'frozen_teacher_unchanged': True,
              'artifacts': {'cache': entry['cache']}}
    monkeypatch.setattr(C, 'read_pin', lambda pin: report)
    monkeypatch.setattr(C, 'pinned', lambda pin: pin)
    assert C.verified_extraction(ctx, entry, 19618, 'extract-pose-train') is report
    ctx['source_sha256'] = 'new-source'
    with pytest.raises(ValueError, match='extraction receipt differs'):
        C.verified_extraction(ctx, entry, 19618, 'extract-pose-train')


def test_concrete_campaign_remains_review_gated_and_all_endpoints_pinned():
    plan=json.loads((ROOT/'results/infra/consolidation_long_20261105/plan.json').read_text())
    C.validate_plan(plan,require_review=False)
    assert plan['source_sha256']==C.source_identity()
    with pytest.raises(ValueError,match='Claude'):C.validate_plan(plan)
    for step in C.DOSES:
        pin=plan['endpoints'][str(step)]['checkpoint']
        assert pin['path'].startswith('/leonardo_work/') and pin['bytes']==454381728
        assert len(pin['sha256'])==64
    changed=copy.deepcopy(plan);changed['reading_rule']['pose_practical_relative_percent']=1
    changed['recipe_sha256']=C.recipe_identity(changed)
    with pytest.raises(ValueError,match='policy'):C.validate_plan(changed,require_review=False)


def test_concrete_launch_matrix_dose_fold_coverage_and_resource_caps():
    launch=json.loads((ROOT/'results/infra/consolidation_long_20261105/launch_plan.json').read_text())
    jobs=launch['jobs'];assert len(jobs)==len({r['id'] for r in jobs})==len({r['out'] for r in jobs})==82
    fits=[j for j in jobs if j['stage']=='fit-classification'];assert len(fits)==48
    logical={}
    for row in fits:
        args=row['args'];step=int(args[args.index('--step')+1]);view=args[args.index('--view')+1]
        ds=args[args.index('--dataset')+1];folds=list(map(int,args[args.index('--folds')+1:]))
        assert folds in ([[0,1],[2,3]] if ds=='ninapro_db2' else [[0,1,2,3]])
        logical.setdefault((step,view,ds),[]).extend(folds)
    assert len(logical)==40 and all(sorted(v)==list(range(4)) for v in logical.values())
    assert all(j['allocation']['gpus']==0 for j in jobs if not j['stage'].startswith('extract'))
    assert sum(j['allocation']['gpus']*j['allocation']['seconds']/3600 for j in jobs)==9
    cores=sum(j['allocation']['cpus']*j['allocation']['seconds']/3600 for j in jobs)+4*2*600/3600
    assert cores==pytest.approx(launch['caps']['allocated_core_hours'])


@pytest.mark.parametrize('wrong', ['classification','pose'])
def test_summary_cannot_accept_same_recipe_fit_as_final_comparison(monkeypatch,wrong,tmp_path):
    import argparse
    import consolidation_long as MAIN
    reports={k:{'complete':True,'source_sha256':'s','recipe_sha256':'r','protocol':C.PROTOCOL,
                'kind':'compare-'+k,'artifacts':{}} for k in ('classification','pose')}
    reports[wrong]['kind']='fit-'+wrong
    ctx={'plan':{'final_reports':{k:{'path':k} for k in reports},'recipe_sha256':'r'},'source_sha256':'s'}
    monkeypatch.setattr(C,'load_context',lambda *a:ctx)
    monkeypatch.setattr(C,'read_pin',lambda p:reports[p['path']])
    monkeypatch.setattr(C,'begin_output',lambda *a,**k:{})
    args=argparse.Namespace(plan='plan',expected_code='code',out=tmp_path,stage='summary')
    with pytest.raises(ValueError,match='mixed final reports'):MAIN.execute(args)
