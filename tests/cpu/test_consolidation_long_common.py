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
    ctx={'plan':{'output_root':str(root),'recipe_sha256':'r'},'plan_pin':{'sha256':'p'},'code_sha':'c','source_sha256':'s',
         'amendment':C.A.record()}
    out=root/'new';report=C.begin_output(ctx,out,'test')
    # Every report written under amendment A1 names the closed input list and the original source.
    assert report['source_sha256']=='s' and report['amendment']=={'id':'A1','input_compatibility':C.A.INPUT_MANIFEST,
                                                                  'input_source_sha256':C.A.ORIGINAL_SOURCE}
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


def bridge(report, kind='extract-pose-train'):
    """A context of amendment A1 whose closed list holds one receipt of the original source."""
    pin = {'path': 'receipt', 'sha256': 'r'*64, 'bytes': 10}
    inputs = {'source_sha256': 'old-source', 'recipe_sha256': 'recipe',
              'phases': {'extract': {'code_sha': 'c'*40, 'plan_sha256': 'p'*64}},
              'receipts': [{'id': 'one', 'phase': 'extract', 'kind': kind, **pin,
                            'fields': {'checkpoint_step': report['checkpoint_step']}}]}
    ctx = {'source_sha256': 'new-source', 'inputs': inputs, 'plan': {'recipe_sha256': 'recipe',
           'endpoints': {'19618': {'checkpoint': {'sha256': 'checkpoint'}}}}}
    return ctx, pin


def original_receipt():
    cache = {'path': 'cache', 'sha256': 'cache-sha'}
    return {'protocol': C.PROTOCOL, 'complete': True, 'source_sha256': 'old-source', 'recipe_sha256': 'recipe',
            'code_sha': 'c'*40, 'plan_sha256': 'p'*64,
            'checkpoint_step': 19618, 'kind': 'extract-pose-train', 'checkpoint_sha256': 'checkpoint',
            'which': 'teacher', 'checkpoint_unchanged': True, 'frozen_teacher_unchanged': True,
            'artifacts': {'cache': cache}}, cache


def test_extraction_is_reused_only_as_a_listed_original_input_of_amendment_a1(monkeypatch):
    report, cache = original_receipt()
    ctx, pin = bridge(report)
    entry = {'receipt': pin, 'cache': cache}
    monkeypatch.setattr(C, 'read_pin', lambda pin: report)
    monkeypatch.setattr(C, 'pinned', lambda pin: pin)
    assert C.verified_extraction(ctx, entry, 19618, 'extract-pose-train') is report
    # The exception is a closed list of exact pins: another file, hash, size or kind is not an input.
    for changed in ({'path': 'other'}, {'sha256': '0'*64}, {'bytes': 11}):
        with pytest.raises(ValueError, match='not a listed amendment input'):
            C.verified_extraction(ctx, {**entry, 'receipt': {**pin, **changed}}, 19618, 'extract-pose-train')
    with pytest.raises(ValueError, match='not a listed amendment input'):
        C.verified_extraction(ctx, {**entry, 'receipt': {k: pin[k] for k in ('path', 'sha256')}}, 19618, 'extract-pose-train')
    with pytest.raises(ValueError, match='not a listed amendment input'):
        C.verified_extraction(ctx, entry, 19618, 'extract-pose-val')
    ctx['inputs']['receipts'].append(dict(ctx['inputs']['receipts'][0], id='twice'))
    with pytest.raises(ValueError, match='not a listed amendment input'):
        C.verified_extraction(ctx, entry, 19618, 'extract-pose-train')
    del ctx['inputs']['receipts'][1]
    assert C.verified_extraction(ctx, entry, 19618, 'extract-pose-train') is report


@pytest.mark.parametrize('field,value', [
    ('source_sha256', 'new-source'), ('source_sha256', 'third-source'), ('recipe_sha256', 'other'),
    ('code_sha', 'd'*40), ('plan_sha256', 'e'*64), ('protocol', 'other'), ('complete', False),
    ('kind', 'extract-pose-val'), ('checkpoint_step', 39237)])
def test_listed_input_must_keep_original_source_recipe_code_plan_and_fields(monkeypatch, field, value):
    report, cache = original_receipt()
    ctx, pin = bridge(report)
    monkeypatch.setattr(C, 'read_pin', lambda pin: report)
    monkeypatch.setattr(C, 'pinned', lambda pin: pin)
    assert C.verified_input(ctx, pin, 'extract-pose-train') is report
    report[field] = value
    with pytest.raises(ValueError, match='amendment input receipt provenance differs'):
        C.verified_input(ctx, pin, 'extract-pose-train')


def test_bridge_never_accepts_inputs_of_the_running_source_or_another_recipe(monkeypatch):
    report, cache = original_receipt()
    ctx, pin = bridge(report)
    monkeypatch.setattr(C, 'read_pin', lambda pin: report)
    monkeypatch.setattr(C, 'pinned', lambda pin: pin)
    # No generic accepted-source list: the running source can never be its own input source.
    ctx['source_sha256'] = 'old-source'
    with pytest.raises(ValueError, match='provenance'):
        C.verified_input(ctx, pin, 'extract-pose-train')
    ctx['source_sha256'] = 'new-source'
    ctx['plan']['recipe_sha256'] = 'another-recipe'
    with pytest.raises(ValueError, match='provenance'):
        C.verified_input(ctx, pin, 'extract-pose-train')
    ctx['plan']['recipe_sha256'] = 'recipe'
    # Later extraction checks are unchanged: dose, checkpoint, immutability, cache coverage.
    entry = {'receipt': pin, 'cache': cache}
    for field, value, message in (('checkpoint_sha256', 'other', 'extraction receipt differs'),
                                  ('frozen_teacher_unchanged', False, 'immutability'),
                                  ('artifacts', {}, 'not covered')):
        changed = {**report, field: value}
        monkeypatch.setattr(C, 'read_pin', lambda pin, changed=changed: changed)
        with pytest.raises(ValueError, match=message):
            C.verified_extraction(ctx, entry, 19618, 'extract-pose-train')


def test_concrete_campaign_remains_review_and_decision_gated_and_all_endpoints_pinned():
    plan=json.loads((ROOT/'results/infra/consolidation_long_20261105/plan.json').read_text())
    C.validate_plan(plan,require_review=False)
    assert plan['source_sha256']==C.source_identity()!=C.A.ORIGINAL_SOURCE
    assert plan['recipe_sha256']==C.recipe_identity(plan)==C.A.ORIGINAL_RECIPE
    assert plan['output_root']==C.OUTPUT_ROOT
    assert plan['output_root'].startswith('/leonardo_scratch/large/userexternal/sbenatti/')
    # The runtime runs only a plan with the converged review and the recorded decision on amendment A1.
    if plan['amendment']['decision'] is None:
        with pytest.raises(ValueError,match='decision of Simone'):C.validate_plan(plan)
    else:C.validate_plan(plan)
    changed=copy.deepcopy(plan);changed['amendment']['decision']=None
    with pytest.raises(ValueError,match='decision of Simone'):C.validate_plan(changed)
    changed=copy.deepcopy(plan)
    changed['amendment']['decision']={'by':'Simone','date':'2026-11-06','quote':'test'}
    C.validate_plan(changed)
    changed['review']['status']='pending'
    with pytest.raises(ValueError,match='Claude'):C.validate_plan(changed)
    for step in C.DOSES:
        pin=plan['endpoints'][str(step)]['checkpoint']
        assert pin['path'].startswith('/leonardo_work/') and pin['bytes']==454381728
        assert len(pin['sha256'])==64
    changed=copy.deepcopy(plan);changed['reading_rule']['pose_practical_relative_percent']=1
    changed['recipe_sha256']=C.recipe_identity(changed)
    with pytest.raises(ValueError,match='policy'):C.validate_plan(changed,require_review=False)
    changed=copy.deepcopy(plan)
    changed['output_root']='/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/consolidation_long_20261105'
    with pytest.raises(ValueError,match='output'):C.validate_plan(changed,require_review=False)


def test_concrete_launch_matrix_dose_fold_coverage_and_resource_caps():
    launch=json.loads((ROOT/'results/infra/consolidation_long_20261105/launch_plan.json').read_text())
    plan=json.loads((ROOT/'results/infra/consolidation_long_20261105/plan.json').read_text())
    jobs=launch['jobs'];assert len(jobs)==len({r['id'] for r in jobs})==len({r['out'] for r in jobs})==103
    assert launch['owner']=='claude' and launch['source_sha256']==plan['source_sha256']
    assert all(row['out'].startswith(C.OUTPUT_ROOT+'/') and row['args'][row['args'].index('--out')+1]==row['out'] for row in jobs)
    fits=[j for j in jobs if j['stage']=='fit-classification'];assert len(fits)==69
    table=plan['amendment']['fit_shards'];logical={};seen=set()
    for row in fits:
        args=row['args'];step=int(args[args.index('--step')+1]);view=args[args.index('--view')+1]
        ds=args[args.index('--dataset')+1];folds=list(map(int,args[args.index('--folds')+1:]))
        assert folds in C.A.fit_shapes(ds,view) and not (step==19618 and view=='pool')
        key='%s/%s/%s/folds%s'%(step,view,ds,'-'.join(map(str,folds)))
        assert row['shard']==key and row['id'].endswith('-20261105-A1') and '/fits_a1/classification/' in row['out']
        # The requested limit of every job is the walltime frozen in the plan for its own shard.
        assert row['allocation']=={'gpus':0,'cpus':8,'memory_gib':30,'seconds':table[key]['walltime_seconds']}
        assert 1800<=row['allocation']['seconds']<=13500 and row['allocation']['seconds']%900==0
        assert row['slurm_job_id'] is None or isinstance(row['slurm_job_id'],int)
        logical.setdefault((step,view,ds),[]).extend(folds);seen.add(key)
    assert seen==set(table) and len(logical)==40 and all(sorted(v)==list(range(4)) for v in logical.values())
    # The pose fits are released before the classification fits; no fifth checkout, no new comparison.
    phase=[j['stage'] for j in jobs if j['phase']=='fit']
    assert phase==['fit-pose']*3+['fit-classification']*69
    assert [j['stage'] for j in jobs if j['phase']=='aggregate']==['compare-classification','compare-pose','summary']
    assert len(launch['phase_checkouts'])==launch['caps']['checkout_count']==4
    # The forty-eight original rows stay in the plan as superseded, with successors covering the same folds.
    old=launch['superseded_jobs'];assert len(old)==48 and all(r['slurm_job_id'] is None for r in old)
    assert not {r['id'] for r in old}&{r['id'] for r in jobs} and not {r['out'] for r in old}&{r['out'] for r in jobs}
    by_id={r['id']:r for r in fits}
    assert sorted(i for r in old for i in r['superseded_by'])==sorted(by_id)
    for row in old:
        folds=list(map(int,row['args'][row['args'].index('--folds')+1:]))
        covered=sorted(f for i in row['superseded_by'] for f in table[by_id[i]['shard']]['folds'])
        assert covered==folds
    # Caps are those approved for the original chain; the amended chain stays inside them.
    assert all(j['allocation']['gpus']==0 for j in jobs if not j['stage'].startswith('extract'))
    assert sum(j['allocation']['gpus']*j['allocation']['seconds']/3600 for j in jobs)==9==launch['caps']['gpu_hours']
    cores=sum(j['allocation']['cpus']*j['allocation']['seconds']/3600 for j in jobs)+4*2*600/3600
    budget=launch['caps']['amendment_A1']
    assert cores==pytest.approx(budget['conservative_total_core_hours'])==pytest.approx(1451.666667)
    assert launch['caps']['allocated_core_hours']==pytest.approx(1711.666667) and cores<=launch['caps']['allocated_core_hours']
    assert budget['classification_fit_allocated_core_hours']==1276 and budget['future_allocated_core_hours']==1317
    assert budget['new_gpu_hours']==0 and budget['compute_jobs']==103 and budget['allocations_left']==77
    original=sum(j['allocation']['cpus']*j['allocation']['seconds']/3600 for j in old)
    assert cores-1276+original==pytest.approx(launch['caps']['allocated_core_hours'])


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
