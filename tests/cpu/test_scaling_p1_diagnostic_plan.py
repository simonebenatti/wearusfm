import copy
import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('p1plan',ROOT/'scripts/scaling_p1_diagnostic_plan.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

@pytest.fixture
def inputs():
    evidence=json.loads((M.BASE/'evidence/preparation/remote_metadata.json').read_text())
    inventory=json.loads((ROOT/'results/infra/tokenizer_evaluation_20261103/probe_plan_19618.json').read_text())
    plan=json.loads((ROOT/'results/infra/scaling_300m_20261104/continuation19618_20261106/evaluation/extraction_plan.json').read_text())
    return [evidence,{'path':'test.json','sha256':'a'*64,'passed':True},inventory,plan]

def test_actual_evidence_and_amended_budget(inputs):
    manifest,budget=M.build(*inputs)
    assert len(manifest['inputs'])==6
    assert [x['pin_kind'] for x in manifest['inputs']]==['diagnostic']*3+['accepted']*3
    assert budget['computed_stage_gpu_hours_max']==pytest.approx(7.264444444444444)
    assert budget['computed_p1_gpu_hours_max']==pytest.approx(3.0708333333333333)
    assert budget['cpu_only_consumed_plus_future_max']==pytest.approx(350.4511113333333)

@pytest.mark.parametrize('mutation',['exit_false','wrong_exit','dirty','grace','payload_pin','sidecar','failed_job','symlink'])
def test_mutated_evidence_is_not_promoted(inputs,mutation):
    e=inputs[0]
    if mutation=='exit_false':e['commands']['sacct']['exit']=False
    if mutation=='wrong_exit':e['commands']['sacct']['exit']=1
    if mutation=='dirty':e['commands']['code_status']['stdout']=' M scripts/tokenizer_preflight.py\n'
    if mutation=='grace':e['commands']['config']['stdout']='OverTimeLimit = 0 min'
    if mutation=='failed_job':e['commands']['sacct']['stdout']=e['commands']['sacct']['stdout'].replace('64596915|FAILED','64596915|COMPLETED')
    if mutation=='payload_pin':inputs[2]['caches']['T100-G']['ninapro_db3']['sha256']='b'*64
    if mutation=='sidecar':e['files'][M.OUTPUT+'p2/epn612.npz.provenance.json']['text']='{}'
    if mutation=='symlink':e['files'][M.OUTPUT+'pose/val.npz']['symlink']=True
    with pytest.raises(ValueError):M.build(*inputs)
