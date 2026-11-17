"""Budget envelope mutations, using the real frozen job table as input."""
import copy
from fractions import Fraction
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import scaling_p1_reader_budget as B
import scaling_plans19618 as S


@pytest.fixture
def tables():
    result={}
    for stage in B.STAGES:
        if stage.startswith('checkout-'):
            result[stage]=[S.checkout_job(stage.removeprefix('checkout-'))]
        elif stage=='probe-fit':
            result[stage]=[]
        elif stage=='probe-aggregate':
            result[stage]=S.jobs(stage,{'protocol':S.C.RESOURCE_PROTOCOL,'shards':[]})
        else:
            result[stage]=S.jobs(stage)
    return result


def ledger(job,slurm='12345'):
    return {'id':job['id'],'stato':'sottomesso','agente':'gpt','campagna':B.CAMPAIGN,
            'job_slurm':slurm,'gpu_ore_stimate':0,
            'sbatch':['sbatch','--parsable','--cpus-per-task='+str(job['cpus']),
                      '--mem='+job['memory'],'--time='+B._time(job['seconds']),'/dev/stdin']}


def accounted(job,slurm='12345',elapsed=30):
    return {'JobIDRaw':str(slurm),'State':'COMPLETED','ExitCode':'0:0','ElapsedRaw':elapsed,
            'AllocCPUS':job['cpus'],'gpus':0,'core_hours':elapsed*job['cpus']/3600,
            'AllocTRES':'cpu=%s,mem=%s,node=1'%(job['cpus'],job['memory']),
            'Timelimit':B._time(job['seconds']),
            'raw_sacct':{'path':'sacct.txt','sha256':'a'*64}}


def with_fits(tables,count=15,seconds=7200):
    tables=copy.deepcopy(tables)
    tables['probe-fit']=[S._job('probe-fit','SCALING-19618-probe-fit-%02d-20261106'%n,
                        8,seconds,'/unused/'+str(n),str(n)) for n in range(1,count+1)]
    return tables


def test_real_job_tables_preserve_approved_global_envelope_without_duplicate_checkouts(tables):
    result=B.evaluate_budget(tables,{}, {},fit_plan_ready=False)
    assert result['complete'] is True and result['within_total_cap'] is True
    assert Fraction(result['remaining_chain']['exact'])==Fraction(2071,6)
    assert Fraction(result['total']['exact'])==Fraction('349.917778')+Fraction((41+34)*2,3600)
    assert result['fixed_prior_reserve']=={'exact':'2396389/500000','core_hours':4.792778}
    assert result['total']['core_hours']>350-1 and result['cap_core_hours']==351
    assert sum(len(tables[k]) for k in B.STAGES if k.startswith('checkout-'))==3
    assert sum(Fraction(result['stages'][k]['exact']) for k in B.STAGES if k.startswith('checkout-'))==Fraction(4,5)
    assert result['unknown_fit_reserve']=={'exact':'246','core_hours':246.0}


@pytest.mark.parametrize('total',[Fraction(350),Fraction(351)])
def test_approved_global_cap_is_351_not_old_350(total):
    assert B.require_total(total) is True


@pytest.mark.parametrize('total',[Fraction(351)+Fraction(1,10**12),Fraction(-1),351.0])
def test_global_cap_has_no_rounding_extension(total):
    with pytest.raises(ValueError):B.require_total(total)


def test_submitted_without_collected_accounting_is_fully_reserved(tables):
    job=tables['probe-benchmark'][0]
    result=B.evaluate_budget(tables,{job['id']:ledger(job)}, {},fit_plan_ready=False)
    assert Fraction(result['total']['exact'])==B.BASELINE
    assert result['jobs'][job['id']]['status']=='reserved'
    assert result['jobs'][job['id']]['submitted'] is True
    assert result['actual_jobs']==0


def test_verified_actual_replaces_reservation_once(tables):
    job=tables['checkout-plans'][0]
    result=B.evaluate_budget(tables,{job['id']:ledger(job)}, {job['id']:accounted(job,elapsed=34)},fit_plan_ready=False)
    expected=B.BASELINE-Fraction(2*480,3600)+Fraction(2*34,3600)
    assert Fraction(result['total']['exact'])==expected
    assert result['actual_jobs']==1
    assert result['jobs'][job['id']]['status']=='actual'


def test_fit_plan_replaces_full_placeholder_with_its_actual_requested_allocations(tables):
    fits=with_fits(tables,count=10,seconds=10800)
    result=B.evaluate_budget(fits,{}, {},fit_plan_ready=True)
    assert Fraction(result['stages']['probe-fit']['exact'])==244
    assert Fraction(result['total']['exact'])==B.BASELINE-2
    assert result['unknown_fit_reserve'] is None
    all_fifteen=B.evaluate_budget(with_fits(tables),{}, {},fit_plan_ready=True)
    assert Fraction(all_fifteen['total']['exact'])==B.BASELINE


@pytest.mark.parametrize('count,seconds',[(16,600),(1,10801),(11,10800),(0,7200)])
def test_fit_count_per_job_and_total_stage_caps_stay_frozen(tables,count,seconds):
    with pytest.raises(ValueError):
        B.evaluate_budget(with_fits(tables,count,seconds),{}, {},fit_plan_ready=True)


def test_missing_fit_plan_cannot_adopt_or_submit_fit_jobs(tables):
    job=with_fits(tables)['probe-fit'][0]
    with pytest.raises(ValueError):
        B.evaluate_budget(tables,{job['id']:ledger(job)}, {},fit_plan_ready=False)
    with pytest.raises(ValueError):
        B.evaluate_budget(with_fits(tables),{}, {},fit_plan_ready=False)


@pytest.mark.parametrize('mutate',[
    lambda t:t.pop('checkout-inputs'),
    lambda t:t.update(extra=[]),
    lambda t:t['pose-benchmark'].pop(),
    lambda t:t['pose-benchmark'].append(copy.deepcopy(t['pose-benchmark'][0])),
    lambda t:t['probe-benchmark'][0].update(seconds=1),
    lambda t:t['pose-validate'][0].update(cpus=8),
    lambda t:t['checkout-plans'][0].update(seconds=600),
    lambda t:t['compare'][0].update(gpus=1),
    lambda t:t['pose-ridge'][0].update(memory='40G'),
    lambda t:t['pose-ridge'][0].update(cpus=True),
    lambda t:t['pose-ridge'][0].update(id='SCALING-new-retry'),
])
def test_scope_cannot_be_reduced_or_expanded_to_manipulate_budget(tables,mutate):
    mutate(tables)
    with pytest.raises(ValueError):B.evaluate_budget(tables,{}, {},fit_plan_ready=False)


@pytest.mark.parametrize('key,value',[
    ('stato','ambiguo'),('stato','rifiutato'),('stato','in_sottomissione'),('stato','prenotato'),
    ('stato','assente'),('agente','unknown'),('campagna','another'),('job_slurm',None),
    ('job_slurm',True),('gpu_ore_stimate',1),('id','unrelated'),
])
def test_uncertain_or_foreign_ledger_stops_the_chain(tables,key,value):
    job=tables['pose-validate'][0];state=ledger(job);state[key]=value
    with pytest.raises(ValueError):B.evaluate_budget(tables,{job['id']:state},{},fit_plan_ready=False)


@pytest.mark.parametrize('option',['--cpus-per-task=8','--mem=4G','--time=00:05:00'])
def test_ledger_resources_cannot_underreserve_actual_submission(tables,option):
    job=tables['pose-validate'][0];state=ledger(job);prefix=option.split('=')[0]+'='
    state['sbatch']=[option if x.startswith(prefix) else x for x in state['sbatch']]
    with pytest.raises(ValueError):B.evaluate_budget(tables,{job['id']:state},{},fit_plan_ready=False)


@pytest.mark.parametrize('key,value',[
    ('State','FAILED'),('State','TIMEOUT'),('State','RUNNING'),('State','PENDING'),('ExitCode','1:0'),
    ('JobIDRaw','12346'),('ElapsedRaw',1801),('ElapsedRaw',0),('ElapsedRaw',True),('AllocCPUS',8),
    ('gpus',1),('AllocTRES','cpu=2,mem=30G,node=1,gres/gpu=1'),
    ('AllocTRES','cpu=2,mem=40G,node=1'),('Timelimit','01:00:00'),('raw_sacct',{}),
    ('core_hours',0),('core_hours',float('nan')),
])
def test_unaccepted_accounting_never_reduces_reserve(tables,key,value):
    job=tables['pose-validate'][0];record=accounted(job);record[key]=value
    with pytest.raises(ValueError):
        B.evaluate_budget(tables,{job['id']:ledger(job)},{job['id']:record},fit_plan_ready=False)


def test_accounting_without_ledger_and_duplicate_scheduler_identity_refused(tables):
    a,b=tables['pose-benchmark'][:2]
    with pytest.raises(ValueError):
        B.evaluate_budget(tables,{}, {a['id']:accounted(a)},fit_plan_ready=False)
    with pytest.raises(ValueError):
        B.evaluate_budget(tables,{a['id']:ledger(a),b['id']:ledger(b)}, {},fit_plan_ready=False)
