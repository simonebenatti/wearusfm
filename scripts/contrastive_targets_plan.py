"""Regenerate the prospective RMS100/fast125 plans; never release or submit them."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from contrastive_prepare import ROOT, PLAN, source_digest

FOLDER = PLAN.parent
PREFLIGHT = FOLDER/'preflight_plan.json'
AMENDMENT = 'contrastive-common-production-20261119-v2'
ROOT_OUT = '/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_targets_20261118'


def encoded(data):
    return (json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()


def science_plan(old):
    p = json.loads(json.dumps(old))
    p.update(status='REVIEW_REQUIRED', executable=False, training_authorized_by_this_file=False, release=None)
    p['scope'] = 'RMS100/fast125 new cohort; common-state parity and paired checkpoint restarts; operational review required'
    p['common']['numeric_profile'] = {'name': 'bf16_math_tf32_off', 'autocast_dtype': 'bfloat16',
        'sdpa': 'math_only', 'matmul_tf32': False, 'cudnn_tf32': False,
        'measure_production_batch_before_training_adoption': True}
    p['common']['benchmark_keep_checkpoints'] = [544, 5000, 19618]
    p['target_recipe_amendment']['known_unresolved_gate'] = (
        'Historical diagnostics concluded and read in faa4a8d7; old negative gates remain historical. '
        'New production parity, restart and throughput/memory receipts remain required.')
    p['target_recipe_amendment']['required_new_evidence'] = ['audit_completed_within_7200s', 'calibration',
        'lambda_CN', 'lambda_CR', 'cuda_check', 'three_full_benchmarks', 'four_checkpoint_reloads', 'benchmark_readout']
    p['numerical_amendment'] = {
        'id': AMENDMENT, 'status': 'REVIEW_REQUIRED', 'science_decision_commit': 'faa4a8d7',
        'preparation_review_commit': '94f50942',
        'neutrality': {'updates_before': [128,256], 'state': 'live C0 control before each update',
            'global_batch': 32, 'mixed_production_stream': True, 'backwards_per_arm': 2,
            'arms': ['C0','CN','CR'], 'required_profile': 'bf16_math_tf32_off',
            'criterion': 'reduced gradients byte-identical within and between arms and ranks',
            'restore': ['student','teacher','queue','batch','masks','CPU_RNG_per_rank','CUDA_RNG_per_rank'],
            'support': 'both contrastive losses supported; absent required support/profile is inconclusive',
            'trajectory_1_256': {'relative_tolerance':1e-4, 'absolute_tolerance':1e-6, 'role':'descriptive_only'},
            'stream_and_zero_weight_invariants': 'mandatory'},
        'resume': {'arms':['CN','CR'], 'fork_update':544, 'end_update':576,
            'continuous_saves_then_stays_live':True, 'fresh_reloads_per_arm':2,
            'same_checkpoint_bytes':True, 'three_pairs':['continuous/reload_a','continuous/reload_b','reload_a/reload_b'],
            'trace_per_rank_per_update':['batch','microbatch','CPU_RNG','CUDA_RNG','active_support'],
            'boundary':'student/teacher/optimizer/queue exact; separate rank CPU/CUDA RNG',
            'student_teacher_bound':.01, 'parameter_denominator':'full displacement from initialization of first member',
            'buffers':'exact', 'Adam_and_queue_float_bound_per_tensor':.01,
            'metadata_counters_order':'exact', 'fork_displacement':'separate descriptive distances from 544',
            'failure':'any violated historical bound blocks; absent profile/support inconclusive'},
        'release': {'first_phase':'audit_only_after_new_code_review', 'audit_state':'COMPLETED',
            'audit_exit_code':'0:0', 'audit_elapsed_max_s':7200,
            'review_binds':['reviewed_source_commit','source_sha256','science_plan_file_sha256','preflight_plan_file_sha256'],
            'historical_review_sufficient':False, 'training':'separate measured budget and successful preflight required'}
    }
    return p


def resources(cells, checkouts):
    gpu = sum(c['gpus']*c['seconds']/3600 for c in cells)
    core = sum(c['cpus']*c['seconds']/3600 for c in cells)+checkouts['count']*checkouts['cpus']*checkouts['seconds_each']/3600
    # Archived maximum and actual consumption are intentionally separate quantities.
    old=json.loads((ROOT/'results/infra/contrastive_comparison_20261116/common_state_diagnostics_20261118/resources.json').read_text())
    used_g=old['before_diagnostics_used']['gpu_hours']+4*(154+289+279)/3600
    used_c=old['before_diagnostics_used']['core_hours']+(4*35+32*(154+289+279))/3600
    extra_g=sum(c['gpus']*360/3600 for c in cells)
    extra_c=sum(c['cpus']*360/3600 for c in cells)+checkouts['count']*4*360/3600
    return {'gpu_hours_cap':gpu,'core_hours_cap':core,'local_hours_cap_assuming_billing_equals_cpus':core,
        'allocation_wall_hours_sum':sum(c['seconds']/3600 for c in cells)+checkouts['count']*240/3600,
        'measured_new_recipe_seconds':None,
        'historical_actual':{'gpu_hours':used_g,'core_hours':used_c,'local_hours':used_c,
            'diagnostics_gpu_hours':4*(154+289+279)/3600,'diagnostics_core_hours':(4*35+32*(154+289+279))/3600},
        'historical_reserved_cap':old['campaign_cap'],
        'historical_unspent_not_new_authorization':{'gpu_hours':old['campaign_cap']['gpu_hours']-used_g,
            'core_hours':old['campaign_cap']['core_hours']-used_c},
        'cumulative_actual_plus_new_requested_max':{'gpu_hours':used_g+gpu,'core_hours':used_c+core},
        'old_unexecuted_target_proposal':{'gpu_hours':14,'core_hours':137.06666666666666,'consumed':False},
        'new_increment_vs_old_target_proposal':{'gpu_hours':gpu-14,'core_hours':core-137.06666666666666},
        'slurm_extra_allowance':{'OverTimeLimit_s':180,'KillWait_s':180,'conservative_sum_per_allocation_s':360,
            'extra_gpu_hours':extra_g,'extra_core_hours':extra_c,
            'new_envelope_including_allowance':{'gpu_hours':gpu+extra_g,'core_hours':core+extra_c},
            'not_scientifically_accepted_runtime':'audit beyond7200s rejected even if COMPLETED'},
        'project_balance':{'value':None,'status':'3-launcher must refresh actual balance and pending reservations before submission',
            'remainder_formula':'fresh local-hour balance minus requested phase maximum and pending campaign writes/reservations'},
        'new_cohort_consumed':{'gpu_hours':0,'core_hours':0,'basis':'no submission by this delivery'},
        'long_training_budget':None}


def preflight(science):
    stages=[('AUDIT','audit',None,None,7200,[]), ('CALIB','calibration',None,None,1800,['audit']),
        ('CUDA','cuda',None,None,1800,['calibration_accepted'])]
    stages += [('BENCH-'+arm,'full',arm,None,1800,['cuda','calibration_accepted']) for arm in ('C0','CN','CR')]
    stages += [('RELOAD-'+arm+'-'+rep,'resume',arm,rep.lower(),1800,['full_'+arm]) for arm in ('CN','CR') for rep in ('A','B')]
    stages += [('READOUT','readout',None,None,3600,['all_three_full_and_four_reloads'])]
    cells=[]
    for label,mode,arm,rep,seconds,deps in stages:
        gpu=mode not in ('audit','readout')
        cells.append({'id':'CONTRAST-'+label+'-targets-v3-20261118','mode':mode,'arm':arm,'repeat':rep,
            'gpus':4 if gpu else 0,'cpus':32 if gpu else 8,'memory_gib':240 if gpu else 30,'seconds':seconds,
            'gpu_hours':4*seconds/3600 if gpu else 0,'core_hours':(32 if gpu else 8)*seconds/3600,
            'depends':deps or ['new_review','audit_checkout'], 'status':'REVIEW_REQUIRED','code_sha':None,'release':None})
    checkout={'count':6,'cpus':4,'memory_gib':2,'seconds_each':240,'core_hours':6*4*240/3600,
        'stages':['audit','calibration','cuda','full','reloads','readout'],
        'logical_ids':['CONTRAST-CHECKOUT-'+s+'-targets-v3-20261118' for s in ('audit','calibration','cuda','full','reloads','readout')],
        'mandatory_sbatch_overrides':['--cpus-per-task=4','--mem=2G','--time=00:04:00'],
        'reason':'separate immutable metadata releases after audit, calibration, CUDA receipt, full checkpoint, final receipts'}
    return {'protocol':'contrastive-target-windows-preflight-20261119-v2','status':'REVIEW_REQUIRED',
        'executable':False,'submission_authorized':False,'training_authorized':False,
        'science_plan':{'path':str(PLAN.relative_to(ROOT)),'sha256':hashlib.sha256(encoded(science)).hexdigest()},
        'source_sha256':source_digest(),'cells':cells,'checkouts':checkout,'resources':resources(cells,checkout),
        'fresh_root':ROOT_OUT,'mandatory_audit_sbatch_override':['--time=02:00:00'],
        'audit_sacct_gate':{'State':'COMPLETED','ExitCode':'0:0','ElapsedRaw_max':7200,'AllocCPUS':8,'TimelimitRaw':120},
        'operator':'3-launcher', 'required_before_first_submission':['new operational code review bound to actual source and both plans',
            'fresh ledger/quota/balance and pending writes checked by 3-launcher','first executable delivery is audit only'],
        'maintenance':{'timezone':'Europe/Rome','start':'2026-11-23T08:00:00+01:00','end':'2026-11-25T08:00:00+01:00',
            'source':'archived project operational instruction; launcher must refresh reservations',
            'rule':'no expected completion guaranteed across closure; queue after restart if phase cannot fit; no implicit retry'},
        'storage':{'minimum_new_headroom_gib':12,'pending_writes_additive':True,'verified_now':False},
        'failure_policy':'Failed or inconclusive gate stops descendants; no retry/time/backend/lambda/threshold change.',
        'old_evidence':'historical files retained unchanged; not qualification evidence for100/125',
        'training_release':'Absent; full-batch math time/memory, successful preflight, measured training budget required'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true'); args=parser.parse_args()
    science=science_plan(json.loads(PLAN.read_text())); operational=preflight(science)
    for path,data in ((PLAN,science),(PREFLIGHT,operational)):
        content=encoded(data)
        if args.check:
            if path.read_bytes()!=content: raise ValueError('stale generated plan: '+str(path))
        else: path.write_bytes(content)
    print(json.dumps({'source_sha256':source_digest(),'allocations':len(operational['cells'])+operational['checkouts']['count'],'resources':operational['resources']},indent=2))

if __name__=='__main__': main()
