"""Two-segment continuation of the frozen historical 300M recipe.

Only orchestration is implemented here. The historical trainer and the already
reviewed continuation utilities are imported from the pinned checkout. Each
process handles one segment and preserves the full resume state.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

NUMPY_BEFORE_TORCH = 'numpy' in sys.modules
import torch  # The original training entry point imports torch before NumPy.

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import scaling_continue as C

R, M, T, P, CLI, require = C.R, C.M, C.T, C.P, C.CLI, C.require
PROTOCOL = 'scaling-continuation49046-20261118-v1'
SEGMENTS = ((19618, 39237, 71100), (39237, 49046, 36000))
CONTROL_SHA256 = 'bfa6141fc2ed6505d2f582a46f0c04f36fd35f867e8cff2b11555b241947a490'
INITIAL_SHA256 = 'd7ebe9d87a08882ae0c1c836b62988ea21c7f1246998a5f82613878a0499387e'
IDENTITY_FILES = C.IDENTITY_FILES
MARKER = 'CONTINUATION49046.json'
EXECUTION_SOURCES = {'scripts/scaling_continue49046.py', 'scripts/scaling_continue.py',
                     'scripts/scaling_train.py', 'scripts/scaling_preflight.py',
                     'scripts/slurm/scaling_continue49046.sbatch'}


def segment(plan):
    start, end = plan.get('start_step'), plan.get('end_step')
    require(type(start) is int and type(end) is int, 'integer endpoints required')
    match = [row for row in SEGMENTS if row[:2] == (start, end)]
    require(len(match) == 1, 'unregistered segment')
    return match[0]


def checkpoint_name(step):
    require(step in (19618, 39237, 49046), 'unregistered checkpoint')
    return {19618: 'checkpoint_step0019618.pt', 39237: 'checkpoint_predecay.pt',
            49046: 'checkpoint_postdecay.pt'}[step]


def derive_recipe(plan, context):
    start, end, wall = segment(plan)
    previous = replace(context['config_300m'], max_steps=start)
    cfg = replace(previous, max_steps=end)
    require(cfg.schedule.total_steps == 49046 and cfg.schedule.decay_start == 39237 and
            cfg.warmup_steps == 1962 and cfg.schedule.decay_fraction == .2,
            'historical schedule differs')
    require(cfg.batch_size == 32 and cfg.accum_steps == 1 and cfg.seed == 0,
            'historical batch or seed differs')
    argv = list(context['comparison']['arm_configurations']['T100-G']['validated_trainer_argv'])
    for flag, value in (('--preset', '300m'), ('--out-dir', plan['run_dir']), ('--max-steps', str(end))):
        require(argv.count(flag) == 1, 'missing/duplicate argument '+flag)
        argv[argv.index(flag)+1] = value
    require(not any(x in argv for x in ('--resume', '--init-from', '--branch-from', '--time-limit-s')),
            'unexpected resume override')
    actual = CLI.build_config(CLI.build_parser().parse_args(argv[1:]))
    require(R.config_to_dict(actual) == R.config_to_dict(cfg), 'recipe differs beyond segment endpoint')
    argv += ['--time-limit-s', str(wall-1800)]
    return previous, cfg, argv


def validate_plan(plan, context):
    start, end, wall = segment(plan)
    require(plan.get('protocol') == PROTOCOL and plan.get('arm') == M.ARM,
            'wrong continuation protocol/arm')
    require(plan.get('additional_steps') == end-start, 'wrong update count')
    require(plan.get('scientific_source_sha256') == context['contract']['scientific_source_sha256'],
            'scientific source contract differs')
    allocation = dict(C.ALLOCATION, walltime_seconds=wall)
    require(plan.get('allocation') == allocation and plan.get('memory') == C.MEMORY,
            'wrong allocation or memory gate')
    require(plan.get('control_checkpoint', {}).get('sha256') == CONTROL_SHA256,
            'wrong 30M terminal control')
    require(plan.get('minimum_free_bytes') == 20 * 2**30, 'storage reserve must cover checkpoint copies')
    require(plan.get('protected_until_utc') == '2026-12-19T00:00:00Z', 'retention horizon differs')
    M.pin_schema(plan.get('resource_policy'))
    previous = plan['previous']
    for name in ('run_dir', 'receipt_dir'):
        path = Path(plan[name])
        require(path.is_absolute() and str(path) == str(path.resolve()) and
                not any(path.is_relative_to(root) for root in C.CODE_ROOTS), 'new canonical output outside code required')
    paths = [Path(plan['run_dir']), Path(plan['receipt_dir']), Path(previous['run_dir'])]
    require(all(not a.is_relative_to(b) and not b.is_relative_to(a)
                for i,a in enumerate(paths) for b in paths[i+1:]), 'overlapping run/receipt paths')
    require(set(previous['files']) >= {'checkpoint.pt', checkpoint_name(start), *IDENTITY_FILES},
            'incomplete source inventory')
    if start == 19618:
        require(previous['files']['checkpoint.pt']['sha256'] == INITIAL_SHA256 and
                previous['files'][checkpoint_name(start)]['sha256'] == INITIAL_SHA256,
                'wrong accepted 19618 checkpoint')
    else:
        require(M.digest(plan.get('predecessor_plan_sha256')) and previous.get('receipt'),
                'second segment requires its predecessor receipt and plan identity')
    for pin in (plan['source_contract'], plan['control_checkpoint'], previous['receipt']):
        M.pin_schema(pin)
    sources = plan.get('execution_source_sha256')
    require(isinstance(sources, dict) and set(sources) == EXECUTION_SOURCES, 'missing exact wrapper source pins')
    for path, sha in sources.items():
        require(M.digest(sha), 'invalid source digest')
        M._contained_source(path, ROOT)
    return derive_recipe(plan, context)


def resolve_previous(plan, code):
    """Bind the second segment to the completed first receipt from this release."""
    if segment(plan)[0] == 19618:
        return plan, None
    raw = plan['previous']
    require(set(raw) == {'run_dir', 'receipt', 'files_from_receipt'} and
            raw['files_from_receipt'] is True and set(raw['receipt']) == {'path'},
            'second segment source must be derived from its first receipt')
    pin = P.file_identity(raw['receipt']['path'])
    receipt = json.loads(Path(pin['path']).read_text())
    job = os.environ.get('SCALING_PREDECESSOR_JOB', '')
    require(job.isdigit() and receipt.get('slurm_job_id') == job, 'predecessor job differs from submitted dependency')
    accounting = subprocess.check_output(['sacct','-n','-P','-X','-j',job,'--format=JobIDRaw,State,ExitCode'],text=True)
    rows = [line.split('|')[:3] for line in accounting.strip().splitlines()]
    require(rows == [[job,'COMPLETED','0:0']], 'predecessor job did not complete successfully')
    require(receipt.get('code_sha') == code and receipt.get('plan_sha256') == plan['predecessor_plan_sha256'],
            'predecessor release differs')
    require(receipt.get('checkpoint') == str(Path(raw['run_dir'])/'checkpoint_predecay.pt'),
            'predecessor native checkpoint differs')
    result = dict(plan)
    result['previous'] = {'run_dir':raw['run_dir'], 'receipt':{k:pin[k] for k in ('path','sha256','bytes')},
                          'files':receipt['run_inventory']}
    previous_receipt(result,receipt)
    return result, pin


def load(plan_path, expected_code):
    require(NUMPY_BEFORE_TORCH is False, 'torch must precede numpy')
    code = P.code_identity(expected_code)
    state = P.file_identity(plan_path)
    plan = json.loads(Path(state['path']).read_text())
    plan, predecessor_pin = resolve_previous(plan, code)
    contract = M.resolve_pin(plan['source_contract'])
    context = M.check_source_contract(contract['path'])
    old, cfg, argv = validate_plan(plan, context)
    return code, state, plan, contract, context, old, cfg, argv


def previous_receipt(plan, receipt):
    start, _, _ = segment(plan)
    expected_protocol = C.PROTOCOL if start == 19618 else PROTOCOL
    require(receipt.get('protocol') == expected_protocol and receipt.get('complete') is True and
            receipt.get('arm') == M.ARM and receipt.get('checkpoint_step') == start and
            receipt.get('scientific_endpoint') is True and receipt.get('previous_run_unchanged') is True,
            'predecessor did not complete the exact endpoint')
    require(receipt.get('resume_checkpoint') == str(Path(plan['previous']['run_dir'])/'checkpoint.pt') and
            receipt.get('resume_checkpoint_sha256') == plan['previous']['files']['checkpoint.pt']['sha256'],
            'predecessor resume checkpoint differs')
    if start == 19618:
        require(receipt.get('slurm_job_id') == '63979977', 'wrong accepted 19618 source job')
    require(receipt.get('checkpoint') == str(Path(plan['previous']['run_dir'])/checkpoint_name(start)) and
            receipt.get('checkpoint_sha256') == plan['previous']['files'][checkpoint_name(start)]['sha256'],
            'predecessor native checkpoint differs')
    if start == 39237:
        require(receipt.get('start_step') == 19618 and receipt.get('additional_steps') == 19619 and
                receipt.get('plan_sha256') == plan['predecessor_plan_sha256'] and
                receipt.get('control_checkpoint_sha256') == CONTROL_SHA256,
                'predecessor segment, plan or control differs')


def verify_endpoint(plan, cfg, signature, before, code, job, plan_sha):
    start, end, _ = segment(plan)
    run, out = Path(plan['run_dir']), Path(plan['receipt_dir'])
    expected = {'checkpoint.pt', checkpoint_name(end), 'config.json', 'fixed_eval.jsonl', 'metrics.jsonl',
                'pose_panel.jsonl', 'pose_provenance.json', 'summary.json', MARKER, *IDENTITY_FILES}
    require(set(p.name for p in run.iterdir()) == expected, 'unexpected or incomplete output inventory')
    summary = json.loads((run/'summary.json').read_text())
    require(summary.get('stopped') == 'max_steps' and summary.get('steps') == end and
            summary.get('world_size') == 4 and summary.get('parameters') == C.PARAMETERS and
            summary.get('accum_steps') == 1 and summary.get('micro_batch') == 8,
            'segment incomplete or batch changed')
    require(json.loads((run/'config.json').read_text()) == R.config_to_dict(cfg), 'output config differs')
    named, final = run/checkpoint_name(end), run/'checkpoint.pt'
    states = [T.checkpoint_identity(p, cfg, end, signature) for p in (named, final)]
    require(states[0]['trained_state_sha256'] == states[1]['trained_state_sha256'], 'native and resume states differ')
    after = C.optimizer_mapping(torch.load(final, map_location='cpu', weights_only=False), cfg, C.PARAMETERS)
    C.check_optimizer_continuity(before, after, end-start)
    for name in IDENTITY_FILES:
        require(R.sha256_file(run/name) == plan['previous']['files'][name]['sha256'], 'identity changed '+name)
    return {'summary': summary, 'named': named, 'final': final,
            'trained_state_sha256': states[1]['trained_state_sha256'],
            'optimizer': {k:v for k,v in after.items() if k != 'adam_steps'},
            'tranche': C.check_tranche_logs(run,plan['previous']['run_dir'],cfg,start,end),
            'memory': C.check_rank_records(out,summary,plan,code,job,plan_sha)}


def verify_identity_sets(cfg, argv, plan):
    """Use the trainer CLI configuration to recompute its diagnostic identities."""
    expected = {name:json.loads((Path(plan['previous']['run_dir'])/name).read_text()) for name in IDENTITY_FILES}
    computed = {}
    def capture(cfg_cli, manifest, roots, out_dir, *, scales=None, window_s=None, **options):
        require(R.config_to_dict(cfg_cli) == R.config_to_dict(cfg), 'identity CLI differs')
        computed.update(C.identity_sets(cfg_cli,manifest,roots,scales,window_s))
        return {'identity_precheck': True}
    old = R.train
    R.train = capture
    try:
        require(CLI.main(C.precheck_argv(argv,Path(plan['receipt_dir'])/'identity_only')[1:]) == 0,
                'identity CLI failed')
    finally:
        R.train = old
    require(expected == computed, 'fixed diagnostic identities differ from accepted source')
    return {name: C.compare_identity(expected[name],computed[name]) for name in IDENTITY_FILES}


def worker(plan_path, expected_code, launch_sha):
    code, state, plan, _, _, _, _, argv = load(plan_path,expected_code)
    rank,local,world = (int(os.environ.get(k,'-1')) for k in ('RANK','LOCAL_RANK','WORLD_SIZE'))
    require(world == 4 and 0 <= rank < 4 and rank == local, 'four single-node ranks required')
    out = Path(plan['receipt_dir'])
    launch = M.resolve_pin({'path':str(out/'launch.json'),'sha256':launch_sha})
    record = json.loads(Path(launch['path']).read_text())
    marker = json.loads((Path(plan['run_dir'])/MARKER).read_text())
    for item in (record,marker):
        require(item['protocol']==PROTOCOL and item['plan_sha256']==state['sha256'] and
                item['code_sha']==code and item['slurm_job_id']==os.environ['SLURM_JOB_ID'], 'worker identity differs')
    require(record['trainer_argv']==argv and not (Path(plan['run_dir'])/'STOP').exists(), 'worker arguments or STOP differ')
    result = CLI.main(argv[1:])
    torch.cuda.synchronize(local)
    P.write_json(out/f'rank_{rank}.json',{'protocol':C.RANK_PROTOCOL,'rank':rank,'local_rank':local,'world_size':world,
        'returncode':result,'code_sha':code,'slurm_job_id':os.environ['SLURM_JOB_ID'],'plan_sha256':state['sha256'],
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(local),'peak_reserved_bytes':torch.cuda.max_memory_reserved(local),
        'device_memory_bytes':torch.cuda.get_device_properties(local).total_memory})
    require(result==0, 'trainer failed')


def execute(plan_path, expected_code, out):
    code, plan_state, plan, contract_state, context, previous_cfg, cfg, argv = load(plan_path,expected_code)
    start,end,wall = segment(plan)
    job = C.verify_allocation(plan)
    out,run = Path(out),Path(plan['run_dir'])
    require(str(out)==plan['receipt_dir'], 'wrong receipt directory')
    for path in (out,run):
        require(not path.exists() and not path.is_symlink(), 'existing output requires investigation')
        require(shutil_disk_free(path.parent)>=plan['minimum_free_bytes'], 'insufficient checkpoint space')
    states = dict(context['states'])
    states['plan'] = plan_state
    states['resource_policy'] = M.resolve_pin(plan['resource_policy'])
    policy = json.loads(Path(states['resource_policy']['path']).read_text())
    expected_segments = [{'start_step':a,'end_step':b,'walltime_seconds':w,'trainer_time_limit_seconds':w-1800} for a,b,w in SEGMENTS]
    require(policy.get('protocol') == 'scaling49046-resources-20261118-v1' and policy['training']['segments'] == expected_segments and
            policy['training']['gpu_hours_requested'] == 119 and policy['training']['gpus'] == 4 and
            policy['training']['cpus'] == 32, 'training resource freeze differs')
    for name,sha in plan['execution_source_sha256'].items():
        states['source:'+name]=M.resolve_pin({'path':str(ROOT/name),'sha256':sha})
    states.update(C.previous_states(plan['previous']))
    old_receipt = M.resolve_pin(plan['previous']['receipt'])
    states['previous_receipt']=old_receipt
    previous_receipt(plan,json.loads(Path(old_receipt['path']).read_text()))
    control = M.resolve_pin(plan['control_checkpoint'])
    states['control_checkpoint']=control
    require(control['bytes'] == 454381728, 'control checkpoint size differs')
    inputs=context['contract']['inputs']
    for name,pin in inputs.items():states['input:'+name]=M.resolve_pin(pin)
    scales=json.loads(Path(inputs['scales']['path']).read_text())['scales']
    window=json.loads(Path(inputs['windows']['path']).read_text())['window_s']
    signature=json.loads(json.dumps(R.calibration_signature(cfg,inputs['manifest']['path'],scales,window)))
    source=Path(plan['previous']['run_dir'])
    control_cfg = replace(context['config_30m'], max_steps=49046)
    control_signature = json.loads(json.dumps(R.calibration_signature(control_cfg,inputs['manifest']['path'],scales,window)))
    T.checkpoint_identity(control['path'],control_cfg,49046,control_signature)
    old_named=T.checkpoint_identity(source/checkpoint_name(start),previous_cfg,start,signature)
    old_resume=T.checkpoint_identity(source/'checkpoint.pt',previous_cfg,start,signature)
    require(old_named['trained_state_sha256']==old_resume['trained_state_sha256'], 'source native/resume states differ')
    before=C.optimizer_mapping(torch.load(source/'checkpoint.pt',map_location='cpu',weights_only=False),previous_cfg,C.PARAMETERS)
    identity_sets=verify_identity_sets(cfg,argv,plan)
    retention=C.check_retention([v['path'] for v in states.values() if 'path' in v],
        datetime.fromisoformat(plan['protected_until_utc'].replace('Z','+00:00')),
        datetime.now(timezone.utc),wall)
    P.check_unchanged(states)
    out.mkdir(parents=True,exist_ok=False)
    seeded=C.seed_run_dir(plan['previous'],run)
    identity={'protocol':PROTOCOL,'arm':M.ARM,'start_step':start,'end_step':end,'code_sha':code,
              'plan_sha256':plan_state['sha256'],'slurm_job_id':job}
    P.write_json(run/MARKER,{**identity,'seeded_files':seeded})
    P.write_json(out/'launch.json',{**identity,'trainer_argv':argv,'identity_sets':identity_sets,
        'import_order':{'numpy_loaded_before_torch':NUMPY_BEFORE_TORCH},
        'earliest_scratch_expiry':retention.isoformat() if retention else None,'verified_inputs':states})
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
             str(ROOT/'scripts/scaling_continue49046.py'),'worker','--plan',plan_state['path'],
             '--expected-code-sha',code,'--launch-sha256',R.sha256_file(out/'launch.json')]
    with (out/'training.log').open('x') as log:
        subprocess.run(command,cwd=ROOT,env={**os.environ,'WEARUSFM_CODE_SHA':code,'WEARUSFM_CODE_DIR':str(ROOT)},
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    checked=verify_endpoint(plan,cfg,signature,before,code,job,plan_state['sha256'])
    P.check_unchanged(states)
    require(set(x.name for x in source.iterdir())==set(plan['previous']['files']), 'source inventory changed')
    P.code_identity(code)
    P.write_json(out/'config.json',R.config_to_dict(cfg))
    P.write_json(out/'summary.json',checked['summary'])
    receipt=T.receipt_for(M.ARM,end,checked['named'],checked['final'],out/'config.json',code,plan_state['sha256'],command)
    inventory={p.name:{k:v for k,v in P.file_identity(p).items() if k in ('sha256','bytes')} for p in run.iterdir()}
    receipt.update(protocol=PROTOCOL,start_step=start,additional_steps=end-start,trainer_argv=argv,
        scientific_endpoint=True,previous_run_unchanged=True,previous_checkpoint_sha256=plan['previous']['files']['checkpoint.pt']['sha256'],
        previous_receipt_sha256=old_receipt['sha256'],scientific_source_sha256=context['contract']['scientific_source_sha256'],
        training_source_contract_sha256=contract_state['sha256'],control_checkpoint_sha256=CONTROL_SHA256,
        trained_state_sha256=checked['trained_state_sha256'],optimizer=checked['optimizer'],tranche=checked['tranche'],
        memory=checked['memory'],run_inventory=inventory,previous_run_dir=str(source),further_training_authorized=False)
    P.write_json(out/'receipt.json',receipt)
    return receipt


def shutil_disk_free(parent):
    import shutil
    while not parent.exists():parent=parent.parent
    return shutil.disk_usage(parent).free


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('run','worker','check-plan'))
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--expected-code-sha',required=True)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--launch-sha256')
    args=parser.parse_args(argv)
    if args.mode=='run':
        require(args.out is not None and args.launch_sha256 is None,'run requires output only')
        execute(args.plan,args.expected_code_sha,args.out)
    elif args.mode=='worker':
        require(args.out is None and M.digest(args.launch_sha256),'worker requires launch pin only')
        worker(args.plan,args.expected_code_sha,args.launch_sha256)
    else:
        require(args.out is None and args.launch_sha256 is None,'check-plan has no outputs')
        checked=load(args.plan,args.expected_code_sha)
        print(json.dumps({'valid':True,'start_step':checked[2]['start_step'],'end_step':checked[2]['end_step']}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
