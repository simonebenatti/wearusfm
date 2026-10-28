"""Offline guards for the single frozen B5000 ->19618 continuation launcher."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT/'scripts/slurm/pose_b19618_continue.sbatch'


def launcher():
    text = SCRIPT.read_text()
    payload = text.split("<<'PY_B19618'\n",1)[1].rsplit('\nPY_B19618',1)[0]
    module = types.ModuleType('b19618_launcher')
    exec(compile(payload,str(SCRIPT),'exec'),module.__dict__)
    return module


def frozen_cli():
    spec = importlib.util.spec_from_file_location('train_jepa_b19618',ROOT/'scripts/train_jepa.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shell_syntax_resource_caps_and_no_alternate_resume():
    subprocess.run(['bash','-n',str(SCRIPT)],check=True)
    text = SCRIPT.read_text()
    for directive in ('--gres=gpu:4','--cpus-per-task=32','--mem=480G','--time=04:00:00','--qos=normal'):
        assert '#SBATCH '+directive in text
    assert '--branch-from' not in text and '--init-from' not in text and 'os.link' not in text
    assert "'--nproc_per_node=4'" in text


def test_exact_B_recipe_only_maxsteps_changes_schedule_identical():
    from wearusfm.training import run as R
    m, cli = launcher(), frozen_cli()
    work = Path('/leonardo_work/IscrB_WearUsFM')
    scratch = Path('/leonardo_scratch/large/userexternal/sbenatti')
    args = cli.build_parser().parse_args(m.trainer_args(work,scratch,Path('/new/run')))
    new = R.config_to_dict(cli.build_config(args))
    old = json.loads((ROOT/'results/infra/pilot_posa/collaudo_b_20261028/config.json').read_text())
    m.check_recipe(old,new)
    assert new == {**old,'max_steps':19618}
    assert args.num_workers == 6 and args.distributed and args.time_limit_s == 13800
    cfg = cli.build_config(args)
    assert cfg.schedule.decay_start == 39237
    assert all(R._lr_at(s,cfg) == 1e-4 for s in (4999,5000,19617,19618))
    assert R._lr_at(49045,cfg) < 1e-4
    bad = deepcopy(new)
    bad['schedule']['total_steps'] = 19618
    with pytest.raises(ValueError,match='beyond max_steps'):
        m.check_recipe(old,bad)
    bad = deepcopy(new)
    bad['warmup_steps'] = 500
    with pytest.raises(ValueError,match='beyond max_steps'):
        m.check_recipe(old,bad)


def test_fresh_destination_rejects_existing_symlink_source_or_code(tmp_path):
    m = launcher()
    source = tmp_path/'B'
    source.mkdir()
    code = tmp_path/'code'
    code.mkdir()
    for destination in (source,source/'child',code/'run',Path('relative')):
        with pytest.raises(ValueError):
            m.fresh_destination(destination,source,[code])
    link = tmp_path/'broken'
    link.symlink_to(tmp_path/'missing')
    with pytest.raises(ValueError):
        m.fresh_destination(link,source,[code])
    m.fresh_destination(tmp_path/'new',source,[code])


def test_seed_copy_is_byteidentical_independent_and_preserves_source(tmp_path):
    m = launcher()
    source, dest = tmp_path/'B',tmp_path/'new'
    source.mkdir()
    # Include optimizer, EMA and all four rank RNG tensors in the byte-copy test.
    state = {'step':5000,'student':torch.randn(4),'teacher':torch.randn(4),
             'optimizer':{'exp_avg':torch.randn(4),'exp_avg_sq':torch.rand(4)},
             'rank_rng':[{'torch_rng':torch.get_rng_state(),'cuda_rng':torch.arange(8,dtype=torch.uint8)} for _ in range(4)]}
    torch.save(state,source/'checkpoint.pt')
    for name in ('pose_panel_set.json','fixed_eval_set.json'):
        (source/name).write_text('{"identity":"frozen"}')
    hashes = {p.name:m.sha(p) for p in source.iterdir()}
    m.seed_destination(source,dest,hashes)
    assert {p.name:m.sha(p) for p in dest.iterdir()} == hashes
    for name in hashes:
        assert (source/name).stat().st_ino != (dest/name).stat().st_ino
    (dest/'checkpoint.pt').write_bytes(b'new future checkpoint')
    assert all(m.sha(source/name) == value for name,value in hashes.items())
    with pytest.raises(FileExistsError):
        m.seed_destination(source,dest,hashes)


def synthetic_state(m):
    return {'step':5000,'config':{'max_steps':5000},'pose_stats_sha256':m.STATS_SHA,
            'stream':{'num_workers':6,'segment':1200,'world_size':4},'world_size':4,
            'rank_rng':[{'torch_rng':torch.get_rng_state(),'cuda_rng':torch.ones(4,dtype=torch.uint8)} for _ in range(4)],
            'student':{'w':torch.ones(2)},'teacher':{'w':torch.zeros(2)},
            'optimizer':{'state':{0:{'exp_avg':torch.ones(2)}},'param_groups':[{'params':[0]}]},
            'data_signature':{'manifest_sha256':m.MANIFEST_SHA},'alarm_counts':(0,0)}


@pytest.mark.parametrize('key,value', [('step',0),('world_size',2),('rank_rng',[]),('optimizer',{}),
                                     ('data_signature',{}),('alarm_counts',()),
                                     ('stream',{'num_workers':5,'segment':1200,'world_size':4})])
def test_incomplete_or_incompatible_resume_state_rejected(key,value):
    m = launcher()
    state = synthetic_state(m)
    m.check_checkpoint(state,{'max_steps':5000},5000,torch)
    state[key] = value
    with pytest.raises(ValueError):
        m.check_checkpoint(state,{'max_steps':5000},5000,torch)


def test_cuda_rng_missing_or_nonfinite_model_rejected():
    m = launcher()
    state = synthetic_state(m)
    state['rank_rng'][3]['cuda_rng'] = None
    with pytest.raises(ValueError,match='RNG'):
        m.check_checkpoint(state,{'max_steps':5000},5000,torch)
    state = synthetic_state(m)
    state['teacher']['w'][0] = float('nan')
    with pytest.raises(ValueError,match='model tensors'):
        m.check_checkpoint(state,{'max_steps':5000},5000,torch)


def test_completion_requires_exact_step_maxsteps_and_frozen_identity():
    m = launcher()
    good = {'steps':19618,'stopped':'max_steps','world_size':4,'code':{'sha':m.CODE_SHA},'pose_stats_sha256':m.STATS_SHA}
    assert m.summary_complete(good)
    for change in ({'steps':19617},{'stopped':'limite di tempo'},{'world_size':1},{'code':{'sha':'wrong'}}):
        assert not m.summary_complete({**good,**change})


def test_frozen_source_guards_match_archived_B_files():
    m = launcher()
    source = ROOT/'results/infra/pilot_posa/collaudo_b_20261028'
    for name,digest in m.SOURCE_HASHES.items():
        if name != 'checkpoint.pt':
            assert m.sha(source/name) == digest
