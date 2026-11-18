"""Registered long segments and exact CPU resume across the native WSD decay."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from wearusfm.training import run as R
from test_pretraining_loader import _tree
from test_scaling_continue import _config, _records, _tree_hash, file_pin, NAMES

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import scaling_continue49046 as L


@pytest.mark.parametrize('start,end,wall',L.SEGMENTS)
def test_registered_segment_and_native_checkpoint(start,end,wall):
    assert L.segment({'start_step':start,'end_step':end})==(start,end,wall)
    assert L.checkpoint_name(end)==('checkpoint_predecay.pt' if end==39237 else 'checkpoint_postdecay.pt')


@pytest.mark.parametrize('pair',[(19618,49046),(5000,19618),(39237,50000),(19618.,39237)])
def test_unregistered_segments_are_refused(pair):
    with pytest.raises(ValueError):L.segment(dict(zip(('start_step','end_step'),pair)))


def test_original_schedule_crosses_decay_without_restart():
    cfg=R.with_wsd(R.small_config(datasets=None),49046,warmup_steps=1962,keep_checkpoints=(5000,19618))
    assert cfg.schedule.decay_start==39237
    assert R._lr_at(39237,cfg)==cfg.lr
    assert R._lr_at(39238,cfg)<cfg.lr
    assert R._lr_at(49045,cfg)>=0


def test_resume_new_directory_across_decay_matches_uninterrupted_training(tmp_path):
    root,manifest=_tree(tmp_path)
    def config(max_steps):
        return replace(_config(max_steps),schedule=R.Schedule('wsd',20,.2,(4,8)),ckpt_every=100)
    whole,early,late=(tmp_path/n for n in ('whole','early','late'))
    R.train(config(20),manifest,[root],whole,log=lambda m:None,num_workers=0)
    R.train(config(16),manifest,[root],early,log=lambda m:None,num_workers=0)
    assert (early/'checkpoint_predecay.pt').exists()
    frozen=_tree_hash(early)
    L.C.seed_run_dir({'run_dir':str(early),'files':{n:file_pin(early/n) for n in NAMES}},late,NAMES)
    R.train(config(20),manifest,[root],late,log=lambda m:None,num_workers=0)
    assert _tree_hash(early)==frozen and (late/'checkpoint_postdecay.pt').exists()
    assert _records(late)==[r for r in _records(whole) if r['step']>16]
    a,b=(torch.load(p/'checkpoint.pt',weights_only=False) for p in (whole,late))
    assert a['step']==b['step']==20 and a['alarm_counts']==b['alarm_counts']
    for part in ('student','teacher'):
        assert all(torch.equal(a[part][k],b[part][k]) for k in a[part])
    assert a['optimizer']['param_groups']==b['optimizer']['param_groups']
    for k,v in a['optimizer']['state'].items():
        assert all(torch.equal(v[n],b['optimizer']['state'][k][n]) for n in ('step','exp_avg','exp_avg_sq'))
    assert torch.equal(a['torch_rng'],b['torch_rng'])
    assert [r['step'] for r in _records(late)]==[17,18,19,20]
