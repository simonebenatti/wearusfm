from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"scripts"))
from pose_temponet import benchmark, controls, data as data_api, linear


def fixture(n=30, time=20, features=8, joints=2):
    rng=np.random.default_rng(71)
    target=rng.normal(size=(n,time,joints)).astype(np.float32)
    return {"config":{"split":"train","which":"teacher","checkpoint_step":49046},
            "lat":rng.normal(size=(n,time,features)).astype(np.float16),
            "target":target,"mask":np.ones_like(target,dtype=bool),
            "std":np.ones((n,joints)),"pose_var":np.ones(n),
            "meta":[{"user":f"u{i//2:02d}","n_patches":time,"window":i} for i in range(n)]}


def test_gpu_benchmark_cannot_substitute_cpu_measurements(monkeypatch,tmp_path):
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    with pytest.raises(RuntimeError,match="no CPU timing fallback"):
        benchmark.benchmark_gpu({},input_loader=lambda:{},output_dir=tmp_path/"b",
                                reference_byte_count=1,reference_proxy_path=tmp_path/"train",allowed_train_users=[],input_identity="pin")


def test_loader_is_real_and_reports_bytes_not_prior_speed():
    d=fixture()
    loaded,evidence=benchmark._load_measurement(lambda:d,"pin")
    assert loaded is d
    assert evidence["input_load_seconds"]>0
    assert evidence["input_bytes"]==sum(v.nbytes for v in d.values() if isinstance(v,np.ndarray))
    with pytest.raises(ValueError,match="loader"):
        benchmark._load_measurement(None,"pin")


def test_benchmark_rejects_development_or_incomplete_population():
    d=fixture(n=80,time=160,features=384,joints=20)
    users={m["user"] for m in d["meta"]}
    folds,pop=benchmark._validate_train(d,users,"pin")
    assert sum(map(len,folds))==len(d["meta"])
    assert pop["validation_windows"]==18548
    d["split"]="development"
    with pytest.raises(ValueError,match="development"):
        benchmark._validate_train(d,users,"pin")
    del d["split"]
    with pytest.raises(ValueError,match="exactly"):
        benchmark._validate_train(d,users|{"new-user"},"pin")


def test_full_feature_tsqr_and_every_mask_pattern_are_measured():
    d=fixture(n=40)
    d["mask"][:10,8,1]=False
    support=data_api.central_support(d["mask"],data_api.lengths_for(d))
    basis={"real":d["target"],"L":d["target"]*.7,"T":d["target"]*.3,"epsilon":d["target"]*.1}
    result,coeff=benchmark._measure_design(d["lat"],basis,support,np.arange(30),np.arange(30,40),context=6,block_windows=4)
    assert len(result["patterns"])==2
    assert all(p["dimension"]==104 for p in result["patterns"])
    assert all(p["saturated_update_measured"] or p["all_blocks_measured"] for p in result["patterns"])
    assert result["projected_seconds"]>0
    assert len(coeff)==14
    assert all(v["real"].shape==(105,2) for v in coeff.values())
    for p in result["patterns"]:
        assert set(map(float,p["solve_seconds"]))=={0.,*(10.**k for k in range(-5,8))}
        assert all(x["normal_residual_max"]<=1e-6 for x in p["solution_diagnostics"].values())


def test_scan_really_evaluates_1936_pairs_times_13_alphas():
    d=fixture(n=12)
    support=data_api.central_support(d["mask"],data_api.lengths_for(d))
    rng=np.random.default_rng(8)
    components={k:rng.normal(size=d["target"].shape) for k in ("L","T","epsilon")}
    basis={"real":d["target"],**components}
    panels=[]
    for fit,held in ((np.arange(6),np.arange(6,12)),(np.arange(6,12),np.arange(6))):
        _,coeff=benchmark._measure_design(d["lat"],basis,support,fit,held,context=0,block_windows=2)
        panel,measure=benchmark._cost_inner_panel(d,components,support,fit,held,coeff,block_windows=2)
        assert measure["sample_held_windows"]==2 and measure["projection_multiplier"]==3
        panels.append(panel)
    scan=benchmark._scan_full_grid(panels)
    assert scan["combinations"]==1936
    assert scan["alpha_evaluations"]==25168
    assert scan["finite_checksum"] and not scan["choice_published"]


def test_real_report_path_exercises_2000_bootstrap_and_twenty_joints(tmp_path):
    d=fixture(n=30,time=160,features=384,joints=20)
    result=benchmark._report_measurement(d,output_dir=tmp_path)
    assert result["bootstrap_replicates"]==2000
    assert result["moment_windows"]==30
    assert result["projected_seconds"]>result["moment_seconds"]
    import json
    payload=json.loads((tmp_path/"cost-report.json").read_text())
    assert payload["complete"]
    assert len(payload["contrasts"])==5


def test_reference_io_reads_all_bytes(tmp_path):
    p=tmp_path/"reference";p.write_bytes(b"actual payload"*100)
    result=benchmark._reference_read([p])
    assert result["reference_read_bytes"]==p.stat().st_size
    assert result["reference_read_seconds"]>0
    with pytest.raises(ValueError):benchmark._reference_read([])


def test_reference_cost_only_reads_train_proxy(tmp_path):
    path=tmp_path/"train-cache.npz"; path.write_bytes(b"actual train bytes"*100)
    result=benchmark._reference_proxy(path,4*path.stat().st_size)
    assert result["reference_io_is_projection"]
    assert not result["development_reference_files_opened"]
    assert result["reference_projection_multiplier"]==4
    assert result["reference_read_seconds"]==4*result["reference_proxy_seconds"]
    with pytest.raises(ValueError): benchmark._reference_proxy(path,0)


def test_tsqr_projection_measures_later_maximum_support_after_saturation():
    d=fixture(n=24,features=4)
    d["mask"][:12,10:14]=False
    support=data_api.central_support(d["mask"],data_api.lengths_for(d))
    basis={"real":d["target"],"L":d["target"]*.7,"T":d["target"]*.3,"epsilon":d["target"]*.1}
    result,_=benchmark._measure_design(d["lat"],basis,support,np.arange(20),np.arange(20,24),context=0,block_windows=2)
    for pattern in result["patterns"]:
        probes=[r for r in pattern["measured_blocks"] if r.get("projection_probe_only")]
        assert probes
        assert probes[0]["already_full_r"]
        assert probes[0]["supported_rows"]==pattern["maximum_supported_rows"]


@pytest.mark.parametrize("change", [{"split":"val"},{"which":"student"},{"checkpoint_step":19618}])
def test_production_cache_config_is_required_even_with_allowed_user_list(change):
    d=fixture(n=80,time=160,features=384,joints=20)
    d["config"].update(change)
    users={m["user"] for m in d["meta"]}
    with pytest.raises(ValueError):benchmark._validate_train(d,users,"pin")


def test_benchmark_requires_forty_users_and_twenty_per_fold():
    d=fixture(n=30,time=160,features=384,joints=20)
    with pytest.raises(ValueError,match="exactly"):
        benchmark._validate_train(d,{m["user"] for m in d["meta"]},"pin")
    d=fixture(n=80,time=160,features=384,joints=20)
    with pytest.raises(ValueError,match="20-user"):
        benchmark._validate_train(d,{m["user"] for m in d["meta"]},"pin",folds=[np.arange(20),np.arange(20,80)])


def test_empty_window_eval_preserves_full_population_and_fp16_storage():
    d=fixture(n=80,time=160,features=384,joints=20)
    d["mask"][0]=False
    d["meta"][1]["n_patches"]=140
    d["mask"][1,140:]=False
    d["lat"][1,140:]=np.nan
    users={m["user"] for m in d["meta"]}
    folds,pop=benchmark._validate_train(d,users,"pin")
    assert pop["train_windows"]==80 and pop["fit_train_windows"]==79
    assert sum(pop["excluded_fitting_windows"])==1
    panel=data_api.subset(d,np.arange(3))
    dataset=benchmark._dataset(d,np.arange(3),"post")
    assert dataset.features.dtype==torch.float16
    assert dataset.size==2 and len(dataset.features)==3
    assert torch.equal(dataset.features[1,140:],torch.zeros_like(dataset.features[1,140:]))
    x,_,_=dataset.batch(torch.arange(dataset.size),torch.device("cpu"))
    assert x.dtype==torch.float32
    class Echo(torch.nn.Module):
        def __init__(self):super().__init__();self.windows=0
        def forward(self,x):self.windows+=len(x);return x[...,:20]
    model=Echo()
    prediction,stats=benchmark._eval_model(model,dataset,panel,.5,device="cpu")
    assert model.windows==3 and prediction.shape==(3,160,20)
    assert stats["window_count"]==2
    np.testing.assert_allclose(prediction[0],d["lat"][0,:,:20].astype(np.float32))
    with pytest.raises(ValueError,match="remain FP16"):
        benchmark._dataset({**d,"lat":d["lat"].astype(np.float32)},np.arange(3),"post")


def test_memory_projection_sums_mask_patterns_and_simultaneous_designs():
    d=fixture(n=80,time=160,features=384,joints=20)
    # Two mask patterns across joints; this doubles R storage even when the
    # cost benchmark constructs one pattern at a time.
    d["mask"][:,20,10:]=False
    support=data_api.central_support(d["mask"],data_api.lengths_for(d))
    folds,_=benchmark._validate_train(d,{m["user"] for m in d["meta"]},"pin")
    memory=benchmark._memory_projection(d,support,folds,baseline_rss=1000)
    outer=memory["folds"][0]["outer_designs"]
    assert all(row["patterns"]==2 for row in outer)
    assert outer[1]["resident_factor_bytes"]==sum(outer[1]["factor_bytes"])
    expected=sum(row["resident_factor_bytes"] for row in outer)+max(row["max_scratch_bytes"] for row in outer)
    assert memory["folds"][0]["simultaneous_working_bytes"]["outer"]==expected
    assert expected>outer[1]["projected_working_bytes"]
    inner=memory["folds"][0]["inner_designs"]
    assert memory["folds"][0]["simultaneous_working_bytes"]["inner"]>max(row["projected_working_bytes"] for row in inner)


def test_many_pattern_memory_gate_fails_without_any_factor_allocation():
    d=fixture(n=80,time=160,features=384,joints=20)
    for joint in range(20):d["mask"][:,20+joint,joint]=False
    support=data_api.central_support(d["mask"],data_api.lengths_for(d))
    folds,_=benchmark._validate_train(d,{m["user"] for m in d["meta"]},"pin")
    result=benchmark._memory_projection(d,support,folds,baseline_rss=0)
    assert not result["runtime_factor_bound_passed"]
    assert not result["within_host_cap"]
    assert result["folds"][0]["outer_designs"][1]["patterns"]==20
