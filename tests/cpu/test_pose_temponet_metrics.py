"""Independent weighted-loss, dynamic-ratio and historical crop parity checks."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"scripts"))
from pose_temponet import metrics as M
from pose_temponet import data as D
from test_pose_temponet_data import fixture


def test_mse_weights_windows_equally_and_masks_nan_before_differentiation():
    p = torch.tensor([[[2.],[float("nan")]],[[4.],[4.]],[[float("nan")],[float("nan")]]],requires_grad=True)
    y = torch.tensor([[[0.],[float("nan")]],[[0.],[0.]],[[float("nan")],[float("nan")]]])
    m = torch.tensor([[[True],[False]],[[True],[True]],[[False],[False]]])
    loss = M.per_window_mse(p,y,m)
    assert loss.item() == 10.  # (4 + (16+16)/2)/2, not 12 pooled.
    loss.backward()
    assert torch.isfinite(p.grad).all()
    torch.testing.assert_close(p.grad,torch.tensor([[[2.],[0.]],[[2.],[2.]],[[0.],[0.]]]))
    with pytest.raises(ValueError,match="no supported"):
        M.per_window_mse(p,y,torch.zeros_like(m))


def test_dynamic_variance_and_threshold_use_full_fitting_targets_and_supported_joints():
    y = np.array([[[0.,0.],[2.,10.],[4.,20.]],[[0.,9.],[1.,8.],[2.,7.]]])
    m = np.ones_like(y,dtype=bool)
    m[0,1:,1] = False
    expected = np.array([np.var([0.,2.,4.]), (np.var([0.,1.,2.])+np.var([9.,8.,7.]))/2])
    np.testing.assert_allclose(M.dynamic_variance(y,m),expected)
    assert M.training_threshold(y,m) == pytest.approx(np.median(expected))
    assert M.training_threshold(np.zeros_like(y),m) == 1e-12
    with pytest.raises(ValueError,match="no temporal variance support"):
        M.training_threshold(y,np.zeros_like(m))


def direct_dynamics(prediction,target,mask,selected):
    error = variance = 0.
    for i in range(len(target)):
        if not selected[i]:
            continue
        for j in range(target.shape[-1]):
            pick = mask[i,:,j]
            if pick.sum() < 2:
                continue
            p,y = prediction[i,pick,j].astype(float),target[i,pick,j].astype(float)
            error += np.sum(((p-p.mean())-(y-y.mean()))**2)
            variance += np.sum((y-y.mean())**2)
    return error,variance


def test_metrics_match_independent_ratios_and_preserve_historical_accumulators():
    data,_ = fixture()
    data["pose_var"][:3] = .1
    prediction = data["target"].astype(np.float64)*.3 + np.arange(8)[:,None,None]*.5
    result = M.evaluate(prediction,data)
    old = D.reference.summarize(D.historical._target_view(data),prediction[:,6:-6])
    for group in ("all","dynamic","static"):
        assert result["groups"][group]["centered"] == old["groups"][group]["centered"]
        assert result["groups"][group]["mae_per_user"] == old["groups"][group]["mae_per_user"]
    mask = D.central_support(data["mask"],D.lengths_for(data),data["key_time_valid"])
    error,var = direct_dynamics(prediction[:,6:-6],data["target"][:,6:-6],mask,data["pose_var"]>=M.THRESHOLD)
    got = result["groups"]["dynamic"]["centered"]["centered_mse_ratio"]
    assert got == pytest.approx(error/var,abs=1e-14)
    errors = np.where(mask,prediction[:,6:-6]-data["target"][:,6:-6],0)**2
    expected_mse = np.mean(errors.sum((1,2))/mask.sum((1,2)))
    assert result["groups"]["all"]["per_window_mse"] == pytest.approx(expected_mse)


def test_empty_temporal_support_is_distinct_from_selected_windows():
    data,_ = fixture()
    data["mask"][0] = False
    data["mask"][0,8,0] = True
    result = M.evaluate(data["pred"],data)["groups"]["dynamic"]
    assert result["n_selected_windows"] == 8
    assert result["n_supported_windows"] == 8
    assert result["centered"]["n_windows"] == 7


def test_selection_statistics_pool_accumulators_not_ratios():
    data,_ = fixture()
    prediction = data["pred"].astype(float)
    full = M.selection_statistics(prediction,data,M.THRESHOLD)
    halves = [M.selection_statistics(prediction[ix],D.subset(data,ix),M.THRESHOLD)
              for ix in (np.arange(3),np.arange(3,8))]
    for key in full:
        assert sum(h[key] for h in halves) == pytest.approx(full[key],abs=1e-10)


def test_reference_gate_rejects_shift_even_with_identical_shape_and_counts(monkeypatch):
    data,_ = fixture()
    ridge = data["target"][:,6:-6].astype(float)*.2
    head = data["pred"][:,6:-6].astype(float)
    rows = [M.evaluate(a,data)["groups"]["dynamic"] for a in (ridge,head)]
    expected = {"post_ridge_E_dyn":rows[0]["centered"]["centered_mse_ratio"],
                "trained_head_E_dyn":rows[1]["centered"]["centered_mse_ratio"],
                "dynamic_selected_windows":8,"dynamic_centered_windows":8,
                "dynamic_centered_elements":rows[0]["centered"]["n_valid_elements"],
                "dynamic_window_joint_pairs":rows[0]["centered"]["n_window_joints"],
                "dynamic_users":4}
    monkeypatch.setattr(M,"HISTORICAL_REFERENCE",expected)
    assert M.historical_reference_gate(data,ridge,head)["passed"]
    with pytest.raises(ValueError,match="crop indices"):
        M.historical_reference_gate(data,ridge,data["pred"][:,:148].astype(float))
    changed = ridge.copy(); changed += np.arange(148)[None,:,None]*.01
    with pytest.raises(ValueError,match="E_dyn parity"):
        M.historical_reference_gate(data,changed,head)
    for key in ("dynamic_users","dynamic_window_joint_pairs"):
        monkeypatch.setattr(M,"HISTORICAL_REFERENCE",{**expected,key:expected[key]+1})
        with pytest.raises(ValueError,match="support counts"):
            M.historical_reference_gate(data,ridge,head)


def test_float64_reference_is_cast_before_subtraction_and_nan_outside_support_safe():
    data,_ = fixture()
    data["target"][~data["mask"]] = np.nan
    pred = data["pred"].copy()
    pred[~data["mask"]] = np.nan
    result = M.evaluate(pred,data)
    assert np.isfinite(result["groups"]["all"]["per_window_mse"])
    assert result["groups"]["dynamic"]["centered"]["centered_mse_ratio"] == pytest.approx(.25)


def test_all_joint_descriptives_sum_to_pooled_accumulators_without_filtering():
    data,_ = fixture()
    result = M.evaluate(data["pred"],data,per_joint=True)["groups"]["dynamic"]
    assert list(result["by_joint"]) == list(M.JOINTS)
    for key,value in result["centered"]["sums"].items():
        assert sum(j["centered"]["sums"][key] for j in result["by_joint"].values()) == pytest.approx(value,abs=1e-9)
    assert sum(j["centered"]["n_valid_elements"] for j in result["by_joint"].values()) == result["centered"]["n_valid_elements"]
