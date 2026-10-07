import numpy as np
import pytest
from wearusfm.harness.spectral_probe import spectral_probe


def arrays():
    x = np.random.default_rng(0).normal(size=(90,4))
    y = np.tile((2*x[:,0]-x[:,1])[:,None], (1,32))
    subject = np.repeat(["ds/a","ds/b","ds/c"],30)
    split = np.repeat(["train","val","test"],30)
    return x,y,np.ones_like(y,bool),subject,split


def test_ridge_learns_and_test_cannot_select_alpha():
    x,y,m,s,p = arrays()
    r = spectral_probe(x,y,m,s,p)
    assert r["coordinates"][0]["r2"] > .99
    y[p=="test"] *= -10
    r2 = spectral_probe(x,y,m,s,p)
    assert r2["coordinates"][0]["alpha"] == r["coordinates"][0]["alpha"]
    assert r2["coordinates"][0]["r2"] < 0  # never truncate negative R2


def test_subject_leakage_constant_and_missing_bands():
    x,y,m,s,p = arrays()
    s[30:60] = "ds/a"
    with pytest.raises(ValueError,match="leakage"):
        spectral_probe(x,y,m,s,p)
    x,y,m,s,p = arrays()
    y[:,0] = 7.; m[:,1] = False; y[:,1] = np.nan
    r = spectral_probe(x,y,m,s,p)
    assert r["coordinates"][0]["r2"] is None and r["coordinates"][1]["r2"] is None


def test_train_scaling_ignores_test_distribution():
    x,y,m,s,p = arrays()
    base = spectral_probe(x,y,m,s,p)
    x[p=="test"] += 1e5
    changed = spectral_probe(x,y,m,s,p)
    assert changed["coordinates"][0]["alpha"] == base["coordinates"][0]["alpha"]
