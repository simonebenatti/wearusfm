import numpy as np
import pytest

from wearusfm.tokenizer_checks import branch_probe as B
from wearusfm.tokenizer_checks import metrics as M
from wearusfm.tokenizer_checks.continuous_features import v3_setup

ALL = ("camargo2021", "capgmyo", "csl_hdemg", "emg2pose", "grabmyo", "putemg")


def _run(seed=0, n_groups=15, n_subj=5, d=16):
    """Run sintetico a 6 dataset. Ramo 0: codici del livello 0 presi da un intervallo PROPRIO del dataset (identita' visibile nell'istogramma).
    Ramo 1: codici a caso (nessuna identita'). Ramo 2: codici a caso ma da insiemi disgiunti e grandi, con vettori del codebook spostati per
    dataset (identita' quasi invisibile nell'istogramma, visibile nel vettore medio). Ramo 3: come il ramo 1. Token uguali in distribuzione per
    tutti i dataset (le 5 bande non vedono niente)."""
    rng = np.random.default_rng(seed)
    per = {}
    for di, name in enumerate(ALL):
        n = n_groups * 256
        codes = rng.integers(0, M.N_CODE, size=(4, 16, n)).astype(np.int32)
        codes[0, 0] = rng.integers(di * 1000, di * 1000 + 50, size=n)
        codes[2, 0] = rng.integers(0, 1300, size=n) + di * 1300
        per[name] = {
            "codes": codes, "tokens": rng.normal(size=(n, 200)).astype(np.float32),
            "group_subject": np.array([f"s{g % n_subj}" for g in range(n_groups)]),
        }
    cb = rng.normal(size=(4, M.N_CODE, d))
    for di in range(len(ALL)):
        cb[2, di * 1300 : (di + 1) * 1300, di % d] += 3.0  # direzione propria del dataset nel codebook del ramo 2
    return per, cb


def test_level0_histogram_and_codebook_mean():
    c = np.zeros((4, 16, 4), dtype=np.int32)
    c[1, 0] = [5, 5, 7, 8190]
    c[1, 1] = [1, 2, 3, 4]  # livelli diversi dallo 0 non contano
    h = B.level0_histogram([c], 1)
    assert h.shape == (1, M.N_CODE) and h.sum() == pytest.approx(1.0)
    assert h[0, 5] == pytest.approx(0.5) and h[0, 7] == pytest.approx(0.25) and h[0, 1] == 0
    cb = np.arange(M.N_CODE * 2, dtype=float).reshape(M.N_CODE, 2)
    assert np.allclose(B.level0_codebook_mean([c], 1, cb)[0], cb[[5, 5, 7, 8190]].mean(axis=0))
    with pytest.raises(ValueError):
        B.level0_codebook_mean([c], 1, cb[:10])


def test_split_is_v3_split_restricted_to_enabled_classes():
    per, _ = _run()
    names, _, _, y, units, part = v3_setup(per, 0)
    full = dict(zip(units, part))
    keep = np.isin(np.asarray(names)[y], B.ENABLED_DEFAULT)
    # la restrizione non cambia la parte di nessun soggetto e toglie solo le classi spente
    assert set(np.asarray(names)[y[keep]]) == set(B.ENABLED_DEFAULT)
    assert all(full[u] == p for u, p in zip(units[keep], part[keep]))


def test_rule_on_planted_signals():
    per, cb = _run()
    r = B.branch_probe(per, cb, seed=0)
    assert r["enabled"] == sorted(B.ENABLED_DEFAULT) and r["bands"]["chance"] == pytest.approx(0.25)
    br = r["branches"]
    assert br["0"]["histogram"]["balanced_accuracy"] > 0.9 and not br["0"]["eligible"]  # identita' nell'istogramma: escluso
    assert br["1"]["eligible"] and br["3"]["eligible"]  # codici a caso: idonei
    # ramo 2: e' il caso della correzione. Il vettore medio vede l'identita' e il ramo e' escluso anche se l'istogramma e' piu' debole
    assert br["2"]["codebook_mean"]["balanced_accuracy"] > 0.9 and not br["2"]["eligible"]
    assert br["2"]["branch_accuracy"] == max(br["2"]["histogram"]["balanced_accuracy"], br["2"]["codebook_mean"]["balanced_accuracy"])
    assert r["eligible_branches"] == [1, 3] and r["anchor_starts"]


def test_no_eligible_branch_means_anchor_does_not_start():
    per, cb = _run()
    for name in per:  # identita' in tutti i rami
        di = ALL.index(name)
        per[name]["codes"][:, 0] = np.random.default_rng(di).integers(di * 1000, di * 1000 + 50, size=per[name]["codes"].shape[-1])
    r = B.branch_probe(per, cb, seed=0)
    assert r["eligible_branches"] == [] and r["anchor_starts"] is False


def test_input_checks():
    per, cb = _run()
    with pytest.raises(ValueError, match="assenti"):
        B.branch_probe({k: v for k, v in per.items() if k != "grabmyo"}, cb)
    with pytest.raises(ValueError, match="forma"):
        B.branch_probe(per, cb[:3])


def test_correction_catches_identity_hidden_from_histogram():
    """Il caso per cui la regola e' stata corretta (review del 30/09/2026): ogni dataset sceglie i codici fra TUTTI gli 8192 con una lieve preferenza
    per quelli il cui vettore del codebook punta in una direzione propria. Con 256 codici per unita' l'istogramma non vede niente (la regola vecchia
    darebbe il ramo per idoneo), la media dei vettori si'."""
    per, cb = _run()
    rng = np.random.default_rng(8)
    cb2 = rng.normal(size=(M.N_CODE, cb.shape[-1]))
    for di, name in enumerate(ALL):
        logit = 0.15 * cb2[:, di % cb.shape[-1]]
        p = np.exp(logit - logit.max())
        per[name]["codes"][2, 0] = rng.choice(M.N_CODE, size=per[name]["codes"].shape[-1], p=p / p.sum())
    cb[2] = cb2
    r = B.branch_probe(per, cb, seed=0)
    v, bands = r["branches"]["2"], r["bands"]["balanced_accuracy"]
    assert M.v3_passes(v["histogram"]["balanced_accuracy"], bands)  # la regola vecchia (solo istogramma) lo avrebbe dato per idoneo
    assert v["codebook_mean"]["balanced_accuracy"] > 0.85 and v["from"] == "codebook_mean"
    assert not v["eligible"] and 2 not in r["eligible_branches"]
