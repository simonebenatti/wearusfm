"""Agente nce, compito (f): test che uccidono i mutanti NON equivalenti sopravvissuti ai test di GPT
(tests/cpu/test_contrastive_core.py e test_contrastive_memory.py). Suggerimento per GPT: non vanno committati cosi' come sono.
Tutti passano sul codice consegnato (db5ac41). Fra parentesi il mutante che ciascuno uccide.
"""
import copy
import inspect
from types import SimpleNamespace

import pytest
import torch

from wearusfm.model.contrastive import ProjectionHead, info_nce, normalize_projection, window_readout
from wearusfm.training.contrastive_memory import CandidateQueue, WindowIdentity, admissible_mask, identities_from_batch


def ident(i=0, *, dataset="A", subject="u", session="s", trial=None, fs=200.0):
    return WindowIdentity(dataset, subject, session, trial, i * 2000, i * 2000 + 800, fs)


# --- proiettore del protocollo: 384 -> 384 -> GELU -> 128, con bias, senza BatchNorm/dropout (C01, C02, C04)
def test_projector_is_the_protocol_architecture():
    head = ProjectionHead(384)
    layers = list(head.layers)
    assert [type(m) for m in layers] == [torch.nn.Linear, torch.nn.GELU, torch.nn.Linear]
    assert (layers[0].in_features, layers[0].out_features, layers[2].in_features, layers[2].out_features) == (384, 384, 384, 128)
    assert layers[0].bias is not None and layers[2].bias is not None
    assert sum(p.numel() for p in head.parameters()) == 384 * 384 + 384 + 384 * 128 + 128
    assert not any(isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout)) for m in head.modules())
    x = torch.linspace(-3, 3, 2 * 384).view(2, 384)
    expected = layers[2](torch.nn.functional.gelu(layers[0](x)))
    assert torch.equal(head(x), expected)
    assert not torch.equal(expected, layers[2](torch.relu(layers[0](x))))


# --- una chiave visibile solo su una patch NON fisica non rende valida la query (C08)
def test_visible_key_on_nonphysical_patch_does_not_validate_window():
    z = torch.ones(2, 3, 2, 4)
    physical = torch.tensor([[True, True, False], [True, False, False]])
    visible = torch.tensor([[False, False, True], [False, True, True]])
    _, valid = window_readout(z, physical, visible)
    assert valid.tolist() == [False, False]
    _, valid = window_readout(z, physical, torch.tensor([[False, True, True], [True, True, True]]))
    assert valid.tolist() == [True, True]


# --- soglia di norma: una proiezione valida quasi nulla e' un errore, non una direzione arbitraria (C10)
def test_norm_floor_rejects_tiny_valid_rows_only():
    tiny = torch.tensor([[1e-9, 0.], [3., 4.]])
    with pytest.raises(ValueError, match="norm floor"):
        normalize_projection(tiny)
    out = normalize_projection(tiny, valid=torch.tensor([False, True]))
    assert torch.allclose(out, torch.tensor([[0., 0.], [.6, .8]]))
    assert torch.allclose(normalize_projection(torch.tensor([[1e-6, 0.]])), torch.tensor([[1., 0.]]))


# --- valori di default del protocollo: temperatura 0,2 e minimo 32 negativi (C25, C26)
def test_info_nce_defaults_are_the_protocol_values():
    params = inspect.signature(info_nce).parameters
    assert params["temperature"].default == 0.2 and params["min_negatives"].default == 32
    g = torch.Generator().manual_seed(3)
    q = normalize_projection(torch.randn(3, 16, generator=g))
    kp = normalize_projection(torch.randn(3, 16, generator=g))
    kn = normalize_projection(torch.randn(40, 16, generator=g))
    mask = torch.zeros(3, 40, dtype=torch.bool)
    mask[0, :31] = True
    mask[1, :32] = True
    mask[2] = True
    value, eligible, metrics = info_nce(q, kp, kn, mask)
    assert eligible.tolist() == [False, True, True] and value[0] == 0 and metrics["negative_counts"].tolist() == [31, 32, 40]
    explicit, _, _ = info_nce(q, kp, kn, mask, temperature=0.2, min_negatives=32)
    other, _, _ = info_nce(q, kp, kn, mask, temperature=0.1, min_negatives=32)
    assert torch.equal(value, explicit) and not torch.allclose(value[1:], other[1:])


# --- identita' dal batch: None nei campi, lunghezze diverse, trial vero conservato (M09, M10, M11)
def _batch(**row):
    base = {"dataset": "A", "subject": "u", "session": "s"}
    base.update(row)
    return SimpleNamespace(rows=[base], windows=[SimpleNamespace(trial=None, start=0, stop=800)], fs=[200.])


@pytest.mark.parametrize("field", ["dataset", "subject", "session"])
def test_none_origin_field_is_an_error_not_the_string_none(field):
    with pytest.raises(ValueError, match="missing"):
        identities_from_batch(_batch(**{field: None}))


def test_origin_length_mismatch_is_rejected():
    b = _batch()
    b.rows = b.rows * 2
    with pytest.raises(ValueError, match="lengths"):
        identities_from_batch(b)
    b = _batch()
    b.fs = []
    with pytest.raises(ValueError, match="lengths"):
        identities_from_batch(b)


def test_integer_trial_is_kept_and_excludes_other_trials_of_the_recording():
    b = _batch()
    b.rows, b.fs = b.rows * 3, b.fs * 3
    b.windows = [SimpleNamespace(trial=3, start=0, stop=800), SimpleNamespace(trial=4, start=0, stop=800),
                 SimpleNamespace(trial=None, start=0, stop=800)]
    ids = identities_from_batch(b)
    assert [i.trial for i in ids] == [3, 4, None]
    assert len(set(ids)) == 3                               # stesso start/stop, prove diverse: finestre fisiche diverse
    far = WindowIdentity("A", "u", "s", 4, 20000, 20800, 200.)
    assert admissible_mask([ids[0]], [ids[1], far, ids[2]]).tolist() == [[False, False, False]]
    other = WindowIdentity("A", "v", "s", 4, 0, 800, 200.)
    assert admissible_mask([ids[0]], [other]).item()


# --- una copia NON valida non entra in memoria e non sostituisce la chiave valida piu' vecchia (M14)
def test_invalid_copy_never_enters_memory_or_replaces_older_valid_key():
    q = CandidateQueue(4, 4)
    q.commit(torch.tensor([[1., 2.]]), [ident(0)], step=0)
    cur = torch.tensor([[float("nan"), 0.], [5., 6.]])
    valid = torch.tensor([False, True])
    snap = q.snapshot(1, cur, [ident(0), ident(1)], valid)
    assert snap.identities == (ident(0), ident(1)) and snap.from_current == (False, True) and snap.steps == (0, 1)
    assert snap.keys.tolist() == [[1., 2.], [5., 6.]]
    q.commit(cur, [ident(0), ident(1)], valid, step=1)
    state = q.state_dict()
    assert [(e["step"], e["key"].tolist()) for e in state["entries"]] == [(0, [1., 2.]), (1, [5., 6.])]


# --- le chiavi in memoria sono copie: ne' il tensore del chiamante ne' lo state_dict le possono cambiare (M16, M27)
def test_stored_keys_do_not_alias_caller_tensor_or_state_dict():
    q = CandidateQueue(4, 4)
    source = torch.tensor([[1., 2.], [3., 4.]])
    q.commit(source, [ident(0), ident(1)], step=0)
    source.zero_()
    empty = torch.empty(0, 2)
    assert q.snapshot(1, empty, []).keys.tolist() == [[1., 2.], [3., 4.]]
    state = q.state_dict()
    for entry in state["entries"]:
        entry["key"].zero_()
    assert q.snapshot(1, empty, []).keys.tolist() == [[1., 2.], [3., 4.]]
    snap = q.snapshot(1, empty, [])
    snap.keys.zero_()
    assert q.snapshot(1, empty, []).keys.tolist() == [[1., 2.], [3., 4.]]


# --- FIFO: alla capienza escono le chiavi piu' VECCHIE, restano le piu' recenti (M24)
def test_capacity_evicts_oldest_keys_first():
    q = CandidateQueue(2, 10)
    q.commit(torch.tensor([[0., 0.], [1., 1.]]), [ident(0), ident(1)], step=0)
    q.commit(torch.tensor([[2., 2.]]), [ident(2)], step=1)
    state = q.state_dict()
    assert [e["identity"]["start"] for e in state["entries"]] == [2000, 4000]
    assert [e["step"] for e in state["entries"]] == [0, 1]
    q.commit(torch.tensor([[3., 3.], [4., 4.], [5., 5.]]), [ident(3), ident(4), ident(5)], step=2)
    state = q.state_dict()
    assert [e["identity"]["start"] for e in state["entries"]] == [8000, 10000] and [e["global_index"] for e in state["entries"]] == [1, 2]
    restored = CandidateQueue(2, 10)
    restored.load_state_dict(copy.deepcopy(state))
    assert restored.snapshot(3, torch.empty(0, 2), []).keys.tolist() == [[4., 4.], [5., 5.]]


# --- la stessa finestra fisica ricommessa sostituisce la copia vecchia: una sola chiave, la piu' recente (M26)
def test_recommitted_window_replaces_its_older_key():
    q = CandidateQueue(4, 10)
    q.commit(torch.tensor([[1., 1.], [2., 2.]]), [ident(0), ident(1)], step=0)
    q.commit(torch.tensor([[9., 9.]]), [ident(0)], step=1)
    state = q.state_dict()
    assert [(e["identity"]["start"], e["step"], e["key"].tolist()) for e in state["entries"]] == [(2000, 0, [2., 2.]), (0, 1, [9., 9.])]
    CandidateQueue(4, 10).load_state_dict(copy.deepcopy(state))      # nessun duplicato nel checkpoint


# --- chiavi valide non finite, cambio di dimensione o di dtype: errore prima di toccare la coda (M40, M42)
def test_bad_current_keys_are_rejected_without_mutation():
    q = CandidateQueue(4, 4)
    q.commit(torch.tensor([[1., 2.]]), [ident(0)], step=0)
    for keys in (torch.tensor([[float("nan"), 0.]]), torch.tensor([[float("inf"), 0.]])):
        with pytest.raises(ValueError, match="nonfinite"):
            q.snapshot(1, keys, [ident(1)])
        with pytest.raises(ValueError, match="nonfinite"):
            q.commit(keys, [ident(1)], step=1)
    for keys in (torch.ones(1, 3), torch.ones(1, 2, dtype=torch.float64)):
        with pytest.raises(ValueError, match="dimension or dtype"):
            q.snapshot(1, keys, [ident(1)])
        with pytest.raises(ValueError, match="dimension or dtype"):
            q.commit(keys, [ident(1)], step=1)
    assert q.last_step == 0 and len(q) == 1


# --- checkpoint della coda: ordine non stretto, oltre la capienza, versione, dtype, coda vuota con stato (M32, M35, M36, M38, M39)
@pytest.mark.parametrize("mutation", ["same_order", "over_capacity", "version", "key_dtype", "uninitialized_dim", "uninitialized_entries"])
def test_more_corrupt_queue_checkpoints_are_rejected(mutation):
    q = CandidateQueue(2, 4)
    q.commit(torch.ones(2, 2), [ident(0), ident(1)], step=0)
    state = copy.deepcopy(q.state_dict())
    if mutation == "same_order":
        state["entries"][1]["global_index"] = 0
    elif mutation == "over_capacity":
        state["entries"].append({"identity": ident(2).to_dict(), "key": torch.ones(2), "step": 0, "global_index": 2})
    elif mutation == "version":
        state["version"] = 2
    elif mutation == "key_dtype":
        state["entries"][1]["key"] = torch.ones(2, dtype=torch.float64)
    elif mutation == "uninitialized_dim":
        state.update(last_step=-1, entries=[])
    elif mutation == "uninitialized_entries":
        state.update(last_step=-1, feature_dim=None, dtype=None)
    restored = CandidateQueue(2, 4)
    with pytest.raises(ValueError):
        restored.load_state_dict(state)
    assert restored.last_step == -1 and len(restored) == 0


# --- snapshot vuoto: zero candidati ma con la dimensione delle chiavi, accettato da info_nce (M55)
def test_empty_snapshot_keeps_key_dimension():
    q = CandidateQueue(4, 4)
    cur = torch.tensor([[float("nan")] * 5])
    snap = q.snapshot(0, cur, [ident(0)], torch.tensor([False]))
    assert snap.keys.shape == (0, 5) and snap.identities == ()
    query = torch.nn.functional.normalize(torch.ones(1, 5), dim=1)
    value, eligible, _ = info_nce(query, query, snap.keys, admissible_mask([ident(0)], snap.identities))
    assert value.tolist() == [0.] and not eligible.any()


# --- lo snapshot torna sul dispositivo delle chiavi correnti (M51): verificabile solo fuori dalla CPU
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="serve un dispositivo diverso dalla CPU")
def test_snapshot_returns_keys_on_current_device():
    dev = torch.device("mps")
    q = CandidateQueue(4, 4)
    q.commit(torch.tensor([[1., 2.]], device=dev), [ident(0)], step=0)
    snap = q.snapshot(1, torch.empty(0, 2, device=dev), [])
    assert snap.keys.device.type == "mps" and snap.keys.cpu().tolist() == [[1., 2.]]


# --- dal loader VERO (albero sintetico dei test): le prove 3D di CapgMyo arrivano all'identita' come trial interi, i dataset 2D come None (M11)
def test_real_loader_trials_reach_identity_and_montage_is_ignored(tmp_path):
    import numpy as np
    from test_pretraining_loader import _cfg, _tree
    from wearusfm.data import pretraining_loader as L
    from wearusfm.data import virtual_montage as VM

    root, manifest = _tree(tmp_path)
    index = L.ManifestIndex.load(manifest, [root])
    loader = L.PretrainLoader(index, _cfg(block_s=30.0, windows_per_block=8, block_pool=16,
                                          virtual=VM.VirtualSpec(classes=("C",), exclude_datasets=())))
    rng = np.random.default_rng(5)
    ids, presented, windows = [], [], []
    for _ in range(40):
        batch = loader.batch(8, rng)
        ids.extend(identities_from_batch(batch))
        presented.extend(batch.presented)
        windows.extend(batch.windows)
    assert all(i.trial == w.trial and i.start == w.start and i.stop == w.stop for i, w in zip(ids, windows))
    cap = [i for i in ids if i.dataset == "capgmyo"]
    assert cap and all(isinstance(i.trial, int) for i in cap) and len({i.trial for i in cap}) > 1
    assert all(i.trial is None for i in ids if i.dataset != "capgmyo")
    # stessa prova, montaggi diversi: una sola identita'; prove diverse della stessa registrazione: mai negativi
    by_ident = {}
    for i, p in zip(ids, presented):
        by_ident.setdefault(i, set()).add(p)
    assert any(len(v) > 1 for k, v in by_ident.items() if k.dataset == "capgmyo")
    distinct = sorted(set(cap), key=lambda i: i.trial)
    assert not admissible_mask(distinct, distinct).any()
