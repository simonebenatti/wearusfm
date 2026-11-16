"""Verificatore NCE: test che uccidono i MIEI mutanti non equivalenti rimasti vivi sia coi test di GPT sia coi 23 test aggiunti dal revisore.
Passano sul codice consegnato (db5ac41). Suggerimento per GPT, da adattare. Fra parentesi il mutante ucciso.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from wearusfm.model.contrastive import info_nce, window_readout
from wearusfm.training.contrastive_memory import CandidateQueue, WindowIdentity, admissible_mask, identities_from_batch


def ident(i=0, *, dataset="A", subject="u", session="s", trial=None, fs=200.0):
    return WindowIdentity(dataset, subject, session, trial, i * 2000, i * 2000 + 800, fs)


# (vM07) il margine vale nei DUE versi: un candidato che FINISCE almeno 4 s prima dell'inizio della query e' ammesso, uno piu' vicino no
def test_gap_rule_holds_for_candidates_before_the_query():
    q = WindowIdentity("A", "u", "s", None, 4000, 4800, 200.)            # 20 s .. 24 s
    before_ok = WindowIdentity("A", "u", "s", None, 2400, 3200, 200.)    # finisce a 16 s: gap 4 s esatti
    before_near = WindowIdentity("A", "u", "s", None, 2401, 3201, 200.)  # gap 3,995 s
    after_ok = WindowIdentity("A", "u", "s", None, 5600, 6400, 200.)     # comincia a 28 s: gap 4 s esatti
    after_near = WindowIdentity("A", "u", "s", None, 5599, 6399, 200.)
    assert admissible_mask([q], [before_ok, before_near, after_ok, after_near]).tolist() == [[True, False, True, False]]
    # simmetria: scambiando query e candidato l'esito non cambia
    for c in (before_ok, before_near, after_ok, after_near):
        assert admissible_mask([q], [c]).item() == admissible_mask([c], [q]).item()


# (vM06) ogni finestra usa la PROPRIA frequenza: candidato a 400 Hz che comincia 3,9975 s dopo la fine della query a 200 Hz
def test_gap_uses_each_window_own_sampling_frequency():
    q = ident()                                                          # 0 .. 4 s a 200 Hz
    near = WindowIdentity("A", "u", "s", None, 3199, 4000, 400.)         # comincia a 7,9975 s
    edge = WindowIdentity("A", "u", "s", None, 3200, 4000, 400.)         # comincia a 8 s
    assert admissible_mask([q], [near, edge]).tolist() == [[False, True]]


# (vM03, vM04) soggetto e sessione dell'identita' vengono dalla riga del manifest: due registrazioni diverse non collassano
def test_identity_takes_subject_and_session_from_the_manifest_row():
    win = SimpleNamespace(trial=None, start=0, stop=800)
    rows = [{"dataset": "A", "subject": "u1", "session": "s1"}, {"dataset": "A", "subject": "u1", "session": "s2"},
            {"dataset": "A", "subject": "u2", "session": "s1"}, {"dataset": "A", "subject": 7, "session": 3}]
    ids = identities_from_batch(SimpleNamespace(rows=rows, windows=[win] * 4, fs=[200.] * 4))
    assert [(i.subject, i.session) for i in ids] == [("u1", "s1"), ("u1", "s2"), ("u2", "s1"), ("7", "3")]
    assert len(set(ids)) == 4 and len({i.recording for i in ids}) == 4
    # stessa posizione in registrazioni diverse: finestre fisiche diverse, negativi ammessi; e la coda le tiene tutte
    assert admissible_mask(ids[:1], ids[1:]).tolist() == [[True, True, True]]
    queue = CandidateQueue(8, 4)
    queue.commit(torch.eye(4), ids, step=0)
    assert len(queue) == 4


# (vC09) finestra con UNA sola patch fisica: la media e' quella patch, non la sua meta'
def test_readout_of_single_physical_patch_is_that_patch():
    z = torch.arange(1 * 3 * 2 * 2, dtype=torch.float64).reshape(1, 3, 2, 2) + 1.
    physical = torch.tensor([[False, True, False]])
    pooled, valid = window_readout(z, physical, physical)
    assert valid.tolist() == [True] and torch.equal(pooled[0], z[0, 1].mean(0))


# (vC24) query_valid deve essere booleana e allineata alle query
def test_query_valid_shape_and_dtype_are_checked():
    q = torch.nn.functional.normalize(torch.ones(2, 4), dim=1)
    kn = torch.nn.functional.normalize(torch.ones(3, 4), dim=1)
    mask = torch.ones(2, 3, dtype=torch.bool)
    for bad in (torch.ones(3, dtype=torch.bool), torch.ones(2), torch.ones(2, 1, dtype=torch.bool)):
        with pytest.raises(ValueError, match="query validity"):
            info_nce(q, q, kn, mask, min_negatives=1, query_valid=bad)


# (vM15) il commit toglie dalla coda le chiavi piu' vecchie dell'eta' massima: lo stato resta ricaricabile dopo molti update senza nuove chiavi
def test_commit_expires_old_keys_and_state_stays_loadable():
    queue = CandidateQueue(4, 2)
    queue.commit(torch.tensor([[1., 2.]]), [ident(0)], step=0)
    empty = torch.empty(0, 2)
    for step in (1, 2):
        queue.commit(empty, [], step=step)
    assert len(queue) == 1                       # eta' 2 = massima: ancora in coda
    queue.commit(empty, [], step=3)
    assert len(queue) == 0 and queue.state_dict()["entries"] == []
    restored = CandidateQueue(4, 2)
    restored.load_state_dict(queue.state_dict())  # nessun «queue order or age is inconsistent»
    assert restored.last_step == 3


# (vM22) alla ripresa un'eta' massima diversa e' un cambio di politica, come la capienza
def test_resume_rejects_changed_max_age():
    queue = CandidateQueue(3, 2)
    queue.commit(torch.ones(1, 2), [ident(0)], step=0)
    other = CandidateQueue(3, 3)
    with pytest.raises(ValueError, match="policy changed"):
        other.load_state_dict(queue.state_dict())
    assert other.last_step == -1 and len(other) == 0
