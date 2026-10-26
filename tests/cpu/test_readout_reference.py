"""Condizione (b1) della seconda ondata (`docs/pilot_posa_v1.md` §3): l'estrazione del readout col codice corrente riproduce bit per bit il
riferimento scritto col codice di 705ec04 (`tests/cpu/readout_reference.py`), e un modello con la testa di posa codifica identico agli stessi
pesi senza testa."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import readout_reference as RR  # noqa: E402
from wearusfm.model.fm import WearUsFM  # noqa: E402


def _reference_model():
    state, outputs = RR.load_reference()
    model = WearUsFM(RR.tiny_config()).eval()
    model.load_state_dict(state, strict=True)
    return model, outputs


def test_reference_file_matches_the_pilot_code_sha():
    z = np.load(RR.REFERENCE, allow_pickle=False)
    assert str(z["code_sha"]) == RR.CODE_SHA and set(RR.LAYERS) <= set(z.files)


def test_extraction_reproduces_the_705ec04_reference_bit_for_bit():
    model, ref = _reference_model()
    out = RR.extract(model)
    for k in RR.LAYERS:
        assert out[k].dtype == ref[k].dtype and out[k].shape == ref[k].shape, k
        assert np.array_equal(out[k], ref[k]), f"{k}: l'estrazione non riproduce il riferimento di 705ec04"


def test_fresh_seed0_model_is_the_reference_model():
    # il modello di riferimento e' quello di seme 0: l'inizializzazione del modello senza testa non e' cambiata
    state, _ = RR.load_reference()
    fresh = RR.fresh_model().state_dict()
    assert set(fresh) == set(state)
    assert all(torch.equal(fresh[k], state[k]) for k in state)


def test_a_model_with_the_pose_head_extracts_the_reference_identically():
    # condizione (b1) del protocollo: la testa non sta nel percorso di encode; i pesi del riferimento caricati in un modello con la testa
    # (la testa resta a zero) danno pool/local/dec identici bit per bit
    from dataclasses import replace

    state, ref = RR.load_reference()
    model = WearUsFM(replace(RR.tiny_config(), pose_head=True)).eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert set(missing) == {"pose_head.norm.weight", "pose_head.norm.bias", "pose_head.out.weight", "pose_head.out.bias"} and not unexpected
    out = RR.extract(model)
    assert all(np.array_equal(out[k], ref[k]) for k in RR.LAYERS)
