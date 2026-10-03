"""Dal dataloader al passo di training: un batch del loader (albero sintetico con montaggi veri) -> ingressi del modello -> perdite JEPA con
ancore e RVQ -> un passo di ottimizzazione. Richiede torch: si salta se non c'e' (gira con ~/.venvs/wearusfm-tok/bin/python -m pytest)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wearusfm.data import pretraining_loader as L  # noqa: E402
from wearusfm.model.fm import FMConfig, WearUsFM  # noqa: E402
from wearusfm.training import run as R  # noqa: E402
from wearusfm.training.jepa import JEPAConfig, jepa_losses, make_teacher, train_step  # noqa: E402

from test_pretraining_loader import _cfg, _tree  # noqa: E402


def test_one_training_step_from_a_real_loader_batch(tmp_path):
    root, mpath = _tree(tmp_path)
    loader = L.PretrainLoader(L.ManifestIndex.load(mpath, [root]), _cfg(filter_band_hz=(20.0, 450.0)))
    batch = loader.batch(4, np.random.default_rng(0))
    inp, visible, rvq_on = L.to_model_inputs(batch)
    torch.manual_seed(0)
    student = WearUsFM(FMConfig(dim=16, n_heads=4, k_latents=4, local_layers=1, backbone_layers=1, decoder_layers=1, muscle_dropout=0.4,
                                rvq_codes=32))
    teacher = make_teacher(student)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
    out = train_step(student, teacher, opt, inp, visible, JEPAConfig("b", 0.2, 0.99), anchor_targets=batch.anchor_targets, rvq_on=rvq_on,
                     rvq_codes=lambda c, w: (c + w) % 32)
    assert all(np.isfinite(v) for v in out.values()) and {"jepa", "log_rms", "band_shape", "log_env", "total"} <= set(out)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="serve un secondo dispositivo (MPS sul Mac)")
def test_no_tensor_is_created_off_the_model_device(tmp_path):
    """Il job 59264349 e' fallito al primo passo su GPU: tensori creati senza `device` restavano sulla CPU. Qui modello e ingressi stanno sulla
    CPU e il dispositivo di default e' MPS: ogni tensore creato senza `device` finisce li' e il calcolo misto fallisce, come sulla GPU."""
    root, mpath = _tree(tmp_path)
    loader = L.PretrainLoader(L.ManifestIndex.load(mpath, [root]), _cfg(filter_band_hz=(20.0, 450.0)))
    batch = loader.batch(4, np.random.default_rng(0))
    inp, visible, rvq_on = L.to_model_inputs(batch)
    torch.manual_seed(0)
    student = WearUsFM(FMConfig(dim=16, n_heads=4, k_latents=4, local_layers=1, backbone_layers=1, decoder_layers=1, muscle_dropout=0.4,
                                rvq_codes=32))
    teacher = make_teacher(student)
    with torch.device("mps"):
        losses = jepa_losses(student, teacher, inp, visible, JEPAConfig("b", 0.2, 0.99), anchor_targets=batch.anchor_targets, rvq_on=rvq_on,
                             rvq_codes=lambda c, w: (c + w) % 32)
        losses["total"].backward()
        d = R.diagnostics(student, teacher, (inp, visible, rvq_on), n_probe=4)
    assert all(v.device.type == "cpu" and torch.isfinite(v) for v in losses.values()) and "rvq" in losses
    assert np.isfinite(d["student_erank"]) and np.isfinite(d["teacher_collapse"]["ratio"])
