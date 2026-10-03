"""Dal dataloader al passo di training: un batch del loader (albero sintetico con montaggi veri) -> ingressi del modello -> perdite JEPA con
ancore e RVQ -> un passo di ottimizzazione. Richiede torch: si salta se non c'e' (gira con ~/.venvs/wearusfm-tok/bin/python -m pytest)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wearusfm.data import pretraining_loader as L  # noqa: E402
from wearusfm.model.fm import FMConfig, WearUsFM  # noqa: E402
from wearusfm.training.jepa import JEPAConfig, make_teacher, train_step  # noqa: E402

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
