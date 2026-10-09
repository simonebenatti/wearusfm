"""Preset dei rung della ladder (`training.run.LADDER`, proposta del 09/10/2026 da firmare; v10 §10.4): conteggi vicini al nominale,
larghezza scalata ovunque con teste da 64, ripartizione fra i moduli, resto della configurazione uguale al sanity."""

import pytest

torch = pytest.importorskip("torch")

from wearusfm.model.fm import WearUsFM  # noqa: E402
from wearusfm.training import run as R  # noqa: E402

NOMINAL = {"30m": 30e6, "100m": 100e6, "300m": 300e6}


def _shares(cfg):
    m = WearUsFM(cfg.model)
    tot = sum(p.numel() for p in m.parameters())
    part = lambda name: sum(p.numel() for n, p in m.named_parameters() if n.split(".")[0] == name) / tot  # noqa: E731
    return tot, part("backbone"), part("local"), part("pool")


@pytest.mark.parametrize("rung", ["30m", "100m", "300m"])
def test_rung_size_shape_and_rest_of_config(rung):
    cfg = R.ladder_config(rung)
    tot, backbone, local, pool = _shares(cfg)
    assert abs(tot / NOMINAL[rung] - 1) < 0.1, tot  # entro il 10% del nominale
    assert cfg.model.dim // cfg.model.n_heads == 64 and cfg.model.k_latents == 64  # teste da 64, K fisso (v10 §10.4)
    assert 0.65 <= backbone <= 0.85 and 0.08 <= local <= 0.15 and pool >= 0.02
    sanity = R.sanity_config()
    assert (cfg.jepa, cfg.loader, cfg.lr, cfg.batch_size, cfg.warmup_steps) == \
           (sanity.jepa, sanity.loader, sanity.lr, sanity.batch_size, sanity.warmup_steps)
    assert cfg.model.grad_checkpoint and cfg.model.muscle_dropout == sanity.model.muscle_dropout


def test_30m_rung_is_the_sanity_architecture_and_sizes_grow():
    assert R.ladder_config("30m").model == R.sanity_config().model
    sizes = [R.count_parameters(R.ladder_config(r).model) for r in ("30m", "100m", "300m")]
    assert sizes == sorted(sizes) and sizes[1] / sizes[0] > 3 and sizes[2] / sizes[1] > 2.5
