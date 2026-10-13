"""Ripresa esatta dello stream dei dati (contratto delle loss, punto 4, rev. 2; review di GPT del 09/10/2026): il batch di indice b ha lo
stesso contenuto in un run continuo e in uno ripreso a qualunque passo, a parita' di worker, processi e segmento. Casi della review:
dentro un segmento e sul bordo, sessioni scartate (scala nulla), scale in cache o da calcolare, montaggi virtuali, piu' worker; e la parita'
fra estrazioni a secco e reali (stesso generatore, stesso pool, stesse sessioni scartate)."""

from dataclasses import fields, replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wearusfm.data import pretraining_loader as L  # noqa: E402
from wearusfm.data import virtual_montage as VM  # noqa: E402
from wearusfm.training import pairing  # noqa: E402
from wearusfm.training import run as R  # noqa: E402

from test_pretraining_loader import _cfg, _tree  # noqa: E402

CONTENT = [f.name for f in fields(L.PretrainBatch) if f.name != "skipped_sessions"]  # il conteggio cumulativo e' una metrica, non contenuto


def _hash(batch) -> str:
    return pairing.batch_sha256({k: getattr(batch, k) for k in CONTENT})


def _setup(tmp_path, *, zero_session=True, virtual=True):
    root, mpath = _tree(tmp_path)
    if zero_session:  # una sessione senza scala (MAD nullo): viene scartata, con estrazioni casuali in piu'
        np.save(root / "ninapro_db2" / "s01" / "session1" / "data_int16.npy", np.zeros((18000, 12), dtype=np.int16))
    cfg = _cfg(block_s=30.0, windows_per_block=3, block_pool=4)
    if virtual:  # montaggi virtuali anche su CapgMyo, l'unica griglia dei dati di prova
        cfg = replace(cfg, virtual=VM.VirtualSpec(p=0.7, exclude_datasets=()))
    return L.ManifestIndex.load(mpath, [root]), cfg


def _stream(idx, cfg, *, start, n_workers, n, segment, scales=None, rank=None):
    """I batch da `start` a `start + n - 1` come li ricompone il DataLoader: un generatore per worker, a turno."""
    gens = [L.batch_stream(L.PretrainLoader(idx, cfg, dict(scales or {})), 2, seed=7, start=start, worker=w, n_workers=n_workers,
                           segment=segment, rank=rank) for w in range(n_workers)]
    out = []
    for t in range(n):
        b = next(gens[t % n_workers])
        assert b.index == start + t
        out.append(_hash(b))
    return out


@pytest.mark.parametrize("n_workers", [1, 3])
def test_resume_at_any_step_gives_the_same_batches(tmp_path, n_workers):
    idx, cfg = _setup(tmp_path)
    segment, n = 5, 17
    full = _stream(idx, cfg, start=0, n_workers=n_workers, n=n, segment=segment)
    assert len(set(full)) == n  # batch diversi fra loro
    for start in (1, 4, 5, 6, 9, 10, 13):  # dentro un segmento e sui bordi (5, 10)
        assert _stream(idx, cfg, start=start, n_workers=n_workers, n=n - start, segment=segment) == full[start:], start


def test_scale_cache_and_rank_do_not_leak(tmp_path):
    idx, cfg = _setup(tmp_path)
    loader = L.PretrainLoader(idx, cfg)
    for r in idx.rows:  # scale calcolate prima (cache piena) contro calcolate al volo (cache vuota)
        try:
            loader.scale(r, loader.view(r), np.random.default_rng(0))
        except ValueError:
            pass
    cold = _stream(idx, cfg, start=0, n_workers=2, n=8, segment=4)
    assert _stream(idx, cfg, start=0, n_workers=2, n=8, segment=4, scales=loader.scale_cache) == cold
    assert _stream(idx, cfg, start=3, n_workers=2, n=5, segment=4, scales=loader.scale_cache) == cold[3:]
    other = _stream(idx, cfg, start=0, n_workers=2, n=8, segment=4, rank=1)  # un altro processo DDP: un altro stream
    assert not set(other) & set(cold)


def test_dry_batch_consumes_like_the_real_one(tmp_path):
    idx, cfg = _setup(tmp_path)
    real, dry = L.PretrainLoader(idx, cfg), L.PretrainLoader(idx, cfg)
    rng_real, rng_dry = np.random.default_rng(3), np.random.default_rng(3)
    for _ in range(4):
        b = real.batch(2, rng_real)
        assert dry.batch(2, rng_dry, dry=True) is None and b is not None
        assert rng_real.bit_generator.state == rng_dry.bit_generator.state  # stesse estrazioni
        assert [(bl.row["session"], bl.data_lo, bl.data_hi, bl.uses_left) for bl in real._pool] == \
               [(bl.row["session"], bl.data_lo, bl.data_hi, bl.uses_left) for bl in dry._pool]  # stesso pool
        assert real.skipped == dry.skipped
    assert all(bl.data is None for bl in dry._pool)  # a secco nessun segnale letto
    assert _hash(real.batch(2, rng_real)) == _hash(dry.batch(2, rng_dry))  # dopo, gli stessi batch veri
    assert "ninapro_db2/s01/session1" in real.skipped_ever  # il caso della sessione scartata c'e' davvero


def test_training_resumed_equals_continuous(tmp_path, monkeypatch):
    root, mpath = _tree(tmp_path)
    seen = {}
    to_inputs = R.L.to_model_inputs

    def record(batch):  # il contenuto di ogni batch consumato dal training (non quello di validazione, senza indice)
        if batch.index is not None:
            seen.setdefault(record.run, []).append((batch.index, _hash(batch)))
        return to_inputs(batch)

    monkeypatch.setattr(R.L, "to_model_inputs", record)
    cfg = replace(R.small_config(datasets=None, max_steps=6), stream_segment=4, ckpt_every=100)
    record.run = "continuo"
    R.train(cfg, mpath, [root], tmp_path / "a", log=lambda m: None)
    record.run = "ripreso"
    R.train(replace(cfg, max_steps=3), mpath, [root], tmp_path / "b", log=lambda m: None)  # interrotto al passo 3 (dentro un segmento)
    R.train(cfg, mpath, [root], tmp_path / "b", log=lambda m: None)  # ripresa fino al 6
    assert seen["continuo"] == seen["ripreso"] and [i for i, _ in seen["continuo"]] == list(range(6))
    a = torch.load(tmp_path / "a" / "checkpoint.pt", weights_only=False)
    b = torch.load(tmp_path / "b" / "checkpoint.pt", weights_only=False)
    for part in ("student", "teacher"):
        for k in a[part]:
            assert torch.equal(a[part][k], b[part][k]), (part, k)  # stessi batch, stessi generatori: stessi pesi
    with pytest.raises(ValueError, match="stream diverso"):  # ripresa con un altro segmento: rifiutata
        R.train(replace(cfg, max_steps=8, stream_segment=5), mpath, [root], tmp_path / "b", log=lambda m: None)


def _has_shared(x) -> bool:
    if isinstance(x, R._Shared):
        return True
    if isinstance(x, (list, tuple)):
        return any(_has_shared(v) for v in x)
    if isinstance(x, dict):
        return any(_has_shared(v) for v in x.values())
    if hasattr(x, "__dataclass_fields__"):
        return any(_has_shared(getattr(x, f)) for f in x.__dataclass_fields__)
    return False


def test_batches_through_shared_memory_are_identical(tmp_path):
    """Con i worker gli array grandi viaggiano in memoria condivisa (collaudo del 12/10/2026): stessi batch, bit per bit, del pickle."""
    idx, cfg = _setup(tmp_path)
    ds = R.BatchStream(idx, cfg, 2, 7, {}, segment=4)
    plain = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=2)
    shared = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=2, collate_fn=R._to_shared)
    it_p, it_s = iter(plain), iter(shared)
    for _ in range(6):
        raw = next(it_s)
        assert _has_shared(raw)  # il passaggio in memoria condivisa e' davvero attivo
        got, ref = R._from_shared(raw), next(it_p)
        assert not _has_shared(got) and _hash(got) == _hash(ref) and got.index == ref.index
        assert all(isinstance(x, np.ndarray) for x in got.signals)


def test_training_with_workers_resumed_equals_continuous(tmp_path):
    root, mpath = _tree(tmp_path)
    cfg = replace(R.small_config(datasets=None, max_steps=6), stream_segment=4, ckpt_every=100)
    R.train(cfg, mpath, [root], tmp_path / "a", log=lambda m: None, num_workers=2)
    R.train(replace(cfg, max_steps=3), mpath, [root], tmp_path / "b", log=lambda m: None, num_workers=2)
    R.train(cfg, mpath, [root], tmp_path / "b", log=lambda m: None, num_workers=2)
    a, b = (torch.load(tmp_path / d / "checkpoint.pt", weights_only=False) for d in ("a", "b"))
    for part in ("student", "teacher"):
        assert all(torch.equal(a[part][k], b[part][k]) for k in a[part]), part
