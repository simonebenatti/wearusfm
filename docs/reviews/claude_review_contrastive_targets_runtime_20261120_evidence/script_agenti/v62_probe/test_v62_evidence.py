"""Scettico 1 per R6-2: read_preflight sostituisce evidence_sha256 di compare_three_paths?

Sonda indipendente. Nessuna modifica al codice rivisto: solo monkeypatch dei pezzi
pesanti (come la fixture del repository) e il compare_three_paths REALE su un fork CPU.
"""
import json
from pathlib import Path

import pytest

import contrastive_benchmark_readout as B
import contrastive_target_resume as C
from test_contrastive_targets_readout import readout_fixture
from test_contrastive_target_resume import common_paths  # fixture di modulo (trainer CPU reale)

REAL_COMPARE = C.compare_three_paths


def test_unit_overwrite_of_compare_evidence(tmp_path, monkeypatch):
    kwargs, *_ = readout_fixture(tmp_path, monkeypatch)
    fake = {ph: {"checkpoint.pt": "K", "trace-0.jsonl": "T0", "trace-3.jsonl": "T3",
                 **({"start.json": "S"} if ph != "continuous" else {})}
            for ph in C.PHASES}
    monkeypatch.setattr(C, "compare_three_paths",
                        lambda *a, **k: {"status": "PASSED", "passed": True, "evidence_sha256": fake})
    result = B.read_preflight(**kwargs)
    for arm in ("CN", "CR"):
        ev = result["arms"][arm]["resume"]["evidence_sha256"]
        print("UNIT", arm, "phases kept:", sorted(ev), "files:", {k: sorted(v) for k, v in ev.items()})
        assert sorted(ev) == ["reload_a", "reload_b"]
        assert not any("trace" in name or name == "start.json" for v in ev.values() for name in v)
    assert result["status"] == "PASSED_FOR_REVIEW"


def _mutate_batch_hash_everywhere(paths, value):
    """Cambio di contenuto coerente nei tre percorsi, ultimo update (non il confine)."""
    originals = {}
    for phase, folder in paths.items():
        p = Path(folder) / "trace-0.jsonl"
        originals[p] = p.read_bytes()
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        rows[-1]["batch_sha256"] = value
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return originals


def test_real_compare_then_readout_and_authenticate(common_paths, tmp_path, monkeypatch):
    paths, cfg, initial_sha, initial, fork_sha, _ = common_paths
    real = REAL_COMPARE(paths, cfg, initial_sha, initial, fork_sha, ranks=1, fork_step=2, end_step=4)
    print("REAL compare status:", real["status"], "evidence per fase:",
          {k: sorted(v) for k, v in real["evidence_sha256"].items()})
    assert real["status"] == "PASSED"
    assert all("trace-0.jsonl" in v for v in real["evidence_sha256"].values())
    assert all("start.json" in real["evidence_sha256"][ph] for ph in ("reload_a", "reload_b"))

    (tmp_path / "fx").mkdir()
    kwargs, *_ = readout_fixture(tmp_path / "fx", monkeypatch)
    monkeypatch.setattr(C, "compare_three_paths",
                        lambda *a, **k: REAL_COMPARE(paths, cfg, initial_sha, initial, fork_sha,
                                                     ranks=1, fork_step=2, end_step=4))
    result = B.read_preflight(**kwargs)
    assert result["status"] == "PASSED_FOR_REVIEW"
    kept = result["arms"]["CN"]["resume"]["evidence_sha256"]
    print("READOUT resume.evidence_sha256:", {k: sorted(v) for k, v in kept.items()})
    assert "continuous" not in kept and not any("trace-0.jsonl" in v or "start.json" in v for v in kept.values())
    receipt = tmp_path / "receipt.json"
    with receipt.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    assert B.authenticate_preflight(receipt) == result  # sanita'

    originals = _mutate_batch_hash_everywhere(paths, "0" * 64)
    try:
        after = REAL_COMPARE(paths, cfg, initial_sha, initial, fork_sha, ranks=1, fork_step=2, end_step=4)
        diff_keys = sorted(k for k in set(real) | set(after) if real.get(k) != after.get(k))
        print("REAL compare dopo il cambio coerente: status", after["status"], "chiavi diverse:", diff_keys)
        assert after["status"] == "PASSED" and diff_keys == ["evidence_sha256"]
        # Codice rivisto: la ricevuta non pinna le tracce -> l'autenticazione NON rileva il cambio
        assert B.authenticate_preflight(receipt) == result
        print("authenticate_preflight dopo il cambio delle tracce: ACCETTATA (nessun rilievo)")
    finally:
        for p, data in originals.items():
            p.write_bytes(data)
