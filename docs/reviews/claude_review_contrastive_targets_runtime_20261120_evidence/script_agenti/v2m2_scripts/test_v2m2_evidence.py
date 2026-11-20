"""Scettico 2 su R2-M2: compare_three_paths REALE (trainer CPU, ranks=1, fork 2, end 4)
inserito in read_preflight, ricevuta scritta su disco, authenticate_preflight dopo
modifiche delle tracce che non cambiano l'esito.

V2M2_FIXED=1: si attende la correzione proposta applicata nel clone temporaneo
(la ricevuta conserva le impronte prodotte) e quindi il rilevamento della modifica.
"""
import copy
import json
import os
from pathlib import Path

import pytest

import contrastive_target_resume as C
import contrastive_benchmark_readout as B
from test_contrastive_targets_readout import readout_fixture
from test_contrastive_target_resume import common_paths  # noqa: F401  (fixture viva, modulo)

REAL = C.compare_three_paths
FIXED = os.environ.get("V2M2_FIXED") == "1"


def _named(value, found, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and (key.startswith("trace-") or key == "start.json"):
                found.append(path + "/" + key)
            _named(item, found, path + "/" + str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _named(item, found, path + "/" + str(index))
    return found


def _rewrite(path, edit):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    edit(rows)
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows) + "\n")


def test_trace_pins_and_postreceipt_edit(tmp_path, monkeypatch, common_paths):
    paths, cfg, initial_sha, initial, fork_sha, _ = common_paths
    live = lambda: REAL(paths, cfg, initial_sha, initial, fork_sha, ranks=1, fork_step=2, end_step=4)
    (tmp_path / "ro").mkdir()
    kwargs, *_ = readout_fixture(tmp_path / "ro", monkeypatch)
    produced = []

    def compare(p, *args, **kw):
        out = live()
        produced.append(copy.deepcopy(out))
        return out

    monkeypatch.setattr(C, "compare_three_paths", compare)
    result = B.read_preflight(**kwargs)
    assert result["status"] == "PASSED_FOR_REVIEW"
    produced_ev = produced[0]["evidence_sha256"]
    published = result["arms"]["CN"]["resume"]["evidence_sha256"]
    print("FIXED", FIXED)
    print("prodotto:", {k: sorted(v) for k, v in produced_ev.items()})
    print("pubblicato resume.evidence_sha256:", {k: sorted(v) for k, v in published.items()})
    names = _named(result, [])
    print("chiavi trace-*/start.json in tutta la ricevuta:", names)
    if not FIXED:
        assert set(published) == {"reload_a", "reload_b"} and names == []
    receipt = tmp_path / "preflight.json"
    receipt.write_text(json.dumps(result, indent=2, allow_nan=False))
    assert B.authenticate_preflight(receipt) == json.loads(receipt.read_text())
    print("authenticate_preflight prima delle modifiche: OK")

    before = live()
    # M1: solo formato (separatori compatti, riga vuota finale) di reload_b/trace-0.jsonl.
    # M2: identita' registrate cambiate in modo coerente nei tre percorsi (RNG CPU dopo il
    #     backward e batch dell'ultimo update): le uguaglianze a coppie restano vere.
    edits = {
        "M1_formato": [("reload_b", lambda rows: None)],
        "M2_rng_batch_coerenti": [(phase, lambda rows: rows[-1].update(
            rng_cpu_after_sha256="0" * 64, batch_sha256="f" * 64)) for phase in C.PHASES],
    }
    originals = {phase: (paths[phase] / "trace-0.jsonl").read_bytes() for phase in C.PHASES}
    try:
        for label, steps in edits.items():
            for phase, edit in steps:
                _rewrite(paths[phase] / "trace-0.jsonl", edit)
            after = live()
            same_outcome = ({k: v for k, v in after.items() if k != "evidence_sha256"}
                            == {k: v for k, v in before.items() if k != "evidence_sha256"})
            changed = sorted(f"{ph}/{name}" for ph in after["evidence_sha256"]
                             for name in after["evidence_sha256"][ph]
                             if after["evidence_sha256"][ph][name] != before["evidence_sha256"][ph][name])
            try:
                B.authenticate_preflight(receipt)
                detected = False
            except ValueError as exc:
                detected = "differs from authenticated recomputation" in str(exc)
            print(label, "esito invariato:", same_outcome, "status:", after["status"],
                  "impronte cambiate nel prodotto:", changed, "rilevata da authenticate_preflight:", detected)
            assert same_outcome and after["status"] == "PASSED" and changed
            assert detected is FIXED
            for phase in C.PHASES:
                (paths[phase] / "trace-0.jsonl").write_bytes(originals[phase])
    finally:
        for phase in C.PHASES:
            (paths[phase] / "trace-0.jsonl").write_bytes(originals[phase])
