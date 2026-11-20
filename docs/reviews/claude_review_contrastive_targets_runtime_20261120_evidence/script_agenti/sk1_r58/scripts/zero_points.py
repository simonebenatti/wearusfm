"""Scettico R5-8: ricevuta a zero punti prodotta dal vero finish() e letta dal vero read_preflight.

Varianti:
  fixture_two_points            -> fixture dei test (due punti, config legato): controllo positivo
  finish_zero_points            -> ricevuta scritta da ControlParityObserver.finish senza measure()
  finish_zero_points_bind_cfg   -> stessa ricevuta, cambiato SOLO config_sha256 al canonico di config.json
  incomplete_c0_run_evidence    -> _run_evidence reale su un run interrotto prima dell'update 128
"""
import json
import sys
import types
from pathlib import Path

import pytest

work = Path(sys.argv[1])
work.mkdir(parents=True, exist_ok=False)

import contrastive_benchmark_readout as B
import contrastive_prepare as P
import contrastive_target_parity as TP
import test_contrastive_targets_readout as T

REAL_RUN_EVIDENCE = B._run_evidence


def run_variant(name, edit):
    mp = pytest.MonkeyPatch()
    out = {"variant": name}
    try:
        tmp = work / name
        tmp.mkdir()
        kwargs, parity_path, parity, _calls, _pairs = T.readout_fixture(tmp, mp)
        c0 = kwargs["full_dirs"]["C0"]
        source, initial, provenance = parity["source_sha256"], parity["initial_state_sha256"], parity["provenance"]
        if edit != "fixture":
            # summary del C0 con i campi che il trainer scrive (initial_sha256, provenance)
            (c0 / "summary.json").write_text(json.dumps(
                {"real": "C0", "contrastive": {"initial_sha256": initial, "provenance": provenance}}))
            parity_path.unlink()
            mp.setattr(P, "source_digest", lambda: source)
            observer = TP.ControlParityObserver(c0)
            ctx = types.SimpleNamespace(active=False, rank=0, is_main=True, world_size=4)
            observer.finish(ctx)  # nessun measure(): zero punti
            disk = json.loads(parity_path.read_text())
            canon = T.canonical(json.loads((c0 / "config.json").read_text()))
            out.update(points=len(disk["points"]), config_sha256=disk["config_sha256"],
                       receipt_status=disk["status"], receipt_inconclusive=disk["gate"].get("inconclusive"),
                       bindings_match={"source": disk["source_sha256"] == source,
                                       "summary": disk["summary_sha256"] == B.sha(c0 / "summary.json"),
                                       "initial": disk["initial_state_sha256"] == initial,
                                       "provenance": disk["provenance"] == provenance,
                                       "config": disk["config_sha256"] == canon})
            if edit == "bind_cfg":
                disk["config_sha256"] = canon
                parity_path.write_text(json.dumps(disk))
        try:
            result = B.read_preflight(**kwargs)
            out.update(readout="returned", status=result["status"],
                       inconclusive_reasons=result["inconclusive_reasons"],
                       failure_reasons=result["failure_reasons"], training_approved=result["training_approved"],
                       parity_assessment=result.get("zero_weight_parity", {}).get("assessment", {}).get("status"))
        except Exception as exc:  # noqa: BLE001 - si registra tipo e messaggio
            out.update(readout="raised", exc_type=type(exc).__name__, exc=str(exc))
    finally:
        mp.undo()
    print(json.dumps(out, sort_keys=True))


def incomplete_run():
    tmp = work / "incomplete_c0_run_evidence"
    tmp.mkdir()
    cfg, provenance, cohort = T.numeric_folder(tmp)
    summary = json.loads((tmp / "summary.json").read_text())
    summary.update(steps=127, stopped="limite di tempo")
    (tmp / "summary.json").write_text(json.dumps(summary))
    rows = [json.loads(line) for line in (tmp / "metrics.jsonl").read_text().splitlines() if line.strip()]
    (tmp / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows if r["step"] <= 127))
    out = {"variant": "incomplete_c0_run_evidence"}
    try:
        REAL_RUN_EVIDENCE(tmp, "C0", cohort, cfg, provenance)
        out.update(run_evidence="returned")
    except Exception as exc:  # noqa: BLE001
        out.update(run_evidence="raised", exc_type=type(exc).__name__, exc=str(exc))
    print(json.dumps(out, sort_keys=True))


for variant, edit in (("fixture_two_points", "fixture"), ("finish_zero_points", "real"),
                      ("finish_zero_points_bind_cfg", "bind_cfg")):
    run_variant(variant, edit)
incomplete_run()
