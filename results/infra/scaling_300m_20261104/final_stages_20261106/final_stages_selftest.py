#!/usr/bin/env python3
"""Regressione del cancello prima dell'invio (`final_stages.py check`): una lettura fallita non e' una coda vuota.

    python -I final_stages_selftest.py

Solo in locale e solo libreria standard: prende lo stato pubblicato delle 10:28:22 e il piano v2, li altera in una cartella
temporanea e chiama `cmd_check` e `checks` di `final_stages.py`. Nessuna connessione, nessun invio, nessuna scrittura nel
repository, nessuno score. Il difetto e' stato riprodotto da GPT nella review del rilascio degli aggregati
(`docs/reviews/gpt_review_scaling_aggregates_release_20261106.md`, commit 1e03fb3): con `squeue` fallito e uscita vuota il
cancello contava zero job e stampava CANCELLO SUPERATO.
"""
import contextlib
import copy
import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_stages as F  # noqa: E402

PLAN = HERE / "aggregate_submission_plan.json"
PLAN_SHA = "abc489613dfbfddc5e33e2150854245bcd3b69292806953bafa622b91a22941c"
STATE = HERE / "live_state_20261106_fits_complete.json"
STATE_SHA = "b75506fd9fbb806886f490d67c22e57d60ae09f559500e6811d41bb600aa5e2c"


def gate(state, places):
    """Esegue il cancello su uno stato: (accettato, righe stampate, messaggio del rifiuto)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.json"
        path.write_text(json.dumps(state))
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                F.cmd_check(str(PLAN), str(path), places)
            return True, out.getvalue().splitlines(), ""
        except SystemExit as e:
            return False, out.getvalue().splitlines(), str(e.code)


def altered(base, name, **changes):
    state = copy.deepcopy(base)
    for key, value in changes.items():
        if value is None:
            state[name].pop(key, None)
        else:
            state[name][key] = value
    return state


def main():
    for path, expected in ((PLAN, PLAN_SHA), (STATE, STATE_SHA)):
        F.require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "%s non e' il file pubblicato e rivisto" % path.name)
    base = json.loads(STATE.read_text())
    queue = base["squeue_serial"]["stdout"].splitlines()
    F.require(len(queue) == 9 and base["squeue_serial"]["exit"] == 0 and base["sacct"]["exit"] == 0, "stato di partenza inatteso")
    four = "\n".join(queue[:4])
    refused_read = "nessun conteggio, nessun cancello"
    # (nome, stato, posti richiesti, accettato atteso, testo atteso nel rifiuto, righe stampate attese: None = non controllate)
    cases = [
        ("stato pubblicato, nove job in coda, sei posti richiesti: rifiuto per i posti", base, "6", False,
         "cancello prima dell'invio non superato", None),
        ("stato pubblicato, un posto richiesto: passa (lo stato e' valido)", base, "1", True, "", None),
        ("quattro job in coda, lettura riuscita: passa con sei posti", altered(base, "squeue_serial", stdout=four), "6", True, "", None),
        ("coda davvero vuota, lettura riuscita: passa", altered(base, "squeue_serial", stdout=""), "6", True, "", None),
        ("squeue fallito e uscita vuota (caso di GPT): rifiuto prima di contare",
         altered(base, "squeue_serial", exit=1, stdout="", stderr="synthetic timeout"), "6", False, "squeue_serial non riuscita (exit 1)", []),
        ("squeue fallito con quattro righe in uscita: rifiuto, l'uscita non conta",
         altered(base, "squeue_serial", exit=1, stdout=four), "6", False, "squeue_serial non riuscita (exit 1)", []),
        ("squeue senza codice d'uscita: rifiuto", altered(base, "squeue_serial", exit=None, stdout=""), "6", False,
         "squeue_serial non riuscita (exit None)", []),
        ("squeue con codice d'uscita false: rifiuto", altered(base, "squeue_serial", exit=False, stdout=""), "6", False,
         "squeue_serial non riuscita (exit False)", []),
        ("sacct fallito con l'uscita intatta: rifiuto", altered(base, "sacct", exit=1), "1", False, "sacct non riuscita (exit 1)", []),
    ]
    failures = 0
    for name, state, places, expected, text, lines in cases:
        accepted, printed, message = gate(state, places)
        ok = (accepted is expected and text in message and (lines is None or printed == lines)
              and (printed[-1:] == ["CANCELLO SUPERATO"]) is expected)
        failures += not ok
        print("PASSA " if ok else "FALLISCE ", name, "->", "CANCELLO SUPERATO" if accepted else message)
    try:  # `plan` passa dagli stessi controlli: con una lettura fallita si ferma prima di scrivere
        F.checks(altered(base, "squeue_serial", exit=1, stdout=""))
        ok = False
    except SystemExit as e:
        ok = refused_read in str(e.code)
    failures += not ok
    print("PASSA " if ok else "FALLISCE ", "checks() con squeue fallito: rifiuto anche per `plan`")
    F.require(failures == 0, "regressione del cancello: %d casi falliti" % failures)
    print("REGRESSIONE SUPERATA: %d casi su %d" % (len(cases) + 1, len(cases) + 1))


if __name__ == "__main__":
    main()
