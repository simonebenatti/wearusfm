# Verifica avversaria R6-5 (guardie del pacchetto senza test)

Esito: il rilievo regge come lacuna di copertura. La gravita' corretta e' **minore**, non importante: le guardie funzionano davvero.

## Eseguito (clone locale usa e getta, al commit 9d34dc69, poi cancellato)
- `--package` -> b0640ca6.../9d5811b6..., come dichiarato.
- Ho fatto il commit di una review con il blocco esatto su un ref che imita REVIEW_REF. `--bind-review` termina con rc 0, `require_binding()` va a buon fine.
- Una seconda `--bind-review` dà FileExistsError (open "x").
- Commit non antenato di REVIEW_REF: CalledProcessError da merge-base (check=True), nessun binding scritto.
- Manomissioni di release_binding.json (tamper.py):
  - sha non ricalcolato: rifiutato con "Invalid review provenance";
  - due blocchi: rifiutato con "precisely one";
  - pin del piano alterato: rifiutato con "does not approve";
  - review_commit inventato ("0"*40) con testo corretto: ACCETTATO;
  - testo ridotto al solo blocco: ACCETTATO.
- run() con monkeypatch (runsim.py, simulazione):
  - fine Slurm lontana: deadline a circa 875 s;
  - fine a 300 s: circa 274 s (fine-25);
  - un TimeoutError del checked_command finto scrive execution_failure.json {INCONCLUSIVE, training_approved: false}.

## Lettura
- È vero che i test non coprono bind_review, commands e run (test_cn_restart_package.py, 69 righe). Una mutazione non verrebbe rilevata.
- Nessuna guardia è però difettosa, e nessuna incide sull'esito scientifico o sull'attribuzione. Il contenuto vero (pin, sha dei FILES al reviewed_commit, blocco unico) è verificato. Il manifest contiene il LAUNCHER (FILES, riga 30).
- Nota di progetto, non è un difetto: require_binding non riautentica review_commit. Un binding fabbricato a mano con il testo giusto passa. Questo è intenzionale (commento alle righe 97-98): l'autenticazione avviene in bind_review prima del checkout congelato. Il binding resta comunque vincolato a FILES e PLAN identici al commit rivisto, quindi non può far passare codice non rivisto.

## Correzione minima (non bloccante)
Aggiungere due test:
- un repository git temporaneo con bind_review -> require_binding, più i casi non antenato, due blocchi e testo manomesso;
- run() con monkeypatch, per la deadline e execution_failure.json.

Opzionale: confrontare in require_binding `review_text` con `git show review_commit:REVIEW_PATH` quando il commit è disponibile.
