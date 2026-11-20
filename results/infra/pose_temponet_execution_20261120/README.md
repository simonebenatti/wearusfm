# Pacchetto eseguibile TCN frozen — seconda consegna, 20/11/2026

> **IN LAVORAZIONE, 20/11 ore 12:35: consegna NON conclusa. Non usare per `release-plan`.**
> Mancano: correzioni ai documenti chieste dai revisori indipendenti, ricevuta di `validate.py`, registro, pin.
> Stato e passi rimanenti: `STATO_PAUSA.md` in questa cartella.

Sostituisce, per i lanci, la consegna in `results/infra/pose_temponet_execution_20261119/`, che resta intatta come traccia del
primo tentativo: piano rilasciato `e3f5b119…`, preflight 64931573 `FAILED 1:0`, prove raccolte dall'operatore.

Stato: **REVIEW_REQUIRED**. Nessun invio, nessun deploy, nessuna operazione di scrittura sul cluster.

- `plan.json`: uscita di `prepare-plan` sui sorgenti di questa consegna. 24 job con suffisso `-20261120`; gli utenti del produttore
  sono gli ID delle cache, le cartelle stanno in `storage_directories`. Proposta, ricetta, input e tetti identici alla prima
  consegna.
- `validate.py` e `validation/`: i 20 file di test TCN e i 5 file storici, con la verifica che `plan.json` sia esattamente ciò che
  `prepare-plan` ricava dai sorgenti.
- `real_input_headers/`: intestazioni e configurazione dei sette `.npz` pinnati, lette in sola lettura su Leonardo il 20/11
  (`read_headers.py`, `headers.json`), e il loro confronto con ciò che il lettore pretende (`check_headers.py`,
  `check_headers.txt`).
- `delivery_pin.json`: SHA del codice, `source_sha256`, hash dei file della consegna.

Istruzioni operative: `docs/pose_temponet_execution_20261120.md`, che aggiorna `docs/pose_temponet_execution_20261119.md`.
Diagnosi del primo tentativo: `docs/reviews/claude_note_pose_temponet_preflight_users_20261120.md`.

Le fixture locali non dichiarano superato nessun benchmark A100, nessun punteggio reale e nessun gate: li decide la catena.
