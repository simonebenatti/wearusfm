# Pacchetto eseguibile TCN frozen — seconda consegna, 20/11/2026

Sostituisce, per i lanci, la consegna in `results/infra/pose_temponet_execution_20261119/`, che resta intatta come traccia del
primo tentativo: piano rilasciato `e3f5b119…`, preflight 64931573 `FAILED 1:0`, prove raccolte dall'operatore.

Stato: **REVIEW_REQUIRED**. Nessun invio, nessun deploy, nessuna operazione di scrittura sul cluster. La correzione è di Claude,
per incarico di Simone; la rilettura indipendente del diff è di GPT.

- `plan.json`: uscita di `prepare-plan` sui sorgenti di questa consegna. 24 job con suffisso `-20261120`; gli utenti del produttore
  sono gli ID delle cache, le cartelle stanno in `storage_directories`. Proposta, ricetta, input e tetti identici alla prima
  consegna.
- `delivery_pin.json`: SHA del codice, `source_sha256`, hash dei file della consegna.
- `validate.py` e `validation/`: tutti i file di test TCN e i 5 file storici, con la verifica che `plan.json` sia esattamente ciò
  che `prepare-plan` ricava dai sorgenti.
- `mutation_check/`: otto mutazioni di prova della correzione, ciascuna fermata da almeno un test nuovo.
- `real_input_headers/`: letture in sola lettura dei file pinnati su Leonardo, fatte il 20/11 senza aprire gli array interi, e il
  loro confronto con ciò che il lettore pretende (`check_headers.py`, `check_headers.txt`):
  - `headers.json`: intestazioni e configurazione dei sette `.npz`;
  - `meta_stats.json`: unicità, tipi e hash dei metadati delle finestre;
  - `train_support.json`: finestre di train con e senza posa valida.
- `independent_review/`: esiti dei due revisori indipendenti che hanno concluso (`review_logic.json`, `review_delivery.json`),
  lo script della prova del preflight su file in formato reale, e gli script lasciati da un terzo revisore fermato prima della
  fine (`partial_realdata/`, senza esito).

Istruzioni operative: `docs/pose_temponet_execution_20261120.md`, che aggiorna `docs/pose_temponet_execution_20261119.md`.
Documento di consegna, con lo SHA del codice per `release-plan`: `docs/reviews/claude_delivery_pose_temponet_users_fix_20261120.md`.
Diagnosi del primo tentativo: `docs/reviews/claude_note_pose_temponet_preflight_users_20261120.md`.

Le fixture locali non dichiarano superato nessun benchmark A100, nessun punteggio reale e nessun gate: li decide la catena.
