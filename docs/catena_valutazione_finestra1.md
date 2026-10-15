# Catena di valutazione della finestra 1: orchestratore (Claude, 15/10/2026)

Accordo GPT-Claude del 15/10/2026 (`docs/accordo_operativo_finestra1_20261015.md`, «Interfaccia della catena di valutazione»): GPT possiede
harness, estrazioni, sonde, confronti e i loro launcher, e scrive il piano di valutazione; Claude possiede `scripts/evaluation_chain.py` e
`tests/cpu/test_evaluation_chain.py` e, per la campagna W1, e' l'unico responsabile delle sottomissioni di training e valutazione.

## Cosa fa

- Legge il piano (schema **provvisorio** `eval-plan-v0` nella docstring dello script, finche' GPT non pubblica
  `docs/evaluation_chain_contract.md`: si adatta a quello), lo valida e ordina i nodi (training -> guardia -> estrazione -> sonde -> confronto).
- **Guardia dei checkpoint** (`scripts/check_checkpoint.py`, `scripts/slurm/check_checkpoint.sbatch`, CPU, minuti): una per checkpoint, in
  afterok dopo il job di training che lo pubblica; verifica run finito (`max_steps`), nessuno STOP, checkpoint presente e non vuoto, passi
  minimi e, se dichiarato, SHA-256. Se fallisce esce con 3 e la valutazione che ne dipende non parte mai (nessun rilancio alla cieca).
- **Dipendenze:** un nodo che legge l'uscita di un altro deve dipenderne direttamente (il confronto parte solo con tutti gli output); piu'
  dipendenze diventano un solo `afterok:a:b` (`submit_job.py` accetta ora piu' `--dopo`).
- **Stato dal ledger del submitter:** `sottomesso` si salta; `ambiguo`, `duplicato`, `rifiutato`, `assente` bloccano il nodo e cio' che ne
  dipende (si chiariscono col submitter); dipendenze non ancora sottomesse -> il nodo aspetta la prossima chiamata (rieseguibile, nessun
  doppione). `limits` per coda ferma le sottomissioni al massimo indicato contando i job gia' in coda.
- **Default dry-run** (stato e comandi nell'ordine); `--submit` sottomette tramite `submit_job.py` (da leonardo-ops). Nessuna scrittura nei
  checkout: uscite assolute e fuori dal codice; `CODE_DIR` e SHA completo per ogni nodo.

Test: `tests/cpu/test_evaluation_chain.py` (piani sintetici, submitter vero con login node finto: validazione, guardie, ordine e dipendenze
multiple, attesa del training, blocco su stati ambigui, limiti di coda, sottomissione rieseguibile senza doppioni, guardia del checkpoint).
