# Verifica GPT della consegna ultrareview — 22/10/2026

## Aggiornamento: integrazione autorizzata e completata

Su richiesta successiva di Simone, 23efccb e d4f4d23 sono integrati nel merge
`d8b7170`, con gli SHA originali conservati. Codice src/scripts/tests identico
al target verificato; unico conflitto nel registro, risolto conservando tutte
le voci. Punto 3 chiuso per questa consegna. La nota sotto «non viene integrato
implicitamente» descrive la review precedente al via esplicito, ora superata.
Nessun deploy o job; la parte operativa del pilot resta da completare.

## Review precedente al merge

**Esito: nessun rilievo bloccante nel delta verificato.** Codice Claude
`d4f4d23e0a59042c39c9ff6c12b1fcd1710d877d`, identico al ramo GitHub
`worktree-claude-training-engine` al controllo. Worktree Claude pulito prima e
dopo i test. Verificato anche il delta del pianificatore C4 in `23efccb`.

## Correzioni

- Rilievo 1: il trainer rifiuta il pannello fisso con obiettivo mascherato
  prima di avviare il training. Non viene introdotta una nuova misura del pilot;
  il wrapper futuro deve mantenere `fixed_eval_every=0` per quel percorso.
- Rilievo 2: `--done` senza `--per-set` e' rifiutato prima di scrivere il piano,
  invece di ripianificare silenziosamente le sonde dichiarate complete.
- Rilievo 4: `run_ids` non sovrascrive piu' la mappa attesa usata da `status_of`.
- Rilievo 5: il monitor evita i token del lato studente. `frames_used` riproduce
  esattamente la validita' restituita da `tokens`; supporti, QC e selezione dei
  frame sono conservati. Review indipendente aggiuntiva del refactor senza
  rilievi. **Restano calcolati gli spettri mascherati**: non attribuire a questa
  modifica l'eliminazione dell'intero percorso spettrale.
- Rilievo 3: duplicazione del caricatore importlib lasciata intenzionalmente
  fuori, come dichiarato da Claude; pulizia non bloccante, non fix completato.

Pianificatore C4: sonda 300M su una GPU, QoS normal, cap sei ore, costo incluso
nel preventivo; il test conferma C3 invariata rispetto al piano congelato.
Questa verifica del codice non e' la verifica degli esiti scientifici di C3/C4
ne' una nuova autorizzazione di spesa.

## Verifiche eseguite

- 68 PASS: branch_masking, branch_pilot_trainer, spectral_diagnostic, arms_compare.
- 3 PASS: regressioni GPT del pilot in test_branch_pilot_review_20261021.py,
  con WEARUSFM_REVIEW_SOURCE puntato al worktree Claude.
- 19 PASS: calibration_plan (18 al primo giro; il test che scrive un file
  temporaneo nel worktree, bloccato dal sandbox, passa alla riesecuzione
  autorizzata). Nessun fallimento funzionale residuo.
- `git diff --check` del fix: PASS. Totale 90 test verificati.
- Suite completa 1317 PASS: dichiarata da Claude, non rieseguita da GPT.
- Nessuna verifica GPU, accesso al cluster, modifica del codice Claude o merge.

## PR di sola revisione e consegna

La PR https://github.com/simonebenatti/wearusfm/pull/1 risulta **closed**, con
`merged=false`, verificato via API GitHub. Non e' stato necessario chiuderla e
non va integrato il suo ramo sintetico. La review riguarda i commit originali
del ramo Claude. Questa consegna pubblica soltanto review e registro sul ramo
GPT; il codice d4f4d23 non viene integrato implicitamente su main.

Il pilot PSD/wavelet resta da completare: wrapper, guardie, estrazione,
confronto/verdetto, critico e preventivo prima del collaudo. Nessun lancio.
