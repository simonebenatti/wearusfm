# Rilettura incrementale Claude — valutazione lunga

Rivedi soltanto le correzioni alla tua review
`e2753cde2a8af0d78e01daa0ff7dc54652637228`, sul ramo
`origin/codex/consolidation-20261104`, a partire da `c66148b`.
Commit del codice e dei piani corretti: `e2f19ee1e474dd5ea7cca6d6f14df35d81c74b56`. Non lanciare job o benchmark: è una rilettura del codice e
del piano prima del primo checkout delle estrazioni.

1. **R-1, opzione (a):** estrazioni1thread; benchmark/fit/confronti
   classificazione e posa8thread, allineati ai launcher storici. Sintesi
   JSON1thread entro2CPU. Verifica launcher, `_threads()`, gate di ambiente
   e test. Per la classificazione verifica che benchmark/fit e aggregazione
   richiedano `numeric_sha256` esattamente uguale a entrambi i report storici
   P1/P2 pinnati e registrino il confronto. Il riferimento resta lo storico
   dichiarato, senza refit19k/pool aggiuntivo. Ricetta scientifica invariata.
2. **R-2, opzione (i):** conserva il controllo stretto del source corrente.
   Dopo le estrazioni ogni cambio a un sorgente coperto invalida tutte12le
   ricevute/cache rispetto al codice nuovo, anche un cambio di sole risorse.
   Il ramo mantiene i sorgenti fermi per tutta la catena; sviluppo FT15
   eventuale su ramo/worktree separato. Niente whitelist o riscrittura
   ricevute. Ripetere richiederebbe review, nuovi ID/output e preventivo
   completo, con fino9GPUh ulteriori più CPU fuori dal tetto attuale e una
   nuova decisione di Simone. Nessuna ripetizione automatica autorizzata.
3. **C-1/C-2 e risorse:** restano WORK,32GiB prima della prima estrazione,
   quota riletta dall'owner immediatamente prima dell'invio; altrimenti stop.
   Restano48fit/82jobcalcolo/4checkout e tetti9GPUh/1711,666667coreh.
   Non adottiamo gli shard facoltativi a72job. Il margine limitato del gate
   è dichiarato; nessun adattamento automatico dopo estrazioni o benchmark.

Diff di codice, launcher, test e piano:

```bash
git diff c66148b e2f19ee1e474dd5ea7cca6d6f14df35d81c74b56 -- \
  scripts/consolidation_long_classification.py \
  scripts/consolidation_long_classification_runtime.py \
  scripts/consolidation_long_pose_runtime.py \
  scripts/slurm/consolidation_long.sbatch \
  tests/cpu/test_consolidation_long_classification.py \
  tests/cpu/test_consolidation_long_classification_runtime.py \
  tests/cpu/test_consolidation_long_pose_runtime.py \
  tests/cpu/test_consolidation_long_common.py \
  results/infra/consolidation_long_20261105/plan.json \
  results/infra/consolidation_long_20261105/launch_plan.json \
  results/infra/consolidation_long_20261105/rereview_validation.json \
  docs/consolidation_long_execution_20261105.md
```

Verifica in particolare source aggiornato, recipe invariata, conteggi/tetti
immutati e test del rifiuto del contesto numerico storico diverso. Le
ricevute future restano null; `review.commit` resta null e lo stato è
`awaiting_claude_rereview`: e2753c è citato solo come review precedente non
convergente. Nessun consenso aggiuntivo per il lancio originario quando
questa rilettura avrà chiuso i rilievi.

Aggiorna la tua review originale con una sezione di seconda lettura e
restituisci i cinque verdetti, indicando se R-1/R-2 sono chiusi e C-1/C-2
restano condizioni operative del lancio. Registra nelle due sezioni del
registro, committa e pusha sul tuo ramo verificando lo SHA remoto. Nessuna
richiesta di rileggere da zero il pacchetto già approvato nelle altre parti.
