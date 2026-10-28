# Analisi della dinamica B, 28/10/2026

Job **62551730**, COMPLETED 0:0, 13s; costo **0 GPU-ore / 0,014444 core-ore**. Protocollo/source/payload congelati
in `dda2b1b` prima del calcolo. [Relazione](../../../../docs/reviews/dinamica_b_results_20261028.md),
[protocollo](../../../../docs/dinamica_b_20261028.md), [review preventiva](../../../../docs/reviews/dinamica_b_prereview_20261028.md).

- `preflight.json`: verifiche prima della sottomissione.
- `submission.json`: argv effettivi, commit e hash, risposta submitter e copia ledger.
- `final/report.json`: rapporto originale; somme per utente, conteggi, MSE e momenti, IC bootstrap, provenienza e controllo E1.
- `final/*.out`, `sacct_utc.txt`, `squeue.txt`: log e contabilità grezzi.
- `final/accounting.json`, `validation.json`, `collection_manifest.json`: riepiloghi operativi e impronte delle copie.
- `analysis/analyze_dynamics.py`, `analysis.json`, `decomposition.png`: ricalcolo indipendente locale da somme e momenti per utente,
  identita' additive, pooling, IC e figura. Nessun import del programma scientifico o nuova lettura della cache.

Riproduzione locale dalla radice del worktree:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  ~/.venvs/wearusfm-tok/bin/python \
  results/infra/pilot_posa/b_dynamics_20261028/analysis/analyze_dynamics.py \
  results/infra/pilot_posa/b_dynamics_20261028/final/report.json --plot
```

Numpy necessario; matplotlib per la figura. Lo script scrive solo nella propria cartella e controlla corrispondenza di
conteggi, componenti, aggregati e IC; non scarta utenti o repliche. Questa riproduzione richiede il solo JSON pubblicato,
non nuove risorse sul cluster. Cache, checkpoint e dati wearable non sono inclusi nel repository.

Il risultato resta una diagnostica standardizzata a 5.000 passi, con correzione perfetta del bias per finestra nel termine
centrato. Non e' un nuovo cancello per B o R3 e non equivale alla qualita' operativa dopo una correzione disponibile in uso.
