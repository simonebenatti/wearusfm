# Stato operativo della catena49k —19/11/2026

Operatore unico: questa chat launcher. Worktree
`/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-scaling-49046-launcher`,
ramo `codex/scaling-49046-launcher-20261118`.

Codice eseguito **87da975bc0fd4824684e166659bb996e118ff107**. Review unica
APPROVATO **f6ba1c925e9619d096eea8d68371b1f8811a36a3**. Tutta la catena fino
al confronto è già autorizzata; nessun altro consenso scientifico o review per
invio. Non modificare i byte del codice; commit successivi solo docs/evidenze.
Nessun nuovo monitor/automazione o mittente concorrente.

| Job | Stato attestato | Risorse |
|---|---|---|
|64835246 checkout-code |COMPLETED0:0, raccolto;18s/0,01coreh |2CPU/4G/5min |
|64837401 train-predecay |PENDING(Priority), letto11:53:31Europe/Rome |4GPU/32CPU/480G/19h45 |

Squeue e sacct usciti0: PENDING(Priority), nessunGPU ancora allocato, log ancora assenti come atteso. Non inferire RUNNING dal solo invio. Secondo segmento non sottomesso. Primo run:
`/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/runs/scaling_49046_20261118/T100-G-300M-39237`.
Ricevute/log del wrapper:
`/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/scaling_49046_20261118/training/39237`.
LogSLURM:
`/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/scaling_continue49046_64837401.out`.

Prossimo passo: controllare il job a intervalli non inferiori a5min mentre la
chat è attiva; nessun processo di invio/monitoraggio persistente nuovo.
Quando è COMPLETED0:0, usare il controller congelato:

```
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src ~/.venvs/wearusfm-tok/bin/python -B scripts/scaling_plans49046.py collect train-predecay
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src ~/.venvs/wearusfm-tok/bin/python -B scripts/scaling_plans49046.py gate train-final
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src ~/.venvs/wearusfm-tok/bin/python -B scripts/scaling_plans49046.py lines train-final
```

Leggere prima ricevuta/summary/rank/log raccolti; complete,39237,max_steps,
previous_run_unchanged,memoria/optimizer/log devono essere quelli accettati dal
wrapper. Eseguire soltanto la riga ammessa dal cancello fresco tramite il
submitter comune. Il secondo segmento verifica anche a runtime il primo job,
ricevuta, piano e inventario. Un errore/TIMEOUT fa fermare lo stadio, nessun retry.

Poi seguire `docs/scaling_49046_execution_20261119.md`: raccolta finale,
archivio tre checkpoint, piano estrazioni da pin effettivi, sei estrazioni,
validazione/benchmark, budget, ridge/fit, sei aggregati, confronto. Nessuna
prosecuzione oltre49.046 né modifica di target/loss/seme o training30M.

La quotaWORK è42,6216GiB nel gate10:38:49UTC. I cancelli la rileggeranno;
non trattarla come una riserva esclusiva. Le soglie restano30GiBWORK,
20GiBSCRATCH, orizzonte19/12; nessuna pulizia automatica autorizzata qui.
Conservare ricevute,sacctgrezzo,accounting e log e pubblicarli conregistro nelle
due sezioni, commit/push e SHA remoto verificato.


## Procedura comune aggiornata21/11

Recepita l'istruzione di Simone in CLAUDE.md su main92d1fc0a, sezione
«Un solo via per le tre chat e i loro launcher». Il via già ricevuto copre
l'intera catena e le riprese previste; consultare il ledger prima di ogni invio,
proseguire alle condizioni approvate senza ulteriori consensi di routine.
Nessun nuovo esperimento, duplicato, retry o cambiamento di budget implicito.

Per gli invii successivi usare direttamente l'interprete persistente
`/Users/simonebenatti/.venvs/wearusfm-tok/bin/python` e lo script persistente
`/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py`.
Confronto locale21/11: script byte-identico a scripts/submit_job.py del codice
approvato87da975b, SHA256814da4bc66dbf18d69ee2ba332675fa0788ab25247d5f5460224cfd24d7e0d82.
Interprete presente. Nessuna modifica al generatore congelato: usare gli argomenti
emessi e ammessi dai cancelli sostituendo solo interprete/percorso del submitter
con quelli persistenti indicati da Simone. Restano checkout, sbatch, flag,
prenotazione/lock e ledger comuni. Verificare ancora la compatibilità prima dell'uso.
Nessun wrapper temporaneo; per coda/accounting usare i comandi diretti consentiti.
Un vero permesso tecnico imposto dall'app si rispetta e si segnala una sola volta
con l'operazione e il motivo effettivi, senza rinnovare il consenso scientifico.
