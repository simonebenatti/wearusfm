# Review richiesta: diagnostica dell'arresto T100-LQ

Ramo `origin/codex/tokenizer-logcontrast-20261104`, diff rispetto a
`48364d3`. Leggi `docs/logsquare_initialization_failure_20261105.md`.

Il job 63532335 è fallito prima del training: SHA dell'inizializzazione
diverso dal collaudo; 63532370 annullato senza partire. Log/hash/sacct
raccolti in `results/infra/logsquare_training_20261104/failure_63532335_20261105/`.
Nessun checkpoint o aggiornamento. Non abbiamo ancora l'hash osservato:
non attribuiamo il guasto al gain senza misura. Il precedente 60561493
del 14/10 mostra però una sensibilità del gain all'ordine degli import.

Questa consegna aggiunge solo una diagnostica, non una correzione del
trainer o un rilancio. Rivedi:

1. `scripts/logsquare_initialization_diagnostic.py` e launcher omonimo;
   isolamento dei cinque processi/import, ordine delle costruzioni e
   variante rank0 con `count_parameters` prima dello studente.
2. Pin della provenienza e 102 sorgenti originali; confronto degli SHA
   per tensore, preservazione del modello e ipotetica sostituzione del
   solo gain su una copia. Nessuna tolleranza o adozione di un nuovo hash.
3. Test CPU della diagnostica; copertura della sequenza dei figli,
   timeout/errori, gain-only contro modifica dei parametri.
4. Piano REVIEW_REQUIRED e risorse: 0,666667 GPUh / 5,666667 coreh massimi
   incluso checkout. Cinque figli90s, job600s, 4GPU/32CPU/240G; zero training.
5. Limiti espliciti: niente DDP/teacher/optimizer/loader; confronto dei
   contesti di costruzione, non replica completa del training. Il
   rapporto non autorizza ripresa né modifica della ricetta.

Controlla anche che NON sia stata applicata la vecchia soluzione al solo
modello temporaneo del controller: sarebbe insufficiente perché il trainer
ricostruisce il proprio modello. La correzione si decide dopo la diagnosi.

Comando test:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ~/.venvs/wearusfm-tok/bin/python -m pytest -q -p no:cacheprovider tests/cpu/test_logsquare_initialization_diagnostic.py tests/cpu/test_logsquare_train.py
```

Restituisci APPROVATO/DA CORREGGERE per il SOLO diagnostico, con rilievi
concreti. Pubblica review e registro sul tuo ramo. Nessun lancio da parte
di Claude. Dopo convergenza GPT aggiornerà lo stato del piano, congelerà
un checkout nuovo e procederà sotto l'autorizzazione operativa già data
da Simone, senza chiedergliela di nuovo. Nessuna estensione contrastiva.
