# Prompt per Claude — chiusura review del confronto contrastivo

Rivedi la consegna GPT **c680e95debad465262d431b18c352581b851696d**, su `origin/codex/tokenizer-logcontrast-20261104`, in risposta alla tua review `0ab9d98` di `db5ac41`.

Diff delle correzioni: `git diff 1d87e8d c680e95`. Gli esempi MATLAB del commit `1d87e8d` sono invariati e fuori perimetro. Non fare job, training, release né modifiche al codice; pubblica la review e il registro nelle tue due sezioni.

Simone ha già deciso esplicitamente:

1. D-1 A: esenzione del solo requisito InfoNCE per dataset secondo rarità da manifest; dati e denominatori globali invariati.
2. D-3 opzione 2: media storica del tratto attivo, fattore 0,1179564123141922; nessuna regolazione automatica dei λ.
3. D-2: E_dyn, P1 ed EPN612 principali; P2 storico e UCI visibili e descrittivi. EPN612 non viene chiamato P2.

Non sono più scelte aperte da riproporre a Simone: verifica che siano coerenti e applicate correttamente. Simone ha autorizzato GPT a lanciare dopo convergenza; questo diff resta REVIEW_REQUIRED e nessun job è stato sottomesso.

Leggi prima:

- `docs/reviews/contrastive_review_response_20261116.md`: risposta punto per punto.
- `docs/contrastive_comparison_protocol_20261116.md`, piano e ultima voce di `docs/decisioni.md`.
- `docs/contrastive_preflight_execution_20261116.md` e `results/infra/contrastive_comparison_20261116/preflight_plan.json`.
- `results/infra/contrastive_comparison_20261116/review_response/`: ricalcolo indipendente, riferimenti, riconciliazione manifest e verifica locale.

La copia locale del manifest differiva da quella sul cluster solo in `code_commit`. Righe, colonne e parametri sono identici; quote ricalcolate identiche. Il piano ora pinna il file reale `6d6d8167…`, gli altri tre input sono stati verificati in sola lettura su Leonardo. Non è stata ripetuta inutilmente la simulazione completa quando tutti i suoi input consumati erano identici: questo limite/procedimento è esplicito nel rapporto di riconciliazione.

Controlla in particolare:

- I-1: modello senza contrastive e serializzazione storica invariati; proiettore fuori da FMConfig. Guardie dei sorgenti storici ancora rigorose, test archiviati adattati dal tuo riferimento a `3263ee9`.
- I-3/I-4: errore coordinato, STOP prima di optimizer/EMA/commit della coda; nessun update applicato senza log.
- I-5/I-6: nessun secondo CLI libero, ricevuta ricalcolata dai file, identità di dati/sorgenti/stato e coorte unica; niente receipt fittizia con `passed=true`.
- I-7/I-8: pannello audit esatto, bound dei descrittori, diagnostiche complete, comandi e ricevute, reale ripresa 544+32, test CUDA per gradienti/STOP e uscita di un rank.
- D-5/I-9/I-10: diagnostica specifica fra finestre, scala relativa E_dyn, etichette, errore Monte Carlo, profili simmetrici e limiti scientifici dichiarati.
- Nuovi strumenti `contrastive_cuda_check.py`, `contrastive_rank_exit.py`, `contrastive_split_benchmark.py`, `contrastive_benchmark_readout.py`, `contrastive_job.py` e i due launcher. Verifica i contratti fra loro, non solo ogni funzione separatamente.

Due dettagli rilevanti per non ripetere falsi allarmi precedenti:

- Il CUDA tecnico usa lo stato **salvato** della calibrazione, non una ricostruzione dipendente dagli import. Configurazione della fixture: batch 8, emg2pose, dropout zero, pesi tecnici 1, minimo un negativo, due update; il benchmark successivo usa invece la ricetta di produzione. Config canonico e pin dei byte sono distinti, perché il config serializzato dal trainer può avere indentazione diversa.
- I pesi effettivi delle loss sono registrati in float64: il lettore può confrontarli esattamente con il programma della rampa. I contatori float dell'ottimizzatore restano esatti nel controllo di ripresa; solo momenti/feature hanno tolleranza. Il tempo `t_step_s` include già il loader.

Sequenza proposta: audit CPU → calibrazione senza update → ricalcolo locale della ricevuta e pubblicazione coorte → CUDA/NCCL → tre benchmark e due riprese → lettura/preventivo. Il CUDA include prima l'uscita intenzionale del rank 0 con quattro attestazioni di inizializzazione e verifica della scomparsa dei processi entro 60 secondi; se fallisce non prosegue. Tetto complessivo **14 GPU-ore e 129,3333 core-ore**, quattro checkout inclusi. Nessun aumento automatico dei tetti o retry.

I λ fuori range fermano l'intera coorte. Il pannello 513–576 ha criteri numerici già scritti. Il piano scientifico non viene modificato dopo la calibrazione; release e stati operativi sono separati. Il preventivo dei training verrà misurato, non preso dai soli tempi storici; gli adattatori/cardinalità/pin della valutazione restano prerequisiti distinti prima dei training e delle estrazioni. Questa consegna non li dichiara già completati.

Verifiche GPT: **361 test mirati contrastivi e direttamente collegati**, più **364 di compatibilità LQ**; sei prove distribuite CPU passate. Le restrizioni sandbox su memoria condivisa/socket/ps hanno richiesto riesecuzione fuori sandbox; i dettagli sono in `verification_v2.json`. Suite intera del repository non eseguita. Nessun CUDA/NCCL reale ancora eseguito.

Per riprodurre, usa un thread e PYTHONPATH del tuo checkout:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ~/.venvs/wearusfm-tok/bin/python -m pytest -q -p no:cacheprovider tests/cpu/test_contrastive_*.py tests/cpu/test_rank_contrast.py tests/cpu/test_arms_collaudo.py tests/cpu/test_wsd_chain.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ~/.venvs/wearusfm-tok/bin/python -m pytest -q -p no:cacheprovider tests/cpu/test_logsquare_*.py tests/cpu/test_tokenizer_logsquare.py
bash -n scripts/slurm/contrastive_preflight_cpu.sbatch
bash -n scripts/slurm/contrastive_preflight_gpu.sbatch
```

Chiedo verdetti distinti: (a) scelte scientifiche recepite; (b) correzioni al runtime/lettori; (c) completezza e fattibilità del collaudo preparatorio con quel tetto; (d) prerequisiti residui prima dei training. Se c'è un blocco, porta una riproduzione mirata. Se è una preferenza, separala dal blocco. Non autorizzare i training sulla sola base dei test locali. Dopo la tua convergenza GPT pubblicherà la release del collaudo e procederà sotto l'autorizzazione operativa già data da Simone, senza chiedergliela di nuovo.
