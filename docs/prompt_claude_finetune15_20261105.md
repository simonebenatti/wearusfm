# Review della preparazione EPN15 — frozen, fine-tuning, scratch

Simone ha approvato questa sequenza: preparare tutto, review Claude, convergere
e poi lanciare. **Solo review: non eseguire esperimenti o sottomissioni.**
Il riferimento è `origin/codex/consolidation-20261104`, SHA completo fornito
nel messaggio di consegna. Preparazione iniziale `dcf3bf8`, diff complessivo
rispetto a `db89202`; il seguito recepisce la tua review `0968bd1` dei risultati5.
Quella review non è ancora una validazione del nuovo runtime FT15.
I runtime precedenti e il report5 pinnato sono invariati.

Contesto verificato: a cinque epoche frozen70,575%, FT93,766%, scratch92,363%;
FT−scratch+1,403pp [0,930;1,837]. F-R2 scatta, scratch[4,5,5,5]. Frozen ha
migliore validation all'epoca5 in4/4fold ed è sotto la logistica storica82,577%.
Simone aggiunge esplicitamente frozen al seguito15epoche già ipotizzato per
FT/scratch. Dopo questo giro vuole allargare i dataset, non proseguire solo EPN.
Recepiti L1–L4 nell'analisi: pipeline migliore-di-due, LR testa selezionati
diversi, due decimali, entità ridotta del segnale F-R2, distanza descrittiva
+11,19 pp dalla logistica e 176/36/102 soggetti meglio/pari/peggio.
Non chiediamo nuovi candidati. Il budget aggiuntivo è già autorizzato da Simone;
restano review e gate del nuovo preventivo, non una nuova richiesta di consenso.

Leggi:

- `docs/finetune15_protocol_20261105.md` e `docs/reviews/finetune15_critic_20261105.md`.
- `results/infra/finetune15_20261105/plan.json`, `resource_projection.json`,
  `launch_plan.json`, `preparation_validation.json`.
- `scripts/consolidation_finetune15.py`, relativo launcher e test.
- `scripts/consolidation_ft15_report.py` e relativo test.
- `docs/downstream_multidataset_next_20261105.md`, solo come proposta della fase
  successiva; non introduce nuovi lanci o pipeline già pronte.

Chiedo verdetti separati su:

1. **Scienza e selezione.** Tutti e tre i regimi a15epoche, stessi quattro fold,
   due LR per regime e griglia della testa comune, seed0/batch8/architettura
   identici. Ripartenza da teacher19.618 o pesi casuali originali, niente resume5.
   Orizzonte cosine15 dichiarato: confronto15−5 non isola la sola dose.
   C0 non legge held; C1 verifica C0, sceglie su validation e predice held una volta
   per cella. È dichiarato il secondo sguardo sulla stessa popolazione held di
   sviluppo, senza scegliere la dose sul held.
2. **Risorse.** Nuova formula esplicita basata sui24fit completi, massimo degli
   otto tempi per regime, margine25%,600s, rallentamento q>=1 dal benchmark
   steady32 dopo4warm-up e aumento misurato dei costi fissi D>=0. Gate ricalcolato,
   memoria come prima. La vecchia proiezione assoluta del microbenchmark rimane
   diagnostica: questa sostituzione va approvata, non è un'eccezione silenziosa.
   24job split:6h frozen,16h FT/scratch. Circa205GPUh per triplicazione osservata,
   265,038 prudenziali a q=1,D=0; tetto totale304,5GPUh/2.436,667coreh incluso
   benchmark30min e due checkout. Nessuna dose/griglia tagliata se il gate fallisce.
3. **Aggregazione.** Primari FT−scratch e FT−frozen a15; secondari15−5 per ciascun
   regime e `transfer_gap15_minus_gap5` = (FT15−scratch15)−(FT5−scratch5),
   con Δ5/Δ15/differenza e IC diretto sulle stesse estrazioni dei sei modelli.
   Verifica anche il test con oracolo letterale e covarianza fra dosi.
   Raw5 obbligatori, snapshot Git/pin/provenienze e pairing esatto. Macrorecall
   pooled e2000bootstrap per soggetto seed20261102, nessuno scarto. Nonfiniti
   espliciti, niente confronti finali su una matrice privata di fold falliti.
4. **Regole di lettura.** Diagnostica validation15 per tutti i regimi:>=3/4 migliori
   epoche alla15 OR mediana val15−14>=0,005. Falso non prova convergenza, vero non
   autorizza altre epoche. Logistica82,577 riferimento descrittivo distinto;
   nessuna soglia V-R1 importata, nessuna utilità generale o regressione inferita.
   Δ15−Δ5 negativo/positivo sostiene restringimento/ampliamento; IC che include
   zero è irrisolto, non equivalenza. Nessuna terza dose automatica.
5. **Operatività e seguito.** Namespace/output WORK nuovi, due checkout immutabili,
   pin del nuovo benchmark prima dei fit,24ID distinti e12dipendenze C1→C0,
   quota/inode/checkpoint/ledger da verificare prima degli invii. Nessun monitor.
   Nuova policy TIMEOUT/NODE_FAIL/OOM: niente resume dai best checkpoint,
   output parziali preservati, afterok non aggirato, nessun retry automatico;
   recupero solo con diagnosi/emendamento concreto e nuovi ID/output/costi.
   Launcher `--no-requeue` e rifiuto restart prima del setup: verificare i test.
   Dopo EPN: proposta UCI e DB2 per classificazione, audit Hyser-forza per
   regressione; distingui dati già presenti nel pretraining e dataset esclusi.

Il conteggio aggiornato dei test è in `preparation_validation.json`; la review
indipendente è integrata in `finetune15_critic_20261105.md`, incluso il
ricalcolo letterale del bootstrap. Nessun benchmark CUDA15 ancora eseguito.

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ~/.venvs/wearusfm-tok/bin/python -m pytest -q -p no:cacheprovider \
  tests/cpu/test_consolidation_finetune15.py tests/cpu/test_consolidation_ft15_report.py \
  tests/cpu/test_consolidation_ft_report.py tests/cpu/test_consolidation_review_controller.py \
  tests/cpu/test_finetuning_review.py tests/cpu/test_consolidation.py \
  tests/cpu/test_consolidation_operations.py
bash -n scripts/slurm/consolidation_finetune15.sbatch
```

Scrivi la review nel tuo ramo, registra entrambi i blocchi del registro,
committa/pusha e restituisci SHA remoto verificato. Non correggere i runtime
in corso. GPT applicherà le correzioni e, dopo convergenza, lancerà la catena
tramite submitter/ledger senza chiedere un nuovo consenso per il piano approvato.
