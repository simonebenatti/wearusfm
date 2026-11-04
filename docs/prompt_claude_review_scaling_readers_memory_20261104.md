# Prompt Claude — review dei lettori scaling e del controllo S3

Fai una review del nuovo pacchetto GPT, senza implementare modifiche o lanciare job.
GPT mantiene implementazione e sottomissioni; Simone ha già autorizzato la campagna.

**Codice da rivedere:** `0d5f0f90074bbaf1142be86d22490e360b632c63`, sul ramo
`origin/codex/scaling-300m-20261104`. Diff dalla consegna del collaudo:

```bash
git diff ab2eb4a 0d5f0f9 -- scripts tests/cpu docs/scaling_300m_protocol_20261104.md results/infra/scaling_300m_20261104
```

La tua review offline `5e4eb8a` resta acquisita: non occorre ripeterla. Il collaudo
63526191 è completato e verificato. Ora sono implementati i lettori mancanti,
S5 e il comparatore, oltre alla prova di memoria sul trainer completo. Il risultato
reale di S3 non è contenuto nel commit del codice; leggere l'eventuale stato
successivo senza confonderlo con il perimetro della review. Nessun endpoint
scientifico 300M a 5.000 è ancora disponibile.

Leggi prima `docs/scaling_300m_readers_memory_20261104.md`, poi il codice:

- `scripts/scaling_projection.py`, `scaling_pose_readout.py`, `scaling_probe.py`,
  `scaling_compare.py` e relativi test;
- `scripts/scaling_production_memory.py`, il diff di `scaling_train.py`,
  `scripts/slurm/scaling_production_memory.sbatch`, i test S3 e readiness;
- `results/infra/scaling_300m_20261104/production_memory_plan.json`,
  `production_memory_resource_policy.json`, `projection_s5/manifest.json`,
  `readers_memory_verification.json` e la review interna
  `docs/reviews/scaling_readers_memory_critic_20261104.md`.

**Domande precise.**

1. Posa: sono preservati supporti, crop, LayerNorm, selezione alpha solo sui
   40 utenti train, separazione dai 15 val, riuso del controllo 30M e bootstrap
   appaiato? Il ramo D384 regredisce numericamente sul lettore storico?
2. Classificazione: scaler e scelta C restano nei fold train, con split/OOF
   storici, corretta gestione delle etichette testuali EPN e convergenza verificata?
   I benchmark misurano costo senza selezionare mediante score di validazione?
3. S5: matrici fisse, manifest e hash rispettano il protocollo? La proiezione
   precede LN/scaler, mantiene distinti locale/backbone e non sostituisce il
   primario naturale con un secondario più favorevole?
4. Comparatore: ricevute/cache/shard/OOF, supporti, runtime e checkpoint sono
   coerenti? Punti e IC sono ricalcolati; il delta posa ha precisione piena;
   artefatti corrotti o convergenza irrisolta non producono raccomandazioni S8?
5. S3: 500 passi esercitano davvero prefetch, pannelli e diagnostiche con
   AdamW residente? Quattro misure reali, schema dei log, runtime storico,
   output freschi e report finale rendono attendibile il gate? Il trainer e i
   103 sorgenti restano intatti? Il checkpoint tecnico resta escluso dal seguito?
   Il tetto di 4 GPU-ore +32 core-ore, oltre al checkout, è coerente?
6. Le precisazioni S8 recepiscono correttamente i tuoi rilievi senza spostare
   soglie o endpoint? Sono distinti implementazione approvabile, memoria ancora
   da misurare e gate operativi ancora necessari prima dei 5.000?

La review interna ha trovato e corretto un difetto: fixed_eval_set.windows è
piatto, 256 righe, non 8×32. È stata aggiunta una regressione sul produttore reale.
Conferma che il test intercetti il problema, senza limitarsi a replicare la fixture.

**Test mirati:** 425 passati in 38,01 s sul Mac.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/cpu ~/.venvs/wearusfm-tok/bin/python -m pytest -q -p no:cacheprovider tests/cpu/test_scaling_preflight.py tests/cpu/test_scaling_train.py tests/cpu/test_scaling_evaluation.py tests/cpu/test_scaling_reading.py tests/cpu/test_scaling_projection.py tests/cpu/test_scaling_pose_readout.py tests/cpu/test_scaling_probe.py tests/cpu/test_scaling_compare.py tests/cpu/test_scaling_production_memory.py tests/cpu/test_scaling_readiness.py tests/cpu/test_tokenizer_train.py tests/cpu/test_tokenizer_probe.py tests/cpu/test_tokenizer_pose_readout.py
```

Non serve rieseguire la suite completa soltanto per ritrovare il fallimento
B19618 già identificato nella tua review: è un problema distinto e preesistente.

Restituisci un verdetto con eventuali blocchi concreti, prova/riproduzione e
correzione minima; separa i suggerimenti facoltativi. Non richiedere nuovi
esperimenti per attestare cose verificabili dal codice. Non dichiarare già pronto
il training scientifico: mancano ancora evidenza S3, piani/launcher e risorse
operativi, controllo/retention e legame dei gate finali.
Scrivi review e registro in entrambe le sezioni nel tuo worktree, commit/push,
e comunica SHA e risultato. Nessun job, nessuna modifica al worktree GPT.
