# Review Claude — valutazione lunga T100-G30M

Revisiona il pacchetto scientifico ed esecutivo di GPT, senza lanciare job.
Ramo: origin/codex/consolidation-20261104.
Commit del codice e del piano: e9d9d7bad7a1c223816f631af2c9b3a4143245d6.
Base del diff: 19c316623c17fceabfb219bda5336e1fc2ebd2e9.
Il commit successivo del prompt cambia soltanto documentazione.

Simone ha già assegnato preparazione e lancio dopo la tua convergenza:
«ok, fai completare alla chat 2 la valutazione del lungo, falla revisionare
a claude e poi lanciatela appena pronta». La review è il passaggio al revisore,
non una nuova richiesta di consenso. Tu non sottomettere job.

Leggi prima docs/consolidation_long_execution_20261105.md, poi:
- docs/consolidation_protocol_20261104.md e V-R1 in
  docs/reviews/consolidation_response_to_claude_20261104.md;
- scripts/consolidation_long.py, consolidation_long_common.py,
  consolidation_long_classification.py e _runtime.py,
  consolidation_long_pose.py e _runtime.py;
- scripts/slurm/consolidation_long.sbatch e cinque nuovi test CPU;
- results/infra/consolidation_long_20261105/{plan,launch_plan,preparation_validation}.json,
  continuation/ e historical_artifacts_verification.json.
La review interna è disponibile, ma non sostituisce la tua.

Scopo: teacher T100-G30M a19.618/39.237/49.046, posa e P1/P2, viste
pre/postbackboneD384 e pool storico2D768. Il pool è post+locale.
Nessun nuovo training FM, loss, seme, dose o fine-tuning. FT15 è separato.
I training63525803/63525805/63525809 sono COMPLETED0:0; copie su WORK,
hash/dose/ricevute verificati,25,283333GPUh consumate. Il runtime delle
estrazioni rilegge lo stato completo e verifica teacher/gain e immutabilità.

Chiedo verdetto distinto sui seguenti cinque punti.

1. Estrazioni e provenienze. Stesso forward per pre/post/pool, stessi crop,
   preprocessing, scale e ordine dei dati. Posa dal loader originale,
   bersagli/maschere e selezioni1907train/18548held preservati.
   Legame dei checkpoint persistenti alle ricevute, ripristino esatto dei
   tensori e gain, guardie dopo estrazione, pin/cache/manifest fail-closed.
   Il campo key_time_valid deve coincidere col supporto reale dichiarato;
   uno scarto ferma la fase e non modifica silenziosamente i bersagli.

2. Classificazione e riuso. Stessa ricetta convergente12000iter,6C,
   scaler sul solo train, selezione validation e refittrain, OOF appaiate.
   Pool19k riusato da cache/OOF storiche verificate: policy esplicita
   declared_historical_cache. Parità atol1e−6/rtol1e−5 registra anche
   uguaglianza esatta; una tolleranza non significa nuove OOF identiche.
   Approva o correggi questa lettura prima degli score nuovi.
   40celle nuove,48job: DB2 in coppie[0,1]/[2,3] fissate ora, altre celle4fold.

3. Posa e aggregazione. Ridge pre/post con ricetta istantanea storica,
   mean_K FP16, LayerNorm FP32, crop6 e CV2fold per utente solo sul train.
   La testa allenata è un lettore separato sul medesimo crop6.
   E2 conserva i partner scelti sui bersagli completi, poi usa l'intersezione
   crop6; non si equipara all'E2 storica fullsupport.
   Primario49k−19k separatamente nei pannelli post_ridge e trained_head;
   pre_ridge secondario, post−pre diagnostico, nessun unico riferimento posa.
   Verifica bootstrap completo dei rapporti, IC, supporti e tutti i contrasti,
   V-R1 a due lati e replica T25-L−B19618 come contesto, seme unico/sviluppo.

4. Risorse e gate.12estrazioni da1GPU/8CPU/100GiB/45min, tetto9GPUh;
   osservato storico per3dosi3,5933GPUh prima dell'overhead nuove viste.
   15benchmark classificazione traincompleto fold0,8CPU/30GiB/30min;
   1benchmark posa trainonly,8CPU/30GiB/15min.
   Classificazione:6C+refit al costo del più lento, pavimento storico,
   margini3×1,5+600s, memoria proiettata85%; tutti48shard ammessi
   prima di qualsiasi fit. DB2 coppie4h/gate13800s, come gli altri job.
   Posa: tempi di lettura, algebra, predizione, E2, bootstrap, scritture
   compresse e hash, proiettati senza aprire il held; gate3300s/1h.
   Tutti i dettagli sono nei const/piani. I benchmark misurati precedono
   il preventivo definitivo e i fit. Cap complessivo9GPUh/1711,666667coreh:
   allocazioni massime, non previsione; solo fitclass storico estrapolato
   114,24coreh. Verifica conti e limiti del trasferimento dei tempi19k
   ai nuovi checkpoint, incluse convergenza, I/O e memoria.

5. Esecuzione e completezza.82jobcalcolo+4checkout separati per fase;
   pin futuri null bloccano le fasi premature, source/ricetta immutati,
   pubblicazione prima dei lanci, submitter/ledger comune, massimo8jobCPU
   outstanding dell'intero utente e quota condivisa fresca.
   Spazio stimato24GiB,32GiB prima dell'avvio; poi8GiB più output ancora
   da scrivere. No retry/requeue/overwrite o scarto di fold, nessun monitor.
   Verifica che sintesi e contrasti richiedano tutti i report giusti.

Test riportati:186nella suite integrata, più4nuovi contratti finali;
190test unici, ultimo pacchetto common10/10. Riesegui i cinque nuovi file e
le regressioni pertinenti con OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts usando
~/.venvs/wearusfm-tok/bin/python -m pytest -q -p no:cacheprovider.
Verifica anche bash−n del launcher. Non fare misure CUDA o nuovi job.

Lavora nel tuo worktree o checkout temporaneo. Scrivi review per punto con
APPROVATO / DA CORREGGERE, difetti dimostrati, test e limiti residui,
in docs/reviews/claude_review_consolidation_long_20261105.md; aggiorna le
due sezioni del registro sul tuo ramo, commit/push e verifica SHA remoto.
Restituisci il commit e un testo breve da inoltrare a GPT. Se trovi blocchi,
chiedi correzioni concrete; non cambiare il protocollo né i checkout in uso.

