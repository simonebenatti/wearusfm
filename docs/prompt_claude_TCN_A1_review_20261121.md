Review mirata TCN A1: tetti e compatibilita delle tre ricevute gia concluse. Nessun lancio e nessun nuovo esperimento; rilettura del solo diff, senza agenti.

Ramo: origin/codex/pose-temponet-launcher-20261120.
Base: c90d43e0b9f905fe13770498a59ffdb7c963413a.
Codice: a1ae9607681c1a181034d7f72770537054af419d.
source_sha256 dei 247 sorgenti: 9aaf1c8b6a050a524dfb60af65df9406313d4fd0e690dab1097bf11df2ec67ea.
Consegna operativa sullo stesso ramo: docs/pose_temponet_execution_A1_20261121.md.
Piano: results/infra/pose_temponet_execution_20261121_A1/plan.json, SHA256 553ce1b5185fc0f996af93c1e6a190813ff56e33c5f05c015e752d1f04ca645f, deliberatamente REVIEW_REQUIRED.

Simone ha autorizzato il budget necessario ×1,5: fit 53,85 GPUh, totale GPU 56,10, CPU-only 115,50 coreh, totale 564,30 coreh. Nessun cambiamento ai walltime, che includono gia fattore 2 e 180 s per job. Scienza, otto input, teste, griglia, dosi, criteri e topologia dei 24 job restano identici.

Preflight 65016127 e benchmark CPU 65016438 / GPU 65016447 sono COMPLETED 0:0. Le originali e le prove sacct stanno in results/infra/pose_temponet_execution_20261120/launcher/. Il nuovo modulo budget_amendment.py ammette solo quelle tre ricevute con hash/byte esatti e il piano base immutabile; i nuovi stadi devono corrispondere alla nuova identita. Non si rifanno preflight o benchmark.

Controlla questi cinque punti:
1. I due controlli di budget usano gli stessi tetti A1; senza A1 restano i vecchi tetti. Il preview passa sulle misure originali, con job, matrice, proiezione e tutti i tempi identici.
2. Il ponte di provenienza non ammette ricevute alterate, non pinnate o di altri job; preserva digest, contesti numerici, dipendenze e vecchia identita dei soli tre predecessori. Nessuna eccezione generale per source/execution hash diversi.
3. Il confronto dell'input_identity dei benchmark contro la base fissa e legittimo: tutti gli algoritmi di benchmark, lettura, modelli, training, controlli, selezione e metriche sono byte-identici. Il nuovo source manifest richiede una nuova review effettiva; la vecchia non rilascia A1.
4. Checkout A1 separato, impossibilita di rigenerare invii dei tre job completati, stesso grafo residuo. Il prontuario include checkout CPU e inventario GPU minimo su un secondo nodo; i due job operativi e i tre pregressi sono contabilizzati nei nuovi tetti (massimo 29 allocazioni). Nessun secondo consenso finanziario necessario.
5. Il piano non e rilasciato e il preview non e un'ammissione. Restano obbligatori: evidenza operatore fresca, release-plan con la review nuova, resource-plan, gate WLS/calibrazione/controlli, confronto hardware prima dei job GPU dipendenti e nessun refresh SCRATCH durante gli stadi.

Test e limiti: validation/local_validation.json distingue 297 test della suite, il passaggio aggiuntivo dei 24 stadi con transizione A1 su sintetico e i 19 test mirati finali. La transizione simula base/pin/review esterni; le tre ricevute reali sono verificate separatamente. Nessuna prova anticipa convergenza delle teste o compatibilita del secondo nodo A100. Bash -n pulito sui quattro launcher invariati.

Per i controlli mirati:
PYTHONPATH=src:scripts PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/.venvs/wearusfm-tok/bin/python -m pytest -q -p no:cacheprovider tests/cpu/test_pose_temponet_budget_amendment.py tests/cpu/test_pose_temponet_resources.py tests/cpu/test_pose_temponet_execution.py tests/cpu/test_pose_temponet_provenance.py tests/cpu/test_pose_temponet_runner.py

Pubblica verdetto e rilievi nel tuo ramo e nelle due sezioni del registro; se APPROVATO, nomina per intero il codice a1ae9607681c1a181034d7f72770537054af419d e il source_sha256 sopra, per release-plan. Restano a GPT i lanci: dopo convergenza procedo senza richiedere di nuovo l'autorizzazione gia data da Simone.
