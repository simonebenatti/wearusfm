# Verifica avversaria R4-B1 (cuda_rng di primo livello nel confronto warm a 576)

Esito: il rilievo REGGE, bloccante. Non sono riuscito a confutarlo.

## Catena verificata (dal codice, file:riga del clone 9d34dc69)
- compare.py:219 `previous = {"ordinary": historical_ordinary, "warm": historical_continuous}`; 241 la warm e' confrontata con il checkpoint del continuo (4743abaf), integralmente, da checkpoint_differences (42-72).
- compare.py:53-54: un tensore non float diverso finisce in metadata_or_schema_different; 248 `require(not endpoints["warm"]["metadata_or_schema_different"], ...)` solleva ValueError; 317-318 lo cattura e scrive INCONCLUSIVE. Il require e' PRIMA di classify (301): l'esito e' INCONCLUSIVE in ogni caso, anche se la traiettoria coincide (SUPPORTED mancato) o differisce (UNSUPPORTED mancato).
- run.py:1438 payload `cuda_rng = torch.cuda.get_rng_state_all()` (tutti i dispositivi visibili al rank 0).
- Continuo: summary.json archiviato parte da step 1 (metrics.jsonl 576 righe dal passo 1), quindi ramo else: run.py:1463 `torch.manual_seed(cfg.seed*1_000_003+rank+1)` = 1 per il rank 0 (config.json: "seed": 0); torch.manual_seed chiama cuda.manual_seed_all -> dispositivi 1..3 del rank 0 a seme 1, mai usati dopo.
- Ripresa (ctx.active): ultimo seed_all e' run.py:1383 `torch.manual_seed(cfg.seed)` = 0; 1446-1450 ripristina solo `torch.cuda.set_rng_state(..., device)` del proprio dispositivo. Dispositivi 1..3 a seme 0.
- Visibilita': contrastive_cn_restart.py:130 e contrastive_job.py:88 usano `torch.distributed.run --nproc_per_node=4` senza CUDA_VISIBLE_DEVICES per rank: il rank 0 vede 4 GPU in entrambi i run.
- Nessun altro controllo lo neutralizza: la gamba (leg.py:46-57) salva/ripristina solo il dispositivo corrente; nessuna esclusione di torch_rng/cuda_rng in checkpoint_differences.
- Precedente storico: contrastive_benchmark_readout.py compare_resume (159-190) confrontava rank_rng e ESCLUDEVA torch_rng/cuda_rng di primo livello; il readout storico CN reload_a vs continuo non riporta cuda_rng tra le differenze proprio per questo.
- L'ordinaria non ne soffre: confronta due riprese (stessa costruzione), e la SHA bb39f339 del file lo conferma.

## Eseguito (simulazione dichiarata)
agents/verifica-r4-confronto-R4-B1-2/sim.py, con la checkpoint_differences vera e stato CUDA simulato uint8[16] (seed u64 + offset u64):
continuo (dev1..3 seme 1) vs warm (dev1..3 seme 0), tutto il resto identico ->
{'floating_tensors_different': [], 'metadata_or_schema_different': ['cuda_rng.1','cuda_rng.2','cuda_rng.3'], 'exact_payload': False}
Controllo: con gli stessi semi exact_payload True.
Il test test_all_checkpoint_leaves_checked_and_classified usa payload senza cuda_rng/torch_rng: non copre il caso.

## Limiti
Non ho i .pt (non sul Mac) ne' la sorgente torch 2.2 di random.py nelle letture del cluster; il comportamento torch.manual_seed -> cuda.manual_seed_all e' quello standard di torch dalle 1.x. Se per qualche ragione i due semi coincidessero la riga 248 passerebbe, ma col seed 0 del config 1 != 0.

## Correzione minima
In checkpoint_differences per la gamba warm escludere `torch_rng` e `cuda_rng` di primo livello (riportarne a parte le impronte, come compare_resume storico), confrontando esattamente `rank_rng` (CPU e CUDA del dispositivo usato da ogni rank). Alternativa: confrontare solo cuda_rng[0]. Test: payload completo in cui differiscono solo cuda_rng[1..3] -> exact_payload vero (o campi esclusi riportati). Ordinaria: confronto integrale e SHA invariati.
