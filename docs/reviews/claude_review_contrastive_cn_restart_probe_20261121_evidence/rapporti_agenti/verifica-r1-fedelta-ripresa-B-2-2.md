# Verifica avversaria di B-2 (cuda_rng.1..3 nella gamba warm) — esito: REGGE, bloccante

Eseguito: agents/verifica-r1-fedelta-ripresa-B-2-2/sim.py (simulazione CPU: stati di generatore CPU al posto dei Philox CUDA,
seminati come nel trainer) con le funzioni vere checkpoint_differences/require/classify del compare al commit 9d34dc69.
Output: {'floating_tensors_different': [], 'metadata_or_schema_different': ['cuda_rng.1','cuda_rng.2','cuda_rng.3'],
'exact_payload': False} -> ValueError "warm endpoint metadata/schema or nonfloat state differs" -> INCONCLUSIVE;
senza la require: classify(...) -> PARTIAL_SUPPORT.

Dedotto dal codice (non eseguito su GPU):
- run.py:1383 torch.manual_seed(cfg.seed) semina tutte le GPU visibili in ogni processo; la ripresa ripristina solo la GPU del
  rank (run.py:1446-1450); il continuo da zero risemina con seed*1000003+rank+1 (run.py:1462-1463). payload (run.py:1482)
  salva get_rng_state_all() del rank 0 -> cuda_rng.1..3 dipendono solo dall'avvio (rank 0 non usa le GPU 1-3).
- Il diagnostico lancia torch.distributed.run --nproc_per_node=4 in un solo task con gres gpu:4 (contrastive_cn_restart.py:130,
  cn_restart_probe.sbatch): ogni processo vede 4 GPU come nella storia. La gamba (leg.py:46-57) tocca solo
  torch.cuda.get/set_rng_state(device) del proprio rank: non altera le GPU 1-3. Quindi warm.cuda_rng.1..3 == CN_A.cuda_rng.1..3
  != continuo, in modo deterministico e indipendente dall'ipotesi.
- Conferma sui checkpoint veri (lettura di Claude): continuo vs A metadata_or_schema_different = ['cuda_rng.1','cuda_rng.2',
  'cuda_rng.3'], cuda_rng.0 uguale.
- Nessun altro controllo lo intercetta prima: compare.py:248 viene subito dopo il controllo dell'ordinaria (246, vs CN_A: passa);
  l'eccezione e' catturata a compare.py:317-318 -> INCONCLUSIVE. Il test test_cn_restart_compare.py non copre lo schema.
- Il guasto e' incondizionato: qualunque esito scientifico, il report esce INCONCLUSIVE dopo l'unico tentativo GPU.

Correzione minima: nel solo confronto warm vs continuo escludere cuda_rng.1..3 (dichiarato nel report e nel protocollo),
richiedere che siano byte-identici a quelli di CN_A/ordinaria; continuare a confrontare cuda_rng.0 e rank_rng; usare
exact_payload calcolato dopo l'esclusione per classify; test con questo schema (len 4, 1..3 diversi per seme d'avvio).
