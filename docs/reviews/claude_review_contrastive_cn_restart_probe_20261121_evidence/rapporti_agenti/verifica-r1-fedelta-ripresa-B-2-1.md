# Verifica avversaria di B-2 (cuda_rng.1..3 nella gamba warm) - esito: REGGE, bloccante

## Eseguito
- Dati veri (lettura di Claude su Leonardo, leonardo/confronto_checkpoint_veri_20261121.out.txt): continuo vs CN_A ->
  metadata_or_schema_different ['cuda_rng.1','cuda_rng.2','cuda_rng.3']; cuda_rng len 4, uguali per dispositivo [True, False, False, False];
  A vs B exact. La funzione checkpoint_differences dello script di lettura e' identica riga per riga a compare.py:42-73 (diff vuoto).
- Simulazione CPU (agents/verifica-r1-fedelta-ripresa-B-2-1/sim.py) con la checkpoint_differences vera: traiettoria identica, solo
  stati di avvio diversi sui dispositivi 1..3 -> metadata_or_schema_different = ['cuda_rng.1','cuda_rng.2','cuda_rng.3'];
  la require di compare.py:248 fallisce.

## Dal codice
- Continuo fresco: run.py:1463 torch.manual_seed(seed*1_000_003+rank+1) semina TUTTE le GPU visibili; rank 0 avanza solo il dispositivo 0.
- Ripresa: run.py:1383 manual_seed(cfg.seed) su tutte, poi run.py:1449-1450 ripristina solo il dispositivo del rank.
- Payload run.py:1482: cuda_rng = get_rng_state_all() del rank 0 (4 stati: ogni processo vede 4 GPU, come mostra len 4 nei dati veri).
- La gamba warm e' una ripresa del trainer di produzione: i dispositivi 1..3 avranno lo stato da cfg.seed, il continuo quello da
  seed*1000003+1. Indipendente dalla traiettoria e dall'ipotesi. Nessun controllo precedente lo intercetta; leg.py tocca solo il RNG
  del dispositivo proprio (leg.py:46-57). compare.py:317-318 trasforma il ValueError in INCONCLUSIVE dopo l'uso delle GPU.
- Anche senza la riga 248, exact_payload (compare.py:72) sarebbe sempre False -> classify (112-123) non puo' dare SUPPORTED
  (recupero esatto classificato PARTIAL_SUPPORT): errata attribuzione.
- Il test test_cn_restart_compare.py:50 non copre questo schema.

## Gravita'
Bloccante: controesempio concreto sui checkpoint veri; il run unico esce INCONCLUSIVE in ogni esito della traiettoria.

## Correzione minima
Nel solo confronto warm vs continuo escludere cuda_rng.1..3, richiedendo che siano uguali a quelli dell'ordinaria (e di CN_A storico),
cosi' si prova che vengono dall'avvio. Mantenere cuda_rng.0, rank_rng, torch_rng e il resto. Dichiararlo in report e protocollo;
test con lo schema esatto (4 stati, differenza solo su 1..3). Ordinary vs CN_A resta sull'intero payload.
