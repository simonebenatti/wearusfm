# Verifica avversaria R3-B1 (cuda_rng[1:] continuo vs riscaldata)

Esito: il rilievo REGGE, bloccante.

Dal codice:
- run.py:1462-1463: nel continuo, che non e' una ripresa, ogni processo fa manual_seed(seed*1_000_003+rank+1), e questo semina tutti i dispositivi.
- run.py:1383 e 1446-1450: nella ripresa c'e' manual_seed(cfg.seed) e poi si ripristina solo il dispositivo proprio.
- run.py:1482: il payload salva get_rng_state_all() del rank 0, cioe' i 4 dispositivi.
- leg.py:46-57: rng_states e restore_rng toccano solo il dispositivo proprio. La gamba riscaldata quindi eredita lo stato della ripresa ordinaria.
- compare.py:44-67: un tensore uint8 diverso va in metadata_or_schema_different.
- compare.py:248 fa require di quella lista vuota. Il ValueError viene catturato in main (riga 318 circa) e il report esce INCONCLUSIVE.
- Il piano e il protocollo non prevedono questa eccezione. tests/cpu/test_cn_restart_compare.py non copre cuda_rng.

Dai dati veri: rev_probe/leonardo/confronto_checkpoint_veri_20261121.out.txt riporta, fra continuo e CN_A, "uguali per dispositivo: [True, False, False, False]" e la lista "metadata_or_schema_different ['cuda_rng.1', 'cuda_rng.2', 'cuda_rng.3']". Fra A e B la lista e' vuota ed exact_payload e' True. La differenza e' quindi deterministica.

Eseguito (simulazione CPU con payload sintetici): checkpoint_differences(cont, warm), con pesi uguali e cuda_rng[1:] seminati come nel trainer, restituisce {'floating_tensors_different': [], 'metadata_or_schema_different': ['cuda_rng.1','cuda_rng.2','cuda_rng.3'], 'exact_payload': False}. Il require della riga 248 fallisce. Togliendo solo il require, classify da' PARTIAL_SUPPORT "warm trajectory does not recover continuous", cioe' un esito attribuito male.

Nessun altro controllo intercetta il caso prima. L'ordinaria contro CN_A passa: e' una ripresa contro una ripresa.

Correzione minima: nel solo confronto riscaldata contro continuo, escludere cuda_rng.1..3 dal confronto esatto e riportarli a parte come differenza attesa. Vanno richiesti uguali cuda_rng.0, rank_rng, torch_rng, step e config. La regola va dichiarata nel protocollo e nel piano, con un test in cui cuda_rng[1:] diversi e tensori uguali danno SUPPORTED. In alternativa la riscaldata puo' allineare esplicitamente lo stato. L'esclusione e' piu' semplice e non tocca il trainer.
