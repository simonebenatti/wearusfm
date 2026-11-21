# Verifica avversaria di B-2 (cuda_rng[1..3] nel confronto warm/continuo)

Esito: il rilievo REGGE ed e' bloccante. La sua portata e' anche piu' ampia di quanto dichiarato: il cancello non dipende da cio' che la warm recupera, quindi il report esce INCONCLUSIVE con qualsiasi esito delle GPU.

## Che cosa ho fatto

- Eseguito: `agents/verifica-r7-critico-B-2-2/sim.py`. E' una SIMULAZIONE CPU che usa il vero `checkpoint_differences` di compare.py. Gli stati sono generati da `torch.Generator.manual_seed`, e la warm recupera tutti i float del continuo, cioe' il caso piu' favorevole. Output: `{'floating_tensors_different': [], 'metadata_or_schema_different': ['cuda_rng.1','cuda_rng.2','cuda_rng.3'], 'exact_payload': False}` e `gate 248 passa: False`.
- Riletto il dato reale del cluster, `leonardo/confronto_checkpoint_veri_20261121.out.txt`, ottenuto con la stessa funzione. Per CN_A contro il continuo: `cuda_rng ... uguali per dispositivo: [True, False, False, False]` e `metadata_or_schema_different ['cuda_rng.1', 'cuda_rng.2', 'cuda_rng.3']`.

## Ragionamento sul codice

- Il checkpoint viene salvato dal rank 0 con `"cuda_rng": torch.cuda.get_rng_state_all()` (run.py:1482): contiene i 4 dispositivi visibili, lunghezza 4 nel dato reale.
- Nel run fresco: `torch.manual_seed(cfg.seed*1_000_003+rank+1)` (run.py:1463) inizializza tutti i device del processo.
- In ogni ripresa, ordinaria o warm: prima `torch.manual_seed(cfg.seed)` (run.py:1383), poi il ripristino del solo device del rank (run.py:1450). I device 1..3 del processo 0 non vengono mai usati, quindi restano allo stato del seme `cfg.seed`.
- La gamba warm non tocca gli RNG degli altri device. `null_backward` e `restore_rng` (leg.py:46-57, 79-98) agiscono solo su `device`, e leg.py non modifica il salvataggio.
- Ne segue che nella warm, come in CN_A, i valori `cuda_rng[1..3]` differiscono sempre dal continuo. Sono tensori uint8, quindi finiscono in `invalid` (compare.py:54-55).
- Il confronto `previous["warm"] = historical_continuous` (compare.py:219, 232, 243) e il `require` della riga 248 vengono eseguiti senza condizioni e prima del ciclo per rank. Il ValueError che ne risulta viene catturato in main (compare.py:317-318), che scrive `status=INCONCLUSIVE`. Il 545 non viene mai classificato.
- Indipendenza da B-1: B-1 sta in `load_observation`, dentro il ciclo per rank, dopo la riga 248. Correggere B-1 non basta, perche' B-2 scatta prima.
- Nessun altro controllo intercetta il caso. Il test esistente (tests/cpu/test_cn_restart_compare.py:50) verifica proprio che i campi non float finiscano in `metadata_or_schema_different`, e nessun test esercita `compare()` con la struttura reale di `cuda_rng`.

## Gravita'

Bloccante. Controesempio concreto: qualunque run vero, compresi un'ipotesi SUPPORTED e una warm perfetta, produce `INCONCLUSIVE: warm endpoint metadata/schema or nonfloat state differs`. Si consuma cosi' l'unico tentativo.

## Correzione minima

Nel confronto della warm col continuo:
- escludere soltanto `cuda_rng.1`, `cuda_rng.2` e `cuda_rng.3`, cioe' i device diversi dal rank 0, e registrarli in un campo descrittivo, per esempio `non_scientific_device_rng`;
- continuare a richiedere l'uguaglianza esatta di `cuda_rng.0`, `rank_rng` e `torch_rng`;
- in alternativa, richiedere che quei tre stati coincidano con quelli della ripresa ordinaria, che li ha uguali per costruzione (A e B identiche nel dato reale).

Servono poi:
- un test con la struttura reale (lista di 4 stati, device 1..3 da `seed` contro `seed*1_000_003+1`) che si aspetti `metadata_or_schema_different` vuoto;
- la regola scritta nel protocollo e in decisioni.md;
- il ricalcolo di package_sha256.
