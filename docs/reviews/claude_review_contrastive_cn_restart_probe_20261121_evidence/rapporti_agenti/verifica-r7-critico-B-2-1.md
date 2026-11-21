# Verifica avversaria di B-2 (chiave verifica-r7-critico-B-2-1)

Esito: il rilievo REGGE. La gravita' giusta e' bloccante.

## Che cosa ho eseguito
- Ho riletto `leonardo/confronto_checkpoint_veri_20261121.out.txt`, la lettura di Claude sui checkpoint veri fatta con torch 2.2 sul cluster e con la stessa `checkpoint_differences`. Riporta `cuda_rng ... uguali per dispositivo: [True, False, False, False]`. Per continuo contro A riporta `metadata_or_schema_different ['cuda_rng.1', 'cuda_rng.2', 'cuda_rng.3']`. Per A contro B riporta `metadati diversi []`.
- Ho rieseguito sul Mac `agents/r7-critico/cuda_rng_sim.py`, una simulazione CPU con stati Philox finti. Usa la vera `checkpoint_differences` e la vera `require` del clone. Uscita: `{'metadata_or_schema_different': ['cuda_rng.1','cuda_rng.2','cuda_rng.3'], ...}` e poi `report -> INCONCLUSIVE: warm endpoint metadata/schema or nonfloat state differs`.

## Che cosa ho dedotto dal codice
- `run.py:1383`: `torch.manual_seed(cfg.seed)` inizializza tutti i dispositivi, in ogni processo.
- Run fresco, `run.py:1463`: `torch.manual_seed(cfg.seed*1_000_003+rank+1)` re-inizializza tutti i dispositivi del processo. Ripresa, `run.py:1447-1450`: ripristina solo il proprio dispositivo. In un processo, i generatori degli altri dispositivi non vengono mai usati. Nel rank 0 restano quindi a `seed*1_000_003+1` nel continuo e a `seed` nella ripresa.
- `run.py:1482`: il payload salva `torch.cuda.get_rng_state_all()` del rank 0, cioe' tutti e 4 i generatori, come tensori uint8.
- La ripresa riscaldata passa dallo stesso percorso di ripresa. `contrastive_cn_restart_leg.py:52-57` (`restore_rng`) tocca solo il dispositivo proprio, quindi i generatori `cuda_rng[1..3]` della riscaldata sono gli stessi di CN_A.
- `compare.py:41-72`: un tensore non in virgola mobile che differisce finisce in `invalid`. `compare.py:241`: confronto fra il checkpoint del continuo e quello della riscaldata. `compare.py:248`: la `require` solleva ValueError, `main` la cattura e il report esce INCONCLUSIVE.
- `checkpoint()` (`contrastive_target_resume.py:57-58`) passa `str(path)` a `torch.load`, quindi il caricamento non cade prima per B-1. La riga 248 si raggiunge davvero. B-1 (`load_observation`) scatta solo dopo, nel ciclo per rank. B-2 e' dunque un difetto indipendente: correggere B-1 non basta.
- L'ordinaria (riga 246-247) supera i controlli: due riprese hanno lo stesso `cuda_rng`, come mostra A contro B.
- Nessun altro controllo intercetta il problema o lo assorbe. Il test `test_cn_restart_compare.py:43-50` prova `checkpoint_differences` solo su payload sintetici e non contiene un `cuda_rng` per dispositivo.
- Il protocollo (riga 18) chiede l'identita' dei «metadati necessari» e la separazione delle «differenze di sola serializzazione». Il codice non fa questa separazione per `cuda_rng`.

## Gravita'
Bloccante. L'esito INCONCLUSIVE e' deterministico, qualunque cosa misurino le GPU. Copre anche gli esiti scientifici `HYPOTHESIS_UNSUPPORTED` e `PARTIAL_SUPPORT`, perche' la riga 248 viene prima della classificazione del 545. Il diagnostico ha un solo tentativo.

## Correzione minima
- Nel confronto della riscaldata con il continuo, escludere `cuda_rng.1-3` (i generatori dei dispositivi non usati dal rank 0) e registrarli in un campo descrittivo.
- Richiedere invece l'uguaglianza esatta di `cuda_rng.0`, `torch_rng` e `rank_rng`.
- Aggiungere un test con la coppia simulata, che si aspetti `metadata_or_schema_different` vuoto dopo l'esclusione.
- Riportare la regola nel protocollo e in decisioni.md, poi ricalcolare `package_sha256`.
