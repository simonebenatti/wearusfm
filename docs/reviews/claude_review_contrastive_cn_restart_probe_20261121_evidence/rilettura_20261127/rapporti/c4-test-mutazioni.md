# C4 - Test e loro forza dopo le correzioni (8c72655c)

Cartella di lavoro: /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_probe2/agents/c4-test-mutazioni
(mutations.py, run_mutations.py, mutation_log.txt, mutation_results.json, test_c4_extra.py).

Verdetto: nessun bloccante. Il codice corretto si comporta bene anche sui percorsi che la suite non raggiunge: l'ho
verificato eseguendo sei casi in piu' sul codice non modificato. La suite resta debole sulle guardie di validita'.

## 1. Ricevuta (eseguito)
- 211 passati nei 12 file (32,5 s). Per file: compare 18, end_to_end 9, leg 4, observe 30, package 12,
  benchmark_readout 26, job 30, numeric_profile 16, readout_handoff 3, release 22, target_resume 18, targets_readout 23.
- leg e observe (Gloo) eseguiti altre due volte: 34 passati entrambe le volte (8,4 s e 7,9 s).

## 2. Test end-to-end (letto ed eseguito)
- Passano davvero da compare.main(): argv della CLI, report scritto e riletto. Le osservazioni sono scritte da
  ReductionObservation.finish vero. Il costruttore e' saltato con object.__new__ e non c'e' DDP.
- Lo shim del torch 2.2 solleva ValueError se mmap riceve un argomento non str. Il test verifica che gli 8 file
  vectors.rank*.pt siano stati letti con mmap e un percorso str.
- I payload hanno rank_rng (4 dict con torch_rng e cuda_rng) e cuda_rng (lista di 4 uint8 da 16 byte), con i tre
  dormienti uguali all'ordinaria. Il formato coincide con run.py:1482 e 1593-1597.
- L'inventario e' di 4 x 270, con i 270 nomi veri e il modello ridotto (dim=16), e i 7 gruppi ricostruiti veri.
  E' verificato solo negli esiti diversi da INCONCLUSIVE.
- Esiti coperti:
  - SUPPORTED;
  - PARTIAL_SUPPORT solo a 576 (different_weight);
  - HYPOTHESIS_UNSUPPORTED solo per eccesso;
  - INCONCLUSIVE per cinque metadati (cuda_rng.3, cuda_rng.0, rank_rng.3.cuda_rng, torch_rng, contatore) e per la
    fedelta' dell'ordinaria (dormiente alterato).
- NON coperti: PARTIAL_SUPPORT a 545, HYPOTHESIS_UNSUPPORTED per gradienti locali e per assenza di scarto, fedelta'
  dell'ordinaria per metriche, dormiente diverso sia dal continuo sia dall'ordinaria. Questi esiti sono verificati solo
  da classify() con booleani.
- Parti finte:
  - ARCHIVE, FORK_SHA ed ENDPOINT sono sostituiti con monkeypatch: i controlli SHA d'archivio diventano tautologici;
  - fork e checkpoint sono sintetici e minuscoli;
  - le ricevute di confine e post-update sono costanti uguali fra le gambe;
  - i gradienti sono costanti, con locale = rank+1 e ridotto = 2,5;
  - le metriche sono identiche a quelle storiche;
  - posizionamenti e UUID sono finti.
  Sono veri il readout storico gz, le tracce CN_A, i profili e le righe DDP del continuo (con bucket_sizes riscritti).
- I 21 rifiuti di load_observation hanno ora un match specifico.
- Timeout: TimeoutExpired e' iniettato in ordinary, warm e compare, e lo stesso test copre il caso senza errori. Fase,
  completed_phases ed elapsed sono verificati. Il checked_command vero non e' esercitato, e nemmeno il TimeoutError
  pre-fase (mutazione N35 sopravvive).

## 3. Mutazioni (eseguito)
Gli script del 21/11 (r6-test, r4-confronto/mutate.py) e le letture di Leonardo non esistono piu': le cartelle sono
vuote. Ho ricostruito il catalogo da zero: 164 mutazioni, eseguite sui 5 file test_cn_restart_* in copie COW
dell'albero.

| Gruppo | Mutazioni | Sopravvissute |
|---|---|---|
| compare, parti vecchie (C) | 66 | 48 |
| parti nuove (N) | 45 | 9 |
| observe (O) | 26 | 5 |
| leg (L) | 17 | 10 |
| cn_restart (R) | 10 | 6 |
| **Totale** | **164** | **78 (48%)** |

- Guardie scientifiche: 49 sopravvissute su 108. Guardie di integrita': 29 su 56.
- Sui gruppi vecchi (C, O, L, R) sopravvivono 69 su 119 (58%), contro 54 su 84 (64%) del 21/11. Il confronto e'
  indicativo, perche' i cataloghi sono diversi.
- Parti nuove sopravvissute: N09, N15, N17, N21, N27, N35, N37, N41, N45.
- Uccise fra le nuove: matches_ordinary non controllato (N01), same_shape ignorato (N02), esenzione sempre
  attiva (N03), esenzione estesa a cuda_rng.0, torch_rng o rank_rng (N04-N06), scala applicata a tutti i tensori
  (N19), scala non usata (N20), rapporto grezzo (N22), scala x4 o dal ridotto o dal solo ultimo rank (N23-N25), ritorno
  INCONCLUSIVE rimosso o cambiato (N30-N32), percorso Path in mmap (N42), TimeoutExpired non catturato (N34).
- Sopravvissute importanti per l'attribuzione: C61, C64, N09, C51, N21. test_c4_extra.py le uccide tutte e cinque e
  passa sul codice non modificato (6 su 6).
- Sopravvissute equivalenti o ridondanti: C50/C52 (hash e payload si coprono a vicenda), C01, C09.
- Restano senza test negativo: le tracce e i confini (C27-C38), gli SHA d'archivio (C22-C26, C32, C43, C58), la
  coerenza fra rank e fra gambe (C53-C57, C59, C60, C63) e le guardie del worker (L05-L16).

## 4. Che cosa i test non provano sul cluster
- NCCL e CUDA: il comm hook e il clone asincrono prima della divisione sono provati solo su Gloo e CPU.
- torch 2.2: lo shim simula solo il tipo del percorso, non torch.save/load della 2.2 ne' mmap con weights_only.
- device_identity con ctypes non e' mai eseguito: gli UUID sono finti.
- Il ciclo delle fasi gira con comandi finti: non ci sono torchrun ne' SLURM_JOB_END_TIME veri.
- Nel test end-to-end le guardie SHA d'archivio non provano nulla.
- La scala dello schermo l'ho verificata solo in simulazione fp32 su CPU: ordine sequenziale (24 permutazioni) e due
  alberi, rapporto massimo 2,3e-7 contro 1e-6. L'ordine di NCCL non e' provato.
