# C2 - Orchestrazione (run), lettore (load_observation), catena CPU - diff 9a2ce7e7..8c72655c

Esito: nessun bloccante. B-1 corretto. Ho trovato due rilievi minori e due note.

## Eseguito
- I 4 file di test CN restart passano: `69 passed in 8.14s`.
- Simulazione di run() (`agents/c2-orchestrazione-lettore/sim_run.py`, output in `sim_run.out`), con il checked_command VERO e figli Python veri. Sono finti solo require_binding, commands, validate_allocation, nvidia-smi e l'ambiente. Esiti in 10 casi:
  - tutto riuscito: 6 righe di tempi, nessun execution_failure.json;
  - uscita 3 in ordinary, warm o compare: RuntimeError, fase e completed_phases corretti;
  - report mancante: ValueError "Missing final comparison", fase compare;
  - timeout in ordinary, warm o compare: TimeoutExpired, fase corretta e figlio ucciso;
  - SLURM_JOB_END_TIME gia' passato: TimeoutError (sottoclasse di OSError), fase ordinary.

  In tutti i casi di errore execution_failure.json e phase_timings.jsonl (start/failed) vengono scritti e l'eccezione risale.
- Catena CPU (`chain.py`, `chain.out`). L'harness r2 non c'e' piu': le cartelle sono vuote dopo il 26/11. Ho usato al suo posto `run_gloo` di `tests/cpu/test_cn_restart_leg.py`: 4 processi Gloo, instrument_leg VERO per ordinary e warm, 270 parametri, finish vero. Poi:
  - load_observation ha girato 8 volte sotto uno shim di torch 2.2 (simulazione: ValueError se mmap=True e f non e' str). Tutte le letture passano un `str`. La vecchia chiamata con Path, sotto lo stesso shim, solleva ValueError: questo conferma B-1 e la sua correzione.
  - Funzioni del confronto: compare_vector, poi screen_reduced_gradients con la mappa degli scalari vuota (il fixture non ha i bias a un elemento), poi warm_endpoint_differences su payload nel formato vero (cuda_rng: 4 uint8; rank_rng: 4 dict; ordinario e continuo diversi solo in cuda_rng.1-3), infine classify. Esito **SUPPORTED**: any_reduced True, any_local False, massimo screen_relative_l2 1,22e-7, exact_scientific_payload True. Le 4x270 righe e l'endpoint passano json.dumps(allow_nan=False).
- Norme (`zero_norm.py`): un gradiente tutto zero o -0.0 scritto da finish viene riletto come norma 0.0 float e accettato. Una norma int scritta a mano e' accettata.
- git diff 9d34dc69..8c72655c: leg.py, sbatch e src/ non cambiano. Nell'osservatore tutti gli hunk cadono in load_observation (righe 144, 161, 169).

## Rilievi
- m-1 (minore), contrastive_cn_restart.py:173 e 186. Le scritture start e completed di phase_timings stanno fuori dal try. Un OSError su quella scrittura (per esempio EDQUOT su WORK) dopo una fase riuscita fa terminare il job senza execution_failure.json. Anche nel ramo except la riga failed viene scritta prima di write_new. L'esito resta comunque "nessun risultato" e non viene attribuito male. Correzione: mettere le scritture dei tempi dentro il try, oppure scrivere execution_failure.json prima della riga failed.
- m-2 (minore), test di run() in tests/cpu/test_cn_restart_package.py. Il test inietta solo subprocess.TimeoutExpired. Non coprono run() i casi RuntimeError (uscita diversa da zero), TimeoutError (scadenza gia' passata) e report mancante. La mia simulazione li verifica; conviene aggiungerli al test.
- n-1 (nota). Se il confronto solleva un'eccezione fuori dalla sua tupla (AttributeError, ZeroDivisionError, TypeError alla scrittura), esce con codice 1 senza report. run() registra allora compare con causa "phase failed (1)" e il dettaglio resta solo nel log di SLURM. L'attribuzione e' corretta (INCONCLUSIVE tecnico).
- n-2 (nota). Con la mappa vera, screen_reduced_gradients esige che i nomi agli id 253 e 257 siano bias a un elemento. Sul modello vero lo verifica il test end-to-end (WearUsFM, names[253] == anchor_heads.rms.bias). Sul fixture della catena la mappa vera solleva ValueError, come previsto.
