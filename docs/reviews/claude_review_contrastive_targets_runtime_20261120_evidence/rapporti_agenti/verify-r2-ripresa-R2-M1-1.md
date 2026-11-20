# Scettico n. 1 su R2-M1 (r2-ripresa) - corsa exists()/mkdir fra rank

## Progressi
- Codice a fbfde797 riletto: scripts/contrastive_run.py:315-323 (controllo exists() per rank prima di init),
  :328-332 (set_device, init_process_group("nccl")), :334-337 (provenance: source_digest + 2 sha), :340 numeric_profile
  (nessuna collettiva), :350-358 mkdir su ogni rank (CN/CR a 352, tutti a 358), :347 barrier NCCL solo nei reload, dopo
  prepare_reload del rank 0 (contrastive_target_resume.py:100, mkdir exist_ok=False). observe_control (parity.py:337) al
  momento dell'ingresso applica solo patch. Base 438c8d65: stesso schema (exists() prima di init, mkdir in R.train run.py:1357).
- Lancio: contrastive_job.py:88 `python -m torch.distributed.run --standalone --nproc_per_node=4`; sbatch GPU
  (scripts/slurm/contrastive_preflight_gpu.sbatch) non imposta TORCH_DIST_INIT_BARRIER ne' TORCH_ENABLE_NCCL_HEALTH_CHECK
  (git grep: nessuna occorrenza nel repo).
- Sorgente torch v2.2.1 (GitHub, tag v2.2.1):
  * local_elastic_agent.py `_start_workers`: `use_agent_store = spec.rdzv_handler.get_backend() == "static"`;
    run.py: `--standalone` -> `rdzv_backend = "c10d"`. Quindi in produzione TORCHELASTIC_USE_AGENT_STORE=False.
  * rendezvous.py `_create_c10d_store`: senza agent store `start_daemon = rank == 0`, TCPStore(world_size=4, multi_tenant=True),
    wait_for_workers di default True.
  * TCPStore.cpp `waitForWorkers`: ogni rank incrementa initKey_, SOLO il server (rank 0) attende che tutti e 4 abbiano
    incrementato.
  * distributed_c10d.py: `_is_barrier_after_init` = int(getenv("TORCH_DIST_INIT_BARRIER","0")) -> nessuna barriera;
    ProcessGroupNCCL::setSequenceNumberForGroup vuota, nessun communicator nel costruttore senza health check.
  => In produzione il rank 0 NON supera init_process_group finche' tutti i rank non hanno fatto il controllo exists().
     I rank 1-3 lo superano appena il server del rank 0 ascolta.
- Emulazione 1 (tmp/verify_r2m1_1/race_emulation.py, torch locale 2.14, init_process_group REALE con env:// e backend pigro
  come NCCL 2.2.1, TORCH_DIST_INIT_BARRIER=0), output in tmp/verify_r2m1_1/emulation_output.txt:
  * S1 full, store sul rank 0 (semantica 2.2.1), rank 2 in ritardo 3 s: rank 1 e 3 fanno mkdir, rank 2 -> ValueError
    "new phase requires a new output directory..."; il rank 0 resta in init: "Timed out ... 3/4 clients joined".
  * S2 full, ritardo sul rank 0: nessun errore (i client aspettano il server). S3 ritardo 0,4 s < finestra 0,6 s: nessun errore.
  * S4 reload, store sul rank 0: init del rank 0 ritorna DOPO il controllo del rank 2 -> nessun errore.
  * S5 reload, agent store (semantica torch >= 2.4): mkdir del rank 0 prima del controllo del rank 2 -> ValueError.
- Emulazione 2, piu' fedele (tmp/verify_r2m1_1/main_harness.py): esegue contrastive_run.main() LETTERALE di fbfde797 su 4
  processi CPU; sostituite solo convalide (con ritardo iniettato), chiamate CUDA, numeric_profile, osservatori e R.train;
  init_process_group reale env:// + backend pigro; TORCHELASTIC_USE_AGENT_STORE=False come in 2.2.1. Output:
  tmp/verify_r2m1_1/harness_output.txt (H1-H4) e harness_output2.txt (H5-H7).
  * H1 calibration, rank 2 in ritardo 3 s: rank 1 e 3 mkdir (riga 358) a 0,464 s; rank 2 a 3,011 s trova out_existed=True ->
    "ValueError: new phase requires a new output directory; continuation must be explicit"; rank 0 "DistStoreError: Timed
    out after 13 seconds waiting for clients. 3/4 clients joined."
  * H2 full CN (riga 352) e H7 full C0 (ritardo sul rank 1): stesso esito.
  * H3 ritardo sul rank 0, H4 nessun ritardo: nessun errore.
  * H5 reload resume_a CN, rank 2 in ritardo 3 s: init del rank 0 ritorna a 3,438 s, dopo il controllo del rank 2 (3,012 s);
    tutti "main() completed". (Una prima esecuzione aveva DistNetworkError artefatto della mia barriera emulata, corretto.)
  * H6 come H5 ma agent store (torch >= 2.4): mkdir del rank 0 a 0,434 s, rank 2 -> "each common-checkpoint restart requires
    a fresh output directory".
- Clone di sola lettura intatto (git status --porcelain --ignored vuoto, nessun __pycache__).

## Esito
Rilievo CONFERMATO nel nucleo (refuted=false), gravita' MINORE, con correzioni:
1. Meccanismo errato per la produzione: con torch 2.2.1 e `torchrun --standalone` NON c'e' agent store; il rank 0 ospita il
   TCPStore(world_size=4) e in init_process_group aspetta tutti i worker. "init_process_group ritorna subito" vale solo per i
   rank 1-3, e solo dopo che il server del rank 0 ascolta.
2. Lo scenario citato (rank 0 a riga 352 mentre il rank 2 e' in verify_reviewed_sources) e' impossibile in produzione. Quello
   reale: un rank non zero (1 o 3) crea out_dir a 352/358 mentre un altro rank non zero e' ancora nella convalida; serve che il
   ritardatario sia in ritardo sia sul rank 0 sia su un altro rank non zero di piu' di set_device + lavoro dopo init.
3. Reload A/B: in produzione la corsa non esiste (non "rischio basso"): l'unico mkdir prima della barriera e' del rank 0
   (contrastive_target_resume.py:100), che non esce da init prima del controllo di tutti (H5). Diventerebbe reale solo con
   torch >= 2.4 (c10d condivide lo store dell'agente per default: H6, dynamic_rendezvous.py locale:1144-1146).
4. Fasi esposte: calibrazione, full C0/CN/CR e, quando rilasciato, training iniziale (stesso percorso 322 -> 358).
5. Probabilita' non misurabile qui: nessun caso storico con lo stesso schema (calibrazione e benchmark a 4 rank completati,
   log in results/infra/contrastive_comparison_20261116/benchmarks_completed_bf16_20261118/*.out). Effetto se capita:
   fallimento chiuso prima delle misure, cartella parziale (numeric_profile.rank1/3.json), nessun retry previsto dal
   prontuario ("un tentativo per ID, nessun retry"): serve un nuovo tentativo approvato con out nuovo. Non BLOCCANTE (non certo).
6. Correzioni proposte valide: TORCH_DIST_INIT_BARRIER=1 nel launcher (in 2.2.1 esegue _store_based_barrier su tutti i rank),
   oppure barriera esplicita subito dopo init prima di ogni mkdir, oppure decisione unica del rank 0 + broadcast.

## Non verificato
- Ambiente reale del modulo cineca-ai/4.3.0 (variabili esportate) e tempi reali per rank su Leonardo (nessun ssh).
- Corpo di ProcessGroupNCCL::init() in 2.2.1 non letto integralmente (comportamento pigro noto, health check spento di default).
