# verify-r2-ripresa-R2-M1-2 (scettico n. 2, R2-M1 corsa exists/mkdir fra rank)

Inizio: 2026-11-20 03:52

## Lettura del codice a fbfde797
- contrastive_run.py:315-323 controllo exists su ogni rank prima di set_device/init (331-332); mkdir: 352 (benchmark CN/CR), 358 (tutte le fasi); reload: rank 0 in prepare_reload (contrastive_target_resume.py:100) prima della barrier 347; i rank non zero fanno mkdir solo dopo la barrier.
- Lancio reale: contrastive_job.py:88 `python -m torch.distributed.run --standalone --nproc_per_node=4` (confermato nei log storici contrastive_gpu_64771457.out).
- sbatch contrastive_preflight_gpu.sbatch: nessun TORCH_DIST_INIT_BARRIER / TORCH_DISTRIBUTED_DEBUG.

## Torch 2.2.1 (versione della build di Leonardo), sorgenti ufficiali al tag v2.2.1
- local_elastic_agent.py `_start_workers`: `use_agent_store = spec.rdzv_handler.get_backend() == "static"`; con `--standalone` il backend e' c10d -> TORCHELASTIC_USE_AGENT_STORE=False: il TCPStore lo ospita il worker di rank 0 (`_create_c10d_store`, start_daemon = rank == 0, world_size passato, wait_for_workers di default).
- TCPStore.cpp `waitForWorkers`: ogni processo incrementa il contatore, "Let server block until all workers have completed": blocca SOLO il server (rank 0) finche' tutti e 4 si sono connessi; i client tornano subito.
- distributed_c10d.py: `_is_barrier_after_init` legge TORCH_DIST_INIT_BARRIER, default 0: nessuna barriera dopo init.
- ProcessGroupNCCL.cpp: costruttore senza operazioni sullo store ne' creazione del comunicatore (salvo TORCH_ENABLE_NCCL_HEALTH_CHECK); `setSequenceNumberForGroup` vuoto.
- Il torch locale 2.14 invece condivide lo store dell'agente anche col c10d (`use_agent_store` -> TORCH_DISABLE_SHARE_RDZV_TCP_STORE != "1"): la descrizione "agent store ... nessun master li aspetta" del rilievo vale per il torch locale, non per Leonardo.
- Conseguenza per Leonardo: rank 0 non puo' uscire da init prima che tutti abbiano superato il proprio controllo exists; i rank 1-3 invece escono da init appena il server di rank 0 e' su, senza aspettarsi fra loro.

## 03:57 Riproduzione in emulazione (Mac, torch 2.14, nessuna GPU)
Script: rev_runtime/tmp/v2r_scripts/{launch.py,worker.py}; clone temporaneo v2r_clone a 0efa4d45 (nessun file modificato).
Ogni rank esegue il VERO contrastive_run.main; sostituiti solo validazioni pesanti (con ritardo iniettato su un rank), CUDA,
numeric_profile, trace, prepare_reload (mkdir exist_ok=False come :100) e R.train (stub con collettiva finale).
init_process_group e' il VERO percorso torch (env://, TCPStore, flag TORCH_DIST_INIT_BARRIER) con un backend pigro registrato
al posto di NCCL (costruttore senza operazioni sullo store, come ProcessGroupNCCL 2.2.1). Semantica di Leonardo:
TORCHELASTIC_USE_AGENT_STORE=False (rank 0 ospita lo store e aspetta i 4 client).
- full CN, rank 2 in ritardo 1 s: rank 3 crea out_dir a 0,025 s; rank 2 a 1,029 s exists=True ->
  ValueError "new phase requires a new output directory; continuation must be explicit"; uscite [-9,-9,1,-9]. RIPRODOTTO.
- calibrazione, rank 2 in ritardo 1 s: stesso errore (mkdir di rank 3 a 0,024 s). RIPRODOTTO.
- full, rank 0 in ritardo 1 s: passa (i rank 1-3 non escono da init prima che il server di rank 0 sia su).
- full con gloo (costruttore sincronizzante) o con TORCH_DIST_INIT_BARRIER=1: passa (correzione proposta efficace).
- reload resume_a, rank 2 o rank 1 in ritardo 1 s: passa (rank 0 crea la cartella solo dopo che tutti si sono connessi,
  gli altri solo dopo la barrier :347). Con semantica agent store (torch locale/backend static): FALLISCE
  ("each common-checkpoint restart requires a fresh output directory"). Il rischio nei reload e' nullo su Leonardo, non "basso".
- Soglia: con 0,5 s di inizializzazione CUDA emulata, ritardo 0,3 s passa, 1,5 s fallisce: la finestra e' CUDA init +
  lavoro dopo init (source_digest, 2 sha) del rank veloce, piu' fino a 1 s di riprova connect (socket.cpp 2.2.1) se arriva prima di rank 0.

## 04:01 Probabilita' e contesto
- Build di Leonardo 2.2.0a0+git6c8c5ad = commit 6c8c5ad5eaf47a62fafbb4a2747198cbffbf1ff0 = tag v2.2.1 (API GitHub): i sorgenti letti sono quelli di produzione.
- socket.cpp v2.2.1: un client che arriva prima del server di rank 0 riprova dopo 1 s ("Wait one second to avoid choking the server"): la finestra cresce fino a 1 s quando il rank veloce precede rank 0.
- Spread fra 4 processi concorrenti alla fine della NUOVA convalida (verify_reviewed_sources 125+2 git show, 5 source_digest), Mac, 8 prove (v2r_scripts/skew.py): 0-15 ms; durata 2,55-2,72 s per rank. Su Lustre puo' essere maggiore: non misurato.
- Storico con lo stesso schema (controllo prima di init, mkdir in R.train dopo init): 6 lanci a 4 rank --standalone (calibrazione, C0/CN/CR full, split CN/CR) senza questo errore. Campione piccolo.
- Il nuovo codice anticipa il mkdir prima di _read(scales/window_seconds) e del preludio di R.train: finestra un po' piu' stretta di prima.
- Conseguenza se scatta: ValueError fail-closed prima delle misure; prontuario "Ledger unico, un tentativo per ID, nessun retry" e failure_policy "Failed or inconclusive gate stops descendants": cella consumata, catena ferma. Nessuna misura falsata.

## Esito
- NON confutato nel nucleo: per calibrazione, full (e training) la corsa esiste ed e' riprodotta in emulazione fedele alla semantica 2.2.1.
- Correzioni al rilievo: (1) su Leonardo non c'e' agent store: rank 0 ospita lo store e aspetta tutti in init, quindi solo i rank 1-3 possono creare la cartella prima del controllo di un altro rank non zero; lo scenario citato (rank 0 alla riga 352 mentre rank 2 e' in verify_reviewed_sources) e' impossibile; (2) nei reload il rischio e' nullo con torch 2.2.1 --standalone (diventerebbe reale solo con store condiviso dell'agente, es. torch piu' recente).
- Gravita': MINORE confermata (difetto reale, evento improbabile, fail-closed, nessun effetto sulle misure). Correzione proposta verificata: TORCH_DIST_INIT_BARRIER=1 (o barriera su store dopo init) elimina il fallimento nell'emulazione.
- Pulizia: rimosso v2r_clone; restano v2r_scripts (20 KB) e i log v2r_runs (560 KB). Copia di riferimento pulita a 0efa4d45.
