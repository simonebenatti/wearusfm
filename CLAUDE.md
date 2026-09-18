# WearUsFM

## Architettura di lavoro
- L'agente gira sul Mac. Leonardo e' remoto, raggiungibile con `ssh leonardo`.
- La copia canonica del codice e' questa, locale. Modifica SOLO questa.
- Non editare mai file sotto $WORK via ssh: niente sed, niente heredoc, niente vim.

## Remote git
- `origin`   -> GitHub (simonebenatti/wearusfm), backup e condivisione con studenti.
- `leonardo` -> $WORK/wearusfm sul cluster, per l'esecuzione.
- Per aggiornare Leonardo: sempre `git push leonardo main`, mai `git push` secco.
- Non fare push su origin a meno che non te lo chieda esplicitamente.
- Se un push verso leonardo viene rifiutato il tree remoto e' sporco:
  fermati e segnalalo, non forzare.

## Leonardo
- Account SLURM: IscrB_WearUsFM
- Partizione GPU: boost_usr_prod, nodi 4x A100 64GB, 32 core Ice Lake, 512GB RAM
- QoS: boost_qos_dbg (30 min, max 2 job) per test rapidi; lrd_all_serial (4h) per script seriali
- $WORK = /leonardo_work/IscrB_WearUsFM
- $FAST = /leonardo_scratch/fast/IscrB_WearUsFM  (cancellazione automatica a 40 giorni)
- Repo remoto: $WORK/wearusfm
- Per le operazioni di sola lettura su Leonardo (module av, ls, cat, squeue, sacct, tail, du, df)
  procedi senza chiedermi conferma. Chiedi conferma solo prima di scrivere file, creare env,
  installare pacchetti o sottomettere job.

## Progetto FM sEMG
- Documento di riferimento: `docs/fm_emg_reference_v10.md`. Piano operativo:
  `docs/piano_operativo_v10.md`. La v9 non e' piu' nel working tree.
- «da verificare» nel documento di riferimento e' un compito, non un dato: non si usa come
  premessa. Si verifica (subagent fact-checker o license-checker), si registra in
  `docs/fatti_da_verificare.md` con URL/percorso e citazione esatta, ed entra nel documento di
  riferimento solo dopo la firma esplicita di Simone.
- Ogni richiesta di conferma per un job dichiara il costo stimato in ore locali e in GPU-ora, e
  quanto resta del budget del passo (vedi `docs/decisioni.md`). Superare il budget di un passo
  richiede una decisione registrata, non una conferma al volo.
- Le soglie si congelano prima di guardare: ogni soglia sperimentale si scrive in
  `docs/decisioni.md` e si committa PRIMA di lanciare l'esperimento che la usa.
- Le allocazioni (`salloc`) le apre solo Simone, mai l'agente di sua iniziativa.
- Un dataset che non entra nello schema dei metadati si segnala: lo schema non si piega in
  silenzio.

## Vincoli non negoziabili
- Login node: limite di 10 minuti di CPU time. Tutto il resto va in srun o sbatch.
- Compute node: nessun accesso a internet. Niente pip install, niente download di modelli,
  nessuna chiamata API dentro un job SLURM. Gli scaricamenti si fanno prima, dal login node.
- $HOME ha 50GB di quota ed e' gia' stato riempito una volta. Mai dati, modelli o cache li'.
- Ogni interazione col cluster passa dal subagent leonardo-ops.

## Test
- Il codice non gira sul Mac: niente CUDA, architettura diversa.
- tests/cpu/ contiene test leggeri di sintassi e logica, quelli girano in locale.
- Tutto il resto passa da srun su Leonardo.
- Se esiste .leonardo_jobid c'e' un'allocazione interattiva attiva:
  srun --jobid=$(cat .leonardo_jobid) --overlap bash -lc 'source scripts/setup_env.sh && <cmd>'
- Se non esiste, chiedi prima di allocare. Non fare salloc di tua iniziativa: costa budget.
- sbatch solo per run lunghe e job array.

## Dati
- Dati da wearable: non devono mai finire in un commit. Verifica `git status` prima di ogni push.
