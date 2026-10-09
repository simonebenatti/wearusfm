# WearUsFM

## Architettura di lavoro
- L'agente gira sul Mac. Leonardo e' remoto, raggiungibile con `ssh leonardo`.
- Non editare mai file sotto $WORK via ssh: niente sed, niente heredoc, niente vim.

### Lavoro in parallelo Claude / GPT (Simone, 09/10/2026: «si a tutti e 4»)
- Il `main` canonico sta in `/Users/simonebenatti/dev/wearusfm`: e' il punto d'integrazione. Si sviluppa in **worktree locali su branch
  separati** dello stesso repo: Claude in `.claude/worktrees/claude-training-engine` (branch `worktree-claude-training-engine`), GPT nel
  suo. Si modificano SOLO i file del proprio worktree; dati e checkpoint restano nelle directory esterne di sempre.
- **Integratore unico su `main`: GPT**, tramite merge (niente squash, rebase o riscrittura dei commit dell'altro: restano gli SHA
  originali). Se un commit non va, torna al suo autore con una nota e si corregge sul suo branch. Quando un'integrazione cambia il
  comportamento della parte di Claude, Claude la rivede prima del deploy.
- **Proprietari dei file.** Claude: `training/run.py`, il comando di training, il dataloader (`data/`), i launcher del training, il
  meccanismo di deploy e di sottomissione. GPT: `harness/`, le sonde, i moduli dei target e delle loss (`model/anchor_targets.py`,
  `model/spectral_targets.py`, `training/jepa.py` e le riduzioni delle loss), i launcher dell'harness. Le interfacce condivise (configurazione
  del modello, formato del checkpoint, uscite dell'encoder, contratto delle loss distribuite) si concordano prima. Un'ottimizzazione dei
  target fatta da Claude deve essere equivalente all'originale (test d'uguaglianza), con review di GPT e passaggio esclusivo temporaneo del
  file; ottimizzazione e cambi di definizione (es. zona di bordo) sono modifiche separate.
- **Registro condiviso** (`docs/claude_GPT_shared`): dopo ogni attivita' ciascuno scrive la sua voce, nelle due sezioni, sul proprio branch;
  l'integratore la conserva testualmente. Prima di lavorare: leggere il registro e i commit nuovi di `main`.

## Remote git
- `origin`   -> GitHub (simonebenatti/wearusfm), backup e condivisione con studenti.
- `leonardo` -> $WORK/wearusfm sul cluster, per l'esecuzione.
- Per aggiornare Leonardo: `git push leonardo main`, mai `git push` secco. Dal 09/10/2026 `main` lo pubblica **solo l'integratore**
  (GPT); un branch di lavoro non va mai su `main`.
- **Riferimenti di deploy:** per creare un checkout di campagna si pubblica un branch o tag `deploy/<nome>` (`git push leonardo
  deploy/<nome>`): non tocca il working tree di `$WORK/wearusfm`, quindi non disturba una campagna in corso. Mai pushare il riferimento
  attualmente estratto in un checkout usato da job in coda o in esecuzione.
- Non fare push su origin a meno che non te lo chieda esplicitamente.
- Se un push verso leonardo viene rifiutato il tree remoto e' sporco:
  fermati e segnalalo, non forzare.

## Leonardo
- Account SLURM: IscrB_WearUsFM
- Partizione GPU: boost_usr_prod, nodi 4x A100 64GB, 32 core Ice Lake, 512GB RAM
- QoS: boost_qos_dbg (30 min, max 2 job) per test rapidi; lrd_all_serial (4h) per script seriali
- $WORK = /leonardo_work/IscrB_WearUsFM (project quota 1TB soft, condivisa da 11 utenti -
  scoperto il 23/09/2026 dopo un EDQUOT reale: usare `lfs quota -hp <project-id> $WORK`, non
  `df -h`, che sulla mount Lustre condivisa mostra numeri sbagliati/in ritardo)
- $FAST = /leonardo_scratch/fast/IscrB_WearUsFM  (cancellazione automatica a 40 giorni,
  quota di progetto 1TB condivisa, quasi sempre piena - non affidabile per download grandi)
- $SCRATCH ($CINECA_SCRATCH) = /leonardo_scratch/large/userexternal/sbenatti (scoperto il
  23/09/2026: scratch PERSONALE, non condiviso col progetto, nessuna quota configurata,
  partizione da 43PB. Cancellazione automatica a 40 giorni come $FAST - usarlo per download
  grandi in corso (dataset raw) quando $WORK e' vicino alla quota, MAI per dati che devono
  restare oltre la finestra di addestramento corrente). **Aggiornamento 30/09/2026 (decisione di
  Simone, 29/09):** $SCRATCH e' il deposito principale dei dati grossi: raw copiato e verificato per
  checksum (7 dataset, gli originali in $WORK/data/raw restano finche' Simone non lancia
  `scripts/slurm/delete_verified_raw.sbatch`) e PROCESSATO grande in `$SCRATCH/data/processed/`
  (Kaifosh, NinaPro DB2/3/4/6/7, Hyser). Il processato piccolo gia' fatto resta in
  `$WORK/data/processed/`. **La data di modifica va rinfrescata prima del purge: scadenza
  25/10/2026 (Simone).** `rsync -a` preserva le date vecchie: dopo una copia si rinfresca
  (vedi `scripts/slurm/verify_touch_scratch_raw.sbatch`). Stato dettagliato per dataset:
  `docs/dataset_access.md`, sezione "Stato dell'ingest".
- Manutenzione programmata di Leonardo: 30/09/2026 08:00 - 01/10/2026 08:00 (Booster, DCGP, nodi di
  login/viz): i job sottomessi restano PENDING fino alla fine. Prossima: DCGP, 15-18/10/2026.
- Repo remoto: $WORK/wearusfm
- **Checkout di campagna (eccezione approvata da Simone il 09/10/2026).** Si possono creare checkout congelati del codice SOLO: da un
  commit gia' pubblicato su leonardo; con la procedura versionata nel repo, eseguita da un job CPU su `lrd_all_serial` (mai a mano sul
  login node, mai editing remoto); fuori dalla working tree di `$WORK/wearusfm`. Un checkout usato da job in coda o in esecuzione non si
  modifica. Ogni launcher riceve `CODE_DIR`, verifica lo SHA atteso e che il checkout sia pulito, e importa codice e configurazioni da
  `CODE_DIR`; log, risultati e checkpoint restano fuori dal codice, come sempre.
- **Sottomissioni:** un solo meccanismo comune ai due agenti (prenotazione sotto lock prima di `sbatch`, ID logico nel `--comment`,
  stato persistente; se la risposta di `sbatch` e' ambigua si riconcilia con `sacct`, mai rilancio alla cieca). Finche' non esiste, vale
  la procedura attuale via leonardo-ops.
- **Budget «preparazione dell'infrastruttura della finestra 1»: 25 GPU-ora** (Simone, 09/10/2026), separato dal budget della finestra 1.
  Ogni prova si presenta comunque con preventivo e si lancia solo col via di Simone; consuntivo in `docs/decisioni.md`.
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
- I job SLURM scrivono i loro output (log, file di risultato) SEMPRE fuori dalla working
  tree del repo: `$WORK/wearusfm_runs/`, mai dentro `$WORK/wearusfm/`. Un job che scrive
  dentro `$WORK/wearusfm/` puo' creare file non tracciati che il prossimo
  `git push leonardo main` dal Mac rifiuta (tree remoto "sporco"). Un risultato entra nel
  repo (`results/<passo>/...`) solo quando viene copiato deliberatamente sul Mac e
  committato da li'.

## Vincoli non negoziabili
- Login node: limite di 10 minuti di CPU time. Tutto il resto va in srun o sbatch.
- Compute node: nessun accesso a internet. Niente pip install, niente download di modelli,
  nessuna chiamata API dentro un job SLURM. Gli scaricamenti si fanno prima, dal login node.
- $HOME ha 50GB di quota ed e' gia' stato riempito una volta. Mai dati, modelli o cache li'.
- Ogni interazione col cluster passa dal subagent leonardo-ops.

## Test
- Il codice non gira sul Mac: niente CUDA, architettura diversa.
- tests/cpu/ contiene test leggeri di sintassi e logica, quelli girano in locale, con l'ambiente
  `~/.venvs/wearusfm-tok` (torch 2.14, Python 3.11). Il pacchetto e' installato in modo editabile dal `main` canonico: in un worktree
  i test vanno lanciati mettendo prima `src/` del worktree nel path, per esempio
  `~/.venvs/wearusfm-tok/bin/python -c "import sys; sys.path.insert(0, 'src'); import pytest; sys.exit(pytest.main(['-q', 'tests/cpu']))"`.
- Tutto il resto passa da srun su Leonardo.
- Se esiste .leonardo_jobid c'e' un'allocazione interattiva attiva:
  srun --jobid=$(cat .leonardo_jobid) --overlap bash -lc 'source scripts/setup_env.sh && <cmd>'
- Se non esiste, chiedi prima di allocare. Non fare salloc di tua iniziativa: costa budget.
- sbatch solo per run lunghe e job array.

## Dati
- Dati da wearable: non devono mai finire in un commit. Verifica `git status` prima di ogni push.
