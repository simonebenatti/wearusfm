# Prova mirata InfoNCE al passo 545: sequenza operativa del lancio (27/11/2026)

Preparata da Claude (chat «3- log quadro e conrtrastive», revisore della linea 3) per la chat Claude
«0- leggi le tabelline», che esegue il lancio su Leonardo. Il protocollo di riferimento resta quello di GPT nel pacchetto
approvato: `docs/contrastive_cn_restart_probe_20261121.md` al commit di release. Questo file aggiunge solo ciò che serve
a eseguirlo con l'operatore indicato da Simone. Non cambia parametri, risorse, ID o criteri.

## Autorizzazione e costi

- **Operatore.** Simone, 27/11, in chat con il revisore: «in realta farai tu il lancio. Anxi quando è tutto pronto, passa
  alla chat 0-leggi le tabelline e lei si occupera di fare il lancio su leonardo».
- **Scelta della prova.** «Prova mirata», Simone, 21/11 alle 17:32 (registro, commit `e0ecbd4b`).
- **Risorse.** Il 20/11 Simone ha firmato la proposta faa4a8d7 e il tetto di 18 GPU-ore («Sì, firmo entrambi»); la
  decisione registrata in `decisioni.md` indica 18 GPU-ore e 169,6 ore locali per le 17 allocazioni del preflight. Le
  due allocazioni di questa prova sono nuove: GPT le ha messe a preventivo sul residuo di quel tetto (`decisioni.md`,
  voce del 21/11). Il job GPU sta sotto la soglia di 5 GPU-ore delle sottomissioni autonome (Simone, 28/10).
- **Numeri.** Consumati 4,74 GPU-ore e 47,3322 ore locali (consuntivo di GPT, non ricalcolato da SLURM). Tetto
  richiesto da questa prova: **1,0 GPU-ora e 8,2667 ore locali** (job GPU: 4 GPU e 32 CPU per 15 minuti; checkout: 4
  CPU per 4 minuti). A piena allocazione restano 12,26 GPU-ore e 114,0011 ore locali. Con i margini di chiusura di
  SLURM l'inviluppo è 1,4 GPU-ore e 11,8667 ore locali, e il residuo prudenziale 11,86 e 110,4011.
- **Review.** `docs/reviews/claude_review_contrastive_cn_restart_probe_20261121.md`, commit
  `76f5d0e7cface6f02bb24bf572ec57b74171a162`: diagnostico approvato su `8c72655c3d894c82fd4918ed523baf0ac9c69e23`.
- **Un solo tentativo.** Nessun rilancio dopo un errore, un timeout o un esito `INCONCLUSIVE`: si raccoglie e si
  riferisce a Simone e a GPT. Nessun esito abilita training.

## Che cosa è già pronto

| Voce | Valore |
|---|---|
| Commit di release (codice di GPT `8c72655c` più il solo `release_binding.json`) | `b58a66f695c8744dfa5fe633829cafe13a5a1fc4` |
| Ramo su `origin` | `claude/cn-restart545-release-20261127` |
| Worktree locale sul commit di release (non va modificato) | `/Users/simonebenatti/dev/wearusfm/.claude/worktrees/cn-restart545-release` |
| Impronta del pacchetto | `e9310d67b8304c064e09d71b0ec55d37b57abf3b4e2090c2d02d4def586f702d` |
| Impronta del piano | `4f538cf183f6e3daf7bd30a5612eecdbbd07106f64d5f348a2f3e759e877853b` |
| Binding verificato da checkout pulito preso da `origin` | superato sul Mac |
| Submitter condiviso e copia al commit di release | stesso sha256 `814da4bc66dbf18d69ee2ba332675fa0788ab25247d5f5460224cfd24d7e0d82` |

Verifiche già fatte dal revisore il 27/11, tutte in sola lettura:

- I due comandi di invio sono stati provati a secco, con un ssh finto e un registro temporaneo: il submitter compone
  le due righe di `sbatch` attese e passa su stdin gli script del commit di release.
- I 27 file storici che il confronto rilegge da WORK (tracce, profili, metriche, configurazioni di `reloads/CN_A` e
  `benchmark/CN`) sono identici alle copie archiviate nel commit di release. Il fork a 544 è presente (463.142.486
  byte).
- I blocchi di controllo A e B di questa sequenza e il comando `rsync` della raccolta sono stati eseguiti davvero
  (l'`rsync` solo a secco).

## Prenotazione spuria nel ledger, già chiusa

Alle 15:33:41 del 27/11 un agente di verifica del revisore ha eseguito per errore il comando di invio del checkout con
un ssh finto. **Nessun job è arrivato a Leonardo**: `squeue` e `sacct` non mostrano job con i due nomi, e non esistono
né il checkout, né la cartella di uscita, né il riferimento di deploy. Nel registro comune è però rimasta la
prenotazione dell'ID del checkout in stato `ambiguo`.

Il revisore l'ha chiusa con la riconciliazione prevista dal submitter, senza cancellare nulla: due controlli d'assenza
validi, alle 15:44:20 e alle @SECONDO_CONTROLLO@, hanno portato l'ID allo stato `assente`. Per questo **il comando del
checkout al passo 4 contiene `--dopo-assenza-verificata`**: è l'unico modo in cui il submitter accetta un ID in stato
`assente`. Non è un rilancio: quel job non è mai esistito. L'ID del job GPU non è stato toccato.

## Parametri fissati dal codice

| Voce | Valore |
|---|---|
| ID logico del checkout | `CONTRAST-CHECKOUT-CN-RESTART545-20261121` |
| ID logico del job GPU | `CONTRAST-CN-RESTART545-20261121` |
| Campagna nel ledger | `CONTRASTIVE-targets-v3-20261118` |
| Checkout di campagna | `/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/cn-restart545-20261121` |
| Cartella di uscita (deve mancare fino al job GPU) | `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_targets_20261118/cn_restart_probe_20261121` |
| Job di checkout | `lrd_all_serial`, 4 CPU, 2 G, 4 minuti |
| Job GPU | `boost_usr_prod`, `boost_qos_dbg`, 1 nodo, 4 GPU, 32 CPU, 240 G, esattamente 15 minuti |
| Log del checkout (un solo file) | `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/campaign_checkout_<jobid>.out` |
| Log del job GPU | `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/cn_restart_probe_<jobid>.out` e `.err` |

## Regole per chi esegue

- **Ordine.** Ogni passo si esegue solo se il precedente ha dato l'esito atteso. Se un controllo non torna o un comando
  dà un errore inatteso: fermarsi e riferire a Simone, senza improvvisare correzioni.
- **Differenze volute rispetto ai comandi del protocollo di GPT:** `--agente claude` al posto di `--agente gpt`;
  invocazione diretta del submitter condiviso al posto del preparatore `mac_launcher_command.py`; SHA scritto per
  esteso; `--dopo-assenza-verificata` nel solo checkout; due righe `date` nella validazione. Il protocollo nomina
  3-launcher e il fermo del 23-25/11: sono frasi superate, ma il file fa parte del pacchetto vincolato e non si modifica.
- **Esiti del submitter.** L'esito buono è la riga `<ID>: sottomesso, job <numero>` con uscita 0. Con `rifiutato`,
  `ambiguo`, `gia' prenotato` o un'uscita diversa da 0: fermarsi e riferire. Non usare `--dopo-rifiuto`. Non usare
  `--dopo-assenza-verificata` fuori dal passo 4. Solo per un esito `ambiguo` si può riconciliare:
  `/Users/simonebenatti/.venvs/wearusfm-tok/bin/python "/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py" --riconcilia <ID logico>`.
- **Comandi su Leonardo.** Sono blocchi `ssh leonardo 'bash -s' … <<'EOS'` con percorsi scritti per esteso: vanno
  copiati così come sono. Il blocco alimenta la shell remota e non crea né modifica file su WORK. Le righe di avviso di
  ssh nello stderr (per esempio sullo scambio di chiavi) sono normali.
- **Uscite operative.** Si salvano sul Mac in `~/wearusfm_local/cn_restart545_20261127/`, fuori dai worktree.
- **Attesa dei job.** Stato con `ssh leonardo squeue -h -j <jobid>`; quando la riga sparisce,
  `ssh leonardo sacct -X -n -j <jobid> -o JobID,JobName%45,State,ExitCode,Elapsed,AllocCPUS`. Un job in attesa non si
  cancella e non si sostituisce: si aspetta, rileggendo ogni pochi minuti con gli strumenti abituali della chat.
- **Certificato ssh.** Alle 15:05 del 27/11 era valido fino al 28/11 alle 02:34. Se scade con un job in coda o in
  esecuzione non serve fare nulla sul job: si chiede a Simone il rinnovo e si raccoglie dopo.
- **Scritture ammesse su Leonardo:** il push del riferimento di deploy e i due invii. Nient'altro.

## Sequenza

### 1. Controlli sul Mac

```bash
mkdir -p ~/wearusfm_local/cn_restart545_20261127
git -C /Users/simonebenatti/dev/wearusfm/.claude/worktrees/cn-restart545-release rev-parse HEAD
git -C /Users/simonebenatti/dev/wearusfm/.claude/worktrees/cn-restart545-release status --porcelain --untracked-files=all
git -C /Users/simonebenatti/dev/wearusfm/.claude/worktrees/cn-restart545-release ls-remote origin refs/heads/claude/cn-restart545-release-20261127
shasum -a 256 "/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py" /Users/simonebenatti/dev/wearusfm/.claude/worktrees/cn-restart545-release/scripts/submit_job.py
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python "/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py" --stato CONTRAST-CHECKOUT-CN-RESTART545-20261121
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python "/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py" --stato CONTRAST-CN-RESTART545-20261121
```

Esiti attesi, nell'ordine:

- `b58a66f695c8744dfa5fe633829cafe13a5a1fc4`;
- nessuna riga (worktree pulito);
- lo stesso SHA seguito dal nome del ramo;
- due righe con lo stesso sha256 `814da4bc66dbf18d69ee2ba332675fa0788ab25247d5f5460224cfd24d7e0d82`;
- `CONTRAST-CHECKOUT-CN-RESTART545-20261121`, poi `assente`, poi `claude` e la campagna;
- `CONTRAST-CN-RESTART545-20261121`, poi `None` (nessuna prenotazione).

Se lo stato del checkout non è `assente` o quello del job GPU non è `None`: fermarsi.

### 2. Riferimento di deploy su Leonardo

Blocco A, in sola lettura: stato del repository canonico prima del push.

```bash
ssh leonardo 'bash -s' > ~/wearusfm_local/cn_restart545_20261127/A_prima_del_push.out.txt 2> ~/wearusfm_local/cn_restart545_20261127/A_prima_del_push.err.txt <<'EOS'
date "+%d/%m/%Y %H:%M:%S"
C=/leonardo_work/IscrB_WearUsFM/wearusfm
echo "ramo estratto: $(git -C "$C" rev-parse --abbrev-ref HEAD)"
echo "HEAD: $(git -C "$C" rev-parse HEAD)"
echo "voci non pulite: $(git -C "$C" status --porcelain --untracked-files=all | wc -l)"
echo "hook non di esempio: $(ls "$C/.git/hooks" | grep -vc '\.sample$')"
echo "receive.denyCurrentBranch: $(git -C "$C" config --get receive.denyCurrentBranch)"
echo "riferimento deploy/cn-restart545-20261121 gia' presente: $(git -C "$C" for-each-ref refs/heads/deploy/cn-restart545-20261121 | wc -l)"
echo "commit di release gia' presente: $(git -C "$C" cat-file -e b58a66f695c8744dfa5fe633829cafe13a5a1fc4^{commit} 2>/dev/null && echo si || echo no)"
EOS
cat ~/wearusfm_local/cn_restart545_20261127/A_prima_del_push.out.txt
```

Atteso (lettura del revisore alle 15:46): ramo estratto `main`; voci non pulite 0; hook non di esempio 0; riferimento
già presente 0; commit di release già presente `no`. Se il ramo estratto non è `main`, se ci sono voci non pulite o
hook, o se il riferimento esiste già: fermarsi.

Poi il push. Crea un riferimento nuovo e non tocca il working tree del canonico.

```bash
git -C /Users/simonebenatti/dev/wearusfm/.claude/worktrees/cn-restart545-release push leonardo b58a66f695c8744dfa5fe633829cafe13a5a1fc4:refs/heads/deploy/cn-restart545-20261121
git -C /Users/simonebenatti/dev/wearusfm/.claude/worktrees/cn-restart545-release ls-remote leonardo refs/heads/deploy/cn-restart545-20261121
```

Atteso dal secondo comando: `b58a66f695c8744dfa5fe633829cafe13a5a1fc4` seguito dal nome del riferimento. Se il push
viene rifiutato: fermarsi e segnalarlo, senza forzare.

### 3. Controlli su Leonardo prima degli invii

Blocco B, in sola lettura. Il primo comando salva il blocco in un file sul Mac, perché serve di nuovo al passo 6; il
secondo lo esegue su Leonardo.

```bash
cat > ~/wearusfm_local/cn_restart545_20261127/blocco_B.sh <<'EOS'
date "+%d/%m/%Y %H:%M:%S"
W=/leonardo_work/IscrB_WearUsFM
R=$W/wearusfm_runs/results/contrastive_targets_20261118
echo "== cartelle che devono mancare"
ls -d "$R/cn_restart_probe_20261121" 2>/dev/null && echo "ATTENZIONE: la cartella di uscita esiste" || echo "cartella di uscita: assente"
ls -d "$W/wearusfm_campaigns/cn-restart545-20261121" 2>/dev/null && echo "il checkout di campagna esiste" || echo "checkout di campagna: assente"
echo "== riferimenti storici"
ls -l "$R/benchmark/CN/checkpoint_step0000544.pt" "$R/benchmark/CN/checkpoint.pt" "$R/reloads/CN_A/checkpoint.pt" | awk '{print $5, $9}'
echo "== quota di progetto"
PID=$(lfs project -d "$W" | awk '{print $1}' | head -1); echo "ID di progetto: $PID"
lfs quota -hp "$PID" "$W" | tail -2
echo "== saldo del progetto"
saldo -b 2>&1 | grep -E "account|IscrB_WearUsFM|^-" | head -8
echo "== job dell'utente"
squeue -u "$USER" -h -o '%i|%j|%T|%r|%P|%q|%l' | head -20
echo "== job dell'account (solo il numero)"
echo "in esecuzione: $(squeue -A IscrB_WearUsFM -h -t RUNNING | wc -l) | in attesa: $(squeue -A IscrB_WearUsFM -h -t PENDING | wc -l)"
echo "== limiti della coda di debug"
sacctmgr -nP show qos boost_qos_dbg format=Name,MaxWall,MaxSubmitPU,MaxJobsPU | head -2
echo "== prenotazioni attive o future (nome, inizio, fine)"
scontrol show reservation | grep -E "^ReservationName" | awk '{print $1, $2, $3}'
EOS
ssh leonardo 'bash -s' < ~/wearusfm_local/cn_restart545_20261127/blocco_B.sh > ~/wearusfm_local/cn_restart545_20261127/B_prima_del_checkout.out.txt 2> ~/wearusfm_local/cn_restart545_20261127/B_prima_del_checkout.err.txt
cat ~/wearusfm_local/cn_restart545_20261127/B_prima_del_checkout.out.txt
```

Criteri (fra parentesi la lettura del revisore alle 15:46):

- cartella di uscita assente e checkout di campagna assente (entrambi assenti);
- i tre checkpoint storici presenti: 463.142.486 byte il fork a 544, 463.190.144 byte gli altri due;
- quota: nella riga con i numeri, il primo valore è lo spazio usato e il secondo la quota (991G su 1T). I giga liberi
  sono 1024 meno quelli usati: erano circa 33. Alla prova servono 4 GB più il checkout, dopo le scritture degli altri
  job dell'account. Se i giga liberi sono meno di 10: fermarsi;
- saldo: la riga `IscrB_WearUsFM` deve mostrare ore residue (2.000.000 totali, 1.350.952 consumate, cioè il 67,5%);
- coda dell'utente: nessun job su `boost_qos_dbg` o su `lrd_all_serial` (c'era un solo job, di un'altra linea, sulla
  coda normale). I limiti per utente sono 2 job in esecuzione e 4 sottomessi su `boost_qos_dbg`;
- prenotazioni: nessuna manutenzione generale che copra l'ora successiva (quattro prenotazioni di altri account; il
  fermo del 23-25/11 è passato).

### 4. Checkout di campagna (CPU, 0 GPU-ore)

```bash
cd /Users/simonebenatti/dev/wearusfm/.claude/worktrees/cn-restart545-release && /Users/simonebenatti/.venvs/wearusfm-tok/bin/python "/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py" \
  --id CONTRAST-CHECKOUT-CN-RESTART545-20261121 --dopo-assenza-verificata \
  --agente claude --campagna CONTRASTIVE-targets-v3-20261118 --gpu-ore 0 \
  --script-stdin scripts/slurm/campaign_checkout.sbatch -- \
  --cpus-per-task=4 --mem=2G --time=00:04:00 \
  "--export=ALL,CAMPAIGN=cn-restart545-20261121,CAMPAIGN_SHA=b58a66f695c8744dfa5fe633829cafe13a5a1fc4,EXPECTED_SHA=b58a66f695c8744dfa5fe633829cafe13a5a1fc4"
```

Atteso: `CONTRAST-CHECKOUT-CN-RESTART545-20261121: sottomesso, job <numero>`. Quel numero è il jobid del checkout.
Attendere la fine come scritto nelle regole. Atteso da `sacct`: `COMPLETED` e `0:0`, in meno di un minuto di esecuzione
(i due checkout precedenti della linea 3: 39 e 41 s).

Poi la verifica del checkout, in sola lettura:

```bash
ssh leonardo 'bash -s' > ~/wearusfm_local/cn_restart545_20261127/checkout_verifica.out.txt 2> ~/wearusfm_local/cn_restart545_20261127/checkout_verifica.err.txt <<'EOS'
date "+%d/%m/%Y %H:%M:%S"
K=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/cn-restart545-20261121
grep -h "checkout di campagna pronto: $K @ b58a66f695c8744dfa5fe633829cafe13a5a1fc4" /leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/campaign_checkout_*.out || echo "RIGA DI CHECKOUT PRONTO NON TROVATA"
echo "HEAD del checkout: $(git -C "$K" rev-parse HEAD)"
echo "voci non pulite: $(git -C "$K" status --porcelain --untracked-files=all | wc -l)"
EOS
cat ~/wearusfm_local/cn_restart545_20261127/checkout_verifica.out.txt
```

Atteso: la riga `checkout di campagna pronto: … @ b58a66f695c8744dfa5fe633829cafe13a5a1fc4`; HEAD uguale allo stesso
SHA; voci non pulite 0. Se il job non è `COMPLETED 0:0` o la riga manca: fermarsi. Un checkout rimasto a metà non si
ripara e non si rilancia.

### 5. Validazione sul nodo di login

È il blocco del protocollo, con lo SHA scritto per esteso e due righe `date` in più.

```bash
ssh leonardo 'bash -s' > ~/wearusfm_local/cn_restart545_20261127/validate_only.out.txt 2> ~/wearusfm_local/cn_restart545_20261127/validate_only.err.txt <<'EOS'
set -eo pipefail
date "+%d/%m/%Y %H:%M:%S"
export RELEASE_SHA=b58a66f695c8744dfa5fe633829cafe13a5a1fc4
export CODE_DIR="$WORK/wearusfm_campaigns/cn-restart545-20261121"
export EXPECTED_SHA="${RELEASE_SHA:?impostare lo SHA di release pubblicato}"
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source "$WORK/venv_wearusfm/bin/activate"
source "$CODE_DIR/scripts/slurm/lib_code_dir.sh"
code_dir_setup
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 "$CODE_DIR/scripts/diagnostics/contrastive_cn_restart.py" --validate-only
date "+%d/%m/%Y %H:%M:%S"
EOS
echo "uscita: $?"
cat ~/wearusfm_local/cn_restart545_20261127/validate_only.out.txt
```

Atteso: `uscita: 0`. Nello stdout, in ordine: la data; la riga `codice: … @ b58a66f695c8744dfa5fe633829cafe13a5a1fc4`;
una riga JSON del runner con `"recipe_valid": true`, `"release_valid": true` ed `"execution_started": false`; il JSON
del driver, su più righe, con `"new_binding_valid": true` ed `"execution_started": false`; la data finale. I comandi
elencati dentro quel JSON non vanno eseguiti: li esegue il job. La validazione rilegge le impronte dei checkpoint
storici e fa girare il validatore del runner: deve restare molto sotto i 10 minuti di CPU del nodo di login.

Se l'uscita non è 0 o manca `"new_binding_valid": true`: non inviare il job GPU, conservare i due file e riferire.

### 6. Job GPU (1,0 GPU-ora nel ledger)

Subito prima, rieseguire il blocco B salvato al passo 3:

```bash
ssh leonardo 'bash -s' < ~/wearusfm_local/cn_restart545_20261127/blocco_B.sh > ~/wearusfm_local/cn_restart545_20261127/B_prima_del_job_gpu.out.txt 2> ~/wearusfm_local/cn_restart545_20261127/B_prima_del_job_gpu.err.txt
cat ~/wearusfm_local/cn_restart545_20261127/B_prima_del_job_gpu.out.txt
```

Criteri: cartella di uscita ancora assente; ora la riga «il checkout di campagna esiste» è attesa; almeno 10 G liberi
nella quota; nessun job dell'utente su `boost_qos_dbg`. Poi l'invio:

```bash
cd /Users/simonebenatti/dev/wearusfm/.claude/worktrees/cn-restart545-release && /Users/simonebenatti/.venvs/wearusfm-tok/bin/python "/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py" \
  --id CONTRAST-CN-RESTART545-20261121 \
  --agente claude --campagna CONTRASTIVE-targets-v3-20261118 --gpu-ore 1.0 \
  --script-stdin scripts/slurm/cn_restart_probe.sbatch -- \
  --cpus-per-task=32 --mem=240G --time=00:15:00 \
  "--export=ALL,CODE_DIR=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/cn-restart545-20261121,EXPECTED_SHA=b58a66f695c8744dfa5fe633829cafe13a5a1fc4,TORCH_DIST_INIT_BARRIER=1"
```

Atteso: `CONTRAST-CN-RESTART545-20261121: sottomesso, job <numero>`. Attendere la fine come scritto nelle regole: la
coda di debug può far aspettare (alle 15:10 aveva 39 job in attesa).

Esiti possibili del job, tutti da raccogliere senza rilanci:

- `COMPLETED 0:0` con `report.json` nella cartella di uscita: il confronto è arrivato in fondo. Lo `status` del report
  può essere anche `INCONCLUSIVE`.
- `FAILED` con `execution_failure.json`: una fase (`ordinary`, `warm` o `compare`) si è interrotta, anche per tempo
  scaduto; il file dice quale e perché.
- `FAILED` o `TIMEOUT` senza `execution_failure.json`: errore prima delle fasi o chiusura forzata. Una cartella di
  uscita con il solo `allocation.json`, o log quasi vuoti, indicano un errore prima delle fasi; la causa è nei due log.

### 7. Raccolta

L'esito scientifico è il campo `status` di `report.json`: `SUPPORTED`, `PARTIAL_SUPPORT`, `HYPOTHESIS_UNSUPPORTED` o
`INCONCLUSIVE`. La raccolta si fa con qualunque esito.

Primo: elenco, impronte e accounting, letti su Leonardo. Le impronte dei file `.pt` si calcolano qui perché quei file
restano su WORK (circa 3,7 GB in tutto).

```bash
ssh leonardo 'bash -s' > ~/wearusfm_local/cn_restart545_20261127/raccolta_remota.out.txt 2> ~/wearusfm_local/cn_restart545_20261127/raccolta_remota.err.txt <<'EOS'
date "+%d/%m/%Y %H:%M:%S"
O=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_targets_20261118/cn_restart_probe_20261121
L=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs
echo "== elenco della cartella di uscita"
ls -lR "$O"
echo "== impronte dei file .pt, che restano su WORK"
find "$O" -name '*.pt' -type f -exec sha256sum {} +
echo "== log di SLURM del job GPU, con impronte"
ls -l "$L"/cn_restart_probe_*.out "$L"/cn_restart_probe_*.err
sha256sum "$L"/cn_restart_probe_*.out "$L"/cn_restart_probe_*.err
echo "== accounting dei due job"
sacct -X -n -P -S 2026-11-27 --name=CONTRAST-CHECKOUT-CN-RESTART545-20261121,CONTRAST-CN-RESTART545-20261121 -o JobID,JobName,State,ExitCode,Elapsed,AllocCPUS,AllocTRES,NodeList,Submit,Start,End
EOS
cat ~/wearusfm_local/cn_restart545_20261127/raccolta_remota.out.txt
```

Secondo: copia sul Mac di tutto ciò che non è `.pt`, dei log e delle ricevute. La destinazione è nel worktree comune
delle chat Claude, sul ramo `worktree-claude-training-engine`. Nel terzo comando `<JOBID_CHECKOUT>` va sostituito con
il jobid del checkout letto al passo 4.

```bash
D=/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-training-engine/results/infra/contrastive_targets_20261118/cn_restart_probe_20261121/execution_20261127
mkdir -p "$D/run" "$D/logs" "$D/operazioni"
rsync -a --exclude='*.pt' leonardo:/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_targets_20261118/cn_restart_probe_20261121/ "$D/run/"
rsync -a 'leonardo:/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/cn_restart_probe_*' "$D/logs/"
rsync -a leonardo:/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/campaign_checkout_<JOBID_CHECKOUT>.out "$D/logs/"
cp ~/wearusfm_local/cn_restart545_20261127/*.out.txt "$D/operazioni/"
cp ~/wearusfm_local/jobs/CONTRAST-CHECKOUT-CN-RESTART545-20261121/state.json "$D/operazioni/checkout.state.json"
cp ~/wearusfm_local/jobs/CONTRAST-CN-RESTART545-20261121/state.json "$D/operazioni/gpu.state.json"
du -sh "$D"; find "$D" -type f -size +5M
```

Che cosa arriva in `run/`: `report.json` oppure `execution_failure.json`; `phase_timings.jsonl` e `allocation.json`;
per ciascuna gamba (`ordinary/` e `warm/`) gli stessi file dell'archivio storico di CN_A già nel repository
(`config.json`, `start.json`, `summary.json`, `metrics.jsonl`, tracce, profili numerici, i file `fixed_eval*`,
`pose_panel*`, `pose_provenance.json`, `origin_sessions.json`) e le ricevute nuove per rank: `observation.rank*.json`,
`boundary.rank*.json`, `post_update.rank*.json`. Nell'archivio storico una gamba senza `.pt` occupa circa 2,7 MB.

Prima del commit:

- `find` non deve elencare file sopra i 5 MB e nessun `.pt` deve essere arrivato sul Mac: nessun dato da wearable;
- nel repository vanno solo i file `.out.txt` delle uscite operative. I file `.err.txt` restano sul Mac: possono
  contenere la riga del certificato ssh, che non va mai committata;
- le uscite contengono solo job dell'utente e conteggi dell'account, non righe di job di altri utenti.

### 8. Registro, commit e resoconto

- **Dove si scrive.** Nel worktree `/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-training-engine`, ramo
  `worktree-claude-training-engine`. Mai su `main`. Il worktree è condiviso fra le chat Claude: prima `git fetch origin`,
  `git status` e rilettura della coda del registro. Si aggiungono solo i percorsi nominati (`git add` della cartella
  `execution_20261127` e di `docs/claude_GPT_shared`), mai `git add -A`. Alla consegna il revisore lo lascia pulito e
  allineato a `origin`.
- **Registro.** In `docs/claude_GPT_shared` una voce nella sezione «1) Cosa ho fatto» (prima del titolo
  «## 2) Problemi trovati») con: ID e jobid dei due job, stato di SLURM, `status` e `reason` del report, tempi per fase
  da `phase_timings.jsonl`, consuntivo, percorso della raccolta. Eventuali problemi in fondo al file, nella sezione 2.
- **Consuntivo.** GPU-ore = 4 × durata del job GPU in ore. Ore locali = 32 × durata del job GPU + 4 × durata del
  checkout. Le durate sono la colonna `Elapsed` di `sacct`.
- **Commit e push** del ramo, poi verifica dello SHA remoto con `git ls-remote origin refs/heads/worktree-claude-training-engine`.
- **A Simone:** esito, consuntivo e residuo, con il testo pronto da incollare a GPT per la lettura scientifica (ramo,
  commit, percorso della raccolta, `status` del report).
- **Recupero del solo lettore.** Se le due gambe sono complete a 576 e l'unico problema è un errore dimostrato nel
  lettore su CPU, la regola registrata il 22/11 prevede una rilettura su CPU dopo review della correzione, sugli stessi
  dati e con gli stessi criteri, senza nuove GPU. Protocollo e `decisioni.md` non coincidono del tutto su quando si
  applica: il punto è aperto con GPT. Non va avviata d'iniziativa: si riferisce e decide Simone.

## Che cosa non si può verificare prima

L'hook di comunicazione su NCCL, il ramo `libcuda` per l'identità della GPU, i sette gruppi sulla scheda e i tempi
reali (stima 450-620 s su circa 830-875 utili) si vedono solo in esecuzione. Se la gamba ordinaria non riproduce il
file storico CN_A o i gruppi non sono quelli attesi, l'esito è `INCONCLUSIVE`: è un esito previsto dal disegno, non un
errore di chi lancia.
