# Prova mirata InfoNCE al passo 545: sequenza operativa del lancio (27/11/2026)

Preparata da Claude (chat «3- log quadro e contrastive», revisore della linea 3) per la chat Claude
«0- leggi le tabelline», che esegue il lancio su Leonardo. Il protocollo scientifico e operativo di riferimento resta
quello di GPT nel pacchetto approvato: `docs/contrastive_cn_restart_probe_20261121.md` al commit di release. Questo file
aggiunge solo ciò che serve a eseguirlo con l'operatore indicato da Simone; non cambia parametri, risorse, ID o criteri.

## Autorizzazione

- **Simone, 27/11, in chat con il revisore:** «in realta farai tu il lancio. Anxi quando è tutto pronto, passa alla chat
  0-leggi le tabelline e lei si occupera di fare il lancio su leonardo».
- **Scelta della prova:** «Prova mirata», Simone, 21/11 alle 17:32 (registro, commit `e0ecbd4b`).
- **Tetto firmato:** 18 GPU-ore e 169,6 ore locali per la coorte («Sì, firmo entrambi», 20/11). Consumati 4,74 GPU-ore e
  47,3322 ore locali (consuntivo di GPT). Tetto richiesto da questa prova: **1,0 GPU-ora e 8,2667 ore locali** (job GPU:
  4 GPU e 32 CPU per 15 minuti, cioè 1,0 GPU-ora e 8 ore locali; checkout: 4 CPU per 4 minuti, 0,2667 ore locali). Con
  i margini di chiusura di SLURM l'inviluppo è 1,4 GPU-ore e 11,8667 ore locali. A piena allocazione restano 12,26
  GPU-ore e 114,0011 ore locali.
- **Review:** `docs/reviews/claude_review_contrastive_cn_restart_probe_20261121.md`, commit `@REVIEW_SHA@`,
  diagnostico approvato su `8c72655c3d894c82fd4918ed523baf0ac9c69e23`.
- **Un solo tentativo per ID.** Nessun rilancio dopo un errore, un timeout o un esito `INCONCLUSIVE`: si raccoglie e si
  riferisce a Simone e a GPT. Nessun training viene abilitato da questa prova.

## Che cosa è già pronto

| Voce | Valore |
|---|---|
| Commit di release (codice di GPT `8c72655c` + il solo `release_binding.json`) | `@RELEASE_SHA@` |
| Ramo su `origin` | `claude/cn-restart545-release-20261127` |
| Worktree locale sul commit di release | `/Users/simonebenatti/dev/wearusfm/.claude/worktrees/cn-restart545-release` |
| Impronta del pacchetto | `e9310d67b8304c064e09d71b0ec55d37b57abf3b4e2090c2d02d4def586f702d` |
| Impronta del piano | `4f538cf183f6e3daf7bd30a5612eecdbbd07106f64d5f348a2f3e759e877853b` |
| `require_binding()` da checkout pulito del commit di release | superato sul Mac |
| Submitter condiviso | stessi byte di `scripts/submit_job.py` al commit di release (sha256 `814da4bc…`) |

Letture su Leonardo del 27/11 alle 15:08, in sola lettura: cartella di uscita e checkout di campagna assenti; repository
canonico su `main` a `8dad7e82`, pulito, senza hook propri; commit di release non ancora presente sul cluster; quota di
progetto 991 G su 1 T; nessuna prenotazione di manutenzione generale (quattro prenotazioni attive di altri account, due
nodi della partizione in manutenzione); nessuna voce nel ledger per i due ID. Vanno rilette al momento del lancio.

Prova a secco dei due comandi di invio (passi 4 e 6), fatta dal revisore con un ssh finto e un registro temporaneo: il
submitter compone le due righe di `sbatch` attese e passa su stdin gli script del commit di release (sha256
`eaafc432…` per il checkout, `a8c8b8b4…` per il job GPU); nel registro scrive `agente claude`, 0 e 1,0 GPU-ore e il
codice `b58a66f6`. Il ledger vero non è stato toccato.

## Parametri fissati dal codice

| Voce | Valore |
|---|---|
| ID logico del checkout | `CONTRAST-CHECKOUT-CN-RESTART545-20261121` |
| ID logico del job GPU | `CONTRAST-CN-RESTART545-20261121` |
| Campagna nel ledger | `CONTRASTIVE-targets-v3-20261118` |
| Nome del checkout | `cn-restart545-20261121` |
| `CODE_DIR` | `/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/cn-restart545-20261121` |
| Cartella di uscita (deve mancare) | `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_targets_20261118/cn_restart_probe_20261121` |
| Job GPU | `boost_usr_prod`, `boost_qos_dbg`, 1 nodo, 4 GPU, 32 CPU, 240 G, esattamente 15 minuti |
| Log di SLURM | `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/cn_restart_probe_<jobid>.out` e `.err` |

Lo script SLURM rifiuta un `CODE_DIR` diverso, un limite diverso da `00:15:00` e l'assenza di
`TORCH_DIST_INIT_BARRIER=1`. Il driver rifiuta una cartella di uscita già esistente.

## Sequenza

Ogni passo si esegue solo se il precedente è riuscito. In tutti i comandi `RELEASE_SHA` è lo SHA della tabella sopra.

### 1. Controlli sul Mac

```bash
cd /Users/simonebenatti/dev/wearusfm/.claude/worktrees/cn-restart545-release
git rev-parse HEAD            # deve essere @RELEASE_SHA@
git status --porcelain        # deve essere vuoto
git ls-remote origin refs/heads/claude/cn-restart545-release-20261127   # stesso SHA
grep -c "CN-RESTART545" ~/wearusfm_local/jobs/ledger.tsv                # deve essere 0
```

### 2. Riferimento di deploy su Leonardo

Il job di checkout richiede che il commit sia già nel repository canonico del cluster. Si pubblica un riferimento
`deploy/…` nuovo, che non tocca il working tree del canonico (oggi su `main`).

```bash
git push leonardo @RELEASE_SHA@:refs/heads/deploy/cn-restart545-20261121
git ls-remote leonardo refs/heads/deploy/cn-restart545-20261121         # deve dare lo stesso SHA
```

Se il push viene rifiutato: fermarsi e segnalarlo, senza forzare.

### 3. Controlli su Leonardo prima degli invii (sola lettura)

- Cartella di uscita e checkout di campagna ancora assenti.
- Quota: `lfs project -d "$WORK"`, poi `lfs quota -hp <ID> "$WORK"`. Devono restare almeno 4 GB oltre al checkout
  (i checkout recenti occupano fino a circa 114 M). Conservare l'uscita.
- Coda dell'utente e posti liberi su `boost_qos_dbg` (al massimo 2 job in esecuzione e 4 sottomessi per utente).
- `scontrol show reservation`: nessuna manutenzione generale che copra la finestra di esecuzione.

### 4. Checkout di campagna (CPU, 0 GPU-ore)

Dalla radice del worktree di release:

```bash
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python "/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py" \
  --id CONTRAST-CHECKOUT-CN-RESTART545-20261121 \
  --agente claude --campagna CONTRASTIVE-targets-v3-20261118 --gpu-ore 0 \
  --script-stdin scripts/slurm/campaign_checkout.sbatch -- \
  --cpus-per-task=4 --mem=2G --time=00:04:00 \
  "--export=ALL,CAMPAIGN=cn-restart545-20261121,CAMPAIGN_SHA=@RELEASE_SHA@,EXPECTED_SHA=@RELEASE_SHA@"
```

Rispetto al protocollo cambiano solo `--agente claude` e l'invocazione diretta del submitter condiviso al posto del
preparatore `mac_launcher_command.py`, che serviva a 3-launcher. Atteso: `COMPLETED 0:0` in meno di un minuto (i due
checkout precedenti della linea 3: 39 e 41 s). Nel log `campaign_checkout_<jobid>.out` deve comparire
`checkout di campagna pronto: … @ @RELEASE_SHA@`. Verificare poi `git -C "$CODE_DIR" rev-parse HEAD` e lo stato pulito.

### 5. Validazione sul nodo di login

Il blocco è quello del protocollo, con lo SHA scritto per esteso e due righe `date` in più. Va eseguito come script,
non incollato in una shell interattiva (inizia con `set -eo pipefail`). Stdout e stderr si salvano fuori dai worktree,
in una cartella locale dedicata alle uscite operative di questo lancio.

```bash
mkdir -p ~/wearusfm_local/cn_restart545_20261127
ssh leonardo 'bash -s' > ~/wearusfm_local/cn_restart545_20261127/validate_only.out.txt 2> ~/wearusfm_local/cn_restart545_20261127/validate_only.err.txt <<'EOS'
set -eo pipefail
date "+%d/%m/%Y %H:%M:%S"
export RELEASE_SHA=@RELEASE_SHA@
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
```

Atteso: uscita 0 e un JSON con `"new_binding_valid": true` ed `"execution_started": false`, con i due comandi delle
gambe e quello del confronto. La validazione rilegge le impronte di tre checkpoint da 463 MB e fa girare il validatore
del runner: deve restare molto sotto i 10 minuti di CPU del nodo di login. Se fallisce: non inviare il job GPU,
conservare le uscite e riferire.

### 6. Job GPU (1,0 GPU-ora nel ledger)

Subito prima: rileggere la quota e verificare che la cartella di uscita manchi ancora.

```bash
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python "/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py" \
  --id CONTRAST-CN-RESTART545-20261121 \
  --agente claude --campagna CONTRASTIVE-targets-v3-20261118 --gpu-ore 1.0 \
  --script-stdin scripts/slurm/cn_restart_probe.sbatch -- \
  --cpus-per-task=32 --mem=240G --time=00:15:00 \
  "--export=ALL,CODE_DIR=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/cn-restart545-20261121,EXPECTED_SHA=@RELEASE_SHA@,TORCH_DIST_INIT_BARRIER=1"
```

Stato: `ssh leonardo squeue -h -j <jobid>`; quando la riga sparisce, `ssh leonardo sacct -X -j <jobid>`. Se la risposta
all'invio è ambigua: riconciliare l'ID con il submitter (`--riconcilia <ID>`), mai rilanciare.

Esiti possibili del job, tutti da raccogliere senza rilanci:

- `COMPLETED 0:0` con `report.json`: il confronto è arrivato in fondo. Lo `status` può essere anche `INCONCLUSIVE`.
- `FAILED` con `execution_failure.json` nella cartella di uscita: una fase (`ordinary`, `warm` o `compare`) si è
  interrotta; il file dice quale e perché.
- `FAILED` o `TIMEOUT` senza `execution_failure.json`: errore prima delle fasi o chiusura forzata; la causa è nei due log
  di SLURM.

### 7. Raccolta

Il completamento di SLURM documenta l'esecuzione; l'esito scientifico è il campo `status` di `report.json`
(`SUPPORTED`, `PARTIAL_SUPPORT`, `HYPOTHESIS_UNSUPPORTED` o `INCONCLUSIVE`). Con qualunque esito raccogliere sul Mac:

- `report.json`, oppure `execution_failure.json` se una fase si è interrotta;
- `phase_timings.jsonl` e `allocation.json`;
- per ciascuna delle due gambe (`ordinary`, `warm`) tutti i file che non sono `.pt`, come nell'archivio storico di CN_A
  già nel repository: `config.json`, `start.json`, `summary.json`, `metrics.jsonl`, `trace-*.jsonl`,
  `trace-*.ddp.jsonl`, `numeric_profile.rank*.json`, i file `fixed_eval*`, `pose_panel*`, `pose_provenance.json`,
  `origin_sessions.json`, e in più le ricevute nuove del diagnostico: `observation.rank*.json`,
  `boundary.rank*.json` e `post_update.rank*.json` (quattro rank ciascuna). Nell'archivio storico una gamba senza `.pt`
  occupa circa 2,7 MB;
- i due log di SLURM `.out` e `.err`, anche se vuoti, con il loro sha256;
- l'accounting: `sacct -X -j <jobid> -o JobID,JobName%45,State,ExitCode,Elapsed,AllocCPUS,AllocTRES%60,NodeList` per i
  due job;
- l'elenco con dimensioni (`ls -lR`) della cartella di uscita.

I file `.pt` (vettori da circa 229 MB l'uno e checkpoint da 463 MB) restano su WORK: le loro impronte e dimensioni sono
già nelle ricevute e nel report. Prima del commit controllare dimensioni e `git status`: nessun dato da wearable.

Percorso proposto per la raccolta, sul ramo `worktree-claude-training-engine`:
`results/infra/contrastive_targets_20261118/cn_restart_probe_20261121/execution_20261127/`.

### 8. Dopo la raccolta

- Voce nel registro `docs/claude_GPT_shared` con ID dei job, esito di SLURM, `status` del report, tempi per fase,
  consuntivo (GPU-ore e ore locali) e percorso della raccolta; commit e push; verifica dello SHA remoto.
- A Simone: esito e consuntivo, con il testo pronto da incollare a GPT per la lettura scientifica.
- **Recupero del solo lettore.** Se le due gambe sono complete a 576 e l'unico problema è un errore dimostrato nel
  lettore su CPU, la regola registrata il 22/11 prevede una rilettura su CPU dopo review della correzione, sugli stessi
  dati e con gli stessi criteri, senza nuove GPU. Non va avviata d'iniziativa: si riferisce e decide Simone.

## Che cosa non si può verificare prima

L'hook di comunicazione su NCCL, il ramo `libcuda` per l'identità della GPU, i sette gruppi sulla scheda e i tempi
reali (stima 450-620 s su circa 830-875 utili) si vedono solo in esecuzione. Se la gamba ordinaria non riproduce il
file storico CN_A o i gruppi non sono quelli attesi, l'esito è `INCONCLUSIVE`: è un esito previsto dal disegno.
