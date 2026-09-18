# Piano operativo — esecuzione del FM sEMG su Leonardo

Allineato a `docs/fm_emg_reference_v10.md` · aggiornato al 18/09/2026.

Le sezioni 1–8 descrivono l'infrastruttura: sono quelle del piano scritto per la v9, invariate
salvo dove indicato. Le sezioni 9–13 sono il piano di esecuzione passo per passo, con le
decisioni che spettano a te e il punto in cui vanno prese.

**Novità rispetto al piano per la v9:**
- §5 conversione del budget in GPU-ora, **scadenza letta come vincolo di calendario**
  (111 giorni), quota mensile e linearizzazione
- §6 aggiunte a `CLAUDE.md`, subagent `fact-checker` (proposto), budget per passo, registro
  delle decisioni usato come pre-registrazione
- §8 stato del repo dopo il passo −1
- §9 **registro delle decisioni** · §10 **calendario a ritroso con tagli pre-decisi** ·
  §11 **passi −1 … 6** con ruoli, output, criteri di accettazione, condizioni di stop e budget
  · §12 decisioni dopo il passo 6 · §13 cosa fare oggi
- **Rev. 3 (18/09/2026):** passo 0 riallineato alla spec riscritta
  (`docs/dataloader_bench_spec.md`): due assi invece di tre bracci, D6 divisa in D6a e D6b,
  budget a 40 GPU-ora; D9 con classe di quota per montaggio e pesi dentro la classe
- **Rev. 2 (18/09/2026):** numeri di `saldo -b` e disponibile reale (§5) · **finestre di
  lancio** e passi 7–9 in parallelo (§10) · vertice speculativo (D13) · D17

**Ruoli.** **TU** = Simone · **AG** = agente principale (Claude Code, sul Mac) ·
**OPS** = `leonardo-ops` · **LIC** = `license-checker` · **FACT** = `fact-checker` (§6).

---

## 1. Architettura di lavoro

**Principio: l'agente (Claude Code) gira sul Mac. Leonardo è uno strumento remoto,
non l'ambiente di sviluppo.**

Motivo tecnico: Claude Code è veloce perché Read/Grep/Glob lavorano su filesystem
locale. Con il codice solo su Leonardo ogni ricerca diventa un round trip SSH su
Lustre, e l'effetto pratico non è solo lentezza — l'agente smette di esplorare,
perché esplorare diventa caro, e inizia a indovinare invece di guardare.

Motivo operativo: il login node CINECA ha un limite di 10 minuti di CPU time, che
uccide qualsiasi processo Node a lunga vita (VS Code Server incluso), e i compute
node non hanno accesso a internet, quindi un agente che chiama un'API da lì non
funziona.

### Tre copie del repo

| Copia | Percorso | Ruolo |
|---|---|---|
| Mac | `~/dev/wearusfm` | **canonica** — unica dove si modifica |
| GitHub | `origin` → `simonebenatti/wearusfm` | backup, condivisione studenti |
| Leonardo | `$WORK/wearusfm` | esecuzione |

Il lato Leonardo ha `git config receive.denyCurrentBranch updateInstead`: un
`git push leonardo main` aggiorna i file **in place**, non solo la history, quindi
VS Code Remote mostra subito il codice aggiornato. Se il working tree remoto è
sporco il push viene rifiutato invece di sovrascrivere — è una protezione, non un
errore.

Non sincronizzati: `data/`, `checkpoints/`, `logs/`, `*.out`, `*.npz`, `*.csv`,
`venv/`, `.env`, `.leonardo_jobid`.

---

## 2. Accesso a Leonardo

### `~/.ssh/config` (sul Mac)

```
Host leonardo
  HostName login.leonardo.cineca.it
  User sbenatti
  ProxyCommand bash -c 'step ssh login "simone.benatti@unimore.it" --provisioner cineca-hpc >&2; exec nc %h %p'
  ServerAliveInterval 60
  ServerAliveCountMax 3
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 10m
```

Il multiplexing (`ControlMaster`/`ControlPersist`) serve all'agente: decine di
`ssh leonardo "squeue ..."` di fila riusano un canale già aperto invece di rifare
handshake e invocare `step` ogni volta.

Il `>&2` nel ProxyCommand sposta stdout su stderr: il canale dati resta pulito ma
gli errori si vedono. La versione precedente con `>/dev/null 2>&1` ha nascosto un
problema banale per settimane.

### Problemi già incontrati e risolti

**Login che smette di funzionare.** Causa: i login node vengono reinstallati e
cambia il fingerprint; `known_hosts` va ripulito e ripopolato con `ssh-keyscan` sui
quattro nodi (`login01-ext`, `login02-ext`, `login05-ext`, `login07-ext`) mappati
sotto il wildcard `login*.leonardo.cineca.it`. Senza wildcard il warning ritorna
appena il load balancer manda su un nodo diverso.

**Certificato smallstep:** dura 12 ore, si perde al riavvio del Mac. Il
`ProxyCommand` lo rinnova da solo (apre il browser e chiede OTP se scaduto).

**VS Code Remote: `Failed to install the VS Code Server / UnpackFailed`.** Non era
SSH: `$HOME` era pieno al 100%. Vedi §3.

---

## 3. Storage e quote — lezione appresa

`$HOME` è su **Lustre** (non GPFS: `mmlsquota` non esiste, si usa
`lfs quota -h -u $USER /leonardo`), quota **50 GB**.

Era saturo per **118 GB di cache HuggingFace** + 5,8 GB di pip. Cancellati, ora
~6 GB. Nota: `df -h ~` può continuare a mostrare 100% anche dopo la cancellazione
(valori stantii su Lustre); il test valido è provare a scrivere con `dd`.

### Redirezione permanente (già nel `.bashrc` su Leonardo)

```bash
export HF_HOME=$FAST/hf
export HF_DATASETS_CACHE=$FAST/hf/datasets
export TORCH_HOME=$FAST/torch
export KERAS_HOME=$FAST/keras
export XDG_CACHE_HOME=$WORK/.cache
export PIP_CACHE_DIR=$WORK/.cache/pip
```

Vanno **ripetute** in `scripts/setup_env.sh`, perché `module purge` e i job SLURM
non ereditano sempre il `.bashrc`.

| Area | Percorso | Note |
|---|---|---|
| `$HOME` | `/leonardo/home/userexternal/sbenatti` | 50 GB. Solo dotfile, codice, binari |
| `$WORK` | `/leonardo_work/IscrB_WearUsFM` | 40 TB, condiviso di progetto, **no backup** |
| `$FAST` | `/leonardo_scratch/fast/IscrB_WearUsFM` | 1 TB, I/O veloce, **pulizia a 40 gg** |
| `$PUBLIC` | — | leggibile da chiunque sul cluster |

**Aggiunta.** `HF_HOME` punta a `$FAST`, che viene pulito a 40 giorni: il **checkpoint
NeuroRVQ** (passo 1-bis) va copiato anche sotto `$WORK/models/`, altrimenti sparisce a metà
progetto. Regola generale: **mai l'unica copia di qualcosa su `$FAST`**.

---

## 4. Ambiente Python — collaudato

```bash
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0
source $WORK/venv_wearusfm/bin/activate
```

**L'ordine è vincolante.** Il venv va attivato **dopo** i `module load`: da solo non
vede i pacchetti di cineca-ai e `import torch` fallisce. Il venv è stato creato con
`python -m venv $WORK/venv_wearusfm --system-site-packages`.

Stack risolto:

| Componente | Versione |
|---|---|
| Python | 3.11.6 (gcc) |
| PyTorch | 2.2.0a0+git6c8c5ad (build CINECA da sorgente) |
| CUDA | 12.1 |
| cuDNN | 8.9.7.29-12 |
| NCCL | 2.19.1-1 |
| Driver | 535.274.02 |

Verificato su nodo GPU: `torch.cuda.is_available()` → `True`,
`NVIDIA A100-SXM-64GB`, matmul 8000×8000 eseguita correttamente.

Versioni `cineca-ai` disponibili: 3.0.0, 3.0.1, 4.0.0, 4.1.0, 4.1.1 (default),
**4.3.0** (scelta). L'ambiente include accelerate, apex, datasets, chromadb, ecc.

---

## 5. Risorse SLURM

| Voce | Valore |
|---|---|
| Account | `IscrB_WearUsFM` |
| Budget totale | 2.000.000 ore locali = 250.000 GPU-ora, dal 07/01/2026 |
| Budget residuo (`saldo -b`, 18/09/2026) | **794.303 ore locali ≈ 99.300 GPU-ora**, per l'intero account |
| Disponibile per il FM | **~70.000–90.000 GPU-ora**: chi condivide l'account smetterà a breve; fino ad allora consuma ~5.500 ore locali al giorno |
| Scadenza | 07/01/2027 (**7 gennaio 2027**), confermata da `saldo -b` |
| Quota mensile | 164.383 h locali ≈ ~20.500 GPU-ora, per l'intero account; al 18/09 ne erano consumate 99.394 |
| Partizione GPU | `boost_usr_prod` — 4× A100 SXM4 64 GB, 32 core Ice Lake, 512 GB RAM |

### Contabilità e calendario — aggiunta

- **Unità.** Il budget è in ore locali (core-ora). Sul Booster le ore addebitate sono
  `durata × nodi × R × 32`, dove R è la frazione di nodo richiesta, presa come **massimo**
  fra core, GPU e memoria. Con 1 GPU, al più 8 core e al più ¼ della RAM del nodo R = 0,25,
  quindi **1 GPU-ora = 8 ore locali**; un nodo intero costa 32 ore locali l'ora. **Anche la
  memoria conta:** chiedere più di ¼ della RAM per GPU alza l'addebito.
- **Quota mensile e linearizzazione.** La quota mensile è budget totale / mesi del progetto.
  Dal primo del mese si lavora a priorità piena; man mano che la quota si consuma i job
  perdono priorità, e oltre la quota girano ancora ma dietro a tutti. **Concentrare il consumo
  a dicembre costa priorità proprio quando serve.**
- **Scadenza = 111 giorni dal 18/09/2026.** Alla scadenza le ore non spese si perdono. **Il
  vincolo non sono le GPU-ora, è il calendario** (§10). Con qualunque calendario realistico
  una larga parte della quota scadrà inutilizzata: è l'oggetto della decisione D0.
- **Finestre di lancio.** La priorità piena torna il primo del mese e cala man mano che
  l'account consuma la quota. I lotti grossi si sottomettono quindi **nei primi due-tre giorni
  del mese**, e dentro la finestra **prima i job più grandi**, i più sensibili alla coda. Le
  finestre di questo progetto sono due: **1–3 novembre** e **1–2 dicembre** (§10). Ogni job
  della finestra si prova prima su `boost_qos_dbg`: un job che muore al terzo minuto si
  rilancia a finestra già consumata.
- Verifica: `saldo -b` (le ultime due colonne sono quota e consumo del mese). Fatta il 18/09;
  OPS la ripete a ogni inizio mese e il giorno prima di ogni finestra, riportando **solo
  l'output**.

### QoS verificate

| QoS | MaxWall | MaxTRESPerUser | Uso |
|---|---|---|---|
| `boost_qos_dbg` | 30 min | 256 cpu, 32 gpu, max 2 job | test rapidi |
| `boost_qos_bprod` | **24 h** | 256 nodi | **default per il training** |
| `boost_qos_lprod` | 4 giorni | 8 nodi, 32 gpu, priority 40 | da rivalutare |
| `lrd_all_serial` | 4 h | 1 nodo, 8 cpu, 30 GB | download, preprocessing seriale |

**Decisione presa: si resta su 24 ore con `bprod`.** `lprod` darebbe 4 giorni ma su
soli 8 nodi, con priorità più bassa e varianza di coda alta (su job grandi si sono
osservate attese fino a 80 ore, e molti job `lprod` finiscono in TIMEOUT netto a 4
giorni). Con 24 ore servono comunque checkpoint frequenti, che vuoi avere in ogni
caso. Da rivalutare solo se i restart diventassero un collo di bottiglia misurato.

### Pattern per i test su GPU

```bash
salloc -A IscrB_WearUsFM -p boost_usr_prod --nodes=1 --gres=gpu:1 \
       --cpus-per-task=8 --time=02:00:00
echo $SLURM_JOB_ID                          # es. 58103602
srun --jobid=58103602 --overlap --pty bash  # shell sul nodo
```

Due proprietà utili:

- `srun --overlap` attacca nuovi step a un'allocazione già in mano, **senza rifare
  coda**. L'agente può quindi lanciare test in secondi mentre tu guardi
  `watch -n1 nvidia-smi` nella tua finestra.
- `srun --pty bash` dà la shell sul nodo **senza passare da ssh**, quindi evita
  tutta la procedura di chiavi generate dentro Leonardo e ProxyJump.

Il job id va scritto in `.leonardo_jobid` (locale, gitignored) perché l'agente lo
legga. `--overlap` è obbligatorio: senza, `srun` aspetta che si liberino le risorse
dello step esistente e si pianta.

Questa allocazione (1 GPU, 8 core, 2 ore) costa **16 ore locali**: è la forma giusta per i
passi 0, 1-bis e 3.

### Vincoli non negoziabili

- **Login node: 10 minuti di CPU time.** Tutto il resto va in `srun` o `sbatch`.
- **Compute node: nessuna rete.** Niente `pip install`, niente download di modelli,
  nessuna chiamata API dentro un job. Gli scaricamenti si fanno prima, dal login
  node o da `lrd_all_serial`.
- Il codice **non gira sul Mac** (no CUDA, architettura diversa): ogni verifica
  passa da `srun`. Conviene un `tests/cpu/` minimale per sintassi e logica.

---

## 6. Configurazione dell'agente

`.claude/settings.json`:

```json
{
  "model": "opusplan",
  "permissions": {
    "allow": ["Bash(ssh leonardo:*)", "Bash(git:*)", "Bash(rsync:*)"]
  }
}
```

`opusplan` = Opus in plan mode, Sonnet in esecuzione.

**Non impostare `CLAUDE_CODE_SUBAGENT_MODEL`**: sovrascrive sempre il campo `model`
di ogni agente, togliendo proprio il controllo fine che serve.

Il wizard `/agents` è stato rimosso da Claude Code: i subagent si definiscono
scrivendo direttamente i file markdown in `.claude/agents/`.

### Subagent definiti (entrambi su Haiku)

**`leonardo-ops`** — `tools: Bash, Read, Grep`. Esegue tutto via `ssh leonardo`.
Regole nel prompt: sul login node solo sbatch/srun/squeue/sacct/scancel/ls/tail/du/df;
dopo `sbatch` riporta il job id e si ferma; se un job fallisce riporta le ultime 50
righe di log **senza interpretarle**; non modifica sorgenti, non fa commit né
checkout sul lato remoto; **non lancia `salloc` di sua iniziativa** (costa budget).

**`license-checker`** — `tools: WebFetch, WebSearch, Read, Write`. Verifica licenze
e condizioni di accesso dei dataset. Deve leggere la pagina ufficiale, mai fonti
secondarie, e scrivere "non trovata" invece di dedurre.

### `CLAUDE.md` — contiene i vincoli non indovinabili

Quale copia è canonica, i due remote e quale usare, il limite di CPU sul login node,
l'assenza di rete sui compute node, la quota di `$HOME`, il pattern
`srun --jobid --overlap` per i test, il divieto di `sbatch` per i test brevi, e la
regola che i dati da wearable non devono mai finire in un commit.

C'è anche una regola di autonomia: operazioni di sola lettura su Leonardo
(`module av`, `ls`, `cat`, `squeue`, `sacct`, `tail`, `du`, `df`) senza chiedere
conferma; conferma richiesta solo prima di scrivere file, creare env, installare
pacchetti o sottomettere job.

### Aggiunte per la v10

**Da aggiungere a `CLAUDE.md`:**
- Il documento di riferimento è `docs/fm_emg_reference_v10.md`; il piano è questo file. La v9
  non è più nel working tree.
- **«da verificare» è un compito, non un dato.** Un fatto così marcato non si usa come
  premessa: si verifica (FACT o LIC), si registra in `docs/fatti_da_verificare.md` con URL e
  citazione esatta, e **entra nel documento di riferimento solo dopo la firma di TU** (§7).
- **Ogni richiesta di conferma per un job dichiara il costo stimato** in ore locali e in
  GPU-ora, e quanto resta del budget del passo (§11). Superare il budget di un passo richiede
  una decisione registrata, non una conferma al volo.
- **Le soglie si congelano prima di guardare.** Ogni soglia di §9 si scrive in
  `docs/decisioni.md` e si committa **prima** di lanciare l'esperimento che la usa: la history
  di git è la prova della pre-registrazione.
- Le allocazioni (`salloc`) le apre solo TU.
- Un dataset che non entra nello schema dei metadati **si segnala**: lo schema non si piega in
  silenzio.

**Subagent proposto: `fact-checker`** — stessa forma e stessi vincoli di `license-checker`
(pagina ufficiale o paper, mai fonti secondarie; "non trovata" invece di dedurre), ma per i
*fatti da verificare* di §12 della v10: frequenze native, montaggi, liste di muscoli, dettagli
del tokenizer. Scrive in `docs/fatti_da_verificare.md` una riga per fatto:
fatto · fonte (URL) · citazione esatta · stato (`raccolto` → `verificato da SB`).
In alternativa si estende il prompt di `license-checker`; tenerli separati evita che un
controllo di licenza si porti dietro il contesto di dieci paper.

---

## 7. Affidabilità dell'agente — tre episodi da tenere presenti

**La delega funziona davvero.** `leonardo-ops` viene invocato come processo separato
(si vede il `SendMessage` con agentId) e l'output verboso resta nel suo contesto
invece di saturare il thread principale. Questo è il beneficio maggiore, più del
risparmio sul costo del modello: `module av` produce 500 righe che non inquinano la
sessione.

**Ha scambiato una credenziale di cortesia per la procedura ufficiale.** Sulle pagine
NinaPro compare un login `reviewers`/`rev2019` lasciato per i revisori dei paper;
l'agente l'ha riportato come il meccanismo di accesso. Ha letto correttamente la
pagina, ha capito male cosa fosse. **Regola: i fatti che entrano in un documento
condiviso vanno verificati da te, non solo raccolti.**

**Ha interpretato quando gli era stato chiesto di non farlo.** A un comando che
chiedeva output numerico grezzo ha risposto con un'analisi qualitativa degli stati
dei job, saltando il calcolo richiesto. In compenso, davanti a un output da 29.000
righe ha correttamente dichiarato di non poterlo restituire integralmente invece di
inventarlo.

**Regole operative che ne derivano:**
- Per i comandi diagnostici, **aggregare lato cluster** (un `awk` che calcola
  mediana e percentili) invece di chiedere dump grezzi: torna una riga sola e non
  c'è spazio per troncamenti o riassunti.
- Scrivere esplicitamente "riportami solo l'output, senza interpretarlo" quando
  serve il dato grezzo — e sapere che potrebbe comunque non bastare.

**Conseguenza per questo piano.** I criteri di accettazione di §11 sono **numeri prodotti da
uno script e salvati in un file** (`results/<passo>/*.json`), non giudizi riportati
dall'agente. Un passo è chiuso quando il file esiste e il numero rispetta la soglia congelata.

---

## 8. Stato del repo — dopo il passo −1

```
~/dev/wearusfm/
├── CLAUDE.md                          ← aggiornato (§6)
├── .gitignore
├── .claude/
│   ├── settings.json
│   └── agents/
│       ├── leonardo-ops.md
│       ├── license-checker.md
│       └── fact-checker.md            ← nuovo, se adottato (§6)
├── docs/
│   ├── fm_emg_reference_v10.md        ← sostituisce la v9, che resta nella history
│   ├── piano_operativo_v10.md         ← questo file
│   ├── decisioni.md                   ← registro: ID, data, decisione, soglie congelate
│   ├── fatti_da_verificare.md         ← fatto · fonte · citazione · stato
│   ├── dataloader_bench_spec.md       ← spec del passo 0 (rev. 2)
│   └── dataset_access.md
├── scripts/
│   └── setup_env.sh
├── src/                               ← vuoto: si parte da qui
└── tests/cpu/
```

---

## 9. Registro delle decisioni

Ogni decisione ha un punto preciso in cui va presa. Quelle marcate **prima** sono soglie da
congelare in `docs/decisioni.md` *prima* di lanciare l'esperimento che le usa.

| ID | Decisione | Dove | Entro | Cosa blocca |
|---|---|---|---|---|
| **D0** | Calendario e continuità: proroga, nuovo progetto, o tagli | passo −1 | oggi | tutto il §10 |
| **D1** | Autonomia dell'agente e budget per passo | passo −1 | oggi | ogni job |
| **D2** | Parametri provvisori del benchmark (quote, patch, contesto) — non vincolanti | passo 0 | prima di lanciare | passo 0 |
| **D6** | Dataloader — (a) montaggi al volo o precompute · (b) layout: padding, packing o bucketing per micro-batch. Regole **prima**, esito dopo | passo 0 | 04/10 | manifest (`D_c`); v10 §2.7 se vince il bucketing |
| **D3** | Dataset di Kaifosh: (a) scaricarlo · (b) ruolo nel manifest | (a) passo 1 · (b) passo 4 | (a) 04/10 · (b) 25/10 | harness, manifest |
| **D5** | Ancora RVQ: (a) soglie X, Y e criterio V4 **prima** · (b) esito | passo 1-bis | (a) prima di lanciare · (b) 11/10 | run 5, quota di slab nel masking |
| **D4** | Firma dei fatti verificati → documento v10.1 | passo 1-bis | 11/10 | ingresso dei fatti nel riferimento |
| **DP** | Preprint autonomo di ri-valutazione: sì/no, e chi lo porta | passo 5 | questa settimana | persone, non GPU |
| **D7** | (a) firma dello schema dei metadati · (b) revisione della tassonomia | passo 2 | (a) **prima di ogni ingest** · (b) 18/10 | tutto l'ingest |
| **D8** | Gate di consistenza: (a) soglie **prima** · (b) se fallisce, insistere o ripiegare | passo 3 | (a) prima di lanciare · (b) 18/10 | tutto ciò che segue |
| **D9** | Manifest: quote per topologia, ruolo Kaifosh, split dei soggetti, target RVQ | passo 4 | 25/10 | **irreversibile lungo la ladder** |
| **D10** | Patch (ms), contesto (s), schedule di masking | passo 6 | prima di scrivere il modello | modello |
| **D11** | §5.6 della v10: target JEPA e decoder a query | passo 6 | prima del passo 7 | ablation SSL |
| D12–D17 | pilot, vertice, WSD, dropout del livello muscolo, asse dei soggetti, training finale | dopo il 6 | §12 | finestre di lancio |

---

## 10. Calendario a ritroso e finestre di lancio

**Proposta: D0 la conferma o la cambia.** Dal 18/09/2026 al 07/01/2027 ci sono 16 settimane
scarse e `src/` è vuoto. **Fino al passo 6 le GPU-ora non comprano nulla:** il tempo lo
decidono lo sviluppo, gli accessi ai dataset e le tue firme, e il calendario è senza margine.
**Dopo il passo 6 il budget compra parallelismo:** i passi 7, 8, 9 e 9-bis partono in blocco
invece che in fila, e la ladder parte tutta insieme. Il margine che prima non c'era viene da lì.

| Settimane | Date | Traguardo (data limite = fine del periodo) |
|---|---|---|
| W1–W2 | 21/09 – 04/10 | passo −1 · passo 0 chiuso con D6 · download avviati · **schema firmato (D7a)** |
| W3 | 05/10 – 11/10 | passo 1-bis chiuso sui dataset aperti · D4 · D5b |
| W4 | 12/10 – 18/10 | **gate di consistenza superato (D8)** · harness che riproduce NeuroRVQ |
| W5 | 19/10 – 25/10 | ingest completo · **manifest congelato (D9)** |
| W6 | 26/10 – 01/11 | modello completo, sanity JEPA (passo 6) · job della finestra 1 provati su `boost_qos_dbg` · D12, D14, D15, D16 |
| **Finestra 1** | **dom 01/11 – mar 03/11** | **lancio in blocco dei passi 7, 8, 9 e 9-bis** |
| W7–W8 | 02/11 – 15/11 | i job girano; analisi man mano che chiudono |
| W9–W10 | 16/11 – 29/11 | decisioni su obiettivo, architettura ed E · run di conferma a 100M se un default è caduto · shakedown del percorso FSDP a 1B · D17 · job della finestra 2 provati su `dbg` |
| **Finestra 2** | **mar 01/12 – mer 02/12** | **lancio in blocco della ladder**: tutti i rung, 3 seed, dal più grande al più piccolo |
| entro il 06/12 | — | vertice speculativo (D13), se le condizioni sono rispettate; dopo non si lancia più |
| W11–W12 | 30/11 – 13/12 | ladder · controlli 2D dai checkpoint pre-decay, appena si vede il flesso |
| W13–W14 | 14/12 – 27/12 | code dei 2D · training finale separato solo se D17 lo chiede · valutazione avviata — periodo festivo, code imprevedibili |
| W15–W16 | 28/12 – 07/01 | valutazione su GPU (passo 13) · **07/01: scadenza** |

### Le due finestre

**Finestra 1 — 1–3 novembre.** In blocco, coi **default di lavoro** della v10 (obiettivo JEPA
+ ancora fisica, target dal decoder a query, concatenazione, K = 64, encoder locale a 2 layer,
E = 4): ablation dell'obiettivo SSL e della collocazione dei target JEPA a 30M; ablation dei
percorsi di identità; calibrazione a 30–100M; pilot sulle epoche a 100M; asse dei soggetti a
100M e 300M. **Ogni esperimento usa i default degli altri.** Se un default cade, in W9–W10 si
fa una run di conferma a 100M con le scelte finali; se cade quello dell'obiettivo, si ripete
il punto E = 4 del pilot e si controlla che la regola D12 dia lo stesso esito. Col WSD il
pilot è **una sola run per seed**, con rami di decay a 1, 2, 4 e 8 epoche: per questo D14 va
chiusa prima della finestra.

**Finestra 2 — 1–2 dicembre.** La ladder tutta insieme: 30M, 100M, 300M, 1B, **3 seed per
rung**, sottomessi **dal più grande al più piccolo**. Il rung a 1B gira sul percorso **FSDP**
anche se DDP basterebbe: così il vertice, se parte, è solo un cambio di configurazione. I
controlli 2D partono dai checkpoint pre-decay appena si vede il flesso.

**Regole comuni.** Job da 24 ore in catena con `--dependency`; ogni job provato prima su
`boost_qos_dbg`; prima i job grandi; `saldo -b` il giorno prima. A novembre la finestra è una
buona abitudine: il FM da solo resta sotto la quota mensile. **A dicembre è necessaria:** col
vertice il consumo del mese supera la quota dell'intero account.

Consumo atteso del FM (stima): sotto le 1.000 GPU-ora fino a fine ottobre; ~6.000–7.000 a
novembre; ~10.000–13.000 a dicembre senza il vertice, ~22.000–29.000 col vertice, contro una
quota mensile di ~20.500. Totale ~17.000–21.000 senza vertice e ~29.000–37.000 con, contro
70.000–90.000 disponibili.

### Tagli pre-decisi

| Se entro il… | non è fatto… | allora |
|---|---|---|
| 18/10 | gate di consistenza (passo 3) | D8b: i front-end separati per frequenza diventano il default e si prosegue |
| 25/10 | accesso a un dataset (NinaPro, CSL-hdemg) | il manifest si congela senza. **NinaPro è il caso sparso, cioè il deployment: l'account va chiesto oggi** |
| 25/10 | verifiche RVQ complete (passo 1-bis) | l'ancora RVQ esce dal manifest, la run 5 non si fa |
| 01/11 | modello completo (passo 6) | la finestra 1 slitta. A novembre la quota lo tollera, ma ogni giorno perso esce dall'analisi di W9–W10 |
| 08/11 | modello completo | via le ablation secondarie e quella dei percorsi di identità (resta la concatenazione) |
| 15/11 | finestra 1 lanciata | niente run di conferma: la ladder parte coi default di lavoro ed E = 4 per prior |
| 29/11 | job della finestra 2 provati su `dbg` | **la finestra 2 non si sposta**: si lancia ciò che è pronto, a partire dai rung grandi |
| 06/12 | condizioni di D13 rispettate | il vertice non si lancia |
| 06/12 | rung a 1B stabile | modello finale sul migliore fra 100M e 300M |

Il passo 5 (harness e preprint autonomo) **non è sul cammino critico** e non dipende dal
corpus: è l'uscita garantita anche se il resto slitta. Per questo parte subito (DP).

---

## 11. Piano passo per passo — fino al passo 6

Budget per passo, da confermare con D1. Totale fino al passo 6: ~520 GPU-ora più l'ingest su
CPU, cioè meno dell'1% del disponibile.

| Passo | Budget | In ore locali |
|---|---|---|
| 0 | 40 GPU-ora | 320 |
| 1 | solo `lrd_all_serial` | — |
| 1-bis | 30 GPU-ora | 240 |
| 2 | ≤ 100 nodo-ore di CPU | ≤ 3.200 |
| 3 | 50 GPU-ora | 400 |
| 5 | 300 GPU-ora | 2.400 |
| 6 | 100 GPU-ora (incluso il sanity JEPA) | 800 |

### Passo −1 — Allineare repo e regole · oggi · 0 GPU-ora

- **AG:** mette `docs/fm_emg_reference_v10.md` e questo file nel repo; `git rm` della v9 (resta
  nella history: l'agente non deve poter leggere la versione sbagliata); crea
  `docs/decisioni.md` e `docs/fatti_da_verificare.md`; aggiorna `CLAUDE.md` (§6).
- **OPS:** `saldo -b` è già fatto (18/09, §5). Resta `saldo --dcgp -b`: se l'account ha budget
  anche su DCGP lo si segna, perché l'ingest del passo 2 è lavoro da CPU.
- **TU:** D0 e D1.

**D0 — Calendario e continuità.** Fatti, da `saldo -b` del 18/09: 111 giorni; residuo 794.303
ore locali ≈ 99.300 GPU-ora, di cui ~70.000–90.000 per il FM; il programma di §10 ne usa
~17.000–21.000 senza il vertice e ~29.000–37.000 con. **Più di metà del disponibile scade
comunque.** Precedente: `IscrB_FM-EEG24` è scaduto il 20/02/2026 con il 52% non speso.
Opzioni, non esclusive: (a) chiedere a CINECA se è possibile una proroga
(`superc@cineca.it`; se sia concessa per un ISCRA B è **da verificare**); (b) preparare un
nuovo ISCRA B per il prossimo bando — di norma ce n'è uno a fine anno, con qualche mese di
valutazione — usando come risultati preliminari il preprint del passo 5 e il pilot; (c)
accettare la scadenza e i tagli di §10. *Proposta:* l'email per (a) questa settimana,
pianificare (b), lavorare come se valesse (c).

**D1 — Autonomia e budget.** Confermare o cambiare i budget della tabella qui sopra e le
regole di §6. *Proposta:* adottarli così.

**Chiuso quando:** v10, piano e registro sono committati; l'output di `saldo -b` del 18/09 è
incollato sotto D0 in `docs/decisioni.md`.

### Passo 0 — Benchmark sintetico del dataloader · W1–W2 · ≤ 40 GPU-ora

Riferimento: **`docs/dataloader_bench_spec.md` (rev. 2)**, che prevale su questo riassunto;
v10 §4.6 e §9. Il contenuto dei dati è casuale, ma si scrive su disco e si rilegge.

- **Due assi, non tre bracci.** *Montaggi*: al volo o precompute — lato CPU e I/O, misurato col
  solo dataloader contro un consumatore simulato. *Layout*: padding + maschera, packing, o
  bucketing per micro-batch — lato GPU, misurato con un modello proxy.
- **AG** scrive il generatore sintetico (montaggi interi, frequenza nativa, classe di origine
  e classe presentata), lo scrittore degli shard, le collate, il consumatore simulato, il
  proxy e `bench/bench_dataloader.py`; i test di logica vanno in `tests/cpu/` e girano sul Mac.
- **TU** apre le allocazioni (§5): una per scrivere gli shard, una successiva per misurare,
  così il primo passaggio è a cache fredda. **AG** lancia con `srun --jobid --overlap`.
- **Requisiti di realismo** (spec §3–§5): il campione è il **montaggio intero**; la collate
  **raggruppa per frequenza** e non padda mai il tempo; nel tempo misurato entra tutta la
  catena, incluse le augmentation di v10 §8 (il time warping è probabilmente lo stadio più
  caro); I/O vero su `$WORK` e su `$FAST`.
- **FlashAttention.** L'import è verificato; **precondizione del braccio packing** è una prova
  forward + backward in bf16 su A100. Il kernel non accetta un bias additivo arbitrario:
  l'encoder locale si scrive come attenzione sui vicini con `gather`, e il kernel varlen serve
  solo alla cross-attention del Perceiver (v10 §4.6, §5.4).

**D2 — Parametri provvisori, non vincolanti** (le scelte vere sono D9 e D10). *Proposta*, spec
§10: base con quote A/B/C 40 / 35 / 25, `p_piena` 0,5, classe C pessimistica (solo Hyser),
contesto 4 s, patch 25 ms, 64 campioni per rank, 8 worker; **un fattore alla volta**, non un
fattoriale. Il parametro a cui il costo è più sensibile è `p_piena`, non le quote.

**D6 — Due regole, da congelare prima di lanciare** (spec §9). Il ritmo richiesto per GPU è

  `finestre/s ≈ MFU · 312·10¹² / (6 · N · K · contesto/patch)`

cioè ~70 finestre/s a 30M con K = 64, contesto 4 s, patch 25 ms e MFU 40%, e ~20 a 100M:
**il caso vincolante è il modello piccolo, non il 5B**.
- **D6a — montaggi.** *Proposta:* al volo resta il default se, col consumatore del 30M, la
  frazione di tempo in attesa (massimo sui 4 rank) è ≤ 2% e il ritmo a vuoto è ≥ 1,5× il
  richiesto, a cache calda e fredda su `$WORK`. Se fallisce, prima si spostano sulla GPU gli
  stadi più cari e si rimisura; **solo se fallisce ancora, precompute** — e allora i montaggi
  virtuali diventano dato fisso e cambia il conteggio di `D_c` (v10 §4.6): va nel manifest.
- **D6b — layout.** *Proposta:* il padding è escluso in partenza se la frazione di token di
  padding supera il 50% (la stima analitica con la base è ~80%). Fra i rimanenti vince chi dà
  più token utili/s nel proxy; a parità entro il 10% vince il più semplice. **Il bucketing per
  micro-batch si adotta solo col tuo assenso:** modifica il «mai bucketing» di v10 §2.7.

**Chiuso quando:** `results/step0/<braccio>/<configurazione>.json` contiene le metriche di
spec §8 — fra cui frazione di tempo in attesa, tempo per stadio, frazione di padding, token
utili/s — e `docs/report_step0.md` applica D6a e D6b; TU firma l'esito.
**Stop:** se al volo non regge nemmeno con gli stadi cari sulla GPU, non si ottimizza a
oltranza: si passa al precompute e si porta a TU la conseguenza su `D_c`.

### Passo 1 — Accessi e download · da oggi · solo `lrd_all_serial`

- **TU, oggi** — sono legati alla tua identità e non li fa l'agente: account NinaPro con
  accettazione dei termini; email per CSL-hdemg con affiliazione.
- **OPS:** download nell'ordine di v10 §2.5, come job `lrd_all_serial` concatenati e
  riprendibili (4 ore di walltime). Prima emg2qwerty, poi i piccoli e il checkpoint NeuroRVQ
  (copia anche in `$WORK/models/`, §3), poi emg2pose e Hyser in background. Tutto sotto
  `$WORK/data/raw/<dataset>/`, nulla in `$HOME`.
- **LIC:** le sei licenze non verificate (v10 §2.2), più accesso, licenza e dimensioni del
  dataset di Kaifosh e licenza del checkpoint NeuroRVQ.
- Nota: `$WORK` non ha backup e il repo di emg2qwerty è archiviato. Se il bucket sparisse,
  l'unica copia sarebbe quella: valutare una seconda copia fuori da Leonardo.

**D3a — Scaricare il dataset di Kaifosh?** Serve comunque all'harness: è la prima colonna
della tabella EMG di NeuroRVQ. *Proposta:* sì, se licenza e dimensioni lo consentono; il
ruolo nel pretraining si decide al passo 4 (D3b).

**Chiuso quando:** per ogni dataset ci sono numero di file, dimensione e checksum in
`data/_manifests/`, e la riga di licenza in `docs/dataset_access.md` viene dalla pagina
ufficiale.

### Passo 1-bis — Checkpoint NeuroRVQ e fatti da verificare · W1–W3 · ≤ 30 GPU-ora

Riferimento: v10 §6.3 e §12.

**D5a — Soglie, da congelare prima di lanciare.** *Proposta:* X = 2 (V2: errore di
ricostruzione mediano fuori dai dataset Meta oltre il doppio di quello su emg2pose); Y = 10
punti (V3: accuratezza dataset-ID dai codici oltre quella dalle potenze di banda); V4: un
livello RVQ è stabile se almeno il 75% dei codici non cambia sotto rumore al noise floor, su
**tutti** i dataset. Sono punti di partenza: contano perché scritti prima, non perché giusti.

Ordine dei lavori:
1. **AG, sul Mac:** clona il repo NeuroRVQ e legge `preprocessing/` e il codice del
   tokenizer. Chiude il fatto n. 4 — banda del pretraining (20–90 Hz?), normalizzazione, se il
   transformer mescola i canali — e quindi il **bivio** di v10 §6.3, cioè quale delle due
   soluzioni per la frequenza vale.
2. **V1** su emg2pose-mini: tokenizzazione un canale alla volta contro multi-canale.
3. **V2–V4** sui dataset aperti già scaricati (CapgMyo, GRABMyo, putEMG, un sottoinsieme di
   Hyser). Il NinaPro Delsys si aggiunge quando arriva l'accesso: è il caso che conta di più,
   ma non blocca l'avvio.
4. **FACT / LIC:** gli altri fatti di v10 §12 in `docs/fatti_da_verificare.md`, con URL e
   citazione esatta.

**D4 — Firma dei fatti.** TU controlla ogni riga sulla fonte e la porta a `verificato da SB`;
AG aggiorna il riferimento a **v10.1**, togliendo o correggendo i «da verificare». È la regola
di §7: raccolto non vuol dire verificato.

**D5b — Esito dell'ancora RVQ:** adottabile, ristretta ad alcuni dataset, o scartata. Con le
soglie congelate è meccanico; TU firma.

**Chiuso quando:** `results/step1bis/*.json` riporta errore di ricostruzione per dataset,
accuratezza dataset-ID da codici e da potenze di banda, stabilità per livello RVQ; D4 e D5b
sono nel registro.
**Stop:** se il 25/10 le verifiche non sono complete, l'ancora RVQ esce dal manifest (§10).

### Passo 2 — Schema, tassonomia, ingest, QC · W2–W5 · CPU

Riferimento: v10 §4.3–§4.5.

- **AG** scrive `src/wearusfm/metadata/schema.py` (con export JSON Schema) con tutti i campi di
  v10 §4.5, e **si ferma**.
- **D7a — Firma dello schema, prima di qualunque ingest.** È la decisione più costosa da
  cambiare dopo. Lista di controllo per la revisione: gruppi di canali con topologia e
  simmetria **del gruppo**; identità anatomica gerarchica con `precisione`, pesi soft e flag
  `nominale`; orientamento della fascia o «ignoto»; frequenza nativa, banda effettiva,
  frequenza di rete; flag raw/inviluppo; chiralità e polarità; calibrazione; validità per
  canale; **identità del soggetto fra dataset** (sovrapposizioni NinaPro); sessione e giorno
  (GRABMyo).
- **D7b — Revisione della tassonomia.** AG prepara l'albero e la funzione atlante da v10 §4.5;
  la revisione umana è obbligatoria e serve competenza anatomica: decidere **chi** la fa.
- **Ingest** un dataset alla volta, dal più piccolo: int16 memory-mapped sotto `$WORK`, QC
  offline, un report per dataset (ore, soggetti, canali, percentuale di canali scartati dal
  QC). Qui si chiudono anche i fatti n. 1–3 (frequenze native, sovrapposizione di soggetti,
  conteggio di soggetti e ore).
- **Dove gira:** è lavoro da CPU. Se c'è budget su DCGP si usa quello; altrimenti nodi Booster
  interi via `sbatch` (32 ore locali l'ora). Con 40 TB su `$WORK` grezzo e preprocessato si
  tengono entrambi: la questione aperta di v10 §2.5 si chiude così, salvo che OPS trovi
  `$WORK` già occupato da altri.

**Chiuso quando:** ogni dataset ha il suo report e i metadati passano la validazione contro
lo schema firmato.
**Stop:** un dataset che non entra nello schema si porta a TU; lo schema non si piega in
silenzio.

### Passo 3 — Preprocessing, front-end, gate di consistenza · W3–W4 · ≤ 50 GPU-ora

Riferimento: v10 §4.2 e §7.1. **È il primo sì/no scientifico del progetto.**

**D8a — Soglie, da congelare prima di lanciare.** *Proposta:* (i) errore relativo RMS delle
feature ≤ 5% nella banda condivisa; (ii) accuratezza del probe A-contro-B ≤ 55%, con
intervallo di confidenza al 95% che contiene il 50%.

- Il gate richiede **un solo dataset a 2 kHz**: si può lanciare appena esiste il front-end,
  su emg2qwerty, senza aspettare l'ingest completo.
- Da controllare per primi se fallisce: il fattore Δt e l'anti-aliasing **per ciascuna
  famiglia di base**.
- Poi: probe dataset-ID come diagnostica **relativa** (contro un banco di filtri fisso), e
  conferma del throughput del dataloader sui dati veri.

**D8b — Se il gate fallisce.** Una settimana per correggere; se il 18/10 non è superato, i
front-end separati per frequenza — che la v10 tiene come controllo in ablation — diventano il
default, e lo si scrive nel registro.

**Chiuso quando:** `results/step3/gate.json` rispetta le soglie congelate.

### Passo 4 — Congelamento del manifest · entro il 25/10

Riferimento: v10 §2.8 e §10.3. **Irreversibile lungo tutta la ladder: firma TU.**

**D9 — cinque scelte:**
- **(a) Quote per topologia.** Il punto delicato è la **ripetizione**: i dati sparsi veri sono
  circa il 5% delle ore. Una quota garantita del 40% li ripete ~8 volte più del resto, cioè
  ~30 passaggi con E = 4 — oltre il punto in cui, sul testo, la ripetizione smette di rendere
  (v10 §10.3). Va deciso se la quota si conta sulla **topologia di origine** o su quella
  **presentata al modello**: i montaggi bipolari virtuali ricavati dalle griglie HD sono
  campioni «tipo sparso» e possono riempire parte della quota senza ripetere i dati sparsi
  veri. *Proposta:* quota sulla topologia presentata, con un tetto esplicito alla ripetizione
  dei dati sparsi veri, e lettura del pilot **per topologia**.
- **(b) Ruolo del dataset di Kaifosh (D3b):** tutto benchmark mai visto, o una parte dei
  soggetti in pretraining. Aggiunge soggetti, non topologie.
- **(a-bis) Classe di quota e pesi dentro la classe.** La classe è **del montaggio** (v10
  §2.8): NinaPro è caso sparso. Dentro la classe va scelto come pesare i dataset — per ore, per
  soggetti, o uniforme — perché decide quanto spesso il modello vede i dataset piccoli.
- **(c) Esito di D6a:** al volo o precompute, con la conseguenza su `D_c`.
- **(d) Split dei soggetti**, congelati col manifest e con seed dichiarato: soggetti di test
  per dataset (split ufficiali dove esistono), più i manifest **sottocampionati per soggetti**
  al 25 e 50% per l'asse di v10 §10.6. Controllare qui la sovrapposizione coi soggetti visti
  dal tokenizer NeuroRVQ.
- **(e) Target RVQ** dentro o fuori, secondo D5b.

**Chiuso quando:** tag git `manifest-v1`, hash del file di manifest nel registro, `D_t`, `D_c`
e consumo realizzato per epoca ricalcolati.

### Passo 5 — Harness di valutazione · in parallelo da W1 · ≤ 300 GPU-ora

Riferimento: v10 §8 e §9 (nota sulla separabilità). Non dipende dal corpus né dai passi 2–4.

**DP — Preprint autonomo: sì o no, e chi lo porta.** Con il calendario di §10 è l'uscita
garantita, ed è il materiale preliminare per la continuità di D0. *Proposta:* sì, con una
persona dedicata che non stia sul cammino critico del FM.

- Split per soggetto, niente classe «rest» dalle pause, normalizzazione stimata solo sul
  train; **due regimi** per tutti i modelli (encoder congelato + probe, e fine-tuning
  completo).
- Primo cliente NeuroRVQ-EMG, poi TinyMyo e PhysioWave; baseline: LDA su feature di Hudgins,
  CNN/TCN da zero, encoder random.
- Dataset: EPN-612 e UCI-EMG subito; DB5 quando arriva NinaPro; Discrete Gestures secondo D3a.

**Chiuso quando:** prima si **riproducono i numeri pubblicati di NeuroRVQ sulla loro
pipeline**, entro una tolleranza dichiarata — è il debug di dati e harness; solo dopo contano
i numeri sui tuoi split.

### Passo 6 — Topologia del modello · codice entro l'01/11 · ≤ 100 GPU-ora

Riferimento: v10 §5, §5.6, §6.4.

**D10 — Patch, contesto, masking.** *Proposta di partenza:* patch 25 ms (coerente con i
3,6·10⁸ time-patch per epoca della v10), contesto 4 s, scale di masking di v10 §6.4; se
l'ancora RVQ sopravvive, una quota di slab ≥ 200 ms su tutti i canali, allineati alla griglia
del tokenizer.

**D11 — §5.6 della v10.** *Default di lavoro:* target dal decoder a query del teacher, con la
diagnostica del collasso da query attiva dal primo giorno; target dall'encoder locale come
braccio di controllo a 30M. Va confermato **prima** del passo 7, che ci poggia sopra.

**Chiuso quando:** un batch con le tre topologie insieme fa forward e backward; i test di
equivarianza passano (rotazione ciclica sugli anelli); la diagnostica del collasso da query è
collegata; i FLOP sono contati per modulo; il **sanity JEPA su un solo dataset omogeneo**
(v10 §10.1) non collassa.

---

## 12. Dopo il passo 6 — decisioni già in calendario

Quattro su sei vanno chiuse **prima della finestra 1**, cioè entro il 1º novembre: i job
partono in blocco e le ereditano.

| ID | Decisione | Entro | Nota |
|---|---|---|---|
| D12 | Soglia della regola decisionale del pilot (v10 §10.3) | finestra 1 | da congelare prima dei risultati |
| D14 | Schedule WSD e policy di continuazione (v10 §10.4) | finestra 1 | due checkpoint per rung, pre- e post-decay; i rami di decay fanno del pilot una sola run per seed |
| D15 | Probabilità del dropout del livello muscolo (v10 §5.2) | finestra 1 | *proposta:* 0,4 |
| D16 | Asse dei soggetti: frazioni, taglie, stratificazione (v10 §10.6) | finestra 1 | *proposta:* 12,5 / 25 / 50 / 100% a 100M e a 300M; i manifest nascono al passo 4 |
| D17 | Training finale separato, oppure modello finale = rung scelto, eventualmente proseguito a 2D | W9–W10 | *proposta:* coincide col rung, salvo che la ladder indichi una configurazione non ancora addestrata. Toglie una fase intera da dicembre |
| D13 | **Vertice speculativo a 3–5B** (v10 §10.4, rev. 2) | 06/12 | parte in parallelo alla ladder se: il 1B gira in FSDP, ha superato il primo 10–20% senza instabilità, si è entro il 06/12. Mai sul cammino critico; se diverge si spegne |

---

## 13. Cosa fare oggi

**TU:**
1. Account NinaPro, con accettazione dei termini — **è sul cammino critico**.
2. Email per CSL-hdemg, con affiliazione.
3. Email a `superc@cineca.it` sulla possibilità di proroga (D0).
4. Chiedere a chi condivide l'account **la data in cui smette**: decide se il disponibile è
   70.000 o 90.000 GPU-ora (§5).
5. D1: confermare budget per passo e regole di §6. DP: chi porta l'harness.
6. Aprire un'allocazione da 2 ore quando AG ha pronto il benchmark del passo 0.

**AG:** passo −1; poi, in parallelo, passo 0 (codice e test su CPU), lettura del repo NeuroRVQ
sul Mac (primo punto del passo 1-bis), scheletro dell'harness (passo 5).

**OPS:** `saldo -b`; download di emg2qwerty e dei dataset piccoli su `lrd_all_serial`.

**LIC / FACT:** licenze non verificate, dataset di Kaifosh, e i fatti di v10 §12.
