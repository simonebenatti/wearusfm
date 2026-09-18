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
| Budget residuo | ~797.000 ore locali ≈ **~100.000 GPU-ora** per l'intero account |
| Quota del progetto FM | ~50% del residuo ≈ **~50.000 GPU-ora** |
| Scadenza | 07/01/2027 (**7 gennaio 2027**) |
| Quota mensile | 164.383 h locali ≈ ~20.500 GPU-ora, per l'intero account |
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
- Verifica: `saldo -b` (le ultime due colonne sono quota e consumo del mese). Da rifare al
  passo −1 per confermare data di fine, quota e consumo.

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
| **D6** | Dataloader: padding / packing / precompute — regola **prima**, esito dopo | passo 0 | 04/10 | manifest (`D_c`) |
| **D3** | Dataset di Kaifosh: (a) scaricarlo · (b) ruolo nel manifest | (a) passo 1 · (b) passo 4 | (a) 04/10 · (b) 25/10 | harness, manifest |
| **D5** | Ancora RVQ: (a) soglie X, Y e criterio V4 **prima** · (b) esito | passo 1-bis | (a) prima di lanciare · (b) 11/10 | run 5, quota di slab nel masking |
| **D4** | Firma dei fatti verificati → documento v10.1 | passo 1-bis | 11/10 | ingresso dei fatti nel riferimento |
| **DP** | Preprint autonomo di ri-valutazione: sì/no, e chi lo porta | passo 5 | questa settimana | persone, non GPU |
| **D7** | (a) firma dello schema dei metadati · (b) revisione della tassonomia | passo 2 | (a) **prima di ogni ingest** · (b) 18/10 | tutto l'ingest |
| **D8** | Gate di consistenza: (a) soglie **prima** · (b) se fallisce, insistere o ripiegare | passo 3 | (a) prima di lanciare · (b) 18/10 | tutto ciò che segue |
| **D9** | Manifest: quote per topologia, ruolo Kaifosh, split dei soggetti, target RVQ | passo 4 | 25/10 | **irreversibile lungo la ladder** |
| **D10** | Patch (ms), contesto (s), schedule di masking | passo 6 | prima di scrivere il modello | modello |
| **D11** | §5.6 della v10: target JEPA e decoder a query | passo 6 | prima del passo 7 | ablation SSL |
| D12–D16 | pilot, vertice, WSD, dropout del livello muscolo, asse dei soggetti | dopo il 6 | §12 | — |

---

## 10. Calendario a ritroso

**Proposta: D0 la conferma o la cambia.** Dal 18/09/2026 al 07/01/2027 ci sono 16 settimane
scarse e `src/` è vuoto. Il calendario qui sotto è quello **senza margine**: ogni slittamento
fa scattare un taglio già deciso, invece di una discussione a dicembre.

| Settimane | Date | Traguardo (data limite = fine del periodo) |
|---|---|---|
| W1–W2 | 21/09 – 04/10 | passo −1 · passo 0 chiuso con D6 · download avviati · **schema firmato (D7a)** |
| W3 | 05/10 – 11/10 | passo 1-bis chiuso sui dataset aperti · D4 · D5b |
| W4 | 12/10 – 18/10 | **gate di consistenza superato (D8)** · harness che riproduce NeuroRVQ |
| W5 | 19/10 – 25/10 | ingest completo · **manifest congelato (D9)** |
| W6 | 26/10 – 01/11 | modello completo, sanity JEPA su un solo dataset (passo 6) |
| W7–W8 | 02/11 – 15/11 | ablation a 30M e calibrazione (passi 7–8) |
| W9–W10 | 16/11 – 29/11 | pilot sulle epoche e asse dei soggetti (passi 9, 9-bis) |
| W11–W12 | 30/11 – 13/12 | ladder fino a 1B e controlli 2D (passo 10) |
| W13–W14 | 14/12 – 27/12 | shakedown e training finale (passi 11–12) — periodo festivo, code imprevedibili |
| W15–W16 | 28/12 – 07/01 | valutazione su GPU (passo 13) · **07/01: scadenza** |

Consumo atteso: sotto le 1.000 GPU-ora fino a fine ottobre, ~3.000–4.000 a novembre,
~8.000–10.000 a dicembre. Dicembre da solo vale circa metà della quota mensile dell'intero
account: sta in piedi, ma **senza spazio per il vertice a 3–5B**.

### Tagli pre-decisi

| Se entro il… | non è fatto… | allora |
|---|---|---|
| 18/10 | gate di consistenza (passo 3) | D8b: i front-end separati per frequenza diventano il default e si prosegue |
| 25/10 | accesso a un dataset (NinaPro, CSL-hdemg) | il manifest si congela senza. **NinaPro è il caso sparso, cioè il deployment: l'account va chiesto oggi** |
| 25/10 | verifiche RVQ complete (passo 1-bis) | l'ancora RVQ esce dal manifest, la run 5 non si fa |
| 08/11 | modello completo (passo 6) | via le ablation secondarie e quella dei percorsi di identità (resta la concatenazione); ladder 30M / 100M / 300M |
| 29/11 | pilot chiuso (passo 9) | E = 4 per prior, senza il braccio a 8 epoche; l'asse dei soggetti ha la precedenza sui controlli 2D |
| 06/12 | ladder avviata (passo 10) | niente rung a 1B: training finale sul migliore fra 100M e 300M |
| — | — | **il vertice a 3–5B non entra in questo calendario**: richiede la continuità di D0 |

Il passo 5 (harness e preprint autonomo) **non è sul cammino critico** e non dipende dal
corpus: è l'uscita garantita anche se il resto slitta. Per questo parte subito (DP).

---

## 11. Piano passo per passo — fino al passo 6

Budget per passo, da confermare con D1. Totale fino al passo 6: ~500 GPU-ora più l'ingest su
CPU, cioè meno del 2% della quota.

| Passo | Budget | In ore locali |
|---|---|---|
| 0 | 20 GPU-ora | 160 |
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
- **OPS:** `saldo -b`, riportando **solo l'output**. Conferma data di fine, quota mensile,
  consumo del mese. Se l'account ha budget anche su DCGP (`saldo --dcgp -b`) lo si segna:
  l'ingest del passo 2 è lavoro da CPU.
- **TU:** D0 e D1.

**D0 — Calendario e continuità.** Fatti: 111 giorni; quota ~50.000 GPU-ora; il programma
realistico senza vertice ne usa ~10.000–12.000 (v10 §10.5); il resto scade. Opzioni, non
esclusive: (a) chiedere a CINECA se è possibile una proroga (`superc@cineca.it`; se sia
concessa per un ISCRA B è **da verificare**); (b) preparare un nuovo ISCRA B per il prossimo
bando — di norma ce n'è uno a fine anno, con qualche mese di valutazione — usando come
risultati preliminari il preprint del passo 5 e il pilot; (c) accettare la scadenza e i tagli
di §10. *Proposta:* l'email per (a) questa settimana, pianificare (b), lavorare come se
valesse (c).

**D1 — Autonomia e budget.** Confermare o cambiare i budget della tabella qui sopra e le
regole di §6. *Proposta:* adottarli così.

**Chiuso quando:** v10, piano e registro sono committati; l'output di `saldo -b` è incollato
sotto D0 in `docs/decisioni.md`.

### Passo 0 — Benchmark sintetico del dataloader · W1–W2 · ≤ 20 GPU-ora

Riferimento: v10 §4.6 e §9. Non richiede un byte di EMG reale.

- **AG** scrive `src/wearusfm/data/synthetic.py`, le tre collate (padding + maschera,
  packing, precompute) e `bench/bench_dataloader.py`; i test di logica delle collate vanno in
  `tests/cpu/` e girano sul Mac.
- **TU** apre l'allocazione (§5); **AG** lancia con `srun --jobid --overlap`.
- **Due requisiti di realismo**, senza i quali il numero non vale:
  1. i campioni sintetici hanno la **frequenza nativa** del dataset che imitano (200, 1000,
     2000, 2048 Hz): la stessa finestra in secondi ha lunghezze in campioni molto diverse, e
     la collate deve **raggruppare per frequenza** — il front-end gira una volta per gruppo,
     poi i token, definiti in ms, si riuniscono;
  2. nel tempo misurato c'è tutto il lavoro di CPU di v10 §4.6: montage dropout,
     sottocampionamento HD, montaggi bipolari virtuali, masking su tre scale più quello
     spaziale.
- **Kernel varlen di FlashAttention** (fatto da verificare n. 9): controllare se è
  nell'ambiente. Se non c'è, **mezza giornata al massimo** per provarci; poi il braccio packing
  usa maschere a blocchi esplicite con SDPA, o si abbandona.

**D2 — Parametri provvisori, non vincolanti** (le scelte vere sono D9 e D10). *Proposta:*
quote sparso 40 / anello 35 / HD 25; patch 25 ms; contesto ∈ {2, 4, 8} s; 8 worker per GPU.

**D6 — Regola, da congelare prima di lanciare.** *Proposta:* si sceglie, nell'ordine padding
→ packing → precompute, il primo braccio che dà almeno **1,5× il ritmo richiesto dal modello
da 30M** con p95/p50 del tempo per batch ≤ 1,5. Il ritmo richiesto per GPU è circa

  `finestre/s ≈ MFU · 312·10¹² / (6 · N · K · contesto/patch)`

cioè ~70 finestre/s a 30M con K = 64, contesto 4 s, patch 25 ms e MFU 40%, e ~20 a 100M:
**il caso vincolante è il modello piccolo, non il 5B**. Se passa solo il precompute, i
montaggi virtuali diventano dato fisso e cambia il conteggio di `D_c` (v10 §4.6): va scritto
nel manifest.

**Chiuso quando:** `results/step0/*.json` contiene, per ogni braccio e ogni punto della
griglia, finestre/s per GPU e p5/p50/p95 del tempo per batch; `docs/report_step0.md` applica
la regola D6 e TU firma l'esito.
**Stop:** se nessun braccio raggiunge il ritmo, non si ottimizza a oltranza: si porta a TU la
scelta fra spostare il masking su GPU e precomputare.

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
- **(c) Esito di D6:** al volo o precompute, con la conseguenza su `D_c`.
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

| ID | Decisione | Prima di | Nota |
|---|---|---|---|
| D12 | Soglia della regola decisionale del pilot (v10 §10.3) | passo 9 | da congelare prima dei risultati |
| D13 | Criterio per il rung a 3–5B (v10 §10.4) | passo 10 | **fuori calendario senza D0**: dovrebbe partire a novembre per non finire in coda a bassa priorità a fine dicembre |
| D14 | Schedule WSD e policy di continuazione (v10 §10.4) | passo 10 | due checkpoint per rung: pre- e post-decay |
| D15 | Probabilità del dropout del livello muscolo (v10 §5.2) | passo 7 | *proposta:* 0,4 |
| D16 | Frazioni e stratificazione dell'asse dei soggetti (v10 §10.6) | passo 9-bis | i manifest al 25 e 50% nascono già al passo 4 |

---

## 13. Cosa fare oggi

**TU:**
1. Account NinaPro, con accettazione dei termini — **è sul cammino critico**.
2. Email per CSL-hdemg, con affiliazione.
3. Email a `superc@cineca.it` sulla possibilità di proroga (D0).
4. D1: confermare budget per passo e regole di §6. DP: chi porta l'harness.
5. Aprire un'allocazione da 2 ore quando AG ha pronto il benchmark del passo 0.

**AG:** passo −1; poi, in parallelo, passo 0 (codice e test su CPU), lettura del repo NeuroRVQ
sul Mac (primo punto del passo 1-bis), scheletro dell'harness (passo 5).

**OPS:** `saldo -b`; download di emg2qwerty e dei dataset piccoli su `lrd_all_serial`.

**LIC / FACT:** licenze non verificate, dataset di Kaifosh, e i fatti di v10 §12.
