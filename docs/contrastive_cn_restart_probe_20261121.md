# Prova mirata sulla prima riduzione InfoNCE dopo la ripresa

Simone ha scelto «Prova mirata» alle 17:32 del 21/11. Si usa il fork pubblicato a 544. Il confronto precedente a 576 ha rilevato scarti fra 1,034% e 1,316% sui quattro primi momenti AdamW del proiettore e deriva più piccola negli altri momenti attivi. Lo stato verificato nel checkpoint è ripristinato esattamente. L’ipotesi da sottoporre a intervento riguarda i gruppi con cui DDP somma i gradienti del primo backward dopo il riavvio. L’uguaglianza delle norme per obiettivo sui 218 tensori condivisi dell’encoder è un’indicazione aggregata. La nuova prova osserva il backward di produzione su tutti i 270 tensori. [Decisione e regole](decisioni.md) · [Piano](../results/infra/contrastive_targets_20261118/cn_restart_probe_20261121/plan.json)

## Intervento

Due processi torchrun distinti eseguono in sequenza le due traiettorie su quattro GPU dentro la stessa allocazione. Si conserva la ricetta CN, compresa l’aritmetica bf16 math con TF32 spento. La ripresa ordinaria esegue gli aggiornamenti 545–576. La seconda prepara il riduttore DDP con un backward aggiuntivo prima del vero 545. Il riscaldamento deve lasciare riconoscibili lo stato del fork e il primo batch reale per ogni rank. L’inventario comprende pesi, buffer, teacher, stati e gruppi di Adam, RNG CPU/CUDA, stream, coda e suo ordine, contatori, stato pendente, gradienti preesistenti e supporto effettivo delle loss. Il vero update 545 usa il batch che la ripresa ordinaria usa al medesimo passo. Si registrano nodo e GPU e si verifica la coerenza dell’ambiente.

Al backward del 545 si salvano per i quattro rank il gradiente locale prima della riduzione, il risultato della riduzione e la corrispondenza fra tensore, nome, dtype, posizione e gruppo, osservata nella comunicazione effettiva. Il confronto usa impronte e valori. L’uguaglianza delle sole norme non dimostra identità dei vettori. I file devono coprire i 270 parametri una volta per rank. Un osservatore di comunicazione può alterare il percorso nativo di Torch. La riproduzione del controllo ordinario è pertanto condizione di validità. Per ogni parametro si registra anche l’impronta del gradiente dopo il clipping e, subito dopo il passo di AdamW, quella del parametro, dei due momenti e del contatore. Le due traiettorie arrivano fino a 576 per poterle confrontare con i due endpoint storici.

## Lettura fissata prima della misura

| Controllo | Lettura |
|---|---|
| Stato, batch e RNG al confine, copertura, ordinaria e gruppi effettivi | Per l’ordinaria si attende la riproduzione del risultato CN_A/CN_B. Si autenticano i byte dei riferimenti storici, si confrontano i tensori e si distinguono i metadati. Nella riscaldata i gruppi al vero 545 devono avere la composizione del continuo. Un controllo obbligatorio mancante o un riscaldamento che tocca lo stato scientifico lascia la prova inconclusiva. |
| Gradienti locali al passo 545 | Devono coincidere bit per bit, per ciascuno dei 270 parametri su ciascun rank. Uno scarto prima della riduzione su un confine altrimenti valido contraddice la riduzione come unica origine. Questo falsificatore vale solo al primo aggiornamento. |
| Dopo la riduzione al 545 | I rank della stessa traiettoria devono avere gradienti ridotti identici. Fra traiettorie ci si attende almeno un tensore diverso. Il numeratore è la norma L2 della differenza, calcolata in float64. Per 268 tensori il denominatore è `max(L2 ordinaria, L2 riscaldata, 1e-12)`. Per i due bias scalari predefiniti il massimo comprende anche la media dei moduli dei quattro addendi locali, come specificato sotto. La soglia falsificante resta 1e-6 in tutti i casi. Si riportano anche massimi assoluti e conteggi degli elementi diversi. Un eccesso contraddice la spiegazione stretta da arrotondamento e da solo non identifica un difetto di ripristino. |
| Esito a 576 | Per la riscaldata si verifica l’identità con il continuo dei tensori scientifici e dei metadati, incluse coda, Adam, studente, teacher e RNG effettivamente usati. Le tre copie dei generatori CUDA inattivi del rank 0 seguono il confronto con l’ordinaria specificato sotto. Le differenze di sola serializzazione vengono separate. Localizzazione al 545 e recupero completo sostengono che l’intervento basta per questo fork. Al passo 545 si confrontano inoltre gli otto campi di gradiente con la traccia storica. Per la riscaldata la norma attesa del continuo è 4,336711902204087, per l’ordinaria è 4,336711902194546. Se i gruppi e i vettori locali sono quelli attesi ma i campi del continuo al 545 differiscono ancora, la localizzazione osservata fra le due riprese resta parziale e il solo raggruppamento non spiega il continuo. Se coincidono al 545 ma rimane una deriva a 576, il primo aggiornamento misurato non basta a spiegare tutta la traiettoria. |
| Interpretazione | L’intervento può preparare anche cache CUDA/NCCL. La coincidenza dei vettori locali e una differenza misurata nei gruppi effettivi circoscrivono la prima divergenza osservata. Una prova su questo fork non dimostra l’esclusività del meccanismo in altri ambienti né l’utilità della loss su posa o classificazione. |

**Rettifica del 22/11, prima delle due riprese.** Per la sola coppia riscaldata/continuo il lettore descrive a parte `cuda_rng.1`, `cuda_rng.2`, `cuda_rng.3`. Il checkpoint conserva qui copie dei generatori delle altre tre GPU viste dal processo del rank 0. Il trainer inizializza queste copie in modo diverso nel continuo e al riavvio. Claude ha misurato sui checkpoint storici che queste tre voci esauriscono gli scarti nei metadati e nei tensori non in virgola mobile. Per ciascuna si richiedono presenza e struttura corrette e identità bit per bit fra riscaldata e ordinaria. L’ordinaria resta vincolata al file storico intero. Si controllano integralmente `cuda_rng.0`, `torch_rng` e `rank_rng` per ciascun rank. Ogni altra differenza non in virgola mobile a 576, cambiamento di schema o mancata corrispondenza di una copia inattiva con l’ordinaria rende l’esito `INCONCLUSIVE`.

Per `253 anchor_heads.rms.bias` e `257 anchor_heads.env.bias`, entrambi di un elemento, quattro contributi locali di segno opposto possono quasi annullarsi. Dai vettori al 545 si calcola in float64 la media dei valori assoluti dei quattro contributi **ordinari prima della riduzione**. Si conserva sia il rapporto originale sia quello con il denominatore ampliato. Soltanto il secondo determina il superamento della soglia sui due bias. Il controllo di identità bit per bit fra locali delle due riprese mantiene la precedenza.

**Recupero del solo lettore.** Se il rapporto è `INCONCLUSIVE` per un errore dimostrato nella sola lettura o se il processo del confronto si interrompe per un errore operativo, 3-launcher conserva rapporto o ricevuta di errore, log e impronte degli input. Per una rilettura occorrono entrambe le gambe complete a 576 con tutti i file grezzi richiesti, le ricevute dei quattro rank e i relativi SHA verificati. La correzione del lettore passa per la review di Claude e si applica su CPU, con gli stessi criteri e gli stessi byte, scrivendo un nuovo rapporto in un percorso distinto e conservando l’originale. Un problema del confine, delle misure, della fedeltà dell’ordinaria o dei dati grezzi resta inconclusivo. Il recupero non prevede nuove esecuzioni GPU.

Il completamento SLURM documenta l’esecuzione. Per il risultato scientifico si legge il campo `status` del report: `SUPPORTED`, `PARTIAL_SUPPORT`, `HYPOTHESIS_UNSUPPORTED` o `INCONCLUSIVE`. Un errore di esecuzione prima del confronto viene raccolto separatamente.

Il verdetto storico resta `FAILED_REQUIRES_REVIEW`; il nuovo report non abilita training. La lettura CPU facoltativa delle norme dei checkpoint (stima 0,3 core-ore) non è una dipendenza né una cella di questa consegna. L’alternativa continuo più ripresa richiederebbe un’altra decisione tecnica se il test con riscaldamento risultasse non interpretabile.

## Risorse e passaggio operativo

| Voce | Stima storica prima della strumentazione | Tetto richiesto |
|---|---:|---:|
| Due riprese, una allocazione da 4 GPU, 32 CPU e 240 GiB | 0,4733–0,5889 GPU-ore e 3,7867–4,7111 ore locali | 15 min, 1 GPU-ora e 8 ore locali |
| Checkout da 4 CPU e 2 GiB | 19–41 s osservati nei checkout recenti, circa 0,021–0,046 ore locali | 4 min, 0,2667 ore locali |
| Totale incrementale | circa 0,47–0,59 GPU-ore e 3,81–4,76 ore locali | 1 GPU-ora e 8,2667 ore locali |

Base effettiva 4,7400 GPU-ore e 47,3322 ore locali, rispetto a 18 e 169,6. A piena allocazione il cumulato previsto raggiunge 5,7400 e 55,5989. Residuo contabile sui tetti 12,2600 e 114,0011. Il billing osservato conta le CPU allocate. Il tempo nel confronto dei tensori, nelle verifiche e nell’I/O rientra nei 15 minuti della stessa allocazione. I soli vettori nativi locali e ridotti occupano circa 1,83 GB decimali complessivi (due varianti per quattro rank). A questi si sommano i checkpoint, le copie temporanee e i log. L’overhead su NCCL resta da verificare. Un timeout dà un’incertezza operativa da documentare e richiede una nuova valutazione. [Accounting precedente](../results/infra/contrastive_targets_20261118/readout_execution_20261121/execution_summary.json)

**Prima della release**, Claude rilegge le rettifiche successive alla review di `9d34dc69` nel commit `fc3eef6b`. Il vecchio blocco di fbfde797 autentica soltanto la base storica. Il nuovo blocco deve vincolare commit, digest di questo pacchetto e digest del piano. Il comando `--package` stampa i digest. `--bind-review` autentica dal ramo di Claude il testo approvato. Dopo il commit del binding e il push verificato, 3-launcher riceve lo SHA definitivo, esegue i controlli prima degli invii ed è l’unico operatore su Leonardo.

Usare il submitter e il ledger comuni per i due ID pubblicati nel piano. Verificare assenza di prenotazioni o uscite preesistenti, saldo netto degli impegni, billing corrente, quota e scritture concorrenti, spazio per osservazioni e checkpoint, slot debug e reservation. Sul login leggere l’ID di progetto del percorso con `lfs project -d "$WORK"`, poi usare `lfs quota -hp "$LUSTRE_PROJECT_ID" "$WORK"` con l’ID numerico appena verificato. Conservare l’output e controllare che restino almeno 4 GB per questa prova dopo gli impegni concorrenti, oltre allo spazio per il checkout se deve ancora essere creato. Il precedente operativo riserva prudenzialmente 180 s di OverTimeLimit e 180 s di KillWait per allocazione. Con gli stessi valori, l’inviluppo delle due nuove allocazioni è 1,4 GPU-ore e 11,8667 ore locali. Questa maggiorazione serve a pianificare la chiusura e si riconcilia sul consuntivo. [Margine operativo storico](contrastive_targets_execution_20261119.md) La manutenzione annunciata è dal 23 novembre alle 08:00 CET al 25 novembre alle 08:00 CET. Prima del fermo, pianificare a ritroso 4 + 15 minuti di allocazione e 6 + 6 minuti di margine storico. Il limite teorico è il 23/11 alle 07:29 CET per un checkout immediato, prima di sottrarre l’attesa in coda, i controlli fra i job e la raccolta. Riverificare sul centro i margini effettivi. Se la finestra non basta, attendere la riapertura.

3-launcher autentica lo SHA del deploy e i riferimenti storici, completa il checkout congelato in sola lettura con il tetto pubblicato e verifica `COMPLETED 0:0`, SHA e pulizia. Sul login esegue il controllo `--validate-only` con il binding definitivo, e trattiene stdout, stderr e ricevute. Per il job GPU, usare lo script `scripts/slurm/cn_restart_probe.sbatch`, `CODE_DIR=$WORK/wearusfm_campaigns/cn-restart545-20261121`, `EXPECTED_SHA` dello stesso checkout e `TORCH_DIST_INIT_BARRIER=1`. Nessun invio implicito dal validatore. Raccogliere `phase_timings.jsonl` con inizio, fine ed elapsed per ordinaria, riscaldata e confronto, i JSON delle eventuali interruzioni, inventari e impronte per rank. Archiviare anche i due file SLURM `$WORK/wearusfm_runs/logs/cn_restart_probe_<jobid>.out` e `.err` (anche vuoto), con SHA e accounting. Conservare i tensori diagnostici su WORK con impronte e percorsi. Il pacchetto richiede una lettura scientifica del nuovo esito.

### Comandi che 3-launcher completerà dopo la review

La variabile `RELEASE_SHA` indica i 40 caratteri del commit pubblicato che contiene il nuovo binding. Il commit del codice candidato e il commit della review sono riportati separatamente nella consegna di release. Sul Mac, dalla radice del worktree verificato del commit di release, 3-launcher prepara separatamente ogni invio usando il preparatore condiviso. I due comandi Mac qui sotto generano le coppie `cmd` e `workdir`. 3-launcher usa i campi prodotti dopo i controlli di ciascuna fase.

```bash
python3 /Users/simonebenatti/dev/wearusfm/scripts/mac_launcher_command.py "$PWD" "${RELEASE_SHA:?}" -- \
  python3 scripts/submit_job.py \
  --id CONTRAST-CHECKOUT-CN-RESTART545-20261121 \
  --agente gpt --campagna CONTRASTIVE-targets-v3-20261118 --gpu-ore 0 \
  --script-stdin scripts/slurm/campaign_checkout.sbatch -- \
  --cpus-per-task=4 --mem=2G --time=00:04:00 \
  "--export=ALL,CAMPAIGN=cn-restart545-20261121,CAMPAIGN_SHA=${RELEASE_SHA:?},EXPECTED_SHA=${RELEASE_SHA:?}"
```

Dopo il checkout, eseguire in una shell Bash sul **login di Leonardo** con `RELEASE_SHA` impostato al commit del nuovo binding pubblicato. Il blocco termina al primo errore. Conservare stdout e stderr della validazione.

```bash
set -eo pipefail
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
```

Dopo l’esito positivo, dal Mac il submitter registra l’allocazione richiesta nel ledger: 4 GPU per 15 minuti, pari a 1,0 GPU-ora.

```bash
python3 /Users/simonebenatti/dev/wearusfm/scripts/mac_launcher_command.py "$PWD" "${RELEASE_SHA:?}" -- \
  python3 scripts/submit_job.py \
  --id CONTRAST-CN-RESTART545-20261121 \
  --agente gpt --campagna CONTRASTIVE-targets-v3-20261118 --gpu-ore 1.0 \
  --script-stdin scripts/slurm/cn_restart_probe.sbatch -- \
  --cpus-per-task=32 --mem=240G --time=00:15:00 \
  "--export=ALL,CODE_DIR=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/cn-restart545-20261121,EXPECTED_SHA=${RELEASE_SHA:?},TORCH_DIST_INIT_BARRIER=1"
```
