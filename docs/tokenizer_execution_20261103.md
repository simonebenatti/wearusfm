# Esecuzione del collaudo tokenizer — 03/11/2026

Simone autorizza la prosecuzione: «va bene, appena fatto lancia pure, hai gia tutti i miei permessi».
Questo piano rende eseguibile il **checkout congelato e un solo collaudo tecnico train-only**.
Il collaudo non produce un checkpoint scientifico e non assegna risorse al training lungo.
L'esito serve a completare il preventivo, insieme ai valutatori ancora da congelare.

Simone autorizza inoltre: «puoi anche sforare il budget, non mi importa, ti autorizzo».
Il disegno da completare mantiene **tutti e tre i bracci nuovi fino a 19.618 passi**, incluso
T25-L. Il precedente residuo resta un riferimento contabile e non è più un limite alla
campagna. I tetti delle allocazioni successive saranno stabiliti dal timing misurato e dal
perimetro delle valutazioni, senza una seconda richiesta di consenso per questo confronto.

La [risposta alla review Claude](reviews/risposta_claude_tokenizer_20261103.md) aggiorna il
[protocollo scientifico](tokenizer_window_nonlinear_protocol_20261102.md). Le configurazioni
del 03/11 sostituiscono quelle del 02/11 per questa esecuzione: **raggio delle query 0 in tutti i
bracci, JEPA 0, ancore 0,2, posa 1, gain scalare storico**. Nessuna standardizzazione per feature.

## Artefatti e provenienze congelate

- [Piano di input](../results/infra/tokenizer_execution_20261103/preflight_plan.json): protocollo
  `tokenizer-preflight-20261103-v1`, configurazioni, percorsi e SHA256 degli input.
- [Configurazioni complete e CLI ricostruita](../results/infra/tokenizer_review_20261103/comparison_configurations.json):
  `T25-L`, `T100-L`, `T100-G`, con hash individuali. Il piano pinna anche questo indice.
- [Tetti di risorse](../results/infra/tokenizer_execution_20261103/resource_policy.json).
- Driver `scripts/tokenizer_preflight.py`; launcher
  [tokenizer_preflight.sbatch](../scripts/slurm/tokenizer_preflight.sbatch).

Il commit esecutivo deve includere tutti questi file, driver e test. La ricevuta registra lo
SHA effettivo; non si usa come SHA esecutivo quello della sola review. Il checkout remoto nuovo,
detached e non scrivibile, viene creato da `scripts/slurm/campaign_checkout.sbatch` dopo la
pubblicazione di un riferimento `deploy/tokenizer-preflight-20261103`. `lib_code_dir.sh`
verifica commit, assenza di modifiche/file non tracciati e sola lettura prima di importare Python.
Il piano deve risiedere in quel checkout. Output e cache restano fuori dal codice.

| Input | Provenienza fissata |
|---|---|
| Corpus train | `manifest_v1_1_nested_1.json.gz`, SHA256 `6d6d8167b882eb4079fcc566748028dc6de81692859f220a0246291c198574ec`, stesso manifest della ricetta B |
| Scale di sessione | `loader_59623700/session_scales.json`, SHA256 dei byte `76486d882b5deeca66f152c3541534e9f35efca35f944f6f8677b6478b877486` |
| Finestre | `window_seconds_59425316.json`, SHA256 dei byte `e660e324a28a339c038ed07dbaa7d4c358f78f613cd32226eea3f522119a48b0` |
| Normalizzazione posa | `emg2pose_pose/pose_stats.json`, SHA256 `757f2aeebf688063747d50e55d983dd6b3c4cea268ce43d2bae06f3e99524a7e` |
| Radici dei segnali | `$WORK/data/processed` e `$SCRATCH/data/processed`, nell'ordine del piano |

I digest del piano sono SHA256 dei file; alcune ricevute storiche hanno anche digest del JSON
canonicalizzato, per esempio `scales_sha256=5bce8624…`. Sono convenzioni diverse: non confrontare
direttamente quei due digest. Il driver deve verificare la convenzione esplicita del piano.
Questi pin attestano manifest/metadati di calibrazione; non equivalgono a un checksum di ogni
segnale grezzo. Hash degli ingressi effettivamente consegnati e target completano l'audit dei 64 batch.

`keep_calibration` e relativo hash sono null, RVQ e rami spettrali spenti. Il gain frontend
storico non viene ristimato sui dati del collaudo. Nessun fit di calibrazione o normalizzatore
sulla validazione. L'harness P1/P2, i CSV di valutazione e i checkpoint B/R3c non vengono letti:
EPN-612 e UCI sono dati di valutazione, non bracci del pretraining. Il corpus può includere
NinaPro DB5 a 200 Hz; la sua quota nominale circa 2,15% non garantisce l'esposizione di un campione
di 64 batch. Il report deve distinguere quota nominale, frequenze osservate e gruppi assenti.

## Contratto del collaudo

CLI Python concordata:

```sh
torchrun --standalone --nproc_per_node=4 scripts/tokenizer_preflight.py \
  --plan <piano congelato> --expected-code-sha <SHA esecutivo> --out <directory nuova>
```

Un solo loader storico distribuito, quattro rank, sei worker per rank, segmento 1200, seed 0:
**64 batch globali da 32, cioè 2.048 finestre consegnate a ciascun braccio**. Le finestre non sono
necessariamente uniche. I tre studenti e i rispettivi EMA sono residenti insieme, ma elaborano
in sequenza lo stesso batch. AdamW, schedule/LR, accumulo delle loss, clipping e EMA seguono il
trainer originale. Il collaudo è una sequenza tecnica da zero; i suoi aggiornamenti non saranno
ripresi nel training scientifico. `schedule.total_steps=49046` resta la ricetta WSD: **non è il
numero di passi da eseguire**. I limiti effettivi sono 64 batch; gli endpoint proposti 5000/19618
appartengono a fasi successive.

Il report deve attestare configurazioni e input verificati, medesimo stream di finestre/maschere,
conteggi e hash delle query/target comuni, inizializzazione condivisa dei moduli compatibili e
frontend identico di T100-L/G prima dell'attivazione. Registrare supporto effettivo delle ancore e
posa, loss finite, gradienti finiti e aggiornamenti reali dei kernel. Una testa posa inizializzata
a zero può rendere nulli alcuni gradienti al primo passo: il gate riguarda la sequenza e i gruppi
pertinenti, non l'obbligo che ogni parametro cambi a ogni passo.

Le scale pre-attivazione e lo scostamento `GELU(x)-0,5x` vanno riportati per feature/famiglia,
dataset/fs, separando ingresso senza maschera e token visibili dello student mascherato.
Un gruppo non osservato resta non osservato; non si aggiungono finestre di validazione né si
cambia il campionamento per riempirlo. Queste misure descrivono il regime della GELU, senza
costituire un criterio di efficacia o un'autorizzazione a riscalarla dopo i risultati.

Timing per braccio, tempi del loader e costo totale devono restare distinti. Escludere startup
e warm-up dal tempo per passo con una regola dichiarata nel report, conservando comunque il
tempo totale allocato per il consuntivo. Tre modelli residenti e strumentazione aggiuntiva
rendono il picco di memoria diverso da un singolo training: riportarlo esplicitamente.
64 batch non rappresentano la distribuzione completa di forme/dataset e non certificano
la memoria massima o il tempo di un training completo.

Solo un report globale completo, raccolto da tutti i quattro rank dopo 64 batch per braccio,
può dichiarare il collaudo riuscito. Report parziali, timeout, OOM, mancata verifica delle
provenienze o mancato aggiornamento dei kernel richiedono diagnosi. Nessun checkpoint, nessuna
valutazione, nessuna ripresa automatica, nessun riutilizzo di una directory output esistente.

## Risorse, sottomissione e unicità

| Job logico | Risorse massime | Tetto GPUh | Tetto coreh |
|---|---|---:|---:|
| `TOKENIZER-checkout-20261103` | `lrd_all_serial`, 1 nodo, 2 CPU, 4G, 10 min |0|0,333333 CPU-only|
| `TOKENIZER-preflight-20261103` | `boost_usr_prod`, QoS `boost_qos_dbg`, 1 nodo, 4 GPU, 32 CPU, 240G, 30 min |2|16 insieme alle GPU|

Riserva totale **2 GPUh + 0,333333 coreh CPU-only**. Dal saldo locale documentato 63,2925 GPUh e
circa 318,821667 coreh CPU-only restano, sottraendo i tetti, **61,2925 GPUh e circa 318,488334 coreh
CPU-only**. I 16 coreh del job GPU vanno rendicontati, senza sottrarli una seconda volta al saldo
CPU-only. Non è una nuova interrogazione del saldo CINECA. L'attesa in coda non consuma GPUh.
Questi residui descrivono il vecchio budget: la nuova autorizzazione ne consente il superamento.
Il tetto di questo collaudo rimane 2 GPUh; il nuovo preventivo completo sarà registrato
separatamente prima dei job lunghi, senza ridurre il numero dei bracci per il vecchio residuo.

Il canale esistente semplicione è stato verificato non disponibile; la nuova autorizzazione di
Simone consente a GPT di eseguire direttamente tramite submitter e ledger comuni. Nel controllo
pre-lancio del 03/11 l'operatore non ha trovato job tokenizer in coda né prenotazioni nel ledger.
Ripetere il controllo subito prima della sottomissione: questa osservazione non è una garanzia
permanente. Un solo operatore presenta i due ID, senza invio parallelo da altre chat.

Esempio operativo, da completare con lo SHA del commit esecutivo pubblicato. I comandi non sono
stati eseguiti dalla redazione di questo documento:

```sh
TOKENIZER_SHA=<SHA-esecutivo-a-40-caratteri>
TOKENIZER_CODE=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/tokenizer-preflight-20261103
TOKENIZER_OUT=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/tokenizer_preflight_20261103

python3 scripts/submit_job.py --id TOKENIZER-checkout-20261103 --agente gpt \
  --campagna tokenizer-20261103 --gpu-ore 0 \
  --script-stdin scripts/slurm/campaign_checkout.sbatch -- \
  --cpus-per-task=2 --mem=4G --time=00:10:00 \
  --export="ALL,CAMPAIGN=tokenizer-preflight-20261103,CAMPAIGN_SHA=$TOKENIZER_SHA"

python3 scripts/submit_job.py --id TOKENIZER-preflight-20261103 --agente gpt \
  --campagna tokenizer-20261103 --gpu-ore 2 -- \
  --export="ALL,CODE_DIR=$TOKENIZER_CODE,EXPECTED_SHA=$TOKENIZER_SHA" \
  "$TOKENIZER_CODE/scripts/slurm/tokenizer_preflight.sbatch" \
  results/infra/tokenizer_execution_20261103/preflight_plan.json "$TOKENIZER_OUT"
```

La seconda sottomissione legge il launcher dal checkout: attendere che il checkout sia creato,
COMPLETED0:0 e verificato. Non dipendere da un ID già completato che SLURM può aver rimosso:
la ricevuta del checkout, lo SHA e la pulizia verificati sono il gate di questa fase.
Il submitter impone job-name/comment con ID logico,
prenota sotto lock e registra lo Slurm ID. In caso di risposta ambigua usare `--riconcilia`:
non ripetere `sbatch`, non cambiare ID per aggirare la prenotazione. Un esito negativo non
autorizza retry ciechi; conservare log/ricevute, diagnosticare e contabilizzare ogni eventuale
nuova allocazione prima del rilancio. Raccogliere `sacct` dei job parent per il consuntivo,
senza sommare nuovamente gli step `batch`/`extern`.

**Esito operativo03/11:** checkout63369832 completato0:0 in34s, SHA ef8016a e pulizia
confermati. Il primo invio del collaudo è stato rifiutato esplicitamente da SLURM con
`Job dependency problem`: nessun ID GPU e nessuna allocazione. Correzione operativa:
stesso ID logico con `--dopo-rifiuto`, senza la dipendenza dal checkout già verificato.
Codice, output, campioni, risorse e budget invariati; non è un retry di un esperimento fallito.

## Condizioni ancora aperte prima del training scientifico

1. Leggere il report completo del collaudo e chiudere ogni difetto tecnico. Usare timing misurati
   e limiti della misura per stimare separatamente 5000 e 19618 passi; la seconda dose include la prima.
2. Congelare e verificare il valutatore teacher della posa: identità delle finestre, supporto
   comune con 6 patch di margine, target centrale 25 ms, maschere/std/pose_var, ridge e denominatori.
   Testare la parità dei target e i fallimenti di provenienza prima dei risultati scientifici.
3. Congelare le sonde convergenti P1/P2 e l'endpoint EPN separato, con firme del solver,
   righe/etichette/soggetti/fold e bootstrap appaiato del protocollo. Il collaudo non li sostituisce.
4. Chiudere un preventivo totale con tetti per ogni training, estrazione, fit CPU e checkout;
   includere salvataggi/validazioni del trainer e margine motivato. Il riferimento storico
   circa 63,45 GPUh includeva un preflight ipotetico 1 GPUh; qui il tetto è 2 GPUh. Quella stima
   non vale come piano esecutivo: il superamento del vecchio residuo è autorizzato, mentre
   le risorse da allocare devono ancora essere misurate e dichiarate.
5. Versionare piano, dose coperta e policy checkpoint: da zero, checkpoint scientifici ai passi
   fissati 5000/19618, prosecuzione soltanto dal checkpoint proprio e stessa ricetta WSD. Nessuna
   scelta del checkpoint sulla val, nessun training 49046, nessuna sostituzione automatica del
   controllo T25 con B storico per risparmiare: il confronto autorizzato comprende tre bracci
   nuovi, tutti alla stessa dose di 19.618 passi.

L'autorizzazione dell'utente consente di completare questi passaggi senza una seconda richiesta
di permesso per il collaudo già approvato. Il lancio lungo richiede comunque un piano concreto
che superi i vincoli tecnici e dichiari i nuovi tetti di costo, anche oltre il vecchio budget.
