# Review Claude — completamento 300M/5.000, rilascio, cinque estrazioni e seguito CPU (GPT c74d207, base già rivista 0d5f0f9) — 5 novembre 2026

**VERDETTO: il lavoro già eseguito è AFFIDABILE. Nessun problema bloccante.** Nessun job da rifare, nessun intervento sui
job attivi, nessuna obiezione all'avvio del seguito CPU nell'ordine previsto.

| Area del prompt | Esito |
|---|---|
| 1. Delta operativo dopo `0d5f0f9` | APPROVATO |
| 2. S3 `63527446` e rilascio a 5.000 | CONFERMATO con ricalcolo indipendente |
| 3. Completamento scientific5000 `63532932` | CONFERMATO con ricalcolo indipendente |
| 4. Piano delle estrazioni e cinque invii | CONFERMATO; esiti: tre estrazioni concluse e verificate, due in corso alla consegna (§4.3) |
| 5. Seguito CPU | launcher APPROVATO; piani e budget misurati non esistono ancora, ed è corretto così |

Otto osservazioni non bloccanti (O-1…O-8) e quattro note (N-1…N-4), nessuna delle quali tocca la validità di un risultato.

**Aggiunta del 05/11, 14:50:** anche le ultime due estrazioni (300M P2 e 300M posa) sono concluse e verificate; dettaglio in §9.

**Natura di questa review.** È una verifica a posteriori dell'affidabilità di ciò che è stato eseguito, non un'autorizzazione
retroattiva. S3, il rilascio scientifico e l'invio delle estrazioni sono stati decisi con review interne di GPT: nessuna di
quelle approvazioni è mia (O-2). La mia copertura precedente arrivava al codice `0d5f0f9` (review `a988a72`) e all'esito
del preflight `63526191` (`a476aad`).

## 0. Perimetro, copertura e metodo

- **Richiesta.** Prompt di GPT `docs/prompt_claude_review_scaling_completion_extractions_20261105.md`, incollato da Simone il
  05/11 («Fai una review indipendente in sola lettura. Non lanciare esperimenti, non cancellare job e non modificare il
  codice»). Riferimenti: ramo `origin/codex/scaling-300m-20261104`, punta `c74d207930aff8834be039247251328ab49665ed`
  (ricontrollata con `ls-remote` prima del verdetto); pacchetto operativo `34b878a`, training eseguito da `dc96559`,
  codice e piano delle estrazioni `aef665b`, comandi `e4ed7ee`.
- **Coperto da questa review:** diff `0d5f0f9..c74d207` di `src/`, `scripts/`, `tests/`; i 103 sorgenti del contratto e i nove
  sorgenti congelati su dieci commit; le prove di S3 (copie nel repo e originali su Leonardo); piano, gate, ricevute, log e
  checkpoint (solo hash) del training a 5.000; piano, comandi, ledger, checkout e ricevute delle estrazioni; il launcher dei
  lettori CPU contro i controller reali.
- **Non coperto:** contenuto dei checkpoint e delle cache (non scaricati: solo hash, dimensioni, ricevute e provenienza);
  esito delle due estrazioni ancora in corso alla consegna; piani CPU con hash effettivi, benchmark e piano delle risorse
  delle sonde, che non esistono ancora; le autorizzazioni di Simone, che GPT riporta in `docs/decisioni.md` e che non ridiscuto.
- **Metodo.** Sola lettura. Checkout detached di `c74d207` nello scratchpad (rimosso alla fine), worktree di GPT non toccato,
  nessun job, nessuna cancellazione, nessuna modifica al codice. Quattro script miei di verifica nello scratchpad (non nel
  repo) per ricalcolare hash, pin, dose, cadenze e comandi. Su Leonardo solo `sacct`, `squeue`, `ls`, `sha256sum`,
  `git rev-parse`/`status` nei checkout di campagna e copia di JSON/JSONL piccoli; letture in
  `~/wearusfm_local/logs/claude_review6_leonardo_{1..6}.txt` e `~/wearusfm_local/logs/review6_remote/`.

## 1. Delta operativo dopo `0d5f0f9` — APPROVATO

- **Dimensione.** `git diff --stat 0d5f0f9 c74d207 -- src scripts tests`: 8 file, +1.165/−1. Fra i file di codice ne cambia
  uno solo già esistente (`scripts/scaling_compare.py`, 9 righe); gli altri sono tre launcher nuovi (`scaling_train.sbatch`
  77 righe, `scaling_evaluation.sbatch` 109, `scaling_readers.sbatch` 269), i loro tre file di test e 43 righe aggiunte a
  `test_scaling_compare.py`. Da `aef665b` a `c74d207` cambiano solo documenti e risultati.
- **103 sorgenti del contratto e nove congelati: coerenti, ricalcolati da me.** Ho ricalcolato gli hash dei 103 sorgenti di
  `training_source_contract.json` (sha `14b044de…`) sui commit `afb644e`, `05d4bf0`, `0d5f0f9`, `34b878a`, `dc96559`, `6c9b145`,
  `b5d4509`, `aef665b`, `e4ed7ee`, `c74d207`: zero differenze su tutti e dieci. I nove sorgenti di
  `scientific_runtime_freeze.json` hanno zero differenze da `34b878a` in poi; rispetto a `0d5f0f9` cambia il solo
  `scaling_compare.py` (correzione F-1). I due insiemi sono disgiunti. Invariati da `0d5f0f9`: `scaling_projection.py`,
  `projection_s5/manifest.json`, `scientific_reading_policy.json` (`da58ec3a…`), `scaling_reading.py`,
  `scaling_production_memory.sbatch`, `lib_code_dir.sh`, `campaign_checkout.sbatch`, `submit_job.py`.
- **Correzione F-1 del comparatore: corretta e minima.** `scripts/scaling_compare.py` l.143–150: la ricevuta delle ispezioni
  della posa si confronta campo per campo escludendo il solo `states` (identità di filesystem: device, inode, date), che
  cambia a ogni copia o `touch`. Restano vincolati contenuto e supporti (`cache_sha256`, `checkpoint_sha256`, utenti, bersagli,
  braccio, split), i pin per contenuto (l.41, «artifact changed between uses») e il controllo finale degli stati (l.267).
  Quattro test di regressione: `touch`, copia ripristinata, byte cambiati («SHA»), supporto cambiato («inspection receipt»).
- **Launcher del training** (`scaling_train.sbatch`): allocazione fissa 4 GPU / 32 CPU / 480G / 05:30:00, controlli su
  ambiente SLURM e allocazione reale, piano relativo al checkout, directory delle ricevute uguale a quella del piano, poi
  `exec python3 scripts/scaling_train.py run` (l.76) con piano, SHA atteso e uscita: nessun argomento di ricetta o di ripresa
  passa dal launcher. **Launcher delle estrazioni** (`scaling_evaluation.sbatch`): cinque celle fisse con il loro walltime,
  controllo di identità, `AllocTRES` e `TimeLimit` reali, protocollo/passo/bracci del piano, una sola GPU visibile, poi
  `exec python3 scripts/scaling_evaluation.py` (l.109). Entrambi delegano tutte le garanzie scientifiche a controller che
  avevo già rivisto e che non sono cambiati.
- **Bootstrap OpenMP/runpy e vincoli di runtime** (`scaling_readers.sbatch` l.163–182, l.261–267): le variabili dei thread
  si impostano prima di ogni import numerico; processo padre e figli precaricano lo stesso modulo (`import scaling_compare`)
  così l'insieme delle librerie numeriche è lo stesso per posa, sonde e comparatore; il controller parte con `runpy` da
  percorso assoluto; stati dei file vincolati e identità del runtime si ricontrollano dopo i figli; `code_dir_setup` gira
  all'inizio e alla fine (l.29, l.269). La scelta va nella direzione giusta: ordine degli import e thread sono già stati
  indicati come possibile causa di differenze all'ultima cifra (T100-LQ, ipotesi non misurata). Limite della prova locale in O-6.
- **Protocollo.** `docs/scaling_300m_protocol_20261104.md` cambia di +10/−3 righe: precisa solo l'ordine di S5 (benchmark
  train-only sulle cache reali dopo le estrazioni e prima dei fit); sonde e criteri invariati.

## 2. S3 `63527446` e rilascio a 5.000 — CONFERMATO

- **Prove di S3 ricalcolate.** Rapporto `production_memory_63527446/report.json` sha `45acd95c…` (9.505 byte), codice
  `0d5f0f9`. Le copie nel repo dei 15 file piccoli sono uguali agli originali su Leonardo, che ho riletto il 05/11 alle 13:09;
  i 12 pin degli originali (8 file del run, 4 file di rank) e il pin del checkpoint tecnico (`3abe7ded…`, 4.673.783.740 byte,
  ricalcolato da me alle 13:31) coincidono. La directory del run S3 contiene solo `checkpoint.pt`: nessun checkpoint
  nominato, nessuno `STOP`.
- **Quattro rank, picco reale, gate.** Picchi allocati 29,611 / 31,242 / 29,710 / 34,544 GiB su 63,42 GiB di device
  (46,7% / 49,3% / 46,8% / 54,5%); gate al 90% = 57,08 GiB, margine 22,5 GiB. Riepilogo e massimo dei rank si riconciliano
  entro 1.536 byte. Il proxy pre-specificato (33,522 GiB) sottostimava il massimo reale del 3,05%: il gate è stato giudicato
  sul reale, come previsto (seguito in O-1). Run: 500 passi contigui, 32 finestre per passo, valori finiti, LR uguale a
  1e-4·s/1962 su tutte le righe, valutazione fissa e pannello ai passi 0 e 500, diagnostiche al passo 500. Consuntivo:
  COMPLETED 0:0, 1.619 s, 1,798889 GPU-ore, 14,391111 core-ore.
- **Il wrapper ha vincolato questi dati prima del training, non solo la presenza dei JSON.** `scripts/scaling_train.py`
  `execute` (l.368–416) non è cambiato da `0d5f0f9`. Ordine: identità del codice → contratto sui byte vivi dei 103 sorgenti →
  ricetta e comando derivati dal piano → allocazione → `check_runtime_gates` (l.287–321) → pin degli input → identità del
  checkpoint di controllo → stati invariati → **solo allora** `launch.json` (l.398) → `torchrun`. Dentro i gate,
  `check_production_memory` (l.251) chiama `validate_report` (`scripts/scaling_production_memory.py` l.291–304), che rilegge
  gli originali di S3 ai percorsi assoluti e ne ricalcola gli hash, checkpoint tecnico compreso, ricostruisce l'evidenza (`inspect_evidence`,
  l.178–288; gate per rank a l.267–269) e pretende che il rapporto dichiari esattamente quei valori. Il `launch.json` reale
  elenca 153 input verificati: i 103 sorgenti; confronto, configurazione 30M, ricevuta di riferimento, contratto e piano di
  esecuzione; i due sorgenti di esecuzione; i tre rapporti dei gate; i 29 sorgenti di valutazione con test, risorse e
  review; i due file della review scientifica; ricevuta e checkpoint di controllo; quattro input. I 148 che stanno nel
  checkout coincidono con i byte del commit `dc96559`. Limite della prova in O-3.
- **Ricetta fresca e controllo 30M.** Il comando eseguito coincide con quello che ho derivato da solo dal piano e differisce
  dall'argv storico del 30M solo per `--preset sanity→300m`, `--out-dir` e `--time-limit-s 19500`; nessun `--resume`,
  `--init-from`, `--branch-from`. La configurazione differisce dal 30M in quattro chiavi del modello (`dim` 384→896,
  `n_heads` 6→14, `local_layers` 2→3, `backbone_layers` 8→18). Il checkpoint di controllo `49dd518b…` (454.381.728 byte) e la
  sua ricevuta `d0a50fe7…` sono vincolati nel piano e verificati dal wrapper prima del lancio.
- **Chi ha approvato.** Il quarto gate (`scientific_review`) e la review della readiness puntano a documenti di GPT; la mia
  review 5 compare solo come `underlying_scientific_review` (sha `345a40a6…`, 23.406 byte, identico al mio file a `a988a72`)
  e copre lettori, S5, comparatore e wrapper S3, non i tre launcher (O-2).

## 3. Completamento scientific5000 `63532932` — CONFERMATO

- **Job e checkout.** COMPLETED 0:0, 05/11 07:16:12–11:22:13, 14.761 s su 19.800 di limite, nodo `lrdn1310`; consuntivo
  **16,401111 GPU-ore e 131,208889 core-ore** (tetti 22 / 176). Checkout `scaling_300m_scientific5000_20261104_v1` a
  `dc96559`, in sola lettura e pulito.
- **Ricevuta.** `receipt.json` `0ac5fe81…`, `config.json` `6120b826…`, `summary.json` `f0975c4d…`, `launch.json` `f1521ce7…`:
  le copie nel repo sono uguali agli originali che ho riletto. La ricevuta lega job, braccio, passo 5.000, codice `dc96559`,
  piano `66aac4e8…`, contratto, sorgenti di esecuzione, hash dei gate uguali a quelli del piano, checkpoint di controllo,
  `scientific_endpoint true`, `automatic_continuation false`. Il comando della ricevuta è uguale a quello di `launch.json` e
  a quello ricostruito.
- **Checkpoint nominato e finale: hash ricalcolato da me.** `checkpoint_step0005000.pt` e `checkpoint.pt`, 4.673.783.740 byte
  ciascuno, hanno entrambi sha `37279995d0d484e241b032eb9b6b23a31c16a5947a0e2cf0c575da0cb70641aa` (`sha256sum` su
  Leonardo, 05/11 13:24). Non li ho scaricati. Nessuno `STOP` nella directory del run.
- **Dose e cadenze, dal log completo (5.000 righe, hash uguale al remoto).** Passi contigui 1…5.000, 32 finestre per passo
  = 160.000 finestre, quattro processi con 8 finestre ciascuno, `accum 1`, nessun valore non finito, nessuna sessione
  saltata; LR uguale a 1e-4·min(s/1962, 1) su tutte le righe; diagnostiche di gradiente e rango ai multipli di 500;
  valutazione fissa a 0, 500, …, 5.000 su 8 batch; pannello posa con 11 righe, hash coerente con l'insieme, 256/256/32.
  Gli insiemi della valutazione fissa e del pannello sono identici byte per byte a quelli di S3 e (valutazione fissa) a quello
  del 30M. Tempo per passo: media 2,796 s contro 3,035 s previsti dalla formula.
- **Runtime.** Ambiente uguale al riferimento 30M (Python 3.11.6, torch 2.2.0a0+git6c8c5ad, CUDA 12.1, moduli). Memoria
  massima 37,752 GiB (59,5% del device): vedi O-1.
- **La loss non è un criterio.** La review interna di GPT lo dichiara
  (`docs/reviews/scaling_scientific5000_completion_review_20261105.md`, l.46 e l.65) e non trovo nessuna
  soglia nuova su loss o pannello nei documenti della consegna. Le diagnostiche descrittive sono in N-1.

## 4. Piano delle estrazioni e cinque invii — CONFERMATO

### 4.1 Piano

- `extraction_plan_5000_20261105.json` ha sha `8b358951182f02dc6af8fafd2621f2dc1ae2eb98939e67c5c914ecc09933528d`, identico
  a `aef665b` e a `c74d207`. `comparison`, `inputs`, `paths`, `pose_references`, `classification_references` sono identici al
  piano storico `tokenizer_evaluation_20261103/extraction_plan_5000.json`.
- **Entrambi i teacher a 5.000.** 300M: checkpoint `37279995…`, ricevuta, configurazione e riepilogo dalle ricevute
  scientifiche. 30M: stesso pin storico `49dd518b…`, con configurazione e riepilogo presi dallo snapshot immutabile in
  `$WORK` (`results/tokenizer_training_20261103/T100-G/5000/`: `d0a50fe7…`, `dd68e900…`, `4fbfa43b…`, riletti da me). La
  scelta è giusta: nella directory del run 30M `config.json` e `summary.json` sono stati riscritti dalla continuazione a 19.618.
- **Controller invariato e già rivisto** (`scripts/scaling_evaluation.py`): solo endpoint nominato, nessuno `STOP`, pin di
  checkpoint/config/ricevuta/riepilogo, caricamento stretto del teacher, guadagno uguale al buffer salvato senza
  ricalibrazione sui dati di valutazione, cache verificate contro i riferimenti B19618, stati invariati alla fine.

### 4.2 Invii

- Tempi dei commit: `aef665b` 12:41:22, `e4ed7ee` 12:42:03 (prima pubblicazione di `evaluation_submission_20261105.json`,
  poi invariato); prima prenotazione nel ledger 12:42:56 (checkout), estrazioni 12:45:08–12:48:11. L'ora del push non è
  verificabile da qui: l'ordine dichiarato è coerente con i tempi dei commit.
- Per tutti e cinque: argv di `sbatch` nella ricevuta uguale a quello pubblicato, `codice_sha` `aef665b`, copia del ledger
  identica, `--time` uguale alla cella dell'inviluppo. Il ledger ha una sola riga `sottomesso` per ID logico: nessun doppio invio.
- Checkout `scaling_300m_evaluation_20261105_v1` a `aef665b`, in sola lettura e pulito (job `63580233`, 53 s, 0,029 core-ore
  sulle 0,333 previste nell'inviluppo CPU-180). Tetti: 2,5 + 2,0 + 2,5 + 0,5 + 0,75 = **8,25 GPU-ore**, 66 core-ore.

### 4.3 Esiti alla consegna

| Job | Cella | Stato | Durata / limite | MaxRSS | Ricevuta |
|---|---|---|---|---|---|
| `63580776` | 30M P1 | COMPLETED 0:0 | 14:27 / 30:00 | 14,6 GiB | `4634ac05…` |
| `63580820` | 30M P2 | COMPLETED 0:0 | 28:09 / 45:00 | 28,2 GiB | `0cf0d507…` |
| `63580603` | 300M P1 | COMPLETED 0:0 | 32:38 / 2:00:00 | 22,4 GiB | `03abef46…` |
| `63580771` | 300M P2 | RUNNING alle 13:41 | 53:30 / 2:30:00 | — | non ancora scritta |
| `63580385` | 300M posa | RUNNING alle 13:41 | 55:37 / 2:30:00 | — | non ancora scritta |

- **Ricevute concluse, lette da me:** `complete`, `checkpoint_unchanged`, `postload_recipe_passed`, sha del piano e del
  checkpoint giusti, codice `aef665b`, guadagno `548.6939187567021` con politica «exact saved buffer; no evaluation-data
  calibration» (lo stesso valore nei due bracci), input grezzi invariati (360 file per P1, 684 per P2), runtime identico.
- **Supporti e popolazioni uguali fra i bracci.** Per i tre dataset P1 la provenienza dei due bracci è identica in tutti i
  campi tranne braccio, cache, checkpoint, configurazione del modello e forma del pool: stessi `ordered_rows_sha256`,
  `fold_roles_sha256`, `input_sha256`, `splits_sha256`, `extraction_recipe`, numero di finestre e di utenti
  (38.544/32, 9.834/8, 33.961/8). Il pool passa da 768 a 1.792 colonne (2×384 → 2×896).
- **Cache 30M della posa riusate:** presenti in `…/tokenizer_evaluation_20261103/5000/T100-G/pose/` (train 257.604.748 byte,
  val 2.484.485.292 byte, del 03/11). A 5.000 passi la campagna tokenizer aveva estratto solo la posa: le cinque cache
  P1/P2 del 30M sono quindi necessarie e nuove, come dichiarato (dodici nuove più due riusate).
- **In corso alla consegna (05/11, 13:41 CET).** `63580771` (300M P2) e `63580385` (300M posa). Della posa è conclusa la
  parte train (12:58:16: 1.907 finestre, 40 utenti, 200 registrazioni, 549.144.173 byte, provenienza coerente con il
  riferimento `c7bc296e…` e con il checkpoint `37279995…`); la val è in corso (O-4). Le due ricevute mancanti le verifico a
  job conclusi e le aggiungo al registro. In ogni caso i lettori CPU vincolano per hash cache, ricevuta e provenienza e ne
  confrontano l'identità (`scaling_probe.py` l.99–142, `scaling_pose_readout.py` l.81–122): una cache senza ricevuta
  completa non può entrare in un piano.

## 5. Seguito CPU — launcher APPROVATO; piani e budget misurati non esistono ancora

- **Stato.** Nel ledger non c'è nessun invio CPU dei lettori; nel repo non c'è nessun piano della posa o delle sonde con hash
  effettivi, nessun benchmark, nessun piano delle risorse: esiste solo l'inviluppo (`evaluation_resource_envelope.json`,
  `observed_benchmarks_300m: false`). Nessun budget misurato è stato inventato.
- **Compatibilità con i controller reali** (i test del launcher intercettano il sottoprocesso, quindi l'ho controllata a
  mano): stadi, argomenti e convenzioni di uscita di `scaling_readers.sbatch` (l.238–259) corrispondono alle CLI di
  `scaling_pose_readout.py`, `scaling_probe.py` (l.553–565) e `scaling_compare.py`; l'ordine delle celle coincide
  (30M naturale 384, 300M naturale 896, 300M matched 384); l'aggregato a 2 CPU non rompe i controlli numerici perché copia
  il contesto degli shard.
- **Validazione e benchmark train-only prima dei fit: imposto dal codice.** Lo stadio ridge (l.184–230) pretende il rapporto
  di validazione delle quattro cache senza metriche scientifiche, il suo accounting riuscito a 2 CPU, i tre benchmark reali
  `train_only_timing` con accounting a 8 CPU, stesso runtime numerico, stesso supporto del fit (40 utenti, 4.813.620
  elementi validi, come nei benchmark storici) e la derivazione esatta
  `ceil60(1,5·(2V + Σ(11S + 13E)) + 120) ≤ 3600`. Il benchmark della posa non calcola nessun punteggio di validazione o di CV
  (`scaling_pose_readout.py` `benchmark_train`, l.201–211). Per le sonde il piano delle risorse deve coprire una e una sola
  volta 3 celle × 5 dataset × 4 fold, con al più 15 shard da 8 CPU / 30 GiB / ≤7.200 s e somma ≤136 core-ore (l.121–141).
- **Formule, RAM e tetti congelati:** stanno nel launcher e nei controller, entrambi vincolati per hash nella readiness.
  Tetti per stadio 1 + 4 + 24 + 8 + 136 + 2 + 4 = 179 core-ore, più checkout 0,333 e riserva 0,667 = 180 (O-8).
- **S5 e S8 invariati:** proiezioni, manifest S5, politica di lettura e `scaling_reading.py` identici a `0d5f0f9`.
- **Fattibilità tecnica e risultato scientifico restano distinti:** nessuna lettura esiste ancora; gli endpoint e le regole
  sono quelli fissati prima dei dati. Le diagnostiche di training non li sostituiscono (N-1).

## 6. Problemi

### 6.1 Bloccanti

Nessuno.

### 6.2 Per Simone: cosa andrebbe rifatto

Niente. S3 (1,80 GPU-ore), il training a 5.000 (16,40 GPU-ore) e le tre estrazioni concluse (1,25 GPU-ore) restano validi;
nessuna delle osservazioni sotto invalida un risultato o richiede di toccare i job in corso.

### 6.3 Non bloccanti

**O-1 — Il picco di memoria della produzione supera quello di S3.**
Prova: `production_memory_63527446/summary.json` `gpu_max_memory_gib` 34,544; `scientific5000_completion_20261105/summary.json`
37,752 (+9,3% su S3, +12,6% sul proxy di 33,522 GiB). Dove: gate in `scripts/scaling_production_memory.py` l.267–269
(allocato ≤ 90% del device; il riservato deve solo essere ≥ allocato). Coinvolti: S3 `63527446`, training `63532932`.
Impatto: nessuno sul 5.000 (59,5% del device, 19,3 GiB sotto il gate). Per una continuazione a 19.618 o run più lunghi i
500 passi di S3 non delimitano il picco di lungo periodo; in S3 il riservato del rank 3 era già 54,49 GiB (85,9% del
device) e per il run a 5.000 il riservato non è registrato. Correzione minima: nessuna sui risultati; per il seguito usare
37,75 GiB come riferimento misurato e registrare allocato e riservato per rank anche nel run scientifico.

**O-2 — I gate di review del rilascio sono stati soddisfatti da review interne di GPT.**
Prova: `scientific5000_plan.json` → gate `scientific_review` → `docs/reviews/scaling_scientific5000_gate_review_20261104.md`;
`evaluation_readiness.json` → `review` → `docs/reviews/scaling_execution_critic_20261104.md`, che scrive «Il launcher training
e i suoi test sono stati implementati da questo agente; la loro verifica qui è una verifica d'implementazione». Dove:
`scripts/scaling_train.py` l.282 controlla solo `review_verdict == "APPROVATO"`. Coinvolti: `63532932` e le cinque
estrazioni. Impatto: nessuno tecnico, perché i controlli che contano (contratto, sorgenti, S3, ricevute) sono meccanici e li
ho rifatti; è un fatto di processo, che GPT ha già registrato in `docs/decisioni.md` il 05/11 («scelta operativa propria
che non sostituisce il ruolo concordato di Claude»). Correzione minima: nessuna retroattiva; nei prossimi gate il file
indichi chi ha fatto la review e, quando è prevista quella dell'altro agente, punti a quella.

**O-3 — `launch.json` non registra la rilettura degli originali di S3.**
Prova: fra i 153 `verified_inputs` la voce `production_memory` è lo stato della copia del rapporto nel checkout; gli stati
dei 13 originali riletti da `validate_report` non entrano nella ricevuta di lancio. Dove: `scripts/scaling_train.py` l.398,
`scripts/scaling_production_memory.py` l.291–304. Coinvolto: `63532932`. Impatto: nessuno sui risultati. La prova è
indiretta ma completa: percorso di codice invariato, job riuscito (un controllo fallito ferma prima di `launch.json`),
originali ancora presenti con hash uguali ai pin. Correzione minima, facoltativa: nella prossima versione del wrapper
scrivere in `launch.json` anche gli stati dei file riletti dai gate.

**O-4 — L'estrazione della posa 300M non mostra l'avanzamento.**
Prova: `val.log` è fermo a 207 byte dalle 12:59:08; anche `train.log` è comparso intero solo a fine estrazione (12:58:16).
Dove: `scripts/scaling_evaluation.py` l.362–364 (uscita del sottoprocesso su file, quindi a blocchi) e
`scripts/slurm/scaling_evaluation.sbatch`, che non imposta `PYTHONUNBUFFERED`; le estrazioni P1/P2 scrivono riga per riga. Coinvolto:
`63580385`. Stima mia: train 653 s per 1.907 finestre; val 18.548 finestre → fine attesa fra le 14:12 e le 14:45, limite
alle 15:15:45, margine 30–63 minuti. Impatto: operativo; un TIMEOUT costerebbe fino a 2,5 GPU-ore e richiederebbe una
decisione registrata per un nuovo invio. Correzione minima: nessuna ora (non si tocca un job in corso); nella prossima
versione del launcher `export PYTHONUNBUFFERED=1`, che si propaga ai sottoprocessi senza toccare i sorgenti scientifici.

**O-5 — Tutti gli stadi CPU sono vincolati a `login08`.**
Dove: `scripts/slurm/scaling_readers.sbatch` l.6 e l.146–147; inviluppo `cpu_node: login08`. Coinvolti: tutti i job CPU del
seguito. Impatto: operativo. Il nodo è condiviso con altri job, quindi i tempi dei benchmark possono non valere sotto contesa (TIMEOUT
fail-closed, nessun rischio numerico); se il nodo non è disponibile (manutenzioni del 10–14/11 e del 23–25/11) i job
restano in coda; cambiare nodo significa rifare i benchmark. Correzione minima: nessuna; tenerne conto nel calendario.

**O-6 — La prima prova reale dell'identità del runtime fra processo padre e figli è lo stadio ridge.**
Prova: `tests/cpu/test_scaling_readers_launcher.py` l.331–374 usa gli import reali ma sul Mac, a un thread, e si ferma a
`--help` («--help stops before any fitting»). Dove: `scaling_readers.sbatch` l.168, l.177–182, l.201, l.215, l.266–267.
Coinvolti: validazione, tre benchmark, ridge, compare. Impatto: se su Leonardo un figlio caricasse una libreria numerica
in più durante il calcolo, il gate rifiuterebbe: nessun effetto scientifico, ma si perderebbero validazione e benchmark
(al più 5 core-ore) e servirebbe un nuovo checkout. Correzione minima: nessuna ora; se il gate rifiuta, fermarsi e
riportare la differenza dei `threadpools`, senza nuovi tentativi.

**O-7 — I file di accounting dei benchmark si producono fuori dai job.**
Dove: `scaling_readers.sbatch` l.191, l.194–198, l.206–211: il launcher controlla coerenza e limiti di `JobIDRaw`, `State`,
`ExitCode`, `AllocCPUS`, `ElapsedRaw`, `MaxRSS_bytes`, non la loro origine; `ElapsedRaw` della validazione entra nella
formula. Impatto: un errore di trascrizione non falsa la scienza (al più un TIMEOUT), ma il «budget misurato» resta
verificabile solo con l'uscita grezza. Correzione minima: pubblicare accanto a ogni JSON l'uscita testuale di `sacct -P`
da cui deriva, come già fatto per S3 (`sacct.txt` e `accounting.json`).

**O-8 — Il tetto cumulativo CPU di 180 core-ore non è imposto dal codice.**
Dove: `scaling_readers.sbatch` l.114–117 (tetti per stadio) e l.137 (136 core-ore per i fit); la riserva dell'inviluppo è
0,667 core-ore. Impatto: ogni ripetizione di uno stadio (benchmark fallito, fit in TIMEOUT) esce dall'inviluppo e richiede
una decisione registrata. Correzione minima: nessuna nel codice; fermarsi al primo fallimento e tenere il consuntivo.

### 6.4 Note

- **N-1 (diagnostiche, non criteri né letture).** Il clipping del gradiente (`grad_clip` 1,0) è attivo sul 100% dei passi in
  entrambi i bracci: norma mediana 4,92 nel 300M e 4,26 nel 30M (ultimi 1.000 passi: 4,01 e 3,38). È una proprietà della
  ricetta comune, non una differenza fra bracci. Valutazione fissa a 5.000 sullo stesso insieme: totale 2,136 (30M) e
  2,167 (300M); posa 0,977 e 0,946, su 256 e 32 finestre. La lettura scientifica resta quella di P1, P2 e posa, ed è di GPT.
- **N-2.** `pip_freeze` è assente nel riepilogo scientifico, come nel riferimento 30M; S3 lo aveva registrato
  (`pip_freeze_s3.txt`, sha `90713fc0…`). L'uguaglianza dell'ambiente è per versioni e moduli.
- **N-3.** I sette invii della catena sono `--agente gpt`. `CLAUDE.md` prescrive ancora le sottomissioni tramite semplicione
  (regola del 28/10); se la regola è superata, va aggiornato con la frase di Simone. Non incide sui risultati.
- **N-4.** Le dimensioni delle cache nuove del 30M sono 88 byte più piccole dei riferimenti B19618 in tutti e cinque i
  dataset: differenza costante, che attribuisco ai metadati testuali nel file (interpretazione mia, non verificata aprendo le
  cache); i supporti sono verificati per contenuto dal controller.

## 7. Test

- Comando registrato da GPT in `execution_verification.json` (16 file), rieseguito su `c74d207` con un mio percorso JUnit:
  **571 passati in 161,76 s**, come dichiarato (571, 154,95 s). Log `~/wearusfm_local/logs/scaling_exec_tests16_c74d207.txt`.
- Intera `tests/cpu` sullo stesso checkout: **2.602 passati, 1 fallito in 26 min 46 s**. Il fallito è quello preesistente
  `test_pose_b19618_launcher.py::test_exact_B_recipe_only_maxsteps_changes_schedule_identical` («continuation recipe
  differs beyond max_steps»), già registrato il 04/11 ed estraneo alla consegna (file non toccati dal diff). Log
  `~/wearusfm_local/logs/scaling_exec_tests_full_c74d207.txt`.

## 8. Testo per GPT

> Review Claude della consegna «completamento 300M/5.000 e rilascio delle estrazioni» sul ramo
> `claude/review-scaling-300m-20261104`, file `docs/reviews/claude_review_scaling_300m_completion_extractions_20261105.md`.
> Verdetto: lavoro eseguito affidabile, nessun problema bloccante, nessun job da rifare; launcher CPU approvato, nessuna
> obiezione ad avviare validazione, benchmark train-only e poi fit nell'ordine previsto.
> Prima dei fit CPU ti chiedo solo: (1) pubblicare piani e pin con gli hash effettivi delle cache concluse prima di ogni
> invio; (2) accanto a ogni JSON di accounting l'uscita testuale di `sacct -P` (O-7); (3) fermarti al primo rifiuto di un
> gate o al primo TIMEOUT e riportarlo, senza nuovi tentativi (O-6, O-8); (4) per un'eventuale continuazione a 19.618 usare
> 37,75 GiB come picco misurato e registrare allocato e riservato per rank (O-1).
> Da registrare senza azione: O-2 (i gate di review del rilascio puntavano a review interne), O-3, O-4, O-5, N-1…N-4.
> Se il job della posa `63580385` andasse in TIMEOUT, segnalalo a Simone prima di qualunque nuovo invio.

## 9. Aggiunta del 05/11/2026, 14:50 — esiti delle ultime due estrazioni

Le due estrazioni in corso alla consegna sono concluse senza errori. Ho letto le loro ricevute con gli stessi controlli delle
altre tre: sono conformi. Le righe corrispondenti di §4.3 e l'osservazione O-4 vanno lette con questa aggiunta; il verdetto
non cambia e l'area 4 è confermata per tutti e cinque gli invii.

| Job | Cella | Stato | Durata / limite | MaxRSS | Ricevuta |
|---|---|---|---|---|---|
| `63580771` | 300M P2 | COMPLETED 0:0 | 1:38:25 / 2:30:00 | 32,3 GiB | `172aa79b…` |
| `63580385` | 300M posa | COMPLETED 0:0 | 1:57:33 / 2:30:00 | 28,9 GiB | `a57c9420…` |

- **Entrambe:** `complete`, `checkpoint_unchanged`, `postload_recipe_passed`, piano `8b358951…`, contratto `14b044de…`, codice
  `aef665b`, checkpoint `37279995…`, guadagno `548.6939187567021` dal buffer salvato, input grezzi invariati.
- **P2.** `epn612.npz` `0c327ceb…` (1.693.610.782 byte) e `uci_emg.npz` `39e121da…` (13.061.126 byte). La provenienza è uguale
  a quella del 30M in tutti i campi tranne braccio, cache, checkpoint, configurazione del modello e forma del pool: stesse
  righe ordinate, stessi fold, 47.100 finestre / 314 utenti e 364 / 19.
- **Posa.** `train_subset.npz` `1e5bdabc…` (549.144.173 byte) e `val.npz` `c1b57991…` (5.321.137.625 byte; 18.548 finestre,
  15 utenti, 1.950 registrazioni, 6.194,5 s). Rispetto alle cache 30M riusate (`145ddfa5…`, `a9e3a8bb…`): stessi split,
  finestre, utenti e registrazioni, stessi hash dei bersagli (`metadata`, `arrays`, `central_mask`) e stessi riferimenti
  B19618 (`c7bc296e…`, `2e1a9d59…`); la configurazione differisce solo per checkpoint e SHA del codice.
- **Insieme completo.** Le dodici cache nuove e le due riusate sono presenti, con supporti uguali fra i bracci. Consuntivo
  delle cinque estrazioni: 17.472 s = **4,853 GPU-ore** su 8,25 e 38,83 core-ore su 66.
- **O-4 chiusa.** Nessun TIMEOUT: la posa è finita alle 14:43:18, 32 minuti prima del limite. Resta il consiglio per il
  prossimo launcher. La riga sul TIMEOUT del testo per GPT (§8) è superata.
- **Punta di GPT dopo la consegna.** Alle 14:38 il ramo di GPT è passato a `c3b2ab6` (due commit: registrazione di questa
  review e raccolta delle ricevute di classificazione). Cambiano solo documenti e risultati, nessun file in `src/`,
  `scripts/`, `tests/`. Non li ho rivisti; constato soltanto che i quattro hash delle ricevute di classificazione e il
  consuntivo che riportano (2,894167 GPU-ore) coincidono con i miei.
