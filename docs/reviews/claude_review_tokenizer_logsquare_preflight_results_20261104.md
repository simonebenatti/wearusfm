# Review di Claude — collaudo T100-LQ eseguito (job 63525730) e seguito preparato (GPT 6d5b353) — 04/11/2026

Richiesta di Simone: «Rivedi il commit 6d5b353 [...] Servono la tua lettura dei risultati e la review del
seguito preparato. Solo review: nessun lancio». Quattro verdetti separati, come chiesto; **nessuna
approvazione implicita dell'intera catena**: prima del training restano gli adattatori di valutazione
implementati e revisionati, il commit di rilascio con i pin e un checkout congelato nuovo.

| Punto | Verdetto |
|---|---|
| 1. Decisione R1: mantenere epsilon = 0,001 | **APPROVATA** come decisione motivata prevista dalla regola, non come assenza di trigger; una precisazione editoriale (R1-a) e una lettura descrittiva da aggiungere al readout del 5.000 (R1-b) |
| 2. Preventivo R2 | **APPROVATO**: rho_update 0,9855573781, stima 16,7052 GPU-ore, tetto 22 GPU-ore e 176 core-ore piu' 0,333 core-ore di checkout, tutto riprodotto da me; nessuna accelerazione dichiarata; rho_infer da misurare |
| 3. Controller `scripts/logsquare_train.py`, launcher e test | **APPROVATO** con quattro note non bloccanti (N1-N4); il piano resta REVIEW_REQUIRED con release nulla e non puo' eseguire |
| 4. Disegno dei valutatori (handoff) | **APPROVATO come disegno**, con tre prerequisiti per l'implementazione (D1-D3); gli adattatori non esistono ancora e vanno rivisti quando consegnati |

## Perimetro e cosa ho eseguito

- Ramo `origin/codex/tokenizer-logcontrast-20261104`, tre commit dopo d48cad5: 43cad1f (S1/S3 nel
  protocollo, piano esecutivo; `git diff --stat d48cad5 43cad1f` tocca solo documenti e lo stato di
  `resource_proposal.json`: codice, launcher, piano e config scientifici identici al d48cad5 approvato),
  4750938 (ricevute), 6d5b353 (lettura, analisi, controller, handoff). Checkout temporaneo detached a
  6d5b353 fuori dai worktree, poi rimosso.
- Letti per intero readout, critico interno, handoff, piano esecutivo, protocollo aggiornato,
  `scripts/logsquare_train.py`, `scripts/logsquare_preflight_analysis.py`, launcher, i test nuovi,
  `training_plan.json`, `resource_proposal.json`, manifest e ricevute.
- **Ricalcolo indipendente dal `report.json.gz`**, con un mio script e senza il codice di GPT: 27 gruppi,
  1.728 righe kernel/gruppo; trigger del pavimento 0 (p massimo 0,29606); trigger della derivata 276 non
  mascherati e 277 mascherati (massimo 0,66035), su 14 kernel distinti [48-52, 54-60, 62, 63], tutti
  della famiglia MLP (indici 48-63 in `frontend.py`: 24 Fourier, 24 spline, 16 MLP); trigger di
  attraversamento e di media del log 0 (massimo 0,16605 std); fase iniziale 10 gruppi, 67.563 token per
  modo, 105/102; fase `trajectory_preupdate` 17 gruppi, 3.884.083 token, 172/174; DB5 a 200 Hz 2.455 e
  52.521 token, p massimo 0,14134 e 0,17858, kernel 58 e poi 50+58; 28 coppie 200/2000 Hz, nessun
  contrasto. Clipping LQ/L 0,987554 / 0,995062 / 1,001502; quota di `W_log` 0,18 / 1,67 / 2,48 %, 2,39 %
  al primo passo. Tutto coincide con il readout e con `analysis.json`.
- Test rieseguiti con la suite integrata: **171 passati in 28,08 s**; `bash -n` del launcher pulito.
- Verifiche di sola lettura su Leonardo: `sacct` dei job 63525639 (27 s, 2 CPU) e 63525730 (148 s, 4 GPU,
  32 CPU, 240G, lrdn0008, COMPLETED 0:0, `--comment` LOGSQ) identico al file archiviato; sha256 del
  `report.json` remoto = 311ac329…5e5b08 come dichiarato; checkout di campagna a 43cad1f e pulito. Ledger
  locale: le due prenotazioni e sottomissioni con `submit_job.py` (righe 990-995).

## 1. Decisione R1 — APPROVATA come decisione motivata

La regola scritta prima del collaudo prevedeva, per un trigger, «motivare il mantenimento di 0,001 oppure
proporre l'emendamento». Il trigger c'e' (553 righe) ed e' riconosciuto senza esenzioni; S1 e' stata
applicata com'era scritta (28 coppie, nessun contrasto, nessun kernel dichiarato inattivo). Il
mantenimento e' giustificato dai dati, non dall'assenza del segnale:

- Il trigger segue l'ampiezza dei kernel dopo il gain condiviso, non il pavimento: nel gruppo piu' grande
  (`trajectory_preupdate/emg2qwerty/2000`, 1.261.067 token) la std scalata mediana e' 5,955 per i
  Fourier, 1,382 per le spline, 0,569 per gli MLP; i 14 kernel MLP con trigger hanno std 0,40-0,69, i due
  senza trigger (53, 61) 1,08 e 0,80; il massimo fuori dalla famiglia MLP e' 0,44. La frazione sotto il
  pavimento dei kernel con trigger e' 0,11-0,19 e la std del log 1,9-2,1, contro 2,4-2,6 delle altre
  famiglie: le loro feature logaritmiche non sono degeneri.
- Aumentare epsilon spingerebbe proprio quei kernel verso il pavimento e ne comprimerebbe l'intervallo
  informativo, senza che il collaudo mostri una patologia: finitezza ovunque, nessun trigger di
  mascheramento, clipping dei due bracci entro ±1,3 %.
- Non si sceglie epsilon sui dati: la decisione conserva il valore pre-registrato e GPT dichiara che
  nessun secondo collaudo e' previsto senza cambiare parametro o codice. Qualunque cambio di epsilon dopo
  risultati di training e' un candidato nuovo, non un emendamento.

**R1-a (editoriale).** Nel readout l'argomento «la derivata resta sotto il suo limite analitico» non
sostiene nulla: e' vero per costruzione. Gli argomenti validi sono quelli sopra; conviene toglierlo.
**R1-b (descrittivo, costo zero).** Il collaudo mostra 64 passi con `W_log` quasi nullo: la conseguenza
del trigger e' un guadagno di retropropagazione verso il banco, attraverso il ramo log, fino a circa
venti volte quello del ramo lineare per unita' di `W_log`, limitato dalla crescita di `W_log` e dal
clipping globale. La prima prova reale e' lo screening a 5.000. Aggiungere al readout del 5.000 e del
19.618, dal checkpoint, le norme delle 64 colonne di `W_log` e il confronto MLP contro le altre famiglie,
come lettura descrittiva e non come criterio. I meccanismi STOP e allarmi del trainer restano quelli
storici (`run.py:1309-1314`).

## 2. Preventivo R2 — APPROVATO

rho_update = 0,70405/0,71437 = 0,9855573781, uguale al campo del report. Durate storiche di T100-L dai
job 63386190 (3.926 s) e 63386337 (11.329 s), presenti in
`results/infra/tokenizer_execution_20261103/19618/sacct_completed.txt`, entrambe con 480G. Stime:
3.869,30 s = 4,2992 GPU-ore e 11.165,38 s = 12,4060 GPU-ore, totale 16,7052; tetti: 5.400 s (6 GPU-ore,
48 core-ore) e 14.400 s (16 GPU-ore, 128 core-ore), totale 22 e 176, con max(1, rho) e arrotondamento a
900 s; checkout 0,333 core-ore; partizione e QoS a 24 h. Consuntivo del collaudo 0,164444 GPU-ore e
1,330556 core-ore, coerente con `sacct`. Il readout dichiara S3 e che rho < 1 non e' un'accelerazione
dimostrata. Limite temporale interno del trainer: tetto meno 300 s, cioe' 5.100 s e 14.100 s contro
stime di 3.869 s e 11.165 s (margini 32 % e 26 %). rho_infer non misurato: nessuna estrazione
preventivabile, come scritto.

## 3. Controller, launcher e test — APPROVATO

Verificato nel codice (`scripts/logsquare_train.py`):

- **Cancello di review** (righe 47-63): il piano deve essere `APPROVED_FOR_TRAINING` con una release
  pinnata che dichiari `training_approved`, epsilon 0,001, gli sha256 di report, analisi e preventivo
  uguali a quelli del piano, gli hash dei due file del controller e la review di Claude; oggi
  `REVIEW_REQUIRED` con release nulla, e `load_plan` fallisce (test).
- **Collaudo come prerequisito** (66-82): protocollo, completezza, codice 43cad1f, epsilon, dose, 4 rank
  tutti `passed`, pairing, sha del piano di collaudo, `W_log` a zero e hash dei pesi condivisi uguale a
  `shared_initial_sha256` = 38cb4cce…, cioe' l'`all_sha256` storico di T100-L; il test usa il report reale.
- **Allocazioni R2** (85-94): tetto ricalcolato dal rho, 1 nodo, 4 GPU, 32 CPU, 480G come il training
  storico, partizione e limite 86.400 s.
- **Parita' della ricetta** (97-112): argv validato storico di T100-L, piu' soli `--tokenizer-logsquare`,
  `--tokenizer-logsquare-eps 0.001`, dose, cartella e `--time-limit-s`; vietati warm start e flag non
  registrati; configurazione effettiva uguale a quella pinnata del candidato; 6 worker, DDP, `torchrun` a
  4 processi (test: argv ricostruito identico all'originale).
- **Inizializzazione condivisa** (115-125): costruzione a seme 0 sotto `fork_rng`, `W_log` (384, 64) a
  zero = 24.576 pesi, hash dei pesi condivisi uguale al collaudo.
- **Ripresa completa 5.000 → 19.618** (150-178, 181-238): ricevuta propria completa con protocollo,
  braccio, codice e piano; STOP assente; identita' del checkpoint nominato e di `checkpoint.pt` con stato
  addestrato uguale; `T.checkpoint_identity` (`tokenizer_train.py:83-103`) controlla passo, config, 4
  stati RNG per rank, AdamW (gruppo unico, ricetta), firma dei dati, contatori di allarme, statistiche di
  posa, finitezza; il controller aggiunge chiavi, forme e dtype completi di student e teacher,
  inventario ordinato dei parametri AdamW e momenti di `W_log` presenti. L'assunzione del gruppo unico
  vale: il trainer crea `AdamW([p for p in student.parameters() if p.requires_grad])` (`run.py:1358`).
  Il trainer riprende da `out_dir/checkpoint.pt` (`run.py:1267-1272`) con `_check_resume_recipe` (805):
  e' la stessa meccanica della continuazione storica. Dopo la fase: `stopped == max_steps`, config,
  checkpoint nominato e finale identici, checkpoint 5.000 immutato, input e codice immutati, ricevuta con
  sha di report e release.
- **Launcher**: stesse risorse del training storico (1 nodo, 4 GPU, 32 CPU, 480G); legge il `TimeLimit`
  reale con `scontrol` e il controller esige che coincida con il walltime del piano, quindi un
  `--time` sbagliato del submitter fallisce subito; checkout congelato verificato da `code_dir_setup`;
  piano dentro il checkout; output fuori dal codice; nessuna sottomissione nascosta.
- **Pin dei sorgenti scientifici**: 102 file con sha256 nel piano, cosi' il commit del training puo'
  differire da 43cad1f mentre la scienza resta identica al byte. Giusto.
- Test: 31 nel file nuovo, comprese dieci corruzioni del checkpoint e i flussi di continuazione.

Note non bloccanti:

- **N1.** Un arresto per `time_limit` nella continuazione lascia `checkpoint.pt` a un passo diverso da
  5.000, che il controller rifiuta di riprendere (controllo del passo): nessun retry, come voluto.
  Scriverlo nel protocollo: in quel caso serve una revisione del piano, non un rilancio.
- **N2.** La release pinna la review di Claude solo come file non vuoto: la decisione di rilascio e' il
  cambio di stato del piano in un commit dedicato, da fare dopo gli adattatori e la loro review.
- **N3.** `run_dir` sta in `$SCRATCH` (purga a 40 giorni): i checkpoint vanno pinnati e, se servono
  oltre, copiati in `$WORK` prima della scadenza; le ricevute con sha256 stanno gia' in `$WORK`.
- **N4.** Il launcher ha `--time=01:30:00` di default: per il 19.618 il submitter deve passare
  `--time=04:00:00`; se lo dimentica il controller si ferma al primo controllo.

## 4. Disegno dei valutatori — APPROVATO come disegno

Le affermazioni dell'handoff sul codice attuale sono vere: `pose_eval.extract` lavora in FP32 sotto
`no_grad` senza autocast (`pose_eval.py:129, 169`), per registrazione con batch fino a dieci finestre;
`extract_channel_readouts` usa autocast bf16 su CUDA (`channel_readout.py:110`) e `readouts` passa da
`forward_layers` con il decoder (75-79); `fm_features.load_model` ricostruisce `FMConfig` dai soli campi
presenti nel checkpoint e carica in modo stretto (`fm_features.py:216-218`), quindi i checkpoint storici
ricevono i due default LQ dalla dataclass. Il disegno giusto e' quello proposto: moduli LOGSQ separati
che riusano le funzioni numeriche verificate, braccio fissato a T100-LQ, nessun monkey patch di `ARMS`,
validazione LQ del checkpoint (flag, epsilon, matrice (384, 64) uguale al checkpoint, `fe_gain` dal
checkpoint e non ricalibrato), canonicalizzazione dei config storici limitata ai due campi, confronto a
quattro celle con i tre contrasti dichiarati, riuso delle OOF storiche con provenienza, benchmark di
inferenza sui percorsi reali. Prerequisiti da chiudere prima dell'implementazione:

- **D1.** Fissare nel piano, prima del benchmark, la regola di aggregazione di rho_infer: per i tetti il
  massimo dei rapporti dei gruppi rappresentativi del job; per la stima puntuale una media pesata con i
  conteggi storici di finestre o token di quel job, presi dai manifest o dalle ricevute delle estrazioni
  storiche. Entrambe scritte prima di misurare.
- **D2.** Il benchmark si prepara al 5.000 del candidato, quindi dopo la prima fase: dichiarare che i
  preventivi delle estrazioni si pubblicano fra le due fasi e che la continuazione non li aspetta.
- **D3.** Ai test elencati aggiungere: con `W_log` non nullo l'estrazione LQ differisce da quella
  lineare sugli stessi ingressi, e con `W_log` nullo coincide; e' il controllo che il ramo e' attivo nel
  percorso d'inferenza, non solo in training.

## Che cosa approvo e che cosa no

- Approvati: la decisione R1 (con R1-a e R1-b), il preventivo R2, il controller con il launcher e i
  test, il disegno dei valutatori con D1-D3.
- Non approvato qui: il rilascio del training come catena. Restano, nell'ordine: adattatori implementati
  e revisionati (D1-D3 compresi), commit di rilascio con stato `APPROVED_FOR_TRAINING`, release pinnata
  e pin dei sorgenti, checkout congelato nuovo, sottomissione da parte di GPT sotto l'autorizzazione
  operativa che Simone gli ha dato; rho_infer misurato prima delle estrazioni. Nessun lancio da parte mia.
