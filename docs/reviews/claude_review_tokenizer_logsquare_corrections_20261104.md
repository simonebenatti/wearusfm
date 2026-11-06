# Review di Claude — correzioni di GPT (d48cad5) alla review 93d89e0 su T100-LQ — 04/11/2026

**Verdetto: APPROVATO. Le correzioni C1-C3 e F1-F2 sono chiuse, le regole R1 e R2 sono scritte
prima del collaudo; il SOLO collaudo preparatorio e' eseguibile com'e' a d48cad5** (2 GPU-ore
massime, 16 core-ore allocate, checkout 0,333 core-ore), con le autorizzazioni di sempre: via di
Simone su quel job, sottomissione di semplicione, checkout congelato da SHA pubblicato, nessun retry.
**Il training scientifico resta NON approvato:** servono i risultati del collaudo letti con R1, il
preventivo da rho_update (R2), orchestratori ed estrattori adattati con pin e una review propria.
Nessun difetto dimostrato. Tre suggerimenti, di cui S1 da scrivere prima di leggere il collaudo.

Richiesta di Simone: «Rivedi le correzioni GPT alla tua review T100-LQ del commit 93d89e0».
Consegna: ramo `origin/codex/tokenizer-logcontrast-20261104`, commit **d48cad5** (unico commit dopo
729093b), diff `git diff 729093b d48cad5`: 14 file, +721/−39. `src/` e launcher invariati, quindi
restano valide le risposte ai punti 1-2 della review precedente e il gate sull'hash iniziale storico.

## Perimetro e cosa ho eseguito

- Checkout temporaneo detached a d48cad5 fuori dai worktree (poi rimosso). Diff letto per intero;
  riletti interi `scripts/logsquare_preflight.py` (345 righe) e il protocollo
  `docs/tokenizer_logsquare_protocol_20261104.md`; letti `logsquare_response_20261104.md`, i tre JSON
  di `results/infra/logsquare_review_20261104/`, i test nuovi e modificati, registro e `decisioni.md`.
- Test rieseguiti col comando del prompt piu' `tests/cpu/test_logsquare_review_corrections.py`:
  **129 passati in 26,60 s** (17 nuovi: 15 nel file nuovo, 2 dalla parametrizzazione di
  `test_tokenizer_preflight.py`); `bash -n` del launcher pulito. Nessun test CUDA/DDP sul Mac.
- Verifiche nel codice condiviso che i test CPU non esercitano: `amp_bf16` e `grad_clip` sono campi
  reali del config (`src/wearusfm/training/run.py:107,113`); `update_arm` usa l'autocast su
  `cfg.amp_bf16` (`scripts/tokenizer_preflight.py:329`) e `clip_grad_norm_(student.parameters(),
  cfg.grad_clip)` con controllo di finitezza (righe 342-343), quindi `grad_norm_before_clip` e' la
  norma totale prima del clipping, identica fra rank dopo la riduzione DDP; `write_json` apre in modo
  `x` (riga 412); il riassunto base pubblica `n_tokens`; il launcher lancia
  `torchrun --standalone --nproc_per_node=4` (`logsquare_preflight.sbatch:81`), che termina il gruppo
  quando un rank esce con errore.
- La copia della mia review nel ramo di GPT e' identica a 93d89e0 (`git diff` vuoto).

## Punto per punto

**C1 — chiuso** (`scripts/logsquare_preflight.py:135-169`, test righe 48-98). Due encode del
controllo con RNG ripristinato prima di ciascuno; prima la finitezza, poi l'uguaglianza esatta del
controllo con se stesso; solo dopo l'encode del candidato. Quattro esiti distinti: `passed`,
`nonfinite_control_forward`, `nondeterministic_control_forward`, `initial_function_mismatch`. Il JSON
per rank (`initial_function_rank{r}.json`) e' scritto **prima** del `require`, quindi sopravvive
all'interruzione; lo scarto massimo e' calcolato in FP64 su `local` e `z`; nessuna tolleranza
introdotta. L'autocast segue ora `cfg.amp_bf16` come in `update_arm` (prima era incondizionato su
CUDA): allineamento corretto. I test usano un encoder finto il cui pescaggio RNG viene riprodotto dal
gate (scarto zero solo se il ripristino funziona) mentre la deriva emulata no. Nota matematica: fra i
bracci il candidato aggiunge `linear(log_features, W_log = 0)`, zeri esatti anche in bf16, quindi
l'uguaglianza esatta e' l'attesa giusta dato un forward deterministico del controllo.

**C2 — chiuso** (righe 172-199, 284, 291; test 113-139). Fattore `min(1, grad_clip/(g + 1e-6))`,
uguale a `torch.nn.utils.clip_grad_norm_` (`clip_coef = max_norm/(total_norm + 1e-6)` limitato a 1),
verificato dall'oracolo torch a rtol 1e-14 per tre soglie; rapporto LQ/L per passo; quota
`(g_log/g_LQ)²` con `None` a norma nulla; frazione di passi clippati; residuo relativo della
decomposizione al primo passo, descrittivo. Liste per rank conservate con l'avvertenza «repliche»:
dato che la norma e' identica fra rank, le quattro liste coincideranno, senza danno. **Accolgo la
precisazione di GPT** sul mio punto 5: i fattori scalano i gradienti prima di AdamW; al primo passo
l'update di Adam e' invariante alla scala del gradiente a meno di eps, e nei passi successivi i
fattori per passo entrano nei momenti con pesi diversi. Il confondimento sui pesi condivisi e' quindi
minore di come l'avevo scritto («aggiornamenti scalati di r») e il report ora contiene cio' che serve
a quantificarlo; resta giusto non toccare il clipping.

**C3 — chiuso** (righe 90-91, 115, 286; test 189-213). Epsilon del piano passato ai collettori di
ogni braccio e all'aggregazione; gate `tokenizer_logsquare_eps == epsilon` fra modello e
diagnostica; test con 0,01 e rifiuto del disaccordo. `load_plan` continua a richiedere epsilon 1e-3
(riga 34): un emendamento resta una modifica di protocollo, come deve.

**F1 — chiuso** (righe 202-207, 253; test 172-186). JSONL per rank, modo `x` al passo 0 e append
dopo, `flush` a ogni riga, scritto **prima** di `verify_pair`, quindi una coppia che fallisce lascia il
proprio record; identita' dei journal nel report (riga 306). `allow_nan=False`: un record non finito
farebbe fallire la scrittura, ma `update_arm` lo ferma prima (riga 343).

**F2 — chiuso** (`scripts/tokenizer_preflight.py:93-96`). Il config storico viene completato solo con
`tokenizer_logsquare=False` e `tokenizer_logsquare_eps=0.001`; il test parametrizzato rifiuta un
legacy privo di `tokenizer_activation` o `keep_readout` anche con pin ricalcolati.

**R1 — scritta prima del collaudo** (protocollo righe 76-107, `preflight_plan.json`
`epsilon_reading_policy`). Statistiche gia' misurate, soglie dichiarate (p ≥ 0,50 non mascherato;
d·√ε ≥ 0,50; attraversamento di 0,50 o |Δ media log| ≥ 1 std fra mascherato e non mascherato),
direzione della revisione per ciascuna, azione (sospensione fino a decisione scritta, nessun cambio
automatico), ambito (per kernel e dataset/fs/fase, braccio LQ, L come contesto), numero di token
pubblicato (`n_tokens` c'e'). Chiara e prospettica. Il punto debole e' la clausola «verificando prima
che non sia un kernel inattivo per ragioni del banco»: a 200 Hz i kernel con banda sopra Nyquist
daranno risposte quasi nulle e p ≈ 1, quindi il primo trigger scattera' quasi certamente per quei
kernel, e l'esenzione verrebbe decisa dopo aver visto i dati. Vedi S1.

**R2 — scritta** (protocollo righe 113-139, `resource_proposal.json`). `rho_update` definito come nel
report (`update_mean_after8_ratio_LQ_over_L`, massimo fra rank per passo, media sui passi 9-64);
stima = durata storica della fase × rho; tetto = walltime storico prenotato × max(1, rho) arrotondato
per eccesso a 900 s; GPU-ore = 4 × walltime / 3600; stop e revisione se oltre partizione o
allocazione; tetti del candidato a `null` fino alla misura; estrazioni con `rho_infer` separato, non
misurato da questo collaudo, 2,75 GPU-ore solo riferimento. Chiara e conservativa. Vedi S3.

## Rilievi

| # | Dove | Rilievo | Proposta | Tipo |
|---|---|---|---|---|
| S1 | protocollo R1, riga 89 | L'esenzione «kernel inattivo per ragioni del banco» non ha un criterio scritto; a 200 Hz scattera' quasi sempre per i kernel sopra Nyquist | Fissare ora il criterio, per esempio: p ≥ 0,50 a 200 Hz con p < 0,50 a 2000 Hz per lo stesso kernel e modo → «inattivo a 200 Hz per il banco», pubblicato, non trigger di ε; ogni altro p ≥ 0,50 resta trigger | prima della lettura del collaudo, meglio prima del checkout |
| S2 | `logsquare_preflight.py:296-309` | I trigger R1 non sono derivati nel report: la lettura resta manuale | Calcolare `epsilon_triggers` nel report dal riassunto (derivazione deterministica), cosi' la sospensione e' meccanica | facoltativo |
| S3 | protocollo R2, righe 120-125 | `rho_update` esclude loader e diagnostica, la durata storica li include: applicare rho all'intera durata sovrastima quando rho > 1 | Una frase che lo dichiari come margine conservativo | facoltativo |

Nessun difetto dimostrato, nessuna voce in «Problemi trovati».

## Che cosa approvo e che cosa no

- **Correzioni chiuse e collaudo preparatorio eseguibile** a d48cad5 senza altre modifiche al codice
  o al piano. S1 riguarda il protocollo e la lettura, non l'esecuzione: va committato prima di leggere
  i risultati, meglio prima di creare il checkout cosi' che lo SHA di campagna lo contenga. Il via sul
  job resta di Simone; la sottomissione di semplicione; nessun lancio da parte mia.
- **Training scientifico non approvato:** collaudo superato e letto con R1 (compreso S1), preventivo
  e tetto pubblicati da rho_update con commit verificato, orchestratori ed estrattori adattati al
  quarto braccio con regressioni contro i tre bracci storici, `rho_infer` misurato prima delle
  estrazioni, pin congelati, review propria.
