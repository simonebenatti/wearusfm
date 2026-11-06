# Review di Claude — T100-LQ: ramo log-quadratico e collaudo appaiato (GPT 729093b) — 04/11/2026

**Verdetto: APPROVATO CON CORREZIONI per il collaudo preparatorio.** Nessun difetto bloccante
dimostrato; tre correzioni piccole prima del lancio del collaudo (una di robustezza CUDA, due di
report) e due requisiti scritti prima del training scientifico. **Il training non e' approvato da
questa review:** lo sara' solo dopo il risultato del collaudo, il preventivo misurato e gli
orchestratori/pin adattati, come il protocollo stesso dichiara.

Richiesta di Simone (prompt GPT `docs/prompt_claude_logsquare_review_20261104.md`): «Simone vuole
soluzione → tua review → convergenza → deploy a cura di GPT». Solo review: nessun job, deploy,
checkout o modifica ai rami di GPT.

## Perimetro e cosa ho eseguito

- Ramo `origin/codex/tokenizer-logcontrast-20261104`, commit di consegna **729093b** (la punta;
  include il prompt), base dell'emendamento **8ee6603**, antenato verificato. Diff letto per intero:
  `patch_tokens.py`, `fm.py`, `run.py`, `train_jepa.py`, `tokenizer_preflight.py`,
  `logsquare_preflight.py`, launcher, i due test nuovi, `preflight_plan.json`,
  `config_T100-LQ.json`, `resource_proposal.json`, `local_validation.json`, protocollo
  `docs/tokenizer_logsquare_protocol_20261104.md`, proposta
  `docs/reviews/tokenizer_logsquare_proposal_20261103.md` con l'emendamento, voci di registro e
  `decisioni.md`. Letto anche `scripts/tokenizer_preflight.py` intero (555 righe), le funzioni
  richiamate di `training/run.py` (`_config_with_defaults`, `_check_resume_recipe`, `_check_branch`,
  `_lr_at`, `_ddp_wrap`, `BatchStream`, `_any_flag`) e `fm.py` (`zero_hidden_samples`, `encode`).
- Test rieseguiti in un checkout temporaneo detached a 729093b con il comando del prompt:
  **112 passati in 25,34 s** (16 in `test_tokenizer_logsquare.py`, 4 in `test_logsquare_preflight.py`);
  `bash -n` del launcher pulito. Nessun test CUDA/DDP: non eseguibili sul Mac.
- Deriva del codice del modello dal SHA di campagna afb644e alla consegna: in `src/` cambiano solo
  `fm.py`, `patch_tokens.py` e `run.py` (il ramo log-quadratico e i default), nessun consumo di RNG
  aggiunto. Quindi il gate che confronta l'hash iniziale dei pesi condivisi con `all_sha256` del
  report storico T100-L (`logsquare_preflight.py:124`, `run()` riga 163) puo' passare: non e' solo
  un auspicio.

## Risposte ai sette punti

**1. Tokenizer e modello** (`patch_tokens.py:29-46, 77-100`, `fm.py:70-80, 168-171`). Il ramo e'
`token = W_lin·x + b + W_log·log(ε + x²)` con `W_log` registrato con `zeros_like(proj.weight)` dopo
`proj` e prima del buffer `fe_gain`: nessun RNG consumato, chiavi e ordine del `state_dict`
invariati quando e' spento (`register_parameter(..., None)`). Le due proiezioni separate
equivalgono alla concatenazione 128→384 con un solo bias. Il log e il quadrato sono in FP32 anche
sotto autocast (`.float()` esplicito; il frontend resta float64, non eleggibile all'autocast), la
proiezione segue l'AMP come quella storica. Caricamento stretto nei due versi testato (legacy senza
chiavi nuove, nuovo con la chiave; `load_state_dict(strict=True)` di un legacy in un LQ fallisce).
`validate_logsquare` rifiuta eps non finito o ≤ 0, tipi sbagliati e GELU+log, prima di consumare
RNG. Test di invarianza ai campioni nascosti con `W_log` non nullo superato a 200/1000/2000 Hz:
l'azzeramento avviene prima del banco (`fm.py:204-208`), quindi il ramo non vede contenuto nascosto
piu' di quanto lo veda il lineare. Corretto.

**2. Trainer e CLI** (`train_jepa.py:68-71, 177-185`, `run.py:190-194`). Default spento;
`--tokenizer-logsquare-eps` senza `--tokenizer-logsquare` rifiutato; GELU+log rifiutato sia dal
CLI sia da `FMConfig`. `_config_with_defaults` aggiunge i due campi ai config storici, quindi
checkpoint e config legacy si ricostruiscono identici; `_check_resume_recipe` e `_check_branch`
rifiutano di riprendere o diramare un checkpoint lineare in una ricetta LQ (test presente). Nessuna
riga di `jepa.py` nel diff: la loss non cambia. Corretto.

**3. Collaudo e launcher** (`logsquare_preflight.py`, `scripts/slurm/logsquare_preflight.sbatch`).
Due aggiornamenti veri per batch con `P.update_arm` (stesso `_accumulate_terms`, clipping, AdamW,
EMA, `_lr_at` della produzione; `student.train()` rimesso a ogni passo, riga 312 del modulo
storico, quindi la `eval()` di `check_initial_function` non contamina gli update); stesso `micro`
per i due bracci con controllo di non mutazione; parita' di supporto query/target/posa e di RNG dopo
ogni passo (`verify_pair`); hash dei pesi condivisi contro il report storico; `W_log` zero;
differenza di parametri esattamente 64·D; funzione iniziale identica su `local` e `z` al passo 0;
finitezza di gradienti, aggiornamenti e parametri; gruppi con gradiente e aggiornamento non nulli
(compreso `logsquare`); copertura 200 Hz e 2000 Hz richiesta; pin degli input e del codice
ricontrollati a fine corsa; output nuovo fuori dai checkout, nessuna ripresa. Launcher identico per
struttura a quello storico: dbg 30 min, 4 GPU, piano dentro il checkout, 4 GPU visibili, scadenza
interna 1500 s. Corretto; vedi C1-C3 sotto.

**4. Momenti, maschere, derivata, memoria, tempi.** Oracolo numpy indipendente per media, std,
frazione sotto il pavimento e RMS della derivata, con fusione fra rank (test). La derivata
`2x/(ε+x²)` e' limitata da `1/√ε ≈ 31,6`, raggiunta a `|x| = √ε ≈ 0,032`: i gradienti verso il banco
attraverso il ramo restano finiti; all'inizio sono nulli perche' `W_log = 0`. Mascherato e non
mascherato sono misurati sullo stesso supporto visibile/QC, come nello storico. La memoria
riportata e' il picco con **due** modelli, teacher e AdamW residenti, dichiarato nel nome del campo;
i tempi sommano i massimi fra rank di forward/backward e clip/AdamW/EMA con barriere fuori dai
timer e diagnostica separata. Buchi CUDA/DDP non provabili sul Mac: (a) l'uguaglianza **esatta**
della funzione iniziale (C1); (b) due DDP nello stesso process group, gia' esercitati dal collaudo
storico con tre bracci; (c) `all_gather_object` di dizionari con hash e bool, come nello storico.

**5. Funzione iniziale identica ≠ primo update identico.** Esatto, ed e' quantificabile senza
nuove prove: al passo 0 i gradienti dei pesi condivisi sono identici nei due bracci, quindi
`‖g_LQ‖² = ‖g_L‖² + ‖g_Wlog‖²` e il rapporto dei fattori di clipping e' `‖g_L‖/‖g_LQ‖` quando il
clipping e' attivo. I record contengono gia' `grad_norm_before_clip` e
`gradient_norms_before_clip["logsquare"]` per passo e braccio: basta derivarne il riassunto (C2).
Non serve un'altra diagnosi prima del training; serve dichiarare la dimensione del confondimento
(«i pesi condivisi di LQ ricevono aggiornamenti scalati di r nei primi passi») insieme al risultato,
senza toccare il clipping, come il protocollo gia' prescrive. Nota: con AdamW e warmup l'effetto
del ramo sulla funzione dopo 64 passi e' trascurabile (`W_log` ≈ Σ lr ≈ 1e-4 per elemento), quindi
il collaudo misura il ramo come gradiente e come diagnostica, non come traiettoria: coerente con
quanto scritto nel protocollo.

**6. Protocollo scientifico.** Primario LQ − T100-L separatamente su E_dyn ed EPN-612; secondari
T25-L e T100-G (emendamento 8ee6603, scritto prima di qualunque risultato di LQ); stessi dati,
seme, stream, dose 5.000 → 19.618 con continuazione completa, teacher congelato, stesse sonde,
split, fold e supporti; IC appaiati 2.000 draw seed 20261102; soglia 2 % posa e margine NI 1 pp su
P1/P2 conservati come riferimenti descrittivi; seme unico, sviluppo riusato, nessuna pretesa
energetica. Il controllo T25-L − B19618 e' dichiarato controllo empirico di ripetibilita', non
pavimento statistico: giusto. Costi distinti e non prenotati: collaudo 2 GPU-h + 16 core-h
allocate, checkout 0,333 core-h, training 22 GPU-h tetto (17 osservate su T100-L), estrazioni 2,75,
lettori CPU 46,33 core-h: coerenti con la convergenza del 03/11. Gate prima del training elencati
(collaudo superato, lettura di ε, orchestratori e pin, preventivo da misura): vedi R1-R2.

**7. Contrastiva.** Giusto tenerla separata e successiva, con base dichiarata adattiva e controllo
che condivide inizializzazione e ogni altra modifica. Scelte da fissare prima di scriverne il
protocollo, in ordine di peso: (i) che cosa e' un positivo, con trasformazioni che conservino
ampiezza e dinamica, perche' le ancore sono bersagli di potenza/forma spettrale e un'invarianza
all'ampiezza lavorerebbe contro di esse; (ii) a quale livello si legge (latente `z` mediato sugli
slot, token locali, teacher o student) e con quale testa di proiezione; (iii) i negativi: con batch
globale 32 i negativi in-batch sono 31 anche raccogliendo i 4 rank, pochi per InfoNCE, quindi o si
accetta un segnale debole o si introduce una coda di memoria, da decidere prima; (iv) falsi
negativi (stesso soggetto o stesso gesto in finestre vicine) e negativi della stessa registrazione;
(v) temperatura e peso fissati una volta, nessuna griglia; (vi) estensione del contratto delle loss
distribuite (numeratori locali, denominatori globali) al termine contrastivo con embedding
raccolti fra rank. Nessuna di queste scelte va presa adesso.

## Rilievi

| # | File:riga | Problema | Impatto | Correzione minima | Tipo |
|---|---|---|---|---|---|
| C1 | `scripts/logsquare_preflight.py:142` | `torch.equal` esatto fra le uscite bf16 di due istanze su CUDA: un kernel non deterministico farebbe fallire il gate con il messaggio «initial encoder function not exactly equal» anche a funzione identica | Falso negativo del collaudo, non provabile sul Mac | Prima del confronto fra bracci, eseguire due volte l'encode del controllo e richiederne l'autouguaglianza; se fallisce, riportare «forward non deterministico» invece di attribuire al ramo; in alternativa riportare il massimo scarto assoluto accanto all'esito | correzione consigliata prima del lancio |
| C2 | `scripts/logsquare_preflight.py:217-227` | Il report non riassume il clipping: `clip_factor = min(1, grad_clip/g)` per passo e braccio, rapporto LQ/L, quota `g_log²/g_LQ²`, frazione di passi con clipping | Il confondimento del punto 5 resta da calcolare a mano dai rank JSON | Aggiungere i quattro riassunti per braccio in `results[name]`; dichiararne la lettura nel protocollo | correzione del report, pre-training |
| C3 | `scripts/logsquare_preflight.py:113, 218` | `LogMoments()` usa l'ε di default della classe, non `plan['epsilon']`; oggi protetto dal gate `== 1e-3` (riga 33) | Un emendamento di ε (previsto dal protocollo) richiederebbe tre modifiche coerenti; il campo `epsilon` del report (riga 229) direbbe il valore del piano mentre le diagnostiche userebbero 1e-3 | `LogMoments(plan['epsilon'])` in entrambi i punti | difetto latente, non bloccante |
| F1 | `scripts/logsquare_preflight.py:176-195` | Nessun record incrementale per passo su disco (lo storico scriveva `rank{r}.jsonl`); se il job cade al passo k restano solo le righe di stdout | Diagnosi di un fallimento piu' povera | Riaprire il JSONL per passo con fingerprint e record dei due bracci | facoltativo |
| F2 | `scripts/tokenizer_preflight.py:93` | Il confronto CLI/config del collaudo storico passa ora per `_config_with_defaults`, che perdona qualunque chiave «successiva» mancante, non solo le due nuove | Allentamento piccolo di un controllo d'identita'; il checkout congelato afb644e non e' toccato | Aggiungere al JSON solo i due default nuovi prima del confronto | facoltativo |
| R1 | `docs/tokenizer_logsquare_protocol_20261104.md:58-62` | «Se mostrano una patologia che richiede un altro epsilon, si documenta un emendamento»: la regola non e' scritta | Scelta di ε a posteriori su diagnostiche gia' viste, anche se train-only | Scrivere prima del collaudo quale statistica (frazione sotto il pavimento per kernel e fs, RMS della derivata, scarto mascherato/non mascherato) e quale direzione farebbero scattare l'emendamento, e che senza emendamento ε resta 0,001 | requisito pre-training |
| R2 | protocollo, §«Training e lettura proposti» | Il preventivo del training va derivato dal rapporto misurato `update_mean_after8_s` LQ/L del collaudo, non dalle 17 GPU-h osservate su T100-L | Tetto 22 GPU-h plausibile ma non ancora motivato dal ramo nuovo | Pubblicare preventivo e tetto dal rapporto misurato, con commit, prima della sottomissione; stesso per le estrazioni (il ramo costa una GEMM in piu' per token) | requisito pre-training |

Nessun rilievo in `## 2) Problemi trovati` del registro: non c'e' un difetto riproducibile oggi.

## Limiti riconosciuti e condivisi

Il collaudo mostra funzione iniziale, gradienti, clipping e diagnostiche delle feature; non la
traiettoria del ramo. `log(ε + x²)` di un singolo coefficiente reale oscilla con la fase della
risposta e ha una coda negativa fino a `log ε = −6,9` nei passaggi per lo zero: e' una trasformazione
deterministica di `x`, non una stima di energia di banda; la lettura «utile o no» vale per questa
ricetta e dose. Seme unico, sviluppo riusato, due primari senza correzione: evidenza esplorativa, come
dichiarato. Lo scarto di replica T25-L − B19618 sulle stesse sonde (EPN −0,24 pp, P1 +0,51, P2 −0,09,
registro 6a67401) e' il contesto empirico con cui leggere un delta LQ − T100-L a seme unico.

## Che cosa approvo e che cosa no

- **Approvato con le correzioni C1-C3** il collaudo preparatorio: codice del ramo, flag, default,
  compatibilita', piano, pin, launcher e test. Costo 2 GPU-h massime, 16 core-h allocate, checkout
  0,333 core-h.
- **Non approvato qui** il training scientifico: richiede collaudo superato e letto (C2, R1),
  preventivo misurato (R2), orchestratori ed estrattori adattati al quarto braccio con regressioni
  contro i tre bracci storici, pin congelati e committati. GPT riprende da questo verdetto; nessun
  lancio da parte mia.
