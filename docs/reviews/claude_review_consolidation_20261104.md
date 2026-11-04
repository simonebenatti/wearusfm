# Review Claude — consolidamento T100-G (19.618 → 49.046), sonde a due profondità e primo fine-tuning EPN (GPT bbca940) — 04/11/2026

Richiesta: Simone, 04/11/2026, «tu fai il commento sul finetuning», con il prompt di GPT `docs/prompt_claude_consolidation_20261104.md`
(ramo `origin/codex/consolidation-20261104`, base 8ee6603, HEAD bbca940). Lo specchietto di coordinamento
(`docs/coordinamento_review_claude_20261104.md`) assegna a questa chat l'intera linea di consolidamento: schedule lungo, sonde ai due
livelli e fine-tuning. Solo review: nessun job, checkout remoto, training, merge o modifica ai worktree di GPT. Verdetti separati per
parte, come chiesto dal prompt; i rilievi sono ordinati per gravità (B bloccante, R requisito da scrivere prima del passo che lo riguarda,
M minore, F facoltativo, N nota) con file:riga e una correzione verificabile.

## Metodo

- Checkout temporaneo detached di bbca940 fuori dai worktree (nessuna scrittura nei checkout di GPT). Letti: protocollo, prompt, review del
  critico interno, `scripts/consolidation_continue.py` e launcher, `continuation_plan.json`, `src/wearusfm/harness/backbone_depth.py`,
  `src/wearusfm/harness/finetuning.py`, `scripts/consolidation_finetune.py` e launcher, `finetune_plan.json`, `tests/cpu/test_consolidation.py`;
  codice riusato: `run.py` (`_check_branch`, `_lr_at`, ripresa, stream, RNG, salvataggio dei checkpoint con nome), `train_jepa.py`,
  `tokenizer_train.py`, `calibration_probe.py` (`p2_internal_subjects`, `fold_roles`), `fm_features.py`, `benchmarks.py`, `spectral_step1.py`
  (`pool_p1p2`), `fm.py`, `channel_identity.py`.
- Test rieseguiti sul Mac: `tests/cpu/test_consolidation.py` **13 passati (4,26 s)**; regressioni `test_training_wsd`, `test_wsd_chain`,
  `test_tokenizer_train` **73 passati (88 s)**; `bash -n` pulito sui due launcher.
- Verifica indipendente del vincolo «zero modifiche ai moduli scientifici»: i 100 file di `scientific_source_sha256` in
  `results/infra/tokenizer_execution_20261103/training_plan.json` hanno in bbca940 lo stesso sha256 del pin (100/100), e il diff
  8ee6603→bbca940 (15 file, 1.266 righe) è fatto solo di aggiunte: nessun file di `src/` preesistente è toccato.

## Verdetti

| Parte | Verdetto | Condizioni |
|---|---|---|
| Continuazione 19.618 → 30.000 → 39.237 → 49.046 | **APPROVATO** | C-M1 e C-M2 sono documentali, da registrare prima della sottomissione |
| Benchmark FT (train-only, fold 0) | **APPROVATO** | dopo F-B1 (cambia `source_identity()`), con B-R1 scritta prima della misura; B-M2 raccomandata |
| Protocollo dei fit (3 regimi × 2 LR × 4 fold) | **DA CORREGGERE** | F-B1 nel codice, F-R2 in `decisioni.md`; F-M3/F-M4 raccomandate prima dei fit |
| Protocollo di valutazione (sonde a due profondità, matrice, OOF/IC) | **APPROVATO come disegno, CONDIZIONATO** | libreria ok; controller della matrice, pin dei checkpoint, preventivo misurato e V-R1 non ancora esistenti |

Nessuna parte autorizza un lancio: l'ordine resta quello fissato da Simone (preparazione → review → correzioni e convergenza → lancio da
GPT tramite submitter e ledger comuni, sottomissione di semplicione).

## Rilievi

### Protocollo dei fit (domande 3 e 4)

- **F-B1 (bloccante prima dei fit) — la griglia della testa non è la stessa nei tre regimi.** `src/wearusfm/harness/finetuning.py:21`
  (`LR_GRID`) e `:70` (`head_lr = 10 * lr if regime == "finetune" else lr`); tabella del protocollo
  `docs/consolidation_protocol_20261104.md:122-126`. Testa: frozen {1e-3, 1e-2}, finetune {1e-4, 1e-3}, scratch {1e-4, 1e-3}. Il contrasto
  primario finetune − frozen (protocollo :154, «isola l'adattamento») confonde l'adattamento dell'encoder con una testa ottimizzata a LR
  10× più basso; la selezione su validation fra due candidati non rimuove la differenza di griglia. Correzione a costo zero (24 candidati
  invariati): stessa griglia della testa {1e-3, 1e-2} in tutti i regimi, cioè `head_lr = {"frozen": 1, "finetune": 100, "scratch": 10}[regime] * lr`;
  aggiornare la tabella del protocollo e il test `tests/cpu/test_consolidation.py:110-133` asserendo che il gruppo della testa in
  `opt.param_groups[0]["lr"]` sia uguale fra regimi a parità di indice della griglia. Alternativa equivalente: abbassare frozen a
  {1e-4, 1e-3}; conta la parità. Da fare PRIMA del benchmark: `source_identity()` (`scripts/consolidation_finetune.py:34-36`) entra nella
  ricevuta del benchmark (`:74`), quindi un benchmark misurato col codice attuale non vale per i fit corretti.
- **F-R2 (requisito prima dei fit) — regola di lettura pre-registrata per «scratch ancora in miglioramento».** Protocollo :141-144 dice
  giustamente che una sconfitta da zero ancora in discesa non chiude il confronto, ma non definisce «in discesa» né il seguito. Scrivere in
  `docs/decisioni.md`, con SHA prima della sottomissione dei fit: (a) criterio operativo, per esempio epoca migliore su validation = 5 oppure
  guadagno di validation fra epoca 4 e 5 ≥ 0,5 pp, mediana sui quattro fold; (b) lettura consentita: «a dose supervisionata fissata (5 epoche,
  ≈19.800 update a batch 8) l'inizializzazione pre-addestrata vale +X pp [IC]», nessuna affermazione sull'ottimo di scratch né su
  «pretraining necessario»; (c) il seguito già dichiarato se (a) scatta: stessa dose maggiore per scratch E finetune (per esempio 15 epoche),
  con preventivo ×3 dei tempi misurati, così l'estensione non è post hoc. Il contrasto finetune − frozen non dipende dalla convergenza di
  scratch (entrambi partono dal teacher) e resta leggibile alla dose fissata una volta corretta F-B1.
- **F-M3 — persistenza per candidato e divergenza come esito, non come crash.** `scripts/consolidation_finetune.py:143-156` tiene i
  risultati in memoria fino a fine job; `finetuning.py:178-181` solleva su loss o gradiente non finiti e il job muore senza report. Con LR
  1e-3 da zero (AdamW, batch 8, clipping 1) una divergenza è plausibile e farebbe perdere anche il candidato 1e-4 già completato (fino a 2 h).
  Correzione: scrivere `candidate_<n>.json` appena `fit_candidate` ritorna; catturare il `ValueError` numerico nel controller e registrarlo
  come esito del candidato (`stopped: "nonfinite"`, `val_bacc: -inf`) proseguendo con l'altro; held predetto solo se almeno un candidato è
  finito. «Stop e diagnosi» (protocollo :169) resta per OOM e sovracosto. Verifica: test CPU con LR fuori griglia forzato a non-finito.
- **F-M4 — il moltiplicatore della testa non è nella ricetta.** `recipe_identity` (`scripts/consolidation_finetune.py:29-31`) e
  `finetune_plan.json` registrano `lr_grid` ma non il fattore della testa (`finetuning.py:70`): aggiungere `head_lr_multiplier` per regime al
  piano, alla tupla `keys` e al controllo di `:66` come per `lr_grid`, poi ricalcolare `recipe_sha256`.
- **F-M5 (facoltativo) — cache delle feature nel regime frozen.** `Classifier.forward` (`finetuning.py:44-50`) ricalcola l'encoder congelato
  a ogni epoca (5 × train + 5 × validation): feature deterministiche (eval, nessun dropout attivo), calcolabili una volta per ritaglio e
  riusate per la testa; circa 5× meno GPU per frozen, risultato invariato (verificabile con `allclose` sui logit).
- **F-M6 — test di parità fra il regime frozen e le feature storiche.** Aggiungere a `tests/cpu/test_consolidation.py` il confronto fra
  `torch.stack(features).mean(0)` di `Classifier(frozen)` e `FF.extract_features` sugli stessi ritagli, come `:74-86` fa per l'estrattore di
  profondità. Oggi la parità è sostenuta dal codice (`pool_p1p2` + media dei due ritagli, `finetuning.py:46-50` contro
  `fm_features.extract_features`) ma non da un test.
- **F-N7** — `scripts/consolidation_finetune.py:157`: la ricostruzione finale usa il seme di default anziché `plan["seed"]`; innocuo perché
  `load_state_dict` sovrascrive tutto, ma per coerenza passare `seed=plan["seed"]`.

Domanda 4, verificato senza rilievi (nessun leakage né vantaggio accidentale): scala per soggetto dalle sole prime 10 finestre non
etichettate del soggetto stesso (`PreparedWindows`, `fm_features.subject_scales`), uguale per i tre regimi e per le sonde storiche, dichiarata
nel protocollo come calibrazione non supervisionata (:146-150); vocabolario e class-weight dal solo train con rifiuto di classi assenti
(`finetuning.py:78-86`); ruoli train/val/held disgiunti per soggetto e partizione completa (`calibration_probe.fold_roles`, test `:136`);
testa inizializzata con lo stesso seme in tutti i regimi, separata dal RNG del modello (`finetuning.py:34-36`, test `:129-131`); ordine dei
batch e permutazioni dallo stesso seme in tutti i regimi (`:160-165`); selezione di epoca e LR solo su validation, pareggi deterministici
(`:191-193`, controller `:150-151`); held letto una volta sola con lo stato selezionato (`:157-163`, test `:150`); nessuna asimmetria
train/eval nell'encoder: nessun modulo Dropout/BatchNorm in `src/wearusfm/model/`, muscle dropout inattivo con anatomia sconosciuta del
montaggio Myo (`channel_identity.py:58-61`, `fm_features.myo8_montage` con `AnatomicalPrecision.UNKNOWN`); bf16 autocast nei due versi come
nelle estrazioni storiche; decoder, ancore e testa posa senza gradiente (test `:128`).

### Benchmark FT (domanda 5)

- **B-R1 (requisito prima del benchmark) — ripiego pre-registrato se una proiezione supera 14.100 s.** Se la popolazione interna è quella
  delle sonde del 29/10 (≈47.100 finestre, 314 soggetti; per fold ≈211 soggetti train, 24 validation, 78-79 held), un candidato fa 3.963
  update per epoca e 19.815 in tutto; `units` (`scripts/consolidation_finetune.py:121-123`) vale ≈23.540 e la proiezione
  `2 · s/update · units · 1,5 + caricamento · 1,5 + preprocessing + 600` passa solo con **≲0,18 s/update** (validation e held compresi). Per
  finetune e scratch (backward su 28 M parametri, due ritagli, batch 8) il margine è stretto; il gate è per regime (`:76`), quindi frozen
  potrebbe passare e gli altri no. Scrivere ora in `decisioni.md` il ripiego: un candidato per job (`--candidate {0,1}`; il secondo job
  legge la ricevuta del primo, seleziona su validation e predice il held), 24 job e tetto 96 GPU-ore; nessuna riduzione di epoche, batch o
  griglia dopo la misura (coerente con :169-170). Il tetto di 48 GPU-ore è correttamente un limite, non una stima.
- **B-M2 — il warm-up CUDA sta nel cronometro.** `:101-111` cronometra 32 update dal primo: inizializzazione CUDA/cuDNN e crescita
  dell'allocatore pesano 1-3 s sul primo update, cioè 0,03-0,09 s/update sulla media, 15-50 % della soglia. Conservativo, ma può bocciare il
  gate per un artefatto. Correzione: 4 update non cronometrati su un'istanza a parte (`max_updates=4`), poi i 32 misurati; riportare
  entrambe le medie (`all` e `steady`) e usare la steady × 1,5 nel gate.
- **B-N3** — i launcher escono con `exit 2` muto su argomenti o allocazione sbagliati (`scripts/slurm/consolidation_finetune.sbatch:18-23`,
  `consolidation_continue.sbatch:14-17`): un `echo >&2` aiuta chi legge il log.

Verificato senza rilievi: proiezione completa e conservativa (validation e held addebitati al costo di training, ×1,5, +600 s, caricamento
×1,5, preprocessing proporzionato all'intero dataset ×1,5, `:118-126`); gate memoria su RSS × rapporto finestre × 1,3 ≤ 85 % di 100 GiB e
CUDA riservata ≤ 85 % della GPU (`:127-130`), coerente col launcher (100 G, 1 GPU, `:6-9,34-37`); solo soggetti train del fold 0 nel forward
(`:88-96`); nessun checkpoint scientifico; costo 0,5 GPU-ora e 4 core-ore; ricevuta pinnata e hash di ricetta e sorgenti obbligatori prima
dei fit (`:71-77`, test `:169,178`).

### Continuazione (domanda 1)

- **C-M1 — ripresa di una fase fermata dal limite di tempo.** `scripts/consolidation_continue.py:61` rifiuta un output esistente e `:73-76`
  esigono la ricevuta completa della fase precedente: se il trainer si ferma per `--time-limit-s` (10.500 s) il `checkpoint.pt` è valido ma
  nessun job può riprendere la stessa fase senza intervento. Margini stimati 23-32 % (8.050 / 7.162 / 7.606 s a 0,775 s/passo contro
  10.500 s), quindi improbabile ma non escluso. Pre-registrare la procedura: nuova revisione del piano (nuovo `plan_sha256`) oppure
  `--attempt N` che accetta la ripresa della stessa fase solo con `summary.stopped` = limite di tempo, `STOP` assente e identità del
  `checkpoint.pt` verificata; mai un retry cieco. Lo stesso vale per il tempo di hashing finale (tre file da 454 MB più gli input) nei 300 s
  residui.
- **C-M2 — conservazione degli endpoint scientifici.** `continuation_plan.json` mette `run_dir` in $SCRATCH (purge a 40 giorni): 39.237 e
  49.046 sarebbero le uniche copie. Registrare la data di purge (data del job + 40 giorni) e la copia o il rinfresco, come per il 19.618
  (purge 13/12/2026), prima delle estrazioni della matrice.
- **C-N3** — stima: 0,775 s/passo (11.334 s per 14.618 passi, avvio e fixed-eval compresi) × 29.428 passi ≈ 22.800 s ≈ 25,35 GPU-ore ✓,
  tetto 36 GPU-ore e 288 core-ore ✓.

Verificato senza rilievi: continuità di ricetta, schedule e stream dimostrata dalle guardie, non solo dichiarata. `command_for`
(`consolidation_continue.py:27-42`) ricostruisce la configurazione dagli argomenti della ricevuta e la richiede uguale a
`19618/T100-G/config.json` a meno di `max_steps` (`:40`), con `decay_start` 39.237 = 49.046 − round(0,2 · 49.046) e `total_steps` 49.046
(`:41`); la ricevuta originale ha già `schedule wsd, total 49.046, decay_fraction 0,2`; `_lr_at(19618) = 1e-4`, `_lr_at(39237) = 1e-4`,
`_lr_at(49045) < 1e-7` (test `:36-45`): nessun riavvio del warmup. Primo segmento con `--branch-from` e cartella nuova (`:34-35,70-71`):
`run.py:823-839` (`_check_branch`) esige configurazione identica tranne schedule e `max_steps`, checkpoint nella fase stabile dell'origine e
del ramo, stato dello stream presente; `run.py:1292-1295` rifiuta uno stream diverso (workers 6, segmento 1.200, world 4: launcher 4 GPU/32
CPU/480 GiB `:7-9,28-31`, `torchrun --nproc_per_node=4`, `--num-workers 6` nella ricevuta); RNG torch/CUDA per rank ripristinati
(`:1366-1373`); stream ripreso da `start=step` (`:1458-1459`); ottimizzatore e quattro RNG validati da `checkpoint_identity` prima e dopo
(`:66-69,77-78,86,92-93`). Segmenti successivi riprendono `run_dir/checkpoint.pt` con `_check_resume_recipe` (regressioni WSD 73 passate).
Checkpoint con nome: `checkpoint_predecay.pt` a `decay_start` e `checkpoint_postdecay.pt` a `total_steps` li salva il trainer dentro il
ciclo (`run.py:1683-1688`), prima del `save()` finale (`:1700`); il 30.000 è una copia esatta confrontata sullo stato addestrato (`:89-93`).
Fase completa solo con `stopped == "max_steps"`, passi esatti e `STOP` assente (`:84-85,79`); nessun retry cieco (`:61`); sorgente in sola
lettura verificata dopo (`:94`); 100/100 sorgenti scientifici identici al pin; `continuation_plan.json` in `awaiting_claude_review` e
controller che rifiuta (`:49`, test `:178`).

### Protocollo di valutazione (domanda 2, matrice e OOF/IC)

- **V-R1 (requisito prima delle estrazioni) — regola di lettura a due lati per i contrasti di dose e di profondità.** Protocollo :90-92
  elenca i contrasti (49.046 − 19.618, 39.237 − 19.618, 49.046 − 39.237, post − pre a ogni dose) e dice di riportarli tutti, ma non
  dichiara quale esito conta come «più esposizione aiuta» né quale farebbe scegliere il riferimento (19.618, 39.237 o 49.046) per i passi
  successivi. Scriverla in `decisioni.md` prima delle estrazioni, con il pavimento di rumore della replica T25-L contro B19618 (0,2-0,9 pp
  a seconda del compito) sotto cui un delta non si legge, come già chiesto per il 300M (S8) e per il log-quadratico (R1).
- **V-R2 (condizione, già dichiarata da GPT)** — controller di estrazione e fitting della matrice con D parametrico, pin dei checkpoint
  effettivi, benchmark CPU e preventivo misurato: niente è lanciabile oggi e i valutatori con 768 cablato non si usano su cache a D colonne.
- **V-F3 (facoltativo) — terza vista «pre normalizzata».** Il contrasto post − pre somma trasformazione del backbone e LayerNorm finale;
  sullo stesso tensore catturato (`backbone_depth.py:24-28`) si può aggiungere `layer_norm(captured.float(), (D,))` senza affine, per
  separare l'effetto della normalizzazione per esempio da quello dei blocchi. Diagnostico, stesso forward, costo nullo.
- **V-N4** — OOF: `fold_roles` (seed 20261015, validation 10 %) è la funzione delle sonde storiche, quindi i held coincidono con i fold OOF
  storici: l'IC appaiato per soggetto vale anche contro la logistica storica (82,58 %) come riferimento separato (protocollo :152-153).

Verificato senza rilievi (domanda 2): hook temporaneo sull'uscita effettiva del Perceiver dentro `model.encode`, rimosso anche in caso
di errore (`backbone_depth.py:24-28`); una sola cattura della stessa forma di `enc.z` (`:29-30`); medie in FP32 sui K slot e, con la stessa
maschera `key_time_valid`, sul tempo valido (`:31-36`); post uguale alle prime D colonne del pool storico, verificato a ogni chiamata
(`:37-39`) e nei test (`:64-65,82`); larghezza presa da `model.cfg.dim`, test a 384 e 896 (`:89-107`); `extract_windows` media i due ritagli
EPN come `extract_features` (test `:74-86`), tiene i ritagli su un asse separato nella cache temporale (`:85`) ed è dichiarato solo per
classificazione, con la posa su `depth_readouts` applicato agli input del loader originale (protocollo :76-79). La lettura «dove i lettori
accedono all'informazione, non perdita irreversibile» (protocollo :21) è quella corretta per un confronto appaiato di profondità.

## Letture sostenibili del primo fine-tuning (domanda 3)

Cinque epoche a batch 8 con due LR per regime sono uno screening ragionevole a dose fissata: stessa esposizione supervisionata, stessa
testa, stessa selezione. Sostenibile, una volta corretta F-B1: il vantaggio dell'inizializzazione pre-addestrata e dell'adattamento
dell'encoder **a questa dose**, con IC appaiati per soggetto sui quattro fold. Non sostenibile: qualunque affermazione sull'ottimo di
scratch, su «pretraining necessario» o su regressione/«FM utile in generale» (protocollo :157-158). Se scratch è ancora in miglioramento
secondo F-R2, il risultato si riporta come efficienza di campione alla dose fissata e il confronto a dose maggiore è quello già
dichiarato, non una scelta fatta dopo i risultati.

## Cosa autorizza questa review

Nessun lancio. Ordine: F-B1 e F-M4 (e F-M3 raccomandata) nel codice → B-R1 e F-R2 in `decisioni.md` → benchmark FT (0,5 GPU-ora) → pin
della ricevuta → fits. Continuazione: approvata; GPT può portare `continuation_plan.json` a `claude_review_converged` dopo aver registrato
C-M1 e C-M2; la sottomissione passa da semplicione con il submitter e il ledger comuni, secondo l'ordine fissato da Simone. Valutazioni
della matrice: non approvate oggi, da ripresentare con controller, pin, preventivo misurato e V-R1.

---

## Rilettura delle correzioni (GPT 2dbd498, diff bbca940 → 2dbd498) — 04/11/2026, sera

Richiesta: Simone con il prompt di GPT `docs/prompt_claude_consolidation_followup_20261104.md` («Rileggi il diff [...] Restituisci verdetti
separati per continuazione, benchmark FT, fit e regole di lettura; indica soltanto eventuali blocchi residui e correzioni necessarie»).
Solo review, nessun job. Metodo: checkout temporaneo detached di 2dbd498 (poi rimosso); letti la risposta punto per punto
(`docs/reviews/consolidation_response_to_claude_20261104.md`), la voce nuova di `docs/decisioni.md`,
`docs/consolidation_continuation_operations_20261104.md`, `review_corrections_validation.json`, i diff di `finetuning.py`,
`consolidation_finetune.py`, `consolidation_continue.py`, dei due launcher, dei due piani e del protocollo, e i tre file di test nuovi.
Verifiche mie: **123 test passati in 82,10 s** (stesso comando dell'autore); `bash -n` pulito sui launcher; `recipe_sha256` del piano
ricalcolato con `recipe_identity` del checkout (uguale), `scratch_learning_rule` e `head_lr_multiplier` del piano uguali alle costanti del
codice; griglie della testa calcolate dal codice: {1e-3, 1e-2} in tutti e tre i regimi; `fit_packaging` sui tre casi (joint/split/blocked);
`fold_roles` restituisce dizionari di liste di stringhe, quindi l'uguaglianza della provenienza fra primo e secondo job split regge al
passaggio per JSON; `balanced_accuracy` ignora le classi assenti (nessun NaN con la validation da 8 finestre del benchmark); `write_json`
apre in esclusiva con `allow_nan=False`; su Leonardo (sola lettura) `/leonardo_work/IscrB_WearUsFM` e `$SCRATCH` non sono symlink, quindi i
controlli con `resolve()` passano.

### Verdetti alla convergenza

| Parte | Verdetto | Residuo |
|---|---|---|
| Continuazione | **APPROVATA, convergenza raggiunta** | solo operativo: pin della quota entro 24 h dalla sottomissione (N-2) |
| Benchmark FT | **APPROVATO, convergenza raggiunta** | `OUT` del benchmark sotto $WORK alla sottomissione (R-1) |
| Protocollo dei fit | **APPROVATO con una correzione prima dei fit** | R-1: esiti scientifici dei fit fuori dalla purge di $SCRATCH |
| Regole di lettura (F-R2, V-R1) | **CONVERGENZA ESPLICITA** sulla formulazione di GPT | nessuno; N-3 e' una nota, non una modifica |
| Valutazione della matrice lunga | invariata: condizionata (V-R2) | controller a D parametrico, pin dei checkpoint, preventivo misurato |

Trattamento dei rilievi, verificato nel diff: F-B1 e F-M4 (`HEAD_LR_MULTIPLIER` 1/100/10, piano, `recipe_identity`, controllo a runtime,
test `test_head_grid_matches_by_candidate_and_encoder_grid_is_unchanged` e `test_head_multipliers_are_part_of_recipe_hash`); F-M3
(`run_candidate` scrive `candidate_<n>.json` e `.pt` appena il candidato finisce; `NumericalDivergence` distinta per loss, gradiente e
predizione, con storia e contesto; OOM ed errori di codice propagano con il loro tipo; `select_candidate` solo su validation con pareggio
al LR minore; held predetto una volta e un non-finito sul held non fa scegliere l'altro LR; entrambi divergenti = fold fallito senza OOF
inventate); F-M6 (parita' dei logit frozen con `extract_features` a due ritagli); F-N7; B-M2 (4 update di warm-up su istanza separata,
medie `all` e `steady`, proiezioni sulla steady); B-R1 (`fit_packaging` per regime, `--candidate`, gate che accetta solo l'impacchettamento
deciso dal benchmark, layout d'uscita congelato, secondo job che legge la ricevuta del primo con pin, provenienza e hash dei file; tetti
48/96 GPU-ore alternativi); F-R2 (`SCRATCH_RULE` nel codice e nel piano, `scratch_still_improving` testata, seguito a 15 epoche da
inizializzazioni originali con orizzonte nuovo, non ripresa); C-M1 (procedura versionata, nessun retry generico, ripresa dopo 39.237 con la
ripresa normale e non con `--branch-from`); C-M2 (`archive_checkpoint` con `.partial` esclusivo, fsync, hash, hard link atomico senza
sovrascrittura, copie in $WORK di 19.618/30.000/39.237/49.046 prima della ricevuta, quota di progetto letta prima del lancio e a runtime);
B-N3. F-M5 e V-F3 rinviati: accettato, non cambiano la matrice.

### Residui

- **R-1 (correzione prima dei fit, una riga di piano).** `finetune_plan.json` → `fit_output_root` punta a `$SCRATCH` (purge a 40 giorni):
  li' finirebbero `report.json`, `candidate_<n>.json`, `held.npz` (le OOF) e il layout obbligatorio `fold<f>/<regime>/<joint|candidate>`
  verificato dal controller. Gli esiti scientifici dei fit devono stare sotto `$WORK/wearusfm_runs/results/consolidation_20261104/finetuning/`
  come dice il protocollo (:175), con i soli `.pt` dei candidati eventualmente in $SCRATCH (24 × ≈113 MB), oppure con una copia
  persistente prima di `complete: true` come per C-M2. `fit_output_root` non entra in `recipe_identity`: la correzione non cambia l'hash
  della ricetta. Lo stesso vale per `OUT` del benchmark, che il controller accetta ovunque sotto `wearusfm_runs`: sotto $WORK, perche' la
  ricevuta viene pinnata nel piano.
- **N-2 (operativo, prima della sottomissione della continuazione).** Lettura reale della quota raccolta da me in sola lettura il
  04/11 alle 17:04 UTC (`~/wearusfm_local/logs/lfs_quota_work_20261104.txt`): progetto 20147512, usati 1.031.361.028 KiB su quota soft
  1.073.741.824 KiB (983,6 GiB di 1 TiB), hard 0, inode senza quota → **40,4 GiB liberi**, gate da 8 GiB soddisfatto, requisito sugli
  inode non applicabile. `check_quota_evidence` del checkout accetta questo testo (anche con `submission_time`): il parser e' valido sul
  formato reale. Il pin in `retention.quota_preflight` lo fa il proprietario; il controllo delle 24 ore non lo fa il submitter comune:
  chi sottomette deve eseguire `check_quota_evidence(plan["retention"]["quota_preflight"], plan["retention"],
  submission_time=datetime.now(timezone.utc))` subito prima di `sbatch` (meglio un'opzione `--check-quota` del controller, da documentare
  nella procedura di semplicione). La quota e' condivisa e quasi piena: una lettura positiva non e' una prenotazione.
- **N-3 (nota su F-R2, nessuna modifica).** Con il cosine a zero il guadagno dell'epoca 5 e' compresso dallo schedule, quindi la clausola
  sulla mediana (≥ 0,5 pp) scatta di rado anche per una rete lontana dall'ottimo; la clausola operativa e' `best_epoch == 5` in ≥ 3 fold.
  La regola e' quindi conservativa nel senso di non far scattare il seguito; un mancato scatto non autorizza comunque affermazioni sulla
  convergenza di scratch (gia' scritto). Resta congelata com'e'.
- **V-R1, convergenza esplicita.** Concordo con la riformulazione: la replica T25-L − B19618 e' un controllo empirico appaiato con i suoi IC,
  riportato accanto ai nuovi contrasti, non un pavimento statistico universale ricavato da una coppia; non si sottrae e non si divide.
  Soglie pratiche a due lati congelate in 2dbd498 prima di qualunque estrazione: 1 pp di balanced accuracy su P1 o P2 con IC appaiato
  inferiore > 0 e non-inferiorita' a −1 pp sull'altro indice; 2 % relativo di E_dyn con IC favorevole; peggioramento pratico speculare;
  IC che attraversano lo zero = inconclusivo, non equivalenza; riferimento per compito (classificazione e posa possono differire), scelta
  dichiaratamente esplorativa; post − pre con gli stessi margini. Le soglie non si abbassano dopo i risultati. Nota descrittiva: 1 pp supera
  tutti i delta osservati nella replica (massimo DB6 +0,86 pp; P1 +0,51 con IC che esclude lo zero), che e' proprio il motivo per cui il solo
  criterio dell'IC non basta.

### Cosa autorizza la rilettura

Convergenza raggiunta su continuazione, benchmark FT e regole di lettura: GPT puo' portare `continuation_plan.json` a
`claude_review_converged` (dopo il pin della quota) e sottomettere i tre segmenti e il benchmark tramite semplicione, submitter e ledger
comuni, nell'ordine fissato da Simone. Per i fit: `fits_status` a `claude_review_converged` dopo R-1 (una riga di piano, senza nuova
rilettura se il diff e' solo quello), poi pin della ricevuta del benchmark, preventivo effettivo e modalita' per regime pubblicati, poi i
fit. Valutazioni della matrice lunga e aggregatore OOF/IC: ancora da consegnare, non autorizzati. Nessun lancio da parte mia.
