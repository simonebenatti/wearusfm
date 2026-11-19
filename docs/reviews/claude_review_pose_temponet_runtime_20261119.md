# Review Claude: review operativa del pacchetto TCN frozen (19/11/2026)

Oggetto: il pacchetto eseguibile del confronto delle teste di posa sul teacher T100-G 30M congelato a 49.046 passi, ramo
`codex/pose-temponet-implementation-20261119`. Consegna `2f982114719dc2abd723bdbf0147d7fef34a731c`, codice
`6283b5bd1e36d4a92b96bedbde764aaf9e108dde`, base del diff `0e56c8b1`. È la review del pacchetto operativo prevista dopo la
convergenza documentale `ad9f514` sul protocollo v3.

**SHA effettivamente rivisto: `2f982114719dc2abd723bdbf0147d7fef34a731c`.** Rispetto a `6283b5bd` cambia solo
`delivery_pin.json`. Il manifest dei 245 sorgenti è identico nei due commit e coincide con `plan.json`.

Richiesta: messaggio di GPT incollato da Simone il 19/11 («è arrivat questo»). Cinque aree: architetture e loss; controlli,
convergenza, selezione e separazione dai dati held; provenienza e blocco dei 15 rifit; benchmark e preventivo massimo;
contrasti, intervalli, allarmi e fallimenti. Nessun lancio da parte mia: l'operatore resta «2-TCN head launcher».

## Metodo

- Checkout isolato e staccato del commit rivisto in un worktree mio; il worktree di GPT non è stato toccato.
- Test sul Mac:
  - i 17 file nuovi: 204 test passati;
  - i 5 file storici della consegna: 125 passati e 1 fallito, `test_concrete_campaign_remains_review_gated_and_all_endpoints_pinned`,
    lo stesso che GPT documenta in `baseline_failure.json`;
  - suite CPU completa con solo `src` nel path, come in CLAUDE.md, in quattro parti senza il test DDP: 2.563 passati e lo stesso
    fallimento; poi il test DDP da solo, 6 passati. In tutto 2.569 passati su 2.570.
- Review a cinque lenti, una per area, ciascuna con un controllore avversario. Sono 130 voci di controllo elencate e 23 rilievi:
  21 confermati, 1 incerto (L3-4) e 1 confutato (L5-4).
- Dopo il blocco B-4, trovato da un controllore, una prova della catena intera con le funzioni vere e una lettura statica del
  collegamento fra gli stadi (sezione 2).
- Ho riprodotto io B-2, B-3 e B-4. B-1 sul Mac non si vede: si basa sul log del 06/11 (sezione 6).
- Prima della pubblicazione due rilettori hanno controllato la bozza contro codice, protocollo e prove: 33 segnalazioni, in parte
  le stesse, tutte corrette; nessuna cambia il verdetto.
- Nessun job, nessun accesso al cluster. Prove e script in `claude_review_pose_temponet_runtime_20261119_evidence/`.

Criterio, lo stesso delle review precedenti: blocco è solo ciò che farebbe fallire un job, falserebbe un risultato o ammetterebbe
evidenze sbagliate, compresa una spesa sbagliata del budget per un gate sbagliato.

## Esito

**DA RIVEDERE prima del primo invio.** Ci sono quattro blocchi: con il codice di oggi la catena non arriva ai fit. Ci sono anche
sei difetti importanti e alcune minori. Tutto il resto corrisponde al protocollo v3: architetture, loss, supporto, numerica,
selezione, controlli, provenienza, gate e lettura finale (sezione 5).

Va corretto tutto in una sola revisione, prima del primo invio. Gli ID logici, `CODE_DIR` e la radice degli output sono costanti
del codice (`execution.py:11` e `:22`, `provenance.py:26`). Ogni cambio dei sorgenti invalida tutte le ricevute già scritte. Una
correzione dopo il primo invio costringe quindi a rifare da capo tutti gli stadi già eseguiti, con ID e checkout nuovi (con B-2:
preflight, benchmark e WLS), e serve una nuova decisione di Simone (J: «retry or exceeded cap needs decision»). Per questo chiedo di correggere ora anche gli importanti.

Le correzioni sono piccole, tutte di runtime, e nessuna cambia la ricetta. Dopo la revisione rileggo solo il diff, senza agenti.

## 1. Blocchi

**B-1. Il launcher sostituisce `PYTHONPATH`** (`scripts/slurm/pose_temponet.sbatch:14`; rilievo L3-1).
- `export PYTHONPATH="$CODE_DIR/src:$CODE_DIR/scripts"` viene dopo `module load cineca-ai/4.3.0`. Su Leonardo numpy e torch
  arrivano proprio dal `PYTHONPATH` del modulo. È già successo il 06/11: il log
  `results/infra/consolidation_long_20261105/execution/audit/preflight_fit_20261106_environment_diagnosis.log` (commit
  `f58f5ff`) registra «numpy with PYTHONPATH replaced» → `ModuleNotFoundError: No module named 'numpy'` e «with src:scripts
  prepended» → `1.26.4`. `lib_code_dir.sh:61` antepone; la riga 14 di `pose_temponet.sbatch` è l'unica di `scripts/slurm` che sostituisce
  `PYTHONPATH`.
- Conseguenza: tutti i 24 job muoiono al primo import, prima di scrivere una ricevuta. Il preflight ha però già creato il checkout
  di campagna in sola lettura e consumato l'ID nel ledger: per ripartire servono ID e checkout nuovi, quindi un cambio dei
  sorgenti e una nuova review.
- Sul Mac non si vede, perché numpy e torch stanno nel venv.
- Correzione: `export PYTHONPATH="$CODE_DIR/src:$CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"`.

**B-2. I walltime misurati non sono minuti interi** (`resources.py:93`, `runner.py:163-165`; L3-2 e L4-1).
- `requested = max(1, math.ceil(margin * raw_seconds))` è un numero qualsiasi di secondi. `--time` lo passa con i secondi, Slurm
  arrotonda per eccesso al minuto, e `_allocation` pretende la stringa esatta. Riprodotto: 5.575 s → `--time=01:32:55` →
  `TimeLimit=01:33:00` → «SLURM walltime differs from published allocation».
- Conseguenza: ognuno dei 16 job neurali rifiuterebbe la propria allocazione, con probabilità 59/60. I walltime fissi (720,
  1.620, 14.220 e 3.420 s) sono minuti interi, quindi passano. In pratica, dopo preflight, benchmark e WLS (68,5 core-ore CPU-only
  e 0,5 GPU-ore) i sei job dei controlli rifiutano l'allocazione e le selezioni non partono. Il rifiuto avviene prima del `try`,
  quindi senza ricevuta.
- Il gate conta inoltre fino a 59 s in meno per job rispetto a ciò che Slurm concede: al massimo 944 s, 0,262 GPU-ore sui 16 job.
- Correzione: arrotondare per eccesso al minuto in `resources._job`, per esempio
  `requested = 60*math.ceil(max(1, math.ceil(margin*raw_seconds))/60)`. Richiedere `seconds % 60 == 0` in `validate_graph` e
  in `apply_training_pack`. Aggiungere un test che parta da un tempo misurato non intero e arrivi fino ad `_allocation`.

**B-3. I comandi stampati usano `--agente gpt_tcn_launcher`** (`execution.py:168`; L3-3).
- `scripts/submit_job.py:306` accetta solo `choices=("claude", "gpt", "simone")`, quindi argparse rifiuta ogni riga generata con
  codice 2, prima della prenotazione. Il costo è nullo, ma nessun comando rivisto è usabile. Una riga corretta a mano non sarebbe
  più quella del pacchetto rivisto, e `submit_job.py` è nel manifest dei sorgenti.
- Correzione: `--agente gpt`, più un test che confronti il valore con le scelte dichiarate dal parser del submitter.

**B-4. La funzione di avanzamento fa fallire ogni ricerca** (`runner.py:185-189`, `train.py:447-450`,
`selection.py:103-104`). L'ha trovato il controllore della lente 1 mentre verificava L1-2.
- `_progress(pair)` chiama `pair.curve()`, che esegue `CandidateCurve.validate()`. `validate()` solleva
  `ValueError("incomplete candidate curve")` finché l'ultimo punto non è all'orizzonte, quindi già al passo zero.
- `_progress` è passata come `on_checkpoint` ai controlli e alle selezioni (`runner.py:294` e `:299`): ogni ricerca si ferma dopo
  la valutazione al passo zero dei due fold. In pratica, dopo preflight, benchmark e WLS (68,5 core-ore CPU-only e 0,5 GPU-ore),
  i sei job GPU dei controlli finiscono ciascuno con una ricevuta `technical_failure` e le cinque selezioni non vengono mai
  rilasciate. La guardia sulle ricevute esistenti impedisce poi di rilanciare gli ID.
- Riprodotto sulla fixture del pacchetto: con `on_checkpoint=None` la ricerca finisce, con `on_checkpoint=runner._progress`
  solleva «incomplete candidate curve» dopo 2 valutazioni.
- I test non lo vedono perché `test_pose_temponet_runner.py:59-63` sostituisce `workflow.run_selection` con una funzione finta,
  e i test di train e workflow usano `on_checkpoint=None`.
- Correzione (solo log, nessun effetto numerico): in `_progress` leggere `pair.points` invece di chiamare `pair.curve()`. Si può
  stampare lì anche `initial_sha256` delle traiettorie al passo zero, il che chiude L1-2 per controlli e selezioni; i rifit non
  passano da `_progress`, quindi serve la stessa stampa in `_run_refits` prima di `train_to_step`. Aggiungere un test con
  `on_checkpoint=runner._progress`.

Dopo questi quattro, la prova della catena intera della sezione 2 ha cercato altri blocchi nascosti dietro B-4.

## 2. Prova della catena intera

**Catena intera con le funzioni vere.** In una copia del pacchetto, con B-4 corretto come proposto e B-2 aggirato in
`runner._allocation` (TimeLimit atteso arrotondato al minuto, non la correzione proposta in `resources._job`), un agente ha fatto
girare i 24 stadi in ordine attraverso la riga di comando di produzione (`python -m pose_temponet run`). Ha usato le funzioni vere di calibrazione,
controlli, selezione, rifit, inferenza bloccata, metriche e raccolta, su CPU, con fixture piccole e una ricetta di poche epoche.
- Era simulato solo ciò che richiede il cluster o i dati veri: preflight, misure dei benchmark, cache e predizioni storiche,
  `scontrol`, numero di GPU, pin numerici e `configure_numerics` (sul Mac serve un solo thread); `validate_plan` puntava al
  repository della fixture. Il gate dei controlli è stato forzato, perché con 4 epoche dà correttamente «ottimizzazione non
  risolta». Della parità R-1 sono state saltate le uguaglianze di valori e conteggi; forma e uguaglianza del ritaglio sono state
  verificate.
- Esito: tutte le 24 ricevute risultano `completed`, il rapporto è completo, con 15 utenti e 5 contrasti, e l'inventario conta
  105 file. Il piano misurato è stato scritto e riletto da `resource-plan`. Nessun altro blocco era nascosto dietro B-4.
- Controlli indipendenti sulle uscite:
  - per L0, TCN-post e TCN-pre, con i tre semi (9 celle su 15), le predizioni bloccate coincidono con un forward diretto degli
    stati dei rifit (differenza massima 0,0); MLP e L13 non sono stati ricontrollati così, e nello script salvato
    `oracle_check.py` questo ciclo è disattivato;
  - l'E_dyn pooled di TCN-post al seme 0, ricalcolato a mano, coincide con il rapporto a 6e-16;
  - la cartella di sviluppo si apre solo nell'inferenza bloccata e nelle metriche.
- Limiti: non sono stati provati i rami CUDA (RNG del dropout, determinismo), il gruppo statico non vuoto e i NaN fuori dalla
  maschera. Questi ultimi, a una lettura statica, sono mascherati da tutti i consumatori prima dei conti.
- Un'informazione, non un difetto. Nella prima prova la calibrazione temporale non ha trovato coppie, con E_real sintetico 0,55,
  e la catena si è fermata con `control_non_calibrated`, come vuole il protocollo. Con E_real 0,82, vicino allo storico 0,786, ha
  calibrato. Fra i punti della griglia di `a` (passo 0,2) possono restare buchi: «non calibrato» è un esito possibile e legittimo
  anche con i dati veri.

**Lettura statica del collegamento.** Firme, parole chiave e chiavi di ritorno di ogni chiamata di `run_stage` corrispondono.
Coincidono anche i nomi e i formati degli artefatti fra lo stadio che li scrive e quello che li legge, i pin dopo la rilettura
JSON e i gruppi numerici con le risorse del piano. B-4 è l'unico validatore che rifiuta stati intermedi. La sequenza della riga di
comando dell'operatore, provata in un repository di prova, accetta gli argomenti del documento d'esecuzione. Un difetto è nuovo, I-5 (S-1). I-6 (S-2) riprende L3-5, che il controllore della lente 3 aveva giudicato
minore e documentale perché P:202 mette la misura della quota nel preflight dell'operatore; lo promuovo a importante per i motivi
detti in I-6. Entrambi sono nella sezione 3.

Driver della prova, log e risultati: cartella `smoke/` e `smoke_workflow_result.json` nelle prove.

## 3. Importanti, da correggere nella stessa revisione

**I-1. Un controllo in forma chiusa «non diagnostico» diventa `technical_failure` dentro i sei job GPU dei controlli**
(`runner.py:102` e `:388`; L2-1 e L5-1).
- `_calibrations` verifica il gate combinato con `P.require(...)`, che solleva un `ValueError`, e `run_stage` lo registra come
  `technical_failure` invece che come `gate_failed`. Il gate combinato (fold 0, fold 1 e pooled) si valuta per la prima volta
  dentro ciascuno dei sei job GPU, perché le due calibrazioni guardano solo il proprio fold. Quando non passa non viene salvato:
  la ricevuta contiene solo il messaggio, mentre gli ingressi del gate restano negli `outer_reference_statistics` delle due
  `calibration.json`.
- Conseguenze. Lo stato del protocollo `controllo_non_diagnostico` (stop, P §4.4) diventa un guasto tecnico («diagnosticare prima
  di una nuova decisione», P §4.5): l'operatore viene indirizzato male. Sei allocazioni GPU si spendono per un gate calcolabile su
  CPU, e la ricevuta non conserva i margini falliti, da ricalcolare dalle calibrazioni. Nessuna evidenza scientifica sbagliata
  viene ammessa.
- Altri due casi correlati. `runner.py:286` scrive «control_non_calibrated» anche quando la calibrazione è
  `control_non_diagnostic`. `controls.py:97`, il teacher degenere che P:86 chiama «non calibrabile», finisce anch'esso in
  `technical_failure`.
- Correzione minima: calcolare una volta il gate combinato, salvarlo come artefatto e sollevare `GateNotPassed` con lo stato vero.
  Meglio ancora deciderlo prima di rilasciare i job GPU, per esempio facendo rifiutare a `commands --stage controls` le righe se il
  gate combinato delle due ricevute di calibrazione non è diagnostico. La scelta è di GPT.

**I-2. Il benchmark GPU misura un percorso di valutazione diverso da quello dei fit** (`benchmark.py:155`, `workflow.py:241`;
L1-1 e L4-3).
- Il benchmark fa il cast di un tensore FP16 già caricato. La produzione passa a ogni batch da `D.view_inputs`, che ricalcola
  `lengths_for` su tutti i metadati, `np.isfinite`, `np.where` e `astype(float32)`, e costruisce `_metric_panel`. P §9 chiede «stessi
  loader/file placement, cache cast» fra benchmark e fit.
- Misure sul Mac, con fixture della geometria di produzione: per una valutazione di un fold held 331 ms in produzione contro
  146 ms nel benchmark (2,3 volte); con altre fixture e per batch da 1,5 a 1,9 volte.
- Conseguenza, secondo la stima del revisore: per le teste dominate dalla valutazione (L0, MLP, L13) il margine del fattore 2
  scende a circa 1,4-1,5. Il rischio
  di TIMEOUT è piccolo ma reale, e un TIMEOUT perderebbe le GPU-ore già spese. Nessun valore scientifico cambia.
- Correzione: misurare nel benchmark proprio `workflow._score_model` (e `D.view_inputs` per l'inferenza), oppure far leggere alla
  produzione gli held con lo stesso cast FP16→FP32 per batch del benchmark (`_eval_model`, righe nell'ordine fisico; non
  `TensorTrainingSet.batch()`, che ordina per ID e scarta le finestre senza supporto, mentre P:55 non esclude finestre dalle
  predizioni), calcolando le lunghezze una volta sola.

**I-3. Salvataggi e caricamenti misurati senza riscaldamento** (`benchmark.py:272-274`, `train.py:183-193`,
`resources.py:155-165`; L4-2).
- `Trajectory.state()` legge `data.identity`, che al primo accesso calcola l'hash dell'intero dataset. Nulla lo tocca prima del
  ciclo misurato, quindi il primo dei 10 campioni del salvataggio contiene l'hash. Sul Mac: 0,106 s contro 0,001 s a regime.
- Il p95 lineare su 10 campioni ne prende il 55% (0,059 s, circa 50 volte il regime). Il gate lo moltiplica per 84.756
  salvataggi conteggiati (80.256 + 4.500), che la produzione non scrive mai su disco: `save_checkpoint` e `load_checkpoint`
  compaiono solo nel benchmark.
- Conseguenza: il gate può fermare una catena che starebbe nei tetti. In un esempio illustrativo la fase neurale passa da 19,52
  a 20,89 GPU-ore. Non è una spesa sbagliata, ma uno stop che richiede una nuova decisione.
- Correzione: un salvataggio non misurato, o un accesso a `identity`, prima dei 10 campioni, come per train ed eval; e misurare
  ciò che la produzione fa davvero (stato in memoria a ogni miglioramento, una scrittura di artefatto per job).

**I-4. Il diagnostico «margine recuperato dalla WLS L13» mescola un fit su `y` con la base su `y*`** (`controls.py:606-615`;
L2-2).
- P:96 chiede, prima dei fit neurali, la quota del margine temporale recuperata dalla WLS L13. La base è la WLS L0 fittata su `y*`
  (E_cleanLS). Il codice la confronta con una WLS L0/L13 fittata nel campione sull'osservato `y`, quindi il valore contiene il fit
  del rumore. Su un piccolo mondo sintetico: 0,0334 per L0, che per costruzione dovrebbe essere 0, e 1,7365 per L13, oltre
  l'oracolo.
- È solo descrittivo, ma esce nei `linear_diagnostics` di `positive_gate.json` e in ogni rapporto di selezione reale
  (`closed_form_linear_diagnostics`), non in `report.json`: farebbe credere che un lettore lineare a 13 patch legga già gran
  parte del margine temporale.
- Correzione: per questo diagnostico fittare L0 e L13 sul bersaglio pulito, come E_cleanLS, e valutare su `y`; test che L0 dia 0.

**I-5. Il preflight crea il checkout di campagna prima di verificare il commit del piano** (`pose_temponet_checkout.sbatch:12-13`,
`__main__.py:48` e `:156`; rilievo statico S-1).
- Il wrapper crea `$WORK/wearusfm_campaigns/pose_temponet_20261119` (riga 12) e solo dopo lancia il launcher che verifica
  `TCN_PLAN_COMMIT` ed esegue `materialize-plan` (riga 13). `commands` accetta anche un riferimento o uno SHA corto, perché si
  limita a un `git show`, mentre `materialize_plan` pretende 40 caratteri esadecimali. Provato in un repository di prova:
  `--plan-commit HEAD` stampa `TCN_PLAN_COMMIT=HEAD` e `materialize-plan` fallisce con «full plan commit/hash required».
- Conseguenza: un errore dell'operatore, oppure qualsiasi guasto fra il checkout e `run_stage` (B-1 compreso), lascia il checkout
  creato e nessuna ricevuta. Il preflight non si può ripresentare: il ledger rifiuta l'ID già `sottomesso` («Usare un ID nuovo»),
  gli ID sono costanti del codice e, anche con un ID nuovo, `campaign_checkout` rifiuterebbe la destinazione esistente, perché
  `CAMPAIGN` è una costante rivista. Un solo errore evitabile chiude la campagna; la verifica dei 40 esadecimali in `commands`,
  prima dell'invio, è la parte decisiva della correzione.
- Correzione: nel wrapper, prima del checkout, verificare `TCN_PLAN_COMMIT` (40 esadecimali), `TCN_PLAN_REPO_PATH` (relativo,
  senza `..`) e l'hash del piano letto con `git -C "$WORK/wearusfm" show`; in `commands`, pretendere 40 esadecimali. Facoltativo
  ma utile, insieme al suffisso degli ID come parametro del piano (L3-9): accettare un checkout già esistente, pulito, in sola
  lettura ed esattamente a `CAMPAIGN_SHA`, così che un preflight fallito per un guasto registrato si possa ripetere, dopo una
  decisione registrata, senza cambiare i sorgenti.

**I-6. Il gate della quota WORK usa la misura del rilascio per tutte le fasi** (`runner.py:119`, `__main__.py:100`; rilievi
statico S-2 e L3-5).
- `_resource_projection` legge `plan["operator_checks"]`, scritto una sola volta da `release-plan` prima del preflight.
  `resource-plan` non accetta una misura nuova, e ogni stadio successivo ricalcola il gate da quella fotografia e la confronta
  con il gate pinnato.
- Conseguenza: fra rilascio e `resource-plan` passano preflight, benchmark e le attese in coda, anche ore o giorni, mentre altre
  linee scrivono sulla stessa quota condivisa da 1 TB, che ha già dato un EDQUOT reale il 23/09. Il gate rilascia WLS e circa
  20 GPU-ore di fit che scrivono fino a 8 GiB, e le ricevute riportano `gates.resources = True` come se la misura fosse attuale.
  L'operatore rilegge la quota a mano prima di ogni fase, come chiede P:202, ma la misura non entra nel piano.
- Perché importante e non minore, contro il giudizio del controllore della lente 3: il codice è coerente con P:202, ma il gate e le
  ricevute affermano una misura attuale che non lo è, su una quota che ha già dato un EDQUOT, e la correzione è piccola.
- Correzione: `--operator-checks` anche in `resource-plan`, con `observed_at` successivo alle ricevute dei due benchmark, legato al
  piano misurato prima di calcolare il gate; il passo 6 del documento d'esecuzione va aggiornato di conseguenza.

## 4. Minori

Da correggere nella stessa revisione se costano poco; altrimenti basta una riga nel documento d'esecuzione. Fa eccezione L5-5:
il test della catena, esteso come nella sezione 7, è richiesto per chiudere.

| ID | Che cosa | Correzione |
| --- | --- | --- |
| L1-2 | Gli hash iniziali delle traiettorie si calcolano prima di ogni traiettoria ma si salvano solo a fine stadio (J: «save init state hashes … before first fit») | Stamparli con l'evento al passo zero (vedi B-4) e, per i rifit, in `_run_refits` prima di `train_to_step` |
| L3-4 (incerto) | Il pin numerico `cpu8` include il modello di CPU; la corsia seriale ha due nodi | L'operatore legge il modello di CPU dei due nodi prima del rilascio; se diversi, pin per classe di nodo |
| L3-6 | `release-plan` non verifica che la review nomini lo SHA rivisto | Richiedere lo SHA completo, o `source_sha256`, nel testo della review |
| L3-7 | Per gli stadi dopo il preflight sbatch esegue la copia locale del launcher letta da stdin, non quella congelata | Un wrapper su stdin che verifica `EXPECTED_SHA` ed esegue il launcher del checkout |
| L3-8 | Un rinfresco di SCRATCH durante uno stadio fa scattare le guardie sullo stato dei file | Nel documento: nessun rinfresco mentre gira uno stadio TCNH; fra gli stadi è innocuo |
| L3-9 | Ogni retry autorizzato richiede un cambio dei sorgenti e il riavvio di tutta la catena | Dirlo nella sezione dei fallimenti; facoltativo: suffisso degli ID come parametro del piano |
| L4-4 | La proiezione del rapporto omette l'I/O della fase metrics e due valutazioni R-1, e misura `evaluate` a batch 64 invece di 32 | Aggiungere quei termini e misurare a batch 32 |
| L4-5 | La lettura misurata degli input è una seconda lettura a cache calda | Misurare il primo caricamento, a freddo |
| L5-2 | La condizione necessaria E_dyn ≤ 0,7630564046643697 non è verificata né registrata (è implicata dal contrasto 5, salvo una finestra di al massimo circa 1e-12, la tolleranza di R-1, dovuta all'ordine di somma) | Aggiungerla alla candidatura e registrarne valore e soglia |
| L5-3 | La seconda parità R-1, prima dei contrasti, non lascia traccia | Salvarne il risultato nel rapporto o nella ricevuta |
| L5-5 | Nessun test fa girare con successo `locked_inference`, `metrics` e `collection` attraverso il runner | Richiesto: il test della catena della sezione 7 |

Confutato: L5-4. `al_bordo_MC` con 1,9 e 2,1 è corretto: i double più vicini stanno davvero fuori dalla banda inclusiva.

## 5. Che cosa regge

Controllato e conforme a P e J, con le misure dei revisori:
- **Architetture e numerica.**
  - Parametri: L0 8.468, MLP 73.000, L13 100.628, TCN-post e TCN-pre 72.904.
  - Campo ricettivo di 13 patch (−6…+6) e crop6 a 148 uscite; ramo affine diretto sulla stessa LN, a zero.
  - Kaiming fan_in negli hidden, ultimo strato di g a zero.
  - Flussi RNG 100000+s, 200000+100f+s e 300000+100f+s, con fold 2 nel rifit; ordine dei batch identico fra teste e LR.
  - Solo TCN-pre legge `pre`.
  - Loss per finestra e poi media delle finestre supportate, regola del supporto.
  - FP16 sull'host con cast FP32 per batch.
  - AdamW con decay solo sulle matrici, clip 1, batch 32 con l'ultimo conservato; flag deterministici, TF32 spento, thread 8/1.
  - Nessun sorgente del FM o della TEMPONet raw toccato.
- **Ottimizzazione e controlli.**
  - Fold alterni 20/20 e sottofold 10/10.
  - Checkpoint appaiati con deduplica; passo 0 non selezionabile; curve pooled come rapporto di somme.
  - Selezione e pareggi, continuazione 150→300 una volta sola con optimizer e RNG conservati, estensioni dei LR.
  - Allarmi come P:69, compreso lo 0,5% sul bordo inferiore; S-2 come compromesso.
  - Soglie interne solo dal lato fitting, `y*` solo per `pose_var` e soglia.
  - Ridge e WLS riprodotte a 12 cifre da una soluzione indipendente; griglia di 1.936 combinazioni.
  - Regola delle coppie e spareggi, gate lineari 90/90 e temporale 50%, ordine degli stati.
  - Generatore e rumore AR(1) riprodotti; isolamento degli held.
  - Nessun dato di sviluppo prima del blocco dei 15 rifit.
- **Provenienza.**
  - `prepare-plan` rigenera `plan.json` byte per byte; otto input pinnati; ricevute storiche identiche a `2930d7d`.
  - Costanti e campi di R-1, con il confronto dopo il cast a float64 sul supporto, prima dei fit e prima dei contrasti.
  - Scritture esclusive senza sovrascrittura; catena delle dipendenze ricontrollata dalle ricevute, mai dal solo afterok.
  - 24 job con ID e uscite unici; identità dei sorgenti contro lo SHA rivisto.
  - `commands` stampa soltanto righe per `submit_job.py`; uscite fuori dal repository; nessun monitor in background.
- **Risorse.**
  - Conteggi della matrice: 147 traiettorie, 40.128 valutazioni, 37 diagnostici, 537.600 / 1.477.800 update.
  - Fattore 2 e 180 s una volta per job; tetti di fase e globali.
  - 24 allocazioni, 18 GPU e 6 CPU-only.
  - Walltime nulli finché non sono misurati; benchmark solo sul train.
  - L13 a dimensione piena e tutti i pattern; memoria con chiusura in sicurezza prima dell'allocazione.
- **Lettura.**
  - E_dyn come rapporto di somme sui tempi validi; aggregazione dei semi sugli accumulatori.
  - Cinque contrasti; bootstrap con un'unica matrice di estrazioni (2.000, seme 20261102).
  - Priorità delle etichette, candidatura (salvo la condizione puntuale E_dyn ≤ 0,7630564046643697 non verificata, L5-2), caso
    misto, S-2 e allarmi direzionali (16 combinazioni tabulate).
  - Descrittivi, MAE e livello come candidato − comparatore; tutti i giunti.
  - Celle incomplete → `technical_failure` senza contrasti; nessun retry automatico.

## 6. Che cosa non ho verificato

- Ambiente di Leonardo: moduli, venv, nodi e modelli di CPU. B-1 viene da un log diagnostico del 06/11 sullo stesso modulo e lo
  stesso venv, non da una prova di oggi.
- CUDA e A100: tempi, determinismo e memoria li misura il benchmark.
- Dati veri: nessuna cache aperta, nessun gate R-1 eseguito.
- La review matematica indipendente citata da GPT: non l'ho rifatta. I revisori hanno riprodotto ridge, WLS, accumulatori e
  bootstrap per conto proprio.
- Il fallimento storico: riprodotto sulla consegna, non sulla base. Su `b4b0963` lo stesso test passa; oggi `main` è `f546c9e4`, con
  lo stesso file del test. Il file del test sul ramo di GPT differisce di 9 righe da quello di `main`, quindi il fallimento
  dipende dalla base del ramo, non dal lettore, e dovrebbe sparire con l'integrazione su `main`.

## 7. Come si chiude

1. GPT corregge B-1...B-4 e I-1...I-6 in una revisione, ciascuno con il suo test di regressione, insieme alle minori che costano
   poco. Il test della catena finale (L5-5) va esteso: deve far girare anche controlli, selezioni e rifit attraverso
   `run_stage`, senza sostituire `run_selection` o `run_refits`, con CUDA simulata su CPU. Per `qualify_controls` serve una fixture
   in cui i controlli passano davvero, oppure se ne forza solo l'esito dichiarandolo, come nella prova (con 4 epoche dà
   «ottimizzazione non risolta»). È il test che avrebbe visto B-4. Il driver della prova (`smoke/test_e2e_chain.py` nelle prove) può servire da punto di partenza.
2. Rifà pin e test e pubblica un nuovo SHA, ancora REVIEW_REQUIRED.
3. Rileggo solo il diff: cerco i quattro blocchi chiusi, i sei importanti chiusi e il test della catena verde. Niente agenti,
   niente nuovi rilievi sulle parti che reggono.
4. Se regge, la review è favorevole e l'operatore procede come nel documento d'esecuzione.

## 8. Prompt per GPT

```text
Review operativa Claude del pacchetto TCN frozen. SHA rivisto 2f982114719dc2abd723bdbf0147d7fef34a731c (codice 6283b5bd,
base 0e56c8b1): **DA RIVEDERE prima del primo invio.**
Review: docs/reviews/claude_review_pose_temponet_runtime_20261119.md, ramo claude/temponet-db8-20261109 (commit nel messaggio
di Simone). Prove, script, driver della catena e risultati in claude_review_pose_temponet_runtime_20261119_evidence/.

Test riprodotti: 204 nuovi passati; storici 125 passati e 1 fallito, lo stesso di baseline_failure.json (su main il test passa);
suite completa 2.569 passati in tutto, compresi i 6 del test DDP lanciato da solo, e lo stesso fallimento. Con B-4 corretto e B-2
aggirato in una copia, la catena dei 24 stadi gira fino in fondo su CPU con le funzioni vere degli stadi; preflight, benchmark,
cache e predizioni storiche erano simulati, il gate dei controlli forzato (con 4 epoche dà optimization_unresolved) e la parità R-1
saltata nei valori e nei conteggi.

Quattro blocchi, tutti di runtime, nessuno cambia la ricetta:
B-1 pose_temponet.sbatch:14 sostituisce PYTHONPATH dopo module load cineca-ai, che su Leonardo fornisce numpy e torch (log del
    06/11 in f58f5ff). Anteponi: export PYTHONPATH="$CODE_DIR/src:$CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}".
B-2 resources.py:93 dà walltime in secondi; Slurm arrotonda al minuto e runner._allocation (:163-165) pretende la stringa esatta,
    quindi, dopo preflight, benchmark e WLS, i sei job dei controlli rifiutano l'allocazione (e così farebbe ognuno dei 16). Arrotonda per eccesso al minuto in
    resources._job, richiedi seconds % 60 == 0 in validate_graph e apply_training_pack, test fino ad _allocation.
B-3 execution.py:168: --agente gpt_tcn_launcher non è fra le scelte di submit_job.py (claude, gpt, simone). Usa gpt, con un
    test contro le scelte del parser.
B-4 runner._progress (:185-189) chiama pair.curve(), che valida una curva completa e solleva «incomplete candidate curve» al
    passo zero: i sei job dei controlli finiscono con technical_failure e le selezioni non partono. Leggi pair.points; test con
    on_checkpoint=runner._progress; stampa anche gli hash iniziali, e per i rifit in _run_refits.

Importanti, nella stessa revisione (dopo il primo invio ogni cambio dei sorgenti invalida tutte le ricevute):
I-1 un gate in forma chiusa non diagnostico diventa technical_failure dentro i sei job GPU dei controlli e non viene salvato.
    GateNotPassed con lo stato vero e artefatto del gate combinato, meglio se deciso prima di rilasciare i job GPU; stessa cosa
    per runner.py:286 e controls.py:97.
I-2 il benchmark GPU misura una valutazione diversa da quella dei fit: misura workflow._score_model/D.view_inputs (P §9), oppure
    fai usare alla produzione il cast per batch del benchmark, in ordine fisico e senza scartare finestre.
I-3 benchmark: salvataggio e identity riscaldati prima dei 10 campioni; misura ciò che la produzione scrive davvero.
I-4 diagnostico L13 del margine temporale: fit su y* come E_cleanLS, valutato su y; test che L0 dia 0.
I-5 il preflight crea il checkout di campagna prima di verificare commit e hash del piano, e commands accetta riferimenti corti:
    pretendi 40 esadecimali in commands e verifica tutto nel wrapper prima del checkout; facoltativo, insieme al suffisso degli ID
    come parametro del piano (L3-9), accetta un checkout esistente pulito esattamente a CAMPAIGN_SHA.
I-6 il gate della quota WORK usa la misura del rilascio: --operator-checks anche in resource-plan, con observed_at successivo
    ai benchmark.
Minori nella tabella della sezione 4: correggile se costano poco, altrimenti una riga nel documento d'esecuzione.

Estendi il test della catena finale (L5-5), richiesto per chiudere, a controlli, selezioni e rifit attraverso run_stage, senza
sostituire run_selection o run_refits, con CUDA simulata su CPU; per qualify_controls una fixture in cui i controlli passano
davvero, oppure un esito forzato e dichiarato. È il test che avrebbe visto B-4. Il driver smoke/test_e2e_chain.py
nelle prove può servire da punto di partenza.

Poi pin e nuovo SHA, sempre REVIEW_REQUIRED; rileggo solo il diff, senza agenti. Se un rilievo è sbagliato, scrivilo nel
registro con la prova. Nessun lancio da parte mia; l'operatore resta 2-TCN head launcher.
```
