# Review di Claude — disegno del protocollo A (efficienza delle etichette su EPN), 07/11/2026

**Oggetto.** Disegno pubblicato da GPT consolidamento sul ramo `codex/consolidation-20261104`, commit
`cfa6462f261fa7e70afb9dc13684d4436756ad65`: `docs/protocol_a_epn_label_efficiency_20261107.md`,
`docs/reviews/protocol_a_implementation_20261107.md`, `results/infra/epn_label_efficiency_20261107/`
(`proposal.json`, `resource_estimate.json`, `population_counts.json`, `static_verification.json`, `design_review.txt`).
Richiesta: `docs/prompt_claude_protocol_a_20261107.md`, stesso commit. Stato dichiarato da GPT:
`awaiting_claude_design_review`, `executable: false`.

**Perimetro.** Review del solo disegno. Nessun benchmark, fit, deployment, checkout o scrittura su Leonardo.
Una sola lettura remota di sola lettura (`saldo -b`, 07/11 17:57) per il punto 5. L'operatore della linea 2
resta Claude; la responsabilita' scientifica del protocollo resta a GPT consolidamento (Simone, 07/11: «A GPT»).

**Metodo.** Rilettura del codice nel checkout comune (`src/wearusfm/harness/finetuning.py`,
`src/wearusfm/harness/backbone_depth.py`, `scripts/consolidation_finetune15.py`,
`scripts/consolidation_long_classification.py`, `scripts/calibration_probe.py`, `scripts/tokenizer_probe.py`);
delle 16 ricevute dei candidati FT/scratch di EPN15 (`results/infra/finetune15_20261105/final/fit_reports/`),
del consuntivo EPN15 (`final/accounting.json`) e delle misure del benchmark EPN15; delle ricevute e del `sacct`
dei fit CPU della valutazione lunga su EPN612 (`results/infra/consolidation_long_20261105/execution/`).
Tutti i conti sono rifatti in modo indipendente (sezione 8). Dove scrivo «lettura mia» e' un'interpretazione,
non un fatto verificato.

## 0. Esito in breve

| Punto | Verdetto | In una riga |
|---|---|---|
| 1 Domanda, popolazione, budget di etichette | **OK con precisazioni** | Conteggi riprodotti; dichiarare che il primario media sui quattro sorteggi del subset |
| 2 Dose e selezione | **DA DECIDERE prima dell'implementazione** | Raccomando la dose costante (A) con le 15 letture posizionate per frazione; variante C se il budget e' vincolante; 15 epoche (B) sconsigliata come primario |
| 3 Primario | **OK, da congelare** | Pesi del trapezio 0,153/0,347/0,347/0,153; margine 1 pp nuovo: in `decisioni.md` prima dei risultati |
| 4 Riuso | **OK condizionato, ponte da scrivere** | Frozen100 riusabile dal lungo; scratch100 riusabile con ponte di configurazione e test di parita', non con 8 fit nuovi |
| 5 Conti | **Conteggi esatti; proiezioni non preventivi** | 766,22 GPU-ore riprodotte (A); B sottostimata (250-255 contro 239,79); CPU allocata ~8 core-ore per GPU-ora |
| 6 Arresti e limiti | **OK, da completare** | Allocazioni, limiti per frazione, confezionamento, calendario delle manutenzioni, spazio e conservazione |

**Pronta a diventare un piano da sottoporre a Simone? Non ancora.** Il disegno e' coerente e non ha difetti che
impongano di rifarlo; mancano quattro decisioni (sezione 7, B-1...B-4): la politica della dose con il suo budget,
il ponte del riuso scritto nel protocollo, il congelamento in `decisioni.md` di primario, margine, letture e numero
di realizzazioni, e la proiezione dei costi rifatta come preventivo. Dopo quelle, la sequenza e' quella scritta da
GPT nella sezione 7 del protocollo: implementazione e test, benchmark con il subset minimo, preventivo completo,
review del codice prima del pin e del deployment, come per la catena lunga.

## 1. Domanda, popolazione e budget di etichette — OK con precisazioni

Verificato:

- Popolazione: 314 soggetti, sei classi, 25 finestre per soggetto e classe, 47.100 finestre; ruoli EPN15
  (`CP.fold_roles`: 4 fold, seme 20261015, validation 10 % per eccesso) con train 211/211/212/212, validation 24,
  held 79/79/78/78. Gli held dei quattro fold partizionano i 314 soggetti (79+79+78+78 = 314): le OOF aggregate
  coprono ogni soggetto una volta, come nel lungo e in EPN15.
- Prefissi `ceil(frazione x N_train)`: 11/22/53/106 sia per 211 sia per 212; finestre 1.650/3.300/7.950/15.900.
  L'annidamento e' garantito dalla costruzione (prefissi di un unico ordinamento per fold).
- Budget di etichette al «5 %»: (11+24)/(211+24) = 14,89 % dei soggetti, identico in finestre (150 per soggetto).
  La proposta lo dichiara: va ripetuto in ogni tabella e figura.
- Ogni soggetto porta 25 finestre per classe: ogni subset e' esattamente bilanciato e i pesi di classe di
  `encode_labels` sul solo train selezionato risultano 1. Va verificato dal manifest, come dice GPT, non assunto.
- Calibrazione: `PreparedWindows` usa le prime 10 finestre di ciascun soggetto, senza statistiche fra soggetti.
  I soggetti train esclusi non contaminano nulla; il test proposto da GPT (perturbare le righe escluse e verificare
  input e gradienti invariati delle righe attive) e' quello giusto.
- `F.role_indices` pretende che train, validation e held coprano tutti i soggetti passati: serve la mappa «dati
  attivi = train selezionato + validation + held» con le righe originali nel manifest (nota GPT, confermata nel codice).

Precisazioni (nessun cambio di disegno):

- L'hash di selezione include il fold: i quattro fold sorteggiano quattro subset diversi di 11 soggetti. Il primario,
  calcolato sulle OOF aggregate, media quindi su quattro sorteggi del train piccolo. E' un'attenuazione parziale del
  limite «una sola realizzazione», da scrivere nel protocollo; non sostituisce piu' realizzazioni, che moltiplicherebbero
  il costo.
- La domanda e' formulata «mantenendo uguale la dose di ottimizzazione»: lega il protocollo alla politica A. Se Simone
  sceglie B o C, la frase va riscritta (sezione 2).
- «Nessun warm start tra frazioni» e «stessi soggetti fra FT, scratch, frozen e candidati LR» sono necessari
  all'appaiamento e vanno mantenuti.

## 2. Dose e selezione — DA DECIDERE; raccomandazione: dose costante con letture posizionate per frazione

Verificato nel codice e nelle ricevute:

- `fit_candidate`: dose = epoche x `ceil(n_train/8)`; `max_updates` entra solo come `min(...)`: e' un tetto, non
  prolunga il train piccolo. Il ciclo a dose costante e' un percorso nuovo (nota GPT, confermata). Warmup
  `min(100, dose//10)`: vale 100 anche a 3.105 aggiornamenti. Coseno sull'intero orizzonte. Permutazione nuova a ogni
  epoca dallo stesso generatore (`default_rng(seed)`), ultimo batch parziale conservato, validation dopo ogni epoca
  con `predict` (no_grad, batch 8), pareggio alla prima epoca con `>` stretto.
- EPN15, 16 candidati FT/scratch: `planned_updates` = `updates` = 59.355 (fold 0/1) o 59.625 (fold 2/3),
  15 letture ai multipli di 3.957 o 3.975, `stopped = fixed_budget`, held una volta per cella nel secondo job.
- Passate sul subset con la dose costante (fold 0/1, U = 59.355): 5 % 286,7; 10 % 143,7; 25 % 59,7; 50 % 29,9;
  100 % 15. Prima lettura (3.957 aggiornamenti) dopo 19,1 passate al 5 %, 9,6 al 10 %, 4,0 al 25 %, 2,0 al 50 %,
  1,0 al 100 %. Fold 2/3: stessi valori entro lo 0,5 %.
- Coppie LR: FT 1e-5/1e-3 e 1e-4/1e-2 (testa x100), scratch 1e-4/1e-3 e 1e-3/1e-2 (testa x10): coincidono con
  `LR_GRID` e `HEAD_LR_MULTIPLIER` di `finetuning.py`.

Lettura delle tre politiche (A e B di GPT; C e' una mia variante):

- **A, dose costante per fold (raccomandata da GPT).** Misura «stessi passi di gradiente». Al 5 % la cella vive per
  centinaia di passate su 1.650 finestre con LR vicino al picco per gran parte dell'orizzonte. Le letture a
  `j x U/15` vedono il subset del 5 % solo dalla passata 19 (10 %: dalla 9,6): uno stato migliore precedente resta
  invisibile alla selezione. Vale allo stesso modo per FT e scratch dentro la frazione, quindi il contrasto resta equo,
  ma il livello della curva alle frazioni piccole e' «il migliore di 15 istantanee rade», con densita' di letture per
  passata che fra frazioni varia fino a 19 volte. Lettura mia: con il coseno la lettura selezionata cadra' spesso nella
  fase finale di annealing, quindi l'effetto pratico puo' essere piccolo; non e' pero' dimostrato.
  Proposta M-1: tenere 15 letture per cella, ma posizionarle per frazione su una griglia fissata prima dei risultati
  (per esempio spaziata in log degli aggiornamenti dalla prima passata a U), uguale per FT e scratch dentro la
  frazione; 100 % invariato (= storico). Costo nullo: 15 letture in ogni caso. Se GPT preferisce `j x U/15` per la
  parita' di orizzonte, e' accettabile con il limite dichiarato.
- **B, 15 epoche per frazione.** Al 5 % scratch riceve 3.105 aggiornamenti da inizializzazione casuale. Indizio
  debole da EPN15 al 100 %: epoca migliore media 9,9 per gli 8 candidati FT (5, 12, 11, 6, 8, 13, 12, 12) e 12,25
  per gli 8 scratch (7, 13, 10, 13, 15, 14, 12, 14): scratch tende a servirsi di piu' orizzonte. Lettura mia: con
  pochi aggiornamenti il regime sottoaddestrato e' piu' probabilmente scratch e `Delta_L` ne risulta gonfiato a favore
  di FT. B risponde a «con la ricetta EPN15 a 15 epoche, quanto aiuta il teacher con pochi soggetti?»: domanda
  legittima ma diversa, da non presentare come dose uguale. Costo sottostimato dalla proiezione lineare (sezione 5).
- **C, dose costante ridotta (variante mia).** Dose U* = 29.820 (le 15 epoche del 50 %, la frazione ridotta piu'
  grande) per le quattro frazioni ridotte in entrambi i regimi, coseno su U*, 15 letture; celle al 100 % alla dose
  storica (scratch100 riusato, FT 49k al 100 % nuovo a U_fold, necessario al confronto con FT 19k storico). La
  proprieta' «dose uguale fra i quattro punti del primario» e' conservata; il riferimento al 100 % e' a dose doppia,
  dichiarata e secondaria. Al 5 %: 144 passate, prima lettura dopo 9,6 passate con `j x U*/15`. Costo ~435 GPU-ore
  (sezione 8), 43 % meno di A.

Raccomandazione: **A** con la proposta M-1, se Simone accetta ~770-800 GPU-ore di soli fit; altrimenti **C**. **B**
solo con la domanda riscritta e la direzione del bias dichiarata, non come primario a dose uguale. In ogni caso la
politica, la griglia delle letture e la dose vanno scritte in `decisioni.md` prima dell'implementazione, non dopo
un esito.

Regole di selezione: conformi a EPN15 e corrette. Migliore balanced accuracy di validation, pareggio al passo piu'
precoce; candidato per sola validation, pareggio al candidato 0; held una volta per cella dopo entrambi i candidati;
nessuna estensione se vince un estremo; tutti i candidati completano la dose. Il confezionamento EPN15
(`split_only`: un job per candidato, il secondo carica la ricevuta del primo, nessun fallback sul held) e' riusabile.
«Senza riavviare optimizer, schedule o RNG» fra le passate e' necessario: il ciclo nuovo deve attingere permutazioni
dallo stesso generatore, come fa il ciclo per epoche.

## 3. Primario — OK, da congelare

- Trapezio su ln(n) con n = (11, 22, 53, 106), normalizzato per ln(106/11) = 2,2655: equivale alla media pesata di
  S nei quattro punti con pesi **0,153 / 0,347 / 0,347 / 0,153** (somma 1). Vanno scritti nel protocollo: il 10 % e
  il 25 % portano il 69 % del primario. L'unita' resta il punto percentuale di balanced accuracy.
- Appaiamento: ruoli fissi, stessi held in ogni cella e regime; bootstrap appaiato per soggetto held sulle OOF
  aggregate (314 soggetti), 2.000 repliche, seme 20261102, percentile al 95 %: stessi parametri del lettore del lungo
  (`tokenizer_probe.py`: `SEED` 20261102, `N_BOOT` 2000). Area e contrasti ricalcolati dentro ogni replica, non media
  dei fold: corretto.
- Lettura a due lati e margine pratico 1 pp: accettabili. Il margine e' nuovo (non V-R1) e va congelato in
  `decisioni.md` con la dicitura «risolto ma piccolo» prima di qualunque fit. Il 100 % come secondario, nessuna
  interpolazione per annunciare etichette risparmiate: giusto.
- Gli IC condizionano su un sorteggio del subset per fold e sul seme 0: dichiarato da GPT; aggiungere la frase
  della sezione 1 sui quattro sorteggi.
- Il confronto storico FT 49k 100 % − FT 19k 100 % e' comparabile al livello della configurazione (stessi ruoli,
  dose, ricetta, held; cambia solo il checkpoint del teacher): passa dallo stesso ponte del punto 4.

## 4. Riuso — OK condizionato; il ponte va scritto nel protocollo

**Frozen100 dal lungo: riusabile.** Nel lettore lungo (`consolidation_long_classification.py`, `fit_fold`) lo
scaler e' addestrato sul solo train; griglia `C_GRID` {0,001; 0,01; 0,1; 1; 10; 100}, lbfgs, `class_weight`
balanced, `max_iter` 12.000, `tol` 1e-4; selezione sulla validation con pareggio al C minore
(`max(C_GRID, key=(score, -c))`); refit sul solo train; held una volta. Ruoli `CP.fold_roles` identici a EPN15. Vista
`x_pool` 768: `backbone_depth.extract_windows` media `pool_p1p2` sui due crop da 4 s, la stessa quantita' che la
testa `Classifier` (LayerNorm(2·dim) + Linear su `pool_p1p2`) vede nel fine-tuning: la corrispondenza pool 768 regge.
Cache `49046/p2/epn612.npz`, SHA256 `f7339e89...`, 294.936.151 byte, copiata e verificata su `$WORK` il 07/11.
Fit storico: `LONG-FIT-G49046-pool-epn612-F0123-20261105-A1`, job 63751328, `COMPLETED 0:0`, 1.783 s su 8 CPU,
RSS di picco 3,46 GiB. Condizioni: l'adapter nuovo riproduce la ricetta per le frazioni ridotte, non usa `fit_fold`
(che pretende la partizione completa), e il manifest cita la ricevuta e la corrispondenza numerica storica; contesto
numerico (8 thread, stesse librerie) dichiarato.

**Scratch100 da EPN15: riusabile con ponte di configurazione, non con fit nuovi.** `build_classifier(..., 'scratch', ...)`
ignora i pesi del teacher: il passaggio da 19.618 a 49.046 non cambia lo scratch. Le celle scratch100 di EPN15
(input `ef0090cb...`, uguale al `data_input_sha256` della proposta; ruoli; seme 0; batch 8; 15 epoche = 59.355/59.625
aggiornamenti; validation a fine epoca = multipli di 3.957/3.975; pareggio alla prima epoca; coppie LR 1e-4/1e-3 e
1e-3/1e-2) sono **esattamente il calcolo della cella 100 %** sotto A, B e C: a 100 % una passata e' un'epoca e le
letture `j x U/15` cadono a fine epoca. Il ponte e' quindi un'identita' di configurazione e di percorso di codice:

1. elenco dei campi identici con hash: input, ruoli, calibrazione, config del modello, testa, pesi di classe, seme,
   optimizer, coppie LR, batch, warmup, orizzonte del coseno, U, passi delle letture, pareggi, regola di selezione;
2. test di parita' del controller nuovo a 100 % contro il ciclo EPN15, su CPU con modello ridotto: stessa sequenza di
   permutazioni, stessi valori di LR per passo, stessi passi di lettura, stessa selezione;
3. citazione delle otto ricevute (`candidate_0/1.json`, `report.json`) e delle quattro OOF selezionate con SHA, senza
   rinominare `source`, `recipe` o `checkpoint` negli artefatti storici; regola di pareggio fra candidati citata dal
   protocollo EPN15.

Cio' che il ponte non puo' fare: una riproduzione bit a bit. In BF16 su GPU con kernel non deterministici, rifare gli
otto fit (~85 GPU-ore, ~680 core-ore allocate) darebbe una **seconda realizzazione** di scratch100, non una verifica,
e imporrebbe di scegliere fra due realizzazioni. Raccomando il riuso con il ponte documentato; fit nuovi solo se il
ponte trova una differenza sostanziale, e allora e' un piano e un budget nuovi, come scrive GPT. Lo stesso ponte serve
al confronto storico FT 19k 100 %.

Adattamenti del controller confermati nel codice: controllo `step == 49046` e config della continuazione al posto di
`step == 19618` e `tokenizer_execution_20261103/19618/T100-G/config.json`; percorso di validation identico
(`predict`, batch 8); dati attivi per `role_indices`; `encode_labels` sul solo train selezionato.

## 5. Conti — conteggi esatti; le proiezioni non sono preventivi

Riprodotto: 20 celle FT (5 frazioni x 4 fold) x 2 = 40 fit; 16 celle scratch (4 frazioni ridotte x 4 fold) x 2 = 32
fit; 8 candidati scratch100 storici; 16 celle frozen nuove (6 C + 1 refit ciascuna) + 4 storiche. 72 fit GPU nuovi,
80 senza riuso.

Per fold (A, per candidato FT 10,6436 e scratch 10,6398 GPU-ore dal consuntivo EPN15):

| Fold | FT: 5 celle x 2 | Scratch: 4 celle x 2 | Totale fold |
|---|---:|---:|---:|
| 0 | 106,44 | 85,12 | 191,55 |
| 1 | 106,44 | 85,12 | 191,55 |
| 2 | 106,44 | 85,12 | 191,55 |
| 3 | 106,44 | 85,12 | 191,55 |
| Somma | 425,74 | 340,47 | **766,22** |

GPT usa un solo valore per candidato in tutti i fold: lecito, gli elapsed EPN15 dei 16 fit stanno fra 37.992 e
38.532 s (differenza massima 1,4 %).

Proiezione B riprodotta (lineare): 239,79 GPU-ore. Ma l'elapsed EPN15 contiene una parte fissa: sottraendo
0,629 s per aggiornamento (benchmark, regime stazionario) restano **743-1.177 s per fit** (media FT 880, scratch 925):
caricamento (257 s), preprocessing (268 s), 15 letture, salvataggi, predizione held. Con quella parte fissa:

| Politica | Fit al 5 % | 72 fit GPU | Core-ore allocate (8 CPU per GPU) |
|---|---:|---:|---:|
| A, dose costante per fold | 10,64 | **766,2** | ~6.130 |
| B, 15 epoche, lineare (GPT) | 0,557 | 239,8 | ~1.920 |
| B, 15 epoche, con parte fissa 743-1.177 s | 0,76-0,85 (+36-53 %) | **250-255** (+4-6 %) | ~2.000-2.040 |
| C, dose 29.820 alle frazioni ridotte | 5,48 | **~435** | ~3.480 |

Altri costi, non nulli: la convenzione del consuntivo EPN15 conta le core-ore allocate come elapsed x CPU allocate
(85,0 per fit FT: 340,08 per i quattro FT C0). Frozen CPU: il fit lungo 49.046/pool su EPN612 e' costato 3,96 core-ore
per i quattro fold al 100 %; le 16 celle ridotte restano sotto le 16 core-ore. Benchmark EPN15: 0,126 GPU-ore e
1,0 core-ore; due checkout: 0,03 core-ore. L'aggregazione e' CPU per minuti. Nel complesso, fuori dai fit, pochi
percento di A; vanno comunque nel preventivo. Nessuna delle proiezioni include margini di allocazione: in EPN15 il
limite per fit era 57.300 s dentro 57.600 s per ~38.300 s usati.

Residuo di budget letto il 07/11 alle 17:57 (`saldo -b`): IscrB_WearUsFM 2.000.000 ore locali, consumate 1.305.624
(65,3 %), residuo 694.376 fino al 07/01/2027; novembre 7.325 su 164.383. Il peso di fatturazione letto da `sacct` (`AllocTRES`) per un job a 1 GPU e 8 CPU e' `billing=8` (job 63579657 e 63581664): 8 ore locali per ora di job, cioe' 8 ore locali per GPU-ora di questi fit. A vale quindi ~6.130 ore locali (0,9 % del residuo), B ~2.000-2.040 (0,3 %), C ~3.480 (0,5 %). La colonna mensile (164.383 totali, 7.325 consumate in novembre), se e' un tetto, non vincola nessuna delle tre.

Calendario: EPN15 ha speso 197 GPU-ore di fit in circa un giorno con piu' job in parallelo; A vale 3,9 volte quel
volume, con le manutenzioni del 10-14/11 (parziale, Booster) e del 23-25/11 (tutto Leonardo, job in attesa). Il piano
deve dirlo.

## 6. Arresti e limiti — OK, da completare

Conformi a EPN15 e al submitter comune: divergenza numerica persistita, candidato non selezionabile, nessuna
predizione da checkpoint non finito, cella non valutabile senza candidati completi; OOM, TIMEOUT e guasti fermano
la fase senza retry o aumento di limite automatico; cancello globale prima del primo invio; nessun primario su
matrice incompleta, artefatti conservati senza cambiare fold, frazioni o supporto.

Da aggiungere al piano eseguibile:

- allocazione per fit GPU (EPN15: 1 GPU, 8 CPU, 100 GB) e limiti per frazione: per A gli stessi di EPN15
  (57.300 s in 57.600 s); per B e C dal benchmark con la formula EPN15 `ceil(3 x M x 1,25 x q + 600 + D)` adattata
  agli aggiornamenti misurati; confezionamento un job per candidato, 300 s riservati;
- frozen CPU in corsia seriale (8 CPU, 30.800 MB, 4 h, 2 job in esecuzione e 10 sottomessi per utente): il fit lungo
  su EPN612 e' stato entro 3,46 GiB e 1.783 s, quindi compatibile;
- radice delle uscite (`$WORK` o `$SCRATCH`), conservazione dei checkpoint dei candidati (72 nuovi, dimensione da
  misurare), spazio in `$WORK` (17,87 GiB liberi alle 16:34 del 07/11, soglia C-1 10,724 GiB) e cancellazione a
  40 giorni su `$SCRATCH`; ricevute, report e manifest raccolti sul Mac e committati;
- calendario delle manutenzioni (sezione 5) e un solo operatore (Claude), ledger comune, ID logici unici, nessun
  rilancio alla cieca.

## 7. Rilievi

Bloccanti per il passaggio a piano eseguibile (non difetti del disegno):

- **B-1** Politica della dose non fissata (A, B o C) con il budget corrispondente: senza, non esistono costo, limiti
  e domanda definitiva. Se non e' A, la domanda «mantenendo uguale la dose di ottimizzazione» va riscritta.
- **B-2** Le proiezioni non sono preventivi: B e' lineare e sottostima i fit piccoli (+36-53 % al 5 %, +4-6 % sul
  totale); mancano CPU allocata, frozen CPU, benchmark, checkout, aggregazione, limiti per frazione, margini e
  calendario. Serve il benchmark con il subset minimo e il costo delle letture separato.
- **B-3** Il riuso di scratch100 e frozen100 e' «condizionato» ma il ponte non e' definito: va scritto nel protocollo
  come identita' di configurazione e di percorso di codice con test di parita' a 100 % e citazione delle ricevute
  (sezione 4). Un rifacimento bit a bit e' impossibile; 8 fit nuovi non sono una verifica.
- **B-4** Da congelare in `decisioni.md` prima di qualunque fit: primario con i pesi espliciti, margine 1 pp, lettura
  a due lati, griglia delle 15 letture per frazione, una sola realizzazione del subset per fold, seme 0, politica
  della dose.

Minori:

- **M-1** Posizione delle letture con A (sezione 2): griglia per frazione fissata prima, uguale per FT e scratch.
- **M-2** Dichiarare che il primario media sui quattro sorteggi del subset (uno per fold).
- **M-3** Adattamenti del controller gia' elencati da GPT e confermati nel codice (sezione 4).
- **M-4** Registrare per cella esempi elaborati, epoche reali e numero di passate alla lettura selezionata.
- **M-5** Il «5 %» sempre con i due conteggi (11 train; 35 train+validation su 235, 14,89 %).
- **M-6** Nessuna misura separata del costo di una lettura di validation: il benchmark deve separarla.

## 8. Conti rifatti

- Prefissi: `ceil(f x 211)` e `ceil(f x 212)` per f = 0,05/0,10/0,25/0,50: 11/22/53/106 in entrambi i casi.
- Aggiornamenti a 15 epoche: 15 x `ceil(finestre/8)`: 3.105, 6.195, 14.910, 29.820, 59.355 (211), 59.625 (212).
- Passate con U = 59.355: U / `ceil(finestre/8)` = 286,7 / 143,7 / 59,7 / 29,9 / 15,0; prima lettura
  3.957 / 207 = 19,1 passate al 5 %.
- Pesi del trapezio: ln 2 / (2 ln(106/11)) = 0,153 agli estremi; (ln 2 + ln(53/22)) / (2 ln(106/11)) = 0,347 in mezzo.
- A: 40 x 10,643611 + 32 x 10,639826 = 766,22. B lineare: per fold 0/1 2 x 10,6436 x 113.385/59.355 + 2 x 10,6398 x
  54.030/59.355 = 60,04; fold 2/3 con 59.625: 59,86; somma 239,79. B con parte fissa O: costo fit =
  (O + s x U_fit)/3.600 con s = (10,6436 x 3.600 − O)/U_fold; O = 760/960/1.115 s: 250,2/253,0/255,1.
  C: 64 x (960 + s x 29.820)/3.600 + 8 x 10,6436 = 435,1.
- Parte fissa per fit: elapsed EPN15 − 0,629 x U (FT) o − 0,628 x U (scratch): FT 743-1.028 s, scratch 717-1.177 s.
- Core-ore allocate: elapsed x 8: 85,1 per fit a dose piena; 72 fit = 6.128.
- Frozen lungo EPN612 49.046/pool: 1.783 s x 8 CPU / 3.600 = 3,96 core-ore.

## 9. Limiti di questa review

Nessun job e nessuna esecuzione di codice del protocollo (non esiste ancora). I costi sono proiezioni dagli elapsed
EPN15, non misure della ricetta nuova. L'uguaglianza fra la config della continuazione 49.046 e quella di T100-G va
provata dai test dell'implementazione, non e' verificata qui. La regola di pareggio fra candidati di EPN15 va citata
dal protocollo EPN15 nel ponte. Le letture «mie» nella sezione 2 sono interpretazioni, non misure.
