# Review di Claude — risultati del fine-tuning EPN a cinque epoche (GPT db89202) — 05/11/2026

Richiesta: prompt di GPT `docs/prompt_claude_finetune_results_20261105.md`, incollato da Simone il 05/11. Review mirata di cinque
punti e parere sul seguito a 15 epoche. Solo review: nessun fit, nessuna inferenza held, nessuna sottomissione, nessuna modifica ai
runtime, alle soglie o ai candidati. Le mie review f70dc72 e 2d157cc sui lanci restano valide e non sono state rifatte.

**Verdetto: APPROVATO.** I cinque punti sono confermati con un ricalcolo indipendente; nessun difetto bloccante nell'aggregatore,
nei numeri o nella lettura. Quattro rilievi non bloccanti sui limiti da dichiarare (L1-L4, sezione 3) e un parere favorevole, con
condizioni da scrivere prima di qualunque lancio, sul seguito a 15 epoche dalle inizializzazioni originali (sezione 4).

## 1. Metodo

- Checkout temporaneo detached di db89202 (poi rimosso). Rispetto al runtime dei fit a2eeb75 il diff su `scripts/`, `src/` e `tests/`
  contiene solo i due file nuovi (`scripts/consolidation_ft_report.py`, `tests/cpu/test_consolidation_ft_report.py`);
  `finetune_plan.json` ha lo stesso hash 2ce4927f… da a2eeb75 a db89202.
- Test con il comando di GPT: **15 passati** (1,10 s). Aggregatore rieseguito sui dati reali in un output nuovo: file **identico
  byte per byte** a `final_20261105/report.json` (sha256 3c4b3601…). Gli hash di `verification.json` ricalcolati coincidono.
- Ricalcolo mio con codice scritto da zero, senza importare l'aggregatore: inventario e hash, provenienze, selezione, pairing,
  macrorecall, bootstrap (tabelle per soggetto costruite con un ciclo esplicito, piu' il ricampionamento letterale delle righe per le
  prime 25 repliche), F-R2 e consuntivo.
- Leonardo in sola lettura, 05/11 alle 10:58 e 11:01 CET: `sacct` dei 29 job e di tutti i job dal 04/11 18:00, `squeue`, elenco dei
  file con dimensioni e date, `sha256sum` dei 108 file sotto `fit_output_root`, conteggio delle righe dei 24 log. Output in
  `~/wearusfm_local/logs/consolidation_ft_results_leonardo_20261105.txt` e `…leonardo2_20261105.txt`.
- Regole congelate: i paragrafi del protocollo su metrica, bootstrap, contrasti e F-R2 sono identici byte per byte in 2dbd498
  (04/11 17:57, prima dei fit partiti alle 19:27), 17dd455, a2eeb75 e db89202. L'aggregatore e' scritto dopo i risultati ma applica
  parametri gia' fissati: non ho trovato gradi di liberta' usati dopo aver visto i dati.

## 2. Verifiche sui cinque punti

### 2.1 Provenienze, selezione, held e pairing — confermato

- **File.** 84 file locali = 84 voci del manifest, hash e dimensioni uguali. Su WORK ci sono 108 file (84 + 24 checkpoint): stesso
  insieme del manifest, nessun file in piu', nessun link. I miei hash remoti coincidono con il manifest per tutti i 108, e i 24 hash
  dei checkpoint coincidono con il `checkpoint_sha256` scritto dal runtime al salvataggio.
- **Identita' dei fit.** I 24 report hanno `code_sha` a2eeb75, `source_sha256` aa857fb3…, `plan_sha256` 2ce4927f…, ricetta
  2ba1b7f5…, checkpoint del teacher a31c8950…, `input_sha256`, fold e regime attesi. I ruoli sono quelli del riferimento storico
  pinnato (ef53b8ab…): train, validation e held disgiunti (211-212 / 24 / 78-79 soggetti), 314 soggetti held senza sovrapposizioni.
- **Selezione sulla sola validation.** Ricostruita dalle curve: epoca = primo massimo, LR = validation piu' alta con pareggio al
  candidato 0, che e' il LR minore perche' le griglie sono crescenti. Coincide con `selected_candidate`, `selected_lr` e
  `selected_best_epoch` dei 12 report C1. Nessun pareggio reale: i margini fra i due LR vanno da 1,6 a 13,0 pp.
- **C0 senza held.** I 12 report C0 hanno `held_evaluated: false` e nessun campo held; su WORK non esiste alcun `held.npz` sotto
  `candidate0`; nel codice il ramo `--candidate 0` non predice il held e `fit_candidate` usa solo `roles["val"]`.
- **C1 con una sola predizione held.** Un solo `held.npz` per cella (12 in tutto), con hash uguale a `held_sha256` del report e data
  di scrittura pochi secondi prima della fine del job C1. La directory di output non puo' preesistere; `sacct -D` mostra un solo
  record per job (nessun requeue); il ledger ha un solo `sottomesso` per ID; dal 04/11 18:00 non esistono altri job FT oltre i 29.
  I log hanno 14 righe ciascuno, senza errori e senza righe per epoca: l'unicita' poggia su codice, inventario, date e accounting.
- **Pairing.** `rows`, `y`, `s` e `fold` identici nei tre regimi; righe 0…47.099 senza buchi ne' duplicati; 314 soggetti, 6 classi,
  esattamente 25 finestre per soggetto e classe.
- **Dati.** Nel commit non ci sono OOF, pesi o segnali. `report.json` contiene conteggi aggregati per soggetto del dataset pubblico
  EPN-612, non predizioni per finestra.

| Regime | Scelto | LR encoder / testa | Validation C0 → C1 nei fold 0-3 (%) | Margine del scelto (pp) | Migliori epoche |
|---|---|---|---|---|---|
| frozen | C1 in 4/4 | fermo / 1e-2 | 68,8→71,6; 65,8→68,0; 67,5→70,6; 65,1→66,9 | da +1,8 a +3,1 | 5, 5, 5, 5 |
| finetune | C1 in 4/4 | 1e-4 / 1e-2 | 88,8→91,4; 91,3→92,8; 91,8→93,8; 88,8→91,8 | da +1,6 a +3,0 | 4, 4, 5, 5 |
| scratch | C0 in 4/4 | 1e-4 / 1e-3 | 89,7→77,5; 91,7→81,3; 92,7→82,3; 89,7→76,7 | da +10,4 a +13,0 | 4, 5, 5, 5 |

### 2.2 Macrorecall pooled e bootstrap — confermato

- **Punteggi.** Macrorecall sulle OOF concatenate, mia implementazione: frozen 70,575372%, FT 93,766454%, scratch 92,363057%;
  contrasti +23,191083 e +1,403397 pp. Uguali al report fino all'ultima cifra. Con 25 finestre per soggetto e classe la macrorecall
  pooled coincide con l'accuratezza semplice e con la media dei punteggi per soggetto.
- **Bootstrap.** 2.000 repliche, seed 20261102, un solo generatore, stessi soggetti estratti per i tre regimi. Riprodotto: IC dei tre
  regimi e dei due contrasti identici (differenza massima 0). La sequenza casuale non dipende dalla suddivisione in blocchi da 128 e il
  ricampionamento letterale delle righe da' gli stessi valori. Nessuna replica scartata ne' con classi mancanti: e' strutturale,
  perche' ogni soggetto ha tutte e sei le classi.
- **Condizionamento.** «IC condizionati a modelli e fold, un solo seme di modello» e' scritto correttamente nell'analisi, nel registro
  e in `decisioni.md`.
- **Controlli miei, descrittivi.** Non sostituiscono l'analisi congelata. Errore standard analitico appaiato per soggetto 0,237 pp
  contro 0,234 del bootstrap; IC normale [0,940; 1,867]; minimo delle 2.000 repliche +0,66 pp.

### 2.3 Lettura di FT−scratch +1,403 pp [0,930; 1,837] — corretta

- «Vantaggio alla dose fissata» e' la formulazione giusta ed e' quella usata in tutti i documenti toccati dal commit: analisi, README,
  registro, `decisioni.md`, piano scientifico, intestazione del protocollo. Non trovo affermazioni di convergenza ne' di vantaggio
  finale o universale.
- **Coerenza interna, descrittiva.** Per fold +1,67 / +1,11 / +1,14 / +1,69 pp, quattro su quattro positivi; sulla validation dei
  selezionati +1,72 / +1,11 / +1,11 / +2,11. Tutte e sei le classi migliorano, da +0,21 (noGesture) a +2,55 (open). Per soggetto: 176
  meglio, 36 pari, 102 peggio; mediana +0,67 pp, cioe' una finestra su 150; estremi −18,7 e +22,0. E' uno spostamento medio piccolo
  e diffuso, non l'effetto di pochi soggetti.
- **Trasferimento.** EPN-612 e' escluso dal pretraining (`src/wearusfm/data/manifest.py:5`, documento di riferimento §2.3): il
  vantaggio riguarda un dataset non visto, con la calibrazione non supervisionata per soggetto gia' dichiarata.
- Limiti da aggiungere alla lettura: L1 e L2 nella sezione 3.

### 2.4 F-R2 e controllo frozen — corretto

- **F-R2.** Ricalcolata dalle storie dei quattro scratch selezionati: migliori epoche [4, 5, 5, 5]; guadagni di validation fra epoca
  4 e 5 di −0,139 / +0,333 / +0,444 / +0,528 pp; mediana +0,389 pp. Scatta per la prima clausola (3 fold su 4); la seconda (≥ 0,5 pp)
  non e' soddisfatta. La regola nel report e' uguale a quella del piano. Nessuna estensione avviata.
- **Storico.** Logistica T100-G 19.618 su EPN: 82,577495% nell'aggregato pinnato, stessi 314 soggetti, 47.100 finestre e ruoli.
  82,577 − 70,575 = 12,002 pp. Confermato.
- **Il +23,191 non e' venduto come guadagno sul miglior frozen** in nessuno dei documenti del commit: ovunque e' riferito a questo
  controllo. Elementi a sostegno, dal codice e dalle curve: la testa frozen legge lo stesso tipo di pool (`pool_p1p2`, due ritagli
  mediati) con LayerNorm per esempio e AdamW per cinque epoche; alla quinta epoca la validation sale ancora di +1,32 pp (mediana) e
  la loss di train e' 0,72-0,74, mentre la logistica storica arriva all'89-91% di balanced accuracy sul train. E' un lettore poco
  allenato; la causa del divario resta non isolata, come scrive GPT.

### 2.5 Consuntivo — confermato

- Il mio `sacct` delle 10:58 CET coincide campo per campo con `sacct.txt` di GPT sui 29 job: tutti COMPLETED 0:0; 1 GPU, 8 CPU e
  100G per fit e benchmark; 2 CPU per i checkout; limiti 4 h (frozen) e 7 h (FT, scratch).
- Secondi: frozen 35.077, FT 105.350, scratch 105.351. Fit: 245.778 s = **68,271667 GPU-ore e 546,173333 core-ore**. Benchmark
  457 + 450 s = 0,251944 GPU-ore e 2,015556 core-ore. Checkout 24 + 18 + 16 s su 2 CPU = 0,032222 core-ore. Totale delle 29 voci:
  **68,523611 GPU-ore e 548,221111 core-ore**.
- Il checkout iniziale 63525708 compare una sola volta. I job della continuazione non sono nel conteggio: alle 10:58 CET 63525803
  COMPLETED, 63525805 RUNNING dalle 09:24, 63525809 PENDING per dipendenza.
- Rispetto ai limiti: 47,4% del tetto di 144 GPU-ore / 1.152 core-ore e 55,7% della proiezione di 122,465 GPU-ore; durata massima
  pari al 31% del limite per frozen e al 53% per FT e scratch.

## 3. Rilievi, nessuno bloccante

- **L1 (limite da aggiungere alla lettura, minore) — LR scelto al bordo della griglia in 12 celle su 12.** Frozen e FT scelgono
  sempre il LR alto, scratch sempre il LR basso (tabella in 2.1). Con due soli punti l'ottimo di validation non e' delimitato in
  nessun regime. Inoltre i due selezionati di FT e scratch hanno lo stesso LR dell'encoder (1e-4) ma LR della testa diverso (1e-2
  contro 1e-3), per costruzione della griglia approvata. Il +1,40 e' quindi una differenza fra pipeline «migliore di due candidati»,
  a pari opportunita' di tuning come previsto dal protocollo, non una differenza a parita' di ottimizzatore ne' fra ottimi di LR.
  Si vede dalla sola validation. Non chiedo ne' scelgo candidati nuovi.
- **L2 (presentazione, minore) — cifre degli IC.** Con il seed prescritto l'IC e' [0,930; 1,837]. Con altri 20 semi di bootstrap il
  limite inferiore varia fra 0,921 e 0,974 e il superiore fra 1,841 e 1,881; con 20.000 repliche [0,943; 1,877]. L'errore Monte Carlo
  sui limiti e' circa ±0,03 pp: la terza cifra decimale non e' informativa e la seconda e' incerta. Il seed prescritto resta il
  riferimento; la conclusione non cambia.
- **L3 (nota su F-R2 e rettifica di una mia nota).** Lo scatto e' valido alla lettera, ma dice poco sull'entita' del possibile
  guadagno. Le tre differenze positive valgono 12, 16 e 19 finestre su 3.600 di validation (24 soggetti), quella negativa 5. Con il
  coseno a zero il LR della quinta epoca e' sotto il 10% del valore di base. Il confronto fra regimi, su validation e train soltanto,
  da' pero' un quadro coerente:

  | Selezionati | Migliori epoche | Guadagno di validation 5−4, mediana | Loss di train all'epoca 5 |
  |---|---|---|---|
  | frozen | 5, 5, 5, 5 | +1,319 pp | 0,72-0,74 |
  | finetune | 4, 4, 5, 5 | −0,056 pp | 0,018-0,020 |
  | scratch | 4, 5, 5, 5 | +0,389 pp | 0,11-0,14 (epoca 4: 0,16-0,20) |

  FT appare saturo a questa dose, scratch migliora ancora lentamente, la testa frozen e' lontana dall'adattamento. La direzione
  attesa a dose maggiore e' una riduzione del divario FT−scratch; l'entita' non e' prevedibile da questi dati. **Rettifica:** nella
  nota N-3 della review 0b2623a avevo scritto che la regola era conservativa nel senso di non far scattare il seguito. Era
  incompleto: lo schedule comprime la clausola sulla mediana, ma favorisce la clausola «migliore epoca = 5». La regola resta
  congelata com'e'.
- **L4 (facoltativo).** Accanto al +23,19 puo' aiutare il lettore la distanza descrittiva dal miglior lettore congelato disponibile:
  93,766 − 82,577 = +11,19 pp, non appaiata e fra lettori diversi (la versione appaiata richiederebbe le OOF storiche su SCRATCH).
  Il protocollo chiede anche la distribuzione per soggetto: e' in `report.json`, non nel testo dell'analisi; i numeri sono in 2.3.

## 4. Parere sul seguito a 15 epoche per FT e scratch

**Favorevole al disegno dichiarato, come proposta nuova con preventivo per Simone. Non e' urgente e non blocca la linea lunga.**

1. **Dalle inizializzazioni originali e non dai modelli a cinque epoche: e' la scelta giusta.** Lo schedule a cinque epoche ha
   portato il LR a zero, quindi una ripresa sarebbe un riavvio a caldo e non una dose di 15 epoche sotto un solo coseno. I checkpoint
   a cinque epoche sono gia' selezionati sulla validation: ripartire da li' porterebbe la selezione dentro l'inizializzazione. FT
   selezionato ha loss di train 0,02: una ripresa misurerebbe soprattutto il suo sovra-adattamento. Ed e' quanto pre-registrato in F-R2.
2. **Che cosa deve restare identico, perche' pre-registrato.** Solo FT e scratch; stessi pesi iniziali e seed 0; stessa griglia a due
   LR per regime; stessi split, testa, batch e warmup; coseno su 15 epoche; selezione di LR ed epoca sulla sola validation; held una
   volta per cella; stesso bootstrap. Una riduzione, per esempio un solo LR per regime, sarebbe una deviazione da dichiarare e da
   motivare sulla sola validation. La sconsiglio: il LR migliore puo' cambiare con l'orizzonte, in particolare FT C0 (1e-5), che ha
   la migliore epoca alla quinta in tre fold su quattro.
3. **Da scrivere e committare prima del lancio.**
   - Contrasto primario FT−scratch a 15 epoche, riportato accanto a quello a cinque, senza scegliere la dose sul held.
   - Lettura a due lati: che cosa si dira' se l'IC a 15 epoche esclude zero e che cosa se lo include; per il confronto con il
     +1,40 consiglio la differenza appaiata per soggetto fra i due contrasti (Δ15 − Δ5) con lo stesso bootstrap.
   - Regola di arresto: se scratch e' di nuovo migliore all'ultima epoca non parte una terza dose in automatico. Con il coseno a
     zero l'ultima epoca tende a essere la migliore; un criterio fra dosi sulla validation (guadagno del selezionato da 5 a 15
     epoche) informa piu' del criterio dentro il run.
   - Il held dei 314 soggetti viene predetto una seconda volta da modelli nuovi: sono due sguardi sullo stesso held, da dichiarare.
4. **Costo indicativo, dal consuntivo; non e' un preventivo.** 3,66 ore per fit a cinque epoche, quindi circa 11 ore a 15; 16 fit
   valgono circa 176 GPU-ore e 1.405 core-ore prima dei margini. Il residuo del tetto dei fit e' 75,7 GPU-ore e 605,8 core-ore: serve
   una decisione di budget registrata. E' 2,6 volte il costo dell'intera matrice a cinque epoche, per una risposta su un solo
   dataset e un solo seme.
5. **Implementazione.** Runtime e aggregatore hanno le cinque epoche cablate (`plan["epochs"] == 5`, `len(history) == 5`,
   `history[4]`). Servono una versione nuova di protocollo e ricetta, limiti di tempo oltre le 7 ore, nuovi pin e test, e la review
   prima del lancio, come previsto dal `followup_gate`. Un fit da 11 ore senza ripresa intermedia perde tutto in caso di TIMEOUT o
   guasto del nodo: va scritto che cosa si fa in quel caso, senza retry cieco.
6. **Che cosa il seguito non risolve.** Il seme unico: la variabilita' fra riaddestramenti resta ignota, i quattro fold concordi sono
   un indizio e non una stima. Il controllo frozen: non fa parte del seguito pre-registrato, e per le rappresentazioni congelate il
   riferimento resta la logistica storica. Il bordo della griglia (L1).

## 5. Limiti della mia verifica

- Niente GPU e nessun checkpoint caricato: non ho rigenerato le predizioni held, come richiesto dal prompt. Il legame fra checkpoint
  selezionato e `held.npz` poggia sul codice del runtime gia' rivisto e sugli hash.
- I log non hanno righe per epoca: l'unicita' della predizione held non e' attestata da un messaggio del programma.
- L'82,577% storico e' letto dall'aggregato pinnato, non ricalcolato dalle sue OOF.
- I controlli di robustezza del bootstrap sono miei e descrittivi; non fanno parte dell'analisi congelata.
- Non ho rieseguito l'intera suite CPU: nessun file del runtime e' cambiato rispetto ad a2eeb75.
- I primi tentativi di collegamento a Leonardo si sono bloccati all'apertura della sessione; le letture sono delle 10:58 e 11:01 CET.

Nessun lancio, cancellazione, inferenza o modifica di checkout, soglie e candidati da parte mia.
