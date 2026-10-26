# Registro delle decisioni

Ogni riga è una decisione presa a un punto preciso, con le soglie o i parametri congelati
**prima** di lanciare l'esperimento o il job che li usa. La history di git è la prova della
pre-registrazione: questo file si committa prima, non dopo aver guardato i risultati.

Riferimento: [`docs/piano_operativo_v10.md`](piano_operativo_v10.md) §9 (registro), §12.

---

## D0 — Calendario e continuità

**Stato:** deciso il 18/09/2026.

Da `saldo -b` del 18/09/2026 (vedi output sotto): residuo 794.303 ore locali ≈ 99.300 GPU-ora,
di cui ~70.000–90.000 stimate disponibili per il progetto FM. Scadenza 7 gennaio 2027 (111
giorni dal 18/09). Il programma di piano_operativo_v10.md §10 ne usa ~17.000–21.000 GPU-ora
senza vertice, ~29.000–37.000 con vertice: più di metà del disponibile scade comunque.

Opzioni proposte nel piano (non esclusive):
- (a) chiedere a CINECA (`superc@cineca.it`) se è possibile una proroga
- (b) preparare un nuovo ISCRA B per il prossimo bando, usando come risultati preliminari il
  preprint del passo 5 e il pilot
- (c) accettare la scadenza e i tagli pre-decisi di §10

**Decisione (Simone, 18/09/2026):** si lavora **come se la scadenza non fosse un problema**,
cioè secondo il calendario del piano (opzione c come base di lavoro quotidiano). La richiesta
di proroga a CINECA (opzione a) si fa **vicino alla scadenza**, non subito: la proroga "la
danno sempre" (valutazione di Simone, non verificata con CINECA). Non pianificare per ora un
nuovo ISCRA B (opzione b) salvo indicazione contraria.

Nota per chi legge questo registro più avanti: se a ridosso del 7 gennaio 2027 la proroga non
arrivasse o arrivasse in ritardo, i tagli pre-decisi di piano_operativo_v10.md §10 restano il
piano di riserva.

### `saldo -b` (18/09/2026)

```
-----------------------------------------------------------------------------------------------------------------------------------------
account                start         end         total        localCluster   totConsumed     totConsumed     monthTotal     monthConsumed
                                             (local h)   Consumed(local h)     (local h)               %      (local h)         (local h)
-----------------------------------------------------------------------------------------------------------------------------------------
CNHPC_1526560       20240711    20260430        640000              557186        557186            87.1             0                  0
IscrB_FM-EEG24      20250120    20260220       2400000             1160939       1160939            48.4             0                  0
IscrB_WearUsFM      20260107    20270107       2000000             1205697       1205697            60.3        164383              99394
```

### `saldo --dcgp -b`

```
-----------------------------------------------------------------------------------------------------------------------------------------
account                start         end         total        localCluster   totConsumed     totConsumed     monthTotal     monthConsumed
                                             (local h)   Consumed(local h)     (local h)               %      (local h)         (local h)
-----------------------------------------------------------------------------------------------------------------------------------------
CNHPC_1526560_0     20240711    20260430         80000                   0             0             0.0             0                  0
```

**Nota.** L'unico account con budget DCGP (`CNHPC_1526560_0`) **non è `IscrB_WearUsFM`**: il
progetto FM sEMG non ha budget DCGP proprio. L'ingest del passo 2 (lavoro da CPU) va quindi
pianificato su nodi Booster interi via `sbatch` (32 ore locali/ora), non su DCGP — aggiorna
piano_operativo_v10.md §11, passo 2, "Dove gira".

`saldo --dcgp -b` del 18/09/2026: nessun budget DCGP disponibile per `IscrB_WearUsFM`. L'unica
riga è `CNHPC_1526560_0`, scaduta il 30/04/2026 con consumo 0. Conseguenza: l'ingest del passo
2 gira su nodi Booster interi (32 ore locali/ora, GPU inutilizzate), oppure su
`lrd_all_serial` per i task seriali leggeri.

---

## D1 — Autonomia dell'agente e budget per passo

**Stato:** deciso il 18/09/2026. **Decisione (Simone):** adottata la proposta del piano
operativo così com'è, budget e regole inclusi.

Budget proposto in piano_operativo_v10.md §11:

| Passo | Budget | In ore locali |
|---|---|---|
| 0 | 40 GPU-ora | 320 |
| 1 | solo `lrd_all_serial` | — |
| 1-bis | 30 GPU-ora | 240 |
| 2 | ≤ 100 nodo-ore di CPU | ≤ 3.200 |
| 3 | 50 GPU-ora | 400 |
| 5 | 300 GPU-ora | 2.400 |
| 6 | 100 GPU-ora (incluso il sanity JEPA) | 800 |

Regole proposte (piano_operativo_v10.md §6):
- «da verificare» è un compito, non un dato: si verifica (FACT o LIC), si registra in
  `docs/fatti_da_verificare.md` con URL e citazione esatta, entra nel documento di riferimento
  solo dopo la firma di TU.
- Ogni richiesta di conferma per un job dichiara il costo stimato in ore locali e GPU-ora, e
  quanto resta del budget del passo. Superare il budget di un passo richiede una decisione
  registrata, non una conferma al volo.
- Le soglie si congelano prima di guardare: ogni soglia si scrive qui e si committa prima di
  lanciare l'esperimento che la usa.
- Le allocazioni (`salloc`) le apre solo TU.
- Un dataset che non entra nello schema dei metadati si segnala: lo schema non si piega in
  silenzio.

---

## D2 — Parametri provvisori del benchmark del dataloader (passo 0)

**Stato:** deciso il 18/09/2026. **Decisione (Simone):** adottata la base proposta, senza
modifiche.

Riferimento: [`docs/dataloader_bench_spec.md`](dataloader_bench_spec.md) §10.

Base: quote A/B/C 40/35/25, `p_piena` 0,5, classe C pessimistica (solo Hyser), contesto 4 s,
patch 25 ms, 64 campioni per rank, 8 worker per GPU — un fattore alla volta, non un
fattoriale completo. Non vincolante: le scelte vere del manifest sono D9 e D10.

---

## D6a — Dataloader: montaggi al volo o precompute

**Stato:** deciso il 18/09/2026. **Decisione (Simone):** confermata la regola proposta, senza
modifiche.

**Esito applicato e firmato il 21/09/2026: AL VOLO.** waiting_fraction 0,0, margine 3,5-4×
il richiesto a cache fredda e calda (job 58137579). Dettagli in
[`docs/report_step0.md`](report_step0.md).

Regola: al volo resta il default se, col consumatore del 30M, la frazione di tempo in
attesa (massimo sui 4 rank) è ≤ 2% e il ritmo a vuoto è ≥ 1,5× il richiesto, a cache calda e
fredda su `$WORK`. Se fallisce: si spostano sulla GPU gli stadi più cari e si rimisura; solo se
fallisce ancora, precompute (con la conseguenza su `D_c` scritta nel manifest, v10 §4.6).

**Limiti noti della misura** (dalla revisione indipendente del 18/09/2026, vedi bugfix
`t_passo_batch_s`): `bench/bench_dataloader.py` misura il tempo di calcolo/lettura dentro i
worker, ma non il trasferimento del batch assemblato dal worker al processo principale
(che una vera `DataLoader` paga via shared memory) - ottimistico. È anche completamente
sincrono, senza sovrapposizione di prefetch - pessimistico. Inoltre un solo pool di worker
su un nodo altrimenti libero non modella la contesa I/O reale fra i 4 loader dei 4 rank
GPU dello stesso nodo. Il numero va letto come indicativo dell'ordine di grandezza, non
come garanzia assoluta a piena scala.

---

## D6b — Dataloader: layout del batch

**Stato:** deciso il 18/09/2026. **Decisione (Simone):** confermata la regola proposta, senza
modifiche.

**Esito applicato e firmato il 21/09/2026: PACKING.** Padding escluso (56,4% > soglia 50%);
packing batte bucketing del 13% (sopra la soglia di parità del 10%). Dettagli in
[`docs/report_step0.md`](report_step0.md).

Regola: il padding è escluso in partenza se la frazione di token di padding supera il
50%. Fra i rimanenti vince chi dà più token utili/s nel proxy; a parità entro il 10% vince il
più semplice, nell'ordine L1 → L3 → L2. Il bucketing per micro-batch (L3) si adotta solo con
l'assenso esplicito di TU: modifica la regola "mai bucketing" di v10 §2.7.

---

## D3a — Scaricare il dataset di Kaifosh?

**Stato:** deciso il 23/09/2026. **Decisione (Simone): sì.**

Riferimento: [`docs/piano_operativo_v10.md`](piano_operativo_v10.md) tabella §9 (D3), §2.1;
fatti raccolti in [`docs/fatti_da_verificare.md`](fatti_da_verificare.md) voce 7b.

Fatti verificati (fonte: repo GitHub `facebookresearch/generic-neuromotor-interface`,
verificato via API GitHub, non solo WebFetch):
- Accesso: script ufficiale del repo, `python -m
  generic_neuromotor_interface.scripts.download_data --task discrete_gestures
  --output-dir ~/emg_data` (anche `--small-subset` per un sottoinsieme di test) — non un
  semplice URL, richiede clonare il repo Python di Meta ed eseguirlo.
- **Licenza: CC-BY-NC-4.0** ("The dataset and the code are CC-BY-NC-4.0 licensed") —
  Non-Commerciale, diversa dal resto del corpus (perlopiù CC BY 4.0 senza restrizioni).
  Vincolo noto e accettato esplicitamente con questa decisione.
- Dimensioni: 51,4h train + 6,2h val + 6,4h test, 100 partecipanti (80/10/10), `.hdf5`, 2 kHz.

Il ruolo nel pretraining (solo benchmark mai visto, o anche una parte dei soggetti in
pretraining) resta una decisione separata, rimandata al passo 4 (**D3b**, non ancora presa).

---

## Aggiunta corpus — dataset Zhang et al. 2026 (Groningen)

**Stato:** deciso il 23/09/2026. **Decisione (Simone): sì, aggiunto al corpus.**

Dataset non presente nella lista originale di `piano_operativo_v10.md`/`fm_emg_reference_v10.md`
§2 - scoperta durante la sessione da un paper fornito direttamente da Simone (PDF), non dal
processo di ricerca dataset standard. Aggiunto a `fm_emg_reference_v10.md` §2.1/§2.2.

Fatti verificati (fonte primaria, non dal solo articolo):
- Paper: Zhang, Y. et al., *Scientific Data* (2026), DOI
  [10.1038/s41597-026-08111-4](https://doi.org/10.1038/s41597-026-08111-4), Università di
  Groningen. L'articolo e' CC BY-NC-ND 4.0, ma **non e' la licenza del dataset**.
- Dataset: Dataverse.nl, DOI [10.34894/QRGIZQ](https://doi.org/10.34894/QRGIZQ), verificato
  separatamente dalla pagina ufficiale - **licenza CC BY 4.0**, nessuna registrazione
  richiesta oltre le norme di citazione.
- 64 partecipanti, 14 gesti × 10 ripetizioni, mano non dominante; **ogni soggetto registrato
  sia in modalita' anatomica (8 elettrodi su muscoli specifici) sia in modalita' anello
  equidistante (8 elettrodi, 40% lunghezza avambraccio dal gomito)** - confronto diretto
  raro in letteratura, rilevante per l'asse di ricerca sulla topologia del montaggio.
- 1.245 file CSV/XLSX/TXT; dimensione totale non dichiarata sulla pagina (singoli file
  sensore ~50-58 MB); il download a pacchetto singolo e' bloccato dalla piattaforma
  (limite 9.3GB per selezione) - va scaricato via API REST di Dataverse.


---

## D7a — Firma dello schema dei metadati

**Stato:** deciso il 23/09/2026 (approvazione informale gia' data da Simone in sessione
precedente - "schema ok" - qui registrata formalmente prima di procedere con l'ingest,
come richiede la regola generale). **Decisione (Simone): approvato.**

Riferimento: [`docs/piano_operativo_v10.md`](piano_operativo_v10.md) passo 2; schema in
`src/wearusfm/metadata/schema.py` (303 righe, 8 dataclass, export JSON Schema), test in
`tests/cpu/test_schema.py`. Copre tutti i campi richiesti da v10 §4.5: gruppi di canali
con topologia/simmetria del gruppo, identita' anatomica gerarchica con `precision`/pesi
soft/flag `nominal`, orientamento della fascia (o ignoto), frequenza nativa/banda
effettiva/frequenza di rete, flag raw/inviluppo, chiralita', calibrazione, validita' per
canale (QC), identita' del soggetto e sessione/giorno per confronto fra dataset.

Sblocca l'ingest (passo 2): un dataset alla volta, dal piu' piccolo.

---

## D7b — Revisione della tassonomia anatomica

**Stato:** bozza dell'agente il 23/09/2026, **revisione di Simone completata e firmata
il 24/09/2026**. **Decisione (Simone): approvata.**

Riferimento: v10 §4.5 dichiara esplicitamente "la prima bozza di mappatura resta un
candidato naturale per un agente, con revisione umana obbligatoria" - coerente con
questa decisione. La bozza (albero regione -> compartimento/settore -> muscolo, funzione
atlante, ID FMA/UBERON) e' in `src/wearusfm/metadata/taxonomy.py`.

**Punto discusso in revisione:** la regione `trunk`/compartimento
`abdominal_wall_anterolateral`, creata dall'agente per l'obliquo esterno di Camargo (v10
§4.5 lo classificherebbe erroneamente sotto "arto inferiore, 11 etichette" insieme agli
altri 10 muscoli veri della gamba). Confermato con Simone: e' un canale EMG reale
(elettrodo fisico, non un errore di dati - verificato da due fonti indipendenti, fatto
n. 6 di `docs/fatti_da_verificare.md`), coerente con la ricerca sulla stabilizzazione del
tronco durante il cammino. La separazione in una regione distinta (invece di forzarlo
sotto "arto inferiore") resta la modellazione corretta - **approvata**.

**Discusso anche e confermato: Camargo 2021 resta nel corpus** (non solo la tassonomia).
Simone ha espresso dubbi sulla coerenza di scope (arto inferiore in un corpus altrimenti
avambraccio/polso, v10 §1), ma ha confermato di tenerlo dopo aver chiarito il suo ruolo
dichiarato (v10 §2.3, §3.5, §5.4, §10.1): e' l'unico dataset del corpus con vicinato
metrico realmente vuoto (11 elettrodi sparsi, nessuna struttura spaziale sfruttabile),
quindi l'unico caso che isola davvero il contributo del percorso di identita' anatomica
dal percorso geometrico nell'ablation di §10.1. Il costo di tenerlo e' trascurabile
(dataset piccolo); il costo di toglierlo sarebbe perdere quel test pulito, non un
risparmio di risorse.

La tassonomia e' ora vincolante: entra nel documento di riferimento v10 come
"verificato" (non piu' bozza). Gli ID ontologici e la lista muscoli di Camargo restano
comunque marcati "da verificare"/"raccolto" nel registro dei fatti dove non derivano da
lettura diretta del paper originale (fatti n. 6 e 14).


---

## D5a — Soglie dell'ancora RVQ (passo 1-bis), CONGELATE

**Stato:** congelate il 29/09/2026, **prima** di qualunque esecuzione di V1-V4 (verificato:
non esiste `results/step1bis/`). **Decisione (Simone): adottate le soglie proposte dal piano,
senza modifiche.**

Riferimento: [`docs/piano_operativo_v10.md`](piano_operativo_v10.md) passo 1-bis; v10 §6.3.

| Verifica | Soglia congelata |
|---|---|
| **V2** — errore di ricostruzione | **X = 2**: l'errore di ricostruzione **mediano** su un dataset fuori dai dataset Meta non deve superare il **doppio** di quello su emg2pose |
| **V3** — dataset-ID dai codici | **Y = 10 punti**: l'accuratezza del dataset-ID dai codici RVQ non deve superare di oltre 10 punti quella ottenuta dalle sole potenze di banda |
| **V4** — stabilita' per livello RVQ | un livello e' stabile se **almeno il 75%** dei codici non cambia sotto rumore al noise floor, su **tutti** i dataset |

Il piano stesso avverte che sono punti di partenza: contano perche' scritti prima, non perche'
giusti. **Non si ritoccano dopo aver visto un risultato**; se una soglia si rivela sbagliata si
apre una nuova decisione, senza riscrivere questa.

**Definizioni operative** (metrica di errore di ricostruzione di V2, livello di rumore "al
noise floor" di V4, baseline delle potenze di banda di V3): scritte e congelate nella sezione
"D5a - Definizioni operative di V1-V4" piu' sotto, il 29/09/2026.

D5b (esito: ancora adottabile, ristretta ad alcuni dataset, o scartata) resta da prendere
entro l'11/10; con queste soglie e' meccanica, la firma e' di Simone. Se il 25/10 le verifiche
non sono complete, l'ancora RVQ esce dal manifest (piano §10).

---

## D5a — Definizioni operative di V1-V4, CONGELATE

**Stato:** proposta di AG del 29/09/2026, **firmata e congelata da Simone il 29/09/2026 senza
modifiche** ("firma e congela D5a"), **prima** di qualunque esecuzione di V1-V4 (verificato:
non esiste `results/step1bis/`). Le soglie X = 2, Y = 10 punti e 75% di D5a restano come sono.
**L'unica soglia nuova e' quella di V1** (punto 2): D5a non ne aveva una, il piano dice solo
"non degradi in modo sostanziale"; congelata insieme al resto.

Come per D5a: **non si ritoccano dopo aver visto un risultato**; se una definizione si rivela
sbagliata si apre una nuova decisione, senza riscrivere questa.

Riferimento: v10 §6.3 (V1-V4), D5a, "Bivio di v10 §6.3" (vista canonica, normalizzazione).

### 1. Dati e preparazione comune

- **Riferimento:** emg2pose (sottoinsieme fisso, seed 0; il checkpoint l'ha visto in
  pretraining, quindi il riferimento e' ottimistico e il criterio X e' piu' severo di quanto
  sembri). **Confronto:** CapgMyo, GRABMyo, putEMG, CSL-hdemg, Camargo (tutti gia' ingeriti);
  Hyser quando sara' ingerito e NinaPro Delsys quando ci sara', ma **non bloccano** l'avvio.
- **Preparazione:** vista canonica (20-400 Hz, 1000 Hz), poi scala. **"Registrazione" = la
  sessione di un soggetto** (tutte le prove dello stesso montaggio), **non la singola prova**:
  una scala per prova cancellerebbe le differenze di ampiezza fra gesti. La deviazione standard
  si calcola su tutti i canali che passano il QC (i canali scartati non entrano).
- **Scala di arrivo:** si sceglie **solo su emg2pose**, prima di guardare gli altri dataset:
  due candidati (varianza unitaria; scala tipica di emg2pose nelle sue unita') e vince quello
  con l'errore di ricostruzione di V2 piu' basso su emg2pose. Poi si congela.
- **Unita' di misura: il token** (un canale x una patch da 200 ms). Per dataset: **20.000 token**
  (= 1.250 campioni da 3,2 s), estratti a caso con seed 0, ripartiti in parti uguali fra i
  soggetti disponibili, griglia di patch ancorata all'inizio della registrazione.

### 2. V1 - un canale alla volta

Modalita' canale-per-volta con **un solo indice spaziale fisso** (l'indice del primo dei 16
elettrodi globali), uguale per tutti i canali di tutti i dataset. Su emg2pose-mini si confronta
l'errore di ricostruzione (metrica di V2) in modalita' multi-canale nativa contro quella
canale-per-volta, sugli stessi token. ***Soglia NUOVA, proposta:*** V1 passa se l'errore mediano
in modalita' canale-per-volta e' **<= 1,5 volte** quello multi-canale. Se non passa, il ripiego
(mappare i canali sui 16 elettrodi, scelta per dataset) richiede una nuova decisione.

### 3. V2 - errore di ricostruzione

Per token, errore quadratico normalizzato **nel dominio standardizzato del tokenizer**
(`std_norm`, come la sua loss): ||x - x_rec||^2 / ||x||^2, con `x_rec` dal decoder rilasciato.
Per dataset, **mediana sui token**. Rapporto = mediana(dataset) / mediana(emg2pose); **passa se
<= X = 2** (congelata). Un dataset sopra soglia esce dall'ancora o l'ancora si scarta, come da
v10 §6.3.

### 4. V3 - dataset-ID dai codici

- **Campione:** 256 token a caso della stessa sessione; **split per soggetto** (train / validazione
  / test con soggetti disgiunti dentro ogni dataset), classi bilanciate.
- **Codici:** istogramma normalizzato dei codici per ciascuna coppia (ramo, livello RVQ), tutti
  i livelli, concatenati (matrici sparse).
- **Baseline:** log-potenza media dei 256 token nelle **5 bande del repo NeuroRVQ** (20-60,
  60-125, 125-200, 200-250, 250-400 Hz; `plotting/plotting_example.py` righe 86-92), stessi
  campioni e stessi split.
- **Classificatori:** per **entrambi** i set di feature, regressione logistica L2 e gradient
  boosting; per ciascun set si riporta il migliore su validazione (una sonda debole
  sottostimerebbe l'identificabilita').
- **Metrica:** accuratezza bilanciata sul test, con il caso e l'intervallo di confidenza
  (bootstrap sui soggetti). **L'ancora si scarta se codici - bande > Y = 10 punti** (congelata).

### 5. V4 - stabilita' per livello RVQ

- **Noise floor per dataset:** RMS mediana dei token nel **10% a energia minore**, nella vista
  canonica dopo la scala per sessione.
- **Perturbazione:** rumore gaussiano bianco a media zero, deviazione standard = noise floor,
  aggiunto **dopo** la vista canonica con la scala di sessione **tenuta fissa**; 3 semi.
- **Metrica:** per ogni coppia (ramo, livello), frazione di token il cui codice a quel livello
  **non cambia** (livelli confrontati in modo indipendente, non condizionati ai precedenti).
- **Regola:** un livello e' stabile su un dataset se in **tutti e 4 i rami** la frazione e'
  **>= 75%** (congelata); e' stabile se lo e' su **tutti** i dataset. Nessun livello stabile =
  ancora scartata.
- **Nota sull'origine di V4 (30/09/2026, su richiesta di Simone):** V4 e' un **criterio del
  progetto**, non uno standard esterno. Nessun documento del repo cita una fonte di letteratura
  per la perturbazione a rumore o per la soglia del 75%, e non risulta un benchmark
  standardizzato equivalente (non verificato in letteratura). Simone dichiara di **non averlo
  proposto**; nei documenti v10 compare nelle bozze di AG, come controllo di coerenza sul
  tokenizer. Peso: la perturbazione (rumore bianco al noise floor), il 75% e i controlli
  (rumore zero = 1,0; dose 0,1x) sono scelte di progetto congelate in D5a il 29/09/2026, prima
  del run. L'esito di V4 va quindi letto come "il criterio che ci siamo dati", non come un
  fallimento rispetto a uno standard; la decisione D5b resta di Simone.

### 6. Scelte firmate da Simone (quelle che possono cambiare l'esito)

1. "registrazione" = sessione (punto 1);
2. soglia nuova di V1, 1,5 volte (punto 2);
3. V4 richiede la stabilita' in tutti e 4 i rami (punto 5), non solo in media;
4. insieme dei dataset di confronto, in particolare CSL-hdemg e Camargo (punto 1);
5. 20.000 token per dataset (punto 1).

Costo (stima di AG, non un impegno): le verifiche V1-V4 usano il tokenizer da 144M in sola inferenza su ~10^5 token in tutto:
stima **< 2 GPU-ora** (< 16 ore locali) su 30 di budget del passo; la stima definitiva va
ridata prima di ogni lancio.

### 7. Autorizzazione al run vero (Simone, 29/09/2026)

Simone autorizza il run vero di V1-V4 **a condizione che il secondo collaudo (con V3 attiva, i
controlli di V4 e il riferimento emg2pose multi-utente) sia pulito**: job finito senza errori;
controllo a rumore zero di V4 = 1,0 su tutti i dataset; V3 calcolabile; nessun dataset escluso
per errore; V1 passa. Se una condizione fallisce **non si lancia** e si riporta il motivo.
Parametri del run: 5 dataset ingeriti (capgmyo, grabmyo, putemg, csl_hdemg, camargo2021),
78 gruppi per dataset, 80 finestre per V1, al piu' 2 sessioni per soggetto, `boost_usr_prod`, 1
GPU, limite 2 ore (<= 2 GPU-ora = <= 16 ore locali, sui 30 del passo). Il report e' un dato: D5b
(adottare, restringere o scartare l'ancora RVQ) resta una decisione di Simone.

---

## D8a — Soglie del gate di consistenza al ricampionamento (passo 3), CONGELATE

**Stato:** congelate il 29/09/2026, **prima** di qualunque esecuzione del gate (verificato:
non esiste `results/step3/`). **Decisione (Simone): adottate le soglie proposte dal piano,
senza modifiche.**

Riferimento: [`docs/piano_operativo_v10.md`](piano_operativo_v10.md) passo 3; v10 §4.2, §7.1.
E' il primo si'/no scientifico del progetto.

Il gate e' superato solo se valgono **entrambe**:

1. **errore relativo RMS delle feature <= 5%** nella banda condivisa;
2. **accuratezza del probe A-contro-B <= 55%**, con **intervallo di confidenza al 95% che
   contiene il 50%**.

Il gate richiede un solo dataset a 2 kHz e si puo' lanciare su emg2qwerty appena esiste il
front-end, senza aspettare l'ingest completo. Da controllare per primi se fallisce: il fattore
Δt e l'anti-aliasing **per ciascuna famiglia di base** (indicazione del piano, non una soglia).

**Non decisa qui:** D8b (se il gate fallisce, insistere o ripiegare). Il piano propone una
settimana per correggere e, se il 18/10 il gate non e' superato, i front-end separati per
frequenza come default; resta una decisione da prendere, con firma di Simone, al momento.

**Aperto, da scrivere QUI prima del lancio del gate:** la definizione operativa di "banda
condivisa", le due frequenze confrontate (A e B), e come si costruisce l'intervallo di
confidenza del probe.


---

## DP — Preprint autonomo di ri-valutazione: NO per ora

**Stato:** deciso il 29/09/2026. **Decisione (Simone): non si fa per ora.**

Riferimento: [`docs/piano_operativo_v10.md`](piano_operativo_v10.md) §9 (DP) e passo 5. Il piano
proponeva di farlo, con una persona dedicata fuori dal cammino critico del foundation model.

**Motivazione (Simone):** vuole avere prima qualcosa di preliminare dal foundation model, e
stima di riuscire ad avere finito l'addestramento e capire a che punto si e' entro fine
novembre.

**Conseguenze da tenere presenti:**
- Il piano attribuiva al preprint il ruolo di "uscita garantita" e di materiale preliminare per
  la continuita' di D0. **Correzione di Simone (29/09/2026): la proroga si puo' chiedere anche
  senza un paper e di solito danno 3 mesi** (esperienza di Simone, non verificata con CINECA,
  coerente con D0). Quindi non dipende dal preprint. Resta comunque l'intenzione di
  **accelerare**: prima si ha una lettura preliminare, meglio e'.
- **Cosa ci sara' davvero a fine novembre, dal calendario del piano (§10):** finestra 1
  (01-03/11) lancia le ablation a 30M; W7-W8 i job girano; W9-W10 (16-29/11) decisioni su
  obiettivo, architettura ed E e conferme a 100M. La **ladder completa parte il 01-02/12** e la
  valutazione finale e' W15-W16 (fino al 07/01). Quindi a fine novembre si avra' una lettura
  preliminare (ablation e conferme), non i modelli grandi.

**Da riaprire:** a fine novembre (W9-W10), con i risultati preliminari in mano. Nessun impegno
di persone ora.

**Non cambia:** l'harness di valutazione (passo 5) prosegue, perche' serve a valutare il
foundation model stesso e non dipende dal preprint. Resta criterio di chiusura del passo 5
riprodurre prima i numeri pubblicati di NeuroRVQ entro una tolleranza dichiarata.


---

## Bivio di v10 §6.3 — banda della vista canonica del tokenizer: 20-400 Hz

**Stato:** deciso il 29/09/2026. **Decisione (Simone): la banda e' 20-400 Hz, "sicuramente,
senza dubbi", per i dati campionati ad almeno 1 kHz.**

Riferimento: v10 §6.3 (bivio "20-90 Hz"), piano passo 1-bis punto 1, fatto n. 4 in
[`docs/fatti_da_verificare.md`](fatti_da_verificare.md).

**Cosa c'era sul tavolo.** La lettura del repo NeuroRVQ (fatto n. 4) non ha chiuso la banda di
pretraining: il repo non contiene il codice di training. Il codice d'esempio filtra a
**20-400 Hz** (Butterworth di ordine 3, `filtfilt`, ricampionamento a 1000 Hz); il **20-90 Hz**
compare solo nel paper, come protocollo del confronto downstream. Nessuna fonte dichiara la
banda del pretraining.

**Cosa segue dal testo di v10 §6.3** (ramo "altrimenti vale la soluzione sopra"; e' una
conseguenza del testo, non una scelta nuova di Simone, da confermare quando si scrive v10.1):
- vista tokenizer canonica per i dataset raw con frequenza nativa >= 1 kHz: passabanda
  **20-400 Hz** -> resampling polifase a 1000 Hz esatti con lo **stesso taglio anti-alias per
  tutti** -> normalizzazione attesa dal tokenizer -> griglia di patch da 200 ms;
- dataset a 200 Hz (DB5, unico nel corpus di pretraining): **ancora RVQ mascherata, nessun
  upsampling**;
- **non** si applica la vista simmetrica a 20-90 Hz (cioe' l'ancora RVQ non e' "di banda
  bassa").

**Normalizzazione: si' (Simone, 29/09/2026: "usiamo normalizzazione, secondo me e' meglio").**
Il codice d'esempio non normalizza in ingresso all'encoder (fatto n. 4, punto b, un'assenza nel
codice e non una dichiarazione degli autori), quindi qui si sceglie di **aggiungerla** nella
vista canonica.
**Tipo di normalizzazione, deciso (Simone, 29/09/2026: "accetto la tua proposta. no z score"):**
un **fattore di scala unico per registrazione** (deviazione standard calcolata su tutti i canali
insieme), **non** uno z-score per canale e non per finestra. I dettagli (scala di arrivo,
canali scartati dal QC, "registrazione" = sessione) sono fissati e congelati nella sezione "D5a -
Definizioni operative di V1-V4", punto 1. Motivi della scelta
(proposti da AG, accettati): il tokenizer stesso standardizza congiuntamente su canali, patch e tempo
(`std_norm`, `NeuroRVQ.py` righe 571-575) e il transformer usa i rapporti di ampiezza fra canali
come informazione spaziale; una normalizzazione per finestra cancellerebbe la dinamica (i picchi
sono informativi). **Da misurare in 1-bis:** l'ampiezza di emg2pose nelle sue unita' non e'
documentata nel codice, quindi va verificato che l'ingresso normalizzato cada nel regime che
l'encoder congelato ha visto (errore di ricostruzione su emg2pose con e senza normalizzazione).

**Passo 5 — preprocessing di NeuroRVQ nella valutazione (Simone, 29/09/2026: "si fallo").**
Il preprocessing dell'harness e' per modello e **separato** dalla vista canonica del tokenizer:
1. **Replica dei numeri pubblicati** (debug di dati e harness, piano passo 5): pipeline degli
   autori, cioe' per NeuroRVQ 20-90 Hz nel confronto downstream (paper, pdf pag. 31, testo
   estratto, da confermare) e il loro split per soggetto 7:1:2 con fine-tuning completo (v10
   §2.1). La tolleranza si dichiara **prima** del lancio.
2. **Ri-valutazione sui nostri split** (il confronto valido): NeuroRVQ riceve il **suo**
   preprocessing (20-90 Hz), non la vista canonica, cosi' non lo si svantaggia. Le regole di
   protocollo valgono per tutti i modelli: split per soggetto, niente classe "rest" dalle
   pause, normalizzazione stimata solo sul train.
Il preprocessing dei nostri modelli nell'harness e' un'altra decisione, non presa qui.

**Non cambia:** la definizione di "banda condivisa" di D8a resta da scrivere.


---

## Autorizzazione notturna per gli ingest (Simone, 30/09/2026)

**Decisione (Simone): "ti autorizzo".** L'agente puo' lanciare da solo, mentre Simone dorme, gli
**ingest gia' scritti e testati** (test su CPU passati e suite completa verde), alle condizioni
proposte dall'agente e accettate:
- job **seriali su CPU** (`lrd_all_serial`), **0 GPU**, al piu' **~6 ore locali** ciascuno;
- **prima un collaudo su un soggetto, poi il pieno**;
- **dal dataset piu' piccolo** (un dataset alla volta);
- un ingest che fallisce non si rilancia in cerca di una correzione senza guardare la causa:
  si legge il log, si corregge il codice sul Mac (test prima), e si rilancia una volta; alla
  seconda rottura ci si ferma e si riporta.
Restano fuori da questa autorizzazione: job GPU, cancellazioni di file, firme (D4, D5b, ...) e
qualunque dataset il cui parser non sia stato scritto e testato.


---

## Zhang 2026 — come rappresentare EMG a frequenze miste: opzione B DECISA (Simone, 30/09/2026)

**Stato:** proposta di AG del 30/09/2026, NON decisa. Non ingerito: serve una scelta di schema (CLAUDE.md: un dataset che
non entra nello schema si segnala, lo schema non si piega in silenzio).

**Cosa si e' visto** (prime cinque righe di un `sensor_data.csv` reale, letto via leonardo-ops il 30/09/2026): 38
colonne per sequenza; 66 cartelle di soggetti, ciascuna con `anatomical/` e `random/`, ciascuna con `sequence_NN/`
(`sensor_data.csv` da ~60 MB e `label.csv`). Le righe di intestazione dicono, per colonna: sensore, tipo di segnale, muscolo,
frequenza, colore/numero. **EMG a due frequenze nello stesso file:** quattro canali a **2000 Hz** (muscoli FDS, PTE, SUP,
EDC) e quattro a **4000 Hz** (ECR, ECU, FCR, FCU); in piu' 6 canali IMU del polso a ~148 Hz e, per ciascuno dei quattro
sensori a 4000 Hz, 6 canali IMU a ~74 Hz (ACC/GYRO: non EMG). I numeri di frequenza usano la virgola decimale
("148,1481 Hz"). Lo schema ha `native_fs_hz` per canale, ma l'array salvato (T, C) ha una sola frequenza.

**Da controllare sul dato prima di scrivere il parser** (cluster irraggiungibile per manutenzione): (1) come sono
disposti nelle righe i due tassi (righe a 4000 Hz con i canali a 2000 Hz ripetuti o vuoti?); (2) se le colonne condividono
la stessa origine dei tempi; (3) se `anatomical` e `random` hanno gli stessi 8 canali EMG; (4) se i valori usano la
virgola decimale.

| Opzione | Cosa fa | Pro | Contro |
|---|---|---|---|
| **B. Portare i 4 canali a 4000 Hz a 2000 Hz** (polifase, filtro anti-alias) e salvare UN array (T, 8) a 2 kHz | 8 muscoli simultanei a 2 kHz, come la maggioranza del corpus (Meta, NinaPro) | un solo array, nessuna modifica allo schema ne' al dataloader; la simultaneita' dei muscoli si conserva; l'FM guarda al massimo 400 Hz (vista canonica 20-400 Hz) | si butta la banda 1-2 kHz di quei 4 canali; e' un ricampionamento fatto all'ingest, prima che il gate D8a dica se ricampionare sia lecito |
| A. Due array per sequenza, uno per frequenza | lossless, ciascun gruppo a frequenza nativa | nessuna perdita | il dataloader deve leggere piu' array come UN campione; cambia il formato del processato e il lettore; oggi non e' previsto |
| E. Due sessioni separate (4 canali ciascuna) | ogni gruppo e' una sessione a se' | semplice | si perdono le relazioni fra i due gruppi nello stesso istante: e' il valore del dataset (8 muscoli anatomici insieme) |
| Escludere Zhang | niente | nessun costo | si perde il confronto anatomico contro anello equidistante sugli stessi soggetti (64) |

**Raccomandazione di AG: B**, con il CSV grezzo lasciato in scratch (nulla si distrugge: si potra' rifare a 4000 Hz).
Motivo: e' la sola opzione che conserva la simultaneita' senza toccare schema e dataloader; la banda persa e' fuori da cio'
che l'FM usa. Il rischio va detto: si ricampiona prima del gate D8a. Mitigazione: registrare nel sidecar
`resampled_from_hz: 4000` per quei canali, cosi' si possono escludere in un'analisi.
Non e' urgente: pesa poco sull'FM (35 GB su ~700 GB di corpus) e non blocca D9 (25/10).

**Decisione (Simone, 30/09/2026): "sicuramente possiamo ricampionare a 2 kHz, quello e' ok".** Si adotta l'opzione B: i quattro
canali a 4000 Hz vengono portati a 2000 Hz (polifase, con filtro anti-alias) e si salva un solo array (T, 8) a 2 kHz; il
CSV grezzo resta in `$SCRATCH`/`$WORK` (nulla si distrugge); nel sidecar `resampled_from_hz: 4000` per quei canali.
**Restano da fare, prima del parser, i quattro controlli sul dato** elencati sopra (disposizione dei due tassi nelle righe,
origine dei tempi, canali di `anatomical` contro `random`, virgola decimale): richiedono il cluster.

**I quattro controlli, fatti il 01/10/2026** (`scripts/inspect_zhang_csv.py`, sola lettura via leonardo-ops; soggetti HG_A468E29 e HG_A684H18, una
sequenza `anatomical` e una `random` ciascuno):
- **intestazione di 7 righe:** sensore (S1-S8, `NaN` per il polso), tipo (EMG, ACC/GYRO X/Y/Z), muscolo, frequenza, colore o numero, un numero
  (70786 per i quattro EMG a 2000 Hz e per l'IMU del polso; 73242, 72970, 72978, 72908 per i quattro a 4000 Hz: ipotesi non verificata, numeri di
  serie dei sensori, e i quattro a 2000 Hz sarebbero un unico sensore a quattro canali), unita' (`mV` per l'EMG, `G`, `deg/s`). Separatore `,`;
- **(1) disposizione dei due tassi:** ogni colonna e' una serie a se', **contigua dall'inizio del file**: i canali a 4000 Hz riempiono tutte le righe,
  quelli a 2000 Hz solo la prima meta' e poi sono vuoti (non interlacciati, non ripetuti); le IMU sono ancora piu' corte. Righe di dati: 602.586 per i
  canali a 4000 Hz e 301.293 per quelli a 2000 Hz (esattamente la meta'); nella sequenza `random`: 527.526 e 263.763;
- **(2) origine dei tempi:** nel file non c'e' una colonna di tempo. Tutte le colonne cominciano alla prima riga e **durano uguale**: 301.293 / 2000 =
  602.586 / 4000 = 150,6465 s; le IMU 150,645 s (22.318 righe a 148,1481 Hz) e 150,646 s (11.159 righe a 74,0741 Hz). Lettura (non dichiarata dalla
  fonte): stessa finestra di registrazione, inizio comune;
- **(3) `anatomical` contro `random`:** stessi 8 canali EMG (stessi sensori S1-S8, frequenze, colori e numeri di serie); cambia solo la riga del muscolo,
  che in `random` vale `NaN` (nessun muscolo mirato). La disposizione dei sensori in `random` (anello equidistante, ordine) va presa dalla fonte;
- **(4) virgola decimale:** solo nelle etichette di frequenza dell'intestazione («148,1481 Hz», campo tra virgolette); i valori usano il punto (0 celle
  con la virgola, 0 valori non numerici nelle colonne EMG).
Conseguenza per il parser (opzione B): i canali a 4000 Hz si portano a 2000 Hz (polifase) e risultano lunghi esattamente come quelli a 2000 Hz; si
allineano campione per campione dall'inizio. Prima del parser restano da prendere dalla fonte la disposizione dei sensori in `random` e il conteggio
dei soggetti (64 partecipanti dichiarati, 66 cartelle viste il 30/09).

**Fonte e conteggi, 01/10/2026** (fact-checker sul README e sulla pagina del dataset; il corpo del paper non e' leggibile dagli strumenti automatici;
citazioni controllate parola per parola sul `README.txt` della copia raw su Leonardo, 9 su 9; conteggi via leonardo-ops):
- **soggetti: 64**, non 66 (il «66» del 30/09 era sbagliato): 64 cartelle `HG_*` piu' `MANIFEST.TXT`, `participants.csv`, `participants.xlsx`,
  `README.txt`. HG_H496O27 e HG_H544X3 non hanno `random/`. Sequenze: `anatomical` 5 in 60 soggetti e 4 in 4; `random` 5 in 58, 4 in 3, 3 in 1, 0 in 2;
  621 `sensor_data.csv`. I numeri delle sequenze vanno da 01 a 10 e non sono consecutivi per soggetto. L'ID HG_M7873Q del CSV corrisponde alla
  cartella HG_M7873Q0;
- **hardware:** «The sEMG acquisition system included four Delsys Trigno Avanti sensors and one Delsys Trigno Quattro sensor, along with a base
  station.» I quattro canali a 4000 Hz sono i 4 Avanti (un elettrodo ciascuno), i quattro a 2000 Hz i 4 elettrodi dell'unico Quattro (l'ipotesi del
  numero di serie comune e' coerente con la fonte, che dice 4 Avanti e 1 Quattro: il numero di serie in se' non e' dichiarato);
- **modo `random`, fonti in contrasto:** README «the sensors are equally spaced on the circumference of the forearm»; descrizione del dataset «sEMG
  sensors were placed randomly around the circumference of the forearm (random mode)»; abstract «8 sensors located in a ring structure around the
  forearm». E: «Due to a technical issue, the order of the sensors in random mode were is unknown for 32 participants.» La numerazione S1-S8 e' nella
  figura del paper, non letta;
- **etichette:** i tempi di `label.csv` sono «counted from the start of the video recording»; nessuna fonte letta dice come allinearli all'EMG;
- **intestazione:** il README dice «From row 9 onwards the files contain the actual data values.»; nei file letti l'intestazione e' di **7** righe
  (602.593 righe = 7 + 602.586): il parser deve riconoscerla, non contarla a priori;
- rete: acquisizione a Groningen (comitato etico dell'UMCG); 50 Hz per deduzione, non dichiarata.

**Decisione (Simone, 01/10/2026: «sì, vanno bene entrambe le proposte su Zhang»):**
- modo `random`: **gruppo sparso senza angoli** per tutti i soggetti (non si dichiara un anello che non si puo' ricostruire: fonti in contrasto e
  ordine ignoto per 32 soggetti); modo `anatomical`: gruppo sparso con gli 8 muscoli dell'intestazione. Se la figura del paper chiarira' l'ordine, si
  potra' passare all'anello D_8 per i soggetti con ordine noto;
- etichette: `label.csv` copiato cosi' com'e' nel processato, segnato come **non allineato all'EMG** (tempi sul video, sincronizzazione non
  dichiarata), non usato finche' la sincronizzazione non e' chiarita.

**Parser scritto (01/10/2026)**, `src/wearusfm/ingest/zhang2026.py`. Scelte di AG da rivedere:
- **supinatore (SUP) non e' nella tassonomia firmata D7b**: annotato al livello di **regione** (avambraccio prossimale), senza piegare la tassonomia.
  Per portarlo al livello di muscolo serve aggiungerlo (OLS4: UBERON:0003228 «supinator muscle», FMA:38512 «Supinator») e scegliere il
  compartimento: e' un muscolo profondo dell'avambraccio prossimale dorsale, e nessuno dei compartimenti firmati gli corrisponde davvero. Decisione
  di Simone. **Decisione (Simone, 01/10/2026: «sì autorizzo collaudo e pieno di Zhang, regione va bene»): per ora regione**; ingest autorizzato;
- abbreviazioni -> tassonomia: PTE = PT (pronatore rotondo), ECR = ECRL (la descrizione del dataset elenca «Extensor Carpi Radialis Longus», non il
  brevis); le altre coincidono. La tabella abbreviazione -> muscolo non e' nella fonte: e' dedotta dai nomi;
- una sessione per soggetto e modo (sequenze concatenate come prove); canali in ordine S1-S8; lato = opposto alla mano dominante (abstract, verificato:
  «64 adult participants performed 14 gestures with their non-dominant hand»); tipo di elettrodo dalla frequenza (4000 Hz = Avanti, 2000 Hz = Quattro);
  nessun dato anagrafico nel processato.

**Pieno (job 59094690, 01/10/2026): 121 sessioni su 124, 61 soggetti, 596 sequenze, 24,3 h; 3 sessioni fallite**, lette sui file:
- **HG_O983O9 anatomical, sequence_09:** canali a 2000 Hz di 309.474 campioni (154,7 s), a 4000 Hz di 503.255-503.256 (125,8 s): gli Avanti non coprono
  la stessa finestra del Quattro e non si sa come allinearli. Correzione (**scelta di AG, da firmare**): una sequenza con durate incoerenti si **esclude** (registrata in
  `excluded_sequences` nel sidecar), le altre della sessione restano; fra canali dello stesso tasso si tollerano 2 campioni di differenza (si taglia).
  Rilanciata solo questa sessione;
- **HG_H496O27 e HG_H544X3 (anatomical; sono i due soggetti senza modo `random` e con la riga vuota in `participants.csv`):** esportazione diversa,
  intestazione di UNA riga con le sole frequenze («2000 Hz, 2000 Hz.1, ...»), senza sensore, muscolo e unita'; in HG_H496O27 l'IMU del Quattro sembra
  esportata come quaternione (4 colonne a 74 Hz di norma circa 1). Abbinare le colonne a sensori e muscoli richiederebbe di assumere l'ordine degli
  altri file, e manca la mano dominante: **lasciati fuori** (~0,4 h su ~25; Simone, 01/10/2026: «sì, lasciamo fuori i due soggetti di Zhang»).


---

## NinaPro DB8 e DB10 — riscarico autorizzato (Simone, 30/09/2026)

Il raw di DB8 e DB10 e' vuoto (`docs/dataset_access.md`: il download non e' mai riuscito). **Simone autorizza il riscarico.** Testo
di Simone: «Ti do comunque autorizzazione a riscaricare o db8 e db10 che non sono venuti per le licenze abbiamo gia' fatto questo discorso e
non ce ne preoccupiamo». **Correzione del 30/09/2026 (review):** la versione precedente di questa voce diceva che Simone aveva dichiarato il
mancato download «legato alle licenze»; la frase si legge piu' naturalmente come «riscarica DB8 e DB10, che non sono venuti; per le licenze il
discorso e' gia' fatto». La causa del mancato download resta **non nota** (il registro dice solo «download mai riuscito").

Condizioni di esecuzione (non cambiano l'autorizzazione, dicono come si applica):
- richiede il cluster (i compute node non hanno internet: il download si fa dal login node, via `leonardo-ops`, verso
  `$SCRATCH`, dove non c'e' quota); prima del lancio si riporta la dimensione da scaricare;
- se il sito chiede di accettare termini o di creare un account, ci si ferma e si riporta: sono azioni di Simone;
- dopo il download: ingest con il parser NinaPro standard solo dopo aver verificato frequenza e canali sui file veri
  (fatto n. 1 del registro: ~1111 Hz per DB8, ~1926 Hz per DB10, montaggi da verificare) — un dataset che non entra nello
  schema si segnala, non si piega.

---

## D5b — l'ancora RVQ NON si scarta (Simone, 30/09/2026)

**Lettura meccanica delle regole D5a sui dati del run 59048369 (registrata, invariata):** V1 passa (1,23 contro 1,5); V2 passa su CapgMyo,
GRABMyo e Camargo, non su CSL-hdemg (2,16) e putEMG (2,24); V3 non passa (0,310 contro 0,10); V4 non passa (nessun livello stabile, nemmeno
il livello 0). Per le regole congelate, V3 e V4 portano a «scartata».

**Decisione di Simone:** «non scartiamo l'RVQ, e' una delle cose che dobbiamo tenere». D5b spetta a lui. E' una **deroga consapevole** alla
lettura meccanica, presa sapendo che V3 e V4 non passano: **non modifica le soglie di D5a**, che restano congelate, e il loro esito resta
scritto sopra e in `results/step1bis/RIEPILOGO.md`.

**Aperto (da decidere, non ancora deciso):**
1. **Forma:** su tutti i dataset, o ristretta a quelli sotto la soglia di V2 (CapgMyo, GRABMyo, Camargo, piu' i dataset Meta)? Il piano
   prevede gia' l'ancora mascherata per dataset (v10 §4.2, §6.3).
2. **Quali codici si predicono.** La regola del piano (v10 §6.3, V4) era «solo i livelli stabili su tutti i dataset»: nessuno lo e', quindi
   la regola non si puo' applicare cosi' com'e'. Serve una **nuova decisione**, con i criteri scritti e committati **prima** di guardare.
3. **Come si giudica:** l'ablation run 4 contro run 5 (v10 §10.1) resta l'arbitro empirico dell'utilita' dell'ancora.

**Conseguenze (lettura di AG, non decisa da Simone):** V1-V4 sono state eseguite, quindi il taglio del 25/10 del piano §10 («se le verifiche
non sono complete l'ancora esce dal manifest») non dovrebbe scattare; il target RVQ resta nel manifest (D9 (e)) nella forma da decidere.
**Buco (review del 30/09/2026):** la proposta firmata dopo richiede altre misure (sonda per ramo, V2 sui dataset nuovi); **non e' scritto cosa
succede se non sono fatte entro il 25/10**. Da decidere.

**Idea collegata (Simone: «la 2 mi piace»):** un **target discreto proprio** — k-means sulle feature fisiche (potenze di banda, RMS) per
riavere la multimodalita' (v10 §6.3) senza un tokenizer esterno. **Stato: candidata, non avviata.** Richiede una nuova decisione con criteri
congelati prima della prova (stabilita' sotto il rumore come V4, controllo dataset-ID come V3). Se stia accanto all'RVQ o lo sostituisca
**non e' deciso**.

---

## Misura del QC dei canali — autorizzata (Simone, 30/09/2026: «sì, misura il QC quando il cluster torna»)

**Cosa:** contare, in **sola lettura**, quanti canali in piu' avrebbe scartato la soglia di «canale piatto» **relativa** (1e-3 x mediana
delle std dei canali della registrazione) rispetto a quella **assoluta** attuale, sui **7** ingest con soglia assoluta: CapgMyo, GRABMyo,
putEMG, CSL-hdemg, Camargo, NinaPro DB5 (dopo il suo ingest) e Kaifosh. Elenco e soglie: `docs/formato_processato.md`, regola 2.
**Perche':** con soglia assoluta e unita' diverse, «0 canali scartati» (Camargo, Kaifosh) non distingue «tutti buoni» da «test troppo
permissivo». **Correzione del 30/09/2026 (review):** la soglia assoluta e' troppo permissiva quando i valori sono **grandi** (Kaifosh: massimo
assoluto fra 362 e 7447 nelle sue unita', quindi 1e-6 non scarta praticamente nulla) e troppo **severa** quando sono piccoli (volt); in chat AG
aveva detto il contrario. **Limite della regola relativa:** scarta solo canali piu' di 1000 volte sotto la mediana (in pratica piatti). Un
elettrodo staccato che raccoglie rumore o rete **non** lo trova, come non lo trova la regola assoluta: la misura risponde a «ci sono canali piatti
non segnalati?», non a «ci sono canali morti?». **Non cambia nessun dato:** se emergono differenze, si correggono solo i flag `qc_valid` nei `metadata.json` (le colonne non
si tolgono mai), e solo dopo aver riportato i numeri a Simone.
**Esecuzione:** un job seriale su CPU (`lrd_all_serial`), via `leonardo-ops`, output in `$WORK/wearusfm_runs/` e non nel repo; zero GPU-ora.
Costo (stima di AG, non un impegno): meno di 1 ora di walltime, circa 0,1-0,3 ore locali; lo ridichiaro in chat prima di sottomettere.

**Aggiornamento (Simone, 30/09/2026, dopo la discussione sui canali morti): «se vedi che sono morti non li usiamo, ma va tutto ovviamente
documentato».** Quindi e' autorizzata anche la **correzione dei flag**, non solo la misura. Come si applica (strumento
`scripts/qc_relative_check.py`, testato in locale su dati sintetici; sbatch `scripts/slurm/qc_relative_check.sbatch`; **mai lanciato sui
dati veri**):
1. **Misura, sola lettura**, su tutti i dataset processati. I dataset gia' ingeriti con la regola relativa (NinaPro DB2/3/4/6/7, ...) sono il
   **controllo**: per costruzione devono dare zero canali nuovi; se no, il difetto e' nello strumento e ci si ferma prima di scrivere.
2. **Applicazione per dataset** (`--apply`) solo se il controllo e' pulito e il **freno di prudenza**, scritto ora, prima di guardare i dati, non
   scatta: nessuna sessione con piu' del 25% dei canali nuovi piatti, e nuovi canali piatti in non piu' del 5% delle sessioni del dataset. Se il
   freno scatta, non si scrive nulla e si riporta a Simone.
3. **Documentazione:** nel sidecar `qc_revisions` (data, regola, colonne aggiunte, riferimento a questa decisione) piu' il backup
   `metadata.pre_qc_revision.json` (reversibile); nel report `$WORK/wearusfm_runs/results/passo2/qc_relative_check_<job>.json`; riepilogo
   committato in `results/passo2/` e una riga qui. I dati `data_int16.npy` non si toccano e le colonne non si tolgono mai; dopo ogni
   scrittura la sessione deve passare `validate_session`, altrimenti il sidecar si ripristina.
4. **Effetto sul run V1-V4 (job 59048369):** e' stato calcolato coi flag di prima e la scala di sessione usa i soli canali validi. Se la
   correzione tocca sessioni di CapgMyo, GRABMyo, putEMG, CSL-hdemg o Camargo usate dal run, lo si dichiara nel RIEPILOGO e si valuta l'effetto
   sui numeri; **nessuna soglia di D5a si riscrive**.
Costo: due job seriali (misura, poi applicazione), meno di 1 ora ciascuno, zero GPU-ora; lo ridichiaro in chat prima di sottomettere ciascuno.

---

## D5b — forma decisa: ancora RVQ RISTRETTA (Simone, 30/09/2026: «ok restringiamo»)

Risolve il punto 1 rimasto aperto in «D5b — l'ancora RVQ NON si scarta»: l'ancora si **restringe ai dataset sotto la soglia di V2**, non e'
estesa a tutti. Il punto 2 (quali codici si predicono) resta aperto: **proposta di AG in `docs/proposta_ancora_rvq.md`, NON firmata**; contiene
anche il trattamento dei dataset mai misurati con V2 (spenti fino alla misura) e una regola di scelta dei rami scritta prima di misurare. Nessuna
misura di quella proposta parte prima della firma e del commit.

---

## Licenze — decisioni d'uso (Simone, 30/09/2026)

- **NinaPro DB4 e DB5 (CC BY-ND 4.0): si INCLUDONO nel corpus.**
- **NinaPro DB2 (CC0 indicata ma non verificata), DB3, DB6, DB7 (nessuna licenza trovata): si includono «senza problemi».** Vale la stessa linea
  gia' presa per DB8 e DB10.
- **Licenza dei pesi NeuroRVQ (fatto n. 5): «non serve»** verificarla; nessun `license-checker` sulla scheda Hugging Face.

Nota di AG, per il registro: sono **decisioni d'uso**, non verifiche. Lo stato dei fatti n. 5 e n. 21 in `docs/fatti_da_verificare.md` non cambia
(le licenze restano «non verificate» dove lo erano). AG non ha valutato la compatibilita' legale della clausola ND con l'uso nel corpus.

---

## D5b — proposta dei codici dell'ancora RVQ: FIRMATA e CONGELATA (Simone, 30/09/2026)

Simone firma le **tre cose** di `docs/proposta_ancora_rvq.md` senza modifiche («firmo le tre cose»):
1. **Dove:** l'ancora e' accesa su CapgMyo, GRABMyo, Camargo (V2 <= 2 misurato) e su emg2pose ed emg2qwerty (dati del tokenizer); **spenta** su
   putEMG, CSL-hdemg (V2 non passa) e DB5 (200 Hz); **spenta fino a V2** su Kaifosh, NinaPro DB2/3/4/6/7 e Hyser, che si accendono solo se V2 <= 2
   (stesso riferimento e stessa scala; X = 2 invariata). Vale anche per i dataset futuri. L'estensione del confronto V2 a quei dataset e' autorizzata
   da questa firma (misura: 1 GPU, stima < 0,5 GPU-ora = < 4 ore locali sui 30 del passo; la stima definitiva si ridichiara prima del lancio).
2. **Cosa si predice:** solo il **livello 0**.
3. **Quali rami:** regola di idoneita' **congelata**. Per ogni ramo, solo livello 0, sonda dataset-ID (stessa sonda e stesso split per soggetto di V3)
   sulle classi accese, contro le 5 potenze di banda sulle **stesse** classi: **idoneo se la differenza e' <= 10 punti**; l'ancora predice il livello 0
   di tutti i rami idonei, uno per testa. **Se nessun ramo e' idoneo l'ancora RVQ non parte** e si torna da Simone (candidato: il target discreto
   proprio). La stabilita' **non** e' un criterio di selezione: si riporta e si accetta come rumore di etichetta dichiarato.

**Prima di questa firma non e' stata lanciata nessuna misura** dell'identita' per singolo ramo (verificato: non esiste alcun risultato in `results/`
per quella misura). Le misure partono quando il cluster torna e gli array del run 59048369 sono copiati: nell'ordine (1) sonda per ramo sulle classi
accese (CPU), (2) V2 sui dataset non ancora misurati (GPU, dopo conferma con costo), (3) di nuovo (1) sui dataset che si accendono, come conferma.
Il criterio di D5a (V1-V4) non cambia; l'ablation run 4 contro run 5 resta l'arbitro dell'utilita' dell'ancora. Scadenza D9: 25/10.

---

## D5b — correzione della regola dei rami (Simone, 30/09/2026: «sì approvo la correzione del punto 7»)

Emersa dalla review del 30/09/2026: la regola firmata premiava i rami su cui la sonda **non** riconosce il dataset, quindi una sonda debole (istogramma
di 256 codici su 8192) rendeva piu' facile passare. **Correzione, approvata prima di qualunque misura** (non esiste nessun risultato della sonda per ramo):
per ogni ramo si usano due rappresentazioni del livello 0, l'istogramma dei codici e la **media dei vettori del codebook** scelti per i 256 token
(`quantize_b.layers.0.embedding.weight`, 128 numeri, standardizzati sul solo train), e si prende **l'accuratezza piu' alta** delle due. Stessa sonda e
stesso split per soggetto di V3; soglia invariata (differenza dalle 5 bande <= 10 punti). Testo completo in `docs/proposta_ancora_rvq.md`, sezione
«Correzione della regola dei rami». Resta aperto il buco segnalato sopra: cosa succede se le misure non sono pronte entro il 25/10.

---

## D5b — esito della regola dei rami: si predice il livello 0 del RAMO 0 (applicazione meccanica, 30/09/2026)

Regola firmata e corretta prima di qualunque misura (`docs/proposta_ancora_rvq.md`), applicata con `scripts/rvq_branch_probe.py` il 30/09/2026 sul Mac,
sugli array del run 59048369 copiati da Leonardo (sha256 verificati cluster = Mac). Risultato: `results/step1bis/rvq_branch_probe.json`.
**Non e' una nuova decisione:** e' l'esito della regola congelata.

**Controlli superati:** codebook del livello 0 estratto dal checkpoint (sha256 verificato) che riproduce il 100% dei codici del run in tutti i rami e
dataset; V3 congelato riprodotto esattamente (5 bande 0,655, codici 0,964).

Classi accese: camargo2021, capgmyo, emg2pose, grabmyo (312 unita', 60 di test, caso 0,25). 5 bande: **0,482** (IC 95% 0,346-0,623).

| Ramo | Istogramma | Vettore medio | Accuratezza del ramo | Ramo − bande | Esito |
|---|---|---|---|---|---|
| 0 | 0,518 | 0,482 | 0,518 | +0,036 | **idoneo** |
| 1 | 0,857 | 0,750 | 0,857 | +0,375 | escluso |
| 2 | 0,679 | 0,732 | 0,732 | +0,250 | escluso |
| 3 | 0,929 | 0,929 | 0,929 | +0,446 | escluso |

**Quindi l'ancora RVQ parte e predice solo il codice del livello 0 del ramo 0**, sui dataset accesi.

Da tenere presente, **senza cambiare l'esito** (la regola usa la stima puntuale, come V3):
- le stime sono larghe: IC 95% del ramo 0 (istogramma) 0,411-0,641, delle bande 0,346-0,623; con 60 unita' di test il margine dalla soglia (0,064) e'
  dentro l'incertezza;
- il ramo 0 e' il meno riconoscibile anche nelle feature continue (0,655 contro 0,988 di tutte le feature insieme: `results/step1bis/continuous_features.json`);
- **etichette rumorose, dichiarate:** al rumore di V4 il codice del livello 0 del ramo 0 resta uguale nel 46% dei token su emg2pose, 33% su CapgMyo, 36% su
  Camargo, **11% su GRABMyo** (`RIEPILOGO.md`, V4). La stabilita' non era un criterio di selezione (proposta, punto 3);
- per i dataset spenti fino a V2 (Kaifosh, NinaPro, Hyser) la regola si **riapplica** come conferma quando si accendono.

---

## V2 sui dataset nuovi — collaudo e run vero autorizzati (Simone, 30/09/2026: «sì autorizzo collaudo e run vero»)

Applica il punto «Dove» della proposta firmata (`docs/proposta_ancora_rvq.md`): V2 su Kaifosh, NinaPro DB2/3/4/6/7 e Hyser, con lo stesso riferimento
emg2pose, lo stesso seed e la stessa soglia X = 2. Script invariato (`scripts/slurm/step1bis.sbatch`), dati da `$SCRATCH/data/processed`.

1. **Collaudo** su `boost_qos_dbg` (30 min al massimo, 1 GPU): `--smoke --n-groups 6 --v1-windows 4 --skip-v3 --max-sessions-per-subject 1 --datasets
   ninapro_db4 ninapro_db6 hyser kaifosh`. Costo previsto 0,1-0,2 GPU-ora (1-1,5 ore locali), massimo 0,5 GPU-ora (4 ore locali).
2. **Run vero**, solo se il collaudo e' pulito: parametri di D5a (78 gruppi, al piu' 2 sessioni per soggetto, 3 semi), `--datasets kaifosh ninapro_db2
   ninapro_db3 ninapro_db4 ninapro_db6 ninapro_db7 hyser`, limite 1 h 30. Costo previsto 0,25-0,5 GPU-ora (2-4 ore locali), massimo 1,5 GPU-ora (12 ore
   locali). Budget del passo 1-bis: 30 GPU-ora, usata finora meno di 1.

**Collaudo pulito (scritto prima di lanciarlo):** job COMPLETED con exit 0; tutti e 4 i dataset analizzati, nessuno escluso per errore; controllo a rumore
zero di V4 = 1,0 su tutti (lo script si ferma altrimenti); report JSON scritto. Se una condizione fallisce il run vero **non parte** e si riporta il motivo.
**Controllo del run vero (scritto prima):** calibrazione, V1 ed emg2pose devono riprodurre il run 59048369: fattore 21,72, errore mediano di emg2pose
0,0352 (entro l'1%), rapporto di V1 1,23. Se non tornano, i rapporti V2 dei dataset nuovi non sono confrontabili: nessun dataset si accende e si riporta.
**Esito:** un dataset con rapporto V2 <= 2 accende l'ancora; sui dataset accesi si riapplica poi la regola dei rami come conferma (CPU, sul Mac).

---

## D5b — conferma della regola dei rami sui dataset nuovi: opzione (b), un dataset alla volta (Simone, 30/09/2026: «scelgo b»)

Chiude il buco della proposta firmata («sui dataset nuovi la si riapplica come conferma», senza dire cosa succede se la conferma non torna). Scritta
**prima** del run di V2 sui dataset nuovi e prima di qualunque sonda su di essi.

- **Base, che non cambia:** l'esito di oggi. Ancora RVQ sul livello 0 del ramo 0, accesa su camargo2021, capgmyo, emg2pose, grabmyo (piu' emg2qwerty,
  dato del tokenizer, quando sara' ingerito).
- **Ogni dataset nuovo con V2 <= 2 si prova DA SOLO**, indipendentemente dagli altri e dall'ordine: classi = le 4 della base + quel dataset; stessa sonda,
  stessa regola (livello 0 del ramo 0, il piu' alto fra istogramma e vettore medio, meno le 5 bande sulle stesse classi, <= 10 punti). **Se il ramo 0
  resta idoneo il dataset entra** (ancora accesa); altrimenti resta spento e la base non cambia.
- Gli altri rami (1-3) non si riprovano: la base li ha esclusi e la regola non li riammette.
- L'insieme finale (base + tutti i dataset entrati) **non** si riprova tutto insieme: sarebbe l'opzione (a), scartata.
- Split per soggetto: quello di V3, calcolato come oggi; lo split dei 4 dataset della base deve restare identico a quello di oggi (i nomi dei dataset
  nuovi vengono dopo in ordine alfabetico; il codice lo verifica e si ferma se non e' cosi').
- Costo: CPU sul Mac, qualche minuto per dataset, nessun costo di cluster.

---

## Buchi di un canale (valore identico per >= 1 s): marcati ed esclusi, non cancellati (Simone, 30/09-01/10/2026)

**Perche'.** Il run vero di V2 sui dataset nuovi (job 59073207) e' crollato su NinaPro DB4 dopo 14 min 33 s (0,24 GPU-ora): una finestra estratta era
tutta a valore costante e l'errore di ricostruzione non e' definito. Kaifosh, DB2 e DB3 erano gia' stati misurati (rapporti 1,05, 1,76, 1,93); il
controllo di confrontabilita' col run di ieri era passato (fattore 21,718, emg2pose 0,0352, V1 1,23). In DB4 (copia sul Mac, hash verificato) ci sono 11
tratti in cui un canale marcato valido vale **esattamente 0** per 3,75-22,5 s (5 soggetti su 10). Verificato con le etichette, su richiesta di Simone:
durante ogni tratto gli altri 11 canali registrano normalmente e in vari casi il soggetto sta eseguendo un movimento per tutta la durata (`stimulus`
diverso da 0). **Non e' riposo: e' un buco di un solo canale** (ipotesi non verificata: pacchetti persi degli elettrodi wireless). Distribuzione dei tratti a
valore identico in DB4: quasi tutti < 10 ms; 50 ms-1 s: circa 270, natura non chiara, **non toccati**; >= 1 s: 14, tutti a 0.

**Decisione di Simone** («ti autorizzo a tutte e 3, dopo che hai verificato la cosa delle etichette»; etichette verificate il 01/10):
1. **Regola, scritta prima di applicarla:** un tratto e' un buco se un canale **QC-valido** ha il valore int16 **identico** (differenza zero fra campioni
   consecutivi) per **almeno 1 secondo** alla frequenza nativa. Non «poco variabile»: identico. Il riposo vero non lo soddisfa.
2. **Marcare ed escludere, mai cancellare:** i tratti si scrivono nel `metadata.json` della sessione (`constant_runs`, con una voce in `qc_revisions`),
   il file dei dati non si tocca; le finestre che li toccano (con un margine di una patch da 200 ms per lato, per il transitorio del filtro) non si
   estraggono, ne' per V2 ne' per l'addestramento.
3. **Ricerca in tutti i dataset**, nello stesso job della misura del QC (sola lettura, poi scrittura dei soli sidecar). Freno di prudenza, scritto ora: se
   in un dataset i buchi coprono piu' dell'1% del tempo dei canali validi, per quel dataset non si scrive nulla e si riporta. Se si trovano buchi nei 5
   dataset del run di ieri, si riporta prima di qualunque rilancio che li riguardi.

**Poi:** rilancio di V2 sui 7 dataset nuovi con gli stessi parametri (autorizzazione del 30/09), salvando gli array dopo ogni dataset e non solo alla fine.

**Esito della scansione in sola lettura (job 59076768, 01/10/2026, 47 min, CPU; report `$WORK/wearusfm_runs/results/passo2/qc_relative_check_59076768.json`,
copia sul Mac con sha256 verificato):**
- **Controllo pulito:** i dataset gia' ingeriti con la regola relativa (NinaPro DB2/3/4/6/7) danno zero canali piatti nuovi.
- **Canali piatti nuovi:** solo CapgMyo, 3 canali in 3 sessioni su 18 (s06 canale 28, s08 e s11 canale 102; deviazione standard 0,00088-0,00098 volte la
  mediana, appena sotto la soglia 0,001). **Il freno scatta** (3/18 sessioni > 5%): per CapgMyo non si scrive nulla, ne' canali ne' buchi. **Revisione di
  Simone richiesta.**
- **Buchi >= 1 s** (tutti sotto l'1%, frazione massima 0,03%): NinaPro DB4 14 in 5 sessioni (un canale per volta); Hyser 6 in 5 sessioni (un canale, 1,1-1,4 s);
  CapgMyo s09 canale 40 per 10 prove intere da 1 s; **Camargo ab06/10_09_18 e NinaPro DB3 s03: tutti i canali insieme** (Camargo 1,2 s e 13,2 s; DB3 8,8 s e
  1,5 s), cioe' pause dell'intera registrazione, non buchi di un canale. Nessun buco negli altri 8 dataset.
- **Camargo e CapgMyo sono dataset del run del 29/09** (e della base della regola dei rami): segnalato qui come previsto; nessun rilancio che li riguardi.
- **Prossimo passo (autorizzato):** scrittura dei buchi nei sidecar di Camargo, Hyser, NinaPro DB3 e DB4 (`--apply --only ...`), poi rilancio di V2.

---

## V2 sui dataset nuovi e conferma della regola dei rami: esito (01/10/2026, applicazione meccanica)

**Run** 59078235 (24 min 49 s, 1 GPU, 0,41 GPU-ora, circa 3,3 ore locali), dopo la scrittura dei buchi (job 59077887). Report in
`results/step1bis/step1bis_59078235.json`; array sul Mac (`~/wearusfm_local/arrays_59078235/`, sha256 verificati). **Confrontabile col run del 29/09:**
stesso checkpoint, fattore 21,718, errore di emg2pose 0,0352, V1 1,231, tutti identici; l'array di emg2pose ha lo stesso sha256 di quello del 29/09.

**V2 (soglia 2, invariata), tutti passano:** Kaifosh 1,05 · NinaPro DB2 1,76 · DB3 1,93 · DB4 1,66 · DB6 1,84 · DB7 1,51 · Hyser 1,36.

**Conferma, opzione (b)** (`scripts/rvq_branch_confirm.py`, risultato `results/step1bis/rvq_branch_confirm_59078235.json`): ogni dataset provato da solo con
le 4 classi della base. Livello 0 del ramo 0 contro le 5 bande, stesse classi: Hyser 0,414 contro 0,557 · Kaifosh 0,443/0,657 · DB2 0,371/0,671 · DB3
0,400/0,643 · DB4 0,414/0,543 · DB6 0,429/0,529 · DB7 0,400/0,557 (caso 0,20). **Tutti entrano** (il ramo 0 riconosce il dataset meno delle 5 bande, da
-0,10 a -0,30). Split della base invariato (verificato dal codice).

**Ancora RVQ (livello 0, ramo 0) accesa su:** camargo2021, capgmyo, emg2pose, grabmyo, hyser, kaifosh, ninapro_db2, ninapro_db3, ninapro_db4, ninapro_db6,
ninapro_db7, piu' emg2qwerty quando sara' ingerito. **Spenta su:** putEMG, CSL-hdemg (V2), NinaPro DB5 (200 Hz).

Da tenere presente, senza cambiare l'esito:
- **Etichette rumorose** (V4, codice del ramo 0 al livello 0 invariato sotto rumore): Kaifosh 0,47 · DB2 0,45 · DB3 0,62 · DB4 0,38 · DB6 0,84 · DB7 0,35 ·
  Hyser 0,43 (base: emg2pose 0,46, CapgMyo 0,33, Camargo 0,36, GRABMyo 0,11).
- **Camargo e CapgMyo** della base sono stati misurati il 29/09, prima che i buchi fossero esclusi (Camargo ab06: 158 s su tutti i canali; CapgMyo s09:
  10 s su un canale): effetto atteso trascurabile, non misurato.
- **CapgMyo:** 3 canali quasi morti e un buco non scritti (freno scattato): in attesa di Simone.
- Costo del passo 1-bis oggi: collaudi 0,05 GPU-ora, run fallito 0,24, rilancio 0,41: circa 0,7 GPU-ora; in tutto il passo resta sotto 2 dei 30.

---

## CapgMyo: scrittura forzata (Simone, 01/10/2026: «sì forza capgmyo»)

Il freno era scattato per i canali piatti (3 sessioni su 18 > 5%). Simone autorizza la scrittura: canali **s06/28, s08/102, s11/102** marcati non validi
(deviazione standard 0,00088-0,00098 volte la mediana) e il buco di **s09 canale 40** (10 prove intere da 1 s). Strumento: `qc_relative_check.py --only
capgmyo --apply --force-dataset capgmyo`, con backup e `qc_revisions`. CapgMyo e' nella base del run del 29/09: si dichiara, nessun rilancio.

## Download di NinaPro DB8 e DB10: dimensioni e causa del mancato download (01/10/2026)

Condizione della decisione del 30/09: riportare la dimensione prima del lancio e fermarsi se servono termini o account.
- **DB8:** la pagina ufficiale linka **36 file `.mat`** (12 soggetti x 3 acquisizioni), **25,4 GB** (somma dei Content-Length). Nessuna registrazione.
  **Causa del mancato download:** `download_ninapro.sbatch` prendeva solo i link `.zip`. Si scarica su `$SCRATCH` ($WORK e' pieno).
- **DB10 = MeganePro** su Harvard Dataverse (4 DOI dalla pagina ufficiale): MDS1 240 GB (163 GB di video `.mp4`, 77 GB di `.mat`), MDS2 33 GB (33 GB
  di video, 0,4 GB di `.mat`), MDS4 18 GB (17,5 GB di video, 0,24 GB di `.mat`), MDSInfo (interviste cliniche e test neurocognitivi degli amputati, un
  file). Termini d'uso del dataset: «This work is licensed under a Creative Commons Attribution 4.0 International License»; nessun file riservato, nessun
  guestbook: **nessuna condizione da accettare**. **Proposta di AG, da confermare:** scaricare solo i `.mat` (circa 78 GB), non i video e non MDSInfo
  (dati clinici, non servono al corpus). Quali `.mat` contengano l'EMG (e a che frequenza, fatto n. 1) va verificato sui file.

## DB8 e DB10: decisioni di Simone (01/10/2026)

- **DB8: opzione A** («A per DB8»). I file sono a 2 kHz ma l'EMG e' stato acquisito a 1111 Hz e sovracampionato (pagina ufficiale: «The sEMG signals were
  sampled at a rate of 1111 Hz, accelerometer and gyroscope data were sampled at 148 Hz, and magnetometer data were sampled at 74 Hz. All signals were
  upsampled to 2 kHz and post-synchronized.»). Si tiene la griglia a 2 kHz come fornita, **dichiarando nel sidecar la banda effettiva fino a 555 Hz**
  (Nyquist di 1111 Hz); le ancore spettrali sopra 555 Hz si mascherano per DB8, come per le bande oltre Nyquist (v10 §4.2). Nessun ricampionamento.
- **DB10: solo i file `.mat`** di MeganePro MDS1, MDS2, MDS4 (circa 78 GB); **non** i video `.mp4` e **non** MDSInfo (interviste cliniche). Destinazione
  `$SCRATCH`. Quali `.mat` contengano l'EMG e a che frequenza si verifica sui file; se sono tabelle MATLAB si convertono sul Mac con MATLAB (Simone,
  01/10: «se ti serve matlab per convertirli puoi usare il mac in locale»).

---

## D8a — definizioni operative del gate di consistenza: FIRMATE e CONGELATE (Simone, 01/10/2026: «firmo D8: 1a, 2, 3, 4a, 5, 6»)

Completa D8a (soglie congelate il 29/09) con le tre cose da scrivere prima del lancio. Testo completo in `docs/proposta_gate_d8.md`.
- **Dataset:** Kaifosh (2 kHz). **Due casi**, entrambi devono passare: 450 Hz su griglia a 2 kHz contro decimato a 1 kHz; 90 Hz su griglia a 2 kHz contro
  decimato a 200 Hz; filtro prima di decimare (Butterworth ordine 8 a fase zero nella prima firma, **sostituito dal passa-basso a muro**: correzione
  firmata sotto). **Banda condivisa:** sotto il taglio del caso; patch allineate in tempo fisico.
- **Errore relativo RMS <= 5% nel totale E in ciascuna delle tre famiglie** di kernel (Fourier, spline, MLP).
- **Sonda A-contro-B:** finestre di 4 s, 20 per soggetto, media e deviazione standard delle feature per finestra; split per soggetto 60/20/20 con le due
  versioni di una finestra nella stessa parte; la piu' forte fra regressione logistica e gradient boosting (scelta sulla validazione); intervallo al 95%
  con bootstrap per soggetto (1000). Passa se <= 0,55 e l'intervallo contiene 0,50.
- **Pesi:** front-end all'inizializzazione, 3 semi, tutti devono passare. **Costo:** CPU, zero GPU-ora.
- **Prerequisito:** il front-end a kernel continui di v10 §4.2 non esiste ancora nel codice; va scritto e testato (fattore Δt, anti-aliasing per famiglia)
  prima del gate. Firmato prima di scriverlo e prima di qualunque misura (non esiste `results/step3/`).

### D8a — difetto trovato nel filtro firmato, prima del gate sui dati veri (01/10/2026): correzione FIRMATA (Simone, 01/10/2026: «firmo la correzione del filtro, lancia il gate»)

Una prova di funzionamento dello script del gate su **dati sintetici** (rumore filtrato 20-400 Hz, 6 soggetti finti; nessun dato vero) ha mostrato che
la frase della proposta firmata «si filtra prima di decimare, quindi A e B hanno esattamente lo stesso contenuto» e' **falsa** col filtro firmato. Un
Butterworth di ordine 8 con taglio a 90 Hz lascia passare ancora il 39% dell'ampiezza a 100 Hz e il 2% a 150 Hz: decimando a 200 Hz (Nyquist 100 Hz) quel
contenuto si ripiega sotto (aliasing) e B non ha piu' lo stesso contenuto di A. Il gate misurerebbe un difetto creato dalla prova, non dal front-end.
Misure sui dati sintetici, front-end seme 0:

| filtro per costruire A | caso 1 kHz | caso 200 Hz |
|---|---|---|
| Butterworth ordine 8 (firmato) | 0,84% | 5,7% |
| passa-basso a muro nel dominio della frequenza (FFT: zero esatto sopra il taglio) | 0,00% | 0,37% |

**Proposta di AG (da firmare):** costruire A con il passa-basso **a muro** (trasformata della finestra con i suoi margini, zero esatto da 450 o 90 Hz in
su, antitrasformata), poi B = A decimata. Cosi' A non ha contenuto sopra il taglio e B lo stesso contenuto su una griglia diversa, come chiede v10 §7.1.
Tutto il resto delle definizioni firmate resta com'e'. Il gate sui dati veri **non e' stato lanciato**.

**Firmata da Simone il 01/10/2026** («firmo la correzione del filtro, lancia il gate»), prima di qualunque misura su dati veri: A si costruisce col
passa-basso a muro (`make_versions` in `src/wearusfm/gate/consistency.py`; test: nulla sopra il taglio). Due correzioni allo script, trovate rileggendo
il percorso dei dati prima del lancio, che non cambiano le definizioni: (1) le finestre si estraggono un soggetto alla volta (prima tutto Kaifosh
restava in memoria in float64, ~30 GB: il job sarebbe stato ucciso); (2) «20 finestre per utente» si conta per soggetto e non per segmento (prima un
soggetto con piu' segmenti ne avrebbe avute 20 per segmento; su Kaifosh, una registrazione per utente, non cambia nulla). Il gate si lancia con
`scripts/slurm/gate_d8.sbatch` (CPU seriale, zero GPU-ora).
Primo lancio (job 59086063, 01/10/2026 10:03): **fuori memoria** dopo 71 s (16 GB), prima di qualunque misura. Causa: ogni finestra era una *vista*
sulla sessione intera, che restava quindi in memoria (~700 MB in float64 per soggetto). Corretto (le finestre sono copie; test che cattura la versione
con la vista); picco misurato sul Mac con sessioni di dimensione vera: 2,2 GB con 2 soggetti, 2,4 GB con 5. Rilanciato una volta, stesse definizioni.
**Esito (dato, 01/10/2026, job 59087481): gate SUPERATO** in entrambi i casi con tutti e 3 i semi (errore massimo 0,43% nel caso 200 Hz;
sonda fra 0,500 e 0,505). Dettagli e note in `results/step3/RIEPILOGO.md`.

---

## NinaPro DB8 — parser scritto (01/10/2026): scelte di AG; gruppi sparsi per gli amputati APPROVATI (Simone, 01/10/2026: «sì autorizzo collaudo e pieno, sparsi va bene»)

Parser `src/wearusfm/ingest/ninapro_db8.py` (opzione A firmata: griglia a 2 kHz, banda effettiva dichiarata 0-555,5 Hz; sidecar con
`acquisition_fs_hz` = 1111). Fonti verificate parola per parola: `fatti_da_verificare.md`, fatto n. 1 (parte DB8, da firmare). Scelte NON dalla fonte:
- un soggetto = una sessione, le tre acquisizioni concatenate come prove (`trials` con `acquisition`); il sidecar segna l'acquisizione 3 come test
  raccomandato dal fornitore;
- colonne 0-7 = una riga, 8-15 = l'altra (ordine non dichiarato; quale riga sia a 3 o a 5,5 cm dal gomito non e' dichiarato);
- normodotati: due anelli D_8 (equispaziatura dichiarata per loro); **amputati 11 e 12: due gruppi sparsi senza angoli e anatomia nominale**, perche'
  la fonte dice solo «a similar configuration» con 13 e 12 sensori. Alternativa: anelli D_8 anche per loro, con le colonne mancanti scartate dal QC;
  **motivazione di Simone (01/10/2026):** «si considera che rispetto ai normodotati dal punto di vista anatomico gli amputati hanno fasce muscolari
  irregolari, quindi va bene sparso»;
- rete 50 Hz per deduzione (Regno Unito), non dichiarata.
Misura sui file (S1 A1 e A3, S11 A1, primi 60 s): la potenza sopra 555 Hz e' fra lo 0,3% e il 3% per canale, non zero. Il report di ingest la
riporta per soggetto.

## DB8: buchi segnati nei sidecar; emg2pose: ingest autorizzato (Simone, 01/10/2026: «sì autorizzo apply su DB8 e collaudo e pieno di emg2pose»)

- **DB8, QC** (job 59090748, sola lettura): 0 canali piatti nuovi; 153 buchi in 9 sessioni su 12, 167 s = 0,035% del tempo dei canali validi (guardia: 1%),
  (153 tratti PER CANALE e 167 CANALE-secondi; in tutto circa 11 eventi simultanei su tutti i canali validi, lunghi 1,03-1,18 s). In s01 il buco finisce esattamente alla fine dell'acquisizione 2 (coda del file
  tenuta ferma); ipotesi, non verificata sulle altre sessioni: stessa origine per tutti. Si scrivono nei sidecar con `--apply` (copia di sicurezza
  `metadata.pre_qc_revision.json`, dati intatti). **Applicato** (job 59092325, 01/10/2026): buchi scritti in 9 sessioni, 9 copie di sicurezza.
- **emg2pose:** collaudo su 3 registrazioni, poi il pieno con `--skip-existing`, riprendibile (budget 2 h 40 min per job, <= 6 ore locali ciascuno;
  stima 1-2 job, ~12 ore locali in tutto, 0 GPU-ora; processato ~100 GB su `$SCRATCH`).

---

## NinaPro DB10 (MeganePro): cosa si ingerisce e come (01/10/2026) — MDS2/MDS4 fuori e ingest di MDS1 autorizzato (Simone, 01/10/2026: «sì, MDS2/MDS4 fuori e autorizzo collaudo e pieno di DB10»)

Dati e fonti: `fatti_da_verificare.md`, fatto 10c (citazioni controllate). Ispezione dei file via leonardo-ops (`scripts/inspect_mat.py`, `whosmat`).
- **MDS1 (esercizio 1, prese):** l'EMG vero, 12 canali Delsys a 1926 Hz (1925,98 dai `ts`), 45 soggetti (30 normodotati, 15 amputati
  transradiali S101-S115), ~56 h. Si ingerisce solo `S<NNN>_ex1.mat`; i `_orig` non contengono EMG.
- **MDS2 e MDS4: FUORI** (confermato da Simone), per la regola gia' scritta in v10 §2.4 (un inviluppo non entra nel front-end, come DB1): la fonte dice
  «rectified via a moving root-mean-square with a window length of 300 samples», nei file a 100 Hz, con 2 elettrodi (uno per avambraccio). Sono
  esercizi di immaginazione motoria e puntamento in cui l'EMG era secondario («the sEMG would serve for control analyses»).
- **Montaggio (scelte di AG, nel parser):** normodotati: anello prossimale D_8 (colonne 1-8, angoli 0-315) e distale D_4 a 45 mm (colonne 9-12, «aligned with the gaps between
  electrodes one and two, three and four»: angoli 22,5, 112,5, 202,5, 292,5); S024 (elettrodo 8 assente): anello prossimale con le posizioni 1-7;
  S039 (7 elettrodi prossimali, spaziatura non dichiarata): sparso; **amputati: sparsi e nominali** (stessa scelta di DB8); lato: destro per i
  normodotati, ignoto per gli amputati (il lato del moncone e' solo nella Tabella 1, non verificata cella per cella).
- **Pause:** la registrazione di un soggetto ha decine di pause nell'asse dei tempi (S010: 50, la piu' lunga 166 s): si spezza in segmenti, come prove.
- Rete 50 Hz (filtro di Hampel a 50 Hz dichiarato; Svizzera e Italia). Un canale con ampiezza molto piu' bassa degli altri (S010, canale 7: 3e-7 V
  contro 5e-6-5e-5) passa il QC relativo: il report elenca i canali sotto il 5% della mediana come «sospetti», senza scartarli.

---

## V2 e conferma dei rami su NinaPro DB8, Zhang e DB10 (Simone, 01/10/2026: «lancia le misure RVQ su DB8, DB10 e Zhang»)

Stessa procedura del run 59078235 (D5b punto «Dove», opzione b), scritta PRIMA dei lanci; soglie invariate (V2: rapporto <= 2 con X = 2 sul riferimento
emg2pose; ramo: livello 0 del ramo 0 entro 10 punti dalle 5 bande, sulle classi della base piu' il dataset nuovo).
1. **Collaudo** su `boost_qos_dbg` (30 min, 1 GPU): `--smoke --n-groups 6 --v1-windows 4 --skip-v3 --max-sessions-per-subject 1 --datasets ninapro_db8
   zhang2026 --processed-root $SCRATCH/data/processed`. Costo previsto 0,1-0,2 GPU-ora, massimo 0,5. **Pulito se:** exit 0, entrambi i dataset analizzati,
   controllo a rumore zero di V4 = 1,0, report scritto.
2. **Run vero** (78 gruppi, al piu' 2 sessioni per soggetto, 3 semi), limite 1 h 30, costo previsto 0,2-0,4 GPU-ora, massimo 1,5: **DB8 subito dopo il
   collaudo; Zhang dopo che i suoi buchi sono stati misurati (job 59099080) e, se ce ne sono, scritti; DB10 dopo il suo ingest.** Il run vero di DB8 e
   Zhang puo' essere uno solo se Zhang e' pronto. **Controllo del run vero:** fattore 21,72, errore mediano di emg2pose 0,0352 (entro l'1%), V1 1,23;
   se non tornano, nessun dataset si accende.
3. **Conferma dei rami** (CPU, `scripts/rvq_branch_confirm.py`) sui dataset con V2 <= 2.
Budget del passo 1-bis: 30 GPU-ora, usati finora meno di 2.

---

## Dati su $SCRATCH: rinfresco periodico delle date (proposta di AG, 01/10/2026; rinfresco subito AUTORIZZATO da Simone: «sì, autorizzo il rinfresco subito su tutto $SCRATCH»)

**Trovato il 01/10/2026** (scansione in sola lettura delle date su `$SCRATCH`): il tar grezzo di emg2qwerty (308 GB) ha data di modifica 25/08/2021 e
i 36 `.mat` di NinaPro DB8 date dal 13/08/2019 (`wget` conserva la data del server): se il purge a 40 giorni conta dalla data di modifica (informazione
di Simone, non verificata), sono **gia' esposti**. Il resto del raw e' del 23-29/09, il processato del 29/09-01/10. Quasi tutto il processato sta su
`$SCRATCH` ($WORK e' pieno), e un manifest congelato (D9) deve puntare a dati che restano per tutta la ladder.
**Proposta:** `scripts/slurm/refresh_scratch.sbatch` (solo `touch` di file e cartelle, con conteggio prima e verifica dopo; non cancella nulla) su
`data/raw`, `data/processed`, `wearusfm_runs` ed `external`: **subito**, poi **ogni 21 giorni** per tutta la durata della ladder (22/10, 12/11, ...).
Costo: CPU seriale, 1 core, minuti; 0 GPU-ora. Il rinfresco non sostituisce le copie. **Correzione (review del 01/10):** non e' vero che i dati difficili da riottenere "restano anche su $WORK":
il raw di emg2qwerty, emg2pose, DB8 e DB10 e quasi tutto il processato stanno SOLO su $SCRATCH (riscaricabili, ma con giorni di lavoro).
**Primo rinfresco fatto il 01/10/2026:** i 37 file gia' scaduti (tar di emg2qwerty, `.mat` di DB8) toccati subito dal login node; poi il job 59105021
(34 min 45 s, exit 0) su `data/raw`, `data/processed`, `wearusfm_runs`, `external`: **0 cartelle con problemi, nessun file piu' vecchio di un giorno
dopo il rinfresco**. Trovati altri file gia' esposti: i **48 file di riferimento di emg2pose per V2** (`wearusfm_runs/emg2pose_reference`) e i 31 di
`emg2pose_mini`, con date del 21/05/2024 (conservate dall'estrazione). **Prossimo rinfresco: 22/10/2026.** Limiti noti dello script (review del codice):
non tocca un bersaglio che sia un file, ne' elementi nascosti al primo livello, ne' il contenuto dietro un link a una cartella; da correggere prima del
prossimo giro.

---

## CapgMyo: seconda applicazione, senza effetto (01/10/2026) — CORREZIONE: la scrittura era gia' stata fatta (sezione «CapgMyo: scrittura forzata»)

Dalla scansione 59076768: 3 canali con deviazione standard 0,00088-0,00098 volte la mediana (s06 canale 28, s08 e s11 canale 102; circa 60 dB sotto gli
altri) e un buco (s09 canale 40, 10 prove intere da 1 s). Il freno era scattato (3 sessioni su 18 > 5%). Si applica con `--force-dataset capgmyo`:
copia di sicurezza dei sidecar, dati intatti. CapgMyo e' fra le classi della base della regola dei rami RVQ (misurata il 29/09 con quei canali validi):
l'effetto atteso sulle misure gia' fatte e' trascurabile (3 canali su 128 in 3 sessioni su 18) e non si rimisura.
**Correzione (AG, 01/10/2026):** AG aveva riproposto la decisione come «in sospeso» leggendo solo la riga «in attesa di Simone» del run 59078235, senza
vedere la sezione «CapgMyo: scrittura forzata» (Simone: «sì forza capgmyo»), applicata prima. Il job 59105224 (`--apply --force-dataset capgmyo`)
ha trovato 0 canali piatti nuovi e 0 buchi nuovi (6 sessioni gia' con canali non validi, 4 copie di sicurezza, 1 sessione con buchi): **non ha
scritto nulla**. Lo stato dei sidecar resta quello della prima applicazione.

### V2 e conferma dei rami su DB8 e Zhang: esiti

Collaudo 59103405 (DB8 + Zhang, `--smoke`, 5 min 46 s, ~0,1 GPU-ora = ~0,8 ore locali): pulito.

**Esito per NinaPro DB8 (01/10/2026, applicazione meccanica):** run vero 59104658 (163 s, 1 GPU, ~0,05 GPU-ora = ~0,4 ore locali; report in
`results/step1bis/step1bis_59104658.json`, array sul Mac in `~/wearusfm_local/reports/step1bis/arrays_59104658/`, sha256 verificati). **Controlli scritti
prima: tutti identici** (fattore 21,718; errore mediano di emg2pose 0,03520; V1 1,231; stesso checkpoint). **V2 di DB8: 1,46 <= 2, passa.** Conferma del
ramo (opzione b, `results/step1bis/rvq_branch_confirm_59104658.json`): livello 0 del ramo 0 0,431 contro 0,723 delle 5 bande sulle stesse classi
(differenza -0,29): **ENTRA**. **Ancora RVQ accesa anche su ninapro_db8.** Zhang: run vero dopo i suoi buchi; DB10: dopo il suo ingest.

---

## Buchi su Zhang ed emg2qwerty: esito (job 59099080, sola lettura, 01/10/2026)

Autorizzato da Simone il 01/10/2026 («sì parti con DB10 e autorizzo i buchi su Zhang ed emg2qwerty»: autorizzazione alla MISURA; una scrittura nei
sidecar, se servisse, si chiede a parte). **emg2qwerty:** 1135 sessioni, 0 canali piatti nuovi, 189 tratti costanti per canale in 14 sessioni, 312,5
canale-secondi (0,0008% del tempo valido dei canali), freni non scattati. **Zhang:** 124 sessioni, 0 canali piatti nuovi, **0 tratti costanti**: non
c'e' nulla da scrivere, e il run vero di V2 su Zhang puo' partire. Per emg2qwerty la scrittura dei 189 tratti e' da autorizzare.
Nota sul vocabolario: qui «buco» e' un tratto di valore costante >= 1 s su un canale; i «salti dell'asse dei tempi» (emg2qwerty 76 registrazioni,
emg2pose 1.668) sono un'altra cosa, gia' elencata nei sidecar.

**Esito per Zhang (02/10/2026, applicazione meccanica):** run vero 59108493 (153 s, 1 GPU, ~0,04 GPU-ora = ~0,3 ore locali; report in `results/step1bis/step1bis_59108493.json`, array sul
Mac in `~/wearusfm_local/reports/step1bis/arrays_59108493/`, sha256 verificati), lanciato dopo la misura dei buchi di Zhang (0 tratti costanti, job
59099080). **Controlli scritti prima: identici** (fattore 21,718; emg2pose 0,03520; V1 1,231; stesso checkpoint). **V2 di Zhang: 1,55 <= 2, passa.**
Conferma del ramo (`results/step1bis/rvq_branch_confirm_59108493.json`): livello 0 del ramo 0 0,371 contro 0,643 delle 5 bande (differenza -0,27):
**ENTRA. Ancora RVQ accesa anche su zhang2026.** Resta DB10, dopo la misura dei suoi buchi.

**Esito per DB10 (02/10/2026, applicazione meccanica):** run vero **59183099** (rilancio autorizzato da Simone dopo il fallimento del 59180438;
20 min 40 s, 1 GPU, ~0,34 GPU-ora = ~2,8 ore locali; report in `results/step1bis/step1bis_59183099.json`, array sul Mac in
`~/wearusfm_local/reports/step1bis/arrays_59183099/`, sha256 verificati), dopo la misura dei buchi di DB10 (0 tratti costanti, job 59173238).
**Controlli scritti prima: identici** (fattore 21,718; emg2pose 0,03520; V1 1,231; stesso checkpoint). Nessun dataset escluso (la correzione
delle prove da 1 campione ha funzionato). **V2 di DB10: 1,84 <= 2, passa** (45 soggetti, 78 gruppi, nessun gruppo mancante). Conferma del ramo
(`results/step1bis/rvq_branch_confirm_59183099.json`): livello 0 del ramo 0 0,386 contro 0,586 delle 5 bande (differenza -0,20): **ENTRA.
Ancora RVQ accesa anche su ninapro_db10** (`RVQ_ON` in `src/wearusfm/data/pretraining_manifest.py`). Chiuse le misure RVQ autorizzate il 01/10
(DB8, Zhang, DB10: entrano tutti e tre). Il report registra il commit d775778 (calcolato alla fine; il job e' partito a 68effc0): fra i due sono
cambiati solo documenti. Budget del passo 1-bis: 30 GPU-ora, usate circa 2,4.

---

## Tre decisioni del 02/10/2026 (Simone: «ok ti do il via», sulla proposta di AG)

1. **emg2pose: le 3.539 registrazioni (58,8 h dai tempi del CSV; 58,6 h di campioni nel manifest) del test ufficiale per fasi nuove** (`split = test`, `held_out_stage = True`, di utenti NON tenuti
   fuori) **escono dal pretraining** e restano per la valutazione, come gli utenti tenuti fuori: il benchmark ufficiale resta pulito. Lo split
   diventa per sessione oltre che per soggetto (`test_sessions` negli split).
2. **emg2qwerty: si scrivono nei sidecar i 189 tratti costanti** misurati dal job 59099080 (`qc_relative_check --only emg2qwerty --apply`, copia di
   sicurezza, dati intatti). **Fatto** (job 59133395, 1 h 42 min, exit 0): 189 tratti scritti in 14 sessioni, 312,5 canale-secondi
   (0,0008% del tempo valido), 0 canali piatti nuovi; report copiato con sha256 uguale.
3. **DB10 ed emg2pose: misura dei tratti costanti**, sola lettura (`qc_relative_check --only ninapro_db10 --only emg2pose`); una scrittura, se serve,
   si chiede a parte. Primo tentativo (job 59133397) in **TIMEOUT** a 2 h senza finire emg2pose (25.253 sessioni, 92 GB):
   nessun report, nulla scritto (sola lettura). Rilancio separato: DB10 da solo; emg2pose da solo con 4 h (massimo della coda seriale).
   **DB10 (job 59173238, 32 min 46 s, exit 0; report `qc_relative_check_59173238.json`, sha256 dec57c08... uguale su cluster e Mac):** 45 sessioni,
   **0 canali piatti nuovi, 0 tratti costanti** (0,0000% del tempo valido), freni non scattati, 0 errori di lettura: nulla da scrivere. Quindi,
   come da procedura («V2 e conferma dei rami su NinaPro DB8, Zhang e DB10»), sottomessi il 02/10 dal repo remoto pulito al commit 653c3fe:
   **run vero di V2 su DB10, job 59180438** (GPU, 0,05-0,4 GPU-ora stimate, autorizzato da Simone il 01/10) e **prima bozza del manifest, job
   59180449** (CPU seriale, 0 GPU-ora). emg2pose (job 59173239) ancora in corso.
   **V2 su DB10 (job 59180438): FALLITO** dopo 2 min 39 s (exit 1; 1 GPU, ~0,04 GPU-ora = ~0,35 ore locali). Controlli scritti prima
   **identici** nel report parziale (fattore 21,718; emg2pose 0,03520; V1 1,231, sha256 del parziale uguale su cluster e Mac). **Causa (verificata):**
   13 sessioni di DB10 hanno in tutto 18 «prove» da **1 campione** (tratti fra due pause, dall'ingest; le altre prove sono lunghe decine di migliaia
   di campioni, letto dai sidecar sul cluster); il filtro passa-banda (`sosfiltfilt`) rifiuta un segnale di 21 campioni o meno, il `ValueError`
   escludeva l'intero DB10, e la sonda V3, rimasta col solo emg2pose, crollava con una classe sola; il motivo dell'esclusione si perdeva perche'
   finiva nel report solo alla fine. **Correzioni (con test, suite CPU verde):** `to_canonical` scarta prima del filtro solo le prove troppo corte
   per il filtro (le sessioni che gia' funzionavano danno lo stesso risultato bit per bit); V3 con meno di 2 dataset dice «non calcolabile»;
   un'esclusione entra subito nel report parziale. **Non rilanciato** (regola del controllo automatico): il rilancio va autorizzato da Simone.
   **Rilancio autorizzato da Simone il 02/10/2026** («sì, autorizzo il rilancio di V2 su DB10»): **job 59183099**, stesso comando, limite 1 h 30,
   costo stimato 0,05-0,4 GPU-ora, sottomesso dal repo remoto pulito al commit 68effc0 (budget del passo 1-bis: 30 GPU-ora, usate circa 2).
   Nota sulla provenienza della bozza del manifest (job 59180449): e' partita al commit 653c3fe e il push di 68effc0 e' avvenuto mentre girava,
   quindi il `code_commit` che scrive alla fine puo' essere 68effc0; fra i due commit `scripts/build_manifest.py` e `src/wearusfm/data/` sono
   identici (controllato con `git diff --stat`), quindi la provenienza resta valida.
   **Bozza del manifest (job 59180449): TIMEOUT** a 1 h 03 min (limite 1 h; CPU, 0 GPU-ora): nessun manifest scritto. Il log contiene solo il
   caricamento dei moduli, perche' lo script stampava tutto alla fine: **dove fosse arrivato non si sa** (un agente ha concluso che il costruttore
   non fosse partito: il log non lo dimostra). *Ipotesi di AG, non verificata:* la lettura in serie di ~27.000 sessioni su Lustre (sidecar e
   intestazioni), mentre la misura di emg2pose leggeva gli stessi dati sullo stesso nodo. La stima «pochi minuti» era sbagliata. **Correzioni**
   (con test): una riga di log per dataset scritta subito, lettura su 8 thread (attesa di I/O; righe identiche a quelle in serie, test), limite
   4 h. **Non rilanciato**: il rilancio va autorizzato da Simone.
   **emg2pose (job 59173239, 3 h 17 min, exit 0; report `qc_relative_check_59173239.json`, sha256 31a44379... uguale su cluster e Mac):**
   25.253 sessioni, **0 canali piatti nuovi**, **2.786 tratti costanti in 576 sessioni** (79 utenti), 4.044,6 s = 0,0166% del tempo valido, freni
   non scattati, 0 errori di lettura. **Tutti i tratti sono sui canali 0 e 1** (1.437 e 1.349), durata mediana 1,3 s, massima 5,3 s. Controllo in
   sola lettura su 12 sessioni a caso (seme 0): **valore esattamente 0** in tutti i 45 tratti guardati e nessun campione a +-32767 sui canali 0-1:
   sono interruzioni del segnale, non saturazione della quantizzazione int16. *Osservazione, non verificata:* nelle stesse 12 sessioni i canali
   0 e 1 hanno massimi e minimi quasi identici (es. 1463/1474, -3016/-3008): possibili quasi-duplicati, da controllare con una correlazione.
   La scrittura (`--apply`) va autorizzata da Simone.
   **Autorizzate da Simone il 02/10/2026** («si»): scrittura su emg2pose, **job 59204293** (`--only emg2pose --apply`, limite 4 h, CPU, 0 GPU-ora),
   e poi la bozza del manifest, **job 59204297**, con `--dependency=afterok:59204293` (parte solo se la scrittura riesce). Repo remoto pulito a 329b119.
   **Scrittura su emg2pose fatta** (job 59204293, 3 h 40 min, exit 0; report `qc_relative_check_59204293.json`, sha256 aa84bf62... uguale su
   cluster e Mac): **2.786 tratti costanti scritti in 576 sessioni, 0 errori**, 0 canali piatti nuovi, freni non scattati; copia di sicurezza del
   sidecar per sessione (`metadata.pre_qc_revision.json`), dati intatti. La bozza del manifest (59204297) e' partita alle 18:03 con il registro
   di avanzamento per dataset.
   **Bozza del manifest fatta** (job 59204297, 16 min, exit 0; `manifest_59204297.json.gz`, 1,6 MB, sha256 924a4c8e... uguale su cluster e Mac,
   copia in `~/wearusfm_local/reports/passo4/`, NON nel repo; hash del contenuto 59e40dbc0831b158; commit registrato 25d1fc4, pushato durante il
   run: rispetto alla partenza sono cambiati solo `training/masking.py` e i documenti, non il codice del costruttore). Parametri della bozza D9:
   quote A 20 / B 75 / C 5, alpha 0,5, tetto 8 passaggi a 4 epoche, patch 25 ms. **Risultati:** 27.560 sessioni; **pretraining 822,9 h**
   (19.034 sessioni), test 210,9 h, benchmark Kaifosh 63,9 h; **D_t 118,5 M, D_c 3,25 G**; per classe pretraining A 90,7 h, B 710,8 h, C 21,3 h;
   quote realizzate A 20,0%, B 75,0%, C 5,0% (nessuna quota inassegnabile: con il tetto la classe C arriva al 5%, contro il timore del punto 1
   della revisione della bozza D9); 10 unita' al tetto di 8 passaggi e 5 fra 7,1 e 7,9 (corretto: nel primo resoconto a Simone AG aveva
   scritto 12, letto a occhio dal log); emg2pose 2,8 ed emg2qwerty 2,7 passaggi, NinaPro DB2 6,0. Nessun soggetto senza sessioni,
   nessuna sessione di test mancante. **Tempo non campionabile** (pretraining): con finestre fisse da 4 s la classe C perde 6,9 h su 21,3
   (CSL-hdemg e CapgMyo per intero, Hyser l'11%), con 2 s 2,1 h, con 1 s nulla; GRABMyo perde tutto solo a 8 s; la classe A non perde nulla.
   **Salti dell'asse dei tempi:** emg2pose 222.477 salti in 1.668 sessioni (3.432 s mancanti, massimo 4,1 s), emg2qwerty 27.017 in 76 (562 s,
   massimo **73,3 s**), Kaifosh 5.098 in 46 (benchmark); spezzare a ogni salto costa a 4 s l'1,2% di emg2pose e lo 0,2% di emg2qwerty (classe B:
   4,1 h). Nessuna sessione con l'elenco dei salti troncato, nessuna posizione fuori dall'array.

---

## Note dalla review dei documenti (02/10/2026)

- **Cancellazioni:** il 01/10 Simone ha scritto «se devi cancellare files e cose gia copiate e duplicate , ti autorizzo». AG **non** ha cancellato
  nulla: la cancellazione definitiva di dati e' fuori da cio' che AG fa anche con autorizzazione; gli originali in `$WORK/data/raw` si cancellano solo
  con `scripts/slurm/delete_verified_raw.sbatch` lanciato da Simone (`dataset_access.md`).
- **Autorizzazioni di misura e di scrittura:** «autorizzo i buchi su Zhang ed emg2qwerty» (01/10) e' stata letta come misura; la scrittura dei 189
  tratti di emg2qwerty e' stata autorizzata a parte il 02/10.
- Le correzioni puntuali segnalate (citazioni, numeri, interpretazioni marcate) sono in questo commit; la bozza D9 si riscrive coi numeri del costruttore.


---

## Passo 6: identita' di canale, primo modulo (02/10/2026, proposta di AG, da rivedere)

Codice: `src/wearusfm/model/channel_codes.py` (numpy: geometria relativa, vicini, codici anatomici dai sidecar) e
`src/wearusfm/model/channel_identity.py` (torch: embedding anatomico gerarchico additivo, concatenazione proiettata), con test sui montaggi veri
degli adattatori (anello emg2pose, sparso Camargo, griglia CapgMyo, anello piu' mirati NinaPro DB2). Non dipende da D10 ne' da D11.
**Interpretazioni di AG da confermare:**
1. il percorso «geometrico relativo» di v10 §5.2 non e' un vettore per canale: per canale entra solo la classe di topologia; gli spostamenti
   relativi (angolo sugli anelli, riga e colonna sulle griglie, solo dentro lo stesso gruppo) entrano a coppie nell'encoder locale, perche' un
   angolo assoluto romperebbe la simmetria ciclica (v10 §3.4) e l'orientamento della fascia e' spesso ignoto;
2. unita': radianti sugli anelli, passi di indice sulle griglie: i sidecar non portano distanze in mm, quindi il bias di §5.3 si apprende per
   topologia;
3. dropout del livello muscolo per canale; il valore p e' un parametro obbligatorio senza default (v10: «p ≈ 0,3–0,5, da fissare»);
4. a parita' di distanza i vicini si scelgono con la geometria relativa (prima lo spostamento angolare positivo), mai con l'indice di colonna:
   l'uscita e' invariante alle rotazioni dell'anello e alle traslazioni della griglia, ed equivariante alle permutazioni delle colonne (test).

**Encoder spaziale locale** (`src/wearusfm/model/local_encoder.py`, insiemi di attenzione in `channel_codes.attention_sets`; 02/10/2026,
proposta di AG). Attenzione di ogni canale sulle sue chiavi raccolte con `gather` (v10 §5.4, rev. 3), con il bias di §5.3. **Scelte da
confermare:**
5. canali di anello e griglia: se stessi e i k vicini metrici dello stesso gruppo; canali sparsi: tutti gli altri canali validi del montaggio,
   senza geometria (il «set encoder su identita' anatomica» di v10 §5.4); i canali metrici NON guardano i canali sparsi dello stesso montaggio
   (la sinergia fra gruppi passa dal Perceiver e dal backbone); i canali scartati dal QC guardano solo se stessi e nessuno li guarda;
6. distanze in **passi d'elettrodo** del gruppo (anello: angolo / passo angolare minimo; griglia: passi di indice), cosi' il bias ha la stessa
   scala su anelli da 8 e da 16 e sulle griglie;
7. bias = componente fissa `-d` (decadimento con la distanza) + basi radiali apprese per testa e topologia, **inizializzate a zero**; sulle
   griglie basi separate per |riga| e |colonna| (asse delle fibre privilegiato, v10 §3.4); il bias dipende dalla distanza e non dal verso, quindi
   e' invariante alla riflessione dell'anello (la direzione di propagazione resta nel segnale, non nel bias);
8. ogni patch temporale e' indipendente nell'encoder locale (attenzione fra canali allo stesso istante); conto dei FLOP per livello incluso.
Test (torch, in locale): rotazione dell'anello senza effetto, permutazione dei canali equivariante, montaggi impacchettati isolati, canali sparsi
che vedono tutto il montaggio, canale scartato ignorato, bias iniziale uguale al solo decadimento fisico.

**Pooling Perceiver** (`src/wearusfm/model/perceiver.py`; 02/10/2026, proposta di AG). **Scelte da confermare:**
9. fattorizzato **per patch temporale** (il proxy del passo 0 attendeva su tutti i token del campione insieme: semplificazione dichiarata
   allora): per ogni campione e istante, K latenti appresi e condivisi fanno cross-attention sui token visibili dei canali di quel campione;
   nessun encoding posizionale sui canali, quindi l'uscita non dipende dall'ordine dei canali;
10. token nascosti (masking JEPA, canali scartati dal QC) fuori dalle chiavi; dove a un istante non resta nessun token visibile (slab su tutti i
    canali, il caso dell'ancora RVQ) i latenti restano quelli appresi, senza contenuto: ricostruire l'istante tocca al backbone temporale;
11. implementazione di riferimento con un ciclo sui campioni impacchettati (offset come `cu_seqlens`); in produzione `flash_attn_varlen_func`
    sulle sequenze (campione, istante), da provare equivalente su Leonardo prima di usarlo.
Primo test d'insieme (torch, in locale): identita' di canale -> encoder locale -> Perceiver su un batch con anello (emg2pose), sparso (Camargo),
griglia (CapgMyo) e misto (NinaPro DB2) impacchettati, avanti e indietro con gradienti a tutti i moduli. Manca il backbone temporale per
chiudere il criterio del passo 6.

**Backbone temporale** (`src/wearusfm/model/backbone.py`; 02/10/2026, proposta di AG). **Scelte da confermare:**
12. blocco fattorizzato: attenzione lungo il tempo per ogni latente, poi fra i K latenti per ogni istante, poi MLP (pre-norm, residui);
13. posizione temporale **relativa** con RoPE sull'indice di patch (la patch e' in ms fissi, D10); nessuna posizione sui latenti (slot appresi);
14. `time_valid` toglie dalle chiavi temporali il padding delle finestre corte (contesto variabile, proposta D10) e gli istanti tutti nascosti;
    questi restano query: se le predizioni si leggono dal backbone o da un decoder a query sugli istanti visibili e' D11 (v10 §5.6);
15. implementazione di riferimento in tensori pieni con padding temporale; in produzione l'attenzione temporale puo' passare a
    `flash_attn_varlen_func` (RoPE prima del kernel), da provare su Leonardo.
**Test d'insieme:** identita' -> encoder locale -> Perceiver -> backbone su un batch con anello, sparso, griglia e misto, con un canale
scartato, un canale nascosto e uno slab: avanti e indietro, gradienti a tutti i moduli. Del criterio di chiusura del passo 6 («un batch con le tre
topologie insieme fa forward e backward») manca il decoder a query (D11); conto dei FLOP per modulo presente in encoder locale, Perceiver e backbone
(manca il front-end).

**Decoder a query e diagnostica del collasso da query** (`src/wearusfm/model/query_decoder.py`; 02/10/2026, proposta di AG). **Scelte da
confermare:**
16. query = identita' del canale (anche per i canali nascosti) all'istante t; cross-attention su tutti i P x K latenti del campione con RoPE sul
    tempo (distanza fra l'istante della query e quello del latente); le chiavi escludono gli istanti non validi per lo studente; piu' livelli
    aggiornano le query. Serve alle due opzioni di D11 (a e b cambiano dove si leggono i target del teacher, non come predice lo studente);
17. diagnostica (v10 §7.1 punto 4): le stesse query di sonda valutate su S campioni; rapporto fra varianza fra campioni a query fissa e varianza fra
    query. Un decoder a cui il contenuto non arriva da' rapporto ~0 (test). **Nessuna soglia ancora:** la soglia d'allarme va scritta qui prima
    del sanity JEPA (regola delle soglie congelate).
**Test d'insieme fino al decoder:** identita' -> encoder locale -> Perceiver -> backbone -> decoder su anello, sparso, griglia e misto, con query
sui canali e sull'istante nascosti: avanti e indietro, gradienti a tutti i moduli. Del passo 6 restano: front-end vero collegato, teste delle
ancore (RMS, bande, inviluppo, RVQ), ciclo JEPA (studente/teacher EMA), sanity JEPA su un dataset omogeneo, FLOP del front-end.

**Front-end collegato e ancore** (`src/wearusfm/model/patch_tokens.py`, `anchor_targets.py`, `anchors.py`; 02/10/2026, proposta di AG). Il
front-end del gate D8 si applica a ogni campione alla sua frequenza nativa, poi una proiezione alla dimensione del modello; finestre piu' corte
diventano padding (`patch_valid`, `time_valid`). **Scelte da confermare:**
18. finestre dei target, centrate sulla patch: log RMS su 25 ms (la patch stessa: dentro i 20-50 ms di v10 e senza guardare le patch vicine,
    magari visibili); forma spettrale su 200 ms (8 patch, la griglia RVQ; risoluzione 5 Hz); log inviluppo su 500 ms; target assente dove la
    finestra esce dal segnale;
19. bande: 5, log-spaziate su 20-450 Hz (20, 37, 70, 130, 242, 450 Hz); una banda e' mascherata se il suo bordo SUPERIORE supera il limite del
    canale (minimo fra Nyquist e banda dichiarata): regola conservativa, che a DB5 (Myo, 100 Hz) lascia solo 2 bande (la terza, 70-130 Hz,
    supera i 100 Hz); la frazione di potenza si normalizza sulle bande disponibili;
20. RVQ: target per (canale, finestra da 200 ms della griglia) solo se la finestra e' tutta nascosta e il dataset ha l'ancora accesa; la testa
    classifica gli 8192 codici del livello 0 del ramo 0 dalla media delle uscite del decoder alle 8 patch (nessun tipo di query nuovo); i codici
    li calcola il tokenizer congelato, da collegare su Leonardo (il clone di NeuroRVQ non e' sul Mac);
21. teste lineari («testa piccola»); il peso complessivo delle ancore (0,1-0,3, v10 §6.2) entra nel ciclo di training.
**Non ancora fatto:** ancore spaziali a peso ridotto (covarianza a lag zero, coerenza in modulo: v10 §6.3), che sono a coppie di canali.
**Test d'insieme dal segnale:** bracciale Meta a 2 kHz e Myo a 200 Hz nello stesso batch, front-end vero, tutta la catena fino alle perdite delle
ancore, gradiente fino ai coefficienti del front-end; DB5 senza finestre RVQ (ancora spenta).

**Correzione al punto 20 (02/10/2026, trovata scrivendo il ciclo JEPA):** nel commit 09b34b5 una finestra RVQ aveva target se era nascosta per
quel canale, anche solo per masking di canale. v10 §6.3 lo esclude («solo intervalli temporali mascherati su tutti i canali, mai masking di
canale»: con i vicini visibili allo stesso istante la fase si ricostruisce). Ora il target RVQ c'e' solo sugli **slab**: finestra nascosta su tutti
i canali validi del campione (`anchor_targets.rvq_target_windows` e `training/jepa.py`, con test). Nessun run l'aveva usato.

**Modello intero e ciclo JEPA** (`src/wearusfm/model/fm.py`, `src/wearusfm/training/jepa.py`; 02/10/2026, proposta di AG). **Scelte da
confermare:**
22. **nessuna fuga del nascosto verso lo studente**, chiusa in tre punti: campioni grezzi delle patch nascoste azzerati prima del front-end (che usa
    100 ms di contesto per lato: senza, una patch visibile vedeva fino a 100 ms del segnale nascosto); nell'encoder locale un token nascosto non e'
    chiave per gli altri canali (prima lo era: corretto qui); Perceiver e backbone gia' lo escludevano. Test: cambiare il segnale dentro le regioni
    nascoste non cambia le predizioni dello studente. Ai bordi di una regione nascosta lo studente vede zeri, come ai bordi del segnale;
23. teacher = copia EMA dell'intero studente, stop-gradient, vede tutto; target (a) o (b) di D11 come parametro **senza default**; target
    normalizzati con LayerNorm senza parametri; perdita = errore quadratico medio (alternativa: errore assoluto, come V-JEPA); query = tutti i token
    nascosti e validi;
24. perdita totale = JEPA + `anchor_weight` x (RMS + bande + inviluppo + RVQ); `anchor_weight` (v10: 0,1-0,3) e momento EMA **senza default**:
    li fissa la configurazione del run (D14 per lo schedule);
25. rango effettivo (exp dell'entropia dei valori singolari, v10 §7.1 punto 3) come diagnostica, accanto a quella del collasso da query.
Del passo 6 restano: generatore di maschere (D10), dataloader vero, collegamento del tokenizer RVQ su Leonardo, ancore spaziali, soglie delle
diagnostiche (da congelare qui prima del sanity), sanity JEPA su un dataset omogeneo.

**Generatore di maschere** (`src/wearusfm/training/masking.py`; 02/10/2026, segue la BOZZA D10 §3, non firmata). **Scelte da confermare con
D10:**
26. tre famiglie distinte: maschera temporale = intervallo su **un solo canale**; slab = intervallo su **tutti** i canali, lunghezza multipla di 8
    patch e inizio sulla griglia (ancora RVQ, solo dove e' accesa); spaziale = canali scelti **per tutta la finestra** (arco contiguo sugli anelli,
    rettangolo fino a 4 x 4 sulle griglie, canale singolo o gruppo dello stesso compartimento sui montaggi sparsi), al piu' meta' dei canali validi.
    **Punto aperto:** su una griglia densa la maschera temporale su un canale e' facile (i vicini sono visibili); l'alternativa e' l'intervallo su un
    gruppo spaziale («tubi», come in V-JEPA);
27. le quote della tabella D10 sono **attese** sui token nascosti, non vincoli per campione: uno slab copre tutti i canali e da solo puo' superare la
    sua quota (uno slab lungo da 24 patch e' il 15% di una finestra da 4 s; il primo tentativo di riempire le quote per campione nascondeva l'82%
    invece del 50%). Si sceglie il tipo con probabilita' proporzionale a quota / dimensione attesa sul campione; una maschera che porterebbe il
    totale oltre budget x 1,1 si scarta. Misurato su 50 campioni per montaggio (budget 0,5, finestra di 160 patch): nascosto 0,50-0,55; quote
    realizzate sui token nascosti corta 15-16%, media 12-14%, lunga 8-11%, slab 17-22%, spaziale 42-43% (proposta: 15 / 12,5 / 10 / 22,5 / 40);
28. la frazione nascosta `ratio` non ha default (regola da congelare prima del sanity JEPA); il generatore restituisce il tipo di ogni token
    nascosto, per le perdite registrate per tipo (v10 §6.4);
29. costo: 3-4 ms a campione su 11-16 canali, ~31 ms su una griglia da 128 (circa mille maschere piccole): da vettorizzare se il dataloader lo
    richiede (Hyser, 256 canali).

**Dataloader del pretraining** (`src/wearusfm/data/pretraining_loader.py`; 03/10/2026, proposta di AG). Dal manifest ai batch per il modello:
sessione estratta coi pesi del manifest; finestra di 1-4 s (proposta D10) con inizio su multipli di 200 ms dall'inizio della prova, dentro una
prova e, se `split_at_gaps` (proposta D9 §f), dentro un tratto senza salti; finestre che toccano un tratto costante su un canale valido riestratte;
filtro opzionale (passa-banda, notch 50 e 60 Hz) con 0,5 s di margine dentro il tratto; scala di sessione = mediana dei MAD dei canali validi
(v10 §4.3) su 32 tratti da 2 s, in cache; target delle ancore, maschere (generatore D10), codici e insiemi di attenzione impacchettati. Test su un
albero sintetico coi montaggi veri e un manifest del costruttore vero, piu' un passo di training intero dal batch del loader. **Da decidere:**
30. **filtraggio**: i dati processati NON sono filtrati all'ingest (nessun filtro negli adattatori), mentre v10 §4.3 vuole il filtraggio offline,
    «mai on-the-fly», per risparmiare CPU. Misura indicativa sul Mac, un processo: 100 ms per batch da 8 senza filtro, 125 ms con il filtro (circa
    65-80 finestre/s per processo, contro un fabbisogno stimato al passo 0 di ~68 finestre/s per GPU a 30M con 8 processi per GPU). Alternativa:
    una copia filtrata su disco (spazio e un job CPU). Da rimisurare su Leonardo con sessioni vere;
31. scala di sessione stimata su tratti sparsi (non sull'intera sessione: emg2qwerty ha sessioni da ~18 minuti a 32 canali); da precalcolare con
    un job CPU e salvare accanto al manifest, perche' ogni processo del dataloader non la ricalcoli;
32. manca ancora: montaggi virtuali e sottocampionamento HD al volo (D6a), misura del ritmo su Leonardo.

---

## D9 firmata, D10 in parte, filtraggio e congelamento del manifest (Simone, 03/10/2026: «firmo 1-11, 13 e 14»)

Sui fogli `docs/fogli_firma_d9_d10.md` (preparati da AG il 03/10). **Firmati:**
1. **quote per classe A 20%, B 75%, C 5%, dichiarate sulla topologia presentata**; il campionatore registra per ogni campione origine e topologia
   presentata (D9 a);
2. **tetto di 8 passaggi per dataset a E = 4, contato sul tempo di origine**; a E = 8 e a 2D i dataset al tetto arrivano a 16 passaggi e il pilot lo
   misura (D9 a, lettura i);
3. **amputati di DB8 e DB10 e Zhang random in classe B** (D9 a-bis);
4. **pesi dentro la classe ∝ ore^0,5**, eccedenza dei dataset al tetto ridistribuita nella classe (D9 a-bis);
5. **Kaifosh tutto benchmark**, con la sovrapposizione fra i dataset Meta e la natura del suo split pubblico dichiarate (D9 b, D3b);
6. **split** come in D9 (d): ufficiali dove esistono (emg2pose anche per sessione), altrove 20% (almeno 2) stratificato, seme 0, CSL-hdemg un
   soggetto di test, DB4/DB5/DB7/DB8 interi nel pretraining, manifest annidati al 12,5/25/50%, sovrapposizioni dichiarate. File firmato:
   `splits/v1/splits_v1.json` (stesso contenuto di `splits/draft/splits_draft.json`, sha256 cfd1e1f4..., con `draft: false` e la firma);
7. **ancora RVQ secondo D5b** nel manifest (15 dataset accesi, 14 nel pretraining perche' Kaifosh e' benchmark; spenta su putEMG, CSL-hdemg, DB5)
   (D9 e);
8. **salti dell'asse dei tempi: segmenti spezzati a ogni salto del sidecar** (D9 f; nel dataloader `split_at_gaps = True`);
9. **rinfresco di `$SCRATCH` ogni 21 giorni per tutta la ladder** (22/10, 12/11, 03/12, 24/12), ogni giro autorizzato da Simone;
10. **patch 25 ms** (D10, 1);
11. **contesto variabile 1-4 s, inizio su multipli di 200 ms dall'inizio della prova** (D10, 2);
13. **filtraggio nel dataloader** (passa-banda 20-450 Hz adattato alla Nyquist, notch a 50 e 60 Hz, margine 0,5 s dentro il tratto): **deviazione
    registrata da v10 §4.3** («offline, mai on-the-fly»), da confermare con una misura del ritmo su Leonardo con sessioni vere (costo da dichiarare
    prima del lancio);
14. **congelamento di `manifest-v1` autorizzato**: costruttore con `--version manifest-v1 --splits splits/v1/splits_v1.json` (CPU seriale, ~16 min,
    0 GPU-ora), tag git `manifest-v1`, hash del file in questo registro, `D_t` e `D_c` ricalcolati.
**Non firmati (03/10):** 12 (masking di D10, compresa la scelta canale singolo / «tubi») e 15-20 (valori del sanity JEPA).
**`manifest-v1` CONGELATO il 03/10/2026** (decisione 14; job 59253155, 9 min 48 s, exit 0):
- file: `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/passo4/manifest_59253155.json.gz` (1.689.911 byte; copia sul Mac in
  `~/wearusfm_local/reports/passo4/`; NON nel repo); **sha256 del file 9ab0fc5f90889973b55635ba6a2d485699b842d747b9629186d2b309b402d781**
  (uguale su cluster e Mac);
- **hash del contenuto b3b4a3cbda375bee8ca6a6edcfe47e02928aefef595bb99fec203e3d10b1b32b** (parametri e righe ordinate, `manifest_hash`);
- codice: commit e053c02 (tag git **`manifest-v1`**); split `splits/v1/splits_v1.json` (sha256 a18e7565..., uguale a `splits_sha256` nel file);
- parametri: quote A 20 / B 75 / C 5, alpha 0,5, tetto 8 passaggi a 4 epoche, patch 25 ms;
- **27.560 sessioni; pretraining 19.034 sessioni, 822,9 h; `D_t` = 118.491.864 time-patch; `D_c` = 3.248.996.413 source-channel-patch**
  (patch da 25 ms, dopo il QC, prima di ogni augmentation: v10 §10.3); quote realizzate A 20,0 / B 75,0 / C 5,0%, nessuna inassegnabile;
- verifica: righe e pesi identici, campo per campo e bit per bit, alla bozza del 02/10 (job 59204297); cambiano solo `version` e l'hash degli
  split; nessun soggetto senza sessioni, nessuna sessione di test mancante.
Il «consumo realizzato per epoca» (piano, «Chiuso quando») si misura dal dataloader, al primo run.

## Masking (D10) e valori del sanity JEPA firmati (Simone, 03/10/2026: «firmo 12 (masking) e 15-20 (valori del sanity)»)

Sui fogli `docs/fogli_firma_d9_d10.md`. **Firmati:**
12. **masking D10**: quote **attese** sui token nascosti (temporale corta 15%, media 25%, lunga 20%, spaziale 40%), meta' del temporale medio-lungo
    come slab RVQ allineati sui dataset con l'ancora accesa, spaziale al piu' meta' dei canali validi; **maschera temporale non slab = intervallo su
    un gruppo spaziale («tubi», proposta b)**: arco d'anello, rettangolo di griglia, canale o compartimento sui montaggi sparsi (implementazione:
    stesso commit di questa registrazione, `training/masking.py`, con test). **D10 e' firmata per intero** (patch, contesto, masking);
15. **frazione nascosta 0,5**, fissa per il sanity JEPA e per la finestra 1, non scelta sui risultati; se il sanity collassa si riapre come
    decisione nuova;
16. **soglie d'allarme** (valutate ogni 500 passi su un batch fisso di validazione): **collasso da query** se il rapporto varianza fra campioni a
    query fissa / varianza fra query e' sotto **0,05** per 3 valutazioni di fila; **rango effettivo** delle uscite del backbone (medie per istante)
    sotto **il 10% della dimensione** del modello per 3 valutazioni di fila. Un allarme ferma il run;
17. **sanity JEPA su emg2qwerty**;
18. **peso complessivo delle ancore 0,2; momento EMA del teacher 0,996 costante** per il sanity (lo schedule vero e' D14);
19. **dropout del livello muscolo 0,4** (D15, anticipata);
20. **target JEPA (b)** per il sanity (uscita del decoder a query del teacher); D11 vera si chiude prima del passo 7, (a) fra le ablation della
    finestra 1.
**Autorizzato anche** (stesso messaggio: «sì, via alla misura con le scale di sessione»): misura del ritmo del dataloader su Leonardo (conferma
della decisione 13) e calcolo delle scale di sessione di tutte le sessioni di pretraining; CPU seriale, 0 GPU-ora.

**Misura del ritmo e scale di sessione (job 59254061, 03/10/2026, 1 h 36 min, CPU, 0 GPU-ora; risultati in
`~/wearusfm_local/reports/passo6/` e su Leonardo in `wearusfm_runs/results/passo6/loader_59254061/`, sha256 verificati):**
- **scale:** 18.839 su 19.034 sessioni; **195 errori** (194 emg2pose, 1 emg2qwerty), tutti «length of the input vector x must be greater than
  padlen, which is 39»: con i salti spezzati restano tratti di pochi campioni fra due salti, e la stima della scala a volte ne pescava uno.
  **Corretto**: la stima usa solo i tratti lunghi almeno `min_window_s` (test). Scale molto diverse fra dataset (es. NinaPro ~3e-6, GRABMyo
  ~3,5e3) per le unita' di misura diverse nei file: la normalizzazione di sessione le assorbe, e' il suo scopo;
- **ritmo: NON BASTA.** 1,5-1,8 finestre/s per processo senza filtro, 1,9-2,4 con il filtro, contro ~8,5 richieste (passo 0: ~68 finestre/s per
  GPU con 8 processi). Il **68% del tempo e' la lettura** delle finestre da `$SCRATCH` (Lustre): letture piccole e sparse, una per finestra. Il
  filtro non e' il collo di bottiglia (con il filtro non e' piu' lento; la differenza e' nel rumore della cache): **la decisione 13 resta
  valida per il costo del filtro**, ma il dataloader com'era non reggeva. Il benchmark del passo 0 (240 finestre/s) leggeva shard sintetici
  contigui, non le sessioni vere;
- **correzione, proposta di AG: lettura a blocchi** (`LoaderConfig.block_s` = 30 s, `windows_per_block` = 8, `block_pool` = 16): ogni processo
  legge una volta tratti contigui di 30 s e ne estrae 8 finestre ciascuno; la probabilita' delle sessioni per finestra resta quella del manifest
  (test), le finestre di un batch vengono da un gruppo piu' piccolo di sessioni. Da rimisurare su Leonardo prima del sanity.

**Catena dei job di training (03/10/2026; problemi trovati da Simone in una chat laterale, verificati da AG):** (1) dopo un allarme o una perdita
non finita i job successivi riprendevano ad allenare: ora `train` scrive `STOP` nella cartella del run, e i job successivi (Python e lo sbatch)
escono subito; (2) lo stato casuale di torch (CPU e GPU) ora sta nel checkpoint; (3) `scripts/slurm/submit_chain.sh` sottomette N job con
`--dependency=afterok`: un job che fallisce ferma la catena, un'uscita pulita al limite di tempo la fa proseguire. Il limite di `boost_qos_dbg`
(«max 2 job») per una catena va verificato sulla configurazione della QoS.

**QoS `boost_qos_dbg` (letta con `sacctmgr`, 03/10/2026):** MaxWall 30 min, **MaxJobsPU 2 e MaxSubmitPU 2** (anche i job in attesa per dipendenza
contano): una catena piu' lunga di 2 job li' non entra. Il sanity vero va su `boost_usr_prod` (QoS normale, fino a 24 h); `boost_qos_dbg` per i
collaudi.

**Sanity JEPA — job 1, collaudo su GPU** (uso del cluster con GPU autorizzato da Simone il 03/10/2026: «se serve il cluster con GPU usalo, ti
autorizzo»): `scripts/slurm/sanity_jepa.sbatch` su `boost_qos_dbg`, 1 GPU, 30 minuti (si ferma da solo a 26), **costo massimo 0,5 GPU-ora = 4 ore
locali**; budget del passo 6: 100 GPU-ora, usate 0 prima di questo job. Preset `sanity` (28,2 M parametri, valori firmati), dataloader a blocchi,
scale del job 59254061 (le 195 mancanti si calcolano al volo), cartella `$WORK/wearusfm_runs/runs/sanity_0310/`. Misura: secondi per passo, memoria
GPU, attese sui dati; se regge, il sanity continua dallo stesso checkpoint su `boost_usr_prod`. **Ancora RVQ spenta** (tokenizer non ancora
collegato al ciclo).

**Esiti dei due job del 03/10 pomeriggio** (letti alle 20:15: la sessione di AG era ferma per il limite d'uso; controllo automatico cancellato):
- **Rimisura del ritmo, job 59264303** (`lrd_all_serial`, 4 CPU, 5 min 20 s, 0 GPU-ora; sha256 `throughput.json` fb1c94a7..., `session_scales.json`
  eccecc00..., uguali su cluster e Mac): dataloader a blocchi **5,9-8,0 finestre/s per processo senza filtro, 5,8-8,5 con il filtro**, contro ~8,5
  richieste: 3-4 volte meglio di 59254061, **ancora sotto** (un processo su quattro alla soglia). La lettura non e' piu' il collo di bottiglia
  (lettura e filtro 10%): **target delle ancore 55-66% del tempo**, maschera 14-17%. Scale: 141 delle 195 mancanti calcolate col metodo nuovo; **54
  sessioni di emg2pose non hanno nessun tratto lungo almeno 1 s fra due salti**: nessuna finestra estraibile, il loader le salta e le conta (tempo
  non campionabile, D9 §f).
- **Sanity JEPA job 1, 59264349: FALLITO al primo passo dopo 35 s** (exit 1, nessun checkpoint, nessun `STOP`): `RuntimeError: Expected all
  tensors to be on the same device` nel front-end. Tensori creati senza `device` restavano sulla CPU: bug di AG, i test giravano solo sulla CPU.
  **Costo: 35 s x 1 GPU = 0,01 GPU-ora**; budget del passo 6: usate 0,01 su 100.
- **Correzione** (commit di questa registrazione): tensori sul dispositivo del modello nel front-end, in `zero_hidden_samples` (ora vettoriale),
  `hidden_queries`, `gather_anchor_targets` (ora vettoriale), nel ramo RVQ e nella diagnostica. Alla ripresa lo stato casuale salvato torna sulla
  CPU: con `map_location` sulla GPU `set_rng_state` falliva (verificato). **Test nuovo:** modello sulla CPU e dispositivo di default MPS (il Mac),
  cosi' ogni tensore creato senza `device` fa fallire il test come farebbe la GPU; fallisce sul codice vecchio, passa sul nuovo. Due test vecchi
  spegnevano il gradiente per tutto il processo (`set_grad_enabled(False)`), e i moduli torch lanciati insieme fallivano: corretti. Ora gli 11
  moduli torch passano insieme (56 test, `~/.venvs/wearusfm-tok`), piu' 540 della suite senza torch.

**Sanity JEPA job 2: collaudo su GPU ripetuto dopo la correzione** (stessa autorizzazione del 03/10: «se serve il cluster con GPU usalo, ti
autorizzo»). Stesso `sanity_jepa.sbatch` su `boost_qos_dbg`: 1 GPU, 30 minuti, **costo massimo 0,5 GPU-ora = 4 ore locali**; budget del passo 6:
usate 0,01 su 100. Scale del job 59264303 (18.980; le 54 sessioni senza tratti lunghi si saltano). Stessa cartella `runs/sanity_0310/`, che non ha
checkpoint. Scopo: secondi per passo, memoria GPU (ora nel log e in `summary.json`), attese sui dati. **Resta un collaudo.** Il dataloader ha ancora
i difetti trovati dalla review del 03/10 (i blocchi deformano la distribuzione delle finestre dentro la sessione; manca il margine di 200 ms attorno
ai tratti costanti), da correggere prima del sanity vero. Il sanity vero ripartira' da zero, non da questo checkpoint. Ancora RVQ spenta.
Correzioni dello stesso commit: lo sbatch rifiuta un `RUN_DIR` relativo, che dopo il `cd` finirebbe dentro il repo.

**Sanity job 2, 59276285: FALLITO dopo 54 s, memoria GPU esaurita** (`torch.cuda.OutOfMemoryError` nell'attenzione sui latenti, 60,4 GiB allocati
su 63,4). **Costo 0,015 GPU-ora**; budget del passo 6: usate 0,025 su 100. La review del modello (Opus, 03/10) l'aveva previsto: ~6 GB di
attivazioni per finestra da 4 s di emg2qwerty in float32 (backbone 3,7, decoder 1,5, encoder locale 0,5), quindi ~190 GB col batch 32. Il preset
del sanity non entra in una A100 com'e': correzione sotto, nessun rilancio prima.

**Review del 03/10, dati e script: correzioni** (istruzione di Simone: «quando hai finito, se ci sono cose sbagliate correggi senza bisogno della mia
conferma»):
- **lettura a blocchi riscritta.** La prima versione cominciava il blocco su un inizio estratto e lo tagliava alla fine del tratto. Su un tratto
  da 60 s copriva i primi 10 s a 0-0,3 volte la media e gli ultimi 5 s a 1,9-3,5 volte; le finestre corte erano il 21% invece del 4,8%, e sul
  sanity (emg2qwerty) il 10,9% invece dello 0,2%. Ora gli inizi accettati di ogni tratto sono divisi in tessere fisse di 30 s, e il blocco e' la
  tessera che contiene un inizio estratto uniformemente nella sessione: **la distribuzione delle finestre e' quella della lettura diretta** (test
  sugli istogrammi degli inizi e sulla quota di finestre corte). Ogni finestra ha il margine intero del filtro (test senza eccezioni);
- **margine di 200 ms attorno ai tratti costanti**, come nella regola firmata il 01/10 («ne' per V2 ne' per l'addestramento»): prima il loader
  controllava solo la sovrapposizione. Gli inizi accettati si calcolano una volta per sessione (vettoriale, test inizio per inizio contro la
  funzione scalare): niente piu' tentativi a vuoto, e una sessione senza inizi accettati si salta e si conta;
- **scala di sessione con il seme dalla chiave** anche quando si calcola al volo: prima ogni processo aveva il suo generatore, e la stessa
  sessione poteva avere scale diverse fra processi e nel batch di validazione;
- **integrita' del manifest:** il loader controlla che il `metadata.json` di ogni sessione abbia lo `sidecar_sha256` della riga di `manifest-v1`,
  altrimenti si ferma;
- **`measure_loader.py`** riscrive `session_scales.json` ogni ~500 sessioni (un TIMEOUT non perde la fase) e ha `--skip-rate`;
- **target delle ancore 2-3 volte piu' veloci** (erano il 55-66% del tempo del loader): RMS e inviluppo con somme cumulative invece delle finestre
  materializzate, spettro in float32 con scipy. 32 canali a 2 kHz: da 37 a 11 ms per finestra; 128 canali: da 115 a 55 ms. Stessi target della
  versione vecchia (test di equivalenza, anche ai bordi).
Restano da decidere (rapporto a Simone): il transitorio del filtro ai bordi dei tratti e l'altro braccio di emg2pose (540 registrazioni con la
mano ferma nel pretraining e quella in movimento nel test, 7,3 h). Sono correzioni minori il ripristino in `qc_relative_check.py` e i test
mancanti sull'esclusione delle righe di test.

**Misura del ritmo e scale, terzo giro** (stessa autorizzazione del 03/10, «Misura del ritmo del loader: ... vai pure quando sei pront. accetto la
proposta»): `measure_loader.sbatch` su `lrd_all_serial` con 8 CPU, 30 GB e 3 h (opzioni da riga di comando), **0 GPU-ora**. Ricalcola **tutte**
le scale col metodo nuovo (solo tratti lunghi): le 18.839 del job 59254061 erano col metodo vecchio. Il seme e' lo stesso, quindi cambiano solo
le sessioni con tratti corti fra salti. Poi misura il ritmo con la lettura a tessere e i target veloci, su 8 processi come su un nodo GPU (8 CPU
per GPU). Soglia invariata: ~8,5 finestre/s per processo.

**Review del 03/10, modello e training (Opus): correzioni** (stessa istruzione di Simone). Quattro rilievi gravi, tutti verificati:
- **A1, identita' di canale senza posizione.** I canali di un anello avevano tutti la stessa identita' (anatomia ignota piu' topologia; su
  emg2qwerty 32 canali, 1 identita'). Il decoder a query vede solo i latenti del Perceiver, senza canali, quindi dava la stessa predizione per
  tutti i canali allo stesso istante (differenze ~5e-7) e non distingueva la mano sinistra dalla destra. Con il target (b) il masking spaziale (40%
  del budget) non chiedeva mai di predire un canale: il sanity avrebbe potuto «non collassare» senza imparare nulla per canale.
  **Correzione (scelta di AG, da confermare):** nell'identita' entra anche il **sistema del sensore** (v10 §3.6). Per gli anelli cos/sin(m
  angolo), m = 1..4, cosi' una rotazione della fascia e' uno sfasamento: simmetria ciclica rappresentabile, non imposta. Per le griglie cos/sin(pi m
  x) su riga e colonna. Si aggiungono l'ordine del gruppo nel montaggio e il **lato dell'arto** (campo `chirality`). I canali sparsi restano
  senza posizione: l'identita' e' anatomica. Era la mia «interpretazione da confermare» del passo 6 (encoding solo relativo), che il revisore ha
  mostrato sbagliata. Test: 32 identita' distinte su emg2qwerty; predizioni del decoder diverse fra canali dello stesso anello.
- **A2, simmetria imposta nell'encoder locale.** Il bias appreso usava solo |spostamento|, quindi un'onda oraria e la sua riflessione davano latenti
  identici: il verso di propagazione non era rappresentabile, contro v10 §3.4 («equivarianza, non invarianza»). **Correzione:** la parte appresa del
  bias lavora sullo **spostamento con segno**: angolare in passi d'elettrodo, riga e colonna. La parte fissa (-distanza) resta simmetrica,
  all'inizio vale solo lei (pesi a zero). Test: con pesi simmetrici la riflessione commuta, con pesi qualsiasi no.
- **A3, memoria.** ~6 GB per finestra, quindi job 59276285 fuori memoria. **Correzione:** `scaled_dot_product_attention` nel backbone e nel
  decoder, che non salva le matrici d'attenzione; **activation checkpointing** per blocco (encoder locale, backbone, decoder); **bf16** in autocast
  su GPU, col front-end che resta in float64 (v10 §5.5). Misura su CPU in fp32, finestra di emg2qwerty da 4 s: tensori salvati per il backward da
  4,0 GiB (gia' con SDPA; 6,0 prima) a **0,47 GiB per finestra**, ~15 GB per il batch 32 in fp32, meno in bf16.
- **A4, il contenuto del segnale pesava ~0,1% del token all'inizializzazione** (il front-end integra, fattore 1/fs: uscite ~1e-3; perdita JEPA
  iniziale ~1e-10). **Correzione:** un guadagno scalare fisso prima della proiezione dei token, calcolato alla costruzione su rumore 20-450 Hz di
  varianza unitaria a 2 kHz, con seme fisso, salvato nel checkpoint: uscite ~1, contenuto del token ~0,8 contro bias 0,07, perdita JEPA iniziale
  ~1,6e-4. Il front-end del gate D8 non cambia.
- **M1, una fuga del contenuto nascosto:** la griglia delle patch in `zero_hidden_samples` era in float32, e in ~20% dei bordi a 2 kHz il primo
  campione nascosto restava visibile. Ora e' in float64 con la stessa espressione del front-end (test su 7 frequenze e tutte le patch; fallisce
  sul codice vecchio).
- **Minori corretti:** la perdita JEPA si registra anche per tipo di maschera e per topologia (v10 §6.4); nelle diagnostiche si registrano anche le
  varianze assolute (rapporto e rango non vedono un'uscita quasi costante: le soglie firmate restano quelle); offset delle query letti una volta
  dalla GPU; ripresa con `map_location="cpu"` (lo `step` di AdamW restava sulla GPU); generatore del dropout su un dispositivo qualsiasi.
- **Aperti, per Simone:** M3, la perdita e' una media per token, quindi a finestra 1 la classe C (HD) peserebbe il 38% della perdita contro il 5%
  delle finestre: serve una decisione su come contare le quote nella perdita. Il sanity non ne risente (un dataset). Restano poi le
  sincronizzazioni residue e il conto dei FLOP del decoder.

**Sanity JEPA job 3: collaudo su GPU dopo le correzioni** (stessa autorizzazione del 03/10). `sanity_jepa.sbatch` su `boost_qos_dbg`, 1 GPU, 30
minuti, **costo massimo 0,5 GPU-ora = 4 ore locali**; budget del passo 6: usate 0,025 su 100. Scale del job 59264303. Il job 59276700, che le
ricalcola, e' ancora in corso: per un collaudo bastano. Cartella nuova `runs/sanity_0310_c3/`: il modello e' cambiato. Misura: secondi per
passo, memoria GPU, attese sui dati, perdite per tipo di maschera. Resta un collaudo: il sanity vero parte dopo l'esito e dopo il rapporto a
Simone.

**Review del 03/10, documenti: correzioni.** Firme e frasi tutte verificate alla lettera sulla trascrizione. Manifest, split, QC, numeri di V2 e
tabelle di D9/D10 tornano col manifest congelato. Corretti, con note datate dentro i testi firmati (il testo firmato non si riscrive):
- v10: §8 asse 5, test degli amputati su DB3 e DB10 (DB7 e DB8 sono interi nel pretraining, decisione 6); trappola NinaPro chiusa dagli split;
  §2.8, DB8, DB10 e Zhang random in B; putEMG e Hyser firmati (fatti 11 e 12);
- D9 e D10: stati «NON firmata» superati; numeri sul solo pretraining (85,1% invece di 86%; 58,6 h di campioni; 22,1% di D_c; 6,1 h); nota che
  **il costruttore non spezza i segmenti ai salti** (~2,4 h della classe B in tratti sotto 1 s sono nelle ore del manifest ma il loader non li
  campiona, ~0,3% della classe B); conteggio dei 96 utenti di emg2qwerty marcato come conteggio di AG;
- fogli: nel foglio 5 la seconda parte della frase sul fatto 23 e' un'interpretazione di AG (nota); decisione 7: 15 dataset accesi, 14 nel
  pretraining;
- `inventario_corpus.md` (Kaifosh benchmark; DB7/DB8 «probabile», senza il «7» inventato), `review_esterna.md` (stato aggiornato).
Corretti anche, dalla review dei dati: il ripristino in `qc_relative_check.py` (su una sessione gia' rivista ripristinava il backup della prima
revisione e cancellava le successive; test) e il test mancante sull'esclusione delle righe di test e sul filtro `nested`.

**Domande aperte per Simone** (dalle tre review del 03/10; nessuna blocca il collaudo, alcune bloccano il sanity vero o la finestra 1):
1. **Quote nel tempo o nelle finestre.** Il manifest calcola pesi e passaggi nel tempo; il loader estrae una finestra per estrazione, lunga 1-4 s.
   CapgMyo (prove da 1 s) avrebbe ~1/4 della quota in tempo. *Proposta di AG:* nel tempo (= nei token, come D_t). Il loader dividerebbe il peso di
   ogni sessione per la lunghezza media delle sue finestre, calcolata una volta come le scale. Finestra 1, non il sanity.
2. **Quote nella perdita (M3).** La perdita e' una media per token: a finestra 1 la classe C (HD, 5% delle finestre) peserebbe ~38% della perdita,
   la A il 7,7% contro il 20%. *Proposta di AG:* la media prima dentro ogni finestra, poi fra le finestre, cosi' la perdita rispetta le quote.
   Finestra 1.
3. **Topologia presentata contro origine (decisione 1).** Oggi presentata = origine, perche' i montaggi virtuali (D6a) non ci sono. Con il tetto,
   la classe C puo' dare al massimo il 5,2%: i montaggi virtuali tolti alla C riducono la C presentata. Da decidere con D6a.
4. **Ancora RVQ nel sanity.** La decisione 17 sceglieva emg2qwerty anche per l'ancora RVQ accesa; il collaudo gira senza, perche' il tokenizer non
   e' collegato. Opzioni: (a) collegarlo prima del sanity vero, con i codici precalcolati per sessione sulla griglia da 200 ms; (b) sanity senza
   RVQ, come deviazione registrata. *Proposta di AG:* (a) se costa meno di un giorno, altrimenti (b).
5. **Batch di validazione (decisione 16).** Il codice lo estrae dal pretraining con un seme fisso, e per il rapporto di collasso usa il minimo fra
   studente e teacher. Le soglie restano quelle firmate. Da confermare.
6. **DB5 nell'harness.** DB5 e' intero nel pretraining (decisione 6), quindi un confronto del nostro FM su DB5 (piano, harness; v10 §11) sarebbe su
   soggetti visti. Opzioni: dichiararlo, togliere DB5 dal confronto, oppure split proprio di DB5 (ma era intero per la sovrapposizione con DB4).
7. **Masking a «tubi» (decisione 12).** Firmata come proposta di AG, «(b) tubi»; il messaggio di firma non nomina la scelta. Da confermare.
8. **D_t congelato = 1,18·10⁸**, un terzo dei 3,6·10⁸ su cui erano pianificati la ladder e il rapporto token/parametri (v10 §10.3, piano). Il
   conto del compute dei gradini va rifatto prima del passo 7.
9. **emg2pose, l'altro braccio:** 540 registrazioni hanno il lato in movimento nel test per fasi nuove e il lato fermo nel pretraining (7,3 h). Lo
   split segue il CSV ufficiale riga per riga. Da dichiarare, o da togliere dal pretraining.
10. **Transitorio del filtro ai bordi dei tratti** (decisione 13: margine dentro il tratto). Nella prima patch l'errore rispetto al filtraggio di un
    segnale lungo e' del 45-55%, nei primi 200 ms del 16-20%: riguarda tutte le finestre di CapgMyo e gli inizi di prova e di segmento. Opzioni:
    accettarlo e dichiararlo, oppure togliere dalle perdite delle ancore le patch entro ~100 ms da un bordo.

**Sanity JEPA job 3, collaudo 59277074: RIUSCITO** (COMPLETED, 26 min 18 s, fermato dal limite di tempo dopo 621 passi; nodo lrdn2960;
`summary.json`, `metrics.jsonl` e log in `~/wearusfm_local/reports/passo6/`, sul cluster in `runs/sanity_0310_c3/`). **Costo 26 min 18 s x 1 GPU =
0,44 GPU-ora**; budget del passo 6: usate 0,47 su 100.
- **Ritmo:** 2,4 s per passo col batch da 32, cioe' ~13 finestre/s per GPU. La GPU e' il collo di bottiglia: attesa sui dati 0,03 s per passo
  dopo il riscaldamento, con 6 processi del dataloader. **Memoria GPU massima 17,9 GiB** su 64.
- **Perdite** (medie su 50 passi, passo 0, 100, ..., 600): totale 0,90 -> 0,23; JEPA 0,146 -> 0,0033; log-RMS 0,69 -> 0,39; forma spettrale
  2,55 -> 0,47; inviluppo 0,52 -> 0,26; norma del gradiente 9,3 -> 1,2 (clip a 1). Nessuna perdita non finita. JEPA per tipo di maschera quasi
  uguale (corta 0,0031, spaziale 0,0030), slab un po' piu' alta (0,0042).
- **Diagnostiche al passo 500** (soglie firmate, decisione 16): rapporto di collasso studente 0,50, teacher 0,22 (soglia 0,05); rango effettivo
  studente 82,5, teacher 89,7 (soglia 38,4). **Nessun allarme.**
- **Da guardare, senza soglia (nessuna soglia si inventa dopo aver guardato):** le varianze assolute delle uscite del decoder sono piccole
  rispetto alla varianza unitaria data dalla LayerNorm finale: fra campioni 0,008 (studente) e 0,003 (teacher), fra query 0,016. Le uscite hanno
  quindi una grande componente comune a tutte le query e a tutti i campioni, ed e' il caso che rapporto e rango non vedono (review del modello,
  M2). Siamo dentro il warmup (1.000 passi): conta come evolvono nel sanity vero. Se Simone vuole un terzo allarme su queste varianze, va scritto e
  congelato prima del lancio.

**Misura del ritmo e scale, terzo giro, job 59276700: COMPLETED** (47 min 55 s, `lrd_all_serial`, 8 CPU, **0 GPU-ora**; sha256 `throughput.json`
ba75b5e5..., `session_scales.json` 29f4404f..., uguali su cluster e Mac; copie in `~/wearusfm_local/reports/passo6/`, sul cluster in
`results/passo6/loader_59276700/`):
- **scale col metodo nuovo per tutte le sessioni:** 18.975 su 19.034; **59 errori, tutti emg2pose, «nessun tratto lungo almeno 1 s»**: sessioni
  senza nessuna finestra estraibile, che il loader salta e conta. Rispetto alle scale di 59264303 sono identiche 18.508 e diverse 467 (442 emg2pose,
  25 emg2qwerty, le sessioni con tratti corti fra salti; scarto massimo 95%). **Da qui le scale valide sono quelle di 59276700.**
- **ritmo, 8 processi su tutto il mix del manifest:** senza filtro 6,4-10,0 finestre/s per processo (media 7,5); con il filtro 6,5-8,9 (media
  7,5). **Sotto la soglia di ~8,5 per processo, scritta prima della misura.** Tempo: apertura della sessione e lettura del blocco 48-60%,
  target delle ancore 23-25% (erano 55-66%), maschere 16-17%, filtro 10%.
- **Confronto con il consumo misurato, che non cambia la soglia:** la soglia veniva dalla stima del passo 0 (~68 finestre/s per GPU). Il
  collaudo 59277074 consuma ~13 finestre/s per GPU (2,4 s per passo col batch 32), e con 6 processi il loader ne da' ~45: su emg2qwerty l'attesa
  sui dati e' 0,03 s per passo. Al modello del sanity il loader basta con margine ~3,5. Se basti alla soglia del passo 0, o se quella soglia vada
  sostituita col consumo misurato, **lo decide Simone**. Il filtro costa il ~10%: **la decisione 13 (filtro nel dataloader) regge**.

## Sanity JEPA vero: via libera (Simone, 04/10/2026: «vai»)

**Lettura di AG, da correggere se sbagliata:** «vai» risponde al punto del 03/10 sera, che proponeva il sanity vero da **10.000 passi** (~6,7
GPU-ora, un job su `boost_usr_prod`, da zero) e, per l'ancora RVQ (domanda 1), di **collegarla con i codici precalcolati se costa meno di un giorno,
altrimenti partire senza** registrando la deviazione. Nessun terzo allarme: Simone non l'ha chiesto, quindi valgono le soglie firmate (16) e le
varianze assolute si registrano soltanto. Le domande 2-10 restano aperte.

**Ancora RVQ collegata** (stima: meno di un giorno, quindi opzione (a)):
- `src/wearusfm/data/rvq_codes.py` riproduce l'ingresso di V2: vista canonica per prova, identica bit per bit a `to_canonical` (test); stesso
  checkpoint (sha256 0d255bcc..., controllato all'avvio); fattore 21,718 di tutti i run di V2; un canale alla volta con l'indice spaziale fisso;
  **blocchi da 16 patch** (i campioni di V2) dall'inizio di ogni prova. Codice = livello 0 del ramo 0 (D5b). Canali scartati dal QC: nessun
  codice;
- allineamento: la finestra che comincia k passi da 200 ms dopo l'inizio della prova ha come blocco w la patch canonica k + w (D10, decisione
  11). Verificato in modo indipendente: un impulso nel blocco nativo p cade nella patch canonica p a 2000, 1926, 1111 e 1000 Hz;
- il dataloader legge i codici per sessione (`LoaderConfig.rvq_codes_root`); la perdita RVQ conta solo gli slab con un codice (-1 = nessun target);
- **scelta di AG, da confermare: entropia incrociata divisa per ln(8192)**, cosi' all'inizio vale ~1 come le altre ancore invece di ~9. Il
  peso complessivo delle ancore resta 0,2 (firmato).

**Job dei codici per emg2qwerty** (`scripts/slurm/precompute_rvq_codes.sbatch`, `boost_usr_prod`, 1 GPU, 8 CPU, limite 2 h): **costo massimo 2
GPU-ora = 16 ore locali**, atteso ~0,5-1 (la vista canonica sulla CPU domina). Budget del passo 6: usate 0,47 su 100. Riprendibile, uscita in
`$WORK/wearusfm_runs/results/passo6/rvq_codes/` (~0,4 GB per emg2qwerty). Il sanity vero si lancia dopo, con il costo ridichiarato.

**`$WORK` saturo** (letto il 04/10 con `lfs quota`: 1T usato su 1T di quota del progetto, **grazia in scadenza fra ~12 ore**; dopo, ogni scrittura su
`$WORK` fallisce, per tutti gli utenti del progetto). **Decisione di Simone (04/10/2026: «vai con la 2»): la cartella del sanity vero va su
`$SCRATCH/wearusfm_runs/runs/`**: fuori dal repo, senza quota, cancellazione automatica a 40 giorni, che per un sanity non pesa. E' un'eccezione per il
sanity alla regola «uscite in `$WORK/wearusfm_runs/`»: lo spirito della regola, mai dentro il repo, resta. Liberare spazio su `$WORK` (per esempio
con `delete_verified_raw.sbatch`, cancellazione definitiva) resta a Simone.

**Sanity JEPA vero, registrato prima del lancio.** Si lancia quando il job dei codici RVQ 59303528 e' finito con i codici di almeno il 95% delle
sessioni di emg2qwerty:
`sbatch --qos=normal --time=09:00:00 --export=ALL,TIME_LIMIT_S=31500,RVQ_CODES=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/passo6/rvq_codes
scripts/slurm/sanity_jepa.sbatch <manifest_59253155.json.gz> <loader_59276700/session_scales.json> $SCRATCH/wearusfm_runs/runs/sanity_0410 10000`.
- Preset `sanity` (valori firmati 15-20) con l'**ancora RVQ accesa**; scale di sessione del job 59276700; **10.000 passi**, cioe' ~1,1 passaggi su
  emg2qwerty. Soglie d'allarme firmate (16), valutate ogni 500 passi; nessuna soglia nuova.
- **Costo:** ~2,4-2,5 s per passo, quindi **~7 GPU-ora attese, massimo 9 GPU-ora** (limite del job 9 h; il run salva e si ferma a 8 h 45 min). Ore
  locali: ~56 attese, massimo 72. Budget del passo 6: usate 0,47 su 100, piu' al massimo 2 per i codici.
- Se non finisce dentro il job (non atteso), riprende dal checkpoint con un secondo job, da ridichiarare.

**Spazio su `$WORK`: originali verificati tolti da Simone** (04/10/2026, lanciati da Simone dal suo terminale; l'agente non esegue cancellazioni
definitive). Prima ha controllato i log di `verify_touch_scratch_raw`: «CHECKSUM IDENTICI» per tutti e 7 i dataset (camargo2021 5 file,
camargo2021_emg_only 3.147, grabmyo 30.716, putemg 298, capgmyo 1.458, ninapro 140, kaifosh_discrete_gestures 2). Poi ha lanciato il rinfresco delle
date delle copie su `$SCRATCH` (`refresh_scratch.sbatch` sui 7 dataset, **job 59303661**), senza il quale lo script di cancellazione rifiuta copie
piu' vecchie di 24 ore, e la cancellazione con dipendenza `afterok` (`delete_verified_raw.sbatch`, **job 59303674**). Lo script ricontrolla per ogni
dataset presenza, dimensioni, numero di file e freschezza della copia prima di cancellare. CPU, 0 GPU-ora. Esito e quota: sotto, quando chiusi.
**Esito della pulizia** (log copiati in `~/wearusfm_local/reports/`): rinfresco 59303661 COMPLETED in 8 min 23 s, nessun file rimasto vecchio;
cancellazione 59303674 COMPLETED in 45 s: **tolti 6 dataset** (camargo2021, camargo2021_emg_only, grabmyo, putemg, capgmyo,
kaifosh_discrete_gestures). **Conservato ninapro**: «NUMERO FILE DIVERSO (140 vs 419)». La copia su `$SCRATCH` ha piu' file dell'originale, perche' ci
sono stati scaricati direttamente altri DB. Lo script vuole lo stesso numero e quindi non cancella, come da progetto. Togliere anche gli originali
di ninapro, accettando una copia che li contiene tutti (rsync non segnala file mancanti o diversi), sarebbe un cambio del criterio di
cancellazione: lo decide Simone. **Quota di `$WORK` dopo: 942,9 GB su 1 TB, grazia chiusa.**

**Codici RVQ, job 59303528: FALLITO per memoria** (OUT_OF_MEMORY dopo 25 min 36 s; MaxRSS ~120 GB su 120). Scritte 178 sessioni su 1.035, che il
rilancio salta, nessun file a meta'. **Costo 0,43 GPU-ora**; budget del passo 6: usate 0,90 su 100. **Causa:** con `imap_unordered` i 6 processi
preparavano viste canoniche (~140 MB a sessione) piu' in fretta di quanto la GPU le consumasse (~8,5 s a sessione), e i risultati si accumulavano nel
processo principale. **Correzione:** al piu' 5 sessioni in volo (`bounded_map`, test), 3 processi e batch del tokenizer 4.096 (era 1.024).
**Rilancio** con lo stesso sbatch, limite 3 h: ~857 sessioni rimaste a ~8,5 s, quindi ~2 h attese. **Costo massimo 3 GPU-ora = 24 ore locali.**
Il sanity vero parte dopo, come registrato sopra.

## Decisioni del 04/10/2026 (Simone: «approvo tutto», sulla lista di 16 proposte di AG)

**Dati e perdita, dalla finestra 1** (nel codice come opzioni spente nel preset del sanity, che resta quello collaudato):
1. **quote nel tempo (in token):** la probabilita' di una sessione e' il suo peso nel manifest diviso per la lunghezza media delle sue finestre,
   calcolata una volta come le scale;
2. **perdita per finestra:** media prima dentro ogni finestra, poi fra le finestre (JEPA e ancore);
3. **bordi dei tratti:** le patch entro 100 ms dal bordo di un tratto (inizio o fine prova, salto) non hanno target delle ancore; il JEPA resta;
4. **ritmo del dataloader:** la soglia «~8,5 finestre/s per processo» (stima del passo 0) e' sostituita dalla regola di D6a del piano: **attesa sui
   dati <= 2% del passo**, col modello vero. Collaudo 59277074: 1,3%;
5. **batch di validazione** (decisione 16): confermato com'e', cioe' estratto dal pretraining con seme fisso, rapporto di collasso = minimo fra
   studente e teacher.
**Scelte di AG ratificate:**
6. identita' del canale con la posizione nel sistema del sensore, il lato e il gruppo;
7. spostamenti con segno nel bias dell'encoder locale;
8. entropia incrociata RVQ divisa per ln(8192);
9. masking a «tubi», (b) della decisione 12.
**Da dichiarare nei risultati, manifest invariato:**
10. **DB5** resta nell'harness, dichiarato «soggetti visti in pretraining»; il confronto pulito con NeuroRVQ e' su EPN-612 e UCI-EMG;
11. **emg2pose:** 540 registrazioni (7,3 h) hanno il lato fermo nel pretraining e quello in movimento nel test per fasi nuove.
**Finestra 1:**
12. **D11:** target dal decoder a query del teacher (b), con la diagnostica del collasso; target dall'encoder locale (a) come braccio di controllo a
    30M;
13. **D15:** dropout del livello muscolo 0,4;
14. **D16:** frazioni 12,5 / 25 / 50 / 100% a 100M e 300M, sui manifest annidati di `manifest-v1`;
15. **D6a:** montaggi virtuali dalle griglie HD al volo nel dataloader, se l'attesa sui dati resta <= 2%; da implementare prima della finestra 1.
    Le quote sulla topologia presentata si decidono con i numeri;
**Operazioni:**
16. **ninapro su `$WORK`:** nuovo criterio di `delete_verified_raw.sbatch`. Si cancella se ogni file dell'originale e' nella copia con la stessa
    dimensione, anche se la copia ha file in piu'; il confronto per checksum e' gia' passato. Lo lancia Simone.
**Restano da preparare** (fogli con i numeri): D12 (soglia della regola del pilot), D14 (WSD e continuazione), conto del compute della ladder con
D_t = 1,18·10⁸.
**ninapro tolto da `$WORK`** (job 59310213 lanciato da Simone, COMPLETED in 4 s, criterio del punto 16): «la copia ha 279 file in piu'
dell'originale (ammesso): ogni file dell'originale c'e', stessa dimensione», 140 file identici e freschi, cancellato. Gli altri 6 risultavano gia'
assenti. Quota letta alla fine del job: 943 GB su 1 TB; su Lustre l'aggiornamento della quota puo' arrivare in ritardo. **Gli originali grezzi dei
7 dataset ora esistono solo su `$SCRATCH`**: il rinfresco ogni 21 giorni (22/10, 12/11, 03/12, 24/12) e' l'unica protezione dal purge.

**Codici RVQ per emg2qwerty, job 59306259: COMPLETED** (1 h 43 min 54 s, exit 0; log e `index_59306259.json` in `~/wearusfm_local/reports/passo6/`):
857 sessioni calcolate su 857 da fare, 0 errori. In tutto **1.035 sessioni su 1.035** con i codici, comprese le 178 del job 59303528; nessun file a
meta'; 150,6 M token; 353 MB in `$WORK/wearusfm_runs/results/passo6/rvq_codes/`. Checkpoint sha256 0d255bcc... e fattore 21,718 come V2. **Costo
1 h 44 min x 1 GPU = 1,73 GPU-ora**; budget del passo 6: usate 2,63 su 100. Condizione del lancio del sanity vero (almeno il 95%) soddisfatta.
**Sanity JEPA vero sottomesso: job 59318048** (04/10/2026 13:20, comando registrato sopra, commit 4a742cc; `boost_usr_prod`, QoS normal, limite 9 h;
cartella `$SCRATCH/wearusfm_runs/runs/sanity_0410`; ancora RVQ accesa con i codici del job 59306259). In coda per priorita'. Il controllo
automatico non e' scattato fra le 11:22 e le 13:16, quindi il lancio e' stato fatto a mano, con gli stessi controlli.

**L'ancora RVQ nel sanity non impara** (04/10/2026, passo ~3.950 di 59318048). La perdita RVQ, normalizzata, e' scesa da 0,995 a 0,986 nei primi
500 passi e poi e' rimasta li'. Su 100 sessioni di codici di emg2qwerty (17,9 M codici, 7.296 su 8.192 usati), **l'entropia della sola frequenza dei
codici e' 0,982**, cioe' la perdita di una testa che predice solo quanto e' frequente ogni codice. La testa non ha imparato nemmeno quella. Il
codice della patch successiva e' uguale nello 0,02% dei casi (caso); quello di un canale vicino alla stessa patch nel 18%. *Interpretazione di AG:*
il codice descrive la forma d'onda istantanea di quei 200 ms, e sugli slab (nascosti su tutti i canali) prevederla dal contesto equivale a prevedere
la forma d'onda (v10 §6.1). C'e' poi il rumore delle etichette noto da D5b.

**Diagnostica autorizzata da Simone** («sì, fai la prova con la maschera per canale, poi vediamo come va»): partendo dai pesi del sanity
(`--init-from` del suo checkpoint, che si legge e non si tocca), ~600 passi in cui la testa RVQ impara sulle finestre **nascoste su un canale ma
non slab**, con dei vicini visibili (`--rvq-targets channel`, **solo diagnostica**: v10 §6.3 lo esclude dal training). Si registrano la perdita e
l'accuratezza top-1 su quelle finestre e, senza gradiente, sugli slab. Confronti scritti prima: entropia della frequenza 0,982; un canale vicino ha
lo stesso codice nel 18% dei casi. **Lettura:** se sulle finestre di canale la perdita scende chiaramente sotto 0,982, l'informazione c'e' e il
problema sono gli slab; se resta li', sono i codici. Job: `sanity_jepa.sbatch` su `boost_qos_dbg`, 1 GPU, 30 min, **costo massimo 0,5 GPU-ora = 4
ore locali**; budget del passo 6: usate 2,63, piu' il sanity in corso (~7). Cartella `$SCRATCH/wearusfm_runs/runs/rvq_diag_channel_0410`.

## D12 e D14 firmate (Simone, 04/10/2026: «firmo D12 e D14»)

Sui fogli `docs/fogli_firma_finestra1.md`, senza correzioni:
- **D12, regola del pilot sulle epoche:** metriche primarie P1 (cross-soggetto sul ramo sparso, sonda lineare, accuratezza bilanciata sui soggetti di
  test della classe A) e P2 (transfer su EPN-612 e UCI-EMG, media dei due); errore standard per bootstrap sui soggetti di test (1.000), combinato
  sui 2 seed. **E = 4**, salvo che su almeno una metrica primaria Δ₄→₈ superi 2 errori standard **e** Δ₄→₈ >= 0,25·Δ₂→₄: allora **E = 8**.
  E = 1 e 2 si riportano, non si scelgono. Prerequisiti prima della finestra 1: harness del passo 5 (P2) e sonda cross-soggetto (P1).
- **D14, schedule WSD:** warmup lineare sul 2% dei passi a D (almeno 1.000); fase stabile al picco scelto dalla calibrazione a 30-100M; **decay
  lineare a zero sul 20% finale**; checkpoint pre-decay (80% di D) e post-decay (D). La continuazione a 2D riparte dal pre-decay, prosegue stabile
  fino all'80% di 2D, poi decay. Pilot con un run per seed (stabile fino a 6,4 epoche, decay fino a 8) e rami di decay da 0,8, 1,6 e 3,2 epoche
  per E = 1, 2, 4: ~9,4 epoche per seed. Teacher EMA costante 0,996; AdamW, weight decay 0,05, clip 1,0, batch 32 al 30M (i rung maggiori dalla
  calibrazione).

**Diagnostica RVQ sul masking di canale, job 59328078: COMPLETED** (26 min 15 s, 628 passi dai pesi del sanity al passo 4.000; **costo 0,44
GPU-ora**; budget del passo 6: usate ~3,1 su 100 piu' il sanity in corso). Sulle finestre nascoste su un canale ma non slab (~5.800 per passo, con dei
vicini visibili allo stesso istante) la perdita RVQ resta **0,983** dal primo all'ultimo blocco di 100 passi; l'accuratezza top-1 resta ~0,05%. Sugli
slab, senza gradiente: 0,983-0,984, ~0,05%. Riferimenti scritti prima: entropia della sola frequenza dei codici 0,982; caso 1/7.000 ≈ 0,014%;
codice piu' frequente 0,03%. **Lettura, secondo la regola scritta prima:** la perdita non scende sotto 0,982 nemmeno coi vicini visibili, quindi il
problema non sono gli slab ma **i codici come target per questo modello**: non sono prevedibili dalle sue rappresentazioni oltre la loro frequenza.
*Interpretazioni di AG, da verificare:* (1) il codice descrive la forma d'onda fine di 200 ms, che JEPA e ancore (ampiezza, spettro) non chiedono di
rappresentare; (2) 8.192 classi quasi equiprobabili e rumorose (D5b: su emg2pose il codice resta uguale solo nel 46% dei casi) sono un target
troppo fine per una testa lineare in poche centinaia di passi. Le opzioni (togliere l'ancora, cambiarne il target, provarla solo in una fase) vanno
a Simone (04/10: «poi vediamo se ha senso provare a modificare l'RVQ, a toglierlo o a capire se si puo finetunare in qualche modo, in coda o in testa»).

## Replica di NeuroRVQ: foglio firmato con due correzioni (Simone, 04/10/2026)

«ok firmo. pero 1) [...] No, teniamo tutto dato che come abbiamo gia deciso non ci interessano le licenze 2) teniamo 20-400Hz di banda. Per il resto
approvo.» Sul foglio `docs/foglio_replica_neurorvq.md`:
- **banda 20-400 Hz** (come il codice d'esempio di NeuroRVQ: Butterworth di ordine 3 a fase zero, taglio alto min(400, fs/2) - 0,5 Hz), non i 20-90
  Hz del paper: scostamento dichiarato nei risultati;
- **il codice di NeuroRVQ che serve alla replica entra nel repo** (`third_party/neurorvq/`, dal commit 926e770, con la licenza CC BY-NC 4.0 e
  l'attribuzione originali), non piu' solo nel clone su `$SCRATCH`. Coerente con le decisioni d'uso sulle licenze del 30/09;
- tutto il resto come proposto: 3 split casuali per soggetto, Tab. 13 per il fine-tuning, testa del paper (concatenazione sui canali, media sul
  tempo), epoca con la miglior accuratezza di validazione, **tolleranza EPN-612 [92,65; 96,65] e UCI-EMG [87,23; 91,63]** sulla media dei 3 split.

## Ancora RVQ: tolta dalla finestra 1 (Simone, 04/10/2026: «accetto quello che proponi su ancora RVQ e finestra 1»)

Sulla proposta di AG dopo le due misure (sanity e diagnostica 59328078: la testa RVQ non va oltre la frequenza dei codici, ne' sugli slab ne' coi
vicini visibili):
1. **la configurazione di default della finestra 1 non ha l'ancora RVQ.** Il sanity in corso la tiene fino alla fine, come lanciato. Per D5b il
   confronto con/senza RVQ era l'arbitro dell'utilita' dell'ancora: con il target attuale, che non si impara, non si fa;
2. **prima di decidere se un target diverso merita un braccio di ablation a 30M** (vettore del codebook in regressione, oppure codici raggruppati),
   un **test sulla CPU**: le rappresentazioni delle patch **visibili** del modello del sanity bastano a leggere il codice? Protocollo e soglie si
   scrivono qui e si congelano prima di lanciare il test (job CPU, 0 GPU-ora).

## Ancore in frequenza: confronto dei target candidati (Simone, 04/10/2026: «benissimo accetto la tua proposta, cosi vediamo se c'e qualcosa di meglio»)

**Protocollo e soglie, congelati prima del test** (job su `boost_qos_dbg`, 1 GPU, 30 min, **costo massimo 0,5 GPU-ora**; si lancia dopo la fine del
sanity 59318048, col suo checkpoint finale; budget del passo 6: usate ~3,1, piu' il sanity ~7):
- **Dati:** sessioni di pretraining di emg2qwerty, 40 utenti divisi per utente in 30 di addestramento e 10 di test (seme 0); finestre da 4 s, filtro e
  scala del dataloader; unita' = (canale, blocco da 200 ms della griglia della prova). Al piu' 100.000 unita' di addestramento e 30.000 di test.
- **Rappresentazioni** del modello del sanity (teacher): (V) uscite del decoder a query sulle patch **visibili**, mediate sulle 8 patch del blocco;
  (H) uscite del decoder sulle patch **nascoste** dalle maschere D10, solo blocchi nascosti per intero su quel canale. Confronto: (B) i target delle
  ancore attuali dello stesso blocco (log-RMS e 5 bande); (R) lo stesso modello all'inizializzazione.
- **Target candidati** per unita': (a) gruppo del codice RVQ (k-means a 64 sui vettori del codebook del livello 0 del ramo 0); (b) vettore del codebook
  del codice (128 numeri); (c) log della potenza in **16 bande** log-spaziate fra 20 e 450 Hz sul blocco; (d) frequenza media e mediana del blocco;
  (e) feature continue dell'encoder di NeuroRVQ prima della quantizzazione, ramo 0 (128 numeri); (f) famiglia spettrale (k-means a 64 sulle
  16 bande, adattato sulle unita' di addestramento).
- **Sonde lineari** addestrate sugli utenti di addestramento e lette sugli utenti di test: regressione ridge (R², media sulle componenti) per b, c, d,
  e; regressione logistica per a, f (accuratezza contro la classe piu' frequente).
- **Regola:** un target e' **prevedibile** se dalle rappresentazioni (H) raggiunge **R² >= 0,3** (regressione) o un'accuratezza di almeno **10
  punti sopra la classe piu' frequente** (64 gruppi). Fra i prevedibili, si propone a Simone quello che aggiunge di piu' alle ancore attuali, cioe'
  con l'R² piu' basso quando lo si predice da (B). Le varianti RVQ (a, b) meritano un braccio di ablation a 30M solo se passano la stessa soglia. Se
  nessun candidato passa, le ancore restano quelle attuali.
**Aggiornamento prima del lancio** (Simone, 04/10/2026: «intanto testiamo comunque le 5 opzioni che mi hai dato prima»): si aggiunge l'opzione 3,
**(g) tempo-frequenza a piu' scale**: log della potenza in 6 bande su ognuno dei 4 sottoblocchi da 50 ms e in 24 bande su 500 ms centrati sul blocco;
la scala da 200 ms e' gia' la (c). Le cinque opzioni proposte sono quindi c, d, g, e, f, piu' le due varianti RVQ (a, b). **Stessa regola e stesse
soglie.** Per «intanto», il test si lancia **subito sull'ultimo checkpoint del sanity** (passo ~6.000), non su quello finale, e si ripete sul finale
quando il sanity chiude. Due job:
- estrazione: `anchor_candidates.sbatch` su `boost_qos_dbg`, 1 GPU, 30 min, **costo massimo 0,5 GPU-ora = 4 ore locali**; si tiene il 15% delle
  unita', a caso con seme fisso, per la dimensione del file;
- sonde: `anchor_candidates_probe.sbatch` su `lrd_all_serial`, CPU, **0 GPU-ora**, in dipendenza `afterok`.
Uscite in `$SCRATCH/wearusfm_runs/anchor_candidates/`. Budget del passo 6: usate ~3,1, piu' il sanity (~7).
**Estrazione 59331712: COMPLETED** (22 min 11 s; checkpoint del sanity al passo 5.850): 191.482 unita', di cui 79.265 nascoste; 613 MB in
`$SCRATCH/wearusfm_runs/anchor_candidates/units_ckpt_latest.npz`. **Costo 0,37 GPU-ora.** Sonde: il job 59331741 e' stato cancellato dal sistema
dopo 28 s («container_p_join: open failed for /scratch_local/tmpfs/59331741/.ns», guasto del nodo login08, non del codice); risottomesso come
**59333079**, uguale (CPU, 0 GPU-ora).
**Esito del confronto dei target candidati** (sonde 59333079, 9 min 40 s, CPU, 0 GPU-ora; report in `~/wearusfm_local/reports/passo6/anchor_candidates/`;
checkpoint del sanity al passo 5.850). 100.000 unita' di addestramento e 30.000 di test; 59.502 e 19.763 nascoste. Regressione: R²; gruppi e
famiglie: accuratezza (classe piu' frequente).

| Target | V visibili | H nascoste | R inizializzazione | B ancore attuali | Prevedibile |
|---|---|---|---|---|---|
| a gruppi RVQ (64) | 4,2% (4,1%) | 3,8% (3,9%) | 3,0% | 4,2% | no |
| b vettore del codebook | 0,000 | 0,000 | 0,000 | 0,000 | no |
| e feature continue NeuroRVQ, ramo 0 | 0,006 | 0,004 | 0,000 | 0,005 | no |
| c 16 bande | 0,752 | **0,559** | 0,267 | 0,782 | si' |
| d frequenza media e mediana | 0,741 | **0,505** | 0,251 | 0,869 | si' |
| g multi-scala (50 + 500 ms) | 0,692 | **0,529** | 0,263 | **0,646** | si' |
| f famiglie spettrali (64) | 23,4% (2,3%) | 12,6% (2,3%) | 4,7% | 30,3% | si', al limite (+10,3 punti) |

**Lettura secondo la regola congelata:**
- **(a), (b) ed (e), cioe' tutto cio' che viene da NeuroRVQ, non sono leggibili nemmeno dalle patch visibili.** Per la regola scritta prima il
  percorso RVQ si chiude, anche per le finestre successive: nessun braccio di ablation. *Cautela di AG:* per (a) e (b) l'allineamento e' provato
  dai test (impulso); per (e) le feature si calcolano sul blocco canonico da 16 patch preprocessato da solo. Un errore di allineamento darebbe
  anch'esso R² ~0, e non e' escluso da un test dedicato.
- **I quattro candidati spettrali sono prevedibili dalle patch nascoste.** Il modello del sanity ha imparato il contenuto in frequenza: V vale
  ~0,75 contro ~0,26 del modello all'inizializzazione.
- **Fra i prevedibili, quello che aggiunge di piu' alle ancore attuali** (R² piu' basso da B) e' **(g) multi-scala**: 0,646, cioe' ~35% della
  sua varianza non e' gia' nelle ancore attuali. Le 16 bande (0,78) e i momenti (0,87) sono quasi gia' contenuti nelle 5 bande e nel log-RMS. Le
  famiglie spettrali si leggono meglio dalle ancore attuali (30,3%) che dal modello (23,4%). **Proposta per la regola: (g).** La decisione e' di
  Simone.
- Da ripetere sul checkpoint finale del sanity (prevista; stesso protocollo).

## Ancora multi-scala nella finestra 1 (Simone, 04/10/2026: «ok per l'ancora multi-scala nella finestra 1»)

Esito del confronto dei target candidati, secondo la regola congelata. Dalla finestra 1 **si aggiunge alle ancore attuali** (log-RMS, forma in 5
bande, inviluppo) la **magnitudine tempo-frequenza a due scale**: log della potenza in **6 bande** su una finestra da **50 ms** e in **24 bande** su
una finestra da **500 ms**. Le bande sono log-spaziate fra 20 e 450 Hz; quelle oltre il limite del canale non contano, come per le 5 bande. Il peso
complessivo delle ancore resta 0,2 (firmato).
*Dettaglio d'implementazione di AG:* le ancore sono per patch da 25 ms, quindi le due finestre sono **centrate sulla patch**. Nel test erano i
4 sottoblocchi da 50 ms e i 500 ms centrati su un blocco da 200 ms: stesse scale, diversa granularita'. Vale anche qui la zona di bordo da 100 ms
(decisione 3). Nel sanity e' spenta. **L'ancora RVQ non c'e'** (decisione del 04/10).
**Implementata** (commit di questa nota; opzione `multiscale_anchor` del modello e del dataloader, accesa da `with_window1_rules`, spenta nel
sanity). Test: il tono a 300 Hz cade nella banda giusta a tutte e due le scale; a 200 Hz le bande alte non ci sono; la zona di bordo vale; in un
training con le regole della finestra 1 compaiono `ms_fast` e `ms_slow` e non compare la perdita RVQ. **Costo:** i target salgono da 12 a 46 ms per
finestra (32 canali, 4 s, 2 kHz, sul Mac), quasi tutto la finestra da 500 ms ricalcolata a ogni patch. Col margine misurato (~45 finestre/s di
dataloader contro ~13 consumate) basta, ma va ottimizzato prima della finestra 1 (passo di 100 ms con interpolazione, o FFT a passo fisso).

**Replica di NeuroRVQ: script e collaudo** (script `scripts/replicate_neurorvq.py`, job `scripts/slurm/replicate_neurorvq.sbatch`; test su dati
sintetici con un FM a caso). Collaudo previsto dal foglio firmato: EPN-612, seme 0, 2 epoche, su `boost_qos_dbg`, **costo massimo 0,5 GPU-ora = 4
ore locali**. Serve a verificare il caricamento del checkpoint pubblico (chiavi mancanti o inattese nel report) e a misurare il tempo per epoca;
il costo dei run completi (3 semi x 2 dataset x 100 epoche) si ridichiara dopo. Budget del passo 5: <= 300 GPU-ora, usate 0 (gli smoke del 23-24/09
erano su CPU).
**Collaudo 1 della replica, 59336240: COMPLETED** (10 min 27 s, **0,17 GPU-ora**; budget del passo 5: usate 0,17 su 300).
- **Checkpoint pubblico caricato:** mancano solo le 8 chiavi `fc_norm_1..4`, usate dal percorso RVQ e non dalla classificazione; ci sono 131
  chiavi in piu', le teste del loro pretraining (`mask_token`, `norm_pre`, `head_pre_*`...).
- EPN-612: 91.800 finestre, 612 soggetti (split col seme 0: 428 / 61 / 123), 6 classi. Accuratezza di test dopo 2 epoche ~46%: non e' un
  risultato, sono 2 epoche su 100.
- **Due problemi trovati, corretti nel commit 493f616:** (1) il loader tagliava tutti i campioni alla lunghezza del piu' corto (2,4 s): ora 5 s
  fissi con `fix_length`, come il loro codice e il paper; (2) 177 s per epoca col ricampionamento sulla CPU a ogni batch, cioe' ~10 ore per seme
  coi campioni da 5 s. Ora i dati a 1 kHz stanno sulla GPU in float16 e il test si valuta solo quando la validazione migliora.
**Collaudo 2** (stessa autorizzazione; `boost_qos_dbg`, 30 min, **massimo 0,5 GPU-ora**): EPN-612 a 5 s, seme 0, 3 epoche, per misurare il tempo
per epoca e ridichiarare il costo dei run completi.
**Collaudo 2 della replica, 59336887: COMPLETED** (24 min 33 s, **0,41 GPU-ora**; budget del passo 5: usate 0,58 su 300). EPN-612 a 5 s, seme 0,
3 epoche: test **88,6% -> 90,2% -> 92,75%**; la migliore in validazione e' la 3 (93,4%). E' gia' dentro la tolleranza ([92,65; 96,65]) con 3
epoche e un seme: la pipeline e' giusta, e le finestre a 5 s erano la differenza (il collaudo 1, a 2,4 s per errore, era al 46%). **Tempo: 373 s
per epoca**, quindi 100 epoche fanno ~10,4 h per seme. **Costo dei run completi, ridichiarato:** EPN-612 3 semi ~31 GPU-ora, UCI-EMG ~1, **totale
~32 GPU-ora** (~256 ore locali), contro i 4-6 stimati nel foglio: si lancia solo col via di Simone.

## Sanity JEPA vero, job 59318048: COMPLETED, nessun allarme (04/10/2026)

10.000 passi in 7 h 13 min 51 s (13:17-20:31), exit 0, 31,5 M parametri (con la testa RVQ), memoria GPU massima 18,0 GiB. **Costo 7,23
GPU-ora**; budget del passo 6: usate ~10,7 su 100 (collaudi 0,47, codici RVQ 2,16, diagnostica RVQ 0,44, estrazione dei target 0,37, sanity 7,23).
**Diagnostiche** (soglie firmate: rapporto 0,05, rango 38,4), studente / teacher:

| Passo | Rapporto di collasso | Rango effettivo | Varianza fra campioni |
|---|---|---|---|
| 500 | 0,43 / 0,22 | 86 / 91 | 0,006 / 0,003 |
| 2.000 | 0,64 / 0,65 | 111 / 109 | 0,013 / 0,012 |
| 4.000 | 0,77 / 0,75 | 129 / 128 | 0,019 / 0,018 |
| 6.000 | 0,82 / 0,82 | 136 / 136 | 0,021 / 0,022 |
| 8.000 | 0,82 / 0,85 | 146 / 139 | 0,022 / 0,024 |
| 10.000 | **0,86 / 0,88** | **147 / 145** | **0,024 / 0,027** |

Minimo dello studente in tutto il run: rapporto 0,43, rango 86. Le varianze assolute, il caso che rapporto e rango non vedono, sono cresciute di
~4 volte: le uscite dipendono sempre piu' dal contenuto.
**Perdite** (medie su 500 passi, inizio -> fine): totale 0,55 -> 0,31; log-RMS 0,49 -> 0,18; forma spettrale 0,73 -> 0,29; inviluppo 0,34 -> 0,08;
JEPA ~0,005-0,007 dopo il passo 1.000; norma del gradiente 3,1 -> 0,41. RVQ 0,995 -> 0,984: non scende sotto l'entropia della frequenza dei codici
(0,982), come gia' misurato. **Attese sui dati:** 0,06-0,10 s per passo fino al passo 8.000, poi 0,2-0,8 s negli ultimi 1.500 passi. *Interpretazione
di AG, da verificare:* in quegli stessi minuti giravano gli altri job sugli stessi dischi (estrazione dei target, collaudi della replica, che
leggono EPN-612). Le scale e la lettura non sono cambiate.
**Lettura:** il criterio del passo 6, «il sanity JEPA su un solo dataset omogeneo non collassa», e' soddisfatto, con tutti gli indicatori in
crescita fino alla fine. Gli altri criteri del «Chiuso quando» sono gia' coperti da test: tre topologie in avanti e indietro, equivarianza, diagnostica
del collasso, FLOP per modulo. **La chiusura del passo 6 la firma Simone.**

**Confronto dei target candidati ripetuto sul checkpoint finale del sanity** (passo 10.000; previsto dal protocollo, stesse unita' e stessa regola).
Estrazione **59340042: COMPLETED** (16 min 0 s, 1 GPU, **0,27 GPU-ora**); sonde **59340045: COMPLETED** (9 min 20 s, CPU, 0 GPU-ora). Report
`report_ckpt_final.json` in `~/wearusfm_local/reports/passo6/anchor_candidates/`. Budget del passo 6: usate ~11,0 su 100. Unita' identiche al
passo 5.850 (191.482; 100.000 / 30.000; nascoste 59.502 / 19.763), quindi R e B identici; cambiano solo V e H.

| Target | V 5.850 -> finale | H 5.850 -> finale | B ancore attuali | Prevedibile |
|---|---|---|---|---|
| a gruppi RVQ (64) | 4,2% -> 4,3% (4,1%) | 3,8% -> 3,8% (3,9%) | 4,2% | no |
| b vettore del codebook | 0,000 -> 0,001 | 0,000 -> 0,000 | 0,000 | no |
| e feature continue NeuroRVQ, ramo 0 | 0,006 -> 0,007 | 0,004 -> 0,005 | 0,005 | no |
| c 16 bande | 0,752 -> 0,779 | 0,559 -> **0,583** | 0,782 | si' |
| d frequenza media e mediana | 0,741 -> 0,785 | 0,505 -> **0,530** | 0,869 | si' |
| g multi-scala (50 + 500 ms) | 0,692 -> 0,717 | 0,529 -> **0,551** | **0,646** | si' |
| f famiglie spettrali (64) | 23,4% -> 26,0% (2,3%) | 12,6% -> 13,9% (2,3%) | 30,3% | si' (+11,6 punti) |

**Lettura:** l'esito non cambia. I candidati spettrali salgono di ~0,02-0,04 di R² negli ultimi ~4.000 passi (il modello continuava a imparare il
contenuto in frequenza, coerente con le diagnostiche); quelli di NeuroRVQ restano a zero. (g) resta quello che aggiunge di piu' alle ancore attuali
(B fisso a 0,646): la scelta firmata dell'ancora multi-scala nella finestra 1 e' confermata. Le famiglie spettrali passano ora piu' chiaramente la
soglia (+11,6 punti), ma restano piu' leggibili dalle ancore attuali (30,3%) che dal modello (26,0%).

## Passo 6 chiuso (Simone, 05/10/2026: «firmo passo 6»)

Firmata la chiusura del passo 6 sulla base del sanity 59318048 e dei test elencati sopra. Consuntivo: **~11,0 GPU-ora su 100**. Restano da fare
prima della finestra 1, fuori dal criterio del passo: ottimizzare i target multi-scala, i montaggi virtuali D6a, la sonda P1 fra soggetti (D12).

## Replica di NeuroRVQ: run completi (Simone, 05/10/2026: «parti con la run completa di replica rvq sulle gpu»)

Protocollo e tolleranza del foglio firmato il 04/10 (`docs/foglio_replica_neurorvq.md`), invariati; costo dichiarato dopo il collaudo 2: **~32
GPU-ora = ~256 ore locali**. Budget del passo 5: usate 0,58 su 300, dopo i run ~33.
- **EPN-612:** 3 job da 1 GPU, uno per seme (0, 1, 2), in parallelo su `boost_usr_prod`, limite **14 h** (stima ~10,4 h per seme col tempo per
  epoca del collaudo 2; il limite non e' il costo, che e' il tempo reale);
- **UCI-EMG:** 1 job da 1 GPU coi 3 semi in fila, limite 4 h (stima ~1 GPU-ora, nessun collaudo sulla GPU: se il percorso UCI ha un errore
  costa minuti).
Uscite in `$WORK/wearusfm_runs/results/passo5/replica_neurorvq/`, un JSON per job; la media sui 3 semi di EPN-612 si calcola dai tre JSON. La
riuscita si giudica sulla media dei 3 split col modello della miglior epoca di validazione, come da foglio.
**Sottomessi** (05/10/2026): EPN-612 semi 0, 1, 2 = **59363702, 59363704, 59363705**; UCI-EMG = **59363706**. Repo remoto a c2f4bc2, pulito.

## Ancora multi-scala: la scala lenta ogni 100 ms (05/10/2026; dettaglio d'implementazione di AG)

Il costo dei target saliva da 12 a 52 ms per finestra (2 kHz, 32 canali, 4 s, sul Mac), di cui ~32 la finestra da 500 ms ricalcolata a ogni patch
da 25 ms, quasi tutto FFT (5.120 FFT da 1.000 punti per finestra di dati). Ora la scala lenta ha **una finestra da 500 ms ogni 100 ms** (4 patch),
centrata sul gruppo di 4 patch e assegnata a tutte e 4: ogni patch riceve lo spettro di una finestra il cui centro dista al piu' 37,5 ms dal suo,
il 7,5% della finestra. La scala rapida (50 ms) resta per patch. **Costo: 52 -> 20 ms per finestra** (1 kHz e 16 canali: 12 -> 7 ms). Non cambia
la decisione firmata (24 bande su 500 ms): cambia il centraggio, che era un dettaglio mio, e resta piu' fine di quello del test dei candidati
(500 ms centrati su blocchi da 200 ms). Test: dentro un gruppo il target e' costante e coincide col calcolo diretto sulla finestra del gruppo.

## D6a: montaggi virtuali implementati, spenti, foglio da firmare (05/10/2026)

Decisione 15 del 04/10. Parametri proposti da AG in `docs/foglio_d6a_montaggi_virtuali.md` (da firmare): classe C, p = 0,5; sottogriglie di una
griglia (4x8, 8x4, 4x4 elettrodi, passo 1 o 2), meta' monopolari e meta' bipolari (compagno meno elettrodo, lungo righe o colonne); topologia
presentata = griglia, quindi quote invariate; scala bipolare della sua derivazione. Codice: `data/virtual_montage.py`, `LoaderConfig.virtual`
(None = spento, com'e' nel sanity e nelle regole attuali della finestra 1); il montaggio si applica **prima del filtro** (lineare: test
dell'uguaglianza); posizioni nel sistema della griglia d'origine (`grid_extent`); nelle metriche per passo, finestre per classe e quante
virtuali; scale bipolari precalcolabili con `scripts/measure_loader.py --bipolar-scales`. Misura sul Mac (sessione sintetica tipo Hyser, 256
canali a 2048 Hz, un processo): griglia intera 3,4 finestre/s, montaggio virtuale 36,6, con p = 0,5 5,7.

## P1 (D12): sonda cross-soggetto scritta, protocollo da firmare (05/10/2026)

Protocollo proposto da AG in `docs/foglio_p1_cross_soggetto.md`: NinaPro DB2, DB3, DB6 (i soli della classe A con soggetti di test ed etichette
allineate all'EMG); `restimulus` senza riposo; finestre da 1 s senza sovrapposizione; sonda addestrata sui soggetti di pretraining (C sul 10% di
loro), letta sui soggetti di test degli split v1; P1 = media pesata per i soggetti di test (8, 3, 2), errore standard per bootstrap stratificato.
Codice: `harness/p1_cross_subject.py`, `scripts/probe_p1.py` (fasi extract su GPU e probe su CPU), job `probe_p1.sbatch` e `probe_p1_probe.sbatch`;
`linear_probe` ora restituisce anche le predizioni di test. Verificato sui dati processati di Leonardo (05/10, sola lettura): DB2 e DB3 numerano
i movimenti 1-49 attraverso i tre esercizi, DB6 le prese fino a 11, riempimento -1 in coda a DB2 E3. *Sul Mac:* sklearn caricato prima di torch
manda in segmentation fault le operazioni di torch a piu' thread (venv del tokenizer): lo script carica torch per primo, il test anche.

## Replica di NeuroRVQ, UCI-EMG: run 59363706 non valido (errore di unita' nei dati), corretto e rilanciato (05/10/2026)

**59363706: COMPLETED** (3 min 2 s, **0,05 GPU-ora**), media sui 3 semi **41,0%**, FUORI tolleranza ([87,23; 91,63]): seme 0 16,5%, seme 2 19,5%
(al caso: perdita ferma a ln 6), seme 1 **87,0%** (si sblocca solo dopo ~60 epoche). 662 finestre da 1 s, 36 soggetti.
**Causa, trovata cercando errori di dati come vuole il foglio:** i file di UCI-EMG salvano i **codici a 8 bit del Myo moltiplicati per 1e-5**
(valori multipli di 1e-5, massimo |x| = 0,00128 = 128e-5; letti su Leonardo il 05/10), mentre EPN-612, stesso bracciale, li salva come sono
(deviazione standard 1-4 per canale). Senza normalizzazione (foglio), NeuroRVQ vedeva UCI ~1e4-1e5 volte piu' piccolo di EPN e del suo
pretraining: l'ottimizzazione non partiva.
**Correzione** (`harness/benchmarks.py`, `to_myo_codes`): UCI-EMG si legge **nei codici del Myo** (x 1e5), le stesse unita' di EPN-612; errore
se i valori non sono multipli di 1e-5. **Non e' una normalizzazione** (nessuna statistica dei dati, una costante del formato) e non cerca
parametri: riporta i due dataset dello stesso bracciale alle stesse unita'. **Da dichiarare** nella replica come scelta sul formato dei dati. La
sonda sul nostro FM (P2) non cambia: la scala per soggetto assorbe la costante.
**Rilancio:** UCI-EMG, 3 semi, stessa configurazione (stima ~0,05 GPU-ora, dentro i ~32 autorizzati per i run completi). Le 662 finestre (~4 passi
per epoca a batch 128) restano come da foglio; se anche coi codici i semi si sbloccano tardi, si riporta.
**Rilancio UCI-EMG, 59368048: COMPLETED** (3 min 0 s, **0,05 GPU-ora**; budget del passo 5: usate ~0,68 su 300 piu' i 3 run di EPN-612 in corso).
Coi codici del Myo l'addestramento parte per tutti i semi (dall'epoca 1 val 57-65%), ma la media sui 3 semi col modello della miglior epoca di
validazione e' **78,66% (± 7,6)**: **FUORI tolleranza** ([87,23; 91,63]). Per seme: 83,5 / 82,6 / 69,9 (miglior epoca 14, 15, 19); all'ultima
epoca 86,5 / 83,5 / 74,8 (media 81,6). La perdita va a ~0 entro l'epoca 20-30: memorizza. 662 finestre in tutto: test ~120-133 finestre (7
soggetti), validazione ~70-90 (4 soggetti), quindi la scelta dell'epoca e' rumorosa e il numero per seme varia di ~15 punti.
**Lettura secondo il foglio:** l'errore di dati trovato (unita') e' corretto; non ne vedo altri nell'harness. La differenza (~11 punti) si
dichiara. *Interpretazione di AG, non verificabile:* le finestre da 1 s senza sovrapposizione erano una mia scelta dove il paper tace; con un
passo piu' corto le finestre sarebbero di piu', ma cambiarlo ora, dopo aver visto il numero, sarebbe cercare i parametri: decide Simone.

## P2 (D12): sonda del nostro FM su EPN-612 e UCI-EMG scritta (05/10/2026)

`scripts/probe_p2.py` (fasi extract su GPU e probe su CPU; job `probe_p2.sbatch` e `probe_p2_probe.sbatch`): stesse finestre e stessi split della
replica (foglio firmato il 04/10), ingresso e feature come P1. *Scelta di AG, da dichiarare:* per dataset, media dell'accuratezza bilanciata sui 3
split; SE = radice della media dei quadrati degli SE di bootstrap degli split (i test si sovrappongono, la media non riduce l'errore); P2 = media
dei due dataset, SE = radice della somma dei quadrati / 2. Costo stimato dell'estrazione: <= 1 GPU-ora per checkpoint (EPN-612: 91.800 finestre
da 5 s, due ritagli da 4 s); da misurare al primo lancio, che chiede la conferma di Simone.

## D6a e P1 firmati; collaudi di P1 e P2 e scale bipolari (Simone, 05/10/2026: «accetto tutti e 4, firma compresa»)

1. **D6a firmato** senza correzioni (`docs/foglio_d6a_montaggi_virtuali.md`): montaggi virtuali **accesi in `with_window1_rules`**
   (`VirtualSpec()`: classe C, p = 0,5, meta' bipolari; test); il sanity resta senza.
2. **P1 firmato** senza correzioni (`docs/foglio_p1_cross_soggetto.md`).
3. **Collaudo di P1 e P2 sul checkpoint finale del sanity** (`$SCRATCH/wearusfm_runs/runs/sanity_0410`), sui dati veri: P1 estrazione su
   `boost_qos_dbg`, **al piu' 0,5 GPU-ora = 4 ore locali**; P2 estrazione su `boost_usr_prod` con limite 1 h, **al piu' 1 GPU-ora = 8 ore
   locali**; sonde su `lrd_all_serial` in dipendenza, 0 GPU-ora. Budget del passo 5: usate ~0,68 su 300 piu' i 3 run di EPN-612 in corso
   (~31). Uscite in `$SCRATCH/wearusfm_runs/p1_sanity_0410/` e `p2_sanity_0410/`.
4. **Scale bipolari** dei montaggi virtuali: `measure_loader.sbatch <manifest> <loader_59276700/session_scales.json> --bipolar-scales
   --skip-rate` (CPU, **0 GPU-ora**); lo sbatch ora passa gli argomenti in piu' allo script.
5. **UCI-EMG:** «accetto tutti e 4» sul punto che offriva due strade (dichiarare la replica non riuscita, oppure una variante delle finestre
   dichiarata a posteriori). Si dichiara non riuscita (78,66%, registrato sopra); la variante **non si lancia** finche' Simone non la sceglie
   esplicitamente.
**Sottomessi** (05/10/2026, repo a 6ff2e57): P1 estrazione **59412499**, sonde **59412507** (afterok); P2 estrazione **59412516**, sonde
**59412535** (afterok); scale bipolari **59412542**.
**Scale bipolari, 59412542: COMPLETED** (18 min 3 s, CPU, 0 GPU-ora): **904 scale bipolari** (226 sessioni di classe C x 4), **nessun errore**
bipolare; i 59 errori sono sessioni di emg2pose senza tratti da 1 s, gia' saltate dal dataloader. File completo (19.879 scale) in
`$WORK/wearusfm_runs/results/passo6/loader_59412542/session_scales.json`: e' quello da usare dalla finestra 1. Rapporto scala bipolare /
monopolare (mediana per sessione): **Hyser 0,25-0,41** (vicini molto correlati: la bipolare e' piu' piccola, come atteso con un riferimento
comune); **CSL-hdemg 1,17-1,35** e **CapgMyo 0,87-1,59** (vicini poco correlati; per CapgMyo lungo le righe oltre 1,41 = sqrt(2), cioe' vicini
correlati negativamente). *Da verificare (fatto da aprire):* se i canali di CapgMyo e CSL-hdemg siano gia' differenziali; in quel caso la
«bipolare» virtuale sarebbe un doppio differenziale. Conferma comunque la scelta della scala per derivazione: con quella monopolare l'ampiezza
cambierebbe di un fattore 0,25-1,6 a seconda del dataset.
**P1, estrazione 59412499: COMPLETED** (13 min 42 s, **0,23 GPU-ora**): DB2 46.452 finestre (28 / 4 / 8 soggetti), DB3 13.681 (7 / 1 / 3), DB6
42.044 (7 / 1 / 2). Sonde 59412507 in corso.

## Replica di NeuroRVQ su EPN-612: RIUSCITA; collaudi di P1 e P2 sul modello del sanity (05/10/2026)

**EPN-612, 59363702 / 59363704 / 59363705: COMPLETED** (9 h 50 min, 9 h 54 min, 9 h 53 min: **29,6 GPU-ora**). Test col modello della miglior
epoca di validazione: **94,32 / 95,28 / 96,56** (epoche 56, 67, 75), **media 95,39**, pubblicato 94,65 ± 2,0: **DENTRO** la tolleranza ([92,65;
96,65]). All'ultima epoca 94,31 / 95,50 / 96,57. Da dichiarare: banda 20-400 Hz (correzione di Simone al 20-90 del paper), split per soggetto
nostri (semi 0, 1, 2). Con UCI-EMG fuori (78,66%, registrato sopra): **la replica riesce su uno dei due dataset**; la chiusura del passo 5 la
firma Simone.
**P1, sonde 59412507: COMPLETED** (22 min, CPU). Sul modello del sanity (31M, 10.000 passi solo su emg2qwerty, nessun dato sparso visto):
DB2 **18,4% ± 2,4** (49 classi, caso 2%; train 37,7%), DB3 **3,1% ± 0,6** (49 classi, ~al caso; train 19,3%), DB6 **38,5% ± 3,7** (7 classi, caso
14,3%). **P1 = 17,97 ± 1,73** (pesata per soggetti di test; a pesi uguali 20,0). Nessuna finestra tolta per classi assenti dal train.
**P2, estrazione 59412516: COMPLETED** (41 min 19 s, **0,69 GPU-ora**); sonde 59412535 (33 min, CPU): EPN-612 **83,88 ± 0,89** (split 83,3 /
83,9 / 84,5), UCI-EMG **84,60 ± 4,64** (88,0 / 86,2 / 79,7). **P2 = 84,24 ± 2,36.**
**Lettura (AG):** e' un collaudo della pipeline, non un risultato: il modello del sanity ha visto un solo dataset ad anello. P2 mostra gia' un
transfer sensato a encoder congelato; P1, il ramo sparso, e' basso (DB3 al caso), coerente col fatto che il sanity non ha visto montaggi sparsi:
la metrica ha margine per distinguere i modelli del pilot. *Non confrontabile* con NeuroRVQ (fine-tuning completo contro sonda lineare). Nei log
delle sonde ci sono avvisi di convergenza di lbfgs (max_iter 3000) per alcuni C: da dichiarare; la C scelta per P1 e' forte (0,001-0,01).
**Costi:** P1 + P2 0,92 GPU-ora (dichiarati al piu' 1,5). Budget del passo 5: **~31,2 su 300** (collaudi 0,58, UCI 0,10, EPN-612 29,6, P1 e
P2 0,92).

## Passo 5 chiuso (Simone, 05/10/2026: «ok, firmo»)

Firmata la chiusura del passo 5: replica di NeuroRVQ **riuscita su EPN-612** (95,39%, tolleranza [92,65; 96,65]) e **non riuscita su UCI-EMG**
(78,66% dopo la correzione delle unita', dichiarata: 662 finestre, scelta delle finestre mia dove il paper tace); harness P1 e P2 provato sui dati
veri col modello del sanity. Consuntivo: **~31,2 GPU-ora su 300**.
**Subito dopo** (stesso messaggio, «Poi lancia il job di 1 e verifica capgmyo e csl»): (1) `window_seconds.sbatch` sul manifest v1 (lunghezza media
delle finestre per le quote nel tempo; CPU, **0 GPU-ora**); (2) verifica col fact-checker della derivazione dei canali di CapgMyo e CSL-hdemg
(monopolari o gia' differenziali; scale bipolari del 05/10), e di Hyser per confronto.

## Derivazione dei canali di CSL-hdemg, CapgMyo e Hyser (05/10/2026; fatti 25-27 raccolti, da firmare)

**window_seconds 59425316** sottomesso (CPU, 0 GPU-ora), in corso. **Verifica** (fact-checker + README di CSL letto dallo zip su Leonardo):
- **CSL-hdemg (fatto 25): gia' bipolare**, catena lungo l'avambraccio (canale i = elettrodo i meno elettrodo i+1, da prossimale a distale; nel
  nostro montaggio lungo le righe). La «bipolare» virtuale di D6a qui e' un **doppio differenziale**: lungo le righe come gli elettrodi Delsys
  «double differential» di NinaPro DB2/DB3, lungo le colonne una differenza fra strisce vicine.
- **CapgMyo (fatto 26): «differential electrode array»**, coppie e riferimento non dichiarati: stessa conseguenza. In piu', **il codice degli
  autori usa 16 righe x 8 colonne** mentre il nostro ingest assume 8 x 16 con `divmod(idx, 16)` (fatto 14 ancora aperto). *Lettura di AG:* se
  il codice degli autori riflette la disposizione fisica (8 moduli da 8 x 2 attorno all'avambraccio), da noi le due colonne di ogni modulo stanno
  una dopo l'altra sulla stessa riga: i vicini lungo l'avambraccio sono giusti tranne al salto 7 -> 8, quelli attorno all'avambraccio no. Peso di
  CapgMyo nel manifest: 0,08% dei campioni. Il rapporto delle scale misurato (indice +1: 0,87; indice +16: 1,47) e' coerente con questa lettura,
  ma non la prova.
- **Hyser (fatto 27):** derivazione **non dichiarata** dalle fonti lette (Quattrocento, OT Bioelettronica); il rapporto 0,25-0,41 e' coerente con
  monopolari a riferimento comune, ma non e' una fonte.
**Da decidere (Simone):** (1) se la bipolare virtuale resta su tutte le griglie (doppio differenziale su CSL e CapgMyo, da dichiarare) o solo su
Hyser; (2) cosa fare della griglia di CapgMyo (togliere CapgMyo dai montaggi virtuali e dichiarare la geometria incerta, oppure correggerla, che
cambia i sidecar e quindi lo sha256 nel manifest congelato).
**Decise** (Simone, 05/10/2026: «si a 1 e 2»): niente bipolari virtuali su CSL-hdemg; CapgMyo fuori dai montaggi virtuali finche' la geometria
non e' corretta. CapgMyo monopolare con riferimento comune (fatto 26, firmato da Simone). In codice (`VirtualSpec`, default) e nel foglio D6a;
test. La correzione della griglia di CapgMyo (16 x 8, cambia i sidecar e lo sha256 nel manifest) resta da fare, quando Simone lo decide.

## Correzione della posizione delle maschere LONG (07/10/2026, GPT; richiesta di Simone)

Correzione d'implementazione della D10 gia' firmata, sul checkout di base `28674c9`.
`training/masking.py` passava meta' della finestra a `draw_tube` come se fosse l'intera finestra: il limite di durata LONG diventava anche
un limite alla posizione. Tutti i tubi LONG non-slab terminavano nella prima meta'; con 40 patch (1 s) la durata minima di 20 patch
costringeva ogni LONG a cominciare da zero. Gli slab usavano gia' l'intera finestra per la posizione.

Ora la durata del singolo tubo resta nell'intervallo `20..min(80, P//2)` e l'inizio si estrae uniformemente fra `0` e `P - durata`,
estremi inclusi, sull'intera finestra valida. Restano le scale, le quote attese, i gruppi spaziali e il budget firmati; le sovrapposizioni
possono ancora formare unioni piu' lunghe del singolo tubo. Il cambiamento delle posizioni modifica le maschere realizzate anche a seme fisso.

Verifica locale CPU: otto regressioni riproducevano il difetto prima della correzione (anello/griglia, P=40/41/80/160).
Esito finale: **31 test passati** fra masking e dataloader, inclusi durata dei singoli tubi, seconda meta' campionabile, raggiungimento
deterministico dell'ultima patch valida, padding, QC, slab e determinismo. Il vecchio test di durata misurava distanze fra indici True,
non lunghezze di intervalli: sostituito. Nessun training eseguito; le misure/checkpoint precedenti restano riferiti al generatore precedente.

## Protezione dai gradienti non finiti nel trainer (07/10/2026, GPT; richiesta di Simone)

Correzione di sicurezza, senza cambiamenti a modello, loss, learning rate, soglia di clipping o decisioni sperimentali firmate.
Base pubblicata `4232246` (il codice del trainer e' quello di `32cf58b`; 4232246 aggiunge soltanto il registro di Claude).
Prima il controllo della loss non proteggeva da gradienti NaN/Inf: il clipping li lasciava passare all'optimizer e all'EMA, e lo stop
alla loss del passo successivo salvava uno stato contaminato. Riprodotto localmente iniettando un gradiente NaN con loss finita.

Ora la norma totale restituita da `clip_grad_norm_` deve essere finita prima di chiamare optimizer ed EMA. Il controllo e' esplicito
(`torch.isfinite(gnorm)`), non un'eccezione non gestita da `error_if_nonfinite=True`: permette di registrare la causa dello stop.
Se la norma non e' finita si scartano i gradienti, non si incrementa il passo, non si registrano metriche di un update non eseguito,
si scrivono `STOP` e `summary.json` e si conserva il checkpoint gia' salvato. La stessa conservazione vale per loss non finita.
Lo stop numerico non riscrive il checkpoint con lo stato RNG avanzato nel passo fallito. Se il primo passo fallisce non crea un checkpoint;
se ci sono passi validi dopo l'ultimo salvataggio, questi non sono recuperabili da quel checkpoint. STOP blocca comunque i job concatenati.
Salvataggi periodici, fine normale, limite di tempo e allarmi conservano il comportamento precedente.

Test di fault injection sul trainer reale con modello/dati sintetici minuscoli: NaN, +Inf e -Inf nel backward, con loss finita, al primo passo,
alla ripresa e dopo un passo valido non ancora salvato. Verificati student/teacher/momenti invariati sul passo difettoso, chiamate a optimizer
ed EMA assenti, contatore/metriche coerenti, checkpoint identico byte per byte e STOP rispettato. Anche lo stop per loss non finita preserva
il checkpoint. Due regressioni selezionate fallivano prima della patch; esito finale: **64 test passati, 1 saltato (MPS non disponibile)**,
inclusi i dieci nuovi casi e le regressioni di trainer, JEPA, Perceiver, dataloader e masking. CPU macOS, torch 2.14.0; nessun job HPC.

Questa verifica non prova contaminazioni nei run ordinari passati e non certifica backend CUDA non eseguiti, checkpoint storici o ogni
possibile overflow durante l'update dell'optimizer. Il Perceiver non e' modificato: il P0 originario non confermato resta distinto dal P2
di pulizia numerica e da questa protezione del trainer.

## Griglia di CapgMyo corretta a 16 x 8 (Simone, 07/10/2026: opzione A)

Esiti gia' registrati da GPT nel registro condiviso: **window_seconds 59425316 COMPLETED** (18 min, 0 GPU-ora, 19.034 sessioni, nessun errore;
`results/passo6/window_seconds_59425316.json`); diagnostica della griglia di CapgMyo (`docs/capgmyo_grid_diagnostic.md`): 16 x 8 con riga =
i // 8 preferita in 72/72 prove. **Opzione A:** `ingest/capgmyo.py` ora da' 16 x 8 (`grid_position`); `scripts/fix_capgmyo_grid.py` riscrive SOLO
`grid_row`/`grid_col` dei 14 sidecar di CapgMyo (backup prima), scrive il **manifest v1.1** (cambia solo lo sha256 dei sidecar di CapgMyo:
pesi, split e sessioni uguali; `params.revision`) e le scale senza le bipolari di CapgMyo, da ricalcolare con `measure_loader.sbatch`. Dopo il
job il manifest v1 viene rifiutato per CapgMyo (sha256 diverso): **dalla finestra 1 si usa il v1.1**. Test: riscrittura, manifest accettato dal
dataloader, vecchio rifiutato, rilancio senza effetti; test dei vicini e dei montaggi aggiornati alla 16 x 8. CapgMyo resta escluso dai montaggi
virtuali finche' il job non ha girato; poi si toglie l'esclusione. Asse: righe attorno all'avambraccio, colonne lungo (lettura di AG da Du
2017); verso prossimale/distale non dichiarato e non necessario al modello.
Job: `fix_capgmyo_grid.sbatch` (CPU, 0 GPU-ora), poi `measure_loader.sbatch <manifest v1.1> <scale senza bipolari CapgMyo> --bipolar-scales
--skip-rate` in dipendenza (CPU, 0 GPU-ora).

## Collaudo della configurazione della finestra 1 (Simone, 07/10/2026: «si va bene. intanto facciamo questa»)

**Cosa:** la prima corsa con tutte le regole della finestra 1 insieme (`--window1-rules`: quote nel tempo, perdita per finestra, zona di bordo
da 100 ms, ancora multi-scala, montaggi virtuali D6a corretti), maschere LONG corrette (91a6d28), protezione dai gradienti non finiti (ec3914d),
**manifest v1.1**, scale di `measure_loader` del job 59623700 (con le bipolari di CapgMyo nuove), `window_seconds_59425316.json`, **tutti i
dataset**. Modello: preset `sanity` (31 M), solo per collaudare la pipeline. `sanity_jepa.sbatch` su `boost_qos_dbg`, 1 GPU, 30 min: **al piu'
0,5 GPU-ora = 4 ore locali**; budget del passo 6 (lavori della finestra 1 provati su `boost_qos_dbg`, piano W6): usate ~11,0 su 100. Cartella
`$SCRATCH/wearusfm_runs/runs/w1_collaudo_0710`. In dipendenza `afterok` dal job delle scale.
**Criteri, fissati prima di guardare:** (1) nessun errore, nessuno stop numerico, nessun OOM; (2) **attesa sui dati <= 2% del passo** (regola di
D6a), misurata dopo i primi 50 passi come somma di `t_data_s` / somma di `t_step_s`; (3) nelle metriche compaiono tutte e tre le classi di quota
e finestre virtuali (`virtual` > 0). Le frazioni realizzate per classe si riportano, non si giudicano: le quote sono nel tempo, non nelle finestre.
Se (2) fallisce si riporta e si guardano le fasi del dataloader, senza cambiare nulla al volo.
**Esiti e lancio (07/10/2026):** `fix_capgmyo_grid` **59623698 COMPLETED** (10 s): **18 sessioni** di CapgMyo (tutte le righe del manifest: 14 di
pretraining + 4 di test), 18 sidecar corretti, 18 righe del manifest v1.1, 56 scale bipolari tolte; backup in
`results/passo6/capgmyo_grid_fix/backup/`. Scale: `measure_loader` **59623700 COMPLETED** (16 s): 19.879 scale, i soliti 59 errori di emg2pose;
file `results/passo6/loader_59623700/session_scales.json`, **da usare dalla finestra 1**. Collaudo della finestra 1: **59624004** (la prima
sottomissione con `afterok` su un job gia' finito e' stata rifiutata da SLURM; rilanciata senza dipendenza). CapgMyo resta escluso dai montaggi
virtuali per questo collaudo: l'esclusione si toglie dopo, per non cambiare il codice mentre il job e' in coda.

**Esito del collaudo 59624004 (07/10/2026, letto dopo la fine del job):** COMPLETED, exit 0:0, 26 min 17 s, fermato dal limite di tempo al
passo **625** su 2000; preset `sanity` da **28,3 M** parametri (non 31 M come scritto sopra). **Costo: ~0,44 GPU-ora**; passo 6: **~11,4 su 100**.
Riassunto, criteri e configurazione in `results/passo6/w1_collaudo_59624004.json`; metriche complete sul Mac
(`~/wearusfm_local/reports/w1_collaudo_0710/`), checkpoint su Leonardo nella cartella del run.
- **(1) superato:** nessun errore, 0 passi con valori non finiti, nessuno stop numerico; memoria GPU massima 37,6 GiB su 64 (27-28 GiB fino al
  passo ~400, poi un salto: probabilmente una finestra lunga di griglia intera, non verificato).
- **(2) NON superato: attesa sui dati 3,59% del passo** dopo i primi 50 (2,455 s per passo). L'attesa e' periodica: un passo ogni 3 esatti, 0,24 s
  in media, 0,55 s al massimo; gli altri passi zero. 6 processi del dataloader su 8 CPU per GPU: la produzione e' di poco (~4%) sotto il consumo
  della GPU. Primo passo 12,2 s (7,8 s di dati). Come fissato: si riporta e si misurano le fasi del dataloader con le regole della finestra 1,
  nessuna modifica al volo. Proposta (da confermare): misura per fasi con `scripts/measure_loader.py` su `lrd_all_serial`, 0 GPU-ora, dopo aver
  verificato che lo script attivi le stesse regole della finestra 1.
- **(3) superato:** 20.000 finestre: A 19,9%, B 72,6%, C 7,5% (riportate, non giudicate: le quote sono nel tempo); **711 virtuali** su 1.508 di
  classe C (0,47; CapgMyo escluso dai montaggi in questo collaudo). Nessuna sessione saltata.
- **Da guardare (fuori dai criteri):** le perdite dell'ancora multi-scala dominano il totale. Medie dei passi 615-625: `ms_fast` 4,44, `ms_slow`
  3,56, contro `log_rms` 0,83, `log_env` 0,66, `band_shape` 0,66, JEPA 0,04; totale ~ JEPA + 0,2 x somma delle ancore (2,07). All'inizio
  `ms_slow` era ~93 (passi 1-10) e ~28 al passo 216, poi e' sceso. Lettura di AG, non misurata: i target multi-scala sono log-potenze non
  standardizzate per banda; perdita grande non vuol dire per forza gradiente grande. Da proporre come foglio separato prima della finestra 1.
- La perdita JEPA scende da 0,87 (passo 50) a 0,03-0,04 (passi 600-625); su 625 passi con warmup di 1000 non dice nulla sul collasso.

## PSD fisica e Step 1 sperimentale (07/10/2026, GPT; richiesta di Simone)

Simone: «inizia con la correzione della PSD, poi vai al passo 1 e riproviamo»; approvati tramite domanda esplicita entrambi i collaudi sotto.
Base `3685255`, checkout canonico Mac condiviso. README_STEP1 allegato usato come proposta, non come prova di esecuzione: il pacchetto di
codice citato nel README non era allegato. Implementazione integrata e testata qui; le ulteriori scelte restano sperimentali.

**PSD multiscala:** Hann simmetrica, detrend=False, nfft=N. Densita' = |rFFT(x*w)|^2/(fs*sum(w^2)), raddoppio dei bin positivi esclusi DC
e Nyquist quando N pari; potenza di banda = somma densita' * fs/N. Log(potenza+1e-8). Riscontro indipendente con periodogram SciPy,
N pari/dispari e float32/64, Parseval comprensivo di DC/Nyquist, guadagno, tono 160 Hz a 1/2 kHz, bande mancanti e bordi.
Non cambiano 50 ms/6 bande/hop25 ms e 500 ms/24 bande/hop100 ms, bande20-450, RMS/env/forma5bande, guard o preprocessing.
Versione target `step1_psd_power_shape_energy_v1`; resume di vecchi run multiscala vietato prima di scrivere output.

**Step 1:** pooling comune P1/P2 [backbone medio su tempo valido/latenti; locale medio nel tempo per canale, poi sui canali QC-validi],
2d coordinate, supporto per conteggi eterogenei e padding; vista mascherata rifiutata. Readout `p1p2_qc_equal_channel_v1`: rigenerare cache.
Target globali: media potenze LINEARI nel tempo per canale, poi canali equiponderati, banda comune; forma log(P_b/sum P_b) ed energia
log(sum P_b+eps), 32 coordinate (6+1+24+1). Silenzio: forma esclusa, energia al floor. Se un canale QC-valido non ha finestre, scala esclusa.
Testa lineare sullo student non mascherato con gradiente verso encoder; SmoothL1 float32 su statistiche del solo train; media prima fra
coordinate di ciascuna famiglia, poi famiglie disponibili e campioni. Parametri/buffer registrati prima di teacher/optimizer, un solo update,
forward extra con RNG CPU/CUDA isolato. MPS non supportato. Default keep_weight=0, keep_readout=False; niente warm-start non verificato.
Per futuri bracci A/B usare la testa in entrambi con calibrazione identica (A=0, B>0), stesso seed: inizializzazione testata identica.

**Sonde preparate, non ancora eseguite su corpus reale:** scripts/spectral_step1.py split/calibrate/extract/probe. ID soggetti qualificati,
scaler sul train, alpha sulla validation, test una volta per protocollo; R2 negativi mantenuti, target costanti/mancanti non valutabili.
Si misurano backbone/locale/concatenazione; nessun bootstrap per soggetto implementato. Split della ridge su soggetti del PRETRAINING:
l'encoder puo' averli osservati senza etichette, quindi non chiamare il risultato generalizzazione dell'encoder a soggetti mai visti.
P1/P2 sul test downstream e un confronto A/B scientifico sono successivi, non inclusi nel collaudo autorizzato. Keep 0,05 e' solo un'ipotesi.
Le 32 coordinate non provano conservazione della dinamica; il guard centrato non certifica l'intero supporto lento rispetto ai transitori.

### Collaudi autorizzati: protocollo congelato PRIMA del lancio

Verifica locale completa: **720 test CPU passati, 1 saltato (MPS)**. Cinque casi downloader bloccati dalla bind localhost nel sandbox
sono stati rieseguiti con permesso e superati; non erano regressioni di codice. Le prove CUDA restano quelle da fare sotto.

Costo massimo approvato da Simone: **1 GPU-ora =8 ore locali**, piu' **1 core-ora CPU (0 GPU-ora)**. Budget passo6: ~11,4/100 usate,
~88,6 residue prima; al massimo ~12,4/100 e ~87,6 residue dopo. Nessun cambio al dataloader/quote/CapgMyo virtuale.
Usare manifest v1.1 e scale59623700/durate59425316; nessun resume dei run storici. Codice e criteri committati prima di sottomettere.

1. **Replica PSD finestra1:** sanity 28,3M, tutti i dataset, RVQ spento, keep spenta, seed0, massimo2000step e1560s, 1GPU/8CPU/30min
   boost_qos_dbg, directory NUOVA `$SCRATCH/wearusfm_runs/runs/w1_psd_0710`. <=0,5GPU-ora. Stessi criteri precedenti: nessun errore/OOM/
   stop numerico; attesa sui dati <=2% dopo50step; tutte le classi A/B/C e virtual>0. Quote riportate, non giudicate. Non confrontare
   direttamente magnitudine loss con vecchia definizione target. Se la soglia dati fallisce si registra, senza ottimizzare al volo.
2. **Calibrazione CPU:** lrd_all_serial 1CPU/1h <=1core-ora, emg2qwerty, split soggetti seed0 (circa80/10/10 entro pretraining), 100
   finestre per soggetto train; cartella NUOVA `$WORK/wearusfm_runs/results/passo6/step1_0710`. Scales/durate e impostazioni W1 identiche
   al braccio smoke. NPZ con y/valid/subject/split, provenienza, versioni/hash di manifest/split/subset/preprocessing/commit. Non scegliere
   soggetti o finestre sul test P1/P2. Le statistiche non si ricalcolano al resume.
3. **Step1 smoke:** sanity con keep_readout=True, emg2qwerty, RVQ spento, seed0, keep_weight=0,05, gradnorm condivise ogni100step;
   massimo200step/1200s poi resume massimo201step/90s nella STESSA allocazione1GPU/8CPU/30min, <=0,5GPU-ora; cartella NUOVA
   `$SCRATCH/wearusfm_runs/runs/step1_smoke_0710`. Prima controlli small su CUDA/bf16/checkpointing: loss/grads finite, encoder riceve
   gradienti, teacher no-grad, RNG CPU/CUDA invariato anche dopo backward, buffer EMA identici. Criteri run: nessun errore/OOM/stop numerico,
   keep e target validi presenti, gradienti condivisi finiti, checkpoint/statistiche/versioni coerenti al resume. Nessuna soglia post hoc
   su loss/P1/P2; throughput/GPU memoria/costo si riportano. L'attesa inferiore nel braccio keep puo' dipendere dalla GPU piu' lenta, non
   dimostra un loader migliorato. Queste prove sono funzionali, non qualita' dell'apprendimento o non-collasso a regime.

### Esiti conclusivi dei collaudi PSD / Step 1 (registrati 08/10/2026, GPT)

Codice eseguito per tutti i job: `0e8ba050553c04530a46e8964eca80438916dacd`. Esiti verificati sul cluster da leonardo-ops e comunicati
in questa chat; nessuna modifica al codice durante i run. Riepilogo machine-readable: `results/passo6/step1_validation_20261008.json`.
Le voci pre-lancio sopra e nel registro restano storiche: questa voce chiude le verifiche CUDA allora ancora da eseguire, NON quelle di qualita'.

- **PSD 59638875: COMPLETED, exit0:0**, 26min16s, 625passi consecutivi su2000, fine normale al limite interno1560s, nessunSTOP/errori/OOM;
  loss e grad_norm tutte finite. Picco37,607GiB, tempo medio2,457s/passo. 20.000finestre: A3978/B14514/C1508, 711virtuali, zero sessioni
  saltate. Criteri1 e3 superati. **Criterio2 NON superato:** dopo50step, sum(t_data)=47,952s, sum(t_step)=1412,955s => **3,394% >2%**.
  Non ottimizzato il loader; non confrontare direttamente le magnitudini loss con i vecchi target non normalizzati.
- **Calib59638890: COMPLETED, exit0:0**, 16min17s. Split seed0:80train/10val/10test;100finestre per ciascun soggetto train =>8000x32,
  float32, tutte coordinate valide/finite, nessun val/test. Versioni/commit/manifest/split/scale/durate/loader coerenti.
  SHA256NPZ:`4640dfcf2c6819fb8ba2870e1506f08e4b024cdf38dda267c4cd37652cbc1f89`;
  SHA256split:`e064a1102bb4a7dd1b01a70caefa8fa2c58d8c9c751bafeafe6bd2892fff7a30`.
  Richiesta1CPU ma accounting2CPU/billing; costo effettivo0,5428ore localiCPU, entro il tetto1ora autorizzato.
- **Smoke59641613: COMPLETED, exit0:0**, 14min19s. CUDA/bf16/activation checkpointing su torch2.2 verificati; RNGCPU/CUDA preservato
  anche dopo backward, encoder riceve gradienti e teacher no-grad. 200passi poi resume da200 a201 nella stessa allocazione; nessunSTOP.
  Keep attiva, gradienti encoder non nulli, 32target attivi; tutte metriche e tensori student/teacher/optimizer finiti, statistiche coerenti
  colNPZ e buffer student/teacher identici. Picco completo29,472GiB dai log; summary finale copre SOLTANTO resume, non l'intero job.

**Costo effettivo:** PSD~0,438GPUh + smoke~0,239GPUh = **0,6764GPUh** (5,411ore locali), piu'0,5428ore localiCPU, entro1GPUh+1hCPU.
Budget passo6: circa**12,1/100GPUh usate,87,9residue** (consumo precedente11,4 arrotondato). Nessun nuovo training/seme/job autorizzato da questa registrazione.

Percorsi completi:
- RunPSD:`/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/runs/w1_psd_0710/`.
- RunSmoke:`/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/runs/step1_smoke_0710/`.
- Split/calib:`/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/passo6/step1_0710/{splits.json,calibration.npz}`.
- Log:`/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/{sanity_jepa_59638875.out,step1_calibrate_59638890.out,step1_smoke_59641613.out}`.

**Passaggio di consegne:** usare sempre `/Users/simonebenatti/dev/wearusfm`, checkout unico condiviso conClaude; la vecchia chat resta
intatta per studio. Log/metriche gia' archiviati fuori Git. Prossimo lavoro: esaminare gradienti/loss e proporre prima sonda
frozen su checkpoint scelto e poi protocollo A/B accoppiato, con nuova stima/autorizzazione e soglie predefinite. P1/P2 e miglioramento
embedding NON ancora verificati. Dataloader3,394% resta aperto; profiling separato e aderenteW1, non assumere vecchio measure_loader
equivalente. Guard completo500ms, dinamica e CapgMyo virtuale restano aperti. Rigenerare featurecache; non riprendere vecchi target/run.

Log/config/summary/metrics archiviati localmente fuori Git in `/Users/simonebenatti/Documents/ChatGPT/FM per EMG/handoff_20261008/`;
nessun checkpoint/NPZ scaricato. Ricontrollati625recordPSD e201smoke tutti finiti, attesa e conteggi coerenti. Norme PESATE sui parametri
encoder comuni ai passi1/101/201: JEPA0,158253/2,686458/1,361323; ancore32,491978/4,550262/1,805742; keep0,646594/0,078914/0,038723.
Sono descrittive, non misure di allineamento fra gradienti o generalizzazione; keep finale0,757865 non valida lambda0,05.
Passaggio di consegne completo: `docs/handoff_step1_20261008.txt`. Nuova chat sullo stesso checkout, vecchia conservata per studio.
Archiviazione conclusa: 9 copie verificate SHA256 contro Leonardo; indice in `handoff_20261008/verification.json`, header/provenienza
in `handoff_20261008/calibration/header_provenance.json` sotto la directory locale indicata sopra. Repo remoto pulito, nessun job in coda.

## Prima sonda spettrale frozen (08/10/2026, GPT; protocollo congelato e job autorizzati)

Su richiesta di Simone di proseguire con lo Step 1, preparato il protocollo della prima misura ancora mancante in
`docs/step1_frozen_spectral_probe.md`. Baseline proposta: teacher EMA finale di `sanity_0410`, 10.000 passi, SHA256 verificato su Leonardo
`fa7b5a7ae8cd6c2c13ef1fe42fa79e07e370d2831fd2782f2a0a1d768fb7cc96`. Si usa emg2qwerty,100finestre per ciascuno dei100soggetti,
split gia' congelato80/10/10 seed0 (SHA256 `e064a1102bb4a7dd1b01a70caefa8fa2c58d8c9c751bafeafe6bd2892fff7a30`), teacher frozen,
readout QC-aware backbone/locale/concatenazione e ridge per le32coordinate; scaler solo train, alpha {0,1;1;10;100} solo validation,
test letto una volta, R2 negativi mantenuti. Questa sonda crea una baseline descrittiva: nessuna soglia di qualita' o confronto A/B.

Preparati `scripts/slurm/spectral_step1_extract.sbatch` (1A100,8CPU,30min, tetto0,5GPUh=4ore locali) e
`scripts/slurm/spectral_step1_probe.sbatch` (8CPU,1h,0GPUh, tetto8core-ora), con output nuovi fuori Git e guardie sugli hash.
Criteri di validita' congelabili: provenienza completa, soggetti disgiunti,10.000righe finite,32coordinate valutabili salvo indisponibilita'
fisica registrata, tre viste/quattro famiglie/dettaglio coordinate completi e nessuna scelta sul test. Budget passo6 dopo il massimo GPU
sarebbe circa12,6/100usate e87,4residue. **Nessun job sottomesso:** tetti e protocollo richiedono approvazione esplicita di Simone.

**Autorizzazione (Simone, 08/10/2026):** «ok allora parti con quello che hai proposto, poi sistemiamo il perceiver». Autorizzati quindi
l'estrazione frozen e il job ridge in dipendenza entro i tetti sopra: massimo0,5GPUh (=4ore locali) piu' massimo8core-oraCPU. Il Perceiver
resta invariato per questa misura; la bonifica P2 `safe_scores` viene affrontata dopo, prima del futuro training A/B.

### Stato Perceiver riesaminato durante la preparazione

Il codice resta quello originale in `model/perceiver.py`: score delle chiavi nascoste a `-inf`, softmax calcolata su tutte le righe e
selezione finale con `torch.where`; nessuna patch `safe_scores`. Diagnostica locale mirata con torch2.14 su un istante interamente nascosto:
la softmax intermedia isolata contiene NaN e il suo gradiente isolato e' NaN, ma nel grafo completo del `PerceiverPooling` output/loss,
gradienti dell'ingresso e di tutti i parametri risultano finiti; sui token nascosti il gradiente e' zero. Questo conferma la classificazione:
**P0 non confermato**, eventuale bonifica come **P2**. I collaudi CUDA gia' conclusi hanno inoltre avuto gradienti finiti, ma non sostituiscono
una regressione CUDA dedicata al caso tutto mascherato. La sonda frozen proposta usa `visible=None`, quindi non attraversa questo ramo e non
dipende dalla scelta di applicare o meno `safe_scores`.

### Esito della prima sonda spettrale frozen (08/10/2026)

Job estrazione59703237 e ridge59703306 entrambi **COMPLETED0:0**; afterok rispettato. Estrazione29min43s,0,4953GPUh; ridge18sCPU;
circa4,00core-ora totali, entro0,5GPUh+8core-ora autorizzate. 10.000righe (X10000x768 float32),32/32coordinate finite e valutabili,
split8000/1000/1000 e soggetti80/10/10 senza sovrapposizioni. Tutta la provenance e gli hash sono coerenti.

R2 macro backbone: forma fast0,8526, energia fast0,9160, forma slow0,7895, energia slow0,9158. Locale:0,7848/0,8905/0,7385/0,9011.
Concatenato P1/P2: **0,8634/0,9142/0,7975/0,9154**; e' il migliore sulle due forme. R2 negativi per soggetto mantenuti; nessun bootstrap.
Criteri tecnici prespecificati tutti superati. Questa e' la baseline del checkpoint storico, non effetto della keep ne' generalizzazione a
soggetti mai visti dal pretraining. Riepilogo machine-readable: `results/passo6/step1_frozen_probe_20261008.json`.

Il report remoto (SHA256 `c946cad5dada1a8976066faa322aad6d4ea413268148801159567ec520826f8b`) e le feature (SHA256
`d2896bf5cc26302306cfa14b3a41f6c8711b51ccd48aedb7b71e9747d7e4ea9a`) sono stati verificati su Leonardo. Due tentativi indipendenti di
copia del solo report/log sul Mac si sono bloccati senza output; nessun file parziale e nessun checkpoint/NPZ scaricato. Budget passo6 ora
circa12,57/100GPUh usate e87,43residue.

### Bonifica P2 del Perceiver preparata localmente dopo la sonda

Applicato `safe_scores` soltanto alle righe senza alcuna chiave visibile: prima della softmax gli score di quelle righe diventano finiti,
poi l'attenzione viene azzerata. Il forward resta invariato; il gradiente verso input/query/chiavi della riga vuota e' zero senza nodi NaN.
Aggiunte regressioni float32/float64 per un istante interamente nascosto e per tutti gli istanti nascosti. Suite CPU:723test unici passati,
1skipMPS; cinque casi localhost inizialmente bloccati dal sandbox, poi i6casi del file superati (uno duplicato). Nessun test CUDA della patch:
il collaudo frozen e' stato eseguito sul commit precedente7655356 e senza masking, quindi resta separato e valido.

**Collaudo CUDA autorizzato (Simone, 08/10/2026):** dopo la chiusura frozen, Simone ha chiesto di provare il Perceiver e poi lavorare
sull'A/B. Protocollo congelato prima del lancio: `scripts/check_perceiver_cuda.py`, Perceiver di taglia produzione d384/H6/K64, due campioni
impacchettati, un istante interamente nascosto nel primo e tutti gli istanti nascosti nel secondo; float32 e autocastbf16. Confronto stessa
inizializzazione/input con forward/backward diretto e ricomputato tramite activation checkpointing. Criteri: output/loss/gradienti finiti,
gradiente input esattamente zero sulle regioni interamente nascoste e nonzero su quelle visibili, equivalenza forward/backward entro tolleranze
prespecificate (2e-5 float32,2e-2 bf16). Job A100,4CPU,20G, massimo5min =0,0834GPUh, entro il tetto autorizzato0,1GPUh. Solo test funzionale,
nessun training/checkpoint/dato reale; esito da registrare senza rilanci o correzioni al volo.

**Esito CUDA:** job59720752 **COMPLETED0:0**,13s,A10064GB,4CPU,20G, MaxRSS~1,03GiB; costo0,00361GPUh entro0,1 autorizzate.
Torch2.2, float32 e autocastbf16: output/loss/tutti i gradienti finiti, gradiente degli input completamente nascosti esattamente zero,
forward e backward con activation checkpointing identici al percorso diretto; differenze massime output/input-grad/parameter-grad tutte0,0.
Tutti i criteri prespecificati PASS, nessun retry. Log SHA256 `52d057adac4cf99ea752552944ee6dcefacd85935f2face38981b35628b251a4`;
riepilogo `results/passo6/perceiver_cuda_20261008.json`. La bonifica P2 e' quindi verificata CPU+CUDA; non e' una misura di qualita' del
training. Budget passo6 circa12,575/100GPUh usate e87,425residue.

## Proposta A/B Step 1 accoppiata (08/10/2026, GPT; NON ancora autorizzata)

Simone: «procedi» dopo la proposta di congelare il protocollo A/B; preparazione completata su base8bc8c64. Riletto anche il confronto
originario nella chat precedente: A=JEPA/ancorePSD normalizzate, B=stessa base+supervisione diretta del readoutP1P2. Nessun nuovo training
o job avviato. Protocollo completo `docs/step1_ab_pilot.md`, ID `step1_ab_4000_v1`; script `scripts/step1_ab.py`, launcher train/compare.

**Scelte da approvare, fissate prima dei nuovi risultati:**4.000update per braccio, semi0/1 accoppiati, Akeep0/Bkeep0,05, testa in entrambi,
emg2qwerty, stessa calibrazioneSHA4640dfc e splitSHAe064a11, batch32/worker6/RVQoff, warmup1000 poi lr3e-4 costante (pilota, NON WSD/D14
o4epocheD12). Unico checkpointteacherEMA finale; nessun warm-start/resume, ricerca di lambda o prosecuzione automatica. Audit opzionale
nel trainer, spento per default: hash iniziali e ogni batch completo, senza consumare RNG; l'audit rifiuta run gia' esistenti e il wrapper
fallisce se non completa4.000update, seSTOP o stati finali non finiti. Nessun cambiamento a perdita/optimizer/preprocessing ordinari.

**Endpoint e criteri:** vista primariaP1P2, media equiponderataR2 formafast/slow, differenzaB-A mediata sui2semi. Ridge train-only, alpha
validation-only, test frozen10soggetti identico in4cache nuove, tutte32coordinate valutabili; bootstrap2000subject-draw seed20261008
accoppiato fra bracci E semi, ricostruzioneR2pooled da n/media/M2/SSE. «Promettente per P1/P2» soltanto se delta>=0,02, CI95%lower>0,
entrambi i semi positivi, nessuna forma peggiora nella media e nessuna famiglia/seme perde>0,01R2. Energie separate, altre viste descrittive,
negativi mantenuti. CI condizionato su due modelli; split della sonda dentro pretraining, NON soggetti mai visti dall'encoder. Soglie pratiche
proposte ora, dopo la baseline storica ma PRIMA dei nuoviA/B; non chiamare questo una conferma indipendente o una vittoria downstream.

**Costo da autorizzare:** atteso16-18GPUh=128-144ore locali GPU (A stimato dal PSD multi-dataset2,457s/step, B smoke51-200=3,95684s/step,
estrazioni da29min43s; non ETA). Massimo23GPUh=184ore locali GPU piu'40core-oraCPU:2A*4h+2B*6h+4extract*45min,4ridge8CPU*1h+
1compare8CPU*1h. Budgetpasso6 ora~12,5753/100usate/87,4247residue; dopo cap~35,5753usate/**64,4247residue**. QoSboost_qos_bprodMax24h
e partizione boost_usr_prod verificate read-only da leonardo-ops; dbg30min non adatta. CPUlrd_all_serial30G. Nessunretry o seme aggiuntivo.
**Non autorizzato finche' Simone non approva questi tetti.** Commit esatto di esecuzione e autorizzazione si registrano prima dei lanci.

P1/P2 e' una tranche successiva solo se promettente, NON inclusa nei23GPUh. Guardia pratica proposta: delta medioB-A non inferiore a-2pp
separatamente perP1 eP2, senza chiamarla equivalenza statistica; nuovi cache/provenance/paired-report da preparare prima dell'approvazione
successiva. Cap stimato ulteriore6GPUh+128core-oraCPU, non autorizzato ora. Dataloader/guard500ms/CapgMyo virtuale/dinamica restano separati.

Verifica locale finale:738test CPU unici passati,1skipMPS. Cinque casi downloader bloccati dal sandboxlocalhost poi superati nel file
rieseguito con permesso (6casi,1duplicato); dopo ultimo refactoraudit/compare19test mirati passati. SintassiBash, CLI e diff verificati.
Audit reale small su3update dimostra init/batch/maschere identici e RNG/stati invariati rispetto al trainer senza audit; comparatore provato
su cache sintetiche, drift/mismatch rifiutati. Nessun training/auditCUDA nuovo, risultatoA/B o jobHPC: quello richiede l'approvazione sopra.

### Autorizzazione A/B acquisita prima dei lanci (08/10/2026)

Simone: «sono partiti? autorizzo», in risposta ai tetti23GPUh (184ore locali GPU) piu'40core-oraCPU. Approvato il protocollo
`step1_ab_4000_v1` gia' congelato nel commitb0ec17d5d9b5f804ffa8986aafc9881861e519f2:4trainingA/B*semi0/1,4.000update ciascuno,
poi4extract SOLO dopo audit valido di tutte le coppie,4ridge e1compare entro i cap indicati. Nessunretry/resume/seme aggiuntivo,
ottimizzazione o P1/P2 downstream. Budgetpasso6 residuo attuale87,4247GPUh, residuo minimo dopo il cap64,4247GPUh.
Si esegue il commit di sola documentazione che registra questa approvazione (codice e criteri invariati). Nessunpush neppure di docs
durante la campagna, per non cambiare expectedcommit/source_commit; jobID/status da registrare temporaneamente fuori Git.
Tutti i lanci via leonardo-ops, senza salloc. Al momento di questa registrazione nessun jobA/B ancora sottomesso.

**Correzione preflight QoS, prima di qualsiasi training:** la prima sbatchA_seed0 conboost_qos_bprod e' stata respinta esplicitamente
`QOSMinCpuNotSatisfied`, senza jobID; coda vuota, nessuna GPUh consumata, altri3run non tentati. Il controllo precedente aveva omesso
MinTRES: bprod impone65nodi/2080CPU/260GPU e non ammette1GPU. Si usa boost_qos_lprod (gia' associata all'account, senza MinTRES),
IDENTICHE risorse1GPU/8CPU/100G e capA4h/B6h; anche extract45min inlprod. Questo e' un adeguamento operativo entro l'autorizzazione
acquisita, non un retry di training, cambio scientifico o aumento di costo. DefaultQoS del launcher corretto prima dei job; nuovo commit
expected per la campagna, senza modificare trainer/target/criteri/semi. La precedente dichiarazione di compatibilita'bprod e' rettificata.
Preflight lprod completo da leonardo-ops: associazioneaccount confermata, MinTRES vuoto; `sbatch --test-only` accetta siaA4h siaB6h.
Gli ID del test non sono job reali; squeue resta vuota. SintassiBash del launcher corretto e diff verificati prima del commit/lancio.

## Esito A/B Step1 e autorizzazione downstream P1/P2 (09/10/2026)

Training59726054/55/56/58 e frozen59766817-25 tutti COMPLETED0:0 al commit69fa2da. Quattro endpointteacher4000,
stati iniziali e8000batch accoppiati identici, metriche finite, nessun allarme. FormaP1P2 deltaB-A+0,0525109691R2,
seed0+0,0608487903/seed1+0,0441731480, CI95%[0,0446510973;0,1240582165]; tutti5gate superati.
Forma fastA0,8503571/B0,9071707; slowA0,7817578/B0,8299661. Energie migliorano entrambe.
Esito promettente per la verifica downstream; nessuna conclusione di miglioramento dei task.
Costo reale16,7847GPUh (training15,2558+extract1,5289),0,1311core-oraCPU. Passo6 circa29,36/100GPUh usate,
70,64residue. Artefatti esterni in step1_ab_0810; SHAcomparison e checkpoint fissati nel nuovo wrapper.

Simone autorizza «procedi», «lancia quando sei pronto», «riparti». Protocollo `docs/step1_downstream_ab.md`,
IDstep1_downstream_4000_v1:8estrazioni nuove+8probe, stessi4teacher, stessi split/seedsonda, nessuntraining.
Cap6GPUh=48ore localiGPU+128core-oraCPU, residuo minimo64,64GPUh. Confronto finale dentro l'ottavo probe
(dipendente dagli altri7), senza costo autorizzato aggiuntivo. Nessunretry/seme/tuning automatico.
Guardia congelata PRIMA delle misure: delta medioB-A>=-2punti percentuali, separatamenteP1 eP2, non test
statistico di non-inferiorita'. Conteggi pooledperclasse/soggetto; bootstrap accoppiato1000seed20261009;
P1stratificato per dataset, P2SEsplitRMS senza riduzione per3split sovrapposti, CI descrittivo normale.
Incertezza condizionata ai2modelli; dettagli e limiti nel protocollo. Hashinput effettivi/cache/report,
provenance e soggetti identici richiesti. Cache storiche non riusate. Commit e push identico prima dei lanci;
checkout congelato fino alla chiusura; jobID/status fuoriGit. Nuovo codice aggiunge audit/statistiche,
riusa encoder/preprocessing/sonde firmate. Preflight remoto read-only passato, nessunjob attivo.
Verifica locale pre-lancio:23testCPU unici passati; P1/P2 completi sintetici conaudit, readout/sonda, pairing/counts,
bootstrap/aggregazione, wrapper/cache/provenienza e confronto finale. Bash/CLI/diffcheck passati. Nessun nuovo testCUDA.

## Lavoro in parallelo Claude / GPT: worktree, integrazione, deploy di campagna, budget (Simone, 09/10/2026: «si a tutti e 4»)

Sulla proposta di GPT (divisione del lavoro) e due giri di review incrociata Claude / GPT. **Approvati:**
1. **Worktree locali su branch separati**; `main` resta nella cartella canonica come punto d'integrazione; **GPT integratore unico** tramite
   merge, senza squash o riscritture (restano gli SHA originali); review di Claude prima del deploy quando un'integrazione cambia la sua parte.
   Divisione: **Claude** = motore di training (multi-GPU, WSD, checkpoint e ripresa, preset 100M/300M, dataloader e misura per fasi, deploy e
   sottomissioni); **GPT** = obiettivi, valutazione, integrazione (A/B, baseline con encoder casuale, sonde fisiche e stratificate, confronto
   JEPA / ancore, zona di bordo dei target). Proprietari dei file e regole in `CLAUDE.md`.
2. **Eccezione di deploy su Leonardo:** checkout di campagna congelati, solo da commit gia' pubblicati (branch o tag `deploy/<nome>`, che non
   toccano il working tree di `$WORK/wearusfm`), con procedura versionata eseguita da un job CPU su `lrd_all_serial`; nessuna modifica del
   codice sul cluster; un checkout usato da job in coda o in esecuzione non si tocca; i launcher verificano `CODE_DIR`, SHA atteso e
   checkout pulito, e importano codice e configurazioni da `CODE_DIR`. Aggiornati `CLAUDE.md` e `.claude/agents/leonardo-ops.md` (che vietava
   ogni checkout remoto; passato anche da Haiku a Sonnet, dopo l'esaurimento del limite settimanale di Haiku del 07/10).
3. **Budget «preparazione dell'infrastruttura della finestra 1»: 25 GPU-ora**, separato dal budget della finestra 1 e da quello del passo 6.
   Non e' un'autorizzazione ai lanci: ogni prova si presenta con preventivo, si lancia col via di Simone, consuntivo qui.
4. **Avvio:** worktree di Claude creato (`.claude/worktrees/claude-training-engine`, branch `worktree-claude-training-engine`, da 8dad7e8),
   poi il contratto delle loss distribuite da concordare con GPT.

**Precisazioni tecniche concordate** (da rispettare nell'implementazione, prima di qualunque prova):
- **Contratto delle loss.** `masked_mean` fa la media dentro ogni finestra e poi fra le finestre con almeno un elemento valido: con piu' GPU,
  la media dei gradienti per GPU non coincide con quella del batch globale. Per ogni componente (JEPA e ciascuna ancora, conteggi distinti,
  pesi delle ancore invariati) si espongono la **somma delle medie per finestra** e il **numero di finestre valide**; il denominatore copre
  **tutte le GPU e tutti i micro-batch dello stesso aggiornamento**. Con `loss_per_window=False` resta il comportamento attuale. Le riduzioni
  in `jepa.py` restano di GPT; l'interfaccia si concorda e si congela prima.
- **Batch 32 = 32 finestre globali** per aggiornamento (es. 4 GPU x 8, o 4 GPU x 4 x 2 accumuli), non 32 per GPU. Clipping, passo
  dell'optimizer ed EMA una sola volta per aggiornamento; **WSD contato in aggiornamenti dell'optimizer**, non in micro-batch.
- **Test d'accettazione DDP:** 1 GPU senza accumulo contro 4 GPU (anche con accumulo) sullo stesso batch globale: loss, gradienti, pesi ed
  EMA entro tolleranze dichiarate prima; maschere eterogenee, una GPU senza target validi, parametri inutilizzati, arresto coordinato,
  ripresa dello stream. La diagnostica dei gradienti con `torch.autograd.grad` (`run.py`) non e' compatibile col normale funzionamento di
  DDP: percorso apposito.
- **Memoria:** DDP replica il modello; il primo test del 300M deve arrivare almeno al primo aggiornamento AdamW (stati dell'optimizer
  creati). Se non entra nemmeno col micro-batch minimo, si rivede il perimetro (oggi: DDP su un nodo, 4 GPU; FSDP e multinodo fuori).
- **Sottomissioni:** prenotazione con lock (`mkdir` atomico sul Mac) e stato persistente (prenotato -> in sottomissione -> sottomesso con
  ID, oppure ambiguo); ID logico nel `--comment`; uno stato ambiguo si riconcilia con `sacct`, mai cancellazione del lock e rilancio alla cieca.
- **Dataloader:** il profilo per fasi si raccoglie gia' nelle prime prove; l'ottimizzazione dopo.
- **Calendario:** prima lista dei job della finestra 1 entro il 14/10 (GPT i bracci scientifici, Claude costo e fattibilita' per job),
  congelamento con la firma di Simone entro il 25/10. Tappe di Claude: ~14/10 deploy di campagna, launcher con `CODE_DIR`, sottomissione con
  lock, DDP su 4 GPU con test d'equivalenza; ~20/10 WSD e preset 100M/300M con memoria e throughput misurati; ~25/10 dataloader sotto il 2%
  se la misura lo permette; 26-31/10 prove su `dbg` dei job della finestra 1.

### Integrazione dell'accordo e precisazioni GPT (09/10/2026)

Simone in questa chat: «va bene, allora io direi di farvi partire, devi committare qualcosa mentre aspetti ?» e poi
«va bene, allora committa l'accordo operativo e cosi lui puo partire». Ha inoltre richiesto di verificare l'aggiornamento di
CLAUDE.md e di leggere/aggiornare sempre il registro condiviso.

- Individuato il commit di Claude `ffda2c3dad22bbfbf72a5758a830523246a5efae` nel worktree gia' creato. Integrato su main locale
  mediante merge fast-forward, preservando SHA, contenuto e voci del registro. Ritirata soltanto la bozza GPT non committata
  sovrapposta. Nessun push remoto, nuovo job o modifica del worktree di Claude; il checkout della campagna P1/P2 resta invariato.
- Il contratto di riduzione include anche le componenti keep e ogni ramo opzionale attivo, con pesi/medie interne invariati.
  I conteggi devono riflettere le stesse maschere di validita' realmente usate nella loss, non approssimazioni dei validi. Rendere
  disponibili i denominatori dell'intero batch effettivo senza conservare tutti i grafi dei micro-batch. Firma/struttura dei risultati,
  trattamento dei conteggi zero e compatibilita' con i chiamanti single-GPU si concordano prima dell'implementazione.
- Precisare il test di resume prima di implementarlo: obiettivo di medesima sequenza di campioni/crop/maschere a world size e worker
  invariati, con stati RNG/stream/prefetch gestiti e confronto continuo contro interrotto/ripreso entro tolleranze prespecificate.
  Nessun requisito di ripresa elastica con world size diverso; una ripresa soltanto stocastica non equivale al requisito e va segnalata.
- Il tetto separato 25 GPU-ora equivale a 200 ore locali GPU secondo la conversione gia' usata nel progetto. Consumo iniziale della
  nuova voce: zero; nessun riaddebito dell'A/B. Preventivo e limite CPU separati anche per il deploy; nessun lancio autorizzato da
  questa sola registrazione. Il budget scientifico W1 e l'adozione della keep rimangono decisioni distinte.
- Preflight del deploy: verificare che ref/configurazione/hook non aggiornino checkout attivi, risolvere lo SHA e creare un checkout
  detached nuovo; nessuna conversione in corsa della campagna P1/P2. Allineata l'eccezione sbatch CPU/dbg nelle regole dei test.
  Preservati il modello Sonnet dell'ops di Claude e le istruzioni per importare src del worktree nei test locali.

Queste sono registrazione e review del contratto, non esiti di collaudo: DDP, accumulo, WSD, resume, nuovi preset e deploy restano
da implementare/verificare. Il commit delle precisazioni e' solo documentale, non modifica trainer, loss, target o soglie scientifiche.

## Review delle prime consegne Claude: contratto e infrastruttura (09/10/2026, GPT)

Riletti CLAUDE.md, registro e commit `c333b9d754a276314103bb7fa10fb66b388c15ec` (contratto) e
`6d7cdef401a390ee6fa613ee8de77aa5b833e675` (deploy/sottomissioni). Durante la review Claude ha aggiunto la correzione
`195fc588342fb406bae4b8cedde4105ba7aff7a2`, anch'essa esaminata, e il merge di main `e7d70eb`.
Worktree GPT di review: `/private/tmp/wearusfm-infra-review.LhAA9s/gpt-review`, branch `codex/review-claude-infra-20261009`, base `a8758df`.
Test eseguiti su checkout isolati dei DUE commit, senza modificare il worktree di Claude, senza push/job/HPC; costo 0 GPU-ora.

### Contratto: struttura accettata, dettagli vincolanti per l'implementazione

- Accettata la struttura `jepa_loss_terms -> LossTerms` con `num`, `den`, `weight`, `log_num`, `log_den`; `jepa_losses` resta
  compatibile con i chiamanti attuali. Denominatori sull'intero batch effettivo, fattore R nel backward, batch32 globale, singolo
  clipping/update/EMA. Chiavi delle componenti/diagnostiche concordate fra rank, conteggio zero per una componente localmente assente:
  non iterare collective su dizionari con chiavi diverse fra rank. I rami spenti conservano il comportamento dei parametri inutilizzati.
- L'opzione dei conteggi senza forward e' praticabile per la geometria attuale: `frontend.forward` usa
  `floor(n_samples/fs/patch_s + 1e-9)`; `encode` combina `patch_valid`, QC e visible. Il dropout muscolare cambia l'identita', non
  queste maschere. `loss_counts` deve ricevere esplicitamente la geometria patch e, con keep attiva, `target_active` della calibrazione:
  `SpectralKeepHead.loss` interseca valid con quel buffer, che non si deduce dai soli target. Esporre anche gli input RVQ quando attivi.
  Condividere la logica di validita' con i termini e verificare conteggi a ogni passo; zero validi/errore di conteggio -> stop coordinato.
- GPT scrive riduzioni/conteggi e test delle loss; Claude scrive ciclo DDP e test distribuiti. Per la keep preservare prima la media
  equiponderata delle famiglie valide DENTRO ogni campione, poi la media fra campioni validi, senza sostituirla con medie globali per famiglia.
- Rev.1 proponeva uno stream nuovo al resume; rev.2 ristabilisce la sequenza identica ed e' coerente col requisito. Il disegno di replay
  segmentato resta da verificare: `_session` puo' saltare una sessione se `scale` fallisce, consumando ulteriori estrazioni casuali.
  Un seed indipendente per le scale non dimostra da solo indipendenza della sequenza dai segnali. Il replay deve ricostruire anche
  scale/skip e pool, con test di cache hit/miss, sessioni scartate, montaggi virtuali, prefetched batch, interno/bordo del segmento.
- Teacher fuori da DDP: inizializzarlo/copiarlo DOPO che lo student e' sincronizzato, oppure sincronizzare esplicitamente anche
  teacher e buffer; uguaglianza al primo forward e al resume, non soltanto dopo EMA. La diagnostica sul rank0 deve isolare hook DDP
  e RNG: chiamare il medesimo student attraverso `.module` non rimuove gli hook. Tolleranze CPU/GPU da fissare prima dei test;
  distinguere test delle riduzioni float64 dai percorsi del modello che fanno cast float32, in particolare il readout keep.

### Esito della review del deploy/sottomissioni: correzioni richieste a Claude prima del merge

- **P1, stato letto prima del lock:** in `submit_job.reconcile` lo stato viene letto prima di `reg.lock`. Se un altro processo lo
  risolve a `sottomesso` fra lettura e lock, la riconciliazione usa la vecchia copia e puo' sovrascrivere l'ID valido con `assente`.
  Riprodotto sia in 6d7cdef sia in 195fc58. Richiesta: acquisire il lock, rileggere stato/storia e verificarne la transizione sotto lock;
  aggiungere un test dell'interleaving. I due controlli distanziati non correggono questa race.
- **P1, verifiche del checkout che non arrestano il job:** `lib_code_dir.sh:37` e `campaign_checkout.sbatch:40` trattano stdout vuoto
  di `git status` come checkout pulito anche quando Git termina con errore. `campaign_checkout.sbatch:42-43` non controlla i fallimenti
  di find/chmod; il job puo' stampare "pronto" ed uscire 0 lasciando codice scrivibile. Riprodotti errore Git e errore chmod su 195fc58.
  Richiesta: controllare esplicitamente gli exit code e fallire senza pubblicare successo; test per ciascun errore e risultato finale.
- **P2, ledger condiviso:** i lock per job non serializzano `ledger.tsv` fra job diversi. Due primi writer producono due intestazioni;
  riprodotto su 195fc58. Richiesta: lock del ledger per header e append, distinto dal lock per job, oppure riepilogo generato da stati.
- Chiuso dalla rev.2 il difetto di 6d7cdef `sacct; squeue` (errore accounting mascherato), insieme al passaggio ad `assente` dopo
  una singola risposta vuota; verificati test nuovi di errori, risposte inattese e controlli distanziati. Un'assenza dopo intervalli
  fissati resta un criterio operativo, non una prova di latenza massima dell'accounting; resta necessario un controllo esplicito prima
  di autorizzare la risottomissione. Nessun uso reale su Leonardo di questi script verificato in questa review.

**Verifiche:** su 6d7cdef, 25 test pertinenti PASS (`test_campaign_deploy`, `test_submit_job`, `test_training_run`), bash -n PASS,
5 riproduzioni locali confermano i difetti iniziali. Su 195fc58, 30 test pertinenti PASS e 4 riproduzioni locali confermano i residui
(race sullo stato, Git status, chmod, header ledger). Le riproduzioni asseriscono la presenza del difetto, non il successo di una patch.
Artefatti esterni: `/private/tmp/wearusfm-infra-review.LhAA9s/test_review_repros.py` e `test_review_rev2.py`, con checkout dei due SHA.
Il conteggio 766 della suite completa e' quello riportato da Claude; GPT ha rieseguito i test pertinenti, non l'intera suite.

L'integrazione dei nuovi commit resta in attesa delle correzioni dell'owner; `ffda2c3` e l'accordo operativo restano gia' integrati.
La presente registrazione non cambia codice o soglie, non autorizza nuovi job e non modifica i checkout della campagna P1/P2.

## Warmup del WSD nei rami: D = 4 epoche (Simone, 09/10/2026: «si D=4 epoche»)

D14 fissa il warmup al 2% dei passi a D (almeno 1.000). Nel pilot sulle epoche (un run a 8 epoche, rami di decay a 1, 2 e 4) e nella
continuazione a 2D il warmup deve essere lo stesso per tutti i rami, altrimenti un ramo non coincide con un run fatto da zero (provato bit
per bit sulla CPU, `tests/cpu/test_training_wsd.py`). **Deciso: D = 4 epoche per tutti i rami di un rung**, cioe' `--steps-at-d` = passi di
4 epoche (al 30M con batch 32: ~92.600 -> warmup 1.852, come nei fogli firmati). Implementazione: `training/run.py` (`Schedule`, `with_wsd`,
`wsd_warmup_steps`, `branch_from`), commit 56bc7e6 sul branch di Claude.

## Preset dei rung della ladder: FIRMATI (Simone, 09/10/2026: «firmo i preset»)

`training.run.LADDER` (commit 772b991 sul branch di Claude). Larghezza della tabella di v10 §10.4, teste da 64, K = 64; profondita' del backbone
che centra il nominale coi nostri blocchi (due attenzioni, tempo e latenti, piu' l'MLP: ~16 d^2 parametri invece dei ~12 d^2 di un blocco
standard, quindi 12 e 18 blocchi ~ 16 e 24 blocchi standard, il bordo basso di v10):

| Rung | d | Encoder locale | Backbone | Decoder | Parametri | Backbone / locale / Perceiver |
|---|---|---|---|---|---|---|
| 30M (= sanity) | 384 | 2 | 8 | 2 | 28,3 M | 67 / 13 / 6% |
| 100M | 640 | 2 | 12 | 2 | 104,8 M | 75 / 9 / 5% |
| 300M | 896 | 3 | 18 | 2 | 292,0 M | 79 / 10 / 3% |

Da dichiarare: al 300M il Perceiver (blocchi fissi per costruzione) resta sotto il 5% di v10. Picco del learning rate e batch dei rung maggiori:
dalla calibrazione della finestra 1 (D14), oggi quelli del sanity. Il preset 1B non e' definito (finestra 2, FSDP fuori dal perimetro iniziale).

## Review e integrazione locale delle consegne Claude (GPT, 12/10/2026)

Su «tocca a te», priorita' al punto 0: esaminati insieme i commit `7ed8b59`, `4fcc1df`, `92e137e`, `56bc7e6`, `772b991`, `1a65a1d`,
incluse le dipendenze c333b9d/6d7cdef/195fc58 e i merge intermedi. Snapshot verificato:
`1a65a1d2b3744d09826da0239bc2c7224ff5a4aa`, checkout GPT isolato `/private/tmp/wearusfm-review-20261012.6LElAx/candidate`.
Integrazione locale con fast-forward: SHA originali conservati, nessuna riscrittura o modifica del worktree di Claude.

**Verifiche GPT:** 98 test pertinenti unici verificati (deploy/sottomissioni, trainer e sicurezza numerica, DDP, replay, WSD, preset,
loader e sonda della memoria), piu' 2 controlli aggiuntivi esterni di accettazione. Il primo giro nel sandbox: 93 PASS e 5 errori di
connessione localhost; il file DDP rieseguito col permesso necessario: 6/6 PASS (un caso gia' passato nel primo giro, non contato due volte).
I due controlli aggiuntivi: training continuo/ripreso con DataLoader reale a 3 worker, segmento non multiplo dei worker; training DDP a
2 processi continuo/ripreso. Student, teacher e RNG finali identici bit per bit. Script riproducibile:
`/private/tmp/wearusfm-review-20261012.6LElAx/test_review_extra.py`. Bash -n sui 6 launcher/libreria e git diff --check PASS.
La suite completa 813 rimane il conteggio dichiarato da Claude: GPT ha rieseguito il sottoinsieme pertinente, non tutta la suite.

**Esito:** chiusi i tre difetti della review precedente: stato riletto sotto lock e transizioni controllate, errori Git/find/chmod
espliciti e sola lettura verificata, lock proprio del ledger. DDP di collaudo, ripresa esatta nelle condizioni provate, WSD e preset
accettati per l'integrazione locale. Il vero training DDP resta rifiutato finche' GPT non consegna somme/conteggi e Claude non collega
riduzione globale e accumulo. Le interfacce del contratto restano quelle concordate nella review del 09/10.

**Limiti da conservare:**
- Cambia lo stream rispetto al codice storico: confronti accoppiati solo fra run del medesimo codice; checkpoint storici senza stream
  non garantiscono una continuazione esatta. Test di equivalenza delle loss globali e NCCL/bf16 ancora da fare.
- WSD discreto: pre-decay dopo 80% degli update; primo update del decay al picco, ultimo a lr_picco/n_decay, valore zero al confine D
  dopo l'ultimo update. Questa convenzione e' esplicita nel codice/test di Claude; non descrivere l'ultimo update applicato come lr=0.
- Restano i limiti operativi del submitter gia' dichiarati da Claude (prenotazione iniziale che puo' lasciare un ID orfano, lock da
  recuperare manualmente dopo SIGKILL, ledger su filesystem locale). Non autorizzano rilanci automatici.
- `collaudo_infra_gpu_1.md` e' integrato come PROPOSTA: 2,5 GPUh + 8,2 core-ora CPU, non come autorizzazione. Usare uno SHA completo
  revisionato e preflight dei ref/hook. Prima del lancio fissare comandi/input/risorse esatti e margine per i due torchrun rispetto al
  walltime (i default 1500+300 secondi non lasciano margine per startup/checkpoint). Nessun job/push fatto in questa attivita'.
- Sonda memoria: Hyser fisso su una GPU e stima dei bucket, non misura DDP/throughput 100M/300M o garanzia per tutti i dataset.
  Associare report a log/SHA del checkout; qualunque OOM, risultato non finito o caso mancante non certifica fattibilita'.

Budget infrastruttura non consumato da questa review (0 GPUh). Picco LR 3e-4 e batch globale32 anche ai rung grandi, come richiesto da
Simone, finche' la calibrazione della finestra 1 li fissa. Prossima consegna GPT: `jepa_loss_terms` / `loss_counts` con regressioni CPU.

## Consegna GPT: termini e conteggi delle loss (12/10/2026)

Implementati `jepa_loss_terms` e `loss_counts` senza modificare training/run.py, dataloader o launcher di Claude. Interfaccia esatta,
riduzione dei contatori e confini di ownership documentati nella sezione6 di `contratto_loss_distribuite.md`.

- Numeratori additivi e conteggi int64 detached per JEPA, RMS/inviluppo/bande, multi-scala, RVQ e keep; media prima dentro ogni finestra
  dove firmata, non media delle medie locali. Keep conserva le famiglie equiponderate dentro ogni campione.
- Conteggi senza forward da FMConfig, geometria/QC, maschere, target e buffer keep calibrato. Se manca geometria/calibrazione, rifiuto;
  nessun consumo RNG nei conteggi. Selezioni RVQ condivise fra precount e forward; codici forniti mediante lookup puro.
- LossTerms espone num/den/weight e diagnostiche detached log_num/log_den, con log_sum per contatori additivi. Container registrato
  come pytree per il percorso DDP. backward_loss calcola R*num/D, non fa collective o update e non divide di nuovo per i micro-batch.
- Il chiamante storico jepa_losses conserva firma, chiavi, medie, RNG e ValueError senza query; i termini nuovi ammettono anche zero
  query locali, con num0 differenziabile/den0. L'assenza di target GLOBALI richiede stop coordinato nel trainer.

Verifiche: 18 regressioni nuove nei3file CPU esistenti; 76 test pertinenti unici PASS complessivi, 1 skip MPS. Compresi confronto
batch globale vs2micro-batch sul modello completo float64 (rtol/atol1e-10 per loss, gradienti, pesi dopo AdamW/clipping, teacher dopoEMA),
precounts contro conteggi del forward con padding/QC/maschere eterogenee, un pezzo senza target RMS, scala lenta assente, targeta/b,
RVQ slab/channel e codici mancanti, keep con target_active e famiglie/validita' disomogenee, assenza locale di query/target,
contatori interi oltre la precisione bf16 e roundtrip pytree.

Inoltre2test esterni contro il sorgente JEPA congelato al commit7e989d6: loss/chiavi, gradienti e RNG bit per bit invariati con
loss_per_window False/True (train mode, dropout attivo, ancore multi-scala/RVQ e diagnostiche). Artefatti:
`/private/tmp/wearusfm-review-20261012.6LElAx/legacy_jepa.py` e `test_legacy_loss_compat.py`.
Regressioni trainer/sicurezza/WSD/replay rieseguite. Non e' la suite completa e non e' il test gloo/NCCL delle nuove loss.

**Passaggio a Claude:** ora puo' collegare il modulo DDP ai termini, sommare denominatori su rank e micro-batch prima del backward,
confrontare conteggi precalcolati/effettivi e coordinare mismatch/zero globali, ridurre metriche senza medie locali, fare un solo
clipping/optimizer/EMA per batch effettivo ed estendere l'equivalenza gloo. Guardia del trainer DDP vero ancora accesa fino a quella
consegna; no flip della guardia, collaudo CUDA, push o nuovo job in questa attivita'. Budget GPU invariato.

## Primo collaudo GPU dell'infrastruttura: AUTORIZZATO (Simone, 12/10/2026: «intanto che aspettiamo, autorizzo il preotocollo gpu»)

Protocollo `docs/collaudo_infra_gpu_1.md` (criteri fissati e committati prima del lancio, comandi e input esatti nella sezione finale).
Quattro job: checkout di campagna (CPU), misura per fasi del dataloader con le regole della finestra 1 (CPU), DDP a 4 GPU sul 30M con le
loss del contratto e ripresa nella stessa allocazione, sonda della memoria dei rung 30M/100M/300M su una GPU. **Costo massimo 2,5 GPU-ora
(20 ore locali) + 8,2 core-ora CPU**, budget «preparazione dell'infrastruttura della finestra 1»: 25 GPU-ora, usate 0 finora, residuo
minimo dopo il collaudo 22,5. *Mia lettura, da correggere se sbagliata:* «intanto che aspettiamo» (la review di GPT del commit 16f5a5d)
indica la strada (b) del protocollo, cioe' il codice del branch di Claude pubblicato con `deploy/collaudo-infra-1`, senza la review di GPT;
nessun push su `main`. SHA del checkout, ID SLURM ed esiti si registrano qui al lancio e alla fine.
**Lancio (12/10/2026):** ref `deploy/collaudo-infra-1` pubblicato su leonardo a `cd335c9740702e10e0823973f386a92bc0ef7eec` (35 file di codice e
documenti rispetto al `main` di leonardo `8dad7e8`, nessun dato; `main` di leonardo invariato). Checkout `infra1_checkout` **60150574**:
COMPLETED 0:0, 11 s, «checkout di campagna pronto», 423 file in sola lettura. **Incidente senza costo:** alla prima sottomissione i tre
launcher sono stati rifiutati da sbatch («Unable to open file»): sbatch legge lo script al momento della sottomissione, e i launcher stavano
nel checkout non ancora creato; nessun job creato. Verificato il checkout dal log, risottomessi senza dipendenza con `--dopo-rifiuto`, come
previsto dal protocollo, stessi comandi e costi: `infra1_loader` **60150682**, `infra1_ddp` **60150683**, `infra1_memory` **60150685**, tutti
RUNNING. *Per i prossimi lanci da un checkout di campagna:* sottomettere i launcher solo dopo che il job di checkout e' finito (oppure passarli
su stdin quando non hanno argomenti). Esiti e consuntivo da registrare alla fine.

**Esito del collaudo GPU dell'infrastruttura (12/10/2026, letto dopo la fine dei job; criteri fissati prima):** tutti COMPLETED 0:0.
- `infra1_ddp` 60150683 (5 min 3 s): **superato**, 4 criteri su 4: nessun errore, stop, valore non finito o OOM; `world_size` 4,
  `batch_per_process` 8, `accum_steps` 1, loss col contratto, codice `cd335c9`; ripresa dal passo 300 fino a 330; un record per passo da 1
  a 330 con 32 finestre. **0,73 s/passo** (mediana dopo il passo 50; media 0,74 senza il passo 301) per 32 finestre su 4 GPU, contro 2,46 s (media) su una GPU nel collaudo
  59624004 (stessa configurazione): ×3,3. Memoria massima 9,7 GiB per GPU. Classi A/B/C 1.924/7.730/906 finestre, 422 virtuali.
  **Attesa sui dati 7,2% (regola di D6a: 2%, non rispettata; si riporta):** 6,9 s su 15,4 vengono dal solo passo 301, il primo dopo la
  ripresa (avvio dei worker e ricostruzione a secco dello stream); senza quel passo 4,1%, mediana 25 ms per passo, costante: e'
  il trasferimento del batch dal worker al processo principale, non la produzione delle finestre (sotto). Da ottimizzare; nessuna modifica
  al volo.
- `infra1_memory` 60150685 (1 min 32 s), A100 64 GB, caso peggiore (Hyser 256 canali, finestre da 4 s, 160 patch), dopo 2 aggiornamenti
  AdamW: memoria riservata + bucket DDP **30M 10,1 / 19,8 GiB, 100M 17,5 / 31,7 GiB, 300M 28,1 / 50,4 GiB** con 4 / 8 finestre per GPU.
  **Tutti e sei i casi stanno sotto i 60 GiB:** la finestra 1 non richiede l'accumulo neppure al 300M con 8 finestre per GPU. Secondi per
  passo nel caso peggiore su una GPU: 0,8 / 1,6, 1,5 / 2,9, 2,8 / 5,5.
- `infra1_loader` 60150682 (2 min 25 s, 8 CPU): con le regole della finestra 1, **6,5-8,7 finestre/s per processo**; fasi: lettura e filtro
  37%, target delle ancore 37%, maschera 17%, sessione e finestra 7%, codici e vicini 1%. Con 6 worker per GPU (~44 finestre/s) contro le ~11
  richieste dal passo DDP misurato: la produzione basta in media.
- Checkout 60150574 gia' registrato sopra. Riepilogo: `results/infra/collaudo_infra_gpu_1_1210.json`.
**Costo reale (sacct):** 0,362 GPU-ora (DDP 0,337, memoria 0,026) = 2,9 ore locali; 0,33 core-ora CPU. Budget «preparazione
dell'infrastruttura»: **usate 0,36 su 25, residuo 24,64**. Nota: DDP avvisa che con le regole della finestra 1 (RVQ e keep spente) non ci
sono parametri inutilizzati: `find_unused_parameters=True` costa un attraversamento del grafo in piu' per passo, da togliere quando le
teste opzionali sono spente (ottimizzazione da fare, non fatta al volo).

## Rami spettrali in parallelo al raw: step 2 (PSD) e step 3 (wavelet) implementati (Claude, 13/10/2026; richiesta di Simone)

Simone, 13/10/2026: «qua dobbiamo fare una roba algorirtmica. secondo me il tokenizer sui dati raw non basta da solo» (lo Step 1 della proposta
risultava gia' integrato da GPT, 07-09/10); poi, col pacchetto `WearUsFM_Step2_Step3_Codice_e_Test.zip` (sha256
`e994af3f5f414494fbba7cea9e22b0935f95f778417853d7f4a706d221b2453c`, usato come riferimento, non copiato): «contrariamente a quello che c'e nella
guida devi farli entrambi come rami, non solo lo step 3 in alternativa, lo voglio in paralello allo step 2 e magari invece di gabor valuta se
usare un banco wavelet».

**Valutazione Gabor contro wavelet (prima di scrivere il ramo; `scripts/eval_wavelet_vs_gabor.py`, `results/steps23/wavelet_vs_gabor_20261013.json`;
EMG sintetico, controllo software).** Il Gabor del pacchetto (centri 40-330 Hz, sigma_f = max(4 Hz, f/12), kernel comune di ~300 ms) lascia
sotto il 10% della risposta mediana il 28% della banda 20-450 Hz nella scala da 6 filtri e il 23% in quella da 24 (frequenze viste da nessun
filtro, sotto 40 e sopra 330 Hz e fra i filtri). Il banco wavelet: 0% (somma delle risposte fra 0,40 e 1,19 della mediana). Leggibilita'
lineare dei 32 target dello Step 1 dalle log-energie medie (ridge, test separato): forma fast/slow R2 **0,956/0,927 Gabor, 0,993/0,986
wavelet**, 0,9996/0,998 PSD. Costo per finestra (16 canali, 4 s, 2 kHz, CPU 1 thread): Gabor 101 ms, wavelet 52 ms, PSD 3,6 ms. Risoluzione
temporale nella scala veloce: equivalente (2 frame sopra meta' del picco per un burst di 20 ms). **Scelto il banco wavelet.**

**Cosa c'e' (codice: `model/spectral_branches.py`; modello, trainer, loader, estrazione).**
- Due rami affiancati al raw (`FMConfig.spectral_branches = ("psd", "wavelet")`, spenti per default): frontend alla fs nativa -> log-potenze
  standardizzate (statistiche del training, congelate) + disponibilita' -> encoder per scala (50 ms ogni 25 ms, 6 bande; 500 ms ogni 100 ms,
  24 bande: proiezione, due convoluzioni temporali, media e deviazione sui frame validi) -> media sui canali validi -> 128 numeri per ramo.
  Embedding P1/P2 = [readout raw dello Step 1 (2d), z_psd, z_wavelet] (`readout`, lo stesso nella loss e nelle sonde).
- PSD: la stessa PSD, le stesse finestre, bande e disponibilita' dei target dello Step 1 (test d'uguaglianza a 200, 1000, 1500, 2000, 2048 Hz).
- Wavelet: Morlet analitiche a Q costante, una per banda dei target (centro geometrico), banda a -3 dB uguale alla banda del target,
  normalizzate a banda equivalente di rumore = larghezza della banda (su rumore bianco energia = potenza di banda, provato), filtraggio nel
  dominio della frequenza, supporto +-3 sigma_t per filtro (da 5 ms a 73 ms nella scala veloce, da 18 a 359 ms nella lenta). Variante con
  centri (+-4%) e larghezze (0,8-1,2) apprendibili come nel pacchetto (`wavelet_learnable`), spenta.
- Loss sul readout con i rami (sostituisce quella dello Step 1, che resta per i modelli senza rami): per campione (1 - alpha) testa sul fuso +
  alpha media delle teste dei singoli rami (alpha 0,5), stessi 32 target e stessa calibrazione; peso complessivo `keep_weight`, confrontabile col
  controllo dello Step 1. I rami vedono solo la vista completa: mai la JEPA mascherata.
- Dentro `WearUsFM` (non un wrapper come nel pacchetto): teacher/EMA, checkpoint, DDP, accumulo e caricamento nelle sonde funzionano senza
  casi speciali. Rami e teste nascono dopo tutto il resto e con un generatore a parte: l'encoder raw e il generatore globale sono quelli del
  modello senza rami (confronti accoppiati, provato).
- Trainer: statistiche d'ingresso dei rami da `branch_calibration_batches` (16) batch del training con un generatore proprio, all'inizio di un
  run nuovo (provenienza in `spectral_branches.json` e nel summary); diagnostica a encoder congelato (`freeze_encoder`, step 2C del pacchetto):
  encoder raw dal teacher di `init_from` (controllo chiave per chiave), si allenano solo rami e teste con la sola loss sul readout, EMA solo su
  di loro (il raw del teacher resta bit per bit quello di partenza); per ora un processo senza accumulo. CLI: `--spectral-branches psd,wavelet`,
  `--wavelet-learnable`, `--branch-alpha`, `--branch-calibration-batches`, `--freeze-encoder`.
- Test preliminare 2A (nessun training): `extract_features(..., feature_set=...)` e `--feature-set` in `probe_p1.py`/`probe_p2.py`: "model",
  "stats" (32 statistiche dello Step 1, 0 se mancanti, + 32 indicatori), "model+stats", e i controlli "mask", "model+mask".
- Differenze volute dal pacchetto: una coordinata mai vista nella calibrazione resta fuori (il pacchetto si fermava con un errore: a meta'
  training ferma il run); un ritaglio senza frame validi da' i flag di presenza spenti (il pacchetto: errore).

**Limite da tenere presente:** con finestre da 1 s (P1, `window_s` 1,0) e il bordo di 0,1 s, le wavelet lente sotto ~48 Hz non hanno frame
validi (serve ~1,2 s di supporto alla banda piu' bassa): in P1 il ramo wavelet lento vede solo le bande sopra ~48 Hz; la PSD lenta si'.

**Scelte nuove, non firmate:** dimensioni 128/48, alpha 0,5, banda e normalizzazione delle wavelet, supporto 3 sigma, limiti della variante
apprendibile, 16 batch di calibrazione (seme [seed, 4242]), bordo di 0,1 s nell'estrazione, zero + indicatore per le statistiche mancanti.

**Verifiche:** 22 test nuovi (`tests/cpu/test_spectral_branches.py`; suite CPU completa 864 passati), compreso il contratto delle loss con i rami (1 processo x 2 micro-batch e
2 processi DDP gloo uguali al batch intero entro 1e-10). Nessuna prova CUDA, nessun EMG reale, **nessun job**. Le prove su Leonardo
(2A sulle sonde, ramo a encoder congelato, training congiunto) richiedono protocollo, soglie e preventivo approvati prima; il codice tocca
file di GPT (`spectral_step1.py`: due righe di aggancio; `fm_features.py`, `probe_p1.py`, `probe_p2.py`): review di GPT prima del deploy.

## Protocollo del test 2A dello step 2: soglie congelate, DA APPROVARE (Claude, 13/10/2026)

Simone, 13/10/2026: «parti dal protocollo del test 2A, intanto giro tutto a GPT». Protocollo `step2a_stats_v1` in `docs/step2a_stats_probe.md`,
scritto e committato PRIMA di qualunque misura. **Nessun job lanciato.**

- **Domanda:** le 32 statistiche spettrali fisse dello Step 1 (+ 32 indicatori) accanto all'embedding dei teacher B_seed0/B_seed1 dell'A/B dello
  Step 1 migliorano P1/P2 con le sonde ufficiali invariate? Nessuna GPU: embedding dalle cache gia' verificate della tranche
  `step1_downstream_4000_v1` (presenti e coerenti, controllo in sola lettura del 13/10), statistiche sulla CPU con l'hash degli ingressi che
  deve coincidere riga per riga con quello delle cache.
- **Criteri congelati (per task, delta = MS - M, media dei due teacher, confronto accoppiato `compare_reports`):** utile se delta >= +0,01, limite
  inferiore dell'intervallo > 0 e delta > 0 con tutti e due i teacher; non dannoso se delta >= -0,01. **2A positivo** = utile in almeno un task e
  non dannoso in tutti e due. Descrittivi: S e K da soli, dataset, teacher, maschere per dataset, riproduzione del report downstream da M.
- Tolto il controllo «embedding + indicatori» del pacchetto: le sonde sono per dataset e dentro un dataset gli indicatori sono costanti (contati
  e registrati dalla fase stats).
- **Costo da autorizzare:** 0 GPU-ora; CPU tetto 132 core-ora (8 job su `lrd_all_serial`, 8 CPU), stima ~75; ~1,5 GB su `$WORK`. Prerequisiti:
  review di GPT (tocca suoi file dell'harness), checkout di campagna da un riferimento `deploy/`, controlli in sola lettura subito prima.

**Aggiornamento dopo la review di GPT del protocollo 2A (13/10/2026; soglie invariate, nessuna misura fatta):** (1) con piu' di una maschera di
disponibilita' in un dataset (una finestra silenziosa basta: GPT ha riprodotto 17 coordinate contro 2) la fase stats si ferma prima delle varianti,
cosi' l'omissione di «embedding + indicatori» resta senza ambiguita' e il tetto resta 132 core-ora; (2) provenienza delle cache storiche
ricontrollata per intero, commit di estrazione `8dad7e8...` e di training `69fa2da...` compresi, e quella del training dei teacher; (3) i report
di B_seed1 portavano il checkpoint di B_seed0: ora ogni report ha il suo teacher. Lettura precisata: un esito positivo dice che aggiungere le
statistiche aiuta la sonda lineare, non che quell'informazione mancasse dall'embedding.

**Precisazione sul confronto Gabor/wavelet (review GPT del 13/10/2026):** il confronto del 13/10 e' fra due banchi concreti (il Gabor del
pacchetto e il banco wavelet scelto: frequenze, larghezze, normalizzazione, supporto e convoluzione contro FFT diversi), non una prova di
superiorita' della famiglia wavelet; l'R2 spettrale su EMG sintetico non dice nulla su P1/P2.

## Loss sul readout con i rami: keep raw dello Step 1 invariata + termine dei rami col suo peso (Simone, 13/10/2026; parere GPT d5c53c1)

Su una nota della review di GPT (con i rami la testa raw dello Step 1 restava inutilizzata e la testa fusa poteva appoggiarsi ai rami), proposta di
Claude (registro, 7bdd31b), parere favorevole di GPT (`d5c53c1`, sul suo branch), Simone: «direi che puoi priocedere». Implementato:
- `JEPA + ancore + keep_weight * keep(testa raw dello Step 1, invariata) + branch_weight * [(1 - alpha) L(fusa su [raw, PSD, wavelet]) +
  alpha * media L(rami)]`; nuovo `JEPAConfig.branch_weight` (default 0), due componenti additive `keep` e `keep_branches` nel contratto delle
  loss distribuite, stesso denominatore perche' stessi target, maschere e calibrazione (controllato su `target_active`), contati separatamente.
- Un solo encode non mascherato per testa raw e fusa; la keep raw e' bit per bit quella dello Step 1 (provato). Il raw conserva la stessa loss
  diretta di B, non lo stesso aggiornamento (la testa fusa manda altri gradienti al raw).
- **Il controllo B e' il modello SENZA rami.** Con `branch_weight` 0 la formula e' quella di B e il raw evolve bit per bit come in B (provato),
  ma l'embedding conterrebbe rami non allenati: e' un altro controllo.
- `freeze_encoder`: solo il termine dei rami (`branch_weight` > 0, `keep_weight` 0). Diagnostica dei gradienti (`keep_grad_every`) con i
  contributi `keep` e `keep_branches` separati sui parametri raw condivisi.
- **Ordine degli esperimenti (GPT):** prima il raw congelato, poi il training congiunto; `branch_weight` (e alpha) fissati prima, sui dati di
  training e guardando i gradienti pesati, senza scegliere sugli esiti P1/P2. Nessun valore proposto ora.
- Punto scientifico aperto: i rami predicono ancora statistiche molto accessibili dai propri ingressi; l'obiettivo self-supervised successivo
  dovra' affrontarlo (definito e versionato a parte).

## Protocollo dei rami a encoder raw congelato (step 2C): proposta, DA APPROVARE (Claude, 13/10/2026)

Simone: «intanto prepara il protocollo del raw congelato», con la sequenza in quattro punti di GPT (2A; rami a raw congelato con controllo
encoder casuale; obiettivo self-supervised dei rami; training congiunto). Protocollo `step2c_frozen_branches_v1` in
`docs/step2c_frozen_branches.md`. **Nessun job; codice del wrapper dopo l'accordo sul disegno.**
- Teacher B_seed0/B_seed1 congelati; un training per teacher con PSD e wavelet (solo termine dei rami, alpha 0,5; lr 1e-3, warmup 100, 3.000
  aggiornamenti, dati e configurazione dell'A/B); contributi di ciascun ramo per sottrazione di colonne; controllo coi rami al passo 0.
- Feature: M dalle cache downstream (il raw congelato e' quello dei teacher B) + uscite dei soli rami estratte sulla CPU, concatenate solo con
  hash degli ingressi e pesi raw verificati. Sonde ufficiali invariate.
- **Criteri congelati:** primario R+PW contro R: utile se delta >= +0,01, limite inferiore > 0, positivo con tutti e due i teacher; non dannoso
  se >= -0,01; positivo = utile in un task e non dannoso nei due. Secondari con la stessa regola: R+PW contro R+S del 2A (meglio delle
  statistiche fisse) e contro R+PW al passo 0 (l'allenamento conta). Descrittivi: R+P, R+W, dataset, teacher, supporto, encoder casuale di GPT.
- **Costo da autorizzare:** tetto 3 GPU-ora (24 ore locali) + 192 core-ora CPU; stima ~1,5 GPU-ora + ~125 core-ora. Da decidere prima: un
  training o tre per teacher; iperparametri dei rami o breve collaudo su dbg; teacher B soli o anche A.

## Step 2C approvato: tre scelte di Simone, collaudo su dbg, teacher A e B, criteri congelati (Claude, 14/10/2026)

Simone, sul disegno di `step2c_frozen_branches_v1`: «per questa mi piace la tua proposta, direi di procedere», e sulle tre scelte aperte:
«1) Quanti training: uno per teacher con i due rami, contributi per sottrazione», «2) un breve collaudo su dbg della curva di loss, circa
0,25 GPU-ora, che misurerebbe anche i tempi», «3) [...] Io proverei entrambe le soluzioni e valutiamo quale è la migliore, non abbiamo problemi
di ore sulla GPU», poi «se hai tutto vai pure». GPT (via Simone): il controllo con encoder casuale e' pronto; review indipendente del codice del
2C prima del lancio. Protocollo aggiornato in `docs/step2c_frozen_branches.md`, codice in `scripts/step2c.py`. **Nessun job lanciato.**
- **Teacher:** A_seed0, A_seed1, B_seed0, B_seed1 congelati, un training per teacher (PSD + wavelet, solo termine dei rami con peso 1, alpha
  0,5, lr 1e-3, warmup 100, costante, weight decay 0,05, clip 1,0, 3.000 aggiornamenti, dati e configurazione del braccio dell'A/B). Encoder del
  run (finale e passo 0) uguale bit per bit al teacher d'origine, o niente estrazione.
- **Collaudo congelato (prima di guardarlo):** B_seed0 su dbg, 30 minuti. Passa se: integrita' (nessuno STOP, encoder intatto, passo 0
  salvato); almeno 600 aggiornamenti; media della loss dei rami sugli ultimi 100 aggiornamenti <= 0,9 x quella dei primi 100; loss finita e media
  mobile su 50 aggiornamenti, dopo i primi 100, mai oltre 1,5 volte il minimo fin li'; allocazione dei run = (avvio + 3.000 x mediana s/passo) x
  1,25 + 10 min, al quarto d'ora, al piu' 2 ore. Non passa -> stop e nuova decisione, nessun ritocco automatico degli iperparametri.
- **Criteri congelati:** regola del 2A (utile: delta >= +0,01, limite inferiore > 0, delta > 0 con i due teacher; non dannoso: >= -0,01;
  positivo: utile in un task e non dannoso nei due). Primario per famiglia (A e B separatamente): R+PW contro R. Secondari: R+PW contro R+PW0
  (per famiglia); R+PW contro R+S del 2A (solo B; rimandato se il 2A non e' concluso, report verificato per hash nel confronto del 2A); quale
  famiglia: R+PW di B contro R+PW di A accoppiati per seme, «B» se la regola vale su B - A, «A» se vale su A - B, altrimenti nessuna differenza
  oltre le soglie. Descrittivi: R+P, R+W, R+PW0 contro R, A/B sul solo raw, loss, supporto, encoder casuale di GPT.
- **Review GPT del 14/10 sul codice dei rami (3dba14c), corrette:** a encoder congelato il totale ora e' `branch_weight` x termine dei rami (prima
  il peso non entrava nel backward); le teste del readout devono avere la stessa calibrazione dei target della testa raw (`stats_fitted`,
  `target_active`, `target_mean`, `target_std`), controllata a ogni passo e quando il trainer prepara o riprende un modello.
- **Costo da autorizzare (in due tempi):** collaudo <= 0,5 GPU-ora (4 ore locali); poi quattro training <= 2 h ciascuno, 8 estrazioni e 16
  sonde CPU. Tetto complessivo 8,5 GPU-ora (68 ore locali GPU) + 432 core-ora CPU; stima ~4,5 GPU-ora + ~270 core-ora. Disco ~15 GB su
  `$WORK`. Dal budget del passo 6.

### Step 2C: via al collaudo senza aspettare la review di GPT (Simone, 14/10/2026)

Simone: «la review arriverà oggi, quindi non aspetterei», «il riferimento di deploy riesci a ricavarlo tu?, gpt non è disponibile», «via al
collaudo per me». Come il 12/10 (strada b): `deploy/step2c-20261014` allo SHA del branch di Claude che pubblica i comandi (in
`docs/step2c_frozen_branches.md`, «Collaudo: comandi esatti»), `main` di leonardo invariato; checkout di campagna `step2c-20261014` (1 CPU,
10 minuti, 0,17 core-ora), poi `step2c_collaudo` (B_seed0, dbg, 30 minuti, **0,5 GPU-ora = 4 ore locali**, budget del passo 6). Preflight in
sola lettura registrato nel protocollo (i teacher dell'A/B stanno su `$SCRATCH`, non su `$WORK` come scritto prima: corretto). Se la review di
GPT trova difetti, si correggono prima dei quattro run. SHA e ID SLURM a seguire.
**Lancio (14/10/2026):** `deploy/step2c-20261014` pubblicato su leonardo a `1c9485260aaf32d5936346b50225f5cff1d1b8ea` (`main` di leonardo
invariato). Checkout `step2c_checkout` **60477316**: COMPLETED 0:0, 11 s, «checkout di campagna pronto», 440 file in sola lettura. Collaudo
`step2c_collaudo` **60477395** sottomesso dopo la fine del checkout (dbg, 30 minuti, nodo lrdn0325). Verdetto e consuntivo a seguire.

**Verdetto del collaudo (14/10/2026, letto dopo la fine del job; criteri congelati): NON PASSA, per la sola stabilita'.** `step2c_collaudo`
60477395 COMPLETED 0:0 in 25 min 23 s; 1.965 aggiornamenti al limite di tempo. Integrita' si' (encoder del teacher identico bit per bit in
studente e teacher, finale e passo 0, 265 tensori; nessuno STOP); aggiornamenti sufficienti si'; apprendimento si' (loss dei rami 0,235 nei
primi 100 -> 0,056 negli ultimi 100, rapporto 0,24 contro <= 0,9); tempo si' (0,69 s/passo di mediana, avvio 90 s: allocazione dei run 1 h);
**stabilita' no: media mobile su 50 aggiornamenti fino a 1,97 volte il minimo raggiunto (soglia 1,5)**, finestra dall'aggiornamento 1577
(media 0,095) contro il minimo 0,048 della finestra dal 1211; 137 finestre su 1.816 oltre 1,5, dalla 471 alla 1734.
*Descrittivo, dopo il verdetto (non cambia il verdetto):* la loss per batch ha code pesanti: 23 batch sopra 0,25 dopo i primi 100 (massimo
0,555 al 1578, con la norma dei gradienti massima 2,74; il 5,6% degli aggiornamenti tocca il clip a 1,0); il batch successivo torna nella norma
(mediana 0,072); autocorrelazione dei residui dopo il 600 circa 0 (0,08, 0,05, -0,01, 0,01 ai ritardi 1, 5, 20, 50); media mobile su 200
0,169, 0,092, 0,082, 0,070, 0,078, 0,068, 0,060, 0,070, 0,065. *Mia lettura, da verificare:* picchi isolati di singoli batch, non una
divergenza; la soglia su 50 aggiornamenti non distingue le due cose quando la loss e' piccola. Come previsto dal protocollo ci si ferma: nessun
run, nessun ritocco degli iperparametri, decisione a Simone. Riepilogo `results/passo6/step2c_collaudo_20261014.json`.
**Costo reale (sacct):** 0,423 GPU-ora (3,4 ore locali) + checkout 11 s su 1 CPU; dal budget del passo 6.

### Step 2C: si procede coi quattro run nonostante il collaudo (Simone, 14/10/2026)

Simone: «vai con l'opzione 1, lancia i quattro run», sulla proposta di Claude: lanciare i quattro run con gli iperparametri invariati,
registrando che il collaudo non e' passato per la sola stabilita' (picchi isolati di singoli batch, lettura di Claude) e che si procede per
decisione di Simone. **Decisione presa dopo aver visto il collaudo**, non un criterio congelato prima; le soglie e i criteri del confronto P1/P2
restano quelli congelati. Run: `step2c_train_<teacher>`, 1 A100, 1 h ciascuno (allocazione misurata dal collaudo), `boost_qos_lprod`, dal
checkout `1c94852`: **4 GPU-ora (32 ore locali)**. La catena CPU (8 estrazioni, 16 sonde, tetto 432 core-ora, gia' nel preventivo
dell'opzione 1) si sottomette a run finiti, dopo la review di GPT se chiede correzioni. Comandi in `docs/step2c_frozen_branches.md`.
**Lancio dei run (14/10/2026):** `step2c_train_A_seed0` **60486353**, `A_seed1` **60486358**, `B_seed0` **60486365** (tre RUNNING subito), `B_seed1` **60486381** (in coda per priorita'). Esiti a seguire.

**Esito dei quattro run (14/10/2026):** tutti COMPLETED 0:0 in 37 min 10-30 s, 3.000 aggiornamenti (`max_steps`), **encoder identico bit per
bit al teacher d'origine** (265 tensori, studente e teacher, finale e passo 0), nessun errore nei log, 0,69-0,70 s/passo. Loss dei rami
(primi 100 -> ultimi 100): A_seed0 0,247 -> 0,090, A_seed1 0,259 -> 0,075, B_seed0 0,235 -> 0,069, B_seed1 0,247 -> 0,065. Il passo 0 di
B_seed0 ha lo stesso SHA di quello del collaudo (`6d4f9e68...`): controllo al passo 0 deterministico. *Descrittivo:* testa fusa negli ultimi
100 aggiornamenti A 0,084 / 0,066 contro B 0,044 / 0,046 (le teste dei soli rami 0,083-0,096 in tutti); picchi di singoli batch piu' frequenti
con i teacher A (65 e 61 batch sopra 0,25 dopo i primi 100, clip dei gradienti nel 21% e 19% degli aggiornamenti) che con B (24 e 30; 5% e
6%); la media mobile su 50 supera 1,5 volte il minimo in tutti e quattro (1,97-2,24), come nel collaudo. Riepilogo
`results/passo6/step2c_runs_20261014.json`. **Costo reale (sacct):** 2,487 GPU-ora (19,9 ore locali); con il collaudo 2,91 GPU-ora dal
budget del passo 6. Punto aperto (Simone: «dopo sistemeremo la cosa della stabilità... magari sentendo chatgpt»): la soglia di stabilita'
del collaudo, da rivedere con GPT.
**Catena CPU (14/10/2026):** review di GPT non ancora arrivata (nessun commit nuovo nel suo worktree), quindi sottomessa come previsto
dall'opzione 1 con `scripts/slurm/step2c_cpu_chain.sh` dallo stesso checkout `1c94852`: 8 estrazioni, 16 job di sonda `afterok` della loro
estrazione, confronto nell'ultimo (P2, B_seed1) `afterok` di tutto il resto, senza il 2A (non ancora lanciato: secondario rimandato).
L'ultimo job ha 3 h invece di 2,5 (due sonde P2 piu' il confronto): tetto CPU 436 core-ora invece di 432. Se la review chiede correzioni al
codice di estrazione, sonde o confronto, i job ancora in coda si fermano e si rifanno da un deploy nuovo.
**Catena CPU fermata a meta' (14/10/2026), nessun retry:** sottomessi 10 job, le 8 estrazioni (`step2c_extract_<task>_<teacher>`
60491778, 60491780, 60491781, 60491783, 60491784, 60491785, 60491786, 60491787) e le sonde `step2c_probe_p1_A_seed0_main` 60491789 e `_rami`
60491790; l'undicesimo (`step2c_probe_p1_A_seed1_main`) **rifiutato da sbatch: `QOSMaxSubmitJobPerUserLimit`** (al piu' 10 job in coda per
utente sulla QoS `normal` di `lrd_all_serial`), e lo script si e' fermato da solo. In coda un solo job alla volta in esecuzione
(`QOSMaxCpuPerUserLimit`: 8 CPU per utente), gli altri in attesa: la catena e' **seriale** (come la tranche downstream del 09/10, sonde una
dopo l'altra). Stima seriale ~34 h di calendario. Inoltre le varianti RPW/RPW0 hanno 1.024 colonne invece di 768: se il tempo della sonda
cresce con le colonne (non misurato), due sonde P1 possono avvicinarsi alle 2 h del job. Decisione a Simone su come proseguire.

### Step 2C: estrazioni e sonde sui nodi GPU (Simone, 14/10/2026)

Simone: «opzione 2, vai su GPU» e, alla domanda sulle estrazioni, «Anche le estrazioni su GPU». La partizione CPU e' seriale per utente (10
job in coda, 8 CPU in esecuzione): ~34 h. Sui nodi GPU ogni job prende un quarto di nodo (1 A100, 8 CPU, 60 GB, `boost_qos_lprod`) e gira in
parallelo; la GPU resta inattiva (estrazione dei rami e sonde sklearn sulla CPU), stessi launcher e stesso checkout `1c94852`, cambia solo la
partizione. Si annullano i 9 job CPU ancora in attesa (7 estrazioni, le 2 sonde di P1/A_seed0); l'estrazione P1 di A_seed0 gia' partita
(60491778) resta e le sue sonde ne dipendono. Sottomissione con `scripts/slurm/step2c_gpu_chain.sh` (`KEEP=p1_A_seed0`): 7 estrazioni
`step2c_gextract_*`, 16 job di sonda `step2c_gprobe_*`, confronto nell'ultimo, senza il 2A.
**Costo:** tempi massimi estrazioni P1 1,5 h, P2 3 h; sonde (due varianti) P1 2,5 h (RPW,RPW0) e 2 h (RP,RW), P2 3 h e 2,5 h, l'ultima col
confronto 3,5 h: **tetto 57 GPU-ora (456 ore locali)**, stima ~38 (estrazioni ~8, sonde ~30). Budget del passo 6: ~33,08 usate prima del 2C
secondo la bozza di GPT (da riconciliare), piu' 2,91 del 2C finora e il tetto di 3 della baseline casuale di GPT: anche nel caso peggiore
~96 su 100. CPU: l'estrazione 60491778 (al piu' 1,5 h x 8 CPU = 12 core-ora).
**Catena su GPU, prima tornata (14/10/2026):** Simone ha annullato dal suo terminale i 9 job CPU in attesa (il permesso di `scancel` era
negato all'agente). L'estrazione CPU P1/A_seed0 60491778 e' finita in **TIMEOUT** dopo 1 h 33 min: ~54 min per modello sulla CPU di
`lrd_all_serial` (modello allenato completo, passo 0 a 2 dataset su 3), quindi le sue sonde sono state rifiutate da sbatch («Job dependency
problem»). Sottomessi 11 job: 7 estrazioni GPU `step2c_gextract_*` 60505198-60505204 (partite alle 14:42; P1 con 1,5 h, a rischio se la CPU
dei nodi GPU e' lenta come quella di `lrd_all_serial`) e le 4 sonde P1 di B 60505205-60505208. Lo script, ora riprendibile (riusa gli ID gia'
sottomessi, risottomette i rifiutati con `--dopo-rifiuto`; estrazioni P1 nuove con 3 h), completera' la catena dopo che Simone avra' tolto la
cartella parziale `step2c_1410/p1/A_seed0` (un'estrazione vuole cartelle nuove).
**Catena su GPU completata nella sottomissione (14/10/2026, 14:52):** Simone ha tolto `step2c_1410/p1/A_seed0`; lo script riprendibile ha
riusato gli 11 job gia' sottomessi e sottomesso i 13 mancanti: `step2c_gextract_p1_A_seed0` 60505763 (3 h) e le sonde 60505774-60505786
(`step2c_gprobe_p1_A_seed0_main` 60505774 risottomesso dopo il rifiuto), l'ultima `step2c_gprobe_p2_B_seed1_main` 60505786 col confronto,
`afterok` dell'estrazione P2/B_seed1 e delle altre 15 sonde. Sui nodi GPU la lettura di EPN-612 (91.800 finestre, 612 soggetti) ha richiesto
meno di 10 minuti.
**Estrazioni (14/10/2026, 16:06):** P1 B_seed0/B_seed1/A_seed1 COMPLETED in 1 h 23-25 min (limite 1 h 30), P2 tutte COMPLETED in 47-50 min;
P1 A_seed0 (60505763) in corso, fine attesa ~16:25. Sonde in partenza; confronto atteso in serata.
**Stabilita' del collaudo, nota di GPT (14/10/2026, via Simone) e verifica di Claude:** togliendo i batch piu' alti il rapporto resta
sopra 1,5 (collaudo 1,77 senza i 3 piu' alti, 1,62 senza i 23 sopra 0,25; mediana mobile su 50 1,66; run 1,80-1,98). **La lettura «picchi
isolati» era sbagliata:** anche la loss tipica su 50 batch oscilla fino a ~1,7-2 volte il minimo; composizione dei batch o regressione vera
non si distinguono con le metriche attuali. Il verdetto resta «non passa». Proposta (Claude, d'accordo con GPT): loss su finestre fisse e
origine dei batch nel trainer, regola di stabilita' da fissare con GPT prima del prossimo collaudo.

### Diagnostiche della stabilita' nel trainer (Simone, 14/10/2026: «sì, implementa i punti 1 e 2»)

Implementate in `training/run.py` (Claude): loss su un insieme fisso di batch del training ogni `fixed_eval_every` aggiornamenti
(`fixed_eval.jsonl`, identita' dell'insieme in `fixed_eval_set.json`) e origine di ogni finestra in `metrics.jsonl` (`log_batch_origin`,
sessioni in `origin_sessions.json`). Spente per default; non cambiano il training (provato bit per bit). La regola di stabilita' del prossimo
collaudo su queste misure si scrive con GPT prima di quel collaudo. Nessun job.

### Esito dello step 2C (14/10/2026, letto dopo la fine della catena; criteri congelati il 14/10 prima del collaudo)

Catena completa: 8 estrazioni e 16 job di sonda COMPLETED 0:0; confronto nel job 60505786 (checkout `1c94852`). Riepilogo completo in
`results/passo6/step2c_comparison_20261014.json`. Delta = accuratezza bilanciata della variante - riferimento, media dei due teacher, in punti.
- **Primario, R+PW contro R: NEGATIVO per tutte e due le famiglie** (nessun task «utile»; non dannoso in nessuno).
  B: P1 +0,34 (IC -0,04..+0,73; teacher +0,37, +0,31), P2 +0,75 (IC -0,76..+2,26; +0,89, +0,60). A: P1 -0,31 (IC -0,80..+0,18), P2 +0,76
  (IC -0,77..+2,29).
- **Secondario, l'allenamento conta (R+PW contro R+PW0): negativo**; l'allenamento dei rami **peggiora**: B P2 -1,59 (IC -3,07..-0,12, tutti e
  due i teacher negativi: «dannoso» per la regola), A P1 -0,84 (IC -1,38..-0,39).
- **Secondario, quale famiglia con i rami: «B»** (P1 +1,04, IC +0,30..+2,08, tutti e due i teacher positivi; P2 +0,81 non dannoso). Sul solo
  raw B - A: P1 +0,38, P2 +0,82, nessuno «utile».
- Secondario contro le statistiche fisse del 2A: rimandato (2A non concluso).
- *Descrittivo (non pre-registrato come test):* i rami NON allenati (passo 0) aggiungono piu' di quelli allenati: R+PW0 contro R, B P2 +2,34 (IC
  +1,42..+3,25; +1,79, +2,88), A P2 +1,40 (IC +0,21..+2,59), A P1 +0,53 (IC +0,22..+0,91); B P1 -0,10. Per dataset, coi rami allenati: EPN-612
  +2,13 in tutte e due le famiglie, UCI-EMG -0,6, NinaPro entro +-0,5. Contributi dei singoli rami (R+P, R+W) tutti sotto 1 punto.
- **Lettura secondo il protocollo:** negativo -> i rami cosi' allenati (target dello Step 1 sul readout) non aggiungono informazione utile alla
  sonda lineare: prima l'obiettivo self-supervised dei rami (punto 3), nessun training congiunto. *Ipotesi di Claude, da verificare, non un
  risultato:* l'ingresso spettrale dei rami porta informazione utile a P2 (il passo 0 e' una proiezione casuale delle log-potenze calibrate),
  e l'obiettivo attuale la comprime verso le 32 statistiche dello Step 1, perdendone una parte. Il 2A (statistiche fisse) dira' se basta
  l'ingresso spettrale senza rami. Incertezza condizionata ai due teacher per famiglia; P2 con intervallo descrittivo; piu' confronti senza
  correzione.
**Costo reale (sacct):** collaudo 0,423 + training 2,487 + estrazioni e sonde sui nodi GPU 26,41 (estrazioni 8,92, sonde 17,49) =
**29,32 GPU-ora (234,6 ore locali)**, entro i tetti (8,5 per collaudo e training; 57 per la catena su GPU); CPU ~15,1 core-ora (estrazione
60491778 scaduta e 60491780 annullata). Dal budget del passo 6.

### Lista comune delle cose critiche (Simone, 14/10/2026: «ok, la lista comune va bene, giro tutto a chat»)

1. Chiudere 2C (fatto) e 2A (GPT) -> scelta fra raw, statistiche fisse e rami appresi. 2. Baseline casuale (GPT; ipotesi di Claude sullo SHA
iniziale nel registro). 3. Integrazione su `main` dei commit del branch di Claude (review e merge GPT), in parallelo. 4. Monitor di stabilita'
(implementato, `5f62444`): review GPT e regola prima del prossimo collaudo. 5. Validare l'obiettivo: bilanciamento dei gradienti, sonde
temporali e fisiche, prima/dopo il bottleneck. 6. Chiudere la finestra 1: ablation, calibrazione LR/batch/K, attesa sui dati <= 2%,
equivalenza NCCL/bf16, configurazioni, budget, piano dei job. Scadenza: rinfresco di `$SCRATCH` entro il 22/10.

### Diagnostica dello stato iniziale per la baseline casuale (Simone, 14/10/2026: «sì, via, lancialo»)

Punto 2 della numerazione unica (baseline casuale, di GPT; Claude in supporto). Un solo job diagnostico, non una seconda catena della baseline:
`scripts/diagnose_initial_state.py` (commit `7d11afc`) su un nodo Booster in dbg, come il training dell'A/B e gli extract r2. Comandi:

    git push leonardo <SHA>:refs/heads/deploy/diag-initial-20261014
    python3 scripts/submit_job.py --id diag_initial_checkout --agente claude --campagna diag-initial-20261014 --gpu-ore 0 \
      --script-stdin scripts/slurm/campaign_checkout.sbatch -- --export=ALL,CAMPAIGN=diag-initial-20261014,CAMPAIGN_SHA=<SHA>
    # dopo COMPLETED e «checkout di campagna pronto»:
    python3 scripts/submit_job.py --id diag_initial_run --agente claude --campagna diag-initial-20261014 --gpu-ore 0.5 -- \
      --export=ALL,CODE_DIR=$WORK/wearusfm_campaigns/diag-initial-20261014,EXPECTED_SHA=<SHA> \
      $WORK/wearusfm_campaigns/diag-initial-20261014/scripts/slurm/diagnose_initial_state.sbatch \
      /leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/runs/step1_ab_0810 \
      /leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/passo6/diag_initial_20261014/report.json

Costo: checkout 1 CPU 10 min (0,17 core-ora; SLURM ne alloca 2); diagnostica 1 A100 su dbg, 30 min: **al piu' 0,5 GPU-ora (4 ore locali)**,
stima 5-10 min. Sola lettura dei run dell'A/B; rapporto in una cartella nuova. <SHA> = il commit che pubblica questi comandi.
**Esito della diagnostica (14/10/2026):** checkout `diag_initial_checkout` 60561424 COMPLETED (11 s), `diag_initial_run` 60561493 COMPLETED 0:0 in
48 s su lrdn0880 (Xeon Platinum 8358, Booster). **Costo: 0,013 GPU-ora + 0,006 core-ora.** Per tutti e due i semi:
- `ab_replica` (thread a 1 dall'avvio, import di `scripts/step1_ab.py`) riproduce lo SHA iniziale di `pairing.json` (A e B, studente e teacher);
- `gpt_like_env8_sklearn_first` (thread a 8, scipy e sklearn importati prima di torch, poi un thread come `initial_teacher`) lo riproduce;
- `torch_only_env1` (solo torch, thread a 1) e `gpt_like_env8` (torch per primo, poi un thread) NON lo riproducono; `env8_plain` neppure.
L'unico tensore diverso e' `tokenizer.fe_gain` (float64, calcolato alla costruzione): 982,425066637557 atteso contro 982,4250666375573 (~3 ULP;
982,4250666375588 con 8 thread). **Non e' il numero di thread: e' l'ordine degli import prima della costruzione** (le librerie caricate a fine
processo sono le stesse in `ab_replica` e `torch_only_env1`; il meccanismo esatto non e' isolato). Rapporto in
`results/passo6/diag_initial_state_20261014.json`. La correzione della baseline e il suo gate restano di GPT.

### Prova GPU/dati della finestra 1 approvata (Simone, 14/10/2026: «sì, via al protocollo GPU e leggi il tetto di nodi»)

Protocollo `w1_gpu_dati_v1` (`docs/protocollo_gpu_dati_finestra1.md`): (A) equivalenza 1 contro 4 GPU NCCL e accumulo in float32 e bf16 su 8
batch fissi (`scripts/ddp_equivalence.py`, test CPU entro 1e-10); (B) attesa sui dati al 30M, batch 32 e 64; (C) s/passo e memoria a 100M e
300M. Criteri precisati prima di qualunque misura: in float32 pesi e teacher entro 1e-2 del loro spostamento (AdamW amplifica le differenze
sui gradienti quasi nulli), loss e gradienti entro 1e-5. Sette job da 4 GPU su dbg, **tetto 14 GPU-ora (112 ore locali)**, stima ~8, dal
budget «preparazione dell'infrastruttura» (25, usate 0,36). Tetto di nodi dell'account: leonardo-ops non esegue `sacctmgr`/`sshare` (fuori
dal suo elenco); comando di sola lettura dato a Simone.

**Esito della prova GPU/dati `w1_gpu_dati_v1` (15/10/2026, criteri congelati prima; letti dopo la fine dei job).** Otto job COMPLETED 0:0
(checkout 60564394; (B)/(C) dal checkout `977e767`, (A) dal checkout `d7e2509`). Riepilogo in `results/infra/w1gd_20261015.json`.
- **(A) equivalenza 1 contro 4 GPU: SUPERATA** in float32 e in bf16 (8 aggiornamenti, 32 finestre, preset 30M, regole della finestra 1).
  Float32: loss <= 1,5e-7 e gradiente <= 9,7e-7 relativi, pesi <= 7,6e-5 e teacher <= 7,7e-4 del loro spostamento (soglie 1e-5, 1e-5,
  1e-2). Bf16: accumulo e DDP restano sotto la distanza bf16-float32 a ogni aggiornamento (gradiente <= 3,4e-3 contro 5,9-6,4e-3). La
  ripetizione identica del riferimento bf16 non e' bit per bit (gradiente fino a 1,2e-3: kernel GPU non deterministici), descrittivo.
- **(B) attesa sui dati al 30M: NON SUPERATA** (soglia 2%, D6a): 2,40% col batch 32 (mediana 17 ms per passo) e 2,81% col batch 64 (35 ms).
  Migliora rispetto al 12/10 (7,2%, 4,1% senza il primo passo dopo la ripresa) ma resta sopra. A 100M 1,24% e 1,73%, a 300M 0,69% e
  0,72%. Come da protocollo: nessuna accensione automatica dei montaggi virtuali, profilo per fasi e proposta da decidere prima dei run.
- **(C) s/passo (mediana dopo il 50) e memoria per GPU, DDP a 4 GPU, regole della finestra 1, tutti i dataset:** 30M 0,707 (batch 32) e
  1,366 (64), 10,0 e 19,3 GiB; 100M 1,390 e 2,692, 17,4 e 32,1 GiB; 300M 2,769 e 5,467 (64 con 2 micro-batch), 28,7 e 37,1 GiB. Il
  costo per finestra non scende raddoppiando il batch (~22 ms/finestra al 30M, ~43 al 100M, ~86 al 300M).
**Costo reale (sacct):** 82,6 minuti di nodo intero = **5,51 GPU-ora (44,1 ore locali)**, entro il tetto di 14; budget «preparazione
dell'infrastruttura»: 0,36 + 5,51 = 5,87 usate su 25 (la diagnostica dello stato iniziale, 0,013, e' nel passo 6).
## Baseline encoder casuale: via al completamento (14/10/2026)

Simone, chiarito che la preparazione GPT non era ancora committata o eseguita: «eh, allora muoviti e completalo».
Protocollo congelato `step1_random_downstream_4000_v1`, in `docs/step1_random_downstream.md`: due inizializzazioni storiche,
zero aggiornamenti, stato iniziale ricostruito e verificato per SHA; identiche finestre, encoder, preprocessing e sonde del downstream.
Confronti descrittivi accoppiati Random→A e Random→B, nessuna nuova soglia di adozione. Nessun peso appreso trasferito nel controllo.

Tetto comunicato: **3 GPU-ora + 64 core-ora CPU**, dal passo6, separato dal 2A e dall'infrastruttura. Checkout CPU aggiuntivo:
1 core,2GiB,10min, massimo0,1667 core-ora. Quattro extract GPU e quattro probe CPU, ultimo probe con confronto; dipendenze afterok,
nessun retry automatico, nessuna modifica al protocollo dopo gli esiti. Preflight fresco e registro comune obbligatori.
29 test CPU e review indipendente superati; nessuna misura reale disponibile al congelamento.

### Sottomissioni CPU della baseline casuale (14/10/2026)

Per il limite effettivo di dieci sottomissioni su lrd_all_serial, le quattro sonde della baseline vengono raggruppate per task in due
job da4h (8CPU,30GiB), con limite2h per ogni seme. Ciascuna usa lo stesso launcher, le stesse feature e la stessa sonda; il confronto
resta dopo tutte e quattro. Tetto invariato64 core-ora, quattro extract invariati3 GPU-ora. Questo e' un adattamento della coda nel
completamento autorizzato da Simone; nessun cambiamento al disegno o alle soglie, nessun retry o GPU aggiuntiva per le sonde.

### Stabilita' Step2C: audit senza cambio retroattivo dei criteri (14/10/2026)

Richiesta di Simone: verificare il log, lanciare il pendente e sistemare il problema della stabilita'. Il criterio v1 e' implementato
fedelmente: collaudo60477395 non passato. Audit locale dei log e review in docs/step2c_stability_review.md; nessuna riclassificazione
del collaudo o modifica degli endpoint P1/P2. Il criterio su batch differenti non distingue da solo deriva e difficolta' degli ingressi.
Monitor prospettico proposto su pannello fisso, piu' provenienza dei batch al trainer: parametri da congelare prima del prossimo run,
non una nuova soglia scelta per approvare questi risultati. Nessun ulteriore job HPC richiesto dall'audit locale.

### Baseline casuale: ricostruzione dell'ambiente storico dopo il gate fallito (14/10/2026)

Il primo tentativo si ferma prima delle feature: quattro extract FAILED per SHA iniziale diverso; sonde cancellate dalle dipendenze.
Costi65s totali =0,018056 GPUh. Diagnosi CPU riprodotta per entrambi i semi: il training storico costruiva con1thread; con2/8 cambia
soltanto il buffer float64 tokenizer.fe_gain (~2e-12), tutti i parametri bit-identici. Ora la costruzione usa temporaneamente1thread,
poi ripristina parallelismo e RNG. SHA iniziali e checkpoint di riferimento invariati; nessun parametro appreso trasferito.
16 test pertinenti PASS. Nuovo checkout/output r2; cap P1extract29min x2 e P2extract60min x2, piu'65s gia' consumati, massimo2,9847GPUh
entro3autorizzate. Sonde sempre64coreh. Prosecuzione del controllo autorizzato con bug ambientale corretto, senza ritocchi sperimentali.

## Riepilogo dei lanci GPT e dei limiti confermati (14/10/2026)

Simone: «verifica cosa ha fatto claude nel log e se hai cose da lanciare lanciale». Le campagne GPT sono interamente sottomesse;
ID, dipendenze e fotografia dello stato alle15:45:43 CEST sono in `docs/claude_GPT_shared`. Il completamento delle sonde resta da osservare.

- **2A**, protocollo `step2a_stats_v1`, codice congelato `3dba14c1176f9ac525aee0e851590ca7db4b4c39`: otto job CPU,
  8CPU/30GiB ciascuno, **tetto132 core-ora e0 GPU-ora**. Riusa le cache dei teacher B storici. Le soglie restano quelle del protocollo
  in `docs/step2a_stats_probe.md` a quel commit; gate di una sola maschera per dataset prima delle varianti. StatsP1 completato e gate
  superato su DB2/DB3/DB6 nell'ultima verifica; nessun esito P1/P2 aggregato ancora disponibile in questa registrazione.
- **Encoder casuale r2**, protocollo `step1_random_downstream_4000_v1`, codice `4fb4208071cd9d641762d683ae581dab6e0432ac`:
  quattro extract GPU e due gruppi CPU che eseguono le quattro sonde. Checkout60509955, extract60510132-60510135,
  gruppi60510137/60510139. **Cap cumulativo2,98472 GPU-ora entro3**, inclusi65s del tentativo fallito; **64 core-ora** per le sonde.
  SHA iniziali e confronti invariati; la correzione riproduce il contesto storico a1thread. All'ultima verifica il checkout e' ancora
  in codaCPU e le estrazioni lo attendono: non e' ancora verificato lo SHA della ricostruzione reale r2.
- **Accounting checkout:** richiesta1CPU/2GiB/10min, ma SLURM alloca2CPU. Cap contabile0,3333coreh per checkout;
  consuntivi osservati60506850 circa0,00611coreh e60508432 circa0,00667coreh. Overhead separato dai job scientifici.
- **Stabilita'2C:** audit dei cinque log committato in `results/passo6/step2c_stability_audit_20261014.json`; il collaudo resta
  non passato secondo v1. Nel collaudo, togliendo il massimo il rapporto resta1,77918 contro1,5. Monitor su pannello fisso e provenienza
  dei batch sono una proposta per il prossimo trainer, ancora da implementare/versionare; nessuna nuova soglia adottata o run autorizzata.

### Aggiornamento della baseline e della prontezza (14/10/2026, verifica16:46 CEST)

I quattro extract casuali r2 sono FAILED sullo stesso gate SHA iniziale; le sonde dipendenti sono cancellate. La correzione locale
dei thread non ha ancora ricostruito lo stato atteso sul cluster. Non esiste un risultato del controllo casuale e non si allentano
gli SHA o si scelgono nuovi valori iniziali. La causa residua richiede diagnosi dei tensori/buffer; nessun rilancio in questa verifica.
2A e sonde2C ancora in corso, senza confronto finale osservato. La scelta del default resta aperta. Il monitor di stabilita' su
pannello fisso resta proposto, da implementare prima del training congiunto e congelare prima dei nuovi esiti.

### Riconciliazione della lista comune ed esito2A (GPT, 14/10/2026, verifica22:53 CEST)

Aggiornamento operativo, non nuove autorizzazioni o soglie. Concordo con la lista comune diClaude: chiusura2A/2C, baseline casuale,
integrazione in parallelo, review/regola del monitor, validazione dell'obiettivo, configurazioni/ablation/calibrazione/budget della
finestra1; equivalenza quantitativaGPU/NCCL/bf16 e attesa dati<=2% ancora pendenti. RinfrescoSCRATCH entro22/10, prima del purge25/10.

**2A concluso: NEGATIVO**, protocollo step2a_stats_v1 e SHA3dba14c invariati. MS-M (punti percentuali di accuratezza bilanciata):
P1 -0,36593, IC95[-0,64305;-0,13331], due teacher negativi; P2 +0,06816, IC descrittivo[-1,24043;+1,37674], teacher di segno opposto.
Nessun task utile; entrambi non dannosi secondo la guardia puntuale>=-1punto. M riproduce i quattro report storici; gate di una maschera
per dataset superato su tutti e cinque i dataset. Otto jobCOMPLETED0:0, **52,7533coreh e0GPUh**, entro il tetto132. Risultato locale in
results/passo6/step2a_comparison_20261014.json; SHAorigine e normalizzazione del newline finale registrati nel file condiviso.

**2C concluso**, risultato Claude2e6be4a letto e finale60505786COMPLETED: primario negativo inA/B; conB, rami allenati-raw P1+0,34/P2+0,75
punti, sotto la soglia utile. Allenati vs passo0 suP2B -1,59punti; primario negativo non significa danno in tutti i task. Confronto
accoppiato rami vsMS del2A ancora pendente. Nessuna adozione o training congiunto automatici: obiettivo dei rami da riesaminare.

Monitor su pannello fisso e origine GIA' implementati daClaude5f62444 dopo il mio precedente snapshot; letti codice/test, non review
completa o confermaGPU. Restano cadenza, pannello e regola del prossimo collaudo; il fallimento storico2C non cambia. Gradienti pesati
JEPA/ancore/keep gia' presenti nelle metricheA/B: da analizzare. Sonde separate locale/backbone utili, ma non isolano da sole il Perceiver.

Baselinecasuale r2FAILED sul gate, gruppi cancellati: nessun risultato valido; **0,015GPUh r2, cumulativo0,033056 conr1**. Gia' suBooster
con1GPU, non su loginCPU: differenze numeriche dei buffer restano da diagnosticare. SHA e assenza di pesi appresi restano vincolanti.
Main5030a19; Claude7a4c965 ha39commit non integrati allo snapshot. Nessun nuovo job/merge/push in questa riconciliazione.

### Numerazione unica e ripartizione del lavoro (Simone, 14/10/2026)

Richiesta: «usiamo una numerazione unica e mettiamola nel file condiviso», con specchietto aggiornato e tranche parallela.
Riferimento operativo: blocco «Coordinamento attivo: numerazione unica dei sei punti critici» in apertura di docs/claude_GPT_shared:
1 confronti2A/2C (GPT per il secondario, Claude per i report2C); 2 baselinecasuale (GPT); 3 review/integrazione (GPT, correzioniClaude);
4 stabilita' (regola condivisa, reviewGPT, trainer/testClaude); 5 obiettivo (gradienti/sonde fisiche e temporali/propostaSSL GPT,
screening locale-backbone Claude); 6 finestra1 (scienza/piano/reviewGPT, fattibilita'/GPU/dataloader/calibrazione/launcherClaude).

Tranche immediata: GPT diagnosi/test baseline e review; Claude protocolloGPU/dati, bozza della regola di stabilita' e fattibilita'.
Sonde separate ancora in attesa come da registroClaude12585fd; nessun lancio implicito. SCRATCH entro22/10 coordinato daClaude.
Un writer per file, passaggio harness limitato ed esclusivo prima delle modifiche, ledger/submitter comuni. Firma finale di default,
configurazioni e budget resta di Simone; la numerazione non altera soglie, budget, autorizzazioni di job o campagne gia' congelate.

### Punto 2: guadagno fisso storico e prosecuzione r3 (GPT, 14/10/2026)

Richiesta corrente di Simone: «diagnosi della baseline (2) e review del trainer/monitor (3-4). PROCEDI!». Diagnosi di supportoClaude
60561493, codice977e767, conclusa in48s: entrambe le inizializzazioni ritrovate esattamente con ab_replica/sklearn_first; unico tensore
discordante nelle altre varianti tokenizer.fe_gain, nessun parametro. Risultato aggregato in results/passo6/diag_initial_20261014.json.

La ricostruzione recupera dal riferimento protetto per hash il SOLO guadagno fisso non appreso del tokenizer, oltre ai quattro buffer
keep gia' previsti. Ogni parametro da seed, goldenSHA completi invariati, nessun caricamento di state_dict appreso o tolleranza.
38testCPUpertinentiPASS; nuovo gate reale ancora da osservare. Questa e' una correzione della ricostruzione tecnica, non dell'esperimento.

Prosecuzione del controllo gia' autorizzato con checkout/output/IDr3 nuovi dopo preflight: 2extractP1cap28min, 2extractP2cap60min,
1GPU/8CPU/100GiB ciascuno; 2groupCPUcap3h55, 8CPU/30GiB, quattro probe identici e timeout2h/seme. afterok come da protocollo.
MassimoGPUr3=2,933333h; sommando r1+r2(119s) E diagnosticaClaude(48s): **2,979722GPUh entro3autorizzate**. GroupCPU62,6667coreh,
checkoutCPU2x10min=0,3333coreh, piccoli consuntivi precedenti: complessivo<64coreh. Nessuna submitr3 prima di questo commit.
Riferimento operativo docs/step1_random_downstream.md; nessun nuovo training, budget aggiuntivo, soglia o cambio delle sonde.

### Punti 2-4: avvio r3 e conclusione della review (GPT, 14/10/2026)

Baseline r3 accodata dal codice7863c5d: checkout60562932, extract60562964/60562980/60562990/60563026, gruppiCPU60563048/60563070
(confronto nell'ultimo). Ops ha verificato tutti gli export/risorse/dipendenze/commit e il passaggio REALE dello SHA iniziale esatto
per entrambi i semi e task; feature e sonde ancora in corso/attesa allo snapshot. Nessun incremento di cap: GPUcumulativo2,979722<3,
CPUseriali gruppi62,6667coreh+checkout0,003889effettive+piccoli overheadpassati<64. MAIN non aggiornato, solo nuovo refdeploy.

Review consegnata in docs/reviews/trainer_monitor_20261014.md, con quattro riproduzioni CPU persistenti e review indipendente.
Quattro P2 da correggere a Claude: protezione seed/ricetta alla ripresa (preesistente); verifica chiavi del forward prima della
backward_loss/flag comuni; protezione effettiva del pannello fisso da scale/input diversi; denominatori validi delle metriche opzionali.
Non e' un errore dimostrato della scala DDP: test gloo della testa opzionale/no_syncPASS; riprese con ricetta invariata non invalidate.
Correzioni sul branch di Claude, niente riscrittura dei suoi SHA o merge non revisionato. I due P2 del monitor vanno risolti prima di
usarlo per il verdetto; v2 di366ab11 resta bozza, delta/k/cadenza non adottati. CPUmonitor5PASS, worker escluso e prove degli agenti
interrotte su quel caso; non fullsuite ne' confermaCUDA/NCCL/bf16. Endpoint/soglie2A/2C e verdetto storico invariati.

### Punto 1 chiuso: secondario rami contro statistiche fisse (GPT, 15/10/2026)

Ricalcolato sul Mac il secondario GIA' previsto dal2C: RPW (raw+rami allenati) meno MS (raw+statistiche fisse2A), teacherB_seed0/B_seed1,
report e sonde invariati. Otto report e due comparison copiati byte-identici e verificatiSHA256 controLeonardo; replay dal codicec1c7a97,
compare_reports ufficiale identico ai dueSHA di campagna,1000bootstrap/seed20261009. Output nuovo results/passo6/step2c_vs_step2a_20261015.json;
nessuna sovrascrittura dei comparison storici o riapertura delle cacheNPZ. Accoppiamento ingressi/righe/split/teacher verificato dai report,
conteggi per soggetto/classe ricontrollati dall'estimatore. Nessun job/0GPU-ora aggiuntive.

| Task | Delta RPW-MS, punti | Intervallo95%, punti | Due teacher positivi | Utile secondo +1punto |
|---|---|---|---|---|
| P1 | +0,708805 | [+0,309728; +1,257906] | si' | NO: stima sotto1 |
| P2 | +0,677052 | [-0,823076; +2,177180], descrittivo | si' | NO: stima sotto1 e intervallo include0 |

Secondario **NEGATIVO**, entrambi non dannosi secondo la guardia puntuale>=-1punto. P1 ha un intervallo positivo, ma non raggiunge la
soglia pratica: non significa identita' delle rappresentazioni ne' assenza di ogni guadagno. P2 resta descrittivo, due teacher e confronti
multipli senza correzione. Primari2A/2C invariati negativi: nessuna adozione/training congiunto con l'obiettivo attuale o cambio delle soglie.

### Punto 2 chiuso: controllo dell'encoder casuale r3 (GPT, 15/10/2026)

Confronto finale step1_random_downstream_4000_v1, source7863c5da34bfa4c261acd41fff742707d2a45f6d, goldenSHA completi invariati per
semi0/1. Encoder iniziali ricostruiti senza pesi appresi; unico recupero aggiuntivo il buffer IMMUTABILE fe_gain storico, gia' documentato
e protetto dall'hash esatto. Stesse sonde, finestre, split, scaler train-only e confronto accoppiato del protocollo; nessuna nuova soglia.
Tutti i7job COMPLETED0:0; ultimo60563070 concluso15/10 02:39:45CEST, zeroSTOP. Finale in
results/passo6/step1_random_downstream_20261015.json, byte-identico al remotoSHA256
dd1f83ef474a24ea4f01bee844071d219e046ae840b106f721b49584183bb119.

| Famiglia | P1 allenato-casuale, punti (IC95%) | P2 allenato-casuale, punti (IC descrittivo95%) |
|---|---|---|
| A | +1,91900 [+0,83175; +3,20508] | +6,68940 [+4,00294; +9,37585] |
| B | +2,30152 [+1,08155; +3,60891] | +7,51249 [+4,31532; +10,70966] |

Casuale: P1 15,95837%, P2 73,09354%; B allenato:18,25989%/80,60603%. Delta aggregati positivi in entrambi i semi, ma DB3 resta
molto debole (B2,58827%, casuale3,57211%; delta-0,98383punti con intervallo che include0). Il pilot apprende rappresentazioni utili
rispetto all'inizializzazione, non prova che la solaJEPA causi il guadagno: A eB hanno entrambi le ancore. Due inizializzazioni,
incertezza condizionata e P2 descrittivo; nessuna proclamazione di default/guadagno su corpus completo.

ConsuntivoR3 verificato daOps:1,845278GPUh;23,17722corehCPUseriale incluso checkout (37,93944 contando ancheCPU nei jobGPU).
Con119s dei tentativi precedenti e48s del diagnosticoClaude, GPUcumulative conservative1,891667<3gia'autorizzate. Nessun nuovo job.

### Punto 5: norme dei gradienti pesati nell'A/B (GPT, 15/10/2026)

Analisi descrittiva dal codicec1c7a97 dei log storici69fa2da, export verificati daOps:4000record/run ma solo8diagnostiche/semeB;
A non le registra perche' keep_weight0. Output results/passo6/step1_weighted_gradients_20261015.json. Su tokenizer/identity/local/pool/backbone,
prima del clipping, le norme includono GIA' i pesi delle loss (ancore0,2, keep0,05):

| Rapporto di norme | Mediana B_seed0 | Mediana B_seed1 | Mediana pooled16batch |
|---|---|---|---|
| Ancore / JEPA | 8,74922 | 9,68113 | 8,96895 |
| Keep / JEPA | 1,27580 | 0,96037 | 1,14472 |

Le ancore superanoJEPA in tutti i16campioni (range2,53170..205,31593; massimo all'inizializzazione). Clipping della norma totale in
12/16diagnostiche, non tasso stimato su4000step. Tutti i valori finiti. Segnale concreto di forte pressione ausiliaria, NON dimostrazione
cheJEPA sia inutile o equivalenza fra norma e aggiornamentoAdamW: mancano direzioni/coseni e attribuzione causale; lo0,2% dei batch non
rappresenta l'intera traiettoria. Nessuna ricalibrazione automatica dei pesi. Il punto5 resta aperto per sonde temporali/fisiche, ablation
dell'obiettivo e nuovoSSL dei rami; niente training congiunto con la loss bocciata.46testCPUpertinentiPASS, non intera suite o provaGPU.

### Presa d'atto degli aggiornamenti Claude (GPT, 15/10/2026, fonte59b8c5b)

Letti fccdc1e/b104fb5: implementate le quattro correzioni della review trainer/monitor;10testCPU mirati unici PASS nella verificaGPT
e del reviewer.930PASS resta la suite completa dichiarata daClaude. Main ancora5030a19: nessun merge o deploy impliciti.

Gli artefatti del protocollo w1_gpu_dati_v1 riportano equivalenzaNCCL1vs4GPU e accumulo superata in float32/bf16 dal source d7e2509;
questo precede le nuove correzioni di ripresa/monitor. Attesa dati30M2,40%/2,81% contro2%: requisito NON soddisfatto, anche se100M/300M
sono sotto soglia. Nessuna deroga adottata qui. Tempi e memoria misurati permettono stime migliori, non chiudono piano/default/budget.
Consuntivo della tranche riportato daClaude5,51GPUh; infrastruttura5,87/25usate, residuo19,13. Nessun nuovo job o costo in questa lettura.

### Prossima tranche parallela, senza attesa reciproca (15/10/2026)

Simone confermaClaude gia' al lavoro sul prefetch del30M (punto6). GPT prende come base fissa di review/integrazione59b8c5b (punto3):
le modifiche successive del trainer restano un delta separato, senza editor concorrenti o riscrittura degliSHA. Regola del monitor4 e
protocolli di ablation/sonde5 possono essere definiti daGPT senza attendere il dataloader. Punti1/2chiusi;6richiede ancora launcher,
manifest annidati, passi da corpus, calibrazioneLR/batch/K, lista/configurazioni e budget firmati. Numerazione e ruoli nel registro comune.

t_data_s misura handoffCPU (next/_from_shared), prima di _prepare sul device: il prefetch deve sovrapporre quel lavoro, con stesso stream,
RNG/aggiornamenti e ripresa anche a batch prefetched in volo. Il tempo totale non si sostituisce spostando l'attesa fuori dal timer.
Bitwise nel percorso deterministico; confrontoGPU/bf16 con non determinismo del riferimento gia' noto e criteri prospettici.
Nessun nuovo job/cap o deroga al2% da questa pianificazione; firma scientifica e budget restano aSimone. SCRATCH entro22/10 aClaude.

### Candidato di integrazione e P2 residuo numerico (GPT, 15/10/2026)

Su richiesta di completare review/integrazione, preparato candidatoadd5359: merge della baseClaude59b8c5b con GPT5a25a5c, preservando
entrambi gli storici e tutte le voci dei registri, senza prefetch o draft non committati. Main canonico resta5030a19: non ancora integrato.

Review finale individua un P2 in run.py506-515: finitezza dei numeratori non garantisce quella della backward_loss dopo prodotti in
floatingpoint. Trainer reale con dati sintetici/keep calibrata e bias finito grande produce lossInf ma gradienti finiti, fa optimizer/EMA
e salva checkpoint senzaSTOP. Non e' un'anomalia osservata nei jobreali o una prova contro i risultati. Non occorrono nuovi jobGPU.

Due regressioni persistenti in docs/reviews/repro_backward_loss_overflow_20261015.py:2FAIL intenzionali sulla base;2PASS con guardia
proposta applicata soltanto in memoria. Trainer invariato; correzione all'autoreClaude come delta separato dal prefetch. Review e patch
in docs/reviews/integration_claude_20261015.md. Prima di aggiornaremain serve chiudere questo rilievo e rieseguire le regressioni.
SuiteCPU generale avviata, non ancora dichiarataPASS; nessun nuovo budget, soglia, deploy o push adottato.

Ripresa dopo interruzioneInternet: candidato e main verificati, nulla perso. Suite generale interrotta a623PASS/1SKIP/6FAIL/5ERROR:
dieci problemi dovuti ai permessi diIPC/localhost e un caso worker in attesa, rerun mirato fuori sandbox avviato. Il FAIL rimanente era
un testGPT che presumeva la pipeline storica nel nuovo checkout: test corretto per verificare snapshotstorico accettato e sorgenti
di encoder/pooling modificati rifiutati (21PASS); guardia di produzione e risultati congelati invariati. Nessun nuovo job/budget.
Main non aggiornato per il distinto P2 della lossInf: correzione minima e regressioni pronte perClaude, come da proprieta' del trainer.

Rerun finale dei sei file locali conIPC/worker/localhost consentiti: **65PASS in187,67s**. Baseline21PASS dopo aggiornamento dei soli
test; la suite generale completa del candidato non e' stata rieseguita fino alla fine e non viene dichiarataPASS. Restano2FAIL
intenzionali della regressione numerica sull'originale e2PASS con la guardia soltanto in memoria. Main5030a19 rimane invariato;
nessun run.py modificato, nessun push/job, nessuna modifica al prefetch diClaude. Punto3 ancora aperto fino alla correzione dell'autore.

### Conferma degli esiti prefetch e del prompt per il P2 (GPT, 15/10/2026)

Letti risultatiClaude85ac845 del source0230775: prova w1_prefetch_v1 PASS, auditinput200/200, attesa30M0,673% e0,649%, tempi medi
0,7057/1,3567s sotto i limiti predefiniti0,721544/1,400466. Consuntivo riportato1,84GPUh, infrastruttura7,71/25usate e17,29residue.
Non e' un nuovo audit raw dei log o una review completa del codice prefetch: quest'ultima resta una consegna separata.

Il trainer attuale diClaude conserva la guardia sui soli numeratori: il P2 di backward_lossInf non e' chiuso da questi job. Confermato
il prompt gia' fornito: fixCPU isolato su59b8c5b, regressioni senza monkeypatch, arresto coordinato; poi integrazione base+fix e review
del prefetch a parte. Nessuna richiesta di ripetere le proveGPU, modificare iperparametri o nuovi lanci. Main5030a19 resta invariato.

### Integrazione della base Claude e chiusura del P2 numerico (GPT, 15/10/2026)

Fix dell'autore a7c87b5 dalla base 59b8c5b verificato e integrato su main locale col merge a719c2c, conservando gli SHA e tutte le voci
dei registri. La struttura locale e' validata prima di costruire backward_loss; la finitezza della loss effettiva entra nello stop
coordinato prima del backward. Review indipendente senza altri rilievi concreti. Punto 3 chiuso per base + fix, non per il prefetch.

Verifica GPT: 48 test CPU PASS in159,49s (comprese le due regressioni originali senza patch in memoria e gli stop gloo su un solo rank
con accumulo1/2), piu'21 test baselinecasuale PASS in3,28s: totale69. Non una nuova suite completa ne' confermaCUDA/NCCL/bf16.
Nessun cambiamento a iperparametri, criteri, default o budget. Prefetch non incluso, review separata; nessun job, push o deploy.

**Prefetch del batch (15/10/2026, Simone: «vai con la 1, il prefetch sulla GPU, poi appena lanciata guarda cosa ha fatto gpt»; nota di
Simone: «quei 17 ms sono nell'handoff CPU dai worker. Il prefetch deve anticipare anche quel lavoro, non soltanto la copia sulla GPU»).**
Protocollo `w1_prefetch_v1` in `docs/protocollo_gpu_dati_finestra1.md`. Un thread anticipa di un batch la ricezione dai worker, la
ricostruzione dalla memoria condivisa, i micro-batch e la copia sul dispositivo (stream CUDA proprio); training bit per bit uguale sulla CPU
(test). Criteri congelati prima del lancio: verifica degli ingressi bit per bit su GPU per 200 aggiornamenti (30M, batch 64, 2 micro-batch);
attesa sui dati al 30M <= 2% (D6a) col batch 32 e 64; tempo totale (media di `t_step_s`, passi 51-600) <= 1,01 x quello di `w1gd`
(vincolo di GPT del 15/10: non spostare l'attesa fuori dal cronometro). 3 job da 4 GPU su dbg, **tetto 6 GPU-ora (48 ore locali)**, stima ~2, dal budget
«preparazione dell'infrastruttura» (usate 5,87 su 25).

**Esito della prova del prefetch `w1_prefetch_v1` (15/10/2026, criteri congelati prima del lancio; letti dopo la fine dei job).** Quattro
job COMPLETED 0:0 (checkout 60608218, `w1pf_audit` 60608329, `w1pf_data_b32` 60608334, `w1pf_data_b64` 60615343), codice `0230775`.
Riepilogo in `results/infra/w1pf_20261015.json`. **SUPERATA.**
- Verifica degli ingressi: 200 passi su 200, ingressi sul dispositivo uguali bit per bit al batch su 4 processi con 2 micro-batch.
- Attesa sui dati al 30M: **0,67%** col batch 32 (era 2,40%; mediana 4,4 ms contro 17) e **0,65%** col 64 (era 2,81%; mediana 8,5 ms
  contro 35), sotto il 2% di D6a.
- Tempo totale: media di `t_step_s` 0,7057 s (limite 0,7215; w1gd 0,7144) e 1,3567 s (limite 1,4005; w1gd 1,3866): -1,2% e -2,2%.
  Descrittivo, tempo di parete per passo (con avvio, valutazioni e checkpoint): +1,2% col 32, -2,1% col 64; stessi batch di w1gd a ogni
  passo, loss entro 1e-7 relativo al passo 1 e 6,6e-6 nei primi 10 passi, poi la deriva del bf16.
Conseguenze, come da protocollo: (B) superata col prefetch; i launcher della finestra 1 passano `--prefetch`; montaggi virtuali nel default
come da D15/D6a. **Costo reale (sacct):** 27,7 minuti di nodo = **1,84 GPU-ora (14,8 ore locali)**, entro il tetto di 6; budget
«preparazione dell'infrastruttura»: 5,87 + 1,84 = 7,71 usate su 25.

**Catene WSD della finestra 1: launcher e pianificatore (15/10/2026, Simone: «sì, vai coi launcher delle catene WSD»).** Dettagli in
`docs/catene_wsd_finestra1.md`. `scripts/slurm/train_ddp.sbatch` vale anche per i job lunghi (`--qos=normal --time=...` davanti allo script):
`BRANCH_FROM` per i rami del WSD (il primo job del ramo esce con 3 se il checkpoint del tronco non c'e' ancora), uscita immediata dei job di
scorta su un run completo. `scripts/wsd_chain.py`: dal piano JSON ai comandi di `submit_job.py` con le dipendenze e al preventivo, coi
controlli di D14 fatti dal codice del trainer. `--keep-checkpoints` accetta anche `:` (sbatch `--export` separa sulle virgole). Test CPU
`tests/cpu/test_wsd_chain.py`. **Collaudo proposto `w1_catena_v1`** (criteri congelati nel documento, prima di qualunque lancio): 30M, tronco
di 1.400 passi in 2 job da 15 minuti + 1 di scorta, ramo da 1.100 a 1.375 + 1 di scorta, QoS `normal`; **tetto 5 GPU-ora (40 ore locali)**,
stima ~1,8; dal budget «preparazione dell'infrastruttura» (usate 7,71 su 25). Da autorizzare.

### Review e integrazione separata del prefetch (GPT, 15/10/2026)

Accettato il delta Claude 8cc6cbf (prefetch0230775 + fix a7c87b5 via merge66b5a0f), con SHA conservati e registri uniti senza perdita
di voci. Due review indipendenti senza rilievi bloccanti; 58 test CPU PASS in376,99s e 6 controlli supplementari PASS su WSD con
prefetch e ripresa DDP, includendo optimizer/RNG: totale64. I due controlli DDP supplementari, inizialmente bloccati dai socket del
sandbox, passati con IPC consentito. Non una nuova fullsuite o provaGPU. Report in docs/reviews/integration_claude_20261015.md.

Punto6: eliminata l'attesa della review del prefetch; Claude puo' preparare launcher/cateneWSD/manifest dal main locale integrato.
Restano calibrazione e piano/configurazioni/budget della finestra1; nessun nuovo lancio o cap autorizzato. GPT passa ai protocolli4/5.
Sul punto5 e' esplicitato un confronto mancante: NeuroRVQ e WearUsFM entrambi congelati sullo stesso protocollo. La replica del
passo5 aveva gia' usato il FM preallenato con fine-tuning, e l'ancora aveva usato il tokenizer; non dimostrano superiorita' o inferiorita'
del riuso del backbone. Gap da specificare, non nuovo job/default adottato; nessuna modifica al verdetto sull'ancoraRVQ.

**Manifest annidati (D16) collegati al training e passi per epoca dal corpus (15/10/2026; Simone: «se non rompe niente tra voi 2 puoi
iniziare coi manifest annidati»).** Dettagli in `docs/manifest_annidati_d16.md`. `RunConfig.nested` / `--nested` (0,125 / 0,25 / 0,5) allena sui
soggetti del sottoinsieme firmato in `splits_v1.json`; `scripts/nested_manifest_report.py` verifica annidamento ed esclusione dei test sul
manifest congelato v1.1 (nessuna verifica fallita; `results/infra/nested_manifests_v1_1_20261015.json`). **Passi per epoca dal corpus: 24.528
col batch 32** (finestra media 3,77 s), non ~23.150: D = 4 epoche = 98.112 passi (+6% sui costi a D). Nei sottoinsiemi, con i pesi del
manifest completo rinormalizzati (la regola attuale del loader), classi entro mezzo punto da 20 / 75 / 5 e passaggi per unita' ~8 come nel
100%. **Da decidere (GPT, Simone):** la regola dei pesi nei sottoinsiemi (attuale; quote di unita' identiche; D9 rifatta sul sottoinsieme).

**Esito del collaudo delle catene WSD `w1_catena_v1` (15/10/2026, criteri congelati in `docs/catene_wsd_finestra1.md` prima del lancio;
letti dopo la fine della catena).** Codice `df4d2f5`, checkout 60623582; job 60623878, 60623935, 60623939, 60623942, 60623946, tutti
COMPLETED 0:0. Riepilogo in `results/infra/w1cat_20261015.json`. **SUPERATO.**
1. Tronco: il job 1 si ferma al limite di tempo al passo 990; il job 2 riprende dal 990 e chiude a 1.400 (`max_steps`); il job di scorta esce
   in 7 s con «gia' completo». Passi 1-1.400 senza buchi ne' duplicati, `lr` di ogni passo uguale allo schedule, presenti
   `checkpoint_step0001100.pt`, `checkpoint_predecay.pt` (1.120) e `checkpoint_postdecay.pt` (1.400).
2. Ramo: parte dopo il job 2 del tronco, `branch.json` con passo 1.100 e SHA-256 del checkpoint d'origine, passi 1.101-1.375, `lr` uguale
   allo schedule del ramo, job di scorta «gia' completo».
3. Nessuno STOP, nessun job FAILED o TIMEOUT.
Descrittivo: 0,70 s/passo; il primo passo dopo una ripresa aspetta ~26 s (ricostruzione a secco dello stream e avvio dei worker), ~65-80 s di
avvio per job: trascurabili su job da 24 h. **Costo reale (sacct):** 23,2 minuti di nodo = **1,55 GPU-ora (12,4 ore locali)**, entro il tetto
di 5; budget «preparazione dell'infrastruttura»: 7,71 + 1,55 = 9,26 usate su 25.

**Decisioni di Simone del 15/10/2026 (dopo il collaudo delle catene).** (1) Calendario della finestra 1: «secondo me puoi usare anche 8 nodi in
parallelo no ? io ti autorizzo se cineca lo consente» -> fino a 8 nodi in parallelo, entro i limiti dell'account (tetto di nodi ancora da
leggere). (2) $SCRATCH: «quando vuoi fai pure partire il refresh» -> giro del rinfresco autorizzato (`refresh_scratch.sbatch`, CPU seriale,
1 core, tetto 2 h = 2 core-ora, 0 GPU-ora; il giro del 01/10 durò 35 minuti). (3) Collaudo dei bracci della finestra 1: «aspettiamo un attimo
che forse c'e' qualcosa di scientifico da sistemare ancora» -> sospeso.

**Limiti SLURM dell'account e delle QoS (letti da Simone con `sacctmgr`, 15/10/2026; `results/infra/slurm_limits_20261015.txt`).** Account
`iscrb_wearusfm` (17 utenti): nessun limite di nodi, GPU o job sull'associazione. QoS `normal` (catene da 24 h su `boost_usr_prod`): nessun
limite per utente o per account. `boost_qos_lprod`: 4 giorni, al massimo 8 nodi per job e **8 nodi per l'intero account** (MaxTRESPA): e' il
`MaxNodePerAccount` del 14/10. `boost_qos_dbg`: 30 minuti, 2 job sottomessi per utente, 8 nodi per utente, 128 nodi in tutto. **Quindi gli 8
nodi in parallelo autorizzati da Simone sono possibili sulla QoS `normal`**; il vincolo pratico e' la coda (priorita' dell'account, consumata
anche dagli altri utenti: 2.774 GPU-ora in 30 ore il 14/10) e il budget dell'account, non un tetto di SLURM.

**Documentazione CINECA sui limiti della Booster (fact-checker, 15/10/2026; citazioni raccolte, non ancora riverificate con
`check_quotes.py`).** https://docs.hpc.cineca.it/hpc/leonardo.html, tabella Booster: `normal` «Max = 64 nodes» per job, «24:00:00», nessun
limite per utente; `boost_qos_lprod` «Max = 8 nodes», «4-00:00:00», «Max resources per Project Account.»; `boost_qos_dbg` «Max = 8 nodes»,
«00:30:00», «Max 2 job running and/or pending per User Account.»: coerente con `sacctmgr`. https://docs.hpc.cineca.it/hpc/hpc_intro.html,
«Budget Linearization»: «To each Project Account is assigned a monthly quota (MQ) calculated as:» «MQ = TB/NM»; «As the budget is consumed,
submitted jobs progressively lose priority until the monthly quota is exhausted.» **Conseguenza per la finestra 1:** gli 8 nodi sono
ammessi; la velocita' della coda dipende dalla quota mensile del progetto, condivisa da 17 utenti (il 14/10 un altro utente ha consumato
2.655 GPU-ora in 30 ore). Serve il saldo del progetto (`saldo -b`) prima di firmare il calendario.

**Saldo del progetto (`saldo -b`, letto da Simone il 15/10/2026; riga del progetto in `results/infra/saldo_20261015.txt`; 1 GPU-ora = 8 ore
locali sulla Booster, come nei consuntivi).** IscrB_WearUsFM, 07/01/2026-07/01/2027: budget 2.000.000 ore locali = **250.000 GPU-ora**;
consumate 1.257.249 = 157.156 GPU-ora (62,9%, col 77% del tempo passato); **restano 92.844 GPU-ora per 2,8 mesi**. Quota mensile 164.383 ore
locali = **20.548 GPU-ora**; consumate in ottobre al 15/10: 5.548 (27%). **Lettura per la finestra 1:** il budget restante basta (finestra 1
~5.100-5.700 GPU-ora, poi la ladder ~13.000), ma la finestra 1 vale il 25-28% della quota di novembre, condivisa da 17 utenti: lanciata all'inizio
del mese parte con la priorita' piena; se gli altri consumano come il 14/10 (~2.200 GPU-ora al giorno) la quota di novembre finisce in ~9 giorni
e le catene ancora in corso perdono priorita'. Utile un accordo con gli altri utenti del progetto per la prima meta' di novembre.

**Rinfresco di $SCRATCH del 15/10/2026 (al posto del giro del 22/10; Simone: «quando vuoi fai pure partire il refresh»).** Job 60643323
(`refresh_scratch_20261015`), COMPLETED 0:0 in 15 min 9 s (15:20-15:35): 32 cartelle di `data/raw`, `data/processed`, `wearusfm_runs`,
`external`, 151.833 file toccati, nessun file piu' vecchio di 30 giorni prima del giro (il piu' vecchio era del 01/10, il giro precedente),
**0 cartelle con problemi, nessun file piu' vecchio di un giorno dopo**. Costo: 15 minuti di un core. Prossimo giro del calendario firmato:
12/11/2026 (28 giorni dopo, entro i 40 del purge).

**Manifest annidati come file congelati (15/10/2026, accordo GPT e Claude riferito da Simone).** Regola: D9 rifatta sulle ore del
sottoinsieme; il residuo di una classe con tutte le unita' al tetto va alle unita' sotto il tetto delle altre classi, in proporzione alla
quota, senza superare il tetto; controlli sulla distribuzione campionata. Costruzione locale dal v1.1: quote 20 / 75 / 5 esatte al 12,5% e al
50%, al 25% classe C 4,97% (residuo 0,0254% ad A e B), passaggi massimi 8,000000 ovunque; pesi del 100% riprodotti bit per bit. Il campo
`hash` del v1.1 e' rimasto quello di v1 (contenuto: 52549ef8...), segnalato. Dettagli in `docs/manifest_annidati_d16.md`. Prossimi: review di
GPT, poi costruzione sul cluster (job CPU da autorizzare).

### Lettura della consegna WSD/manifest e priorita' successive (GPT, 15/10/2026, fonteClaude225b380)

Artefatto results/infra/w1cat_20261015.json sul ramoClaude: collaudo w1_catena_v1 PASS sui criteri congelati, codice df4d2f5;
tronco990->1400, ramo1100->1375, scorte uscite gia' complete. Costo registrato1,55GPUh, infrastruttura9,26/25usate e15,74residue.
La continuazione2D del planner e il collegamento nested sono consegne successive da review; nessuna integrazione di quel delta qui.

La stima aggiornata della finestra1 e'5.000-5.600GPUh, non budget firmato. Restano bracci non implementati e catena di valutazione,
oltre a regola di stabilita', protocollo scientifico, calibrazione, tetto nodi e SCRATCH22/10. Proposta: review del delta in parallelo
alla definizione4/5, non nuovi training prima di protocollo/costi. Sono proposte, non nuove autorizzazioni.

Pesi nested ancora da firmare: GPT preferisce valutare D9 riapplicata ai sottoinsiemi (quote20/75/5, sqrtore, tetto8), non adottata.
Il report mostra B74,133% al12,5% (-0,867punti, non mezzo punto) e massimo8,031passaggi al50%. Il conteggio delle epoche deve inoltre
replicare l'esclusione/rinormalizzazione del sampler per62sessioni senza finestre: scarto~0,04% tenendo lo stesso budget di ore,
non un nuovo valore finale di passi approvato. Distinguere ore nominali/campionabili e congelare una sola regola di arrotondamento.

Nessuna soglia«1% di loss» adottata dal confronto bf16; norme dei gradienti non attribuiscono causalmente l'apprendimento alle ancore.
Queste cautele non cambiano verdetti, soglie pregresse o architettura; nessun lancio, nuovo budget o decisione scientifica firmata.

### Parere sulla proposta nested D9 e calendario a 8 nodi (GPT, 15/10/2026, fonteClaude ecdfc50)

Favorevole a manifest congelati per frazione con D9 riapplicata, stessi ingressi e provenienza completa; non ancora adozione delle
quote finali. Al25% il residuoC rende la massa totale0,99974648745: la normalizzazione globale porta il cap atteso a8,00203.
Lo script stampa quote normalizzate ma passaggi grezzi: servono gestione esplicita del residuo e verifica sul sampler finale, comprese
le sessioni senza finestre. Non e' una tolleranza o una deroga firmata al tetto8; riportare quote effettivamente realizzabili.

Calendario8nodi6,8-7,6giorni e costo~5.100-5.700GPUh restano scenario/stima: non comprendono attese di coda, cap account e disponibilita'
non verificati. Riferimento30M a due semi gia' nella bozza: nessun run aggiuntivo deciso. Coseni e sonde intermedie sono proposte utili,
non sostituti delle ablation. Nessuna nuova soglia, campagna, autorizzazione di spesa o modifica del default adottata qui.

### Preparazione della prossima tranche e assegnazioni (15/10/2026)

Simone: «decidiamo e preparami il prompt per Claude e il tuo». Specifica concreta e prompt in
`docs/accordo_operativo_finestra1_20261015.md`: Claude implementa manifest/planner/orchestrazione; GPT mantiene loss, protocolli,
sonde/launcher dell'harness, criteri e review. Per W1 un solo submitter, Claude, dai piani revisionati da GPT.

Regola proposta per rendere fattibile il nested al25%: tentare20/75/5 con D9; se C satura, conservare A20% e assegnare il residuo a B
rispettando tutti i cap. Se A o B non hanno capacita', fermare la costruzione. Ore delle sessioni campionabili, righe escluse a peso0,
quote/passaggi finali verificati nel sampler e provenienza versionata; floor unico dell'endpoint. Estensione esplicita da verificare nel
report prima di congelare i nuovi derivati; nessuna modifica retroattiva a v1.1 o firma attribuita a Simone su quote non ancora misurate.

Griglia da preparare/preventivare: batch32; 30M LR1e-4/3e-4/1e-3 a1epoca seme0, due finalisti ripetuti seme1; due LR100M a0,5epoche;
conferma300M a0,25. Otto run, circa156,7GPUh di training con3% di margine alle misure attuali, valutazione esclusa. Budget W1 da chiudere,
non voce del residuo infrastruttura. Fold interni P1 e selezione da protocollo GPT prima del lancio; niente test ufficiale per scegliere.

Aggiornamento dai logClaude fba794f: SCRATCH60643323 chiuso,151.833file,0problemi, prossimo12/11; limiti sacctmgr compatibili con8nodi normal.
Nessun nuovo accesso cluster, lancio o budget approvato in questa attivita'. I nuovi prompt avviano preparazione e verifiche locali.

**Aggiornamento sul manifest consegnato95f441b (15/10/2026).** Su richiesta di Simone il promptClaude e' stato ristretto al lavoro residuo:
conservare builder/file/report/guardie esistenti; allineare residuo C->B, ore campionabili, derivato100%, floor e verifiche finali alla
specifica c6f59b9. La ripartizione A+B della consegna e' stata sviluppata in parallelo; nessuna nuova decisione scientifica o variazione
del tetto. Report finali da rigenerare dal commit corretto; review statica qui, non nuova suite o costruzione sul cluster.

**Manifest derivati allineati all'accordo del 15/10/2026 (codice `56cea75`, rapporti dal commit pulito).** Regola dell'accordo
(`pretraining_manifest.DERIVED_RULE`). Derivati 12,5 / 25 / 50 / 100% dal v1.1 e da window_s congelato: dopo il sampler classi 20 / 75 / 5
esatte, al 25% 20 / 75,0254 / 4,9746 (residuo di C a B), passaggi massimi 8,0000, endpoint a 4 epoche col batch 32: 13.625 / 25.628 /
49.926 / 98.094 passi (floor sul totale). Il v1.1 campiona 20,008 / 74,990 / 5,002 perche' pesa anche le 62 sessioni senza finestre (0,47 h).
Dettagli e preventivo della costruzione sul cluster (0 GPU-ora, ~0,7 core-ora) in `docs/manifest_annidati_d16.md`; da lanciare dopo la
review di GPT e col via di Simone.

**Calibrazione C1-C4: pianificatore e preventivo (15/10/2026; accordo GPT-Claude, sezione «Calibrazione»).** `scripts/calibration_plan.py`:
finalisti come decisioni esplicite di GPT (fasi bloccate senza la decisione, conferma fallita che ferma C3/C4, nessuna scelta dalla training
loss), passi col floor dal derivato al 100%, warmup di D (1.962 passi) in tutte le fasi, decay sul 20% di ogni endpoint; `--lr` aggiunto al
comando di training. Preventivo: **solo training 152,1 GPU-ora; attese ~160, tetto 208** (8 run, un job ciascuno), esclusi estrazioni,
sonde e selezione (GPT). Dettagli in `docs/calibrazione_c1_c4_piano.md`. Nessun lancio: servono i criteri di GPT e il via di Simone.

**Costi e calendario aggiornati dopo l'accordo del 15/10/2026.** Finestra 1 ~5.070-5.700 GPU-ora (calibrazione 157 compresa); con 8 nodi sulla
QoS normal 7,2-9,8 giorni a seconda dell'attesa in coda per job (0-3 h), con 4 nodi 13,6-18,1; attese delle decisioni della calibrazione a
parte. Tabella in `docs/finestra1_fattibilita.md`, simulazione in `scripts/finestra1_calendario.py`.

**Costruzione dei manifest derivati sul cluster autorizzata senza attendere la review di GPT (Simone, 15/10/2026: «puoi partire con la cosa del
manifest senza aspettare gpt»).** Preventivo di `docs/manifest_annidati_d16.md`: checkout di campagna e `build_nested_manifests.sbatch` su
`lrd_all_serial`, 1 core, **0 GPU-ora, tetto ~0,7 core-ora**; uscita `$WORK/wearusfm_runs/results/passo6/derived_v1_1/`, v1.1 intatto. La review
di GPT resta da fare prima dell'uso dei derivati nei run; esito confrontato con la costruzione locale (hash di contenuto).

**Manifest derivati costruiti sul cluster (15/10/2026).** Checkout 60661456 e costruzione 60661475 COMPLETED 0:0 da `11f868b` (checkout pulito),
verifiche superate, hash di contenuto identici alla costruzione locale; file in `$WORK/wearusfm_runs/results/passo6/derived_v1_1/` (SHA dei
file: 0,125 a622978f..., 0,25 90ce5564..., 0,5 5968179b..., 1 6d6d8167...). Consuntivo: 48 s di CPU in due job, 0 GPU-ora. Uso nei run dopo
la review di GPT.

**Ultrareview della tranche (15/10/2026, lanciata da Simone contro `review-base-w1` = main e02ac74).** 10 rilievi, tutti verificati e corretti
con test (dettaglio nel registro condiviso): i tre di gravita' normale erano nella catena di valutazione (guardia col python di sistema,
afterok su job gia' dimenticati da SLURM, guardie condivise fra requisiti diversi); gli altri: limiti di coda senza conteggi reali, loader
rigido negli strumenti, stati del ledger non previsti, argomenti liberi che alteravano lo schedule nel pianificatore, etichette dei LR con
perdita, scrittura non atomica del costruttore, costanti duplicate. Nessun job da rifare: catena e guardie non erano ancora state usate.

**Collaudo della catena di valutazione sul cluster `w1_valutazione_v1` (15/10/2026; Simone: «va bene procedi»).** Criteri congelati in
`docs/catena_valutazione_finestra1.md` prima del lancio: guardia senza afterok su un training finito da ore e catena fittizia
estrazione -> sonde -> confronto in ordine; guardia con SHA sbagliato che esce con 3 e blocca il nodo a valle senza lasciarlo in coda;
nessun doppione rieseguendo. Checkpoint veri di `w1cat`, nodi fittizi (`scripts/slurm/chain_dummy.sbatch`). **0 GPU-ora, tetto ~1 core-ora.**

**Esito del collaudo della catena di valutazione `w1_valutazione_v1` (15/10/2026, criteri congelati prima del lancio): SUPERATO.** Guardia
senza afterok su un training finito da ore (sacct), COMPLETED col rapporto `ok`; catena fittizia estrazione -> due sonde -> confronto
COMPLETED nell'ordine delle dipendenze; guardia con SHA sbagliato FAILED 3:0 e nodo a valle bloccato senza mai entrare in coda; nessun doppione
rieseguendo. Job 60664574-60664599 (7 CPU, 40 s, 0 GPU-ora). `results/infra/w1val_20261015.json`, `docs/catena_valutazione_finestra1.md`.

**Calibrazione C1-C4: protocollo di ammissibilita' e selezione PROPOSTO (15/10/2026), da firmare prima di C1.** Simone: «come al solito chat
gpt si è mangiato tutti i token , riessci tu a fare i task per C1 C4 ?», «mi raccomando se trovi sistema tutto». Proposta di Claude al posto
di GPT (accordo del 15/10, punto 5), in `docs/calibrazione_c1_c4_selezione.md`: validation interna sui soli soggetti di pretraining P1
(DB2/DB3/DB6, 4 fold di soggetti, seme 20261015, sonda e feature della P1, S fuori fold coi pesi della P1); ammissibilita' A1 completo,
A2 numerico, A3 non-collasso (soglie firmate, decisione 16), **A4 pannello fisso ogni 500 passi: mai 3 valutazioni consecutive oltre 1,10
volte il minimo, L finale <= 0,9 L_0 (soglie nuove, da firmare)**; selezione per S con confronto accoppiato, **parita' entro 1 SE al LR piu'
basso (da firmare)**; C3 con un solo LR ammissibile: continuare segnalandolo oppure fermarsi (**da decidere**). Preventivo C1-C4 ~163
GPU-ora attese, tetto 214,5 (1.716 ore locali), CPU ~199 core-ora; C1 ~61 / 79,5. Il budget della finestra 1 non e' ancora fissato: serve
una decisione prima di C1. Nessun job lanciato.

**Calibrazione C1-C4: C3 con un solo LR ammissibile = stop (Simone, 16/10/2026).** Simone: «in C3 con un solo LR ammissibile ci si ferma
(come dice l'accordo), salvo diversa firma mia». Chiude il punto «da decidere» della proposta del 15/10: in C3 meno di due LR ammissibili al
100M (uno o nessuno) fermano il passaggio a C4, come l'accordo del 15/10 («Se mancano due candidati ammissibili [...] interrompere il passaggio
alla fase successiva»). `scripts/calibration_select.py` e `docs/calibrazione_c1_c4_selezione.md` aggiornati; restano da firmare A4 e la regola
di parita'. Nella stessa correzione (nessuna soglia cambiata): A1 confronta il checkpoint della sonda con quello del run come file e non come
stringa (una barra finale in `run_root` avrebbe reso inammissibili tutti i run).

**Calibrazione C1-C4: A4 e regola di parita' FIRMATE (Simone, 16/10/2026).** Simone: «firmo A4 e parità, prepara il preventivo del collaudo
su w1cat». Congelate prima di qualunque run di C1 (`docs/calibrazione_c1_c4_selezione.md`, `scripts/calibration_select.py`): **A4** pannello
fisso (8 x 32 finestre) ogni 500 passi dal passo 0 senza buchi, run inammissibile con 3 valutazioni consecutive in cui la loss totale supera
1,10 volte il minimo raggiunto, o con loss all'ultima valutazione oltre 0,9 volte quella del passo 0; **parita'**: differenza accoppiata di S
entro 1 errore standard -> vince il LR piu' basso. Insieme allo stop di C3 con un solo ammissibile (16/10) il protocollo di selezione e'
completo. Resta da decidere il budget della finestra 1 prima del lancio di C1. Un cambio di queste soglie dopo C1 e' una decisione nuova.

**Collaudo della catena reale `w1_selezione_v1`: protocollo e preventivo (16/10/2026), da autorizzare.** Simone: «firmo A4 e parità,
prepara il preventivo del collaudo su w1cat». Estrazione, sonda a fold e selezione (fase `collaudo`) sui post-decay `w1cat-T` e `w1cat-B`;
criteri congelati in `docs/calibrazione_c1_c4_selezione.md` prima del lancio (esito atteso della selezione: nessuno ammissibile, pannello
assente in `w1cat`). **GPU attese ~0,4 GPU-ora (3,2 ore locali), tetto 1,0 (8 ore locali)** sul budget infrastruttura (residuo 15,74 su 25,
dopo il tetto almeno 14,74); CPU attese ~20 core-ora, tetto ~51. Nessun job lanciato.

**Collaudo `w1_selezione_v1` autorizzato e lanciato (16/10/2026).** Simone: «procedi con tutti e 3 i punti e vai, lancia il collaudo su
w1cat». Deploy `deploy/w1sel-20261016` (7cf4f08), checkout 60688786 (CPU, 12 s), catena 60688822-60688830 (7 job; tetto 1,0 GPU-ora sul
budget infrastruttura, residuo prima del collaudo 15,74 su 25). Consuntivo all'esito.

**Ruoli dal 16/10/2026: la parte scientifica passa a Claude, GPT fa le verifiche (Simone).** Simone: «sì a questo punto esautoriamo gpt
dalla parte scientifica perché è troppo lento e gli facciamo fare le verifiche, modifica i documenti necessari per questa procedura».
Claude prende protocolli e bracci delle ablation, sonde, harness, target e loss (`training/jepa.py`, `model/anchor_targets.py`,
`model/spectral_targets.py`), regola di stabilita', contratto di valutazione e lista dei job della finestra 1; GPT fa review,
riesecuzione dei test e controlli indipendenti, senza modificare i file di Claude. Invariati: GPT integratore su `main`, firme e lanci di
Simone. *Proposta di Claude, da confermare:* la verifica di GPT non blocca i lanci; prima del lancio passa il subagent `critico`.
Aggiornati CLAUDE.md, l'accordo del 15/10 (sezione del 16/10, con il prompt per GPT) e il registro.

**Verifiche e revisore prima del lancio (Simone, 16/10/2026).** Simone: «1) si non bl,occa 2) passiamo il revisore suila parte scientifica e algorirmica, nei test possiam fidarci delle verifiche.» Conferma la proposta registrata nella voce sui ruoli: (1) la
verifica di GPT non blocca i lanci; (2) il subagent `critico` rivede prima del lancio la parte scientifica e algoritmica (protocolli,
criteri, loss, target, sonde, selezione). *Interpretazione di Claude:* «nei test» = i collaudi e i test (CPU, launcher, criteri congelati
dei collaudi), per i quali bastano le verifiche gia' previste senza il `critico`. CLAUDE.md e l'accordo del 15/10 aggiornati.

**Manifest derivati: review del `critico` prima del primo uso nei run (Simone, 16/10/2026).** Simone: «firmo 2 e 8, fai rivedere i
manifest al critico». (2) I manifest derivati (frazioni 12,5/25/50/100%, costruiti sul cluster dal job 60661475, codice 11f868b), segnati
nel registro «da rivedere da GPT prima dell'uso nei run», li rivede il subagent `critico` al posto di GPT, prima di C1; l'esito si
registra qui. (8) Conferma la regola gia' registrata nella voce «Verifiche e revisore prima del lancio» dello stesso giorno: `critico`
prima del lancio sulla parte scientifica e algoritmica, verifica di GPT non bloccante.

**Budget della finestra 1 e della calibrazione FIRMATI (Simone, 16/10/2026).** Simone: «firmo 220 per la calibrazione e 6.000 per la
finestra 1».
- **Finestra 1: tetto complessivo 6.000 GPU-ora (48.000 ore locali), calibrazione compresa.** Base: stima del calendario del 15/10
  5.070-5.700 GPU-ora; piano operativo ~6.000-7.000 del FM a novembre; saldo dell'account al 15/10 ~92.844 GPU-ora, quota mensile ~20.548
  condivisa con gli altri utenti. E' un tetto, non un'autorizzazione: ogni tranche arriva con protocollo, preventivo e via di Simone, e il
  superamento del tetto (o di quello di una voce) richiede una decisione registrata. Ripartizione del resto (~5.780) con la lista dei job,
  dopo le firme sui bracci (`docs/bracci_obiettivo_finestra1.md`).
- **Calibrazione C1-C4: 220 GPU-ora (1.760 ore locali)**, voce della finestra 1. Preventivo: ~163 attese, tetto 214,5 (training 160,2 /
  208,0; valutazione 2,6 / 6,5), piu' il margine per i tetti delle valutazioni a 100M e 300M da rivedere dopo il collaudo `w1_selezione_v1`.
  Consuntivo per fase qui sotto. Separata dal budget «preparazione dell'infrastruttura» (25 GPU-ora, residuo 15,74 prima del collaudo).
- Consumato sulla finestra 1 finora: 0 GPU-ora.

**Manifest derivati: review del `critico` (16/10/2026) DA RIVEDERE, rilievi corretti.** Bloccante: il derivato al 25% non era
deterministico (ordine di un set in `allocate`, PYTHONHASHSEED): ora ordine fisso, scelto fra quelli fissi perche' riproduce i quattro file
costruiti sul cluster il 15/10, che restano validi (ricostruiti con sei semi: 80a214fd / 13d3dce8 / 52ec61f0 / 167923c3). Minori: verifiche
dopo il sampler complete (A, B + residuo, C - residuo, somma dei pesi, durate); il training rifiuta un derivato senza sampler pesato nel
tempo, con durate o configurazione delle finestre diverse da quelle della costruzione, o con un file diverso da quello del piano
(`--manifest-sha256`); test negativi; rapporto del cluster in pianificatore e documenti; identita' di un derivato documentata
(`docs/manifest_annidati_d16.md`). Nessuna soglia ne' regola dei pesi cambiata. Suite 1166 PASS. Seconda review del `critico` a seguire.

**Manifest derivati: review del `critico` CHIUSA, APPROVATO (16/10/2026).** Seconda review su cfd7d47: DA RIVEDERE (determinismo
confermato con 14 semi e hash uguali al cluster; pesi e training non derivato invariati; comandi di C1 accettati), cinque rilievi su test e
dettagli corretti in 8872d5c (test che isola l'annidamento, pianificatore legato al rapporto costruito sul cluster, rifiuto di un derivato
filtrato per dataset, docstring, quattro casi negativi). Terza review su 8872d5c: **APPROVATO**, con due rilievi minori chiusi subito (test
anche per la meta' «sessioni» dell'annidamento; questa registrazione). I file del cluster in `derived_v1_1/` restano quelli del 15/10; il
training ne verifica identita' e durate all'avvio. Per C1 i manifest sono pronti.

**Esito del collaudo `w1_selezione_v1` (16/10/2026, criteri congelati prima del lancio): SUPERATO.** Guardie, estrazioni (17:51 e 16:34),
sonde (1 h 44 e 1 h 51) e selezione COMPLETED 0:0; solo soggetti di pretraining; decisione «nessuno ammissibile» per il pannello assente,
come atteso; confronto accoppiato calcolato sui dati veri; nessun doppione rieseguendo. Job 60688786-60688830. **Consuntivo 0,57 GPU-ora**
sul budget «preparazione dell'infrastruttura»: 9,83 usate su 25, residuo 15,17. Stime delle valutazioni di C1 aggiornate (estrazione 0,3 h,
sonda 1,8 h a 8 core; sonde in fila per il limite di core per utente). `results/infra/w1sel_20261016.json`.

**Calibrazione C1 AUTORIZZATA (Simone, 16/10/2026).** Simone: «se è tutto verificato e a posto, via per C1, sonde sulla CPU». Verificato
prima del lancio: protocollo e regole firmati (A4, parita', stop di C3), review del `critico` su calibrazione, bracci e manifest derivati
(APPROVATO), collaudo `w1_selezione_v1` superato, budget firmati. Codice `72cf24abec5b8332011d83ce2b2fc7ae6bcc9cec` (deploy
`deploy/w1cal-20261016`, checkout `wearusfm_campaigns/w1cal-20261016`). 3 run da 24.523 passi (LR 1e-4, 3e-4, 1e-3, seme 0, 30M, un nodo
4 GPU, QoS normal, un job da 6 h 30 ciascuno), poi la catena di valutazione con le sonde sulla CPU seriale. **Preventivo C1: ~61 GPU-ora
attese, tetto 79,5 (636 ore locali)**, sul budget della calibrazione (220, finora 0 usate): dopo il tetto ne restano almeno 140,5.

**Bracci dell'obiettivo (punto 5) FIRMATI (Simone, 16/10/2026).** Simone: «firmo i bracci come proposti». Nella forma proposta in chat
lo stesso giorno (dettaglio in `docs/bracci_obiettivo_finestra1.md`, «Firma»): bracci A, JEPA-only, ancore-only (ancore attuali), keep
separata (opzione 1, A' a peso 0 e B a 0,05, calibrazione della keep ricalcolata sul corpus intero); 30M, 4 epoche, 2 semi, LR di 30M da
C1-C2 (con due confermati: S migliore al seme 1, parita' al piu' basso); un braccio batte A con almeno +1 punto su P1 e P2 in media sui due
semi; braccio fermato dall'allarme = «collassato», non batte A; allarme di collasso (0,05) valido in tutti i bracci; letture «al LR di A».
Via al pianificatore dei bracci e al piano di valutazione (lavoro locale). Nessun lancio dei bracci autorizzato da questa firma.

**Bracci dell'obiettivo: regola di lettura e scelta del default corrette (Simone, 16/10/2026).** Simone: «iniziamo a risolvere 1 e 2,
apporvo le soluzioni proposte, se riesci firmo e implementa eventuali corrzioni dove non gi afatte». Dalla critica del punto 5 (`critico`,
589be8c, `docs/reviews/punto5_pianificatore_p1p2_critica_20261016.md`). (1) Su P1/P2 ufficiali un braccio batte A solo con la regola del
2A/2C: confronto accoppiato per seme (`compare_reports`), per P1 e per P2 delta medio >= +0,01, limite inferiore dell'IC al 95% > 0, delta
> 0 in entrambi i semi; sostituisce il «+1 punto sulla media dei due semi» della firma dei bracci. (2) Il default non si sceglie sui
soggetti di test: si sceglie con la validation interna S (P1: `calibration_probe.py`; P2: analoga sui soggetti fuori dai test dei tre
split), con la stessa regola applicata a S; P1/P2 ufficiali sono conferma e risultato; in disaccordo il default segue S e si registra.
*Interpretazione:* costruire anche la S della P2 (dettaglio in `docs/bracci_obiettivo_finestra1.md`). Nessun risultato dei bracci esiste.
Confermata la S della P2 (Simone: «per p2 bene òa proposta tua»): validation interna su EPN-612 e UCI-EMG coi soggetti fuori dai test dei
tre split ufficiali, in `scripts/calibration_probe.py --task p2`.

**C1, training finiti (consuntivo, 16/10/2026).** 60715596 (LR 1e-4) 5 h 18, 60715602 (3e-4) 5 h 13, 60715603 (1e-3) 5 h 18: COMPLETED 0:0,
`max_steps` 24.523, nessuno STOP, checkpoint pre e post-decay. **63,2 GPU-ora** (4 GPU x 15 h 48; attese 60,3, tetto 78) sul budget della
calibrazione: 63,2 usate su 220, residuo 156,8. La valutazione (estrazioni, sonde, selezione) parte da un checkout nuovo con le correzioni
della selezione dopo la review del `critico`.

**Valutazione di C1 lanciata (19/10/2026; Simone: «ok procedi, avvisami quando parte la valutazione»).** Codice della valutazione 9ab781e
(`deploy/w1cal-eval-20261019`, checkout 61378990; review del `critico` APPROVATO), piano `results/infra/w1cal_c1_20261016/valutazione_C1.json`.
Guardie 61379130/61379133/61379134 COMPLETED 0:0 (checkpoint post-decay verificati), estrazioni 61379154 (1e-4), 61379217 (3e-4),
61379811 (1e-3), sonde 61379181/61379306/61379874 (CPU, in fila), selezione 61379881. L'estrazione di 1e-3 era stata rifiutata da sbatch
(«Job dependency problem»: afterok su una guardia gia' finita e uscita dalla memoria di SLURM; nessun job creato): risottomessa senza
dipendenza col `--dopo-rifiuto` del submitter, dopo aver verificato la guardia. Tetto della valutazione 1,5 GPU-ora sul budget della
calibrazione (63,2 usate su 220).

**C1 decisa con le regole firmate (selezione 61379881, 19/10/2026 16:21): finalisti 1e-4 e 3e-4.** Ammissibilita': 1e-4 si' (pannello
massimo 1,08 volte il minimo, rango finale 118), 3e-4 si' (una valutazione sopra 1,10, massimo 1,12, rango 85), **1e-3 no: A4, 7
valutazioni consecutive oltre 1,10 volte il minimo fino al passo 8.000 (massimo 1,58)**. Regola di C1: 3e-4 ammissibile -> finalisti 3e-4 e
la migliore alternativa ammissibile, cioe' 1e-4 (unica). S della P1 (descrittivo, non decide qui): 18,21 +- 0,78 / 17,46 +- 0,73 / 15,86 +-
0,60; confronti accoppiati descrittivi: 1e-4 - 3e-4 = +0,74 punti (SE 0,32), 1e-4 - 1e-3 = +2,35 (0,40), 3e-4 - 1e-3 = +1,61 (0,30). Identita'
dei tre run verificata (99 campi, codice 72cf24a). File delle decisioni: `results/infra/w1cal_decisioni.json`; rapporti e registro in
`results/infra/w1cal_c1_20261016/valutazione/`.
Consuntivo della valutazione: 3 estrazioni da 11 min (**0,57 GPU-ora**), sonde 1 h 09 / 1 h 17 / 1 h 46 in fila (~34 core-ora), selezione
3 min. Calibrazione: 63,8 GPU-ora usate su 220.
**Preventivo C2 (da autorizzare):** 1e-4 e 3e-4 al seme 1, 24.523 passi, un job da 6 h 30 per run: training 40,2 attese / 52,0 tetto,
valutazione 0,6 / 1,0: **~41 / 53 GPU-ora (424 ore locali)**; dopo il tetto restano almeno 103 GPU-ora della calibrazione per C3 e C4
(tetti 52 + 26 + valutazioni 6). Training dallo stesso checkout di C1 (72cf24a), stesso codice dei run da confermare.

**Calibrazione C2 AUTORIZZATA (Simone, 20/10/2026).** Simone: «via per C2, prepara i comandi». Due run al seme 1, LR 1e-4 e 3e-4
(`W1CAL-C2-30M-lr0p0001-s1`, `W1CAL-C2-30M-lr0p0003-s1`), 24.523 passi, warmup 1.962, decay da 19.618, un job da 6 h 30 per run, QoS
normal, dallo stesso checkout congelato di C1 (`wearusfm_campaigns/w1cal-20261016`, 72cf24abec5b8332011d83ce2b2fc7ae6bcc9cec). Piani in
`results/infra/w1cal_c2_20261020/` (`calibrazione_C2.json`, `comandi_training_C2.txt`, preventivo, riepilogo), generati con le decisioni
di C1; il piano di C1 rigenerato e' identico byte per byte a quello committato, e i comandi di C2 differiscono da quelli di C1 solo per ID,
campagna, seme e cartella del run. Preflight in sola lettura (20/10): coda vuota, cartelle dei run di C2 assenti, checkout (sola lettura) e
ingressi con le date di C1; SHA e pulizia del checkout, SHA del manifest e delle durate li verifica il job all'avvio (`lib_code_dir.sh`,
`check_input_files`). **Preventivo C2: ~41 GPU-ora attese, tetto 53 (424 ore locali)**, training 40,2 / 52,0 e valutazione 0,6 / 1,0, sul
budget della calibrazione (63,8 usate su 220): dopo il tetto restano almeno 103,2. Sottomissione da parte di Simone. La valutazione di C2
parte da un deploy nuovo, dopo la correzione dell'afterok della catena.
**C2 sottomessa da Simone (20/10/2026 11:44):** 61526225 (`W1CAL-C2-30M-lr0p0001-s1-j01`) e 61526229 (`W1CAL-C2-30M-lr0p0003-s1-j01`),
sottomissioni pulite (uscita 0, stato in `~/wearusfm_local/jobs/`); alle 11:46 entrambi PENDING per priorita', senza partenza stimata.

**Catena di valutazione: afterok sui nodi della stessa chiamata corretto (20/10/2026).** Il difetto visto nella valutazione di C1 (sbatch
aveva rifiutato l'estrazione di 1e-3 con «Job dependency problem»: afterok su una guardia finita in 5 s e gia' dimenticata da SLURM) e'
corretto in `scripts/evaluation_chain.py`: con `--submit` i nodi si sottomettono uno alla volta e prima di ciascuno lo stato dei
predecessori si rilegge con `sacct`, compresi quelli sottomessi nella stessa chiamata (finito bene -> nessun afterok, fallito -> nodo
bloccato, in coda o appena accettato da sbatch -> afterok). Il dry-run non cambia. Test nuovo sul cluster finto, che ora rifiuta come SLURM
l'afterok su un job dimenticato: fallisce con la catena di prima e passa con la correzione; suite 1194 PASS (1193 + 1). La valutazione di C2 usera'
questa versione (deploy nuovo). Collaudo infrastrutturale: nessuno, si vede alla valutazione di C2 (stessa catena, un solo cambiamento).

**Review indipendente GPT di C1-C4 (16/10/2026, patch 41ce569 e merge locale 41b80af).** Hardening della provenienza delle cache e
del confronto accoppiato, checkpoint verificato per byte/SHA, bootstrap invalido bloccante, merge delle decisioni dopo stop rifiutato,
rango zero per feature costanti. Nessuna modifica ad A4, parita', griglia o budget firmati. Restano due raccordi segnalati a Claude:
finalizzazione di candidati falliti prima della sonda e verifica della configurazione effettiva contro il piano. Dettagli e prove in
`docs/reviews/c1c4_review_20261016.md`. Patch nel branch GPT, non ancora su main o nei checkout di campagna; lettura di Claude prima
del deploy. Le cache di calibrazione senza i nuovi metadati non si attestano retroattivamente. Non revoca il via a C1 e non cambia
i ruoli del 16/10; nessun lancio o costo in questa review.

**Punto 3: integrazione della consegna corrente su main CONCLUSA (Simone, 16/10/2026).** Richiesta: «fai il merge tu sul main del
punto 3 ? cosi lo chiudiamo». Main canonico pulito aggiornato in fast-forward da 06b0d34 a 1aa33c9, con Claude fino ad a7edd55 e
correzioni GPT 41ce569, conservando tutti gli SHA. Codice gia' verificato dai 158 test mirati della review. I due raccordi residui
della selezione restano aperti nel punto 6; nessuna soglia o budget cambiato. Registro aggiornato in entrambe le sezioni. Nessun push,
deploy o lancio in questa attivita'; le campagne congelate mantengono il proprio codice.

**Review GPT del punto 5, implementazione JEPA-only/ancore-only (16/10/2026): favorevole.** Esaminati b1238a1 e fe67339 nel checkout
2dc1197; letta anche la firma del protocollo 0d45a2f sul ramo Claude. Nessun difetto confermato; 45 test mirati passati, incluse prove
aggiuntive di equivalenza delle loss, gradienti, pesi ed EMA fra batch globale, accumulo e due processi CPU gloo. Dettagli in
`docs/reviews/objective_arms_review_20261016.md`. Restano pianificatore, piano P1/P2, collaudo GPU breve e risultati scientifici;
per la keep separata, ricalibrazione sul corpus intero. Nessuna nuova scelta scientifica o autorizzazione a lanciare in questa voce.

**Merge di `main` nel ramo di Claude (20/10/2026): review C1-C4 di GPT (41ce569) letta e integrata.** Simone, consegna sui rami: «Lavora
nel tuo worktree». Prima della diagnostica dei rami (che riusa sonda e confronti della calibrazione) e della valutazione di C2. Patch di GPT
tenute tutte (ricetta d'estrazione e SHA degli split nelle cache, righe OOF e loro impronta, `check_pairing` prima di ogni decisione, SHA del
checkpoint contro la sonda, bootstrap invalido bloccante, `merge` dopo stop o conferma incompleta rifiutato, rango zero per feature costanti),
unite alla P2 interna e alla selezione col piano del ramo. Correzioni del merge: la cache e il rapporto della P2 interna registrano anche loro
ricetta e identita' degli split (`EXTRACTION_RECIPE_P2`, `P2_SPLITS_SHA256`: la definizione degli split, non c'e' un file); il controllo
dello SHA del checkpoint non si applica a un candidato senza sonda (prima del merge avrebbe sollevato un'eccezione); `arms_compare` chiama
`check_pairing` sui rapporti di S. Test di GPT adattati alla selezione col piano, nuovi test su P2 e bracci; suite 1212 PASS. Le cache di C1
non hanno i campi nuovi: C1 resta decisa com'e' (risultato storico); le estrazioni di C2 e dei bracci si fanno col codice integrato.

**Diagnostica dei rami PSD e wavelet, protocollo `rami_diagnostica_v1` PROPOSTO (20/10/2026; Simone: «Procedi col punto 1», consegna
preparata da GPT il 20/10).** Punto 5 della tabella; verdetti 2A/2C invariati. Domanda: il training dei rami del 2C peggiora i rami anche da
soli o soprattutto nella concatenazione col raw? Solo cache gia' estratte (2C `step2c_1410`, tranche downstream `step1_downstream_0910`),
verificate sul cluster: coprono la validation interna (P1: soggetti di pretraining, le righe di test non si leggono; P2: soggetti fuori dai
test dei tre split), **nessuna estrazione nuova**. Metrica S della calibrazione invariata (4 fold di soggetti, normalizzazione sul train del
fold, C sui soggetti di validazione del fold). Insiemi PW0, PW, R, RPW0, RPW sui quattro teacher dell'A/B. **Lettura proposta (si congela col
commit che precede il lancio, dopo la review del `critico`; nessuna sonda prima):** per famiglia e compito, confronti accoppiati PW0->PW (rami da soli) e RPW0->RPW (concatenazione)
letti con la regola dei bracci («migliora»: delta >= +0,01, limite inferiore dell'IC > 0, delta > 0 con i due teacher) e la sua speculare
(«peggiora»: delta <= -0,01, limite superiore < 0, delta < 0 con i due teacher), altrimenti «entro le soglie»; tabella delle quattro risposte
in `docs/rami_diagnostica_v1.md`. R->RPW0 e R->RPW, rango effettivo, CKA e R^2 dei rami dal raw: descrittivi. Le letture indicano dove si
vede il peggioramento, non la causa; non adottano ne' escludono nulla. **Preventivo (da autorizzare):** 16 job seriali da 8 core + confronto
+ checkout, **~241 core-ora attese, tetto 481 core-ora (481 ore locali), 0 GPU-ora**, ~30 h di calendario, `--nice` dietro ai job della
finestra 1; ~80 MB su `$SCRATCH`. Prima del lancio: review del `critico`. Codice `scripts/spectral_diagnostic.py`,
`scripts/slurm/spectral_diagnostic.sbatch`; 6 test nuovi PASS (suite 1218 PASS su c631c6d).

**C2 partita (20/10/2026 12:03):** 61526225 (1e-4) su lrdn3052 e 61526229 (3e-4) su lrdn2272. Primi 2.350 / 800 passi regolari: perdite
finite, passo 1 identico nei due run (6,1213: stesso seme), 0,61-0,95 s/passo (media ~0,70, come la stima), dati 0,00-0,01 s, memoria
massima 10,2 GiB. Fine prevista verso le 17:15, dentro il limite del job (21.000 s).

**Diagnostica dei rami: review del `critico` (20/10/2026, su c631c6d) DA RIVEDERE, rilievi chiusi; lettura proposta aggiornata (sostituisce
la tabella delle quattro risposte della voce precedente; nessuna sonda lanciata).** Risposta per famiglia e compito = solo quale dei due
confronti chiave (PW0->PW, RPW0->RPW) supera la regola dei bracci o la sua speculare: «peggioramento oltre le soglie», «miglioramento oltre le
soglie», «entro le soglie (non esclude una differenza sotto 1 punto)»; nessun meccanismo ne' «soprattutto». Il confronto fra i due delta e'
un descrittivo senza soglia: (RPW - RPW0) - (PW - PW0) con IC dallo stesso ricampionamento accoppiato. Nota: l'effetto che motiva la domanda
puo' stare sotto la soglia (2C, A/P1 ufficiale -0,84). 16 letture senza correzione per confronti multipli. **Lancio solo dopo la valutazione
di C2**, a tranche con `evaluation_chain.py` (al piu' 3 job seriali dell'utente in coda o in esecuzione, finestra 1 compresa; confronto come
ultimo nodo; `--nice=10000`); ritardo massimo su un job seriale della finestra 1: il resto della sonda in esecuzione, al piu' 4 h; nessuna
tranche nuova a ridosso delle valutazioni di C3 e C4. Budget: ore locali CPU dell'account, nessun budget GPU; tetto 481 ore locali da
autorizzare. Disco ~250 MB su `$SCRATCH`. Correzioni nel codice: uscita del confronto fissa, argomenti di uscite/radici/split rifiutati dal
launcher, nessuna abbreviazione; run del 2C controllato (stesso run per passo 0 e allenati, completo, non di collaudo); `--nice` (solo
positivo) nell'elenco chiuso di `evaluation_chain.py`. Il numero dei test nuovi della voce precedente era sbagliato (7): sono 6, suite 1218
PASS su c631c6d (corretto nella voce).

**Diagnostica dei rami: seconda review del `critico` (20/10/2026, su 10d3731) DA RIVEDERE per un rilievo aperto in parte, chiuso; lettura
proposta precisata (nessuna sonda lanciata).** «Entro le soglie» = **regola non superata** (stima sotto 1 punto, intervallo che tocca lo zero
o teacher discordi): la frase della risposta e' «regola non superata (stima, intervallo o accordo dei teacher insufficienti): non esclude una
differenza, neanche oltre 1 punto», sostituisce «non esclude una differenza sotto 1 punto» della voce precedente; delta, IC e delta per
teacher sempre accanto. Limite di coda della diagnostica: **2** job seriali dell'utente (finestra 1 compresa; uno alla volta gira comunque).
Ritardo massimo su un job seriale della finestra 1: al piu' 4 h **se il nice prevale sull'eta'** (pesi di priorita' non letti: `scontrol` e
`sprio` non sono fra i comandi ammessi a leonardo-ops), **caso peggiore ~8 h**. Soggetti di UCI-EMG verificati sulla provenienza della
tranche downstream: 36, 19 interni, 662 finestre in tutto (EPN-612: 612, 314 interni, 91.800 finestre).

**Diagnostica dei rami: terza review del `critico` (20/10/2026, su 84bcbe0) APPROVATO.** Un minore chiuso (commento di `NICE` allineato al
ritardo condizionato). **Lettura congelata con questo commit**, prima di qualunque sonda: quella delle voci del 20/10 come precisata dalla
seconda review («entro le soglie» = regola non superata, non esclude una differenza neanche oltre 1 punto; differenza dei delta descrittiva
senza soglia). **Lancio da autorizzare** (preventivo ~241 core-ora attese, tetto 481 ore locali, 0 GPU-ora, ore locali CPU dell'account),
comunque dopo la valutazione di C2.

**Pesi di priorita' di SLURM letti (Simone, 20/10/2026: `ssh leonardo 'scontrol show config | grep -i priority'`,
`results/infra/slurm_priority_20261020.txt`).** multifactor; eta' 20.000 (massimo a 7 giorni), fair-share 25.000, dimensione 10.000.000
(`SMALL_RELATIVE_TO_TIME`, `PriorityFavorSmall = no`), QoS 300.000, partizione 0. Per la diagnostica dei rami: il nice di 10.000 prevale
sull'eta' di una sonda in attesa (~476 punti in 4 h, pareggio dopo ~84 h), quindi **il ritardo massimo su un job seriale della finestra 1 e'
il resto della sonda in esecuzione, al piu' 4 h**; il caso peggiore di ~8 h della voce precedente non si applica.

**Diagnostica dei rami AUTORIZZATA (Simone, 20/10/2026: «vai pure quando pronto»),** alle condizioni presentate: preventivo ~241 core-ora
attese, tetto 481 ore locali (ore CPU dell'account), 0 GPU-ora; lancio dopo la valutazione di C2, a tranche con `evaluation_chain.py` (al
piu' 2 job seriali dell'utente, finestra 1 compresa), `--nice=10000`, nessuna tranche nuova a ridosso delle valutazioni di C3 e C4. Lettura
congelata in b597786. Lo stesso deploy serve alla valutazione di C2 (codice integrato con la review GPT e la correzione dell'afterok).

**Merge della review C1-C4: review del `critico` APPROVATO (20/10/2026, su 58c043a).** Nessuna regola firmata cambiata; percorso reale di C2
provato end-to-end in una cartella temporanea (estrazione P1 vera con modello finto, sonda, `select("C2")`, candidato senza sonda, `merge` sul
file vero delle decisioni); P2 interna accoppiabile; nessuna perdita nella risoluzione dei conflitti. Tre minori chiusi: (1) senza sonda, un
checkpoint mancante non conta piu' fra i problemi di A1, cosi' un run altrimenti ammissibile senza sonda resta l'errore del protocollo firmato
(«la valutazione si completa, non si salta») invece di diventare un esito; (2) i test d'estrazione controllano che P1 e P2 scrivano i campi
che la sonda verifica; (3) il protocollo `docs/calibrazione_c1_c4_selezione.md` elenca i controlli bloccanti della review GPT. Test collegati
120 PASS. Questo codice e' quello del deploy della valutazione di C2 e della diagnostica dei rami.

**Deploy della valutazione di C2 e della diagnostica dei rami (20/10/2026).** Preflight in sola lettura pulito (ref e cartella nuovi assenti,
hook solo `.sample`, `updateInstead` solo su `main`; in coda solo i due training di C2, su un altro checkout). `deploy/w1cal-eval-20261020` =
62bec0b71e89cdcff54de7c7e0f93d7046019213 (codice con il merge della review GPT, `critico` APPROVATO, e la correzione dell'afterok), checkout
di campagna job 61540094 (1 core, 10 min). Piano `results/infra/w1cal_c2_20261020/valutazione_C2.json` dalla campagna
`results/infra/w1cal_campaign_c2_eval_20261020.json` (piano di training di C2 rigenerato identico byte per byte), dry-run della catena: 2
guardie, 2 estrazioni (0,5 GPU-ora di tetto ciascuna), 2 sonde CPU, selezione. Sottomissione a training finiti.

**Diagnostica dei rami: review GPT (20/10/2026, 4203d59 sul suo ramo, `docs/reviews/spectral_diagnostic_review_20261020.md`) favorevole al
disegno, due P2 di robustezza corretti, nessun cambio del protocollo scientifico.** (1) Rami costanti: R^2 e CKA non definiti si registrano
come null con lo stato «non definito» (prima R^2 NaN faceva fallire il JSON e il job); rango effettivo 0. (2) ID dei job unici per campagna
(`RAMI-<campagna>-<teacher>-<job>`, `--campaign` obbligatoria), stabili nelle tranche della stessa campagna. Minori: IC della differenza dei
delta anche per dataset; `compare` controlla protocollo, compito e teacher di `redundancy.json`. Le due regressioni di GPT passano sul ramo;
test mirati 71 PASS. La diagnostica partira' da un deploy nuovo con queste correzioni (la valutazione di C2 resta sul deploy 62bec0b, che
non usa questo script).

**C2, training finiti (consuntivo, 20/10/2026).** 61526225 (1e-4) 5 h 03 e 61526229 (3e-4) 5 h 04: COMPLETED 0:0, 24.523 passi
(`max_steps`), checkpoint pre e post-decay, codice 72cf24a. **40,4 GPU-ora** (4 GPU x 10 h 07; attese 40,2, tetto 52) sul budget della
calibrazione: **104,2 usate su 220**, residuo 115,8.
**Valutazione di C2 sottomessa (20/10/2026 17:3x)** dal checkout `w1cal-eval-20261020` (62bec0b): guardie 61558827 / 61558828 (senza afterok:
training gia' finiti bene secondo sacct), estrazioni 61558829 (1e-4) / 61558831 (3e-4), sonde 61558830 / 61558832, selezione 61558834.
Nessun rifiuto di sbatch (correzione dell'afterok del 20/10 al primo uso). Tetto 1,0 GPU-ora.

**Rami spettrali, punto 2: scelte di Simone (20/10/2026).** Simone: «A, solo famiglia B, includi il controllo di LR». Obiettivo A
(predizione di rappresentazioni locali mascherate nei rami, raw congelato), teacher della sola famiglia B (B_seed0, B_seed1), con un braccio
di controllo dell'LR sull'obiettivo vecchio. Protocollo, codice e preventivo da preparare; review del `critico` prima del lancio; il lancio
dopo l'esito della diagnostica del punto 1, che ne orienta la lettura.

**Rami spettrali, punto 2: review del disegno del `critico` (20/10/2026) DA RIVEDERE; braccio in piu' scelto da Simone.** Tre bloccanti: fuga
attraverso le code delle wavelet (lo studente calcola gli spettri sul segnale coi tratti nascosti azzerati), anti-collasso tarato male
(riferimento = predittore costante, soglie relative al passo 0 per ramo e scala, varianza assoluta dei token), obiettivo confuso con gli
strati allenati. Per l'ultimo Claude ha proposto due rimedi: un controllo ibrido a costo GPU zero (token del vecchio obiettivo, strati finali
del passo 0) e due run dell'obiettivo vecchio con gli strati finali congelati. Simone: «se consigli di aggiungere, aggiungi», «vai»: si
aggiungono i due run (PWf) e il primario diventa il confronto fra obiettivi a parita' di strati allenati e LR. Protocollo in revisione.

**C2 decisa con le regole firmate (selezione 61558834, 20/10/2026 20:12): confermati 1e-4 e 3e-4.** Entrambi ammissibili al seme 1:
1e-4 pannello massimo 1,04 volte il minimo, rango finale 124; 3e-4 massimo 1,03, rango 87; nessuna valutazione sopra 1,10. S della P1 18,01
+- 0,77 (1e-4) e 17,95 +- 0,71 (3e-4); confronto descrittivo 1e-4 - 3e-4 = +0,06 punti (SE 0,27, parita'). Identita' dei due run verificata
(99 campi, codice 72cf24a). `results/infra/w1cal_decisioni.json` = C1 finalisti + C2 confermati; rapporti, registro e config/summary in
`results/infra/w1cal_c2_20261020/`. Consuntivo della valutazione: 2 estrazioni da 11 min 43 (**0,39 GPU-ora**), sonde 1 h 16 e 1 h 20 (~21
core-ora). **Calibrazione: 104,6 GPU-ora usate su 220.**
**Preventivo C3 (da autorizzare):** 100M, 0,5 epoche (12.261 passi, warmup 1.962, decay da 9.809), LR 1e-4 e 3e-4, seme 0, un job da 6 h 30
per run: training 39,8 attese / 52,0 tetto, valutazione 1,3 / 3,0 (CPU fino a ~66 core-ora): **~41 / 55 GPU-ora (440 ore locali)**; dopo il
tetto restano almeno 60,4 GPU-ora per C4 (tetto 26 + valutazione 3).

**Diagnostica dei rami: deploy e piano (21/10/2026).** Dopo la valutazione di C2, come autorizzato. Preflight in sola lettura pulito;
`deploy/rami-diag-20261021` = aba1f2ef3a94b68b8e44bce005adf7b005a92011 (ultimo commit con la suite completa verificata, 1241 PASS, con le
correzioni della review GPT; i commit successivi toccano il trainer, non usato dalla diagnostica), checkout 61582077. Piano
`results/rami_diagnostica/piano_20261021.json` (17 nodi, campagna `rami-diag-20261021`, al piu' 2 job seriali dell'utente, `--nice=10000`),
dry-run: a coda vuota entrano i primi due job di B_seed0. Rapporti in `$SCRATCH/wearusfm_runs/rami_diagnostica_20261021`.
**Diagnostica dei rami, prima tranche (21/10/2026 ~00:10):** checkout 61582077 pronto (aba1f2e, 595 file); sottomessi 61582132 (B_seed0 P1:
R, PW0, PW e ridondanza, 4 h) e 61582137 (B_seed0 P1: RPW0, 3 h 30), `--nice=10000`; gli altri 15 nodi in attesa del limite di coda.

**Review GPT della seconda tranche (3272398, 20-21/10/2026): tre rilievi corretti, consegnati per il merge (21/10/2026).** Simone: «Sistema i tre
rilievi della review GPT `3272398`, poi consegna i commit per verifica e merge su `main`». Nessuna regola firmata cambiata, nessun training da
ripetere. **R1:** un candidato senza sonda si esclude solo con una prova di fallimento terminale registrata nel risultato (STOP del trainer o
evidenza da sacct; in coda, in corso, TIMEOUT, finito o ignoto: nessuna decisione), `--failed-runs RUN=PROVA`. **R2:** PASSI vincolati al
piano congelato. **R3:** QoS normal esplicita per le estrazioni dei bracci e controllo del tempo sulla QoS effettiva. Sul ramo di Claude:
07e120c (R1+R2) e 32b5718 (R3), suite 1272 PASS. **Consegna separata dal pilot dei rami:** branch `consegna/review-3272398` da c21ca40 (il punto della
review) coi soli due fix (8e920bb R1+R2, e1929be R3), test suite 1254 PASS e 140 test collegati PASS; le tre regressioni di GPT passano (`WEARUSFM_REVIEW_SHA`
= e1929be).
**Diagnostica dei rami, tranche 3 (21/10/2026 02:45):** 61582137 (B_seed0 P1 RPW0) COMPLETED 0:0 in 1 h 37 (stima 1 h 45); 61582132 (B_seed0
P1 R, PW0, PW) in esecuzione; sottomesso 61587820 (B_seed0 P1 RPW, 3 h 30). Nessun fallimento.

**Diagnostica dei rami: tranche sospese (21/10/2026 04:45), stime sbagliate per i soli rami.** Misure su B_seed0, P1: R (768 colonne) 1 h 26,
RPW0 (1.024) 1 h 37, ma **PW0 (256) oltre 1 h 30 e non ancora finito**: la regressione logistica sui soli rami arriva al limite di 3.000
iterazioni (avvisi di convergenza di lbfgs). Il job 61582132 (R, PW0, PW e ridondanza, 4 h) non entrera' nel suo limite (05:49): R e' salvato,
PW0, PW e la ridondanza a rischio. Lo stesso raggruppamento e' pianificato per gli altri tre teacher, e i job P2 mettono 5 insiemi in 4 h.
Sospese le tranche automatiche (nessun job nuovo); i job gia' sottomessi continuano (61587820, RPW di B_seed0, in coda). Proposta a Simone:
un job per insieme con tempi misurati (stima ~435 core-ora complessive, tetto ~980, oltre il tetto autorizzato di 481) oppure la sola
famiglia B ora (~200 / ~490), A dopo. Nessuna modifica del protocollo scientifico.

**Diagnostica dei rami: tempi misurati e piano v2 pronto (21/10/2026 06:15), in attesa della scelta di Simone.** 61582132 TIMEOUT alle
05:51 (4 h 03) con R (1 h 27) e PW0 (2 h 10, 7.800 s) salvati, PW interrotto, ridondanza non fatta; 61582137 (RPW0) 1 h 37; 61587820 (RPW
di B_seed0) in corso dalle 05:51. Piano v2, un job per insieme (`spectral_diagnostic.py plan --per-set`, tempi massimi P1 R 3 h, PW0 e PW 4
h, RPW0 e RPW 3 h; P2 2-3 h; la ridondanza col primo job rimasto), campagna `rami-diag-20261021b`, insiemi gia' fatti saltati, confronto
dipendente anche da 61587820: **opzione B** (`results/rami_diagnostica/piano_20261021b_B.json`, famiglia B, 16 sonde, tetto 368 core-ora
nuove, ~441 con quelle gia' usate, entro le 481 autorizzate; stima ~220 complessive) oppure **A+B** (`..._AB.json`, 36 sonde, tetto 840
nuove, ~913 complessive, stima ~437, oltre il tetto autorizzato). Nessun job sottomesso.

**Diagnostica dei rami: opzione B autorizzata (Simone, 21/10/2026: «va bene, procedi con B»).** Piano v2 per la sola famiglia B
(`results/rami_diagnostica/piano_20261021b_B.json`, campagna `rami-diag-20261021b`, 16 sonde e confronto `--families B`), tetto ~441 core-ora
compresi i job gia' usati, entro le 481 autorizzate. La famiglia A resta da decidere dopo l'esito di B. Tranche automatiche riattivate sul
piano v2 (al piu' 2 job seriali dell'utente, `--nice=10000`).
**Diagnostica dei rami, piano v2, prima tranche (21/10/2026 ~08:10):** 61587820 (RPW di B_seed0, P1) COMPLETED 0:0 in 1 h 30; sottomessi
61592919 (B_seed0 P1 PW e ridondanza, 4 h) e 61592922 (B_seed0 P2 PW0, 3 h). Gli altri 14 nodi in attesa del limite di coda.

**Calibrazione C3 AUTORIZZATA (Simone, 21/10/2026: «Potresti partire gia con C3 ?», sul preventivo presentato).** 100M, 0,5 epoche
(12.261 passi, warmup 1.962, decay da 9.809), LR 1e-4 e 3e-4 (confermati da C2), seme 0, un job da 6 h 30 per run, QoS normal, dallo stesso
checkout di training di C1 e C2 (`w1cal-20261016`, 72cf24a). Piani in `results/infra/w1cal_c3_20261021/` (piani di C1 e C2 rigenerati
identici byte per byte). **Preventivo: training 39,8 attese / 52 tetto, valutazione 1,3 / 3,0: ~41 / 55 GPU-ora (440 ore locali)** sul
budget della calibrazione (104,6 usate su 220): dopo il tetto restano almeno 60,4 per C4. Sottomissione di Simone (come C1 e C2). La
diagnostica dei rami non sottomette tranche nuove a ridosso della valutazione di C3.
**C3 sottomessa da Simone (21/10/2026):** 61593336 (`W1CAL-C3-100M-lr0p0001-s0-j01`) e 61593337 (`W1CAL-C3-100M-lr0p0003-s0-j01`).

**Bracci dell'obiettivo: piani, collaudo e preventivo pronti (21/10/2026, Simone: «va bene procedi» sul passo consigliato), da autorizzare.**
LR = 1e-4 con la regola firmata (C2 conferma 1e-4 e 3e-4; al seme 1 S 0,18011 contro 0,17951, differenza accoppiata +0,0006, SE 0,0027:
parita' entro 1 SE, il piu' basso), derivato da `scripts/arms_plan.py` da `results/infra/w1cal_decisioni.json` e `registro_C2.json` (SHA nel
riepilogo). Piani in `results/infra/w1arm_20261021/` (campagna `results/infra/w1arm_campaign_20261021.json`): 6 run (A, JEPA-only,
ancore-only, semi 0 e 1), 98.094 passi, un job da 23 h ciascuno, dal checkout di training di C1-C3 (`w1cal-20261016`, 72cf24a). Review del
`critico` sugli strumenti dei bracci: DA RIVEDERE, 1 bloccante (il confronto non verificava identita' dei run e provenienza dei rapporti) e
5 minori, corretti con test (`docs/bracci_obiettivo_finestra1.md`, «Lancio del gruppo dell'obiettivo»); re-review prima del lancio.
**Collaudo `w1arm-collaudo-v1`** dei due bracci nuovi (1.300 passi ciascuno, `boost_qos_dbg`, 4 GPU, 30 min): criteri congelati in
quel documento e in `scripts/arms_collaudo.py`; **2,6 GPU-ora attese, tetto 4,0 (32 ore locali)** sul budget «preparazione
dell'infrastruttura» (9,83 usate su 25: dopo il tetto restano almeno 11,17). **Preventivo del gruppo:** training 463,8 / 552,0 GPU-ora (484
attese col passo di C2; 4.416 ore locali di tetto), valutazione 10,5 / 27,0 con sonde sulla seriale (+192 core-ora, ~24 h in fila) o 34,5 /
87,0 con sonde su GPU. Voce proposta sulla finestra 1 (6.000, calibrazione compresa; consumato fuori dalla calibrazione: 0): **tetto 579
GPU-ora** (sonde sulla seriale) **o 639** (sonde su GPU). Keep: bloccata fino alla calibrazione sul corpus intero. Nessun job sottomesso.

**Bracci dell'obiettivo: re-review del `critico` (21/10/2026, su c3ef10b) DA RIVEDERE, rilievi corretti.** Bloccante: il confronto non legava
l'etichetta del braccio al run (A e JEPA-only scambiati per intero passavano; un braccio si poteva omettere): ora il piano deve essere
quello di `arms_plan.arm_runs` per gruppo e LR, la spec copre tutti i bracci del gruppo e ogni etichetta e' il run del suo braccio. Minori:
provenienza di P1/P2 ufficiali solo dichiarata (scritto come limite; il default si sceglie su S), controllo del compito dei rapporti di S,
rischio sul tempo dichiarato, piano lanciato congelato, esito del collaudo scritto anche coi non finiti (e composizione della loss con
componenti finite: criterio 5 piu' stretto, prima del lancio), test del percorso relativo. Dettagli in `docs/bracci_obiettivo_finestra1.md`.
**Piani lanciabili congelati:** `results/infra/w1arm_20261021/bracci_objective.json` SHA-256
`dbe0f8feff7ebc3604dfc3c414646b51ddf3160bcfb699d8a5ed7021e6940b52` (training dei 6 run) e `collaudo_bracci.json`
`a9e835400f5dc5a659952761dea570b961fcc3c11d639f9f4e38caa331954d6f` (collaudo); alla valutazione `arms_plan.py --frozen-plans
results/infra/w1arm_20261021` rifiuta un piano rigenerato diverso. **Correzione del preventivo:** il passo di A a 30M misurato va da 0,740 s
(C2) a 0,778 s (C1): training atteso 484-509 GPU-ora (tetto invariato 552), run da 20,2-21,2 h, un job solo finche' il passo medio resta
sotto ~0,81 s; un run fermato per tempo si completa con un job di ripresa da presentare. Suite 1301 PASS su c3ef10b. Nessun job sottomesso.

**Bracci dell'obiettivo: terza passata del `critico` (21/10/2026, su bd907a0): APPROVATO**, cinque minori corretti: margine del tempo scritto
con la convenzione della soglia (0,9-1,9 h dopo i 15 minuti di avvio, non 1,1-2,0); passi di D nella spec (da `arms_plan`, dal rapporto dei
derivati) confrontati col piano; test del gruppo keep nel confronto; spiegazione corretta del difetto dei non finiti (tolleranza relativa
infinita); limite di provenienza di P1/P2 ufficiali esteso all'encoder dichiarato. Piani congelati invariati (`dbe0f8fe...`, `a9e83540...`).
Strumenti dei bracci pronti per il collaudo; il collaudo aspetta il via di Simone.

**Collaudo dei bracci: controllo in sola lettura pulito (21/10/2026 ~11:15).** Checkout di training e input presenti, cartelle dei run del
collaudo e dei bracci assenti, in coda solo C3: i 2 posti di `boost_qos_dbg` sono liberi. Il collaudo aspetta il via di Simone.

**Collaudo `w1arm-collaudo-v1` sottomesso da Simone (21/10/2026), sul preventivo presentato (2,6 / 4,0 GPU-ora, budget infrastruttura).**
61607555 = `W1ARMC-JEPA-only-30M-s0-j01`.
61607561 = `W1ARMC-ancore-only-30M-s0-j01`.

**Collaudo `w1arm-collaudo-v1`: SUPERATO (21/10/2026, criteri congelati, `scripts/arms_collaudo.py`).** 61607555 (JEPA-only) e 61607561
(ancore-only) COMPLETED 0:0 in 17:54 e 17:59 (11:28-11:46), 1.300 passi, `max_steps`, codice 72cf24a, DDP a 4 GPU: 8 criteri su 8 per
entrambi (pesi effettivi JEPA 1 / ancore 0 e JEPA 0 / ancore 0,2, composizione della loss esatta a ogni passo, record completi e finiti,
termine allenato in calo: JEPA 0,0090 -> 0,0010, ancore x 0,2 4,86 -> 1,26; nessun allarme, diagnostiche a 500/1.000/1.300, checkpoint
pre e post-decay). Esiti in `results/infra/w1arm_20261021/collaudo/`. Descrittivi: 0,71 s/passo (mediana), memoria GPU 11,9 GiB, attesa
sui dati tipica 0,6% (media 2,8-3,0% per una decina di passi con pause di secondi, gli stessi nei due run, che hanno lo stesso flusso di
dati). **Da notare (descrittivo, nessuna soglia nuova):** nel JEPA puro il rapporto di collasso scende da 1,30 a 0,58 (studente) e da 1,61 a
0,65 (teacher) fra i passi 500 e 1.300, rango effettivo ~52; in ancore-only resta ~2 (rango ~71). L'allarme firmato e' a 0,05: lontano, ma
la tendenza dice che il JEPA puro e' il candidato al collasso nei 98.094 passi (la firma lo conta come «non batte A»).
**Consuntivo del collaudo: 2,39 GPU-ora** (attese 2,6, tetto 4,0); budget infrastruttura **12,22 usate su 25, residuo 12,78**.

**C3, training finiti (consuntivo, 21/10/2026).** 61593336 (1e-4) e 61593337 (3e-4) COMPLETED 0:0 in 4 h 57 (08:13-13:11), 12.261 passi,
`max_steps`, 1,453 s/passo, 23,2 GiB, codice 72cf24a; config e summary in `results/infra/w1cal_c3_20261021/runs/`. **39,63 GPU-ora**
(attese 39,8, tetto 52): calibrazione **144,2 usate su 220, residuo 75,8**. Prossimo: deploy e catena della valutazione di C3.

**Diagnostica dei rami, famiglia B:** 61592919 (B_seed0 P1 PW e ridondanza) COMPLETED in 2 h 28 e 61592922 (B_seed0 P2 PW0) in 14 min; il
dry-run della catena ha pronti B_seed0 P2 PW e R, 12 sonde e il confronto ancora da fare. Tranche a richiesta di Simone.

**Bracci dell'obiettivo: 6 run sottomessi da Simone (21/10/2026), dopo il collaudo superato e sul preventivo presentato** (training
484-509 GPU-ora attese, tetto 552, 4.416 ore locali; un job da 23 h per run, checkout 72cf24a, piano `dbe0f8fe...`; Simone ha eseguito i
comandi di `results/infra/w1arm_20261021/comandi_training_objective.txt`): 61622377 (`W1ARM-A-30M-s0-j01`), 61622387 (A s1), 61622389
(JEPA-only s0), 61622391 (JEPA-only s1), 61622414 (ancore-only s0), 61622418 (ancore-only s1). **Voce di budget «Bracci dell'obiettivo»
(tetto 639 GPU-ora proposto, 579 con le sonde sulla seriale): da confermare esplicitamente da Simone**; la sottomissione autorizza il
training (tetto 552), non ancora la valutazione. Valutazione dopo i training, rigenerata con `--frozen-plans`.
**Diagnostica dei rami, piano v2, tranche di Simone (21/10/2026 ~14:50):** 61622429 (B_seed0 P2 PW, 3 h) e 61622430 (B_seed0 P2 R, 2 h);
restano 12 sonde e il confronto.

**Integrazione GPT (21/10/2026):** seconda tranche e fix di consegna integrati nei merge 6484432 (e1929be) e 93d5541 (bbf824a),
preservando gli SHA originali. Pilot dei rami escluso; nessun cambio di regola firmata o risultato C1/C2, nessun nuovo lancio.
Verifica indipendente sul codice integrato: 132 test distinti PASS (dettaglio e correzione dell'invocazione nella review).

**Review GPT del pilot temporale dei rami (21/10/2026, target 134b77c): da rivedere.** 47 test esistenti PASS e tre regressioni dei
rilievi FAIL attesi: metrica indefinita nel logging, consecutivi del monitor con lacune, monitor dei token assente per PWf. Protocollo
ancora PROPOSTA, wrapper e controlli di esecuzione da completare; nessuna nuova soglia o autorizzazione di spesa firmata qui.
Consegna a Claude in docs/reviews/branch_pilot_review_20261021.md. Non riguarda i collaudi JEPA-only/ancore-only o C3.

**Valutazione di C3 AUTORIZZATA (Simone, 21/10/2026: «punto 2 approvo. vai pure.»)**, sul preventivo gia' firmato con C3 (1,3 GPU-ora
attese, tetto 3,0; sonde CPU 48 / 65,7 core-ora). Merge di `main` (6fbe54d, review GPT del pilot) nel ramo: 98c5832, nessun cambio del
codice del ramo (le correzioni R1-R3 di `main` sono le stesse gia' presenti). Deploy previsto `deploy/w1cal-eval-20261021` = 98c5832 dopo la
suite completa; controllo in sola lettura pulito (ref e cartella assenti, hook solo `.sample`, `updateInstead` solo su `main`). Piano
`results/infra/w1cal_c3_20261021/valutazione_C3.json` dalla campagna `results/infra/w1cal_campaign_c3_eval_20261021.json` (piano di
training di C3 rigenerato identico): 2 guardie, 2 estrazioni (QoS normal, 1 h 30), 2 sonde sulla seriale (4 h), selezione.
**Deploy della valutazione di C3 (21/10/2026):** suite completa 1309 PASS sul merge; `deploy/w1cal-eval-20261021` = 98c5832 (SHA remoto
verificato), checkout di campagna job 61624224 (1 core, 10 min, 0 GPU-ora).

**Pilot dei rami: correzioni della review GPT del 21/10 (`docs/reviews/branch_pilot_review_20261021.md`, R1-R3), prima di qualunque collaudo
del pilot.** R1: il log scrive «non definita» per una misura non definita, il summary la conta (`masked_normalized_undefined`), e nella
lettura una perdita normalizzata non definita fra le ultime 100 rende il criterio «PWm ha imparato» non verificabile. R2: i consecutivi
dell'allarme sono controlli consecutivi; un controllo senza misura interrompe la sequenza e si conta; un monitor atteso senza riferimento e'
«non definito» e il non-collasso del braccio non e' verificato (`collapse_monitor.verified`). R3: PWf (e PWl con `branch_monitor`) ha gli
stessi monitor del PWm sugli stessi frame, senza gradiente: uguaglianza esatta col PWm e training identico bit per bit provati nei test. Le
tre regressioni di GPT passano sul ramo; test del 2C, del pilot e dell'identita' 127 PASS. Protocollo aggiornato
(`docs/rami_pilota_obiettivo_v1.md`, ancora PROPOSTA): passa dal `critico` col resto del pilot (launcher, estrazione, verdetto) prima del
collaudo. Nessun job.

**Verifica e integrazione GPT (21/10/2026): R1-R3 del pilot chiusi su d7310be, merge 3a3c595.** Tre regressioni indipendenti passate,
incluse nei 23 test di maschere/trainer; 83 test sul percorso precedente e selezione passati. Verificati anche gli strumenti collegati
dei bracci/diagnostica (143 test, parzialmente sovrapposti; un test rieseguito col contesto Git dello snapshot). SHA preservati,
nessun nuovo lancio o modifica alle soglie; protocollo ancora PROPOSTA, wrapper e campagna da completare da Claude.
**Bracci: JEPA-only seme 1 COLLASSATO (21/10/2026 15:52, allarme firmato, decisione 16).** 61622391 fermato al passo 5.000: «rango effettivo
sotto 10% della dimensione per 3 valutazioni» (studente 35,9 / 32,8 / 31,4 ai passi 4.000 / 4.500 / 5.000, soglia 38,4; il rango scende
in modo continuo da 70,1 al passo 500; rapporto di collasso ~0,08, sopra la soglia di 0,05). Firma dei bracci, punti 6-7: un braccio fermato
dall'allarme e' «collassato» e conta come «non batte A»; `arms_compare` conta collassato il braccio se lo e' un seme. Consuntivo 4,22
GPU-ora (1 h 03 x 4 GPU). Diagnostiche in `results/infra/w1arm_20261021/runs/W1ARM-JEPA-only-30M-s1/`. Il seme 0 (61622389) continua.
**Valutazione di C3 sottomessa (21/10/2026 ~16:20)** dal checkout `w1cal-eval-20261021` (98c5832): guardie 61631403 / 61631404 (senza
afterok: training finiti bene secondo sacct), estrazioni 61631405 (1e-4) / 61631408 (3e-4), sonde 61631407 / 61631410, selezione 61631411.
Tetto 3,0 GPU-ora.
**Bracci: JEPA-only seme 0 COLLASSATO (21/10/2026 16:15, allarme firmato).** 61622389 fermato al passo 7.000: «collasso da query: rapporto
sotto 0.05 per 3 valutazioni» (min fra studente e teacher 0,0485 / 0,0339 / 0,0325 ai passi 6.000 / 6.500 / 7.000; il rapporto scende da
1,64 al passo 500; il rango, sceso a 43 al passo 5.000, risale a 135). Consuntivo 5,79 GPU-ora. **JEPA-only collassa in tutti e due i semi**
(per rango il seme 1, per rapporto il seme 0): «collassato», non batte A (firma, punti 6-7). Costo del braccio 10,0 GPU-ora invece di ~160.
Simone (21/10): «se collassa inutile sprecare ore gpu»: nessuno scancel necessario, l'allarme li ha fermati. Alla valutazione i due run
entrano come `--failed-runs` (niente nodi di valutazione; nel confronto contano collassati dallo STOP). Restano 4 run (A e ancore-only).

**C3 decisa con le regole firmate (selezione 61631411, 21/10/2026 20:23): LR proposto per il 300M = 1e-4.** Entrambi ammissibili (A1-A4):
S 0,1755 (SE 0,0079) per 1e-4 e 0,1716 (SE 0,0068) per 3e-4; differenza accoppiata +0,0040 con SE 0,0030, oltre 1 SE: vince 1e-4 (non
parita'). Per dataset: db2 0,188 / 0,185, db3 0,030 / 0,025, db6 0,344 / 0,339. Pannello A4: massimo 1,101 e 1,103 volte il minimo, una
sola valutazione oltre 1,10 (servono 3 consecutive): ammissibili ma vicini alla soglia. Collasso: rapporto finale 2,78 / 3,42, rango 129 /
92. `results/infra/w1cal_decisioni.json` = C1 finalisti + C2 confermati + C3 `proposed_300m_lr` 1e-4 (merge di `calibration_select.py`);
rapporti e registro in `results/infra/w1cal_c3_20261021/valutazione/`. Guardie 34-35 s, estrazioni 15:53 / 15:52, sonde 1 h 47 / 2 h 02
(in fila), selezione 2 min. **Consuntivo della valutazione: 0,53 GPU-ora** (tetto 3,0) e 30,5 core-ora; **calibrazione 144,7 usate su
220, residuo 75,3**. Prossimo: C4 (300M al LR 1e-4, 0,25 epoche, seme 0): la sonda del 300M (attese 4,2 h) supera le 4 h della seriale,
da spostare su GPU o dividere prima del piano. Descrittivo, nessuna decisione: la S del 100M a 0,5 epoche (17,55) e' sotto quella del 30M a
1 epoca (C1 seme 0: 18,21); esposizioni diverse, la calibrazione sceglie il LR, non la taglia.
**Bracci, 22/10 10:50 (controllo dal vivo):** i 4 run sani (A e ancore-only, semi 0 e 1) ai passi 95.400-96.900 su 98.094 dopo 20 h, rango
144-146 (A) e 139-143 (ancore-only), rapporto 1,8-2,0 (A, in lenta discesa da 2,8) e 1,26 (ancore-only, stabile); fine attesa entro le 11:30.
**Diagnostica dei rami, tranche di Simone (22/10/2026 ~11:15):** 61754844 (B_seed0 P2 RPW0 e ridondanza) e 61754849 (B_seed0 P2 RPW): con
queste il seme 0 e' completo; restano le 10 sonde di B_seed1 e il confronto.

**Bracci: valutazione con le sonde su GPU (Simone, 22/10/2026: «ok», alla proposta «sonde su GPU, come ti consiglio»; interpretazione: via alla
valutazione dei 4 run sani col preventivo presentato, 23,0 GPU-ora attese, tetto 58,0, voce della finestra 1).** Training: A s1 e ancore-only
s0 / s1 COMPLETED 0:0 (11:06-11:14), 98.094 passi in 20,3-20,4 h, 0,744-0,750 s/passo; A s0 in chiusura. Identita' delle `config.json` vere
dei 4 run contro il piano congelato `dbe0f8fe...` col codice di oggi: 102 campi uguali, codice 72cf24a. Spec del confronto e piano della
valutazione in `results/infra/w1arm_20261021/valutazione/` (`arms_plan.py --frozen-plans results/infra/w1arm_20261021 --probes-on-gpu
--failed-runs` per i due JEPA-only collassati): 32 nodi di estrazione e sonda su GPU e il confronto, campagna
`results/infra/w1arm_campaign_eval_20261022.json`; deploy nuovo che contiene la spec, poi piano rigenerato con lo SHA del deploy.
**Voce «Bracci dell'obiettivo» sulla finestra 1:** training tetto 552 (autorizzato dalla sottomissione; speso finora 10,0 per JEPA-only, ~325
attese per i 4 run sani) + valutazione 58 = 610 GPU-ora di tetto.

**C4: piano e preventivo pronti, da autorizzare.** Regola nuova della sonda (`calibration_plan.py`): se la durata attesa supera il limite
della seriale (4 h) la sonda va su un nodo GPU (1 GPU, 6 h; stesse opzioni delle sonde dei bracci); le valutazioni di C1-C3 rigenerate sono
identiche a quelle usate (test). C4: 300M al LR 1e-4 (C3), 0,25 epoche (6.130 passi, warmup 1.962, decay da 4.904), seme 0, un job da
6 h 30 dal checkout di C1-C3 (`w1cal-20261016`, 72cf24a); piani di C1-C3 rigenerati identici. **Preventivo: training 19,9 / 26,0 GPU-ora,
valutazione 5,7 / 9,0 (estrazione 1,5 / 3,0 e sonda su GPU 4,2 / 6,0): ~25,6 / 35,0 GPU-ora (280 ore locali)** sul budget della
calibrazione (144,7 usate su 220): dopo il tetto restano almeno 40,3. Piano e comandi in `results/infra/w1cal_c4_20261022/`.
**Bracci, training finiti (22/10/2026):** A s0 COMPLETED alle 11:21 (98.094 passi, `max_steps`, 20.55 h di training);
con A s1 e ancore-only s0 / s1 i 4 run sani sono completi; summary e config in `results/infra/w1arm_20261021/runs/`. Consuntivo dei tre
run con sacct: 244.2 GPU-ora (4 GPU x 20 h 17-20 h 26); A s0 da sacct alla prossima lettura. Suite 1316 PASS su 23efccb.
**Deploy della valutazione dei bracci:** `deploy/w1arm-eval-20261022` = 23efccb (contiene spec e piano congelato; SHA remoto verificato),
checkout job 61761132; piano rigenerato con quello SHA (spec identica) e dry-run: 4 giri da 8 job GPU (estrazioni e sonde di un run per
giro), il primo con le 4 guardie.
**Valutazione dei bracci: primo giro rifiutato senza danni (22/10/2026 11:45).** La catena e' stata lanciata prima che il checkout
`w1arm-eval-20261022` (job 61761132, in coda per il limite di CPU della seriale) esistesse: sbatch ha rifiutato la prima guardia
(«Unable to open file .../check_checkpoint.sbatch»), la catena si e' fermata, nessun job creato. Recupero previsto dal submitter: a checkout
pronto, la stessa guardia con `--dopo-rifiuto`, poi la catena (che blocca i nodi rifiutati e non li rilancia da sola).
**C4 sottomessa da Simone (22/10/2026), sul preventivo presentato (~25,6 / 35 GPU-ora: training 19,9 / 26, valutazione 5,7 / 9 con la
sonda su GPU), dal checkout `w1cal-20261016` (72cf24a):** 61761908 = `W1CAL-C4-300M-lr0p0001-s0-j01` (300M, LR 1e-4, 6.130 passi, un job da
6 h 30). La valutazione di C4 si prepara a training finito dalla campagna di valutazione (sonda su GPU).
**Valutazione dei bracci, primo giro sottomesso (22/10/2026 ~12:10).** Checkout 61761132 COMPLETED (12:00, 673 file, 23efccb). Guardia di A s0
risottomessa con `--dopo-rifiuto` (61769168); catena: guardie 61769194 / 61769201 / 61769207 (senza afterok, training finiti bene), A s0
61769211-61769260 (estrazioni e sonde su GPU di P1, P2, S della P1 e l'estrazione di S della P2). Il resto (28 nodi, compreso il confronto)
in attesa del limite di 8 job della coda GPU nella catena, che conta anche C4: giri successivi a richiesta. Nessun rifiuto.
**Valutazione dei bracci, secondo giro (Simone, 22/10/2026 ~14:20):** 61789204 (sonda di S della P2 di A s0: A s0 completo) e A s1
61789208-61789264 (estrazioni e sonde di P1, P2, S della P1). In attesa: S della P2 di A s1, ancore-only s0 e s1, confronto (17 nodi).
**Diagnostica dei rami, tranche di Simone (22/10/2026 ~14:25):** seme 0 completo (61754844 P2 RPW0 + ridondanza 54 min, 61754849 P2 RPW
40 min); sottomessi 61789455 (B_seed1 P1 PW0, 4 h) e 61789465 (B_seed1 P1 PW, 4 h). Restano 8 sonde di B_seed1 e il confronto.
**Ultrareview della PR di revisione (22/10/2026).** `/ultrareview 1` lanciata da Simone (crediti d'uso) su simonebenatti/wearusfm#1
(base 79db894, revisione 7cfad41: solo codice). 5 rilievi, tutti minori e verificati: pannello fisso con l'obiettivo mascherato del pilot,
`--done` ignorato senza `--per-set` nella diagnostica, caricatore importlib duplicato, variabile `ids` riusata in `arms_compare`, monitor dei
rami che calcola token inutili. Nessuno tocca run fatti o in corso; correzioni proposte, da fare sul ramo di Claude se Simone le approva.
**Ultrareview, correzioni (Simone, 22/10/2026: «si, procedi»).** Rilievi 1, 2, 4, 5 corretti sul ramo di Claude: pannello fisso rifiutato
con l'obiettivo mascherato del pilot, `--done` rifiutato senza `--per-set`, `run_ids` in `arms_compare`, monitor dei rami senza i token del
lato studente (valori identici). Il 3 resta com'e'. Suite 1317 PASS. Rami `review/` cancellati su GitHub (PR #1 chiusa con essi).
**Stato alle 17:00 (22/10/2026, squeue e sacct dal vivo).** Bracci: A s0 8/8, A s1 6/8, restano 19 nodi (catena ferma dalle 15:17: il
giro delle 15 non l'ho dato). Diagnostica B_seed1: P1 PW0 finita (2 h 05), P1 PW in corso; restano 8 sonde e il confronto. C4 in coda.
**Valutazione dei bracci, terzo giro (Simone, 22/10/2026 ~17:05):** 61813378, 61813380 (S della P2 di A s1: A completo nei due semi) e
ancore-only s0 61813385-61813407 (P1, P2, estrazione di S della P1). Restano 12 nodi, compreso il confronto.
**Diagnostica dei rami (Simone, 22/10/2026 ~17:10):** 61813410 (B_seed1 P1 R). Restano 7 sonde di B_seed1 e il confronto.

**Integrazione GPT (Simone, 22/10/2026: «Verifica e integra su main 23efccb ... e d4f4d23 ... cosi chiudiamo anche 3»).**
Consegna integrata col merge `d8b7170`, senza riscrivere gli SHA. Review GPT: 90 test PASS; suite completa 1317 PASS riportata da Claude.
Codice d'integrazione identico alla consegna verificata. Punto 3 chiuso fino a d4f4d23; il rilievo 3 importlib resta una pulizia non
bloccante. Registro aggiornato con entrambe le tabelle. Priorita' confermata da Simone: prima PSD/wavelet, poi l'eventuale recupero di
JEPA-only. Nessuna nuova soglia, default o budget, nessun job o deploy; i checkout di campagna restano congelati.
**Controllo dell'integrazione (Claude, 22/10/2026, su richiesta di Simone).** main 4680617 contiene 23efccb e d4f4d23 (merge d8b7170,
SHA conservati), codice identico a d4f4d23: punto 3 chiuso fino a d4f4d23. main su leonardo a 8dad7e8, non usato dalle campagne. Ramo di
Claude allineato con un merge di main (bf63bbf).
**Richiesta per GPT (Simone, 22/10/2026: «potresti segnare di chiarire questa cosa poi lo fa lui»).** Nella tabella dei sei punti, chiarire
che il pilot sta nel punto 5 (PSD-2) e che il punto 3 riguarda solo review e integrazione del codice. Registrata nel registro; la fa GPT.
**C4 inammissibile: allarme di collasso al passo 1.500 (23/10/2026).** 61761908 (300M, LR 1e-4) fermato dall'allarme firmato (16): rango
effettivo dello studente 77,6 / 75,0 / 66,7 ai passi 500 / 1.000 / 1.500 contro la soglia 89,6 (10% di 896); 1.500 passi su 6.130.
Con le regole firmate C4 non passa (A1, A3) e la calibrazione si ferma senza sostituire LR ne' allargare la griglia; il LR della campagna
resta una decisione di Simone. Valutazione di C4 non preparata (risparmiate le ~5,7 GPU-ora previste). Consuntivo 4,82 GPU-ora:
calibrazione ~149,5 / 220. Descrittivo per la decisione (`results/infra/w1cal_c4_20261022/diagnostiche_erank_taglie.json`): nei primi 4.000
passi il rango effettivo e' ~67-87 in tutte e tre le taglie, mentre la soglia cresce con la dimensione (38,4 / 64 / 89,6).
**Valutazione dei bracci, quarto giro (Simone, 23/10/2026 ~9:00):** 61919606-61919648 (ancore-only s0 completato, ancore-only s1 P1, P2 e
estrazione di S della P1). Restano 4 nodi, compreso il confronto.
**Diagnostica dei rami (Simone, 23/10/2026 ~9:05):** 61919664 (B_seed1 P1 RPW0 e ridondanza), 61919669 (B_seed1 P1 RPW). Restano le 5 sonde
della P2 di B_seed1 e il confronto.
**Pilot dei rami: codice operativo (23/10/2026, Simone: «no fermarlo, vediamo che succede»).** `scripts/branch_pilot.py`, due launcher,
sonde del pilot nella diagnostica; protocollo con le aggiunte dichiarate (controllo della configurazione del 2C, PWf a un passo nel collaudo,
primario per primo, opzione GPU per la valutazione). Configurazione del 2C riprodotta senza differenze col codice di oggi. Suite 1333 PASS.
Review del `critico` in corso; nessun job, nessuna soglia nuova (la lettura resta quella del protocollo, da congelare col commit del lancio).
**Valutazione dei bracci, ultimo giro (Simone, 23/10/2026 ~9:45):** 61925267, 61925272, 61925282 e il confronto 61925288 (afterok). Tutti i
nodi del piano sottomessi.
**Pilot dei rami (`rami_pilota_v1`): lettura e criteri del collaudo congelati prima del lancio (23/10/2026).** Dal protocollo
(`docs/rami_pilota_obiettivo_v1.md`), con le aggiunte del 23/10 dichiarate li', e col codice `scripts/branch_pilot.py` di questo commit:
- **Primario:** PWf -> PWm su RPW (validation interna, regola dei bracci di `spectral_diagnostic.reading`); positivo = «migliora» in almeno un
  compito e delta >= -0,01 sulla stima puntuale nell'altro. Si legge solo se PWm ha imparato con tutti e due i teacher (media della perdita
  normalizzata sugli ultimi 100 aggiornamenti <= 0,9; una misura non definita, anche di una sola componente ramo x scala, rende il criterio non
  verificabile) e PWf e PWm sono non collassati (`collapse_monitor.verified`) con tutti e due. Braccio collassato = non batte il riferimento.
  Accanto PW0 -> PWm; descrittivi come nel protocollo; nessuna adozione automatica, nessuna variante nuova automatica se il primario non aiuta.
- **Collaudo** (PWm, B_seed0, dbg 30 min; prima PWf per un passo nello stesso job): integrita' (nessuno STOP; configurazione del 2C salvo il
  braccio; raw, passo 0 e strati fissi uguali al 2C; perdita del passo 1 di PWf entro 1e-4 relativo), apprendimento come sopra, monitor
  verified, copertura >= 50% degli aggiornamenti per ramo x scala, allocazione dei run ((avvio + 3.000 x mediana s/passo dal 51esimo) x 1,25 +
  10 min, al quarto d'ora) <= 1 h 30; stabilita' descrittiva. Passa -> preventivo dei sei run e via di Simone; non passa -> stop e proposta.
- **Run:** controlli del passo 0, del raw, degli strati fissi, della perdita del passo 1 (PWl, PWf), configurazione del 2C; un controllo
  fallito = run non valido, nessuna estrazione. Uscite dei rami sempre su CPU (fp32, come il 2C).
- **Preventivo:** training 9,5 GPU-ora di tetto (stima ~4,6) dal residuo 26,21 del passo 6; valutazione su coda seriale ~1.060 core-ora di
  tetto (stima ~475), ~60 h di calendario (~18 h per il primario, per primo). L'opzione su nodo GPU (tetto 134 GPU-ora) supera il residuo:
  solo con una decisione di budget. Review del `critico` del 23/10: DA RIVEDERE (2 bloccanti, 10 minori), tutti corretti in questo commit;
  seconda review prima del collaudo. Nessun job lanciato; il collaudo richiede il via di Simone.
**Pilot dei rami: seconda review del `critico` e soglie dell'allarme congelate (23/10/2026).** La voce precedente diceva «tutti corretti»: la
seconda review (DA RIVEDERE, 2 bloccanti e 7 minori su 4af5715) ha trovato che l'esito «collassato» non scattava su un vero STOP del trainer
(«allarme: allarme di collasso dei rami: ...», non all'inizio della riga) e che il protocollo prometteva per un ibrido mancante un
comportamento che il codice non ha. Corretti nel commit di questa voce, con le minori. Regole che la lettura usa, congelate qui:
- **Allarme di collasso dei rami** (`branch_masking.COLLAPSE`, invariato dal 21/10): per ramo e scala, riferimento = media dei passi 1-10;
  un controllo ogni 10 aggiornamenti dopo i primi 100; allarme e STOP con diversita' < 0,05 x riferimento, oppure varianza assoluta < 0,05 x
  riferimento, oppure rango effettivo < 0,10 x riferimento in 3 controlli consecutivi (una lacuna interrompe la sequenza). Collassato = non
  batte il riferimento; il primario non si legge se PWf o PWm collassa o non e' «verified».
- **Run mancanti o falliti per altro** (compreso un ibrido PWh): nessun esito, stop e proposta. PWl collassato: il confronto completo si lancia
  a mano e salta i descrittivi di PWl.
- **Misura non definita:** fa fede `learned` (media o componente presente non finita); il contatore del trainer conta solo le medie.
**Pilot dei rami: fuga attraverso il preprocessing (23/10/2026, misura preliminare).** Il filtro del loader agisce prima della maschera: con
contenuto nascosto estremo le log-potenze di contesto dello studente cambiano di nat interi anche a 500 ms dal tratto
(`results/rami_pilota/fuga_preprocessing_20261023.json`). Il pilot non si lancia finche' non si decide come mascherare prima del filtro.
**Bracci dell'obiettivo: esito (23/10/2026), default A per la regola firmata.** Nessun braccio batte A su S della P1 e della P2 (JEPA-only
collassato; ancore-only +0,51 punti su P1 con IC sopra zero ma sotto la soglia di 1 punto, -0,40 su P2). Test ufficiali descrittivi: P1 +0,07,
P2 -0,89 per ancore-only. Il JEPA non mostra valore aggiunto misurabile sulle sole ancore a 30M; la scelta resta A per la regola (il riferimento
vince alla pari). `results/infra/w1arm_20261021/valutazione/esito_confronto_objective.json`.
**Pilot dei rami SOSPESO (Simone, 23/10/2026: «si, parti con 1 e 2 e sospendi il pilot»).** Motivi: fuga attraverso il filtro del loader
(serve mascherare prima del filtro: modifica del loader) e risultato dei bracci (il JEPA non aggiunge valore alle ancore). Nessun job del pilot
lanciato; codice e misure restano. Priorita' nuove: (1) regola del rango effettivo e nuova C4 (300M), (2) baseline classica e rete casuale
sulla stessa validazione. Diagnostica PSD/wavelet: si chiude come previsto.
**Allarme sul rango effettivo: soglia assoluta, PROPOSTA (Claude, 23/10/2026; da firmare prima di una nuova C4).** La regola firmata (16)
ferma un run col rango effettivo dello studente sotto il 10% della dimensione per 3 valutazioni: 38,4 al 30M, 64 al 100M, 89,6 al 300M.
Dati (`results/infra/w1cal_c4_20261022/diagnostiche_erank_taglie.json`, `results/infra/w1arm_20261021/runs/W1ARM-JEPA-only-30M-s1/
diagnostiche.json`), ai passi 500-1.500 e oltre:
| run | rango effettivo | min / prima valutazione | rapporto di collasso | varianza fra campioni |
|---|---|---|---|---|
| 30M C2 1e-4 (sano) | 68-125 | 0,95 | 1,87-3,59 | 0,13-0,23 |
| 100M C3 1e-4 (sano) | 72-129 | 0,90 | 1,95-3,74 | 0,08-0,15 |
| 300M C4 1e-4 (fermato) | 67-78 | 0,86 | 1,94-2,35 | 0,07-0,10 |
| 30M JEPA-only s1 (collassato) | 31-70 | 0,45 | 0,08-0,13 | 0,008-0,014 |
Il rango delle diagnostiche non cresce con la larghezza; C4 ha rapporto e varianze dei run sani e scavalca solo la soglia proporzionale.
**Proposta:** soglia assoluta di 38,4 (il valore firmato al 30M) a tutte le taglie (`AlarmRule.erank_min`, `train_jepa.py --erank-min
38.4`); invariati la soglia del rapporto (0,05) e i 3 controlli. Al 30M le decisioni sono identiche a quelle della regola firmata (JEPA-only
fermato allo stesso passo), quindi nulla di retroattivo; al 100M la soglia scende da 64 a 38,4 (C3 mai sotto 71). **Limite dichiarato:** la
proposta nasce dopo aver visto C4 (i dati sopra), quindi va congelata prima della nuova C4 e non salva la C4 fermata, che resta inammissibile.
**Nuova C4:** run nuovo (ID nuovo), stesso piano (300M, LR 1e-4, 6.130 passi, sonda su GPU), ~19,9 GPU-ora attese (tetto 26) piu' la
valutazione (~5,7 / 9): calibrazione da ~149,5 a ~175 / 220. Codice: default `None` = regola firmata (nessun run gia' fatto cambia).
**Baseline classica e rete casuale: protocollo e preventivo PROPOSTI (23/10/2026, Simone: «parti con 1 e 2»).** Lettura descrittiva, nessuna
soglia di adozione. **Classica (`baseline_classica_td_v1`):** le feature di Hudgins del progetto (`harness.features_hudgins`: MAV, ZC, SSC,
WL per canale, soglia di ZC e SSC 0,05 in unita' della scala del soggetto), sullo stesso segnale delle feature del modello (filtro del
pretraining, scala per sessione in P1 e per soggetto in P2, zone di bordo di 0,1 s); differenze dichiarate: finestra intera (il modello
media due ritagli da 4 s), canali scartati dal QC con feature a 0, sonda logistica del modello e non la LDA di v10 §8. Valutazione: la
validazione interna della calibrazione (`scripts/classical_baseline.py`, stessa sonda, fold, soggetti e pesi di `calibration_probe`, provata
uguale nei test; predizioni fuori fold salvate per confronti accoppiati con A) e i test ufficiali (`probe_p1`/`probe_p2 --feature-set td
--checkpoint none`). Si riportano S e accuratezza ufficiale accanto a quelle di A (interna P1 18,31, P2 81,34; ufficiale P1 21,31, P2 85,97).
**Costo:** 2 job CPU seriali (`scripts/slurm/classical_baseline.sbatch`, P1 e P2), 8 core, al piu' 3 h ciascuno: tetto 48 core-ora, 0
GPU-ora. **Rete casuale:** checkpoint con l'architettura di A e pesi dai semi 0 e 1 (`scripts/random_checkpoint.py`; non e' il passo 0 di A,
che non e' registrato), valutato con le stesse estrazioni e sonde dei bracci: piano e preventivo a parte. Review del `critico`: DA RIVEDERE (4
bloccanti, 6 minori), corretti; seconda review prima del lancio.
**Baseline classica: correzioni della seconda review del `critico` (23/10/2026), da leggere con la voce precedente.** (1) Soglia di ZC e SSC
**0** (non 0,05): e' quella di default di `harness.features_hudgins` e di `HudginsLDABaseline`; lo 0,05 della voce precedente non aveva una
provenienza. (2) La ricetta td (feature, soglia, bordi 0,1 s, banda 20-450, QC a 0, finestra intera) e i canali dei dati (uguali al
montaggio e alle altre sessioni) si registrano nella cache e si verificano in `check_cache`. (3) Questa e' la sonda logistica del modello
sulle feature di Hudgins, NON la «LDA su feature di Hudgins» di v10 §8: quella resta dovuta (`harness.baselines.HudginsLDABaseline`). (4)
Lettura **solo affiancata** ad A (S e accuratezza ufficiale); un confronto accoppiato, se servira', con `load_oof` e `paired_difference` dopo
aver verificato che `input_sha256` e `oof_rows_sha256` coincidono con quelli dei rapporti di A (nei test l'impronta degli ingressi della
baseline e' uguale a quella dell'estrazione della calibrazione). (5) Costo atteso ~20 core-ora (~1-1,5 h per job: lettura dei dati e
sonde su poche colonne), tetto 48 core-ora, ore CPU dell'account IscrB_WearUsFM, 0 GPU-ora. (6) `--feature-set td` vuole `--checkpoint none`.
**Baseline classica: terza review del `critico` (23/10/2026).** Bloccante trovato: in `probe_p1` il numero di canali della cache td restava
da un dataset al successivo, quindi la P1 ufficiale si sarebbe fermata su DB6 (16 colonne contro 12 di DB2 e DB3) dopo aver speso DB2 e DB3.
Corretto (canali per dataset) con un test su due dataset di larghezza diversa; provati anche pesi, S e S_se uguali a `calibration_probe` con
due dataset P1, e l'impronta degli ingressi uguale a quella dell'estrazione della calibrazione in P1 e in P2.
**Baseline classica: deploy (23/10/2026).** Controllo in sola lettura del cluster (riferimenti, hook solo d'esempio, campagna libera,
`receive.denyCurrentBranch = updateInstead`: un riferimento `deploy/` non tocca il checkout di `main`); pubblicato
`deploy/baseline-td-20261023` = 56984cf (SHA remoto verificato). Comandi in `results/infra/baseline_td_20261023/comandi.txt`: checkout di
campagna, poi i job p1 e p2 (CPU seriale, tetto 48 core-ora, stima ~20, 0 GPU-ora), rapporti in `$SCRATCH/wearusfm_runs/baseline_td_20261023/`.
**Baseline classica: checkout sottomesso da Simone (23/10/2026):** 61971050 (`BASELINE-TD-20261023-checkout`). I due job dopo
«checkout di campagna pronto».
**Allarme sul rango effettivo e nuova C4: FIRMATO (Simone, 23/10/2026: «firmo,»), sui punti (a)-(d) proposti da Claude dopo la review del
`critico`.** (a) **Regola:** soglia assoluta del rango effettivo 38,4 (il valore firmato al 30M) sia per l'allarme del trainer sia per A3
della selezione, a tutte le taglie; invariati la soglia del rapporto (0,05) e i 3 controlli consecutivi. Limite dichiarato: la soglia vale
per il batch diagnostico attuale (16 finestre, seme del run); se il batch cambia si rivede. Al 30M decisioni identiche alla regola firmata
(16), che questa modifica per le taglie maggiori. (b) **Deroga** al «C4 instabile: ci si ferma» del protocollo C1-C4: UNA sola ripetizione
di C4; se fallisce anche questa, il 300M si ferma definitivamente e il piano W1 si fa fino al 100M. La C4 fermata resta inammissibile e
registrata. (c) **Ambito:** C4 e tutti i run della finestra 1. (d) **Costo:** checkout nuovo (CPU, minuti); C4 tetto 26 GPU-ora (stima ~20),
valutazione tetto 9 (stima ~5,7): calibrazione da ~149,5 a un tetto di 184,5 / 220, residuo 35,5. I primi 1.500 passi della nuova C4
ripetono quelli gia' visti (~4,8 GPU-ora): dichiarato, ripresa dal passo 1.500 scartata (vietata da `_check_resume_recipe`, servirebbe un
cambio di codice). Alternative valutate (relativa, varianza, combinata) nella voce del registro del 23/10. Prossimi passi: pianificatore
(tentativo nuovo di C4 con ID e cartella nuovi e `--erank-min 38.4`), A3 letta dalla configurazione del run, test, `critico`, deploy,
prova d'uguaglianza della configurazione del 300M con C4 (salvo `alarm.erank_min`), comando a Simone.
**Baseline classica: checkout pronto (23/10/2026).** 61971050 COMPLETED 0:0 in 15 s, «checkout di campagna pronto»
(`wearusfm_campaigns/baseline-td-20261023` @ 56984cf, 696 file in sola lettura). Diagnostica: 61919669 (B_seed1 P1 RPW) COMPLETED in 2 h 36;
restano le 5 sonde della P2 di B_seed1 e il confronto, dopo le baseline.
**Rango effettivo: codice della firma del 23/10 (pianificatore e A3).** `calibration_plan.py --c4-repeat` (`C4_REPEAT`: ID, cartelle e campagna
«C4b», `--erank-min 38.4`; i piani di C1-C4 restano identici, provato nei test), `calibration_select.erank_floor` (A3 con la soglia della
configurazione del run, altrimenti la regola (16); chiave del rapporto ora `erank_floor`), test del flag (`--erank-min` 0, negativi, NaN
rifiutati; allarme del rapporto invariato). Correzione di formulazione rispetto alla proposta: il rango non cresce *in proporzione* alla
larghezza nei primi 1.500 passi; piu' avanti il 100M sta sopra il 30M (batch diverso).
**Baseline classica: job P1 sottomesso da Simone (23/10/2026):** 61972380 (`BASELINE-TD-20261023-p1`).
**Baseline classica: job P2 sottomesso da Simone (23/10/2026):** 61972407 (`BASELINE-TD-20261023-p2`). Le due baseline sono in coda.
**Diagnosi del pretraining e idea di soluzione (Claude, 23/10/2026; Simone: «dobbiamo capire perche la parte del tokenizer, degli embedding e
quindi il pretraining non va, e immaginare possibili soluzioni»).** `docs/diagnosi_pretraining_20261023.md`: 12 fatti gia' nel repo (ancore-only
= A; JEPA-only collassato sulla query; +2/+7 punti sul casuale; embedding = statistiche spettrali, 2A e 2C; rango ~70-87 uguale a 30M, 100M e
300M; readout medio sui canali). In una frase: l'obiettivo insegna cio' che le ancore chiedono e il termine JEPA (target b) o collassa o non
aggiunge nulla di misurabile; cause in ordine: scala del target (25 ms), target b, readout medio, esposizione del caso A. Non e' il problema:
front-end, RVQ, LR, taglia (F4: una nuova C4 misura lo stesso obiettivo). Proposta in tre mosse: readout per canale (3.1), pilot degli
obiettivi a 30M/1 epoca (3.3: target a, target per regione, ancora di posa su emg2pose, combinazione; ~76 GPU-ora attese), ancora di posa come
cambio di principio rispetto a v10 §6.1 (decisione di Simone). **Scelta di Simone:** «Parti dal readout per canale sui checkpoint esistenti
(3.1), insieme alla baseline classica; il pilot degli obiettivi dopo».
**Readout per canale sui checkpoint esistenti: protocollo `readout_per_canale_v1` PROPOSTO, lettura CONGELATA con questo commit (Claude,
23/10/2026).** `docs/readout_per_canale_v1.md`; codice `harness/channel_readout.py`, `scripts/channel_readout.py`, launcher
`channel_readout_{extract,probe}.sbatch`; `harness/probe.py` NON toccato (vincolato byte per byte ai rapporti della diagnostica dei rami,
`tests/cpu/test_branch_pilot.py`). Cosa: dalla stessa passata in avanti di A s0 e s1 (post-decay, teacher, gli stessi dei bracci), per
finestra e per canale la media nel tempo dell'encoder locale (`local`) e del decoder a query su ogni (canale, patch) valida (`dec`), piu' il
readout medio (`pool`) come controllo d'uguaglianza; validazione interna della calibrazione (stessi soggetti, finestre, fold, C, pesi e
sonda); a larghezza pari (standardizzazione e PCA a k = 2d = 768 sul solo train del fold, le componenti alla sonda invariata) e, sulla sola
P2, intera. Nessun training, nessun numero di test, nessuna adozione: le sonde ufficiali restano quelle di D12. **Lettura congelata qui,
prima di qualunque sonda e prima di aprire i rapporti della baseline classica (non letti):** regola dei bracci cosi' com'e' implementata
(`arms_compare.s_comparison`: bootstrap congiunto dei soggetti sui quattro rapporti, 1.000, seme 20261009, intervallo dei percentili;
«migliora» = delta medio dei due semi >= +0,01, limite inferiore > 0, delta > 0 in tutti e due i semi; «peggiora» la speculare; altrimenti
«regola non superata»). (1) `local` e `dec` a k 768 contro i rapporti storici di S di A (s0 P1 0,1834 / P2 0,8192; s1 0,1828 / 0,8075; readout
medio), per compito, 4 letture senza correzione: «il readout medio nasconde informazione» se uno strato «migliora» in un compito senza
«peggiorare» nell'altro. (2) Un candidato scelto prima di aprire la baseline: fra `local` e `dec` a k 768 quello con la S della P1 media dei
due semi piu' alta, se supera il `pool` storico (0,1831); altrimenti il `pool`; parita' = `pool`; contro `interna_p1.json` / `interna_p2.json`
della baseline (stesso rapporto per i due semi): «l'encoder batte le feature classiche» solo se «migliora» in tutti e due i compiti
(distorsione dichiarata: la P1 sceglie e confronta). (3) Descrittivi: `local` contro `dec`, larghezza intera sulla P2, varianza spiegata,
rango effettivo, S per dataset. (4) Controllo d'uguaglianza: `pool` di A s0 sulla P1 = 0,1834 entro 1e-3, se no nessuna lettura.
Tranche 2 (rete casuale con l'architettura di A, larghezza intera sulla P1 divisa per dataset) si decide prima di leggere la 1.
**Costo, da autorizzare:** 4 estrazioni GPU (1 A100), tetto 5,0 GPU-ora (40 ore locali), attese ~2,3; 13 sonde CPU seriali da 8 core e 4 h,
tetto 416 core-ore (52 ore locali), attese ~70; ~6 GB su `$SCRATCH`. Voce proposta: la valutazione dei bracci (tetto 58 GPU-ora, Simone
22/10/2026; consuntivo non registrato, stima dal `sacct` del 23/10 ~12 usate, residuo ~46); in alternativa l'infrastruttura (12,78 / 25). Ore CPU
dell'account. Review del `critico`: prima DA RIVEDERE (3 bloccanti: accoppiamento della P2, regola non quella dei bracci, confronto sul Mac;
8 minori), tutti corretti; seconda review in corso alla scrittura di questa voce, esito nel registro. Nessun job lanciato, nessun deploy.
**Sonda senza nuova scala e vincolo su `harness/probe.py`: DECISO (Simone, 23/10/2026: «devi dire all'altro agente che adesso le modifiche
principali sono quelle che fai tu, se modifichi cose poi le riporti a come erano prima perdiamo tempo»).** Il readout per canale usa
`linear_probe(scale=False)` sulle componenti della PCA (nessuna nuova scala, come nel protocollo presentato al `critico`); il parametro e'
aggiunto con default invariato e il percorso di default (pilot, diagnostica, calibrazione) e' provato uguale nei test. Il vincolo byte per
byte di `tests/cpu/test_branch_pilot.py` su `harness/probe.py` passa all'hash del commit di questa voce, con la decisione citata nel test;
gli altri quattro file restano quelli di aba1f2e. L'altra sessione Claude (C4, pilot) e' avvisata: le modifiche all'harness le fa questa.
**Baseline classica, P1: esito (23/10/2026, letto dopo il commit 63ac0dd della lettura congelata del readout per canale).** Job 61972380
COMPLETED in 20:32, 61972407 (P2) COMPLETED in 18:42 (P2 da leggere). Validazione interna S, stesse finestre e stesse righe fuori fold di A
(verificato): **TD 14,30 ± 0,55** (DB2 14,07, DB3 3,02, DB6 32,14; 48/48/64 colonne) contro A s0 18,34 e s1 18,28. Test ufficiale P1: TD 16,17 ±
1,44 (DB2 16,19, DB3 3,14, DB6 35,67) contro A 21,31. Confronto accoppiato descrittivo (regola dei bracci con `channel_readout.py compare`,
bootstrap congiunto dei soggetti, riferimento la baseline per tutti e due i semi): **A - TD = +4,01 punti, IC95 [+2,97; +5,03], per seme +4,04
e +3,98: «migliora»**; per dataset DB2 +5,71 [+4,43; +7,02], DB3 +0,07 [-1,14; +1,26] (tutti e due al caso), DB6 +3,10 [-0,39; +6,73]. Lettura:
sulla P1 l'encoder allenato, letto col readout medio, batte le feature di Hudgins con la stessa sonda; il distacco sta in DB2. Non e' la LDA di
v10 §8 (ancora dovuta) e non cambia nessuna regola. Esito in `~/wearusfm_local/readout_canale_20261023/confronti/td_vs_A_pool_p1.json`
(dati, fuori Git); la copia dell'esito entra in `results/` col commit dei confronti del readout.
**Baseline classica, P2: esito (23/10/2026).** Validazione interna S (soggetti fuori dai test dei tre split, stesse finestre e righe di A,
verificato): **TD 67,48 ± 2,17** (EPN-612 66,61, UCI-EMG 68,35) contro A s0 81,92 (87,34 / 76,49) e s1 80,75 (86,09 / 75,42). Test ufficiale
P2: TD 71,58 ± 3,35 (EPN-612 65,19, UCI-EMG 77,98) contro A 85,97. Confronto accoppiato descrittivo (come per la P1): **A - TD = +13,86 punti,
IC95 [+11,00; +17,03], per seme +14,44 e +13,27: «migliora»**; EPN-612 circa +20, UCI-EMG circa +7,5. Log dei due job senza errori ne'
avvisi di convergenza (P1: 495 avvisi «y_pred contains classes not in y_true», gia' previsti dal protocollo). Lettura dei due compiti insieme:
col readout medio l'encoder allenato batte le feature di Hudgins con la stessa sonda su P1 (+4) e P2 (+14); e la rete casuale dello Step 1
(ufficiale P2 73,1) stava gia' al livello delle feature classiche (71,6): l'architettura col readout medio vale da sola quanto Hudgins, il
training aggiunge il resto. Resta da misurare quanto il readout medio nasconde (readout per canale) e la LDA di v10 §8. Esito in
`~/wearusfm_local/readout_canale_20261023/confronti/td_vs_A_pool_p2.json`.
**Readout per canale: seconda review del `critico` (DA RIVEDERE, 2 bloccanti e 8 minori) e correzioni (23/10/2026), con una voce di
correzione.** Bloccanti: (1) k = 768 non era applicabile su UCI-EMG (~220-240 finestre di train per fold: il codice rifiutava e le quattro
sonde P2 si sarebbero fermate) -> k effettivo = min(k, righe di train - 1), dichiarato nel rapporto; su UCI il readout per canale e' di fatto a
larghezza intera e la «pari larghezza» non vincola nulla; (2) la normalizzazione dei rapporti per `s_comparison` poteva nascondere candidati
dei due semi di tipo diverso (strato, k) o riferimenti uno con modello e uno senza -> controlli espliciti prima della normalizzazione, con
test. Minori corretti: `dec` descritto come target dell'opzione b del teacher esteso a tutte le posizioni (non cio' che leggono le teste delle
ancore, che sono sullo studente mascherato); ore locali (le core-ore sono le ore locali del budget CPU: 416 di tetto e ~70 attese; 52 e 9
erano ore di orologio; vale anche per la voce di 63ac0dd); una sola soglia d'uguaglianza (1e-3 su S) e il comando `--stage equality`; ramo
irraggiungibile tolto e test del «due file dello stesso checkpoint». **Correzioni a 63ac0dd:** la voce del protocollo diceva «`harness/probe.py`
NON toccato» e «componenti alla sonda invariata»: era vero quando e' stata scritta e falso al commit, perche' la decisione successiva nello
stesso commit ha ripristinato `scale`; fa fede la voce «Sonda senza nuova scala ... DECISO». Il messaggio di commit e il registro dicevano «12
test»: `test_channel_readout.py` ne ha 11. La frase «DECISO (Simone ...)» sul vincolo di `probe.py` e' la mia lettura delle parole di Simone
(«le modifiche principali sono quelle che fai tu», «se modifichi cose poi le riporti a come erano prima perdiamo tempo», «tua nel senso che
usi il file probe.py che hai modificato oggi»): *interpretazione*, non una firma sul vincolo; se Simone la smentisce si ripristina l'hash di
aba1f2e e il parametro resta inutilizzato dal pilot. Il punto (4) della voce della baseline classica («lettura solo affiancata; accoppiato
con `load_oof`/`paired_difference` se servira'») e' sostituito dal confronto 2 del protocollo (`s_comparison`, regola dei bracci,
descrittivo). Questi emendamenti sono scritti dopo l'apertura dei rapporti della baseline (P1 e P2) e non cambiano soglie, regola o candidati.
**Opzione proposta (ablation):** ancore-only s0/s1 nella stessa campagna (+4 estrazioni, P1 e P2 per seme, tetto 5,0 GPU-ora, attese ~2,3; +8
sonde, tetto 256 core-ore): se col readout per canale A supera ancore-only, il JEPA aggiunge informazione che la media non mostra. Suite
completa su 63ac0dd: 1369 PASS (21 min 50).
**Readout per canale: AUTORIZZATO (Simone, 23/10/2026: «Via al deploy deploy/readout-canale-20261023, al checkout e alle estrazioni, compresa
l'opzione ancore-only s0/s1 (tetto totale 10 GPU-ora dalla voce della valutazione dei bracci); poi le sonde CPU.»).** Codice deployato
6242bc3 (comandi in `results/infra/readout_canale_20261023/comandi.txt`, sezione «OPZIONE APPROVATA» per ancore-only). Budget: 8 estrazioni
GPU (A s0/s1 e ancore-only s0/s1, P1 e P2), tetto **10,0 GPU-ora** (attese ~4,6) dalla voce della valutazione dei bracci (tetto 58, stima
~12 usate prima di oggi); 21 sonde CPU seriali, tetto 672 core-ore (attese ~110), ore CPU dell'account. Sequenza: preflight in sola lettura,
push del riferimento, checkout di campagna, estrazioni, sonde; uguaglianza e confronti sul Mac. Lettura congelata in 63ac0dd ed emendata in
6242bc3; il confronto A contro ancore-only col readout per canale e' descrittivo (stessa regola dei bracci, riferimento ancore-only).
**Readout per canale: deploy, checkout ed estrazioni sottomessi (23/10/2026).** Preflight pulito; `deploy/readout-canale-20261023` = 6242bc3
(SHA remoto verificato); checkout 61982223 COMPLETED in 13 s; estrazioni A s0 61982369 / 61982398, A s1 61982414 / 61982432, ancore-only s0
61982457 / 61982465, s1 61982480 / 61982484 (P1 / P2; tetto 10,0 GPU-ora). Sonde dopo le estrazioni.
**Readout per canale: estrazioni finite, consuntivo (23/10/2026).** Le 8 estrazioni COMPLETED senza errori: P1 12-14 min l'una, P2 ~28 min l'una;
**2,74 GPU-ora** in tutto (tetto 10,0, attese 4,6), dalla voce della valutazione dei bracci (stima: ~15 usate su 58 dopo oggi). Cache P1 ~2 GB
e P2 per checkpoint su `$SCRATCH/wearusfm_runs/readout_canale_20261023/features/`. Le 21 sonde CPU sottomesse in due tornate (7 dopo le P1,
14 dopo le P2); JobID nel registro. **Limite scoperto: la coda seriale ammette 10 job per utente** (`QOSMaxSubmitJobPerUserLimit`) oltre a
farne girare uno alla volta: 10 sonde in coda, una rifiutata e segnata dal submitter, 11 da sottomettere a tornate man mano che la coda si
svuota. **Controllo d'uguaglianza (punto 4) superato:** `pool` di A s0 sulla P1 = 0,18337 come lo storico, zero predizioni diverse.
**Pilot degli obiettivi, braccio R5 «finestra 150 ms a passo 25 ms»: PROPOSTO e codice pronto (Simone, 23/10/2026: «tipo 150ms con overlap di
125.ms. che ne pensi? va esplorata?»; alla proposta di farne un braccio: «si ottimo!»).** Il kernel del front-end vive su 150 ms centrati sulla
patch; la griglia dei token (e quindi sequenza, maschere, ancore, RVQ) resta a 25 ms; le basi scalano per conservare il contenuto in frequenza
fisico; la perdita conta solo i token interamente nascosti (3 patch per lato). Da firmare per questo braccio: `share_short` 0 (le maschere da
1-2 patch non producono token interi), quote rinormalizzate 0,294 / 0,235 / 0,471: scostamento dichiarato da D10. Prerequisito: gate D8
riaperto per il banco da 150 ms con le soglie firmate (D8a). Default del codice invariato bit per bit (test). Nessun job. Dettagli in
`docs/pilot_obiettivi_v1.md`.
**Pilot degli obiettivi: durata DECISA, 2 epoche per tutti e 4 i run (Simone, 23/10/2026: «scelgo le 2 epoche per tutti i 4 run»).** Run R0 (A
rifatto), R1 (target a), R2 (target per regione), R5 (finestra 150 ms): 30M, seme 0, LR 1e-4, 2 epoche WSD (49.046 passi) piu' un ramo di decay
dal checkpoint a 1 epoca (due endpoint validi per run, ~2,2 epoche di calcolo, ~44 GPU-ora l'uno). Motivo (proposta di Claude accolta): A e'
piatto da 1 a 4 epoche, i bracci nuovi possono essere piu' lenti; due endpoint danno la pendenza. C1 s0 resta il controllo di coerenza
dell'endpoint a 1 epoca di R0. Regola di lettura proposta (da congelare prima del lancio): passa chi all'endpoint a 2 epoche migliora S P1 e
S P2 di +1 punto rispetto a R0 col readout designato; seme 1 solo per chi passa. Preventivo prima ondata: ~176 / 230 GPU-ora di training,
~8 / 16 di estrazioni, ~170 core-ore di sonde; famiglia «Ablation a 30M» o voce nuova (da decidere). Prerequisiti: R2 da scrivere, gate D8
per R5, firma dello scostamento da D10 per R5, review del `critico`, congelamento in questo file. Nessun job.
**Readout per canale, confronto 1 sulla P1, strato `local` (24/10/2026, regola congelata in 63ac0dd): «PEGGIORA».** `local` per canale a k 768
contro il readout medio storico di A, due semi, bootstrap congiunto: delta -2,00 punti, IC95 [-2,60; -1,36], per seme -1,99 / -2,00 (DB2 -3,41,
DB3 -0,26, DB6 +1,07 con intervallo che include lo zero). Sulla P1 cross-soggetto la media sui canali non nasconde informazione utile
dell'encoder locale: la toglie dove e' dannosa (il posizionamento per soggetto). Nessuna adozione. Seguono `dec`, ancore-only e la P2.
**Readout per canale, confronto 1 sulla P1, strato `dec` (24/10/2026, regola congelata): «REGOLA NON SUPERATA».** `dec` per canale a k 768 contro
il readout medio di A, due semi: delta -0,42 punti, IC95 [-1,20; +0,33], per seme -0,28 / -0,56 (DB2 -0,40, DB3 -0,33, DB6 -0,65). Sulla P1
nessuno strato per canale migliora il readout medio: `local` peggiora, `dec` e' alla pari. La domanda del protocollo resta aperta sulla P2.
**Correzione della voce della firma del rango (Claude, 23/10/2026).** La voce «Allarme sul rango effettivo e nuova C4: FIRMATO» rimanda
alle «Alternative valutate (relativa, varianza, combinata) nella voce del registro del 23/10»: nel registro quella valutazione non c'era (le
alternative risultavano «non discusse»). Ora c'e', scritta dopo la firma coi dati gia' nel repo (registro, voce «Rango effettivo: le
alternative coi dati e le correzioni della seconda review»): non cambia la regola firmata. Correzioni della seconda review del `critico`
(DA RIVEDERE, 2 bloccanti di registrazione): costante `W1_ERANK_MIN`; `--c4-repeat` rifiuta codice che non contiene 1b7b1c2; test della C4b
dalla selezione al merge, STOP compreso; comando a mano della selezione per lo STOP ed emendamento in testa a
`docs/calibrazione_c1_c4_selezione.md`; byte esatti del `config.json` della C4. Prossimo: terza review, poi deploy e comando a Simone.
**Readout per canale, opzione ancore-only, strato `local` sulla P1 (24/10/2026, regola congelata, lettura descrittiva per protocollo): «REGOLA
NON SUPERATA».** Riferimento ancore-only s0/s1, candidato A s0/s1, stesso strato `local` a k 768, bootstrap congiunto: delta +0,20 punti, IC95
[+0,01; +0,39], per seme +0,31 / +0,09 (DB2 +0,16, DB3 +0,24, DB6 +0,28, intervalli che includono lo zero). Il JEPA aggiunge alle ancore sole
~0,2 punti anche col readout per canale: il verdetto «ancore-only = A» dei bracci regge. `dec` appena finisce il seme 1. Strumento:
`--stage compare --models` (emendamento 8 del protocollo), aggiunto dopo l'apertura dei rapporti e prima del confronto, dichiarato; regola e
bootstrap invariati. Sonde: 17 job finiti senza errori, 1 in esecuzione, 9 in coda (quinta tornata: ancore-only s1 P2 local), 3 da sottomettere.
**Readout per canale, confronto 1 sulla P2 e risposta alla domanda del protocollo (24/10/2026, regola congelata in 63ac0dd).** `local` a k 768
contro il readout medio di A, due semi, bootstrap congiunto: -2,49 punti, IC95 [-4,59; -0,66], per seme -3,12 / -1,87: «PEGGIORA» (EPN-612
-7,40; UCI +2,41 con intervallo che include lo zero). `dec`: +1,33 punti, IC95 [-0,93; +3,54], per seme +0,72 / +1,93: «REGOLA NON SUPERATA»
(EPN-612 +1,41 [+1,03; +1,83]; UCI +1,25 [-3,26; +5,74]). Quattro letture (P1 e P2 x local e dec): nessuno strato «migliora» in un compito ->
**con la regola congelata il readout medio non nasconde informazione.** Nessuna adozione, nessun cambio del readout di default. Sfumatura
descrittiva registrata (decoder per canale +1,4 su EPN-612 a montaggio condiviso, in perdita sui montaggi per soggetto di DB2/DB6): orienta la
scelta del readout del pilot degli obiettivi, non e' una prova.
**Readout per canale, confronto 2 (24/10/2026):** candidato scelto con la regola del protocollo (S P1 media dei due semi: local 16,31, dec
17,89, pool storico 18,31) = `pool` storico; contro la baseline classica gli esiti del 23/10 (P1 +4,01 [+2,97; +5,03], P2 +13,86 [+11,00;
+17,03], tutti e due «migliora»): «l'encoder batte le feature classiche», descrittivo, nessuna soglia di adozione.
**Readout per canale, opzione ancore-only, strato `dec` sulla P1 (24/10/2026, descrittivo): «REGOLA NON SUPERATA».** A - ancore-only = -0,78
punti, IC95 [-1,12; -0,45], per seme -0,78 / -0,78 (DB2 -1,17 [-1,60; -0,78]). Con `local` (+0,20) e `dec` (-0,78) il verdetto dei bracci
«ancore-only = A» regge col readout per canale. Sonde: tutte le 21 sottomesse (ultime tre 62128193, 62128210, 62128222); 13 finite senza
errori, 8 in coda o in esecuzione, tutte P2 (larghezza intera e ancore-only), descrittive.
**Sintesi «cosa e' successo» al pretraining (Claude, 24/10/2026; `docs/sintesi_pretraining_20261024.md`).** Il passo 1 della diagnosi del 23/10
e' chiuso: l'encoder batte le feature classiche (P1 +4,01, P2 +13,86); il readout medio non nasconde informazione con la regola congelata
(quattro letture, nessuno strato «migliora»); A contro ancore-only col readout per canale: local +0,20, dec -0,78, regola non superata. La
causa 3 (readout) cade come causa principale; restano le cause 1 e 2 (scala del target, target b): *interpretazione*, il collo di bottiglia
piu' probabile e' nell'obiettivo.
Readout per il pilot: `pool` come criterio, `dec` per canale a 768 descrittivo. La C4 a 300M misura lo stesso obiettivo a taglia maggiore:
resta in attesa della decisione di Simone.
**Pilot degli obiettivi: PROPOSTA pronta dopo la review del `critico`, regola di lettura DA CONGELARE (Claude, 24/10/2026;
`docs/pilot_obiettivi_v1.md`; Simone: «scrivi la sintesi e prepara il pilot degli obiettivi (codice R2, pianificatore a 2 endpoint, gate D8 per
R5, regola di lettura, critico); nessun job senza il mio via»).** Bracci R0 (A rifatto), R1 (target a), R2 (target per regione: ogni regione
pesa quanto i suoi token, quote di D10 invariate), R5 (finestra 150 ms); 30M, seme 0, LR 1e-4, `--erank-min 38.4`, tronco a 2 epoche
(49.046 passi, checkpoint tenuto a 19.618) piu' ramo di decay a 1 epoca come in D14 (da 19.618 a 24.523, decay su 4.905 passi: lo schedule di
C1 s0). **Regola di lettura, testo unico col protocollo §5, da firmare:**
«**Regola di lettura del pilot degli obiettivi.** Criterio: per ciascun braccio R_i (R1, R2, R5) e per ciascun compito (P1, P2) della validazione
interna, all'endpoint a 2 epoche (teacher di `checkpoint_postdecay.pt` del tronco, run completo senza STOP), col readout `pool` estratto e
sondato con `scripts/channel_readout.py` (`--stage extract`, poi `--stage probe --layer pool`: la sonda della calibrazione, controllo
d'uguaglianza del 23/10 su P1, A s0), il confronto accoppiato con R0 allo stesso endpoint con `scripts/channel_readout.py --stage compare --single-seed
--models` (soggetti ricampionati insieme, 1.000 ricampionamenti, seme 20261009, intervallo dei percentili, `arms_compare.s_comparison`). R_i
**passa** se in P1 **e** in P2 il delta e' >= +1 punto col limite inferiore dell'intervallo al 95% > 0. Sei letture del criterio (3 bracci x 2
compiti) senza correzione per confronti multipli: dichiarato. Un solo seme: il criterio dell'accordo fra i semi della regola dei bracci non si
applica e il rapporto lo dichiara (`seeds` 1). Un run fermato dall'allarme sul rango (38,4 / 0,05 / 3 controlli) o incompleto non passa.
Provenienza: il rapporto di confronto registra percorso e SHA-256 dei due checkpoint; prima di leggere si verifica che siano i
`checkpoint_postdecay.pt` dei run giusti (R0 e R_i, tronco a 2 epoche) e che `arms_compare.run_status` dia «completo» per entrambi, e lo si
scrive nel registro. **Conferma:** per ogni braccio che passa si lanciano il suo seme 1 e, una volta sola, R0 seme 1 (stesso disegno, tronco
e ramo), e si rilegge con la regola dei bracci intera (due semi: `compare --models` senza `--single-seed`); se piu' bracci passano anche la
conferma, si registra l'ordine per delta medio di P1 e P2 e la scelta del default e' di Simone. **Descrittivi, fuori dal criterio:** `dec` per
canale a 768, l'endpoint a 1 epoca, la pendenza 1 -> 2, le loss e le diagnostiche. Nessun braccio passa -> si registra e il default non
cambia; la seconda ondata la decide Simone. **Non si decide qui:** K, taglie, LR della campagna, default della finestra 1.»
**Scostamento da D10 per R5, da firmare:** `share_short` 0, quote 0,294 / 0,235 / 0,471 (`--mask-shares 0:0.294:0.235:0.471`). **Preventivo
prima ondata** (s/passo 0,7 misurato su A, slack 1,15): collaudo dbg di R2 e R5 2,6 / 4,0 GPU-ora; training 175,6 / 228 (4 tronchi 12 h + 4 rami
2 h 15, un nodo ciascuno); estrazioni ~5,6 / 20; gate D8 ~1 core-ora (tetto 4); sonde ~270 core-ore; **~184 / 252 GPU-ora = ~1.472 / 2.016 ore locali**; conferma del primo braccio che passa (seme 1 suo e di R0, tronco e
ramo) ~88 / 114 GPU-ora (704 / 912 ore locali) piu' ~2,8 / 10 di valutazione, +44 / 57 per ogni altro. Budget: finestra 1 (6.000 GPU-ora firmate
il 16/10, calibrazione compresa); spese registrate finora ~496 (calibrazione ~149,5 con C4; bracci: 4 run sani ~321, JEPA-only ~10, valutazione
~15 col readout per canale): residuo indicativo ~5.500, da confermare col consuntivo prima della firma; voce «Ablation a 30M» o voce nuova
(Simone). **Scostamento da D10 per R5, da firmare anche nella ripartizione della perdita** (solo token interi; misura del `critico` col
generatore vero): media 15,8 -> 10,2%, slab 19,6 -> 14,1%, spaziale 52,3 -> 62,3% dei token su capgmyo (emg2pose e DB2 simili), token interi /
nascosti ~0,83.
Prerequisiti prima dei job: firma della regola e dello scostamento D10, deploy `deploy/pilot-obiettivi-20261024`, checkout di campagna, gate D8
sul banco da 150 ms (NON SUPERATO -> R5 non parte), collaudo dbg (s/passo > 0,81 -> job di scorta), via di Simone. Nessun job.
**Codice del pilot (Claude, 24/10/2026).** R2: `generate_mask_regions` (numero della maschera estratta per token; `generate_mask` invariato
bit per bit), `PretrainBatch.region`, `JEPAConfig.region_target`, `jepa.region_means` (LayerNorm della media del teacher grezzo per regione,
media delle uscite dello studente; medie in float32), `loss_counts` coerente, `--jepa-region-target`; test: default identico, regione per token
= perdita per token, contratto delle loss (numeratori, conteggi, micro-batch), dal loader vero. **Correzione a R5 (problema trovato prima di
ogni job):** `loss_counts` non applicava il raggio dei token interi (`core_radius`), mentre `_jepa_compute` si': col contratto delle loss
distribuite il run R5 si sarebbe fermato al primo passo («conteggi del forward diversi da quelli precalcolati»); corretto con test. Launcher
del gate D8 con la regola CODE_DIR (prima `cd $WORK/wearusfm`). Pianificatore: nessuna estensione necessaria (tronco con `keep_checkpoints`
e ramo `branch_of`, come `w1cat`); piano e preventivo in `results/infra/pilot_obiettivi_20261024/` (SHA segnaposto, comandi da rigenerare al
deploy). Test: nella voce del registro del 24/10 (righe di riepilogo di pytest per file e suite completa sul commit finale).
**Pilot degli obiettivi: review del `critico` (DA RIVEDERE: 4 bloccanti, 9 minori) e correzioni (Claude, 24/10/2026).** Bloccanti:
(1) R2 pesava le regioni alla pari: col generatore vero il 70% della perdita sarebbe finito sulle maschere corte da 1-2 patch e il 2% sulle
spaziali, l'opposto dell'ipotesi e uno scostamento da D10 non dichiarato -> ogni token porta l'errore della sua regione (quote di D10 invariate,
conteggi del contratto uguali a quelli per token), test; (2) il ramo «a 1 epoca» stava a 1,25 epoche (24.523 -> 30.654) contro D14 -> ramo da
19.618 a 24.523 (lo schedule di C1 s0), piano e preventivo rigenerati; (3) la conferma col seme 1 richiede anche R0 seme 1 -> nella regola e nei
costi, col caso di piu' bracci che passano; (4) la sintesi affermava oltre i dati -> «+0,20, intervallo che esclude lo zero, sotto la soglia,
solo P1», interpretazioni marcate. Minori: testo della regola unico nei due file (sei letture = 3 bracci x 2 compiti; allarme; provenienza con
`run_status` e SHA dei checkpoint); `--erank-min 38.4` nel piano; `jepa_hidden_tokens` nel contratto (token per regione anche in DDP); il JSON
del gate registra il banco; test del contratto a micro-batch con le regioni e di R5 con ancore, QC e lunghezze diverse; s/passo di R2 e R5 non
misurati -> collaudo dbg prima del lancio e soglia 0,81 s/passo; strumento unico `channel_readout.py`; gate ~1 core-ora attesa (4 di tetto);
residuo del budget dichiarato; segnaposti del registro riempiti. Esito della seconda review nel registro.
**Pilot degli obiettivi: regola di lettura e scostamento da D10 per R5 FIRMATI; deploy, checkout, gate D8 e collaudo AUTORIZZATI (Simone,
24/10/2026: «Firmo la regola di lettura del pilot (testo del §5) e lo scostamento da D10 per R5. Via al deploy deploy/pilot-obiettivi-20261024,
al checkout di campagna, al gate D8 sul banco da 150 ms e al collaudo dbg di R2 e R5 (tetto 4 GPU-ora). I 4 training aspettano il mio via dopo
gate e collaudo.»).** Firmati: (1) la regola di lettura, cioe' il blocco «Regola di lettura del pilot degli obiettivi» qui sopra (identico al §5
di `docs/pilot_obiettivi_v1.md`), scritta prima di qualunque job: criterio `pool` a 2 epoche, delta >= +1 punto in P1 e in P2 col limite inferiore
dell'intervallo al 95% > 0, un seme dichiarato, conferma col seme 1 del braccio che passa e di R0, sei letture senza correzione, allarme sul
rango = non passa; (2) lo scostamento da D10 per R5: `share_short` 0, quote 0,294 / 0,235 / 0,471 (`--mask-shares 0:0.294:0.235:0.471`) e la
ripartizione della perdita sui soli token interi che ne segue (capgmyo: media 15,8 -> 10,2%, slab 19,6 -> 14,1%, spaziale 52,3 -> 62,3%; token
interi / nascosti ~0,83). Autorizzati ora: deploy del codice (commit di questa voce, stesso codice di 7da3d9e con la suite completa 1387 PASS),
checkout di campagna (CPU seriale, secondi), gate D8 sul banco da 150 ms (CPU seriale, tetto 2 core x 2 h = 4 core-ore, 0 GPU-ora; NON SUPERATO
-> R5 non parte, gli altri tre si'), collaudo dbg di R2 e R5 (2 job su `boost_qos_dbg`, 1.300 passi, 30 min l'uno su un nodo da 4 A100: **tetto
4,0 GPU-ora**, attese 2,6; dentro la prima ondata del pilot, ~184 / 252 preventivati; voce di budget della finestra 1 da nominare al via dei
training). Ordine scelto da Claude dentro l'autorizzazione: checkout -> gate D8 e collaudo di R2 insieme -> collaudo di R5 dopo l'esito del gate
(se NON SUPERATO il collaudo di R5 non si lancia e le sue GPU-ore restano). Soglia del collaudo (congelata il 24/10): oltre 0,81 s/passo il run
passa a `spare_jobs` 1 in `piano.json` prima del lancio. **I 4 training (tronchi e rami) restano in attesa del via di Simone dopo gate e collaudo.**
**Pilot degli obiettivi: deploy e checkout sottomessi (24/10/2026, 18:25).** Preflight in sola lettura: hook del canonico solo `.sample`,
cartella e ref assenti, `main` di leonardo (8dad7e8) non toccato; `receive.denyCurrentBranch` non riverificato oggi (il subagent non esegue
`git` sul login node): vale la verifica del 23/10 e la regola riguarda solo il ramo estratto. `deploy/pilot-obiettivi-20261024` =
**705ec0465420e80d5d818a8bbc7acc3a92c06efc** (SHA remoto verificato; codice del cluster identico a 7da3d9e, suite completa 1387 PASS; il commit
aggiunge la firma e l'agente di monitoraggio). Piani e comandi rigenerati con lo SHA vero (`results/infra/pilot_obiettivi_20261024/`, identici
alle bozze a meno dello SHA). Checkout di campagna `PILOT-OBIETTIVI-20261024-checkout` = job **62148779** (serial, 0 GPU-ora), in coda dietro
le sonde P2 del readout (limite di CPU per utente sulla seriale: una sonda alla volta, ~2,5 h l'una, 1 in corso e 4 in attesa). Il gate,
sottomesso nella stessa tornata con afterok sul checkout, e' stato rifiutato da sbatch (il launcher sta nel checkout non ancora creato; nessun
job, nessun costo): gate e collaudo di R2 si sottomettono dopo «checkout di campagna pronto», il collaudo di R5 con afterok sul gate. Monitoraggio
con l'agente leggero `leonardo-watch` (richiesta di Simone del 24/10; haiku), che sveglia la sessione a job finito. Nessuna GPU-ora spesa. **Coda (18:29):** Simone ha messo in hold le 4 sonde P2 del readout in attesa (62100961, 62128193, 62128210, 62128222) per far passare prima checkout e gate (squeue: `JobHeldUser`); rilascio dopo il gate autorizzato da Simone come eccezione alla regola di leonardo-ops.
**Pilot degli obiettivi: checkout pronto, gate D8 e collaudi dbg sottomessi (24/10/2026, 19:34).** Checkout job 62148779 COMPLETED
(«checkout di campagna pronto: .../wearusfm_campaigns/pilot-obiettivi-20261024 @ 705ec0465420e80d5d818a8bbc7acc3a92c06efc», 719 file).
Seconda tornata con `submit_job.py`: gate D8 a 150 ms = **job 62154852** (`--dopo-rifiuto` sull'ID rifiutato alle 18:25; seriale, 0 GPU-ora);
collaudo di R2 `PILOTC-R2-regione-30M-s0-j01` = **job 62154863** (dbg, 30 min, 1.300 passi, 2 GPU-ora prenotate); collaudo di R5
`PILOTC-R5-finestra150-30M-s0-j01` = **job 62154880** con `afterok:62154852` (parte solo a gate SUPERATO; altrimenti resta
`DependencyNeverSatisfied` a costo zero e si cancella col via di Simone). Prenotate 4,0 GPU-ora dentro il tetto di 4,0 del collaudo, attese
2,6. Alle 19:35 gate e collaudo di R2 erano gia' RUNNING (la sonda 62080282 che occupava la seriale e' finita prima del previsto), R5 in
attesa del gate, le 4 sonde del readout ancora in hold. L'agente di monitoraggio aspetta gate e collaudo di R2; dopo il gate si rilasciano le
4 sonde in hold (eccezione autorizzata da Simone). **I 4 training restano in attesa del via di Simone.**
**Pilot degli obiettivi: collaudo dbg di R2 SUPERATO (24/10/2026, job 62154863).** 1.300 passi completati in 998,8 s su 4 A100 col codice
705ec04 del checkout; mediana **0,706 s/passo** sui passi 51-1300 (soglia congelata 0,81) -> **R2 resta con `spare_jobs` 0**; contratto delle
loss con le regioni rispettato a tutti i passi (`jepa_regions`, `jepa_hidden_tokens`, totale = jepa + 0,2 x ancore entro 1e-6, nessun NaN);
memoria GPU max 11,9 GiB; rango effettivo 66,8-74,0 (minimo 38,4), nessun allarme. Verificatore `scripts/arms_collaudo.py` esteso con
l'obiettivo A (pesi dei bracci del pilot): 8 criteri su 8, esito in `results/infra/pilot_obiettivi_20261024/collaudo/esito_R2.json`. Costo
reale ~1,15 GPU-ora su 2,0 prenotate (consuntivo con sacct alla chiusura). Gate D8 in corso; R5 parte dopo il gate; **i 4 training restano in
attesa del via di Simone.**
**Pilot degli obiettivi: gate D8 sul banco da 150 ms SUPERATO (24/10/2026, job 62154852), sonde del readout rilasciate, collaudo di R5 in
corso.** Gate dal checkout 705ec04 (44 min sulla seriale, ~1,5 core-ore su 4 di tetto, 0 GPU-ora): 6 casi su 6 superati con le soglie
firmate il 01/10 (errore relativo <= 5%: misurato 0,0000 a 1 kHz e 0,0010-0,0012 a 200 Hz; sonda <= 55%: 0,500-0,5025); il JSON registra
`frontend.window_ms` 150 e `bank` «finestra del kernel di 150 ms a passo 25 ms (R5)», archiviato in `results/step3/gate_62154852.json`.
Quindi **R5 e' ammesso al pilot**: il suo collaudo (62154880) e' partito da solo alle 20:20 per la dipendenza afterok. Le 4 sonde P2 del readout
(62100961, 62128193, 62128210, 62128222) sono state rilasciate alle 20:23 dalla sessione principale con `scontrol release`, per l'eccezione
autorizzata da Simone il 24/10 (il subagent leonardo-ops l'ha rifiutato, come da sua regola): tornate in coda normale. **I 4 training restano
in attesa del via di Simone.**
**Pilot degli obiettivi: collaudo dbg di R5 SUPERATO (24/10/2026, job 62154880); gate e collaudi chiusi; consuntivo; via ai training da
chiedere a Simone.** R5: 1.300 passi in 1.012 s su 4 A100 col codice 705ec04, `window_ms` 150 e quote 0 / 0,294 / 0,235 / 0,471 come firmato;
mediana **0,709 s/passo** sui passi 51-1300 (soglia 0,81) -> **anche R5 resta con `spare_jobs` 0**; contratto coi token interi rispettato a tutti
i passi (totale = jepa + 0,2 x ancore entro 1e-6, nessun NaN); memoria GPU max 16,3 GiB; rango effettivo 71,7-81,7 (minimo 38,4), nessun allarme;
verificatore 8 criteri su 8 (`results/infra/pilot_obiettivi_20261024/collaudo/esito_R5.json`). **Consuntivo (sacct):** gate ~1,5 core-ore su 4
di tetto e 0 GPU-ora; collaudi 1,138 + 1,149 = **2,287 GPU-ora su 4,0 di tetto** (attese 2,6). `piano.json` invariato. **Da decidere (Simone):**
via ai 4 training (`comandi_training.txt`: 4 tronchi + 4 rami, 175,6 GPU-ora attese, tetto 228, 12,8 GB di checkpoint) e la voce di budget
dentro la finestra 1 (famiglia «Ablation a 30M» oppure una voce nuova). Fino al via nessun job di training viene sottomesso.
**Pilot degli obiettivi: via di Simone ai 4 training (25/10/2026); voce di budget: finestra 1, famiglia «Ablation a 30M».** Testo di
Simone: «Via ai 4 training del pilot degli obiettivi (8 job di comandi_training.txt, 175,6 GPU-ora attese, tetto 228), voce di budget: finestra
1, famiglia «Ablation a 30M».» Si sottomettono gli 8 job di `results/infra/pilot_obiettivi_20261024/comandi_training.txt` (4 tronchi da 48
GPU-ora prenotate + 4 rami da 9, afterok sul tronco; 228 GPU-ora di tetto, 175,6 attese; 12,8 GB di checkpoint su $SCRATCH) sul checkout di
campagna 705ec04, nell'ordine del file e fermandosi al primo non «sottomesso». Il consuntivo si fa con `sacct` a run finiti e si scrive qui;
la lettura degli esiti segue la regola del §5 di `docs/pilot_obiettivi_v1.md` firmata il 24/10. Nessun altro job oltre gli 8 senza un nuovo via.
**Pilot degli obiettivi: 8 job sottomessi il 25/10/2026 alle 08:04 (tronco R0) e 08:07 (gli altri 7)** (tronchi 62233554, 62233766, 62233768, 62233770; rami `-e1` in afterok sul proprio tronco: 62233765, 62233767, 62233769, 62233771),
tutti sul checkout di campagna 705ec04, QoS `normal`, 4 A100 per job; 228 GPU-ora prenotate (175,6 attese) sulla voce **finestra 1, famiglia
«Ablation a 30M»**. Ledger e `squeue` in `~/wearusfm_local/` (file nel registro). L'agente leggero aspetta i tronchi; il consuntivo e la
lettura (regola del §5, a 2 epoche sui rami) seguono a run finiti. Nessun altro job oltre questi 8 senza un nuovo via.
**Readout per canale: campagna CHIUSA (25/10/2026; Simone: «Sì: leggi le 4 sonde P2 del readout e la sonda 62080282 e chiudi il readout per
canale, senza toccare i training.»).** Letture finali con la regola congelata in 63ac0dd: A contro ancore-only sulla P2 (opzione del protocollo,
descrittiva, `--models`): `local` +0,65 [-0,01; +1,26], `dec` -0,46 [-1,52; +0,47], «regola non superata» in tutti e due; con la P1 (+0,20 /
-0,78) il verdetto «ancore-only = A» regge col readout per canale su entrambi i compiti e gli strati. Larghezza intera sulla P2 (descrittivo):
-0,01 / +0,06 punti contro k 768 (la PCA a 768 non costa niente a livello di compito); contro il readout medio le letture del confronto 1 non
cambiano (`local` -2,51 «peggiora», `dec` +1,39 regola non superata). Le tre risposte della sintesi del 24/10 restano: l'encoder batte le
feature classiche; il readout medio non nasconde informazione; il JEPA non aggiunge piu' di qualche decimo di punto lineare alle ancore.
Consuntivo: 30 job COMPLETED, **2,74 GPU-ora** (tetto 10) e **238,9 core-ore** (tetto 672; attese ~110: le sonde P1 a k 768 sono durate ~2 h 20
l'una invece di ~50 min). Esiti in `results/readout_canale_20261023/` (README con la tabella dei verdetti). Nessun altro job; il pilot degli
obiettivi non e' stato toccato.
**Pilot degli obiettivi: preventivo della valutazione (25/10/2026, in attesa del via).** Comandi in
`results/infra/pilot_obiettivi_20261024/comandi_valutazione.txt`, costi in `preventivo_valutazione.json`, dal checkout di campagna 705ec04
(`channel_readout.py` fra 6242bc3 e 705ec04 cambia solo in `compare`; `load_model` ricostruisce il modello dal checkpoint, `window_ms` compreso).
Blocchi, ciascuno col via di Simone: **A1 = criterio del §5** (endpoint a 2 epoche dei 4 tronchi: 8 estrazioni GPU 2,8 attese / 10,0 di tetto
GPU-ora + 8 sonde `pool` 46 / 256 core-ore; risposta ~6-7 h dopo la sottomissione perche' la seriale fa girare una sonda alla volta); **A2**
descrittivo `dec` (8 sonde, 88 / 256 core-ore); **B** endpoint a 1 epoca dei rami (8 estrazioni 2,8 / 10,0 GPU-ora + 16 sonde 134 / 512
core-ore); **C** confronti sul Mac (6 letture del criterio con `--single-seed --models`, R0 riferimento, dopo il controllo di provenienza e di
`run_status`). Totale 5,6 / 20,0 GPU-ora e 269 / 1.024 core-ore = la riga «valutazione» del §7 del protocollo; voce di budget: finestra 1,
famiglia «Ablation a 30M». Consuntivo parziale dei training: tronco R1 COMPLETED per `max_steps` in 9 h 40 (38,6 GPU-ora su 48 prenotate).
**Pilot degli obiettivi: via di Simone alla valutazione, tutti i blocchi (25/10/2026, dopo i 4 tronchi COMPLETED per `max_steps`).** Simone:
«seguo il consiglio, via a tutti!» (consiglio: i descrittivi A2 e B sono gia' nel protocollo e costano 2,8 GPU-ora in piu'). Autorizzati i blocchi
A1 (criterio), A2 e B di `results/infra/pilot_obiettivi_20261024/preventivo_valutazione.json`: 16 estrazioni GPU, 5,6 attese / 20,0 di tetto
GPU-ora; 32 sonde CPU, 269 / 1.024 core-ore; voce di budget: finestra 1, famiglia «Ablation a 30M». Sequenza: A1 subito, A2 a estrazioni di A1
COMPLETED, B a rami COMPLETED. Tronchi: R0 10 h 03, R1 9 h 40, R2 10 h 03, R5 10 h 13, tutti `max_steps`, ~160 GPU-ora (consuntivo `sacct` a
rami finiti). Lettura del criterio con la regola del §5 (firmata il 24/10), dopo il controllo di provenienza e di `run_status`.
**Pilot degli obiettivi: training COMPLETATO, consuntivo `sacct` 176,5 GPU-ora (25/10/2026).** Gli 8 job (4 tronchi a 2 epoche, 4 rami a
1 epoca) sono COMPLETED per `max_steps`, nessun allarme: tronchi 159,9 GPU-ora (R0 40,2, R1 38,6, R2 40,2, R5 40,9; attese 156,4, tetto 192),
rami 16,6 (attese 19,2, tetto 36), totale **176,5 contro 175,6 attese e 228 di tetto** (`results/infra/pilot_obiettivi_20261024/consuntivo_training.json`).
Voce di budget: finestra 1, famiglia «Ablation a 30M». La valutazione (blocchi A1, A2, B1, B2 autorizzati da Simone il 25/10) e' in corso:
criterio A1 in coda sulla seriale, risposta attesa la mattina del 26/10.
**Pilot degli obiettivi: criterio letto (26/10/2026, 00:24): NESSUN BRACCIO PASSA; il default non cambia; la seconda ondata la decide Simone.**
Regola §5 firmata il 24/10 applicata dopo il controllo di provenienza e di `run_status` (`results/pilot_obiettivi_20261024/provenienza_criterio.json`:
4 tronchi «completo», checkpoint giusti e distinti). Sei letture `pool` a 2 epoche contro R0 (`--single-seed --models`, 1.000 ricampionamenti,
seme 20261009), punti di S: R1 -0,76 [-1,25; -0,28] su P1 e -1,27 [-2,63; +0,15] su P2; R2 -0,27 [-0,72; +0,16] e +0,55 [-1,53; +2,66];
R5 -0,53 [-1,07; -0,02] e +0,35 [-2,14; +2,68]: nessuna con delta >= +1 e limite inferiore > 0, nessuna «peggiora». Conseguenze fissate dalla
regola: nessuna conferma col seme 1; il default della finestra 1 (obiettivo A, bersaglio di default, banco a 1 s) resta; non si decide qui K,
taglie o LR. I descrittivi (`dec` k 768, endpoint a 1 epoca, pendenza 1 -> 2) e il consuntivo `sacct` della valutazione seguono a code vuote.
Dettaglio in `results/pilot_obiettivi_20261024/README.md` e nel registro (voce del 26/10).

**Seconda ondata del pilot degli obiettivi: PROPOSTA di R3 (ancore + testa di posa su emg2pose, senza JEPA) e preventivo; regola di lettura e due deroghe DA FIRMARE (Claude, 26/10/2026; `docs/pilot_posa_v1.md`, `results/infra/pilot_posa/preventivo.json`).**
Simone, 26/10: «Sì, prepara la proposta della seconda ondata con R3 (ancore + testa di posa su emg2pose, senza JEPA) e il preventivo; nessun job
senza il mio via.» Disegno: come il pilot (30M, seme 0, tronco a 2 epoche + ramo a 1 epoca), riferimento R0 del pilot gia' fatto, testa lineare
di posa sulla media dei latenti del backbone per istante, target = 20 angoli di emg2pose standardizzati per giunto (mocap a 60 Hz interpolato a
2 kHz; fallimenti IK = righe tutte a zero), peso 1,0, `--jepa-weight 0`; R3c = ancore-only a 2 epoche come controllo consigliato e separabile.
Tre review del `critico` (la prima DA RIVEDERE con 8 bloccanti e 9 minori, tutti ripresi; la seconda DA RIVEDERE (3 bloccanti: condizione (b) del §3 incompatibile col codice previsto, cancello del collaudo mal posto, margine della regola di validita' non misurabile; 6 minori), la terza DA RIVEDERE con un solo bloccante (la (b), riscritta attorno a tre controlli d'uguaglianza: estrazione bit per bit da un checkpoint di riferimento, sonda con la stessa S, confronto del pilot riprodotto identico) e 5 minori (atteso del cancello, margine della discesa, perimetro della registrazione, testata, ore dello split), tutti ripresi nella versione committata; il `critico` rivede protocollo e codice insieme prima di qualunque job (§6, punto 3)) e un giro di fact-checker
(`docs/fatti_da_verificare.md`, riga 28). **Soglie congelate con la proposta, da firmare insieme alla regola:** peso della posa 1,0, testa
inizializzata a zero; cancello del collaudo dbg letto da `metrics.jsonl` sui passi con conteggio di posa > 0: (1) media dei passi 1-100, posa
pesata >= 0,1 x ancore pesate (x 0,2); (2) media dei passi 500-1.300, posa pesata <= 2 x ancore pesate; (3) media del termine di posa sui passi
1.201-1.300 <= 0,9 x media sui passi 500-599; (4) conteggio di posa > 0 in almeno il 90% dei passi 1-1.300; s/passo al collaudo <= 0,81, altrimenti
piano rigenerato con un job di scorta (+51 GPU-ora di tetto); standardizzazione per lato se per almeno un giunto |media sinistra - media destra|
> 0,5 deviazioni standard congiunte o rapporto fra le deviazioni dei due lati > 1,5, altrimenti insieme; regola di validita' dei campioni da
congelare qui dopo il collaudo dell'ingest (nucleo: riga coi 20 angoli tutti a zero non valida; margine ai bordi misurato dal profilo di
|differenza prima|). **Costi:** solo R3 ~49,7 / 67,0 GPU-ora e ~79 / 275 core-ore; con R3c ~98,1 / 132 GPU-ora e ~149 / 531
core-ore; job di scorta +51 di tetto solo se il collaudo da' s/passo > 0,81; conferma, solo se R3 passa, 94 / 120 + 2,8 / 10 GPU-ora e 141,6 / 512
core-ore; voce: finestra 1, famiglia «Ablation a 30M» (~530 GPU-ora spese finora dalla famiglia; residuo indicativo della finestra ~5.300, da
confermare col consuntivo); 4-5 giorni di calendario dal via (5-6 con R3c). **Da firmare prima di qualunque job:** (1) la regola di lettura qui
sotto; (2) deroga a v10 §6.1 («Tutti i target sono osservativi, calcolati dal raw»: la posa e' misurata dal motion capture); (3) deroga a v10
§6.5/§3.7 (peso di loss sul termine, ritardo zero senza lead time: il backbone e' bidirezionale); (4) registrazioni a mano non sollecitata tenute
nel termine e nelle statistiche (23% delle ore di emg2pose nel pretraining); (5) con o senza R3c; (6) dopo il collaudo dell'ingest: regola di
validita' dei campioni e standardizzazione per lato o insieme. **Non si decide qui:** K, taglie, LR, default della finestra 1, R4, A + posa.
Nessun job, deploy o codice fino alla firma. Testo della regola, identico al §5 del protocollo:

«**Regola di lettura della seconda ondata (R3).** Criterio: per ciascun compito (P1, P2) della validazione interna, all'endpoint a 2 epoche
(teacher di `checkpoint_postdecay.pt` del tronco, run completo senza STOP), col readout `pool` estratto e sondato con `scripts/channel_readout.py`
(`--stage extract`, poi `--stage probe --layer pool`), il confronto accoppiato di R3 con **R0 del pilot degli obiettivi allo stesso endpoint**
(`PILOT-R0-A-30M-s0`, `checkpoint_postdecay.pt`, SHA-256 registrato in `results/pilot_obiettivi_20261024/provenienza_criterio.json`) con
`scripts/channel_readout.py --stage compare --single-seed --models` (soggetti ricampionati insieme, 1.000 ricampionamenti, seme 20261009,
intervallo dei percentili, `arms_compare.s_comparison`). R3 **passa** se in P1 **e** in P2 il delta e' >= +1 punto col limite inferiore
dell'intervallo al 95% > 0. Due letture del criterio senza correzione per confronti multipli: dichiarato. Un solo seme: il criterio dell'accordo
fra i semi non si applica e il rapporto lo dichiara (`seeds` 1). Un run fermato dall'allarme sul rango o incompleto non passa. Provenienza: prima
di leggere si verifica che i due checkpoint siano i `checkpoint_postdecay.pt` dei run giusti (R0 del pilot e R3, tronco a 2 epoche), che
`arms_compare.run_status` dia «completo» per entrambi, che lo SHA di R0 coincida con quello registrato dal pilot e che le condizioni di
comparabilita' del §3 del protocollo (commit della firma) siano verificate: percorso di default bit per bit identico senza le opzioni della posa
(test); i tre controlli d'uguaglianza superati (estrazione identica bit per bit da un checkpoint di riferimento, sonda con la stessa S e zero
predizioni diverse, confronto R1 contro R0 del pilot riprodotto identico col codice di R3); `git diff` fra 705ec04 e il deploy di R3 sulla
chiusura degli import di estrazione, sonda e confronto vuoto fuori dai quattro file dichiarati nel §3 e, dentro quelli, limitato alle aggiunte
della posa con le funzioni usate dall'estrazione e dal confronto identiche; stesso manifest, scale, finestre e seme; ambiente registrato nel log
del job di R3. Tutto si scrive nel registro; una condizione che manca blocca la lettura. **Conferma:** se R3 passa si lanciano R3 seme 1 e R0 seme 1 (stesso disegno,
tronco e ramo), e si rilegge con la regola dei bracci intera (due semi: `compare --models` senza `--single-seed`); se anche la conferma passa, la
configurazione candidata al default della finestra 1 e' R3 cosi' com'e' (ancore 0,2 + posa 1,0, senza JEPA) e la decisione e' di Simone.
**Descrittivi, fuori dal criterio:** `dec` per canale a 768, l'endpoint a 1 epoca, la pendenza 1 -> 2, i confronti con R3c (R3 contro R3c, R3c
contro R0) se R3c e' stato lanciato, l'errore di posa sul pannello fisso, le loss e le diagnostiche. R3 non passa -> si registra e il default non
cambia; la mossa successiva la decide Simone. **Non si decide qui:** K, taglie, LR della campagna, default della finestra 1, R4, A + posa.»

**Seconda ondata del pilot degli obiettivi: regola di lettura, deroghe, soglie e mano non sollecitata FIRMATE; CON il controllo R3c; via al codice e ai test; nessun job senza il via (Simone, 26/10/2026).**
Testo di Simone: «Firmo la regola di lettura del §5 di docs/pilot_posa_v1.md, le deroghe a v10 §6.1 e §6.5/§3.7, le soglie congelate e la scelta sulla mano non sollecitata; seconda ondata CON il controllo R3c. Via al codice e ai test; nessun job senza il mio via.» Restano congelati il testo della regola e le soglie della voce precedente («Seconda ondata del pilot degli obiettivi:
PROPOSTA di R3 ...»); la regola di validita' dei campioni di posa si congela qui dopo il collaudo dell'ingest, con la firma di Simone, prima
dell'ingest intero. Budget della seconda ondata con R3c: ~98,1 GPU-ora attese, tetto 132,0 (piu' eventuale job di scorta +51 di tetto), ~149 /
531 core-ore; voce: finestra 1, famiglia «Ablation a 30M». Ogni job si sottomette solo col via di Simone.
