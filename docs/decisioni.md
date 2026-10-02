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

---

## Tre decisioni del 02/10/2026 (Simone: «ok ti do il via», sulla proposta di AG)

1. **emg2pose: le 3.539 registrazioni (58,8 h) del test ufficiale per fasi nuove** (`split = test`, `held_out_stage = True`, di utenti NON tenuti
   fuori) **escono dal pretraining** e restano per la valutazione, come gli utenti tenuti fuori: il benchmark ufficiale resta pulito. Lo split
   diventa per sessione oltre che per soggetto (`test_sessions` negli split).
2. **emg2qwerty: si scrivono nei sidecar i 189 tratti costanti** misurati dal job 59099080 (`qc_relative_check --only emg2qwerty --apply`, copia di
   sicurezza, dati intatti). **Fatto** (job 59133395, 1 h 42 min, exit 0): 189 tratti scritti in 14 sessioni, 312,5 canale-secondi
   (0,0008% del tempo valido), 0 canali piatti nuovi; report copiato con sha256 uguale.
3. **DB10 ed emg2pose: misura dei tratti costanti**, sola lettura (`qc_relative_check --only ninapro_db10 --only emg2pose`); una scrittura, se serve,
   si chiede a parte. Primo tentativo (job 59133397) in **TIMEOUT** a 2 h senza finire emg2pose (25.253 sessioni, 92 GB):
   nessun report, nulla scritto (sola lettura). Rilancio separato: DB10 da solo; emg2pose da solo con 4 h (massimo della coda seriale).

---

## Note dalla review dei documenti (02/10/2026)

- **Cancellazioni:** il 01/10 Simone ha scritto «se devi cancellare files e cose gia copiate e duplicate , ti autorizzo». AG **non** ha cancellato
  nulla: la cancellazione definitiva di dati e' fuori da cio' che AG fa anche con autorizzazione; gli originali in `$WORK/data/raw` si cancellano solo
  con `scripts/slurm/delete_verified_raw.sbatch` lanciato da Simone (`dataset_access.md`).
- **Autorizzazioni di misura e di scrittura:** «autorizzo i buchi su Zhang ed emg2qwerty» (01/10) e' stata letta come misura; la scrittura dei 189
  tratti di emg2qwerty e' stata autorizzata a parte il 02/10.
- Le correzioni puntuali segnalate (citazioni, numeri, interpretazioni marcate) sono in questo commit; la bozza D9 si riscrive coi numeri del costruttore.

