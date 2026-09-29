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

**Aperto, da scrivere QUI prima del lancio di 1-bis** (non sono soglie nuove, sono le
definizioni operative che le rendono misurabili): la metrica di errore di ricostruzione di V2,
il livello di rumore "al noise floor" di V4 e il baseline delle potenze di banda di V3.

D5b (esito: ancora adottabile, ristretta ad alcuni dataset, o scartata) resta da prendere
entro l'11/10; con queste soglie e' meccanica, la firma e' di Simone. Se il 25/10 le verifiche
non sono complete, l'ancora RVQ esce dal manifest (piano §10).

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
vista canonica. Decisa solo *la scelta di normalizzare*. **Il tipo di normalizzazione NON e'
ancora deciso**: va scritto con le definizioni operative di D5a, prima del lancio di 1-bis.
*Proposta di AG, da confermare (non congelata):* un **fattore di scala unico per registrazione**
(deviazione standard calcolata su tutti i canali insieme), non uno z-score per canale e non per
finestra. Motivi: il tokenizer stesso standardizza congiuntamente su canali, patch e tempo
(`std_norm`, `NeuroRVQ.py` righe 571-575) e il transformer usa i rapporti di ampiezza fra canali
come informazione spaziale; una normalizzazione per finestra cancellerebbe la dinamica (i picchi
sono informativi). **Da misurare in 1-bis:** l'ampiezza di emg2pose nelle sue unita' non e'
documentata nel codice, quindi va verificato che l'ingresso normalizzato cada nel regime che
l'encoder congelato ha visto (errore di ricostruzione su emg2pose con e senza normalizzazione).

**Non cambia:** il passo 5 riproduce i numeri pubblicati di NeuroRVQ con **la loro** pipeline
(nel paper: 20-90 Hz per NeuroRVQ nel confronto downstream); e' una pipeline di valutazione, non
la vista canonica del tokenizer. La definizione di "banda condivisa" di D8a resta da scrivere.
