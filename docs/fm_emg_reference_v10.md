# Foundation model per sEMG — documento di riferimento (v10)

Sintesi delle decisioni prese, delle questioni aperte e del razionale tecnico.
Pensato per essere incollato come contesto all'inizio di una nuova conversazione, o passato
come contesto a un agente.

**Convenzione.** «da verificare» marca un fatto riportato a memoria o da fonte secondaria: va
trattato come **compito di verifica, non come dato**. L'elenco completo è in §12.

**Rev. 1 (18/09/2026).** Conversione del budget in ore locali, scadenza e quota mensile (§1,
§10.4, §10.5). Calendario a ritroso e decisioni operative: `docs/piano_operativo_v10.md`.

**Rev. 2 (18/09/2026).** Numeri da `saldo -b`: disponibili ~70.000–90.000 GPU-ora, non
~50.000 (§1, §10.5). Di conseguenza: passi 7–9 e ladder lanciati in parallelo nelle due
finestre di inizio mese (§9), 3 seed per rung (§10.2), **vertice a 3–5B da «non si esegue» a
speculativo e parallelo** (§10.4), asse dei soggetti a due taglie (§10.6).

**Rev. 3 (18/09/2026).** Dalla revisione della spec del passo 0: il campione è il **montaggio
intero** e la **classe di quota è del montaggio** (§2.8, §3.4); il benchmark del dataloader ha
**due assi, non tre bracci** (§4.6); i kernel flash non accettano il bias geometrico: encoder
locale con attenzione sui vicini (§5.4); Hyser e putEMG con dati raccolti da fonte ufficiale
(§2.1, §2.2). Spec: `docs/dataloader_bench_spec.md`.

**Novità rispetto alla v9:**
- §1 **compute aggiornato** (rev. 2: ~70.000–90.000 GPU-ora) · **scope corretto** (il cingolo
  scapolare non è coperto da nessun dataset) · rung a 3–5B **condizionale**
- §2 **NinaPro DB9 rimosso** (solo cinematica) · frequenze native corrette · montaggio NinaPro
  riconosciuto come **misto** · dataset di Kaifosh et al. come voce da decidere · **costo in
  soggetti** dell'esclusione di EPN-612 · «zero-shot» → **transfer su dataset mai visto**
- §3.4/§4.5 **topologia dichiarata per gruppo di canali**, non per dataset
- §4.2/§7.1 **gate del front-end riscritto**: consistenza al ricampionamento; il probe
  dataset-ID diventa diagnostica relativa · fattore Δt · impronta 50/60 Hz
- §4.5/§5.2 **tassonomia anatomica gerarchica** con precisione per canale, funzione atlante
  per anelli e fasce, embedding additivi e dropout del livello muscolo: **questione chiusa**
- §4.6 **packing** come terzo braccio del benchmark del dataloader
- §5.6 **decoder a query e collocazione dei target JEPA**: nuova questione aperta,
  prioritaria, con la diagnostica del **collasso da query** (§7.1)
- §6.3 **ancora RVQ riscritta**: fatti sul checkpoint, soluzione per la frequenza
  (**chiusa**), quattro verifiche preliminari a costo quasi zero, e il ripiego «intra-canale»
  **sostituito** perché non ferma il leakage di fase
- §8/§11 **confrontabilità con NeuroRVQ corretta**: i loro numeri sono fine-tuning completo
- §9 passi **1-bis** e **9-bis**, nota sull'avvio
- §10.4 **vertice condizionale** (rev. 2: speculativo e parallelo) · §10.5 contabilità
  rifatta · §10.6 **asse dei soggetti**, nuovo
- §12 questioni chiuse e nuove aperte, più l'**elenco dei fatti da verificare**

---

## 1. Obiettivo e vincoli

**Obiettivo.** Foundation model per sEMG, architettura transformer-based, pre-addestrato su
un corpus ampio ed eterogeneo di dataset pubblici (HD-EMG incluso, task di regressione e di
classificazione). Deve produrre una rappresentazione su cui agganciare teste semplici per
downstream task generici.

**Principio guida trasversale.** Il modello non deve essere una scatola nera scollegata
dalla realtà del segnale. Vincoli fisici e geometrici vanno inseriti per costruzione dove
possibile, e la rappresentazione va **verificata** contro osservabili fisici (§3, §7).

**Scope.** Arto superiore — **in pratica avambraccio e polso**, più bicipite e tricipite dove
i montaggi NinaPro li includono. Il cingolo scapolare, dichiarato nella v9, **non è coperto da
nessun dataset del corpus** ed esce dallo scope rivendicato. **Unica eccezione dichiarata:**
Camargo 2021 (arto inferiore) resta come **caso di controllo anatomico** (§2.3), non come
pretesa di generalità multi-regione. L'estensione ad altre regioni corporee è lavoro futuro.

**Target di deployment.** Montaggi sparsi e fascette lineari — protesi, clinica, HCI.
L'HD-EMG è un lusso disponibile in training, non il caso d'uso.

**Taglia target.** Ordine dei miliardi, design per 1–5B. Decisione presa consapevolmente:
classe di taglia deployabile via distillazione e silicio dedicato; volontà di superare di
ordini di grandezza i FM EMG esistenti; evidenza interna, in altri domini, che modelli più
grandi ricostruiscono meglio i contesti. **La controevidenza è forte e va affrontata nel
paper, non scoperta in review** (§11). Il rung a 3–5B è **condizionale** (§10.4): le GPU-ora
ci sono; i costi veri sono il tempo di ingegneria e la priorità in coda.

**Licenza e uso.** Il corpus include dataset sotto CC-BY-NC-SA 4.0 (emg2pose, emg2qwerty) e
sotto CC BY-NC 4.0 (putEMG).
L'uso è **esclusivamente non commerciale e di ricerca**; i pesi sono strumentali ai numeri
del paper e non sono destinati a rilascio pubblico. Se un rilascio dovesse diventare
rilevante, la clausola ShareAlike va riesaminata prima, non dopo. Il deployment industriale
citato come motivazione della taglia è **direzione tecnica, non percorso di rilascio di
questi pesi**.

**Compute.** **~70.000–90.000 GPU-ora** su Leonardo (CINECA), progetto ISCRA-B
`IscrB_WearUsFM`. Da `saldo -b` del 18/09/2026: totale 2.000.000 ore locali (250.000 GPU-ora),
residuo **794.303 ore locali ≈ 99.300 GPU-ora**; sul Booster 1 GPU-ora = 8 ore locali, se per
GPU si chiedono al più 8 core e un quarto della RAM del nodo. Chi condivide l'account consuma
oggi ~5.500 ore locali al giorno e smetterà a breve: la forbice dipende da quando. Partizione
`boost_usr_prod` (4× A100 SXM4 64 GB per nodo). Programma completo stimato fra ~17.000 e
~37.000 GPU-ora, a seconda dell'MFU e del vertice (§10.5): **le GPU-ora non sono il vincolo**.
Il margine va speso prima in seed, controlli 2D e asse dei soggetti (§10.6), poi nel vertice
speculativo (§10.4).

**Calendario.** Il budget **scade il 7 gennaio 2027** e le ore non spese si perdono. La quota
mensile dell'account (164.383 ore locali ≈ 20.500 GPU-ora) fa perdere priorità man mano che
la si consuma, e la priorità piena torna il primo del mese: i lotti grossi si sottomettono
nelle **finestre di inizio novembre e inizio dicembre**. **Dopo i dati, il secondo vincolo
reale è il tempo**: calendario a ritroso, finestre e tagli pre-decisi in
`docs/piano_operativo_v10.md`.

**Vincolo reale.** I dati. Corpus pubblico stimato ~2.200 ore dopo le esclusioni, e
**nell'ordine di 600 soggetti** (stima, da ricalcolare all'ingest). **Regime data-bound**, e
con ogni probabilità **subject-bound** (§10.6). Il fallimento atteso non è underfitting, è
collasso o degenerazione della rappresentazione.

---

## 2. Corpus

### 2.1 Incluso — dati e caratteristiche

| Dataset | Scala | Topologia | Note |
|---|---|---|---|
| emg2pose (Meta) | 193 sogg., 370 h, 16 ch @ 2 kHz | anello | + hand pose da mocap |
| emg2qwerty (Meta) | 108 utenti, 346 h, 2×16 ch @ 2 kHz | anello | bracciali **speculari** |
| NinaPro DB2/3/4/6/7/8/10 | ~80–100 h | **mista**: anello a 8 + mirati (§3.4) | Delsys/Cometa, 2 kHz nominali; **DB8 ~1111 Hz, DB10 ~1926 Hz (da verificare)**. DB3/DB7 amputati (anche DB8/DB10: da verificare) |
| NinaPro DB1 | 100 Hz | sparsa | **inviluppo RMS Otto Bock** → percorso §2.4 |
| NinaPro DB5 | 200 Hz, 2× Myo | anello | 8-bit, banda ridotta |
| GRABMyo | 43 sogg., 3 giorni, **2048 Hz**, 16+12 ch | anello (fasce su avambraccio e polso) | multi-giorno |
| putEMG (+ Force) | 44 sogg., 24 ch (3 fasce da 8, a 45°, primo elettrodo sull'ulna), **5120 Hz**, **monopolare** | anello (3 fasce) | label di forza |
| EPN-612 | 612 sogg., Myo 8 ch @ 200 Hz | anello | **escluso dal pretraining** (§2.3) |
| CapgMyo (DBa) | 128 ch @ 1 kHz, **18 sogg.** (corretto il 23/09/2026 - non 23: verificato contando i file .mat reali del dataset scaricato, DBa specificamente, 1440 file = 18 sogg. × 8 gesti × 10 trial) | griglia 2D | HD |
| CSL-hdemg | 168 ch @ **2048 Hz**, 5 sogg. | griglia 2D | HD |
| Hyser | 20 sogg. × 2 sessioni, **256 ch** (4 griglie da 64) @ **2048 Hz** | griglia 2D | il più grande per volume |
| Camargo 2021 | 25 sogg. (AB06-AB30, corretto il 23/09/2026 - non 22 come riportato da una fonte terza non ufficiale), 11 muscoli @ 1 kHz | **sparsa anatomica** | **controllo anatomico** (§2.3) |
| Zhang et al. 2026 (Groningen) | 64 sogg., 14 gesti × 10 ripetizioni, mano non dominante, 8 ch @ 4 kHz (4x Avanti) + 4 ch @ 2 kHz (1x Quattro) | **mista, appaiata**: anatomica sparsa (8 muscoli mirati) **e** anello equidistante (8 elettrodi, 40% lunghezza avambraccio dal gomito) sullo **stesso soggetto, stessi gesti** | aggiunto il 23/09/2026; confronto diretto sparso-vs-anello raro in letteratura, utile all'asse di ricerca sulla topologia del montaggio (§2.6-§2.8); CSV, licenza dataset CC BY 4.0 (indipendente dalla CC BY-NC-ND dell'articolo) |
| UCI-EMG (Lobov) | 8 ch | anello | **escluso dal pretraining** (§2.3) |

**NinaPro DB9 rimosso.** Sono i dati cinematici ricalibrati dei 77 soggetti già rilasciati in
DB1, DB2 e DB5: nessun sEMG nuovo. Promemoria generale: la sovrapposizione di soggetti fra i
DB NinaPro **va controllata all'ingest**, perché tocca sia gli split per soggetto (§8) sia il
conteggio di D (§10.3).

**Il montaggio NinaPro (Delsys/Cometa) è misto, non sparso puro.** Nel montaggio standard a 12
elettrodi, 8 sono equispaziati attorno all'avambraccio all'altezza dell'articolazione
radio-omerale — un anello rado — e gli altri sono mirati su flessore ed estensore superficiale
delle dita, bicipite e tricipite. Le varianti (DB6 a 14 elettrodi, DB8 a 16: da verificare)
vanno mappate all'ingest. Conseguenza di schema in §3.4 e §4.5.

**Da decidere: dataset rilasciato con Kaifosh et al. (Nature 2025)** — gesti discreti,
scrittura, polso; stesso bracciale Meta a 16 canali, 2 kHz. È la **prima colonna della tabella
EMG di NeuroRVQ** (§11), quindi serve comunque all'harness del passo 5. Opzioni: tenerlo tutto
come benchmark mai visto, oppure usarne una parte dei soggetti in pretraining. Aggiunge
soggetti ma non topologie (§2.6). Accesso, licenza e dimensioni: **da verificare**.

### 2.2 Accesso, licenze e dimensioni — verificato

| Dataset | Accesso | Licenza | Dimensione |
|---|---|---|---|
| emg2pose | S3 diretto, no registrazione | **CC-BY-NC-SA 4.0** (+ dipendenze NC) | **431 GB** |
| emg2qwerty | S3 diretto; **repo archiviato dal 1/8/2026** | **CC-BY-NC-SA 4.0** (LICENSE da verificare: cita NC-4.0) | ~346 h, 1.136 file |
| Hyser | PhysioNet, open access | CC BY 4.0 | **135 GB compressi / 143 GB** |
| GRABMyo | PhysioNet, open access | CC BY 4.0 | 9,1 GB / 9,4 GB |
| EPN-612 | Zenodo, libero | da verificare | 5,5 GB |
| CapgMyo | figshare, libero | da verificare | 1,3 GB |
| CSL-hdemg | **richiesta via email** con affiliazione | da verificare | >2 GB |
| putEMG | libero via cloud PUT Poznań | **CC BY-NC 4.0** (raccolto dalla pagina ufficiale, da firmare) | non indicata |
| NinaPro DB1–8, DB10 | **account e accettazione dei termini** | da verificare | non indicata |
| UCI-EMG | UCI ML Repository, libero | da verificare | 17 MB |
| Camargo 2021 | Mendeley Data, 3 parti, libero | CC BY 4.0 | non indicata |
| Kaifosh et al. (se adottato, §2.1) | da verificare | da verificare | da verificare |
| Zhang et al. 2026 (Groningen, anatomico vs equidistante) | Dataverse.nl, libero, nessuna registrazione | CC BY 4.0 (verificata sulla pagina del dataset, indipendente dalla licenza CC BY-NC-ND dell'articolo) | 1.245 file CSV/XLSX/TXT, dimensione totale non dichiarata (singoli file sensore ~50-58 MB) |

**Due avvertenze sulla tabella.** Sulle pagine di istruzioni NinaPro compare una credenziale
`reviewers`/`rev2019`: è una cortesia per i revisori dei paper, **non la procedura di
accesso**, che richiede account e accettazione dei termini. E il repository emg2qwerty è
archiviato in sola lettura: il bucket S3 dovrebbe restare, ma non c'è manutenzione futura,
quindi **va scaricato presto**.

### 2.3 Esclusi dal pretraining, ed esclusi del tutto

**Esclusi dal pretraining, per rendere verificabile il claim di transfer su dataset mai
visto** (§8, asse 4):
- **EPN-612** — Myo 200 Hz, 8 canali, 612 soggetti
- **UCI-EMG** — piccolo, ed è già benchmark in NeuroRVQ. **Attenzione alla confrontabilità:**
  i loro numeri EMG sono **fine-tuning completo** con una testa lineare e split per soggetto
  7:1:2, non encoder congelato. Il confronto valido è solo quello prodotto ri-valutando il
  loro checkpoint col protocollo di §8, **non la loro tabella** (§11)

**Nota terminologica.** Un probe allenato sul dataset target **non è zero-shot**: è *transfer
su dataset mai visto in pretraining*. «Zero-shot» resta riservato ai casi senza alcuna
etichetta del target (es. classi di gesto condivise fra dataset), se mai verranno testati.

**Costo dell'esclusione, da dichiarare.** EPN-612 ha 612 soggetti: circa quanti ne ha tutto il
resto del corpus (§1). Tenerlo fuori dimezza la diversità di soggetti del pretraining. La
scelta resta — un benchmark Myo mai visto vale il prezzo — ma il prezzo va **quantificato**
con l'asse dei soggetti di §10.6.

**Escluso del tutto: Gait120.** Fornisce 12 canali EMG processati come inviluppo, non raw.
Non è un problema di banda ridotta ma di **osservabile diverso** (§2.4). Non vale il costo di
tassonomia, QC e metadati per un dataset che può alimentare una sola ancora su tre.

**Escluso del tutto: NinaPro DB9.** Solo cinematica, degli stessi soggetti di DB1, DB2 e DB5
(§2.1).

**Camargo 2021 resta, con un ruolo dichiarato.** È l'unico caso nel corpus in cui il vicinato
metrico è realmente vuoto: 11 muscoli distribuiti sull'arto inferiore, dove ciò che identifica
un canale è solo l'identità del muscolo. Serve come **controllo del percorso anatomico**
(§3.5, §10.1). Costa poco (22 soggetti) e senza di lui l'ablation sui percorsi di identità
perderebbe il suo caso più netto.

### 2.4 Dataset a inviluppo — percorso separato, non input

**Regola.** Un dataset fornito come inviluppo (raddrizzamento + passa-basso a pochi Hz) **non
entra nel front-end**. Riguarda NinaPro DB1 e chiunque altro nella stessa condizione.

**Perché non come input.** L'inviluppo non è una codifica alternativa dello stesso segnale,
è una proiezione a perdita non invertibile. Sopravvive solo l'ampiezza a bassa frequenza;
spariscono contenuto spettrale, fase, ritardi inter-canale e coerenza in banda EMG. In più il
raddrizzamento è non lineare e **non commuta con il banco di filtri** di §4.2: non esiste una
griglia su cui valutare `k(t)` per ottenere qualcosa di confrontabile con il raw. Ammetterlo
in input richiederebbe un percorso dataset-specifico, cioè esattamente ciò che §4.2 esiste per
evitare e che le sonde di §7.1 sorvegliano.

**Dove entrano invece.** L'inviluppo è un osservabile che il modello **già calcola**: è
l'ancora RMS di §6.3. Quindi questi dataset alimentano la **testa dell'ancora RMS** con
target reali, senza mai entrare nell'input. Il front-end resta uno solo, le sonde di §7.1 non
si sporcano, e soggetti e movimenti altrimenti assenti danno supervisione a una testa su tre.

**Conseguenza contabile:** non contribuiscono a `D_c` (§10.3), non aiutano JEPA, non aiutano
le ancore spettrali e spaziali.

### 2.5 Logistica di scaricamento

**~600 GB prima di contare putEMG, NinaPro e Camargo**, dominati da emg2pose (431 GB) e
Hyser (135 GB). Contro `$WORK` da 40 TB la capienza non è un problema; il tempo lo è.

**Vincoli operativi su Leonardo:**
- I nodi di calcolo **non hanno rete**. Si scarica solo dal login node.
- Il login node ha un limite di **10 minuti di CPU time**: un download da centinaia di GB va
  lanciato dalla partizione `lrd_all_serial` (4 h di walltime), non da shell interattiva.
- Destinazione `$WORK`, mai `$HOME` (quota 50 GB, già saturata una volta) né `$SCRATCH`
  (pulizia automatica a 40 giorni).

**Ordine di priorità:**
1. **emg2qwerty** — repo archiviato, rischio di sparizione più alto
2. **CSL-hdemg** e **NinaPro** — richiedono email o account, hanno latenza umana: avviare subito
3. I piccoli (UCI-EMG, CapgMyo, EPN-612, GRABMyo) — utili per far girare la pipeline presto —
   più il **checkpoint NeuroRVQ** da HuggingFace (centinaia di MB): serve al passo 1-bis
   (§9), e anche questo si scarica solo dal login node
4. **emg2pose** e **Hyser** — grossi, in background su `lrd_all_serial`
5. Dataset di Kaifosh et al., **se adottato** (§2.1): prima verificare accesso, licenza e
   dimensioni

**Da calcolare prima di scaricare tutto:** la dimensione della copia preprocessata in int16
memory-mapped (§4.3). Se grezzo + preprocessato sfiora il terabyte, conviene processare e
cancellare il grezzo per i dataset più grossi invece di tenere entrambi.

### 2.6 Razionale sull'eterogeneità

Con il corpus centrato su una regione corporea, la regola di conteggio cambia.

- **Eterogeneità di sensore → invarianza, non informazione.**
- **Eterogeneità di topologia di montaggio → informazione vera** (anello, griglia, sparso).
- **Eterogeneità di popolazione → informazione vera** (normodotati, amputati, multi-giorno).
- **Eterogeneità di spazio di output → informazione vera** (gesti, pose, forza, testo).
- **Eterogeneità di regione corporea → fuori scope**, salvo il controllo di §2.3.

**Regola pratica: contare topologie, popolazioni e spazi di output — non i bracciali.**

### 2.7 HD-EMG — percorso unico

**Un solo percorso, che accetta anche le griglie complete.** Il sottocampionamento a
montaggi bipolari virtuali (≤32 canali) è **augmentation stocastica**, non riduzione
obbligatoria.

Scartate: due percorsi separati (creerebbero cammini dataset-specifici); training ausiliario
su griglie piene (secondo obiettivo da bilanciare, rischio di specializzazione del ramo).

Non costa: il bottleneck Perceiver è indipendente da *C*, l'attenzione ai vicini è O(C·k).

**Batching: padding + maschera**, mai bucketing per numero di canali **fra batch**.
Alternative da misurare al passo 0 (§4.6): il **packing**, che conserva la miscela di topologie
per batch, e il **bucketing per micro-batch con accumulo del gradiente**, che la conserva per
passo di ottimizzazione. Misurare il secondo non impegna; adottarlo modifica questa regola ed è
una decisione da prendere coi numeri in mano.

### 2.8 Quote di campionamento — il caso sparso è prioritario

Il caso sparso **non è il ramo degenere: è il caso di deployment.** Quota **garantita per
topologia**, non lasciata alle proporzioni naturali del corpus. Le quote fanno parte del
**manifest**, congelato lungo tutta la ladder (§10.3).

**La classe di quota è del montaggio, non del gruppo di canali.** Un montaggio misto conta per
ciò che è in deployment: NinaPro (anello a 8 + mirati) è **caso sparso**, come nella v9. Tre
classi: **A** radi/anatomici (NinaPro, Camargo) · **B** anelli e fasce (bracciali Meta, DB5,
GRABMyo, putEMG) · **C** griglie HD. Anche i **pesi dentro la classe** (per ore, per soggetti o
per dataset) sono un parametro del manifest, da fissare al passo 4.

---

## 3. Vincoli fisici e simmetrie

**Criterio: simmetrie e geometria della misura SÌ, modelli generativi del segnale NO.**

### Escluso esplicitamente

Decomposizione in unità motorie come task o input, forme d'onda MUAP come basi imposte,
simulatori biofisici, dati sintetici. **Il corpus resta interamente reale.**

Se la velocità di conduzione emerge da una sonda lineare senza essere stata insegnata, è un
risultato. Se gliel'hai messa dentro, non lo è. **Vincola anche la scelta delle ancore
(§6.3), inclusa quella RVQ.**

*Zona grigia da non far scivolare dentro:* augmentation prodotta da un modello generativo
addestrato su dati reali non violerebbe l'esclusione dei simulatori biofisici, ma
introdurrebbe segnale non misurato nel corpus. Se mai diventasse rilevante, è una decisione
da riprendere esplicitamente.

### 3.1 Geometria nell'attenzione tra canali

L'ampiezza decade con la distanza e il volume conduttore agisce da **filtro passa-basso
spaziale**. Bias di attenzione tra canali funzione della distanza inter-elettrodica — **dove
una distanza esiste** (§3.4). Forma del bias: §5.3.

### 3.2 Guadagno — doppio percorso

**Il logaritmo non produce invarianza al guadagno**: trasforma una moltiplicazione in un
offset additivo, che va comunque rimosso o modellato.

- **percorso normalizzato** → morfologia, spettro, relazioni spaziali;
- **token/feature separata** per scala globale e dinamica d'ampiezza.

| Grandezza | Natura | Trattamento |
|---|---|---|
| Scala di sessione da hardware | nuisance puro | **rimossa, MAI data in input** |
| Ampiezza fisiologica entro sessione | segnale | preservata dalla normalizzazione a scala condivisa (§4.3) |
| Ampiezza calibrata (%MVC), dove fornita | fisicamente interpretabile | **input esplicito** |

Dare in pasto la scala di sessione non calibrata sarebbe un identificatore quasi univoco di
sessione. Nei metadati serve un campo **calibrazione disponibile / assente**.

### 3.3 Polarità — canonicalizzazione, non invarianza

**Il flip indipendente per-canale è escluso.** Le relazioni di fase e segno **tra** canali
contengono informazione su propagazione e direzione delle fibre.

1. Orientamento noto (griglie HD, specularità di emg2qwerty) → **canonicalizzare**.
2. Orientamento ignoto → al massimo **flip globale di montaggio**, mai per-canale.
3. Griglie HD usate per la velocità di conduzione → **nessun flip**.

### 3.4 Topologie di montaggio — tre casi, tre simmetrie

D_n è il gruppo di simmetria del **poligono regolare**, cioè dell'anello. Non delle griglie:
quelle hanno *meno* simmetria, perché la direzione delle fibre rompe l'isotropia.

| Topologia | Simmetria reale | Struttura rilevante | Esempi |
|---|---|---|---|
| **Anello** | ciclica + riflessione (**D_n**) | spostamento angolare relativo | Myo, DB5, EPN-612, emg2pose, emg2qwerty, **anello a 8 di NinaPro** |
| **Griglia 2D** | traslazione lungo gli assi; **asse fibre privilegiato** | spostamento 2D relativo | CapgMyo, CSL-hdemg, Hyser |
| **Sparsa / anatomica** | **nessuna** | identità del muscolo | Camargo, **elettrodi mirati di NinaPro**, protesi, clinica |

Su montaggi sparsi permutare i canali **non è una simmetria, è un errore**.

**La topologia si dichiara per gruppo di canali, non per dataset.** Il montaggio NinaPro a 12
elettrodi (§2.1) è il controesempio: 8 canali formano un anello rado (simmetria D_8,
spostamento angolare relativo), gli altri sono mirati su muscoli noti (nessuna simmetria,
identità del muscolo). Un montaggio è un **insieme di gruppi di canali**, ciascuno con la
propria topologia, geometria e simmetria dichiarata (§4.5). Vale anche per i doppi bracciali
(DB5, emg2qwerty) e per le fasce multiple (putEMG, GRABMyo).

È un **metadato** per encoding posizionale e simmetrie: **il campione che entra nel modello
resta il montaggio intero**, con tutti i suoi gruppi. Spezzarlo per gruppo toglierebbe al
modello proprio la sinergia fra i gruppi. La classe di quota si assegna al montaggio (§2.8).

**Principio: la simmetria è un dato dichiarato nei metadati per montaggio, non
un'assunzione architetturale.** L'architettura deve essere *symmetry-capable* senza essere
*symmetry-imposing*.

**Meccanismo:** un encoding posizionale **puramente relativo in uno spazio metrico** fa
emergere il gruppo giusto in ciascun caso — angolare per gli anelli, 2D per le griglie,
inesistente per i montaggi sparsi (⇒ §3.5).

**Riflessione (anelli):** il mirroring L↔R è augmentation legittima, ma **inverte la
chiralità della propagazione**, quindi serve **equivarianza, non invarianza**. Flag di
chiralità nei metadati.

### 3.5 Identità di canale — doppio percorso

Su montaggi sparsi ciò che identifica il canale non è *dove sta* ma **su quale muscolo sta**.

- **percorso geometrico relativo** — distanze e spostamenti nello spazio metrico del sensore
- **percorso anatomico** — identità anatomica **gerarchica** (regione → compartimento →
  muscolo), al livello di precisione **realmente noto per canale**, da una **tassonomia
  condivisa** (§4.5); dominante sui montaggi sparsi

**Combinazione: concatenazione come default** (§5.2), gating sulla topologia come braccio di
ablation (§10.1).

**Nota sulla forza del test, dopo la restrizione di scope.** Sull'arto superiore i vicinati
metrici non sono mai vuoti: le distanze inter-elettrodiche sull'avambraccio sono dell'ordine
di pochi centimetri, quindi il percorso geometrico ha sempre qualcosa da dire e il percorso
anatomico parte svantaggiato. **Camargo 2021 è l'unico caso in cui il vicinato è davvero
vuoto**, ed è il motivo per cui resta nel corpus (§2.3). Se il percorso anatomico non mostra
guadagno, è un risultato negativo onesto, non un bug. Il test su Camargo **richiede il
livello muscolo**: a livello di compartimento più canali collassano sulla stessa etichetta e,
con vicinato vuoto, diventano scambiabili (§4.5).

### 3.6 Tre sistemi di coordinate — da separare nei metadati

- **sistema del sensore** (indice di canale, riga/colonna, angolo sul ring)
- **coordinate anatomiche** (muscolo, compartimento, lato, orientamento delle fibre)
- **trasformazioni da riposizionamento** (rotazione, traslazione, chiralità)

Una permutazione ciclica ideale e uno spostamento fisico reale del bracciale **non sono lo
stesso fenomeno**.

### 3.7 Vincoli temporali EMG→movimento

**Ritardo elettromeccanico**: l'attività elettrica precede la forza di ~30–100 ms.

- Teste di regressione con **lead time appreso ma vincolato** in intervallo fisiologico.
- Il passaggio drive neurale → forza è un **passa-basso** con costanti di decine-centinaia
  di ms. Vincolare il decoder a essere causale e passa-basso è prior gratuito. **Non** un
  modello di Hill completo.

---

## 4. Front-end, preprocessing, ingest

### 4.1 Principio: scala temporale fisica comune

**Una rappresentazione comune non richiede una frequenza di campionamento comune; richiede
una scala temporale fisica comune. La patch si definisce in millisecondi, non in campioni.**

- **HD e dati per velocità di conduzione: nativi**, 2 kHz / 2048 Hz. A 1 kHz la
  discretizzazione è 1 ms, confrontabile con i ritardi da misurare.
- **Dati convenzionali: 1 kHz.**
- **Myo (200 Hz): banda limitata**, senza fingere che l'upsampling generi informazione.
- **Inviluppo: fuori dal front-end** (§2.4).

Le frequenze native del corpus sono più di tre — 200, 1000, ~1111, ~1926, 2000, 2048,
~5120 Hz (alcune da verificare, §2.1). Al principio non cambia nulla: è il motivo per cui
esiste.

### 4.2 Front-end a kernel continui

Proiezioni lineari **separate** per 200 Hz / 1 kHz / 2 kHz offrirebbero percorsi
dataset-specifici.

**Soluzione: banco di filtri continui parametrizzati in tempo fisico**, valutato sulla
griglia di campionamento del dataset. Un kernel `k(t)` campionato a `t = 0, 1/fs, 2/fs, …`:
**i pesi sono gli stessi, cambia solo la griglia di valutazione.**

**Parametrizzazione: basi eterogenee, decisa.** Il banco è composto da tre famiglie di
kernel concatenate lungo i canali di uscita — base di Fourier, base a spline, kernel
parametrizzati da MLP su `t` (stile CKConv). Ogni famiglia ha proprietà spettrali e di
regolarizzazione diverse; il modello usa quelle che servono. Il costo parametrico è
trascurabile rispetto al backbone e non introduce ambiguità di attribuzione: è un banco
multi-base, non tre architetture alternative.

**Requisito di anti-aliasing.** Il kernel va passabassato alla Nyquist della frequenza target
prima del campionamento, **per ciascuna famiglia di base**. Fisicamente corretto — il Myo
davvero non vede sopra i 100 Hz.

**Fattore Δt.** La somma discreta che approssima l'integrale di convoluzione va **scalata per
Δt = 1/fs**. Senza, l'ampiezza in uscita cresce con la frequenza di campionamento e il
front-end consegna al resto del modello un identificatore di dataset gratuito.

Front-end separati per frequenza restano come **controllo in ablation** (§10.1).

**Gate associato: consistenza al ricampionamento** (§7.1). Il probe dataset-ID subito dopo il
front-end **non può fare da gate**: distingue Myo, Delsys e griglia monopolare quasi al 100%
con qualunque kernel, perché l'identità della sorgente è **fisicamente nel segnale** — banda,
noise floor, tipo di elettrodo, quantizzazione a 8 bit, rete a 50 o 60 Hz con le armoniche
(§4.3). Come gate confonderebbe «il kernel continuo non funziona» con «gli hardware sono
diversi», e fallirebbe sempre. Ciò che questa sezione rivendica — stessi pesi, griglia
diversa, **stesse feature** — si verifica sulle **stesse registrazioni** portate su griglie
diverse. Il probe dataset-ID resta, come **diagnostica relativa**.

### Mascheramento dei target spettrali oltre banda — non opzionale

Sul Myo la Nyquist è 100 Hz: le bande alte sono **identicamente nulle**. Senza mascheratura
la loss ancora su quei termini è una costante che il modello azzera subito, e il peso
effettivo dell'ancora diventa **diverso per dataset** — bias invisibile nei log aggregati.

### 4.3 Preprocessing

- **Passabanda** 20–450 Hz (adattato alla Nyquist effettiva).
- **Notch sempre a 50 e 60 Hz**, anche dove non serve. **Limite noto:** le armoniche restano,
  quindi la frequenza di rete è un'**impronta geografica** del dataset (60 Hz negli USA, 50 Hz
  in Europa e Cina) che i due notch non cancellano. Non si estende il notch alle armoniche —
  costerebbe troppa banda utile — ma va ricordato leggendo il probe dataset-ID (§7.1).
- **Storage** int16 memory-mapped o webdataset shard su `$WORK`. Non `$SCRATCH` (pulito dopo
  40 giorni), non `$HOME` (quota 50 GB).
- **Filtraggio offline**, mai on-the-fly: 32 core / 4 GPU sul Booster.

**Normalizzazione: per traccia/sessione, scala CONDIVISA tra canali** (mediana dei MAD).
Normalizzare ogni canale per il proprio MAD distruggerebbe l'ampiezza **relativa**, cioè il
pattern spaziale che il masking di canale deve far imparare. MAD e non deviazione standard:
code pesanti e burst. Mai per finestra.

### 4.4 QC dei canali — offline, all'ingest

Passaggio offline che marca canali piatti, saturi o con noise floor anomalo.

Motivo: sulle griglie HD i canali con contatto scadente sono la norma. Con il **40% del
budget di masking sul masking di canale**, un canale morto trasforma "predici il canale k
dagli altri" in "predici zero".

**Il QC precede il conteggio di D** (§10.3).

### 4.5 Schema metadati di sensore

Decisione più costosa da cambiare a posteriori:
- tipo di elettrodo, distanza inter-elettrodica, **frequenza nativa**, banda effettiva,
  **frequenza di rete** del luogo di acquisizione (50/60 Hz, §4.3)
- **flag raw / inviluppo** (§2.4) — determina se il dataset entra nel front-end
- **gruppi di canali**: un montaggio è un insieme di gruppi, e **topologia** (anello /
  griglia 2D / sparsa) e simmetria dichiarata (§3.4) sono proprietà **del gruppo**, non del
  dataset
- geometria e i **tre sistemi di coordinate** di §3.6; per anelli e fasce, **orientamento
  della fascia** rispetto a un repere anatomico, oppure «ignoto»
- **identità anatomica del canale — gerarchica**: etichetta al livello più fine realmente
  noto, campo `precisione`, eventuali pesi soft sui compartimenti, flag `nominale` (sotto)
- **flag di chiralità** (§3.4)
- **orientamento della coppia bipolare / canonicalizzazione di polarità** (§3.3)
- **calibrazione disponibile / assente**, riferimento MVC se disponibile (§3.2)
- flag di validità per canale (QC)
- regione corporea

#### Tassonomia anatomica condivisa — gerarchia, decisa

**Albero: regione → compartimento (al polso: settore) → muscolo.** I livelli grossolani sono
funzione deterministica di quelli fini.

**Regola di annotazione.** Ogni canale si annota **una volta sola, al livello più fine che si
conosce davvero**, con un campo `precisione` ∈ {muscolo, compartimento, settore, regione,
ignoto}. Gli antenati si derivano per lookup. **Mai etichette inventate.** È il motivo per cui
la gerarchia non raddoppia il lavoro di mappatura: lo riduce rispetto al «solo muscolo»,
perché nessuno deve più indovinare un muscolo sotto un Myo.

**Perché non «solo compartimento».** Rompe l'unico esperimento per cui il percorso anatomico
esiste. Su Camargo (lista dei muscoli da verificare) a livello di compartimento vasto mediale,
vasto laterale e retto femorale collassano in una sola etichetta, bicipite femorale e
semitendinoso in un'altra, gastrocnemio e soleo in una terza. Con il vicinato metrico vuoto
quei canali diventano **scambiabili**: è l'ambiguità di permutazione che §3.4 chiama errore, e
il controllo più netto smette di essere leggibile.

**Perché non «solo muscolo».** Su anelli e fasce metà del corpus avrebbe etichette inventate.

**Perché la precisione è per canale e non per dataset.** Il montaggio NinaPro a 12 elettrodi
(§2.1) contiene, nello stesso dataset, canali «posizione angolare nota, muscolo ignoto» e
canali «muscolo noto». Qualunque granularità globale è sbagliata per una parte dei canali.

**Funzione atlante per anelli e fasce.** Per Myo, bracciali Meta, anello NinaPro, GRABMyo,
putEMG e le strisce di CapgMyo **non si mappa a mano**. Si scrive una volta una funzione

  `(angolo rispetto al repere, livello prossimo-distale) → pesi soft sui compartimenti`

da sezioni trasverse dell'avambraccio, e la si riusa ovunque. Serve solo l'orientamento della
fascia. Dove l'orientamento è ignoto (EPN-612: varia per utente) l'etichetta è «compartimento
ignoto» e l'identità arriva dal **percorso geometrico** — la divisione del lavoro corretta fra
i due percorsi: la geometria dà la disposizione *relativa*, l'anatomia l'ancoraggio
*assoluto*, quando c'è. Le griglie HD si etichettano per griglia o per porzione (lato
flessorio o estensorio), con la stessa funzione se la posizione circonferenziale è nota.

**Restano a mano poche decine di canali:** gli elettrodi mirati di NinaPro e gli 11 di
Camargo.

**Due cautele.**
- **Polso.** Sotto i bracciali Meta (emg2pose, emg2qwerty: un terzo delle ore) ci sono tendini
  e ventri distali, non ventri muscolari: lì anche «compartimento» è un'approssimazione. Si
  usano **settori** — volare/dorsale × radiale/ulnare — come livello intermedio.
- **Amputati** (DB3, DB7). L'anatomia del moncone è alterata: le etichette sono **nominali**.
  Flag `nominale` nei metadati; conta per l'asse cross-popolazione di §8.

**Vocabolario.** ID **FMA** o **UBERON** per i muscoli, così l'estensione ad altre regioni non
richiede di rifare i nomi. Compartimenti **revisionati e firmati da Simone il 24/09/2026**
(D7b, `docs/decisioni.md`; implementazione in `src/wearusfm/metadata/taxonomy.py`, 27 ID
FMA/UBERON verificati via OLS4):

| Regione | Compartimento / settore | Muscoli di riferimento |
|---|---|---|
| Avambraccio prossimale | flessori-pronatori, lato ulnare | FCU, palmare lungo, FDS (parte ulnare) |
| Avambraccio prossimale | flessori-pronatori, lato radiale | FCR, pronatore rotondo |
| Avambraccio prossimale | gruppo radiale | brachioradiale, ECRL, ECRB |
| Avambraccio prossimale | estensori dorsali | EDC, EDM, ECU |
| Avambraccio distale | dorsale-radiale, profondi affioranti | APL, EPB |
| Polso | 4 settori: volare/dorsale × radiale/ulnare | — (tendini e ventri distali) |
| Braccio | anteriore · posteriore | bicipite, brachiale · tricipite |
| Arto inferiore (solo Camargo) | coscia (anteriore/mediale/posteriore) · gamba (anteriore/posteriore) · glutei | gastrocnemio mediale, tibiale anteriore, soleo, vasto mediale, vasto laterale, retto femorale, bicipite femorale, semitendinoso, gracile, medio gluteo (10 muscoli, verificati dalle colonne del dataset reale + sito del laboratorio, fatto n. 6) |
| Tronco (solo Camargo) | parete addominale anterolaterale | obliquo esterno destro — **non arto inferiore**: l'11º canale di Camargo è un muscolo del tronco (correzione D7b, non nella bozza originale) |

Ordine attorno all'avambraccio prossimale, partendo dal margine sottocutaneo dell'ulna in
direzione volare (bozza per la funzione atlante): FCU → palmare lungo / FDS → FCR → pronatore
rotondo → brachioradiale → ECRL → ECRB → EDC → EDM → ECU → ulna.

**Uso a valle.** Embedding gerarchico e dropout del livello muscolo: §5.2. La sonda di §7.3 si
valuta a livello di compartimento ovunque, e a livello di muscolo dove `precisione` lo
consente — è ciò che la rende verificabile. La prima bozza di mappatura resta un candidato
naturale per un agente, con revisione umana obbligatoria.

### 4.6 Nota di ingegneria con conseguenza sul protocollo

Il dataloader è il punto di rischio specifico di questo progetto. Ogni campione richiede:
lettura da shard, montage dropout o sottocampionamento HD, costruzione dei montaggi bipolari
virtuali, masking su tre scale temporali più quello spaziale, e collate a **C variabile con
padding e maschera** — quest'ultima non standard, perché la forma variabile impedisce la
pre-allocazione e sposta lavoro vero sulla CPU (8 core per GPU).

Le quote per topologia impongono inoltre una miscela prescritta in ogni batch, e un campione
HD a 256 canali costa in preparazione molto più di uno a 8: il tempo per batch dipende dalla
composizione, e con quote fisse quella varianza non si media via. Si manifesta come
throughput oscillante, non come "dataloader lento" — difficile da diagnosticare a metà run.

**Retroazione sul protocollo:** se la collate a forma variabile non regge, la risposta è
precomputare i montaggi virtuali offline. Ma un montaggio precomputato è dato fisso, uno
generato al volo è augmentation stocastica — e questo cambia il conteggio di `D_c`, che va
congelato al passo 4 della roadmap. **Va misurato prima di congelare il manifest.**

**Misurabile senza dati** (§9, passo 0): tensori casuali con la distribuzione di canali che
il manifest prevede, quote per topologia applicate, throughput con 8 worker per GPU. **Due
assi, non tre bracci** (rev. 3): *montaggi* al volo o precompute — lato CPU e I/O, misurato col
solo dataloader contro un consumatore simulato — e *layout* padding, packing o bucketing per
micro-batch — lato GPU, misurato con un modello proxy. Il contenuto è casuale, ma i dati si
**scrivono su disco nel formato dell'ingest e si rileggono**: senza I/O il precompute non si
vede. La distribuzione ha frequenza nativa e contesto, non solo canali: una finestra di Hyser
pesa oltre 150 volte una di DB5. Spec completa: `docs/dataloader_bench_spec.md`.

**Alternativa da misurare: packing.** Invece di padding + maschera, concatenare i token di
canale di più campioni in un'unica sequenza con attenzione a blocchi (stile NaViT, «Patch n'
Pack»). Conserva la miscela di topologie in ogni batch — il motivo per cui il bucketing è
escluso (§2.7) — senza far pagare 256 canali a chi ne ha 8. Con il padding al massimo del
batch, lo stadio pre-bottleneck lavora su C token per passo temporale contro i K del
backbone: con le quote provvisorie circa l'80% dei token pre-bottleneck sarebbe padding.
Vincolo pratico: il PyTorch 2.2 di `cineca-ai/4.3.0` non ha FlexAttention. `flash_attn` 2.5.5 è
installato e `flash_attn_varlen_func` si importa; **manca la prova funzionale** su A100. E il
kernel accetta maschera causale, finestra scorrevole e pendenze ALiBi, **non un bias additivo
arbitrario**: il bias geometrico di §5.3 non ci passa. Quindi: encoder locale con attenzione
sui vicini via `gather` (§5.4), dove il packing è gratuito; kernel varlen solo per la
cross-attention del Perceiver.

---

## 5. Architettura

```
                  ┌─ caso A: sparso / anatomico, C = 8–32   ← CASO DI DEPLOYMENT
ingresso ─────────┼─ caso B: anello, C = 8–32
                  └─ caso C: griglia HD completa, C = 64–256
                            (+ sottocampionamento a montaggi
                             virtuali come augmentation stocastica)
        ↓
patch temporali per canale — definite in ms, kernel continui a basi
eterogenee condivisi (§4.2)
        ↓
embedding di identità di canale: geometrico relativo ⊕ anatomico (§5.2)
        ↓
encoder spaziale locale geometry-aware        ← punto di misura 1 (§7.4)
   · caso A → set encoder su identità anatomica (nessun vicinato metrico su Camargo)
   · caso B → vicinato angolare, equivarianza ciclica
   · caso C → vicinato 2D, gradienti spaziali, propagazione
        ↓
pooling latente Perceiver  (K latenti)         ← punto di misura 2 (§7.4)
        ↓
backbone temporale globale                     ← punto di misura 3 (§7.4)
   (attenzione fattorizzata tempo/canale)
        ↓
decoder a query (stile Perceiver IO)           ← APERTO (§5.6)
   query = identità di canale ⊕ tempo → predizioni per (canale, tempo):
   predictor JEPA e teste delle ancore
```

### 5.1 K — risoluzione spaziale, non banda informativa

La capacità informativa del bottleneck è **K × d_model**, non K. Tenere fisso K×d
reintrodurrebbe la proiezione che si vuole evitare.

**K è la risoluzione spaziale del bottleneck** — quanti pattern spaziali distinti possono
essere rappresentati separatamente. Vincolo **strutturale e geometrico**, non cap informativo.

**Corollario contabile:** i token latenti del backbone dipendono da K, quindi **non possono
essere usati come budget dati** (§10.3).

### 5.2 Combinazione dei percorsi di identità — concatenazione

**Default: concatenazione.** I due embedding (geometrico relativo, anatomico) sono
concatenati e proiettati. È la scelta meno impositiva: il modello riceve entrambi e decide
da sé come pesarli.

**Embedding anatomico gerarchico, additivo** (§4.5):

  `e_anat = E_regione + E_compartimento + E_muscolo`

Ai livelli ignoti un embedding **UNK appreso per livello**; con etichette soft (funzione
atlante) l'embedding di compartimento è la somma pesata. Un canale Myo e un canale Delsys sul
flessore radiale condividono così il vettore di compartimento, e il livello muscolo agisce da
raffinamento residuo.

**Dropout del livello muscolo** (p ≈ 0,3–0,5, da fissare) sui canali dove il muscolo è noto.
Due ragioni: rende **in-distribuzione** la condizione «solo compartimento», che è il caso di
deployment; e impedisce che «etichetta fine presente» diventi una **scorciatoia per
l'identità del dataset**, con cui è perfettamente correlata.

**Perché non tutti e tre i meccanismi insieme.** Concatenazione, gating sulla topologia e
attenzione separata non sono tre implementazioni della stessa cosa: sono tre ipotesi
mutuamente esclusive su come geometria e anatomia si combinano. Fonderle rende **mal definita
l'ablation** di §10.1 ("togliere il percorso anatomico" non ha più un significato univoco) e
rende **inattribuibile** un fallimento della sonda anatomica di §7.3.

**Attenzione separata scartata:** raddoppia i blocchi di attenzione nell'encoder locale, che
§10.4 prescrive di tenere piccolo per evitare overfitting sui pochi dataset HD.

**Gating sulla topologia** resta come terzo braccio di ablation (§10.1).

### 5.3 Bias di attenzione geometrico — ibrido

**Bias = componente funzionale fissa + componente appresa su una base di distanze.**

La funzionale codifica il decadimento con la distanza e il passa-basso spaziale del volume
conduttore (§3.1), e fa da inizializzazione fisicamente sensata; l'appresa corregge dove i
dati lo richiedono. È il pattern standard dei positional bias relativi, e ha un vantaggio
sulle due alternative pure: dà il comportamento giusto anche dove i dati sono scarsi, cioè
sui montaggi con pochi canali.

### 5.4 Altri elementi

- **Encoder spaziale locale**: graph transformer o attenzione ai vicini fisici, vicinato
  definito da un **raggio nello spazio metrico**.
  - **Implementazione (rev. 3):** attenzione sui k vicini raccolti con `gather`, col bias
    geometrico di §5.3 calcolato sulle loro distanze. I kernel flash non accettano un bias
    additivo arbitrario (§4.6), e in questa forma i canali variabili non richiedono padding.
  - **Il caso A non è degenere: è il target.** Percorso di prima classe con quota garantita.
  - Su Camargo il vicinato metrico è vuoto: il blocco funziona interamente sull'identità
    anatomica, **a livello muscolo** (§4.5). È il caso che rende leggibile l'ablation di
    §10.1.
- **Equivarianza**: encoding relativo circolare (anelli), relativo 2D (griglie), nessuna
  (sparsi). Flag di chiralità per la riflessione.

### 5.5 Note di sistema su Leonardo

Booster: 3.456 nodi, 4× A100 SXM4 64 GB, Ice Lake 32 core, 512 GB RAM, 2× HDR 100 Gb/s,
Dragonfly+. Niente FP8, quindi bf16 + activation checkpointing. Fino a ~1B DDP basta; sopra
ZeRO-2/3 o FSDP. Walltime 24 h → checkpoint/restart concatenati.

Ambiente: `module load profile/deeplrn && module load cineca-ai/4.3.0`, venv su `$WORK` con
`--system-site-packages`. PyTorch 2.2.0a0, CUDA 12.1.

### 5.6 Decoder a query e collocazione dei target JEPA — APERTA, prioritaria

I K latenti **non corrispondono ai canali**. Predire una patch mascherata (canale, tempo)
richiede quindi un **decoder a query** — query = identità di canale ⊕ tempo, cross-attention
sui latenti — e la stessa domanda si pone per il teacher: **dove si leggono i target?**

| Opzione | Target | Pro | Contro |
|---|---|---|---|
| (a) | uscita per-canale dell'encoder spaziale locale del teacher | corrispondenza canale↔target naturale, semplice | target poco profondi, contestualizzati solo localmente |
| (b) | uscita del decoder a query del teacher sull'input non mascherato, alle stesse query dello studente | target profondi e contestualizzati | apre il **collasso da query** |
| (c) | latenti post-Perceiver del teacher | — | **scartata**: mascherare canali cambia il contenuto dei latenti in modo non strutturato, target e predizione non sono allineati |

**Collasso da query.** Con (b) esiste una soluzione degenere a loss bassa: un'uscita che
dipende dalla **query** (identità di canale, tempo) e non dal **contenuto**. **Il rango
effettivo non la vede** — tante query distinte danno rango alto. Diagnostica dedicata in §7.1.
Le ancore a target esterno la contrastano, ma non vanno date per sufficienti.

**Default di lavoro, non decisione:** (b) con la diagnostica attiva dal primo giorno, (a) come
braccio di controllo a 30M (§10.1). Va deciso al **passo 6**, prima dell'ablation del passo 7
che ci poggia sopra. Le teste delle ancore leggono dallo stesso decoder.

---

## 6. Obiettivo di pretraining

### 6.1 Principio

**Il target non è mai la waveform grezza.** Il segnale è somma di potenziali d'azione da
unità motorie che scaricano stocasticamente: la realizzazione specifica non è predicibile
nemmeno in linea di principio. L'MSE sul raw ottimizza rumore e produce un denoiser.

Predicibile: inviluppo di ampiezza (~5–10 Hz), forma spettrale, pattern spaziale tra canali,
struttura temporale alla scala del movimento.

**Tutti i target sono osservativi, calcolati dal raw.**

### 6.2 Design: JEPA con ancora fisica

**Loss primaria — predizione latente mascherata.** Studente su contesto, insegnante EMA,
stop-gradient. Il target è appreso: la rappresentazione di una patch rumorosa è essa stessa
povera di informazione, quindi lo studente non viene punito per non predire il rumore.

**Loss ancora — target espliciti**, testa piccola, peso complessivo 0.1–0.3:
1. **Non può collassare** (target esterno e fisso). JEPA puro fallisce per collasso
   silenzioso con loss bassissima.
2. **Garantisce** che ampiezza, spettro e struttura spaziale restino nella rappresentazione.
3. È **interpretabile**.

**Perché RVQ non è la loss primaria.** Come obiettivo principale aggiungerebbe uno stadio di
training congiunto con fallimenti propri (codici morti, collasso del codebook) — troppi pezzi
mobili a questa scala. **Come ancora con tokenizer congelato quei fallimenti non si
presentano**, ed è la forma in cui vale la pena testarlo (§6.3).

Nota che RVQ resterebbe la scelta giusta per un modello **generativo** di EMG — campionamento
di tracce, imputazione di canali, predizione autoregressiva sul vocabolario discreto. Non è
l'obiettivo attuale, che è un encoder di rappresentazione.

*Premessa da non perdere:* il testo è già discreto, quindi tokenizzarlo recupera una struttura
esistente; l'sEMG è continuo e stocastico, quindi quantizzarlo **impone** una discretizzazione
che non c'è. Il risultato di NeuroRVQ lo riflette: sull'EMG la riduzione dell'errore di
ricostruzione della configurazione completa rispetto alla baseline è 4,5×, contro 43×
sull'EEG e 17× sull'ECG (è l'intera configurazione, non la sola loss di fase), e la
convergenza è più graduale per via della natura a banda larga. Il metodo funziona meno bene
proprio su questa modalità.

### 6.3 Ancore — multirisoluzione, taglio fase/modulo, e ancora RVQ

**La finestra del target NON deve coincidere con la lunghezza della patch mascherata.**

A 25 ms la risoluzione in frequenza è 40 Hz: la banda 20–40 Hz **non è risolvibile**. Danno
collaterale: le bande basse sono dove vive lo spostamento della frequenza mediana con la
fatica, che è una sonda fisica di §7.2.

| Target | Finestra | Note |
|---|---|---|
| RMS / energia locale | 20–50 ms | dinamica veloce; **unica ancora alimentata dai dataset a inviluppo** (§2.4) |
| Forma spettrale (4–5 bande) | **≥ 100–250 ms** | risoluzione 4–10 Hz |
| Inviluppo | finestre più lunghe | contesto di attivazione |

Bande **log-spaziate su 20–450 Hz** o dalla PSD media del corpus.
**Mascherate oltre la banda disponibile** per dataset (§4.2).

**Ancore spaziali (peso ridotto).** RMS e PSD non dicono nulla sulla struttura spaziale. Ma
inserire i ritardi nell'obiettivo renderebbe **tautologica** la sonda sulla velocità di
conduzione.

| Quantità | Contiene fase? | Determina la CV? | Uso |
|---|---|---|---|
| Covarianza spaziale a lag zero | no | no | **ancora** |
| Coerenza in **modulo** | no | no | **ancora** |
| Struttura dei ritardi tra canali | sì | sì | **solo sonda** |
| Fase del cross-spettro | sì | sì | **solo sonda** |

#### Ancora RVQ — condizionata alle verifiche preliminari e all'ablation

**Terza ancora, accanto a RMS e bande spettrali:** predizione dei codici del **tokenizer
NeuroRVQ-EMG pre-addestrato e congelato**, come target discreto sulle regioni mascherate.
Testa piccola, peso basso.

Il tokenizer EMG di NeuroRVQ è **già rilasciato** (144M, HuggingFace), quindi non va
addestrato: diventa un target esterno come gli altri.

**Cosa aggiunge che RMS e PSD non danno.** La cross-entropy su codebook cattura la
**multimodalità della distribuzione predittiva** — l'MSE su target continui la media via.
È una lacuna reale del set di ancore attuale.

**Perché in questa forma non ha i problemi dell'RVQ come primaria.** Tokenizer congelato ⇒
nessun training congiunto, nessun codice morto, nessun collasso del codebook durante il
pretraining principale. Target esterno e fisso ⇒ stesse proprietà anti-collasso delle altre
ancore.

**Fatti sul checkpoint** (dal paper v4 e dal README di una conversione di terze parti dei
pesi; il codice del repo **non è stato letto**):
- lavora a **1000 Hz** con patch da **200 ms**. I 200 campioni sono **cablati
  nell'architettura**: gli embedding hanno la dimensione della patch e il decoder ricostruisce
  ampiezza e fase dello spettro di Fourier di quella patch
- 4 rami temporali × 16 livelli RVQ × 8192 codici da 128 dimensioni: **64 codici per
  patch-canale**. Circa metà dei 144M sono codebook (4·16·8192·128 ≈ 67M): la rete è piccola
- addestrato **solo su emg2pose ed emg2qwerty** — un solo dispositivo, al polso
- embedding spaziali indicizzati sugli elettrodi del database di pretraining (le 16 posizioni
  del bracciale Meta); per come lo descrive il paper, il transformer mescola patch di canali
  e istanti diversi (**da verificare nel codice**)
- nei downstream gli autori hanno ricampionato tutto alla frequenza del modello e filtrato a
  **20–90 Hz**. **Non verificato** se anche il tokenizer è stato addestrato su quella banda
  (→ bivio, sotto)

##### Frequenza fissa — soluzione, chiusa

**Il trucco di §4.2 qui non è disponibile.** «Stessi pesi, griglia diversa» richiederebbe di
riaddestrare: i 200 campioni sono nell'architettura. Ricampionare a 1 kHz è inevitabile, e la
domanda diventa **quali dataset possono entrare nel tokenizer senza che i codici si portino
dietro l'identità del dataset**.

1. **Vista tokenizer canonica**, offline, unica per tutti i dataset raw con frequenza nativa
   ≥ 1 kHz: stesso passabanda → resampling polifase a 1000 Hz esatti con **lo stesso taglio
   anti-alias per tutti** (il minimo del gruppo, non il Nyquist di ciascuno) → la
   normalizzazione attesa dal tokenizer → griglia di patch da 200 ms ancorata alla
   registrazione. Serve **solo a calcolare i target**: il front-end continua a vedere i dati
   nativi.
2. **Dataset a 200 Hz: ancora RVQ mascherata, nessun upsampling.** Nel corpus di pretraining
   il 200 Hz è **un solo dataset: DB5, dieci soggetti** (EPN-612 e UCI-EMG sono fuori, DB1 non
   entra nel front-end). È la stessa logica del mascheramento delle bande oltre Nyquist
   (§4.2), ed è coerente con §4.1.
3. **Pesi.** I pesi delle singole ancore restano fissi; **nessuna rinormalizzazione del totale
   su DB5** — gonfierebbe in silenzio RMS e bande basse sul Myo. Log per dataset.

**Perché non «solo i rami lenti» sul Myo.** I quattro rami differiscono per dimensione del
kernel ma vedono tutti l'intera banda e passano per lo stesso transformer: niente garantisce
che un sottoinsieme di codici corrisponda alla banda bassa.

**Perché è un'asimmetria accettabile.** È **dichiarata**, dello stesso tipo di quelle già
accettate (bande mascherate, dataset a inviluppo), e il modello non vede la maschera di loss
in input. Quella da evitare è l'asimmetria **nascosta**, in cui è il target stesso a
codificare il dataset — ed è il tema delle verifiche qui sotto.

**Bivio da chiudere leggendo il repo** (cartella `preprocessing`). Se il tokenizer è stato
addestrato su dati filtrati a 20–90 Hz, il suo dominio nativo è già la **banda comune a tutto
il corpus, Myo incluso**. In quel caso la soluzione simmetrica è dare a tutti la vista
20–90 Hz — DB5 compreso, qui sì con upsampling — e accettare che l'ancora RVQ sia un'**ancora
di banda bassa**. Altrimenti vale la soluzione sopra.

##### Il problema più grosso della frequenza: il tokenizer è fuori distribuzione

Fuori dai due dataset Meta il tokenizer è **fuori distribuzione**, mentre un terzo delle ore
del corpus è dentro. Se i codici sono etichette pulite sul polso Meta e rumorose altrove, il
peso effettivo dell'ancora è **diverso per dataset** — il bias invisibile di §4.2, in forma
più profonda. Quattro verifiche, tutte eseguibili col checkpoint pubblico **prima di qualunque
pretraining** (§9, passo 1-bis):

- **V1 — un canale alla volta.** Tokenizzare un canale per volta, con un **indice spaziale
  fisso uguale per tutti** i canali di tutti i dataset. Sparisce la scelta dataset-specifica
  di mappare i canali sui 16 elettrodi Meta, e i codici diventano intra-canale per
  costruzione. È fuori distribuzione rispetto all'addestramento multi-canale: controllare che
  la ricostruzione su emg2pose non degradi in modo sostanziale.
- **V2 — errore di ricostruzione per dataset**, col decoder rilasciato. È il misuratore del
  rumore sulle etichette. Se Delsys e griglie monopolari ricostruiscono molto peggio di
  emg2pose, l'ancora pesa diversamente da dataset a dataset.
- **V3 — probe dataset-ID sugli istogrammi dei codici**, confrontato con lo stesso probe sulle
  potenze di banda. Se i codici identificano la sorgente molto meglio delle ancore continue,
  l'ancora **insegna al modello l'identità del dataset**.
- **V4 — stabilità dei codici per livello RVQ** sotto perturbazioni innocue (rumore al noise
  floor del dataset). Si predicono **solo i livelli stabili su tutti i dataset**: i livelli
  profondi cambiano codice per perturbazioni minime e sono i più specifici dell'hardware.

**Criteri di arresto, da congelare prima di guardare** (soglie in §12). V2: se l'errore
mediano fuori dai dataset Meta supera di un fattore X quello su emg2pose, l'ancora si
restringe ai dataset sotto soglia, o si scarta. V3: se l'accuratezza dataset-ID dai codici
supera di Y punti quella dalle potenze di banda, l'ancora si scarta. V4: se nessun livello è
stabile su tutti i dataset, l'ancora si scarta.

##### Vincoli di applicazione

- **Scala di masking.** Un target esiste solo per finestre da 200 ms della griglia del
  tokenizer **interamente contenute** in una regione mascherata: l'ancora lavora solo alle
  scale di masking media e lunga (§6.4).
- **Solo intervalli temporali mascherati su tutti i canali, mai masking di canale** (vedi
  leakage, sotto).
- **Augmentation.** Il target si calcola sul segnale pulito: le trasformazioni geometriche
  (time warping) si applicano anche al segnale da cui si calcola il target, il rumore no. Un
  montaggio bipolare virtuale è un segnale nuovo: i suoi codici si calcolano online — la rete
  del tokenizer è piccola — o si precomputano se lo sono i montaggi (§4.6).
- **Soggetti visti dal tokenizer.** Ha visto soggetti di emg2pose ed emg2qwerty: verificare la
  sovrapposizione con i soggetti di test di §8. Se non è ricostruibile, va dichiarato come
  limite del braccio RVQ.

##### Leakage di fase — il ripiego «solo intra-canale» non basta

Il tokenizer è addestrato con una **loss phase-aware**: i suoi codici contengono per
costruzione informazione di fase. Secondo il taglio qui sopra, la fase deve restare **fuori**
dalle ancore per non rendere tautologica la sonda sulla velocità di conduzione.

**Perché il ripiego della v9 non regge.** Un ritardo inter-canale è la **differenza fra due
fasi per-canale**: i codici intra-canale di due canali bastano, in linea di principio, a
ricostruirlo. E se si predice il codice phase-aware di un canale mascherato **mentre i vicini
sono visibili nello stesso istante**, l'unico modo di farlo bene è modellare il ritardo di
propagazione: si sta insegnando la velocità di conduzione.

**Sostituto: l'ancora RVQ è attiva solo su intervalli temporali mascherati su tutti i canali
insieme (*slab*), mai sul masking di canale.** Senza nessun canale visibile nello stesso
istante la fase è imprevedibile, e teste indipendenti per canale non catturano la struttura
congiunta fra canali. La parte predicibile del codice si riduce ad ampiezza, forma spettrale
e **multimodalità** (parte un burst oppure no) — il motivo per cui l'ancora esiste. Margine
ulteriore, a costo quasi nullo: escludere dall'ancora RVQ le registrazioni HD usate per la
sonda sulla velocità di conduzione.

**Il test cambia oggetto.** Verificare che «dai codici non si ricostruiscano i ritardi» non
ha più senso: dai codici di due canali i ritardi si ricostruiscono *per costruzione*, perché
il tokenizer ricostruisce le forme d'onda. Conta se l'ancora **trasferisce**
quell'informazione alla rappresentazione: si confronta la decodificabilità dei ritardi
inter-canale sulle griglie HD fra run 4 e run 5 (§10.1).

**Asimmetria del test, da non sottovalutare.** Si sta cercando di dimostrare un'**assenza** di
trasferimento, e un probe lineare che non vede differenze è l'evidenza più debole possibile.
Se la fase è codificata non linearmente, il probe lineare passa, l'ancora viene adottata, e
la sonda sulla velocità di conduzione diventa tautologica senza che nessuno se ne accorga.
**Serve il probe più forte costruibile** (MLP profondo, o gradient boosting sulle
rappresentazioni), non il più semplice, e l'esito va dichiarato negativo solo se non vede
differenze nemmeno quello.

Se la decodificabilità dei ritardi sale in modo significativo con l'ancora RVQ, l'ancora va
**scartata**. Ha precedenza la sonda, non l'ancora.

**Tokenizer rate-agnostic riaddestrato sul corpus:** solo come passo successivo, e solo se la
run 5 (§10.1) mostra un guadagno. Reintroduce uno stadio con i fallimenti propri dell'RVQ e,
soprattutto, un tokenizer proprio **non è più un target esterno**.

### 6.4 Masking multiscala — e due task spaziali diversi

**Temporale:** 20–50 ms (contenuto spettrale locale) · 100–300 ms (attivazione muscolare) ·
500 ms–2 s (contesto motorio). Span molto corti sono interpolabili dall'inviluppo ai lati:
non vanno usati come unica scala.

**Vincolo dall'ancora RVQ, se adottata (§6.3):** si attiva solo su intervalli ≥ 200 ms
mascherati **su tutti i canali insieme**, allineati alla griglia del tokenizer. Una quota del
masking temporale medio-lungo va quindi riservata a slab su tutti i canali.

**Spaziale (~40% del budget):** canali singoli, gruppi contigui, porzioni geometriche della
griglia.

**Il masking di canale è due task diversi, da loggare separatamente:**

| Topologia | Cosa chiede davvero | Difficoltà |
|---|---|---|
| Griglia densa | **interpolazione spaziale** — risolvibile con la conduzione di volume | bassa |
| Sparsa / anatomica | **predizione di sinergia** — coordinazione motoria, non interpolazione | alta |

**Senza logging separato la loss aggregata è dominata dal caso facile.**

### 6.5 Teste supervisionate

Attivate per-batch secondo il dataset di provenienza. Bilanciamento **per campionamento, non
per pesi di loss**. Piccole: troppa capacità assorbe l'adattamento che si vorrebbe nel
backbone. Le teste di regressione portano il lead time vincolato di §3.7.

---

## 7. Diagnostiche e sonde — in CI dal primo giorno

### 7.1 Sonde di integrità

1. **Consistenza al ricampionamento — gate del front-end** (§4.2). Si prendono registrazioni
   native a 2 kHz (o 2048 Hz) e, per ciascuna banda target (es. < 90 Hz per il caso Myo,
   < 450 Hz per il caso 1 kHz), se ne costruiscono due versioni con **lo stesso contenuto su
   griglie diverse**: (A) filtrata passa-basso alla banda target ma lasciata sulla griglia
   nativa; (B) la stessa, decimata a 200 Hz o a 1 kHz. Criteri: (i) le feature del front-end
   coincidono entro una soglia (da congelare, §12); (ii) un probe A-contro-B resta al caso.
   **Se fallisce, il kernel continuo di §4.2 non funziona e non ha senso proseguire.** Non
   richiede il corpus: basta un dataset a 2 kHz.
2. **Linear probe su dataset ID — diagnostica relativa, in due punti:**
   - **subito dopo il front-end** → **non è un gate**: in assoluto riesce sempre, perché
     l'identità della sorgente è nel segnale (§4.2), inclusa l'impronta 50/60 Hz (§4.3). Si
     legge **per differenza**, contro lo stesso probe (a) su un banco di filtri fisso e (b)
     sul braccio a front-end separati (§10.1).
   - **sugli embedding finali** → diagnostica. Fallimento **silenzioso**: la loss SSL scende
     benissimo mentre succede.
3. **Rango effettivo delle rappresentazioni.**
4. **Collasso da query** (§5.6). Varianza di target e predizioni **a query fissa fra campioni
   diversi**, confrontata con la varianza **fra query**. Se la prima crolla rispetto alla
   seconda, l'uscita dipende dalla query e non dal contenuto. **Il rango effettivo non lo
   vede.**
5. **Linear probe cross-soggetto**, **riportato per topologia**, non aggregato.

### 7.2 Sonde fisiche

- ampiezza RMS e sua dinamica
- frequenza mediana dello spettro (cala con la fatica per calo della velocità di conduzione)
- **velocità di conduzione**, dal ritardo di propagazione tra righe di una griglia HD
  allineata alle fibre. **Vale solo se i ritardi restano fuori da tutte le ancore, inclusa
  quella RVQ** (§6.3), richiede risoluzione temporale nativa (§4.1) e griglie piene viste nel
  pretraining (§2)
- covarianza spaziale e struttura di coerenza
- distanza inter-elettrodica, numero di canali attivi

### 7.3 Sonde di rilevanza per canale

**Misurare sì, pesare no.** Nessun gate appreso di rilevanza per canale: imparerebbe quali
canali sono informativi *in media sul corpus*, cioè un prior sulla configurazione degli
elettrodi.

Strumenti: **ablazione di canale** (esatta, model-agnostica, C forward pass) · **pesi di
cross-attention al bottleneck** · **integrated gradients**.

**I due test:**
1. **La mappa ruota quando ruota il bracciale?** Verifica quantitativa dell'equivarianza
   ciclica — solo sugli anelli. Analogo per riflessione e traslazione sulle griglie.
2. **La mappa è coerente con l'anatomia funzionale?** Richiede la tassonomia di §4.5: si
   valuta a livello di compartimento ovunque, a livello di muscolo dove il campo `precisione`
   lo consente.

### 7.4 Sonda stratificata — permanente

Le sonde fisiche di §7.2 valutate in **tre punti** (§5): output dell'encoder spaziale locale ·
dopo il Perceiver · dopo il backbone.

- sonda buona prima del Perceiver, degradata dopo → problema di **K o bottleneck**
- sonda già debole dopo l'encoder locale → problema di **profondità o raggio locale**
- sonda buona ovunque ma downstream piatto → problema di **backbone o dati**

Precedente diretto: *Beyond Accuracy* (arXiv 2605.17562) fa probing block-wise e robustezza a
rumore e channel dropout sui FM EEG.

---

## 8. Valutazione

### Assi di generalizzazione — riportati separatamente, sempre

1. Intra-sessione, cross-ripetizione — regime gonfiato; solo come sanity check,
   **dichiarandolo**.
2. Cross-sessione, stesso soggetto. DB6, GRABMyo.
3. Cross-soggetto, stesso dataset.
4. **Transfer su dataset mai visto in pretraining** (encoder congelato + probe; non
   «zero-shot», §2.3) — EPN-612 e UCI-EMG.
5. **Cross-popolazione: normodotati → amputati. DB3, DB7** (più DB8 e DB10, se la presenza di
   amputati è confermata: §2.1). Con lo scope ristretto, è la generalizzazione più lontana
   che il corpus permette di testare: peso relativo aumentato.
6. **Cross-topologia**: pretraining con HD, valutazione su montaggi sparsi. **Asse portante**
   del paper, ora che cross-regione non è più rivendicato.

### Trappole da chiudere per iscritto

- **Niente classe "rest" derivata dalle pause tra gesti.** Label leakage via padding
  documentato da NeuroRVQ su NinaPro DB5.
- **Normalizzazione stimata solo sul train**, e se per sessione, dai primi N secondi.
- **Sovrapposizione di soggetti fra DB NinaPro:** da controllare all'ingest, prima di
  definire gli split (§2.1).
- **Soggetti visti dal tokenizer NeuroRVQ** (emg2pose, emg2qwerty): se si usa l'ancora RVQ,
  verificare la sovrapposizione coi soggetti di test (§6.3).

### Trasformazioni del sensore — test distinti per topologia

| Test | Topologia | Cosa verifica | Augmentation? |
|---|---|---|---|
| **Rotazione ciclica** | anelli | equivarianza architetturale | non serve — è nella struttura |
| **Riflessione L↔R** | anelli | equivarianza diedrale, chiralità | ammessa, ma deve trasformarsi |
| **Traslazione lungo gli assi** | griglie | equivarianza traslazionale | ammessa in parte |
| **Traslazione lungo l'asse dell'arto** | tutte | robustezza allo **shift reale** | **mai** — asse di valutazione |
| **Permutazione di canale** | sparse | — | **mai**: non è simmetria, è errore |

### Baseline non negoziabili

- **NeuroRVQ-EMG-FM ri-valutato col protocollo** — è il **riferimento principale**, non
  TinyMyo. Checkpoint pubblico (§11). **Il confronto valido è questo, non la loro tabella:**
  i loro numeri EMG sono fine-tuning completo con testa lineare e split per soggetto 7:1:2.
- **TinyMyo e PhysioWave ri-valutati col protocollo.**
- LDA su feature di Hudgins.
- CNN/TCN allenata da zero sul dataset target.
- **Lo stesso modello con encoder random**, non pre-allenato, stessa testa.

La ri-valutazione dei checkpoint pubblici col protocollo onesto è **metà del contributo del
paper**, e non richiede né il corpus, né l'architettura, né Leonardo: vedi §9, nota sulla
separabilità.

**Due regimi, per tutti i modelli:** encoder congelato + probe (il regime del protocollo)
**e** fine-tuning completo (il regime della letteratura). Così ogni numero è leggibile sia
contro il protocollo sia contro le tabelle pubblicate. Benchmark da aggiungere all'harness:
**Discrete Gestures** (Kaifosh et al.), prima colonna della tabella EMG di NeuroRVQ (§2.1).

### Altre augmentation valide

Montage dropout, rumore realistico (drift sub-20 Hz, powerline, contaminazione ECG per tronco
e muscoli prossimali), time warping, sottocampionamento di montaggi virtuali da griglie HD.
**Non**: guadagno (§3.2), polarità per-canale (§3.3), traslazione lungo l'arto, permutazione
su montaggi sparsi. Moltiplicatore effettivo onesto: 3–10×.

---

## 9. Roadmap

0. **Benchmark sintetico del dataloader** (§4.6) — non richiede dati. Tensori casuali con la
   distribuzione di canali prevista, quote per topologia, collate a forma variabile,
   throughput con 8 worker per GPU. Risponde alla domanda precompute-vs-on-the-fly, che
   retroagisce su `D_c`. Due assi: montaggi (al volo / precompute) e layout (padding /
   packing / bucketing per micro-batch). Spec: `docs/dataloader_bench_spec.md`.
1. **Scaricamento dataset** secondo l'ordine di priorità di §2.5. Avviare subito le richieste
   che hanno latenza umana (CSL-hdemg via email, account NinaPro) e scaricare presto
   emg2qwerty (repo archiviato).
   - **1-bis. Verifiche sul checkpoint NeuroRVQ** (§6.3: V1–V4, bivio 20–90 Hz, lettura del
     repo). Bastano il checkpoint, emg2pose-mini, i dataset piccoli e un NinaPro Delsys;
     GPU-ora trascurabili. Esito — ancora RVQ adottabile, ristretta o scartata — **prima** di
     spendere la run 5.
2. **Schema metadati + tassonomia anatomica** + ingest in int16 memory-mapped + QC canali
3. Preprocessing, front-end a kernel continui a basi eterogenee, **gate di consistenza al
   ricampionamento** (§7.1: basta un dataset a 2 kHz) e probe dataset-ID come diagnostica
   relativa. **Confermare il throughput del dataloader** sui dati veri
4. **Congelamento del manifest** e calcolo definitivo di D_t e D_c (§10.3)
5. Harness di valutazione, con **NeuroRVQ-EMG come primo cliente**
6. Topologia del modello — **incluso il decoder a query e la collocazione dei target JEPA
   (§5.6): da decidere qui, prima del passo 7**
7. **Ablation dell'obiettivo SSL a 30M** (§10.1)
8. **Calibrazione dell'architettura a 30–100M** (§10.1)
9. **Pilot sulle epoche a 100M** (§10.3)
   - **9-bis. Asse dei soggetti a 100M** (§10.6) — condivide col pilot la run al 100%
10. **Scaling ladder** (§10.2–10.4)
11. Shakedown a piena scala
12. Training finale
13. Valutazione e distillazione

*Nota sull'ordine.* Riprodurre NeuroRVQ **sulla loro pipeline** ha senso al passo 5 come
debug di dati e harness. Riprodurlo **coi tuoi split** è un risultato scientifico e richiede
l'harness pronto. Non serve Leonardo: il checkpoint è da 5,9M.

*Nota sulla separabilità.* Il passo 5 nella sua forma estesa — tre FM EMG pubblici rimessi
sullo stesso protocollo subject-independent, con il caso TinyMyo (89,4% dichiarato contro
25,26% misurato) come risultato centrale — **è un preprint autonomo**. Non richiede corpus,
architettura né compute HPC, si chiude in settimane invece che in mesi, costruisce credibilità
sul protocollo prima di chiedere a un revisore di crederti sulla scala, e resta in piedi anche
se il FM grande deludesse. In un'area dove escono paper ogni pochi mesi, è anche la mossa che
riduce il rischio di tempistica.

*Nota sull'avvio.* Eseguibili subito, senza dipendere dal corpus completo: il passo 0; le
richieste con latenza umana e il download di emg2qwerty (passo 1); il passo 1-bis, appena
scaricati il checkpoint e i dataset piccoli; il gate di consistenza del passo 3 su un solo
dataset a 2 kHz; la chiusura dei **fatti da verificare** di §12.

*Nota sul parallelismo (rev. 2).* Fino al passo 6 il tempo lo decidono sviluppo, accessi e
revisioni: le GPU-ora non comprano nulla. Dopo, comprano parallelismo. I passi 7, 8, 9 e 9-bis
si lanciano **in blocco** nella finestra di inizio novembre, ciascuno coi default di lavoro
degli altri, con una run di conferma a 100M se un default cade. La ladder del passo 10 si
lancia **tutta insieme** nella finestra di inizio dicembre, dal rung più grande al più
piccolo. Col WSD il pilot sulle epoche è **una sola run per seed** con rami di decay a 1, 2, 4
e 8 epoche. Dettagli e date nel piano operativo.

---

## 10. Esperimenti, budget dati e ladder

### 10.1 Ablation dell'obiettivo e calibrazione

**Ablation dell'obiettivo SSL — 30M.** Cinque run, stesso corpus, stesso masking, valutate
sul linear probe cross-soggetto **per topologia**:

1. MSE sulla waveform grezza (baseline da battere — misurato, non argomentato)
2. target espliciti mascherati (multirisoluzione + ancore spaziali)
3. JEPA puro
4. **JEPA + ancora fisica** ← ipotesi principale
5. **JEPA + ancora fisica + ancora RVQ** (§6.3) ← ~75 GPU-ora

La run 5 chiude con un numero la questione se il codebook aggiunga qualcosa alle ancore
continue. Precondizione: verifiche V1–V4 superate (§6.3). La run gira con l'ancora RVQ **solo
sui dataset ≥ 1 kHz, solo sui livelli RVQ stabili, solo su intervalli temporali mascherati su
tutti i canali**. Il confronto run 4 → run 5 è anche il **test di leakage di fase** (§6.3):
se la decodificabilità dei ritardi inter-canale sale con l'ancora, l'ancora si scarta.

**Ablation della collocazione dei target JEPA — 30M, due bracci** (§5.6):

1. target dal **decoder a query del teacher** (default di lavoro)
2. target dall'**uscita dell'encoder locale del teacher**

Letta sul linear probe cross-soggetto per topologia **e** sulla diagnostica del collasso da
query (§7.1). Va eseguita **prima** dell'ablation dell'obiettivo, o almeno insieme: le cinque
run qui sopra ereditano la scelta.

**Ablation dei percorsi di identità — 30M, tre bracci:**

1. solo percorso geometrico
2. **concatenazione** (default, §5.2)
3. gating sulla topologia

Costo nell'ordine delle decine di GPU-ora, trascurabile.

**Metrica di lettura, da fissare prima di lanciare:** linear probe cross-soggetto **sul solo
ramo sparso**, con Camargo riportato separatamente. Sull'aggregato i tre bracci differiscono
poco per costruzione, e il ramo denso domina.

**Esito atteso, dichiarato prima di guardare:** con l'arto superiore e vicinati metrici mai
vuoti, le tre curve dovrebbero somigliarsi. Se è così è un risultato negativo pulito, si
scrive in tre righe e si risparmia complessità architetturale. Se il gating vince nettamente
sui montaggi sparsi, è evidenza diretta a favore della rivendicazione topology-aware di §11.

**Altre ablation secondarie:** con/senza encoder spaziale locale · kernel a basi eterogenee vs
front-end separati per frequenza · contributo delle singole famiglie di base nel banco (§4.2)
· dropout del livello muscolo acceso/spento (§5.2).

**Se in una run la loss ancora e la JEPA divergono, è il segnale più informativo
dell'intero esperimento.**

**Sanity check consigliato prima del corpus completo:** far girare l'obiettivo JEPA su **un
singolo dataset EMG omogeneo**, replicando uno split pubblicato. Non riproduce nessun
risultato, ma verifica che il training non collassi in condizioni semplici. Se collassa lì il
problema è nell'obiettivo; se collassa solo sul corpus completo è nell'eterogeneità.

**Calibrazione dell'architettura — 30–100M.** Encoder locale 2 vs 4 layer (**letto
separatamente per topologia**) · K ∈ {32, 64, 128} · bottleneck 1 vs 2 blocchi · raggio del
vicinato. **Criterio:** la configurazione **più piccola** che non mostra perdita evidente fra
pre- e post-bottleneck, letta con la sonda stratificata di §7.4.

### 10.2 Regime della ladder — dati fissi

**Sequenza:** (1) fissare D · (2) fissare contesto, sampler, masking, K e topologia — cioè il
manifest · (3) definire la regola di crescita di N · (4) eseguire.

> **Regime della scaling ladder — dati fissi.** La ladder principale varia il numero di
> parametri *N* mantenendo fisso il budget di dati osservati *D*. Tutti i rung ricevono lo
> stesso insieme e la stessa distribuzione di finestre di training, con identica durata del
> contesto in secondi, composizione per dataset e montaggio, budget di masking e numero di
> esposizioni. La ladder misura quindi la prestazione in funzione di *N* a informazione ed
> esposizione costanti: identifica il punto oltre il quale il corpus non sostiene più un
> aumento utile di capacità. Non è una ladder compute-optimal in stile Chinchilla, che
> richiederebbe un aumento di *D* non disponibile nel presente regime data-bound. Per
> distinguere saturazione informativa da semplice sotto-ottimizzazione, alcuni rung
> selezionati vengono prolungati a *2D* come controllo diagnostico. Con sequenza, sampler e
> topologia fissati, i FLOP crescono approssimativamente con *N*; vengono comunque riportati
> i FLOP misurati o stimati per ogni configurazione.

**Con D = 4 epoche nominali, 2D = 8 epoche sullo stesso corpus, non dati nuovi.**
**Seed:** 3 per rung, fino a 1B (rev. 2: il budget lo consente).

### 10.3 Budget dati D

#### D è un manifest, non un numero

Né i token del backbone né i channel-patch sono misure di informazione indipendente: i primi
dipendono da K; i secondi non sono osservazioni indipendenti (canali HD vicini correlati,
montaggi virtuali dagli stessi elettrodi, quote per topologia che cambiano la media di canali).

**L'oggetto congelato è il manifest**: stessi segmenti temporali, stesse quote per topologia,
stessa distribuzione dei montaggi, stessa molteplicità di esposizione.

| Contabilità | Una epoca | Quattro epoche | Significato |
|---|---|---|---|
| Time-patch | 3,6·10⁸ | 1,44·10⁹ | supporto temporale |
| **Source-channel-patch** | ~9·10⁹ | ~3,6·10¹⁰ | **volume sensoriale osservato** |
| Token del backbone | ~2,3·10¹⁰ | ~9,2·10¹⁰ | **solo costo architetturale** |

*Da ricalcolare dopo le esclusioni di §2.3 e la rimozione di Gait120 e di NinaPro DB9.*

`D_t` = time-patch unici, `D_c` = source-channel-patch unici; `D̃_t = 4·D_t`, `D̃_c = 4·D_c`.

**"Source" è vincolante.** Si conta **dopo il QC** ma **prima di**: channel masking, montage
dropout, sottocampionamenti HD, montaggi bipolari virtuali, ogni altra augmentation.

**Ambiguità nota:** contando prima del sottocampionamento, una griglia da 256 canali
contribuisce 256 anche se il modello ne vede ~32. `D_c` misura il **volume disponibile nel
manifest**, non quello consumato. Riportare anche una colonna di **consumo realizzato per
epoca**.

#### Quattro epoche: prior, non legge

Muennighoff et al. trovano che, **nei loro esperimenti linguistici**, ripetere fino a ~4
epoche produce differenze trascurabili rispetto a dati freschi; ~16 epoche è il punto stimato
di forte decadimento; il regime privo di valore è più verso le ~40. **È un risultato su
testo** — nel sEMG la ridondanza potrebbe essere molto maggiore.

#### Pilot sulle epoche — 100M

E ∈ {1, 2, 4, 8}, almeno due seed, leggendo: cross-soggetto **sul ramo sparso** ·
cross-dataset · sonde fisiche ai tre punti · dataset-ID probe · rango effettivo · loss JEPA e
ancore · **per topologia**, mai solo aggregati.

**Regola decisionale, congelata prima dei risultati.** Es.: si mantiene E = 4 se
`Δ₄→₈ < 0,25·Δ₂→₄` e nessun miglioramento supera due errori standard sulle metriche primarie.

**Assunzione da dichiarare:** il pilot valida E a 100M ma E si usa a tutti i rung. Si assume
il numero ottimale di epoche approssimativamente indipendente da N in questo intervallo; il
controllo 2D sul rung più grande è la mitigazione parziale.

#### Nota su Chinchilla

Il rung da 30M vede ~9·10⁹ source-channel-patch unici, molto oltre ciò che un rapporto di ~20
token per parametro assegnerebbe. **Serve a mostrare che il rung piccolo è fortemente
sovraesposto — condizione voluta in uno studio di saturazione — non a determinare il numero
di epoche, e non va presentato come previsione quantitativa.**

Con la contabilità conservativa (supporto temporale 3,6·10⁸ patch, canali HD vicini non
indipendenti) a 1B si è intorno a ~1,4 token per parametro su quattro epoche. **Sommato al
risultato di NeuroRVQ (§11), l'esito più probabile della ladder è saturazione precoce, forse
già fra 30M e 100M.** Il paper va scritto fin dall'inizio perché quello sia la tesi, non il
fallimento — e la giustificazione del rung a 3–5B va riesaminata a valle del pilot (§10.4).

Il risultato interessante è **stimare una data-reuse curve specifica per il sEMG**.

### 10.4 Regola di crescita di N, e controllo 2D

**Larghezza condivisa scalata ovunque; profondità scalata soprattutto nel backbone; K fisso.**

| Componente | Cosa scala | Cosa resta quasi fisso |
|---|---|---|
| Encoder spaziale locale | d_model, head, FFN | 2–4 layer, raggio del vicinato |
| Perceiver bottleneck | d_model, proiezioni | numero di cross-attention block, K |
| Backbone temporale | d_model, FFN, head, **soprattutto profondità** | struttura dei blocchi |
| Tokenizzazione | nulla | patch in ms, banda, masking |
| Contesto | nulla — asse separato | stessa durata in secondi |

**Perché non solo il backbone.** Con encoder a 384 e backbone a 1536 serve una proiezione
384→1536, e il modello da 1B riceve una rappresentazione già compressa da un front-end da
30M. Alla saturazione non si distingue fra quattro ipotesi diverse: **la ladder misurerebbe
il collo di bottiglia, non il valore della scala.**

**Perché non tutto proporzionalmente.** L'encoder locale applica un inductive bias geometrico
semplice; troppa capacità lì lo farebbe adattare a hardware e montaggio, con overfitting sui
pochi dataset HD.

| Taglia | d_model | Encoder locale | Perceiver | Backbone | K |
|---|---|---|---|---|---|
| 30M | 384 | 2 layer | 1 cross + 1 latent | 10–12 layer | 64 |
| 100M | 640 | 2 layer | 1 + 1 | 16–20 layer | 64 |
| 300M | 896 | 2–3 layer | 1 + 1 | 24–28 layer | 64 |
| 1B | 1.280–1.536 | 3 layer | 1 + 2 | 30–40 layer | 64 |
| 3–5B *(condizionale)* | 2.048–2.560 | 3–4 layer | 2 + 2 | 48–64 layer | 64–128 se necessario |

**Da ricalcolare** dopo aver fissato l'architettura: il banco a basi eterogenee (§4.2) e il
bias ibrido (§5.3) spostano il conteggio dell'encoder locale, e il budget prescritto è
75–85% backbone · 8–15% encoder locale · 5–10% Perceiver. **Verifica sul ramo sparso**, non
sull'aggregato.

**Aspect ratio al vertice.** A 5B con d=2560 e 48–64 layer fattorizzati si arriva a 96–128
layer di attenzione effettivi, rapporto d/L ~40–50: profondo e stretto, difficile da
stabilizzare e poco parallelizzabile. Tenere la profondità al bordo basso e compensare in
larghezza, o warmup e normalizzazione più aggressivi.

**Riesame del vertice — speculativo e parallelo (rev. 2).** Nella v10 il default era «non si
esegue», per un argomento di informazione per unità di costo: ~1,4 token per parametro a 1B
(§10.3) dicono che l'esito più probabile è la saturazione. L'esito atteso non cambia; **è
cambiato il costo**. Il disponibile è ~70.000–90.000 GPU-ora, scade il 7 gennaio 2027, e ciò
che non si spende si perde: il costo del vertice in GPU-ora è quasi nullo. Restano due costi
veri: il **tempo di ingegneria** (FSDP a 5B, stabilità di un modello profondo e stretto) e la
**priorità in coda** (a dicembre, col vertice, l'account supera la quota mensile). Il vertice
si lancia quindi **insieme alla ladder, non dopo**, a tre condizioni:

1. il rung a 1B gira già sullo **stesso percorso FSDP** (anche se DDP basterebbe, §5.5), così
   il vertice è solo un cambio di configurazione;
2. il rung a 1B ha superato il primo 10–20% del training **senza instabilità**;
3. il lancio avviene **entro il 6 dicembre**; dopo non si lancia più.

**Mai sul cammino critico:** non blocca la ladder né il modello finale, e se diverge si spegne
senza tentativi di recupero. Seed, controlli 2D attorno al flesso e asse dei soggetti (§10.6)
hanno comunque la precedenza: danno più risoluzione sul risultato che interessa. Il vertice
serve a mostrare la saturazione su un ordine di grandezza in più, non a smentirla.

**Quando aumentare K:** solo con una combinazione di — sonde buone prima del bottleneck ma
peggiori dopo · il backbone più grande non migliora il full-grid HD · latenti saturi o
condivisi · ricostruzione di porzioni di griglia molto peggiore di quella dei singoli canali ·
il modello migliora sui montaggi piccoli ma non su 128–256 canali. In quel caso ramo
controllato: stesso backbone, K=64 vs 128, stesso numero di token, compute comparabile.

#### Controllo 2D — strategia e condizione di validità

1. **100M fino a 2D** come curva di riferimento;
2. ladder completa a D;
3. estensione a 2D dei **due rung ai lati del flesso apparente**;
4. **più il rung più grande**, se non coincide.

**Condizione: il learning-rate schedule deve permettere una vera continuazione.** Non
completare a D un cosine decay quasi a zero, ripartire dallo stesso checkpoint e concludere
che il modello è saturo — si misurerebbe la ripartenza dell'ottimizzatore. Serve **warmup →
lunga fase stabile → decay separato** (WSD), o una policy di continuazione definita prima.

**Due checkpoint conservati per rung:**
- **post-decay** → è quello valutato a D;
- **pre-decay** → è quello da cui riparte la continuazione a 2D.

#### Metriche in funzione di N

Linear probe cross-soggetto **per topologia** · transfer su dataset mai visto e
cross-topologia ·
**sonde fisiche ai tre punti di §7.4** — in particolare: *la velocità di conduzione diventa
più linearmente decodificabile al crescere della scala?*

#### Regola di igiene

**Non cambiare insieme parametri, contesto, patch, masking e risoluzione del bottleneck.**

### 10.5 Contabilità del compute

**Budget: ~70.000–90.000 GPU-ora** (§1).

Stime preliminari: ~2,3·10¹⁰ token latenti per epoca; rung a 5B ~6,9·10²⁰ FLOP per epoca.

**Ma 6ND è un'approssimazione per transformer linguistici densi.** Front-end locale,
cross-attention del Perceiver e attenzione fattorizzata hanno conteggi diversi. Calcolo
finale **modulare**:

`FLOP_tot = FLOP_front-end + FLOP_locale + FLOP_Perceiver + FLOP_backbone + FLOP_teste`

o misurato col profiler. **Riportare i FLOP per ogni configurazione.**

**Ordine di grandezza in GPU-ora** — A100 in bf16 (312 TFLOP/s di picco), MFU 30–40% con
activation checkpointing, quattro epoche. **Stima, da sostituire coi FLOP e l'MFU misurati.**

Costi unitari a D: 5B ~6.000–8.000 · 1B ~1.200–1.600 · 300M ~400–500 · 100M ~120–160
GPU-ora. A 2D raddoppiano.

| Blocco | GPU-ora |
|---|---|
| Ablation a 30M e calibrazione 30–100M | ~1.500–2.000 |
| Pilot sulle epoche (100M, 2 seed) | ~1.000 (meno, coi rami di decay del WSD) |
| Asse dei soggetti (100M e 300M, 2 seed, §10.6) | ~3.000 |
| Run di conferma e shakedown del percorso FSDP | ~500–1.000 |
| Ladder 30M → 1B con 3 seed, più controlli 2D | ~10.000 |
| Training finale separato, se serve (a 1B) | ~1.500–3.000 |
| **Totale senza il vertice** | **~17.000–21.000** |
| Vertice a 3–5B, a D e a 2D | ~12.000–16.000 |
| **Totale col vertice** | **~29.000–37.000** |

Anche col vertice si resta sotto metà del disponibile: il resto scade comunque (piano
operativo, D0). Lanciato in parallelo alla ladder nella finestra di inizio dicembre, il
vertice entra nel calendario; resta condizionale per ragioni di ingegneria e di priorità in
coda, non di costo (§10.4). Il
padding al massimo del batch (§4.6) abbassa l'MFU dello stadio pre-bottleneck: altro motivo
per misurare il packing al passo 0.

### 10.6 Asse dei soggetti — 100M e 300M

La ladder varia N a D fisso e il pilot varia le epoche: manca l'asse che nel sEMG quasi
certamente lega, **il numero di soggetti**. Kaifosh et al. ottengono la generalizzazione fra
persone con dati di migliaia di partecipanti (curve di scaling nel numero di partecipanti:
**da verificare** sul paper). Il corpus qui ne ha nell'ordine di 600 (§1).

**Esperimento.** A 100M **e a 300M** (rev. 2), manifest **sottocampionato per soggetti**:
12,5 / 25 / 50 / 100%, stratificato per dataset e per topologia — le quote di §2.8 restano
quelle del manifest — con 2 seed. Due taglie dicono se l'esponente nei soggetti **dipende da
N**, che è la domanda vera di una scaling law. Stessa molteplicità di esposizione E: i passi
scalano con la frazione. A 100M la run al 100% è quella del pilot (§10.3), non va ripetuta.

**Controllo.** La frazione 25% anche **a passi uguali al 100%** (16 epoche nominali), per
separare «meno soggetti» da «meno passi di ottimizzazione».

**Lettura.** Le metriche del pilot in funzione del numero di soggetti; pendenza in log-log sul
cross-soggetto **per topologia** e sul transfer su dataset mai visto.

**Perché vale il costo** (~3.000 GPU-ora, §10.5). Trasforma «saturazione precoce in
N» in «ecco l'esponente nei soggetti»: un risultato azionabile per chi raccoglie dati, e la
risposta alla domanda che un revisore farà comunque. Quantifica anche il prezzo
dell'esclusione di EPN-612 (§2.3). Con N ed epoche, è ciò che rende la scaling law di §11 una
legge su più di una fetta.

---

## 11. Stato dell'arte

| Lavoro | Taglia | Note |
|---|---|---|
| **NeuroRVQ (Barmpas et al., Imperial + Cogitat)** | tokenizer EMG **144M**, EMG-FM **5,9M** | **Codice e pesi rilasciati.** Tokenizer RVQ multi-scala con loss phase-aware, **a 1 kHz con patch da 200 ms, addestrato solo su emg2pose + emg2qwerty** (§6.3). Ha documentato il problema degli split e del label leakage via padding su DB5 |
| TinyMyo (Fasulo et al., 2025) | 3.6M | Nel suo paper: DB5 89,4%. **Sotto valutazione terza: 25,26%** |
| PhysioWave (Chen et al., 2025) | 5M / 15M / 37M | Wavelet-transformer, ~823 GB di pretraining. Il 37M batte Moment (385M) |
| EMGNet (bioRxiv 2025) | 1.75M | 197 h, 1.667 soggetti |
| Moment | 385M | generalista time-series, battuto da modelli 10× più piccoli su EMG |
| HEAR (EEG) | 3.1M tiny / base | Saturazione tiny→base attribuita ai dati disponibili |
| Kaifosh et al. (Nature 2025) | — | Interfaccia neuromotoria generica di Meta: generalizzazione fra persone ottenuta con dati di migliaia di partecipanti. Riferimento per l'asse dei soggetti (§10.6); il suo dataset è la prima colonna qui sotto |

**Risultati EMG di NeuroRVQ**, sotto valutazione con split subject-independent — **fine-tuning
completo** con testa lineare, split per soggetto 7:1:2, non encoder congelato (§8):

| Modello | Gesti discreti (BAcc) | EPN-612 (Acc) | NinaPro DB5 (Acc) | UCI-EMG (Acc) |
|---|---|---|---|---|
| PhysioWave | 54,70 | 90,30 | 24,91 | 56,52 |
| TinyMyo | 39,70 | 84,68 | 25,26 | 85,99 |
| **NeuroRVQ-EMG (5,9M)** | **70,80** | **94,65** | **41,36** | **89,43** |

Due letture. Primo: **DB5 al 25% per TinyMyo contro l'89,4% del suo paper** — conferma da
terzi che fuori dal suo esperimento non regge, ed è il tipo di risultato che il protocollo di
§8 è costruito per produrre.

Secondo, e scomodo: **la tesi esplicita degli autori è che l'ingrediente critico sia la
fedeltà della tokenizzazione, non la scala del modello o la complessità architetturale** — con
un FM da 5,9M. **È la controevidenza più forte alla scommessa sulla scala di questo
progetto**, più netta di PhysioWave, e va affrontata nel paper invece che scoperta in review.

Nota anche che **Dario Farina è fra i coautori** — stesso riferimento della letteratura
metodologica sulla velocità di conduzione (§7.2).

**Rivendicazione di paper, più difendibile della taglia:** un FM *topology-aware* per sEMG
dell'arto superiore — che tratta la simmetria come un **dato dichiarato per montaggio** e non
come un'assunzione architetturale, supportando anelli, griglie HD e montaggi anatomici sparsi
**nello stesso modello e nello stesso spazio di rappresentazione** — con simmetrie fisiche
corrette, front-end a kernel continui indipendente dal sample rate, rappresentazioni
**verificate** contro osservabili fisici, protocollo di valutazione onesto, **la prima scaling
law a dati fissi per il sEMG — in N, in epoche e nel numero di soggetti (§10.6)** — e **la
prima data-reuse curve specifica per il sEMG**.

Con lo scope ristretto, è il "nello stesso modello" a portare il peso della rivendicazione.

### Riferimenti e risorse

- **NeuroRVQ**: arxiv.org/abs/2510.13068 · github.com/KonstantinosBarmpas/NeuroRVQ ·
  HF: ntinosbarmpas/NeuroRVQ
- **Beyond Accuracy** (arXiv 2605.17562) — probing block-wise, robustezza a rumore e channel
  dropout sui FM EEG. Precedente diretto della sonda stratificata di §7.4
- **PhysioWave** (Chen et al., 2025)
- **Muennighoff et al.**, *Scaling Data-Constrained Language Models* — origine del prior a 4
  epoche (§10.3), da validare fuori dal dominio linguistico
- **data2vec** (Baevski et al.) — insegnante EMA con target latenti su segnali continui;
  template più vicino di I-JEPA. data2vec 2.0 e HuBERT/BEST-RQ sono i precedenti per la
  combinazione target discreti + predizione latente (§6.3)
- **Perceiver / Perceiver IO** (Jaegle et al.) — bottleneck latente
- **CKConv** (Romero et al.) — continuous kernel convolutions, base per §4.2
- **Farina & Merletti** — metodi di stima della velocità di conduzione da sEMG di superficie.
  Non ML, e proprio per questo necessario: la sonda di §7.2 misura una grandezza con una
  letteratura metodologica quarantennale
- **LUNA, HEAR** — FM EEG channel-agnostic; l'EEG ha affrontato il montaggio variabile prima
- **Kumar et al.**, *Fine-Tuning can Distort Pretrained Features and Underperform
  Out-of-Distribution* — a favore di encoder congelato + testa piccola
- **Kaifosh, Reardon et al.**, *A generic non-invasive neuromotor interface for
  human-computer interaction* (Nature, 2025) — generalizzazione fra persone con dati di
  migliaia di partecipanti; riferimento per l'asse dei soggetti (§10.6) e sorgente del
  benchmark Discrete Gestures
- **NaViT** (Dehghani et al., *Patch n' Pack*) — packing di sequenze a lunghezza variabile
  in un unico batch (§4.6)
- **FMA / UBERON** — ontologie anatomiche per il vocabolario della tassonomia (§4.5)

Codebase JEPA: `facebookresearch/ijepa` (immagini) · `facebookresearch/jepa` (V-JEPA video,
mask collator e attentive probe a backbone congelato) · `facebookresearch/vjepa2` ·
`facebookresearch/eb_jepa`. **Da leggere, non da forkare**: sono tutti costruiti su griglie
2D/3D e la pipeline dati non è riutilizzabile con canali variabili. Quel che serve è la
meccanica del training loop — schedule dell'EMA, stop-gradient, design del predictor,
dettagli anti-collasso.

---

## 12. Questioni aperte

**Chiuse rispetto alla v9:**
- **Tokenizer a frequenza fissa (§6.3):** vista tokenizer canonica per i dataset ≥ 1 kHz,
  ancora mascherata su DB5, pesi non rinormalizzati — salvo il bivio 20–90 Hz, che è una
  verifica e non una scelta
- **Granularità della tassonomia anatomica (§4.5):** gerarchia ad albero, annotazione al
  livello più fine realmente noto, precisione per canale
- **Gate del front-end (§4.2, §7.1):** consistenza al ricampionamento; il probe dataset-ID
  resta come diagnostica relativa
- **Ripiego «solo intra-canale» dell'ancora RVQ (§6.3):** sostituito da «solo intervalli
  temporali mascherati su tutti i canali»

**Restano aperte:**

- **Decoder a query e collocazione dei target JEPA (§5.6)** — nuova e prioritaria: va decisa
  al passo 6, e l'ablation del passo 7 ci poggia sopra
- **Ancora RVQ (§6.3):** esito delle verifiche V1–V4; soglie X (errore di ricostruzione) e Y
  (dataset-ID dai codici), da congelare prima di guardare; bivio 20–90 Hz; test di leakage di
  fase run 4 → run 5 **con il probe più forte costruibile, non lineare**
- **Dataset di Kaifosh et al. (§2.1):** adottarlo, e in che ruolo — benchmark mai visto o
  anche pretraining
- Quote di campionamento per topologia (§2.8) — determinano la media di canali e quindi `D_c`,
  i gradienti al ramo sparso, e la frequenza con cui l'encoder locale vede griglie dense.
  **Ultimo parametro con conseguenze in tre direzioni; da fissare al passo 4**
- Dataloader (§4.6), **misurabile già al passo 0**: montaggi virtuali al volo o precompute,
  che retroagisce su `D_c`; e layout del batch — padding, packing o bucketing per micro-batch,
  l'ultimo solo modificando §2.7
- Pesi di campionamento dentro la classe di quota (§2.8), da fissare al passo 4
- Frequenza al front-end dei dataset non-HD a 2 kHz: 1 kHz come da §4.1, o nativa
- Soglie del gate di consistenza al ricampionamento (§7.1), da congelare prima dei risultati
- Soglia numerica della regola decisionale del pilot (§10.3), da congelare prima dei risultati
- **Condizioni per il lancio speculativo del rung a 3–5B (§10.4)**: confermarle o cambiarle
  prima della finestra di inizio dicembre
- Parametri dello schedule WSD e della policy di continuazione (§10.4)
- Frazioni, stratificazione e controllo a passi uguali dell'asse dei soggetti (§10.6)
- Dimensione di patch (in ms), finestra di contesto, schedule di masking — con il vincolo dei
  200 ms e della quota di slab su tutti i canali, se l'ancora RVQ sopravvive (§6.4)
- Probabilità del dropout del livello muscolo (§5.2)
- Vocabolario dei compartimenti e funzione atlante (§4.5): la bozza va rivista da una persona
- Licenze non ancora verificate per sei dataset (§2.2)
- Dimensione della copia preprocessata, che determina se tenere o buttare il grezzo (§2.5)

**Fatti da verificare** — riportati a memoria o da fonte secondaria; sono compiti, non dati:

1. Frequenze native di NinaPro DB8 (~1111 Hz) e DB10 (~1926 Hz); presenza di amputati in DB8
   e DB10; varianti del montaggio NinaPro (DB6 a 14 elettrodi, DB8 a 16); disposizione
   interna delle fasce di GRABMyo. (putEMG e Hyser: raccolti da fonte ufficiale, da firmare)
2. Sovrapposizione di soggetti fra i DB NinaPro
3. Conteggio dei soggetti del corpus (~600) e delle ore dopo la rimozione di DB9
4. Dal codice del repo NeuroRVQ: banda di filtraggio usata nel pretraining del tokenizer
   (20–90 Hz?), normalizzazione attesa in ingresso, e se il transformer mescola canali
   diversi
5. Sovrapposizione fra i soggetti visti dal tokenizer NeuroRVQ e i soggetti di test di §8;
   licenza del checkpoint
6. Lista degli 11 muscoli di Camargo 2021
7. Kaifosh et al.: presenza e forma delle curve di scaling nel numero di partecipanti;
   accesso, licenza e dimensioni del dataset rilasciato
8. Orientamento della fascia documentato, dataset per dataset, per tutti gli anelli e le
   fasce (serve alla funzione atlante di §4.5)
9. `flash_attn_varlen_func` in `cineca-ai/4.3.0`: l'import è verificato, manca la **prova
   funzionale** forward + backward in bf16 su A100 (§4.6)
10. MFU reale sul Booster, per sostituire la stima di §10.5
