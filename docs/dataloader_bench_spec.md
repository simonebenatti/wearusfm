# Benchmark sintetico del dataloader — spec (Passo 0), rev. 2

Fonti: [`docs/fm_emg_reference_v10.md`](fm_emg_reference_v10.md) rev. 3 — §2.1, §2.7, §2.8,
§3.4, §4.1, §4.6, §5, §6.4, §8, §9 — e [`docs/piano_operativo_v10.md`](piano_operativo_v10.md),
passo 0, D2 e D6.
Solo specifica. Nessun codice in questo documento.

**Cosa cambia rispetto alla rev. 1:**
- i «tre bracci» erano **due assi ortogonali** confusi in uno: come si costruiscono i montaggi
  (lato CPU e I/O) e come il batch arriva alla GPU (lato GPU). Le metriche sono diverse per
  ciascun asse
- l'unità di campionamento è il **montaggio intero**, non il gruppo di canali
- la distribuzione ha **frequenza nativa, contesto e pesi**, non solo conteggi di canali
- i dati sintetici si **scrivono su disco e si rileggono**: senza I/O il braccio precompute è
  vuoto
- c'è un **ritmo richiesto**, un consumatore simulato, e due regole di decisione da congelare
  prima (D6a, D6b)
- FlashAttention: l'import non è una verifica, e il kernel **non accetta bias additivi
  arbitrari** — conseguenza per l'encoder locale (§6)

---

## 1. Le due domande

| | Domanda | Dove si vede | Si misura con |
|---|---|---|---|
| **Asse M** — montaggi | I montaggi bipolari virtuali si calcolano **al volo** o si **precomputano**? | CPU dei worker e I/O | solo il dataloader, contro un consumatore simulato |
| **Asse L** — layout | Il batch a canali variabili arriva alla GPU con **padding + maschera**, **packing**, o **bucketing per micro-batch**? | tempo di passo dello stadio pre-bottleneck | un modello proxy su GPU |

L'asse M retroagisce su `D_c` (v10 §4.6, §10.3): un montaggio precomputato è dato fisso, uno
generato al volo è augmentation stocastica. Il benchmark misura; non decide la semantica di
`D_c`, che va riportata a chi legge.

**Non sono domande di questo benchmark** le quote finali (D9) né patch, contesto e masking
(D10): qui sono parametri da variare, per sapere quanto la risposta ne dipende.

## 2. Unità di campionamento e classi di quota

**Un campione è una finestra temporale di un montaggio intero:** tutti i canali registrati
insieme, con tutti i loro gruppi. «La topologia si dichiara per gruppo di canali» (v10 §3.4,
§4.5) è un metadato per encoding posizionale e simmetrie; non spezza il campione. Spezzare
NinaPro in un anello da 8 e un gruppo mirato da 4 toglierebbe al modello la sinergia fra i due
gruppi, e produrrebbe un caso sparso con C = 4, sotto il range 8–32 di v10 §5.

**La classe di quota (v10 §2.8) è del montaggio:**

| Classe | Montaggi |
|---|---|
| **A — radi / anatomici** (il caso di deployment) | NinaPro DB2/3/4/6/7/8/10, Camargo |
| **B — anelli e fasce** | emg2pose, emg2qwerty, NinaPro DB5, GRABMyo, putEMG |
| **C — griglie HD** | CapgMyo, CSL-hdemg, Hyser |

Il generatore etichetta ogni campione con la **classe di origine** e con la **classe
presentata** al modello (un montaggio virtuale ricavato da una griglia HD nasce in C ma si
presenta come A o B): servono entrambe per la scelta di D9a.

## 3. Distribuzione sintetica

Solo i dataset che entrano nel pretraining (esclusi EPN-612, UCI-EMG, DB1, DB9, Gait120).

| Montaggio | Classe | C nominale | Gruppi | fs nativa (Hz) | Stato del dato |
|---|---|---|---|---|---|
| emg2pose | B | 16 | 1 anello | 2000 | v10 §2.1 |
| emg2qwerty | B | 32 | 2 anelli da 16, speculari | 2000 | v10 §2.1 |
| NinaPro DB5 | B | 16 | 2 anelli Myo sullo stesso avambraccio | 200 | v10 §2.1 |
| GRABMyo | B | 28 | fasce su avambraccio (16) e polso (12) | 2048 | v10 §2.1; disposizione interna da verificare |
| putEMG | B | 24 | 3 fasce da 8, a 45°, primo elettrodo sull'ulna | 5120 | **raccolto** dalla pagina ufficiale, da firmare |
| NinaPro standard | A | 12 | anello a 8 + 4 mirati | 2000 | v10 §2.1 |
| NinaPro DB6 / DB8 | A | 14 / 16 | da verificare | 2000 / ~1111 | da verificare |
| Camargo 2021 | A | 11 | 11 mirati | 1000 | v10 §2.1 |
| CapgMyo | C | 128 | 8 strisce 2×8 | 1000 | v10 §2.1 |
| CSL-hdemg | C | 168 | 1 griglia | 2048 | v10 §2.1 |
| Hyser | C | 256 | 4 griglie da 64 | 2048 | **raccolto** da PhysioNet, da firmare |

**Frequenza al front-end.** Default secondo v10 §4.1: HD nativi, Myo a 200 Hz, tutto il resto
a 1 kHz. Variante di sensibilità: **tutto nativo**, che è il caso peggiore in byte. Se i dataset
non-HD a 2 kHz entrino nel front-end a 1 kHz o nativi è da confermare (§13).

**Perché la frequenza conta.** Il costo per campione scala con C × fs × contesto, non con C.
Una finestra di 4 s di Hyser (256 × 2048 Hz) sono ~2,1 milioni di valori; una di DB5
(16 × 200 Hz) ~12.800. **Il rapporto è oltre 150, non 32.**

**Pesi dentro la classe.** Sono un parametro del manifest (D9), e le ore per dataset non sono
ancora note. Per il costo contano solo in classe C, dove i montaggi differiscono molto; si
usano quindi due scenari che delimitano la risposta: **pessimistico** (solo Hyser) e **misto**
(uniforme sui tre). Nelle classi A e B: proporzionale alle ore dove note (emg2pose 370,
emg2qwerty 346, NinaPro ~90), uniforme altrove.

**Trasformazioni che decidono il C presentato:**
- **montage dropout** (v10 §8): per canale e per gruppo intero (es. un solo bracciale di
  emg2qwerty). Probabilità di lavoro: 0,2 per gruppo, 0,1 per canale — da confermare
- **griglie HD:** con probabilità `p_piena` il campione resta a griglia piena; altrimenti
  diventa un montaggio bipolare virtuale con C ≤ 32 (v10 §2.7). `p_piena` è **il parametro a
  cui il costo è più sensibile**, più delle quote: si varia in {0,25 · 0,5 · 0,75}

## 4. Lavoro per campione da simulare

Nel tempo misurato entra tutta la catena, stadio per stadio, con il tempo di ciascuno:

1. lettura della finestra da disco (§5) e conversione da int16
2. montage dropout, oppure sottocampionamento HD e costruzione dei montaggi bipolari virtuali
3. **augmentation di v10 §8:** rumore realistico (drift sotto i 20 Hz, rete, contaminazione
   ECG) e **time warping** — un'interpolazione su C × campioni, probabilmente lo stadio più
   caro. Nella rev. 1 mancava
4. maschere: tre scale temporali (20–50 ms · 100–300 ms · 500 ms–2 s) più masking spaziale
   (~40% del budget: canali singoli, gruppi contigui, porzioni di griglia); opzionale la quota
   di slab ≥ 200 ms su tutti i canali per l'ancora RVQ (default: **spenta**)
5. collate: **raggruppamento per frequenza** — il front-end gira una volta per gruppo di fs e i
   token, definiti in ms, si riuniscono dopo. **Il tempo non si padda mai**
6. pinning e trasferimento sulla GPU (in int16: la conversione a virgola mobile si può fare
   sulla GPU e dimezza il trasferimento)

Il tempo per stadio serve a rendere **azionabile** un fallimento: dice cosa spostare sulla GPU
prima di ricorrere al precompute.

## 5. I/O reale

Tensori casuali generati in RAM non fanno I/O: il precompute, il cui vantaggio è leggere
~8 volte meno byte di una griglia a 256 canali, non si vedrebbe.

- Il contenuto resta casuale, ma gli shard sintetici si **scrivono su disco nel formato
  previsto per l'ingest** (int16 memory-mapped, v10 §4.3) e si rileggono dal percorso vero.
- Su **`$WORK`** (dove staranno i dati) e su **`$FAST`**, per sapere quanto vale spostarli.
- Gli shard si scrivono in un **job precedente**: così il primo passaggio del benchmark è
  davvero a **cache fredda**. Il secondo passaggio dà la misura a **cache calda**. Si riportano
  entrambe.
- Dimensione di lavoro: ~100–200 GB. Layout su disco: tempo × canali (una finestra = una
  lettura contigua). Il layout canali × tempo, che favorisce la lettura di sottoinsiemi di
  canali, è una variante secondaria solo per il braccio al volo in classe C.
- Braccio precompute: un insieme fisso di montaggi virtuali già pronti, stesso formato.

## 6. I bracci

**Asse M — montaggi** (misurato col solo dataloader):
- **M1 — al volo:** si legge la griglia piena e si costruisce il montaggio nel worker.
- **M2 — precompute:** si legge il montaggio già pronto.

**Asse L — layout** (misurato col proxy di §7, con i dati del braccio M1):
- **L1 — padding + maschera:** padding in C fino al massimo del batch (v10 §2.7).
- **L2 — packing:** lista piatta di token con indici di campione. L'**encoder locale** si
  scrive come attenzione sui vicini con `gather`: non c'è padding per costruzione e il bias
  geometrico di v10 §5.3 si calcola sulle distanze dei vicini raccolti. `flash_attn_varlen_func`
  serve **solo** per la cross-attention del Perceiver (K latenti verso i C token di ciascun
  campione, con `cu_seqlens` distinti per query e chiavi).
- **L3 — bucketing per micro-batch con accumulo del gradiente:** micro-batch omogenei in C, e
  la miscela di topologie si conserva **per passo di ottimizzazione** invece che per
  micro-batch. Niente padding, niente kernel speciali. **Misurarlo non impegna; adottarlo
  modifica il «mai bucketing» di v10 §2.7 ed è decisione di TU.** Da misurare anche lo
  squilibrio fra rank, che qui è il rischio.

**FlashAttention — stato.** `flash_attn` 2.5.5 è installato e `flash_attn_varlen_func` si
importa. **Non è ancora una verifica:** precondizione del braccio L2 è una prova forward +
backward in bf16 su A100, in cross-attention con lunghezze diverse per query e chiavi.
Dall'interfaccia pubblica, il kernel accetta maschera causale, finestra scorrevole e pendenze
ALiBi (un bias lineare nella distanza *di indice*), **non un bias additivo arbitrario**: per
questo il bias geometrico non passa da lì, e l'encoder locale usa il `gather`.

## 7. Modello proxy e consumatore simulato

**Consumatore simulato (asse M).** Un processo per rank che consuma un batch ogni `t_passo`,
con una barriera fra i 4 rank a ogni passo: in DDP il rank più lento ferma gli altri, quindi
l'attesa si misura come **massimo sui rank**. `t_passo` per il 30M e per il 100M si ricava dal
ritmo richiesto (§9).

**Proxy su GPU (asse L).** Front-end per gruppo di fs (una convoluzione con gli stessi FLOP del
banco di v10 §4.2), encoder locale a 2 layer con attenzione sui vicini, cross-attention verso
K = 64 latenti; il backbone è sostituito da prodotti di matrici con FLOP equivalenti, così il
tempo di passo è realistico. Forward + backward in bf16, a d = 384 e d = 640 (rung da 30M e
100M, v10 §10.4). Non serve che impari: serve che costi quanto il modello vero.

## 8. Metriche

**Asse M — lato dataloader:**
- finestre/s per GPU, **token/s** (C × patch per finestra) e MB/s letti: le finestre/s da sole
  non sono confrontabili fra miscele diverse
- **frazione di tempo in attesa** del consumatore simulato, massimo sui 4 rank, a `t_passo` del
  30M e del 100M — **è la metrica che decide.** La varianza del tempo per batch non lo è: la
  coda di prefetch l'assorbe, e ciò che ferma il training è la coda che si svuota
- p50 / p95 / p99 dell'intervallo fra batch
- tempo per stadio (§4), p50
- occupazione di CPU per worker; memoria dell'host di picco
- tutto a cache **fredda** e **calda**, su `$WORK` e su `$FAST`

**Asse L — lato GPU:**
- **frazione di token di padding**, prima analitica — 1 − media(C) / max(C) per batch, dalla
  sola distribuzione, senza GPU — poi misurata. Con le quote provvisorie, `p_piena` = 0,5 e
  classe C pessimistica vale **circa l'80%** (C medio ~48 contro un massimo di 256): è il
  numero da smentire
- token utili/s del proxy; tempo di passo p50 / p95
- memoria di picco sulla GPU
- squilibrio fra rank: massimo / media del tempo di passo

**Worker.** 8 worker per GPU più i 4 processi principali fanno 36 processi su 32 core: si
misurano **6, 7 e 8** worker per GPU, col processo principale vincolato agli stessi 8 core del
suo rank, come farebbe SLURM.

## 9. Ritmo richiesto e regole di decisione

Ritmo richiesto per GPU (piano operativo, passo 0):

  `finestre/s ≈ MFU · 312·10¹² / (6 · N · K · contesto/patch)`

cioè ~70 finestre/s a 30M e ~20 a 100M con K = 64, contesto 4 s, patch 25 ms, MFU 40%.
**Il caso vincolante è il modello piccolo.** Stima da rifare quando esiste il modello vero.

**Regole proposte — TU le conferma o le cambia, e si committano in `docs/decisioni.md` prima di
lanciare (D6):**

- **D6a — montaggi.** M1 (al volo) resta il default se, col consumatore del 30M, la frazione
  di tempo in attesa è ≤ 2% **e** il ritmo a vuoto è ≥ 1,5× il richiesto, a cache calda e
  fredda, su `$WORK`. Se fallisce: si spostano sulla GPU gli stadi più cari secondo il tempo
  per stadio, e si rimisura. **Solo se fallisce ancora, M2** — con la conseguenza su `D_c`
  scritta nel manifest.
- **D6b — layout.** L1 è escluso in partenza se la frazione di padding misurata supera il 50%.
  Fra i rimanenti vince chi dà più token utili/s nel proxy; a parità entro il 10% vince il più
  semplice, nell'ordine L1 → L3 → L2. L3 si può adottare solo con l'assenso di TU (§6).

## 10. Griglia dei parametri (D2)

Non un fattoriale completo: **una base, e un fattore alla volta**.

| Parametro | Base | Varianti |
|---|---|---|
| Quote A / B / C | 40 / 35 / 25 (piano, D2) | 25 / 50 / 25 · 10 / 50 / 40 |
| `p_piena` (classe C) | 0,5 | 0,25 · 0,75 |
| Composizione della classe C | solo Hyser (pessimistica) | uniforme sui tre |
| Contesto | 4 s | 2 s · 8 s |
| Patch | 25 ms | — |
| Frequenza al front-end | default v10 §4.1 | tutto nativo |
| Slab per l'ancora RVQ | spento | acceso |
| Batch per rank | 64 campioni | 32 · per L2 anche a budget di token |
| Worker per GPU | 8 | 6 · 7 |

**Durata.** 50 batch di riscaldamento scartati, poi almeno 500 batch per configurazione. La
misura è valida se il p95 della prima metà e quello della seconda differiscono meno del 10%.

**Budget.** ≤ 40 GPU-ora = 320 ore locali, cioè ~10 ore di nodo. Il piano ne prevedeva 20: va
alzato in D1, e il costo resta irrisorio rispetto al disponibile.

## 11. Hardware e ambiente

- 1 nodo `boost_usr_prod`, 4× A100 SXM4 64 GB, 32 core, 512 GB di RAM. Il nodo intero costa
  32 ore locali l'ora anche se l'asse M non usa le GPU.
- `module load profile/deeplrn && module load cineca-ai/4.3.0`, venv su `$WORK` con
  `--system-site-packages` ([scripts/setup_env.sh](../scripts/setup_env.sh)). PyTorch
  2.2.0a0, CUDA 12.1, `flash_attn` 2.5.5.
- Scrittura degli shard sintetici: job separato e precedente (§5).

## 12. Output

`results/step0/<braccio>/<configurazione>.json`, con tutte le metriche di §8 e i parametri di
§10; `docs/report_step0.md` applica D6a e D6b e riporta, senza deciderla, la conseguenza su
`D_c`. **Un passo è chiuso quando i file esistono e i numeri rispettano le soglie congelate**
(piano operativo §7), non quando l'agente riferisce che è andato bene.

---

## 13. Stato dei fatti e cose da decidere

**Raccolto da fonte ufficiale, da firmare (D4):**
- Hyser: 256 canali a 2048 Hz, 20 soggetti, due sessioni in giorni diversi (PhysioNet). Le 4
  griglie da 64, due sul lato flessorio e due sull'estensorio, vengono da una fonte secondaria.
- putEMG: 24 elettrodi in 3 fasce da 8, a 45°, primo elettrodo di ogni fascia sull'ulna,
  numerazione oraria; 5120 Hz; registrazione **monopolare**; licenza **CC BY-NC 4.0** (pagina
  del dataset e paper).
- `flash_attn`: l'interfaccia espone maschera causale, finestra scorrevole e pendenze ALiBi;
  nessun argomento per un bias additivo arbitrario (repository ufficiale).

**Da verificare:** disposizione interna delle fasce di GRABMyo; montaggi di NinaPro DB6 e DB8;
ore per dataset, per i pesi dentro la classe; prova funzionale di `flash_attn_varlen_func`.

**Da decidere (TU):**
1. Confermare la base e le varianti di §10 (D2) e le probabilità di montage dropout.
2. Congelare le soglie di D6a e D6b.
3. Se L3 è adottabile, nel caso vinca: modifica v10 §2.7.
4. Se i dataset non-HD a 2 kHz entrano nel front-end a 1 kHz (v10 §4.1) o nativi: cambia di 2×
   i byte di un terzo del corpus.
5. Alzare il budget del passo 0 a 40 GPU-ora (D1).
