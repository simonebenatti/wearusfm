# Proposta — D10: patch, contesto, masking (passo 6): BOZZA da firmare prima di scrivere il modello

**Stato: bozza di AG del 02/10/2026, NON firmata.** D10 va chiusa al passo 6, «prima di scrivere il modello» (piano, §9), e il codice del passo 6
va finito entro l'01/11. Riferimenti: piano operativo §Passo 6 (proposta di partenza); v10 §4.1-4.2 (front-end in ms), §6.3 (ancore e ancora RVQ),
§6.4 (masking multiscala), §10.3 (`D_t`); D2 (parametri provvisori del passo 0: contesto 4 s, patch 25 ms, «non vincolanti»); gate D8 (superato
con patch 25 ms). **Le parti segnate «[numeri dal costruttore]» aspettano la tabella del job 59180449** (tempo perso con finestre da 1, 2, 4 e 8 s,
per dataset): finché non c'è, quelle proposte sono condizionate.

**Proposta di partenza del piano** (passo 6): «patch 25 ms (coerente con i 3,6·10⁸ time-patch per epoca della v10), contesto 4 s, scale di masking
di v10 §6.4; se l'ancora RVQ sopravvive, una quota di slab ≥ 200 ms su tutti i canali, allineati alla griglia del tokenizer». L'ancora RVQ è
sopravvissuta (D5b firmata, accesa su 14 dataset; DB10 in misura): il vincolo degli slab vale.

Cosa si decide qui: **(1)** lunghezza della patch; **(2)** contesto, cioè la lunghezza della finestra vista dal modello; **(3)** lo schema di masking
(scale, proporzioni, quota degli slab RVQ, frazione mascherata). Non si decide: dove si leggono i target JEPA (D11, v10 §5.6), le finestre dei target
spettrali (v10 §6.3: «La finestra del target NON deve coincidere con la lunghezza della patch mascherata»), la frequenza al front-end dei dati convenzionali a 2 kHz (1 kHz o nativa: elenco aperto di
v10 §12), le finestre di valutazione dell'harness (passo 5).

## (1) Lunghezza della patch

**Vincoli, con i numeri del corpus:**

- **Griglia del tokenizer.** NeuroRVQ lavora a 1 kHz con patch da 200 ms (v10 §6.3). Gli slab dell'ancora devono essere allineati a quella griglia:
  la patch deve dividere 200 ms. Lo fanno 20, 25, 40 e 50 ms; non 30 o 60.
- **Campioni per patch** (v10 §4.1: la patch è in ms, la griglia è quella del dataset; il front-end a kernel continui gestisce numeri non interi):

  | fs al front-end | Dataset | 20 ms | 25 ms | 40 ms | 50 ms |
  |---|---|---|---|---|---|
  | 200 Hz | NinaPro DB5 (Myo) | 4 | 5 | 8 | 10 |
  | 1000 Hz | CapgMyo, Camargo; tutti i convenzionali se portati a 1 kHz (v10 §4.1) | 20 | 25 | 40 | 50 |
  | 1926 Hz | NinaPro DB10 | 38,5 | 48,2 | 77,0 | 96,3 |
  | 2000 Hz | Meta, NinaPro DB2/3/4/6/7/8, Zhang (se nativi) | 40 | 50 | 80 | 100 |
  | 2048 Hz | Hyser, CSL-hdemg, GRABMyo | 41,0 | 51,2 | 81,9 | 102,4 |
  | 5120 Hz | putEMG | 102,4 | 128 | 204,8 | 256 |

  Il caso stretto è DB5: 5 campioni a 25 ms, 4 a 20 ms, con una banda di 100 Hz.
- **Scala corta del masking** (v10 §6.4: 20-50 ms, «contenuto spettrale locale»): la maschera più corta è una patch. A 25 ms la scala corta è 1-2
  patch; a 50 ms coincide con una patch sola e smette di essere una scala a sé.
- **Gate D8**: misurato con patch 25 ms (e 100 ms di contesto del kernel). `results/step3/RIEPILOGO.md`: «se D10 cambia la lunghezza della patch,
  se rifare il gate e' una decisione di Simone». Il gate è girato su CPU, 0 GPU-ora.
- **Costo.** Passi temporali per finestra = contesto / patch (4 s: 200 a 20 ms, 160 a 25 ms, 100 a 40 ms, 80 a 50 ms); i token del backbone sono K
  per passo. A calcolo fissato, le ore di dati viste per FLOP crescono con la patch. Il ritmo richiesto misurato al passo 0 (~68 finestre/s per il
  30M, margine del dataloader 3,5×) è calcolato a 25 ms.
- **`D_t`** (v10 §10.3) conta time-patch: scala come 1/patch. Il costruttore del manifest lo calcola a 25 ms (`--patch-ms 25`).

**Opzioni:** 20 ms (+25% di token, scala corta più fine, DB5 a 4 campioni); **25 ms** (piano); 40 o 50 ms (−37,5% o −50% di token, la scala corta
del masking sparisce come scala distinta).

**Proposta di AG: 25 ms.** Divide la griglia da 200 ms (8 patch per finestra del tokenizer), sta dentro la scala corta di v10 §6.4, il gate D8 e il
passo 0 sono già misurati lì: cambiarla costerebbe un nuovo gate e un nuovo conto del ritmo, senza un motivo dai dati.

## (2) Contesto

**Vincoli:**

- **Masking lungo** (v10 §6.4: 500 ms-2 s, «contesto motorio»): una maschera da 2 s in una finestra da 4 s lascia metà finestra visibile; in una da 2 s
  non resterebbe niente. Con 4 s le tre scale ci stanno tutte.
- **Tratti contigui più corti della finestra.** Una finestra non può attraversare il confine di una prova (né, se si firma D9 §f, un salto dell'asse
  dei tempi): il tempo in tratti più corti del contesto non si campiona mai. **Verificato:** CapgMyo ha prove da 1 s esatto (1.440 prove, 0,4 h,
  `WINDOW_SAMPLES = 1000` nell'ingest): con un contesto fisso di 2 s o più, **CapgMyo sparisce tutto** (classe C). DB10 ha 18 prove da 1 campione
  (job 59180438): trascurabili. Gli altri dataset: **[numeri dal costruttore]**.
- **Ancora RVQ:** un target esiste solo per finestre da 200 ms interamente dentro una regione mascherata (v10 §6.3): nessun vincolo ulteriore sul
  contesto.
- **Costo:** lineare nel contesto per i token, quadratico solo nell'attenzione temporale del backbone (160 passi a 4 s: piccolo). Il packing scelto
  in D6b concatena campioni senza padding; con lunghezze temporali diverse va rimisurato.
- **Valutazione:** le finestre dell'harness non sono ancora fissate (passo 5). Se saranno più corte del contesto di pretraining, un modello che ha
  visto solo finestre da 4 s le vedrà fuori distribuzione.

**Opzioni:**
- (i) **fisso 4 s** (piano e D2): semplice; perde per intero i dataset con tratti più corti (CapgMyo di sicuro);
- (ii) **variabile, fino a 4 s**: la finestra è lunga 4 s dove il tratto lo permette, più corta dove il tratto è più corto, **mai sotto una
  lunghezza minima**; il packing (D6b) concatena campioni senza padding, ma è stato misurato solo a contesto fisso; le scale di masking si tagliano sulla finestra (una maschera
  lunga non supera metà finestra). Costo: il conto del ritmo e il proxy del passo 0 vanno ripetuti con lunghezze variabili; la contabilità di `D_t`
  non cambia (conta il tempo, non le finestre);
- (iii) **fisso 2 s**: perde meno dei tratti corti, ma la scala lunga del masking (fino a 2 s) non ci sta più intera.

**Proposta di AG: (ii), con massimo 4 s e minimo 1 s.** La lunghezza minima la fisso adesso, prima di vedere la tabella del costruttore, con un
argomento che non dipende dai dati: una finestra deve contenere almeno una maschera della scala lunga (500 ms) con altrettanto contesto visibile,
cioè 1 s = 40 patch = 5 finestre del tokenizer. *Da dichiarare:* che CapgMyo abbia prove da 1 s esatto rende la soglia comoda per CapgMyo; il motivo
della scelta resta il masking, e una soglia diversa andrebbe argomentata allo stesso modo. Se il costruttore mostra che a 4 s fisso si perde poco
ovunque tranne CapgMyo, l'alternativa onesta è (i) più l'esclusione dichiarata di CapgMyo dal pretraining: la scelta è tua.

## (3) Masking

**Già scritto in v10 §6.4** (non si ridecide): tre scale temporali (20-50 ms, 100-300 ms, 500 ms-2 s; per gli span molto corti «non vanno usati come
unica scala»); spaziale **~40% del budget** (canali singoli, gruppi contigui, porzioni geometriche della griglia); il masking di canale è due compiti
diversi (interpolazione sulle griglie, sinergia sui montaggi radi) **da loggare separatamente**; con l'ancora RVQ, **slab su tutti i canali, ≥ 200 ms,
allineati alla griglia del tokenizer**, e l'ancora RVQ attiva solo lì (v10 §6.3, mai sul masking di canale).

**Da decidere qui, proposte di AG (numeri di lavoro, non da v10):**

| Componente | In patch da 25 ms | Quota del budget di masking | Nota |
|---|---|---|---|
| temporale corta | 1-2 patch | 15% | mai da sola: sempre insieme ad almeno una scala più lunga nello stesso campione |
| temporale media | 4-12 patch (100-300 ms) | 25% | metà come slab RVQ (8 patch allineate), sui dataset con l'ancora accesa |
| temporale lunga | 20-80 patch (0,5-2 s), al più metà finestra | 20% | metà come slab RVQ (multipli di 8 patch allineati) |
| spaziale | canali singoli, archi di anello, rettangoli di griglia, gruppi anatomici (classe A) | 40% | v10 §6.4 |

- **Allineamento degli slab.** La griglia del tokenizer è ancorata all'inizio di ogni prova (`tokenizer_checks/sessions.py`: «la griglia e'
  ancorata all'inizio di ogni prova»). Proposta: **l'inizio di ogni finestra cade su un multiplo di 200 ms dall'inizio della prova**, così gli slab
  si allineano contando patch dentro la finestra. Si perde al più 175 ms di libertà nell'inizio della finestra: irrilevante.
- **Dataset senza ancora RVQ** (putEMG, CSL-hdemg, DB5, e DB10 se non entra): la quota degli slab torna al masking temporale normale.
- **Montaggi con pochi canali** (classe A, 8-12 canali): il masking spaziale non supera metà dei canali validi, così resta qualcosa da cui predire.
  *Interpretazione di AG:* con 8 canali mascherarne 4 è già il compito difficile di v10 §6.4 (sinergia), e oltre resterebbe troppo poco.
- **Frazione mascherata per campione.** v10 non dà un numero. Proposta: un valore di lavoro nel codice, e **la regola per sceglierlo congelata in
  `decisioni.md` prima del sanity JEPA** del passo 6 (piano, passo 6, «Chiuso quando»: il sanity JEPA su un solo dataset omogeneo, v10 §10.1,
  «non collassa»).
  Non propongo un numero dalla letteratura senza averlo verificato.

## Cosa misurare prima della firma, e quanto costa

1. **Tabella del costruttore** (job 59180449, in corso; CPU, 0 GPU-ora): tempo perso per dataset con finestre da 1, 2, 4 e 8 s. Decide fra (i) e
   (ii) del contesto.
2. **Se si firma il contesto variabile:** ripetere il conto del ritmo e il proxy del passo 0 con lunghezze variabili (dataloader su CPU; proxy su
   GPU in `boost_qos_dbg`, stima < 1 GPU-ora, da confermare col costo prima del lancio). Budget del passo 6: ≤ 100 GPU-ora (piano), usato 0.
3. **Se la patch cambia da 25 ms:** rifare il gate D8 (CPU, 0 GPU-ora) è una tua decisione.
4. Nessuna misura per il masking prima della firma: lo schema entra nel codice, e il sanity JEPA del passo 6 lo mette alla prova.

## Cosa si firma

- [ ] (1) patch 25 ms
- [ ] (2) contesto variabile, massimo 4 s, minimo 1 s, finestre allineate a 200 ms dall'inizio della prova — oppure fisso 4 s con CapgMyo escluso
      e dichiarato
- [ ] (3) masking: scale e quote della tabella (60% temporale con tre scale, 40% spaziale), metà del temporale medio-lungo come slab RVQ allineati
      sui dataset con l'ancora accesa, masking spaziale al più metà dei canali validi, frazione mascherata con regola congelata prima del sanity JEPA
