# Foglio di firma — D6a, montaggi virtuali dalle griglie HD (bozza di AG del 05/10/2026; FIRMATO da Simone il 05/10/2026)

**Perche'.** Decisione 15 del 04/10 (Simone): «montaggi virtuali dalle griglie HD al volo nel dataloader, se l'attesa sui dati resta <= 2%; da
implementare prima della finestra 1. Le quote sulla topologia presentata si decidono con i numeri». v10 §2.7: il sottocampionamento a montaggi
virtuali (<= 32 canali) e' **augmentation stocastica, non riduzione obbligatoria**; le griglie intere restano. I parametri non sono scritti da
nessuna parte: sono qui sotto, marcati come proposta.

**Implementato e spento** (`src/wearusfm/data/virtual_montage.py`, opzione `LoaderConfig.virtual`, None di default; test in
`tests/cpu/test_virtual_montage.py` e `test_pretraining_loader.py`). Con la firma si accende nelle regole della finestra 1. **Acceso** in
`training.run.with_window1_rules` dopo la firma (05/10); il sanity resta senza.

| Punto | Proposta di AG | Alternativa |
|---|---|---|
| 1. A quali campioni | classe C del manifest (CapgMyo, CSL-hdemg, Hyser: il 5% dei campioni, di cui Hyser 3,75%) | — |
| 2. Quanti | **p = 0,5**: meta' dei campioni di griglia presentati come montaggio virtuale, meta' interi | p piu' alta (0,75) se si vuole piu' peso sugli ingressi «da deployment»; le griglie intere restano comunque |
| 3. Forma | una **sottogriglia** di una sola griglia (Hyser: una delle 4): finestra di **4x8, 8x4 o 4x4 elettrodi**, presi **ogni 1 o 2** passi della griglia, posizione a caso; solo le forme che ci stanno (CSL-hdemg ha 7 righe: niente 8x4). <= 32 canali | anche montaggi da due griglie di Hyser insieme (flessori + estensori) |
| 4. Tipo | **meta' monopolari, meta' bipolari.** Bipolare: «compagno meno elettrodo», col compagno a 1 o 2 passi lungo le righe o lungo le colonne (a caso); canale nel punto medio, valido se lo sono tutti e due gli elettrodi, banda = intersezione | solo monopolari |
| 5. Topologia presentata | **resta la griglia** (classe C): le quote non cambiano, presentata = origine. Risponde alla domanda aperta 3 del 03/10 | presentarli come sparsi (classe A): **sconsigliato**, gli elettrodi di griglia non hanno identita' di muscolo e da sparsi sarebbero indistinguibili per il modello (v10 §3.5) |
| 6. Scala | monopolare: quella della sessione. Bipolare: **la scala della sua derivazione** (mediana dei MAD su tutte le coppie di quell'asse e passo, stessi tratti), una per sessione, asse e passo | la scala monopolare: i bipolari avrebbero un'ampiezza diversa da quella dei dati veri, normalizzati sulla loro (con elettrodi vicini correlati, minore; non misurato) |
| 7. Posizioni | **coordinate della griglia d'origine**: una sottogriglia ogni 2 passi ha i vicini a 2 passi, come sono; posizione nel sistema del sensore uguale a quella dell'elettrodo nella griglia intera | rinumerare la sottogriglia (perde la spaziatura vera) |
| 8. Conti | `D_c` invariato (si conta prima delle augmentation, v10 §10.3). Nelle metriche, per passo: finestre per classe e quante virtuali | — |

**Da dichiarare (non si risolvono qui):**
- il modello non ha un ingresso «monopolare / bipolare»: lo deve capire dal segnale, come per i dati veri (`electrode_type` non entra nel
  modello);
- la polarita' bipolare e' canonica **nel sistema della griglia** (righe, colonne), non in direzione anatomica: per le griglie con l'orientamento
  rispetto all'arto non dichiarato, «+riga» non e' la stessa direzione fra dataset (Hyser e' marcato «polarity_unspecified» dall'adattatore);
- che un montaggio virtuale valga per il modello come un montaggio sparso vero non e' misurato (era gia' scritto nella proposta D9); il pilot
  letto per topologia serve anche a questo.

**Costo** (misura sul Mac, sessione sintetica tipo Hyser, 256 canali a 2048 Hz, filtro e ancora multi-scala accesi, un processo): griglia intera
**3,4 finestre/s**; montaggio virtuale **36,6 finestre/s** (il montaggio si applica prima del filtro, che e' lineare: si filtrano solo i canali che
servono); con p = 0,5 **5,7 finestre/s**. Le finestre di griglia intera restano le piu' care (target delle ancore su 256 canali, ~200 ms l'una), ma
sono il ~2,5% dei campioni. La regola di D6a (attesa sui dati <= 2% del passo) si verifica su Leonardo col modello vero, nel primo collaudo della
finestra 1. **Scale bipolari:** se non sono precalcolate, ogni processo del dataloader le stima al primo uso (sul Mac ~3 s per sessione per le
4 scale, sessione sintetica tipo Hyser). Proposta: precalcolarle con `scripts/measure_loader.py --bipolar-scales --scales-from <scale attuali> --skip-rate` (226 sessioni di
classe C x 4 = 904 scale; **job CPU su `lrd_all_serial`, 0 GPU-ora**), da lanciare con la tua conferma.

**Tua decisione:** [x] firmo · [ ] firmo con correzione · [ ] non firmo — **FIRMATO da Simone il 05/10/2026** («accetto tutti e 4, firma compresa»)

**Correzioni di Simone del 05/10/2026** («si a 1 e 2», dopo i fatti 25-27): (1) **niente bipolari virtuali su CSL-hdemg**, gia' bipolare
(README): solo sottogriglie; (2) **CapgMyo fuori dai montaggi virtuali** finche' la geometria della griglia (16 x 8, fatti 14 e 26) non e'
corretta. Hyser (monopolare, paper TNSRE 2021) resta com'era. Codice: `VirtualSpec.exclude_datasets`, `no_bipolar_datasets`.
