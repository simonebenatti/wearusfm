# Pilot degli obiettivi di pretraining — `pilot_obiettivi_v1` (BOZZA)

**Stato: BOZZA di Claude (23/10/2026), da discutere con Simone e da passare al `critico`; niente e' congelato, nessun codice nuovo, nessun job.**
Origine: `docs/diagnosi_pretraining_20261023.md` §3.2-3.3 (Simone: «poi lavorare sulle tue soluzioni per il tokenizer/embeddings»). Si legge
dopo l'esito del readout per canale (`docs/readout_per_canale_v1.md`): la scelta del readout con cui si misura il pilot dipende da quello.

## Domanda

Un obiettivo che chiede la struttura alla scala del movimento (regione nascosta, non patch da 25 ms), o che toglie il percorso query -> target
(opzione a), o che chiede esplicitamente le variabili motorie (posa), produce un encoder che una sonda lineare legge meglio di A, a pari
esposizione? Misura: S della validazione interna (P1 e P2) coi readout del protocollo del readout per canale.

## Bracci (prima ondata: solo codice gia' quasi pronto)

| Braccio | Cosa cambia rispetto ad A (JEPA target b + ancore 0,2) | Codice |
|---|---|---|
| **R0** riferimento | **A rifatto a 2 epoche** col ramo di decay a 1 (Simone, 23/10: «scelgo le 2 epoche per tutti i 4 run»): stesso seme 0 e LR 1e-4 di C1 s0, che resta il controllo di coerenza dell'endpoint a 1 epoca (S P1 18,21) | pronto (e' A); C1 s0 ha solo la S P1 |
| **R1** target a | `JEPAConfig.target = "a"`: target = uscita per canale dell'encoder locale del teacher, non il decoder a query | esiste nel codice; manca il flag `--jepa-target` in `train_jepa.py` (oggi fisso "b" in `sanity_config`) e il test che "b" resti identico |
| **R2** target per regione | il target JEPA di ogni regione nascosta (tubo, slab o maschera spaziale) e' la **media del teacher sulla regione**; lo studente predice la media delle sue uscite sulla stessa regione; perdita per regione, media per finestra come oggi; ancore invariate per token | nuovo: `generate_mask` ritorna anche `region` (C, P) (id della maschera estratta), `PretrainBatch.region`, `_jepa_compute(region_target=True)`; test d'uguaglianza bit per bit con `region_target=False`; DDP: numeratori/denominatori per regione nel contratto delle loss |

| **R5** finestra 150 ms, passo 25 ms (Simone, 23/10: «fare una finestra un po piu lunga con overlap, in modo che lo scorrimento sia 25ms. tipo 150ms con overlap di 125ms») | il kernel del front-end vive su 150 ms centrati sulla patch, la griglia dei token resta a 25 ms (sequenza, maschere, ancore e RVQ invariati); basi scalate per lo stesso contenuto in frequenza fisico (156 armoniche, 120 nodi, omega 36); **la perdita conta solo i token interamente nascosti** (nessuna patch visibile entro 3 patch per lato: JEPA, ancore e RVQ); le maschere corte da 1-2 patch non producono token interi, quindi la quota `share_short` (15%) passa a 0 e le altre si rinormalizzano (0,294 / 0,235 / 0,471): **scostamento dichiarato da D10**, da firmare per questo braccio | **fatto il 23/10** (`FMConfig.window_ms`, `frontend.window_kwargs`, `jepa.core_hidden`, `--window-ms`, `gate_d8.py --window-ms`; test: default bit per bit identico, griglia invariata, coerenza fra fs, token interi); manca il **gate D8 riaperto** per il banco da 150 ms (job CPU, `scripts/gate_d8.py --window-ms 150`, stesse soglie firmate: errore relativo <= 5% per famiglia, sonda <= 55%) prima del run |

R5 e R2 provano la stessa ipotesi (un'unita' meno rumorosa, alla scala dell'inviluppo) in due posti diversi: R2 nella loss, a codice quasi nullo e
col tokenizer di sempre; R5 nel tokenizer, che migliora anche l'ingresso ma cambia identita' al front-end (gate D8, training da zero). Ordine
proposto: R2 prima o insieme a R5; se R2 non aiuta, R5 difficilmente aiutera'.

Seconda ondata (dopo l'ingest degli angoli, da preventivare a parte): **R3** ancore + testa di posa su emg2pose senza JEPA; **R4** R2 + testa di posa.
Richiede: rilettura del tar di emg2pose per i `joint_angles` (462 GB, job CPU lungo; oggi l'ingest li salta), `labels.npz` a 20 colonne float
a 50 Hz, loader con i target per campione e maschera di validita', testa di regressione sulle uscite del decoder (o del backbone) attiva solo
sui batch di emg2pose, termine nel contratto delle loss, log separato. E' un cambio di principio rispetto a v10 §6.1: lo decide Simone prima
che si scriva il codice.

## Disegno (proposta)

- **Taglia e durata (Simone, 23/10/2026: «scelgo le 2 epoche per tutti i 4 run»):** 30M (preset `sanity`), **2 epoche** del manifest v1.1 al 100%
  (49.046 passi, batch 32, WSD D14: warmup 1.962, fase stabile, decay lineare sul 20% finale, dal passo 39.237) **piu' un ramo di decay dal
  checkpoint a 1 epoca** (passo 24.523, nella fase stabile; decay su 4.905 passi = 20% di un'epoca): due endpoint validi per run, a 1 e a 2
  epoche, ~2,2 epoche di calcolo. LR 1e-4, seme 0, stesso manifest, stesso sampler, stesso loader (vedi «fuga»). L'endpoint a 1 epoca di R0
  deve coincidere con C1 s0 a meno del ramo di decay: controllo di coerenza, descrittivo.
- **Perche' due epoche:** A e' piatto da 1 a 4 epoche (18,21 -> 18,34), ma i bracci nuovi possono imparare piu' lentamente (R2 ha un target
  per regione invece che per token; la posa impara da un terzo dei batch): con due endpoint si legge la **pendenza** 1 -> 2 di ogni braccio
  accanto a quella di R0, non solo un punto.
- **Valutazione (Simone, 23/10: «valuti le loss dopo le 2 epoche o valuti anche altro?»):** le loss NON sono il criterio: fra obiettivi diversi
  non sono confrontabili (target a e b sono grandezze diverse, la media per regione cambia l'unita', R5 conta solo i token interi). A ciascuno
  dei due endpoint: (1) **S interna P1 e P2** del teacher post-decay, encoder congelato e sonda lineare, con **due readout**: il medio (`pool`,
  il riferimento storico) e quello per canale che il protocollo del readout designa (local o dec a k 768), estratti e sondati con
  `channel_readout.py`: e' il criterio; (2) la **pendenza 1 -> 2** di S accanto a quella di R0, descrittiva; (3) le **diagnostiche di non
  collasso** del trainer ogni 500 passi (rango effettivo di studente e teacher, rapporto di collasso da query, varianza fra campioni) con
  l'allarme firmato (38,4 / 0,05 / 3 controlli): un run fermato dall'allarme non passa; (4) le **loss come monitor**, non per decidere: JEPA
  per tipo di maschera e per topologia, ancore (RMS, forma spettrale, inviluppo, multiscala: target fisici uguali fra i bracci; per R5 sui
  soli token interi, con `core_hidden_fraction`), pannello fisso ogni 500 passi; (5) **riferimenti esterni** sulla stessa validazione: baseline
  classica e rete casuale. Per il braccio che passa: seme 1, poi i test ufficiali P1/P2 e le 4 epoche prima di toccare un default.
- **Semi:** seme 0 per tutti; **seme 1 solo per il braccio che passa** la regola (un run in piu', a 2 epoche).
- **Regola di lettura (da congelare prima del lancio):** un braccio «passa» se **all'endpoint a 2 epoche** migliora S della P1 **e** S della P2
  di almeno +1 punto rispetto a R0 con lo stesso readout (regola dei bracci, bootstrap congiunto dei soggetti; con un solo seme il «delta > 0
  nei due semi» non c'e': si dichiara, e il seme 1 serve a confermare); si legge col readout designato; il `pool` e l'endpoint a 1 epoca
  (pendenza) restano descrittivi. Nessun braccio passa -> si registra e non si cambia il default; si va alla seconda ondata solo se Simone lo
  decide.
- **Non si decide qui:** K, taglie, LR della campagna, default della finestra 1.

## Prerequisiti e punti aperti

1. **Fuga attraverso il filtro** (`results/rami_pilota/fuga_preprocessing_20261023.json`): il filtro passa-banda del loader agisce prima della
   maschera, quindi i token visibili entro ~100 ms da una regione nascosta portano una traccia del contenuto nascosto (ringing del filtro;
   log-potenze di contesto cambiate fino a ~3 nat al 99° percentile con contenuto nascosto estremo). Vale per A e per tutti i bracci allo
   stesso modo. Due strade: (a) pilot col loader attuale, confrontabile con C1 s0, e «mascherare prima del filtro» come ablation separata dopo;
   (b) loader nuovo per tutti i bracci, ma allora R0 va rifatto (+1 run). *Proposta: (a).*
2. **S P2 di C1 s0:** non serve piu' come riferimento (R0 si rifa' a 2 epoche); C1 s0 resta il controllo di coerenza sulla sola P1.
   **Rami di decay:** la catena WSD (`scripts/wsd_chain.py`, D14) prevede checkpoint da tenere (`--keep-checkpoints`) e rami di decay: il
   piano del pilot usa quelli, e il pianificatore va esteso ai 4 run con 2 endpoint (test di uguaglianza coi piani esistenti).
3. **Readout designato:** si aspetta l'esito del readout per canale; se non e' concluso si usa il `pool`.
4. **Review del `critico`** su protocollo e codice prima del lancio; soglie in `decisioni.md` prima dei job.

## Costi (stima dalle misure di C1: 0,707 s/passo a 4 GPU, 4,77 h per epoca; 2,2 epoche per run = ~10,5 h su un nodo, ~44 GPU-ora)

| Voce | Run | GPU-ora attese / tetto |
|---|---|---|
| R0 (A), R1, R2, R5 a 2 epoche col ramo di decay a 1 | 4 | ~176 / 230 |
| Gate D8 per il banco da 150 ms (R5), CPU | 1 | 0 GPU-ora, ~10 core-ore |
| Seme 1 del braccio che passa (2 epoche col ramo) | 0-1 | ~44 / 58 |
| Valutazione: estrazioni GPU di 8 endpoint (4 run x 2), P1 e P2 | 16 | ~8 / 16 |
| Sonde CPU (pool, readout designato; P1 e P2; 8 endpoint) | ~32 job seriali (10 alla volta in coda, 1 in esecuzione) | 0 GPU-ora, ~160 core-ore |
| **Prima ondata** | | **~184 / 246 GPU-ora** (+44 / 58 col seme 1) |

Voce di budget: famiglia «Ablation a 30M» della finestra 1 (~2.100 GPU-ora stimate in `docs/finestra1_fattibilita.md`), oppure una voce nuova
decisa da Simone. Calendario: i due run in un'ondata (~5 h su due nodi), valutazione ~1 giorno sulla seriale.

## Cosa NON fa questo pilot

Non tocca il tokenizer (front-end, identita' di canale) ne' K: se R1 e R2 non passano e la seconda ondata non e' voluta, la mossa successiva e'
sul tokenizer (token da 100-200 ms sopra le patch, lettura non lineare per canale) con l'analisi sui pesi come primo passo a costo zero.
