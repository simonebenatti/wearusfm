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
| **R0** riferimento | niente: **C1 seme 0 al LR 1e-4** esiste gia' (`W1CAL-C1-30M-lr0p0001-s0`, 1 epoca = 24.523 passi, S P1 18,21) | fatto; manca la **S P2** di quel checkpoint (estrazione ~1,5 GPU-ora + sonda CPU) |
| **R1** target a | `JEPAConfig.target = "a"`: target = uscita per canale dell'encoder locale del teacher, non il decoder a query | esiste nel codice; manca il flag `--jepa-target` in `train_jepa.py` (oggi fisso "b" in `sanity_config`) e il test che "b" resti identico |
| **R2** target per regione | il target JEPA di ogni regione nascosta (tubo, slab o maschera spaziale) e' la **media del teacher sulla regione**; lo studente predice la media delle sue uscite sulla stessa regione; perdita per regione, media per finestra come oggi; ancore invariate per token | nuovo: `generate_mask` ritorna anche `region` (C, P) (id della maschera estratta), `PretrainBatch.region`, `_jepa_compute(region_target=True)`; test d'uguaglianza bit per bit con `region_target=False`; DDP: numeratori/denominatori per regione nel contratto delle loss |

Seconda ondata (dopo l'ingest degli angoli, da preventivare a parte): **R3** ancore + testa di posa su emg2pose senza JEPA; **R4** R2 + testa di posa.
Richiede: rilettura del tar di emg2pose per i `joint_angles` (462 GB, job CPU lungo; oggi l'ingest li salta), `labels.npz` a 20 colonne float
a 50 Hz, loader con i target per campione e maschera di validita', testa di regressione sulle uscite del decoder (o del backbone) attiva solo
sui batch di emg2pose, termine nel contratto delle loss, log separato. E' un cambio di principio rispetto a v10 §6.1: lo decide Simone prima
che si scriva il codice.

## Disegno (proposta)

- **Taglia e durata:** 30M (preset `sanity`), 1 epoca del manifest v1.1 al 100% (24.523 passi, batch 32, WSD con warmup 1.962 e decay dal
  passo 19.618), LR 1e-4, seme 0: identico a C1 s0, cosi' R0 non si rifa'. Stesso manifest, stesso sampler, stesso loader (vedi «fuga»).
- **Valutazione:** S interna P1 e P2 del teacher post-decay, con **due readout**: il medio (`pool`, il riferimento storico) e quello per canale
  che il protocollo del readout designa (local o dec a k 768), estratti e sondati con `channel_readout.py`; baseline classica accanto.
- **Semi:** seme 0 per tutti; **seme 1 solo per il braccio che passa** la regola (un run in piu').
- **Regola di lettura (da congelare prima del lancio):** un braccio «passa» se migliora S della P1 **e** S della P2 di almeno +1 punto rispetto
  a R0 con lo stesso readout (regola dei bracci, bootstrap congiunto dei soggetti; con un solo seme il «delta > 0 nei due semi» non c'e': si
  dichiara, e il seme 1 serve a confermare); si legge col readout designato; il `pool` resta descrittivo. Nessun braccio passa -> si registra
  e non si cambia il default; si va alla seconda ondata solo se Simone lo decide.
- **Non si decide qui:** K, taglie, LR della campagna, default della finestra 1.

## Prerequisiti e punti aperti

1. **Fuga attraverso il filtro** (`results/rami_pilota/fuga_preprocessing_20261023.json`): il filtro passa-banda del loader agisce prima della
   maschera, quindi i token visibili entro ~100 ms da una regione nascosta portano una traccia del contenuto nascosto (ringing del filtro;
   log-potenze di contesto cambiate fino a ~3 nat al 99° percentile con contenuto nascosto estremo). Vale per A e per tutti i bracci allo
   stesso modo. Due strade: (a) pilot col loader attuale, confrontabile con C1 s0, e «mascherare prima del filtro» come ablation separata dopo;
   (b) loader nuovo per tutti i bracci, ma allora R0 va rifatto (+1 run). *Proposta: (a).*
2. **S P2 di C1 s0:** da estrarre e sondare (non c'e': la calibrazione usava solo la P1). Stessa catena del readout per canale.
3. **Readout designato:** si aspetta l'esito del readout per canale; se non e' concluso si usa il `pool`.
4. **Review del `critico`** su protocollo e codice prima del lancio; soglie in `decisioni.md` prima dei job.

## Costi (stima dalle misure di C1: 0,707 s/passo a 4 GPU, 4,77 h per epoca)

| Voce | Run | GPU-ora attese / tetto |
|---|---|---|
| R1, R2 (prima ondata) | 2 | ~40 / 52 |
| Seme 1 del braccio che passa | 0-1 | ~20 / 26 |
| S P2 di R0 + valutazione dei bracci (estrazioni GPU) | 3 x 2 | ~4 / 8 |
| Sonde CPU (pool, readout designato; P1 e P2) | ~12 job seriali | 0 GPU-ora, ~60 core-ore |
| **Prima ondata** | | **~64 / 86 GPU-ora** |

Voce di budget: famiglia «Ablation a 30M» della finestra 1 (~2.100 GPU-ora stimate in `docs/finestra1_fattibilita.md`), oppure una voce nuova
decisa da Simone. Calendario: i due run in un'ondata (~5 h su due nodi), valutazione ~1 giorno sulla seriale.

## Cosa NON fa questo pilot

Non tocca il tokenizer (front-end, identita' di canale) ne' K: se R1 e R2 non passano e la seconda ondata non e' voluta, la mossa successiva e'
sul tokenizer (token da 100-200 ms sopra le patch, lettura non lineare per canale) con l'analisi sui pesi come primo passo a costo zero.
