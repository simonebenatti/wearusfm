# Foglio di firma — replica dei numeri di NeuroRVQ (passo 5; bozza di AG del 04/10/2026; FIRMATO da Simone il 04/10/2026 con due correzioni)

**Perche'.** Il passo 5 si chiude quando «si riproducono i numeri pubblicati di NeuroRVQ sulla loro pipeline, entro una tolleranza dichiarata»
(piano); la tolleranza si dichiara **prima** del lancio (Simone, 29/09). Il fact-checker (fatto 24, `fatti_da_verificare.md`, raccolto il 04/10) ha
letto il paper (arXiv 2510.13068v4, App. J.3 e Tab. 13) e il repo: **il protocollo e' dichiarato solo in parte e il codice EMG downstream non e'
pubblico**. Non si puo' rifare alla lettera: si fa una replica con le parti mancanti scelte e dichiarate. Le scelte sono qui sotto, marcate.

**Dataset della replica:** EPN-612 e UCI-EMG, i due mai visti dal nostro pretraining (decisione 10 del 04/10). DB5 (visto in pretraining) e
Discrete Gestures dopo.

| Punto | Dal paper (fatto 24) | Scelta di AG dove il paper tace |
|---|---|---|
| Modello | checkpoint pubblico `NeuroRVQ_EMG_foundation_model_v1.pt` (gia' su Leonardo) | Tab. 13 da' dimensione 40, lo yml 200: si usa il checkpoint pubblico com'e' |
| Frequenza | ricampionamento a 1000 Hz | `resample_poly` (EPN-612 200 -> 1000 Hz; UCI-EMG e' gia' a 1 kHz, fatto 15) |
| Filtro | passa-banda 20-90 Hz | **Correzione di Simone: 20-400 Hz**, come il loro codice d'esempio (Butterworth di ordine 3, fase zero, taglio alto min(400, fs/2) - 0,5 Hz: a 200 Hz, EPN-612, diventa 99,5 Hz). Scostamento dichiarato dal 20-90 Hz del paper |
| Normalizzazione | nessuna dichiarata per NeuroRVQ | nessuna |
| Finestre | EPN-612 5 s; UCI-EMG 1 s | EPN-612: i campioni da 5 s etichettati del dataset; UCI-EMG: finestre da 1 s **senza sovrapposizione** dentro un tratto con una sola etichetta, classi **1-6** (interpretazione di «6-class»: la 7, palmo esteso, non e' stata eseguita da tutti i soggetti e la 0 non e' marcata, secondo la descrizione dei file letta il 23/09, `bench/harness_smoke_uci_emg.py`; non e' un fatto firmato) |
| Canali | — | gli 8 canali sugli elettrodi c1..c8 del modello |
| Split | 7:1:2 solo per soggetto | soggetti ignoti: **3 split casuali** per soggetto (semi 0, 1, 2) |
| Testa e aggregazione | lineare; embedding concatenati sui canali e mediati sul tempo | come il paper (il codice pubblico appiattisce: conflitto dichiarato) |
| Fine-tuning | completo; Tab. 13 «Finetuning (EMG)»: batch 128, lr 1e-3, cosine fino a 1e-6, warmup 5 epoche (fattore 0,01 -> 1), weight decay 0,001, Adam (0,9, 0,95), 100 epoche, nessun layer decay | entropia incrociata pesata per classe come il loro modulo; bf16 |
| Modello valutato | non dichiarato | **l'epoca con la miglior accuratezza di validazione**; si riporta anche l'ultima |
| Metrica | Acc (e F1) | accuratezza; F1 macro come informazione |

**Tolleranza, da firmare prima del lancio:** la replica riesce se la **media sui 3 split** dell'accuratezza di test cade entro **max(2 x il loro ±; 2
punti)** dal numero pubblicato:
- EPN-612: 94,65 ± 2,0, quindi **[92,65; 96,65]**;
- UCI-EMG: 89,43 ± 2,2, quindi **[87,23; 91,63]**.
Fuori tolleranza non si «aggiusta» il protocollo cercando i parametri: si riporta, si cercano errori di dati e di harness, e la differenza si dichiara.

**Costo:** il loro FM ha 6 M parametri. EPN-612 ha ~92.000 finestre da 5 s: 100 epoche a batch 128 sono ~72.000 passi per split, stimati ~1 ora di
GPU; UCI-EMG e' molto piu' piccolo. **3 split x 2 dataset: ~4-6 GPU-ora**, da ridichiarare dopo un collaudo su `boost_qos_dbg` (0,5 GPU-ora).
Budget del passo 5: <= 300 GPU-ora.

**Il nostro FM sugli stessi dati** (metrica P2 di D12): stesse finestre e stessi split, preprocessing **suo** (quello del pretraining,
`harness/fm_features.py`), encoder congelato e sonda lineare (`harness/probe.py`); il fine-tuning completo e' il secondo regime, dopo.

**Tua decisione:** [ ] firmo · [x] firmo con correzione · [ ] non firmo — **FIRMATO da Simone il 04/10/2026** («ok firmo. pero [...] 2) teniamo 20-400Hz di banda. Per il resto approvo.»). Correzioni: (1) banda 20-400 Hz invece di 20-90 Hz; (2) il codice di NeuroRVQ che serve alla replica si tiene nel repo («teniamo tutto dato che come abbiamo gia deciso non ci interessano le licenze»; licenze, decisioni d'uso del 30/09), con l'attribuzione e la licenza originali.
