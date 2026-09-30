# Formato dei dati processati (contratto degli ingest)

Scritto il 30/09/2026 leggendo il codice degli ingest, non i dati sul cluster. Il validatore che lo applica e' `wearusfm.data.processed.validate_session`; il controllo su un intero albero e' `scripts/check_processed_layout.py` (sola lettura, va lanciato su Leonardo: i dati non sono sul Mac). **Finche' non gira sui dati veri, questo documento descrive l'intenzione, non uno stato verificato.**

## Sessione

Una sessione e' una cartella con:

| file | obbligatorio | contenuto |
|---|---|---|
| `data_int16.npy` | si | int16, `(T, C)` oppure `(n_prove, T, C)`; C = colonne = canali nell'ordine del montaggio |
| `metadata.json` | si | sidecar, vedi sotto |
| `labels.npz` | no | etichette per campione, `(T,)` ciascuna, solo per layout 2D; `-1` = campione senza etichetta |

Percorso: `<radice>/<soggetto>/<sessione...>/`. Il soggetto e' il primo livello; la sessione e' il resto (CapgMyo non ha livello di sessione: la cartella del soggetto e' la sessione). Nomi effettivi per dataset: `kaifosh` `u<UUU>/dataset<DDD>`; `ninapro_db2/3/4/5/7` `s<NN>/session1`; `ninapro_db6` `s<NN>/D<d>_T<t>`; `hyser` `s<NN>/session<k>_<chiave>`; `emg2qwerty`, `emg2pose` `u<utente>/<sessione o file>`; CapgMyo `s<NN>`; GRABMyo `p<NN>/session<k>`; putEMG `p<NN>/<traiettoria>_<timestamp>`; CSL-hdemg `s<NN>/session<k>`; Camargo `ab<NN>/<data>`.

## Sidecar `metadata.json`

Chiavi obbligatorie: `montage` (schema in `src/wearusfm/metadata/schema.py`, con `groups[].channels[].qc_valid` in ordine di colonna), `native_fs_hz` (> 0), `shape` (uguale a quella dell'array).

Chiavi condizionali:
- `int16_scale`: assente = int16 nativo senza scala (GRABMyo); numero = uguale per tutti i canali; lista di C numeri = una scala per canale (Hyser, i cui file hanno guadagni diversi). Ricostruzione: `dato = codice / scala`. Sempre finita e > 0.
- `trials`: lista di prove. Con array 3D: una voce per prova (`len == shape[0]`). Con array 2D concatenato: ogni voce ha `offset` e `n_samples`, contigue dall'offset 0, e la somma dei `n_samples` e' T.
- `n_channels_discarded_by_qc`: se presente deve uguagliare il numero di `qc_valid` falsi.

Tutto il resto (`label_fields`, `time_axis`, `laterality`, `attrs`, ...) e' informativo per il singolo dataset: il validatore non lo guarda.

## Regole che il formato incorpora

1. **Nessun dato EMG si tronca mai** per far quadrare le etichette: se le etichette sono piu' corte di T si riempiono con `-1` (tolleranza `max(5, 1%)` campioni, oltre e' errore).
2. **QC dei canali**: un canale e' invalido se piatto o saturo (>1% dei campioni entro lo 0,1% del proprio massimo assoluto). I canali invalidi restano nell'array (le colonne non si tolgono) e si marcano in `qc_valid`. La soglia di "piatto" NON e' uniforme, e questo e' un dato di fatto del codice, non una scelta: **assoluta** (in unita' del dato) in CapgMyo, Camargo, CSL-hdemg, NinaPro DB5 e Kaifosh (1e-6), putEMG (1e-3), GRABMyo (1.0, su int16 grezzi); **relativa** (`1e-3 x` mediana delle std dei canali) in NinaPro DB2/3/4/7, DB6, Hyser, emg2qwerty, emg2pose. Con la soglia assoluta, "0 canali scartati" (Camargo, Kaifosh, DB5 non ancora ingerito) non distingue "tutti buoni" da "soglia troppo permissiva" quando i valori sono **grandi** rispetto alla soglia (Kaifosh: massimo assoluto 362-7447); quando sono piccoli (volt) la soglia assoluta e' invece troppo severa. Non verificato caso per caso. **Limite di entrambe le regole:** trovano solo canali (quasi) piatti, non elettrodi staccati che raccolgono rumore. (Correzione del 30/09/2026: prima questo paragrafo diceva il contrario sulle unita'.)
3. **Unita'**: l'ampiezza e' relativa; la normalizzazione (una scala per registrazione) avviene nella vista canonica, non nell'ingest.
4. **Frequenza nativa**: si conserva, non si ricampiona nell'ingest.

## Cosa il validatore NON controlla

Il contenuto dei dati (saturazione, canali morti reali, buchi temporali: per Kaifosh sta in `time_axis`), la coerenza fra sessioni dello stesso soggetto, la correttezza delle etichette, il rispetto degli schemi dei metadati oltre alle chiavi sopra. Per lo schema del montaggio vale la validazione fatta all'ingest (`validate_montage_dict`).
