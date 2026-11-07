# Protocollo A: fattibilità implementativa EPN, 07/11/2026

Sola ricognizione locale; nessun codice del repository modificato, nessun job o accesso remoto. Disegno e costi restano una proposta da revisionare. Componenti esaminati: `finetuning.py`, controller/aggregatore EPN15, lettore di classificazione lungo e relativi piani/ricevute. Conteggi controllati anche leggendo le quattro OOF scratch15 locali e verificandone gli SHA contro le ricevute finali; nessuna nuova predizione. Dettaglio numerico in `results/infra/epn_label_efficiency_20261107/population_counts.json`.

## Popolazione concreta e sottoinsiemi annidati

EPN sviluppo: **314 soggetti, 47.100 finestre, sei classi, 25 finestre per soggetto e classe**. Gli identificatori includono la cartella (`trainingJSON/user…` o `testingJSON/user…`): non rimuovere questo prefisso. I 314 escludono l'unione dei test ufficiali dei tre split P2, come già firmato; mantenere esattamente `CP.p2_internal_subjects` e i quattro `CP.fold_roles` storici.

| Fold | Train utenti / finestre | Validation utenti / finestre | Held utenti / finestre |
|---|---:|---:|---:|
| 0,1 | 211 / 31.650 | 24 / 3.600 | 79 / 11.850 |
| 2,3 | 212 / 31.800 | 24 / 3.600 | 78 / 11.700 |

Un ordinamento deterministico dei **soli soggetti train originali**, mediante hash come specificato nel protocollo, poi prefissi di lunghezza `ceil(frazione × n_train)`, produce:

| Frazione nominale | Utenti train | Finestre train | Finestre train + validation | Aggiornamenti con 15 epoche, batch8 |
|---|---:|---:|---:|---:|
| 5% | 11 | 1.650 | 5.250 | 3.105 |
| 10% | 22 | 3.300 | 6.900 | 6.195 |
| 25% | 53 | 7.950 | 11.550 | 14.910 |
| 50% | 106 | 15.900 | 19.500 | 29.820 |
| 100% | 211/212 | 31.650/31.800 | 35.250/35.400 | 59.355/59.625 |

La stessa selezione va condivisa fra FT e scratch, i due LR e il controllo congelato. Conservare l'ordine originale delle righe dopo la selezione; la permutazione serve a scegliere gli utenti, non a riordinare i dati. Manifest: popolazione originale, train selezionato, train escluso, val/held invariati, righe originali, frazione realizzata, seme di selezione distinto dal seme del training, hash della selezione. Fissare prima quante permutazioni si vogliono: una sola non stima la variabilità della scelta dei pochi soggetti.

**Limite della domanda scientifica:** con validation fissa di 24 utenti, il 5% usa più utenti etichettati per il tuning che per il train. Le etichette train+val sono circa il 14,9% del pieno, non il 5%. È una curva rispetto ai soggetti usati per aggiornare i pesi, condizionata a una validation esterna fissa; non chiamarla 5% delle etichette totali. Le 150 finestre di ogni soggetto sono tutte incluse; non è una prova di pochi esempi per classe né di adattamento a un nuovo singolo utente.

## Preprocessing e assenza di uso dei soggetti esclusi

- `PreparedWindows` applica la mediana dei MAD sulle prime **10 finestre non etichettate di ciascun soggetto**, senza statistiche fra soggetti. Conservare questa calibrazione anche per val/held: è il protocollo storico dichiarato, non uso supervisionato degli held. Se si mantengono tutte le finestre dei soggetti selezionati, le prime10 e l'ordine restano identici. I due crop di4s della finestra5s condividono la stessa etichetta e non contano come due esempi indipendenti.
- `F.role_indices` richiede che train/val/held coprano tutti i soggetti passati. Non basta accorciare train lasciando l'intero array314: oggi viene rifiutato. Soluzione pulita: dati attivi = train selezionato + val + held, in ordine originale, con mapping esplicito alle `rows` originali e `unused_train` nel manifest. Il 100% deve ricostruire esattamente gli array storici.
- `encode_labels` deve ricevere soltanto gli indici train selezionati per vocabolario e pesi. EPN ha tutte le sei classi per ogni soggetto, quindi i pesi risultano1; verificarlo, non usare i conteggi del train pieno come scorciatoia. Nessuna esclusione silenziosa di esempi val/held per classi mancanti.
- Il frozen logistico deve rifare `StandardScaler.fit` esclusivamente sulle righe del train selezionato. La normalizzazione delle feature non va imparata sul cache intero. Calibrare separatamente ogni soggetto non è la stessa cosa che stimare uno scaler globale sul held.
- I dati dei soggetti train esclusi possono essere letti per verificare il fingerprint della sorgente originale, ma non devono entrare in scaler, pesi, optimizer o calibrazioni di altri soggetti. Mettere alla prova questa distinzione con un test che perturbi righe escluse e lasci immutati input/gradienti delle righe attive.

## Dose di ottimizzazione: scelta da fissare nel protocollo

**15 epoche per frazione** riusa più direttamente il trainer: numero di aggiornamenti nella tabella, warmup100 in tutti questi casi e coseno sull'orizzonte più corto. FT e scratch sono appaiati a ogni frazione, ma frazioni diverse hanno calcolo diverso; un calo può includere sottotraining. Non dedurre un confronto a calcolo costante.

**Aggiornamenti fissi per fold**: `U=59.355` per fold0/1 e `U=59.625` per fold2/3, con warmup100 e stesso coseno, è concretamente possibile. Però `fit_candidate(max_updates=U)` oggi applica `min(epoche×ceil(ntrain/batch), U)`: è soltanto un limite superiore, non prolunga il train piccolo. Serve un percorso nuovo che cicli sul subset fino aU e registri esposizioni/epoche reali.

In quel percorso, fissare **15 letture della validation agli stessi passi**: multipli3957 o3975 per fold. Validare ogni epoca reale darebbe circa287 opportunità di selezione al5% contro15 al100%, un vantaggio accidentale. Lasciare batch8 e l'ultimo batch parziale secondo la ricetta storica; registrare anche esempi realmente elaborati, perché uguali aggiornamenti non equivalgono sempre a identico numero di esempi. Il 100% deve riprodurre il ciclo storico, incluso l'ordine delle operazioni e il tie-break prima epoca.

Due candidati attuali: FT encoder `{1e−5,1e−4}` e testa `{1e−3,1e−2}`; scratch encoder `{1e−4,1e−3}` e testa `{1e−3,1e−2}`. Sono coppie legate dai moltiplicatori100 e10, non due griglie indipendenti. AdamW, weight_decay0,01, clipping1, batch8, seed0, BF16 e selezione di LR/checkpoint sulla sola validation sono riusabili. La divergenza numerica resta un esito persistito; OOM/guasti non diventano divergenze e non attivano fallback basati sul held.

## Riuso concreto

**Scratch100 EPN15** è riusabile senza fit o nuovo held se: stessa architettura/config modello, stessa inizializzazione seed0, split/ordine/vocabolario/pesi/calibrazione identici, due LR e regola di selezione identici, batch/warmup/coseno/U/15 occasioni validation identici. `build_classifier(...,'scratch',...)` usa la config modello e ignora i pesi del teacher: cambiare il checkpoint nominato da19k a49k non cambia matematicamente lo scratch, purché la config modello e il codice coincidano. Serve un bridge dichiarato verso le **otto ricevute originali** e le quattro OOF selezionate; non rinominare source/recipe/checkpoint negli artefatti storici. Se cambiano LR, seme, dose, selezione o preprocessamento, niente riuso automatico.

Con una selezione annidata, quattro fold, due LR e FT49k+scratch: 40 candidati FT, 40 scratch, dei quali8 potenzialmente già disponibili a100%; **72 fit nuovi**, oppure80 se non si dimostra l'equivalenza. Sono conteggi, non un preventivo. Una replica del sottoinsieme non rende indipendente la stessa OOF100 riusata.

**Teacher49k persistente:**
`/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/consolidation_20261104/continuation/checkpoints/checkpoint_postdecay.pt`
SHA256 `bfa6141fc2ed6505d2f582a46f0c04f36fd35f867e8cff2b11555b241947a490`.

**Frozen logistica 49k senza nuova estrazione GPU:** la cache esiste nella consegna pubblicata ed è nella copia WORK verificata dai manifest (nessuna nuova lettura del payload remoto):
`/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/consolidation_long_20261105/49046/p2/epn612.npz`
SHA256 `f7339e8950b27bb649b7b09dcc4f76b154f99891157c446b00aa5d21931fe70b`, 294.936.151byte. Pin in `results/infra/consolidation_long_20261105/execution/manifests/cache_manifest.json`, voce `classification/49046/epn612`; ricevuta `LONG-G49046-p2-20261105.json`. Usare **x_pool768**, la vista corrispondente al pooling dell'encoder fine-tuned; non cambiare su pre/post dopo i risultati.

`M.load_cache` è riusabile per hash e identità completa. `M.fit_fold` invece verifica che tutti i fold partizionino tutto l'universo: non accetta train ridotti con utenti ignorati. Serve un adapter subset dedicato che conservi scaler/train-only, logistica L2/lbfgs, class_weight balanced, griglia C storica a6punti, max_iter12000, tolleranza1e−4, refit train-only e selezione sulla stessa validation. Nessun fit del FM per questo controllo. Il punto100% frozen49k è potenzialmente riusabile dalle OOF del lungo se la ricetta resta esattamente quella; i16 fold/subset inferiori sono CPU nuovi. La logistica è un controllo distinto, non una testa a pari ottimizzatore con FT.

## Componenti riusabili e modifiche necessarie

Riusabili: loader EPN, esclusione dei test ufficiali, ruoli originali, `PreparedWindows`, modello/testa e optimizer, calcolo predizioni, bootstrap appaiato per soggetto, audit delle OOF. **Non modificare retroattivamente i controller congelati EPN15 o del lungo.**

Nuovo controller/protocollo: supporti annidati e righe originali; caricamento validato del teacher49k (EPN15 impone oggi `step==19618` e config storica); eventuale ciclo fixed-updates con validation fissata; nuova provenance contenente frazione, seed di selezione, manifest, inizializzazione e budget; bridge scratch100/frozen100; risorse misurate e output nuovi. Nuovo aggregatore deve verificare supporti held identici per frazione/regime, includere tutti i punti fissati e la dipendenza del riuso100, applicare la lettura concordata senza scegliere la frazione migliore post hoc. Il bootstrap degli utenti held non copre la variabilità dei subset train né dei semi di addestramento.

Controlli essenziali prima dei job: nesting e copertura; train esclusi mai usati per statistiche/gradiente; classi presenti senza drop held; mapping delle OOF all'ordine originale; parità100 con la ricetta storica; numero identico di opportunità di selezione FT/scratch; teacher49k effettivo per FT; cache congelato corrispondente; gate benchmark con costo validation/preprocessing anche alle piccole frazioni. Costi e scelta dose definitiva da fissare prima della review e dell'implementazione operativa.
