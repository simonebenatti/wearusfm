# Piano di diagnosi della distanza dal paper: TEMPONet su NinaPro DB8

Claude, 09/11/2026, su richiesta di Simone («Yes, prepare it», alla domanda se preparare un piano di diagnosi). **Proposta: nessun job lanciato.**
Soglie e cancelli diventano vincolanti solo dopo l'approvazione, con una voce in `docs/decisioni.md` committata prima del primo job.

## Punto di partenza

Run 63988277 (`docs/temponet_db8_replication.md`, sezione «Esito»): MAE primaria (fold, EMA, 3 semi) **8,795 gradi** contro **6,89**,
non replicata. Distanza sistematica: semi entro 0,12 gradi, dispersione fra soggetti come quella pubblicata (2,21 contro 2,08). Il modello
`full` (sessioni 1 e 2 insieme, seme 0) arriva a 7,778 ed e' migliore dei fold in 12 soggetti su 12. Validazione dei fold col seme 0 (MAE con
EMA sull'altra sessione, dati di training): **9,906**, ferma dall'epoca ~10; loss di training (con dropout) 9,3 alle ultime epoche, vicina
alla validazione: sottoadattamento, non overfitting.

## Ipotesi

| | Ipotesi | Indizi oggi | Come si verifica |
|---|---|---|---|
| H1 | I modelli valutati da [Z21] sul test erano allenati sulle sessioni 1 e 2 unite (la 2-fold CV serviva a scegliere taglia ed EMA) | `full` 7,78 contro 8,75, 12/12 soggetti | domanda agli autori; baseline classiche nei due protocolli (passo 1) |
| H2 | Pipeline o dati diversi (versione di DB8, matrice DoA, definizione di test o MAE) | nessuno diretto | baseline classiche contro 7,28 e 7,14 (passo 1) |
| H3 | Bersaglio normalizzato (z-score per DoA) o inizializzazione dell'uscita, non dichiarati | uscita a zero dopo un'epoca, sottoadattamento con Adam a 1e-4 | variante V1 in validazione (passo 2) |
| H4 | Dettagli d'architettura (padding simmetrico come Q-PPG invece che causale) | conteggio dei parametri esatto, quindi struttura giusta | variante V3 in validazione (passo 2) |
| H5 | Dose di training (passo delle finestre, schedule del learning rate) | da passo 32 a 8: −0,86 su S1; da 8 a 4: −0,07 | gia' quasi saturo su S1; solo domanda agli autori |
| H6 | Quantizzazione int8 con QAT | non puo' migliorare un modello fp32 | esclusa |

## Passo 0 — domanda agli autori (costo zero, consigliato per primo)

Simone e' coautore di [Z21]: le risposte di Zanghieri chiuderebbero H1, H3 e H5 senza GPU. Bozza da inviare (da Simone; io non invio nulla):

> Ciao Marcello, sto replicando la TEMPONet di regressione del vostro COINS 2021 su NinaPro DB8 e arrivo a circa 8,8 gradi invece di 6,89.
> Mi chiariresti alcuni dettagli? (1) Passo delle finestre di training (in test 16 ms, come da IWASI 2023). (2) Il 6,89 viene da modelli allenati
> su un fold della 2-fold CV (una sessione o meta' delle ripetizioni?) o sulle sessioni 1+2 unite? Come erano fatti i fold? (3) Bersaglio in
> gradi o normalizzato? EMG in volt grezzo o scalato? (4) Learning rate costante 1e-4 o con schedule? Weight decay? (5) Alpha dell'EMA scelto su
> quali predizioni? (6) MAE su tutta la sessione 3, riposi compresi? (7) Nell'architettura: le convoluzioni dilatate hanno gli stessi canali del
> blocco (16/32/64)? FC I ha 64 unita' su 4x64 ingressi? Padding causale o simmetrico? (8) Il codice e' disponibile? Grazie!

Con risposte complete, il passo 2 si riduce a un solo run mirato e il passo 1 resta come controllo della pipeline.

## Passo 1 — controllo della pipeline con le baseline classiche di [Z21] (solo CPU)

Stessa catena della TCN (cache, DoA, split, EMA, metriche); modelli di Tab. I di [Z21]: Waveform Length (WL) per canale su finestre di 60 ms
(120 campioni), 16 feature scalate a varianza unitaria sui dati di training; **SVR RBF** (gamma `scale`, C per DoA scelto in validazione sulla
griglia 0,1, 1, 10, 100) con finestre a passo 100 ms; **MLP** a 3 strati nascosti (100, 100, 100, scelta mia: [Z21] non da' le taglie) con
passo 25 ms. Bersaglio all'ultimo campione della finestra; EMA con alpha dalla validazione. Due protocolli per ciascun modello: **fold** (A1 ->
A2 e viceversa, come la primaria) e **full** (A1 + A2, C e alpha dai fold). Test su A3, 12 soggetti; un seme per l'MLP (la SVR e' deterministica).

Lettura congelata (con l'approvazione), tolleranza ±0,30 come per la TCN, sulla SVR (l'MLP e' meno specificato ed e' descrittivo):

| Esito della SVR | Lettura | Conseguenza |
|---|---|---|
| fold entro 7,28 ± 0,30 | pipeline e protocollo fold coerenti col paper: la distanza e' nel training della TCN | passo 2 |
| fold sopra, full entro 7,28 ± 0,30 | indizio per H1 (modelli finali sulle sessioni unite) | passo 2, poi passo 3 con `full` come grandezza principale |
| fold e full entrambi sopra 7,58 | indizio per H2 (pipeline o dati diversi) | stop alla TCN: si torna ai dati con gli autori |
| altri casi | non conclusivo | passo 2, lettura prudente |

Costo: un job su `lrd_all_serial`, 8 CPU, limite 4 ore: al massimo **32 core-ore**, 0 GPU-ore. La SVR su ~11.500 finestre per fold e ~23.000
per `full` e' la parte lenta: il job registra i tempi per fit e salta la griglia oltre un tempo massimo dichiarato (nessun risultato parziale
spacciato per completo).

## Passo 2 — varianti della TCN scelte in validazione (GPU)

Fold1 e fold2, seme 0, 12 soggetti, protocollo del run 63988277 (passo di training 8, 19 epoche, Adam 1e-4, cuDNN veloce). Varianti:

- **V1**: bersaglio standardizzato per DoA con media e deviazione standard delle finestre di training; loss L1 sul bersaglio standardizzato,
  uscite riportate in gradi prima di EMA e metriche (H3);
- **V3**: padding simmetrico delle convoluzioni dilatate, come Q-PPG (H4);
- **V13**: V1 + V3.

Base V0 gia' disponibile: validazione 9,906 (run 63988277, seme 0). **Regola di scelta congelata:** si sceglie la variante con la MAE di
validazione (fold, EMA, media su 12 soggetti e 2 fold) piu' bassa, solo se migliora V0 di almeno 0,20 gradi; altrimenti resta V0. Nessun numero
di test entra nella scelta: il job li calcola, ma la sintesi del passo 2 riporta solo la validazione, e il test si apre al passo 3.

Costo: 72 modelli di fold (~94 s l'uno) in un job da 1 GPU, limite 2 h 30 min: al massimo **2,5 GPU-ore**.

## Passo 3 — conferma sul test (GPU)

Configurazione scelta al passo 2, 12 soggetti x 3 semi, test aperto una volta sola. Composizione secondo il passo 1:

- caso «fold coerente» o non conclusivo: fold x 3 semi + `full` col seme 0, come il run 63988277 (~2,5 h, limite 3 h);
- caso H1: fold e `full` x 3 semi (~3,9 h, limite 4 h 30 min), con `full` come grandezza principale.

Criterio congelato: riferimento 6,89, tolleranza ±0,30, stesse etichette della replica (replicata / non replicata / non coerente);
normodotati e amputati separati, descrittivi. Il verdetto del run 63988277 non si riscrive: il passo 3 e' una seconda analisi con i dettagli
scelti in validazione, e si dichiara come tale.

## Costi e tetto proposto

| Passo | Risorse | Tetto |
|---|---|---|
| 0 | domanda agli autori | 0 |
| 1 | CPU, 8 core, ≤ 4 h | 32 core-ore |
| 2 | 1 GPU, ≤ 2 h 30 min | 2,5 GPU-ore |
| 3 | 1 GPU, ≤ 3 h oppure ≤ 4 h 30 min | 3 oppure 4,5 GPU-ore |
| totale nuovo | | **≤ 7 GPU-ore e ≤ 35 core-ore** (56 ore locali di GPU con billing 8) |

Il tetto attuale dell'attivita' (5 GPU-ore, 40 core-ore) ha 2,09 GPU-ore e ~39,7 core-ore residue: le core-ore bastano, per la GPU serve un
nuovo tetto, **12 GPU-ore in tutto** (2,907 spese + 7 + margine), per decisione di Simone. Ogni job resta sotto le 5 GPU-ore.

## Codice da scrivere dopo l'approvazione

- `scripts/db8_classical_baselines.py` (WL, SVR, MLP sulla stessa cache e con le stesse funzioni di `harness/db8_regression.py`) e il suo
  launcher CPU; test su dati finti, come per la TCN.
- `--target-norm` in `scripts/temponet_db8.py` (V1; V3 esiste gia' con `--symmetric-padding`), un modo `screen` del launcher che produce solo
  la sintesi di validazione, e i test.
- Ogni passo: commit, deploy, checkout congelato, job dal submitter comune, ricevute in `results/`, voce di registro; nessun rilancio senza
  diagnosi.

## Cosa non si fa

Nessuna selezione sul test; niente quantizzazione int8 o QAT; nessuna variante oltre V1, V3 e V13 senza una nuova proposta; nessuna
riscrittura dell'esito del run 63988277.

## Esiti

### Passo 0 — 09/11/2026

Simone: «non rispondono vai avanti». Nessuna risposta degli autori; i passi 1-3 procedono coi cancelli congelati.

### Passo 1 — job 64001955, 09/11/2026

`TEMPONET-DB8-classical-20261109` = 64001955 (codice 9584631), COMPLETED 0:0 in 57 min 37 s (8 CPU, 7,68 core-ore allocate, 5 h 56 min di CPU),
12 soggetti su 12. Ricevute in `results/temponet_db8/classical_20261109/`.

**Cancello: «indizio H1».** SVR fold **8,279** (sopra 7,58), SVR full **7,366** (dentro 7,28 ± 0,30): il numero pubblicato della SVR torna con
il training sulle sessioni 1 e 2 unite, non per fold. Conseguenza congelata: passo 2, poi passo 3 con `full` come grandezza principale.

| Modello | fold con EMA | full con EMA | acc. 10 / 15 (fold) | pubblicato |
|---|---|---|---|---|
| SVR | 8,279 | **7,366** | 0,733 / 0,859 | 7,28 (0,795 / 0,883) |
| MLP (descrittivo) | 7,691 | 6,824 | 0,778 / 0,894 | 7,14 (0,799 / 0,889) |
| TEMPONet (run 63988277) | 8,795 | 7,778 (seme 0) | 0,691 / 0,822 | 6,89 (0,814 / 0,900) |

Normodotati e amputati (descrittivo): SVR full 7,232 e 8,032; MLP full 6,581 e 8,038. S12 e' il soggetto peggiore anche per SVR e MLP (9,8-10,4),
S11 fra i migliori: la difficolta' di S12 non e' propria della TCN.

Limiti, da dire senza correggere a posteriori: la SVR ha scelto C = 100, il bordo superiore della griglia congelata, in 97 fit su 120 (C = 10 in
21, C = 1 in 2): con C piu' grandi SVR fold e full potrebbero scendere ancora, quindi i numeri della SVR sono un limite superiore. Nessun MLP ha
raggiunto la convergenza di scikit-learn in 200 iterazioni (l'MLP era descrittivo). Letture descrittive: con le stesse finestre e la stessa
EMA, l'MLP su WL batte la TEMPONet in entrambi i protocolli (7,69 contro 8,80 fold; 6,82 contro 7,78 full), cosa che il paper non riporta (TCN
6,89 contro MLP 7,14): resta un difetto proprio del training della TCN, oggetto del passo 2.

### Passo 2 — job 64005316, 09/11/2026

`TEMPONET-DB8-screen-20261109` = 64005316 (codice 9584631), COMPLETED 0:0 in 1 h 52 min 14 s (1 GPU: **1,871 GPU-ore**), 72 modelli di fold su
72 (3 varianti x 12 soggetti x 2 fold, seme 0), ~93 s l'uno. Ricevute in `results/temponet_db8/screen_20261109/`, solo validazione: `screen.json`,
`screen.md` e i `run.json`. Il log del job (`temponet_db8_64005316.out` in `wearusfm_runs/logs`, fuori dal commit per la regola `*.out`) non contiene
numeri di test grazie a `--hide-test-in-log`, controllato prima della copia. Le unita' con le metriche di test restano su Leonardo, chiuse: il test
si apre solo al passo 3.

| Variante | validazione (fold, EMA) | fold1 | fold2 | da V0 | soggetti meglio / peggio di V0 |
|---|---|---|---|---|---|
| V0 (run 63988277) | 9,906 | 9,666 | 10,145 | | |
| V1 bersaglio standardizzato | 9,594 | 9,488 | 9,700 | −0,312 | 11 / 1 |
| V3 padding simmetrico | 9,941 | 9,704 | 10,178 | +0,035 | 5 / 7 |
| V13 entrambi | **9,590** | 9,458 | 9,721 | −0,316 | 11 / 1 |

**Regola congelata: scelta V13** (validazione piu' bassa, guadagno 0,316 sopra la soglia di 0,20). Lettura descrittiva: il guadagno viene dal
bersaglio standardizzato (H3); V1 e V13 differiscono di 0,005, entro il rumore, e il padding simmetrico da solo non migliora (H4 non sostenuta).
Quanto di questo guadagno passi al test, e quanta della distanza dal paper chiuda, lo dice solo il passo 3.

### Passo 3 — job 64010753, 09-10/11/2026

`TEMPONET-DB8-confirm-20261109` = 64010753, codice 5a26a19 (checkout `temponet-db8-5a26a19`, job 64005365), variante V13, fold e `full` x 3
semi su 12 soggetti, `full` grandezza principale (caso H1). Inviato il 09/11 alle 20:53, COMPLETED 0:0 alle 00:34 del 10/11 in 3 h 40 min 4 s
(**3,668 GPU-ore**), 36 unita' su 36. Ricevute in `results/temponet_db8/confirm_20261109/`: unita', `run.json`, sintesi del job (`summary_job.*`)
e sintesi corretta (`summary.*`, vedi sotto).

**Esito secondo il criterio congelato: NON REPLICATA.** MAE `full` con EMA **7,585 gradi** (semi 7,651 / 7,507 / 7,597) contro 6,89 ± 0,30,
differenza +0,695. Stesso valore ricalcolato a parte sul Mac direttamente dalle unita'. E' una seconda analisi coi dettagli scelti in
validazione: il verdetto del run 63988277 (8,795, fold) resta com'e'.

| | V0 (run 63988277) | V13 (passo 3) | pubblicato |
|---|---|---|---|
| `full` con EMA | 7,778 (solo seme 0) | **7,585** (seme 0: 7,651) | 6,89 |
| fold con EMA | 8,795 | 8,509 | |
| accuratezza a 10 / 15 gradi, `full` | 0,729 / 0,873 (seme 0) | 0,743 / 0,889 | 0,814 / 0,900 |
| alpha dell'EMA, `full` | 0,932 ± 0,023 (seme 0) | 0,929 ± 0,027 | 0,862 ± 0,044 |
| dev. std. sui 60 valori soggetto x DoA, `full` | | 2,05 | 2,08 |

Descrittivi: `full` migliore del fold in 12 soggetti su 12; normodotati 7,394 e amputati 8,540 (S11 6,574, S12 10,506); per DoA da 5,70
(flessione del pollice) a 9,97 (rotazione del pollice).

**Lettura.**
- La catena (cache, DoA, split, EMA, metriche) e' coerente col paper: la SVR del passo 1 col protocollo `full` torna a 7,37 contro 7,28.
- Il protocollo spiega circa un grado: fold 8,51, `full` 7,59.
- V13 toglie circa 0,1-0,3 gradi: fold da 8,795 a 8,509, `full` col seme 0 da 7,778 a 7,651. In validazione il guadagno veniva dal bersaglio
  standardizzato, e il padding simmetrico da solo non aiutava; sul test si e' misurata solo V13.
- Restano **0,70 gradi propri del training della TCN**. Con la stessa catena e lo stesso protocollo, l'MLP su Waveform Length (6,824,
  descrittivo) fa meglio della TEMPONet di 0,76 gradi; nel paper e' il contrario.
- Interpretazione: l'alpha piu' alto del pubblicato (0,93 contro 0,86) indica uscite grezze piu' rumorose, che chiedono piu' lisciatura.

Ipotesi rimaste, non verificate: dose e schedule del training (H5: epoche, learning rate e sua schedule) e altri dettagli non dichiarati
(augmentation, weight decay, normalizzazione dell'ingresso). Come stabilito dal piano, nessuna variante oltre V1, V3 e V13 senza una nuova proposta.

**Correzione della sintesi, dopo il run (presentazione, non il verdetto).** Con `--primary full`, sotto la primaria `full` la sintesi del job
riportava righe del fold senza dichiararlo: per seme (8,517 / 8,547 / 8,463), per soggetto, per DoA, deviazione standard, accuratezze e alpha.
`scripts/temponet_db8_summary.py` ora calcola le stesse descrittive anche per il `full` (`full_details`, e `full_acc_*` per gruppo). La sintesi
in markdown dichiara il modello di ogni riga e mette fold e `full` affiancati nelle tabelle. Test d'uguaglianza sulle ricevute reali:
- modo fold (run 63988277): markdown identico byte per byte;
- nei due modi: chiavi JSON preesistenti identiche, si aggiungono solo le chiavi nuove;
- nuovo test CPU con fold e `full` diversi; 53 test passati.

La sintesi del job resta come ricevuta (`summary_job.*`).

**Consuntivo della diagnosi.**
- Passo 1: 7,68 core-ore (solo CPU).
- Passo 2: 1,871 GPU-ore.
- Passo 3: 3,668 GPU-ore.
- Due checkout: 0,04 core-ore.
- Totale: **5,538 GPU-ore e 7,72 core-ore**.

Con la replica, l'attivita' arriva a **8,445 GPU-ore sul tetto di 12** (67,6 ore locali con billing 8) e **8,04 core-ore sul tetto di 40**.
Rettifica al consuntivo della replica: i checkout allocano 2 CPU, non 1, quindi i job solo CPU della replica valgono 0,32 core-ore e non 0,27.

Spazio: risultati 198 MB su `$WORK` (`confirm` 72, `screen` 57, `full` 56 MB) e cache 5,0 GB su `$SCRATCH`.
