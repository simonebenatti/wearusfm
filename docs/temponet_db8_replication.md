# Replica di TEMPONet sulla regressione della cinematica della mano (NinaPro DB8)

Claude, 09/11/2026. Ramo `claude/temponet-db8-20261109`. Criterio e tetto congelati in `docs/decisioni.md` (voce del 09/11/2026) prima di
qualunque lancio.

## Mandato

Simone, 09/11/2026: «allora qua in autonomia testiamo la TCN temponet. Da internet e dai repository ti scarichi la rete, la alleni e la testi
sul suo dataset di riferimento guardando i paper che trovi in letteratura e replichi i risultati per un task di regressione (guarda prevlentemente
i lavori di zanghieri e orlandi). Poi una volta che abbiamo l codice funzionante vediamo.»

Collocazione nel progetto: e' la baseline «CNN/TCN allenata da zero sul dataset target» di `docs/fm_emg_reference_v10.md` (baseline non
negoziabili), elencata fra i debiti dell'harness v10 in `docs/finestra1_job_plan.md`. Questa consegna e' solo la replica sul dataset di
riferimento della rete; l'uso come baseline del FM e' un passo successivo da decidere.

## Fonti lette

| Sigla | Lavoro | Che cosa ne prendo |
|---|---|---|
| [Z21] | Zanghieri, Burrello, Kartsch, Benatti, Meattini, Palli, Melchiorri, Benini, «sEMG-based Regression of Hand Kinematics with Temporal Convolutional Networks on a Low-Power Edge Microcontroller», IEEE COINS 2021, DOI 10.1109/COINS51742.2021.9524188, postprint https://hdl.handle.net/11585/841320 | architettura (sez. II-C, Fig. 1), protocollo (II-D), metriche (III-A), risultati da replicare (Tab. I) |
| [Z20] | Zanghieri, Benatti, Burrello, Kartsch, Conti, Benini, «Robust Real-Time Embedded EMG Recognition Framework Using Temporal Convolutional Networks on a Multicore IoT Processor», IEEE TBioCAS 14(2), 2020, postprint https://hdl.handle.net/11585/712563 | TEMPONet originale per classificazione (sez. III-B1, Fig. 3): canali, campo ricettivo, protocollo a 2 fold |
| [Z23] | Zanghieri, Benatti, Benini, Donati, «Event-based Low-Power and Low-Latency Regression Method for Hand Kinematics from Surface EMG», IWASI 2023, postprint https://hdl.handle.net/11585/946796 | passo d'inferenza della TCN di [Z21] (16 ms), deviazione standard del MAE di [Z21] (6,89 ± 2,08, ottenuta dagli autori riproducendo il setup), esperimenti per soggetto |
| [Z24] | Zanghieri, Rapa, Orlandi, Donati, Benini, Benatti, «sEMG-driven Hand Dynamics Estimation with Incremental Online Learning on a Parallel Ultra-Low-Power Microcontroller», IEEE TBioCAS 18(4), 2024, postprint https://hdl.handle.net/11585/972134 | lavoro con Orlandi: regressione di forza su HYSER con una rete piccola diversa da TEMPONet (codice pulp-bio/incremental_hyser); non e' il bersaglio di questa replica |
| [K19] | Krasoulis, Vijayakumar, Nazarpour, «Effect of User Practice on Prosthetic Finger Control With an Intuitive Myoelectric Decoder», Front. Neurosci. 13:891, 2019, e Data Sheet 1 | definizione delle 5 DoA (eq. S2) e delle tre acquisizioni (10, 10 e 2 ripetizioni) |

Codice cercato (cloni in sola lettura, nulla eseguito, nulla copiato nel repo):

- `pulp-bio/event-based-semg-regression` @ `1920531b733aa3af7340f013c3d718b8ba318cb2` (LGPL-2.1): caricatore di DB8 degli autori, con la
  matrice DoF->DoA trascritta da [K19] eq. S2 e `scipy.io.loadmat` su `emg` e `glove`.
- `EmbeddedML-EDAGroup/Q-PPG` @ `ddf3866da6d5f9dda4da7d7884b4f1f3b809a6ba` (Apache-2.0): TEMPONet in PyTorch di un coautore (Burrello) per la
  PPG, `precision_search/model/TCN_variants.py`, classe `TCN_network_float`: ordine dei layer e lunghezza finale 4 con finestre di 256 campioni.
- `pulp-bio/incremental_hyser` @ `060d1e3877442e0334404a21eed00d9b23a04f68`, `pulp-bio/hdsemg-force-regression` @ `f3aafbf2`,
  `MarcelloZanghieri2/tcn_under_review` @ `4239ea2c`: reti diverse da TEMPONet.
- Nessun codice pubblico della TEMPONet di regressione di [Z21] trovato (organizzazione pulp-bio, profilo dell'autore, ricerca web). La rete e'
  quindi reimplementata da zero in `src/wearusfm/model/temponet.py` dalle fonti sopra.

Matrice DoF->DoA verificata sul PDF dell'editore (Data Sheet 1, pagina 4, eq. S2): la stampa ha **19 righe** per x in R^18; l'ultima e' tutta a
zero. Si usano le prime 18, come il codice degli autori; il refuso non cambia nulla se la riga in piu' e' l'ultima, come nel codice degli autori.

## Risultati pubblicati da replicare ([Z21] Tab. I)

| Modello | Formato | MAE | accuratezza 10 gradi | accuratezza 15 gradi |
|---|---|---|---|---|
| SVM (WL) | fp32 | 7,28 | 0,795 | 0,883 |
| MLP (WL) | fp32 | 7,14 | 0,799 | 0,889 |
| LSTM (Koch et al. 2020) | fp32 | 7,04 | - | - |
| **TEMPONet, Block III a 64** | **int8** | **6,89** | **0,814** | **0,900** |

Altri numeri di [Z21]: 70,9 kB in int8, 3,16 MMAC, alpha dell'EMA medio 0,862 con deviazione standard 0,044 fra soggetti e DoA; [Z23]: MAE
6,89 ± 2,08 gradi.

## Architettura ricostruita

Dal testo di [Z21] sez. II-C2: tre blocchi, ciascuno con due convoluzioni causali dilatate (kernel 3, dilatazione 2, 4, 8, «full padding») e una
convoluzione con kernel 5 e stride 1, 2, 4 seguita da average pooling 2; canali dei blocchi 16, 32, 64 (dimezzati rispetto ai 32 e 64 di [Z20],
Block III esplorato fra 32 e 192, scelto 64); FC I, FC II a 32 unita', FC III a 5 unita'; BatchNorm e ReLU ovunque tranne FC III; dropout 0,5 su
FC I e FC II; ingresso 16 x 256 grezzo.

Due ambiguita' del testo, risolte con i numeri dello stesso paper:

1. **Canali dentro il blocco.** [Z20] Fig. 3 ha convoluzioni dilatate a 16·2^(i-1) canali e la convoluzione con stride che li raddoppia; [Z21]
   parla di un solo numero di canali per blocco. Con tutte e tre le convoluzioni allo stesso numero di canali i conteggi tornano (punto 2).
2. **FC I.** «FC I has 4× units as Block III's channels»: letta alla lettera (256 unita') darebbe 125.504 pesi e, con Block III a 192, 940.608,
   contro i 70,9 kB e i «less than 500 k parameters» del paper. Letta come riferita all'ingresso (4 passi temporali x 64 canali = 256 ingressi,
   64 unita') i conteggi tornano esattamente.

Con queste letture (convoluzioni e FC I/II senza bias perche' seguite da BatchNorm; FC III con bias; BatchNorm con gamma e beta):

| Block III | parametri | MAC (stride applicato) |
|---|---|---|
| 32 | 29.701 | 1.881.248 |
| 48 | 47.237 | 2.110.112 |
| **64** | **70.917** = 70,9 kB int8 | 2.410.656 |
| 96 | 136.709 | 3.226.784 |
| 128 | 227.077 | 4.329.632 |
| 192 | 481.541 (< 500 k) | 7.395.488 |

Le dimensioni seguono anche le ascisse di [Z21] Fig. 2. I MAC dichiarati (3,16 M per Block III a 64) non coincidono col conteggio a stride
applicato; tornano (3,23 M) contando le due convoluzioni con stride come se fossero calcolate a stride 1: conteggio di [Z21] non dichiarato,
lo si riporta solo come osservazione. Campo ricettivo: 378 campioni ([Z20] Fig. 3: 377).

## Protocollo e scelte dove il paper tace

Dati: raw ufficiale gia' su Leonardo (`$SCRATCH/data/raw/ninapro/DB8`, 36 `.mat`, rinfrescati il 15/10, prossimo rinfresco il 12/11), letti una
volta dal job di cache (`scripts/temponet_db8_cache.py`): `scipy.io.loadmat` impiega minuti per file da 1 GB, troppo per rileggerli a ogni
training (sul login node un solo file ha consumato 4 minuti di CPU il 09/11: lettura interrotta). Fissato da [Z21]: segnali gia' filtrati
10-500 Hz alla fonte, nessun altro preprocessing; finestre di 256 campioni; loss L1, Adam, lr 1e-4, batch 64, 19 epoche fp32; EMA con alpha per
soggetto e DoA dai soli dati di training; test sull'acquisizione 3; MAE e accuratezze a 10 e 15 gradi.

Scelte dichiarate (non specificate da [Z21]):

1. **Fold.** «Sessions 1 and 2 [...] merged and used in a 2-fold cross-validation setup»: i due fold sono le due sessioni (10 ripetizioni per
   movimento ciascuna, [K19]). `fold1`: training su A1, validazione su A2; `fold2`: il contrario. Come in [Z20], i modelli dei fold si valutano sul
   test e si media. Analisi secondaria: modello `full` su A1 + A2, con alpha dalle predizioni fuori fold.
2. **Bersaglio**: DoA all'ultimo campione della finestra (regressione causale).
3. **Passo delle finestre**: 32 campioni (16 ms) in validazione e test, il passo d'inferenza di [Z21] secondo [Z23]; in training **8 campioni
   (4 ms)**, scelto sulla validazione di S1 nei collaudi (emendamento del 09/11 in `docs/decisioni.md`): con Adam a lr 1e-4 e 19 epoche conta
   il numero di passi, e a passo 32 la rete resta sottoaddestrata.
4. **Unita' dell'ingresso**: EMG in **microvolt** (x 1e6 rispetto ai volt del file). E' un cambio di scala costante, non una normalizzazione.
   Con l'inizializzazione di PyTorch la varianza dopo la prima convoluzione e' circa 0,33 x var(x); le statistiche della cache (`cache_v2`,
   36 acquisizioni x 16 canali) danno deviazione standard mediana 12,5 microvolt, 3,7 al 5° percentile, 41,4 al 95°. In volt quella varianza
   sarebbe ~5e-11, in millivolt ~5e-5 al canale mediano e ~5e-6 al 5° percentile, vicina o sotto l'`eps` 1e-5 della BatchNorm; in microvolt e'
   ~50, ben sopra. La prima stesura diceva millivolt con ampiezze stimate «di decine di microvolt» da confermare sulla cache: corretta a
   microvolt il 09/11, prima di qualunque training, sulle sole statistiche dei dati.
5. **Padding**: causale per le convoluzioni dilatate (eq. 1 e testo di [Z21]); 2 per la convoluzione con stride. Q-PPG usa padding simmetrico:
   resta disponibile con `--symmetric-padding`, non usato.
6. **Ordine nel blocco con stride**: Conv -> AvgPool -> BatchNorm -> ReLU, come Q-PPG (in inferenza pool e BatchNorm commutano).
7. **Learning rate costante** (nessuno schedule dichiarato); **niente epoca di quantization-aware training int8**: si valuta il modello fp32 dopo le
   19 epoche fp32. TF32 disattivato.
8. **EMA**: y'_0 = y_0, griglia di alpha 0,00-0,99 a passo 0,01, pareggi all'alpha piu' piccolo, EMA che riparte a ogni registrazione.
9. **Registrazioni intere**, riposi compresi, nessuna rimozione dei transitori (DB8 e' fatto per la regressione dei transitori, [Z21] sez. II-B).
10. **Amputati**: 16 ingressi anche per S11 e S12, con i canali senza sensore (a zero) tenuti come sono (S11: canali 14-16; S12: 13-16,
    deviazione standard nulla nella cache). Ripetizioni per movimento nella cache: A1 10 (S5 ne ha 3 per almeno un movimento: registrazione
    piu' corta, 1.041 s), A2 10 o 11, A3 2; nove movimenti ovunque.
11. **Semi**: tre (0, 1, 2), per stimare la variabilita' dovuta al training; [Z21] riporta un solo valore. Il modello `full` solo col seme 0.
13. **cuDNN non deterministico**: con `cudnn.deterministic=True` le convoluzioni dilatate passano per im2col/col2im di PyTorch, 13 volte piu'
    lente; si rinuncia alla riproducibilita' bit per bit (tre semi). Passo di training catturato in un CUDA graph, verificato contro l'eager
    (stessi parametri, differenza 0 in modo deterministico).
12. **Valori non finiti**: in tutto il raw (36 file controllati il 09/11) c'e' un solo campione non finito, l'ultimo di `S5_E1_A1`, con NaN nelle 18
    colonne del guanto. Regola: si scarta una coda finale non finita lunga al massimo 1 s (qui un campione), registrata nel manifest della cache
    (`trimmed_tail_samples`); qualunque altro valore non finito ferma il job. Trovato dal primo job di cache (63985267, fallito sul controllo).

## Grandezza primaria e criterio

Fissati in `docs/decisioni.md` (09/11/2026) prima di qualunque lancio: MAE in gradi sull'acquisizione 3, dopo l'EMA, media su 12 soggetti x 5 DoA
x 2 modelli di fold x 3 semi. Riferimento 6,89 gradi, tolleranza ±0,30. Tutto il resto e' descrittivo.

## Esecuzione e costi

| Passo | Job | Risorse | Tetto |
|---|---|---|---|
| checkout di campagna | `campaign_checkout.sbatch` | 1 CPU, 10 min | 0,17 core-ore |
| cache .npy | `temponet_db8_cache.sbatch` | `lrd_all_serial`, 8 CPU, 1 h | 8 core-ore, 0 GPU-ore |
| collaudi (tre) | `temponet_db8.sbatch smoke` | 1 GPU, `boost_qos_dbg`, 30 min ciascuno: velocita', equivalenza eager/graph, curve di S1 | 0,434 GPU-ore spese |
| replica | `temponet_db8.sbatch full ... benchmark` | 1 GPU, 3 h; 36 unita' (12 soggetti x 3 semi, `full` solo col seme 0), un processo | 3 GPU-ore |

Uscite fuori dal codice: cache in `$SCRATCH/wearusfm_runs/temponet_db8/cache_v1` (rigenerabile dal raw), risultati in
`$WORK/wearusfm_runs/temponet_db8/<run>` (`run.json`, `units/`, `predictions/`, `models/`, `summary.json`, `summary.md`); sintesi e ricevute si
copiano sul Mac e si committano in `results/temponet_db8/`.

## Fuori perimetro

Quantizzazione int8 e QAT con NeMO, deployment su GAP8, baseline SVM/MLP su Waveform Length, altri dataset (HYSER di [Z24]), uso della rete come
baseline del FM sui benchmark del progetto.

## Esito del run completo (09/11/2026, job 63988277)

`TEMPONET-DB8-full-20261109` = 63988277, codice 92bceb5, COMPLETED 0:0 in 2 h 28 min 21 s (2,47 GPU-ore); 36 unita' su 36. Ricevute in
`results/temponet_db8/full_20261109/` (`run.json`, `units/`, `summary.json` e `summary.md` con i gruppi, `summary_job.*` calcolata dal job,
identica sulle chiavi comuni); modelli e predizioni restano in `$WORK/wearusfm_runs/temponet_db8/full_20261109` (56 MB).

**Verdetto secondo il criterio congelato: non replicata.** MAE primaria **8,795 gradi** contro 6,89 (+1,905; limite 7,19).

| Grandezza | Replica | [Z21] / [Z23] |
|---|---|---|
| MAE fold1/fold2, EMA, 3 semi (primaria) | **8,795** (semi 8,754 / 8,877 / 8,755, dev. std. 0,071) | **6,89** |
| dev. std. sui 60 valori soggetto x DoA | 2,21 | 2,08 |
| accuratezza a 10 e 15 gradi | 0,691 e 0,822 | 0,814 e 0,900 |
| alpha dell'EMA | 0,931 ± 0,030 | 0,862 ± 0,044 |
| senza EMA (fold) | 9,356 | - |
| `full` (A1 + A2, seme 0, descrittivo) | 7,778 (senza EMA 8,241) | - |
| fold1 (training su A1) e fold2 (su A2) | 9,202 e 8,389 | - |
| riferimenti del paper | - | SVM 7,28; MLP 7,14; LSTM 7,04 |

Per DoA (fold, EMA): rotazione del pollice 11,39; flessione del pollice 6,47; indice 8,29; medio 9,48; anulare+mignolo 8,34.

Normodotati e amputati separati (richiesta di Simone del 09/11; descrittivo, nessun verdetto per gruppo):

| Gruppo | Soggetti | fold con EMA | senza EMA | `full` | acc. 10 | acc. 15 | alpha |
|---|---|---|---|---|---|---|---|
| normodotati | S1-S10 | 8,680 | 9,213 | 7,611 | 0,695 | 0,826 | 0,927 |
| amputati | S11-S12 | 9,373 | 10,069 | 8,613 | 0,671 | 0,803 | 0,952 |

Fra gli amputati S11 e' fra i migliori (7,53) e S12 il peggiore di tutti (11,22). Per soggetto (media su fold e semi) i normodotati vanno da
7,57 (S2) a 9,59 (S9).

Letture descrittive, nessuna e' un criterio:

- La distanza e' sistematica: tre semi entro 0,12 gradi, dispersione fra soggetti e DoA come quella pubblicata (2,21 contro 2,08).
- Piu' dati per modello aiutano: `full` (due sessioni) e' migliore della media dei due fold in 12 soggetti su 12 col seme 0 (7,78 contro 8,75);
  fold2, allenato sulla sessione che precede il test, e' migliore di fold1 (8,39 contro 9,20). Anche `full` resta fuori tolleranza.
- Alpha piu' alto del pubblicato (0,93 contro 0,86): le uscite grezze sono piu' rumorose e l'EMA le liscia di piu'.

Ipotesi sulla distanza (da verificare, nessuna e' accertata): dettagli di training non dichiarati da [Z21] (passo delle finestre di training,
che nei collaudi conta attraverso il numero di passi di Adam; schedule del learning rate, «initial learning rate»; normalizzazione del
bersaglio), modello finale allenato sulle due sessioni unite invece che per fold, differenze nella definizione del MAE o del tempo di test.
Le letture dell'architettura sono vincolate dal conteggio esatto dei parametri e sono la causa meno probabile.

## Seconda analisi: diagnosi della distanza (09-10/11/2026)

Piano e passi in `docs/temponet_db8_diagnosis_plan.md`, approvato da Simone («Approve the plan»), cancelli congelati in `docs/decisioni.md`.
Cosa e' emerso:
- La SVR su Waveform Length torna al valore pubblicato (7,37 contro 7,28) solo col modello allenato sulle sessioni 1 e 2 unite («indizio H1»).
- In validazione si e' scelta la variante V13: bersaglio standardizzato e padding simmetrico.
- Con V13 e la stessa grandezza calcolata sul modello `full`: **7,585 gradi**, ancora **non replicata** (+0,695). Il fold scende a 8,509.

Il verdetto della sezione precedente resta com'e'; la distanza residua e' propria del training della TCN.

## Terza analisi: schedule del learning rate (H5, 10/11/2026)

Proposta in `docs/temponet_db8_h5_plan.md`, approvata da Simone («Approvo»), regola congelata in `docs/decisioni.md`. Su V13 si sono provati in
validazione due schedule: coseno sulle 19 epoche (W1) e un'epoca finale a 1e-5 (W2). Il miglioramento e' di 0,06 e 0,10 gradi, sotto la soglia
di 0,20, quindi nessuna conferma sul test. Lo schedule riduce le oscillazioni fra epoche, ma l'EMA ne toglieva gia' gran parte.
**La replica si chiude a 7,585 gradi contro 6,89: non replicata.** I verdetti precedenti restano.
