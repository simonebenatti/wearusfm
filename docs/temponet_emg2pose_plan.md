# Proposta: TEMPONet su emg2pose (regressione della posa della mano)

Claude, 10/11/2026. Alla domanda «Preparo la proposta per TEMPONet su emg2pose?» Simone ha risposto «Sì, preparala».
**E' una proposta: nessun job lanciato, nessun codice scritto.** Regole e soglie diventano vincolanti solo dopo l'approvazione, con una voce in
`docs/decisioni.md` committata prima del primo job.
**Stato, 11/11 pomeriggio:** approvata («Approvo», un seme), passi 0-3 eseguiti, cancello superato, test letto una volta: TEMPONet fra NeuroPose
e SensingDynamics in tutte e tre le condizioni. Il passo 4 non e' approvato. Vedi «Esito».

## Scopo

- **Misura nuova, non una replica.** Una ricerca del 10/11 non ha trovato risultati pubblicati di TEMPONet su emg2pose.
- **Valore per il progetto.** E' la baseline «CNN/TCN allenata da zero sul dataset target» di `docs/fm_emg_reference_v10.md` («Baseline non
  negoziabili»), su emg2pose, con la ricetta della replica su DB8 (conclusioni firmate il 10/11).
- **Riferimento:** il compito di **regressione** del benchmark ufficiale, cioe' la posa senza conoscere quella iniziale ([E24]: Salter et al.,
  emg2pose, NeurIPS 2024 D&B, arXiv 2412.02725, Tab. 4, letta il 10/11). Errore angolare medio assoluto in gradi, media ± dev. std. fra utenti:

| Test | vemg2pose | NeuroPose | SensingDynamics |
|---|---|---|---|
| utenti nuovi (`user`) | 12,2 ± 1,3 | 13,2 ± 1,1 | 15,5 ± 1,4 |
| fasi nuove (`stage`) | 15,2 ± 1,6 | 17,2 ± 1,7 | 18,8 ± 1,6 |
| utenti e fasi nuovi (`user_stage`) | 15,8 ± 1,4 | 17,5 ± 1,5 | 18,7 ± 1,6 |

- Il compito di **tracking** (posa iniziale nota: vemg2pose 7,7 / 11,2 / 11,0) e' fuori: TEMPONet non usa uno stato iniziale.
- [H26] (Hadidi et al., arXiv 2603.08212, marzo 2026) dichiara un nuovo stato dell'arte fra i modelli in streaming; i suoi numeri non sono
  nell'abstract e vanno verificati prima di usarli.

## Cosa c'e' gia' su Leonardo (verificato il 10/11)

- **Raw:** tar da 462,8 GB su `$SCRATCH`, verificato byte per byte.
- **EMG processato:** `$SCRATCH/data/processed/emg2pose/u<utente>/<registrazione>/data_int16.npy` (int16, scala per registrazione),
  allineato riga per riga all'HDF5.
- **Pose:** blocchi a 200 Hz (medie di 10 campioni) con validita' in `$SCRATCH/data/processed/emg2pose_pose/` per tutti i 193 utenti (ingest
  62410740, 27/10). Blocchi validi 82,1%: la validita' toglie i fallimenti della cinematica inversa con un margine di 280 campioni (firmato).
  `pose_stats.json` ha le statistiche per giunto sullo split di training, con standardizzazione unica per i due lati (`per_side = false`).
- **Angoli:** 20 gradi di liberta' in radianti, da motion capture a 60 Hz con cinematica inversa offline, interpolati a 2 kHz e filtrati a 15 Hz
  (fatto n. 28 di `docs/fatti_da_verificare.md`).
- **Split ufficiali** (CSV dei metadati, firmati il 02/10; il test e' fuori dal pretraining del FM):

| Split | Condizione | Registrazioni | Ore | Utenti |
|---|---|---|---|---|
| train | - | 17.136 | 288,9 | 158 |
| val | utenti nuovi / utenti e fasi nuovi | 1.618 / 332 | 27,2 / 5,4 | 15 |
| test | utenti nuovi | 2.172 | 36,4 | 20 |
| test | fasi nuove (utenti del training) | 3.539 | 58,8 | 158 |
| test | utenti e fasi nuovi | 456 | 7,5 | 20 |

## Protocollo ufficiale di valutazione (letto nel codice di emg2pose il 10/11)

- `metrics.AngleMAE`: L1 sugli angoli in radianti negli istanti senza fallimento della cinematica inversa. Un fallimento e' una riga coi 20 angoli
  tutti a zero (`np.isclose`, `utils.get_ik_failures_mask`). Le maschere non sono nel file: si ricalcolano dagli angoli.
- `test_analysis`: split `test` raggruppato per `generalization`; finestre di 10.000 campioni a passo 10.000 (5 s, senza sovrapposizione e senza
  jitter), solo dentro tratti senza fallimenti lunghi almeno 10.000 campioni (`skip_ik_failures=True`); il contesto del modello e' preso prima
  della finestra.
- Il paper riporta media e dev. std. fra utenti (didascalia della Tab. 4).

## TEMPONet: la configurazione della replica su DB8

- **Rete:** architettura ricostruita da [Z21], con uscita a 20 angoli invece di 5 DoA: 71.412 parametri invece di 70.917. V13, cioe' bersaglio
  standardizzato per giunto (`pose_stats.json`) e padding simmetrico; learning rate costante, perche' lo schedule (H5) non ha dato guadagni.
- **Training:** loss L1, Adam 1e-4, batch 64, 19 epoche in float32, cuDNN in modo benchmark, passo catturato in un CUDA graph con il controllo
  eager/graph all'inizio. Un solo modello per i 158 utenti e i due lati, come le baseline ufficiali.
- **Ingresso:** 16 canali a 2 kHz nelle unita' native (int16 diviso per la scala della registrazione), finestra di 256 campioni (128 ms), nessun
  filtro in piu': l'EMG di emg2pose e' gia' filtrato passa-alto alla fonte. Una sola scala costante 10^k, scelta sullo split di training perche'
  la deviazione standard mediana dei canali cada fra 3 e 30 (su DB8, in microvolt, era 12,5): e' la protezione dall'`eps` della BatchNorm usata
  su DB8, non una normalizzazione.
- **Bersaglio:** la posa all'ultimo campione della finestra. In training, finestre che finiscono all'ultimo campione di un blocco valido a 200 Hz,
  a passo 8 blocchi (40 ms), con 255 campioni di storia nella registrazione. Sono circa 21 milioni di finestre e 334 mila passi di Adam per epoca.
- **Post-processing:** EMA causale per giunto con alpha scelto sulla griglia 0,00-0,99 sulla validazione, ripartita a ogni finestra di
  valutazione. Si riportano anche le uscite grezze, perche' le baseline di [E24] non usano l'EMA.
- **Valutazione:** finestre ufficiali; istanti ogni 10 campioni (200 Hz) dall'inizio di ogni finestra, con almeno 255 campioni di storia nella
  registrazione; bersaglio esatto a 2 kHz in quell'istante; errore in gradi (x 180/π).
  - Le finestre che iniziano al campione 0 perdono i primi 255 campioni (128 ms su 5 s).
  - I salti dell'asse dei tempi (1.668 registrazioni) non si trattano, come nel codice ufficiale; si contano e si riportano.

## Passi

**Passo 0 — CPU, `lrd_all_serial`, 2 core, limite 2 h (≤ 4 core-ore).** Dal tar, per le 8.117 registrazioni di validazione e test:
- tratti ufficiali senza fallimenti (inizio e fine a 2 kHz) e angoli esatti negli istanti di valutazione (float32, ~7,8 GB su `$SCRATCH`,
  rigenerabili);
- controllo incrociato con le medie a 200 Hz dell'ingest.

Dal processato: deviazione standard dei canali su 1.000 registrazioni di training estratte col seme 0, per la scala 10^k. L'ingest intero delle
pose ha letto tutte le 25.253 registrazioni in 1 h 57 min su 2 core.

**Passo 1 — GPU, `boost_qos_dbg`, 30 min (≤ 0,5 GPU-ore). Collaudo, nessun numero di test.** Contenuto:
- controllo eager/graph;
- memoria e velocita' del caricatore dalla RAM;
- training breve su un sottoinsieme di utenti di training;
- valutazione e EMA sulla sola validazione.

Regola meccanica, solo sulla velocita': se il passo supera 1,6 ms, il passo delle finestre di training sale al primo multiplo di 10 campioni con
cui 19 epoche stanno in 2 h 45 min.

**Passo 2 — GPU, 1 A100, limite 3 h 30 (≤ 3,5 GPU-ore). Training col seme 0.** Previsti ~6,3 milioni di passi (~2,3 h a 1,3 ms) piu' il
caricamento (~5 min) e la validazione grezza per epoca, solo descrittiva, su istanti a 50 Hz (~6 min in tutto). Si usa il modello finale, senza
scelta dell'epoca, come in [Z21].

**Passo 3 — GPU, limite 45 min (≤ 0,75 GPU-ore). Valutazione del seme 0:**
1. alpha dell'EMA sulla validazione;
2. cancello sulla validazione (sotto);
3. solo se passa, il test, una volta.

**Passo 4 — facoltativo: semi 1 e 2**, training e valutazione come nei passi 2 e 3 (≤ 8,5 GPU-ore).

## Regole da congelare con l'approvazione

1. **Grandezza principale:** MAE angolare in gradi con EMA, per condizione di test (utenti nuovi, fasi nuove, utenti e fasi nuovi). Per ogni utente
   e' la media su tutti gli istanti delle sue finestre ufficiali; si riportano media e dev. std. fra utenti, come in [E24]. Con tre semi, la
   media dei semi. In piu', descrittivi: uscita grezza, media pesata per istanti, per lato e per mano in movimento, landmark non calcolati.
2. **Cancello prima del test**, sulla validazione (15 utenti, finestre ufficiali, media fra utenti): MAE di TEMPONet con EMA / MAE del
   predittore costante (la posa media del training per giunto) **< 0,9**. Se non passa, il test non si guarda e si apre una diagnosi.
3. **Lettura descrittiva**, per condizione: dove cade TEMPONet rispetto alle tre baseline della Tab. 4. Le categorie sono: sotto vemg2pose; fra
   vemg2pose e NeuroPose; fra NeuroPose e SensingDynamics; sopra SensingDynamics. E' un confronto con numeri pubblicati e non un verdetto, perche'
   contesto e training sono diversi.
4. **Test una volta per seme**, dopo il cancello; nessuna scelta dopo averlo guardato. Un passo incompleto non si legge; nessun rilancio senza
   diagnosi.

## Due scelte per Simone

1. **Quando guardare il test.** Subito dopo il cancello, oppure congelare il modello e valutarlo insieme al FM, senza conoscere prima il numero
   della baseline. Raccomando subito: e' l'uso normale di una baseline, e il FM non ha scelte che dipendano da questo numero.
2. **Quanti semi.** Uno (passi 0-3, ≤ 4,75 GPU-ore) oppure tre (anche il passo 4). Su DB8 i semi differivano di 0,07-0,15 gradi; qui la
   variabilita' fra utenti e' di 1-2 gradi. Raccomando di partire col seme 0 e decidere il passo 4 dopo il cancello.

## Costi e tetto

| Passo | Risorse | Tetto |
|---|---|---|
| 0 | CPU, 2 core, ≤ 2 h | 4 core-ore |
| 1 | 1 GPU, dbg, 30 min | 0,5 GPU-ore |
| 2 | 1 GPU, 3 h 30, 8 CPU, 120 GB di RAM | 3,5 GPU-ore |
| 3 | 1 GPU, 45 min | 0,75 GPU-ore |
| 4 (facoltativo) | come 2 e 3, due volte | 8,5 GPU-ore |
| checkout | 2 CPU, pochi minuti | ~0,05 core-ore |

Tetto proposto: **5 GPU-ore** con un seme (4,75 previste), **14 GPU-ore** con tre (13,25); core-ore ≤ 10. Ogni job resta sotto le 5 GPU-ore.
Con billing 8 una GPU-ora vale 8 ore locali.

## Rischi

- **Caricatore nuovo.** Lo split di training occupa 66,6 GB in int16 e non sta in una GPU da 64 GB, come invece su DB8. Le finestre si raccolgono
  dalla RAM del nodo a blocchi, con un thread in parallelo alla GPU; servono circa 58 mila finestre al secondo. Lo misura il passo 1.
- **Memoria del job.** 120 GB per 1 GPU invece dei 100 di DB8; il passo 1 controlla che il billing resti 8.
- **Dati solo su `$SCRATCH`.** Processato e pose dipendono dal rinfresco del 12/11 (gia' in calendario, da non saltare).
- **Manutenzioni.** Dal 10 al 14/11 una parte dei nodi Booster e' ferma, quindi le code si allungano; dal 23 al 25/11 e' fermo tutto Leonardo.
- **Contesto corto.** TEMPONet vede 128 ms, mentre vemg2pose lavora su finestre di 11.790 campioni (5,9 s,
  `config/experiment/regression_vemg2pose.yaml`). Nella regressione della posa assoluta e' il limite principale da aspettarsi: e' probabile
  che TEMPONet resti sopra vemg2pose. Non azzardo un numero.

## Codice da scrivere dopo l'approvazione

- `scripts/emg2pose_eval_targets.py` e il suo launcher CPU: tratti ufficiali e angoli esatti per validazione e test.
- `src/wearusfm/harness/emg2pose_regression.py`, con queste parti:
  - elenco delle registrazioni dal CSV;
  - archivio in RAM (int16 piu' scala per registrazione);
  - indice delle finestre di training;
  - finestre ufficiali, EMA per giunto, metrica in gradi per utente e condizione, predittore costante.
- `scripts/temponet_emg2pose.py`, con due modi:
  - training: TEMPONet a 20 uscite, stepper col CUDA graph su un buffer di batch statico, raccolta delle finestre in un thread;
  - valutazione: alpha su validazione, cancello, test.
- `scripts/slurm/temponet_emg2pose.sbatch` con i modi `smoke`, `train` ed `eval`.
- Test CPU su dati sintetici: mascherina dei fallimenti e finestre ufficiali contro una reimplementazione diretta delle funzioni di emg2pose;
  allineamento finestra-bersaglio; EMA; metrica in gradi; cancello; caricatore contro un'estrazione diretta.
- Per ogni passo: commit, deploy, checkout congelato, job dal submitter comune, ricevute in `results/`, voce di registro.

## Cosa non si fa

- nessuna scelta sul test;
- nessuna modifica dell'architettura (per esempio una finestra piu' lunga) e nessuna QAT int8;
- niente tracking e niente landmark: servirebbero il modello cinematico della mano di emg2pose;
- nessun uso dei dati di test nel pretraining.

Una TEMPONet con piu' contesto sarebbe un altro modello, da proporre a parte.

## Esito (11/11/2026)

Passi 0-3 eseguiti con le regole congelate in `docs/decisioni.md`: approvazione ed emendamento delle finestre del 10/11, rilancio del passo 0
dell'11/11. Training e valutazione dal codice 0c19c4b (checkout `temponet-emg2pose-0c19c4b`, job 64120902). Ricevute in
`results/temponet_emg2pose/`, con sha256 uguali a quelli su Leonardo.

| Passo | Job | Esito | Durata | Consumo |
|---|---|---|---|---|
| 0, primo tentativo | 64096497 | `FAILED 1:0` (I/O, vedi `docs/decisioni.md`) | 1 h 50 min 08 s | 3,671 core-ore |
| 0, rilancio | 64165740 | `COMPLETED 0:0` | 25 min 31 s | 0,851 core-ore |
| 1, collaudo | 64165743 | `COMPLETED 0:0` | 7 min 13 s | 0,120 GPU-ore |
| 2, training | 64169771 | `COMPLETED 0:0` | 3 h 04 min 46 s | 3,079 GPU-ore |
| 3, valutazione | 64169772 | `COMPLETED 0:0` | 18 min 47 s | 0,313 GPU-ore |

Tre checkout CPU (64096213, 64097008, 64120902) da 36-38 s: 0,062 core-ore. Il collaudo 64097115 e' stato cancellato da SLURM senza partire
(afterok del primo tentativo del passo 0).

**Training** (seme 0, 19 epoche, passo delle finestre 8 blocchi):
- 17.136 registrazioni, 61,75 GiB caricati in 2.190 s; 21.039.763 finestre, 328.746 passi per epoca (6.246.174 in tutto) a 1,085 ms.
- Controllo eager/graph: differenza massima 0.
- La loss di training (L1 sul bersaglio standardizzato) scende a ogni epoca, da 0,734 a 0,707.
- La validazione grezza per epoca (istanti a 50 Hz, solo descrittiva) scende da 16,00 a 15,63 gradi. Nelle ultime dieci epoche resta fra 15,63
  e 15,71, tranne l'epoca 18 (16,01).
- Memoria massima 78,15 GB su 120.
- Circa 28 minuti, fra il caricamento e il controllo eager/graph, sono andati negli indici delle finestre: lettura delle pose di 19.086
  registrazioni. Il dato e' dedotto dalle date dei file e non misurato a parte.
- Si usa il modello finale, come stabilito.

**Cancello** (validazione: 15 utenti, 1.609 registrazioni, 12.468 finestre ufficiali):
- MAE con EMA 15,116 gradi (dev. std. fra utenti 1,19) contro 18,191 del predittore costante: rapporto **0,831 < 0,9, superato**.
- Grezza 15,387.
- Alpha per giunto da 0,89 a 0,97, mediana 0,96: costanti di tempo da 43 a 164 ms sugli istanti a 200 Hz.

**Test**, letto una volta subito dopo il cancello (5.884 registrazioni, 48.299 finestre). MAE in gradi, media ± dev. std. fra utenti:

| Condizione | Utenti | Finestre | TEMPONet con EMA | grezza | pesata per istanti | costante | rapporto con la costante | vemg2pose / NeuroPose / SensingDynamics | Lettura |
|---|---|---|---|---|---|---|---|---|---|
| utenti nuovi (`user`) | 20 | 15.346 | **14,55 ± 1,54** | 14,86 | 14,35 | 17,51 | 0,831 | 12,2 / 13,2 / 15,5 | fra NeuroPose e SensingDynamics |
| fasi nuove (`stage`) | 158 | 29.146 | **17,67 ± 1,60** | 18,06 | 17,58 | 19,73 | 0,896 | 15,2 / 17,2 / 18,8 | fra NeuroPose e SensingDynamics |
| utenti e fasi nuovi (`user_stage`) | 20 | 3.807 | **17,60 ± 1,76** | 17,97 | 17,44 | 19,58 | 0,899 | 15,8 / 17,5 / 18,7 | fra NeuroPose e SensingDynamics |

**Lettura** (regola 7, descrittiva, non un verdetto). In tutte e tre le condizioni TEMPONet cade **fra NeuroPose e SensingDynamics**.
- Le categorie restano le stesse con l'uscita grezza, che e' il confronto piu' diretto, perche' le baseline di [E24] non usano l'EMA.
- Distanze da vemg2pose: +2,35, +2,47 e +1,80 gradi.
- Distanze da NeuroPose: +1,35, +0,47 e +0,10 gradi.
- L'EMA toglie 0,31, 0,38 e 0,37 gradi.

Descrittivi sul test intero (condizioni insieme, con EMA):
- per lato: destra 17,21 ± 1,82, sinistra 17,62 ± 1,95 (178 utenti);
- per mano: in movimento 17,50 ± 1,66 (178 utenti), ferma 10,58 ± 2,06 (20 utenti, 6,9% degli istanti).

Osservazioni (interpretazioni dopo il test, nessuna scelta ne dipende):
- **Fasi nuove.** TEMPONet toglie il 17% dell'errore del predittore costante sugli utenti nuovi (2,97 gradi), ma solo il 10% sulle fasi nuove
  (2,06) e su utenti e fasi nuovi (1,98). Il salto da utenti nuovi a fasi nuove (+3,13 gradi) e' simile a quello delle baseline (vemg2pose +3,0,
  NeuroPose +4,0, SensingDynamics +3,3).
- **Contesto corto.** Era il limite atteso (128 ms contro i 5,9 s di vemg2pose, sezione «Rischi»), ma questi numeri non lo isolano: servirebbe
  un modello con piu' contesto, cioe' un altro modello.
- **Un solo seme.** In `user_stage` la distanza da NeuroPose (0,10 gradi) e' dell'ordine della variabilita' fra semi vista su DB8 (0,07-0,15):
  in quella condizione la categoria e' fragile. Le altre distanze dalle soglie delle categorie vanno da 0,47 a 1,35 gradi.

**Consuntivo:**
- **3,513 GPU-ore sul tetto di 5** (28,1 ore locali con billing 8): collaudo 0,120, training 3,079 e valutazione 0,313, ciascuno sotto il
  proprio budget (0,5, 3,5 e 0,75).
- **4,583 core-ore sul tetto di 10**: passo 0 4,522 su 5,7 e checkout 0,062.

Conservazione:
- Modello finale in `$WORK/wearusfm_runs/temponet_emg2pose/train_20261111/model_final.pt` (313.772 byte, sha256 `d2d4bed9...3582`), con i
  checkpoint per epoca nella stessa cartella. Cartelle di training e valutazione su `$WORK`: 5,5 MB.
- Bersagli del passo 0 su `$SCRATCH`, rigenerabili e soggetti alla cancellazione a 40 giorni.

Il passo 4 (semi 1 e 2) resta non approvato. Una scelta presa adesso verrebbe dopo aver guardato il test.
