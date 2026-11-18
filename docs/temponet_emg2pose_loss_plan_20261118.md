# TEMPONet su emg2pose: studio delle loss (18/11/2026)

Richiesta di Simone, 18/11/2026: «riprendiamo temponet che abbiam gia provato e vorrei studiare diversi tipi di loss per la regressione su
temponet»; poi «puoi vedere cosa succederebbe usando le loss che abbiam nel FM ? [...] oppure quella con MSE della posa e MSE della velocità
che si trova in alcuni papers ?», con il paper di riferimento arXiv 2610.32674 ([S26], Sîmpetru et al. 2026, tabella S1 del supplemento).
Approvazione: alla domanda «Approvi i tre bracci, smooth L1, MSE e MSE più velocità, un seme, circa 12 GPU-ore?» Simone ha risposto «Sì,
approvo».

Codice: `scripts/temponet_emg2pose.py` (`--loss`), `scripts/slurm/temponet_emg2pose.sbatch` (`LOSS`, `TRAIN_TIME`, modo `losssmoke`), test in
`tests/cpu/test_temponet_emg2pose.py`. Commit del codice `0342170`. Il percorso L1 e' identico bit per bit a quello dell'11/11 (`ee118a2`) su
CPU: `results/temponet_emg2pose/loss_study_20261118/check_l1_identity*`.

## Domanda

A ricetta fissa, quanto cambia l'errore di TEMPONet sul test di emg2pose se cambia solo la funzione di loss?

## Bracci

| Braccio | Loss sul bersaglio standardizzato (`pose_stats.json`) | Origine |
|---|---|---|
| A, fatto | L1 | ricetta dell'11/11: training 64169771, valutazione 64169772 |
| B | smooth L1 con beta 1 (in deviazioni del giunto; default di PyTorch) | [S26] Tab. S1: Smooth-L1 per TCN, Transformer e CNN+TCN dei confronti |
| C | MSE | il termine della posa del FM (`training/jepa.py`, errore quadratico); senza le ancore, che chiedono finestre da 100-500 ms e TEMPONet ne vede 128 |
| D | MSE della posa + 0,1 x MSE della velocita' | [S26] Tab. S1, Transformer di emg2pose: «Coordinate MSE + 0.1× velocity MSE», errori divisi per la deviazione dei bersagli del training |

Fuori dallo studio la loss «logaritmica» di [S26]: e' la ricostruzione del tokenizer (smooth L1 sullo spettro di potenza logaritmico), non una
loss della posa.

Braccio D in dettaglio:
- La velocita' e' la differenza fra la posa di una finestra e quella di una **compagna** che finisce 4 blocchi prima (20 ms), nella stessa
  registrazione. La compagna vale se tutti i blocchi in mezzo sono validi e la sua finestra ha la storia intera; altrimenti ha peso 0 (al suo
  posto c'e' la finestra stessa). Nessuna divisione per il tempo: predizione e bersaglio hanno la stessa scala.
- Velocita' predetta = (z - z_compagna) x sd; bersaglio = y - y_compagna, in radianti; errore diviso per sd_v, la deviazione per giunto della
  velocita' bersaglio sulle coppie valide del training (della popolazione, calcolata nel job e scritta in `run.json`).
- Termine della posa solo sulle finestre di partenza: sono le stesse dei bracci A-C.
- Differenze inevitabili rispetto a C, dichiarate: le due finestre passano insieme nel modello, quindi BatchNorm normalizza su 2 x 64
  finestre e le coppie si sovrappongono per 216 campioni su 256; le maschere del dropout della finestra e della compagna sono indipendenti,
  quindi il termine della velocita' penalizza anche la variazione dovuta al dropout.

## Cosa resta fisso

Tutto il resto della ricetta dell'11/11: dati, split, finestre ogni 8 blocchi, V13, Adam 1e-4 costante, batch 64, 19 epoche, seme 0 (stessa
inizializzazione e stesso ordine delle finestre del braccio A), cuDNN benchmark, CUDA graph con controllo eager/graph, modello finale. Stessa
valutazione: alpha dell'EMA per giunto sulla validazione, cancello < 0,9, test una volta.

## Lettura (congelata in `docs/decisioni.md` prima del lancio)

Metrica principale: MAE di test con EMA, media fra utenti, per condizione (user, stage, user_stage), come l'11/11. Per ogni braccio e
condizione, Δ = braccio - A:
- |Δ| ≤ 0,30 gradi: **non distinguibile** da L1 con un seme;
- Δ < -0,30: **migliore** di L1, segnale da confermare con altri semi;
- Δ > +0,30: **peggiore** di L1, segnale da confermare.

La soglia 0,30 e' la tolleranza della replica su DB8; li' la deviazione fra tre semi era 0,071-0,073 gradi e l'escursione fino a 0,14
(`docs/temponet_db8_replication.md`). Su emg2pose non c'e' una misura fra semi: e' un'ipotesi dichiarata, non una stima.

Descrittivi, senza soglia: utenti migliori e peggiori (confronto diretto dei valori per utente di `per_user_deg`, conteggi interi, pareggi a
parte); MAE di validazione con EMA e cancello; MAE grezza; curve di validazione per epoca; per D i due termini della loss.

Un braccio che non passa il cancello si riporta come tale, senza test e senza nuovo tentativo. Nessuna loss si adotta su questo run: per
adottarla serve una proposta con piu' semi.

## Esecuzione e costi

Ordine: checkout di campagna; collaudo `losssmoke` su `boost_qos_dbg` (al massimo 0,5 GPU-ore): un'epoca breve per loss, con il controllo
eager/graph; training B e C con limite 3:30 e valutazione `afterok` con limite 0:45; poi training D con il limite scelto dal collaudo, e la
sua valutazione.

Tempi dell'11/11 (braccio A): 11.074 s in tutto, di cui 4.025 s di caricamento e indici e 19 epoche da 357 s di training e 14 di
validazione. B e C costano come A. Per D, r = ms per passo di `mse_vel` / ms per passo di `l1` nel collaudo; tempo atteso
D_s = 1,05 x 4.025 + 19 x (357 r + 14) secondi.

Regole meccaniche, fissate prima del collaudo:
1. Se il controllo eager/graph del collaudo fallisce per una loss, quel braccio non parte: si apre una diagnosi.
2. Limite di D: il piu' piccolo `TRAIN_TIME` della tabella dello sbatch con tempo massimo (limite - 900 s) ≥ 1,10 x D_s.
3. Spesa attesa = B + C + D_s/3600 + tre valutazioni (0,31 l'una, come l'11/11) + collaudo. Se supera **12,5 GPU-ore**, D non parte e si
   torna da Simone.

Approvato: circa 12 GPU-ore. Previsione con r = 1,5: 3,08 + 3,08 + 4,07 + 0,93 + 0,25 = circa 11,4 GPU-ore. I limiti di tempo prenotati
sono piu' larghi (3,5 + 3,5 + il limite di D + 3 x 0,75 + 0,5), ma l'addebito e' sul tempo effettivo: i limiti contano solo se un job si
blocca. Nessun nuovo tentativo alla cieca.
