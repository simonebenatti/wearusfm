# TEMPONet residua su emg2pose: piano di un seme (18/11/2026)

**Stato:** piano e regola di lettura congelati prima del lancio (voce del 18/11 in `docs/decisioni.md`). Nessun risultato.
Ramo `claude/temponet-residual-419618`, partito da `ee118a25` (ramo `claude/temponet-db8-20261109`).

## Richiesta

- Simone, 18/11/2026: «apri un tuo fratello e fagli provare una versione della temponet in cui pero inseriamo un residual block per ognuno
  dei 3 blocchi».
- Poi, sullo schema di [Z20] Fig. 3, col gruppo «x 3» cerchiato: «io i residual li metterei nel x3, in ognio blocco cerchiato».
- **Mia lettura (interpretazione):** un salto attorno a ciascuno dei tre blocchi interi. Il blocco comprende le due convoluzioni dilatate
  (Sec. 1 della figura) e la riduzione (Sec. 2: convoluzione con stride, pooling, BN e ReLU).
- La lettura sostituisce la proposta di partenza della sessione che mi ha aperto, che metteva il salto attorno alle sole due dilatate.
- **Riferimento:** la TEMPONet L1 dell'11/11 (codice 0c19c4b, training 64169771, valutazione 64169772), con conclusioni firmate il 12/11.
  Test con EMA, media ± dev. std. fra utenti: `user` 14,55 ± 1,54, `stage` 17,67 ± 1,60, `user_stage` 17,60 ± 1,76. Cancello 0,831.

## Architettura: l'unico fattore che cambia

Blocco con canali c_in -> c, dilatazione d e stride s (1, 2, 4):

    y = ReLU( BN(AvgPool_2(Conv_5,s(TC2(TC1(x))))) + BN(Conv_1x1(AvgPool_2s(x))) )

- **Ramo principale:** invariato. Cambia solo il posto dell'ultimo ReLU, che passa dopo la somma come nel blocco di ResNet [H16].
- **Salto:** a proiezione, cioe' 1x1 senza bias e BN (eq. 2 di [H16]). Prima della 1x1 c'e' un average pooling, come in ResNet-D [H19].
  - Kernel e passo del pooling valgono 2s, cioe' 2, 4 e 8. Le lunghezze tornano 128, 32 e 4, come nel blocco; il costruttore lo verifica.
  - Il pooling media i campioni. Una 1x1 con passo 2s ne scarterebbe 7 su 8 nel blocco 3.
- **Proiezione anche nel blocco 1** (16 -> 16), dove l'identita' sarebbe possibile. Con l'identita' il salto sommerebbe l'EMG grezzo, non
  normalizzato, al ramo normalizzato dalle BN. La rete perderebbe cosi' l'invarianza alla scala globale dell'ingresso (10^k), che la rete
  senza salti ha: le sue convoluzioni senza bias sono seguite da BN. Un test CPU verifica l'invarianza per le due reti.
- **Allineamento:** il centro temporale del salto coincide con quello del blocco nel blocco 1. Nei blocchi 2 e 3 se ne scosta di 0,5 e 1,5
  campioni del rispettivo ingresso, cioe' 1 e 12 campioni a 2 kHz, contro un campo ricettivo di 378 campioni.
- **Dimensioni:** 74.452 parametri con 20 uscite (+3.040, +4,3%); MAC +57.344 (+2,4%); campo ricettivo invariato (378).
- **Inizializzazione appaiata:** i rami del salto si creano per ultimi. Col seme 0 i pesi in comune hanno quindi gli stessi valori iniziali
  della L1, e l'ordine dei dati e' lo stesso.
- **Default invariato** (`residual=False`): identico al codice dell'11/11. L'ho verificato bit per bit contro `ee118a25`, con chiavi e valori
  dello stato per i semi 0, 5 e 777, a 5 e 20 uscite, e col forward in training.

Codice: `src/wearusfm/model/temponet.py` (`residual`); `scripts/temponet_emg2pose.py` (`--arch plain|residual`, scritta in `run.json`;
la valutazione la rilegge e ricarica il checkpoint con la stessa rete); `scripts/slurm/temponet_emg2pose.sbatch` (`ARCH`, `TRAIN_TIME`).

## Ricetta e valutazione: identiche alla L1

- **Training:** L1 sul bersaglio standardizzato (`pose_stats.json`), Adam 1e-4 costante, batch 64, 19 epoche in float32, seme 0.
  - Finestre a passo 8 blocchi, V13 (padding simmetrico).
  - cuDNN benchmark, CUDA graph con controllo eager/graph all'inizio.
  - Si usa il modello finale.
- **Valutazione:** finestre e bersagli ufficiali del passo 0 dell'11/11 (manifest `eval_targets_20261111`).
  - EMA con alpha per giunto scelto sulla validazione.
  - Cancello < 0,9 sulla validazione, poi il test una volta.

## Tempo e risorse

- **Riferimento:** 1,085 ms per passo; 4.040 s di parte fissa (caricamento e indice); 370 s per epoca; 3 h 04 min in tutto.
- **Il salto costa tempo.** Aggiunge kernel a ogni passo: pooling, 1x1, BN, somma e i loro gradienti. In piu' l'ottimizzatore passa da 35 a
  44 tensori. Stimo il passo il 15-20% piu' lento; la misura arrivera' dal log.
- **Con 3:30 non basta.** Il limite di 3:30 non fa partire epoche dopo 12.000 s. A 1,27 ms per passo le 19 epoche finirebbero con 184 s di
  margine, meno della variabilita' del caricamento.
- **Training:** limite **4:00**, nessuna epoca nuova dopo 13.500 s (900 s di margine, come nello studio delle loss). Le 19 epoche ci stanno
  fino a circa 1,55 ms per passo.
- **Nessun collaudo su dbg.** Il controllo eager/graph (tolleranza 1e-5) gira all'inizio del training come l'11/11; se fallisce, il job si
  ferma prima del training.
- **Valutazione:** 45 min, in afterok.
- **Costo massimo: 4,0 + 0,75 = 4,75 GPU-ore** (38 ore locali con billing 8). Resta sotto le 5 della regola del 28/10 e dentro l'involucro
  della consegna (3:30 + 0:45 + collaudo).
- **Checkout di campagna:** CPU, 1 core per al massimo 10 minuti.
- **Nessun rilancio,** nessun altro seme, nessun braccio in piu'.

## Regola di lettura (congelata in `docs/decisioni.md` prima del lancio)

Per ogni condizione di test (`user`, `stage`, `user_stage`) si calcola Δ = MAE residua − MAE L1, con EMA e media fra utenti.

- |Δ| ≤ 0,3 gradi: **non distinguibile con un seme**.
- Δ < −0,3: **segnale di miglioramento**, da confermare con piu' semi.
- Δ > +0,3: **segnale di peggioramento**, da confermare con piu' semi.

Altre regole:

- **Cosa si riporta:** le tre condizioni con dev. std. fra utenti, il cancello, la validazione e, come descrittivi, l'uscita grezza e la
  curva di validazione per epoca.
- **Un verdetto per condizione,** nessun verdetto complessivo.
- **Nessuna scelta dopo il test.**
- **Cancello non superato** (≥ 0,9): il test non si guarda e si apre una diagnosi.
- **Training incompleto:** non si legge niente.

La soglia di 0,3 e' una convenzione comune allo studio delle loss, non un test statistico. Su emg2pose il rumore di un seme non e' misurato.
Su DB8 lo stesso seme rifatto differiva di 0,076 gradi per il solo cuDNN; tre semi stavano entro 0,12, con deviazione standard 0,045
(`docs/temponet_db8_h5_plan.md`, r. 39-40; `docs/temponet_db8_replication.md`, r. 179).

## Job

| ID logico | cosa | risorse |
|---|---|---|
| `TEMPONET-E2P-RES-checkout-20261118` | checkout di campagna congelato del commit del piano | CPU, 10 min |
| `TEMPONET-E2P-RES-train-20261118` | training, `ARCH=residual`, `TRAIN_TIME=04:00:00` | 1 GPU, 4 h, 8 CPU, 120 GB |
| `TEMPONET-E2P-RES-eval-20261118` | valutazione in afterok, `ARCH=residual` | 1 GPU, 45 min |

Uscite: `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/temponet_emg2pose/residual_20261118/{train,eval}`. Ricevute da copiare sul Mac in
`results/temponet_emg2pose/residual_20261118/`.

## Fonti

- [Z20], [Z21]: vedi la docstring di `src/wearusfm/model/temponet.py`.
- [H16] K. He, X. Zhang, S. Ren, J. Sun, «Deep Residual Learning for Image Recognition», CVPR 2016.
- [H19] T. He, Z. Zhang, H. Zhang, Z. Zhang, J. Xie, M. Li, «Bag of Tricks for Image Classification with Convolutional Neural Networks»,
  CVPR 2019.
