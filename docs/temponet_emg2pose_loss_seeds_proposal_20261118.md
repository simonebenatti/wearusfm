# TEMPONet su emg2pose: conferma con altri semi del confronto L1 / MSE (proposta, 18/11/2026)

Richiesta di Simone, 18/11/2026: alla domanda «Preparo la proposta di conferma con altri semi (L1 contro MSE)?» ha risposto «Sì,
preparala». Proposta per la sua firma: nessun job finche' non la approva.

## Da dove parte

Studio delle loss del 18/11 (`docs/temponet_emg2pose_loss_plan_20261118.md`; esito e consuntivo in `docs/decisioni.md`): con il seme 0
l'MSE e' peggiore della L1 di +0,50 / +0,47 / +0,55 gradi (user / stage / user_stage, test con EMA, media fra utenti), oltre la soglia di
0,30. La lettura congelata dice «da confermare con altri semi». La soglia stessa era un'ipotesi: la dispersione fra semi su emg2pose non
e' mai stata misurata.

## Domande

1. La differenza MSE - L1 regge cambiando il seme?
2. Quanto varia la L1 fra semi su emg2pose? Il numero serve a tarare la soglia degli studi futuri; non cambia la lettura di questo.

## Bracci

L1 e MSE con i semi 1 e 2. Il seme 0 c'e' gia' per entrambi: A (64169771 / 64169772) e C (64776581 / 64776582). In tutto 4 training e
4 valutazioni.

Il seme fissa inizializzazione, ordine delle finestre e maschere del dropout. A parita' di seme L1 e MSE partono dagli stessi pesi e
vedono le finestre nello stesso ordine; la funzione di loss non usa numeri casuali. Il confronto si fa quindi **appaiato per seme**:
Δ_s = MSE_s - L1_s.

Ricetta identica ad A e C: stessi dati, finestre ogni 8 blocchi, V13, Adam 1e-4 costante, batch 64, 19 epoche, CUDA graph con controllo
eager/graph, stessa valutazione (EMA, cancello < 0,9, test una volta). Cambia solo `--seed`.

## Codice

Una riga nello sbatch: `SEED` (intero, default 0) passato a `--seed` nel modo train. Con `SEED` non impostato il comando equivale a quello
di oggi. Nessun collaudo: lo script non cambia; `run.json` registra il seme usato e la raccolta lo controlla.

## Lettura (da congelare in `docs/decisioni.md` prima del primo job)

Per ciascuna condizione (user, stage, user_stage), con Δ_s dai `mean_over_users_deg` non arrotondati degli `eval.json`:
- **MSE peggiore di L1, confermato**: Δ_s > 0 per tutti e tre i semi e media dei tre Δ_s > +0,30 gradi;
- **MSE migliore di L1, confermato**: Δ_s < 0 per tutti e tre i semi e media < -0,30;
- altrimenti **non confermato**.

Senza differenza vera, il segno uguale nei tre semi capita una volta su quattro nell'una o nell'altra direzione (1/8 per direzione). La
soglia sulla media scarta le differenze piccole.

Descrittivi, senza soglia:
- media e deviazione fra i tre semi per L1 e MSE in ogni condizione;
- deviazione dei Δ_s;
- utenti migliori e peggiori per seme;
- confronto della deviazione fra semi della L1 con la soglia di 0,30.

Un braccio che non passa il cancello si riporta come tale, senza test e senza nuovo tentativo. Fallimenti tecnici: nessun nuovo invio
senza Simone.

## Costi

Tempi osservati il 18/11 per L1 e MSE (B e C): training circa 10.340-10.390 s, valutazione circa 1.240-1.290 s. L'11/11 A aveva impiegato
11.074 s, con un caricamento piu' lento.
- Attesi: 4 x (circa 3,0 + 0,36) = **circa 13 GPU-ore**, circa 105 ore locali (fattore 8).
- Limiti: training 4:30 e valutazione 0:45, come B e C; esposizione massima 4 x 5,25 = 21 GPU-ore.
- Checkout CPU su `lrd_all_serial`: 0 GPU-ore.
- Saldo del progetto letto alle 18:55 del 18/11: 658.634 ore locali residue.
- Calendario: i quattro training in parallelo, una GPU ciascuno; circa 3,5 ore se partono subito. Prima della manutenzione totale del
  23/11.

## Fuori dalla proposta (aggiunte possibili)

- Smooth L1 con i semi 1 e 2: + circa 6,5 GPU-ore. Con il seme 0 non era distinguibile dalla L1.
- MSE + velocita': non serve, con il seme 0 era gia' nettamente peggiore (+0,85-1,09).
- TEMPONet residua con i semi 1 e 2: + circa 7 GPU-ore. Con il seme 0 non era distinguibile dalla L1 (sessione `temponet-residual`).
