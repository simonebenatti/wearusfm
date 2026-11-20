# TEMPONet su emg2pose: conferma con altri semi del confronto L1 / MSE (proposta, 18/11/2026)

Richiesta di Simone, 18/11/2026: alla domanda «Preparo la proposta di conferma con altri semi (L1 contro MSE)?» ha risposto «Sì,
preparala». Alla domanda «Approvi la conferma con i semi 1 e 2, circa 13 GPU-ore?» ha risposto «Sì, approvo».

La versione approvata e' `d0bf277`. Prima del primo job la review del critico (workflow `wf_27ae4d80-d4f`, «DA RIVEDERE») ha portato a
correggere la regola di lettura e la dichiarazione dei costi. I job e il costo non cambiano. Le correzioni sono qui sotto e in
`docs/decisioni.md`.

## Da dove parte

Studio delle loss del 18/11 (`docs/temponet_emg2pose_loss_plan_20261118.md`; esito e consuntivo in `docs/decisioni.md`). Con il seme 0
l'MSE e' peggiore della L1 di +0,50 / +0,47 / +0,55 gradi (user / stage / user_stage, test con EMA, media fra utenti), oltre la soglia di
0,30. La lettura congelata dice «da confermare con altri semi». La soglia stessa era un'ipotesi: la dispersione fra semi su emg2pose non
e' mai stata misurata.

## Domande

1. La differenza MSE - L1 regge cambiando il seme?
2. Quanto varia la L1 fra semi su emg2pose? Con tre semi, cioe' due gradi di liberta', si ottiene solo un ordine di grandezza: non basta a
   tarare la soglia degli studi futuri.

## Bracci

L1 e MSE con i semi 1 e 2. Il seme 0 c'e' gia' per entrambi: A (64169771 / 64169772) e C (64776581 / 64776582). In tutto 4 training e
4 valutazioni.

Il seme fissa inizializzazione, ordine delle finestre e maschere del dropout. A parita' di seme L1 e MSE partono dagli stessi pesi e
vedono le finestre nello stesso ordine; la funzione di loss non usa numeri casuali. Il confronto si fa quindi **appaiato per seme**:
Δ_s = MSE_s - L1_s.

Ricetta identica ad A e C: stessi dati, finestre ogni 8 blocchi, V13, Adam 1e-4 costante, batch 64, 19 epoche, CUDA graph con controllo
eager/graph, stessa valutazione (EMA, cancello < 0,9, test una volta). Cambia solo `--seed`.

## Codice

Lo sbatch passa `SEED` (intero) a `--seed` nel modo train. I job partono dal checkout `temponet-emg2pose-seeds-d0bf277`, con `SEED`
esplicito per ogni job. In quel checkout una `SEED` vuota diventerebbe seme 0; dopo la review, nel ramo, una `SEED` vuota e' un errore.

La raccolta controlla il seme con `scripts/temponet_emg2pose_seeds_compare.py`: legge il `run.json` del training, perche' `args.seed`
dell'`eval.json` e' quello della chiamata di valutazione, e rifiuta seme, loss o checkpoint diversi da quelli attesi. Nessun collaudo:
lo script del training non cambia.

## Lettura (congelata in `docs/decisioni.md` prima del primo job)

Per ciascuna condizione (user, stage, user_stage), Δ_s viene dai `mean_over_users_deg` non arrotondati degli `eval.json`. Contano **solo i
semi nuovi 1 e 2**:
- **MSE peggiore di L1, confermato**: Δ_1 > 0, Δ_2 > 0 e media dei due > +0,30 gradi;
- **MSE migliore di L1, confermato**: Δ_1 < 0, Δ_2 < 0 e media dei due < -0,30;
- **non leggibile**, se a uno dei quattro bracci nuovi manca il test (cancello non superato, run incompleto, guasto);
- altrimenti **non confermato**.

Il seme 0 non entra nella regola, perche' e' il risultato da confermare.

Probabilita' della regola, in un modello normale con Δ_1 e Δ_2 indipendenti e deviazione fra semi σ. Per σ il valore plausibile e'
0,10-0,30 gradi: su DB8 la deviazione fra semi era 0,07, e la differenza fra due training vale circa 0,10.

| σ | falsa conferma (Δ vero 0), per direzione | conferma con Δ vero +0,50 | con Δ vero +0,30 |
|---|---|---|---|
| 0,10 | 0,002% | 99,8% | 50% |
| 0,20 | 1,7% | 92% | 50% |
| 0,30 | 7,2% | 80% | 48% |

Descrittivi, senza soglia:
- Δ_0 e media dei tre Δ_s;
- media e deviazione fra i tre semi per L1 e MSE, in ogni condizione;
- deviazione dei Δ_s;
- utenti migliori e peggiori per seme.

Un braccio che non passa il cancello si riporta come tale, senza test e senza nuovo tentativo.

**Baseline firmata.** La baseline dell'harness resta quella firmata il 12/11: la misura del seme 0, con protocollo e seme unico
dichiarati. Sostituirla con una media su piu' semi richiede una nuova firma di Simone.

## Costi e risorse

Tempi trascorsi il 18/11:
- C (MSE): training 2:52:37 e valutazione 0:21:45, cioe' 3,24 ore;
- B (smooth L1, stessa velocita' per passo della L1 nel collaudo): 2:53:31 + 0:21:00, cioe' 3,24 ore;
- A (L1) l'11/11: 11.074 s di training (tempo interno di `run.json`), con un caricamento piu' lento.

| Voce | GPU-ore | Ore locali (fattore 8) |
|---|---|---|
| Attesi: 4 x circa 3,25-3,35 | circa 13-13,4 | circa 104-107 |
| Esposizione massima con i limiti: training 4:30 e valutazione 0:45, 4 x 5,25 | 21 | 168 |

- **Tetto dell'attivita': 14 GPU-ore spese e 1 core-ora.** Nessun nuovo invio se un job fallisce: si torna da Simone.
- **Checkout** `TEMPONET-E2P-seeds-checkout-20261118` = 64789455: `lrd_all_serial`, 1 CPU, 2 GB, limite 10 minuti (al massimo 0,17
  core-ore). `COMPLETED 0:0` in 41 s, circa 0,01 core-ore.
- **Saldo** letto con `saldo -b` alle 23:59 del 18/11: 1.341.366 ore locali consumate su 2.000.000, residuo 658.634. Il saldo non conta
  ancora i job del 18/11: lo studio delle loss ha speso circa 84 ore locali.
- **Calendario**: i quattro training in parallelo, una GPU ciascuno; circa 3,5 ore se partono subito. Prima della manutenzione totale del
  23/11.

## Fuori dalla proposta (aggiunte possibili)

- Smooth L1 con i semi 1 e 2: + circa 6,5 GPU-ore. Con il seme 0 non era distinguibile dalla L1.
- MSE + velocita': non serve, con il seme 0 era gia' nettamente peggiore (+0,85-1,09).
- TEMPONet residua con i semi 1 e 2: + circa 7 GPU-ore. Con il seme 0 non era distinguibile dalla L1 (sessione `temponet-residual`).
