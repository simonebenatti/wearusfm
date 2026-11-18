# TEMPONet su emg2pose: training lungo (proposta, 18/11/2026)

**Stato:** APPROVATA da Simone il 18/11 («Approvo B»), regole congelate in `docs/decisioni.md` (voce del 18/11). Codice: `EPOCHS`,
limiti 13:00 e 15:00 e `CHECKPOINT` nello sbatch, `--checkpoint` nella valutazione, con test CPU. Critico APPROVATO; inviati il 18/11 alle 23:56 dal codice `32c5123`.

## Domanda

- Simone, 18/11/2026: «19 epoche di training? non ono poxhine? soliamernte si fanno 100 almeno».
- Alla domanda «Preparo una proposta con un training piu' lungo?» ha risposto «Sì, preparala».

## Cosa sappiamo

- **Da dove vengono le 19 epoche:** sono la ricetta di [Z21] (L1, Adam 1e-4, batch 64, 19 epoche), seguita su DB8 per la replica e tenuta su
  emg2pose per non cambiarla.
- **Su emg2pose un'epoca e' grande:** 21,0 milioni di finestre, cioe' 328.746 passi. Le 19 epoche fanno 6,2 milioni di aggiornamenti.
  - Finestre di 128 ms ogni 40 ms: ogni istante del segnale cade in circa 3 finestre per epoca.
- **Curve della L1 (11/11):**
  - la loss di training scende a ogni epoca, da 0,734 a 0,707, ma nell'ultima solo da 0,7076 a 0,7074;
  - la validazione grezza resta fra 15,63 e 15,71 gradi nelle ultime dieci epoche, tranne l'epoca 18 (16,01).

  La variante residua del 18/11 ha curve dello stesso tipo.
- **Su DB8:** due schedule del learning rate hanno dato 0,06 e 0,10 gradi in validazione, sotto la soglia di 0,20.
- **Non misurato:** che cosa fa un training piu' lungo su emg2pose. Dalle curve mi aspetto poco con il learning rate costante, ma e' una
  lettura, non una misura.

## Proposta

Un solo fattore: il numero di epoche, **100 invece di 19**. Tutto il resto resta come nella L1: L1 sul bersaglio standardizzato, Adam 1e-4
costante, batch 64, seme 0, passo 8 blocchi, V13, CUDA graph col controllo eager/graph. Stessa valutazione: EMA per giunto, cancello < 0,9.

- **Un run per architettura.** Il learning rate e' costante, quindi le prime 19 e le prime 50 epoche di un run da 100 seguono la stessa
  ricetta dei run piu' corti. Non sono identiche bit per bit, per cuDNN in modo benchmark.
  - Il training salva gia' un checkpoint e la validazione grezza a ogni epoca: la curva mostra dove si ferma il miglioramento.
- **Test di tre checkpoint dichiarati prima:** epoche 19, 50 e 100 dello stesso run, ciascuno guardato una volta e tutti riportati, senza
  sceglierne uno.
  - Il confronto principale e' fra l'epoca 100 e l'epoca 19 dello stesso run. Toglie la differenza fra run diversi, che su DB8 era di
    0,076 gradi per il solo cuDNN.
  - Il cancello vale per ciascun checkpoint: chi non lo supera non va al test.
- **Regola di lettura da congelare,** per condizione (`user`, `stage`, `user_stage`), con Δ = MAE all'epoca 100 − MAE all'epoca 19:
  - |Δ| ≤ 0,3 gradi: le 19 epoche bastano, nel senso di «non distinguibile con un seme»;
  - Δ < −0,3: segnale che il training lungo migliora, da confermare con piu' semi;
  - Δ > +0,3: segnale che peggiora (sovra-adattamento), da confermare.

  Descrittivi: l'epoca 50; l'epoca 19 del run lungo contro la L1 dell'11/11, dove mi aspetto una differenza piccola; la curva di
  validazione.

## Codice da aggiungere (dopo l'approvazione, con test CPU e critico)

- **sbatch:** `EPOCHS` (default 19, invariato) e due limiti nella tabella di `TRAIN_TIME`, 13:00:00 e 15:00:00, con 900 s di margine.
- **Valutazione:** `--checkpoint epoch_NN.pt`, per valutare un checkpoint dichiarato invece del modello finale; scritto in `eval.json`.
- Il resto dello script non cambia.

## Opzioni, tempi e costo

Stime dalle ricevute: parte fissa circa 4.000 s; 370 s per epoca per la rete senza salti, 426 s per la residua; valutazione 12-19 minuti.
Il costo massimo e' calcolato sui limiti richiesti.

| Opzione | job | durata attesa | costo massimo |
|---|---|---|---|
| **A** (consigliata) | rete senza salti, 100 epoche (limite 13:00) + 3 valutazioni (45 min l'una) | 11,4 h + ~1 h | **15,25 GPU-ore** (122 ore locali) |
| **B** | A + residua, 100 epoche (limite 15:00) + 3 valutazioni | +12,9 h + ~1 h | **32,5 GPU-ore** (260 ore locali) |

- **Perche' A:** risponde alla domanda per la rete di riferimento, che e' anche la base dello studio delle loss. A 19 epoche la residua non
  si distingue dalla L1; B serve solo se vuoi sapere se la residua guadagna di piu' da un training lungo.
- **Budget del progetto:** restano 658.634 ore locali su 2.000.000 (`saldo -b`, 18/11 alle 23:26). Entrambe le opzioni superano la
  soglia di 5 GPU-ore della regola del 28/10, quindi servono un tetto approvato e una voce in `docs/decisioni.md`.
- **Calendario:** Leonardo e' fermo dal 23/11 alle 08:00 al 25/11 alle 08:00, e un job da 13-15 ore deve finire prima. Per stare nei tempi
  il codice va rivisto e il job inviato entro il 21/11.
- **Nessun rilancio:** un run incompleto non si legge e non si rilancia senza diagnosi.

## Decisioni per Simone

1. **Opzione:** A o B.
2. **Test:** i tre checkpoint 19, 50 e 100, oppure solo il modello finale (un solo sguardo, ma senza il confronto dentro lo stesso run).
3. **Tetto:** 15,25 GPU-ore per A, 32,5 per B.

## Esecuzione (opzione B)

| ID logico | cosa | risorse |
|---|---|---|
| `TEMPONET-E2P-LONG-checkout-20261118` = 64790540 | checkout `temponet-emg2pose-long-32c5123` | CPU, 10 min; `COMPLETED 0:0` |
| `TEMPONET-E2P-LONG-train-plain-20261118` = 64790588 | 100 epoche, `ARCH=plain`, `EPOCHS=100`, `TRAIN_TIME=13:00:00` | 1 GPU, 13 h |
| `TEMPONET-E2P-LONG-train-res-20261118` = 64790605 | 100 epoche, `ARCH=residual`, `EPOCHS=100`, `TRAIN_TIME=15:00:00` | 1 GPU, 15 h |
| `TEMPONET-E2P-LONG-eval-{plain,res}-e{19,50,100}-20261118` | sei valutazioni in afterok (`CHECKPOINT=epoch_19.pt`, `epoch_50.pt`; modello finale) | 1 GPU, 45 min l'una |

Uscite in `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/temponet_emg2pose/long_20261118/{plain,residual}/{train,eval_e19,eval_e50,eval_e100}`.

Valutazioni (afterok, 45 min): senza salti 64790591 (e19), 64790603 (e50), 64790604 (e100); residua 64790608 (e19), 64790609 (e50),
64790610 (e100). Inviate il 18/11 alle 23:56 dal codice `32c5123`.
