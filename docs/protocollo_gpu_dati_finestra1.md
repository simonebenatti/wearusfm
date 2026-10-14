# Prova GPU e dati per la finestra 1: equivalenza 1–4 GPU (NCCL, bf16), attesa sui dati, throughput dei rung (protocollo `w1_gpu_dati_v1`)

Proposta di Claude del 14/10/2026, punto 6 della numerazione unica (registro condiviso, `c4a0302`: «Protocollo ed esecuzione equivalenza
NCCL/bf16 e attesa dati <=2%; memoria/throughput/LR-batch, costi e launcher»). Simone: «protocollo GPU/dati (6)». **Da approvare; nessun job,
codice dell'equivalenza da scrivere dopo l'accordo.** Criteri fissati qui prima di qualunque misura.

## Cosa prova e cosa no

Prova: (A) che il training DDP su 4 GPU con NCCL, in float32 e in bf16, calcola le stesse loss, gli stessi gradienti ridotti, gli stessi pesi
dopo AdamW e lo stesso teacher dopo l'EMA di un processo solo sullo stesso batch globale, e che l'accumulo dei gradienti pure; (B) che con le
regole della finestra 1 il passo non aspetta i dati per piu' del 2% del suo tempo (regola di D6a); (C) i secondi per passo e la memoria dei rung
30M, 100M e 300M in DDP a 4 GPU con i dati veri, per la fattibilita' dei job (`docs/finestra1_fattibilita.md`).
Non prova: la convergenza dei run lunghi, la scelta di learning rate e batch (calibrazione, protocollo a parte), FSDP o piu' nodi (fuori dalla
finestra 1), RVQ e keep in DDP (spenti nel default).

Gia' provato: sulla CPU, DDP `gloo` a 2 e 4 processi e accumulo in float64 uguali al batch intero entro 1e-10 (`docs/contratto_loss_distribuite.md`
§5, anche coi rami); collaudo GPU del 12/10 funzionale (DDP a 4 GPU, 300 passi e ripresa, memoria dei rung). Il contratto rimandava NCCL/bf16 a
«una conferma su Leonardo [...] tolleranza da fissare dopo il rumore del bf16».

## (A) Equivalenza numerica su GPU

Il DDP da' a ogni processo il suo stream di batch (`training.run.BatchStream`): fra 1 e 4 GPU i run non vedono gli stessi dati, quindi non si
confrontano due training. Si confrontano **gli stessi batch globali** calcolati in modi diversi, come nel test CPU.
- **Script nuovo** `scripts/ddp_equivalence.py` (torchrun, 4 processi, un nodo): 8 batch globali FISSI da 32 finestre (regole della finestra 1,
  tutti i dataset, montaggi virtuali, generatore proprio), costruiti uguali su ogni processo; modello 30M (preset firmato, ancora multiscala,
  RVQ e keep spenti) dal seme 0. Per ciascuno degli 8 aggiornamenti, sugli stessi batch e dallo stesso stato:
  1. riferimento: un processo, batch intero (32), senza accumulo, sulla GPU 0;
  2. un processo con 4 micro-batch da 8 (accumulo);
  3. DDP 4 x 8;
  4. DDP 4 x (2 micro-batch da 4).
  In due precisioni: **float32 con TF32 spento** (matmul e cuDNN) e **bf16** (autocast del trainer). Piu' una ripetizione del riferimento bf16
  sullo stesso processo (rumore del bf16 fra due calcoli identici, descrittivo).
- **Quantita' confrontate**, per aggiornamento: loss totale e per componente; gradiente dopo la riduzione (norma L2 relativa della differenza,
  coseno); pesi dopo clipping e AdamW; teacher dopo l'EMA (norma relativa della differenza).
- **Criteri (congelati):**
  - float32: ogni differenza relativa fra 2, 3, 4 e il riferimento <= 1e-5 in tutti gli 8 aggiornamenti (restano solo l'ordine delle riduzioni
    NCCL e i kernel non deterministici);
  - bf16: per ogni quantita' e aggiornamento, ||variante bf16 - riferimento bf16|| <= ||riferimento bf16 - riferimento float32|| (il DDP e
    l'accumulo non aggiungono piu' errore del bf16 stesso), e loss totale entro l'1% relativo;
  - **superata** se valgono tutti; altrimenti si ferma la finestra 1 su DDP finche' non e' capita la differenza (nessuna tolleranza allargata
    dopo la misura).
- Test CPU dello script prima del lancio: `gloo` a 2 processi in float64 entro 1e-10 (stesso confronto del contratto).

## (B) Attesa sui dati in DDP

- `scripts/slurm/train_ddp.sbatch`, preset 30M (il rung piu' veloce, quindi il caso peggiore per l'attesa), regole della finestra 1, tutti i
  dataset, 6 worker per GPU, memoria condivisa fra worker e processo (`49509a2`, mai misurata su GPU), 600 aggiornamenti, cartella nuova.
  Due bracci: batch globale 32 (8 per GPU, il sanity) e 64 (16 per GPU, candidato della calibrazione).
- **Misura:** frazione d'attesa = somma di `t_data_s` (il processo piu' lento) / somma di `t_step_s`, sugli aggiornamenti 51-600.
- **Criterio (D6a, congelato):** <= 2% in tutti e due i bracci -> superata; montaggi virtuali accesi nel default come da D15/D6a. Altrimenti:
  profilo per fasi (`measure_loader.sbatch`) e proposta (piu' worker, cache), da decidere prima dei run; i montaggi virtuali non si accendono
  in automatico.
- Riferimento: collaudo del 12/10, 7,2% (4,1% senza il primo passo dopo la ripresa), mediana 25 ms per passo di sola ricezione; misura locale
  dopo la memoria condivisa ~6 ms.

## (C) Throughput e memoria dei rung

- Stesso launcher (`PRESET=100m`, `PRESET=300m`), regole della finestra 1, tutti i dataset, 300 aggiornamenti (200 al 300M con batch 64):
  100M e 300M, batch globale 32 e 64 (al 300M il 64 con 2 micro-batch da 8 per GPU: la memoria a 16 finestre per GPU non e' stata misurata).
  Si registrano: secondi per passo (mediana dopo il 50), memoria massima per GPU, attesa sui dati. Descrittivo, nessun criterio: alimenta i
  costi.

## Job, risorse e costo

| ID logico | Cosa | Risorse | Tempo massimo | Stima |
|---|---|---|---|---|
| `w1gd_checkout` | checkout di campagna | `lrd_all_serial`, 1 CPU | 10 min | 0,17 core-ora |
| `w1gd_equiv` | (A), 4 varianti x 2 precisioni x 8 aggiornamenti | 1 nodo, 4 GPU, `boost_qos_dbg` | 30 min | ~10-15 min |
| `w1gd_data_b32`, `w1gd_data_b64` | (B), 30M | 1 nodo, 4 GPU, dbg | 30 min ciascuno | ~12 min ciascuno |
| `w1gd_tp_100m_b32`, `_b64`, `w1gd_tp_300m_b32`, `_b64` | (C) | 1 nodo, 4 GPU, dbg | 30 min ciascuno | 10-25 min |

**Tetto: 7 job da 4 GPU x 30 min = 14 GPU-ora (112 ore locali)**, stima ~8 GPU-ora; dal budget «preparazione dell'infrastruttura della
finestra 1» (25 GPU-ora, usate 0,36). dbg ammette 2 job alla volta: si sottomettono a coppie.

## Codice da scrivere (dopo l'accordo)

`scripts/ddp_equivalence.py` (riusa `_accumulate_terms`, `_ddp_wrap` e la costruzione del trainer; nessuna modifica a `train()`), il suo test
CPU `gloo`, e il launcher `scripts/slurm/ddp_equivalence.sbatch`. Per (B) e (C) bastano il trainer e `train_ddp.sbatch` esistenti.
