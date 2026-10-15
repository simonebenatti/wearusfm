# Calibrazione C1-C4 della finestra 1: pianificatore e preventivo (Claude, 15/10/2026)

Accordo GPT-Claude del 15/10/2026 (`docs/accordo_operativo_finestra1_20261015.md`, sezione «Calibrazione»): GPT protocollo, ammissibilita' e
selezione (congelate prima di C1, fold interni alle partizioni train delle sonde P1, nessun test ufficiale); Claude planner e costi. Questo
documento non autorizza lanci.

## Pianificatore: `scripts/calibration_plan.py`

Dal rapporto dei manifest derivati (`results/infra/derived_manifests_v1_1_build_20261015.json`, derivato al 100%) e dal file delle decisioni
di GPT: configurazioni, passi, warmup, decay, checkpoint, memoria, tempo, GPU-ora attese e tetto, e i piani per `scripts/wsd_chain.py`
(stessi controlli di D14 col codice del trainer; comandi di `submit_job.py` con `--lr` e `--nested 1`). I finalisti sono dipendenze di
decisione esplicite: senza la decisione che la precede una fase e' bloccata col motivo; finalisti fuori griglia o non distinti sono
rifiutati; una conferma fallita in C2 ferma C3 e C4. Il pianificatore non sceglie mai un LR dalla training loss. Aggiunto `--lr` a
`scripts/train_jepa.py`. Test: `tests/cpu/test_calibration_plan.py`.

**Regole applicate.** Obiettivo A (JEPA target b + ancore 0,2; keep e rami spenti), batch globale 32, un nodo 4 GPU, prefetch, regole della
finestra 1, manifest derivato al 100%. Passi = floor(epoche x finestre per epoca / 32), un solo floor. Warmup riferito a D = 4 epoche:
98.094 passi a D -> **1.962** (`max(1000, 2%)`), uguale in tutte le fasi; decay sul 20% finale di ciascun endpoint. Stesso seme entro
taglia: inizializzazione e stream accoppiati fra i candidati. Un job per run, dimensionato con s/passo x 1,15 piu' avvio e margine, senza
scorta.

## Preventivo (s/passo misurati col prefetch: 30M 0,70, 100M 1,39, 300M 2,77; memoria w1gd)

| Fase | Run | Passi | Warmup / decay da | Memoria per GPU | Ore per run | Job | GPU-ora attese / tetto |
|---|---|---|---|---|---|---|---|
| C1 | 30M, LR 1e-4 / 3e-4 / 1e-3, seme 0 | 24.523 | 1.962 / 19.618 | 10,0 GiB | 4,77 | 3 x 6 h 30 | 60,3 / 78,0 |
| C2 | 30M, i due finalisti di C1, seme 1 | 24.523 | 1.962 / 19.618 | 10,0 GiB | 4,77 | 2 x 6 h 30 | 40,2 / 52,0 |
| C3 | 100M, i due finalisti confermati, seme 0 | 12.261 | 1.962 / 9.809 | 17,4 GiB | 4,73 | 2 x 6 h 30 | 39,8 / 52,0 |
| C4 | 300M, il LR proposto, seme 0 | 6.130 | 1.962 / 4.904 | 28,7 GiB | 4,72 | 1 x 6 h 30 | 19,9 / 26,0 |
| **Totale** | 8 run | | | | | | **160,2 / 208,0** |

Solo training **152,1 GPU-ora** (l'ordine di grandezza dell'accordo); con avvio e valutazioni del trainer **~160 attese, tetto 208**
(1.664 ore locali). Esclusi: estrazioni, sonde e selezione dei finalisti (GPT), collaudi. Voce della finestra 1, non del residuo
infrastruttura. Tempo di calendario: C1 in parallelo ~5 h, poi le decisioni di GPT, C2 ~5 h, C3 ~5 h, C4 ~5 h: con le attese delle
decisioni e delle code, ~2-3 giorni prima della campagna (con 8 nodi C1 entra in un'ondata sola).

Uso: `python scripts/calibration_plan.py results/infra/derived_manifests_v1_1_build_20261015.json [--decisions decisioni.json]
[--campaign campagna.json] [--out cartella]`; senza `--campaign` e' un dry-run con percorsi segnaposto.
