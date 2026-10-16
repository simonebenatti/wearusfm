# Calibrazione C1-C4 della finestra 1: pianificatore e preventivo (Claude, 15/10/2026)

Accordo GPT-Claude del 15/10/2026 (`docs/accordo_operativo_finestra1_20261015.md`, sezione «Calibrazione»): GPT protocollo, ammissibilita' e
selezione (congelate prima di C1, fold interni alle partizioni train delle sonde P1, nessun test ufficiale); Claude planner e costi. Questo
documento non autorizza lanci.

## Pianificatore: `scripts/calibration_plan.py`

Dal rapporto dei manifest derivati costruiti sul cluster (`results/infra/derived_manifests_v1_1_build_cluster_20261015.json`, derivato al
100%; quello della costruzione locale ha lo SHA di un file che sul cluster non esiste) e dal file delle decisioni
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

Uso: `python scripts/calibration_plan.py results/infra/derived_manifests_v1_1_build_cluster_20261015.json [--decisions decisioni.json]
[--campaign campagna.json] [--out cartella]`; senza `--campaign` e' un dry-run con percorsi segnaposto.

## Aggiornamento del 15/10/2026: selezione, valutazione e preventivo completo

Con GPT senza token, Simone: «riessci tu a fare i task per C1 C4 ?». Protocollo di ammissibilita' e selezione (proposta da firmare prima di
C1), validation interna e catena di valutazione in `docs/calibrazione_c1_c4_selezione.md`. Il pianificatore ora: accende il pannello fisso
nei run (`--fixed-eval-every 500`, serve al criterio di stabilita' A4); accetta «stop» per C1 e C3 e l'esito di C4; rifiuta un LR per il 300M
che non sia uno dei due di C3; con `--out` e una campagna con "eval" scrive anche `valutazione_<fase>.json` per `scripts/evaluation_chain.py`.
Preventivo con la valutazione: **~163 GPU-ora attese, tetto 214,5** (training 160,2 / 208,0; valutazione 2,6 / 6,5) e CPU fino a ~199
core-ora; C1 da sola ~61 / 79,5.

## Pacchetto di lancio di C1 (16/10/2026; Simone: «Pacchetto di lancio C1»)

Configurazione della campagna con i percorsi veri del cluster: `results/infra/w1cal_campaign_20261016.json`. Piani in dry-run in
`results/infra/w1cal_c1_dryrun_20261016/` (`calibrazione_C1.json`, `valutazione_C1.json`, riepilogo, i 3 comandi di sottomissione del
training e il dry-run della catena di valutazione: tutti i nodi «in attesa» del training, come deve essere). **`expected_sha` e `code_dir`
sono provvisori** (b671901): al lancio si rigenerano allo SHA del deploy.

| Ingresso | Percorso sul cluster | Identita' |
|---|---|---|
| Manifest derivato al 100% | `$WORK/wearusfm_runs/results/passo6/derived_v1_1/manifest_v1_1_nested_1.json.gz` | file 6d6d8167..., contenuto 167923c3... (job 60661475) |
| Durate delle sessioni | `$WORK/wearusfm_runs/results/passo6/window_seconds_59425316.json` | e660e324..., lo stesso della costruzione dei derivati |
| Scale di sessione | `$WORK/wearusfm_runs/results/passo6/loader_59623700/session_scales.json` | come `w1cat` |
| Run | `$SCRATCH/wearusfm_runs/runs/w1cal/<run>` | 3 x 1,4 GB di checkpoint |
| Feature / rapporti | `$SCRATCH/wearusfm_runs/w1cal/features/`, `$WORK/wearusfm_runs/results/w1cal/` | |

Run di C1: `W1CAL-C1-30M-lr{0p0001,0p0003,0p001}-s0`, 24.523 passi, warmup 1.962, decay da 19.618, un job da 6 h 30 per run su QoS `normal`
(4 GPU), argomenti `--window1-rules --datasets all --prefetch --fixed-eval-every 500 --lr <lr> --nested 1`. Preventivo C1: training 60,3
GPU-ora attese / 78,0 tetto, valutazione 0,6 / 1,5, CPU fino a 74 core-ora: **~61 / 79,5 GPU-ora (636 ore locali)**.

**Sequenza al lancio:**
1. Dopo l'esito di `w1_selezione_v1`: aggiornare stime e tetti delle valutazioni con le durate misurate (commit), se serve.
2. Decisioni prima di C1: **budget della finestra 1** (la calibrazione e' una sua voce) e review dei manifest derivati, che il registro segna
   «da rivedere da GPT prima dell'uso nei run» (si puo' affidare al subagent `critico`).
3. Preflight in sola lettura, `deploy/w1cal-<data>` allo SHA del commit, checkout di campagna (job CPU).
4. Rigenerare i piani con `expected_sha`/`code_dir` veri (`calibration_plan.py --campaign ... --out ...`) e committarli.
5. Sottomettere i 3 training (`wsd_chain.py --commands`, comandi via leonardo-ops, uno per riga con `< /dev/null`).
6. Sottomettere la catena di valutazione quando i training sono finiti (seconda chiamata di `evaluation_chain.py --submit`: le guardie partono
   senza afterok e nessun nodo resta in coda con una dipendenza impossibile se un training fallisce).
7. Copiare decisione e registro della selezione sul Mac, `calibration_select.py merge`, commit; C2 solo col via di Simone.

**Primo uso del manifest derivato in un training:** finora l'hanno letto solo i test CPU e i rapporti del costruttore. Un errore di caricamento
si vede nei primi minuti del job (costo minimo); controllare i primi passi dei tre run subito dopo la partenza.
