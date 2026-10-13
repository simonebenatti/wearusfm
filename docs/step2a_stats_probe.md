# Step 2, test preliminare 2A: statistiche spettrali fisse accanto all'embedding (protocollo `step2a_stats_v1`)

Proposta di Claude del 13/10/2026 su richiesta di Simone («parti dal protocollo del test 2A»). **Da approvare: nessun job lanciato.** Soglie e
criteri sono fissati qui e in `docs/decisioni.md` PRIMA di qualunque misura. Codice: `scripts/step2a.py`, `scripts/slurm/step2a_stats.sbatch`,
`scripts/slurm/step2a_probe.sbatch`, `harness/fm_features.py` (`feature_set`), `scripts/probe_p1.py` / `probe_p2.py` (`--feature-set`); test in
`tests/cpu/test_step2a.py`.

## Domanda

Prima di allenare i rami spettrali (step 2B-3), la proposta del 07/10 chiede un controllo senza training: **l'informazione spettrale esplicita
aggiunge qualcosa, sui task P1/P2, all'embedding che abbiamo gia'?** Si confrontano, con le sonde ufficiali, l'embedding (M) e l'embedding con
accanto le 32 statistiche spettrali fisse dello Step 1 (MS). Se MS non migliora su M, il caso a favore di un ramo spettrale appreso e' piu'
debole (non impossibile: un ramo puo' imparare cose che le 32 medie non contengono). Se migliora, conviene provare i rami.

## Ingressi (gia' esistenti, verificati il 13/10 in sola lettura)

- **Embedding:** le cache della tranche `step1_downstream_4000_v1` (`$WORK/wearusfm_runs/results/passo6/step1_downstream_0910`), teacher EMA
  finali **B_seed0 e B_seed1** dell'A/B dello Step 1 (keep 0,05, 4.000 update, solo emg2qwerty; SHA256 dei checkpoint in
  `scripts/step1_downstream.py`). Presenti, commit di valutazione `8dad7e8`, hash dei checkpoint corrispondenti; P1: DB2 33.427/5.117/7.908,
  DB3 8.610/1.224/3.847, DB6 30.036/3.925/8.083 finestre train/val/test; P2: EPN-612 91.800, UCI-EMG 662. Si leggono, non si riscrivono.
  Si usano i teacher B perche' lo step 2 estende la configurazione dello Step 1 con la keep; i teacher A restano fuori (eventuale tranche a parte).
- **Statistiche (S):** per ogni finestra, i 32 target globali dello Step 1 (forma e energia, 50 ms e 500 ms; `global_spectral_targets` sulle
  ancore multiscala) calcolati sullo stesso ingresso delle feature: stesso filtro, stessa scala per soggetto o sessione, stessi ritagli, bordo
  di 0,1 s ai due lati come nel pretraining. Mancanti = 0, piu' 32 indicatori di presenza: 64 colonne. Nessun valore stimato sui dati.
- **Allineamento:** la fase stats registra l'hash degli ingressi realmente letti (`--audit-inputs`). Deve coincidere con quello delle cache
  degli embedding, e devono coincidere etichette, soggetti e ordine di ogni ruolo; altrimenti la catena si ferma senza scrivere varianti.
- **Identita' delle cache (review GPT del 13/10):** ogni cache nuova registra l'insieme di feature e le colonne dell'embedding
  (`harness/feature_sets.py`); la sonda rifiuta una cache il cui insieme o la cui larghezza non e' quello dichiarato. Le cache della tranche
  downstream, senza registrazione, valgono solo come "model".
- **Supporto:** le statistiche sono i target dello Step 1, con la loro regola di bordo (si escludono le patch che toccano la zona di bordo;
  su 1 s, 6 frame lenti). Il ramo PSD dei rami spettrali e' piu' severo (finestra intera fuori dal bordo: 4 frame), ma non entra nel 2A.

## Varianti e sonde

| Variante | Feature | Dove |
|---|---|---|
| M | embedding (768) | cache downstream, per teacher; la sonda rifatta deve riprodurre il report downstream (controllo) |
| MS | embedding + statistiche (832) | nuova, per teacher |
| S | statistiche (64) | nuova, una per task (non dipende dal modello) |
| K | soli indicatori (32) | nuova, una per task: controllo |

Non c'e' M + indicatori (il controllo `raw_plus_mask` del pacchetto): le sonde sono per dataset, e dentro un dataset fs e banda sono le stesse per
tutte le finestre, quindi gli indicatori sono costanti e M + indicatori coinciderebbe con M. La fase stats conta le maschere diverse per dataset
e lo registra; K lo verifica sulle accuratezze (atteso: livello del caso).

Sonde **invariate**, come firmate e come nella tranche downstream: encoder congelato, scaler solo sul train, regressione logistica bilanciata, C in
{0,001; 0,01; 0,1; 1; 10; 100} scelto sulla validation, `max_iter` 3000, seme 0; P1 su NinaPro DB2/DB3/DB6 (finestre da 1 s, split v1 con
validation 10%, pesi 8/13, 3/13, 2/13); P2 su EPN-612 e UCI-EMG (split per soggetto 70/10/20 ai semi 0, 1, 2; media dei tre split, poi dei due
dataset). Il test non sceglie nulla.

## Endpoint, incertezza e criteri (congelati)

Per task (P1, P2): **delta = accuratezza bilanciata(MS) - accuratezza bilanciata(M)**, media sui due teacher, con il confronto accoppiato gia'
usato (`harness.downstream_ab.compare_reports`, braccio «A» = M, «B» = MS): bootstrap accoppiato sui soggetti di test (1.000 estrazioni, seme
20261009); P1 intervallo percentile 95% stratificato per dataset; P2 intervallo descrittivo delta +- 1,96 SE (SE RMS sugli split sovrapposti).

- **Utile in un task:** delta >= +0,01 (1 punto), limite inferiore dell'intervallo > 0, delta > 0 con tutti e due i teacher.
- **Non dannoso in un task:** delta >= -0,01.
- **2A positivo:** utile in almeno un task e non dannoso in tutti e due. Altrimenti **2A negativo**.

Descrittivi, senza soglie: S e K da soli, delta per dataset e per teacher, maschere diverse per dataset, riproduzione del report downstream da
parte della sonda rifatta su M (attesa identica: stesse feature, stesso codice delle sonde).

**Lettura.** Positivo: l'informazione spettrale esplicita e' utile e non gia' contenuta nell'embedding -> si prepara il protocollo dei rami a
encoder congelato (step 2C), con preventivo. Negativo: lo si registra; decidere se provare comunque i rami spetta a Simone. In nessun caso e'
adozione automatica. Limiti: incertezza condizionata ai due teacher; due task senza correzione per confronti multipli; encoder allenati solo su
emg2qwerty (P1 di partenza basso, ~0,18: un guadagno in P1 puo' dire soprattutto che l'encoder trasferisce poco a NinaPro); le 32 statistiche
sono medie della finestra, non dinamica temporale.

## Esecuzione

**Prerequisiti:** review di GPT del codice dei rami e del 2A (tocca suoi file: `fm_features.py`, `probe_p1.py`, `probe_p2.py`); commit su un
riferimento `deploy/step2a-<data>` e checkout di campagna (job CPU di `campaign_checkout.sbatch`, ~1 minuto); controlli in sola lettura subito
prima: cache e provenienze invariate, dati P1 in `$WORK/data/processed` e `$SCRATCH/data/processed`, EPN-612 e UCI-EMG grezzi nei percorsi del
launcher, spazio su `$WORK` (servono ~1,5 GB; liberi 154 GB il 13/10).

Uscite nuove, fuori da Git: `EVAL = $WORK/wearusfm_runs/results/passo6/step2a_<data>`; `RUNS = $SCRATCH/wearusfm_runs/runs/step1_ab_0810`
(checkpoint B_seed0, caricato dalle sonde in estrazione ma non usato per le statistiche); `DOWN` = la cartella downstream sopra. Tutti i job su
`lrd_all_serial`, 8 CPU, 30 GB, nessuna GPU, sottomessi con `scripts/submit_job.py` (CODE_DIR ed EXPECTED_SHA del checkout), solo `afterok`,
nessun retry; un fallimento, un timeout o un controllo non superato si registra e ferma la catena.

| # | Job | Dipende da | Tempo massimo | Stima |
|---|---|---|---|---|
| 1 | `step2a_stats.sbatch p1` | checkout | 1 h | ~0,5 h |
| 2 | `step2a_stats.sbatch p2` (`--time=02:00:00`) | checkout | 2 h | ~1 h |
| 3-4 | `step2a_probe.sbatch p1 M,MS B_seed0` / `B_seed1` (`--time=02:30:00`) | 1 | 2,5 h ciascuno | ~1,5 h |
| 5-6 | `step2a_probe.sbatch p2 M,MS B_seed0` / `B_seed1` (`--time=03:00:00`) | 2 | 3 h ciascuno | ~1,8 h |
| 7 | `step2a_probe.sbatch p1 S,K -` | 1 | 1 h | ~0,3 h |
| 8 | `step2a_probe.sbatch p2 S,K -` con `COMPARE=1` (`--time=01:30:00`) | 2-7 | 1,5 h | ~0,5 h |

Stime dalla tranche downstream (sacct del 09/10: estrazioni P1 14 minuti e P2 42 su nodo GPU, in gran parte lettura e filtro; sonde P1 39-43
minuti, P2 48-53, piu' ~10% per le 64 colonne in piu'), non ETA di coda.

## Costo da autorizzare

**0 GPU-ora.** CPU: **tetto 132 core-ora** (somma dei tempi massimi x 8 CPU), stima ~75 core-ora; dal budget del passo 6. Disco: ~1,5 GB su
`$WORK`. Nessun training, nessuna estrazione GPU, nessun seme o variante in piu' rispetto a questa tabella.

## Verifica locale prima del congelamento

`tests/cpu/test_step2a.py` (11): P2 da capo a fondo su dati sintetici con le sonde vere (statistiche, varianti, sonde, provenienze); cache con
insieme di feature o larghezza diversi da quelli dichiarati rifiutate (il caso riprodotto da GPT); rifiuto di
cache su finestre diverse, di provenienze cambiate e di cache modificate dopo; regola di decisione su sei casi; confronto accoppiato e controlli
descrittivi. `tests/cpu/test_spectral_branches.py::test_feature_sets_of_the_2a_diagnostic`: le statistiche sono quelle dello Step 1 sullo stesso
ingresso. Nessuna misura sui task reali prima dell'approvazione.
