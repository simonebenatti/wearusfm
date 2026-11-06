# EPN a 15 epoche — raccolta finale e aggregazione, 06/11/2026

Raccolta e aggregazione eseguite da Claude, operatore della linea 2 dal 05/11 sera, con la procedura di
`docs/handoff_line2_claude_20261105.md` («Raccolta finale e aggregazione: procedura concreta per Claude»).
Codice e responsabilità scientifica restano a GPT consolidamento, che verifica dopo, senza bloccare.

24 fit `COMPLETED 0:0`, nessun riavvio; tutti i candidati terminano 15 epoche. Questa cartella contiene ricevute,
curve, log e risultati derivati: nessun segnale wearable, nessun peso, nessuna predizione OOF. Gli originali restano in

`/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/finetune15_20261105/epn/`

La copia locale usata per l'analisi è esterna al repo e temporanea:
`/private/tmp/wearusfm-ft15-claude-final-20261105/` (`epn15/`, `epn5/`, piano eseguito). WORK resta la fonte.

| File | Contenuto |
|---|---|
| `manifest.json` | 84 file di risultati e 24 checkpoint con SHA256, dimensioni e percorso remoto; pin della popolazione storica e della chiusura a 5 epoche |
| `remote_inventory.json` | Hash e dimensioni calcolati sul cluster il 06/11 alle 09:41 UTC |
| `fit_reports/` | 24 report, 24 candidati JSON e 24 metriche JSONL; nessuna OOF, nessun checkpoint |
| `report.json` | Aggregazione di `scripts/consolidation_ft15_report.py`: scelte, curve, celle, contrasti, coda di validation, confronto con 5 epoche |
| `operator_independent_check.json` | Ricalcolo di celle e contrasti dalle predizioni con codice separato, fatto dall'operatore |
| `verification.json` | Controlli di completezza fatti prima di leggere i punteggi, test, hash dei file |
| `sacct.txt`, `accounting.json` | 27 job: 24 fit, benchmark, due checkout |
| `logs/` | 27 log SLURM, con hash e dimensioni remoti in `verification.json` |

## Controlli prima di leggere i punteggi

27 job su 27 `COMPLETED 0:0`; 24 report completi con codice `1359b924`, piano `8a07e975…`, sorgenti `13de2f31…`,
ricetta `db975bd1…`, input `ef0090cb…` e teacher `a31c8950…`. I C0 non hanno letto il held. Ognuno dei 12 C1 fissa il
report del C0 della stessa cella con l'hash riletto sul cluster. I 24 hash dei checkpoint candidati riletti sul cluster
coincidono con quelli scritti nei report; nessun checkpoint è stato caricato. Tutti i candidati hanno eseguito gli
aggiornamenti pianificati (59.355 nei fold 0-1, 59.625 nei fold 2-3). I held sono 79, 79, 78 e 78 soggetti, disgiunti,
314 in tutto.

## Risultati

Held di sviluppo, 314 soggetti e 47.100 finestre, accuratezza bilanciata sulle sei classi, bootstrap appaiato per
soggetto con 2.000 repliche e seed 20261102, intervalli percentili al 95 %.

| Regime | 15 epoche | IC 95 % | 5 epoche |
|---|---:|---|---:|
| frozen | 71,420 % | [70,042; 72,873] | 70,575 % |
| finetune | 93,894 % | [93,051; 94,737] | 93,766 % |
| scratch | 93,059 % | [92,149; 93,947] | 92,363 % |

| Contrasto | Differenza (pp) | IC 95 % (pp) |
|---|---:|---|
| Primario: finetune − scratch a 15 | +0,834 | [+0,416; +1,263] |
| Primario: finetune − frozen a 15 | +22,473 | [+21,212; +23,641] |
| Secondario: frozen 15 − 5 | +0,845 | [+0,637; +1,043] |
| Secondario: finetune 15 − 5 | +0,127 | [−0,187; +0,450] |
| Secondario: scratch 15 − 5 | +0,696 | [+0,293; +1,093] |
| Secondario: (finetune − scratch) a 15 meno a 5 | −0,569 | [−1,051; −0,051] |

Il distacco finetune − scratch era +1,403 pp a 5 epoche ed è +0,834 pp a 15.

Coda di validation (regola congelata: almeno 3 fold con epoca migliore 15, oppure mediana del miglioramento fra le
epoche 14 e 15 di almeno 0,005):

| Regime | Fold con epoca migliore 15 | Mediana 15 − 14 | Segnale |
|---|---:|---:|---|
| frozen | 2 | +0,0019 | no |
| finetune | 0 | −0,0001 | no |
| scratch | 1 | +0,0007 | no |

LR scelti sulla validation: finetune 1e−4 in quattro fold su quattro (estremo alto della griglia), scratch 1e−4 in
quattro su quattro (estremo basso), frozen 0,01 in tre su quattro e 0,001 nel fold 3.

## Lettura nei limiti congelati (dell'operatore, da verificare da GPT)

- I due contrasti primari sono positivi con intervallo che esclude lo zero.
- Con 15 epoche scratch e frozen salgono; non emerge un guadagno risolto del fine-tuning passando alla ricetta a 15
  (+0,127 pp, IC95 [−0,187; +0,450]: un intervallo che comprende zero non prova uguaglianza o convergenza;
  formulazione rettificata il 06/11 su indicazione della verifica di GPT `bb08cd0`); il distacco
  finetune − scratch si riduce di 0,569 pp; l'estremo superiore dell'intervallo è −0,051 pp, vicino allo zero.
- Nessun regime fa scattare il segnale della coda di validation. Per la regola congelata questo non prova la
  convergenza e non autorizza una terza dose.
- Frozen a 71,420 % resta sotto la logistica storica a 82,577 %: confronto solo descrittivo.
- Limiti dichiarati nel protocollo: non è un effetto isolato della dose, perché cambia anche l'orizzonte del coseno;
  un solo seme; griglia di LR a due punti, con le scelte di finetune e scratch sull'estremo; secondo sguardo sui held
  di sviluppo; nessun test ufficiale; nessun verdetto generale su FM, P1/P2, regressione o V-R1.

## Verifica dei conti

`operator_independent_check.json`: celle e contrasti ricalcolati dalle predizioni con codice separato coincidono con
l'aggregatore (differenza massima 0,0 pp). Le estrazioni del bootstrap generate con lo stesso generatore e seme
coincidono con quelle dell'aggregatore, quindi il confronto degli intervalli verifica lo stimatore e l'appaiamento,
non lo schema di estrazione. Test dell'aggregatore: 32 superati.

## Consuntivo

| Gruppo | Job | GPU-ore | Core-ore |
|---|---:|---:|---:|
| frozen | 8 | 26,879167 | 215,033333 |
| finetune | 8 | 85,148889 | 681,191111 |
| scratch | 8 | 85,118611 | 680,948889 |
| benchmark | 1 | 0,125833 | 1,006667 |
| checkout | 2 | 0 | 0,031667 |
| **Totale** | **27** | **197,272500** | **1.578,211667** |

Tetto approvato: 304,5 GPU-ore e 2.436,666667 core-ore. Nessun tentativo fallito, nessun rilancio.

## Riproduzione

Dalla radice del worktree, con la raccolta esterna già presente e un output nuovo (il programma non sovrascrive):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  ~/.venvs/wearusfm-tok/bin/python scripts/consolidation_ft15_report.py \
  --input-root /private/tmp/wearusfm-ft15-claude-final-20261105/epn15 \
  --input-root-five /private/tmp/wearusfm-ft15-claude-final-20261105/epn5 \
  --plan /private/tmp/wearusfm-ft15-claude-final-20261105/executed_plan.json \
  --manifest results/infra/finetune15_20261105/final/manifest.json \
  --out /private/tmp/wearusfm-ft15-claude-final-20261105/reproduced_report.json
```

Il piano eseguito si estrae con
`git show 1359b924d25601f92e9035b8bd621c2c6a93f947:results/infra/finetune15_20261105/plan.json`.
Il codice non carica checkpoint, non predice il held e non esegue fit. Nessuna estensione o nuova sottomissione è
stata avviata da questa analisi.
