# EPN fine-tuning — raccolta e analisi conclusiva, 05/11/2026

24 fit COMPLETED 0:0; tutti i candidati terminano cinque epoche.
Questa cartella contiene ricevute, curve e risultati derivati, non segnali
wearable, pesi o predizioni OOF native. Gli originali persistono in:

`/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/consolidation_20261104/finetuning/epn/`

La copia locale per l'analisi è esterna al repo:
`/private/tmp/wearusfm-ft-results-20261105/epn/` (temporanea, WORK resta la fonte).

| File | Contenuto |
|---|---|
| `manifest.json` | 84 file di risultati + 24 checkpoint con SHA256/dimensioni/percorso remoto; pin della popolazione storica |
| `remote_sha256.txt`, `remote_sizes.txt` | Evidenza di hash e dimensioni calcolati sul remoto |
| `fit_reports/` | Copia dei 24 report, 24 candidati JSON e 24 metriche JSONL; nessuna OOF o checkpoint |
| `report.json` | Aggregazione verificata, scelte/curve, statistiche pooled/per soggetto/per classe e F-R2 |
| `independent_review.json` | Calcolo indipendente del critico prima del confronto con l'aggregatore |
| `verification.json` | Test, esito della review e hash dei file di analisi/evidenza |
| `sacct.txt`, `accounting.json` | 29 job: 24 fit, due benchmark, tre checkout; escluso training lungo |

Riproduzione da root del worktree, con i dati esterni già raccolti e un output
nuovo (il programma rifiuta di sovrascriverne uno esistente):

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ~/.venvs/wearusfm-tok/bin/python scripts/consolidation_ft_report.py \
  --input-root /private/tmp/wearusfm-ft-results-20261105/epn \
  --plan results/infra/consolidation_20261104/finetune_plan.json \
  --manifest results/infra/consolidation_20261104/final_20261105/manifest.json \
  --out /private/tmp/wearusfm-ft-results-20261105/reproduced_report.json
```

Il codice non carica checkpoint, non predice held e non esegue fitting.
Verifica tutti gli input e ricostruisce la selezione dalla validation.
Bootstrap: 2.000 repliche appaiate per soggetto, seed 20261102, zero scarti.

Risultato FT−scratch +1,403pp [0,930;1,837], condizionato alla dose di cinque
epoche. F-R2 scatta; frozen70,575% sotto logistica storica82,577%. Lettura e
limiti completi in `docs/reviews/finetune_results_20261105.md`.
Nessuna estensione a 15 epoche o altra sottomissione avviata da questa analisi.
