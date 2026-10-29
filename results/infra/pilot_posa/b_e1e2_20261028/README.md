# E1/E2 di B student a 5.000, 28/10/2026

Job 62539724 (estrazione) e 62540295 (rapporto): COMPLETED, 0:0. Lettura diagnostica, nessun verdetto scientifico
all'endpoint di 49.046 passi. [Analisi](../../../../docs/reviews/b_e1e2_results_20261028.md),
[protocollo](../../../../docs/valutazione_b_e1e2_20261028.md).

- `preflight.json`, `*_submission.json`, `startup*`: evidenze storiche prima del completamento; non sovrascritte.
- `final/report.json`: risultato originale, inclusi i booleani legacy del programma frozen. E2 e' presente qui,
  mentre il riepilogo stdout omette la sua sezione.
- `final/*provenance.json`: controlli e impronte salvati nei compute node prima delle due fasi. L'estrazione contiene
  identita' delle registrazioni e metadati/file size, non gli array dei wearable. Non certifica il rehash di tutti gli array.
- `final/*.out`, `sacct_utc.txt`, `squeue.txt`, `ls_l.txt`: log e accounting grezzi, raccolti in sola lettura.
- `final/collection_manifest.json`: percorsi di origine, comandi di lettura e SHA256 delle copie locali.
- `final/accounting.json`, `validation.json`: riepiloghi derivati da accounting e provenienze.
- `analysis/analyze_e1e2.py`, `analysis.json`, `per_user_e1_e2.png`: elaborazione locale del JSON; bootstrap appaiato
  del gap fissato prima dei risultati (1.000 repliche, seme 20261009, pesi degli elementi validi).

Riproduzione dalla radice del worktree, senza GPU, import del trainer o accesso remoto:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  ~/.venvs/wearusfm-tok/bin/python \
  results/infra/pilot_posa/b_e1e2_20261028/analysis/analyze_e1e2.py \
  results/infra/pilot_posa/b_e1e2_20261028/final/report.json --plot
```

Lo script richiede numpy; la figura richiede anche matplotlib. Scrive solo nella propria cartella. Con dati insufficienti
o incoerenti indica `evaluable=false` e la causa; non elimina utenti o repliche per forzare un IC.
Ricalcola pooling, gap e IC E2 dai valori per utente. E1 conserva IC e aggregato del rapporto: il JSON E1 non contiene
i conteggi degli elementi validi per utente, necessari per ricalcolarli indipendentemente. Verifica rapporto MAE/costante,
rapporti individuali, mediana e conteggio delle finestre; non sostituisce i pesi degli elementi con il numero delle finestre.

Checkpoint e cache rimangono sul cluster nei percorsi riportati nelle provenienze; nessun dato grezzo, NPZ o checkpoint
in questa consegna. Costo consuntivo: **0,435278 GPU-ore e 3,595556 core-ore**, entro 0,666667 e 8.
