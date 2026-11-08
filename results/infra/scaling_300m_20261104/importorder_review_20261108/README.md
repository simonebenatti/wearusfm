# Verifica locale del recupero import-order, 08/11/2026

Review e piano: `docs/reviews/gpt_review_scaling_importorder_recovery_20261108.md`.
Nessun job, SSH, deploy, nuovo training/forward o modifica ai worktree rivisti.

Sorgenti:

- diagnosi: `913b4ae4bdfd8de709641ef39383a9fa7e0020f0`;
- proposta: `0c5b974af2483faf767a2e171183ae1af0dcdb19`;
- controllo estrazione 30M: `453ce6c496745f2d73aa3007406721586e6dd767`;
- lettura scientifica lunga 30M: `57c4d7ea564949cbf74506085578b77a4d033d10`.

Contenuti:

- `importorder-review-audit.py` / `.json`: 42 verifiche indipendenti
  di diff, AST, identità storiche e diagnosi. Verifica anche la presenza dei
  risultati dei due comandi pytest riportati sotto; non rilancia pytest.
- `importorder-review-launcher-tests.txt`: 40 PASS, 45,62 s.
- `importorder-review-v1-pin-tests.txt`: 7 FAIL attesi sul piano storico
  incompatibile col nuovo wrapper, 110 deselezionati, 3,32 s. Non è una suite
  completa verde e non è un rilascio operativo del vecchio piano.
  Copia del log con soli spazi di fine riga rimossi; contenuto invariato.
- `importorder-evaluation-review-20261108.md` e relativo evidence JSON:
  mappa dei processi/guardie e pin dei nove file storici. I riferimenti
  `/private/tmp` nella nota indicano il luogo originario delle prove; le
  copie definitive sono in questa cartella.
- `importorder-science-costs-20261108.py` / `.json`: costo riprodotto da
  freeze, summary e accounting storico; fallimento e diagnosi sono calcolati
  dalle durate pubblicate, non da una nuova lettura scheduler.
- `critic.txt`: controllo indipendente della proposta e collegamenti operativi.
- `manifest.json`: hash di queste evidenze e riferimenti della consegna.

Riproduzione (percorsi di esempio; i checkout di diagnosi e proposta devono
essere puliti e sui commit esatti):

```bash
python3 importorder-review-audit.py \
  --diagnosis-repo /percorso/diagnosi \
  --fix-repo /percorso/proposta \
  --out /tmp/importorder-audit.json

python3 importorder-science-costs-20261108.py \
  --repo /percorso/diagnosi --out /tmp/importorder-costs.json
```

Test eseguiti dalla directory della proposta, con Python del venv WearUsFM,
`src/` del checkout in testa al path e thread singolo. Comandi equivalenti
per riprodurre gli stessi gruppi:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  ~/.venvs/wearusfm-tok/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/cpu/test_scaling_continue_launcher.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  ~/.venvs/wearusfm-tok/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/cpu/test_scaling_continue.py \
  -k 'published_plan_is_valid or job_seeds_a_new_directory or job_without_a_complete'
```

Il secondo comando deve riprodurre i sette rifiuti di pin su 0c5b974.
Nel pacchetto di recupero coerente dovrà invece passare con il nuovo piano.
L'audit e il calcolo dei costi sono stati rieseguiti dal revisore principale;
JSON ottenuti identici alle copie archiviate. Nessun payload wearable è
copiato qui; solo metadati e statistiche della verifica.
