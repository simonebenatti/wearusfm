# Verifica scettica n. 2 di R3-M1 (r3-release)

Stato: in corso (20/11/2026). Copia di sola lettura a 0efa4d45; mutazioni solo in un clone temporaneo.

## Letture del codice (fbfde797)
- `contrastive_release.py:58-64,76-87`: l'inventario confrontato col commit rivisto e' quello di `source_digest` (src/wearusfm/**/*.py, scripts/contrastive_*.py, scripts/slurm/contrastive*.sbatch, train_jepa.py, lib_code_dir.sh). `validate_review_binding` (106-134) non ha controlli di ascendenza o di diff.
- `contrastive_job.py:62-73`: unico punto di esecuzione; chiama `validate_review_binding` con verify_git=True; nessun confronto di `WEARUSFM_CODE_SHA` col commit rivisto.
- `lib_code_dir.sh:37-61` (invariato rispetto a 438c8d65): HEAD = EXPECTED_SHA, albero pulito, sola lettura, poi `PYTHONPATH=$CODE_DIR/src`. Nessuna ascendenza.
- `campaign_checkout.sbatch` e `submit_job.py`: nessuna verifica di ascendenza o di diff (submit_job registra solo EXPECTED_SHA).
- Launcher `contrastive_preflight_{cpu,gpu}.sbatch:28/30` (invariati): `exec python3 scripts/contrastive_job.py` senza -I/-P/-S/-E.

## Riproduzione indipendente (script mio: rev_runtime/tmp/v2m1_scripts/repro.sh, uscita repro.out)
Clone temporaneo `--shared` a 0efa4d45; review finta in un commit laterale scritto da me (reviewed_commit 0efa4d45, hash correnti:
source 40d39f56...cde0, plan b03c0bc6...021a, preflight e4b39591...6dd92); `bind-review` rc 0; commit di release figlio con copia + binding.
`check-binding` (PYTHONPATH=src, cwd = checkout, come il launcher):
- M0 solo metadati: rc 0 (controllo positivo).
- M7 + `src/sitecustomize.py`: rc 0, stampa "V2M1 INJECTED sitecustomize eseguito".
- M8 + `scripts/argparse.py` (oscura la stdlib, poi carica quella vera): rc 0, entrambi i marcatori.
- N1 modifica di `scripts/contrastive_job.py`: rc 1 "review does not bind the current sources and both plans" (respinto).
- N2 nuovo `scripts/contrastive_extra.py`: rc 1, stesso messaggio (respinto).
- A1 commit orfano con l'albero della release iniettata: rc 0, `merge-base --is-ancestor 0efa4d45 HEAD` = no.
- A2 stesso albero con genitore 438c8d65: rc 0, ascendenza no.
- `python scripts/contrastive_job.py --help` nel checkout iniettato: i due marcatori compaiono prima dell'usage (codice eseguito prima di ogni controllo).
- Binding con `review_commit` = "2"*40 e `original_review_path` inesistente: check-binding rc 0 (provenienza non riverificata a runtime).
=> I fatti del rilievo sono RIPRODOTTI.

## Ricerca di controlli compensativi (nessuno trovato)
- Nessun `merge-base`/`is-ancestor`/`diff --name-only` nella catena contrastiva; precedente nel repo: `calibration_plan.py:172-179` (`_require_commit`).
- `contrastive_run.py:181-183` richiama la stessa `validate_review_binding`; `contrastive_cuda_check.py:287` e `run.py:1825` registrano solo `WEARUSFM_CODE_SHA` (provenienza, non cancello).
- `campaign_checkout.sbatch`: solo esistenza del commit, HEAD, pulizia, sola lettura. `submit_job.py:190-192`: registra EXPECTED_SHA.
- Prontuario 126-127: bind-review controlla byte della review, sorgenti/piani nel commit rivisto, copia nel checkout; 129-131: "non si pretende una firma crittografica d'autore"; 133-136: "figlio del commit rivisto con soli metadati operativi" come compito dell'operatore, senza comando di verifica. Prompt GPT 58-60: "checkout con soli metadati e sorgenti invariati". Preflight: checkouts = "separate immutable metadata releases"; nessuno dei sei e' riletto da Claude.
- Test `test_contrastive_release.py:57`: "Later metadata commits are permissible; execution bytes remain identical" (prova solo un release.json).

## Perche' l'impatto reale e' basso
- Nessun percorso accidentale: la catena importa solo moduli dell'inventario (+ stdlib/terzi dal venv; grep degli import, anche locali); src/ contiene solo wearusfm, nessun file non .py; le risorse del repo lette a runtime sono pinnate (BASELINE_SHA256, HISTORICAL_INITIAL_SHA256, policy_reference via plan); nessun nome in scripts/ collide con `sys.stdlib_module_names`; `__pycache__/` in .gitignore. Un commit di base sbagliata viene respinto se tocca l'inventario o i piani (N1/N2).
- Serve un'azione deliberata (sitecustomize, moduli omonimi, .so/.pyc). Il modello dichiarato non difende da un operatore sleale: io stesso ho creato una review APPROVED e un binding accettati; a runtime passa anche un review_commit inesistente. L'iniezione non aggiunge capacita' rispetto a questo.
- L'ascendenza da sola non cambia l'esecuzione: con lo stesso albero i byte sono identici; contano solo i file aggiunti.

## Esito
Rilievo non confutato nei fatti (riprodotto, nessun controllo compensativo, percorso di produzione). Gravita' reale: NOTA (non MINORE):
proprieta' "soli metadati" dichiarata come compito dell'operatore e non meccanizzata; nessuna conseguenza senza manomissione deliberata,
che il modello di provenienza non crittografico gia' non copre. Correzione utile e a basso costo: in verify_git `merge-base --is-ancestor
<reviewed_commit> HEAD` + `git diff --name-only <reviewed_commit> HEAD` limitato ai percorsi di metadati, oppure il comando nel prontuario.
Non verificato: ambiente dei moduli cineca-ai (se aggiunge un proprio sitecustomize la via src/sitecustomize.py su Leonardo potrebbe non scattare; l'oscuramento da scripts/ resta certo); come 3-launcher passa il launcher a sbatch (copia del checkout o stdin).
