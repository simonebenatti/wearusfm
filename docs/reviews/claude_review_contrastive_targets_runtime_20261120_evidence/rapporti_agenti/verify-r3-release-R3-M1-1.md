# verify-r3-release-R3-M1-1 — scettico n. 1 sul rilievo R3-M1

Avvio: 2026-11-20. Compito: confutare R3-M1 (release: codice fuori inventario e discendenza dal commit rivisto non imposti).

## Letture iniziali
- contrastive_release.py letto per intero a 0efa4d45 (= fbfde797 per i sorgenti). source_paths() righe 58-64, verify_reviewed_sources 76-87, ramo verify_git 124-128: confermato che confronta solo inventario e due piani.
- Prontuario righe 133-136: "checkout immutabile figlio del commit rivisto con soli metadati operativi" (regola di procedura).

## Riproduzione indipendente (clone tmp/v1m1_clone a 0efa4d45, comandi miei)
- Hash a 0efa4d45: source 40d39f56...cde0, plan b03c0bc6...021a, preflight e4b39591...6dd92 (= delivery_pin), inventario 125 file.
- Review simulata 3432312f (ramo v1-review, figlio di 0efa4d45) con blocco APPROVED, reviewed_commit fbfde797. bind-review CLI rc 0.
- Release onesta H d8f4ac0b (0efa4d45 + copia review + binding): check-binding rc 0.
- Release iniettata X cf228390 (H + src/sitecustomize.py + scripts/argparse.py), checkout pulito, HEAD=X: check-binding rc 0, stdout con "APPROVED", stderr:
  "V1 INJECTED src/sitecustomize.py executed (unreviewed)" e "V1 INJECTED scripts/argparse.py executed instead of stdlib".
  Stesso esito con `python -B scripts/contrastive_job.py --help` (PYTHONPATH=src, cwd=checkout come lib_code_dir.sh:60-61): il codice iniettato gira all'avvio del dispatcher, prima di build_command.
- Commit-tree con l'albero di H: orfano db499d53 e figlio di 438c8d65 a945b7c6: check-binding rc 0, `merge-base --is-ancestor fbfde797` = no. Orfano con l'albero di X (21997c1d): rc 0 e codice iniettato eseguito.
- Unici chiamanti di validate_review_binding: contrastive_job.py:70, contrastive_run.py:182, make_job, check-binding, bind_review; tutti con verify_git=True e nessun controllo aggiuntivo. grep is-ancestor|merge-base|rev-list|diff --name in scripts/src: solo calibration_plan.py:176 (altra linea), contrastive_local_validation.py:49 (storico). campaign_checkout.sbatch e lib_code_dir.sh: solo HEAD=EXPECTED_SHA, pulizia e sola lettura.
- Variante src/ fuori da wearusfm: release Y 1e2dcf69 (H + src/csv.py): check-binding rc 0, stderr "V1 INJECTED src/csv.py executed instead of stdlib" (csv importato da contrastive_release.py:9).
- Vettore ulteriore (utile per la correzione): con `-m` sys.path[0] = cwd = CODE_DIR (lib_code_dir.sh:60). Un pacchetto torch/ alla radice viene importato da `python -m torch.distributed.run` (stderr "V1 ROOT torch/ package imported by -m from cwd"): i comandi torchrun di contrastive_job.py:88 sono esposti a moduli alla radice. L'alternativa "inventario esteso a src/ e scripts/" non lo coprirebbe; il controllo di diff si'.

## Ricerca di motivi per confutare
- Controllo altrove: nessuno. lib_code_dir.sh (HEAD=EXPECTED_SHA, pulizia, sola lettura), campaign_checkout.sbatch (commit presente, HEAD, pulizia, chmod), submit_job.py (registra EXPECTED_SHA nel ledger), dispatcher e runtime (stesso validate_review_binding): nessun controllo di ascendenza, di diff o di moduli caricati.
- Percorso di produzione: si'. Launcher `exec python3 scripts/contrastive_job.py` dopo code_dir_setup (contrastive_preflight_cpu.sbatch:19,28; gpu:21,30), poi execv di script in scripts/ o torchrun con -m.
- Dichiarazioni: prontuario 133-136 chiede "figlio del commit rivisto con soli metadati operativi" come procedura; nessun limite dichiarato in proposito (a differenza di N-6/N-7). Il commento di contrastive_release.py:129-131 riguarda l'autore della review, non il contenuto della release. preflight_plan.required_before_first_submission: "review bound to actual source and both plans" (soddisfatto). R-2 di 94f50942 chiedeva solo il legame a sorgenti e piani: implementato; il rilievo va oltre la correzione richiesta.
- Attenuanti: nessun modulo del checkout fuori inventario e' importato dalla catena (scansione dinamica dei 17 moduli contrastive_* + train_jepa: 0; import statici di moduli scripts/ non contrastive: 0; src/ contiene solo wearusfm, nessun file non-.py; nessuna collisione di nomi stdlib/terzi in scripts/). Quindi una base sbagliata ma onesta cambia file d'inventario (respinta) o file non eseguiti (innocua): il danno richiede l'aggiunta deliberata di file con nomi speciali. Un sitecustomize deliberato girerebbe comunque prima di qualunque controllo in-process: il controllo di diff va eseguito anche come comando git semplice, fuori dal Python del checkout.
- Ascendenza: con lo stesso albero il contenuto e' identico; la sola discendenza non ha effetto funzionale. Con un orfano, il push del deploy non porterebbe fbfde797 su Leonardo e verify_git fallirebbe chiuso se nessun altro ref lo contiene (inferenza, non verificata: niente Leonardo).

## Esito
- NON confutato: tutte le affermazioni di fatto del rilievo riprodotte con prove mie; nessun controllo compensativo.
- Gravita': MINORE confermata (difetto reale fra regola di release e cancello meccanico; nessun effetto BLOCCANTE nel flusso previsto, serve un'azione deliberata).

## Chiusura
- Clone tmp/v1m1_clone e cartella tmp/v1m1_work cancellati. Copia di riferimento rev_targets/clone: HEAD 0efa4d45, status pulito, nessun ramo v1-review.
