# Verifica scettica n. 1 del rilievo P3-M1 (comparatore finale non legato a coorte/ricetta)

Copia: clone @ 438c8d65 (detached, pulita dopo le prove). Solo lettura, nessun job.

## Esito
Fatti tecnici RIPRODOTTI; come difetto MINORE della consegna CONFUTATO. Gravita' reale: NOTA
(requisito della futura release di estrazione/confronto), non MINORE e non BLOCCANTE.

## Riproduzione indipendente (via from_manifest, non solo compare)
Script mio: tmp/verify_p3m1_1/repro_manifest.py (fixture sintetiche di test_contrastive_compare,
manifest pinnati scritti su disco, uscite sintetiche poi cancellate).
Comando (dalla radice della copia):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD/src:$PWD/tests/cpu:$PWD/scripts \
  ~/.venvs/wearusfm-tok/bin/python -B ../tmp/verify_p3m1_1/repro_manifest.py $PWD ../tmp/verify_p3m1_1/run
Output essenziale:
- mixed_T100G_as_C0_with_amendment_field: ACCEPTED decision_valid=True protocol=contrastive-comparison-20261116-v2
  (C0 = SHA reale del checkpoint T100-G 19.618 a31c8950..., cella annotata 25/50/v1; CN/CR 100/125/v3;
  manifest che dichiara falsamente amendment=contrastive-target-windows-20261118-v1).
- mixed_T100G_as_C0_no_amendment: ACCEPTED, stesso protocol.
- homogeneous_new_no_amendment: ACCEPTED, stesso protocol (indistinguibile per `protocol` dal caso misto).
- pose_vs_audit_checkpoint_mismatch: REJECTED "classification/pose checkpoint identity differs" (unica guardia d'identita').
- L'uscita ricopia il manifest in `input_pins` + `input_manifest_sha256`: un campo amendment del manifest
  compare nell'uscita, ma non e' richiesto ne' validato.
- `grep -n -E "rms_window|ms_fast|target_version|amendment|cohort|config" scripts/contrastive_compare.py` -> exit 1.
- pytest tests/cpu/test_contrastive_compare.py: 32 passed (basetemp cancellato).
Precisazione: oltre a protocol/step/which/griglia/SHA checkpoint, from_manifest+compare controllano anche pin
SHA dei file, appaiamento OOF (row/y/s/fold), supporto/target/soglia posa, conteggi e identita' audit: tutte
guardie di appaiamento dei dati, nessuna di ricetta. Il "solo" del rilievo e' riduttivo ma non cambia la sostanza.

## Perche' non e' un difetto di questa consegna
1. Dichiarazione esplicita di perimetro: docs/contrastive_comparison_protocol_20261116.md:166 (identica a c680e95d:160,
   non toccata dall'emendamento) "Il comparatore verifica OOF appaiati, supporto posa e checkpoint comuni, ... 420 record
   di fit ... Provenienza/leakage degli estrattori sono prerequisiti distinti." Il comparatore rispetta la sua specifica;
   il legame con coorte e ricetta e' assegnato agli estrattori. docs/contrastive_preflight_execution_20261116.md:76:
   "Le estrazioni e i fit downstream hanno un preventivo separato da pubblicare prima dei loro invii."
2. Percorso non eseguibile oggi: scripts/contrastive_job.py:14-15 MODES = cuda/calibration/full/split/audit/readout,
   nessuna modalita' di confronto; plan.json executable=false, release=null, training_authorized_by_this_file=false;
   nessuno script contrastivo produce celle posa/OOF (resources.extraction_cells=12 e' solo preventivo).
3. Controlli a monte: scripts/contrastive_run.py:29-52 validate_recipe (training: config == calibrated_configs da
   build_review_configs, quindi 100/125/v3), :54-92 validate_cohort (tre bracci dallo stesso bundle/ricevuta), :165
   APPROVED_FOR_TRAINING, validate_resume lega initial_sha256/provenance della coorte. Nessun APPROVED_FOR_TRAINING in
   results/infra/contrastive_comparison_20261116 (git grep): nessun C0/CN/CR storico a 19.618 da mescolare.
4. Parte invariata e fuori dal diff: `git diff c680e95d 438c8d65 -- scripts/contrastive_compare.py tests/cpu/test_contrastive_compare.py`
   = 0 righe; il file non e' in `git diff --stat e388a1de 0efdc29c`. La stessa proprieta' valeva per la coorte storica:
   non e' introdotta dall'emendamento. Richiesta di Simone: "senza chiedere giri ripetuti su parti invariate".
5. Coerente col modello della linea 3: nel lettore LQ (scripts/logsquare_pose_readout.py:88-115) provenienza,
   model_config e rifiuto di un checkpoint storico stanno nel piano/lettore, non nel confronto appaiato.
6. Tracciabilita': input_manifest_sha256 e input_pins pinnano le celle posa che portano checkpoint_sha256 per braccio;
   manca solo l'etichetta nel campo `protocol`. Nessun codice (nemmeno i lettori del preflight) legge `amendment`:
   l'identita' nuova e' affidata ai pin SHA di piano e config.

## Scenario di errore
Ogni manifest misto richiede che un estrattore futuro etichetti male braccio o passo (T100-G come C0, un 576 storico
come 19.618): difetto di una consegna non ancora scritta. Con estrattori corretti il comparatore non da' uscite sbagliate.

## Gravita' reale
NOTA: nella futura release di estrazione/confronto il manifest deve legare cohort.json, config e checkpoint di ogni
braccio (terna 100/125/fast125_v3 uguale per i tre) e registrare l'emendamento. Nessuna correzione ora, nessun nuovo giro.

## Non verificato
- Cluster: assenza effettiva di checkpoint contrastivi a 19.618 su Leonardo (verificata solo l'assenza di release nel repo).
- Estrattori e manifest futuri (non esistono).
