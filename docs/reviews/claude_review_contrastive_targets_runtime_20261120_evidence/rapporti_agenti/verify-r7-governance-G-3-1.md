# verify-r7-governance-G-3-1 (scettico n. 1 su G-3)

Avvio: 20/11. Clone 0efa4d45 (sola lettura). Obiettivo: confutare G-3 (consumatori di validate_review_binding / validate_audit_accounting non protetti da test) se possibile.

## Riproduzione (clone temporaneo g3v1clone a 0efa4d45; script fbfde797 == 0efa4d45, diff solo delivery_pin.json)
- Base: job/run/release/targets_readout 89 passed; + historical_callsites 94 passed. Coincide col rilievo.
- Mutazione A (R-2): `pass` al posto di contrastive_job.py:70-72 (validate_review_binding + confronto review) e contrastive_run.py:182-184 (validate_review_binding + review_pin==review_commit).
  - 7 file (job/run/release/targets_readout/historical_callsites/targets_guards/benchmark_readout): 125 passed.
  - INTERO insieme dichiarato dalla ricevuta validation_final (29 file, -o tmp_path_retention_policy=none): 465 passed in 100 s, come la ricevuta (465). Nessun test rileva la rimozione. CONFERMATO.
- Mutazione A come nel rilievo (job 70-72, run 181-184 con l'import): 4 file job/run/release/targets_readout 89 passed = base. CONFERMATO.
- Mutazione B (R-4), due varianti:
  - sola chiamata: job.py:81, run.py:195, release.py:242 -> INTERO insieme di 29 file: 465 passed in 105 s = ricevuta. CONFERMATO.
  - con i rami: job.py:80-81, run.py:193-195 (anche il controllo di presenza `audit_sacct`), release.py:240-242 -> 5 file (con historical_callsites) 94 passed = base. CONFERMATO.
- A+B insieme: 5 file 94 passed.
- Estensione: release.py:221 `declared = validate_review_binding(binding)` sostituita con `binding["review_payload"]` -> release/job/run 66 passed. Anche la chiamata R-2 di make_job non ha test negativi (test_contrastive_release.py:144-151 usa solo un binding valido e una cella audit, quindi non raggiunge nemmeno :240-242).

## Verifiche positive
- Codice corretto oggi: sonda mia /private/tmp/claude-501/-Users-simonebenatti-dev-wearusfm/af7c0377-8fec-4211-a7d6-d075a9892443/scratchpad/rev_runtime/tmp/g3v1_tests/test_g3_consumer_wiring.py (spia che registra e rifiuta, con le fixture esistenti): 11 passed sul codice intatto (build_command chiama il binding in audit/cuda/readout/full e l'accounting in cuda/readout/calibration/full; rifiuta review diversa dal binding; validate_release chiama binding e accounting e rifiuta review_pin != review_commit). Con A+B: 11 failed. Il test proposto dal fix e' fattibile e distingue la regressione.
- Lettore finale protetto: readout.py:421 sostituita -> historical_callsites 1 failed (riga 128). Conforme al rilievo.
- Riferimenti del rilievo esatti a fbfde797: job.py:69-81, run.py:181-195, release.py:240-242, test_job.py:26-27, test_run.py:73-74.
- Test fuori dal digest: source_digest (contrastive_prepare.py:192-201) e source_paths (contrastive_release.py:58-64) coprono solo src/wearusfm, scripts/contrastive_*.py, scripts/slurm/contrastive*.sbatch, train_jepa.py, lib_code_dir.sh; il blocco di review lega solo sorgenti e due piani. delivery_pin.json non e' letto da nessun codice (grep vuoto); si aggiornerebbe solo la ricevuta locale.

## Ricerca di motivi per confutare (nessuno regge)
1. Controllo altrove: i launcher contrastive_preflight_{cpu,gpu}.sbatch eseguono solo contrastive_job.py (r.28/30), nessun check-binding/check-audit separato. Per audit, cuda e readout build_command e' l'unico controllo del binding a runtime (audit/cuda_check/readout non chiamano validate_review_binding). Nessun altro test in tests/ (esiste solo tests/cpu; i 17 file che importano i moduli sono tutti nei 29).
2. Percorso non in produzione: no. build_command <- contrastive_job.main <- sbatch; validate_release <- contrastive_run.main (r.309) <- torchrun da build_command; make_job <- CLI make-job del prontuario (r.159).
3. Dichiarazione esplicita: nessuna. Prontuario r.143-145 «Il gate è eseguito nella release, nel dispatcher e nel runtime della calibrazione»; risposta R-2 (r.11) senza test citati, R-4 (r.13) test negativi solo della funzione (esistono, test_release.py:109-157). Nessun limite dichiarato sull'integrazione.
4. Codice congelato: mitigazione reale (il pacchetto rivisto e' corretto e il digest lega i byte), ma la protezione serve alla prossima revisione dei sorgenti; non rende il rilievo documentale.
5. Precedente: la review 94f50942 ha classificato MINORE (M-1) proprio «barriere di secondo livello senza test sul codice corrente». G-3 e' lo stesso tipo di lacuna sulle nuove barriere R-2/R-4: MINORE coerente.

## Esito
NON confutato. Gravita' MINORE confermata: nessun effetto sulla catena attuale (codice corretto, sonda 11/11), lacuna di verifica reale e dimostrata per mutazione sull'intero insieme di 29 file della ricevuta.
Pulizia: clone g3v1clone e basetemp g3v1_pt_* cancellati; conservati come prova g3v1_tests/test_g3_consumer_wiring.py e g3v1_scripts/mutate.py.
