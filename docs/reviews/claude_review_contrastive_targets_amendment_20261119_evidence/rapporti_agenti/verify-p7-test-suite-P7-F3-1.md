# Verifica scettica n. 1 di P7-F3 (ricevuta della suite senza i log; targeted_tests senza comando) — 19/11/2026

Copia: clone a 438c8d65; clone temporaneo tmp/vp7f3-0efd a 0efdc29c per i conteggi (poi rimosso).

## Progressi e prove (comandi miei)
- `json.load(verification.json)['evidence_sha256']` -> 5 voci (cpu_suite.json, collection.log, disk_affected_files.json,
  recovery_remaining.json, recovery_remaining.xml), righe 24-30. `ls full_suite/` -> 12 file: fuori dai pin cpu_part0-3.log,
  cpu_ddp.log, recovery_remaining.log (6). Gli sha256 dei 6 log (shasum) non compaiono in nessun file del repo a 438c8d65
  (`git grep` su hash interi e prefissi da 12): legati solo dal commit 438c8d65, che li aggiunge tutti insieme alla ricevuta.
- cpu_suite.json non ha hash dei log: per gruppo solo part/files/returncode/tail (4 righe; 2 per ddp).
- Tail: ultime 4 righe di cpu_part0..3.log == tail di cpu_suite.json (True x4); i log sono lunghi 3306/1328/2254/1470 righe.
- PRECISAZIONE: cpu_ddp.log == "\n".join(tail ddp)+"\n" e recovery_remaining.log == "\n".join(recovery_remaining.json.tail)+"\n"
  (byte per byte, True). Questi due log hanno contenuto integralmente dentro file pinnati e 0 righe FAILED/ERROR.
  Gli elenchi FAILED/ERROR stanno solo nei 4 cpu_part*.log: 130/44/59/12 righe (= failed+errors dei sommari), 131 nodi distinti
  in 13 file; insieme dei file == disk_affected_files.json == verification.recovery_files (True). Il perimetro del recupero deriva
  quindi da 4 log non pinnati, e senza di essi i soli file pinnati (conteggi dei sommari, collection.log, xml) non permettono di
  verificare che nessun nodo fuori dai 13 file sia fallito: i log sono portanti per `unresolved_failures: 0`.
- Nessun consumatore in codice della ricevuta (`git grep "full_suite\|verification.json" -- scripts src tests` vuoto;
  PASSED_FULL_CPU_SCOPE solo in JSON): nessun gate dipende dai pin -> nessun effetto da BLOCCANTE/MINORE.
- Ricevuta gemella target_windows_integration_20261118 (f546c9e4): anche li' pin solo su JSON/XML, ma il recupero rifa' gruppi
  interi, quindi gli XML pinnati bastano; qui il recupero e' per file e richiede gli elenchi dei log.
- targeted_tests: local_validation.json:78-81 solo {passed:121, seconds:12.52}; nel registro solo «Test mirati121verdi»;
  nessun comando/elenco in docs, script, commit (0efdc29c ha messaggio di una riga). La ricevuta storica dello stesso autore
  (contrastive_comparison_20261116/verification.json) registrava `files` per ogni giro mirato. A 0efdc29c (collect-only) piu'
  combinazioni danno 121: prepare43+job29+benchmark_readout26+run14+historical_calibration9 = 121, oppure
  prepare43+job29+readout26+hist_calibration9+hist_callsites5+mmap7+callsite_oracle2 = 121; i 17 test_contrastive*.py = 307
  (= contrastive_suite_initial.log). Insieme non ricostruibile in modo univoco.
- Codice del perimetro identico fra 0efdc29c e 65bc6022 (diff aggiunge solo i diagnostici e i loro test): la suite completa a
  65bc6022 supera davvero il conteggio mirato.

## Esito
NON CONFUTATO. Gravita' reale: NOTA (tracciabilita'/documentazione; nessun codice legge la ricevuta, nessun effetto su ricetta,
evidenze, confronti o tetti). Correzione di precisione: i log effettivamente non pinnati nel contenuto sono 4 (cpu_part0-3.log),
non 6; cpu_ddp.log e recovery_remaining.log sono copie integrali di tail pinnati. Fix proposto valido, basta pinnare i 4 part log
(o tutti e 6 per uniformita') e registrare file/comando del giro mirato o togliere il campo.
