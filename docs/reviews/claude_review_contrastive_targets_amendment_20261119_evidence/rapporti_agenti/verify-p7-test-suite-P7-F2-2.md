# Verifica scettica n. 2 del rilievo P7-F2 (testo descrittivo del readout non protetto da test)

Copia: clone a 438c8d65 (sola lettura); mutazioni in tmp/v-p7f2-2 (clone --shared, detached a 438c8d65).

## Progressi
- letti: scripts/contrastive_benchmark_readout.py:95-111 e :360 (read_preflight inserisce initial_calibration_description nella ricevuta), test_contrastive_benchmark_readout.py:79-85, diff e388a1de..0efdc29c del readout (solo la stringa di riga 111), emendamento :46-47.
- git grep a 438c8d65 su tests/: nessun test legge 'interpretation' del readout; unico chiamante di initial_calibration_description nei test e' la riga 83.
- il readout e' nel perimetro di source_digest (scripts/contrastive_*.py, contrastive_prepare.py:192-201); nessun test lega il digest pinnato in preflight_plan/local_validation al codice (grep 'contrastive_targets_20261118' nei test: zero).
- riferimento storico (sha 367afd99..., pinnato): campi grad_norm, log_rms, log_env, band_shape, ms_fast, ms_slow; il suo campo 'limits' dice gia' «Only update1 can potentially match exactly».

## Riproduzione (comandi miei)
- Mutazione: nel clone tmp/v-p7f2-2 sostituita la sola riga 111 con quella di e388a1de; `diff <(git show e388a1de:scripts/contrastive_benchmark_readout.py) <mutato>` vuoto (file identico al blob 90945903).
- M0 (senza mutazione), tutti i 19 tests/cpu/test_contrastive_*.py: 314 passed in 110,64 s.
- M6 (mutazione), stessi 19 file: 314 passed in 103,90 s. Nei 9 file diretti del revisore: 173 passed in 13,82 s.
- Il readout mutato produce davvero il testo vecchio: initial_calibration_description su metrics sintetiche -> descriptive_only True, 'passed' assente, interpretation «Only update 1 may match exactly; ...».
- Conclusione fattuale: RIPRODOTTO. Nessun test della suite contrastiva legge 'interpretation'; l'unico test della funzione (righe 79-85) controlla descriptive_only, assenza di 'passed' e mean_ratios.

## Ricerca di motivi per confutare
- Percorso di produzione: SI'. preflight_plan.json ha la cella CONTRAST-READOUT-targets-v3-20261118 (mode readout); contrastive_job.py:77 lancia il readout; read_preflight (riga 360) mette la descrizione nella ricevuta. Non e' codice morto.
- Nessuna dichiarazione che il testo sia testato: local_validation.json (18 checks, targeted_tests 121) e voci del registro non lo affermano; l'emendamento (righe 46-47) dichiara solo la lettura descrittiva.
- Guardie meccaniche che limitano l'effetto (non sono test): source_digest() a 438c8d65 = 643a8937... = preflight_plan/local_validation.source_sha256; con la mutazione diventa 335496e9...; build_command (contrastive_job.py:64-68) rifiuta un job con source_sha256 != source_digest(). Un ritorno del testo richiede quindi una nuova identita' sorgente e un nuovo rilascio. Dopo la ricevuta, authenticate_preflight ricalcola e confronta l'intera ricevuta (testo incluso).
- Nessun effetto su evidenze vecchie: la ricevuta storica benchmarks_completed_bf16_20261118/preflight_readout.json contiene il testo vecchio, ma resta rifiutata per identita' sorgente/coorte, non per il testo.
- Sovrastima nel rilievo: il testo vecchio dice «may match exactly» (possibilita'), non afferma una corrispondenza. Nella ricevuta storica (25/50) l'update 1 coincideva solo a ~1e-7 relativo (6,6e-8..3,0e-7), non bit a bit. Con 100/125 cambiano log_rms e ms_fast (e grad_norm); per log_env, band_shape, ms_slow i target sono dichiarati invariati (CLAUDE.md; anchor_targets.py:43 per ms_slow), quindi un accordo all'update 1 non e' escluso: il testo vecchio sarebbe incompleto/fuorviante, non falso per tutti i campi. Il riferimento storico pinnato (campo 'limits') conserva comunque la formulazione vecchia.

## Esito
- refuted = false: la lacuna di test esiste ed e' riproducibile anche sull'insieme piu' ampio (19 file, 314 test).
- Gravita' reale: NOTA, non MINORE. Il comportamento che conta (descrittivo, nessun gate, nessun 'passed') e' testato; la parte non protetta e' una frase in un campo descriptive_only, senza effetto su gate, confronti, evidenze o tetti; un ritorno richiederebbe nuova identita' sorgente e nuovo rilascio. La correzione proposta (asserzione su RMS100/fast125 e «need not match») resta utile e costa una riga.
- Righe verificate: contrastive_job.py:62 (build_command), :65 (source_sha256 != source_digest()), :68 (rifiuto), :77 (comando readout); readout :360 (descrizione nella ricevuta), :401/:410 (authenticate_preflight confronta l'intera ricevuta); preflight_plan.json:157 (cella readout). La consegna non tocca tests/cpu/test_contrastive_benchmark_readout.py (diff e388a1de..0efdc29c vuoto).
- Pulizia: rimossi tmp/v-p7f2-2 e le cartelle -bt-m0, -bt-m6, -bt-m6d, -logs, -run; clone a 438c8d65 pulito.
