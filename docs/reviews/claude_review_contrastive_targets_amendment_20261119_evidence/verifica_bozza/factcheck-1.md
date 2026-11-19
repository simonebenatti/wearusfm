# Factcheck n. 1 della bozza claude_review_contrastive_targets_amendment_20261119.md

Stato: completato (vedi fine file).

## Errori trovati (in aggiornamento)

### E1 — ERRORE (R-2 e prompt per GPT): `6081832...` non e' lo SHA di preflight_plan.json
- Frase: «cita `preflight_plan.json` `6081832...` e i sorgenti `643a8937...`»; nel prompt «(preflight 6081832..., sorgenti 643a8937...)».
- Problema: 6081832627cee150... e' lo sha256 di `results/infra/contrastive_targets_20261118/plan.json` (il piano scientifico, pinnato dentro preflight_plan.json come `science_plan.sha256`, riga 9). Lo sha256 di `preflight_plan.json` e' ea2cef0fad2101df... (uguale a 0efdc29c e 438c8d65).
- Prova: `git show 438c8d65:results/infra/contrastive_targets_20261118/preflight_plan.json | shasum -a 256` -> ea2cef0fad2101df...; idem per plan.json -> 6081832627cee150...; `sed -n 7-10p preflight_plan.json` -> science_plan.path plan.json, sha256 6081832... L'errore viene dal critico (how_to_check della lacuna 1).
- Corretto: «cita `plan.json` `6081832...` (o `preflight_plan.json` `ea2cef0f...`) e i sorgenti `643a8937...`».

### E2 — ERRORE (R-3): non tutti i checkout storici hanno usato --time=00:04:00
- Frase: «Tutti i checkout storici della linea sono partiti con `--cpus-per-task=4 --mem=2G --time=00:04:00`.»
- Problema: il primo checkout CONTRAST-CHECKOUT-AUDIT-v2-20261117 (job 64652044) e' stato inviato con `--time=00:05:00`. CPU 4 e memoria 2G valgono per tutti.
- Prova: results/infra/contrastive_comparison_20261116/submission_20261117/checkout_state.json:14 `--time=00:05:00`; releases/audit_submission_plan_20261117.json:23 idem; gli altri cinque state.json (M30, CALIB, BENCH, bf16, mmap) e il piano readout hanno 00:04:00. Il critico lo annotava («uno con 00:05:00»).
- Corretto: «Tutti con `--cpus-per-task=4 --mem=2G`; tutti con `--time=00:04:00` tranne il primo (64652044, 00:05:00).»

### E3 — IMPRECISIONE (R-4): 64652559 e' andato oltre l'OverTimeLimit
- Frase: «Un audit oltre 7.200 s ma dentro l'OverTimeLimit verrebbe accettato. E' cio' che accadde a 64652559: 3.787 s su 3.600.»
- Problema: 3.787 - 3.600 = 187 s, piu' dei 180 s di OverTimeLimit (3 min). Il rapporto e' stato scritto alle 09:10:55, 186 s dopo il limite: dentro il margine complessivo (OverTimeLimit + KillWait, o la latenza del controllo), non dentro l'OverTimeLimit.
- Prova: audit_completed_20261117/accounting_and_integrity.txt (start 08:07:49, end 09:10:56, 3787 s, Timelimit 60; audit.json mtime 09:10:55; OverTimeLimit = 3 min; KillWait = 180 sec).
- Corretto: «dentro il margine di SLURM (OverTimeLimit 3 min piu' KillWait 180 s)».

### E4 — IMPRECISIONE (Richiesta): 0d91297 tocca 30 file, non 26
- Frase: «i 26 file che tocca sono identici byte per byte in entrambi.»
- Problema: `git show --name-only 0d912977` elenca 30 file. 26 sono identici a main e alla punta; 4 no: CLAUDE.md (diverso alla punta), docs/claude_GPT_shared e docs/decisioni.md (voci successive), e src/wearusfm/training/run.py, identico in main ma non alla punta, dove contiene le modifiche di 0d91297 fuse con il codice contrastivo.
- Prova: confronto dei blob 0d912977 / main (f546c9e4) / 438c8d65 per ciascuno dei 30 file; `git diff 0d912977 main -- src/wearusfm/training/run.py` vuoto; `git diff --stat main 438c8d65 -- src/wearusfm/training/run.py` 111+/14-.
- Corretto: «dei 30 file che tocca, 26 sono identici byte per byte in main e alla punta; run.py alla punta contiene le sue modifiche fuse con il codice contrastivo; CLAUDE.md, registro e decisioni hanno voci successive».

### E5 — IMPRECISIONE (Perimetro, Registro)
- Frase: «il registro della punta conserva ogni riga dei commit precedenti; cambia solo l'ordine.»
- Problema: 438c8d65 rimuove una riga (il titolo doppio «## 2) Problemi trovati», 2 occorrenze a afb89b45, 1 alla punta) e aggiunge 23 righe (due voci GPT nuove e la nota editoriale). Le voci sono conservate (come dice il critico: «voci conservate testualmente»), non ogni riga.
- Prova: multinsieme delle righe di docs/claude_GPT_shared afb89b45 vs 438c8d65: 1 riga mancante ('## 2) Problemi trovati'), 23 aggiunte; nota editoriale alla riga 14317 della punta.
- Corretto: «conserva ogni voce dei commit precedenti; sposta il blocco delle attivita', toglie il titolo doppio della sezione 2 e aggiunge due voci nuove».

### E6 — IMPRECISIONE (Verifiche positive 2 e M-1): la calibrazione storica e' fermata dalla barriera di ricetta, non dalla guardia delle sorgenti
- Frasi: «l'audit, la calibrazione 64701482 (λ 0,1194 e 0,3304) e la ricevuta canonica, alla guardia delle sorgenti»; in M-1 «Oggi le evidenze vecchie sono fermate prima dalla guardia delle sorgenti, che e' testata.»
- Problema: sul percorso reale calibration_report controlla la ricetta (contrastive_prepare.py:273-274) prima della guardia delle sorgenti (:288); calibration_report, authenticate_calibration e calibrated_configs rifiutano run e ricevuta storici con «calibration run config differs from the frozen 512-batch recipe». La guardia rifiuta la calibrazione solo se chiamata direttamente. Vale per l'audit (validate_audit chiama prima la guardia).
- Prova: rapporti_agenti/p2-probe_b.out.json voci 2a/2b (guardia diretta) e 2c/2d/2e (percorso reale, messaggio di ricetta); verify P2-N1-1: «guard_calls=[] (la guardia non viene mai chiamata)»; codice contrastive_prepare.py:266-290.
- Corretto: «l'audit alla guardia delle sorgenti; run e ricevuta della calibrazione 64701482 alla barriera di ricetta (e alla guardia se chiamata direttamente)».

### E7 — IMPRECISIONE (M-1): i due scettici confermano due barriere su tre
- Frase: «Il revisore le ha disattivate tutte e tre insieme e i 337 test contrastivi passano ancora; due scettici confermano.»
- Problema: i due scettici di P2-M1 hanno riprodotto solo le due barriere dell'audit (config_sha256 in validate_audit e read_preflight: X1+X2 e MD+MG, 337 passati). La guardia della ricevuta (MF) l'ha riprodotta un terzo scettico, quello di P2-N1 (337 passati con F). Nessuno scettico ha riapplicato le tre insieme.
- Prova: verify-p2-nessuna-eredita-P2-M1-1.md (X1, X2, X1+X2), P2-M1-2.md (MD+MG), P2-N1-1.md (F da sola, 337 passed); script_agenti/p2/results/all_contrastive_MD_MG_MF.txt (337 passed, solo revisore).
- Corretto: «due scettici confermano le due barriere dell'audit, un terzo la guardia della ricevuta».

### E8 — ERRORE (fatto superato: tabella dell'Esito, punto 6, e prompt per GPT, R-1)
- Frasi: tabella «diagnostica non ancora lanciata»; prompt «(non ancora deployata ne' lanciata)».
- Problema: la lettura delle 10:07 era corretta, ma il ledger comune locale registra dopo la bozza (salvata alle 11:23:41) gli invii della diagnostica da parte di GPT: CONTRAST-CHECKOUT-common-20261118 = 64835236 (11:32:01), CONTRAST-COMMON-cuda-20261118 = 64835286 (11:32:20), resume-CN = 64835297 (11:32:26), resume-CR = 64835304 (11:32:28), tutti `sottomesso` al commit 365cb1e2. Il checkout implica il push del ref di deploy. Lo stato su Leonardo non l'ho verificato (niente ssh).
- Prova: `grep -n "CONTRAST-COMMON\|CONTRAST-CHECKOUT-common" ~/wearusfm_local/jobs/ledger.tsv` (righe 1793-1802).
- Corretto: datare l'affermazione («alle 10:07 non deployata») e aggiungere che alle 11:32 GPT ha inviato checkout e tre celle (ID sopra); rileggere stato e ledger prima di pubblicare.

### E9 — IMPRECISIONE (Limiti e Metodo): N-3 non e' una lacuna del critico
- Frasi: «Ho verificato da solo le quattro piu' rilevanti (R-1...R-3, N-3), senza un secondo giro di scettici.»; nel Metodo «le quattro lacune del critico che entrano qui sotto».
- Problema: N-3 viene dal revisore del punto 6 (P6-N3) ed e' gia' stato confermato da uno scettico (verify-p6-diagnostica-P6-N3-1.md, NON CONFUTATO). Le nove lacune del critico non parlano di db66b507. Inoltre le lacune del critico che entrano nella review sono piu' di quattro: matrice, appaiamento, stato di Leonardo (R-1), pin di review (R-2), checkout e ore locali (R-3), local_validation (N-5), documenti operativi (N-2), titolo di P2-M1 (M-1).
- Prova: sintesi_workflow.json, critico.gaps (9 voci); dimensioni p6, P6-N3 survives true, upheld 1.
- Corretto: indicare le quattro lacune verificate da solo (es. R-1, R-2, R-3 e N-2 o N-5) e trattare N-3 come rilievo di P6 confermato da uno scettico.

### E10 — IMPRECISIONE (R-3): «Cumulato della linea 3» copre solo il confronto contrastivo
- Frase: «Cumulato della linea 3, training escluso: GPU-ore: 5,284444 + 6 + 14 = 25,284444; core-ore: ... = 236,704445.»
- Problema: gli addendi sono la campagna CONTRASTIVE-v2-20261116 (catena storica 5,284444/51,371111, diagnostica, nuovo preflight). La linea 3 e' «Log-quadratico e contrastiva» (CLAUDE.md, tabella) e ha anche job non di training della parte log-quadratica (ledger: LOGSQ-preflight-20261104 2 GPU-ore stimate, LOGSQ-initdiag-run 1, LOGSQ-infer-5000 0,5, LOGSQ-pose-5000 0,83, LOGSQ-extract 19618 2,25, piu' i job CPU logsquare-cpu-a/b). Aritmetica esatta.
- Prova: benchmarks_completed_bf16_20261118/accounting.txt («consuntivo_catena: 5.284444 GPU-ore, 51.371111 core-ore»); common_state_diagnostics_20261118/resources.json (incremental_cap 6 / 48,266667); ledger, campagne logsquare-*.
- Corretto: «Cumulato del confronto contrastivo della linea 3 (campagna CONTRASTIVE-v2-20261116), training escluso».

### E11 — IMPRECISIONE (Verifiche positive 3): C0−T100-G non compare solo nelle limitazioni
- Frase: «C0−T100-G non e' calcolato da nessun codice: compare solo nelle limitazioni dichiarate.»
- Problema: la prima meta' e' vera (nessun codice lo calcola). Ma il confronto compare anche in plan.json `target_recipe_amendment.historical_comparison`, nell'emendamento (righe 27-29) e nel protocollo par. 6, riga 132, dove resta un «controllo descrittivo di compatibilità» da riportare. E' il punto di N-1 della stessa bozza.
- Prova: plan.json (limitations[5] e target_recipe_amendment.historical_comparison); docs/contrastive_comparison_protocol_20261116.md:132; rapporto p3-confronti.md, sezione (c).
- Corretto: «non e' calcolato da nessun codice; compare solo come lettura descrittiva dichiarata (limitazioni, emendamento, protocollo par. 6)».

### E12 — IMPRECISIONE (Verifiche positive 3): non tutte le differenze fra i piani sono aggiunte
- Frase: «Ogni differenza fra i piani e' un'aggiunta dichiarata.»
- Problema: il campo `scope` e' riscritto, non aggiunto («Design and implementation for Claude review, no submissions» -> «New RMS100/fast125 preparation; no submissions; old artifacts remain historical only»). Tutte le altre differenze sono aggiunte.
- Prova: diff campo per campo dei due plan.json (contrastive_comparison_20261116 vs contrastive_targets_20261118); p3-confronti.md: «CHANGED scope (testo descrittivo)».
- Corretto: «Ogni differenza e' dichiarata: aggiunte piu' il testo di `scope`».

### E13 — IMPRECISIONE (Esito, tabella): N-2 non compare nella tabella
- Problema: N-2 e' etichettata «(punti 1 e 2)», ma le righe 1 (N-6, N-7) e 2 (R-2, R-4, M-1) non la elencano.
- Corretto: aggiungere N-2 alle righe 1 e 2.

### E14 — IMPRECISIONE (Perimetro): 438c8d65 non aggiunge solo evidenze
- Frase: «`438c8d65`: aggiunge solo evidenze.»
- Problema: oltre a full_suite/ e README aggiunge docs/prompt_claude_contrastive_targets_20261118.md (il prompt per Claude) e voci in docs/decisioni.md e nel registro, riordina il registro e modifica local_validation.json (da `"full_cpu_suite": "pending"` al blocco completo, piu' `independent_preparation_review`). Non tocca codice.
- Prova: `git diff --name-status afb89b45 438c8d65`.
- Corretto: «non tocca codice: aggiunge evidenze, il prompt e le voci di registro e decisioni, aggiorna local_validation.json».

## Verificato senza errori (sintesi)
Perimetro: afb89b45 senza src/scripts/tests; 65bc6022 porta identici i 5 file di codice e test di 365cb1e2 (e i pin); 0d91297 antenato di main e della punta.
R-1: 9 celle e dipendenze uguali, salvo i 3 prerequisiti dell'audit; split invariato da 8905e346; cancello alla riga 395 (rel 1e-4, abs 1e-6, total e grad_norm); amp_bf16 true come prima; 146/135 norme (results.json e registro r.9442); citazione GPT al registro r.9424 (18/11); bracci diversi solo per contrastive.arm; stato delle 10:07 conforme al file; 5 celle = 10 GPU-ore.
R-2: righe 62-69 e 22-31; audit senza release; sonda del critico; 643a8937 = source_digest alla punta e nel piano; contrastive_job.py e' nel digest.
R-3: 4 CPU/30 GiB/240 s dichiarati; launcher 1 CPU/2G/10 min alle righe 5-7; solo `--time=02:00:00` obbligatorio; 138,666667 e 137,066667; 30.800 MiB; billing 32/8/4; addendi e somme; OverTimeLimit 3 min e KillWait 180 s; stima 3.900-4.000 s e 6.032 s; manutenzione 23-25/11.
R-4: preflight_plan.json:196; audit senza limite interno, scrittura finale; validate_audit senza stato o durata.
M-1: righe 100, 333 e 384-385; 337 test con MD+MG+MF; test senza match; test_pinned a 8905e346 e commento letterale; test fuori dal digest; messaggi di match esistenti.
Note N-1...N-7: righe e citazioni (protocollo :132 nel par. 6; db66b507 :22 e citazione; break del driver; PLAN :30; readout :111; nessun test su interpretation o sul nuovo piano; verification.json con 5 pin; targeted_tests 121 senza comando; validate_recipe accetta il file senza target_version, che diventa v3; run_frozen da 8905e346; diff MDRT vuoto).
Verifiche positive: config_from_dict :46; 47 config storici letti 25/50/v1 senza modifiche (T100-G, 4, 36, 6); 4 config nuovi = storici salvo 3 campi; rigenerazione byte per byte; :117; nessun helper largo; 19 job e 16 release rifiutati; evaluation identica; compare invariato; coefficiente e formula (:349); ALLOCATED_SECONDS :17; 13/14/137,066667; digest 643a8937 contro 46938f31 (ricalcolato a 365cb1e2 e 8905e346); cuda_check invariato; db66b507 solo documenti.
Test: 173 + 32 passati (rieseguiti da me); 3.723 nodi in 233 file identici alla collection.log (collect-only da me); 131 nodi in 13 file; 263/263 nel recupero; impronte giuste; nessun cambiamento di sorgenti o test fra 65bc6022 e 438c8d65.

Stato: COMPLETATO.
