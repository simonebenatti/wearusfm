# Fact-check n. 2 della bozza claude_review_contrastive_targets_amendment_20261119.md
Fonti: clone 438c8d65, evidenze _evidence/. Solo lettura.

## Esiti (aggiornati man mano)

### F1 (IMPRECISIONE) Registro "conserva ogni riga ... cambia solo l'ordine"
- Bozza r.17. Multiinsieme righe afb89b45 vs 438c8d65: manca 1 riga (titolo duplicato "## 2) Problemi trovati", 2 occorrenze -> 1) e ci sono 23 righe aggiunte. Lo dice anche il critico (critico-completezza.md:56-57).

### F2 (ERRORE/IMPRECISIONE) "0d91297 ... i 26 file che tocca sono identici byte per byte in entrambi"
- 0d912977 tocca 30 file (26 in src/scripts/tests + CLAUDE.md, docs/claude_GPT_shared, docs/decisioni.md, docs/target_windows_fix_20261118.md).
- src/wearusfm/training/run.py: 0d=main ma main != tip (alla punta ha in piu' le modifiche contrastive: campo `contrastive`, config_to_dict, _accumulate_terms). Gli hunk di 0d91297 ci sono (reverse patch si applica).
- Identici in main e punta: 26 file = 25 di codice + docs/target_windows_fix_20261118.md.

### OK verificati (parziale)
- Citazioni del prompt GPT (prompt_claude_contrastive_targets_20261118.md:4-5, 23, 32-33) letterali.
- CLAUDE.md alla punta: «sul punto 3 passiamo a 100/100» (r.25); ruoli GPT scientifico/3-launcher (r.37-44); Claude non lancia job in linea 3 (r.58).
- afb89b45 non tocca src/scripts/tests. 65bc6022: 5 file di codice/test della diagnostica identici a 365cb1e2 (anche alla punta).
- db66b507 tocca solo 5 documenti; non e' antenato della punta (sta su codex/tokenizer-logcontrast-20261104). N-3 r.22 e testo vecchio letterali.

### F3 (ERRORE) R-2 e Prompt: "preflight_plan.json `6081832...`"
- 6081832627... e' lo sha256 di plan.json (pinnato in preflight_plan.json science_plan.sha256, r.9). preflight_plan.json ha sha256 ea2cef0fad21... (identico in 0efdc29c, 65bc6022, afb89b45, 438c8d65).

### F4 (IMPRECISIONE) Limiti: "Lacune del critico ... (R-1...R-3, N-3)"
- N-3 non e' una lacuna del critico: e' P6-N3 del revisore p6, confermato da uno scettico (verify-p6-diagnostica-P6-N3-1.md, NON CONFUTATO). Le lacune del critico (9) finiscono in R-1 (2,3,8), R-2 (1), R-3 (4,5), N-5 (6), N-2 (7), M-1 (9).

### F5 (IMPRECISIONE) Tabella Esito: N-2 "(punti 1 e 2)" non compare nelle righe 1 e 2 della tabella.

### F6 (ERRORE) Verifiche positive 2: "19 job e 16 release storici"
- p2-probe_b.out.json: 21 voci "3a build_command(...)", tutte rifiutate (21 file *.job.json in releases/). 16 release ok. Il 19 viene dal rapporto p2 (r.18), la cui stessa lista audit/calibration/cuda/benchmarks(5)/mmap(6)/bf16(6)/readout(1) somma 21.

### F7 (IMPRECISIONE) Verifiche positive 2: calibrazione 64701482 "alla guardia delle sorgenti"
- Nel percorso di codice (calibration_report/authenticate_calibration/calibrated_configs) la calibrazione storica si ferma prima alla barriera di ricetta: "calibration run config differs from the frozen 512-batch recipe" (probe 2c/2d/2e; test_unmodified_historical_calibration_does_not_qualify_new_targets con match). La guardia sorgenti la rifiuta solo se chiamata direttamente (2a/2b). Vale anche per M-1 "fermate prima dalla guardia delle sorgenti" (vero per l'audit, non per la calibrazione).

### F8 (IMPRECISIONE) M-1: "due scettici confermano" (le tre barriere)
- I due scettici di P2-M1 hanno riprodotto solo le due barriere config_sha256 (validate_audit, read_preflight: X1/X2 o MD+MG, 337 passati). Nessuno dei due ha provato la guardia della ricevuta in authenticate_calibration (MF): solo il revisore p2 (MD+MG+MF 337/337) e il critico (lacuna 9).

### F9 (IMPRECISIONE) R-3: "Tutti i checkout storici della linea sono partiti con --cpus-per-task=4 --mem=2G --time=00:04:00"
- 6 su 7: CONTRAST-CHECKOUT-AUDIT-v2-20261117 con --time=00:05:00 (releases/audit_submission_plan_20261117.json; submission_20261117/checkout_state.json; contrastive_preflight_release_20261117.md:11 "4 CPU, 5 minuti"). Il critico lo dice ("uno con 00:05:00").

### F10 (IMPRECISIONE) R-4: "dentro l'OverTimeLimit ... E' cio' che accadde a 64652559: 3.787 s su 3.600"
- 3787-3600 = 187 s > 180 s di OverTimeLimit: finito 7 s oltre i tre minuti, nella finestra KillWait (contrastive_audit_results_20261117.md:53-55: «sette secondi oltre quei tre minuti»; accounting_and_integrity.txt r.1, 7, 9).

### F11 (IMPRECISIONE minima) R-3: core-ore cumulate "236,704445"
- Valore esatto 51,371111...+48,266666...+137,066666... = 236,704444 (resources.json; p5-preventivo.md riporta 236,704444). 236,704445 e' la somma degli addendi arrotondati.

### F12 (IMPRECISIONE) N-4: "riportati ai valori vecchi, la suite resta verde"
- Mutazioni M7/M6 rieseguite solo su sottoinsiemi (173+32, 64, 229, 314, 322 casi); suite completa esplicitamente non eseguita (verify-p7-...-F1-1 r.29, F1-2 r.40); il resto e' grep.

### F13 (IMPRECISIONE minore) Verifiche 3: "Ogni differenza fra i piani e' un'aggiunta dichiarata"
- Oltre alle aggiunte, il campo `scope` di plan.json e' riscritto ("Design and implementation for Claude review, no submissions" -> "New RMS100/fast125 preparation; ..."); limitations: +2 voci.

### F14 (IMPRECISIONE minore) riferimenti di riga senza file
- N-4 "(`:111`)" accanto a contrastive_prepare.py:30: la riga 111 e' in scripts/contrastive_benchmark_readout.py (in contrastive_prepare.py la 111 e' vuota). Punto 4 "(`:349`)" senza file: e' scripts/contrastive_prepare.py:349.

### F15 (IMPRECISIONE minore) Perimetro: "438c8d65: aggiunge solo evidenze"
- Aggiunge anche docs/prompt_claude_contrastive_targets_20261118.md (+34), 12 righe di docs/decisioni.md, riordina il registro e modifica local_validation.json (full_cpu_suite "pending" -> ricevuta, + independent_preparation_review). Nessun file in src/scripts/tests (vero).

### F16 (IMPRECISIONE minore) R-1: "le cinque celle di training (10 GPU-ore)"
- Sono le cinque celle di benchmark (3 full + 2 split, mode full/split); il preflight dice "training_release: Absent".

### Verificati OK (selezione)
- 9 celle/dipendenze vs vecchio preflight; split_benchmark invariato da 8905e346; readout:395 rel 1e-4 abs 1e-6 total/grad_norm; amp_bf16 True; 146/135 e citazione GPT r.9421 letterali.
- Bracci nuovi differiscono solo per contrastive.arm; 4 config nuovi vs storici: solo 2 finestre + target_version.
- contrastive_job.py:62-69, 22-32, :17; audit senza release (r.103-104); probe del critico rieseguita: pin review 17/11 e config accettati, stop su FileNotFoundError.
- campaign_checkout.sbatch:5-7 = 1 CPU/2G/10 min; preflight checkout 4/30/240; override obbligatorio solo audit; 138,666667 vs 137,066667; billing 32/8/4.
- cumulato 5,284444/51,371111 + 6/48,266667 + 14/137,066667 (resources.json); OverTimeLimit 3 min, KillWait 180 s; 64652559 TIMEOUT 3787 s; stima 3.900-4.000 e 6.032 s (p5).
- contrastive_run.py:100; readout:333-334; prepare:384-385; messaggi «audit stream/config differs», «different inputs/config/source», "current execution source inventory" esistono.
- 337 test contrastivi raccolti (20 file); 173+32=205 passati rieseguiti da me; collect-only tip: 3723 nodi, 233 file, identici a collection.log; 131 nodi distinti non passati, 13 file, 263/263; impronte di verification.json corrette; log non pinnati.
- protocollo r.132 par. 6 «controllo descrittivo di compatibilità» letterale; tre documenti operativi con 0 rimandi; N-3 testo db66b507 r.22 e driver break su "failed".
- evaluation identica; contrastive_compare/cuda_check invariati; coefficiente 0,1179564123141922 (gradient_reference.json); formula :348-349; 64701482 λ 0,11942686/0,33044627.
- Leonardo 10:07: nessun ref deploy/contrastive-common-*, checkout e uscite assenti; ledger fermo al 18/11 18:37.
- 38 agenti = 7 revisori + 30 scettici + 1 critico; votesFor 3/2/1.

## Sintesi finale
- ERRORE: F2 (run.py non identico alla punta), F3 (6081832... e' plan.json, non preflight_plan.json; anche nel Prompt), F4 (N-3 non e' lacuna del critico, ha avuto uno scettico), F6 (21 job storici, non 19).
- IMPRECISIONE: F1, F5, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16.
- Circa 115 affermazioni controllate; nessuna modifica a file della copia (git status pulito), basetemp cancellate.
