# Verifica scettica P5-N3 (n. 1) - vincoli di coda e semantica dell'override

HEAD 438c8d65, consegna e388a1de..0efdc29c. Solo lettura, nessun job.

## Fatti riprodotti (corretti)
- remote_evidence.txt:18-19 (cuda_completed_bf16_20261118) `Name|MaxJobsPU|MaxSubmitPU` / `boost_qos_dbg|2|4`.
- preflight_plan.json: celle 4-8 (3 full + 2 split) con depends ['cuda','calibration_accepted']; 5 > MaxSubmitPU 4.
- Corsia seriale: claude_GPT_shared:11841-11842 (MaxTRESPU cpu=8, mem 30800M, MaxJobsPU 2); logsquare_cpu_execution_20261107.md:78 "i fit da 8 CPU/30G procedono quindi uno alla volta".
- Manutenzione 23/11 08:00-25/11 08:00 in CLAUDE.md: un job da 2 h dopo ~06:00 del 23/11 resta PENDING.
- Dispatcher: pytest test_contrastive_job.py -k "fixed_resources or swapped_or_extended" -> 17 passed: audit 3600/7199/7201 rifiutati, solo 7200 ammesso; main() chiama build_command (contrastive_job.py:161) prima di validate_allocation (:162) e di ogni lettura dati -> senza override fallisce in pochi secondi.

## Motivi per cui non e' un difetto della consegna
1. (2) e (3) gia' scritti come controlli obbligatori prima del rilascio: emendamento:81 "Prima del rilascio: quota e saldo reali, scritture pendenti, corsia, manutenzioni e ledger"; preflight_plan.json:190-195 required_before_first_submission "check ledger, quota, maintenance, available balance; no duplicates". Effetto peggiore: attesa in coda, nessun costo, nessun fallimento.
2. (1) e' il formato di release gia' usato dalla stessa linea il 18/11: releases/bf16_20261118/submission_plan.json:144-162 `benchmark_groups` [[C0,CN],[CR,RESUME-CN],[RESUME-CR]] e `qos_observed` {boost_qos_dbg, 2, 4, must_recheck_all_user_jobs_before_each_group: true}; handoff_20261117.md:115 "a gruppi secondo piano e quota di tutti i job dell'utente (MaxJobsPU=2, MaxSubmitPU=4 da ricontrollare)"; review bf16:40 "Invii a gruppi contando tutte le linee". Anche il piano preflight storico (eseguito) non conteneva ondate: stanno nella release, che qui non esiste ancora (executable=false, "le celle non contengono release"). Controllo meccanico: un rifiuto MaxSubmitPU non crea job; submit_job.py:12 stato `rifiutato` -> `--dopo-rifiuto` (precedente logsquare:76-77). Nessun limite superabile.
3. Corsia condivisa gia' nota all'operatore della linea 3: handoff_20261117.md:110 "Checkout può attendere quota CPU linea 1". Interferenze fra linee: ruolo di GPT coordinamento (CLAUDE.md).
4. (4) semantica gia' determinata: override obbligatorio dichiarato (preflight_plan.json:187-188; emendamento:74-75 "richiede l'override ... e controlla il tempo effettivo"); un job creato e fallito non riusa l'ID (submit_job.py:12 `sottomesso` mai risottomesso) e l'emendamento:82-83 "Nessun checkpoint, ID o output esistente viene riutilizzato per un nuovo tentativo"; procedura per ogni fallimento: contrastive_preflight_execution_20261116.md:57 ("se non basta il job fallisce e si legge il log, senza alzarlo automaticamente") e :70 ("si archiviano evidenze e consuntivo e si richiede una nuova review mirata"); precedente di rettifica operativa senza evidenze (decisioni.md:7848-7850, preflight storico `operational_correction`). "No time extension" non puo' scattare: il dispatcher ammette solo 7200 s, quindi ripresentare con l'override del piano non estende nulla. La lettura piu' restrittiva ("gate fallito, nessun retry") porta solo a fermo e review: conservativa, nessun effetto della scala BLOCCANTE/MINORE.

## Esito
refuted=true. Numeri corretti ma nessun difetto della consegna: tre punti su quattro sono gia' requisiti scritti prima del rilascio o formato di release consolidato con limiti imposti da SLURM; il quarto e' risolto da submitter, emendamento e procedura documentata. Gravita' se reale: NOTA al massimo.
