# r2-ripresa — Punto 2, ripresa a loss attive (fbfde797)

Inizio: 14:03

## 14:20 test esistenti
- tests/cpu/test_contrastive_target_resume.py a fbfde797/0efa4d45: 18 passed in 15,46 s (copia pulita dopo l'esecuzione).
- Il test "live" usa un solo processo non distribuito (ranks=1): non esercita rank_rng, DDP, barrier, gloo/nccl.
## 14:35 lettura del codice (a, c)
- (a) keep_checkpoints: calibrated_configs aggiunge 544 solo per endpoint 544/576 (contrastive_prepare.py:404-405); validate_cohort rimette [5000, 19618] per 5000/19618 (contrastive_run.py:86-87); piano: benchmark_keep_checkpoints [544, 5000, 19618]. Ricetta review CN: ckpt_every 500, eval/fixed/panel 500, prefetch True, log_batch_origin True, grad_dense 513-576 (tutti i 32 update ripresi hanno il pannello denso dei gradienti).
- (a) fork: trainer save(names) a 544 scrive checkpoint.pt e lo copia byte per byte in checkpoint_step0000544.pt (run.py:1589-1604, 1790-1795), il processo continua. Reload: prepare_reload copia con open("xb") e verifica SHA sorgente/copia, poi barrier e SHA su ogni rank (contrastive_run.py:341-349); il trainer riprende da out_dir/checkpoint.pt.
- (c) historical_numeric_pair confrontata riga per riga con compare_resume a 438c8d65 (readout:159-198): stesso schema, buffer esatti, _tensor_distance con riferimento initial e pavimento 1e-12, _nested_equal(.01) per optimizer e contrastive_state, esatti per stream/config/data_signature/pose_stats/provenance/initial/alarm_counts/rank_rng. _tensor_distance e _nested_equal non cambiano fra 438c8d65 e fbfde797 (il diff del readout parte da _run_evidence). RELATIVE_LIMIT .01 = RESUME_RELATIVE_DISPLACEMENT. Unica differenza: eccezioni esatte diventano pair failed (FAILED_REQUIRES_REVIEW) invece di eccezione: fail-closed.
- validate_target_fork (contrastive_run.py:247-266) non ha alcun test (grep in tests/cpu).
## 14:50 esperimento CPU a 4 rank gloo (trainer reale, DDP, prefetch, muscle_dropout .4, fixed_eval ogni 2, fork 4, fine 8)
Script: rev_runtime/tmp/r2_scripts/multirank.py; clone temporaneo r2_clone con UNA sola modifica di prova: env R2_CPU_SIM salta i due controlli che esigono cuda_rng (validate_fork:81, compare_three_paths:378), impossibili su CPU.
- Senza mutazioni: status PASSED, 0 boundary mismatch, 0 differenze di traccia; 4 righe per rank e fase. continuo/A e continuo/B: studente 3,50e-07, teacher 3,99e-06 (relativi allo spostamento dall'init); A/B: 0,0 esatto.
- Prima divergenza continuo/reload all'update 5 (primo dopo il fork): total identica, grad_norm 147.1406308099424 vs 147.14062985990338; A e B identici. Tracce DDP: nel continuo bucket gia' ricostruiti (iteration 4), nei reload ricostruzione solo dall'update 7 (iteration 2). Riproduce su CPU la causa localizzata nei diagnostici storici: l'ordine della riduzione cambia fra processo vivo e processi nuovi anche senza alcuna non-determinazione numerica.
- Mutazioni in reload_b (tutte rilevate, FAILED_REQUIRES_REVIEW):
  - Adam exp_avg azzerato dopo il load: boundary optimizer_moments su 4 rank.
  - RNG CPU del rank 2 non ripristinato: boundary rng_cpu_before (reload_b, rank 2), 8 differenze di traccia, rank_rng.2.torch_rng esatto diverso.
  - LR spostato di un passo (warmup 10, quindi LR variabile nella finestra): boundary optimizer_groups_at_next_lr su 4 rank + lr metadata finale. (Con LR costante nella finestra la stessa mutazione e' innocua: PASSED, atteso.)
  - coda senza la voce piu' vecchia: boundary contrastive_state + 8 differenze di traccia.
  - teacher x(1+1e-6) dopo il load: boundary teacher su 4 rank; il solo cancello numerico finale avrebbe dato 0,91% < 1% (passato). Il controllo esatto al confine e' quindi indispensabile ed e' efficace.
- Osservazione: un errore di confine in reload_b rende passed=false anche la coppia continuo/reload_a, con failure_reasons e differenze vuote (pair["passed"] AND globale di boundary_errors/support_missing, contrastive_target_resume.py:412).

## 2026-11-20 03:02 ripresa dell agente (sessione precedente interrotta dopo le 14:50 del 19/11)
- Copia di riferimento ancora a 0efa4d45 e pulita; r2_clone con la sola modifica R2_CPU_SIM in contrastive_target_resume.py (righe 81, 378). Continuo da (b), (c) residuo, (d), (e); riverifico i punti chiave.
## 03:40 equivalenza della regola numerica (c), prova incrociata
- Script rev_runtime/tmp/r2_scripts/equiv.py: carica compare_resume dal file di 438c8d65 (git show in r2_equiv/scripts/readout_438.py) e historical_numeric_pair di fbfde797; stessi checkpoint finali CPU (step forzato a 576 solo per il vecchio controllo d'ingresso).
- 7 esperimenti x 3 coppie = 21 confronti: esito, distanze studente/teacher (fino all'ultima cifra) ed elenco dei tensori oltre l'1% IDENTICI; dove il vecchio solleva (rank_rng.2.torch_rng differs exactly, optimizer.param_groups.0.lr metadata differs) il nuovo solleva lo stesso messaggio (poi convertito in coppia fallita da compare_three_paths).
- teacher_noise (x(1+1e-6)): continuo/B teacher 0,909% e queue_age: entrambe PASSANO la regola numerica (vecchia e nuova); le blocca solo il confronto esatto al confine/tracce. Conferma che il confine e' indispensabile.
## 03:45 dimensioni e tempi (d)
- 28.559.133 parametri; state_dict 114.270.852 B per modello; Adam 2 x 114.270.772 B; coda ~5,8 MB (calibrazione storica 234.339.066 B = 2 state_dict + coda). Checkpoint full ~463 MB.
- Scritture della coorte per full+reload: C0/CN/CR ciascuno checkpoint.pt + checkpoint_step0000544.pt (anche C0: calibrated_configs aggiunge 544 a tutti i bracci) = 3 x 0,93 GB; 4 reload x 0,46 GB; .tmp transitori 0,46 GB per salvataggio (max 2 job dbg in parallelo). Totale ~4,6-5,6 GB < 12 GiB.
- Storico bf16 auto: full 638-742 s (trainer), 661-775 s (sacct); resume storico 32 update 96,4-96,7 s nel trainer. Reload nuovi: ~100 s + convalide; ampio margine su 1770/1800 s. Full: margine ~1000 s; serve un rallentamento math >2,3x per sforare (non misurato).
- verify_reviewed_sources (125 file, git show ciascuno): ~2,2 s per chiamata sul Mac; si esegue su ogni rank prima del controllo out_dir.exists().
## 03:55 percorso di produzione con diagnostica densa (b)
- Ricetta reale: grad_dense 513-576 (tutti i 32 update tracciati calcolano global_gradients dentro _accumulate_terms), warmup 1962 (LR varia nella finestra: 2,7727e-05 -> 2,9358e-05), fill/ramp 256/256, coda 1024/256.
- Nuovo esperimento 4 rank gloo, trainer reale, R2_DENSE (grad_dense 1..8) + warmup 10: PASSED, 0 confini diversi, A/B 0,0, continuo/reload 5,55e-07 (studente); contrast_global_grad_base presente in 5..8 in tutti e tre i percorsi. Una sola chiamata di _accumulate_terms per update anche con la diagnostica densa: la traccia non si rompe.
- pytest: tests/cpu/test_contrastive_target_resume.py + test_contrastive_targets_readout.py: 41 passed in 29,76 s (copia pulita).
## 04:00 rilievi candidati
- MINORE: readout:489 sovrascrive check['evidence_sha256'] (impronte di trace-{rank}.jsonl, start.json, origin e del continuo calcolate in contrastive_target_resume.py:347-361,423) con i soli pin dei due reload: la ricevuta pubblicata non lega i 12 file di traccia per braccio.
- MINORE (plausibile, non riproducibile senza NCCL): controllo out_dir.exists() su ogni rank prima di init_process_group; con torchrun (agent store) e NCCL pigro di torch 2.2 nessuna sincronizzazione; nel full il primo rank crea out_dir (contrastive_run.py:352/358) mentre un rank in ritardo nella convalida (ora con 125 git show per rank) puo' ancora eseguire il controllo -> ValueError prima delle misure. Nei reload la finestra e' molto piu' larga a favore (rank 0 rifa' validate_fork prima del mkdir).
- NOTA: pair['passed'] = AND con errori globali di confine/supporto: una coppia sana risulta passed=false senza motivi propri.
## 04:10 validate_target_fork (non coperta dalla suite)
- Script rev_runtime/tmp/r2_scripts/vtf.py: fork sintetico 544/576 (dal fork CPU reale con step/config/provenienza/coda/rank_rng adattati), source_digest reale della copia.
- 9 casi: ok ACCETTATO; STOP, summary 575, world_size 1, braccio diverso, nome 543, cartella diversa dal summary: RIFIUTATI (complete matching continuous576); cohort diverso: RIFIUTATO (provenance); SHA del pin errato: RIFIUTATO (checked_pin). Logica corretta.
## 04:25 prova della sovrascrittura delle impronte (readout:489)
- rev_runtime/tmp/r2_scripts/test_r2_evidence_overwrite.py (riusa readout_fixture, compare_three_paths sostituito da uno stub che restituisce le impronte di trace-0..3 e start.json): 1 passed. Prodotto: fasi continuous/reload_a/reload_b con start.json e trace-0..3; pubblicato nella ricevuta: solo reload_a/reload_b con le chiavi di _run_evidence. Le impronte delle tracce e del continuo spariscono.
- test_contrastive_job.py + test_contrastive_run.py in clone temporaneo: 44 passed in 5,84 s.
## 04:30 pulizia e chiusura
- Rimossi r2_clone (patch R2_CPU_SIM salvata in r2_scripts/r2_cpu_sim.patch, 22 righe), r2_tclone, cartelle basetemp di pytest, r2_vtf. Restano solo gli esiti piccoli r2_exp_* (~50 MB) e gli script. Copia di riferimento pulita.
- Esito: nessun BLOCCANTE nel punto 2. MINORE: (1) corsa out_dir.exists()/mkdir fra rank senza barriera dopo init (torchrun + NCCL pigro), (2) readout:489 perde le impronte delle tracce. NOTE: passed per coppia congiunto con errori globali; copertura dei test (un rank, pannello denso spento, validate_target_fork e percorso del runner non testati); tracce DDP non lette ne' legate.
