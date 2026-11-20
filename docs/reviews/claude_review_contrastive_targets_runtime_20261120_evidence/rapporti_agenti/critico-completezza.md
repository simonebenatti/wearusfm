# Critico di completezza — pacchetto contrastive targets (fbfde797 / 0efa4d45)

Avvio: 2026-11-20 05:20

## Progressi
- Letti: prompt di GPT (docs/prompt_claude_contrastive_targets_runtime_20261119.md) e i sette rapporti r1..r7 con le verifiche avversarie (elenco in reports/). Copia di riferimento a 0efa4d45, pulita.
- Metodo: per ciascuno dei 5 punti del prompt confronto cio' che GPT chiede con verified_ok/not_checked dei revisori; cerco affermazioni del prontuario non verificate, file non letti, percorsi non coperti e rischi di cluster; faccio controlli rapidi di sola lettura.
- Controllo rapido 1 (audit 7.200 s, ricetta 100/125): anchor_targets multiscala sul Mac, segnale sintetico (cc_scripts/time_targets.py): costo 25/50 -> 100/125 x1,14-1,18 (fs 1-4 kHz). Storico 64652559: lavoro finito entro 3.787 s (.batch COMPLETED, allocazione TIMEOUT). Stima peggiore se tutto fosse target: ~4.470 s < 7.200 s (margine ~1,6x). Nessun revisore l'aveva stimato.
- Controllo rapido 2 (modello): dim 384, 6 teste, K=64, 8 livelli backbone, 2 decoder, model.grad_checkpoint True (config storico resume_CN del 19/11). Attenzione con attn_mask booleana (backbone.py:59, query_decoder.py:48): in math la matrice d'attenzione si materializza; i commenti del codice stimano ~1,3 GB/livello e ~630 MB/blocco/finestra se salvata. Con il checkpointing la memoria resta limitata a un blocco alla volta; nessuna misura math a batch 32 (oracolo CUDA e diagnostica a batch globale 8).
- Controllo rapido 3 (diagnostica storica faa4a8d7:.../common_results_20261119/cuda/bf16_math.json): gradienti identici sui 6 casi ma LOSS diverse fra repliche (C0a 7.261735688894987, CNa 7.261735825985671, CRa 7.261735766381026); in bf16_math_deterministic loss identiche. Quindi nel profilo candidato il forward non e' deterministico (index_add float: anchors.py:65, jepa.py:165-168, commento jepa.py:26); l'identita' dei gradienti a batch 8 regge solo se quei valori non entrano nel gradiente. A batch 32 non e' garantita: rischio non valutato da nessun revisore.
- Controllo rapido 4 (corsa out_dir, R2-M1): 4 processi concorrenti con verify_reviewed_sources (127 git show) sul Mac: scarto di arrivo 0,00 s in 3 prove (cc_scripts/skew.py). Sul Mac non si riproduce; su Lustre lo scarto non e' misurato. La finestra di rank 0 (exists -> mkdir) per i full e' ~ inizializzazione CUDA, simile allo storico (mkdir a run.py:1357): i 9 job GPU del 17/11 non l'hanno incontrata.
- Controllo rapido 5 (orchestrazione): checkout = git worktree del canonico (campaign_checkout.sbatch), quindi fbfde797 e' leggibile da git show se il deploy discende da 0efa4d45; un solo AdamW nel trainer (run.py:1429), quindi il vincolo "unique production AdamW" della traccia regge anche in produzione; torchrun --standalone (contrastive_job.py:86).
- Controllo rapido 6: config della coorte (build_review_configs): batch 32, grad_checkpoint True, region_target False, RMS 100, fast 125, v3, finestre 1-4 s (P <= 160 patch). Test preesistenti modificati (benchmark_readout, historical_calibration/callsites, mmap, run): solo campi aggiunti, guardie nuove sostituite da stub e un match piu' stretto; nessuna asserzione indebolita.
- Controllo rapido 7: l'audit registra per step eligible_nce, rnc_windows, rnc_pairs (contrastive_audit.py:70-73, chiave "records"): il supporto a 127/255 e 544-575 si puo' verificare prima della spesa GPU; nessun codice lo richiede. L'audit non ha limite di tempo interno.
- Controllo rapido 8: nessun generatore di release APPROVED_FOR_PREPARATION ne' di accounting (grep: solo validatori, contrastive_job.py:134, contrastive_run.py:168, contrastive_release.py:164); sacct letto solo per l'audit; pin di decisione/budget solo per il training (contrastive_run.py:160).

## Esito: lacune (sintesi; dettaglio nell'uscita strutturata)
1. P1/P4 memoria e tempo math a batch 32: nessuna stima di memoria dai revisori; mia stima analitica (checkpointing attivo) +1-2 GB su 13,72 GiB e f ~1,1-1,2; prima misura reale = calibrazione.
2. P1 evidenza storica non letta: in bf16_math loss diverse fra repliche ma gradiente unico; identita' per byte strutturale finche' le somme atomiche entrano linearmente (region_target False).
3. P1/P2 primo uso NCCL di osservatore, traccia e prepare_reload nelle celle di produzione; l'oracolo CUDA non li esercita (evidenze storiche parziali favorevoli: chiavi DDP in torch 2.2, confine RNG combinato identico).
4. P2 fattibilita' dell'1% per tensore continuo/reload con asimmetria dei bucket DDP non rimossa dal profilo math: non stimata.
5. P1/P2 supporto a 127/255 e 544-575 verificabile dall'audit e non richiesto.
6. P3 audit entro 7.200 s con 100/125: stimato da me, margine ~1,6-1,9x (target x1,14-1,18).
7. P3 nessun controllo preliminare dei job prima di sbatch (make-job non valida config/inputs/roots; --validate-only non prescritto) con un tentativo per ID.
8. P3 release di esecuzione, evidenze, coorte e accounting senza generatore; percorso positivo provato solo con stub.
9. P3/P4 G-1: non valutata la separazione audit CPU / fasi GPU; una condizione nella review non e' vincolante (nessun cancello di budget per la preparazione).
10. P4 consumo effettivo e limiti GPU senza sacct; avvio fuori dai 1.770 s; full nell'OverTimeLimit accettato.
11. P4 cammino critico rispetto alla manutenzione 23-25/11 non calcolato.
12. P5 orchestrazione di contrastive_run.main (326-372) mai eseguita; nessun read_preflight su uscite reali.
13. P1 ambiente non registrato oltre torch/CUDA (numpy/scipy, cuDNN, NCCL, driver, host); venv mutabile.
14. P2 R2-M1 non quantificata su Lustre (Mac: scarto 0,00 s; storico 9/9 con finestra simile).
15. API torch 2.2 fuori dal file di sonda (torch.load mmap, fork_rng, AdamW, is_bf16_supported prima di set_device) verificabili con un comando sul login node.
Nessuna modifica alla copia (status pulito a 0efa4d45); nessun clone creato; script in rev_runtime/tmp/cc_scripts.
