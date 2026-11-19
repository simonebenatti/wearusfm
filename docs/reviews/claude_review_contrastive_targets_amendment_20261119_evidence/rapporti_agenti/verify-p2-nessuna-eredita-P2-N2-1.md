# Verifica scettica P2-N2 (scettico n. 1)

Rilievo: la non-eredita del TIMEOUT e solo dichiarata (NOTA).
Copia: 438c8d65. Inizio lavoro (prova riprodotta con comandi propri).

## Riproduzione (script mio tmp/skep1_p2n2/scripts/probe_timeout.py, output tmp/skep1_p2n2/probe_out.json)
- Schema del rapporto reale 5c720fad (job 64652559, TIMEOUT): 15 chiavi di primo livello, nessuna di tempo/stato/job; nessuna nei record. contrastive_audit.py:99-105 e 181-185 non scrivono durata o stato.
- Audit sintetico per la ricetta nuova (sorgente corrente, config 100/125/v3, input coerenti): validate_audit ACCETTA; con marcatori espliciti slurm_state=TIMEOUT, elapsed_s=7386 ACCETTA ugualmente (campi ignorati).
- validate_release(stage calibration) con quell'audit e una release scritta a mano: ACCETTA.
- read_preflight (a monte simulato): supera il cancello dell'audit (r.320-334) e arriva a _cuda_evidence.
- Controllo non vacuo: config_sha256 sbagliato -> rifiutato da entrambi i cancelli.
- grep in scripts/contrastive_*.py, sbatch contrastivi, lib_code_dir.sh, src/.../contrastive_*.py e test contrastivi: nessun audit_recovery/APPROVED_EXACT/slurm_state/sacct/TIMEOUT; elapsed solo per benchmark (risorse) e rank-exit (limite 60 s).
- Fatti del rilievo confermati: contrastive_job.py:104 comando audit senza limite interno; launcher CPU passa solo TimeLimit (scontrol) e il dispatcher verifica 7200 = allocazione, non durata; accounting_and_integrity.txt:1-2,9 (TIMEOUT, .batch COMPLETED 0:0, 3787 s, OverTimeLimit 3 min).

## Ricerca di motivi per confutare
- Controllo altrove? Nessuno. Il passaggio audit -> calibrazione richiede una release APPROVED_FOR_PREPARATION scritta a mano (audit_sha256, review_pin). Nessun builder di release nel codice. afterok sull'audit non e' praticabile: la release pinna lo SHA del rapporto, quindi l'audit e' gia' concluso prima del checkout; la storia (contrastive_calibration_release_20261117.md:52-53) vieta afterok verso job TIMEOUT o gia' scaduti. submit_job.py --dopo serve solo da afterok.
- Percorso non di produzione? No: audit -> release di calibrazione -> validate_release/validate_audit -> read_preflight e' il percorso previsto da preflight_plan.json (calibrazione depends ["audit"]).
- Dichiarazione esplicita? Esiste, ma e' la premessa del rilievo: preflight_plan.json:196 ("no ... time extension"; "Archived audit TIMEOUT exception is not inherited") ed emendamento r.35. Il requisito "job principale COMPLETED 0:0" e' scritto solo nei documenti storici (contrastive_calibration_release_20261117.md:21). Il piano nuovo non lo ripete e non dice chi controlla sacct. Emendamento r.75 "controlla il tempo effettivo": nel codice c'e' solo il controllo dell'allocazione (launcher r.20-22 TimeLimit da scontrol; contrastive_job.py:67). Frase ambigua.
- Design dichiarato: contrastive_preflight_execution_20261116.md:57 "il walltime SLURM/cgroup e' il limite esterno ... se non basta il job fallisce e si legge il log". Con OverTimeLimit il job fallisce (TIMEOUT) ma il rapporto esiste e il log e' pulito (slurm.out storico senza riga CANCELLED). Resta il solo sacct, letto dall'operatore. Storicamente l'ha fatto: verification.json, execution_gate FAILED_TIME_LIMIT, calibration_released false.
- Asimmetria: le fasi GPU hanno un limite interno (1770 s, run.py:1609) e i lettori esigono stopped=max_steps; rank-exit rifiuta elapsed_s>60. L'audit non ha limite interno (contrastive_audit.py senza opzione di tempo) ne' campo di durata.

## Precisazioni sul rilievo
- 'OverTimeLimit = 3 min' e' in accounting_and_integrity.txt:9, non in slurm.out (0 occorrenze). Citazione imprecisa, fatto vero (snapshot del 17/11).
- La non-eredita' del record storico e' meccanica: l'audit 5c720fad e' rifiutato da guardia sorgenti e config, audit_recovery non e' letto da nessun codice. E' solo procedurale il rifiuto di un audit NUOVO in TIMEOUT. Lo era anche nella catena storica.
- Sulla correzione: un elapsed_s misurato nel processo esclude module load, venv e dispatcher, quindi non coincide con l'Elapsed SLURM. Piu' fedeli: un limite interno che fallisce chiuso sotto 7200 s (come 1770/1800 delle GPU) oppure il pin di sacct State=COMPLETED nella release di calibrazione.

## Esito
NON CONFUTATO. Fatti riprodotti con prova mia. Gravita' NOTA confermata. Nessuna accettazione automatica: serve la release manuale con review_pin. Il rapporto scritto nel margine e' completo per contenuto (open 'x' dopo i controlli d'integrita'). Nessuna evidenza vecchia accettata e nessun tetto superato dimostrato. Il divario e' fra politica dichiarata e applicazione meccanica, e il piano nuovo non scrive esplicitamente il requisito COMPLETED 0:0 ne' chi lo verifica.
Non verificato: OverTimeLimit attuale su Leonardo (niente ssh), durata reale dell'audit 100/125, consegna dei segnali SLURM (dedotta dall'accounting storico).
Pulizia: work sintetico (52M) rimosso, basetemp pytest rimosso, copia git pulita a 438c8d65. Restano script e output in tmp/skep1_p2n2/.
