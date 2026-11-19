# Scettico n. 1 - P5-N2 ("Totale massimo" senza tolleranza SLURM)

HEAD 438c8d65, solo lettura. Esito: **NON confutato. Gravita' reale: NOTA, marginale.**

## Riproduzione (comandi miei)
- `tail -4 .../audit_completed_20261117/accounting_and_integrity.txt`: KillWait 180 sec, MinJobAge 30 sec,
  OverTimeLimit 3 min, UnkillableStepTimeout 120 sec. Riga 1: 64652559 TIMEOUT, Elapsed 3787, Timelimit 60, billing=8;
  `.batch` COMPLETED 0:0.
- `releases/audit_recovery_20261117.json`: elapsed 3787, requested 3600, audit_actual_core_hours 8,415556.
- Ricalcolo con Fraction da `preflight_plan.json`: 13 allocazioni, 14 GPUh, 2056/15 = 137,066667 coreh;
  i cap coincidono esattamente con la somma dei limiti richiesti (margine zero).
- Tolleranza su ogni job, caso teorico: +180 s => +1,4 GPUh / +12,8 coreh; +360 s => +2,8 GPUh / +25,6 coreh;
  audit da solo +0,4 / +0,8 coreh.
- `scheduler_review7.txt`: boost_usr_prod e lrd_all_serial hanno `OverTimeLimit=NONE` a livello di partizione,
  quindi vale il valore globale di 3 min per entrambe.
- Storico: i due soli TIMEOUT nei risultati, 63593508 (5584 contro 5400 s) e 64652559 (3787 contro 3600 s),
  hanno sforato di 184 e 187 s. E' compatibile con OverTimeLimit piu' qualche secondo; un contributo di KillWait
  non e' dimostrato.
- `contrastive_job.py:118` expected_time 1800/1770; `:136` `--total-time-limit-s 1800`; `:138` `--time-limit-s 1770`;
  `contrastive_split_benchmark.py:108` deadline = 1800-30. Confermati.
- Durate GPU storiche: calibrazione 774 s; cuda bf16 141 s; bench 661/683/769 s; split 775/759 s. Lettura 124 s.

## Motivi contrari cercati (riducono il valore, non eliminano il rilievo)
- CLAUDE.md:146-148: "Per il preventivo si usa numero di GPU allocate per walltime massimo richiesto ... Registrare
  anche ... consuntivo". La somma dei limiti e' dunque la convenzione del progetto, non un errore di calcolo.
- Emendamento: la sezione s'intitola "Preventivo"; la tabella ha la colonna "Limite"; la riga 76 dice "Sono tetti
  proposti, non misure"; la riga 72 cita esplicitamente 3.787 s contro 3.600 richiesti.
- Linea 3 storica: preflight_execution_20261116.md:28 "tetti prenotati, non una misura di consumo";
  calibration_release_20261117.md:34-35 "Nessun consumo si nasconde nel margine"; decisioni.md:7868
  "OverTimeLimit ... non usarlo come margine garantito".
- failure_policy del nuovo preflight_plan: un gate fallito ferma i discendenti; niente retry ne' estensioni
  di tempo; l'eccezione TIMEOUT archiviata non viene ereditata. Copre gia' la seconda meta' della correzione
  proposta, come ammette il rilievo stesso.

## Perche' resta una NOTA
- Nessun testo nuovo dice che il consumo fatturato puo' superare il totale per la tolleranza di SLURM.
  L'elenco delle esclusioni alla riga 76-77 (training, diagnosi, retry, downstream) non la nomina.
- decisioni.md (voce del 18/11, in questa consegna) dice "costa al massimo 14 GPUh e 137,066667 coreh" e il
  README dice "massimo". Il 17/11, nella stessa linea, uno sforamento di 0,4156 coreh porto' la proiezione a
  129,461 contro 129,333 e richiese una rettifica dei checkout. Nella linea 1 lo stesso tema e' stato dichiarato
  in modo esplicito, con la colonna "Margine scheduler" (scaling_300m_cpu_recovery_execution_20261105.md:20-37).
- Nessun effetto operativo: un TIMEOUT ferma i discendenti, e ogni job completato consuma molto meno del suo
  limite (GPU storiche al massimo il 43%). Uno sforamento del totale richiede uno scenario patologico.

## Imprecisioni del rilievo (non ne cambiano l'esito)
- La cella cuda non ha una scadenza interna complessiva: `contrastive_cuda_check.py` ha solo il timeout NCCL di
  10 min e la sonda di 90 s. Le scadenze 1770 s e 1800-30 s proteggono 6 celle GPU su 7.
- "grazie a OverTimeLimit e KillWait": per l'audit e' provato solo OverTimeLimit (+187 s, batch COMPLETED 0:0).
  Il valore di 0,8 coreh e' un limite teorico.

## Suggerimento
Tenere la nota, accorpata alle altre. Una sola frase nell'emendamento e nella voce del 18/11: i totali sono la
somma dei limiti richiesti, non includono la tolleranza di SLURM (OverTimeLimit di 3 min); ogni TIMEOUT o
sforamento si registra nel consuntivo e ferma i discendenti, secondo failure_policy.
