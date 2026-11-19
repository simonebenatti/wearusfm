# Scettico n. 1 — rilievo P5-N4 (residuo del budget e cumulato)

HEAD 438c8d65; emendamento e preflight_plan.json invariati da 0efdc29c (unico commit che li tocca).

## Riproduzione
- Assenza: grep di `5.28|51.37|11.28|99.63|129.33|cumulative|campaign_cap|budget|residu|saldo|consuntiv` in
  docs/contrastive_targets_amendment_20261118.md, plan.json, preflight_plan.json, local_validation.json, README.md:
  solo preflight_plan.json:194 ("available balance") e amendment:81 ("quota e saldo reali"). Nessun cumulato, nessun residuo.
  Anche registro e decisioni del commit 0efdc29c: "Preventivo del solo preflight 14GPUh/137,066667coreh".
- Nessun codice calcola cumulati o tetti contrastivi (grep gpu_hours_cap/core_hours_cap in contrastive_*.py: nulla).
- Consuntivo vecchia catena ricalcolato dai sacct: 9 job GPU (774+67+127+141+661+683+769+775+759 s) x 4 GPU / 3600 = 5,284444 GPUh,
  uguale a accounting.txt:20 e resources.json before_diagnostics_used.
- Somma esatta (Fraction): 1189/225 + 6 + 14 = 25,284444 GPUh; 23117/450 + 724/15 + 2056/15 = 236,704444 coreh.
- Il nuovo tetto non entra nel residuo del vecchio (14 - 11,284444 = 2,715556 GPUh): e' per forza un tetto nuovo, non dichiarato come tale.

## Motivi cercati per confutare (non sufficienti)
- La regola (CLAUDE.md:262-264) vale per la richiesta di conferma, che e' futura: il piano la impone
  ("publish and present resource proposal to Simone"). Ma la sezione si intitola "Preventivo da presentare prima degli invii":
  e' il testo da presentare, e non contiene il residuo.
- "available balance" copre il saldo del conto, non il budget del passo ne' il cumulato.
- Prassi della stessa linea: docs/contrastive_common_diagnostics_20261118.md:53 e resources.json dichiarano il cumulato
  (11,28/99,64 su 14/129,33); decisioni.md:7911 e seguenti riportavano sempre "massimo totale con consuntivi ... entro 14/129,333333".

## Precisazioni
- Il "cumulato della linea 3" e' solo del preflight contrastivo: la linea 3 comprende anche T100-LQ
  (decisioni.md:7751: training 17,065556 GPUh, estrazioni 1,739444 GPUh, CPU 13,143333 coreh).
- "<=" vale sui tetti richiesti: l'audit vecchio ha sforato (decisioni.md:7867, 8,415556 coreh su 8).
- La parte "saldo del conto" della correzione e' gia' nel piano (preflight_plan.json:194, amendment:81).

## Esito
Non confutato. Gravita' NOTA (documentazione della presentazione; nessun invio possibile senza la presentazione a Simone).
