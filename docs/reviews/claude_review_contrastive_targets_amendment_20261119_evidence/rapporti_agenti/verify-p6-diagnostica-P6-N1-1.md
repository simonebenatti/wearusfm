# Verifica scettica n. 1 del rilievo P6-N1 (cancello diagnostic_resolution solo procedurale)

Avvio: 2026-11-19. Copia: clone @ 438c8d65 (sola lettura).

## Progressi (1)
- Letti: rapporto p6-diagnostica.md, preflight_plan.json (cella audit depends r.23-27, required_before_first_submission r.190-195),
  emendamento docs/contrastive_targets_amendment_20261118.md, scripts/contrastive_job.py (build_command r.62-69).

## Progressi (2) - riproduzione dei fatti
- `git grep -n 'diagnostic_resolution\|required_before_first_submission' 438c8d65 -- scripts src tests` -> exit 1 (nessun risultato): confermato
  che nessun codice legge il cancello. `contrastive_targets_20261118` e' citato nel codice solo da contrastive_prepare.py:30 (PLAN).
- preflight_plan.json:26 etichetta nuda; :190-195 voce generica "resolve historical bf16/... with separately prepared common-state
  diagnostics". C0.config.json:144 amp_bf16 true (ereditato da T100-G, contrastive_prepare.py non tocca amp_bf16). decisioni.md:7846 esiste.

## Progressi (3) - motivi per confutare
- Tipo di registrazione NOMINATO: emendamento r.49-51 "va letta la diagnostica da stato comune ... poi registrata la risoluzione del
  problema o un emendamento esplicito". La parte "ne' il tipo di registrazione" del rilievo e' inesatta.
- Identita' dei diagnostici univoca nello stesso albero: submission_plan.json frozen_commit 365cb1e2 e reports (3 percorsi);
  prompt_3_launcher r.3 (365cb1e2...) e r.8 (SHA-256 dei rapporti e commit delle evidenze da restituire); registro r.183
  (35ea94ba / 365cb1e2); README r.15-17 (non eseguirli da questo checkout, "i commit e i loro pin"). Unico pacchetto "common-state".
- Cronologia: 0efdc29c 22:10:32 (consegna), 0afd0864 22:26 (approvazione Claude, non antenato), db66b507 23:29 (scheda, altro ramo,
  non antenato). La consegna non poteva nominare scheda o approvazione; gli sha256 dei report non esistono ancora.
- I report si legano da soli al codice congelato: contrastive_common_state_cuda.py:238-239 (digest) e :333 source_sha256, :396
  training_approved False; contrastive_common_checkpoint.py:36/278.
- Applicazione del cancello nel percorso reale: nessun generatore scrive APPROVED_FOR_PREFLIGHT (solo contrastive_job.py:64 lo
  legge); job a mano in commit nuovo, review pinnata obbligatoria (contrastive_job.py:69), release con review_pin "converged review"
  (contrastive_run.py:177-178). Autorizzazione permanente solo "after convergence" (plan.json:308;
  contrastive_preflight_execution_20261116.md:11 "dopo la convergenza con Claude").
- Precedente 17/11: eccezione esplicita di Simone "per questo giro"; la release dice "sostituisce l'attesa della nuova review di
  Claude, non i cancelli tecnici" (reviews/contrastive_preflight_release_20261117.md:3); dopo, gate rispettati (audit TIMEOUT:
  "nessun nuovo job finche' la rettifica non e' registrata", decisioni ~7873; recupero solo col via esplicito di Simone).
- Tetti: probe tmp/p6n1v1/probe_precision.py -> config con solo amp_bf16=False rifiutato da validate_recipe ("configuration differs
  from the frozen comparison recipe"); 3600 s su cella full rifiutati ("wrong mode, release, source or allocation").
  Un cambio di precisione/determinismo richiede codice nuovo (generatore o src) -> digest nuovo != 643a8937 -> pacchetto e review nuovi;
  i tetti non si possono superare (ALLOCATED_SECONDS fissi, failure_policy senza estensioni) e la riproposta delle risorse a Simone e'
  gia' obbligatoria (preflight_plan r.191; plan.json target_windows.execution "present new costs before any launch").

## Progressi (4) - controlli aggiuntivi
- Stile di progetto: anche il preflight storico (contrastive_comparison_20261116/preflight_plan.json) usa etichette nude in depends
  ("calibration_accepted", "cuda", "all_five_benchmarks"); i pin stanno nelle release/job. Anche "cuda" prima dei benchmark non e'
  controllato dal codice al lancio (cuda_report solo nella lettura, contrastive_job.py:78). diagnostic_resolution segue lo stesso schema.
- Il lancio richiede job nel checkout congelato (contrastive_job.py:159), APPROVED_FOR_PREFLIGHT scritto a mano, review pinnata.
- tests/cpu/test_contrastive_job.py alla punta: 29 passed (1,18 s); basetemp cancellato; copia pulita (git status vuoto).
- Probe (tmp/p6n1v1/probe_precision.py, conservato): validate_recipe(calibration.config con amp_bf16=False) -> ValueError
  "configuration differs from the frozen comparison recipe"; build_command(full, 3600 s) -> "wrong mode, release, source or allocation";
  source_digest = 643a8937... = preflight_plan.source_sha256.

## Esito
refuted = true. Fatti di base riproducibili (etichetta nuda, nessun controllo nel codice, precedente del 17/11 esistente), ma:
(1) il tipo di registrazione e' nominato (emendamento r.49-51); (2) i diagnostici sono identificati in modo univoco nello stesso
albero (submission_plan frozen_commit/reports, prompt_3_launcher, registro, README); scheda db66b507 e sha dei report non esistevano
alla consegna; (3) cancello procedurale per costruzione e conforme allo schema storico, applicato dalla release nuova con review pinnata
e dall'autorizzazione "dopo la convergenza"; (4) il precedente e' un'eccezione esplicita di Simone che conservava i cancelli tecnici;
(5) un cambio di precisione/determinismo impone codice e digest nuovi, i tetti non sono superabili e la riproposta delle risorse a
Simone e' gia' obbligatoria. Residuo: al piu' un rimando incrociato mancante in preflight_plan (cosmetico). Gravita' se reale: NOTA.
