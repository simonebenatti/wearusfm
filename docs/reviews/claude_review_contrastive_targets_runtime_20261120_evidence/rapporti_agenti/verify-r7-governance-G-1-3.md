# Scettico n. 3 per G-1 (r7-governance)

Avvio: verifica indipendente del rilievo G-1 (proposta faa4a8d7 e tetto 18 GPUh senza firma di Simone).

## Progressi
- Fatti documentali riprodotti (git show): faa4a8d7 si intesta «Decisione proposta a Simone» (r.3) e chiude «proposta da discutere prima degli invii» (r.52); voce decisioni «Proposta per Simone ... training e invii sospesi». fbfde797 decisioni.md:8280 «attua la decisione scientifica faa4a8d7»; :8292 «Preventivo proposto ... 18 GPUh / 169,6 coreh richieste».
- 438c8d65 preflight_plan.json:191 «publish and present resource proposal to Simone» in required_before_first_submission; a fbfde797 :315-319 sostituito da review/ledger/audit-only. plan.json:328 conserva «Prepare and present new costs before any launch».
- Ultimo via di Simone con tetto: 17/11 decisioni:7846 (14 GPUh / 129,3333), «per questo giro»; release 17/11: «Il budget approvato dal presente via riguarda questo collaudo».
- Nessuna citazione di Simone su budget 18 GPUh, profilo math, parita' su stato comune, fork 544 in 3cc74ad6/8e38eb71/faa4a8d7/fbfde797/0efa4d45 (grep sulle righe aggiunte).
- Codice riprodotto: contrastive_release.review_payload rifiuta chiavi extra ("conditions") e status diversi da APPROVED; ignora la prosa fuori dal blocco (probe locale: ok ACCEPTED, extra_conditions_key REJECTED, status_conditional REJECTED, prose_condition_outside_block ACCEPTED). make_job imposta APPROVED_FOR_PREFLIGHT (:236; test_job_builder_releases_only_the_fixed_reviewed_cell passa). contrastive_job.build_command: audit, cuda e readout senza file di release; calibration/full/resume con release APPROVED_FOR_PREPARATION senza pin di decisione o budget (test_stage_commands_... 6 passed). Pin scientific/budget solo per training (contrastive_run.py:160-161, preesistenti a 438c8d65).
- submit_job.py non applica tetti di campagna (solo --gpu-ore dichiarato). preflight_plan.json "submission_authorized": false non e' letto da alcun codice.

## Tentativi di confutazione (falliti)
1. Autorizzazione permanente 16/11 «launches after convergence» (plan.json:321): valeva per lo stesso piano col tetto 14 GPUh; il piano nuovo stesso dice «Prepare and present new costs before any launch» (:328), le intestazioni storiche dicono che le vecchie release «non autorizzano nuovi invii», il residuo e' «not_new_authorization»; CLAUDE.md:35 «Nessun lancio implicito»; CLAUDE.md:267-269 decisione registrata per superare il budget.
2. Mandati del 19/11 («Ti affido l'implementazione ... e la preparazione del pacchetto», «Gli invii ... spettano a 3-launcher») assegnano ruoli, non approvano tetto o criteri; faa4a8d7 (copiata byte per byte nel pacchetto) dice «proposta da discutere prima degli invii».
3. Autonomia 28/10 (<5 GPUh per job): copre l'audit CPU, ma richiede «rispettando i budget della campagna» e vieta il frazionamento; fasi full (6) e reload (8) superano 5 GPUh; nessun budget approvato per la coorte nuova.
4. Simone come inoltro fra chat: incidentale (tool GPT->3-launcher fallito il 18/11, invio diretto era previsto), non decisione registrata.
5. Attenuanti reali: CLAUDE.md:183 e' parzialmente superato dal 28/10; la parte "criteri non firmati" e' piu' debole perche' la R-1 di 94f5094 affidava la scelta a GPT scientifico con regola scritta prima; il pacchetto non attribuisce a Simone approvazioni false ("Preventivo proposto", "Tetto richiesto").

## Esito
Non confutato. Gravita': BLOCCANTE per l'emissione del blocco contrastive-release-review (non per l'approvazione del codice): con il blocco, la procedura documentata (prontuario r.121-147, prompt r.58-61, CLAUDE.md r.3) porta fino a 18 GPUh richieste senza decisione registrata di Simone, oltre il tetto approvato 14 (residuo 7,913333). Rimedio procedurale, nessuna modifica di codice necessaria.
