# verify-r4-risorse-R4-N2-1 — scettico n. 1 su R4-N2 (NOTA)

Inizio: lettura di contrastive_targets_plan.py, contrastive_job.py, contrastive_release.py, contrastive_run.py,
launcher contrastive_preflight_{cpu,gpu}.sbatch, lib_code_dir.sh, prontuario, test del piano, piano di base 438c8d65.

## Riproduzione dei fatti (copia 0efa4d45, sola lettura)
- preflight_plan.json: depends dei reload = ['full_CN'] / ['full_CR']; solo l'audit ha 'audit_checkout' (fallback `deps or [...]`,
  contrastive_targets_plan.py:102). checkouts: count 6, stages [audit, calibration, cuda, full, reloads, readout], un solo
  logical_id CONTRAST-CHECKOUT-reloads-targets-v3-20261118 (:103-107). CONFERMATO.
- Il job JSON deve stare nel checkout: contrastive_job.py:180 (is_relative_to(ROOT)), launcher: path relativo a CODE_DIR,
  checkout pulito senza file non tracciati e in sola lettura (lib_code_dir.sh:41-58). CONFERMATO.
- Il job di reload contiene i pin di fork_checkpoint e continuous_summary (contrastive_job.py:153-157), e la release richiede
  summary a 576 update, stopped=max_steps, stesso braccio (contrastive_run.py:246-258): il JSON del reload di un braccio
  si scrive solo dopo la fine del full di quel braccio. Con un solo checkout dei reload (6 checkout a piano), quel commit
  contiene i quattro JSON e quindi nasce dopo i full CN e CR. CONFERMATO (lettura del codice).

## Prove eseguite (clone temporaneo n2v1_clone a 0efa4d45, poi cancellato)
- `contrastive_job.py --job <file fuori dal checkout> --kind gpu --allocated-seconds 1800` ->
  `ValueError: job specification must be in frozen checkout` (contrastive_job.py:180).
- tests/cpu/test_contrastive_targets_plan.py: 3 passed (il test :15 asserisce solo depends == ['full_'+arm]).
- `contrastive_targets_plan.py --check`: ok; git status del clone vuoto.

## Motivi per cui NON e' un difetto
1. Il prontuario dichiara esplicitamente la barriera di fase: :181 «audit→calibrazione→CUDA→tre full→quattro reload→lettura»;
   :182 aggiunge al full del proprio braccio «e dal checkout con i pin544 appena misurati»; :87 un solo checkout reload;
   :187 «I sei checkout congelano solo le nuove evidenze tra fasi». Il piano: checkouts.stages in ordine e reason
   «...after ... full checkpoint ...» (contrastive_targets_plan.py:104,107). Il rilievo attribuisce a :182 la sola
   dipendenza dal proprio braccio: inesatto. La correzione proposta e' gia' nel prontuario.
2. `depends` non e' letto da nessun codice (grep: solo generatore e test); i comandi definitivi con afterok vengono emessi
   nella consegna post-review (prontuario :178-179). Le guardie (job nel checkout, pin del fork del proprio braccio,
   summary a 576) rendono impossibile un reload prima del suo full: nessun rischio di esecuzione scorretta.
3. Lo stile (solo l'audit nomina il checkout) e' ereditato invariato da 438c8d65: audit ['new_review','new_checkout',
   'diagnostic_resolution'], le altre celle senza checkout; review 94f5094 «stesse dipendenze». Parte storica.
4. Calendario: dbg MaxJobsPU 2, MaxSubmitPU 2 (decisioni.md:1395) o 4 (handoff 17/11): i tre full vanno comunque a ondate;
   inviando prima CN e CR il checkout unico non aggiunge attese rispetto a checkout per braccio; altrimenti l'attesa e' la
   barriera dichiarata. Nessun effetto su gate, budget, attribuzione.

## Esito
refuted = true. Fatti meccanici riprodotti, ma il comportamento e' dichiarato (prontuario :181-182, :87, :187; piano
checkouts.stages/reason), il campo depends e' descrittivo, ereditato e non consumato. Gravita' se reale: NOTA.
Fine. Cloni temporanei cancellati.
