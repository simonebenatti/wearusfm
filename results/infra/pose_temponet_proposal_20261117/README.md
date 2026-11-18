# TCN ispirata a TEMPONet con ramo lineare — proposta revisionata18/11/2026

**REVIEW_REQUIRED. Non eseguibile: nessun codice, benchmark, job o deploy autorizzato.** Questa revisione documentale sostituisce c15aac9, dopo review Claude32e5452 e seguito8d34f9c.

- `proposal.json`: IDv2, cinque teste, ricerca finita, due controlli positivi train-only, WLS descrittiva, criteri e tetti proposti.
- `analytic_checks.json`: soli conti verificabili;103 traiettorie iniziali/max147, massimo prudente1.477.800 aggiornamenti, non misure di training.
- `raw_source_identity.json`: provenienza immutata; chiavi dei report raw ora complete.
- `design_review.txt`: letture storiche conservate, con avviso di superamento e nuova verifica documentale.

Protocollo: `docs/pose_temponet_frozen_protocol_20261117.md`;13 rilievi e50note: `docs/reviews/pose_temponet_response_20261118.md`. Prompt aggiornato: `docs/prompt_claude_pose_temponet_review_20261117.md`.

Ramo affine in MLP/TCNpost/TCNpre:73.000/72.904/72.904 parametri. Quattro LR iniziali/max6,150→300, selezioneMSE,3semi rifit. Due controlli positivi devono passare prima dei fit reali; nessun esito negativo dell'encoder da un controllo fallito.

**Tetto proposto24GPUh,80coreh CPU-only,272coreh allocate**, max8GiB output. Il benchmark deve ammettere anche tutte le estensioni. Nessun saldo/quota/hash remoto verificato oggi; input principaliWORK ma tre prerequisiti ancoraSCRATCH, dichiarati e pinnati. Vecchio tetto4/12/44 superato, non un'autorizzazione aggiuntiva.

Nessuna modifica a src/, scripts/, tests/ o config; nessun dato wearable, pesi o predizioni aggiunti al repository.
