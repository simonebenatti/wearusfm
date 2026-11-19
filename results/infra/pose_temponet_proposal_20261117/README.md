# TCN frozen: proposta v3 del 19/11/2026

ID `pose-temponet-frozen-encoder-readout-20261117-v3`. **REVIEW_REQUIRED**.
La review documentale di Claude richiesta riguarda il diff da `3e007c8`, dopo `ee118a25`.
Simone ha approvato il seguito del confronto nella chat coordinamento il 19/11; restano la vera
convergenza, la review del pacchetto operativo e i cancelli prima dei rispettivi invii.

- `proposal.json`: architetture, dati, criteri, calibrazione, ammissione e risorse.
- `delivery_pin_20261118.json`: consegna storica v2, conservata; non è il pin della v3.
- `delivery_pin_20261119.json`: pin della consegna v3 dopo il commit dei contenuti.
- `document_checks_20261119.json`: verifiche documentali; nessuna misura A100 o sui latenti reali.
- `analytic_checks.json`: distingue massimo raggiungibile e limite prudente del gate.
- `raw_source_identity.json`: provenienza della TEMPONet raw, senza importare quel ramo.

Fonti: `docs/pose_temponet_frozen_protocol_20261117.md`,
`docs/reviews/pose_temponet_response_20261119.md`. I riferimenti della v2 restano storici.
Runtime separato: `codex/pose-temponet-implementation-20261119`, file sotto scripts/pose_temponet.
Operatori: GPT scientifico prepara, Claude rivede, 2-TCN head launcher invia dopo i cancelli.
Nessun nuovo training del FM, loss o estrazione; cinque teste sulle cache T100-G 49.046.
