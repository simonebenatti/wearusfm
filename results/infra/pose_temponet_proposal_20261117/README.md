# Proposta TEMPONet come lettore congelato della posa — 17/11/2026

**REVIEW_REQUIRED.** Non è un piano di lancio. Nessun runtime implementato, job o deploy. Encoder T100-G 30M teacher49k congelato, cinque nuove teste e due riferimenti storici.

- `proposal.json`: disegno, matrici, input e tetti proposti, non risorse autorizzate.
- `analytic_checks.json`: parametri, campo ricettivo, centri, fit e spazio calcolati; nessun test del modulo futuro.
- `raw_source_identity.json`: modello e script emg2pose identici fra il code_sha raw e la punta del ramo Claude; modello identico al commit di introduzione. Solo Git, senza merge.
- `design_review.txt`: tre letture indipendenti locali; precisazioni recepite. Non sostituiscono la review di Claude richiesta dal prompt.

Protocollo: `docs/pose_temponet_frozen_protocol_20261117.md`. Prompt di review: `docs/prompt_claude_pose_temponet_review_20261117.md`.

I pin storici vengono dalle ricevute e dalla conservazione pubblicata, non da una nuova verifica su Leonardo. Payload NPZ e checkpoint non sono aggiunti a Git. Il preventivo è un inviluppo da validare sul train, non una stima misurata: tetto4GPUh,12corehCPU-only,44coreh totali. Il gate impedisce fit parziali o riduzioni della matrice per farla rientrare.
