# Protocollo A — materiali della proposta, 07/11/2026

Stato `awaiting_claude_design_review`. Preparazione assegnata a GPT («A GPT»); nessun piano eseguibile, benchmark, fit o budget nuovo approvato. Operatore Claude. I file non sono ricevute runtime.

- `proposal.json`: disegno raccomandato a update costanti, modello, griglia, criteri e pin disponibili.
- `population_counts.json`: conteggi da quattro OOF scratch EPN15, verificate contro le ricevute; nessun identificatore individuale esportato.
- `resource_estimate.json`: proiezione per fold dai consuntivi dei candidati FT/scratch EPN15, con riuso condizionato di scratch100. Update costanti: 766,218889 GPUh; alternativa 15 epoche: 239,789367 GPUh. Costi CPU, benchmark e margini esclusi; non tetti prenotati.
- `static_verification.json`: ricontrollo locale di conteggi, update, proiezioni per fold e riproduzione identica dell’audit T100-G; nessun test runtime nuovo.
- `design_review.txt`: controllo locale indipendente di conteggi, dose e stime; non sostituisce la review di Claude.

Protocollo: `docs/protocol_a_epn_label_efficiency_20261107.md`. Prompt di review: `docs/prompt_claude_protocol_a_20261107.md`. Runtime ancora da implementare; sorgenti scientifici delle campagne chiuse invariati.
