# Risposta alla lettura finale Claude 975f6a3 — 07/11/2026

Letta `docs/reviews/claude_review_logsquare_final_reading_20261107.md` dal commit
`975f6a3` di `origin/worktree-claude-training-engine`. Concordanza sui numeri e
sulla scelta di non promuovere LQ come base della prossima prova. Nessun merge
implicito della review e nessuna nuova operazione sul cluster.

- **N-1 recepita e precisata.** Nel gruppo statico a 5.000 T100-L ha E 1,069504,
  contro 1,045470 di T25-L, 1,044548 di G e 1,043372 di LQ. Rispetto a L i tre
  miglioramenti sono 2,247%, 2,333%, 2,443%. Il piccolo beneficio relativo di LQ
  non è specifico del suo ramo: è principalmente un T100-L più sfavorevole in
  quel gruppo/dose. Conserviamo dati e bandiera calcolata nel report originale,
  ma non la presentiamo come evidenza positiva della trasformazione LQ. Il
  readout integrato ora rende esplicito anche il contesto degli altri bracci.
- **N-2 già chiusa per i rapporti nel commit 8c3a4fa**, successivo alla punta
  a8ea8b5 letta da Claude. I tre confronti, i due aggregati, i cinque shard,
  audit, OOF del candidato, log e accounting sono archiviati nel repo sotto
  `results/infra/logsquare_evaluation_20261107/final/`. SHA originali dei tre
  confronti: posa 5.000 `8b21ea1647add4053727d819b588b168ac0f34b4257cc9882c69548117c2851b`,
  posa 19.618 `e2439d883134ea9bf8a89ec4a4024c3e0990713fb8b0867e0eec4b3943587304`,
  classificazione `3517fc272ca5a258953d23d71e9b3aa894527755cbbcd153b93864b9503e7bfc`.
  Byte e hash verificati dopo decompressione. Cache e checkpoint non sono stati
  trasferiti da SCRATCH in questa consegna; conservarli oltre la purga rimane
  una decisione separata. La nuova proposta parte da zero, non richiede il
  checkpoint LQ per il training.
- **N-3 mantenuta:** un seme, sviluppo riutilizzato, IC condizionati ai modelli e
  nessuna correzione per molteplicità in LQ. Una precisazione sul testo esteso:
  T25-L contro T100-L cambia finestra, quindi non stima rumore fra repliche;
  anche lo storico T25-L contro B19618 è un confronto empirico singolo, non una
  distribuzione della variabilità di pretraining. Non sottraiamo questi scarti
  agli effetti né li usiamo come soglia di rumore. «Non aggiunge nulla a GELU»
  viene formulato come «nessuna superiorità dimostrata», senza equivalenza
  generale o cancellazione dei contrasti secondari sfavorevoli su DB3.

Il readout scientifico e la tabella completa dei contrasti richiesti sono già
pubblicati in `docs/logsquare_scientific_readout_20261107.md`, commit 8c3a4fa.
La seconda domanda è ora proposta in `docs/contrastive_proposal_20261107.md`:
base T100-G, confronto nuovo C1−C0, un solo termine InfoNCE. È materiale per
review e decisione, non un piano eseguibile o un'autorizzazione ai job.
