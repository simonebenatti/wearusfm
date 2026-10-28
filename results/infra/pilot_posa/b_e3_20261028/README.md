# E3 B student a 5.000 passi, 28/10/2026

Protocollo: `docs/e3_b_20261028.md`; risposta alla review Claude:
`docs/reviews/risposta_review_claude_e3_b_20261028.md`. Piano e comandi nella cartella superiore.

- `cluster_preflight.json`: disponibilita' input, risorse e coda prima del lancio.
- `extract_review.json`: review tecnica della guardia di estrazione.
- `technical_review.json`: verifica finale payload/argv/hash/dipendenza e output nuovi.

41 test CPU passati, critico scientifico favorevole. Questa preparazione precede i risultati;
le evidenze di sottomissione e i risultati sono stati aggiunti con commit successivi separati.
Cache e checkpoint rimangono sul cluster; nessun dato wearable viene pubblicato nel repository.

Protocollo pubblicato prima dei job in `520158e`. Sottomissioni: **62561780** (estrazione train),
**62561881** (rapporto CPU, afterok:62561780), entrambe confermate dal submitter.
`extract_submission.json` e `report_submission.json` conservano argv e risposte;
`interrupted_launch_local_reconciliation.json` documenta l'assenza di job dopo l'interruzione del primo permesso runtime.
Nessuna esecuzione duplicata o retry automatico.

## Risultati conclusi

Entrambi COMPLETED 0:0: estrazione 255 s, rapporto 106 s. Consuntivo **0,070833 GPU-ore / 0,802222 core-ore**.
`final/` conserva rapporto originale, provenienze, accounting, log e manifest SHA; file copiati e verificati deliberatamente.
`independent_recalculation.json` ricalcola dai momenti per utente i tre delta e IC appaiati, senza import dell'analisi originale.
`present_results.py` riproduce il ricalcolo e la figura PNG/PDF.

Primario ridge meno testa **+0,002832, IC95 [+0,000898; +0,004602]**: questo rifit non recupera altra dinamica.
Migliorano il livello medio e la MSE totale; non si conclude che la ridge sia peggiore in generale o che l'encoder sia saturo.
Lettura completa, elenco delle modifiche e prossima proposta: `docs/reviews/e3_b_results_20261028.md`.
Nessun training successivo lanciato.
