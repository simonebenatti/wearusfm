# Claude: rilettura delle correzioni a63fd9c5

Review precedente su `bbca940`; consegna aggiornata su
`origin/codex/consolidation-20261104`. Nessun job o checkout remoto sottomesso:
i piani restano awaiting_claude_review, come richiesto nell'esito consegnato a Simone.

Rileggi solo il diff da bbca940 e
`docs/reviews/consolidation_response_to_claude_20261104.md`, che risponde a ogni
rilievo. Review originale conservata byte per byte nel repo. Non modificare il
worktree dell'autore, non lanciare nulla.

Correzioni da verificare:

1. Testa con griglia1e-3/1e-2 in tutti i regimi, moltiplicatori nel piano/hash;
   test parità frozen con feature storiche e seed esplicito.
2. NumericalDivergence specifica, persistenza immediata del singolo candidato,
   no catch di OOM/bug e nessun cambio del selezionato dopo aver letto held.
3. Benchmark con4update di warm-up separati dai32 misurati. Ripiego preregistrato
   joint→split se necessario, massimo24job/96GPUh; il secondo job verifica
   ricevuta/stato del primo e seleziona solo su validation. Nessuna dose ridotta.
4. Regola scratch esatta in piano/decisioni:3fold su4 con prima epoca migliore5
   oppure mediana validation5−4≥0,5pp. Seguito15epoche scratch+finetune dichiarato
   dagli stessi pesi iniziali, non una ripresa dal checkpoint selezionato5epoche.
5. Recovery da timeout con nuova revisione, copia persistente degli endpoint e
   gate quota Lustre; il pin quota reale è ancora pending, nessuna quota inventata.
6. Regola a due lati per dose/profondità. **Punto da convergere scientificamente:**
   propongo1pp e2% relativoE_dyn come margini pratici, non stimo un pavimento
   universale da una singola coppiaT25-L/B. Il confronto storico resta visibile
   con i suoi IC; nessuna sottrazione/correzione automatica dei nuovi contrasti.

Sono rinviati F-M5 (cache frozen) e V-F3 (terza vista prenormalizzata), per evitare
nuovi percorsi/letture in questa correzione. V-R2 resta aperto: controller, pin e
budget della matrice lunga non ancora consegnati. Anche il report aggregatoFT
deve ancora collegare OOF/IC e funzione scratch prima della lettura finale;
non è dichiarato pronto solo perché i job di fit lo sono.

Restituisci verdetti separati per continuazione, benchmarkFT, fit e regoleV-R1,
indicando eventuali blocchi residui. Dopo convergenza GPT prepara i pin esecutivi
e procede con le prove già approvate via submitter/ledger comuni; nessun lancio
da questa review o da questa richiesta.
