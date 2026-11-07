# Prompt per Claude — review del disegno del protocollo A

Ramo GPT: `codex/consolidation-20261104`; usa il commit che pubblica questo file. La preparazione è stata affidata a GPT da Simone («A GPT»). Questa consegna chiede una review del disegno, non un lancio: non esiste ancora un piano eseguibile o un budget approvato per A. Nessun nuovo benchmark, fit o deployment in questa fase. L'operatore della linea 2 resta Claude.

## Conservazione chiusa

GPT ha verificato le prove di entrambe le copie: LONG_ROOT (`424f0ec`, job 63781257) e T100-G 19.618 (`2930d7d`, job 63785874). Totale 380 file, 21.086.120.024 byte; hash e alberi originali/copie concordi. Consuntivo lungo più due copie: 109 allocazioni, 176,836111 coreh, 3,373611 GPUh. Nota aggiornata in `docs/reviews/consolidation_long_preservation_20261107.md`; audit T100-G riproducibile con 195 controlli. È una verifica dei manifest archiviati, non una nuova lettura dei payload remoti. Nessun job di GPT.

## Cosa rivedere

- `docs/protocol_a_epn_label_efficiency_20261107.md`: disegno completo, primario, selezione, riusi, costi e arresti.
- `results/infra/epn_label_efficiency_20261107/proposal.json`: proposta strutturata, non eseguibile.
- `results/infra/epn_label_efficiency_20261107/population_counts.json` e `resource_estimate.json`: conteggi e proiezioni dai consuntivi EPN15.
- `docs/reviews/protocol_a_implementation_20261107.md`: componenti riusabili e modifiche necessarie; non codice implementato.

Proposta concreta: teacher T100-G 49.046, fine-tuning contro scratch, controllo congelato con logistica CPU sul pool 768 già estratto. Quattro fold EPN storici, train annidati 5/10/25/50/100%, validation e held fissi, due LR per regime e seme 0. Il 5% ha 11 soggetti train più 24 validation: esplicitato il budget effettivo di etichette. Nessun FT19k aggiuntivo a train ridotto e nessuna prova su nuovi dataset inclusa tacitamente.

Raccomando per la domanda scientifica primaria gli stessi aggiornamenti del 100% a ogni frazione, con 15 occasioni validation fisse. **Il costo cambia sostanzialmente rispetto alla stima iniziale nella discussione:** 72 nuovi fit, circa 766,22 GPUh prima di benchmark, CPU e margini. Le 8,21 GPUh usate allora includevano i frozen più veloci: FT e scratch costano circa 10,64 GPUh per candidato. L'alternativa a 15 epoche per frazione costa circa 239,79 GPUh, con meno ottimizzazione nei subset piccoli; il suo significato è scritto separatamente. Entrambe sono proposte senza budget autorizzato. Non ridurre la griglia tacitamente per fare tornare il costo.

## Domande della review

1. Domanda, popolazione e label budget: annidamento, val fissa, dati di sviluppo e unico seme sono descritti senza promettere un risparmio di etichette non misurato?
2. Dose e selezione: update costanti, shuffle e batch parziali storici, nessun reset tra passaggi, 15 checkpoint validation, due LR e held una volta sono coerenti? Valuta esplicitamente il vantaggio scientifico rispetto al costo dell'alternativa a 15 epoche; raccomanda una politica prima dell'implementazione.
3. Primario: area normalizzata su log(n), FT−scratch per n=11/22/53/106, 100% secondario, bootstrap appaiato, regola pratica nuova a 1 pp e lettura a due lati. Cerca ambiguità o inferenze non sostenute; nessuna sostituzione post hoc dei primari firmati del lungo.
4. Riuso: identifica condizioni mancanti per scratch100 EPN15 e frozen100 LONG49k. Il riuso è subordinato a un bridge verificato; se fallisce, niente fit aggiuntivi impliciti. Controlla anche scaler train-only e corrispondenza pool 768 con il modello del fine-tuning.
5. Conti: 40 FT + 32 scratch nuovi, 8 scratch storici; 16 celle frozen nuove più quattro storiche. Ricalcola la proiezione per fold e distingui fit, celle e allocazioni ancora da dimensionare. Il costo CPU e l'overhead non sono zero o già inclusi.
6. Arresti e limiti: divergenza numerica distinta da guasto/OOM/TIMEOUT, gate globale prima dei fit, niente primario su matrice incompleta, nessuna estensione automatica. Dimmi se la proposta è pronta a diventare un piano da sottoporre a Simone o richiede cambi di disegno.

Consegna una review sul tuo ramo con verdetti separati per i sei punti, rilievi bloccanti e raccomandazioni motivate. Registra la voce nelle due sezioni del registro, commit/push con SHA remoto verificato. Nessuna nuova autorizzazione per i vecchi job è richiesta; il protocollo A è un esperimento nuovo ancora in preparazione. Dopo la convergenza del disegno e la decisione di Simone si prepareranno codice, benchmark, tetti completi e consegna operativa.
