# Prompt per Claude — controllo dei risultati B

B e' terminato. GPT ha raccolto e analizzato il job 62523672 sul ramo pubblicato
`origin/codex/r3-long-collaudo-20261028`. Esegui la revisione reciproca dal tuo worktree;
non modificare quello GPT. Nessun deploy, estrazione, training o nuovo job in questa richiesta.

Codice eseguito e lettore congelati: `24139ab901c052357a46a65bc20466f2eb1e8358`, dalla tua base
`afd2e4d6aca7a417fef59efcd3df2c048fa23653`. Il tip aggiunge evidenze, analisi e registro;
non cambia trainer, ricetta o soglie dopo il risultato.

1. Leggi `docs/reviews/collaudo_b_results_20261028.md`, `docs/collaudo_r3_b_20261028.md`,
   `results/infra/pilot_posa/collaudo_b_20261028/README.md` e le nuove voci GPT del registro.
2. Riesegui localmente i due comandi del README: lettore congelato e analisi descrittiva.
   Il verdetto applicabile e' `verdetto.json`: SUPERATO, MSE 1,160059 → 1,024176, ratio 0,882866,
   IC95 utenti [0,839470; 0,932231], 23/32 utenti in miglioramento. Verifica config e provenienza,
   5.000 passi, 11 pannelli invariati, bootstrap, conteggi globali e checkpoint nel listing.
   Il primo `verdetto_listing_file_assoluto.json` aveva un input `ls` non conforme alla CLI,
   descritto nel README; non scambiarlo per il verdetto definitivo o per un difetto del run.
3. Rivedi l'interpretazione: B dimostra un calo sul train; non generalizzazione, movimento corretto
   o causalita' esclusiva della durata del dbg. IC95 positivo non significa miglioramento >=10%
   con confidenza 95%. E4 resta locale; le quote di energia delle predizioni non sono quote del
   guadagno MSE. Nove utenti peggiorano. Supporti e stratificazioni sono descrittivi post hoc.
4. Valuta la proposta scientifica: prima del costo di R3 completo, sola inferenza E1/E2 sullo
   **student a 5.000**, su `val`, col controllo disallineato a supporto comune. Lo script ha
   default teacher: va specificato student. Nessun train/E3 necessario per questa lettura.
   Il checkpoint a 5.000 sarebbe diagnostico, distinto dall'endpoint scientifico a 49.046;
   encoder senza maschera come E1/E2, mentre B usa maschere fisse. Non introdurre un nuovo gate
   retroattivo o una selezione dei modelli su `val`. La proposta non e' autorizzata al lancio.
5. Controlla il preventivo indicativo del rapporto: 1 A100 / 8 core / 60G / 40 min piu' 8 core / 30G / 20 min,
   tetto 0,667 GPU-ore e 8 core-ore, senza nuovo checkout se riutilizzabile quello congelato.
   E2 con testa non e' stato cronometrato da B; non assumere i 60s del rapporto R3c senza testa.
6. Pubblica review e rilievi verificati sul tuo ramo, nel registro, indicando se cambiano il
   verdetto, la sola interpretazione o la proposta successiva. Niente correzioni preventive
   non motivate, nessuna riscrittura dei risultati. I 123 test della preparazione restano validi;
   riesegui soltanto quelli pertinenti a un dubbio concreto.

Consuntivo B: 4,23778 GPU-ore, 33,91167 core-ore checkout compreso. R3 completo, E3 su R0 e J2 bis
restano fermi. Restituisci a Simone un verdetto chiaro sul risultato e sulla raccomandazione.
