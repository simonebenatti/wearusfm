# Step2C: review della stabilita' (14/10/2026)

Il collaudo `60477395` resta **non passato** secondo il criterio congelato v1. Il successivo via ai quattro run e' una decisione
registrata dopo quel risultato. Questa review non cambia il verdetto, i checkpoint o gli endpoint P1/P2.

## Che cosa misura v1

La formula implementata in `step2c.collaudo_verdict` coincide con il protocollo: massimo rapporto tra media mobile50 e minimo storico
della stessa media. Il riferimento e' selezionato e le circa1.800 finestre si sovrappongono: misura una risalita da una finestra favorevole,
senza distinguere cambiamenti del modello e variazioni degli ingressi. Il codice non ha un errore che spieghi il fallimento.

Le sintesi di Claude mostrano loss prime100->ultime100 in calo in tutti i run (circa0,25->0,065-0,090), encoder intatto e medie su200
basse nel finale. Non dimostrano una deriva persistente, ma non certificano da sole stabilita' dell'ottimizzazione.
Il singolo batch0,555, con altri49 a0,05, darebbe media0,0601: non spiega da solo la media critica0,0952. Va esaminata tutta la finestra.
Clipping piu' frequente con A (~19-21%) rispetto a B (~5-6%): descrittivo, non prova causale o risultato downstream.

## Audit riproducibile sui log esistenti

`scripts/audit_training_stability.py` riproduce v1, registra la finestra critica e il minimo, i dieci batch piu' alti, media e mediana,
la media dopo rimozione del solo massimo (descrittiva), blocchi non sovrapposti200 e norme dei gradienti prima del clipping.
Non usa una nuova soglia per approvare i run. Zero/NaN/Inf restano espliciti. Provenienza: SHA256 di ogni JSONL, stessi aggiornamenti.
Tre test distinguono picco isolato e risalita persistente, riproducono v1 e controllano valori non validi/indici.

## Correzione proposta per le campagne future

Separare gli arresti numerici (loss/gradienti/pesi non finiti) dal monitor di regressione e dalla descrizione dei batch difficili.
Per la regressione, usare periodicamente un pannello fisso di finestre, selezionato da dati train/validation senza usare P1/P2 o test:
studente in eval, stessi ingressi e target, RNG preservato. Il confronto cosi' varia con il modello e non con l'identita' del batch.
Dimensione, cadenza, tolleranza e numero di regressioni consecutive devono essere congelati prima del prossimo run, con nuova versione;
non stimarli dalle cinque curve per farle passare. La sola mediana puo' nascondere una coda: riportare anche media, quantili e massimo.

Al trainer di Claude aggiungere provenienza per aggiornamento (sessioni pseudonimizzate, ritagli, target validi), loss per testa e norma
pre-clipping. I log esistenti non consentono attribuzione precisa alle sessioni: non dichiarare quale dataset/soggetto causa i picchi.
Nessun cambio di learning rate, loss, scala dei target o clipping motivato soltanto da questo allarme. Prima collegare i picchi ai dati
e alla loss sul pannello fisso; le nuove campagne richiedono tale diagnosi e un protocollo prospettico.
