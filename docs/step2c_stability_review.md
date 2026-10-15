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

### Esito sui cinque JSONL verificati per SHA remoto/locale

L'audit riproduce esattamente1,973731 del collaudo. Finestra critica1577-1626: media0,09518, mediana0,07083, minimo storico0,04822.
Togliendo il solo massimo0,55490, la media resta0,08580 e il rapporto1,77918, ancora sopra1,5. I cinque batch piu' alti fanno32,6%
della somma della finestra; ce ne sono anche a0,29482 e0,28665. Quindi l'allarme non dipende da un unico batch.

| Run | Rapporto v1 | Rapporto togliendo il massimo | Gradiente al clip |
|---|---:|---:|---:|
| Collaudo | 1,974 | 1,779 | 5,6% |
| A_seed0 | 2,238 | 1,953 | 20,9% |
| A_seed1 | 2,108 | 1,675 | 18,8% |
| B_seed0 | 1,974 | 1,779 | 4,9% |
| B_seed1 | 1,976 | 1,867 | 5,7% |

A_seed0 e B_seed0 hanno la stessa finestra critica e gli stessi aggiornamenti dei picchi piu' alti; e' compatibile con un effetto
degli ingressi accoppiati, ma senza gli hash dei batch/sessioni non e' un'attribuzione dimostrata. Le ultime medie a blocchi200 e le
loss ultime100 restano basse: non c'e' una crescita persistente evidente nel finale. Nessuno dei run passa il criterio v1.
Risultato completo: `results/passo6/step2c_stability_audit_20261014.json`, con hash, finestre e loss per testa.

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
