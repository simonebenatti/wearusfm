# T100-LQ: review critica dei risultati finali — 07/11/2026

**Verdetto: APPROVATO il readout scientifico e la chiusura della matrice a
19.618 passi. Nessun rilievo P0/P1 e nessuna nuova prova necessaria per leggere
questa matrice.** L'approvazione riguarda analisi e raccomandazioni in
`docs/logsquare_scientific_readout_20261107.md`; non rilascia una continuazione,
una loss contrastiva o una modifica della ricetta.

## Perimetro e verifiche indipendenti

Ho confrontato protocollo congelato, report originali, piani esecutivi,
consuntivi, readout integrato e analisi indipendenti di posa/classificazione.
Non ho ripetuto i bootstrap estesi già ricostruiti dagli altri due revisori:
posa dai sufficienti per utente; classificazione dalle OOF del candidato e dai
conteggi storici. Entrambe le analisi riportano ricostruzione indipendente senza
import delle funzioni numeriche di produzione. Il limite della verifica posa
è esplicito: i grandi array remoti non sono stati scaricati e ricalcolati qui.

Ho verificato direttamente decompressione, dimensione e SHA256 dei tre originali:

| Report | SHA256 originale | Byte | Job finale |
| --- | --- | ---: | --- |
| Posa 5.000 | `8b21ea1647add4053727d819b588b168ac0f34b4257cc9882c69548117c2851b` | 263400 | 63786157 |
| Posa 19.618 | `e2439d883134ea9bf8a89ec4a4024c3e0990713fb8b0867e0eec4b3943587304` | 263158 | 63786158 |
| Classificazione 19.618 | `3517fc272ca5a258953d23d71e9b3aa894527755cbbcd153b93864b9503e7bfc` | 878936 | 63775886 |

Gli SHA dei tre piani coincidono con i file pubblicati. Posa eseguita sul
checkout B `9a53c7a65f59de683be093a4ad1ac450a473bdf2`; classificazione sul
checkout A `61f0021753fedf170eb1a7d40fde3fa40a638861`. I cinque sorgenti della
posa e i riferimenti locali storici ricontrollati mantengono gli hash attesi.
Le ricevute finali legano rapporti, artefatti e accounting; si aggiungono alle
verifiche precedenti di nove cache, endpoint, benchmark e release dei fit.

Il riepilogo di convergenza è completo: 140 fit nuovi e 420 storici risolti,
nessun warning né limite raggiunto; otto selezioni ridge risolte. Non c'è un
difetto documentato del lettore che renda questi esiti scientificamente sospesi.

## Rilievi prioritizzati e interpretazione

**P0/P1: nessuno.** La raccomandazione principale è proporzionata ai dati:
beneficio EPN rispetto a T100-L, assenza del beneficio dinamico aggiuntivo
cercato, nessuna superiorità generale dimostrata. T100-L resta il primario;
GELU non è stata sostituita al controllo dopo aver visto il risultato.

**P2, cautele già recepite nel readout; nessuna correzione bloccante:**

- EPN LQ−L = +1,5308 pp, IC95 [1,0127; 2,0149], è un risultato positivo di
  sviluppo. LQ−G = +0,0552 pp [−0,5117; 0,6030] non dimostra superiorità né
  equivalenza. Il bootstrap è condizionato a un solo pretraining per braccio
  e alle sonde fittate; non misura variabilità fra addestramenti indipendenti.
- Sulla posa a 19.618, delta E_dyn LQ−L = +0,0007257
  [−0,0011904; 0,0027240]. Non soddisfa il beneficio pratico registrato del 2%.
  LQ migliora con la dose insieme al controllo; non è una prova di utilità
  aggiuntiva del ramo. LQ−G = +0,0051558 [0,0030395; 0,0072968] favorisce
  GELU nel confronto secondario, senza autorizzare una conclusione generale
  sul FM o sull'endpoint a 49k non osservato.
- Il 16,19% è il rapporto delle varianze temporali predetta/bersaglio; la radice
  è circa 40,24% in ampiezza RMS centrata aggregata. Il testo distingue
  correttamente varianza, ampiezza, correlazione e livello medio. Il vantaggio
  delle statiche a 5.000 è conservato nella verifica completa della posa;
  non viene promosso al posto del primario dinamico e scompare a 19.618.
- P1/P2 sostengono NI entro 1 pp contro L e T25, non superiorità. Contro G,
  NI è sostenuta soltanto per P1: P2 resta irrisolta. Gli aggregati non
  garantiscono i singoli dataset; il peggioramento secondario su DB3 è
  dichiarato. Gli intervalli dei molti confronti non sono corretti per
  molteplicità e restano esplorativi.
- W_log non nullo mostra che il ramo è stato aggiornato, non il contributo
  funzionale delle sue famiglie. Il readout conserva questa distinzione,
  epsilon fisso e il confondimento di ottimizzazione previsto: nel collaudo
  rapporto mediano dei fattori di clipping 0,995062 e quota mediana
  g_log²/g_LQ² 1,6681%. Non si estrapolano quelle misure al training intero.
  Il risultato riguarda l'intervento LQ completo; non isola un meccanismo
  di energia di banda o un effetto causale della sola frequenza a 200 Hz.

## Risorse e completezza

Ricalcolo da `final/accounting.json`: **18/18 COMPLETED 0:0**, 16 job scientifici
e due checkout, **13,143333 core-ore** contro 51; zero GPU. I report finali
coincidono con gli ID e gli stati dell'accounting.

Ricalcolo dai secondi del consuntivo training: (21 + 3970 + 11368) × 4 / 3600
= **17,065556 GPU-ore**, incluso il tentativo fallito prima degli update;
136,524444 core-ore, contro 22/176. Le quattro estrazioni sommano
**1,739444 GPU-ore e 13,915556 core-ore**, senza i loro checkout.

Il readout le presenta correttamente come tranche distinte, non come costo
totale della campagna: collaudo, diagnostica, benchmark d'inferenza e altri
checkout restano nelle rispettive evidenze. Nessun costo osservato è sostituito
dal tetto prenotato o viceversa.

## Raccomandazione

Chiudere la matrice autorizzata e pubblicare l'esito misto. Non dare priorità a
LQ49k per la sola posa è una decisione ragionevole di allocazione, non una prova
che un seguito non possa mai funzionare. Un eventuale confronto futuro deve
conservare il controllo primario alla medesima dose. La linea GELU/fine-tuning
resta separata e non è stata verificata operativamente da questa review.

La contrastiva resta una proposta da fissare e revisionare su una base
esplicita, non una conseguenza automatica del risultato. Un nullo di questa
ridge sul teacher medio K64 non esclude altre viste, lettori o fine-tuning,
né chiude la diversa domanda delle energie strutturate.

Review locale; nessun job, monitor, modifica di sorgente/piano o commit.
