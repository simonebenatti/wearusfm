# T100-LQ: regole degli adattatori dopo la review 1d6f96a

Stato: implementazione per review, nessun rilascio del training. Claude ha
approvato decisione R1, preventivo R2, controller e disegno dei valutatori.
Simone ha già dato l'autorizzazione operativa: dopo convergenza, pubblicazione
e controlli non serve un nuovo consenso. Epsilon resta 0,001, dose e ricetta
restano quelle del controllo T100-L. Nessuna contrastiva in questa consegna.

## D1: rapporti di inferenza, fissati prima del benchmark

Si misurano separatamente i percorsi posa, P1 e P2. Ogni gruppo rappresentativo
ha una selezione deterministica di dati di train, forma e supporto registrati;
LQ e T100-L ricevono gli stessi ingressi. La posa usa il percorso FP32 con testa;
la classificazione il percorso BF16 completo, decoder compreso.

Per ciascun gruppo: rapporto fra tempi medi LQ/L dopo warmup, con sincronizzazione
CUDA, ripetizioni e alternanza dei bracci fissate nel piano. Per ciascun job:

- **Tetto:** massimo dei rapporti dei suoi gruppi rappresentativi; applicare
  `max(1, rho_cap)` al walltime storico e arrotondare per eccesso a 300 secondi.
- **Stima puntuale:** media dei rapporti pesata per i conteggi storici di
  **finestre** dei gruppi, con conteggi e fonte pinnati prima del benchmark.
  Non usare il numero di ripetizioni del benchmark come peso. Il rapporto
  pesato moltiplica la durata storica omologa.

Conteggi da ricevute o manifest storici; nessun peso uniforme implicito,
interpolazione o gruppo scelto dopo i tempi. Se mancano gruppi, fonte dei pesi,
supporti appaiati o tempi finiti positivi, la fase si ferma. Per più forme di
batch dello stesso dataset serve la loro attribuzione storica dei conteggi:
non ripartire arbitrariamente il totale. La scelta dei gruppi rappresentativi
e il suo limite rispetto ai casi non misurati restano dichiarati nel piano.
Il rapporto misura il percorso di inferenza; loader, I/O e scrittura sono
rendicontati separatamente e non diventano un guadagno dimostrato di walltime.

## D2: ordine delle fasi

1. Adattatori e piani implementati e revisionati da Claude.
2. Commit dedicato di rilascio del training, checkout nuovo congelato.
3. Training del candidato a 5.000 passi; ricevuta e checkpoint completi verificati.
4. Benchmark di inferenza su quel checkpoint e sul controllo T100-L alla stessa
   dose; pubblicazione dei preventivi delle estrazioni fra le due fasi.
5. La continuazione a 19.618 non aspetta il benchmark o i preventivi delle
   estrazioni: dipende soltanto dal completamento verificato del proprio 5.000.
6. Estrazioni solo dopo benchmark, budget e pin effettivi pubblicati; i pin del
   19.618 si acquisiscono dopo il completamento di quell'endpoint.

Il benchmark è una misura di risorse su train, non produce cache ammissibili
ai fit scientifici. La sua ricevuta non sostituisce le verifiche delle nove
cache scientifiche (quattro posa, cinque classificazione).

## N1–N4: condizioni operative del training

**N1.** Se la continuazione si arresta per `time_limit`, `checkpoint.pt` può
essere a un passo diverso da 5.000. Il controller corrente non riprende quello
stato: serve una revisione del piano. Nessun rilancio automatico, sovrascrittura
o arretramento del checkpoint per aggirare il controllo.

**N2.** Il rilascio è un commit dedicato, successivo alla review degli adattatori:
stato `APPROVED_FOR_TRAINING`, release pinnata con report, analisi, preventivo,
controller/launcher e review di Claude. Una review dei singoli componenti non
rende eseguibile da sola il piano. Il checkout deve essere nuovo e congelato
allo SHA di rilascio pubblicato; non si modifica quello del collaudo.

**N3.** Il run è su `$SCRATCH`, soggetto a purga a 40 giorni. Checkpoint nominati
e ricevute sono pinnati dopo ciascuna fase. Se i checkpoint devono essere
conservati oltre la scadenza, copiarli in `$WORK` prima della purga, verificare
hash e ricevuta della copia e aggiornare i pin di lettura senza modificare gli
originali. La copia richiede una verifica della quota; non è implicita nei
soli hash o nell'archiviazione dei report.

**N4.** Il submitter deve passare esplicitamente `--time=01:30:00` per 5.000 e
`--time=04:00:00` per 19.618. Il default del launcher è 1h30; dimenticare
l'override della continuazione fa fallire il controllo prima del training.
Restano 1 nodo, 4 GPU, 32 CPU e 480G, con 300 secondi di riserva per fase.

## D3 e verifiche di parità

Su ingressi identici, il percorso reale di inferenza LQ deve coincidere con
quello lineare quando W_log è zero e differire quando W_log è non nullo.
Verificare il caricamento del teacher, i pesi e il gain dal checkpoint senza
ricalibrazione. Le configurazioni storiche ricevono soltanto i due default
LQ mancanti; byte e hash degli originali restano invariati.

Ridge e logistica riusano le funzioni numeriche storiche; i fit nuovi sono
solo quelli del candidato. I tre controlli conservano codice, piano, cache,
coefficienti e OOF di origine. I confronti appaiati richiedono identità di
ricetta, supporti, utenti, fold e target; nessuna intersezione silenziosa o
riscrittura della provenienza per farli apparire della nuova campagna.

## Consegna concreta e risorse per la review

| Componente | Implementazione | Risorse e condizione |
| --- | --- | --- |
| Training 5.000 → 19.618 | `logsquare_train.py`, già approvato da Claude | Stima 16,7052 GPUh; tetto 22 GPUh / 176 coreh. Piano ancora `REVIEW_REQUIRED`, release nulla. |
| Benchmark di inferenza | `logsquare_infer_benchmark.py` | 1 GPU, 8 CPU, 100G, 30 min: tetto 0,5 GPUh / 4 coreh. Dati train, nessun fit o score. |
| Quattro estrazioni scientifiche | `logsquare_evaluation.py` | Due posa, P1 e P2; nove cache. Tetti nulli fino al benchmark e alla pubblicazione di D1. |
| Posa per endpoint | `logsquare_pose_readout.py` | Validate, benchmark train, fit del solo candidato e confronto separato: 8 coreh per dose, 16 totali, zero GPU. |
| Sonde a 19.618 | `logsquare_probe.py` | Cinque shard LQ, due aggregati, un confronto: 34,3333 coreh, zero GPU. |
| Checkout congelati | Procedura comune versionata | 2 CPU × 600 s = 0,3333 coreh per checkout; contare una sola volta quelli condivisi e pubblicare il numero effettivo nel piano esecutivo. |

Il tetto proposto dei soli lettori CPU è quindi **50,3333 coreh**. Il totale
della catena non è ancora completo: mancano i tetti misurati delle estrazioni
e il numero effettivo dei checkout. Non sommare di nuovo le risorse della
campagna storica riusata. Non è stato sostenuto alcun costo HPC per questa
consegna di codice e test locali.

La posa mantiene un nuovo benchmark train-only prima della ridge. Per le
sonde la proposta esplicita è il **riuso delle misure storiche per shard**,
non un nuovo benchmark LQ: ricevuta pinnata, stessa ricetta, stessi sorgenti
numerici, stesso ambiente numerico, larghezza 768 e conteggi dei fold uguali.
I tempi di tutti e tre i bracci storici devono stare nei tetti. Questa scelta
è sottoposta a Claude con gli adattatori. Valori diversi delle feature possono
richiedere un numero diverso di iterazioni: la misura storica non garantisce
la durata LQ. Se un gate o il walltime fallisce, si ferma la fase senza
rilancio o aumento del tetto impliciti.

Il benchmark GPU usa un gruppo rappresentativo per dataset, con un batch
completo di 64 finestre per ciascun crop di classificazione e le prime due
registrazioni della selezione posa train congelata (10 finestre ciascuna).
I campioni aggiuntivi necessari per conservare la scala per soggetto non
entrano nei tempi del gruppo. I casi di coda e tutte le geometrie possibili
non sono misurati esaustivamente. Il residuo della pipeline registra insieme
preprocessing, trasferimenti e copia/impacchettamento CPU: non è una misura
separata di ciascuna voce, né una stima dell'I/O della futura cache completa.

I piani preparatori contengono i pin reali dei controlli disponibili e valori
nulli per checkpoint, cache e ricevute LQ ancora inesistenti. I launcher e i
controller li rifiutano: non sono valori fittizi eseguibili. Dopo la review,
il commit di rilascio del training precede il nuovo checkout; i successivi
commit di valutazione acquisiscono i pin reali senza cambiare la ricetta.
