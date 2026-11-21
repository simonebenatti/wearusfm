# Review Claude: ripresa InfoNCE della coorte RMS100/fast125, secondo parere scientifico (`918245b6`)

21/11/2026. Revisore: Claude. Linea 3: scienza e codice a GPT, lanci a 3-launcher. Claude non ha lanciato nulla.
Il verdetto storico resta `FAILED_REQUIRES_REVIEW`: questa review lo spiega, non lo riscrive.

## Richiesta

Messaggio di GPT incollato da Simone in chat il 21/11: «i quattro scarti di InfoNCE indicano un problema nella ripresa oppure una
differenza numerica ripetibile che il criterio dell’1% rende visibile? Qual è il controllo minimo per distinguerli?»

Il prompt completo è in `docs/prompt_claude_contrastive_targets_resume_science_20261121.md` a `918245b6`. Distingue tre
ipotesi («un difetto di ripristino, una differenza numerica ripetibile o una sensibilità del criterio») e chiede tre
cose: verificare i quattro momenti, mettere alla prova la diagnosi, valutare la prova minima.

Perimetro e metodo:
- **Fonti.** Evidenze a `f23b7cca`; lettura di GPT a `adb6e3d6`, con la sezione «Elementi da sottoporre a Claude» aggiunta
  a `918245b6`. Il JSON originale decompresso ha 3.081.874 byte e sha256
  `d4300a2e31c85b39f1f36b8f3146886811d70fe9e24c2d0a9d07e683e84cd4b8`: ricalcolato da me.
- **Sorgenti.** L'inventario dei sorgenti a `918245b6` è identico a quello del pacchetto rivisto `fbfde797`: digest
  ricalcolato `40d39f56…`, uguale al `source_sha256` del rapporto.
- **Lettura mia** di rapporto, metriche per update e tracce, con script e uscite nella cartella delle prove.
- **Sei analisi indipendenti** su mappatura, stato ripristinato, decorso, meccanica DDP, prova proposta e confutazione.
- **Due letture in sola lettura su Leonardo** (21/11, 16:03): nodi dei sette job con `sacct`, e ordine delle operazioni
  DDP nel torch del cluster.
- **Rilettura della bozza.** Due verificatori indipendenti hanno controllato contro le fonti circa 125 e 58 affermazioni
  e segnalato 33 rilievi, alcuni in comune: 4 errori, 12 conclusioni più forti delle prove e 17 imprecisioni. Il più
  importante: la bozza ometteva il limite registrato il 20/11 (sezione 3). Un terzo controllo sul testo riscritto,
  circa 100 affermazioni, ha chiesto altri 11 ritocchi. Tutti i rilievi sono corretti qui.

## Verdetto

**I quattro scarti sono una differenza numerica ripetibile che il criterio dell'1% rende visibile. Non indicano un difetto
nel salvataggio o nel ripristino dello stato.** Il giudizio poggia su prove osservative e aggregate, descritte sotto con
i loro limiti.

- **Che cosa si può chiudere.** Lo stato salvato al passo 544 è ripristinato esattamente (sezione 2.1). Lo stato di
  processo che il checkpoint non contiene, cioè il raggruppamento dei gradienti in DDP, alla ripresa è diverso: lì nasce
  la differenza.
- **Dove nasce lo scarto.** Al 545 differiscono, per un arrotondamento, solo le grandezze calcolate dopo la riduzione DDP:
  2,2e-12 sulla norma del gradiente. È l'unico update in cui la ripresa riduce con un raggruppamento diverso dal
  continuo. L'indicazione è indiretta: viene da norme globali, non da gradienti per rank e per tensore (sezione 2.5).
- **Come cresce.** Lo scarto salta di circa sette ordini di grandezza un update dopo in Rank-N-Contrast (546) e due
  update dopo in InfoNCE (547, dopo un 546 con norme identiche). Poi lo scarto della norma del gradiente cresce
  lentamente (sezione 2.2); quello delle chiavi della coda resta allo stesso livello (sezione 2.4).
- **Perché solo InfoNCE supera l'1%.** Tutti i primi momenti differiscono, in entrambi i bracci. I quattro fuori soglia
  sono i parametri del proiettore: formano un gruppo distinto in cima alla distribuzione e ricevono soltanto il
  gradiente InfoNCE, che diverge più di quello di base. In Rank-N-Contrast gli stessi tensori hanno momenti nulli, quindi
  il suo esito non li mette alla prova. Il fattore proprio del proiettore non è dimostrato (sezione 2.3).
- **Regola registrata.** Il limite scritto il 20/11 chiede di localizzare lo scarto al primo passo «per rank, tensore e
  gruppo DDP» e tratta il cambio dell'ordine di riduzione come «ipotesi da verificare». Gli artefatti archiviati non
  hanno quella granularità. Proseguire senza una prova mirata è quindi una deroga, che spetta a Simone dichiarare.
- **Controllo minimo.** Se si fa la prova, conviene partire dal fork esistente con due riprese, una ordinaria e una con
  un'iterazione di riscaldamento DDP: circa metà del costo della proposta di GPT, la granularità richiesta e un esito
  binario sull'ipotesi (sezione 3.4).

## 1. I quattro momenti: mappatura e denominatori

### Mappatura

**La mappatura di GPT è corretta.**
- **Ordine.** Il trainer attacca il proiettore dopo `WearUsFM` (`run.py:1387-1390`) e passa ad AdamW i parametri
  allenabili in ordine di costruzione (`run.py:1429`). L'ID di `optimizer.state.<ID>` è la posizione in quella lista.
- **Ricostruzione su CPU** dalla `config.json` di CN: 270 tensori, 28.559.133 elementi.

| ID | Parametro | Forma | Elementi |
|---|---|---|---|
| 266 | `contrastive_head.layers.0.weight` | 384 × 384 | 147.456 |
| 267 | `contrastive_head.layers.0.bias` | 384 | 384 |
| 268 | `contrastive_head.layers.2.weight` | 128 × 384 | 49.152 |
| 269 | `contrastive_head.layers.2.bias` | 128 | 128 |

- **Peso nel modello.** Il proiettore ha 197.120 elementi, lo 0,69% del totale.
- **Iperparametri.** AdamW ha un solo gruppo: lr 1e-4, beta 0,9 e 0,999, weight decay 0,05. Il proiettore non ha
  iperparametri propri.
- **Impronta dell'ordine.** `864a86de…9837` compare in 24 tracce su 24 al confine. Uno degli analisti l'ha riprodotta
  solo con le chiavi di AdamW di torch 2.2, la versione del run. Prova un gruppo di 270 parametri con gli stessi
  iperparametri, non i nomi: i nomi vengono dall'ordine di costruzione del codice, che è quello dei sorgenti del run.

**In Rank-N-Contrast gli stessi ID hanno errore zero esatto.** Il proiettore esiste anche lì, ma entra solo nella loss
InfoNCE, che ha peso 0: `contrast_projector_grad_norm` vale 0 in tutti i 576 update. Momenti nulli, errore nullo.

### Denominatori

La descrizione di GPT è corretta. Formula unica in `scripts/contrastive_benchmark_readout.py:114-127`: norma L2 della
differenza divisa per una norma di riferimento, con pavimento 1e-12.

| Grandezza | Riferimento al denominatore | Ruolo |
|---|---|---|
| Ogni tensore dei momenti di Adam e ogni chiave della coda | norma dello stesso tensore nel primo membro, al passo 576 | cancello, 1% per tensore |
| Studente e teacher | spostamento del primo membro dall'inizializzazione autenticata (checkpoint di calibrazione) | cancello, 1% |
| Studente e teacher «dal passo 544», totale e per blocco | spostamento del primo membro dal fork | descrittivo |

- **Primo membro.** È il continuo nelle coppie continuo/A e continuo/B, la ripresa A nella coppia A/B.
- **Valori.** Tutti quelli citati da GPT coincidono con il JSON: 1,222%, 1,034%, 1,316% e 1,108% per i primi momenti;
  da 0,155% a 0,168% per i secondi; 0,08975% e 0,00821% per studente e teacher di InfoNCE; 0,6454% e 0,6164% dal 544.

**I denominatori non sono omogenei, e questo conta per la lettura:**
- **Primi momenti.** Con beta1 = 0,9, dopo 32 update i gradienti successivi al fork pesano il 96,6% nella media mobile.
  È un peso, non una quota della norma. La differenza fra continuo e ripresa viene tutta da quei gradienti.
- **Secondi momenti.** Con beta2 = 0,999 lo stesso peso è il 3,1%. È coerente con i loro errori, molto più piccoli.
- **Pesi.** Sono giudicati contro 576 update di spostamento. Con il metro omogeneo, lo spostamento dal 544, lo studente
  differisce dello 0,65%: lo stesso ordine dei primi momenti.

### Che cosa leggere dai checkpoint

Norme assolute, norma della differenza e coseno aiutano a giudicare la scala, non la causa. Il coseno da solo aggiunge
poco: se le due norme sono uguali, un errore relativo dell'1,3% implica un coseno di circa 0,99991.

Misure utili, per i quattro tensori e per tutti i 270 come confronto, in entrambi i bracci:

| Misura | Esito atteso se è rumore numerico | Esito che indicherebbe un errore sistematico |
|---|---|---|
| Norme dei due momenti e loro rapporto | rapporto entro l'1% da 1 | rapporto lontano da 1, per esempio vicino a un fattore di beta1 |
| Differenza scomposta in parte radiale e tangenziale | quasi tutta tangenziale | parte radiale dominante, differenza proporzionale al momento |
| Parte del momento successiva al fork come denominatore | il proiettore si allinea agli altri tensori | il proiettore resta isolato |
| Norma del momento rispetto alla radice del secondo momento | più bassa nel proiettore; l'errore cresce dove è bassa | nessuna relazione |
| Direzione dell'update di Adam nei due run | coseno almeno 0,9999 | coseno basso o scala anomala |

La quarta riga risponde alla domanda di GPT sul denominatore: se il primo momento del proiettore è piccolo rispetto alla
scala tipica dei suoi gradienti, la sua norma al denominatore amplifica l'errore relativo.

## 2. La diagnosi alla prova

### 2.1 Stato al confine: lo stato salvato è ripristinato esattamente

- **Impronte al confine.** All'update 545 le impronte di studente, teacher, momenti e gruppi di AdamW e stato
  contrastivo coincidono con il file del fork su 4 rank e 3 percorsi, in entrambi i bracci. Lo studente comprende il
  proiettore; i momenti comprendono il contatore `step`; lo stato contrastivo comprende le chiavi della coda.
  `boundary_mismatches` è vuoto.
- **Generatori casuali.** Le impronte CPU e CUDA per rank coincidono prima e dopo ogni backward, per 32 update.
- **Dati.** Batch, input, ordine e lunghezza della coda, copertura, pesi delle loss e conteggi coincidono per 32 update
  su 4 rank. Uno degli analisti ha confrontato 5.965 conteggi per braccio senza trovare differenze.
- **Forward del 545.** Le loss `contrast_nce`, comprese le componenti per dataset, `contrast_rnc` e `pose` sono
  identiche bit per bit fra continuo e riprese.
- **Chiavi della coda.** Le 4.979 chiavi anteriori al fork ancora presenti a 576 hanno errore zero esatto.
- **Coda replicata.** Le chiavi sono raccolte da tutti i rank e ogni rank tiene la stessa copia: salvarla dal rank 0 è
  corretto, e l'impronta al confine è uguale sui 4 rank.

**Restano fuori dalle impronte gli stati di processo che il checkpoint non contiene:** lo stato interno di DDP, la
comunicazione NCCL, le librerie CUDA, l'allocatore. Fra questi il raggruppamento DDP risulta diverso (sezione 2.5). Le
impronte ignorano anche il dispositivo dei tensori: quello dei momenti di AdamW dopo il caricamento non è verificato a
runtime, ed è ritenuto improbabile come causa.

### 2.2 Decorso update per update

Differenza relativa fra continuo e ripresa A; la ripresa B dà gli stessi numeri.

| Update | InfoNCE: norma del gradiente | InfoNCE: norma del gradiente del proiettore | InfoNCE: loss `contrast_nce` | Rank-N-Contrast: norma del gradiente |
|---|---|---|---|---|
| 545 | 2,2e-12 | 1,0e-13 | 0 | 3,4e-13 |
| 546 | 0 | 0 | 0 | 3,4e-6 |
| 547 | 2,2e-5 | 6,5e-5 | 2,4e-6 | 1,3e-4 |
| 548 | 9,1e-5 | 1,1e-3 | 6,0e-5 | 3,1e-5 |
| 548-561, media geometrica | 1,6e-4 | 8,6e-4 | | 7,4e-5 |
| 562-576, media geometrica | 5,0e-4 | 7,3e-4 | | 3,9e-4 |

- **Seme.** Al 545 la norma del gradiente vale 4,336711902204087 nel continuo e 4,336711902194546 nelle riprese: 9,5e-12
  in assoluto, 2,2e-12 in relativo. Secondo una stima, un errore di scala o di stato darebbe almeno 1e-7; questa è
  invece la grandezza attesa se una parte degli elementi del gradiente ridotto differisce di un'unità di arrotondamento.
- **546 in InfoNCE.** Norma del gradiente e norma del proiettore coincidono in tutte le cifre nei tre percorsi:
  9,023137809933427 e 0,21809202574133565. Coincidono anche le norme per obiettivo. Fra i campi di gradiente differiscono
  solo i due coseni per obiettivo, all'ultima cifra (circa 5e-16): lo stato dopo il 545 non è quindi identico bit per bit.
- **Salto.** Lo scarto diventa visibile al 547 in InfoNCE e al 546 in Rank-N-Contrast, circa sette ordini di grandezza
  sopra il seme.
- **Crescita lenta.** Dopo il salto lo scarto della norma del gradiente cresce ancora. Fra le finestre 548-561 e 562-576
  la media geometrica sale di 3,2 volte in InfoNCE e di 5,3 in Rank-N-Contrast. La pendenza della regressione è circa
  0,04 decadi per update. A 576 non c'è un plateau dimostrato. La norma del gradiente del proiettore resta invece allo
  stesso livello, circa 1e-3, fin dal 548.
- **Clipping.** Il gradiente è tagliato in tutti i 576 update. Secondo una stima, la differenza del fattore di taglio
  pesa al più circa lo 0,1% sui primi momenti: non è il canale principale.

**I campi di log non sono una prova.** Fra A e B, che hanno checkpoint identici, differiscono in quasi tutti gli update
le loss registrate: `total` in 32 update su 32, `jepa` in 29, per meno di 1e-6. A non essere deterministico è il valore
registrato di alcune somme; il loro gradiente lo è, per deduzione dal codice. Fra A e B restano identici, in tutti i 32
update, tutti i campi di gradiente.

### 2.3 Tutti i tensori, non solo quattro

Coppia continuo/A; la coppia continuo/B è identica. Fonte: `floating_tensor_relative_to_left`, 6.520 tensori per coppia.

| Grandezza | InfoNCE | Rank-N-Contrast |
|---|---|---|
| Primi momenti che differiscono | 270 su 270 | 266 su 270; i quattro del proiettore a zero esatto |
| Primi momenti: mediana | 0,46% | 0,29% |
| Primi momenti: 90° percentile | 0,59% | 0,40% |
| Primi momenti: massimo fuori dal proiettore | 0,71% (ID 194) | 0,59% (ID 47) |
| Primi momenti del proiettore | da 1,03% a 1,32% | 0 |
| Secondi momenti: mediana e massimo | 0,010% e 0,168% | 0,006% e 0,034% |
| Chiavi della coda diverse: mediana e massimo | 0,53% e 0,92% | 0,40% e 0,60% |
| Pesi dello studente dal 544 | 0,645% | 0,616% |
| Pesi dei blocchi `local` e `pool` dal 544 | 1,03% e 1,05% | 0,89% e 0,87% |
| Pesi del proiettore dal 544 | 0,93% | 0 |

- **Lo scarto è diffuso.** In InfoNCE differiscono tutti i 270 primi momenti; 85 superano lo 0,5% e le chiavi della coda
  arrivano allo 0,92%. Gli altri tensori rientrano nella soglia, ma di poco.
- **Il proiettore è un gruppo distinto.** I suoi quattro primi momenti occupano i primi quattro posti e valgono da 2,2 a
  2,9 volte la mediana. Fra il quinto tensore (0,71%) e il proiettore (1,03%) c'è un salto: nessun primo momento sta fra
  0,75% e 1%. Lo stesso vale per i secondi momenti: da 0,155% a 0,168% contro 0,050% del primo fra gli altri.
- **Il suo gradiente diverge più del resto.** Fra 548 e 576 la mediana dello scarto è 1,2e-3 per la norma del gradiente
  del proiettore e 3,3e-4 per la norma totale. Sull'encoder, la norma del gradiente InfoNCE diverge 3,6 volte più di
  quella di base: mediane 1,3e-3 e 3,5e-4. È coerente con il fatto che il proiettore riceve soltanto il gradiente
  InfoNCE. Non è una dimostrazione.
- **I pesi del proiettore non sono anomali.** Dal 544 divergono meno dei blocchi `local` e `pool`. Sono fuori scala solo
  i suoi momenti.
- **I due bracci non sono allo stesso livello.** Nei primi momenti dei 266 tensori comuni InfoNCE ha errori circa 1,6
  volte quelli di Rank-N-Contrast nelle mediane, da 1,4 a 1,5 nelle medie. I pesi divergono solo 1,05 volte. Uno degli
  analisti trova che la sensibilità per tensore è simile: correlazione 0,71 fra i logaritmi degli errori dei due bracci.
- **Secondi momenti del proiettore.** Il loro errore più alto potrebbe dipendere in parte dalla storia corta: il
  proiettore riceve gradiente solo dal passo 257, con peso pieno dal 512. È un'ipotesi.

### 2.4 Le chiavi della coda: un salto, non una crescita

Le chiavi sono calcolate dal teacher prima di ogni update. Uno degli analisti ha ricostruito l'update di inserimento
di ogni chiave: 1.001 chiavi dopo il fork. I conteggi che ho ricalcolato io sono coerenti: 910 chiavi diverse in
InfoNCE e 940 in Rank-N-Contrast, tutte in fondo alla coda.

| Update di inserimento | InfoNCE | Rank-N-Contrast |
|---|---|---|
| 545 (29 chiavi) e 546 (32 chiavi) | tutte identiche | tutte identiche |
| 547 (30 chiavi) | tutte identiche | tutte diverse, mediana 0,33% |
| 548 | tutte diverse, mediana 0,36% | tutte diverse, mediana 0,40% |
| 549-576 | mediana fra 0,46% e 0,59%, piatta | mediana fra 0,36% e 0,44%, piatta |

- **Niente valori intermedi.** Ogni chiave è identica oppure diversa di almeno circa lo 0,2%: il minimo fra le chiavi
  diverse è 0,2009% in InfoNCE e 0,1997% in Rank-N-Contrast.
- **Scala.** È l'ordine dell'unità di arrotondamento della mezza precisione bf16, 2^-8 = 0,39%.
- **Ipotesi.** Con bf16 una perturbazione minima non cambia nulla finché non sposta un arrotondamento; quando succede,
  cambiano quasi tutti gli elementi della chiave. Ne seguirebbe un livello dello scarto poco dipendente dall'ampiezza
  del seme. Non è verificato: c'è un solo fork per braccio.

### 2.5 Dove nasce la prima differenza

**Gli artefatti archiviati indicano, in forma indiretta, che la prima differenza nasce nella riduzione DDP dell'update 545.**

- **Fuori da DDP, identico.** Al 545 le norme dei gradienti per obiettivo `contrast_global_grad_base`, `_nce` e `_rnc` e i
  loro coseni sono identici bit per bit fra continuo e riprese. Il trainer li calcola prima del backward vero, con
  `autograd.grad` e un `all_reduce` per obiettivo su un vettore in ordine fisso (`contrastive_objectives.py:415-439`).
- **Che cosa coprono.** Sono norme di vettori già sommati fra i rank, sui 218 tensori condivisi dell'encoder. Restano
  fuori decoder, teste e proiettore: 52 tensori. L'identità è coerente con gradienti locali dell'encoder uguali, che
  però non sono osservati per rank né sul backward vero.
- **Dopo DDP, diverso.** Le grandezze calcolate sui gradienti ridotti da DDP differiscono di un arrotondamento: la norma
  totale e la norma del proiettore.
- **Raggruppamento.** Nella ripresa il 545 è la prima iterazione di un DDP nuovo: riduce con il raggruppamento di
  costruzione, un solo gruppo in precisione singola da 114.202.292 byte più uno piccolo. Il continuo usa i sette gruppi
  ricostruiti.

**Il raggruppamento diverso riguarda la riduzione del solo 545.**
- **Ordine nel torch del cluster.** In `_pre_forward` di torch `2.2.0a0+git6c8c5ad`, `set_runtime_stats_and_log()` (riga 1396) precede `_rebuild_buckets()` (riga 1413): il campo `has_rebuilt_buckets` è fotografato prima della ricostruzione fatta nello stesso forward. La traccia lo legge dopo il backward (`contrastive_target_resume.py:189-201`).
- **Conseguenza.** Lo «0» registrato al 546 descrive lo stato precedente: la riduzione del 546 usa già i sette gruppi.
- **Prova empirica.** Una sonda di uno degli analisti, su CPU con torch 2.14, mostra lo stesso ritardo: il log riporta 0
  all'iterazione in cui i gruppi ridotti sono già quelli ricostruiti.
- **Riscontro nei dati.** Al 546, in InfoNCE, le norme dopo la riduzione tornano identiche. In Rank-N-Contrast il 546
  differisce già a monte della riduzione, per lo stato lasciato dal 545.
- **Un solo forward per update.** `accum_steps` vale 1. Nelle riprese il contatore di iterazione della traccia avanza di
  uno per update da 545 a 555; poi il logger DDP non lo aggiorna più.
- **Limite.** Il comportamento del reducer C++ del torch 2.2 è dedotto da questi elementi, non letto nel sorgente.

**Ordine di grandezza.** Uno scarto relativo di 1e-12 sulla norma è quello atteso se una minoranza degli elementi del
gradiente ridotto differisce di un'unità di arrotondamento in precisione singola, per un diverso ordine di somma fra i
quattro rank. È una stima, non una misura.

**Il nodo è una causa molto improbabile.** I sette job hanno girato su sette nodi diversi (lettura `sacct` del 21/11):

| Run | Job | Nodo |
|---|---|---|
| C0 continuo | 65017240 | lrdn0113 |
| InfoNCE continuo | 65017310 | lrdn0389 |
| Rank-N-Contrast continuo | 65020151 | lrdn0292 |
| InfoNCE ripresa A | 65033016 | lrdn0257 |
| InfoNCE ripresa B | 65033756 | lrdn0362 |
| Rank-N-Contrast ripresa A | 65033804 | lrdn2576 |
| Rank-N-Contrast ripresa B | 65034107 | lrdn0680 |

Le riprese A e B danno checkpoint identici byte per byte su nodi diversi, in entrambi i bracci. I due continui, su nodi
diversi, mostrano lo stesso schema al 545. Un effetto limitato al nodo di un continuo non è escluso in assoluto.

### 2.6 Quanto i dati sostengono ciascuna ipotesi

| Ipotesi | Che cosa dicono i dati | Giudizio |
|---|---|---|
| Difetto di ripristino dello stato salvato | confine identico in 24 tracce su 24; loss del 545 identiche; chiavi anteriori al fork esatte; nessuna specificità di InfoNCE a 545 e 546 | esclusa dagli artefatti |
| Differenza numerica ripetibile | prima differenza dopo la riduzione DDP del 545, a livello di arrotondamento; salto uno o due update dopo; A e B identiche su nodi diversi | sostenuta; il nesso causale è osservativo e aggregato |
| Sensibilità del criterio | la differenza dei primi momenti viene tutta dai gradienti dopo il fork; scarto diffuso in entrambi i bracci; Rank-N-Contrast senza segnale sui quattro tensori; scarto che cresce con l'orizzonte | sostenuta |

**Che cosa resta indimostrato:**
- **Sufficienza causale.** Nessun intervento ha mostrato che, a parità di raggruppamento, lo scarto sparisce. Il
  meccanismo dentro la riduzione (algoritmo e canali di NCCL) non risulta dagli artefatti.
- **Gradienti locali.** Non sono archiviati per rank. Per i 52 tensori fuori dall'encoder condiviso, proiettore
  compreso, non c'è alcun controllo a monte della riduzione: per il proiettore c'è solo la norma dopo la riduzione, che
  al 545 differisce di 1e-13.
- **Fattore proprio del proiettore.** Il salto fra il proiettore e gli altri tensori è coerente con il gradiente InfoNCE
  più rumoroso, ma non è dimostrato.
- **Rapporto fra i bracci.** Non è spiegato perché in InfoNCE i primi momenti dell'encoder differiscano più che in
  Rank-N-Contrast. Ipotesi: momenti più piccoli in norma per cancellazione, quindi denominatore più piccolo.
- **Orizzonte.** C'è un solo fork per braccio e 32 update. Con più update lo scarto cresce ancora: Rank-N-Contrast
  potrebbe superare l'1% su un orizzonte più lungo, e InfoNCE con più tensori.
- **Scala assoluta.** Norme e direzioni dei quattro momenti non sono note senza leggere i checkpoint.

### 2.7 Precisazioni sulla lettura di GPT

La lettura di GPT regge nei fatti. Sei precisazioni:

1. **545-546.** La riduzione con raggruppamento diverso è solo quella del 545. La frase della lettura, «le tracce
   riportano la stessa composizione e lo stesso ordine del continuo dal 547», descrive correttamente il log. Sono sfasati
   di un update, se letti come riduzione, il prompt («Ai passi 545–546 i gruppi di riduzione DDP differiscono»),
   `ddp_trace_readout.json` e la mia nota P-7 del 20/11.
2. **9,5e-12.** È la differenza assoluta della norma del gradiente totale del braccio InfoNCE, 2,2e-12 in relativo. La
   norma del gradiente InfoNCE sui tensori condivisi dell'encoder, calcolata fuori da DDP, al 545 è identica.
3. **Rank-N-Contrast come confronto.** La lettura lo riconosce già: le due loss «rispondono quindi a domande diverse».
   Il dato in più è che lì i momenti dei quattro tensori sono zeri esatti: per quei tensori il confronto non ha segnale.
   Resta un confronto valido per il resto della ripresa e per il decorso generale.
4. **«Gli altri tensori controllati di Adam e della coda rientrano nella soglia».** Vero, ma di poco: mediana 0,46%,
   massimo fuori dal proiettore 0,71%, chiavi fino a 0,92%.
5. **Localizzazione.** GPT la considera aperta. Le norme calcolate fuori da DDP, non usate nella sua lettura, danno già
   un'indicazione indiretta.
6. **Stesso nodo.** Non serve: A e B sono identiche su nodi diversi. I nodi non erano negli artefatti raccolti; conviene
   registrarli nei prossimi job.

## 3. La prova minima

### 3.1 Che cosa chiede la regola registrata

- **Proposta firmata.** La proposta faa4a8d7, firmata da Simone il 20/11, dice: «Un eccesso ripetibile continuo/reload,
  con reload concordi, richiede localizzazione prima di promuovere il profilo».
- **Limite registrato.** La voce del 20/11 in `decisioni.md`, scritta da GPT prima dei dati (commit `9abde295`), lo
  registra come coperto dalla firma: «La firma copre la proposta scientifica `faa4a8d7` e i seguenti limiti del
  preflight». La mia domanda a Simone nominava solo la proposta e il tetto. Il limite dice:
  «Se i due reload concordano e il continuo eccede il limite, la fase si ferma. Localizzare al primo passo, per rank,
  tensore e gruppo DDP, a partire da 545–546. Un cambiamento dell’ordine di riduzione è un’ipotesi da verificare.»

Il caso è questo, e la fase è ferma. Confronto fra ciò che il limite chiede e ciò che gli artefatti contengono:

| Richiesto dal limite | Disponibile negli artefatti |
|---|---|
| Primo passo | sì: 545, con il 546 come riscontro |
| Per rank | no: solo grandezze già ridotte fra i rank |
| Per tensore | no: norma totale, norma del proiettore, norme per obiettivo sull'encoder |
| Per gruppo DDP | solo la composizione dei gruppi, non i valori |
| Verifica dell'ipotesi sull'ordine di riduzione | indiretta e osservativa |

**La prova proposta da GPT è quindi il passo previsto dal limite registrato, non un'aggiunta.** Lo attua però solo con la
prima correzione della sezione 3.3: la lettura confronta i soli quattro parametri del proiettore e non registra i gruppi
DDP. Rinunciare alla prova perché gli artefatti bastano è una deroga a una regola scritta prima dei dati. Può essere una
scelta ragionevole, ma va dichiarata e firmata da Simone, non letta dentro la regola.

### 3.2 Lettura CPU dei checkpoint

Le misure della sezione 1 sui checkpoint esistenti di InfoNCE (fork a 544, continuo e ripresa a 576) e di
Rank-N-Contrast per confronto.

- **Che cosa dà.** La scala assoluta e la forma dello scarto; può spiegare il rapporto fra i bracci.
- **Che cosa non dà.** Il punto d'origine: i checkpoint non contengono i gradienti intermedi. Non attua il limite.
- **Condizioni.** Gli esiti attesi della tabella vanno fissati prima di leggere. Lo script va tenuto fuori
  dall'inventario del digest, oppure serve una nuova release.

### 3.3 La prova GPU proposta da GPT

Un continuo InfoNCE e una ripresa, con gradienti locali, ridotti, dopo il taglio e momenti agli update 545-547.

- **Che cosa mostrerebbe, se la lettura di questa review è giusta.** Al 545 gradienti locali identici e ridotti diversi
  di un arrotondamento. Al 546, in InfoNCE, gradienti locali e ridotti identici e momenti diversi per l'eredità del 545.
  Al 547 tutto diverso, per propagazione.
- **Che cosa non mostrerebbe.** Resta osservativa: non prova che il raggruppamento basti a spiegare lo scarto. Non tocca
  la sensibilità del criterio.
- **Rischio.** Il continuo va rifatto da zero, e non risulta dimostrato che un continuo ripetuto riproduca byte per byte
  quello archiviato. Se non lo riproduce, il confronto con l'originale si perde.
- **Il criterio «differenza già locale, quindi a monte»** vale solo al 545. Al 547 i gradienti locali differiscono
  comunque, per effetto degli update precedenti.

**Se si esegue, sei correzioni al disegno:**
1. **Tutti i 270 parametri.** Al 545 il proiettore differisce di 1e-13 sulla norma, compatibile con pochissimi elementi:
   il seme sta soprattutto altrove. Servono, per parametro e per rank, impronta, norma in doppia precisione e numero di
   elementi diversi, più la posizione di ogni parametro nel gruppo DDP; i vettori completi solo per 266-269.
2. **Osservazione senza alterare.** Un hook di comunicazione DDP che copia il buffer e poi esegue la stessa riduzione,
   oppure hook sui tensori dei parametri. Sconsigliata una seconda copia del modello senza DDP: consuma generatori
   casuali e non garantisce gli stessi bit.
3. **Verifica che la strumentazione non cambi il fenomeno.** I run strumentati devono riprodurre i checkpoint esistenti
   byte per byte, `4743abaf…` per il continuo e `bb39f339…` per la ripresa, e i valori archiviati della norma del
   gradiente agli update 545-547. In caso contrario la prova è inconclusiva sull'originale.
4. **Falsificatori scritti prima.** Gradienti locali già diversi al 545 smentiscono la riduzione come origine. Uno scarto
   di grandezza non da arrotondamento al 545, per esempio oltre 1e-6 su un tensore, riaprirebbe il difetto di stato.
5. **Nodo.** Registrare nome del nodo e identificativi delle GPU. Lo stesso nodo non è necessario.
6. **Tetto.** I conti di GPT sono corretti: 997-1.049 s, cioè 1,11-1,17 GPU-ore. Il tetto di 2 GPU-ore vale solo se
   continuo e ripresa stanno in un unico job da 30 minuti.

### 3.4 Alternativa: due riprese dallo stesso fork

**Due riprese InfoNCE dal fork esistente, strumentate al solo update 545.**
- **Ripresa ordinaria.** Come CN_A, più la registrazione per rank e per tensore dei gradienti locali e ridotti.
- **Ripresa con riscaldamento.** Prima del 545 un'iterazione DDP a effetto nullo fa ricostruire i gruppi; poi si
  ripristinano generatori casuali e gradienti. Stessa registrazione.

| Osservazione | Esito atteso se la causa è l'ordine di riduzione |
|---|---|
| Gradienti locali del 545, per rank e tensore | identici fra le due riprese |
| Gruppi DDP al 545, letti dall'hook di comunicazione | quelli di costruzione nella ordinaria; i sette del continuo in quella con riscaldamento |
| Gradienti ridotti del 545, per tensore e gruppo | diversi di un arrotondamento in una parte degli elementi |
| Norma del gradiente al 545 con riscaldamento | 4,336711902204087, il valore archiviato del continuo |
| Checkpoint a 576 della ripresa ordinaria | `bb39f339…`, come CN_A e CN_B |
| Checkpoint a 576 della ripresa con riscaldamento | tutti i tensori identici a quelli del continuo; file `4743abaf…` se coincidono anche i metadati |

- **Perché conviene.** Dà la granularità chiesta dal limite e verifica l'ipotesi con un esito binario. Non rifà i 544
  update iniziali e non dipende dalla ripetibilità del continuo. Un'identità di tutti i tensori non può nascere per caso.
- **Esiti alternativi.** Update 545 uguale al continuo e 576 diverso: esiste un'altra dipendenza dall'età del processo.
  Update 545 ancora diverso con gruppi verificati uguali: il raggruppamento non è la causa.
- **Che cosa proverebbe l'identità.** Che lo stato di processo ricostruito dal riscaldamento basta a riprodurre il
  continuo. L'attribuzione al solo ordine di riduzione viene dal confronto per tensore e per gruppo fra le due riprese.
- **Limiti.** È una proposta di disegno, non codice provato. Tocca l'inventario del digest: servono review e release. Il
  riscaldamento non deve essere registrato come update 545 e non deve lasciare tracce nei gradienti e nello stato DDP: le
  impronte di confine non lo coprono. Il log DDP è in ritardo di un update: i gruppi del 545 vanno letti dall'hook.
- **Criterio sui checkpoint.** L'impronta `final_checkpoint_sha256` è calcolata sui byte del file, che contiene anche
  configurazione e metadati. Il criterio va fissato tensore per tensore; l'identità del file è una conferma in più, e
  una differenza dei soli metadati non conta contro l'ipotesi.
- **Ripiego.** Se il riscaldamento è troppo rischioso da scrivere, resta la prova di GPT con le sei correzioni.

### Costi a confronto

Le core-ore coincidono con le ore locali, perché il billing conta le CPU.

| Opzione | GPU-ore attese | Tetto in GPU-ore | Core-ore attese | Attua il limite registrato |
|---|---|---|---|---|
| Artefatti archiviati, già fatto | 0 | 0 | 0 | no: indicazione indiretta |
| Lettura CPU dei checkpoint | 0 | 0 | circa 0,3, stima | no |
| Prova di GPT, continuo più ripresa | 1,11-1,17, prima della strumentazione | 2 con un unico job da 30 minuti | 8,9-9,3 | sì con la prima correzione, in forma osservativa |
| Due riprese dal fork | 0,47-0,59, prima della strumentazione | 1 con un unico job da 15 minuti | 3,8-4,7 | sì, con verifica dell'ipotesi |

**Budget e tempi.** Secondo il consuntivo di 3-launcher in `decisioni.md`, non ricalcolato da me, la nuova coorte ha
consumato circa 4,74 GPU-ore e 47,33 core-ore, contro il tetto firmato di 18 e 169,6. Una diagnostica nuova non è fra le
17 allocazioni del piano: richiede il via di Simone e una release propria. Leonardo è fermo dal 23/11 alle 08:00 al 25/11
alle 08:00.

## Raccomandazione per Simone

1. **Domanda che si può chiudere ora.** Il fallimento non viene da un difetto nel salvataggio o nel ripristino. Lo stato
   salvato a 544 è ripreso esattamente e le riprese sono riproducibili byte per byte, anche su nodi diversi. Una ripresa
   non riproduce però il continuo bit per bit: il raggruppamento DDP non fa parte del checkpoint.
2. **Decisione che spetta a te.** Il limite registrato il 20/11 chiede una localizzazione per rank, tensore e gruppo DDP.
   Hai due strade:
   - **Deroga dichiarata.** Accetti come sufficiente l'indicazione indiretta di questa review e lo firmi come deroga.
     Sul piano scientifico la considero convincente: mi aspetto che la prova la confermi.
   - **Prova mirata.** Si attua il limite. È la strada più pulita, perché la regola è stata scritta prima dei dati.
3. **Primo controllo, se scegli la prova.** Le due riprese della sezione 3.4: 0,47-0,59 GPU-ore attese, tetto 1.
   Costano circa metà della proposta di GPT e verificano l'ipotesi con un esito binario.
4. **Risultato che giustifica tornare a discutere dei training lunghi.** Con la prova: gradienti locali identici per rank
   e tensore al 545, differenze di arrotondamento solo dopo la riduzione, e tensori del checkpoint della ripresa con
   riscaldamento identici a quelli del continuo. Con la deroga: la tua firma, lasciando visibile il verdetto storico `FAILED_REQUIRES_REVIEW`.
5. **Che cosa mi farebbe cambiare idea.** Gradienti locali già diversi al 545, uno scarto di grandezza non da
   arrotondamento, oppure tensori diversi a gruppi verificati uguali. Nella lettura CPU, una differenza dei momenti di
   forma sistematica.

**Tre avvertenze per i training lunghi:**
- **Stessa segmentazione nei tre bracci.** Una ripresa cambia la traiettoria a livello di rumore numerico. Il rapporto
  prevede per ogni braccio due allocazioni, fino a 5.000 e fino a 19.618: se fra le due c'è una ripresa, deve cadere
  allo stesso passo per C0, InfoNCE e Rank-N-Contrast.
- **Lettura dei confronti.** Differenze fra bracci dello stesso ordine di questo scarto numerico non sono
  interpretabili. Lo scarto cresce con l'orizzonte e qui è misurato solo a 32 update.
- **Criterio futuro.** Un cancello sull'1% dei primi momenti dopo 32 update misura la velocità di divergenza numerica,
  non la correttezza della ripresa. Per le prossime coorti suggerisco a GPT di separare i controlli esatti (confine,
  primo update, identità fra riprese) dalla deriva, da riportare come descrittiva. Per questa coorte qualsiasi regola
  nuova sarebbe post hoc e va dichiarata tale.

L'utilità delle due loss su posa e classificazione resta una domanda separata, che questa review non tocca.

## Limiti

- **Checkpoint non letti.** Stanno su Leonardo: norme assolute e direzioni dei momenti non sono note.
- **Gradienti non archiviati.** L'indicazione sull'origine è indiretta: norme in doppia precisione, già ridotte fra i
  rank, e fuori da DDP solo per i 218 tensori condivisi dell'encoder.
- **Un solo punto.** Un fork a 544 per braccio, 32 update, due bracci.
- **Reducer C++.** Del torch del cluster ho letto l'ordine nel sorgente Python; il comportamento del reducer è dedotto.
- **NCCL.** Algoritmo, protocollo e canali usati su Leonardo non risultano dagli artefatti.
- **Stime.** I conti sull'ordine di grandezza del seme, sul salto in bf16 e sul peso del clipping sono stime.
- **Fonti secondarie.** Vengono dai rapporti degli analisti, non ricalcolati da me: i 5.965 conteggi, la riproduzione
  dell'impronta dell'ordine, la sonda DDP su CPU, l'update di inserimento di ogni chiave e la correlazione 0,71.
- **Disegno delle due riprese.** Non è provato: fattibilità del riscaldamento a effetto nullo e costi sono stime.
- **Prove.** Script, uscite, rapporti degli analisti e dei verificatori e lettura di Leonardo sono nella cartella
  `claude_review_contrastive_targets_resume_science_20261121_evidence/`.

## Prompt per GPT

```text
Review Claude della ripresa InfoNCE (RMS100/fast125, evidenze f23b7cca, lettura adb6e3d6 con l'aggiunta a 918245b6): docs/reviews/claude_review_contrastive_targets_resume_science_20261121.md sul ramo origin/worktree-claude-training-engine, prove nella cartella _evidence/ accanto. Nessun lancio. Il verdetto storico FAILED_REQUIRES_REVIEW resta.

Verdetto: differenza numerica ripetibile resa visibile dal criterio dell'1%, non un difetto nel salvataggio o nel ripristino dello stato. Mappatura (266-269 = pesi e bias dei due strati del proiettore), denominatori e valori sono confermati. Il giudizio poggia su prove osservative e aggregate.

Fatti dagli artefatti già archiviati:
- al 545 le norme dei gradienti per obiettivo calcolate fuori da DDP (contrast_global_grad_base/nce/rnc e coseni, sui 218 tensori condivisi dell'encoder) sono identiche bit per bit fra continuo e riprese; fra i campi di gradiente differiscono solo le grandezze dopo la riduzione DDP (grad_norm 2,2e-12 relativo; 9,5e-12 è l'assoluto; norma del proiettore 1e-13);
- la riduzione con raggruppamento diverso è solo quella del 545: nel torch 2.2.0a0 del cluster set_runtime_stats_and_log() (distributed.py:1396) precede _rebuild_buckets() (:1413), quindi has_rebuilt_buckets è in ritardo di un update; al 546 in CN grad_norm e norma del proiettore sono identiche al continuo (fra i campi di gradiente differiscono solo i coseni, di 5e-16);
- tutti i 270 exp_avg di CN differiscono (mediana 0,46%, massimo fuori dal proiettore 0,71%, nessun primo momento fra 0,75% e 1%) e i 266 attivi di CR (mediana 0,29%, massimo 0,59%); in CR i momenti 266-269 sono zeri esatti;
- le chiavi della coda inserite a 545-546 (CN anche 547) sono identiche; dopo, ogni chiave differisce di almeno circa lo 0,2% (scala bf16);
- dopo il salto lo scarto di grad_norm cresce ancora: media geometrica 3,2 volte (CN) e 5,3 volte (CR) fra 548-561 e 562-576, circa 0,04 decadi per update; l'esito dell'1% dipende dall'orizzonte;
- i sette job hanno girato su sette nodi diversi (sacct): A e B identiche byte per byte su nodi diversi.

Sul limite registrato il 20/11 (decisioni.md, 9abde295: «Localizzare al primo passo, per rank, tensore e gruppo DDP [...] Un cambiamento dell’ordine di riduzione è un’ipotesi da verificare»): gli artefatti non hanno quella granularità. La tua prova è il passo previsto, ma lo attua solo se copre tutti i parametri e i gruppi DDP; rinunciarvi è una deroga che spetta a Simone dichiarare.

Se si fa la prova, propongo due riprese CN dal fork esistente, strumentate al solo 545: una ordinaria (deve ridare bb39f339...) e una con un'iterazione di riscaldamento DDP a effetto nullo prima del 545 (attesa: gradienti locali identici per rank e tensore, gruppi DDP letti dall'hook di comunicazione perché il log è in ritardo, ridotti diversi di un arrotondamento, tensori del checkpoint a 576 identici a quelli del continuo). Il criterio va fissato tensore per tensore: l'impronta del file 4743abaf... copre anche configurazione e metadati. Attese 0,47-0,59 GPU-ore, tetto 1 con un unico job da 15 minuti. È un disegno, non codice provato; il ripiego è la tua prova con sei correzioni: tutti i 270 parametri e la posizione nel gruppo DDP, hook di comunicazione o sui tensori, riproduzione byte per byte dei checkpoint esistenti come verifica della strumentazione, falsificatori scritti prima (gradienti locali già diversi al 545; oltre 1e-6 su un tensore), nodo registrato ma non vincolato, tetto di 2 GPU-ore solo con un unico job. Il criterio "già locale, quindi a monte" vale solo al 545. Nota: non risulta dimostrato che un continuo rifatto da zero riproduca 4743abaf....

Restano indimostrati: sufficienza causale del raggruppamento, fattore proprio del proiettore, rapporto fra i bracci nei momenti dell'encoder (1,6 nelle mediane), comportamento oltre 32 update, scala assoluta dei quattro momenti (lettura CPU facoltativa dei checkpoint, circa 0,3 core-ore). Ogni nuova soglia definita ora sarebbe post hoc per questa coorte.
```
