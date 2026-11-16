# Review Claude: confronto contrastivo InfoNCE / Rank-N-Contrast, consegna di GPT `db5ac41` (linea 3), 16/11/2026

Richiesta di Simone, 16/11: «Ho autorizzato GPT a preparare entrambe le contrastive e il confronto. Ti chiedo una review prima dei
lanci, senza sottomettere job.» e, in chiusura: «Distingui approvazione del disegno, implementazione e prerequisiti del collaudo.
Il training non è rilasciato: mancano le misure dei benchmark e il preventivo esecutivo. Pubblica rilievi riproducibili e verdetto
sul tuo ramo, aggiornando entrambe le sezioni del registro. Nessun lancio.» Ho seguito il prompt di GPT
`docs/prompt_claude_contrastive_comparison_20261116.md` (`77b7003`). Oggetto: ramo `origin/codex/tokenizer-logcontrast-20261104`,
codice `db5ac41`, diff da `924e19d` (30 file, 4.462 righe aggiunte). Ruolo: revisore della linea 3. Nessun job, nessun file della
linea 3 modificato, nessun cambio di ricetta: tutto è stato eseguito sul Mac, in copie esterne al repository, salvo una lettura
sul nodo di login di Leonardo (sezione 9).

## Verdetti

| Che cosa | Verdetto | In breve |
| --- | --- | --- |
| **Disegno** | **Schema approvato; tre regole DA RIVEDERE prima dei dati** | C0/CN/CR con stessa inizializzazione, dati, dose e testa è un confronto valido. Tre regole vanno riviste. Il cancello di copertura non è superabile (D-1) e la regola di lettura dà «irrisolto» quasi sempre (D-2): entrambi dimostrati sui numeri veri. Il riferimento dei pesi è il massimo transitorio della base (D-3): misurato sul log; la conseguenza vale se la norma del termine non cala |
| **Implementazione** | **APPROVATA CON CORREZIONI** | Nessun errore di calcolo trovato in loss, memoria, trainer, gradienti distribuiti, ripresa e comparatore. Dieci gruppi di rilievi e una precisazione (I-1...I-11), tutti minori nel calcolo. Uno, I-1, è un blocco di processo: va chiuso prima del collaudo, perché la correzione tocca sorgenti e config che la release fissa |
| **Prerequisiti del collaudo** | **Non ancora soddisfatti** | La calibrazione, così com'è, finisce di certo in `FAILED_REQUIRES_REVIEW` e ferma il collaudo. Elenco ordinato in sezione 7 |

Sui cinque punti del prompt di GPT:

| Punto | Esito |
| --- | --- |
| 1. Loss e correzioni | **Approvato.** RnC corretta conforme al paragrafo 3 in 14 punti su 14; InfoNCE conforme nei punti controllati, con un punto ambiguo (regola dei trial, D-11) e uno scarto al confine dei 4 s (I-10a); oracoli indipendenti entro 1e-13. Restano test deboli (sezione 6) |
| 2. Trainer e distribuzione | **Approvato con correzioni minori.** Gradiente distribuito = gradiente globale a 2 e 4 processi, anche con lunghezze diverse e un rank senza posa. Dei sei test distribuiti 5 passano e 1 fallisce per la tolleranza del test (I-2). Il percorso storico resta identico, ma scattano le guardie delle campagne chiuse (I-1): da chiudere prima del collaudo. Poi gli arresti (I-3, I-4) |
| 3. Peso e copertura | **Da rivedere.** Il percorso di misura è corretto; le due regole vanno riviste: D-1 e D-3 |
| 4. Confronto scientifico | **Da rivedere.** Il comparatore calcola ciò che il protocollo dice; la regola di lettura, con le ampiezze vere, dà «irrisolto» quasi sempre: D-2. Difetti minori del comparatore in I-9 |
| 5. Preparazione e rilascio | **Corretto non rilasciare il training.** Il rifiuto della fase `training` è robusto. Mancano strumenti e criteri del collaudo (I-5, I-6, I-7, I-8) e il budget (sezione 7) |

**I tre punti che contano, con i numeri.**

1. **Il cancello di copertura non è superabile da `capgmyo`** (D-1). Sul flusso vero 0 query ammissibili su 1.461, 0 blocchi
   superati su 18, e nel pannello di calibrazione 13 finestre contro le 32 richieste. Non è un errore del codice: è la soglia per
   dataset.
2. **Il «10%» dei pesi è riferito al massimo transitorio della base** (D-3). La norma del gradiente di base scende da 29,8 a 6,7
   prima che il peso contrastivo si accenda, e a 2,65 a fine corsa. A norma del termine costante il rapporto vale 0,44-0,49
   all'inizio del peso pieno (update 513-1.024) e circa 1 in media sul tratto a peso pieno.
3. **La regola di lettura non distingue un braccio invariato da uno peggiore su P2** (D-2). Con le ampiezze dei bracci storici la
   non inferiorità su P2 è sostenuta in 1 contrasto su 12, il guadagno utile in 0 su 12, e 24 terne storiche su 24 sono
   «irrisolte».

**Stato alla consegna.** Punte rilette con `git ls-remote` alle 18:58 del 16/11: ramo di GPT `1d87e8d`, coordinamento `8eba5e4`,
`main` `e114a78`. Durante la review GPT ha pubblicato `1d87e8d` (16:58): aggiunge solo `examples/matlab_contrastive_losses/` e due
voci di registro; `git diff db5ac41 1d87e8d -- src scripts tests results/infra/contrastive_comparison_20261116` è vuoto. Il codice
rivisto è quindi quello in punta. **Gli esempi MATLAB sono fuori dal perimetro di questa review: non li ho letti né eseguiti.**

## 1. Materiali, metodo, livelli di prova

- **Alberi.** `db5ac41` e `924e19d` estratti con `git archive` in cartelle esterne; nessun file scritto nel repository di GPT.
  Ambiente: Mac, CPU, torch 2.14.0 (Leonardo ha torch 2.2), `gloo` per i processi multipli. Soli tensori sintetici. Del flusso
  vero solo indici e metriche: log del controllo 30M con la ricetta T100-G (19.618 update, campi `origin`, `pose_windows`,
  `grad_norm`) e manifest v1.1. Per la regola di lettura, i conteggi per soggetto dei bracci storici già nel repository. Nessun
  segnale da wearable letto.
- **Che cosa ho fatto di persona** (script e uscite nella cartella di prove, `mie/`): i 153 test locali di GPT e i sei
  distribuiti; la diagnosi del test fallito; la copertura InfoNCE sull'intero flusso; i conteggi del pannello di calibrazione; le
  norme del gradiente del run vero e il rapporto che ne segue; la regola di lettura sui bracci storici col bootstrap congelato; la
  suite completa sui due alberi; il costo CPU della memoria; i 203 test aggiuntivi proposti dagli agenti (200 passati, 1 saltato,
  2 marcati come difetto atteso); la lettura delle righe che cito.
- **Venti agenti in dieci dimensioni** (RnC, InfoNCE, innesto nel trainer, distribuito, parità col percorso storico, calibrazione,
  copertura, comparatore, disegno, rilascio). Per ogni dimensione un agente ha cercato i rilievi e un secondo, con codice e dati
  propri, ha provato a confutarli uno per uno. I rapporti completi sono nella cartella di prove, `rapporti_agenti/`.
- **Rilettura della bozza.** Prima di pubblicare, quattro lettori indipendenti hanno confrontato questa review con le fonti:
  numeri, citazioni, riferimenti al codice, completezza. Hanno trovato 8 errori e 71 imprecisioni, tutti corretti (sezione 9).
- **Etichette.** **[io]** = eseguito o letto da me. **[2A]** = trovato da un agente e confermato da un secondo con prove
  indipendenti. **[1A]** = un solo agente, non riverificato. Dove non c'è etichetta vale [2A] per il rilievo; i dettagli aggiunti
  dal solo verificatore sono [1A], e lo scrivo nei punti che contano.

Esito delle verifiche avversarie, rilievo per rilievo:

| Dimensione | Rilievi | Confermati | Parziali | Rifiutati | Rilievi nuovi del verificatore |
| --- | ---: | ---: | ---: | ---: | ---: |
| RnC | 9 | 8 | 1 | 0 | 3 |
| InfoNCE e memoria | 9 | 8 | 1 | 0 | 4 |
| Innesto nel trainer | 8 | 7 | 0 | 1 | 1 |
| Distribuito | 7 | 6 | 1 | 0 | 1 |
| Parità col percorso storico | 8 | 7 | 1 | 0 | 2 |
| Calibrazione | 9 | 6 | 3 | 0 | 3 |
| Copertura | 8 | 6 | 2 | 0 | 3 |
| Comparatore | 10 | 9 | 1 | 0 | 3 |
| Disegno | 10 | 8 | 2 | 0 | 3 |
| Rilascio | 12 | 10 | 2 | 0 | 2 |

In tutto 90 rilievi verificati: 75 confermati, 14 confermati in parte, 1 rifiutato; 25 rilievi nuovi dei verificatori. Uso sempre
la versione corretta dal verificatore. Tre correzioni cambiano il senso di affermazioni che altrimenti avrei riportato (dettaglio
in sezione 9): che C0 sia uguale al percorso storico bit per bit vale solo per certi batch; che dopo un'eccezione la catena
riparta dal checkpoint è falso, si ferma; che l'esito favorevole della regola di lettura sia irraggiungibile è troppo forte, è
improbabile.

## 2. Che cosa regge

**Loss.**
- **RnC corretta.** Conforme al paragrafo 3 del protocollo in 14 punti su 14, fra cui: solo positivi `all`, esclusione simmetrica
  dello stesso istante fisico, pareggi nello stesso denominatore. Nel calcolo non c'è più la sentinella: logsumexp cumulativo sul
  solo prefisso valido, a gruppi di righe con lo stesso numero di candidati. Contro un oracolo numpy scritto dal protocollo, nel
  punto di chiamata vero: errore relativo al più 4,4e-16 sul valore, gradiente nel riempimento esattamente zero, supporto uguale a
  quello della MSE di posa. Le coppie escluse coincidono con quelle a target identico sulla griglia vera del loader.
- **InfoNCE.** Valore e gradiente contro un oracolo scalare: scarto 8,8e-14. Denominatore sulle 32 finestre globali anche con un
  rank vuoto. Regole dei negativi (stesso dataset, non la stessa finestra, stessa registrazione solo stesso trial a 4 s) conformi
  nei punti controllati; due eccezioni minori in D-11 e I-10a. Memoria: FIFO, età, capienza, deduplica e ripresa uguali a un
  oracolo sulla storia in 400 flussi e 24.000 update, 0 differenze.
- **I due rilievi principali del prototipo sono chiusi.** P-1 e P-2 della mia review del prototipo (`b3a3a94`) non si
  ripresentano: `cross` è rifiutato (`model/rank_contrast.py:31`) e i candidati sono globali.

**Trainer e percorso distribuito.**
- **Gradiente distribuito = gradiente globale.** I sei test nuovi girano sul Mac: 5 passati, 1 fallito per la tolleranza del test
  (I-2) **[io]**. Media dei gradienti locali contro gradiente pieno: entro 1,4e-13 a 2 e 4 processi. 24 lanci e 72 passi di un
  secondo agente, con finestre da 1 a 4 s, un rank senza posa, RnC vuota e una sola finestra con posa: tutti uguali al processo
  singolo (gradienti entro 2,8e-13). I 10 test distribuiti indipendenti proposti passano sul codice consegnato **[io]**.
- **Algebra dei fattori.** Numeratore RnC replicato, denominatore uguale al numero di processi, la raccolta consegna R volte la
  derivata globale, la `all_reduce` della raccolta precede ogni gradiente di parametro (36 controlli su 36). I quattro mutanti dei
  fattori sono rilevati dai test di GPT.
- **Senza contrastive il codice consegnato dà gli stessi run della base.** Base `924e19d` contro `db5ac41`, sei casi (config vero
  di T100-G rimpicciolito, accumulo 2 e 3, DDP a 3 e 4 processi): 18 checkpoint, 9.471 tensori, 0 diversi. Uguale anche contro il
  codice storico del run T100-G (`afb644e`).
- **C0 è un controllo appaiato valido.** Stessi dati, maschere e generatore nei tre bracci; update a peso zero identici bit per
  bit; gradiente del proiettore in C0 esattamente zero. Sotto lo stesso percorso di calcolo, contratto senza contrastive e C0
  coincidono bit per bit in 32 batch su 32.
- **Rampa.** Update 1-256 a peso zero, 257 = λ/256, 512 = λ pieno, 255 valori intermedi esatti.
- **Ripresa.** Bit per bit a un processo e a 2 processi, stato contrastivo compreso, anche con arresto per limite di tempo dentro
  un segmento dello stream. A 4 processi su `gloo` solo entro l'arrotondamento (scarto relativo 1,5e-7 - 2,1e-7): accade identico
  nella base senza codice contrastivo (sezione 7).
- **Activation checkpointing e diagnostica.** Gradienti identici bit per bit con e senza activation checkpointing, con e senza
  diagnostiche; norme e coseni registrati ricompongono la norma del gradiente applicato entro 1,9e-6.

**Calibrazione, comparatore, rilascio.**
- **Percorso di misura della calibrazione.** Pesi fermi bit per bit, optimizer mai usato, memoria che avanza, norme uguali a un
  processo e a 4 processi con activation checkpointing **[io per il test a 4 processi]**. Trainer vero in calibrazione sotto DDP a
  4 processi per 512 update, poi lettore vero: hash iniziale uguale a student e teacher finali, 0 passi di optimizer, 8.192 query
  nel blocco, `PASSED_FOR_REVIEW` (modello minuscolo) [1A]. Su 45 ingressi avversi il lettore ne ferma 33 (26 con eccezione, 7 con
  `FAILED_REQUIRES_REVIEW`), ne accetta 10 a ragione e 2 per lassità innocue; non applica clamp.
- **Comparatore.** Le balanced accuracy ricalcolate dai conteggi pubblicati, con lo schema del comparatore, coincidono con quelle
  pubblicate **[io]**; con le funzioni vere del comparatore lo confermano due agenti. Una reimplementazione replica per replica
  coincide entro 4,8e-14 su 48 contrasti; le estrazioni sono identiche fra numpy 1.26.4 (cluster) e 2.3.5 (Mac) [1A].
- **Rilascio.** I 31 + 4 hash della consegna coincidono. La fase `training` è rifiutata in 10 tentativi su 10. I config a pesi
  `null` non costruiscono il runtime. Nessun lancio risulta dal registro comune dei job.
- **Test di GPT.** 153 passati, 6 deselezionati, in 129 s **[io]**.

## 3. Tre decisioni da prendere prima dei dati

Le decisioni sono di GPT e di Simone. Io porto fatti, numeri e opzioni; dove aggiungo una lettura mia lo dico.

### D-1. Cancello di copertura: `capgmyo` non può superarlo. Blocca calibrazione e collaudo

**Dove.** Regola: protocollo §4 riga 69 («90% complessivo e 80% per dataset dopo il riempimento, con almeno 32 query osservate per
dataset a quota positiva») e §8 riga 131 («La calibrazione usa il proprio blocco train di 256 batch e deve superare i criteri
prima del rilascio»). Codice: `src/wearusfm/training/contrastive_objectives.py:252-263` (riga 261), lettore
`scripts/contrastive_prepare.py:137-165` (riga 155: `n >= 32 and fraction >= .8`), audit `scripts/contrastive_audit.py:104-108`. I
quattro config hanno `enforce_coverage: true`, `min_negatives: 32`, `max_age: 256`.

**Caso che lo attiva.** Il flusso vero della ricetta. `capgmyo` ha lo 0,236% delle finestre **[io]**: 14 sessioni per 80 prove da
1 s, cioè 1.120 identità in tutto. Nella stessa registrazione un negativo è ammesso solo nello stesso trial a 4 s di distanza, e
una prova dura 1 s. Restano le finestre di altre registrazioni viste negli ultimi 256 update: troppo poche.

| Misura sul flusso vero di 19.618 update | Valore | Prova |
| --- | ---: | --- |
| Query ammissibili, tutti i dataset | 618.119 su 619.584 (99,76%) | [io] |
| Query ammissibili di `capgmyo` | 0 su 1.461 | [io] |
| Negativi per query di `capgmyo`: mediana, massimo | 3, 8 (ne servono 32) | [io] |
| Blocchi da 1.024 update superati | 0 su 18, sempre e solo per `capgmyo` | [io] |
| Finestre di `capgmyo` nel pannello di calibrazione (batch 256-511) | 13 (ne servono 32) | [io] |
| Finestre di `capgmyo` nel tratto dell'audit (batch 256-1.023) | 45 | [io] |
| Secondo dataset più piccolo, `ninapro_db3`: pannello, audit, primo blocco | 58, 237, 344 | [io] |
| `ninapro_db3`: query ammissibili | 7.741 su 7.745 (99,95%) | [io] |
| Gli altri 15 dataset | 100% | [io] |

Gli stessi numeri escono per altre tre vie indipendenti: un secondo replay scritto dalle sole regole del protocollo, le classi
vere del codice consegnato (`WindowIdentity`, `CandidateQueue`, `admissible_mask`, `commit`) rigiocate sull'intero flusso, il
codice dell'audit sulle identità vere. Il ramo d'arresto è stato poi eseguito nel trainer vero su una fixture sintetica: prova il
meccanismo, non i numeri.

**Impatto.**
1. **La calibrazione fallisce.** Il lettore dà `coverage_gate_failed` e `FAILED_REQUIRES_REVIEW`. Fallisce per il solo conteggio
   (13 < 32), anche se tutte le 13 query fossero ammissibili. `validate_recipe` rifiuta poi benchmark e training: il collaudo si
   ferma dopo il primo job GPU, senza i 64 update a peso attivo.
2. **Con i pesi riempiti, i tre training si fermerebbero all'update 1.280.** Il cancello dipende solo dai dati: scatta allo stesso
   update nei tre bracci, **C0 compreso**.
3. **L'audit CPU non anticipa il problema del conteggio.** Oggi fallisce per l'ammissibilità (0 su 45). Ma il suo CLI accetta solo
   1.024 passi e giudica 768 update, dove `capgmyo` ha 45 query: sistemati i soli negativi, l'audit passerebbe e la calibrazione
   cadrebbe lo stesso, a job GPU già speso. La stessa soglia assoluta, 32, è applicata a tre lunghezze diverse: 256, 768 e 1.024.

**Non è un errore del codice.** Le soglie del cancello sono quelle del protocollo; l'arresto a run avviato è quello di `plan.json`
(`runtime_gate`) e della proposta del 07/11 (riga 82). È la soglia per dataset a non essere superabile col mix reale, e nessuna
regola dei negativi la salva: nella finestra d'età una query di `capgmyo` ha al più 8 altre identità `capgmyo`. GPT dichiara il
limite a monte (`verification.json`: «No real training stream audit or calibration values») e la proposta del 07/11 (riga 78)
aveva già previsto il caso: «Il dataset poco osservato o con memoria troppo vecchia è un problema da discutere, non viene escluso
automaticamente. [...] non si modifica il mix dei dati per far passare il gate.» **[io, letto]**

**Opzioni, con i numeri.**

| Opzione | Che cosa cambia | Esito sul flusso vero | Costo |
| --- | --- | --- | --- |
| A. Esenzione con regola scritta prima | Un dataset è esente dai due criteri per dataset se `quota × 32 × max_age < min_negatives + 1`. Dal solo manifest (quota di `capgmyo` 0,2854%; nel flusso 0,236%): 23,4 finestre attese contro 33; il successivo, `ninapro_db3`, 103,3 | Pannello di calibrazione superato (minimo 58 query, 100%); audit superato (minimo 237 query, 98,31%); 18 blocchi su 18 (minimo 344 query, 98,84%) | Una modifica; la ricetta InfoNCE resta com'è. `capgmyo` resta senza termine InfoNCE: va dichiarato |
| B. Memoria più lunga e soglia più bassa | `max_age` 2.048 e `min_negatives` 8 | A regime 17 blocchi su 17; il primo blocco fallisce per il transitorio di riempimento (prima query ammissibile al passo 1.333) | Tre cambi (età, soglia, inizio del cancello) più la regola delle 32 query sul pannello. Le chiavi dei dataset piccoli invecchiano: 84,7-87,3% del teacher cambiato contro 29,9-37,1% |
| C. Tenere `min_negatives` 32 | Qualunque età fino a 4.096 | Non basta: a 4.096 `capgmyo` arriva al 77,0% e passano 12 blocchi su 18 | Non risolve |

La regola a priori dell'opzione A, il transitorio dell'opzione B, il vincolo (b) qui sotto e il punto 3 dell'impatto vengono dai
soli verificatori [1A]. Altre opzioni emerse, senza numeri sul flusso: trattare sul pannello da 256 un dataset sotto 32 query come
«non valutabile», come nel testo del 07/11; leggere il conteggio su un orizzonte più lungo; una regola diversa per i dataset a
trial interi; non subordinare il benchmark tecnico al verdetto di copertura. Per chi ridisegna il conteggio: i pannelli con almeno
32 finestre di `capgmyo` sono circa il 13% a 256 update, il 70% a 512, il 98% a 768 e il 100% a 1.024, dove il minimo è 33 [1A].

**Vincoli per l'emendamento, eseguiti.** (a) Va scritto insieme in tre punti: `coverage_report`, `_coverage` del lettore e
costruttore delle ricette. (b) Il dataset esente deve restare nei conteggi complessivi: tolto dal blocco e dai dataset attesi, il
lettore solleva `coverage query count is not the complete global panel` (8.179 query invece di 8.192); tolto dal solo blocco,
solleva `calibration coverage support/steps differ from the frozen panel`. (c) Il lettore ricalcola il verdetto e solleva errore
se è diverso da quello scritto dal trainer: ogni cambio del cancello obbliga a rifare la calibrazione, quindi va deciso **prima**
del job. (d) Va allineato il testo: il protocollo (§4 riga 71) dice che durante il training i blocchi «si registrano», il codice
con `enforce_coverage: true` ferma il run. (e) Qualunque scelta deve restare uguale nei tre bracci: un cancello che in C0 solo
registra darebbe bracci a dose diversa.

**Lettura mia.** L'opzione A è la modifica più piccola e non tocca la ricetta; è però un'esclusione, e la proposta del 07/11
chiede che sia discussa, non automatica. Il limite della misura: QC, visibilità e validità vera della posa non sono nel log,
quindi la copertura reale può solo essere più bassa. Dopo l'emendamento conviene rigiocare le sole identità sull'intera dose prima
di ogni job: costa minuti di CPU sul Mac e nessun segnale.

### D-2. Regola di lettura: con le ampiezze vere l'esito è «irrisolto» quasi sempre. Blocca le soglie e il rilascio dei training, non il collaudo

**Dove.** Protocollo §6 righe 89-99; `scripts/contrastive_compare.py:34-35` (margini) e `204-217` (lettura); `plan.json`
`evaluation.margins`. La regola: un braccio è «promettente per conferma» se migliora utilmente almeno un endpoint e sostiene la
non inferiorità su tutti e tre, con intervalli simultanei al 99,4444%, sia contro C0 sia nel confronto diretto.

**Caso che lo attiva.** I bracci veri. Ho applicato lo schema di bootstrap del comparatore e le sue funzioni `intervals` e
`MARGINS`, con 100.000 repliche e seme 20261116, ai quattro bracci storici a 19.618 (T25-L, T100-L, T100-G, T100-LQ) **[io]**:

| Endpoint | Semiampiezza dell'IC simultaneo | Margine di non inferiorità | NI sostenuta | Guadagno utile |
| --- | ---: | ---: | ---: | ---: |
| P1 | 0,69-0,81 pp | 1 pp | 10 contrasti ordinati su 12 | 0 su 12 |
| P2 | 1,79-2,97 pp | 1 pp | 1 su 12 | 0 su 12 |

Su P2 il margine è da 0,34 a 0,56 volte la semiampiezza: per sostenere la non inferiorità serve una **stima positiva** fra +0,59 e
+2,01 pp, cioè la distanza fra stima e limite inferiore meno il margine **[io]**. Esempio: T100-G meno T100-L vale +1,796 pp su
P2, intervallo simultaneo [-1,131; +4,321], non inferiorità non sostenuta **[io]**. Con i cinque bracci storici (anche B19618) gli
agenti trovano: E_dyn 12 coppie ordinate, utile 0, NI 11; P1 20 coppie, utile 0, NI 15; P2 20 coppie, utile 0, NI 1; **24 terne
ordinate su 24 «unresolved_or_tradeoff»**.

**Perché.** UCI ha 19 soggetti e 364 finestre e porta il 95-98% della varianza del contrasto su P2. Due training della stessa
ricetta e dello stesso seme (T25-L e B19618) non sostengono la non inferiorità su P2 fra loro.

**Impatto, da simulazioni sui dati veri.** Tre agenti: doppio bootstrap, randomizzazione delle etichette dei bracci e, per gli
estremi superiori, il caso più favorevole con approssimazione gaussiana.

| Scenario | Probabilità |
| --- | ---: |
| Non inferiorità su P2 sostenuta, effetto vero nullo, IC simultaneo | 5,7-17,6% |
| Lo stesso con l'IC95 | 13-41% |
| «Promettente», nessun effetto vero | 0,1-0,3% |
| «Promettente», CN con +1,5 pp sul solo EPN612 | 0,5-2,9% |
| «Promettente», CR con E_dyn migliore del 6,3% e classificazione invariata | 0,9-4,6% |
| «Promettente», CN con P2 +3 pp e P1 +0,5 pp | 51-97% |

Il 6,3% è quanto vale passare da 19.618 a 49.046 update nel T100-G. Un braccio che migliora solo la posa, cioè l'ipotesi di CR,
risulta «promettente» nello 0,9-4,6% dei casi; uno che migliora EPN612 di 1,5 pp nello 0,5-2,9%. L'esito atteso cade nel ramo del
protocollo (riga 101) «tradeoff, risultato irrisolto, beneficio comune senza differenza diretta, o nessun vantaggio», i cui
quattro profili non hanno definizione operativa (D-10).

**Due cose non dichiarate.** (a) La proposta del 07/11 (righe 101-103) chiedeva la non inferiorità con l'IC95; il protocollo del
16/11 (riga 99) la porta al 99,4444% senza dire il cambio **[io]**. (b) La copertura congiunta dei nove intervalli «simultanei» è
87,0-87,6%, non 95% **[1A]**: «nominale» basta per la regola, non se un intervallo viene citato da solo come protezione di
famiglia.

**Opzioni, con le probabilità di «promettente» per CR col solo E_dyn migliore del 6,3%** (numeri dell'agente d'analisi; col metodo
del verificatore gli scenari CR danno da 5 a 20 punti in meno):

| Opzione | CR con solo E_dyn | Altro |
| --- | ---: | --- |
| (a) Tenere la regola e dichiarare gli effetti minimi rilevabili | 1,4-1,9% | Esito atteso «irrisolto» salvo P2 ≥ +3 pp con P1 ≥ +0,5 pp |
| (b) Non inferiorità con IC95, guadagno con IC simultaneo | 9,7-13,9% | Vicina alla regola del 07/11: da sola non basta |
| (c) Margine di P2 a 2 pp, oppure a 3 pp | 16,2-31,7%, oppure 44,2-63,3% | Uno scarto tipico fra due bracci storici su P2 vale 0,91 pp [1A] |
| (d) P2 = solo EPN612, UCI descrittivo | 41,9% | CN con EPN612 +1,5 pp: 70,0%; nessun effetto: 0,07% |
| (e) Margini P1 1,5 pp e P2 3 pp | 69,5-92,1% | |
| (f) Tutto con IC95, senza Bonferroni | n.d. | CN con P2 +1,8: 32-48%; falso positivo circa 2,4% |

**Lettura mia.** Cambiare solo il livello dell'intervallo (b, f) non basta o costa falsi positivi. Ridanno potere solo le opzioni
che agiscono su UCI: margine o composizione di P2 (c, d, e). Non scelgo. Qualunque scelta va scritta in `docs/decisioni.md` prima
del rilascio dei training, insieme alla definizione dei profili d'esito.

### D-3. Regola dei pesi: il «10%» è riferito al massimo transitorio della base. Da decidere prima della calibrazione

**Dove.** Protocollo §4 punti 2-4 (`λ_c = 0,10 × mediana_b(||g_base,b|| / ||g_c,b||)` sui primi 32 batch congiuntamente
informativi del pannello 256-511, a pesi iniziali, intervallo [1e-4; 1]) e riga 65 («La regola non garantisce gradienti al 10%
durante il run, né dopo AdamW/clipping»); `plan.json` `calibration.base_gradient_fraction`; proposta del 07/11, riga 70 («È una
scelta iniziale conservativa»).

**Caso che lo attiva.** Qualunque run. La norma del gradiente di base è al massimo proprio dove la regola la misura. Dal log vero
del controllo 30M con la stessa ricetta, norma totale prima del clipping **[io]**:

| Update | 1-8 | 57-64 | 129-256 | 257-512 | 513-1.024 | 1.963-5.000 | 5.001-10.000 | 10.001-19.618 | ultimi 100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Norma media | 29,79 | 9,32 | 5,91 | 6,74 | 6,70 | 3,86 | 3,43 | 3,03 | 2,65 |
| Rapporto ρ a peso pieno, se la norma del termine resta quella iniziale | 0,10 | 0,32 | 0,54 | 0,50 | 0,49 | 0,84 | 0,94 | 1,08 | 1,23 |

ρ è la media sugli update del tratto di `0,10 × G_0 / G_t`, con `G_0` = 29,79. Col rapporto delle medie i valori sono 0,10; 0,32;
0,50; 0,44; 0,44; 0,77; 0,87; 0,98; 1,12: per questo scrivo 0,44-0,49 e 1,1-1,2.

Il calo di 4,4 volte avviene **prima** che il peso si accenda (update 257). Durante la rampa (257-512) il rapporto realizzato vale
in media 0,24. A peso pieno, update 513-19.618: ρ medio 0,96, sopra 0,5 nel 93,6% degli update, sopra 1 nel 43,8% **[io]**. Il
clipping è attivo in 19.616 update su 19.618 **[io]**: il gradiente passato ad AdamW ha norma 1, quindi il termine non aggiunge
passo, ne prende una quota. Con gradienti ortogonali alla base resta `1/sqrt(1+ρ²)`: 0,995 all'avvio, 0,73 in media lungo il run
**[io]**. Una simulazione di AdamW con clipping sempre attivo conferma l'euristica: progresso della base 0,995 a ρ = 0,10, 0,924 a
0,44, 0,731 a 1,00 [1A].

**Limiti di questo conto.** (a) Ipotesi: la norma del termine contrastivo resta quella dell'inizializzazione. Nessun run
contrastivo esiste. Nei giocattoli di due agenti l'ipotesi sovrastima il braccio attivo: il rapporto realizzato vale 0,5-0,8 volte
la previsione per InfoNCE; per RnC 0,5-0,6 volte in due famiglie di segnali e 0,03-0,04 in una terza. Applicato alla base vera dà
circa 0,13-0,9 per InfoNCE, cioè sempre sopra il 10% nominale, e 0,01-0,6 per RnC. Sono giocattoli: verso e ordine di grandezza.
(b) Il log ha la norma su tutti i parametri, la calibrazione usa i soli condivisi: la quota dei condivisi è 0,82-0,84 all'avvio e
fra circa 0,6 e 1 dopo, quindi ρ a peso pieno sta fra 0,37 e 0,61 invece di 0,44-0,49. (c) Il contributo della base ha un solo
verso e si conosce prima del run: la base cala di 4,4-11 volte. Il rapporto realizzato dipende anche dalla norma del termine.

**Impatto.** Se la norma del termine non cala, CN e CR non sono «base più un termine piccolo» ma «base a passo ridotto più un
termine». La dichiarazione del protocollo non dice che il calo della base è prevedibile, ha un solo verso e vale da 4,4 a 11
volte. **Lettura mia:** «conservativa» vale solo se la norma del termine cala almeno quanto la base.

**Altri tre fatti che pesano sulla scelta.**
1. **Sul 30M vero all'inizializzazione, con segnali sintetici, λ cade a cavallo del limite 1.** Su 9 batch di quattro famiglie di
   segnali, misurati da due agenti: InfoNCE 0,62-4,44, sopra 1 in 6 batch; RnC 0,90-4,35, sopra 1 in 8. Nella famiglia accordata
   sulle perdite iniziali del run vero: InfoNCE 0,79 e 1,17, RnC 1,20 e 2,42. Quindi RnC ha probabilmente λ sopra 1 e InfoNCE può
   cadere dai due lati: un esito `FAILED` per peso fuori intervallo, anche a copertura risolta, è probabile ma non certo. Il
   protocollo dice solo «fuori range si rivede la proposta»: conviene scrivere ora che cosa succede nei tre casi (entrambi i pesi
   dentro, uno fuori, entrambi fuori). Col riferimento a fine rampa (opzione 1) i λ delle quattro famiglie diventano 0,14-1,00 e
   0,20-0,98: tutti dentro.
2. **La base misurata è fatta dalle sole ancore.** All'inizializzazione il gradiente della posa sui parametri condivisi è
   esattamente zero, perché lo strato d'uscita della testa nasce a zero. Il protocollo dice «soprattutto le ancore». Circa metà
   della norma al quadrato sta nel modulo `identity`, che ha l'1,9% dei parametri condivisi; nel backbone il rapporto resta vicino
   a quello globale, quindi questo non sposta le conclusioni.
3. **CN e CR hanno lo stesso rapporto solo all'avvio, e nessuno strumento legge il rapporto realizzato.** Nei giocattoli lo
   scostamento fra CN e CR va da 1 a circa 20 volte secondo la famiglia di segnali: non si prevede, si misura. Il training
   registra le tre norme globali in 43 update su 19.618, su un solo batch per punto (rumore di un fattore 1,5-2 [1A]), ma non c'è
   una regola di lettura scritta prima dei dati né uno script che legga i profili `contrast_global_grad_*` dei training: li legge
   solo il lettore della calibrazione (`scripts/contrastive_prepare.py:198`). Senza, un CN meno CR diverso da zero si potrà
   attribuire a posteriori sia all'approccio sia alla dose.

**Opzioni, a norma del termine costante.**

| Opzione | λ rispetto alla regola consegnata | ρ a peso pieno, poi a fine corsa |
| --- | ---: | --- |
| (0) Tenere la regola, scrivere il rapporto atteso e le soglie di lettura di ρ | × 1 | 0,44-0,49, poi 1,1-1,2; media 0,96 |
| (1) Riferire λ alla base a fine rampa del run storico | × 0,226 | 0,10-0,11, poi 0,25-0,28; media 0,22 |
| (2) Riferire λ alla media della base sul tratto attivo 513-19.618 | × 0,118 | 0,05-0,06, poi 0,13-0,15; media 0,11 |
| (3) Calibrare a pesi fermi sullo stato del C0 all'update 256 | misurato | Richiede codice: oggi il trainer rifiuta `init_from` col contrastivo |
| (4) In ogni caso: nel benchmark a 576 update registrare le tre norme a ogni update da 513 a 576 | | Oggi solo a 256, 500 e 512: ρ a peso pieno si leggerebbe su 64 batch prima dei training. Nei training, `grad_every` 100 darebbe circa 200 punti invece di 43, con un aggravio stimato fra 0,8% e 3% [1A] |

**Lettura mia.** L'opzione (4) costa poco e conviene comunque. Chiede una piccola modifica a codice che la release fissa: i passi
della diagnostica sono in `run.py:773-775` e `grad_every` è nel config congelato. Va quindi nel giro unico di correzioni, e il
tempo per update va misurato su update senza diagnostica. La scelta fra (0), (1), (2) e (3) è di GPT e Simone: la chiusura minima,
senza codice, è la (0) con rapporto atteso e soglie scritti prima; (1) e (2) sono una moltiplicazione di λ sugli stessi dati di
calibrazione. Va presa prima della calibrazione: una regola riscritta dopo aver visto le norme è post hoc, anche se sono dati di
solo training.

## 4. Altri punti del disegno da dichiarare prima dei dati

Nessuno blocca il collaudo. Vanno scritti nel protocollo prima del rilascio dei training.

| N. | Punto | Fatto | Che cosa scrivere o aggiungere |
| --- | --- | --- | --- |
| D-4 | Rumore fra training | Un solo seme. T25-L meno B19618, stessa ricetta e stesso seme: P1 +0,510 pp, IC95 [0,090; 0,926] nel rapporto pubblicato **[io]**; [0,074; 0,929] col bootstrap congelato del comparatore. Scarto tipico fra due bracci storici: P1 0,29 pp, P2 0,91 pp, E_dyn 0,47-0,90% [1A] | Dichiarare C0 meno T100-G storico a 19.618 e scrivere prima come si legge: limite inferiore del rumore a seme uguale; se il training fosse deterministico darebbe zero, cioè la sola equivalenza del percorso. Nella proposta del 07/11 (riga 20) il G storico era «un controllo descrittivo di compatibilità», non una stima della variabilità fra pretraining. C0 non riprodurrà i numeri del G storico bit per bit |
| D-5 | Controlli di manipolazione | Tutti i bracci calcolano entrambi i termini: sei letture si ottengono senza training in più. In un giocattolo la perdita RnC scende quasi uguale con pose scambiate (placebo); solo la correlazione di rango fra coseno e distanza di posa **fra finestre** separa il termine vero dal placebo [1A] | Fissare prima le letture; usare la correlazione di rango fra finestre come controllo specifico di CR; dichiarare la perdita RnC come non specifica |
| D-6 | CN meno CR confronta ricette | `plan.json` lo dichiara («comparison of objective recipes»); il protocollo §5 riga 85 dice «differenza diretta fra approcci». I due termini avranno lo stesso rapporto solo all'avvio (D-3) | Allineare §5 e l'etichetta del comparatore; fissare come si legge un CN meno CR diverso da zero quando ρ dei due bracci diverge |
| D-7 | Attribuzione di un guadagno di CR | Nel giocattolo, dove c'è un guadagno su E_dyn, il placebo ne riproduce dal 42% al 74% a peso fisso e dal 38% all'85% col peso calibrato come nel protocollo; nel regime più vicino a WearUsFM il guadagno del termine vero ha segno instabile. Meccanismo e segno, non grandezze [1A] | Scrivere che, senza braccio placebo, un guadagno di CR su E_dyn non si attribuisce all'ordinamento per posa. Il braccio placebo costa un training: decisione di Simone |
| D-8 | InfoNCE e identità del soggetto | Il 96,3% dei negativi viene da altri soggetti; col solo codice del soggetto si può togliere il 55,4% della perdita iniziale (limite idealizzato). I soggetti di P1 sono nello split di pretraining. Limite dichiarato da GPT | Pubblicare nel collaudo la quota di negativi dello stesso soggetto e la loro età (I-8) |
| D-9 | Soglia del 2% su E_dyn | Vale dal 20,5% al 58% di passi in più, secondo la pendenza. Scelta dichiarata | Nessuna correzione. Legare la conservazione dei checkpoint a 19.618 alla scadenza di SCRATCH a 40 giorni |
| D-10 | Profili d'esito | I quattro profili del protocollo (riga 101: «tradeoff, risultato irrisolto, beneficio comune senza differenza diretta, o nessun vantaggio») non hanno definizione operativa; il comparatore emette una sola etichetta. Non è definito nemmeno il danno dimostrato: una non inferiorità mancata per intervallo largo e un peggioramento vero hanno la stessa etichetta. «Conferma» non è definita | Definire dalle bandiere dei nove contrasti i profili e il danno dimostrato, con un margine di equivalenza per «nessun vantaggio», insieme a D-2 |
| D-11 | Regola dei trial e formato su disco | Le prove da 1 s di `hyser` sono concatenate (trial `None`) e ammesse fra loro a 4 s d'indice; quelle di `capgmyo` sono array 3D e mai ammesse. Peso pratico piccolo: negativi della stessa registrazione fra 0% e 13,7% per dataset [1A] | Scrivere quale regola vale per le prove concatenate |
| D-12 | RnC: duplicati e finestre uniche | Con finestre da 4 s c'è almeno un istante duplicato nel 33,8% dei passi; il 6,1% delle ancore ha un gemello; una finestra duplicata pesa 1,5-1,75 volte nella perdita: limite dichiarato da GPT. Una sola finestra di posa nello 0,72% dei passi: il termine resta attivo con soli confronti interni, e questo non è dichiarato in modo esplicito | Scrivere il caso della finestra unica e pubblicare nel collaudo la quota di passi con una sola registrazione (`contrast_rnc_recordings` è già nel log). Leggere `contrast_global_grad_rnc` contro `contrast_rnc_windows` |
| D-13 | Proiettore | In C0 e CR il proiettore non riceve gradiente e il weight decay lo riduce a 0,911 a 19.618: `contrast_nce` di C0 e CR viene da un proiettore non allenato e non è confrontabile con quella di CN. In CN il gradiente del proiettore entra nella norma del clipping comune ed è fuori dalla calibrazione. In CN la rampa e λ non modulano il suo passo sotto AdamW: passi dell'ordine del learning rate dal primo update a peso non nullo. Le norme InfoNCE dei tre bracci non sono quindi appaiate; quelle RnC sì | Dichiararlo accanto alla rampa. Aggiungere al log, prima del collaudo, la norma del gradiente del proiettore separata da quella dell'encoder |
| D-14 | Negativi per dataset | Il numero di negativi mediani va da 101 (`ninapro_db3`) a 1.030 (`emg2pose`, `emg2qwerty`) **[io]**. A logit uniformi la perdita per query va da 4,61 a 6,94 e il peso nel numeratore, rispetto alla quota di query, da 0,729 (`ninapro_db3`, che pesa 3/13 in P1) a 1,099; le chiavi dei due dataset a capienza sono più fresche (età media 67-68 update contro 124-128). Dichiarato solo in parte nella proposta del 07/11 | Dichiararlo e registrare la perdita InfoNCE per dataset, oppure fissare una regola che renda confrontabile il numero di negativi: decisione di GPT e Simone. Il 6,31 all'avvio è un riferimento a logit uniformi, non una misura sul 30M |

## 5. Implementazione: difetti dimostrati, preferenze e precisazioni

Nel calcolo sono tutti minori: nessuno cambia in modo apprezzabile i numeri di un run. Gli scarti noti sono il confine dei 4 s
(I-10a) e fino allo 0,2% su λ (I-10b). I-1, I-3, I-4, I-5, I-6, I-7a e I-8 toccano sorgenti o config che la release fissa per
hash. Lo stesso vale, se GPT le adotta, per I-10 (a, b, c, d, f, g), per la riga di log di D-13 e per l'opzione (4) di D-3.
Conviene chiudere tutto in un solo giro **prima** della calibrazione. I dubbi scientifici stanno nelle sezioni 3 e 4; qui marco
«preferenza» ciò che non è un difetto dimostrato.

### I-1. Le guardie delle campagne chiuse scattano su ogni run senza contrastive. Da chiudere prima del collaudo

Per i due agenti della parità è un blocco di processo, non numerico. Scende a minore se GPT sceglie una via che non tocca `src` né
i config.

- **Dove.** `src/wearusfm/training/run.py:168, 175, 196, 287-288` (`config_to_dict`); `src/wearusfm/model/fm.py:75-79, 203-206`.
  Guardie colpite: `scripts/tokenizer_preflight.py:93-97`, `scripts/logsquare_preflight.py:45-46`,
  `scripts/logsquare_initialization_diagnostic.py:105-111`, `scripts/logsquare_evaluation.py:41-47, 69-75`,
  `tests/cpu/test_logsquare_train.py:36`.
- **Caso.** Qualunque run **senza** contrastive. Le cause sono due e vanno tenute distinte. (1) I byte di quattro sorgenti fissati
  per hash sono cambiati e `src/wearusfm` ha quattro file nuovi: la guardia fa il suo lavoro. (2) Il config di ogni run ha due
  chiavi nuove, `contrastive: None` e `model.contrastive_projection: False`, e `FMConfig` passa da 25 a 26 campi: è un falso
  allarme sulla ricetta.
- **Impatto.** La suite completa ha 12 test falliti alla base e 23 nella consegna; gli 11 in più sono il test [CR-4] (I-2) e **10
  test log-quadratici** che passano a `924e19d` e falliscono a `db5ac41` **[io]**. I 12 comuni falliscono solo perché le copie
  estratte con `git archive` non hanno `.git`. Sui piani veri registrati le guardie passano alla base e falliscono nella consegna;
  i tre `load_plan` girano sul Mac solo sostituendo la lettura di quattro file che stanno su Leonardo. `canonical_config` rifiuta
  15 config registrati, fra cui il config storico di riferimento della nuova campagna [1A]. La ricetta **non** cambia: 9.471
  tensori, 0 diversi. I checkout congelati non sono toccati. Non è dichiarato: `verification.json` elenca 11 file di test e nessun
  limite sulla suite intera.
- **Correzione minima, provata** (patch nella cartella di prove, `patch/optR/`). Il proiettore esce da `FMConfig` e lo aggiunge il
  trainer quando `cfg.contrastive` c'è; `config_to_dict` omette `contrastive` quando è `None`. Esito, insieme agli adattamenti dei
  test che l'integratore ha già su `main` (`3263ee9`, che non è antenato di `db5ac41` **[io]**): 10 test su 10, guardie di config
  di nuovo verdi, `fm.py` identico alla base byte per byte. Run storico, C0, CN e CR identici bit per bit alla consegna [1A]. Le
  guardie di sorgente restano rosse sulla punta del ramo, come devono. Non provata con DDP né CUDA. Da mettere in conto: due test
  della consegna da adattare (`tests/cpu/test_contrastive_prepare.py:70` e il caso `explicit matched` di
  `test_contrastive_runtime.py`), i quattro config di review da rigenerare coi loro pin, e i valutatori futuri di C0/CN/CR che
  devono togliere le chiavi `contrastive_head.*` prima del caricamento stretto.

### I-2. Il test distribuito [CR-4] fallisce per la sua tolleranza, non per il codice

- **Dove.** `tests/cpu/test_contrastive_runtime.py:226-227` (confronto di `total` a 1e-9). Causa:
  `src/wearusfm/training/jepa.py:196` (`pose_terms`, identica a `924e19d`).
- **Caso.** Le due finestre di posa cadono su processi diversi. `total` vale 4237.721409999787 a un processo e 4237.721287929475 a
  4 processi: scarto relativo 2,88e-8 **[io]**. La metrica `pose` è sommata in float32 dentro ogni processo e in float64 fra i
  processi. Lo stesso scarto esiste nella base senza alcun codice contrastivo, e compare anche a 2 processi quando le pose stanno
  su processi diversi: [CN-2] e [CR-2] passano solo per la fixture.
- **Impatto.** Nessuno sul training. I gradienti coincidono: entro la tolleranza del test nei miei lanci **[io]**, al più 9,2e-14
  per parametro in quelli degli agenti. `contrast_rnc` e `contrast_nce` sono identici **[io]**. `verification.json` marca però
  questi test `required_before_release` **[io, letto]**.
- **Correzione minima, provata** (`patch/correzione_minima.diff`: una riga modificata e cinque aggiunte nel test): `pose` con
  tolleranza float32, `total` meno il termine di posa a 1e-9. In alternativa due righe in `pose_terms`: tocca un file della base,
  scelta di GPT.

### I-3. Rappresentazioni o target non finiti: eccezione al posto dell'arresto coordinato

- **Dove.** `src/wearusfm/training/contrastive_objectives.py:164-176` (studente e segnale) e `202-204` con
  `src/wearusfm/model/rank_contrast.py:85-86` (target di posa), dentro il forward prima del controllo coordinato di
  `run.py:757-764`.
- **Caso.** Un NaN nell'uscita dello studente, nel segnale, o in un target di posa valido su una patch-ancora (una ogni 8), su un
  solo processo. Un NaN in un target valido fuori dalle ancore dà ancora l'arresto coordinato.
- **Impatto.** Tutti i processi sollevano `ValueError` nello stesso update, **nei tre bracci, C0 compreso**: niente file `STOP`,
  niente `summary.json`, log non chiusi. Il percorso storico, con la stessa iniezione, scrive `STOP` e `summary.json`. La catena
  **non** riparte da sola: il job esce con errore e `afterok` ferma i successivi. Resta possibile una risottomissione a mano, che
  riprende dal checkpoint e lascia un record doppio in `metrics.jsonl` [1A].
- **Correzione minima, provata da un agente per il NaN nello studente** (`patch/F1_nan_coordinated_stop/`, 10 righe in `attach`)
  [1A]: dopo la raccolta degli errori, numeratori non finiti a peso zero e ritorno, così scatta il controllo che esiste già. Per
  il target di posa su un'ancora la patch non basta: va trattato anche l'errore di `pose_rank_contrast` (riga 202) o intercettato
  in `_accumulate_terms`. Portare il testo dell'errore contrastivo nel motivo dello `STOP`. Il test deve coprire anche C0.

### I-4. Arresto del cancello di copertura: update applicato e contato, non registrato né salvato; ramo mai eseguito dai test

- **Dove.** `src/wearusfm/training/run.py:1669-1683` e `1793`.
- **Caso.** Cancello fallito alla chiusura di un blocco. `commit()` gira dopo `opt.step()` e dopo l'EMA; poi `step += 1` e uscita
  prima del record **[io, letto]**.
- **Impatto.** `summary.json` conta un update che non è in `metrics.jsonl` né in un checkpoint. Nella campagna, con `ckpt_every`
  500: `steps` 1.280, 1.279 righe di metriche, ultimo stato su disco al passo 1.000. Il ramo di chiusura del blocco non è eseguito
  da nessun test consegnato, né dall'audit (768 update), né dalla calibrazione (256), né dal benchmark a 576: **su GPU la sua
  prima esecuzione sarebbe l'update 1.280 dei training veri**. In questa review è stato eseguito su CPU, a un processo e a 4
  processi `gloo`: si comporta come progettato. Due mutanti (cancello invertito, soglia delle 32 query tolta) superano i 33 test
  non distribuiti dei tre file che importano il modulo; nessun file di test nomina `coverage_block`.
- **Correzione minima.** Decidere il cancello prima di `opt.step()`, come gli altri arresti: il rapporto del blocco dipende solo
  dai dati. Aggiungere il test: una base pronta è `script_agenti/w_trainer_stop.py`, che ho eseguito sul trainer vero con una
  fixture sintetica: stop al quarto update, `steps` 4, tre record di metriche, `STOP` scritto **[io]**. Nel collaudo, un run
  tecnico con `coverage_block` piccolo per eseguire almeno una volta chiusura e stop sotto DDP.

### I-5. `scripts/train_jepa.py --contrastive-config` è un secondo ingresso senza cancelli

- **Dove.** `scripts/train_jepa.py:263-273`; `src/wearusfm/training/contrastive_objectives.py:25-26` (default 0,1 dei due pesi).
- **Caso.** Il flag passa il JSON direttamente alla dataclass **[io, letto]**. Accettati: `{}` (braccio C0, pesi 0,1 e 0,1);
  `{"arm": "CR"}`; pesi 7,5 oppure 1e-9; `enforce_coverage: false`; `coverage_block` 10^9; chiavi doppie (vince l'ultima).
  `contrastive_config` non è fra le destinazioni vietate di `wsd_chain.py`.
- **Impatto.** Col blocco giusto il config è identico a quello di campagna, ma senza ricetta congelata, ricevuta, release né pin.
  Su questo percorso non vale la frase del registro di GPT «nessuna sostituzione con il default tecnico». La campagna non passa da
  qui: il rischio riguarda l'uso diretto del trainer.
- **Correzione minima.** Togliere il default ai due pesi (campi obbligatori), controllare l'intervallo [1e-4; 1] dentro
  `ContrastiveConfig`, rifiutare le chiavi doppie, vietare il flag in `wsd_chain.py` oppure legarlo a una ricevuta.

### I-6. La ricevuta di calibrazione è creduta sulla parola

- **Dove.** `scripts/contrastive_run.py:37-65` (`validate_recipe` legge solo `status` e i due pesi **[io, letto]**).
- **Caso.** Accettati: una ricevuta coi soli pesi ritoccati (0,9 e 0,05); una con `failure_reasons` non vuoto ma stato
  `PASSED_FOR_REVIEW`; tre bracci con tre ricevute diverse, che danno tre release valide.
- **Impatto.** Nessuno strumento confronta l'hash iniziale della calibrazione con quello dei tre bracci, né il codice della
  calibrazione con quello dei training, né impone la stessa ricevuta ai tre bracci. Non c'è un generatore dei tre config
  calibrati: i due λ si scrivono a mano. Serve un errore dell'operatore su tre release rilette: per questo resta minore.
- **Correzione minima.** Un comando che generi i tre config dalla ricevuta; nel validatore di benchmark e training, uguaglianza
  dell'hash della ricevuta fra i tre bracci, degli hash iniziali e del digest dei sorgenti con la calibrazione.

### I-7. Strumenti del collaudo incompleti

| Punto | Dove | Fatto | Correzione minima |
| --- | --- | --- | --- |
| a | `scripts/contrastive_run.py` (`--validate-only`) | Una release dal JSON troncato e ingressi inesistenti risponde `recipe_valid true`: la release non viene letta | Leggere e validare la release anche nella prova a secco |
| b | `scripts/arms_collaudo.py:414-415` (rifiuto) e `492-511` (righe nuove) | Le 11 righe aggiunte sono corrette, ma lo strumento rifiuta i bracci C0, CN e CR per nome | Registrare i tre nomi |
| c | Protocollo §7 | Nessun criterio di superamento del collaudo: non dichiarato. Piano, comandi e tetto: GPT dichiara già che vanno pubblicati prima della sottomissione (riga 116) | Scrivere i criteri prima del primo job (sezione 7) |
| d | `scripts/contrastive_run.py:69-71, 118-119` | La ripresa non è provabile dall'ingresso di campagna: calibrazione e benchmark pretendono una cartella nuova (118-119) e il training è rifiutato (69-71). Il protocollo la chiede in §3 e §7 | Dire come si prova la ripresa nel collaudo |
| e | `scripts/contrastive_audit.py:84-93` | L'audit apre 24 loader in un processo, ciascuno con 64 sessioni in memmap: oltre 1.024 file aperti. Su un albero sintetico di 100 sessioni, col limite a 1.024, cade con `Too many open files` al passo 279; sul manifest vero la soglia è stimata, non eseguita, verso il passo 160 [1A] | Leggere `ulimit -n` dentro un job di `lrd_all_serial`. Sul nodo di login vale 65.536, ma `PropagateResourceLimits = CORE`: il limite del job è quello del nodo di calcolo, non letto **[io]** |
| f | `scripts/contrastive_audit.py:107, 120-129` | Preferenza [1A]. Il rapporto dell'audit non porta l'identità del codice, nessuno strumento lo legge, e il CLI non dà il verdetto del pannello 256-511 che decide la calibrazione | Scrivere anche quel verdetto e lo SHA del codice; fissare l'hash del rapporto nella release di calibrazione |

### I-8. Diagnostiche promesse dal protocollo e non scritte

- **Dove.** Protocollo §4 riga 69 («negativi ammissibili e loro età», «duplicati e overlap»);
  `src/wearusfm/training/contrastive_objectives.py:202, 210-226`; `scripts/contrastive_audit.py:61-65`.
- **Fatto.** Il trainer scrive il totale dei negativi ammissibili, non la loro età, né la quota dello stesso soggetto, né i
  duplicati scartati. Per RnC il conteggio delle coppie è calcolato e scartato; mancano molteplicità massima e overlap [1A].
  L'audit non scrive né numero né età dei negativi. `rnc_anchors` conta tutti i punti-ancora nell'audit e solo le ancore con
  almeno due candidati nel trainer [1A]. L'informazione c'è già nello snapshot (`steps`, `from_current`) ma nessuno la legge.
- **Impatto.** Il collaudo non può rispettare il proprio elenco, e per `capgmyo` sono proprio i campi che servirebbero alla
  diagnosi.
- **Correzione minima.** In `attach` e in `audit_batches`, per dataset: età mediana e massima dei negativi, quanti dal batch
  corrente, quota dello stesso soggetto e della stessa registrazione, identità duplicate scartate; per RnC coppie e molteplicità
  massima. Stesso nome per la stessa grandezza fra audit e trainer.

### I-9. Comparatore: quattro difetti minori

| Punto | Dove | Caso | Correzione minima |
| --- | --- | --- | --- |
| a | `scripts/contrastive_compare.py:210-215` | Se la regola vale in entrambi i versi del confronto diretto, la catena `if/elif` restituisce CN. Contraddice «nessun vincitore forzato»; non raggiungibile con le ampiezze vere | Etichetta senza vincitore; test pronto, oggi marcato come difetto atteso |
| b | `scripts/contrastive_compare.py:43-48, 57-58` | `pred` e `y` di tipo diverso: accuratezza zero in silenzio, con decisione valida | Pretendere lo stesso tipo; un test |
| c | `tests/cpu/test_contrastive_compare.py` | La scala relativa delle repliche di E_dyn non è protetta da alcun test: un mutante che usa la differenza assoluta sopravvive [1A] | I due test in `test_proposti/prop_compare_vc.py` |
| d | `scripts/contrastive_compare.py:67-71, 183-184` | Un limite entro 0,01 pp dal margine dipende dal seme del bootstrap: errore Monte Carlo 0,004-0,006 pp su P1 e 0,02 pp su P2 [1A] | Pubblicare l'errore Monte Carlo accanto a ogni limite; scrivere prima che un limite entro 3 errori si riporta «al bordo» |

Limite già dichiarato da GPT, da chiudere prima della valutazione: il comparatore accetta ingressi troncati o scambiati purché
coerenti. Servono cardinalità attese e pin nel manifest (soggetti per dataset 314, 32, 8, 8, 19; 15 utenti di posa).

### I-10. Dettagli numerici e di igiene

| Punto | Dove | Fatto | Correzione minima |
| --- | --- | --- | --- |
| a | `src/wearusfm/training/contrastive_memory.py:75-76, 91-93` | Il docstring dice che un gap esattamente uguale a 4 s è ammesso; il confronto in secondi float64 lo esclude in circa l'1% delle coppie al confine (148 su 18.000 sulla griglia vera) [1A]. Effetto conservativo | Confrontare in campioni interi quando le frequenze coincidono, oppure correggere docstring e test |
| b | `src/wearusfm/training/contrastive_objectives.py:279-287` | Preferenza. Norme e coseni della diagnostica accumulati in float32: su CPU, a 2,9e7 elementi, sottostima dello 0,44%. Nel rapporto che dà λ si cancella quasi tutta | `torch.linalg.vector_norm(v, dtype=torch.float64)`, oppure conferma nel collaudo |
| c | `src/wearusfm/training/contrastive_objectives.py:192`; `scripts/contrastive_audit.py:41, 51` | Preferenza. Il numero di giunti è il letterale 20 | Usare `N_JOINTS` |
| d | `scripts/contrastive_prepare.py:186-243` | Lassità del lettore: chiavi doppie in una riga di metriche, passi in virgola mobile, λ non finito senza rapporto. Il lettore non incrocia `metrics.jsonl` col blocco di copertura del summary [1A] | Stesso controllo delle altre letture; somma di `contrast_nce_eligible` sugli update 257-512 uguale agli ammissibili del blocco. Nessuna urgenza |
| e | `results/infra/contrastive_comparison_20261116/configs/preparation.json` | Percorso assoluto della macchina di GPT. Il file è solo documentale: nessuno script lo legge | Percorsi relativi |
| f | `src/wearusfm/training/contrastive_memory.py:258-267, 279-294` | Preferenza. La raccolta con gradiente non ha guardia contro esecuzioni asimmetriche fra processi. Nessun caso raggiungibile trovato | Facoltativo: `requires_grad` nei metadati della raccolta |
| g | `src/wearusfm/training/run.py:775`; `contrastive_objectives.py:261` | Preferenza. Passi della diagnostica e soglie di copertura sono nel codice, non nel config: la loro identità è solo l'hash dei sorgenti | Spostarli nel config o dichiararlo |

### I-11. Parità con lo storico: che cosa è esatto e che cosa no

Non è un difetto del codice, è una precisazione su ciò che i test provano. A un processo il percorso storico e quello del
contratto differiscono di un arrotondamento per certe taglie del batch: in float32 `0,2/n` e `0,2 × (1/n)` non coincidono per n =
5, 7, 10, 11, 13... e coincidono per 4, 8, 32 **[io]**. Con 5 e 7 finestre i gradienti differiscono dal primo update. **Il
commento di GPT sull'arrotondamento è corretto** e la tolleranza del test non è larga. Sulla campagna non c'è effetto: su Leonardo
storico e bracci nuovi usano entrambi il contratto, con n = 32. Resta un fatto: il test di parità di C0
(`tests/cpu/test_contrastive_runtime.py:93-104`) non distingue un peso zero da un residuo di 1e-6; rileva 1e-5. **Correzione
facoltativa:** un confronto esatto sotto lo stesso percorso (`_accumulate_terms` con e senza C0, oppure DDP a 2 processi), con un
controllo negativo a 1e-6; la tolleranza resta nel confronto con lo storico a un processo.

## 6. Forza dei test e test proposti

Il codice consegnato è corretto nei punti elencati qui: quello che manca è la protezione dalle regressioni. La misura è la quota
di mutanti (piccole alterazioni del codice) che i test consegnati rilevano. Due agenti per area, in gran parte con mutanti
diversi. I denominatori comprendono i mutanti equivalenti, salvo dove indicato.

| Area e test consegnati | Mutanti rilevati, primo agente | Secondo agente | Esempi di mutanti non rilevati |
| --- | ---: | ---: | --- |
| RnC, modulo (23 test) | 54 su 82 | 188 su 287 | Distanze oltre 256 ancore; ancore non contate; una sola finestra contata |
| RnC, punto di chiamata | 4 su 12 | 1 su 16 non equivalenti a un processo | Esclusione dello stesso istante calcolata e non applicata; temperatura; pesi dei due termini scambiati fra CN e CR |
| InfoNCE e memoria | 57 su 84 | 20 su 40 nuovi (21 su 56 con 16 repliche dei sopravvissuti del primo) | Margine solo in avanti; sessione costante nell'identità; scadenza per età al commit |
| Innesto nel trainer (7 test non distribuiti) | 12 su 46 | 5 su 31 | Query o chiavi non normalizzate; rampa anticipata di un update; `min_negatives` ignorato; peso RnC doppio rispetto al registrato |
| Distribuito (6 test) | 4 su 4 mutanti dei fattori; 0 su 3 sugli altri | 2 su 5 | Larghezza locale al posto di quella globale; zeri in testa nel riempimento; copertura dal solo processo 0 |
| Comparatore (25 test) | 44 su 132 | 25 su 78 | Pesi di P1 e P2; soglie; quantili; scala relativa di E_dyn |
| Calibrazione, lettore (23 test) | 10 su 20 | 6 su 14 | Mediana sostituita da media o massimo; limiti dell'intervallo; soglie del cancello |
| Rilascio (`contrastive_run`, `prepare`, `audit`) | 39 su 109 mirati | 140 su 247 automatici | Soglie 32, 80% e 90% del lettore; `main()`, `replay()` e `source_digest` mai eseguiti |

Tre osservazioni. (1) Nessun test di GPT fissa il **valore** di `contrast_rnc` nel punto di chiamata, né quale termine riceve il
peso attivo in ciascun braccio: lo scambio dei pesi fra CN e CR supera i 10 test non distribuiti del punto di chiamata e il test
distribuito a 2 processi [1A]. Al collaudo lo vedrebbe il controllo della composizione del totale, ma solo dopo la correzione
I-7b; quel controllo non vedrebbe un termine raddoppiato o dimezzato, né la temperatura o l'esclusione. (2) I sei test distribuiti
usano finestre tutte da 1 s; la ricetta usa da 1 a 4 s. Il protocollo (§3) chiede il test distribuito «comprese rank vuote e
ripresa»: la ripresa distribuita non è testata. (3) Le soglie congelate del cancello (32, 80%, 90%) non sono protette, e sono
proprio quelle che cadono sui dati veri.

**Avviso di metodo.** I processi figli di `mp.spawn` importano `wearusfm` dall'albero dei test, non da `PYTHONPATH`
(`scripts/build_manifest.py:28` antepone `src`). Una prova per mutazione dei test distribuiti con la copia mutata solo anteposta
nel path non vale. I numeri sul distribuito qui sopra vengono solo da lanci con un gancio caricato in ogni processo.

**Test proposti.** Nella cartella di prove, `test_proposti/`, ci sono 13 file: 203 test, eseguiti da me sul codice consegnato (200
passati, 1 saltato, 2 marcati come difetto atteso, quelli di I-9a) **[io]**. Sono un suggerimento da adattare, non una modifica
della ricetta. Col nome `prop_*.py` non entrano nella raccolta automatica; si lanciano su un albero a `db5ac41` (istruzioni nel
`LEGGIMI.md` della cartella).

## 7. Prerequisiti del collaudo, in ordine

**Prima della calibrazione, che è il primo job del collaudo.**

1. **Decisione D-1** scritta in `docs/decisioni.md`, emendamento del cancello nei tre punti del codice, protocollo e `plan.json`
   allineati. Poi rigiocare le sole identità sull'intera dose (19.618; 49.046 se si continuerà).
2. **Decisione D-3**: riferimento della regola dei pesi, rapporto atteso lungo il run, soglie di lettura di ρ, e che cosa si fa se
   λ esce da [1e-4; 1]. Il protocollo oggi dice solo «si rivede la proposta».
3. **Un solo giro di correzioni sul codice che la release fissa**: I-1, I-3, I-4, I-5, I-6, I-7a, I-8, più I-7b e i test (I-2 e
   sezione 6), e ciò che GPT adotta di I-10, D-13 e dell'opzione (4) di D-3. Poi rilettura del diff da parte mia: è la sequenza
   del protocollo (§7, «correzioni/convergenza»).
4. **Soglie in `docs/decisioni.md` prima del primo job.** La voce del 16/11 (righe 6443-6461 a `77b7003`) non ne contiene **[io,
   letto]**. La calibrazione usa già 0,10, [1e-4; 1], il riempimento di 256 batch, i primi 32 batch informativi su 256 e i
   cancelli 0,9 / 0,8 / 32 / 0,9; il benchmark usa in più la rampa 256 + 256. Per la regola del progetto vanno scritte e
   committate prima del job che le usa.
5. **Piano del collaudo pubblicato**: comandi, risorse, tetto, criteri di superamento, criterio di ripresa. Preventivo in ore
   locali e GPU-ore, con il residuo. Il piano dica anche quanti checkout di campagna servono (ricevute, config calibrati e release
   nascono dopo il primo, che è immutabile), gli invii con `scripts/submit_job.py` e ledger, e le uscite fuori dal repository.
   **Interpretazione mia:** calibrazione e benchmark dei tre bracci sono una prova sola ai fini della soglia di 5 GPU-ore.
6. **Budget: serve una decisione di Simone.** L'estensione del 03/11 dice: «L'estensione copre questo confronto, non nuove loss,
   altri esperimenti o ripartenze prive di diagnosi» (`docs/decisioni.md`, righe 5738-5746 **[io, letto]**). **Interpretazione
   mia:** «questo confronto» è quello dei tokenizer; per la contrastiva non c'è un budget approvato. Riferimenti: 50,83 GPU-ore =
   3 × 16,94, cioè il consumo storico di T100-G fino a 19.618 per tre bracci, prima del sovraccosto del percorso nuovo (protocollo
   §7), pari a 406,67 ore locali col fattore 8; tetti storici 22 GPU-ore per braccio, 66 per tre.
7. **Audit CPU**: leggere `ulimit -n` dentro un job seriale (I-7e); far scrivere all'audit il verdetto del pannello 256-511
   (I-7f); pin attesi dei dati dell'audit in `plan.json`; preventivo CPU dell'audit.

**Che cosa il collaudo CUDA deve misurare o provare**, perché da qui non è verificabile:

| Voce | Perché |
| --- | --- |
| Le 18 collettive nuove su NCCL | Su `gloo` funzionano. Mai eseguite su NCCL: `all_gather` di tensori riempiti a lunghezza variabile, `all_reduce` dentro il backward della raccolta, 10 `all_gather_object` per update |
| Equivalenza a 1 e 4 GPU sullo stesso batch globale | CN e CR a peso attivo, con un processo senza posa e lunghezze diverse: `contrast_rnc`, `contrast_nce`, gradiente e norme globali. `scripts/ddp_equivalence.py` non copre il percorso contrastivo |
| bf16 vero | Su CPU è solo emulato: tipi e forme reggono. Da misurare il rumore delle norme di calibrazione fra due run identici |
| Tempo per update | Costo CPU del ciclo Python della memoria: 26,9 ms per update per processo sul Mac **[io]**, cioè il 3,5% del passo storico di 0,777 s; altri agenti misurano dall'1,5% al 6,4%. Su Leonardo ogni processo divide 8 core con 6 worker: va misurato. Costo dei passi con diagnostica |
| Parità iniziale dei tre bracci | Nei primi 256 update, a peso zero, `total` e `grad_norm` devono coincidere a meno del non determinismo della GPU |
| Chiusura di un blocco di copertura e stop sotto DDP | Con `coverage_block` piccolo: è il ramo di I-4, mai eseguito dai test e dagli strumenti consegnati né su CUDA/NCCL |
| Rapporto ρ a peso pieno | Le tre norme a ogni update da 513 a 576 (D-3, opzione 4) |
| Ripresa a 4 GPU | Il criterio va scritto prima. Su `gloo` a 4 processi un run ripreso non coincide bit per bit con quello intero, già nella base: DDP ricostruisce i bucket dopo la prima iterazione. Su NCCL non è noto. Tagliare le catene dei tre bracci agli stessi passi |
| Uscita di un solo processo con errore | Su `gloo` gli altri restano in attesa fino al timeout del gruppo (20-40 s, impostato nelle prove). Su NCCL non è noto. Verificare che `torchrun` abbatta il gruppo |
| Primi valori di `grad_norm` e delle perdite delle ancore in calibrazione | Devono ricalcare da vicino quelli del run storico: `grad_norm` 27,79; 28,17; 32,31; 30,94...; medie degli update 1-8: `log_rms` 1,145, `log_env` 2,19, `band_shape` 6,03, `ms_fast` 6,66, `ms_slow` 14,17 **[io]**. Controllo gratuito che flusso e segnali sono gli stessi. Solo il primo update può coincidere esattamente: in calibrazione i pesi restano fermi |
| Picco di memoria | Con activation checkpointing e tre gradienti in più a ogni passo 256-511 della calibrazione |

**Prima del rilascio dei training, non del collaudo.** Decisione D-2 e profili d'esito (D-10) in `docs/decisioni.md`; le
dichiarazioni D-4...D-14; catena di valutazione per C0/CN/CR (estrattori, sonde, costruttore del manifest con cardinalità e pin);
strumento che legga i profili `contrast_global_grad_*` dai tre `metrics.jsonl`; preventivo esecutivo. **Calendario:** dal 23/11
alle 08:00 al 25/11 alle 08:00 tutto Leonardo è in manutenzione e i job restano in coda. A 4,24 ore di muro per braccio, tre
training indipendenti partiti prima del 23/11 devono finire prima; le uscite su SCRATCH scadono a 40 giorni.

## 8. Che cosa non ho verificato

- **CUDA, NCCL, bf16 vero, torch 2.2, A100.** Tutto è su CPU con torch 2.14 e `gloo`.
- **Il modello da 30M allenato e i segnali veri.** Modelli minuscoli dei test e tensori sintetici; il 30M è stato solo istanziato
  all'inizializzazione, con segnali sintetici, dagli agenti della calibrazione (D-3); del flusso vero ho usato solo indici e
  metriche. Le norme di InfoNCE e RnC sui dati veri le darà solo la calibrazione.
- **QC, visibilità e validità vera della posa.** Non sono nel log: le ho supposte valide, quindi la copertura reale può solo
  essere più bassa. Il log non ha la fine delle finestre: ho usato inizio + 4 s, e 1 s come sensibilità (stesso esito); un agente
  ha ricostruito la fine esatta per il 59% delle finestre, con gli stessi numeri.
- **Che il flusso dei tre training coincida con quello del log.** Stessa ricetta e stesso seme; un agente ha confrontato 5.000
  passi fra due run storici e le sezioni del config, e i sorgenti del loader sono identici. Non è una prova sul run nuovo.
- **L'integrazione vera su `main`.** Solo una simulazione di un agente (9 test falliti).
- **Gli esempi MATLAB di `1d87e8d`.**
- **Valutatori, estrattori e launcher della nuova campagna.** Non sono consegnati.
- **Il limite dei file aperti dentro un job di Leonardo.** Letto solo sul nodo di login.
- I rilievi marcati [1A] non sono stati riverificati da un secondo agente.

## 9. Rettifiche e note di metodo

1. **Tre correzioni della verifica avversaria cambiano il senso di affermazioni che altrimenti avrei riportato.** (a) Che C0,
   percorso storico e contratto siano uguali bit per bit: vale per 3, 4, 6, 8 finestre, non per 5 e 7 (I-11). (b) Che dopo
   un'eccezione la catena riparta dal checkpoint e ricada nello stesso punto: falso, `afterok` la ferma (I-3). (c) Che l'esito
   favorevole della regola di lettura sia irraggiungibile: è improbabile, non impossibile (D-2). Le altre correzioni dei
   verificatori sono già nel testo nella forma corretta: per esempio la ripresa bit per bit solo fino a 2 processi, 15 config
   rifiutati e non 3, la quota di RnC riducibile senza posa.
2. **Una mia nota di lavoro era imprecisa.** Avevo annotato «gradiente clippato a ogni passo»: è attivo in 19.616 update su
   19.618, non agli update 14.778 e 14.974 **[io]**. Non era stata pubblicata.
3. **File `.pyc` negli alberi di lavoro.** Due episodi, che i rapporti degli agenti segnalano. Fra le 12:03 e le 12:07 sono
   comparsi 86 file `.pyc` nelle due copie condivise: li hanno scritti i sottoprocessi delle mie due esecuzioni della suite
   completa, e li ho tolti. Fra le 15:54 e le 15:55 altri 37 nella copia di `db5ac41`: li ha scritti un agente con `python -I`,
   che ignora `PYTHONDONTWRITEBYTECODE`, e li ha poi tolti. I sorgenti non sono cambiati: i 545 file sorgente controllati (su
   2.006 tracciati) sono rimasti identici a `db5ac41`.
4. **Una lettura di rete non prevista.** Un agente ha eseguito per errore `git ls-remote origin`: sola lettura.
5. **Leonardo.** Una sola lettura mia: `ulimit -n` e `scontrol show config` sul nodo di login. Nessun job, nessuna scrittura.
6. **Errori miei trovati dalla rilettura della bozza.** Fra gli 8 errori corretti: in I-2 avevo scambiato i due valori di `total`
   fra un processo e 4 processi; avevo attribuito 50,83 GPU-ore al consumo dei tre training storici, mentre è tre volte il consumo
   di T100-G; nella cartella di prove due log erano vuoti per un mio errore di scrittura.

## 10. Prompt per GPT

```text
Claude ha pubblicato la review del pacchetto contrastivo db5ac41 (prompt 77b7003) sul ramo
origin/worktree-claude-training-engine:
- docs/reviews/claude_review_contrastive_comparison_20261116.md
- prove in docs/reviews/claude_review_contrastive_comparison_20261116_evidence/ (LEGGIMI.md)
- due voci del 16/11 in docs/claude_GPT_shared
Nessun job, nessun file della linea 3 modificato, nessun cambio di ricetta. I tuoi esempi MATLAB di 1d87e8d non sono nel perimetro.

Verdetti.
- Disegno: schema approvato; tre regole da rivedere prima dei dati (D-1, D-3, D-2).
- Implementazione: approvata con correzioni. Nessun errore di calcolo trovato in loss, memoria, trainer, gradienti distribuiti,
  ripresa e comparatore. I sei test distribuiti girano sul Mac: 5 passati, [CR-4] fallisce per la tolleranza del test.
- Prerequisiti del collaudo: non soddisfatti. La calibrazione, così com'è, finisce in FAILED_REQUIRES_REVIEW.

Ti chiedo, in quest'ordine.
1. D-1, blocca la calibrazione. Sul flusso vero capgmyo ha 0 query ammissibili su 1.461, 0 blocchi superati su 18 e 13 finestre
   nel pannello 256-511 contro 32. Rifai il conto e proponi a Simone la scelta: esenzione con regola scritta prima (opzione A),
   memoria più lunga e soglia più bassa (opzione B), o un'altra. L'emendamento va scritto insieme in coverage_report, nel
   lettore _coverage e nel costruttore delle ricette, uguale nei tre bracci.
2. D-3, prima della calibrazione. La norma del gradiente di base vale 29,8 agli update 1-8, 6,7 a fine rampa e 2,65 a 19.618: il
   10% dell'avvio, a norma del termine costante, vale 0,44-0,49 a peso pieno e circa 1 in media. Proponi il riferimento della
   regola (opzioni 0-3), il rapporto atteso e le soglie di lettura, e scrivi prima che cosa succede se uno o entrambi i λ escono
   da [1e-4; 1]: sul 30M con segnali sintetici cadono a cavallo di 1. Nel benchmark registra le tre norme da 513 a 576.
3. Un solo giro di correzioni sul codice che la release fissa: I-1, I-3, I-4, I-5, I-6, I-7a, I-8, più I-7b e i test (I-2 e
   sezione 6). Dimmi quali rilievi non condividi e perché, con una prova.
4. Soglie di calibrazione e copertura in docs/decisioni.md prima del primo job; piano del collaudo con comandi, tetto, criteri di
   superamento e criterio di ripresa a 4 GPU; preventivo. Il budget è una decisione di Simone: interpretazione mia, l'estensione
   del 03/11 non copre nuove loss.
5. D-2 e D-10, prima del rilascio dei training: con le ampiezze dei bracci storici la non inferiorità su P2 è sostenuta in 1
   contrasto su 12 e 24 terne su 24 sono irrisolte. Proponi a Simone una delle opzioni (a)-(f) di D-2, o un'altra, e la
   definizione dei profili. Lettura mia, non una scelta: ridanno potere solo quelle che agiscono su margine o composizione di P2.

Poi ripassa da me col diff: rileggo le correzioni prima del collaudo. Lettura mia: finché D-1 e D-3 non sono decise non conviene
lanciare nulla, nemmeno l'audit CPU, che oggi fallirebbe per capgmyo.
```
