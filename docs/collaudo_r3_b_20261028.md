# R3: collaudo lungo B — proposta del 28/10/2026

**Stato aggiornato il 28/10/2026: AUTORIZZATO AL LANCIO da Simone.** Dopo la consegna completa con tetto di 6 GPU-ore,
Simone chiede: «puoi lanciare tu i test? cosi a claude faccio fare solo il controllo». GPT esegue il checkout e il singolo collaudo B;
Claude svolge il controllo. Restano invariati codice `24139ab`, ricetta, criteri e preventivo presentati. Nessun altro job incluso.
La proposta originale riportata sotto e' ora il criterio applicabile a B. Il dbg resta fallito secondo la regola allora firmata.


Riferimenti: [valutazione scientifica del 28/10](reviews/pose_scientific_assessment_20261028.md),
[pilot della posa](pilot_posa_v1.md), [emendamento diagnostico](emendamento_pilot_posa_v1.md).
Configurazione e provenienza definitive saranno congelate in
`results/infra/pilot_posa/piano_b_20261028.json`; i comandi in `results/infra/pilot_posa/comandi_b_20261028.txt`.
Questi file devono identificare il commit completo approvato, il checkout di campagna, gli ingressi e gli output. Nessuno SHA mobile
o ancora da definire vale come autorizzazione all'esecuzione.

## 1. Domanda e limiti della prova

B verifica se la ricetta R3 riduce la MSE della posa su un pannello fisso del train dopo 5.000 aggiornamenti con lo schedule del run
vero. L'audit precedente non ha rilevato difetti nei controlli sui 20 casi esaminati; non ha dimostrato «allineamento 10/10» o misurato
un rapporto segnale/rumore q. E3 su R3c misura una ridge su un encoder gia' addestrato: non identifica il limite raggiungibile da R3.
Queste evidenze consentono di preparare B senza altri audit o estrazioni come prerequisiti.

La prova non isola causalmente H1: aumentando gli aggiornamenti cambia anche la rappresentazione, quindi ottimizzazione e
accessibilita' possono contribuire insieme. Un passaggio significa **riduzione della loss sul pannello nella ricetta estesa**;
non dimostra da solo dipendenza dal movimento, generalizzazione a utenti nuovi o trasferimento agli altri dataset. Anche la correzione
di bias puo' ridurre la MSE. E1/E2 e i confronti di trasferimento restano le letture scientifiche del candidato effettivo.

## 2. Ricetta e durata

- R3 a 30M, seme 0, batch globale 32 su quattro A100 (8 finestre per rank, `accum_steps=1`): ancore con peso 0,2, posa con peso 1,0,
  JEPA con peso 0. Testa e loss immutate.
- M = 280, mano non sollecitata inclusa, statistiche di posa congelate e stesso manifest/campionamento di R3. Il piano registra
  le identita' degli ingressi e l'hash delle statistiche; nessuna ricalibrazione sulla base del pannello. SHA256 atteso delle statistiche:
  `757f2aeebf688063747d50e55d983dd6b3c4cea268ce43d2bae06f3e99524a7e`.
- Run nuovo dall'inizializzazione prevista da R3. **Non riprendere dal checkpoint decaduto del dbg.**
  Directory dedicata `PILOT-POSA-R3-B-30M-s0-20261028`, nel percorso esterno al codice specificato dal piano. Il launcher con
  `REQUIRE_NEW_RUN=1` deve rifiutare una directory gia' esistente e opzioni di init/branch/resume incompatibili con questa partenza nuova.
- WSD: `total_steps=49046`, `steps_at_d=98094`, warmup 1.962, LR di picco 1e-4, stop operativo `max_steps=5000`.
  Il decadimento resta quello del run vero, successivo al collaudo; `total_steps=5000` sarebbe una configurazione diversa e va rifiutato.
- Diagnostica E4 ogni 500 passi e pannello della posa ogni 500 passi. Il config effettivo e' verificato contro il piano prima della lettura.

La somma dei LR fino a 5.000 vale 0,40195: 4.019,5 aggiornamenti al LR di picco come semplice integrale, contro 671 nel dbg.
Non sono aggiornamenti Adam equivalenti e non prevedono il guadagno atteso.

Una futura continuazione da questo checkpoint richiede una decisione separata: config compatibile, stesso pannello, controllo della
provenienza e della ripresa. Il superamento di B non autorizza la continuazione o la campagna completa.

## 3. Pannello fisso del train

**Dimensione richiesta: 32 utenti × 2 registrazioni × 4 finestre da 4 s = 256 finestre.** «Registrazioni» non significa sessioni
distinte. L'appartenenza a sessione, fase e lato viene registrata; non si assume che due registrazioni dello stesso utente rappresentino
contesti indipendenti. L'unita' di ricampionamento statistico e' l'utente.

### Selezione deterministica

1. Considerare soltanto registrazioni emg2pose ammesse dal manifest effettivo con probabilita' di campionamento nel training positiva.
   Deduplicare le identita' delle registrazioni prima di costruire i candidati.
2. Costruire finestre da 4 s sulla geometria e sulla griglia del loader. Sono eleggibili solo finestre con almeno un elemento di posa
   valido secondo l'intersezione fra maschera dei target e patch esistenti, la stessa usata dalla loss. Non introdurre una soglia
   arbitraria sulla frazione valida; salvare i conteggi effettivi.
3. Per registrazione ordinare i candidati per fine e poi per inizio, con ordinamento deterministico degli eventuali ulteriori pareggi.
   Costruire un insieme senza sovrapposizione con la regola greedy della prima fine disponibile. Da tale insieme estrarre quattro
   finestre senza reinserimento. Una registrazione che non ne offre quattro non e' eleggibile.
4. Usare un generatore dedicato, indipendente dallo stream di training, con seme composto dal seme del run e dalla costante del
   pannello 7272. Permutare in modo deterministico utenti e registrazioni; procedere ai candidati successivi se una registrazione
   non e' eleggibile o un utente non fornisce due registrazioni. Registrare esclusioni e motivi, senza scegliere in funzione degli errori
   o delle predizioni del modello. Il pannello deve essere completo prima del primo aggiornamento.
5. Se non si raggiungono esattamente 32 utenti, 64 registrazioni distinte e 256 finestre eleggibili, fermare la preparazione del run:
   nessun ridimensionamento silenzioso, cambio di seme opportunistico o allentamento dei vincoli.

Il pannello e' bilanciato per utente e condizionato all'ammissibilita': non replica le probabilita' del flusso di training e non stima
la sua loss media. Le finestre non si sovrappongono all'interno di una registrazione; eventuali acquisizioni simultanee fra registrazioni
rimangono nello stesso gruppo utente e vanno rese riconoscibili dai metadati. Il pannello appartiene al train, non e' un holdout scientifico.

### Materializzazione e immutabilita'

Usare preprocessing, scale, geometria, maschere e target del loader di training; le maschere estratte per il pannello restano fisse.
Valutare lo student in modalita' eval, senza gradiente e senza dropout del modello. Non ricampionare ingressi o maschere a ogni misura.
Costruzione e valutazione devono lasciare invariati stream, stato casuale, modo del modello e traiettoria del training.

`pose_panel_set.json` conserva utenti, registrazioni, sessione/fase/lato e prompted/unprompted quando disponibili, coordinate delle
finestre, seme, esclusioni, supporti validi e hash del contenuto materializzato. L'identita' deve essere confrontata alla ripresa.
Un'identita' diversa non si accetta come lo stesso pannello.

## 4. Misure e aggregazione

Valutazioni obbligatorie ai passi **0, 500, 1.000, 1.500, 2.000, 2.500, 3.000, 3.500, 4.000, 4.500 e 5.000**.
Il passo indica il numero di aggiornamenti completati. Salvare in `pose_panel.jsonl` gli errori per finestra e i dati necessari a
ricostruire gli aggregati, non soltanto la media. Nessuna selezione del checkpoint migliore fra le valutazioni.

Per una finestra, la MSE e' la media degli errori quadratici standardizzati su tutti gli elementi validi patch/giunto, come nel
contratto della posa. Per un utente, media delle sue otto MSE di finestra. La metrica primaria L(t) e' la media delle 32 medie utente.
Il rapporto primario e' **L(5000)/L(0)**, non la media dei 32 rapporti individuali. Con copertura completa e quote uguali la media
delle 256 finestre coincide con questa aggregazione; il lettore deve comunque verificare le quote.

Descrittivi, senza nuove soglie di passaggio:

- MSE e MAE standardizzate per finestra e per utente; MAE standardizzata per giunto. MAE in radianti per finestra, ricostruita con
  le statistiche congelate; non viene salvata una MSE separata per giunto.
- MSE/MAE del predittore costante zero nello spazio standardizzato, ossia la media train per giunto secondo le statistiche congelate.
- Media delle predizioni per giunto; **varianza temporale calcolata per giunto** sul supporto valido di ogni finestra, di cui si salva
  la media sui giunti con almeno due istanti validi (ddof=0; zero se nessun giunto ha supporto sufficiente).
  Non usare come misura di dinamica una varianza che mescola i giunti: venti bias diversi ma costanti nel tempo possono produrla.
  Riportare i conteggi, compresi i casi con supporto temporale insufficiente a interpretare la varianza.
- Conteggi degli elementi e delle patch valide, distribuzione della frazione valida per finestra, quote di lato/fase/unprompted.
- Andamento fisso 2.000→5.000 e loss di training nei blocchi **2.001–2.500** e **4.501–5.000**, escludendo dalla media della posa
  i passi senza posa e riportandone sempre i conteggi. Non sono confronti appaiati delle stesse finestre.

## 5. Cancello proposto

### Integrita' e controlli tecnici

- File e config integri; provenienza di codice/dati/statistiche, pesi, DDP 4×8 senza accumulo, numero e sequenza dei passi devono
  coincidere con il piano. Log di training completi ai passi 1–5.000, valori finiti, chiusura per completamento di 5.000 aggiornamenti.
- Tutte le 256 finestre devono contribuire a ogni misura con identita', maschere, supporti e quote invariati; tutti i valori necessari
  alla lettura devono essere finiti e L(0) strettamente positiva. Controllare esattamente 32 utenti e otto finestre per utente.
- Al passo zero le predizioni della testa inizializzata a zero devono essere zero e MSE(0) deve coincidere con la MSE del costante,
  entro la tolleranza numerica dichiarata nei test del lettore. E' un controllo d'integrita', non una seconda soglia di apprendimento.
- Conservare i controlli di finitezza, stabilita'/non collasso (valutazioni attese presenti, nessun allarme), composizione della loss e
  ritmo del collaudo R3: **mediana del passo
  ≤0,81 s**, con la stessa esclusione dell'avvio del lettore. Questa mediana non include necessariamente tutti i costi del job.
- Scala iniziale: nei passi 1–100, posa pesata almeno 0,1 volte la somma delle ancore pesate. Tetto: nei passi 500–5.000, posa pesata
  al massimo due volte la somma delle ancore pesate. Per entrambi usare passi con posa presente. Conteggio di posa positivo in almeno
  il 90% dei 5.000 passi. Restano controlli tecnici; non si reintroduce una seconda discesa obbligatoria della posa sul flusso.
- Endpoint a 5.000 e `checkpoint.pt` non vuoto presenti. Non richiedere checkpoint predecay/postdecay: B si arresta prima del
  decadimento dello schedule vero. La futura ripresa richiede anche la verifica dello stato del checkpoint, non soltanto la presenza
  del file. La discesa della loss totale fra primi e ultimi 50 passi e' descrittiva, non un ulteriore gate di apprendimento.

### Riduzione della loss sul pannello

Il gate richiede entrambe le condizioni:

1. **L(5000) ≤ 0,9 × L(0).**
2. **Limite superiore dell'IC95 appaiato del rapporto <1.** Bootstrap percentile: 1.000 ricampionamenti di 32 utenti con reinserimento,
   seme **20261027**, percentili 2,5 e 97,5. Ogni utente estratto porta le medesime finestre e i due endpoint insieme; calcolare per
   ogni replica il rapporto delle medie aggregate con i medesimi pesi del punto stimato. Repliche con denominatore nullo o valori
   non finiti invalidano la lettura, non si scartano selettivamente.

Non si richiedono un ulteriore −10% dopo il warmup, un gate aggiuntivo contro il medesimo costante o una varianza delle predizioni
superiore a 0,01. Il riferimento zero evita di bocciare un miglioramento gia' avvenuto entro 2.000. Il bootstrap descrive la variabilita'
fra gli utenti del pannello train: non e' un intervallo di generalizzazione a utenti mai visti.

Queste soglie sono una scelta operativa proposta prima di B, non derivate dai pochi punti del dbg. Non cambiano le regole di E1/E2/E3
o il criterio di trasferimento del pilot firmato.

### Lettura degli esiti

- **Superato:** controlli tecnici e d'integrita' soddisfatti, piu' entrambe le condizioni sul pannello. Apre una decisione sui risultati;
  non autorizza R3 completo. Se il flusso appare piatto, registrare la discordanza e la differenza fra i due campionamenti.
- **Non superato:** dati completi ma almeno un criterio non soddisfatto. R3 completo resta fermo; nessun aumento automatico del peso,
  cambio di target o nuova variante. Se il flusso scende e il pannello no, il gate non e' superato.
- **Non valutabile/incompleto:** endpoint assente, stop per tempo, pannello o provenienza non confrontabili, supporti mancanti o
  dati necessari non disponibili. Distinguere la causa tecnica da una mancata discesa della loss; non promuovere il run e non rilanciarlo
  automaticamente. Un errore di integrita' dimostrato va registrato come tale, non come evidenza contro l'apprendibilita' della posa.

## 6. Preventivo proposto e limiti operativi

**Proposta aggiornata: quattro A100, 32 core, 90 minuti di tetto = 6 GPU-ore e 48 core-ore.** Sostituisce per questa proposta il
precedente preventivo di 75 minuti / 5 GPU-ore / 40 core-ore; non e' gia' approvata. Motivazione: 5.000 passi a 0,711 s richiedono
circa 59,25 minuti di soli passi; alla soglia tecnica di 0,81 s richiedono 67,5 minuti. Preparazione del pannello, tutte le valutazioni,
avvio e checkpoint devono avere un margine esplicito.

Impostare **`TIME_LIMIT_S=5100`** nel trainer e limite SLURM **90 minuti = 5400 s**: 300 s riservati alla chiusura rispetto al limite
interno. Alla soglia di ritmo restano 1.050 s entro il limite interno rispetto ai 4.050 s dei soli passi. Il margine comprende avvio,
pannelli e checkpoint; non e' una garanzia di completamento. Il limite interno e la riserva devono essere verificati nel percorso di
uscita e salvataggio: la sola mediana del passo non garantisce che il job chiuda entro SLURM. Registrare tempi complessivi e tempo del
pannello; niente prolungamenti impliciti se il margine non basta.

| Attivita' | Tetto di risorse | Costo massimo |
|---|---|---|
| Checkout di campagna nuovo | 2 core ×10 min, senza GPU | 0,3333 core-ore |
| B, singolo job | 4 A100 e 32 core ×90 min | 6 GPU-ore e 48 core-ore |
| **Totale proposto** | Checkout piu' B | **6 GPU-ore e 48,3333 core-ore** |

Memoria e altri parametri operativi sono quelli dichiarati nei comandi definitivi, da verificare contro i limiti della partizione/QoS
prima della firma. Riconciliare il residuo del budget nel ledger e registrarlo nel piano: il tetto proposto non amplia da solo il budget
della seconda ondata. Il costo consuntivo comprende anche attese delle altre GPU durante valutazioni sul solo rank principale.

Nessun retry automatico, nuovo seed, job di scorta, E3 su R0 o J2 bis e' incluso. La lettura locale dei risultati non autorizza nuove
estrazioni o training. Un'esigenza diversa di risorse o numero di job richiede preventivo aggiornato e autorizzazione separata.

## 7. Condizioni della consegna prima della firma

Congelare codice, piano e comandi dopo test mirati: selezione non sovrapposta e completa; target/supporti e identificazione del pannello;
uguaglianza del training con pannello spento/acceso; conservazione di RNG e modalita' del modello; logging del passo zero e dell'endpoint;
ripresa con identita' verificata; aggregazione per utente e bootstrap; rifiuto dei log incompleti o incoerenti; schedule e controlli tecnici.
Verificare il comportamento distribuito quando soltanto il rank principale prepara/valuta il pannello, comprese le uscite per errore.

Il checkpoint e il dataset reale non sono sostituiti dai test sintetici: i test verificano il contratto del collaudo, non anticipano il
suo risultato. Review tecnica e scientifica della consegna finale, firma esplicita di Simone su criterio e preventivo, poi eventuale
sottomissione con la procedura comune. Il via successivo di Simone e' registrato in apertura e nel piano; non estenderlo ad altri esperimenti.
