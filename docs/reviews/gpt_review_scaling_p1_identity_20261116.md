# Linea 1, P1 a 19.618: B con modifiche

16/11/2026. Consegna `c2ae7ee02b3df5bdc3310b90d6608e211753dd07`, evidenze
`90ce085`, `7c5c886`, `ca18841`, `3143e52`; letto anche l'aggiornamento
documentale `01f853e` (suite Claude: 3.555 passati). Registro remoto letto
sul ramo `claude/review-scaling-300m-20261104`. Questa review non riesegue
quella suite né attesta payload remoti non raccolti.

**Verdetto: B CON MODIFICHE.** Favorevole a una diagnostica circoscritta,
con i criteri allegati. Non approvo un terzo tentativo identico né una
correzione globale di `file_identity/check_unchanged`. Preservare pose e P2;
il rilancio della sola P1 è subordinato all'esito diagnostico, alla correzione
verificata e all'emendamento operativo/budget. Nulla parte prima della
decisione di Simone. Claude resta autore e unico operatore; GPT non ha
eseguito SSH, raccolte, deploy, invii o modifiche ai checkout rivisti.

## 1. Diagnosi: plausibile, non chiusa

Il fatto dimostrato è il rifiuto di una coppia di attributi durante un hash,
dopo la conclusione del sottoprocesso di estrazione. Non è dimostrato che
il contenuto fosse invariato: mancano sia gli stat prima/dopo sia il digest
del controllo fallito. Il messaggio non identifica il campo cambiato.

Le due allocazioni fallite sono su nodi diversi (`lrdn0248`, `lrdn0707`).
I tempi della tabella sono ricostruiti da birth/mtime, provenance e fine job;
non sono misure dirette di ciascun hash né della durata del flush. Anche
«2,19 GB scritti in 1–2 s» descrive i metadati osservati, non la durabilità
dei byte. Il secondo caso indebolisce la spiegazione limitata al file appena
scritto; non prova da solo che scrivere DB6 alteri gli attributi di DB3.

I «7/9» sono esiti per cache raggiunta, non nove prove indipendenti: ogni
cache validata viene hashata almeno due volte. Il secondo DB6 non è stato
raggiunto. Il codice confronta `st_mtime_ns` e `st_ctime_ns`, non secondi
arrotondati. **`atime` non è nel confronto**: il suo normale aggiornamento
durante la lettura non basta a spiegare l'errore.

| Ipotesi | Cosa la distingue |
|---|---|
| Aggiornamento/revalidazione degli attributi del client Lustre, eventualmente fra precisioni temporali diverse | Stesso oggetto, lunghezza e SHA; cambiano solo mtime/ctime. Registrare interi ns e parte frazionaria, eventuali arretramenti, stat ordinario e fstat; confronto forzato col server solo dopo l'osservazione primaria |
| Scrittura ancora attiva o processo esterno | Byte/SHA o dimensione cambiano, errori I/O, PID/FD ancora scriventi; registrare writer unico, chiusura, ritorno del processo, attività di manutenzione conosciuta. L'assenza di FD locali non esclude writer su altri nodi |
| Sostituzione del percorso, symlink o inode | Differenza fra oggetto aperto e percorso, dev/inode/FID/link. Il controllo attuale riapre il percorso per l'hash e non confronta `st_dev` |
| Touch, chmod, chown, refresh o altra modifica dei metadati | Contenuto uguale ma timestamp/permessi/proprietà diversi; non attribuire automaticamente a Lustre. La creazione del sidecar modifica la directory, non dovrebbe modificare il file NPZ vicino |
| Problema I/O, client o manutenzione | Errori lettura/CRC/hash, differenze fra client/versioni; raccogliere versione/mount e log accessibili. Confrontare la manutenzione con una versione precedente soltanto se quella versione è documentata |

La documentazione Lustre descrive attributi distribuiti fra MDT/OST,
LSoM e prefetch di attributi tramite statahead/AGL. **Non afferma che un
normale `stat` su file chiuso possa essere trattato come LSoM approssimato**:
LSoM è un'interfaccia specifica. La stripe unica riduce la complessità, ma
non esclude problemi di timestamp o del client. Sono piste da misurare,
non una diagnosi per nome. [Manuale Lustre, §§21 e39.4.2.2](https://doc.lustre.org/lustre_manual.pdf).

`stat` non è uno snapshot transazionale di tutti i campi e `statx` offre
una richiesta esplicita di sincronizzazione col server; nessuna delle due
cose dimostra la causa qui. [stat(2)](https://man7.org/linux/man-pages/man2/stat.2.html),
[statx(2)](https://man7.org/linux/man-pages/man2/statx.2.html).

## 2. Diagnostica pronta da congelare

Il protocollo completo è in
`docs/reviews/scaling_p1_identity_diagnostic_protocol_20261116.md`.
Prevede un solo job, 27 minuti richiesti più il margine congelato di
180 s, **massimo contabile 0,5 GPUh / 4 coreh**, una GPU allocata senza
inferenza, 8 CPU, 32 GiB. Sei cicli sintetici prestabiliti, file freschi e
invecchiati almeno 12 minuti, controllo originale e osservazioni complete;
due giri di sola lettura su sei cache reali, con entrambi i percorsi dei
lettori. Nessun tuning di Lustre, drop-cache, modifica di file accettati,
fit o score.

La prova distingue esplicitamente: instabilità dei soli timestamp a byte
uguali; cambiamento di oggetto/contenuto; rischio sui lettori; mancata
riproduzione. Un esito nullo o incompleto è **inconclusivo**, non un via al
terzo tentativo identico. La correzione circoscritta richiede almeno una
riproduzione osservata del primo caso e tutti i controlli d'integrità
prescritti. Non serve attribuire definitivamente il difetto a uno specifico
sottosistema Lustre per correggere una guardia di pubblicazione dimostrata
troppo sensibile ai timestamp; serve sapere che cosa stiamo accettando.

## 3. Correzione: barriera I/O e contratto d'identità sono distinti

`os.sync()` non certifica l'immutabilità: è globale alle filesystem del
nodo e non fornisce l'errore specifico di una singola scrittura. Un `fsync`
mirato è più circoscritto e restituisce errori, ma non è un lock né una
garanzia contro sostituzioni o writer successivi. Non scegliere ora quale
risolve il caso; confrontare i percorsi previsti nella diagnostica.
[sync(2)](https://man7.org/linux/man-pages/man2/sync.2.html),
[fsync(2)](https://man7.org/linux/man-pages/man2/fsync.2.html).

**«Seconda lettura stabile con lo stesso SHA» da sola non basta.** La
politica candidata deve distinguere identità dei byte e metadati osservati:

1. Si applica soltanto alle nuove uscite del producer concluso, in directory
   esclusiva. Ingressi, checkpoint, codice e piani pinati restano protetti
   dalle guardie esatte attuali.
2. Oggetto regolare e percorso canonico invariati: dev/inode, lunghezza,
   byte effettivamente letti, tipo, permessi/proprietà e link devono restare
   coerenti. Verificare anche il legame FD/percorso. Nessuna sostituzione o
   variazione di size è classificabile come assestamento benigno.
3. Solo mtime/ctime possono essere trattati come osservazioni che si
   assestano, e soltanto nel caso sostenuto dalla diagnostica. Conservare
   ogni valore e il motivo d'accettazione; non eliminare `ctime` dal mondo.
4. Se il primo hash rileva quell'instabilità, una sola rilettura completa
   dopo 2 s, con identità stabile durante la rilettura e SHA identico al
   primo. SHA diverso, EOF corto, errore I/O o seconda instabilità: stop.
   Nessun ciclo «finché passa» e nessuna sostituzione del primo digest con
   il secondo. Un pin preesistente, ove pertinente, deve coincidere anch'esso.
5. Il controllo copre anche il confine **dopo la validazione NPZ**: stessi
   byte e stesso oggetto prima/dopo, osservazioni temporali separate. Non
   aggiungere il fallback al primo hash lasciando poi un confronto opaco
   dell'intero vecchio dizionario che annulli la politica appena introdotta.
   Al massimo una rilettura aggiuntiva per cache nell'intera validazione;
   tempi e byte letti entrano nel report e nel preventivo.

La serializzazione di receipt/provenance deve dire «contenuto e oggetto
verificati, transizione temporale osservata», non «nessun attributo è
cambiato». Il percorso ordinario conserva i due hash già previsti prima e
dopo la validazione; non occorre raddoppiare ogni lettura di tutti i file.

**Dove intervenire:** nuovo helper I/O esplicito e wrapper di recupero
circoscritto, con schema di audit e pin. `tokenizer_preflight.py` è incluso
nei **103 sorgenti fissati dal contratto del training**, SHA
`3bdd90d0a30c1089953345c0b63415f8c40974939af6dd05da63f11d59affe69`.
Modificarlo e ripinnarlo globalmente non è il fix minimo: rompe la verifica
del run accettato. Non usare monkeypatch, flag per ignorare errori o modifiche
al checkout `df2de35`. Per `scaling_evaluation19618.py`, derivazione e nuovo
entry point devono dichiarare esattamente l'emendamento operativo; sorgenti
numeriche, ordine d'importazione e contratto scientifico restano identici.

## 4. Commit misti: sì, con provenienza per task

Preferisco **conservare pose/P2 da df2de35 e rifare soltanto P1**, se ammessa
dai criteri. Claude deve prima raccogliere e verificare ricevute e pin dei
due successi: in questa consegna abbiamo lo stato scheduler, non una nuova
verifica dei loro payload.

I lettori già controllano `extraction_code_sha` per cache e richiedono un
checkpoint comune per braccio. Il vincolo a un unico commit è nel costruttore
operativo `build_reader_plans/_extraction_identity`, non una necessità
scientifica. Prova locale con fixture: builder attuale rifiuta la miscela,
schema del lettore la accetta con identità per cache coerenti.

Il ponte deve elencare per task **ID produttore, receipt SHA, commit, piano,
cartella e pin delle cache**, consentendo soltanto la coppia di revisioni
pubblicata. Verificare byte-identici i sorgenti numerici, stesso checkpoint,
gain, preprocessing, selezioni, supporti, fold e ambiente. Il nuovo commit
incorporato negli NPZ può cambiare i byte del contenitore; non autorizza un
cambiamento delle feature. Test sintetico di equivalenza della sola modifica
I/O e mutazioni di provenienza devono passare.

Non riestrarrei tutte e tre solo per uniformare il commit: consumeremmo
GPU senza una nuova domanda scientifica. Occorre un nuovo checkout di
recupero e un emendamento circoscritto alle guardie di rilascio; non rendere
`release_check/rereview_check` permissivi verso qualsiasi futuro sorgente.

## 5. Lettori: verificare entrambi i percorsi prima dei fit, non indebolirli in anticipo

Classificazione usa `file_identity/check_unchanged`; posa usa anche
`tokenizer_pose_readout.verify_file` e `pose_temporal_readout.file_state`,
con confronto di dev/inode/size/mtime/ctime **dopo l'intero fit**. Cambiare
il solo helper comune non coprirebbe tutto e invaliderebbe il contratto.

La diagnostica prova questi percorsi su cache già chiuse, all'inizio e alla
fine. Seguono comunque la validazione completa e i benchmark già previsti
prima dei fit. Se una cache vecchia cambia metadati a hash fermo, fermare
il rilascio dei lettori e preparare una correzione separata, applicata
simmetricamente a 30M/300M e al confronto. Se cambia il contenuto o l'oggetto,
è un problema d'integrità, non un falso positivo da assorbire.

Un test di mezz'ora non garantisce 43 ore senza guasti: consente però di
intercettare il rischio noto senza una riscrittura preventiva. Annotare
eventuali refresh programmati durante i fit. Un refresh può lasciare gli
SHA identici ma cambiare lo stato confrontato a fine job; non va confuso
con corruzione dei byte né nascosto dalla guardia.

## 6. Budget: tre emendamenti espliciti, nessun costo nascosto

Accounting ricalcolato dalle sole righe di allocazione di sacct, senza
`.batch/.extern`. I due job P1 restano FAILED e contano nel consuntivo.

| Voce | GPUh | Coreh nei job GPU |
|---|---:|---:|
| Quattro job già eseguiti | 4,714444 | 37,715556 |
| Dei quali i due P1 | 1,020833 | 8,166667 |
| Diagnostica 27 min + 180 s | 0,500000 | 4,000000 |
| Un nuovo P1, 2 h + 180 s | 2,050000 | 16,400000 |
| Totale possibile dello stadio | **7,264444** | **58,115556** |

**Il 6,764444 della proposta omette la diagnostica.** Chiedendo invece
30 minuti nominali più 180 s si arriverebbe a 7,314444 GPUh/58,515556 coreh.
Anche la versione con diagnostica entro 0,5 supera sia 7/56 nominali sia
7,15/57,2 con il vecchio margine. Non spostarla tacitamente su un'altra linea.

Riga P1: con un nuovo tentativo al limite diventa **3,070833 GPUh,
24,566667 coreh, tre tentativi**, contro 2/16 e un job originari. Il rilancio
unico precedente è consumato. Un nuovo via deve identificare il tentativo
successivo e il criterio che lo ammette, non autorizzare retry indistinti.

CPU-only: 349,917778 è un **inviluppo di consumi e massimi futuri**, non
quanto già consumato. Un checkout ulteriore vale 0,266667 con margine:
350,184444; due valgono **350,451111**. Contare separatamente checkout della
diagnostica e del codice corretto; i checkout piani/budget/input già inclusi
non vanno contati due volte. Le 4 coreh della diagnostica stanno nella
colonna dei job GPU, non qui.

Per rendere la decisione concreta propongo, se si approva l'intera B
condizionata, tetti **7,30 GPUh / 58,40 coreh nei job GPU**, **P1 cumulativa
3,10 GPUh / 24,80 coreh e massimo tre tentativi**, **CPU-only 351 coreh**,
comprensivi dei margini sopra, con al massimo due checkout aggiuntivi.
Nessuna nuova riserva da spendere automaticamente. È un emendamento proposto,
non approvato da questa review. Se basta un checkout, si pubblica il conto
minore. Se il piano concreto richiede altro, si ferma e lo dichiara.

## 7. Alternative senza alterare i checkout storici

- **Recupero delle cache già prodotte:** possibile soltanto con un protocollo
  e una ricevuta di recupero separati, nuova cartella, pin attuali e limite
  storico dichiarato. Embedded metadata e sidecar parziali aiutano; non
  ricostruiscono gli stati in RAM e i controlli finali mai persistiti.
  H1=H2 oggi non rende il vecchio job COMPLETED. Non sceglierei questa
  maggiore complessità come prima opzione per risparmiare circa mezz'ora
  di inferenza, ma non la escludo per principio. Nessun mix opportunistico
  dei dataset dei due tentativi o selezione sulla base degli score.
- **Attestazione dopo estrazione in processo CPU separato:** conserva il
  producer numerico, ma richiede comunque codice/protocollo di pubblicazione
  nuovo. Il CLI attuale non valida cache esistenti: `stage=validate` verifica
  gli ingressi, non completa quelle uscite.
- **Staging locale di input già pinati:** può isolare un problema Lustre dei
  lettori, verificando hash della copia e capacità disco; ha costi, percorsi
  e provenienza da pianificare. Non sana le ricevute mancanti.
- Un'attesa o `sync` esterni **prima** del wrapper non inseriscono una barriera
  fra il suo producer interno e il controllo che fallisce. Un altro invio
  identico ha già fallito e non è il passo successivo raccomandato.

## 8. Consegna e verifica

Diagnostica → lettura rispetto ai criteri congelati → eventuale patch I/O,
ponte per task e piano aggiornato → verifica del pacchetto → unico P1
condizionato → catena dei lettori con i cancelli previsti. Non introdurre
review fra ogni singolo invio già coperto dal pacchetto approvato. Un nullo
diagnostico o un errore d'integrità non soddisfano il criterio per proseguire.

Prove locali e note indipendenti in
`results/infra/scaling_300m_20261104/p1_identity_review_20261116/`.
La review non riguarda nuovi score, loss, training o scelta del checkpoint.
Sorgenti, criteri e consumi storici restano conservati senza riscritture.
