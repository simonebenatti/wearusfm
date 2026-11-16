# P1 19.618: review indipendente delle prove e dei costi, 16/11/2026

Verdetto del perimetro: **B con modifiche**, non un altro rilancio identico. La causa Lustre resta un'ipotesi, non una diagnosi chiusa. Nessun job o accesso remoto effettuato.

## Prove accertate

29 controlli offline superati (`p1-evidence-budget-20261116.py` / `.json`). Checkout esatto c2ae7ee; sacct originali raccolti da Claude, log e letture stat pinati nel JSON.

- Quattro allocazioni distinte: pose64571999 COMPLETED0:0,7.330s; P164572000 FAILED1:0,1.882s; P264572001 COMPLETED0:0,5.967s; P1r164596915 FAILED1:0,1.793s. Ogni allocazione1GPU/8CPU. Nessun doppio conteggio batch/extern. Gli intervalli Start/End riproducono ElapsedRaw.
- Entrambi i P1 hanno finito il subprocess (log fine estrazione); falliscono nel primo file_identity della cache, prima della verifica di quella cache. Primo suDB6, secondo suDB3, due nodi diversi lrdn0248/lrdn0707. DB6 del rilancio non raggiunto.
- I7successi su9 sono2pose+2P2+3P1, contro2rifiuti. Non sono9prove indipendenti né il numero totale di hash: ogni cache accettata è hashata almeno una seconda volta dopo la verifica NPZ, oltre agli hash del riferimento/ingressi.
- La tabella dei tempi deriva da birth/mtime dei file, mtime delle provenance e finejob, a granularità di secondo. Non registra le chiamate stat/hash direttamente. Gli intervalli di controllo includono verifica NPZ e secondo hash; non isolano la durata o il timestamp del primo hash fallito. Anche «scritta in1–2s» non è una misura del tempo di flush sul dispositivo.
- Mancano i due stat che hanno dato il mismatch e il digest della lettura fallita. Le letture successive hanno mtime/ctime alla scrittura e atime al controllo, ma non consentono di sapere cosa vedesse il primo stat. Il codice non confronta atime. Mancano prove per concludere che il contenuto non sia cambiato.

## Conto esatto

| Voce | GPUh | CPU allocate nei jobGPU |
|---|---:|---:|
| Stadio già eseguito |4,714444|37,715556|
| Soli due tentativiP1 |1,020833|8,166667|
| UlterioreP1,2h+180s |2,050000|16,400000|
| Stadio+P1, SENZA diagnostica |6,764444|54,115556|
| Stadio+P1+diagnostica con tetto comprensivo0,5GPUh |**7,264444**|**58,115556**|
| Stadio+P1+diagnostica richiesta30min e180sdi margine |**7,314444**|**58,515556**|

La proposta cita6,764444 come se coprisse B intera, ma omette la diagnostica. Anche diagnosticacon tetto complessivo0,5 supera7GPUh/56coreh dello stadio e7,15/57,2 includendo il vecchio margine dello stadio.

Per mantenere letteralmente diagnostica≤30min e≤0,5GPUh anche con il margine180s: richiestaSLURM≤27min, deadline interna precedente e budget0,5. Se si chiedono30min, pubblicare0,5nominali+0,05margine; non promettere un massimo0,5. Serve comunque emendamento esplicito per l'insieme B se P1rimane2h. La diagnostica può avere una riga straordinaria separata, ma non sparire dal totale di campagna o essere attribuita tacitamente a un'altra linea.

La riga cumulativaP1 con il nuovo tentativo al limite è3,070833GPUh/24,566667coreh, trejob, contro2/16eunjob del freeze. Il residuo di altre estrazioni non modifica automaticamente questo sottotetto: emendare esplicitamente quantità e numero di tentativi, preservando gli accounting dei falliti. Il rilancio unico precedente è consumato.

CPU-only:349,917778 è **inviluppo cumulativo misto consuntivi+massimi futuri**, non consumo già sostenuto. Un checkout2CPU×(300+180)s aggiunge0,266667, portando a350,184444, eccesso0,184444su350. Questo conto vale solo perUNcheckout nuovo: distinguere checkout diagnostica, runtimecorretto, piani/budget/inputs già inclusi e qualunque aggiunta. La decisione deve approvare il nuovo tetto esplicito; non nascondere l'eccesso nell'arrotondamento. I4/4,4coreh della diagnostica1GPU/8CPU stanno nei CPU dei jobGPU, non nel contoCPU-only.

Una riestrazione di tutte le tre fasi al tetto7+0,15, dopo gli attuali4,714444 e diagnostica0,55, prenoterebbe12,414444GPUh: non giustificata dal solo diverso commit di un controllo d'integrità. Valutare separatamente la compatibilità scientifica dei sorgenti; conservarne i pin per artefatto.

## Recupero delle cache già scritte

Nessuno score aperto o nuova inferenza. Il formato NPZ conserva code_sha, checkpoint_sha256, input_sha256, ricetta/split/soggetti; le cache con provenance già emessa hanno superato i loro controlli in quel momento. Questo permette una verifica successiva utile, ma non ricostruisce l'intera ricevuta mancante.

Nel wrapper, raw_before e il suo confronto dopo subprocess restavano in RAM; il fallimento precede check_unchanged finale su checkpoint/sorgenti, ST.check_source_contract, controlloCode/STOP e scrittura runtime/receipt. Una nuova lettura non può attestare retroattivamente l'assenza di modifiche durante tutto il job. Non inventare receipt storica, non cambiare FAILEDinCOMPLETED e non scrivere nelle cartelle fallite.

Un'adozione sarebbe possibile solo come protocollo distinto esplicitamente approvato: copia/verifica in nuovo output, provenienza del job fallito e nuova attestazione, limiti della garanzia storica dichiarati, gate/piani adattati. La politica congelata attuale la esclude. Non è il percorso più semplice per recuperare~30min di inferenza; restano più sicuri diagnostica circoscritta e riestrazione soloP1 dopo causa o comportamento sufficientemente documentato.

## Raccomandazione all'integratore

Pubblicare emendamento che separi: (1) diagnosi delimitata, (2) correzione sostenuta dalle evidenze, (3) eventuale nuovoP1 con criterio d'accesso, (4) nuovi checkout e CPU-only. La diagnostica non conferma a priori os.sync né autorizza un retry al primo PASS generico. Nessun ulteriore job richiesto o eseguito da questa review.
