# Due prove sul confronto contrastivo a quattro GPU

Stato della consegna: **REVIEW_REQUIRED**. Nessun job sottomesso. La review di
Claude precede checkout, deploy e invii dei diagnostici.

Il rapporto 64775571 ha rilevato differenze fra run con entrambi i pesi contrastivi a zero e fra i checkpoint finali di due esecuzioni della stessa ricetta. Per InfoNCE e Rank-N-Contrast le traiettorie del precedente test di ripresa si separavano già prima del salvataggio. Queste due prove localizzano la prima differenza riproducibile. I risultati si leggono per braccio e per condizione.

## Precisione, backend e selezione del braccio

Sul modello a 30M usiamo lo stato iniziale autenticato della calibrazione. Ogni coppia di esecuzioni riceve gli stessi tensori, bersagli, maschere e stato dei generatori per rank. C0 è il controllo, CN seleziona InfoNCE, CR seleziona Rank-N-Contrast. Durante il confronto i due coefficienti sono zero. La replica C0/C0 misura quanto varia il calcolo senza cambiare braccio. Il confronto dei **vettori** di gradiente prima del clipping individua anche differenze di direzione con norme scalarmente uguali.

Prima di attribuire uno scarto al braccio o alla precisione, verifichiamo per rank le impronte di stato, batch e RNG. Un disaccordo rende il confronto corrispondente inconclusivo e identifica il primo controllo da correggere.

| Osservazione nella prova controllata | Lettura fissata prima dell'esecuzione |
|---|---|
| Tutti i bracci coincidono con C0 e le repliche C0 coincidono | Il percorso provato è ripetibile su questa fixture. Resta aperta l'origine della deriva sulle traiettorie lunghe. |
| bf16 e FP32 danno due risultati ripetibili, diversi tra loro, con lo stesso backend | Si misura l'effetto della precisione sul valore del gradiente. Per spiegare la variabilità tra due run bf16 serve anche uno scarto tra repliche bf16. |
| La replica C0/C0 varia col backend automatico e diventa ripetibile con backend controllato | Il backend o il modo di eseguire le riduzioni contribuisce nella prova. La differenza si quantifica sulle stesse coppie di tensori. |
| C0/CN o C0/CR differiscono sistematicamente mentre C0/C0 coincide, in particolare nel profilo deterministico | Verificare consumo RNG e primo blocco con gradiente diverso. Si apre un sospetto sul percorso del braccio a peso zero. |
| Un profilo fallisce per memoria, operatore non supportato o problema distribuito | Registrare profilo, eccezione e ultimo punto completato. Le coppie che richiedono il profilo mancante restano inconclusive. |

Con due misure per braccio e una fixture piccola, l'ampiezza osservata descrive solo i casi misurati. Il profilo FP32 serve da riferimento a parità di backend. Il confronto con i backend automatici dipende anche dal kernel effettivamente scelto. Tutti i profili condividono `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

La cella esegue quattro profili in gruppi di processi nuovi. Prova bf16 con selezione automatica del backend, bf16 con backend SDPA math, FP32 con lo stesso backend math e bf16 math con algoritmi deterministici. Il confronto FP32/bf16 math condivide anche l'impostazione TF32 disattivata. Ciascun profilo misura due backward per braccio in ordine C0, CN, CR, CR, CN, C0. Si usano otto finestre emg2pose su quattro GPU e il dropout muscolare della ricetta (0,4). La portata è quella di questo batch e di un singolo aggiornamento a peso zero. Le misure delle traiettorie storiche rimangono il riferimento per il comportamento nel tempo e sugli altri dataset.

La prova CUDA ricostruisce il contesto di un update a peso zero (255). Usa i pesi fissi della calibrazione, le chiavi reali della coda con età relative preservate e verifica nelle misure che entrambe le loss abbiano supporto nel batch. Questo disegno misura quanto varia un backward quando si ripristinano i suoi ingressi. Non ricostruisce lo stato storico addestrato al passo 255.

## Ripresa da uno stato comune

Il punto di partenza di ogni braccio è il checkpoint continuo storico al passo 576. Il rapporto CPU completo contiene le impronte dei due file. Su Leonardo abbiamo verificato la presenza del checkpoint di calibrazione (224 MiB) e dei due checkpoint continui (circa 442 MiB ciascuno).

| Braccio | SHA-256 storico completo a 576 |
|---|---|
| CN | `668bb2119d728126c7250903882376099e5be607bfe253a0552c44dedc7eddd5` |
| CR | `574220fa015a341a5caa9786a65085fb960f40ba144f8b4b39ac5dad347b31c8` |

I passi 577–580 producono un unico checkpoint per braccio. Il processo che lo ha scritto prosegue in memoria fino a 588. Due nuovi gruppi di quattro processi caricano i byte conservati a 580 e ripetono i passi 581–588. Il confronto fra le due riprese misura la ripetibilità del caricamento e degli otto aggiornamenti. I due confronti con la prosecuzione in memoria misurano l'effetto complessivo del riavvio.

Per ottenere il salvataggio a 580 il driver prepara in una cartella nuova una copia diagnostica dello stato a 576. Aggiunge 580 alla lista dei checkpoint da conservare. Rilegge la copia dal disco, ripristina la lista originale nel confronto in memoria e verifica l'impronta del contenuto completo contro lo stato storico. Il trainer salva il checkpoint intermedio con il proprio metodo ordinario. Le distanze finali dei modelli si leggono rispetto allo spostamento fra 580 e 588 del primo membro della coppia. La soglia storica dell'1% era riferita allo spostamento dall'inizializzazione fino a 576.

Leggere per primi input e RNG al passo 581, poi ai sette passi successivi. Una differenza del batch con lo stesso indice porta a esaminare loader, ricostruzione dello stream e prefetch. Una differenza del RNG prima del primo forward porta a esaminare il ripristino e la costruzione del nuovo processo. Con batch e RNG identici, confrontare le prime divergenze di loss e norma e gli stati finali di student, teacher, optimizer e memoria. Le due repliche ricaricate concordi e lontane dalla prosecuzione in memoria localizzano un effetto del riavvio. Le ipotesi comprendono anche il nuovo gruppo DDP e l'ordine delle riduzioni. Una differenza di ampiezza simile in tutte e tre le coppie lascia aperta l'attribuzione con questa quantità di repliche.

La misura finale dei modelli include parametri e buffer dello `state_dict`. Nel gate storico la distanza aggregava i parametri e verificava i buffer separatamente per uguaglianza esatta.

## Verifiche sul Mac

Passati 69 test CPU mirati (7 sui nuovi driver e 62 sui controlli CUDA, sul lettore e sulla compatibilità delle fonti esistenti). La prova aggiuntiva con quattro processi Gloo, un piccolo modello CN e un passo dopo il salvataggio ha verificato lo stato iniziale di ogni prosecuzione, i tensori ricevuti e il RNG. Le due riprese hanno prodotto pesi identici. Fra continuo e ripresa la distanza dello student è risultata 7,52 × 10⁻⁷ dello spostamento nel singolo passo. Nella prosecuzione in memoria DDP aveva già ricostruito i gruppi dei gradienti, mentre nei due processi appena avviati usava ancora i gruppi iniziali. La concomitanza rende questo un meccanismo da verificare su Leonardo. I dettagli sono nelle tracce `*.ddp.jsonl` e nell'evidenza locale `smoke_gloo_result.json`. Nella fixture sintetica si è alzata la pazienza dell'allarme per consentire il confronto dopo che il primo tentativo era terminato per collasso. Nelle celle CUDA restano attivi gli allarmi delle ricette originali.

## Decisione e consegna

Queste misure sono diagnostiche. Il rapporto storico conserva l'esito `FAILED_REQUIRES_REVIEW`. Una decisione sulle prove a 5.000 e 19.618 passi richiede la lettura congiunta dei nuovi rapporti. La consegna a **3-launcher** comprende un checkout congelato e tre celle GPU indipendenti dopo il checkout, una per la precisione e due per la ripresa. Al termine vanno archiviati rapporti per cella e profilo, hash, eventuali errori e consuntivi.

Ogni cella prenota 4 GPU, 32 CPU e 240 GiB per un massimo di 30 minuti. Il checkout richiede 4 CPU per 4 minuti. Il tetto incrementale è di 6 GPU-ore e 48,27 core-ore. Se le tre celle consumano per intero le allocazioni, il cumulativo raggiunge 11,28 GPU-ore e 99,64 core-ore sui tetti di 14 e 129,33. I job possono procedere in parallelo secondo la quota debug condivisa dagli altri esperimenti.

I 30 minuti costituiscono un tetto di prenotazione prudenziale, non una durata
misurata su CUDA dei nuovi script. Walltime e cgroup SLURM limitano tutti i
discendenti. Un timeout chiude la cella: raccogliere anche log, profili completati,
tracce e checkpoint parziali; nessun rilancio automatico. Un report mancante rende
la cella incompleta. I file storici vengono letti e verificati tramite hash;
ogni output richiede una cartella nuova.

Il generatore `common_state_diagnostics_20261118/prepare_handoff.py --frozen-sha
<SHA-pubblicato> --dry-run` stampa deploy e argv per il submitter comune, senza
eseguirli. Congelare soltanto la versione approvata dalla review. I diagnostici
mantengono sempre `training_approved=false`: `MEASURED_FOR_REVIEW` e
`CONTROL_INPUTS_VERIFIED` indicano disponibilità e coerenza delle misure, non
superamento del gate storico o via libera ai training.

## Verifiche locali della consegna

I due file pytest passano: **7 test**, 19,65 s. `bash -n` passa sul launcher.
Il digest dei sorgenti storici rimane
`46938f319181d408ee852cfaacfdc92cc5e44eaa1cfa770825ad96956eec9320`.
Smoke aggiuntivo a quattro processi Gloo: invarianti di confine, input e RNG
coincidono; reload_a e reload_b sono bit-identici. Prosecuzione/reload mostra
uno scarto student di 7,5231e-7 e teacher di 3,4473e-9, rispetto allo spostamento
del seguito. Optimizer non esattamente uguale, struttura e stato della coda uguali.
L'evidenza e i dati DDP sono in `smoke_gloo_result.json` nel pacchetto.
È un modello sintetico piccolo CPU con un solo update dopo la separazione;
non replica CUDA/NCCL né gli otto update della prova prevista.
