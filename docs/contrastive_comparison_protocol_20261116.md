# InfoNCE e Rank-N-Contrast: confronto controllato — 16/11/2026

**Stato: REVIEW_REQUIRED.** Simone autorizza il disegno e l'implementazione delle due prove; chiede la verifica di Claude prima dei lanci. Questo protocollo prepara la review, non è una release di training. Nessun seguito automatico a 49.046, nessun cambio di loss a run iniziato.

## 1. Domanda, base e controllo

**A pari architettura, dati e dose, quale aggiunta alla ricetta ancore+posa offre un miglioramento utile nella regressione dinamica e/o nella classificazione, senza un deterioramento rilevante degli altri compiti?** Le due idee usano segnali di supervisione diversi: InfoNCE discrimina istanze del mix train; RnC ordina rappresentazioni degli istanti etichettati secondo gli angoli. Gli angoli sono già disponibili a tutti i bracci tramite la MSE di posa: non si aggiungono dati etichettati soltanto a RnC.

La base resta **T100-G 30M**: finestra 100 ms, hop 25 ms, overlap 75 ms, GELU; nessun ramo log-quadratico. È una scelta adattiva fondata sui risultati di sviluppo già visti, dichiarata prima di questi risultati.
Ricetta storica di riferimento: `results/infra/tokenizer_execution_20261103/19618/T100-G/config.json` (SHA256 `f9d0318462fd30a5592087ef72c63d3036a88ff1d07fc0dad8624e36253c2cfa`). Il riferimento è la configurazione, non un checkpoint da cui ripartire.

| Braccio nuovo | Loss di base | Termine aggiunto | Dati usati dal termine |
| --- | --- | --- | --- |
| C0 | 0,2 ancore + posa | Nessuno | — |
| CN | Identica | λ_N(t) InfoNCE | Finestre ammissibili del mix train |
| CR | Identica | λ_R(t) Rank-N-Contrast | Istanti con posa valida, candidati dal batch globale |

**Tre training nuovi**, stessa inizializzazione non addestrata, seme 0, stesso stream/ordine di finestre, montaggi, maschere e quota dei dataset; batch globale 32, quattro rank, accumulo **1**, sei worker del loader per rank come nella ricetta storica (24 identità di worker). AdamW, LR 1e−4, clipping 1, weight decay 0,05, EMA 0,996 e resto della ricetta storica invariati. JEPA resta a peso 0. WSD totale 49.046, warmup 1.962, decay finale 20%; letture a 5.000 e 19.618.

Tutti i bracci hanno lo stesso proiettore registrato e calcolano entrambi i termini e la manutenzione della memoria; cambiano soltanto i pesi. Il proiettore usa RNG isolato e non sposta l'inizializzazione dei parametri condivisi né lo stream casuale. Si riusano gli encode student/teacher già calcolati. Il costo del controllo include quindi l'infrastruttura nuova: confronto a pari dose e percorso di calcolo, non promessa di costo identico al G storico. Il teacher e la sua copia del proiettore seguono l'EMA normale.

## 2. CN: InfoNCE con teacher e memoria

Conserva il disegno del [07/11](contrastive_proposal_20261107.md), salvo il nuovo confronto a tre e la calibrazione comune sotto.

- Query: student sulla vista mascherata storica; positivo: teacher EMA completo sulla stessa finestra fisica e stesso montaggio. Nessuna nuova trasformazione di ampiezza, tempo o canali.
- Readout: media dei 64 slot, poi delle patch fisicamente esistenti con QC valido, prima del decoder. Le patch interamente mascherate sono query calcolate e restano nel supporto; una finestra senza alcuna chiave student visibile valida non contribuisce a NCE.
- Proiettore 384→384→GELU→128, bias, niente BatchNorm/dropout; coseno normalizzato FP32, temperatura **0,2**. Query con gradiente; chiavi detached. Niente secondo student forward o simmetrizzazione.
- FIFO **1.024 chiavi per dataset**, età massima **256 update**. Candidati = memoria precedente più batch globale corrente. Stesso dataset; non escludere sistematicamente lo stesso soggetto.
- Escludere copie della stessa finestra fisica indipendentemente dal montaggio; nella stessa registrazione/trial escludere overlap o gap inferiore a **4 s**. Stessa sessione ma trial non confrontabili: esclusione conservativa. `trial=None` identifica l'asse continuo, non un campo assente.
- Deduplicazione deterministica: chiave più recente, corrente prima della FIFO, pareggio per ordine globale rank/indice. Identità e tempi mancanti sono errori.
- Minimo **32 negativi** ammissibili; altrimenti contributo zero, registrato. Denominatore della loss **32 finestre globali**, non numero di query ammissibili.
- Chiavi teacher calcolate prima dell’update; optimizer ed EMA una volta, poi commit della memoria con quelle chiavi già fissate. L’ordine EMA/commit non ricalcola né altera le chiavi. Stato completo nel checkpoint; nessuna modifica durante eval o ricomputazione del backward.

Restano possibili falsi negativi e scorciatoie di soggetto/sessione. NCE non supervisiona direttamente l'ordine temporale né promette recupero del movimento.

## 3. CR: RnC corretta e globale

Il nucleo di Claude, verificato nella [review del 16/11](reviews/rnc_claude_review_20261116.md), viene corretto nei due punti principali.

- Readout temporale `mean_K(enc.z)` dello student mascherato, stesso supporto `pose_time_valid` della MSE posa, incluse patch nascoste/slab; padding e target invalidi esclusi.
- Un'ancora ogni **8 patch**, fase **4**: nominalmente 200 ms di passo. Nessun campionamento casuale aggiuntivo.
- Candidati provenienti dal **batch globale**. Raccolta delle feature con gradiente anche dal lato candidato; target e maschere detached. Un rank senza pose partecipa comunque alle collettive. Accumulo >1 non supportato in questa prima prova e rifiutato.
- Tutti i candidati ammissibili sono anche positivi nei rispettivi termini RnC: **`all`**, non la variante `cross` che conservava la stessa finestra solo nei denominatori. Esclusa la diagonale e le coppie senza giunti validi in comune. Nessun filtro di identità della finestra applicato soltanto a uno dei due ruoli.
- Si escludono simmetricamente anche le coppie che rappresentano lo stesso istante fisico: dataset, soggetto, sessione, trial, frequenza e indice nativo `start + ceil((p+0,5) × patch_ms × fs / 1000 − 1e−9)`. Le copie restano ancore rispetto agli altri punti: è esclusione della coppia duplicata, non deduplicazione completa o ripesatura della popolazione. Molteplicità riportata nel log.
- Distanza target RMS sugli angoli standardizzati validi in comune; **risoluzione 0** (ordine esatto e parità esatte). Similarità coseno dopo LayerNorm senza parametri; temperatura **0,1**.
- Per ciascuna coppia i,j, denominatore sui k con distanza target da i almeno pari a quella di j. Tutti i pareggi inclusi; mascheratura numericamente sicura, senza il sentinel finito −1e4 che falsava temperature estreme.
- Normalizzazione per ancora al valore con similarità uniformi; almeno due positivi per far contare un'ancora, conteggio dalle sole maschere. Media delle ancore nella finestra e poi fra le finestre con supporto. Finestre senza posa non ricevono questo termine.
- Loss globale identica calcolata su ciascun rank; backward della raccolta differenziabile somma i contributi e la successiva media DDP restituisce il gradiente globale. Non si applica di nuovo la compensazione world-size usata per numeratori locali. Il test distribuito deve verificare questa algebra, comprese rank vuote e ripresa.

RNC **non usa** il proiettore o la memoria teacher nel proprio valore matematico; CR ne esegue comunque il percorso a peso zero per mantenere la stessa infrastruttura del confronto. Non si interpretano molti istanti correlati della stessa finestra come molti esempi indipendenti. Nel collaudo si pubblicano duplicati fisici, overlap, registrazioni/soggetti distinti e coppie/ancore utili; non si modifica il mix del candidato per aumentare le pose.

La geometria è rispetto agli angoli, non alla velocità o alla forza. La stessa postura può accompagnarsi a contrazioni differenti: l'effetto sul trasferimento generale resta da misurare.

## 4. Peso, attivazione e copertura

**Una calibrazione train-only comune, senza optimizer step o selezione sulle sonde.** Pesi e teacher fissi allo stato iniziale; la memoria avanza secondo regole congelate.

1. Primi **256 batch globali** dello stream train congelato: riempimento e diagnostica.
2. Batch **256…511**: gradienti completi globali di base, InfoNCE e RnC sull'encoder condiviso (tokenizer, identity, local, pool, backbone; esclusi teste, decoder e proiettore).
3. Selezionare i **primi 32 batch congiuntamente informativi**: entrambi i termini hanno supporto, tutte e tre le norme sono finite e positive. Pubblicare indice e ragione di esclusione di ogni batch. Se non ne esistono 32 entro il pannello, stop e revisione; non proseguire lo stream alla ricerca di esempi favorevoli.
4. Per ciascun termine c: `λ_c = 0,10 × mediana_b(||g_base,b|| / ||g_c,b||)`, sugli stessi 32 batch. Entrambi i pesi devono cadere in **[1e−4; 1]**; fuori range si rivede la proposta, senza clamp o griglia.
5. Pubblicare λ, norme, coseni, supporto e pin. Reset completo di modello, optimizer, teacher, memoria, contatori e RNG prima dei tre training.

La testa posa parte a zero: la norma iniziale della base riflette soprattutto le ancore. La regola non garantisce gradienti al 10% durante il run, né dopo AdamW/clipping. Non si usa l'E4 locale dell'ultimo micro-batch come surrogato del gradiente globale.

CN e CR: primi 256 update a peso zero; successivi 256 rampa lineare fino al rispettivo λ; dal completamento dell'update 512 peso fisso. C0 sempre zero. Stesse tempistiche nei tre bracci.

Audit train e collaudo devono riportare almeno: copertura NCE complessiva/per dataset, negativi ammissibili e loro età; finestre posa/ancore/utenti/registrazioni per batch globale; quote senza termini attivi, duplicati e overlap; gradienti globali, coseni e clipping. Si conserva per NCE il gate proposto del 90% complessivo e 80% per dataset dopo il riempimento, con almeno 32 query osservate per dataset a quota positiva. Per RnC: almeno **90% dei batch globali** dopo il riempimento con una o più finestre contate; nessuna promessa che il 90% degli esempi del mix abbia posa. Il gate di calibrazione congiunta sopra resta distinto.

Queste soglie si verificano sul train prima del rilascio; un fallimento richiede diagnosi e proposta aggiornata, non una scelta automatica di coda/stride/peso. Durante il training si registrano blocchi di 1.024 update, evitando che un tratto iniziale favorevole nasconda una perdita successiva di supporto.

## 5. Che cosa confrontiamo

Teacher congelato **prima del proiettore**, stesse finestre, split e ricette dei lettori storici. Posa: ridge istantanea sulla media K64; classificazione: pool768 e logistica convergente senza nuova PCA o ricerca di pooling. Nessuna classificazione calcolata dalla proiezione contrastiva.

| Dose | Letture | Ruolo |
| --- | --- | --- |
| 5.000, tutti i bracci | Posa: E_dyn, correlazione, varianza predetta/target, MAE e separazione livello/dinamica | Descrittivo; non seleziona ricetta o continuazione |
| 19.618, tutti i bracci | **E_dyn, P1, P2** | Tre endpoint prespecificati del confronto |
| 19.618 | Gruppi posa tutti/statici, cinque dataset separati, profili di gradienti e costo | Descrittivi; niente promozione sul solo dataset migliore |

P1 usa pesi DB2/DB3/DB6 **8/13, 3/13, 2/13**; P2 EPN612/UCI **1/2, 1/2**. Balanced accuracy da conteggi aggregati per classe come la ricetta storica, non media non pesata delle accuracy per soggetto. E_dyn è somma degli errori quadratici centrati divisa per somma della variabilità target sulle finestre dinamiche, non media dei rapporti per utente.

Tre coppie prespecificate: **CN−C0, CR−C0, CN−CR**. Questo separa il miglioramento rispetto alla base dalla differenza diretta fra approcci. Non basta che CN batta CR se entrambi peggiorano rispetto a C0.

## 6. Regola di lettura senza vincitore obbligatorio

Tutti gli effetti sono orientati come beneficio positivo: P1/P2 in punti percentuali, posa come **riduzione relativa percentuale di E_dyn**. Per ogni coppia e ogni endpoint si riportano stima, IC95 descrittivo e IC simultaneo nominale **99,4444%** (Bonferroni: nove confronti, α famiglia 0,05). È una scelta conservativa fissata prima dei risultati, non un aggiustamento dopo aver visto gli intervalli.

Bootstrap appaiato **100.000 repliche, seme 20261116**: soggetti entro ogni dataset, stessi draw per i tre bracci; utenti per la posa. Ordine alfabetico dei dataset. Classi assenti in un draw: macro-recall sulle classi presenti, tutti i draw conservati, come la lettura storica. Le code 0,2778% degli intervalli simultanei motivano 100k invece di 10k; non aumentano il numero di partecipanti.

| Endpoint | Guadagno pratico minimo, oltre a IC simultaneo sopra zero | Margine di non inferiorità |
| --- | ---: | ---: |
| E_dyn | Riduzione relativa ≥2% | Aumento relativo <2% |
| P1 | ≥+0,5 pp | Perdita <1 pp |
| P2 | ≥+0,5 pp | Perdita <1 pp |

Un approccio è **promettente per conferma** rispetto a un altro se migliora utilmente almeno un endpoint e sostiene non inferiorità su tutti e tre, usando gli stessi intervalli simultanei. Per preferire CN o CR occorre soddisfare questa regola **sia contro C0 sia nel confronto diretto**. Le letture inverse del confronto diretto derivano dagli stessi intervalli/draw, non costituiscono tre ipotesi aggiuntive.

Altrimenti si riporta il profilo: tradeoff, risultato irrisolto, beneficio comune senza differenza diretta, o nessun vantaggio. **Nessuna somma arbitraria di posa e classificazione**, nessun vincitore forzato. Le soglie restano criteri di screening, non prova di equivalenza o adozione definitiva.

Il comparatore implementato è `scripts/contrastive_compare.py`. Legge OOF e statistiche sufficienti di posa pinnate, verifica l'appaiamento completo, ricostruisce i conteggi e controlla i 420 record dei fit delle sonde (3 bracci × 5 dataset × 4 fold × 7 fit). Fit irrisolti, selezione ridge al bordo irrisolta o bootstrap diverso da quello congelato mantengono le stime ma sospendono il verdetto. Il comparatore non sostituisce i controlli di provenienza e leakage degli estrattori/lettori: questi devono essere verificati prima di produrre il manifest finale.

Limiti: un solo seme di pretraining per braccio; IC condizionati ai checkpoint e ai lettori, **non** variabilità fra training. Sviluppo già osservato, nessun test finale intatto. La posa è già supervisionata in tutti i bracci: un miglioramento non prova trasferimento a una regressione nuova. Se il risultato merita, conferma e nuovo task si propongono separatamente.

## 7. Verifiche e costi prima della release

Test da includere nella review: RnC senza repulsione dipendente dalla finestra, pareggi e maschere, temperature ammesse, oracolo numerico; NCE e filtri di identità/overlap; conteggi/zero supporto; gradienti distribuiti a uno/quattro rank anche con rank vuota; nessun gradiente teacher; FIFO identica e ripresa; inizializzazione/RNG condivisi e peso zero compatibile; nessun aggiornamento della memoria in eval; metriche appaiate e correzione delle nove letture.

Il collaudo CUDA successivo alla review deve includere calibrazione, almeno 64 update tecnici per CN e CR con il rispettivo termine a peso attivo, oltre al controllo C0, parità iniziale, timing, memoria, ripresa e dose; i suoi pesi vengono scartati. Gli endpoint dei training partono soltanto dopo lettura del collaudo e preventivo congelato.

| Voce | Riferimento preliminare | Prima della sottomissione |
| --- | --- | --- |
| Tre training a 19.618 | 3 × 16,9444 = **50,8333 GPU-ore** storiche, prima dell'overhead comune | Misurare costo del nuovo percorso nei tre bracci, pubblicare stime e tetti separati |
| Collaudo/calibrazione | Da misurare/preventivare sul percorso completo | Piano concreto, comandi, dati e tetto pubblicati |
| Dodici estrazioni | 3 bracci × (posa 5k, posa 19.618, P1, P2) | Endpoint completi, pin e benchmark di inferenza |
| Ridge, sonde, comparatore | 6 letture di posa, 6 aggregati P1/P2; 420 fit di classificazione | Benchmark CPU e tetti, inclusi 100k bootstrap |
| Checkout e conservazione | Voci separate | Pin e consuntivi come nelle campagne precedenti |

La tabella è un ordine di grandezza, **non un budget operativo già approvato**. Non promettere 50,83 GPU-ore dopo aver aggiunto il nuovo calcolo a tutti i bracci. I tre training sono indipendenti e possono occupare nodi separati da quattro GPU; non cambiare GPU per singolo modello, batch o dose per accelerare il confronto.

Sequenza: design/implementazione e test locali → review Claude dell'intera consegna → correzioni/convergenza → collaudo e preventivo esecutivo → release congelata → training/valutazioni. Nessun monitor in background in questa preparazione.

## 8. Strumenti consegnati e confini della verifica

`model/rank_contrast.py` corregge il prototipo; `model/contrastive.py` contiene le primitive NCE. `training/contrastive_memory.py` gestisce identità, memoria e raccolta distribuita; `training/contrastive_objectives.py` collega i due termini al trainer. `scripts/train_jepa.py --contrastive-config` attiva il percorso; assenza del flag conserva quello storico. La compatibilità del totale è inclusa in `arms_collaudo.py`, ma i suoi vecchi criteri scientifici non diventano i criteri della nuova campagna.

`scripts/contrastive_audit.py` riproduce in sola CPU i quattro rank e i sei worker logici per rank, senza encoder; usa segnali/maschere/target train dal loader, non stima la validità delle proiezioni. `scripts/contrastive_prepare.py` prepara configurazioni in review e legge la calibrazione congelata; `scripts/contrastive_compare.py` produce il confronto finale. I config scientifici contengono gli stessi due λ calibrati: cambia soltanto `arm`, che seleziona il peso effettivo.

Il pannello `fixed_eval` resta limitato ad ancore/posa/JEPA, esplicitamente etichettato come obiettivo di base; non aggiorna la memoria. Nel training i gradienti globali si registrano ogni 500 update e ai punti 256, 512, 1.962, 5.000 e 19.618. La copertura viene verificata su blocchi completi da 1.024 update dopo il riempimento; il blocco parziale si riporta separatamente. La calibrazione usa il proprio blocco train di 256 batch e deve superare i criteri prima del rilascio.

Le verifiche locali e i loro limiti sono nel file `verification.json` della consegna. I nuovi test distribuiti a più processi sono scritti ma richiedono esecuzione fuori dalla sandbox del Mac; non sono presentati come superati in assenza del loro esito. CUDA/NCCL, costo reale, copertura sul loader reale, estrazioni e fit scientifici restano passi successivi alla review. Questo è un pacchetto di implementazione e verifica, non un checkout esecutivo già rilasciato.

`scripts/contrastive_run.py` verifica le ricette e prepara l'ingresso dei collaudi con pin di codice, configurazione e dati. La fase scientifica `training` è deliberatamente rifiutata dall'entry point di campagna: prima del deploy vanno collegati i rapporti misurati dei tre benchmark a 576 passi e il validatore dei preventivi/tetti. Cambiare soltanto lo stato del piano non abilita il lancio. Il trainer sottostante implementa già tutti e tre i bracci; questa limitazione riguarda il rilascio operativo, successivo alla review e alle misure. Il CLI dell'audit accetta soltanto il config di calibrazione congelato e verifica gli hash degli ingressi prima e dopo il replay.
