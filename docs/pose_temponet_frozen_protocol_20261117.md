# TEMPONet come lettore della posa del FM — proposta

17/11/2026. **Stato: REVIEW_REQUIRED. Nessun codice runtime, merge, deploy o job autorizzato da questa proposta.** Ramo `codex/pose-temponet-proposal-20261117`, derivato dalla consegna consolidamento `cfa6462`. Risponde al prompt consegnato da Simone e alla richiesta «leggilo e dimmi come impostare il confronto con la testa attuale usando l’implementazione che abbiamo già».

Questo è il **livello A del prompt TEMPONet: encoder congelato**, distinto dal protocollo A sull'efficienza dei dati etichettati, messo da parte per questa prova. La proposta non modifica le conclusioni firmate del lungo o i confronti delle altre linee.

## 1. Domanda e scelta iniziale

**Una testa temporale non lineare legge meglio la dinamica della posa dalle rappresentazioni già disponibili?** Partire dal teacher T100-G 30M a **49.046 passi**, riferimento esplorativo per la posa firmato da Simone. Si adattano soltanto teste sulle cache; nessun aggiornamento del FM e nessuna estrazione nuova. Il 300M, le altre dosi e la loss contrastiva restano fuori.

Concordo con il controllo a encoder congelato proposto nel prompt. Non concordo con una sua lettura troppo forte: una testa migliore non dimostrerebbe il valore generale del pretraining o il trasferimento a una regressione nuova. La posa è già supervisionata nel FM. Un esito nullo non dimostrerebbe assenza di informazione, né inutilità di una testa temporale durante l'addestramento congiunto.

Il backbone è bidirezionale e ciascun suo latente ha già accesso alla finestra di quattro secondi. La TCN aggiunge una lettura temporale strutturata, non il primo accesso al tempo; non rende causale il sistema.

## 2. Uso della TEMPONet esistente

Fonte locale verificata: `src/wearusfm/model/temponet.py` nel ramo Claude, commit `8917e2a911287d8939e44d210adf566e4c8e8561` (la punta aggiunge il prompt, non un nuovo modello). Provenienza: [Zanghieri et al., COINS 2021](https://iris.polito.it/handle/11583/2978570). Quel lavoro e il codice esistente sono il riferimento della TEMPONet su segnale grezzo, non una validazione di questa nuova testa.

Il modello risale a `60b35bd` e non è cambiato. Verificata identità byte per byte di modello e `scripts/temponet_emg2pose.py` fra `0c19c4bc32a96e3b72d15a09770d204a747fa13b` (code_sha dei report train/eval) e `8917e2a`. Launcher raw: `scripts/slurm/temponet_emg2pose.sbatch`; test esistenti `tests/cpu/test_temponet.py` e `tests/cpu/test_temponet_emg2pose.py`, non rieseguiti come se coprissero la nuova testa. [Verifica di provenienza](../results/infra/pose_temponet_proposal_20261117/raw_source_identity.json). Nessun merge o modifica ai file del ramo Claude.

La classe esistente riceve `(B,C,T)` con lunghezza fissa e restituisce una posa per finestra. Stride, tre pooling e flatten impediscono di usarla invariata per una posa a ogni patch. I suoi 128 ms su EMG a 2 kHz non corrispondono a 256 latenti a 40 Hz. La prima convoluzione 384→16 introdurrebbe inoltre una compressione poco adatta a questa domanda.

**Correzione tecnica al prompt:** il helper `receptive_field()` restituisce 378 passi per un elemento finale convoluzionale, non per tutta l'uscita dopo flatten. Con T=160 il flatten combina due elementi distanti 64 passi: unione nominale di 442, prima dei limiti dell'input e del padding. In training la BatchNorm introduce ulteriori dipendenze dalle statistiche di batch/tempo. In entrambi i casi il trapianto diretto non ha il significato temporale dei 128 ms raw.

Propongo quindi **una TCN compatta ispirata a TEMPONet**, mantenendo convoluzioni temporali dilatate e ReLU ma adattando esplicitamente risoluzione, normalizzazione e uscita. Non la chiameremo replica TEMPONet. Non si riusano i pesi del modello raw: gli ingressi 16 canali grezzi e 384 feature non sono compatibili. I blocchi privati `_TemporalConv`, `_StridedConv`, `_Dense` fissano BN/pooling, quindi non sono riutilizzabili invariati: si adatta il codice in un modulo sperimentale separato, con attribuzione e test, senza modificare la replica di Claude.

## 3. Specifica unica della testa e controlli

Input: media dei 64 slot, sequenza `(B,160,384)`. Nessuna media sul tempo. Cast FP16→FP32, **LayerNorm affine per patch, eps=1e−5**, uguale nelle nuove teste. La testa attuale ha proprio LN affine: mantenendola, L0 usa la stessa `PoseHead`, rifittata da zero sulle cache.

TCN, ordine completo:

`LN384 → Conv1×1(384,64) → ReLU → Conv k3,d1(64,64) → ReLU → Conv k3,d2(64,64) → ReLU → Conv k3,d3(64,64) → ReLU → Dropout0.1 → Conv1×1(64,32) → ReLU → Dropout0.1 → Conv1×1(32,20)`.

Tutte le convoluzioni hanno bias, stride 1, dilatazioni 1 per i kernel 1. Padding simmetrico 1/2/3 sui tre kernel 3; nessun pooling, BatchNorm, normalizzazione temporale, residuale o filtro delle predizioni. Ultimo strato lineare con pesi e bias inizializzati a zero, come `PoseHead`; interni con inizializzazione Torch standard e versioni pinnate. LN inizia con gamma=1, beta=0. Non azzerare tutti gli strati.

Campo ricettivo locale: `1 + 2*(1+2+3) = 13` patch, tutti gli offset −6…+6. **±150 ms tra centri**, 300 ms tra centri estremi, 325 ms contando i 13 bin da 25 ms. È il contesto aggiunto dalla testa, non il campo ricettivo dell'intero FM. I centri 6…153 di una finestra da 160 patch non dipendono dal padding: esattamente il crop6 storico.

| Braccio nuovo | Ingresso | Lettore | Parametri | Domanda |
|---|---|---|---:|---|
| L0 | `lat`, dopo backbone | `PoseHead`: LN384→Linear384,20 | 8.468 | Controllo principale con la testa attuale, rifittata alla pari |
| MLP | `lat` | LN→384→96→96→96→64→32→20, ReLU sui cinque hidden | 65.300 | Quanto basta aumentare la capacità non lineare per istante? |
| L13 | `lat` | LN→Conv384,20,k13,pad6 | 100.628 | Quanto basta una lettura lineare con lo stesso contesto? |
| TCN-post | `lat` | TCN definita sopra | 65.204 | Ipotesi principale: lettura temporale non lineare |
| TCN-pre | `pre`, prima backbone | Stessa identica TCN | 65.204 | Controllo descrittivo dell'utilità della vista dopo il backbone |

MLP e TCN hanno entrambi **64.128 pesi nelle trasformazioni**, sei trasformazioni, cinque ReLU e 10.260.480 MAC per finestra (esclusi bias/LN/ReLU). La differenza di 96 parametri riguarda i bias. Nella MLP dropout 0,1 dopo i hidden da 64 e 32, negli stessi punti della TCN. Questo rende il controllo di capacità preciso senza una ricerca di larghezze.

L13 è una mappa affine delle feature normalizzate, a rango non limitato da un collo di bottiglia da 16 canali. Non è a capacità pari né una ridge chiusa: LN apprendibile e AdamW cambiano il fitting. Perciò un confronto TCN/L13 riguarda questa ricetta e la sua parametrizzazione, non un teorema sulla non linearità.

**Due riferimenti storici, senza nuovo fitting:** ridge post 49k e testa già allenata del teacher, con predizioni pinnate sullo stesso supporto. Sono il confronto pratico con ciò che già funziona, mentre TCN-post/L0 è il confronto controllato di nuove teste. La ridge storica minimizza SSE aggregata sugli elementi validi, la `PoseHead` del training MSE media per finestra: non usare la ridge da sola per attribuire il guadagno all'architettura. La testa storica ha visto 158 utenti nel pretraining mascherato, con aggiornamenti EMA; le nuove teste vedono 40 utenti e latenti teacher non mascherati.

## 4. Dati, maschere e selezione

Cache del lungo già copiate su WORK: **1.907 finestre / 40 utenti train** e **18.548 finestre / 15 utenti val**, con `pre`, `lat`, target, maschere, lunghezze, predizioni e statistiche. Hash, dimensioni e riferimenti sono in [proposal.json](../results/infra/pose_temponet_proposal_20261117/proposal.json). Riutilizzo in loco, senza duplicare 4,720 GiB di input. Prima dell'esecuzione l'operatore deve verificarne l'integrità attuale: qui sono verificate soltanto le prove archiviate.

Tutti i bracci nuovi hanno le stesse finestre, etichette e supporti. Si conservano la media/deviazione standard della posa calcolate sul train158 e gli indici originali. Nessuna normalizzazione delle feature stimata sui 15 val, nessun nuovo lag, EMA d'uscita o selezione di giunti. Nel percorso delle feature non entra la disponibilità delle etichette vicine.

Supporto comune: target valido del giunto centrale e `6 ≤ t` e `t+6 < N_i`, con patch di contesto realmente esistenti secondo lunghezze/`key_time_valid`. Il crop dipende dalla geometria EMG; la maschera target serve alla loss/metrica. Finestre senza centri supervisionati si dichiarano ed escludono in tutti i bracci allo stesso modo, mai in base alle predizioni.

Loss per tutti i nuovi fit: media dell'errore quadratico standardizzato sugli elementi validi di ciascuna finestra, poi media sulle finestre supportate. Si allena su tutte le finestre, non soltanto dinamiche; nessun termine aggiuntivo sulla velocità o correlazione. Il bersaglio resta quello originale per patch a ritardo zero.

Ricetta proposta, unica:

- AdamW, beta=(0,9;0,999), eps=1e−8, weight decay 0,01 sulle sole matrici/kernel, esclusi bias e LN; clipping norma 1; FP32, autocast e TF32 spenti.
- Batch 32 finestre, ultimo batch conservato, shuffle per epoca da un generatore dedicato, indipendente da inizializzazione/dropout e identico fra bracci a parità di fold/seme; LR `{1e−4,1e−3}`, costante, senza warmup; massimo **50 epoche** per candidato. Nessuna ricerca di dropout, RF, larghezze o altra loss.
- Due fold da 20 utenti ciascuno, alternando gli ID train ordinati come nella ridge storica. Seme 0 per ogni fit di selezione, pesi iniziali nuovi. I 15 utenti val restano chiusi.
- Per ogni testa scegliere insieme LR ed epoca minimizzando la media delle MSE per finestra su tutti gli utenti interni held dei due fold: somma delle perdite per finestra divisa per il numero complessivo di finestre supportate, non media non pesata dei fold. Pareggi: epoca precedente, poi LR minore.
- Rifit da zero su tutti i 40 utenti con la coppia scelta, semi **0,1,2** e numero di epoche fisso. Nessuna selezione del seme o arresto sulla val. LR costante: il rifit riproduce il prefisso della ricetta, senza riscalare un coseno sull'epoca selezionata.

**20 fit di selezione + 15 rifit = 35 fit di teste**, non 35 pretraining del FM. Prima di aprire le predizioni val si congelano scelte, pesi, semi e hash di tutti i bracci. Una sola valutazione finale comune sui 15 utenti; si possono riprodurre i report dalle predizioni congelate, non usare l'esito per scegliere una variante.

I fold interni sono nuovi solo per la testa: encoder e statistiche di posa hanno già visto quei 40 utenti. È selezione del lettore condizionata a quel FM, non validazione dell'intero sistema su utenti mai visti. I 15 val sono sviluppo già consultato; il test ufficiale è già stato letto dalla baseline raw e qui non si usa. **Non c'è una conferma indipendente incontaminata in questa prova.**

Se il migliore è all'ultima epoca o al bordo LR si riporta; non si estende automaticamente la ricerca. NaN, divergenza, OOM o fit incompleto fermano la fase, con conservazione degli artefatti e diagnosi distinta; nessun rimpiazzo del braccio, riduzione dei semi o retry cieco.

## 5. Metriche, famiglia e decisione proposta

Primario: **riduzione relativa di E_dyn sulle finestre dinamiche, TCN-post contro L0**. Minore E_dyn è meglio. Si conserva la soglia storica `pose_var ≥ 0,2574231028556824` sulla varianza originale dell'intera finestra, senza ricalcolarla sul crop o sui risultati.

E_dyn: centrare separatamente previsione e target sui medesimi tempi validi di ogni coppia finestra/giunto, almeno due punti; sommare SSE centrata e somma dei quadrati centrati del target, poi fare il rapporto. Non fare la media dei rapporti per finestra o per utente. I tre semi sono riassunti mediando i rispettivi accumulatori per utente prima del rapporto e del bootstrap: **prestazione media di tre teste, non ensemble di predizioni**. Pubblicare anche i tre esiti separati.

Famiglia fissa di cinque contrasti, sempre TCN-post meno il riferimento in termini di miglioramento `100*(E_ref−E_TCN)/E_ref`:

1. L0: primario architetturale.
2. MLP: informazione utile nella lettura temporale rispetto alla non linearità per istante a capacità comparabile.
3. L13: risultato rispetto al controllo lineare temporale.
4. Ridge post storica: utilità rispetto alla miglior lettura lineare disponibile.
5. Testa storica allenata: utilità rispetto all'uscita già disponibile del FM.

Bootstrap appaiato sugli stessi 15 utenti, 2.000 repliche, seme 20261102. Stessi sorteggi fra bracci, semi e riferimenti. **IC percentile 99% per ciascuno dei cinque contrasti**, Bonferroni per una copertura simultanea nominale 95%. È un'approssimazione esplorativa con 15 unità e circa dieci repliche per coda: non una garanzia esatta. Gli IC restano condizionati ai tre fit e a un solo encoder preallenato, non includono la variabilità del pretraining.

Regola a due lati, proposta da congelare prima dei fit:

- **Beneficio pratico risolto:** stima della riduzione ≥2% e limite inferiore aggiustato >0. Non significa che l'intero IC superi 2%.
- **Deterioramento pratico:** stima ≤−2% e limite superiore aggiustato <0.
- **Beneficio di almeno 2% non sostenuto dall'intervallo:** limite superiore aggiustato <2%, riportando comunque direzione e ampiezza. Non chiamarlo equivalenza.
- **Inconclusivo:** l'IC include effetti di almeno 2% ma il criterio di beneficio non passa. Il pilot si chiude senza promozione; nessuna ricerca automatica per ottenere un sì.

Si considera la TCN candidata a **migliore lettore della dinamica centrata** di questo checkpoint solo se passa il primario **e** i due confronti storici (4 e 5). E_dyn rimuove il livello medio: questo non basta a raccomandare la sostituzione operativa della testa se MAE o livello peggiorano. Quel compromesso va discusso prima dell’adozione; nessuna non-inferiorità del livello è implicita e nessuna nuova soglia si sceglie dopo gli esiti. Se batte L0 ma non i riferimenti già disponibili, l'architettura è promettente nel confronto controllato ma non offre ancora un miglior lettore operativo. Per attribuire utilità specifica al contesto servono anche TCN/MLP; per sostenere il vantaggio della ricetta non lineare temporale sul controllo lineare, TCN/L13. Si pubblicano tutti e cinque, senza scegliere il confronto favorevole.

**TCN-post/TCN-pre è descrittivo**, IC 95% senza cancello inferenziale. Stessa testa e dati, ma le due viste hanno diverso contesto già incorporato: misura accessibilità dopo il backbone, non valore del pretraining o origine dell'informazione. Nessun sesto verdetto automatico.

Secondari: E_dyn per tutti/statiche; correlazione centrata, rapporto delle varianze, errore del livello e MAE in radianti e gradi con statistiche originali. Per correlazione, rapporto delle varianze, livello e MAE si pubblicano i valori per seme e la loro media aritmetica, senza chiamare media delle correlazioni quella ricavata da momenti aggregati; nessun ensemble. Gli IC aggiustati della famiglia riguardano soltanto E_dyn. Per giunto e pollice secondo nomi dello schema, non scegliendo i giunti che si muovono di più. Nessuna misura selettiva su tratti/utenti del test già osservati. Per questo pilot la misura di inseguimento resta quella centrata sulle finestre di quattro secondi, con correlazione/varianza affiancate; una nuova metrica di tratto richiederebbe una proposta distinta.

## 6. Controlli rinviati e confronto raw

Non aggiungo ora encoder random, TCN raw da zero o nuova inferenza TEMPONet raw: non sono necessari per decidere se cambiare il **lettore** su un encoder fissato. Restano necessari a una futura affermazione sul beneficio del pretraining rispetto a nessun encoder appreso; la vista `pre` non li sostituisce. Non si introduce una deroga silenziosa alle baseline v10: la conclusione qui è più ristretta.

Per un futuro confronto addestrato da zero: stessi 40 utenti, finestre/tempi, loss, selezione e metriche per una parità dell'adattamento, dichiarando che il FM ha comunque ricevuto supervisione da 158 utenti. Un raw con 158 utenti sarebbe un confronto fra pipeline con diversa dose di etichette, da riportare separatamente. Nessuna nuova estrazione sui 158 utenti in questo livello A.

TEMPONet raw già allenata può diventare un riferimento descrittivo, senza ritarare EMA o lag e senza pretendere parità di loss/dati/contesto. **Non basta mettere le MAE pubblicate accanto a E_dyn.** Per confrontare esattamente i target FM, `window_pose` assegna alle patch i blocchi da dieci campioni a 2 kHz secondo i loro centri e ne media gli angoli. Occorre produrre le previsioni raw sui medesimi campioni nativi, mediare ogni blocco, poi usare gli stessi blocchi validi e la stessa media per patch, standardizzazione e crop6. Cinque predizioni puntuali a 200 Hz per patch sono un'approssimazione dichiarata, non lo stesso bersaglio temporale.

Il passato raw `[q−255,q]` rientra nel margine di 150 ms per i centri ammessi, da verificare con indici assoluti e blocchi che attraversano il bordo. Le etichette non entrano nell'input raw; maschere solo per aggregazione/valutazione. L'inferenza esatta richiede circa dieci volte gli endpoint della lettura ufficiale a 200 Hz: costo separato, nessun invio in questa proposta.

## 7. Livello B: sostituzione nel pretraining, decisione successiva

Non lo lancerei insieme ad A. Se la TCN migliora il lettore, il passo successivo può verificare se usarla durante il pretraining migliora anche le rappresentazioni. Se A è nullo, B resta logicamente possibile ma richiede una motivazione nuova: non è il seguito automatico.

B deve fissare inizializzazione del resto del FM (testa sotto `fork_rng`), dati/ordine, maschere, loss e pesi, dose/schedule, EMA, e un controllo `PoseHead` appaiato. Una testa temporale può ricostruire anche tratti mascherati: il livello A, teacher non mascherato, non misura questo cambiamento del compito. Valutare sia ciascuna testa propria sia lettori comuni rifittati, compreso quello lineare; una loss più bassa non basta per affermare latenti migliori.

Se B fosse a 19.618, A andrebbe letto anche a quella dose con protocollo distinto prima di B. Il T100-G storico non basta come controllo appaiato; il C0 della linea 3 è riusabile solo con identità dimostrata di inizializzazione, ricetta, ordine e percorso di calcolo, senza cambiare i suoi nove contrasti o checkpoint. Altrimenti due training nuovi: controllo e candidato. **Ordine di grandezza storico 2×16,94 = 33,88 GPUh**, più estrazioni/letture/overhead; non preventivo né budget di B.

La testa proposta è circa 0,23% del 30M, ma quasi otto volte `PoseHead`: la percentuale piccola non dimostra che non assorba il compito. Il v10 §6.5 va affrontato esplicitamente se la testa entra nel pretraining. Nel livello A non si cambia la testa supervisionata del FM e non serve una deroga a quel vincolo; ritardo zero e backbone non causale restano quelli firmati, nessun nuovo lead time appreso. Eventuali cambi di B passano da un emendamento, separati dalla prova del lettore.

## 8. Codice, ruoli, prove prima dell'esecuzione

Proposta di continuità: **GPT consolidamento autore del protocollo e del codice; Claude revisore e unico operatore** per deployment, invii, log e raccolta, con submitter/ledger comuni. L'affidamento della nuova attività e il budget li conferma Simone; non viene chiesto di nuovo il consenso per catene già approvate. Ora si consegnano soltanto documenti.

Implementazione futura in nuovi file di esperimento sotto `scripts/` e relativi test CPU, su ramo distinto dalla linea 3. Nessun merge del ramo TEMPONet, nessuna modifica ai suoi file; nessun cambiamento a `src/wearusfm`, `FMConfig`, `PoseHead`, config storici o codice congelato delle altre linee. La classe originale rimane il riferimento di provenienza; l'adattamento non viene venduto come wrapper invariato. I digest delle release altrui restano quelli dei loro checkout. Anche gli input riusati richiedono pin dei file e bridge esplicito SCRATCH→WORK, non hash riscritti.

Test da preparare dopo l'accordo sul disegno:

- forme per patch, conteggi dei parametri e stessa capacità MLP/TCN;
- RF13 senza buchi, crop6 equivalente al calcolo senza padding, MLP/L0 dipendenti solo dal centro delle feature;
- maschere centrali, lunghezze variabili, nessun uso delle etichette vicine o leakage fra finestre/fold;
- stesso input, target, statistiche e riduzione della loss in tutti i bracci;
- zero iniziale e gradienti finiti dopo l'avvio: al primo backward l'uscita a zero blocca i gradienti interni, comportamento atteso, non un test da falsare;
- scelta LR/epoca sui soli fold train, fit mancanti bloccanti, semi fissati, apertura val successiva al manifest dei modelli;
- metriche e media fra semi da accumulatori confrontate con un oracolo semplice; cinque contrasti, quantili e supporti appaiati;
- cache/encoder invariati, parametri originali e piano sorgenti invariati, rifiuto di pin o output incompatibili.

Nessun test runtime nuovo è dichiarato passato in questa consegna: sono stati verificati codice esistente, conti di parametri/RF e disegno, senza implementare o addestrare le nuove teste.

## 9. Risorse e conservazione: tetto proposto, da misurare

**Tetto proposto del solo livello A: 4 GPUh, 12 coreh CPU-only, 44 coreh allocate totali.** Il costo delle CPU dei job GPU è incluso nelle 44 (8 CPU per GPU), non nelle 12. Ore locali massime condizionate al billing corrispondente: 44. Non è una misura del runtime né un budget autorizzato.

| Fase | Allocazione proposta | Limite cumulativo | GPUh | Coreh CPU-only |
|---|---|---:|---:|---:|
| Checkout, manifest/hash e supporti | 2 CPU | 15 min | 0 | 0,5 |
| Benchmark CPU train-only | 8 CPU | 15 min | 0 | 2 |
| Benchmark GPU train-only | 1 GPU + 8 CPU | 15 min | 0,25 | — |
| Selezione + rifit delle cinque teste | 1 GPU + 8 CPU | 3 h | 3 | — |
| Inferenza val bloccata sui 15 modelli | 1 GPU + 8 CPU | 30 min | 0,5 | — |
| Riferimenti, metriche e bootstrap | 8 CPU | 1 h | 0 | 8 |
| Raccolta e conservazione | 2 CPU | 15 min | 0 | 0,5 |
| Riserva entro il totale | stesse risorse | — | 0,25 | 1 |
| Totale proposto | | | **4** | **12** |

RAM proposta: 24 GiB CPU seriale (entro i 30.800 MB del limite riportato), 32 GiB per job GPU; VRAM da misurare. Il costo resta quello delle risorse effettivamente allocate: se CPU/billing/minimi della partizione differiscono, ricalcolare prima di impegnare il tetto.

Il benchmark non usa la val scientifica: train soltanto, con una porzione train come sostituto del costo della validation. Misura caricamento, epoca forward/backward per ogni architettura e dimensione fold/full, valutazione interna, salvataggio, RSS/VRAM. Warmup CUDA fuori dal cronometro dei batch ma incluso nel costo totale del job. Nessuna metrica held durante il benchmark.

Gate prima di qualunque fit: raddoppiare la proiezione dell'intera matrice usando **50 epoche anche per tutti i rifit**, includendo 20 selezioni, 15 rifit, validation, I/O e avvii; deve rientrare nelle 3 GPUh della fase. Anche la proiezione di inferenza scalata a 18.548 finestre × 15 modelli, raddoppiata e con I/O, deve rientrare in 0,5 GPUh, CPU e memoria nei rispettivi tetti. Se un gate fallisce, stop prima dei fit: niente diminuzione post hoc di epoche, semi, bracci o precisione. Riserva non autorizza retry automatici. Nessun tempo del vecchio ridge o della TEMPONet raw viene presentato come benchmark delle nuove teste.

Output proposti: `$WORK/wearusfm_runs/results/pose_temponet_20261117/`, **massimo 8 GiB persistenti**, senza copie degli input. Le 15 predizioni val FP32 non compresse occupano circa 3,32 GiB a 160 patch; si conservano anche pesi di tutti i rifit, curve/selezione train, supporti, predizioni/riferimenti, ambiente, ricevute e manifest. Report e manifest sul Mac e in Git, NPZ/PT fuori Git. NPZ compressi: decomprimere una volta le sole viste necessarie entro la RAM della fase, poi fare il cast FP16→FP32 a lotti. Non si presume mmap efficiente dei membri compressi; il benchmark include il loader reale. Nessuna nuova copia persistente della cache per comodità.

Prima dell'invio Claude verificherà quota attuale e saldo: i 933,1 G/1 T del 16/11 e il residuo del 09/11 sono storici e non attestano disponibilità oggi. I payload A sono su WORK e non dipendono dal rinfresco SCRATCH del 03/12; raw/random o B avrebbero dipendenze diverse. Il fermo 23/11 08:00–25/11 08:00 è quello riportato nel prompt, da ricontrollare operativamente: nessuna promessa di completamento prima del fermo. Nessuna lettura operativa del cluster eseguita da GPT per questa proposta.

## 10. Decisioni per Simone, dopo la review

| Decisione | Proposta concreta |
|---|---|
| Livello | Solo lettore su encoder congelato; pretraining B separato |
| Checkpoint | T100-G 30M teacher 49.046, niente altre dosi/300M ora |
| Bracci | L0, MLP, L13, TCN-post, TCN-pre; due riferimenti storici riusati |
| Ricetta e lettura | Sezioni 3–5: RF13, stessi 40 utenti, MSE per finestra, 2 LR, 50 epoche max, 3 semi, cinque IC aggiustati |
| Budget | Tetti 4 GPUh + 12 coreh CPU-only, totale 44 coreh; gate misurato sull'intera matrice |
| Autore / revisore / operatore | GPT codice; Claude review e unico operatore, senza duplicazioni |

Queste sono proposte da approvare, non istruzioni di lancio. Dopo review e decisione sul disegno: codice e test, review reciproca, benchmark entro il tetto concordato e gate globale, poi esecuzione da Claude secondo il piano concreto. Nessun esperimento della linea 2 viene avviato da GPT con questa consegna.
