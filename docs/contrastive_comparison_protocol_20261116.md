# InfoNCE e Rank-N-Contrast: confronto controllato v2 — 16/11/2026

**Protocollo `contrastive-comparison-20261116-v2`. Decisioni scientifiche approvate da Simone; correzioni e rilascio esecutivo ancora da verificare.** Sono approvati l'esenzione di copertura basata sul manifest (D-1/A), il riferimento dei pesi al tratto storico attivo (D-3/2) e la famiglia decisionale E_dyn/P1/EPN612 con P2 e UCI descrittivi (D-2). L'autorizzazione ai lanci dopo la convergenza resta valida; questo documento e il `plan.json` in stato `REVIEW_REQUIRED` non sono da soli una release esecutiva. Comandi, risorse, criteri tecnici e ricevute sono nel [piano del collaudo](contrastive_preflight_execution_20261116.md).

La dose scientifica termina a 19.618 update: nessuna estensione automatica a 49.046, nessun braccio placebo aggiunto, nessun cambio di ricetta a run iniziato o selezione della continuazione dai risultati a 5.000.

## 1. Domanda, ricetta e tre bracci

**A pari inizializzazione, stream e dose, quale ricetta aggiuntiva offre un beneficio utile sui compiti decisionali, conservando gli altri entro i margini fissati?** InfoNCE discrimina istanze del mix train; RnC ordina rappresentazioni degli istanti con posa secondo gli angoli. La MSE posa è già presente in tutti i bracci: CR non riceve dati etichettati esclusivi.

Base **T100-G 30M**: finestra 100 ms, hop 25 ms, overlap 75 ms, GELU, senza ramo log-quadratico. Questa scelta usa risultati di sviluppo già osservati. Riferimento: `results/infra/tokenizer_execution_20261103/19618/T100-G/config.json`, SHA256 `f9d0318462fd30a5592087ef72c63d3036a88ff1d07fc0dad8624e36253c2cfa`. Si riparte dall'inizializzazione, non da un checkpoint storico.

| Braccio | Loss di base | Termine aggiunto |
| --- | --- | --- |
| C0 | 0,2 ancore + posa | Nessuno |
| CN | Identica | λ_N(t) InfoNCE sulle finestre train ammissibili |
| CR | Identica | λ_R(t) RnC sugli istanti con posa valida del batch globale |

Tre training nuovi: seme 0, stessa inizializzazione non addestrata, stessi dati, ordine di finestre, montaggi, maschere e quote; batch globale 32, quattro rank, accumulo 1, sei worker per rank. AdamW, LR 1e−4, clipping globale 1, weight decay 0,05, EMA 0,996; JEPA a peso zero. Schedule WSD di riferimento: 49.046 update, warmup 1.962, decay finale 20%, con arresto scientifico a 19.618 e checkpoint anche a 5.000.

Tutti i bracci costruiscono lo stesso proiettore e calcolano entrambi i termini e la memoria; cambia soltanto il peso attivo. Il proiettore usa RNG isolato e non altera inizializzazione condivisa o stream; il trainer lo aggiunge prima dell'optimizer e della copia teacher. Si riusano i forward student/teacher. Il controllo comprende il costo dell'infrastruttura nuova: non si assume costo identico al T100-G storico.

CN−CR confronta **ricette**, con supervisione, unità, candidati e traiettorie di gradiente differenti. La calibrazione iniziale non assicura dose di gradiente identica durante il training. Nessuna conclusione di superiorità universale di una famiglia di loss.

## 2. InfoNCE: identità, negativi e proiettore

- Query: student sulla vista mascherata storica; positivo: teacher EMA completo sulla stessa finestra fisica e stesso montaggio. Nessuna trasformazione aggiuntiva di ampiezza, tempo o canali.
- Readout: media dei 64 slot, poi delle patch fisiche con QC valido, prima del decoder. Le patch interamente nascoste restano nel supporto; una finestra priva di chiavi student visibili valide non contribuisce a NCE.
- Proiettore 384→384→GELU→128, bias, senza BatchNorm/dropout; coseno normalizzato FP32, temperatura **0,2**. Query con gradiente, chiavi detached; niente secondo student forward o simmetrizzazione.
- FIFO **1.024 chiavi per dataset**, età massima **256 update**. Candidati = memoria precedente + batch globale corrente, nello stesso dataset. Lo stesso soggetto non viene escluso sistematicamente.
- Copie della stessa finestra fisica escluse indipendentemente dal montaggio. Nella stessa registrazione, candidati ammessi soltanto sullo stesso asse di trial e con gap tra gli intervalli **≥4 s**; overlap sempre escluso. Trial diversi della stessa registrazione sono esclusi.
- `trial=None` identifica l'asse di campioni della registrazione su disco. Per prove concatenate, come Hyser, il gap si misura su quell'asse e può ammettere prove concatenate distanti almeno 4 s d'indice: non prova 4 s di separazione reale fra acquisizioni. CapgMyo conserva trial separati; le sue prove da 1 s non diventano negativi reciproci della stessa registrazione. Questa asimmetria di formato resta dichiarata e congelata.
- Identità e tempi mancanti sono errori. Deduplicazione deterministica dei candidati: chiave più recente, corrente prima della FIFO, pareggio nell'ordine globale rank/indice. Query duplicate conservate nel denominatore globale.
- Minimo **32 negativi** ammissibili per query; altrimenti contributo zero registrato. Loss divisa per **32 finestre globali**, non per le sole query ammissibili.
- Chiavi teacher calcolate prima dell'update; optimizer ed EMA una volta, poi commit della memoria con quelle chiavi. Nessun aggiornamento in eval o ricomputazione del backward. Memoria e stato di avanzamento nel checkpoint.

Restano falsi negativi e scorciatoie di soggetto/sessione. Si pubblicano per dataset numero ed età dei negativi, quote dello stesso soggetto/registrazione e del batch corrente, perdita NCE. Numero e freschezza differiscono: stessa quota di query non significa stesso contributo alla loss. Il riferimento a logit uniformi non è una misura sull'encoder reale. Nessun cambio del mix o dei candidati per pareggiare queste quantità.

**Le loss NCE dei tre bracci non sono direttamente confrontabili come efficacia.** In C0 e CR il proiettore non riceve gradiente di apprendimento NCE, ma può evolvere per weight decay e tramite EMA teacher; in CN è addestrato. Il gradiente del proiettore entra nel clipping globale di CN ed è escluso dalla calibrazione sull'encoder: la sua norma viene registrata separatamente. La rampa della loss non impone una rampa proporzionale al passo AdamW; dal primo peso non nullo l'adattamento può produrre passi dell'ordine del learning rate.

## 3. Rank-N-Contrast globale

- Readout temporale `mean_K(enc.z)` dello student mascherato, sul supporto `pose_time_valid` della MSE, comprese patch nascoste/slab. Padding e target invalidi esclusi.
- Un'ancora ogni **8 patch**, fase **4**: passo nominale 200 ms, senza estrazione casuale aggiuntiva.
- Candidati dal **batch globale**, raccolta delle feature differenziabile anche dal lato candidato; target e maschere detached. Rank senza posa partecipano alle collettive. Accumulo >1 rifiutato.
- Tutti i candidati ammissibili sono anche positivi: modalità **`all`**. Diagonale e coppie senza giunti validi comuni escluse. Nessun filtro di finestra applicato a un solo ruolo; variante asimmetrica `cross` rifiutata.
- Esclusione simmetrica delle coppie dello stesso istante fisico: dataset, soggetto, sessione, trial, frequenza e indice nativo `start + ceil((p+0,5) × patch_ms × fs / 1000 − 1e−9)`.
- Le repliche restano ancore e candidati rispetto agli altri punti: esclusione della coppia duplicata, senza deduplicazione o ripesatura della popolazione. Si riportano molteplicità massima, coppie duplicate escluse e overlap.
- Distanza target RMS degli angoli standardizzati validi in comune; **risoluzione 0**, ordine e parità esatti. Coseno dopo LayerNorm senza parametri, temperatura **0,1**.
- Per i,j il denominatore comprende k a distanza target da i almeno pari a quella di j, inclusi tutti i pareggi. Calcolo sicuro, senza sentinella finita −1e4.
- Normalizzazione per ancora al valore con similarità uniformi; almeno due positivi per contarla. Media delle ancore nella finestra, poi delle finestre con supporto. Finestre senza posa escluse dal termine.
- Loss globale replicata sui rank: la raccolta somma le derivate e la media DDP restituisce il gradiente globale. Nessuna seconda compensazione world-size.

**Con una sola finestra di posa RnC può restare attiva con soli confronti interni.** Istanti correlati della stessa finestra non sono esempi indipendenti. Si pubblicano finestre/ancore utili, registrazioni e soggetti distinti, quota di update con una sola finestra o registrazione, e norma RnC rispetto al numero di finestre. Il gate di attività non promette varietà tra registrazioni.

RnC non usa proiettore o memoria teacher nel proprio valore matematico; CR esegue comunque NCE a peso zero. La geometria riguarda angoli, non velocità o forza. Il calo della loss RnC è un controllo **non specifico**: può derivare anche dalla base supervisionata o da regolarizzazione generica. **Senza placebo non si attribuisce un guadagno di CR all'ordinamento per posa.** Il disegno approvato resta di tre bracci.

Controllo di manipolazione prespecificato, descrittivo e senza soglia di rilascio: nei soli passi diagnostici, correlazione di Spearman fra coseno e **negativo della distanza RMS di posa**, usando coppie valide di finestre diverse, triangolo superiore, con la stessa esclusione dei duplicati fisici e ranghi medi per i pareggi. Valori positivi indicano maggiore accordo geometrico; non provano un meccanismo causale. Si pubblicano numero di coppie e flag di validità. Supporto vuoto o variabile costante: flag 0, correlazione registrata 0 come sentinella esplicita, non interpretata come assenza di relazione.

## 4. Calibrazione comune, riferimento storico e rampa

Calibrazione **solo train**, student/teacher allo stato iniziale e senza optimizer step; la memoria avanza secondo la ricetta congelata.

1. Batch globali 0…255: riempimento e diagnostica.
2. Batch 256…511: gradienti globali completi di base, NCE, RnC sui parametri condivisi `tokenizer`, `identity`, `local`, `pool`, `backbone`; escluse teste, decoder e proiettore.
3. Selezione dei **primi 32 batch congiuntamente informativi**: supporto di entrambi i termini e tre norme finite e strettamente positive. Indici e ragioni di esclusione pubblicati. Se entro il batch 511 non sono 32, stop; nessuna estensione del pannello.
4. Sugli stessi 32 batch, per ciascun termine c: **`λ_c = 0,1 × k × mediana_b(||g_base,b|| / ||g_c,b||)`**, con **`k = 0,1179564123141922`**.
5. Entrambi i λ finiti e in **[1e−4; 1]**, estremi inclusi. Se uno o entrambi sono fuori, stop del rilascio dell'intera coorte: niente clamp, benchmark del solo superstite, retry o cambio automatico della regola. Ogni modifica richiede revisione e nuova calibrazione comune.
6. Ricevuta con norme, supporto, coseni, pesi e pin; ricette dei tre bracci generate dalla stessa ricevuta autenticata. Reset completo di modello, teacher, optimizer, memoria, contatori e RNG prima di ogni nuovo run.

L'opzione D-3/2 congela k come media storica di `grad_norm` agli update **513…19.618** (**3,5140269780276796**) divisa per la media agli update **1…8** (**29,790894018271878**). Fonte e hash: `review_response/gradient_reference.json`, fissato dal `plan.json`. Il rapporto storico attenua la frazione iniziale a **0,01179564123141922**; non sostituisce i gradienti realmente misurati in calibrazione.

Lo strato d'uscita della testa posa nasce a zero: **il gradiente iniziale della posa sui parametri condivisi è esattamente zero**, quindi la base condivisa iniziale proviene dalle ancore. La testa posa stessa può avere gradiente non nullo. Il log storico riguarda tutti i parametri prima del clipping, la calibrazione solo quelli condivisi: questa differenza limita l'interpretazione di k.

Solo ipotizzando norma contrastiva costante e traiettoria della base uguale allo storico, il rapporto attivo avrebbe media **0,057** agli update 513…1.024, **0,114** sull'intero tratto attivo e **0,145** negli ultimi 100. Sono scenari di scala, **non gradienti contrastivi misurati**, previsioni garantite o rapporti fra passi AdamW. Il rapporto delle medie sul tratto attivo è 0,1; la media dei rapporti non coincide con esso.

Update 1…256 a peso zero; 257…512 con rampa lineare da λ/256 a λ; successivamente peso pieno fisso. C0 sempre zero. Nessun uso dell'E4 locale dell'ultimo micro-batch come sostituto del gradiente globale.

## 5. Copertura e rapporti di gradiente

**D-1/A.** Con `time_weighted=true`, q_d è la somma per dataset di `weight / window_seconds` sulle sessioni di pretraining, normalizzata sul totale; durata nulla dà quota zero. Manifest e durate pinnati in `review_response/coverage_manifest_reference.json`. Esenzione dai soli criteri per dataset quando **`q_d × 32 × 256 < 32 + 1`**.

L'unico esente è **CapgMyo**: q=0,0028543868474759163, 23,3831 finestre attese contro 33; il successivo dataset ne ha 103,3445. È uno screening della fattibilità della memoria, non garanzia di chiavi distinte o di superamento per gli altri dataset.

| Gate | Regola |
| --- | --- |
| InfoNCE globale | Ammissibili / tutte le query ≥90%, **compreso CapgMyo** |
| InfoNCE per dataset non esente | ≥32 query osservate e frazione ammissibile ≥80% |
| RnC | ≥90% dei batch globali con una o più finestre contate |
| Calibrazione congiunta | Primi 32 batch informativi nel pannello, oltre ai gate di copertura |

CapgMyo resta in pretraining, report, conteggi e denominatori globali; nessun cambio di campionamento o supporto artificiale. Nel replay storico ha 0 query NCE ammissibili su 1.461, mediana 3 e massimo 8 negativi; nel pannello di calibrazione compare 13 volte. Questa limitazione è parte dell'esperimento. Politica identica in C0/CN/CR, audit e lettore della calibrazione.

La calibrazione valuta il pannello 256…511; l'audit deve riportarlo separatamente. In training i **blocchi completi di 1.024 update dopo il riempimento sono gate di arresto**, non solo log; il blocco finale parziale è descrittivo. Fallimento: diagnosi e revisione, senza variazioni automatiche di coda, stride, quota o peso. Gestione comune della coorte e dose appaiata.

Il replay indipendente passa da 0/18 a 18/18 blocchi con l'esenzione. Usa indici storici, validità QC/visibilità idealizzata e durate 1/4 s: non sostituisce l'audit reale. La sensibilità sulla durata cambia una query DB3 e nessun verdetto. Identità dello stream e supporto effettivo dei nuovi run vanno verificati nei job tecnici.

**Benchmark, tutti i 64 update completati 513…576.** Norme globali condivise di base/NCE/RnC e coseni; per il termine attivo del rispettivo braccio **`ρ_c = λ_c × ||g_c|| / ||g_base||`**, prima del clipping, con base finita e strettamente positiva. Limiti inclusivi congelati nel `plan.json`:

- mediana ρ_N di CN e mediana ρ_R di CR entrambe in **[0,005; 0,5]**;
- percentile 95 di ciascun rapporto **≤1**;
- rapporto fra le mediane CN/CR in **[0,25; 4]**.

Fuori limite, non finitezza o pannello incompleto: stop del rilascio della coorte per review, senza aggiustare λ. Sono limiti tecnici di intervento non nullo/non dominante, non leggi empiriche o dose equivalente. Un termine che si riduce rapidamente può violare il limite inferiore pur essendo matematicamente valido. Il percentile 95 su 64 batch riguarda circa tre osservazioni di coda: pubblicare tutta la sequenza, senza promessa di precisione.

Diagnostiche globali ogni **100 update**, anche a 256, 512, 1.962, 5.000 e 19.618, più il pannello denso 513…576. I rapporti tardivi sono **descrittivi** nei blocchi 513…1.024, 1.025…1.962, 1.963…5.000, 5.001…10.000, 10.001…19.618. ρ>0,5 significa norma aggiunta sostanziale; ρ>1 norma maggiore della base. Non misura causalità o frazione del passo assegnata alla loss. Divergenza CN/CR limita l'attribuzione alla sola forma dell'obiettivo e non attiva retuning.

Il collaudo pubblica anche negativi/età e frazioni per soggetto/registrazione, NCE per dataset, finestre/ancore/coppie RnC, duplicati/overlap, quota di batch con una finestra/registrazione, allineamento di rango, norma del proiettore, clipping, memoria e costo. Le loss grezze non sostituiscono questi controlli o i risultati downstream.

## 6. Endpoint, lettori e riferimento storico

Teacher congelato **prima del proiettore**, stessi split e lettori storici: ridge istantanea sulla media K64 per posa; pool768 e logistica convergente per classificazione, senza nuova PCA o ricerca del pooling.

| Dose | Letture | Ruolo |
| --- | --- | --- |
| 5.000, tutti i bracci | E_dyn, correlazione, varianza predetta/target, MAE, livello/dinamica | Descrittivo; nessuna selezione o stop scientifico |
| 19.618, tutti i bracci | **E_dyn, P1, EPN612** | Tre endpoint decisionali |
| 19.618 | **P2 storico, UCI**, dataset separati, gruppi posa tutti/statici, diagnostiche/costo | Descrittivo |

P1 conserva i pesi DB2/DB3/DB6 **8/13, 3/13, 2/13**. **P2 conserva nome e definizione: 1/2 EPN612 + 1/2 UCI. EPN612 non si rinomina P2.** Restano i raggruppamenti di estrazioni e fit P1/P2; cambia la famiglia decisionale. I margini P2 conservati nel piano sono metadata storici descrittivi e non entrano nelle decisioni v2.

Balanced accuracy dai conteggi aggregati per classe, non media per soggetto. E_dyn = somma degli errori quadratici centrati / somma della variabilità target sulle finestre dinamiche, non media dei rapporti per utente. La riduzione relativa della posa si calcola **in ogni replica bootstrap** rispetto al braccio di confronto della stessa replica.

L'emendamento è informato da sviluppo già osservato. Ricalcolo indipendente, 100.000 repliche sui quattro bracci storici: semiampiezze simultanee P1 0,69…0,81 pp, P2 1,79…2,97 pp, EPN612 0,72…0,92 pp; UCI determina circa 95…98% della varianza del contrasto P2. Nessun allargamento dei margini per adeguarli al rumore. Scelta precedente ai nuovi risultati, ma non validazione su test intatto.

Coppie prespecificate: **CN−C0, CR−C0, CN−CR**. Vantaggio diretto insufficiente se non si conserva la base. Si riporta **C0−T100-G storico a 19.618** come controllo descrittivo di compatibilità: una ripetizione a seme uguale non stima la variabilità fra pretraining e gli IC per soggetto non ne diventano una misura. Percorso numerico e non determinismo possono contribuire; nessuna promessa di score storico bit per bit.

I soggetti P1 sono presenti nel pretraining: non si dimostra trasferimento a soggetti mai visti. La posa è già supervisionata in tutti i bracci: non è una regressione nuova. UCI/P2 descrittivi non consentono una dichiarazione di NI generale della classificazione fondata soltanto su EPN612.

## 7. Inferenza e profili senza vincitore universale

Beneficio positivo: P1/EPN612 in pp, posa come riduzione relativa percentuale di E_dyn. **100.000 repliche, seme 20261116**; soggetti entro dataset e utenti per posa, draw appaiati fra bracci, dataset in ordine alfabetico. Macro-recall sulle classi presenti nel draw, tutte le repliche conservate. Bootstrap condizionato ai checkpoint/lettori, un seme per braccio: non misura variabilità fra training.

Famiglia: **tre endpoint × tre coppie = nove contrasti**. IC95 descrittivo e IC percentile Bonferroni **nominale 99,4444%**, α famiglia 0,05. La correzione nominale non garantisce copertura congiunta effettiva 95% nel campione finito. Il verso inverso CN/CR usa gli stessi draw, senza tre ipotesi aggiuntive. P2/UCI hanno effetti e IC95 descrittivi, senza bandiere decisionali.

| Endpoint decisionale | Guadagno utile | NI | Intervallo di equivalenza pratica |
| --- | --- | --- | --- |
| E_dyn | Stima di riduzione ≥2% e limite inferiore >0 | Limite inferiore >−2% | IC interamente in (−2%; +2%) |
| P1 | Stima ≥+0,5 pp e limite inferiore >0 | Limite inferiore >−1 pp | IC interamente in (−1; +0,5) pp |
| EPN612 | Stima ≥+0,5 pp e limite inferiore >0 | Limite inferiore >−1 pp | IC interamente in (−1; +0,5) pp |

**Monte Carlo dei limiti decisionali.** Dai quantili 0,158655 e 0,841345 della Binomiale(n,p), p probabilità di coda, il comparatore ricava un intervallo di ranghi nelle repliche ordinate. Metà dell'ampiezza dei relativi valori stima un errore standard del limite. Nessun nuovo seme o aumento adattivo delle repliche. Limite distante **al più tre errori standard** dalla soglia: `boundary`, relativa bandiera non sostenuta. Stime, IC ed errore restano pubblicati. È incertezza Monte Carlo condizionata alla distribuzione bootstrap, non correzione del bias o prova di copertura.

Per endpoint/coppia: beneficio utile e NI secondo tabella, senza confine Monte Carlo pertinente; **danno materiale dimostrato** se il limite superiore è sotto −NI, lontano dal confine; **NI irrisolta** se non sostenuta ma senza danno dimostrato; **equivalenza pratica** se tutto l'IC è nell'intervallo pratico e i due limiti sono lontani dai confini. «Nessun vantaggio pratico sostenuto» richiede equivalenza su **tutti** gli endpoint decisionali: la semplice assenza di significatività non basta.

Promettente rispetto a un altro = almeno un beneficio utile e NI su tutti e tre gli endpoint. Il comparatore `scripts/contrastive_compare.py` valuta nell'ordine:

| Situazione | Profilo |
| --- | --- |
| Entrambi i versi CN/CR soddisfano beneficio+NI su endpoint diversi | `bidirectional_benefit_tradeoff_no_winner`, senza precedenza a CN |
| Un solo candidato soddisfa beneficio+NI contro C0 e nel confronto diretto | `CN_supported_for_confirmation` oppure `CR_supported_for_confirmation` |
| Una coppia presenta beneficio utile e danno materiale dimostrato | `tradeoff_with_demonstrated_material_harm` |
| Entrambi migliorano C0, nessun beneficio utile/danno materiale risolto nel diretto | `common_benefit_without_resolved_recipe_difference` |
| Entrambi migliorano C0 ma il diretto non determina una preferenza complessiva | `common_benefit_with_unresolved_recipe_preference` |
| Tutte le coppie, incluso il verso inverso CN/CR, praticamente equivalenti | `practically_equivalent_on_decision_endpoints` |
| Altri casi | `unresolved_or_tradeoff`, con benefici, danni e NI irrisolte espliciti per coppia |

Nessuna somma di posa e classificazione, nessun vincitore universale. «Per conferma» = proporre separatamente verifica della variabilità fra training e/o un compito nuovo; non autorizza adozione, estensione o ulteriori job.

Il comparatore verifica OOF appaiati, supporto posa e checkpoint comuni, famiglia numerica/stringa delle etichette e **420 record di fit** (3 bracci × 5 dataset × 4 fold × 7 fit). Solver irrisolti, ridge al bordo irrisolta o bootstrap non conforme mantengono le stime e sospendono l'interpretazione (`interpretation_withheld`). Provenienza/leakage degli estrattori sono prerequisiti distinti.

## 8. Rilascio tecnico, costi e conservazione

Correzioni/test locali → review Claude del diff e convergenza → audit reale, calibrazione e benchmark secondo [piano esecutivo](contrastive_preflight_execution_20261116.md) → lettura delle ricevute e preventivo misurato → release congelata → training/valutazioni. Soglie prima dei nuovi dati; pesi dei job tecnici scartati.

Collaudo C0/CN/CR fino a 576 update: 64 update a peso pieno, parità iniziale a peso zero, norme/clipping, memoria/timing, ripresa a quattro GPU. Fasi esplicite con taglio a 544 e conclusione a 576 secondo il piano. Distinguere tempi senza diagnostica da quelli con le derivate aggiuntive; il costo completo include entrambi e il loader.

| Voce | Riferimento | Prima del training scientifico |
| --- | --- | --- |
| Tre training a 19.618 | 3 × 16,9444 = **50,8333 GPU-ore** storiche prima dell'overhead | Tre benchmark misurati e tetti per braccio |
| Audit/calibrazione/benchmark | Risorse e limiti nel piano esecutivo | Review convergente, dati/codice fissati, ricevute autentiche |
| Dodici estrazioni | 3 × (posa 5k, posa 19.618, gruppo P1, gruppo P2) | Checkpoint/pin, adattatori verificati e benchmark inferenza |
| Valutazioni CPU | Sei letture posa, sei aggregati storici P1/P2, 420 fit e bootstrap | Audit solver, benchmark e tetti CPU |
| Checkout/conservazione | Voci separate | Checkout immutabile, custodia degli output necessari |

50,83 GPU-ore non è il costo misurato del confronto o il suo tetto esecutivo. Ogni training conserva quattro GPU, batch e dose. Checkpoint a 19.618 e statistiche necessarie devono essere conservati oltre l'eventuale scadenza a **40 giorni di SCRATCH**; il piano di custodia va chiuso prima della release.

## 9. Strumenti e stato delle verifiche

Loss in `model/rank_contrast.py` e `model/contrastive.py`; identità/FIFO/raccolta in `training/contrastive_memory.py`; collegamento al trainer e diagnostiche in `training/contrastive_objectives.py`. Il trainer aggiunge il proiettore solo in presenza della configurazione contrastiva. `FMConfig` resta compatibile con la ricetta storica; la serializzazione omette il campo contrastivo quando assente. I vecchi checkout/hash non sono alterati; i nuovi sorgenti richiedono nuovi pin di release.

Il flag generale `train_jepa.py --contrastive-config` è rimosso. Ingresso di campagna: `scripts/contrastive_run.py` con ricetta, fase, codice, ingressi, audit, calibrazione e coorte autenticati. `--recipe-only` controlla solo la ricetta; `--validate-only` verifica la release completa. Training subordinato anche ai tre benchmark e alla ricevuta del budget misurato: cambiare lo stato del piano non basta.

`contrastive_audit.py` usa il loader train e riproduce in CPU quattro rank/sei worker logici per rank, senza encoder; non misura qualità delle proiezioni. `contrastive_prepare.py` genera ricette, legge calibrazione e genera la coorte; `contrastive_compare.py` applica endpoint/profili. `fixed_eval` resta di base (ancore/posa/JEPA), senza aggiornamento della memoria contrastiva. `arms_collaudo.py` riconosce i bracci per le verifiche tecniche; i vecchi criteri scientifici non sostituiscono quelli correnti.

La review Claude del 16/11 riguarda la consegna storica `db5ac41`/`1d87e8d`, non certifica il diff v2. Vi sono prove CPU/gloo di loss e gradienti; i test distribuiti sono eseguibili sul Mac, correggendo la dichiarazione precedente. Ripresa a quattro processi gloo entro arrotondamento, non sempre bit per bit: nessuna prova NCCL.

Per v2 le ricevute aggiornate di test/review devono identificare lo SHA rilasciato. Disponibili ricalcolo indipendente della copertura storica/bootstrap e regressioni del comparatore in `results/infra/contrastive_comparison_20261116/review_response/`; verifica completa delle correzioni e review restano da consolidare. **CUDA/NCCL, bf16 reale, gradienti contrastivi su segnali veri, supporto reale, prestazioni, estrazioni e fit dei nuovi bracci non sono attestati dai test CPU.** Il protocollo fissa requisiti, non certifica anticipatamente il superamento delle implementazioni in preparazione.
