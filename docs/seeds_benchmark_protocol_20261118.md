# SEEDS: preparazione del confronto di regressione esterno

**18/11/2026 — proposta di protocollo; acquisizione e audit autorizzati, nessun training scientifico avviato.** Simone: «Teniamolo fuori dal pretraining e prepariamo il confronto tra FM congelato, fine-tuning e TEMPONet». Questo documento prepara quei tre bracci. Non modifica le campagne in corso e non aggiunge automaticamente un quarto braccio o una nuova loss.

Il confronto diventerà eseguibile dopo l'audit dei file, l'implementazione degli adattatori mancanti, la review del pacchetto e il preventivo misurato. Le scelte seguenti precedono qualsiasi risultato SEEDS. Un problema dell'audit si registra prima dei fit, senza sostituire soggetti o scegliere un protocollo dai punteggi.

**Gate di copertura dei metadati superato, payload ancora da verificare:** l'inventario completo attraversa i tre componenti ufficiali OSF `wa3qk` (01–10), `e7m5k` (11–20) e `yp4vq` (21–25). `results/infra/seeds_20261118/source_manifest.json` registra5.880file:5.850trial,25calibrazioni,4helper e1PDF demografico,59.737.275.276byte (55,635GiB), licenzaCCBY4.0 per tutti i tre nodi. Ogni soggetto ha234trial e una calibrazione. La root da sola mostrava dieci persone: la discrepanza era dovuta ai componenti, non a soggetti mancanti. Il15/5/5 resta quello proposto; gli hash e i contenuti di tutti i file devono ancora essere verificati dopo il download.

## 1. Domanda scientifica e dati

**Su persone di SEEDS mai incontrate dal pretraining né dall'adattamento supervisionato, quanto valgono le rappresentazioni del FM, quanto migliora adattare l'encoder e come si confrontano con una rete temporale addestrata direttamente sull'EMG?**

Fonte primaria: [Scientific Data 2019, DOI 10.1038/s41597-019-0200-9](https://www.nature.com/articles/s41597-019-0200-9). Secondo il Data Descriptor si attendono 25 soggetti, 13 movimenti, tre sessioni e sei ripetizioni, quindi 234 registrazioni per soggetto e 5.850 complessive. Le tre sessioni e tutte le ripetizioni di una persona appartengono allo stesso split.

| Voce | Contratto da verificare sui file |
|---|---|
| EMG | 134 canali, 2.048 Hz: 126 della matrice ad alta densità e otto degli estensori |
| Cinematica | 18 canali del guanto acquisiti a60Hz, distribuiti a256Hz dopo ricampionamento; mantenere ordine e nomi originali |
| Unità dei bersagli | Escursione normalizzata dalla calibrazione degli autori, nominalmente 0–1; valori esterni sono legittimi e non vanno tagliati; **non gradi e non radianti** |
| Allineamento | Sincronizzazione fornita dagli autori; verificare tempi, lunghezze, rapporto delle frequenze e convenzione del primo campione |
| Dati distribuiti | Versione elaborata dagli autori: riferimento medio separato per i due gruppi e filtraggio 10–500 Hz descritti nel paper; non definirla EMG non filtrato |
| Compito | Regressione della sequenza dei 18 segnali del guanto, non classificazione dei 13 movimenti |

Le matrici di calibrazione min/max non sono da sole una conversione in angoli fisici. Non riusare i 20 giunti, le statistiche, le maschere o le soglie dinamiche di emg2pose. Un canale del guanto non viene dichiarato equivalente a un suo giunto senza documentazione.

## 2. Esclusione dal pretraining e split irrevocabile

**Tutto SEEDS resta esterno al pretraining**, compresi i 15 soggetti destinati al futuro adattamento. Nell'allowlist attuale di `src/wearusfm/data/pretraining_manifest.py` non compare SEEDS. La garanzia operativa richiede anche controllo di manifest reali, sorgenti e alias: il solo nome assente non è prova sufficiente di assenza di duplicati.

Conservare raw, derivati e manifest SEEDS in una radice dedicata al benchmark, fuori dalle radici attraversate automaticamente dal loader di pretraining. Ogni voce porta `purpose=external_regression_benchmark`, `pretraining_allowed=false`. Un controllo deve rifiutare intersezioni per percorso, identificativo sorgente e hash con i manifest di pretraining. Non aggiungere SEEDS all'allowlist per rendere comodo un loader.

Split proposto **15 train / 5 validation / 5 test**, condizionato alla copertura completa dei25 soggetti e fissato senza leggere prestazioni o distribuire finestre casualmente. Per gli identificativi canonici `01`…`25`, ordinare per SHA256 UTF-8 di `SEEDS-subject-split-v1|20261118|AA`; primi 15 train, cinque successivi validation, ultimi cinque test. Ordinare numericamente ciascuna lista per presentarla:

| Split | Soggetti |
|---|---|
| Train | 05, 06, 07, 08, 11, 12, 14, 15, 16, 18, 20, 21, 22, 23, 24 |
| Validation | 01, 02, 04, 17, 19 |
| Test sigillato | 03, 09, 10, 13, 25 |

Questi numeri richiedono conferma della corrispondenza uno-a-uno con gli ID pubblicati. Un ID mancante, duplicato o diverso ferma il congelamento del manifest: nessuna nuova permutazione per ottenere un insieme più conveniente. Se l'archivio risultasse limitato a10 soggetti, si presenta prima una rettifica motivata di potenza e split, senza adattare tacitamente questo15/5/5 o chiamare benchmark completo un sottoinsieme. Gli audit di integrità possono leggere struttura, dimensioni, finitezza e identità di tutti i file; selezione di iperparametri, finestre diagnostiche e grafici qualitativi usa train, poi validation. Nessuna metrica predittiva sul test fino al congelamento finale dei tre bracci.

Il modello apprende una testa su SEEDS train: questo è **trasferimento con adattamento supervisionato verso nuovi soggetti**, non previsione zero-shot. La validation non viene aggiunta al train nel primo confronto, così il modello valutato è esattamente quello selezionato e verificato.

## 3. Segnale, calibrazione e bersagli comuni

### Ingresso e significato operativo della calibrazione

Primario: tutti i 134 canali nella stessa disposizione, frequenza nativa 2.048 Hz, senza selezione delle sole derivazioni che funzionano meglio. L'identità di elettrodo/geometria deve venire dai file e dal disegno dell'array, mai da associazioni muscolari inventate. Non usare il sottocampionamento casuale HD del pretraining durante estrazione o confronto. Canali invalidi si trattano con una regola di qualità congelata e identica per i tre bracci; prima vanno contati, senza escluderli perché correlano poco con la posa.

La normalizzazione deve essere compatibile con il checkpoint e usare la **stessa versione del segnale** per tutti i bracci. Proposta operativa: una scala robusta per sessione, condivisa fra i canali, stimata su una riserva di sola EMG separata dalla valutazione; mediana dei MAD per canale, non normalizzazione indipendente che cancelli i rapporti fra canali. Fissare nel manifest10secondi riservati, accumulati senza creare continuità fra trial, ed escludere interamente da fit e score i trial toccati. La data pubblicata è soltantoYYYYMMDD e movimenti/velocità erano randomizzati: **non è ancora ricostruito un prefisso cronologico**. Se non emerge l'ordine di acquisizione, scegliere e congelare una riserva deterministica prima dei fit, chiamandola tale; l'interpretazione operativa sarà calibrazione su dati separati della sessione, non sui primi10secondi di utilizzo reale.

Questo ammette **calibrazione EMG non supervisionata di una nuova sessione**. È permessa anche ai soggetti test, usa solo quel prefisso e non legge il guanto; non usa l'intera sessione né aggiorna pesi o statistiche BatchNorm sul test. Un'eventuale scala nulla/non finita è un errore esplicito, non una costante sostitutiva scelta a posteriori. La calibrazione del guanto degli autori definisce invece il target ed è parte delle condizioni del benchmark: non si afferma funzionamento senza tale calibrazione e non si ricalcolano min/max dai bersagli test.

Il filtraggio già presente nei file non va applicato una seconda volta per errore. Occorre distinguere preparazione comune del segnale, normalizzazione e front-end interno del FM, che conserva il proprio anti-aliasing e tokenizer. La politica comune finale di preprocessing va verificata contro il checkpoint prima di estrarre cache. Nessuna ricerca di ritardo sui punteggi: allineamento fornito, ritardo zero.

### Finestra e supporto temporale

Proposta comune: finestre complete di **4 secondi, 8.192 campioni EMG**, partenza ogni secondo entro lo stesso trial, senza concatenare ripetizioni/sessioni e senza padding di un trial corto per farlo sembrare lungo. Target ogni **25 ms**, 160 istanti per finestra; tokenizer del FM a 100 ms con 75 ms di overlap, GELU, invariato.

**La convenzione del target non è ancora congelata.** Il campione reale letto contiene32.678campioni EMG e4.085del guanto: non ha rapporto intero8. L'helper ufficiale MATLAB interpola linearmente gli indici1…N_glove suiN_emg punti equispaziati fra gli estremi e non usa l'argomento `orig_fs`. Quello Python usa ricampionamento FFT a8× e poi tronca alla lunghezza EMG. I due metodi non sono equivalenti, e il rapporto nominale2048/256 non decide quale asse temporale fosse inteso per ogni file.

Prima di creare target/task si documentano le convenzioni di entrambi gli helper su segnali noti e su campioni train. Si sceglie una trasformazione canonica unica per tutti i bracci, poi si congela la definizione della media nei bin semiaperti `[t,t+25ms)` sulla sequenza allineata. Un oracolo indipendente deve verificarne tempi, estremi e maschere, senza scegliere il metodo dai punteggi test. **Non è autorizzato assumere fin d'ora blocchi fissi di6/7campioni del guanto o di8campioni EMG**, né presentare l'accordo delle lunghezze come prova di allineamento fisiologico. Anche il front-end a2.048Hz deve conservare la griglia temporale e non accumulare patch da51campioni.

Si conservano sequenze, maschere di validità e chiavi `(subject,session,trial,tempo,canale)`. Supporto comune proposto: crop di sei patch per lato della finestra, gli stessi 148 centri per tutti i bracci. I trial corti e le code non coperte vengono contati e pubblicati; l'audit deve quantificare la copertura per soggetto/movimento/sessione prima di confermare che una finestra di 4 s è praticabile.

In validation/test le finestre sovrapposte non moltiplicano le osservazioni: ogni timestamp assoluto ha un solo proprietario, la finestra il cui centro è più vicino; pareggi alla partenza precedente. La scelta dipende solo dai tempi, non dai bersagli o dalle predizioni. Fit e maschere usano lo stesso manifest per tutti i bracci. Non filtrare le predizioni con EMA scelta sulla validation soltanto per TEMPONet; nell'esperimento primario non c'è post-filtro.

**Limite da dichiarare:** il sistema è offline e il backbone del FM è bidirezionale. Stesso buffer di 4 s non significa necessariamente stesso campo ricettivo effettivo delle diverse reti. Nessuna promessa di causalità o confronto di latenza di una protesi.

## 4. Tre bracci e checkpoint dichiarati prima dei risultati

| Braccio | Inizializzazione | Parti allenate | Lettura |
|---|---|---|---|
| FM congelato | Teacher T100-G 30M a 49.046 passi | Solo nuova testa SEEDS | Media dei 64 slot per patch, LayerNorm384, Linear384→18; nessuna media nel tempo |
| FM fine-tuning | **Lo stesso** stato teacher e stessa testa iniziale | Encoder e testa, senza decoder/obiettivi di pretraining inattivi | Identica testa, vista, supporto e target del congelato |
| TEMPONet da zero | Inizializzazione nuova, nessun peso preaddestrato | Rete temporale e uscita SEEDS | EMG comune a 134 canali, stessi 4 s, stessi 160 target e supporto di valutazione |

La testa 384→18 non riutilizza semanticamente i 20 output emg2pose. Inizializzazione proposta: LN gamma=1/beta=0, bias e pesi dell'uscita zero; identico stato iniziale in frozen e FT per ciascun seme. In FT si aggiorna la catena che produce la rappresentazione; teacher significa **origine dei pesi**, non aggiornamento EMA o loss JEPA durante l'adattamento. Quantificare parametri allenati e costo separatamente.

Checkpoint di riferimento, provenienza già registrata in `results/infra/consolidation_long_20261105/continuation/49046/receipt.json`:

- WORK: `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/consolidation_20261104/continuation/checkpoints/checkpoint_postdecay.pt`.
- SHA256: `bfa6141fc2ed6505d2f582a46f0c04f36fd35f867e8cff2b11555b241947a490`.
- 49.046 passi; 28.362.013 parametri nel run storico, denominazione 30M; sorgente training `17dd455f8d40d5430f87c2154c571a2f9e683baa`.

Il percorso/hash è una provenienza da ricevuta, non una nuova verifica remota effettuata da questo documento. Prima dell'uso: esistenza e checksum attuale, caricamento completo dello stato teacher e della configurazione 100 ms/GELU, nessun fallback allo student o a un'altra dose. Il checkpoint viene dichiarato adesso, non selezionato sui risultati SEEDS. Nessun 300M o log-quadratico aggiunto alla matrice.

### TEMPONet: adattamento ancora da qualificare

La sorgente raw di Claude `src/wearusfm/model/temponet.py`, ramo `claude/temponet-db8-20261109` (letto a `8d34f9c`), implementa **una posa per finestra**, originariamente 16×256 campioni, con convoluzioni, stride, pooling, BatchNorm, flatten e strati densi. Cambiare solo `in_channels=134` e `out_dim=18` non la rende un lettore di sequenze. Non usare 128 ms per TEMPONet contro 4 s del FM e chiamarlo confronto controllato del solo pretraining.

Candidato operativo da revisionare: **TEMPONet densa adattata**, mantenendo i tre blocchi convoluzionali raw, sostituendo flatten e densi globali con trasformazioni condivise nel tempo 64→64→32→18; riallineamento delle uscite alla griglia di 40 Hz mediante timestamp espliciti dei centri. Con il downsampling originale di fattore64, l'uscita intermedia è a32 Hz: interpolazione e bordi devono essere definiti e verificati, non identificati per indice con160 target. I livelli intermedi e i parametri/normalizzazioni/dropout del lettore devono essere congelati prima dei fit. Il campo ricettivo locale resta diverso da quello del FM: va misurato e riportato.

Questa è una **proposta di adattatore, non un'architettura già implementata né una replica letterale della TEMPONet pubblicata**. Prima di confermarla occorrono review scientifica, prova di allineamento, controllo di capacità sul train e benchmark; se non soddisfa il confronto richiesto si rettifica il protocollo prima di addestrare, senza scegliere fra varianti dai punteggi test. Nessun riuso automatico di L1, finestre 128 ms, filtro EMA o iperparametri selezionati nelle precedenti campagne emg2pose/DB8.

### Relazione con la proposta di testa TCN del 17–18 novembre

`docs/pose_temponet_frozen_protocol_20261117.md` sul ramo `codex/pose-temponet-proposal-20261117`, letto a `3e007c8`, propone una **TCN sui latenti più ramo affine**, 72.904 parametri per 20 target, su emg2pose. È una proposta documentale separata con runtime assente, non il modello raw qui sopra. Non eredita una firma per SEEDS e non è già disponibile come testa dell'FT.

Il confronto minimo SEEDS fissa LN+Linear in entrambi i FM per isolare l'adattamento dell'encoder. Un risultato scarso del congelato resta condizionato a questa testa. Se si decide prima dell'esecuzione di adottare invece una testa temporale più ricca, va usata **identica** nei due FM e aggiornata questa specifica, senza sceglierla dopo il test o introdurre una quarta linea nascosta.

## 5. Loss, scelta finita e controlli prima del test

Loss comune proposta: **MSE sui 18 target normalizzati**, media sugli elementi validi di ogni finestra, poi media sulle finestre. Nessuna loss contrastiva, velocità aggiunta, L1 o pesatura dinamica diversa in un singolo braccio. Nessuna standardizzazione dai target validation/test. La MAE resta metrica di reporting e non implica usare L1 per il fit.

La ricetta eseguibile deve fissare, prima dei fit, una ricerca contenuta e comparabile: griglia degli LR di testa e backbone, ottimizzatore/decadimento/clip, epoche massime, selettore e regola di arresto, batch effettivo ed eventuale accumulo. Valori ragionevoli non equivalgono a un benchmark di costo: **non sono ancora pinnati**. Il congelato e FT partono dalla stessa testa; FT ha LR del backbone dichiarato separatamente. TEMPONet non va penalizzata imponendole un LR ottimale soltanto per il FM.

Selettore proposto: minimo della MSE macro per soggetto sulla validation, stessi timestamp deduplicati; pareggi alla dose minore e poi al LR minore. Curve train/validation complete, miglior checkpoint e checkpoint finale conservati. Tre semi dell'adattamento (0,1,2), medesime liste di finestre e ordine appaiato fra frozen/FT; nessun rimpiazzo di un seme sfortunato. Se il costo rende impraticabili tre semi, si aggiorna il preventivo e il protocollo **prima** dei fit, non dopo averne visto il primo.

Controlli obbligatori prima di aprire il test:

1. Integrità della sorgente, zero sovrapposizione dei soggetti, zero attraversamento di trial/sessione nelle finestre; isolamento della calibrazione e assenza di SEEDS dal pretraining.
2. Convenzione canonica degli helper dichiarata, oracolo indipendente dei tempi2.048/256/40Hz, target sintetici a rampa/impulso e controllo della corrispondenza ai canali del guanto; crop e deduplicazione identici. Nessunassunto di ratio8 esatto.
3. Input realmente uguali dopo preprocessing; tutte le134 derivazioni presenti o esclusioni QC comuni dichiarate, nessun sottocampionamento HD nascosto.
4. Frozen: pesi/BN del FM immutati, cache con sequenza; FT: gradiente presente nel backbone e testa, assenza di update del decoder inattivo. Nessun fit/test di BatchNorm sui soggetti esclusi.
5. Controllo di memorizzazione di un piccolo insieme train e recupero di target sintetici con informazione nota; curve sufficienti per non attribuire al pretraining una mancata ottimizzazione del comparatore. Un fallimento è diagnostico, non un risultato negativo del FM o della TEMPONet.
6. Hash di selezioni, stati, split, config e supporti prima dell'unica lettura finale. Tutti i bracci completati o errore dichiarato: nessuna vittoria contro un fit in timeout o non convergente.

## 6. Report della regressione: livello e movimento separati

**Primario:** MAE normalizzata, macro sui18 canali e poi sui5 soggetti test, tutte le osservazioni valide sul supporto comune. Presentarla anche come percentuale della scala normalizzata, per esempio0,05=5 punti della scala, non «5 gradi». Riportare MAE per soggetto e per canale; MSE normalizzata e curva validation completano il quadro.

Per il movimento, pubblicare descrittivamente sullo stesso supporto:

- errore del livello medio per finestra/canale;
- `E_dyn = somma((pred−media_pred)−(target−media_target))² / somma(target−media_target)²`, centrate su identici tempi validi, rapporto di somme e non media di rapporti; almeno due punti per canale e denominatore positivo;
- correlazione centrata e rapporto varianza predetta/vera, per verificare traiettorie troppo piatte;
- MAE ed E_dyn distinti per finestre dinamiche e poco dinamiche, più numero di finestre e soggetti che contribuiscono;
- risultati per movimento, sessione e velocità prevista dal protocollo degli autori, tutti i gruppi dichiarati, senza selezionare quelli favorevoli.

La soglia dinamica si ricava **solo dai target train**, prima dei fit: mediana della varianza per finestra mediata sui canali, sul supporto comune, poi viene congelata e applicata a validation/test. Le finestre sovrapposte usano gli accumulatori dei soli timestamp loro assegnati, così il report non conta più volte una stessa osservazione; quelle con supporto insufficiente sono contate come non valutabili. **Con ownership e passo1s il tratto centrato è normalmente circa1s, pur avendo contesto4s:** questaE_dyn non coincide con la metrica storica centrata sui4s di emg2pose. Riportare la distribuzione delle durate effettive e non accostare numeri come se misurassero lo stesso supporto. E_dyn totale/dinamico/statico e varianza pred/target sono descrittivi, non nuovi selettori. Finestre a denominatore nullo vanno contate, non nascoste con un epsilon che simuli buona regressione.

**Quiete non significa semplicemente assenza di movimento:** una contrazione isometrica può avere posa stabile. Se il dataset non fornisce intervalli espliciti di riposo, usare «bassa dinamica» e non attribuire un'etichetta fisiologica di riposo. Mostrare esempi train scelti per ID prima dei risultati; per il test, un elenco fisso e uguale per i tre modelli, senza scegliere la traiettoria più bella.

Tre contrasti dichiarati: FT−frozen, FT−TEMPONet, frozen−TEMPONet. Differenze appaiate per soggetto, media aritmetica sui tre semi di adattamento, SD dei semi separata dagli intervalli fra utenti; non ensemble. Con cinque soggetti test riportare tutti i valori individuali e intervalli descrittivi bootstrap appaiati degli utenti (10.000 repliche, seed20261118), segnalando la forte discrezione e incertezza del piccolo campione. Nessuna soglia di «passa/non passa», nessuna pretesa di copertura garantita o di equivalenza da un intervallo che include zero.

## 7. Cosa il confronto può e non può concludere

| Esito | Lettura consentita |
|---|---|
| FT migliora frozen | Adattare l'encoder aiuta con questa testa, dose e split |
| Frozen predice bene nuovi soggetti SEEDS | Informazione trasferibile leggibile con una testa supervisionata su altri soggetti SEEDS |
| TEMPONet supera i FM | Questa pipeline specializzata, con la propria architettura, è migliore in questo confronto; non dimostra che ogni pretraining sia inutile |
| FM supera TEMPONet | Vantaggio del sistema FM considerato, non attribuzione causale del guadagno al solo pretraining |
| Tutti migliorano il livello ma non E_dyn | La dinamica resta un limite anche se la MAE aggregata è buona |

Per isolare il **valore del pretraining** serve in seguito la **stessa architettura FM da zero**, stessa testa e adeguata ottimizzazione. TEMPONet da zero è un riferimento applicativo con diversa architettura, non quel controllo. Lo si registra come controllo futuro necessario alla relativa affermazione, senza aggiungerlo automaticamente ai tre bracci richiesti.

## 8. Stato della preparazione e gate operativi

| Elemento | Stato al momento di questa proposta | Passo necessario |
|---|---|---|
| Fonte, download e inventario | Metadati dei tre nodi verificati:25ID/5.850trial+25calibrazioni,5.880filetotali,CCBY4.0 | Download e verifica di tutti i checksum/contenuti, ricevute ed eccezioni |
| Esclusione dal pretraining | SEEDS assente dall'allowlist letta | Verifica manifest effettivi/alias e guardia di non contaminazione |
| Split | Liste e algoritmo proposti sopra; copertura dei25ID confermata nei metadati OSF | Confrontare identità interne dei file e manifest pinnato, nessuna riscelta dai risultati |
| Frequenze/unità/allineamento | Specifiche della pubblicazione, non ancora certificato sui dati locali | Audit dei file e oracolo indipendente |
| Loader SEEDS | Non presente nel ramo base ispezionato | Adattatore dedicato, senza modificare i dataset storici |
| Checkpoint FM | Ricevuta con percorso persistente e hash disponibile | Verifica remota attuale, config/stato teacher |
| Inferenza134canali | Front-end progettato per numero variabile, ma questa combinazione non collaudata | Picco memoria, chunking che non cambi il modello, metadata/identità corretti |
| Testa18/fine-tuning | Componenti generiche disponibili, harness SEEDS assente | Medesima testa, timestamp e loss; test di congelamento/gradienti |
| TEMPONet SEEDS | Raw esistente produce una posa; versione densa solo proposta | Definizione completa, review, implementazione e controlli temporali |
| Ottimizzazione e costi | Nessun numero misurato SEEDS, nessun job preparato | Benchmark sul solo train, griglia/dose/tetto pubblicati |
| Test finale | Nessuna prestazione letta | Aprirlo soltanto dopo congelamento dei tre modelli e review |

Prima di ogni eventuale lancio scientifico: pacchetto congelato, review, risorse e criterio pubblicati, identificativi e output unici tramite submitter e ledger comuni. Nessun costo GPU viene dedotto da una TEMPONet a16 canali o da un FM su un altro montaggio. Questa preparazione non altera i job delle altre linee e non accede alle loro directory di output.

### Riscontri di sola lettura sui vincoli del codice

- **Non c'è un limite fisso a128 canali nel FM:** token, identità, packing e collate dipendono dal numero di canali del montaggio. `MAX_GROUPS=8` in `channel_codes.py` è il vocabolario dei gruppi, non il numero di canali. Questo non certifica memoria o qualità su134.
- **C'è invece un costo di memoria concreto per il montaggio misto.** `channel_codes.attention_sets` assegna a ogni canale `sparse` tutti gli altri canali validi, poi usa una larghezza comune pari al massimo. Se gli otto estensori sono correttamente descritti come `sparse` e tutti134 canali sono validi, la larghezza è134 anche per le righe della matrice HD. `LocalEncoderLayer.forward` materializza `kv_g` di forma `(134,134,160,2,384)` per una finestra: **8,22GiB inFP32 o4,11GiB inFP16**, solo per questo tensore e prima di scores, backward e altri livelli. Il banco134ch non può ereditare senza benchmark il batch dei16ch. Preflight a batch1; eventuale microbatching e checkpointing devono conservare la funzione e il batch effettivo. Non etichettare falsamente gli estensori come griglia né sottocampionare canali per aggirare il costo.
- **Formato del loader:** `SessionView.open` richiede `metadata.json` e `data_int16.npy`, canali sull'ultimo asse, `qc_valid`, `native_fs_hz`, montaggio e banda per canale. I MATLAB SEEDS non sono già quel formato. Serve un adattatore dedicato o una conversione verificata, preservando trial e target a256Hz; se si quantizza da floating point aint16, misurare saturazioni/errore e conservare la sorgente. Non dare per acquisita una conversione senza perdita.
- **Il loader dei target emg2pose non è generico:** `pose_targets.py` impone20 giunti,2000Hz, blocchi di10 campioni e file `pose_200hz_f16.npy`; `window_pose` rifiuta esplicitamente2048Hz. Non modificare queste costanti globali: scrivere la catena SEEDS separata18/256/2048 con il proprio oracolo.
- **Preprocessing/calibrazione del pretraining non è una politica di test:** il default storico aggiunge bandpass20–450Hz e notch50/60Hz; `estimate_session_scale` sceglie32 tratti da2s sparsi nella sessione. Non riusarlo sul testSEEDS al posto del prefisso separato, né rifiltrare inavvertitamente i dati già elaborati dagli autori. `LoaderConfig.virtual` deve essereNone per conservare l'array intero.
- **Lo script raw non è riutilizzabile solo cambiando il config:** `temponet_emg2pose.py` alloca dati16canali, finestre256, target20 ai blocchi200Hz, usaL1 e converte in gradi; la lettura finale selezionaEMA sulla validation. Tutti questi contratti sono specifici alla campagna storica e restano immutati. Il nuovo benchmark richiede un harnessSEEDS, non la sostituzione di un percorso nel launcher storico.

Sono osservazioni statiche sui sorgenti letti, non esiti di un collaudo a134canali o di un fit. Nessun modulo è stato implementato da questa nota.
