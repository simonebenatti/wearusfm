# SEEDS: verifica delle fonti primarie — 18/11/2026

**Esito:** SEEDS è adatto a essere esaminato come dataset esterno per classificazione e regressione continua, mantenendolo escluso dal pretraining. La fonte ufficiale dei dati è **OSF, DOI `10.17605/OSF.IO/WA3QK`**. La scheda Figshare associata all'articolo contiene metadati e non è il deposito delle registrazioni. Questa verifica riguarda le fonti: non attesta ancora completezza, checksum o contenuto degli archivi scaricati, né autorizza risultati sperimentali.

## Fonti e provenienza

- Matran-Fernandez et al., *SEEDS, simultaneous recordings of high-density EMG and finger joint angles during multiple hand movements*, Scientific Data 6, 186 (2019), [articolo, DOI 10.1038/s41597-019-0200-9](https://www.nature.com/articles/s41597-019-0200-9).
- [Testo integrale primario in XML, Europe PMC, PMC6768861](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6768861/fullTextXML), letto direttamente per Methods, Data Records, Technical Validation e Usage Notes. Copia di lavoro locale: `/private/tmp/seeds_paper.xml`; non è un artefatto da versionare.
- Riferimento 15 dell'articolo: Matran-Fernandez, Rodríguez Martínez, Poli, Cipriani e Citi (2018), *SEEDS dataset*, [OSF, DOI 10.17605/OSF.IO/WA3QK](https://doi.org/10.17605/OSF.IO/WA3QK), [deposito](https://osf.io/wa3qk/).
- [File dei metadati Figshare associati all'articolo](https://ndownloader.figshare.com/files/22056342): fonte accessoria per individuare il deposito, non sostituto dei dati OSF.

**Licenza del dataset verificata tramite OSF:** la relazione `license` del [nodo `wa3qk`](https://api.osf.io/v2/nodes/wa3qk/) rinvia alla [risorsa licenza `563c1cf88c5e4a3877f9e96a`](https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96a/), che dichiara `CC-By Attribution 4.0 International` e collega il [testo CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode). Il payload della licenza acquisito dal coordinatore è stato riletto in `/private/tmp/seeds_osf_license.json`; la ricevuta di acquisizione deve conservarne una copia persistente con il legame al nodo. L'indisponibilità della pagina nel browser non equivaleva a indisponibilità della metadata API via download.

La licenza dell'**articolo** è anch'essa [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), mentre il suo avviso assegna CC0 ai **metadati associati all'articolo**. La CC0 dei metadati Figshare non sostituisce la licenza CC BY 4.0 del dataset OSF. Conservare nella provenienza attribuzione agli autori, DOI del dataset e dell'articolo, collegamento alla licenza e indicazione delle trasformazioni effettuate.

## Che cosa è effettivamente registrato

| Aspetto | Dato attestato dall'articolo | Conseguenza per WearUsFM |
|---|---|---|
| Partecipanti | 25 volontari senza disturbi scheletrici/neuromuscolari noti; 22 destrimani, 2 mancini, 1 ambidestro | Non è una coorte di amputati; generalizzazione clinica non dimostrata |
| Lato acquisito | **Avambraccio e mano sinistri per tutti**, per disponibilità del solo guanto sinistro | Lato acquisito e mano dominante sono metadati distinti; niente mirroring implicito |
| EMG anteriore | 126 elettrodi HD, **9 righe × 14 colonne**, distanza centro-centro 4 mm; area di registrazione 33 × 53 mm² | È una griglia 126 canali, non una griglia 128 canali |
| EMG posteriore | 8 ulteriori elettrodi monopolar, sui muscoli estensori | In totale **134 canali**; gruppo separato dalla griglia |
| Frequenza EMG | Acquisizione originale 16.384 Hz; distribuzione 2.048 Hz | L'ingresso pubblico da verificare è a 2.048 Hz |
| Cinematica | CyberGlove III, 18 sensori: 10 di flessione delle dita, 4 abduzioni, thumb crossover, palm arch, wrist flexion e wrist abduction | Sono **18 letture di sensore**, non automaticamente 18 angoli anatomici equivalenti ai 20 target emg2pose |
| Frequenza guanto | Acquisizione originale **60 Hz**; distribuzione **256 Hz** dopo sincronizzazione e ricampionamento | La serie a 256 Hz non contiene altrettanta informazione temporale indipendente |
| Compiti | 13 movimenti, 3 sessioni, 6 ripetizioni per movimento/sessione; velocità slow/normal, salvata fast/slow | Movimento, sessione, ripetizione e velocità vanno preservati; non assumere bilanciamento delle velocità senza inventario |

Il gruppo HD è posto sulla regione dei flessori dell'avambraccio, mirando al ventre del flexor carpi radialis; gli otto elettrodi posteriori mirano a extensor carpi ulnaris, extensor digitorum comunis e extensor carpi radialis longus. Il posizionamento usa la contrazione volontaria per localizzare i muscoli. Le etichette agli angoli della griglia sono MA1, MA9, MN1 e MN9; la **Figura 1** dà la disposizione. L'ordine reale va letto da `channels_emg`: non basta dedurlo dall'indice di riga. CMS e DRL sono riferimenti hardware, non due dei 134 ingressi da aggiungere al modello.

## Bersagli: normalizzati, non gradi

La sezione Experimental protocol chiarisce esplicitamente che il guanto non registra gradi assoluti di flessione, ma un segnale analogico da calibrare. Per ciascun partecipante e sensore viene applicata la trasformazione `(X_orig - m) / (M - m)`, usando estremi ottenuti dalla calibrazione iniziale. Gli estremi sono forniti separatamente.

**Non si possono chiamare MAE in gradi o errori angolari fisici le metriche su questi bersagli.** Invertire la normalizzazione restituisce la lettura originaria del CyberGlove, non dimostra una conversione in gradi. Per il primo confronto usare unità normalizzate, con definizione esplicita dell'aggregazione per sensore e soggetto; eventuale standardizzazione del learner va stimata solo sul fit. Non riutilizzare la testa emg2pose a 20 uscite come se i target coincidessero.

Usage Notes segnala inoltre che la calibrazione non ha coperto tutti i giunti: **valori fuori dall'intervallo [0,1] possono essere legittimi**. Vanno quantificati, non automaticamente tagliati o classificati come corruzione. L'eventuale clipping cambierebbe il benchmark. La calibrazione è specifica del partecipante e precede l'esperimento: il protocollo deve dichiarare che usa le letture già calibrate fornite, senza stimare nuovi estremi dalle registrazioni di test.

## Preprocessing e sincronizzazione già effettuati

L'archivio pubblico descritto è **preprocessato e segmentato**, non il flusso originale grezzo. I flussi alle frequenze originali sono disponibili dagli autori su richiesta.

1. Common average reference applicata separatamente alla griglia HD di 126 canali e agli 8 monopolar.
2. Filtro Butterworth passa-banda di terzo ordine, zero-phase, 10–500 Hz, applicato all'EMG.
3. Normalizzazione del guanto con gli estremi della calibrazione per partecipante/sensore.
4. Sincronizzazione offline mediante trigger TTL: l'Arduino invia un trigger ogni nuovo campione del guanto; il sistema EMG registra il trigger. I campioni cinematici vengono posti ai cambiamenti del trigger e quindi ricampionati alla frequenza EMG.
5. Segmentazione delle singole ripetizioni; riduzione delle frequenze a 2.048 Hz per EMG e 256 Hz per guanto per la distribuzione.

Non applicare una seconda CAR o un secondo passa-banda solo perché il nostro ingest tratta il file come "raw". Un eventuale ricampionamento aggiuntivo richiede un filtro antialias appropriato e un asse temporale condiviso. Il filtro originale zero-phase rende questi dati già preprocessati offline: il solo uso di una testa causale non autorizza una conclusione di deploy strettamente causale.

Gli autori forniscono `resample_glove` per portare il guanto allo stesso numero di campioni dell'EMG conservando la sincronizzazione. **Il codice concreto del helper va letto prima di scegliere interpolazione, endpoint e trattamento delle lunghezze**. Il solo rapporto nominale 2048/256 = 8 non prova quale convenzione temporale abbia ogni file. La figura di sincronizzazione dell'articolo è una verifica illustrativa, non un oracolo indipendente di ogni registrazione scaricata.

## Inventario atteso, da confrontare con gli archivi

Il paper descrive 25 cartelle, ciascuna con 234 trial MATLAB: `13 × 3 × 6`. Attesi quindi **5.850 trial**, più **25 file di calibrazione**, oltre a helper e materiali di supporto. È un conteggio atteso dalla pubblicazione, non ancora una ricevuta dell'inventario scaricato.

**Struttura OSF verificata il 18/11:** il listing della radice `osfstorage`, salvato dal coordinatore in `/private/tmp/seeds_osf_root.json` e riletto qui, contiene 15 elementi totali con `next=null` e `per_page=100`: le dieci cartelle `subj01`…`subj10`, quattro cartelle di supporto (`functions`, `figures`, `stimuli`, `Wiki images`) e `Demographic Data.pdf`. Il listing dei provider contiene solo `osfstorage`. Gli altri partecipanti sono in **due componenti figli ufficiali**, individuati tramite [API children del nodo](https://api.osf.io/v2/nodes/wa3qk/children/) e riletti da `/private/tmp/seeds_osf_children.json`:

| Nodo | Contenuto dichiarato |
|---|---|
| [wa3qk](https://osf.io/wa3qk/) | Radice, `subj01`…`subj10` e supporti |
| [e7m5k](https://osf.io/e7m5k/) | `Participants 11-20` |
| [yp4vq](https://osf.io/yp4vq/) | `Participants 21-25` |

La prima apparente discrepanza dieci/venticinque è quindi spiegata dall'organizzazione in componenti; non è un motivo per ridurre lo split a dieci persone. **Il download deve attraversare tutti e tre i nodi.** Il contenuto dei componenti, i 5.850 trial attesi, le calibrazioni e le licenze/relazioni ereditate restano da attestare nell'inventario ricorsivo prima del congelamento del manifest.

- Trial: `detop_exp01_subjAA_SessB_CC_DD.mat`, con soggetto 1–25, sessione 1–3, movimento 1–13 e ripetizione 1–6. Non congelare lo zero-padding o le maiuscole prima di vedere i nomi reali.
- Variabili: `subject`, `session`, `date` (YYYYMMDD), `movement`, `fs_emg`, `fs_glove`, `speed`, `emg`, `channels_emg`, `glove`, `channels_glove`.
- `emg`: array canali × campioni, 134 righe; prime 126 HD e ultime 8 monopolar. `glove`: 18 × campioni.
- Calibrazione: `subjAA_calibration_values.mat`, campi `calibration_min` e `calibration_max`, 18 valori ciascuno nell'ordine di `channels_glove`.
- Cartella `Functions`/`functions`: helper MATLAB/Python, `resample_glove`, `load_data`, esempio d'uso e script della verifica classificativa. Il case e il contenuto effettivo restano da leggere nel deposito.
- Materiali descritti: video di calibrazione, spiegazione dei movimenti e stimoli alle due velocità; grafici della ripetibilità. Non sono necessari al primo download delle registrazioni, salvo chiarire semantica o tempi.

## Implicazioni e controlli ancora aperti

Il paper verifica la classificazione entro partecipante con kNN e feature manuali, finestre da 200 ms e overlap 25%, variando il taglio iniziale e la durata. **Non fornisce in quelle sezioni un protocollo esterno train/validation/test per il confronto FM vs fine-tuning vs TEMPONet.** I suoi numeri non costituiscono direttamente la baseline del nostro confronto.

Per il confronto richiesto servono gli stessi soggetti, segmenti, bersagli e informazioni temporali per tutti i bracci. I soggetti esterni al pretraining possono comunque servire a fit/validation/test del task: i tre ruoli vanno fissati esplicitamente. Le finestre sovrapposte non vanno distribuite casualmente fra ruoli; tenere soggetti interi per una misura su nuovi utenti. La testa congelata e quella del fine-tuning devono differire per l'aggiornamento dell'encoder, non per un vantaggio implicito nel bersaglio o nella finestra. TEMPONet va adattata e dichiarata per 134 ingressi e 18 uscite, non presentata come replica identica di un modello su un altro dataset.

Prima dei fit restano da attestare sul payload:

- ricevuta persistente della licenza del deposito già verificata, elenco/versioni dei file, dimensioni e checksum;
- 5.850 chiavi trial univoche e 25 calibrazioni, o elenco esplicito delle discrepanze;
- forma/dtype/unità EMG, frequenze, finitezza, canali, lato e dati mancanti;
- corrispondenza fra nomi dei file e metadati, durata e lunghezze EMG/guanto;
- `calibration_max > calibration_min`, eventuali degenerazioni e distribuzione dei target fuori [0,1];
- comportamento effettivo di `resample_glove`, allineamento su esempi di ogni soggetto e convenzioni ai bordi;
- identità del mapping dei 18 sensori fra file/soggetti; nessuna conversione non documentata in gradi;
- esclusione di SEEDS e dei suoi soggetti dal corpus di pretraining e manifest separato del benchmark.

Questa nota non lancia job, non aggiunge SEEDS al corpus di pretraining e non conclude che la regressione migliorerà: identifica ciò che il dataset può misurare e i controlli necessari per confronti interpretabili.

## Seguito verificato dal coordinatore — inventario e campione locale

Inventario ricorsivo completato e congelato: `results/infra/seeds_20261118/source_manifest.json`,
SHA256 `fcfb6308b5155fdd14e63109b200a2208c88326c602e8a963c746b76e14856d7`.
Confermati tutti25soggetti con234trial e una calibrazione ciascuno, più quattro helper e
unPDF:5.880file,59.737.275.276byte. API originali archiviate in `api_inventory.json.gz`;
licenzaCCBY4.0 dichiarata separatamente da tutti e tre i nodi. Questo chiude la verifica
dell'inventario; l'integrità dell'intero payload richiede ancora il download/auditCPU.

Helper letti: `resample_glove.m` fa interpolazione lineare sugli estremi e ignora `orig_fs`;
`functions_osf.py` usa ricampionamentoFFT a8×seguito da troncamento. Le funzioni non sono
identiche. Il trial campione ha32.678campioniEMG e4.085campioni del guanto, non rapporto8esatto.
Un trial e una calibrazione hanno checksum e struttura corretti; audit dichiaratoPARTIAL,
nonPASScompleto. I7.221valori del guanto sottozero sono conservati senza clipping.
