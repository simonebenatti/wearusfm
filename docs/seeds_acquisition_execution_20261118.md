# SEEDS: acquisizione e verifiche autorizzate — 18/11/2026

Simone: «scarichiamolo e facciamo le verifiche. Teniamolo fuori dal pretraining e
prepariamo il confronto tra FM congelato, fine-tuning e TEMPONet». GPT è autore e
operatore di questa acquisizione separata; nessuna operazione sulle altre campagne.

Inventario ufficiale completato sui tre nodi OSF wa3qk, e7m5k, yp4vq:
**5.880 file, 59.737.275.276 byte (55,635 GiB)**. Sono 5.850 trial,
25 calibrazioni, quattro helper e un PDF demografico. Ogni soggetto ha234 trial e
una calibrazione; i tre nodi dichiarano CC BY4.0. Figure, video degli stimoli e
immagini wiki non sono segnali e sono esplicitamente esclusi dal download.
`results/infra/seeds_20261118/source_manifest.json` congela URL, versione OSF,
dimensione, SHA256 e MD5 di ciascun file. Il conteggio del manifest non sostituisce
la verifica del payload.

## Destinazioni e isolamento

- Segnali originali: `$SCRATCH/data/external_benchmarks/seeds`, mai sotto `processed`.
- Ricevute e audit: `$WORK/wearusfm_runs/results/seeds_20261118`.
- Codice: checkout detached nuovo `seeds-acquisition-20261118`, SHA pubblicato prima
  della sottomissione e verificato dal launcher.
- Nessun aggiornamento di manifest, quote, split o loader del pretraining. Marker
  obbligatorio `pretraining_allowed=false`; la CLI rifiuta altre radici di dati.
- SCRATCH è soggetto alla politica di cancellazione del cluster. La fonte pubblica,
  il manifest, gli hash e i report conservati permettono il recupero; non si dichiara
  un backup permanente locale dei60GB. I checkpoint storici non vengono modificati.

## Risorse e catena

| Fase | ID logico | CPU | RAM | Limite | Massimo core-ore | GPU-ore |
|---|---|---:|---:|---:|---:|---:|
| Checkout congelato | SEEDS-checkout-20261118 | 1 | 2GiB | 10min | 0,167 | 0 |
| Download e checksum | SEEDS-download-20261118 | 2 | 4GiB | 4h | 8 | 0 |
| Audit MAT completo | SEEDS-audit-20261118 | 2 | 16GiB | 2h | 4 | 0 |

**Tetto12,167 core-ore, zero GPU-ore.** Il limite corrente della partizione e QoS
`lrd_all_serial` è4h, verificato con scheduler il18/11; il download rispetta questo limite. Sono tetti di allocazione, non tempi previsti;
la velocità del deposito OSF e la coda non sono garantite. Quattro trasferimenti al
massimo, due retry aggiuntivi soltanto per errori di trasporto/429/5xx, entro il job.
Un hash errato ferma nuovi trasferimenti, conserva l'errore e non viene ritentato.
Almeno5GiB liberi oltre ai byte restanti, senza duplicare gli archivi compressi.

Submitter comune `scripts/submit_job.py`, ledger `~/wearusfm_local/jobs/ledger.tsv`.
Download solo dopo checkout riuscito; audit con dipendenza `afterok` sul download.
Nessun job duplicato, retry SLURM automatico, training o valutazione predittiva.
Se il download fallisce, si leggono causa e ricevuta prima di qualsiasi seguito.

## Controlli e limiti

Il download verifica size/SHA256/MD5 e riusa soltanto file già integri; un file
completo corrotto non viene sovrascritto. L'audit rilegge tutti i trial e controlla
identità, combinazioni, frequenze, forme, canali, valori finiti, calibrazioni,
durate e duplicati. Il campione locale è distinto dalla verifica completa.

I18 target sono letture del guanto normalizzate, non gradi. Valori fuori0–1 sono
ammessi e non vengono troncati. Guanto60Hz acquisiti,256Hz distribuiti; EMG2048Hz.
I due helper ufficiali differiscono: MATLAB interpola linearmente fra gli estremi,
Python ricampiona tramite FFT a8× e tronca alla lunghezza EMG. Si conservano i file
originali e si documenta la differenza; integrità strutturale non dimostra assenza
di ritardo elettromeccanico. Nessuna scelta del ritardo dai risultati test.

Il [protocollo del confronto](seeds_benchmark_protocol_20261118.md) prepara i tre
bracci con soggetti separati15/5/5 e checkpoint30M T100-G49k prespecificato. Il fit
richiede ancora gli adattatori SEEDS, una convenzione temporale comune verificata,
il collaudo della memoria con134canali e la ricetta/costo di ottimizzazione. Questa
acquisizione non afferma che quei fit siano già pronti o già eseguiti.

## Sottomissioni e seguito

Codice congelato **b8c0718133963436b48eb2972e9ce1d4910cffea**, pubblicato suorigin e sul
riferimento remoto `deploy/seeds-acquisition-20261118`, senza aggiornaremain.
Review `docs/reviews/seeds_acquisition_review_20261118.md`: favorevole all'acquisizione,
70test verdi; confronto scientifico ancora da qualificare.

| Fase | Job | Stato iniziale verificato |
|---|---:|---|
| Checkout |64759644|PENDING, QOSMaxCpuPerUserLimit|
| Download |64759680|PENDING, Dependency sul checkout|
| Audit completo |64759758|PENDING, Dependency sul download|

L'invio audit ha perso la rispostaSSH; la riconciliazione del submitter ha trovato il job
esistente e salvato la ricevuta. Nessuna risottomissione. Tutte le ricevute sono in
`results/infra/seeds_20261118/submission/`. La coda non consuma le core-ore prenotate.

Alla conclusione leggere accounting parent, log e JSON:
`$WORK/wearusfm_runs/results/seeds_20261118/download_64759680.json` e
`$WORK/wearusfm_runs/results/seeds_20261118/audit_64759758.json`.
Verificare statusdownloadcomplete,5880file verificati, manifestSHA
`fcfb6308b5155fdd14e63109b200a2208c88326c602e8a963c746b76e14856d7`,
auditfull_dataset_passtrue,5850trial+25calibrazioni,COMPLETED0:0 e consuntivi.
UnFAILED/TIMEOUT/hashdifforme richiede diagnosi, non riavvio automatico. Riportare anche
le durate e le anomalie senza descrivere i soli checksum come prova di allineamento.
Raccogliere gli aggregati sulMac e aggiornare registro/risultati concommit/push.
Nessun monitor persistente o automazione di completamento è stato creato.

## Recupero diagnosticato del 19/11/2026

Il checkout64759644 è COMPLETED0:0; il download64759680 è FAILED1:0 dopo772s.
L'audit64759758 è CANCELLED prima di partire per la dipendenza fallita.
La ricevuta registra **1.049/5.880 file integri,10.651.199.162byte**. L'unico errore
è HTTP502 per `subj05/detop_exp01_subj05_Sess2_05_01.mat`: tre risposte502 in5,28s,
zero byte ricevuti. Nessun errore di checksum e nessuna evidenza di dati corrotti.

Il19/11 lo stesso oggettoOSF `vw2cq`, versione1, è stato riscaricato con un controllo
mirato sulMac:13.010.647byte, SHA256 eMD5 identici al manifest congelato. Anche i
metadati ufficiali correnti coincidono. Evidenze, ricevuta originale e log in
`results/infra/seeds_20261118/recovery_20261119/`. Il payload diagnostico resta fuoriGit.

È quindi motivata una **ripresa dell'acquisizione già approvata**, senza modifiche
al downloader, al manifest o ai criteri. Si riusa il checkout congelato b8c0718,
verificato pulito suLeonardo con lo stesso hash del manifest; non esistono altri jobSEEDS
attivi. Il downloader rilegge gli hash dei1.049 file completi e li salta; scarica i
**4.831 mancanti,49.086.076.114byte**, compreso l'oggetto temporaneamente indisponibile.
I file completi non vengono sovrascritti. Un nuovo errore richiede un'altra diagnosi,
senza retrySLURM automatici. I32 test del downloader sono stati rieseguiti: tutti verdi.

| Ripresa | ID logico | Risorse | Tetto |
|---|---|---|---:|
| Download mancante e verifica dei presenti | SEEDS-download-resume-20261119 |2CPU,4GiB,4h|8coreh|
| Audit completo, afterok della ripresa | SEEDS-audit-resume-20261119 |2CPU,16GiB,2h|4coreh|

Consuntivo precedente0,456111coreh, **massimo complessivo consuntivato più residuo
12,456111coreh**, zeroGPUh. Si estende di0,289444coreh il vecchio tetto12,166667;
non si sommano le prenotazioni inutilizzate dell'audit cancellato. Nota contabile:
lo scheduler ha allocato2CPU al checkout, pur richiedendone1; il consuntivo usa le
2CPU effettive. Preventivo e diagnosi sono pubblicati prima dei nuovi invii tramite
submitter/ledger comuni. Nessuna nuova scelta scientifica, fit o pretraining suSEEDS.

**Invii del19/11 confermati:** download **64827659**, giàRUNNING nello snapshot delle
10:33CET; audit **64828230**, sottomesso con `afterok:64827659`. Piano e diagnosi
pubblicati prima degli invii nel commitf7fc04df, SHAoriginverificato. Ricevute dei nuovi
IDlogici e stato in `results/infra/seeds_20261118/recovery_20261119/`.

Il prossimo controllo deve leggere i soli nuovi report
`$WORK/wearusfm_runs/results/seeds_20261118/download_64827659.json` e
`$WORK/wearusfm_runs/results/seeds_20261118/audit_64828230.json`, insieme a log e
accounting dei parent. Il download deve certificare5.880file verificati e lo stesso
manifest; l'audit deve essere completo, noncampione. Non duplicare i due job, non
confondere il download riuscito con una verifica cinematica già conclusa e non
avviare i fit dal solo esito di acquisizione. Nessuna automazione persistente.

### Esito della ripresa: limite richieste OSF, nessun ulteriore invio

Il controllo successivo corregge lo statoRUNNING sopra: **64827659 FAILED1:0** dopo383s,
**64828230 CANCELLED** prima di partire. La ripresa ha riusato i1.049file integri e
scaricato altri207: **1.256/5.880file verificati**,12.733.870.349byte. L'oggetto che
aveva dato502 è stato superato conhashvalidi. Ora due trial disubj06 hanno restituito
**HTTP429 Too Many Requests** per tre tentativi ciascuno, senza ricevere byte.

Questo è un limite di frequenza imposto dalla fonte: **ulteriori invii fermi**.
Prima di un'altra ripresa serve una correzione operativa verificata del downloader:
ritmo globale più lento, concorrenza ridotta e rispetto di `Retry-After`, senza
aggirare il limite né cambiare fonte o hash. Il codice attuale ritenta dopo1e2s:
troppo presto per il limite osservato. Nessuna modifica è stata ancora applicata,
nessun terzo download inviato. Ricevuta/log e diagnosi finale nel medesimo percorso.

Consuntivo dell'intera acquisizione finora **0,668889coreh,0GPUh**; nessun auditMAT
completo eseguito. Il download parziale non certifica disponibilità del benchmark.
