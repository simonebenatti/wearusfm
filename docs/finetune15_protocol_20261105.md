# EPN a 15 epoche: frozen, fine-tuning e scratch

**Preparazione per la review di Claude; nessun job sottomesso, neppure il benchmark.**
Simone approva di preparare tutto, farlo revisionare da Claude e poi lanciare.
L'owner resta GPT, con submitter e ledger comuni. Questa consegna non cambia
i runtime della continuazione lunga, i risultati a cinque epoche o le altre linee.

## Domande e motivazione

A cinque epoche: frozen70,575%, fine-tuning93,766%, scratch92,363%;
FT−scratch+1,403pp IC95[0,930;1,837]. F-R2 scatta con scratch[4,5,5,5].
Il frozen è12,002pp sotto la logistica storica82,577%, con migliore validation
all'ultima epoca in tutti i fold. Il seguito deve chiarire due domande:

1. La nuova testa frozen recupera con una ricetta di ottimizzazione più lunga?
2. Il vantaggio del pretraining rispetto a scratch resta a quindici epoche?

F-R2 aveva proposto FT/scratch; **l'aggiunta di frozen è esplicitamente richiesta
e approvata da Simone il05/11**, dopo aver visto il suo risultato. È sviluppo
guidato dai risultati precedenti, non una conferma indipendente pre-registrata
prima della prima campagna. I criteri sotto sono fissati prima dei nuovi fit.

La review Claude `0968bd1` approva i risultati a cinque epoche e propone un
seguito FT/scratch. Il presente piano comprende anche frozen per la successiva
istruzione di Simone; non attribuisce questa estensione alla vecchia F-R2.
L'autorizzazione operativa e al superamento del vecchio budget è già disponibile;
resta da chiudere la review di questo codice e preventivo, non da richiedere
un secondo consenso per lo stesso esperimento.

## Disegno fissato

Stessi dati EPN interni P2,314soggetti/47.100finestre, fingerprint
`ef0090cb748e9eb4b9e9f474f084b8dbbbb6a9ca5f6594d91ceaf216204ee0c0`.
Esclusa l'unione dei test ufficiali semi0/1/2; quattro fold soggetto seed20261015,
validation10% dei soggetti restanti. Nessun test ufficiale valutato.
Preprocessing e calibrazione non supervisionata delle prime dieci finestre
per soggetto identici; due crop4s, readout storico pool768, testa LayerNorm+Linear.
Non è una valutazione su nuovi utenti senza calibrazione.

| Regime | Inizializzazione encoder | LR encoder candidati | LR testa candidati |
|---|---|---|---|
| Frozen | Teacher T100-G30M a19.618, congelato | Nessun aggiornamento | 1e-3 / 1e-2 |
| Fine-tuning | Stesso teacher, encoder aggiornabile | 1e-5 / 1e-4 | 1e-3 / 1e-2 |
| Scratch | Architettura identica, inizializzazione casuale seed0 | 1e-4 / 1e-3 | 1e-3 / 1e-2 |

**24 candidati:4fold ×3regimi ×2LR.** Non si conserva soltanto il LR vincente
a cinque epoche. Identica testa iniziale, batch8, seed0, AdamW weight_decay0,01,
clipping1, stessa cross-entropy pesata e nessuna nuova augmentation.
Il file `src/wearusfm/harness/finetuning.py` rimane byte-identico.

Quindici epoche complete, warmup fino100update e cosine con orizzonte15epoche.
Si riparte dalle inizializzazioni originali, non dai modelli downstream
selezionati a cinque epoche. Il riferimento teacher resta il checkpoint19.618
SHA256 `a31c895061c1f273eaac2d796fc00cf76cffa4318b521a20be4d6ba65b63a056`,
letto in sola lettura; non si passa al checkpoint del pretraining lungo.

**15 contro5 cambia anche la traiettoria del learning rate**, non soltanto il
numero di aggiornamenti. La prova confronta due ricette con diverso orizzonte;
non è un'ablazione che isola causalmente la sola dose. Non garantisce convergenza.

## Selezione, persistenza e lettura

Tutti i candidati completano15epoche, salvo divergenza numerica esplicita o
errore operativo. Migliore epoca su validation, pareggi alla prima; migliore LR
sulla stessa validation, pareggi al candidato inferiore. C0 salva il risultato
senza predire held; C1 verifica ricevuta/pesi di C0, seleziona dopo entrambi e
predice held una volta. Nessuna selezione o ripetizione guidata dai punteggi held.

È un **secondo sguardo sugli stessi 314 soggetti held di sviluppo**, con modelli
nuovi per la dose 15, dopo aver letto le OOF della dose 5. «Una volta» vale per
ciascuna nuova cella, non per tutta la storia del progetto. Il confronto non è
una conferma su un test intatto; la dose non viene selezionata in base al held.

Nonfiniti restano esiti espliciti; OOM/errori software non diventano divergenze
scientifiche. Se manca una cella, il relativo confronto non è valutabile:
nessuna eliminazione silenziosa di soggetti, fold, candidati o repliche bootstrap.
Report, curve, OOF e pesi persistono sotto WORK in directory nuove.

Primari a15: **FT−scratch e FT−frozen**, entrambi riportati. Secondari:
15−5 per ciascuno dei tre regimi e **Δ15−Δ5**, dove Δe=FTe−scratche.
Il contrasto `transfer_gap15_minus_gap5` misura il cambiamento del vantaggio
FT−scratch: riporta Δ5, Δ15 e la loro differenza con IC appaiato. Si richiede
pairing esatto di righe/target/soggetti/fold e ricontrollo delle OOF5 originali.
La logistica storica82,577% resta un
riferimento descrittivo distinto: non sostituisce frozen e non riceve un
intervallo appaiato ricavato dal solo punteggio aggregato.

Metrica: macrorecall pooled sulle OOF; IC95% percentile descrittivi da2000
bootstrap appaiati per soggetto, seed20261102. Stessi campionamenti per i
regimi e i contrasti pertinenti, nessuna replica scartata. Per Δ15−Δ5 si
sottraggono i contrasti entro ogni stessa replica,
non gli estremi degli IC e non campionamenti indipendenti fra le due dosi.
Gli IC sono condizionati a
modelli/fold, senza riaddestramento o incertezza fra semi. Si riportano curve,
scelte, distribuzioni per soggetto e recall per classe; nessun vincitore unico
scelto ex post fra i contrasti. Le soglie V-R1 P1/P2/posa non vengono importate.

Lettura a due lati: un IC positivo FT−scratch sostiene il vantaggio del
pretraining in questa ricetta; un IC che include zero lascia il vantaggio
irrisolto; un IC negativo favorisce scratch. Non sono affermazioni sui rispettivi
ottimi. Per frozen15−frozen5 vale la stessa lettura della differenza;
avvicinarsi alla logistica sostiene un ruolo dell'ottimizzazione, senza
identificare quale parte della ricetta abbia causato il recupero.

Per Δ15−Δ5: un IC interamente negativo sostiene un restringimento del
vantaggio FT−scratch; interamente positivo, un ampliamento; se include zero,
il cambiamento resta irrisolto, non si conclude equivalenza. Δ5 e Δ15 sono
riportati comunque, senza scegliere il più favorevole. L1 resta valido anche
qui: confrontiamo pipeline selezionate fra due LR, non ottimi di LR né ricette
di ottimizzazione identiche fra FT e scratch.

Per tutti i regimi, diagnostica sulla sola validation dei quattro candidati
selezionati: almeno3fold con prima epoca migliore alla15 **oppure** mediana
validation15−validation14 ≥0,005 (0,5pp). Se incompleto/divergente: non valutabile.
Flag falso non prova convergenza; flag vero non avvia altre epoche e non
garantisce ulteriori guadagni. È l'adattamento dichiarato della diagnostica
precedente, non una soglia di utilità o una regola di arresto durante il fit.
Si mostrano inoltre le validation dei selezionati a 5 e 15 per fold/regime,
senza convertirle in un nuovo gate. **La catena termina alla dose 15**: nessuna
terza dose, nuova griglia o nuovo seme automatici, anche se la migliore epoca
fosse ancora l'ultima. Dopo questa lettura la priorità è ampliare i dataset.

## Risorse: emendamento operativo esplicito

Il precedente microbenchmark sovrastimava il costo medio dei fit completi di
circa1,8×. Ora abbiamo tutti i24fit finiti: li usiamo come evidenza operativa
pinnata, mantenendo il nuovo benchmark per rallentamento, memoria e provenienza.
Non si usano i punteggi per scegliere quali tempi prendere.

Per ogni regime r:

```text
M_r = massimo elapsed SLURM degli otto candidati completi a cinque epoche
q_r = max(1, steady_seconds_per_update_nuovo / steady_seconds_per_update_v2)
F = 1,5 × load_seconds + projected_full_preprocessing_seconds
D = max(0, F_nuovo − F_v2)
P_r = ceil(3 × M_r × 1,25 × q_r + 600 + D)
gate: P_r <= walltime_r − 300 secondi
```

`steady` misura32update dopo4warm-up, sul medesimo campione train-only del
vecchio benchmark, batch8. Il nuovo benchmark non può ridurre la proiezione
se risulta più veloce. La triplicazione include già i costi fissi del vecchio
fit; si aggiungono25%,600s ed eventuali aumenti dei costi fissi misurati.
La vecchia proiezione assoluta dal microbenchmark resta diagnostica e non è
un secondo gate concorrente. Questo cambio di formula passa dalla review Claude.

| Regime | Max osservato5epoche | Proiezione15 con q=1,D=0 | Allocazione/job | Job | Tetto GPUh |
|---|---:|---:|---:|---:|---:|
| Frozen | 4.519s | 17.547s,4,874h | 6h | 8 | 48 |
| Fine-tuning | 13.359s | 50.697s,14,083h | 16h | 8 | 128 |
| Scratch | 13.446s | 51.023s,14,173h | 16h | 8 | 128 |
| Totale fit | | **265,038GPUh** | | **24** | **304** |

La semplice triplicazione del consuntivo è204,815GPUh: stima indicativa,
distinta dalla proiezione prudenziale e dal tetto prenotato. Tutti i job usano
1GPU/8CPU/100GiB; split fisso, nessun raggruppamento o troncamento automatico.
Benchmark nuovo30min: massimo0,5GPUh/4coreh. Due checkout da10min,
2CPU prudenziali: massimo0,667coreh. **Tetto totale304,5GPUh/2.436,667coreh**.

Prima dei fit: benchmark completo0:0, pin/hash/codice/ricetta/input/checkpoint
validi,32update+4warm-up, gate tempo ricalcolato per regime e memoria sotto85%
di host/GPU come prima. L'equivalenza dei sorgenti scientifici5epoche viene
controllata; il nuovo controller ha identità distinta. Se un gate fallisce si
ferma la fase interessata e si documenta il problema, senza retry cieco,
riduzione di epoche/batch/griglia o uso della vecchia ricevuta come autorizzazione.

## Esecuzione e consegne

Piano, preventivo e24ID logici distinti in `results/infra/finetune15_20261105/`.
Namespace FT15, ownerGPT, submitter `scripts/submit_job.py`, ledger comune.
Prima review Claude, poi stati convergenti e pubblicazione con SHA remoto
verificato; checkout benchmark immutabile, misura, pin/preventivo misurato
pubblicati e secondo checkout immutabile per i fit. C1 dipende dal proprio C0.
Un checkout già verificato e rimosso dalla memoria SLURM resta prerequisito
soddisfatto: si conserva la ricevuta senza duplicarlo per ricreare afterok.

Output aggiuntivi:24pesi ≈2,73GB/2,54GiB, oltre a curve/report/OOF. Tutto su
WORK, nessuna dipendenza dalla purge di SCRATCH per i risultati. Prima degli
invii quota e inode reali aggiornati, almeno6GiB liberi e controllo del margine
per l'intera matrice; non è una prenotazione della quota condivisa. Verificare
di nuovo il checkpoint sorgente e l'assenza di ID già presenti nel ledger.

Nuovo aggregatore15 già parte della consegna: verifica SHA del codice/piano
effettivamente eseguiti, manifest, ricevute, selezione, dose e OOF5/15 prima
dei contrasti. I pin di esecuzione effettivi saranno pubblicati dopo la review
e la misura; non sono sostituiti da SHA fittizi nella preparazione.
Nessun monitor automatico. Nessuna ulteriore autorizzazione per la stessa
catena dopo convergenza; nessun lancio prima della review richiesta da Simone.

## TIMEOUT, guasto del nodo e ripetizioni

Il launcher richiede `--no-requeue` e rifiuta `SLURM_RESTART_COUNT` diverso da
zero prima del setup. Un TIMEOUT, NODE_FAIL, OOM o errore software non è un
risultato scientifico completo. Si conservano directory parziali, log,
ricevute e accounting; si diagnostica il problema prima di qualsiasi recupero.
Le dipendenze afterok non vengono aggirate: se fallisce C0, C1 non è eseguibile.
Una matrice incompleta non produce i contrasti finali su un sottoinsieme scelto.

I checkpoint dei candidati contengono il modello selezionato, non uno stato
completo di optimizer, schedule e RNG: **non consentono una ripresa fedele**.
Questa versione non implementa resume e può perdere il lavoro del candidato
interrotto. Non sovrascrive né riutilizza come completo un output parziale.

Un eventuale nuovo tentativo richiede prima diagnosi, emendamento concreto
revisionato, costo aggiuntivo e nuovi ID/output collegati al tentativo fallito;
non è incluso nei 24 job del piano corrente. Se ammesso, riparte dalle
inizializzazioni originali per tutte le 15 epoche. Un held già prodotto viene
preservato e il suo stato dichiarato nell'emendamento, senza ripeterlo in
silenzio o scegliere fra tentativi in base al punteggio. Nessun retry,
requeue, cambio di protocollo o terza dose automatici.

## Priorità successiva: uscire dal solo EPN

Simone richiede di passare ad altri dataset dopo questo giro. EPN resta il banco
di sviluppo del fine-tuning; le sonde storiche su altri dati non sostituiscono
un confronto FT/scratch sugli stessi dati. La prossima fase deve comprendere
altri task di classificazione e una regressione diversa dalla posa emg2pose
già supervisionata nel pretraining. Inventariare prima target, dati e split
effettivamente disponibili, distinguendo dataset presenti/esclusi dal pretraining.

Per ogni nuovo dataset si fissano prima dei risultati popolazione, metrica,
ricetta e pari opportunità di tuning per frozen/FT/scratch. Una ricetta EPN non
viene assunta ottimale ovunque. Dati già usati per lo sviluppo restano sviluppo;
eventuale conferma indipendente va identificata esplicitamente. Nessuna nuova
campagna su altri dataset è inclusa nei24fit o nel budget di questa consegna.

[Ricognizione e sequenza proposta](downstream_multidataset_next_20261105.md):
UCI-EMG escluso dal pretraining, NinaPro DB2 per diverso montaggio/compito,
Hyser-forza subordinato all'audit dei bersagli. Nessuno dei tre è già autorizzato
come nuova campagna dal solo inserimento in questa nota.
