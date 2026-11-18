# Preparazione offline dello screening 300M — 4 novembre 2026

Simone conferma il guasto del nodo di login di Leonardo e chiede di avanzare sui
requisiti scientifici mentre il cluster non è accessibile. Riferimenti della
review Claude: `befab4a` e complemento `2f3207b`. Il collaudo tecnico rimane quello
già approvato al commit `05d4bf0`; questa consegna non ne cambia i sorgenti,
il piano, il launcher, la dose o le risorse.

| Punto | Preparazione locale | Limite prima dell'esecuzione |
|---|---|---|
| S4, ripiego memoria | Già pubblicato prima del collaudo: due microbatch, batch globale32 e4rank. | Se necessario richiede test equivalenza, collaudo separato e costo; non è un retry automatico. |
| F5/F8, test dell'update | LR di stress confrontato direttamente con il picco; RNG dell'oracolo avanzato indipendentemente. | Test CPU sul percorso dell'update; non sostituiscono misure GPU. |
| S1, training5000 | Nuovo wrapper `scripts/scaling_train.py` e contratto dei sorgenti rispetto ad `afb644e`, stessi dati/ricetta e preset300m. | Richiede artefatti reali di collaudo, memoria di produzione, readiness delle valutazioni, review e preventivo. |
| D896/pool1792 | Nuovo `scripts/scaling_evaluation.py`: checkpoint, teacher, configurazione, cache posa e classificazione, estrazioni alla dose5000 per i due bracci. | Sono validatori ed estrazioni; non equivale ad avere già pronti tutti i lettori e gli aggregatori. |
| S8, lettura a due lati | Protocollo, policy JSON e `scripts/scaling_reading.py`, con test di segni, confini, matrice e costi. | Consuma riepiloghi già verificati; non calcola le sonde né certifica da solo la provenienza dei risultati. |
| S2, controllo30M | Pin storici e requisito di disponibilità fino a estrazioni/confronto. | Esistenza, retention e spazio per una copia vanno ricontrollati su Leonardo. |

## Identità e regressioni

Il controllo 30M è T100-G a5.000, teacher EMA, D384/pool768. Il candidato è
T100-G300M a5.000, teacher EMA, D896/pool1792. La derivazione della configurazione
deve differire solo nei campi LADDER e nelle destinazioni operative consentite;
restano WSD49.046, warmup1.962, picco1e-4, clip1, batch globale32, quattro rank,
sei worker, posa1 e ancore0,2, JEPA0. Non si confrontano dosi5.000 e19.618.

I test della valutazione confrontano i risultati del validatore D384 con quello
storico, esercitano D896 e i fallimenti su supporti, target, righe, soggetti, fold,
precisione e finitezza. Le etichette testuali EPN non passano da `isfinite`.
Il caricamento del teacher è provato su un checkpoint reale piccolo in CPU;
la guardia pubblica separata impone i preset completi della campagna.

Il riepilogo del controllo30M va letto dallo snapshot di fase in
`$WORK/wearusfm_runs/results/tokenizer_training_20261103/T100-G/5000/summary.json`,
con pin proprio, come config e ricevuta. Il `summary.json` mobile nella directory
del training è stato sostituito dalla continuazione a19.618 e non è una fonte
valida per il riepilogo5.000. Il checkpoint nominato resta in scratch; la scadenza
indicata13/12/2026 è un limite operativo da verificare, non una garanzia di retention.

## Cosa resta prima del training scientifico

Il collaudo approvato può essere sottomesso quando l'accesso torna operativo,
senza attendere queste implementazioni. Il training a5.000 ha invece questi gate:

1. Collaudo concluso, hash/parità/completezza verificati e risorse misurate.
2. Margine di memoria della produzione, comprese diagnostiche e prefetch.
3. Lettori ridge e sonde per la nuova matrice a due taglie, aggregazione appaiata,
   controllo secondario a dimensione pareggiata e audit di convergenza.
4. Controllo30M e input preservati fino alle letture; pin completi dei nuovi artefatti.
5. Benchmark CPU, preventivo, launcher congelati e review della catena completa.

Non si crea una ricevuta di readiness per requisiti che non sono completati.
Le nuove guardie sono progettate per rifiutare documenti incompleti o non coerenti;
non trasformano una preparazione CPU in una verifica eseguita sul cluster.

F1–F4 e F6 restano facoltativi differiti per preservare il collaudo approvato.
Il rapporto dei tempi (F7) è già fissato sui112 update misurati. Le loss e le
baseline di memoria disponibili nei record saranno analizzate senza inventare
metadati runtime mancanti o trasformare un conteggio di byte in prova di memoria.

## Stato operativo

Nessun comando Leonardo, job, allocazione o monitor eseguito da questa preparazione
offline. Il piano di invio del collaudo resta in
`docs/scaling_300m_execution_20261104.md`; l'ultimo stato operativo registrato è
`results/infra/scaling_300m_20261104/submission_status.json`.

Le verifiche CPU e la review finale della consegna sono registrate in
`results/infra/scaling_300m_20261104/offline_verification.json` e
`docs/reviews/scaling_scientific_offline_critic_20261104.md`.

Validazione finale: **248 test passati in19,52secondi**, eseguiti dal critico sulla
suite integrata; nove hash di codice/policy/test ricontrollati dopo l'esecuzione.
Diff vuoto rispetto05d4bf0 per src/trainer, codice e launcher del collaudo, piano
e resource_policy. Nessun rilievo aperto alla consegna offline per Claude.
