# Tokenizer: tre training e valutazioni autorizzati — 03/11/2026

Il collaudo tecnico **63370721 è COMPLETED 0:0**, 199 secondi su quattro A100:
**0,221111 GPU-ore**. I tre bracci hanno elaborato gli stessi 64 batch globali da32,
2.048 finestre per braccio, con parità di stream/query/target/supporto della posa.
Gli hash dei quattro report di rank coincidono con il report aggregato. I file di rank
sono pubblicati compressi senza perdita, con hash dei byte originali e degli archivi;
si tratta di statistiche aggregate e tracciati tecnici, non segnali wearable.

Si esegue il confronto già autorizzato: **T25-L, T100-L, T100-G da zero fino a5.000,
poi ciascuno dal proprio stato fino a19.618**. Il checkpoint del collaudo tecnico non esiste
e non viene riutilizzato. Il confronto mantiene WSD totale49.046/warmup1.962, stessi dati,
seed0, batch32, quattro rank, sei worker/rank; ancore0,2+posa1, JEPA0 e raggio query0.
Le sole differenze registrate sono finestra nominale25/100ms e identity/GELU.

Simone ha autorizzato lancio diretto e superamento del precedente budget. Nessun nuovo
consenso è necessario. Si usano submitter e ledger comuni; un solo operatore, nessun
duplicato o retry cieco. Nessuna nuova loss o estensione della domanda scientifica.

## Tempi e tetti pubblicati prima delle allocazioni

| Braccio | Tempo medio aggiornamento nel collaudo | Rapporto con25-L |
|---|---:|---:|
| T25-L |0,695264s|1,000000|
| T100-L |0,699735s|1,006430|
| T100-G |0,705888s|1,015279|

La media esclude i primi8 aggiornamenti, conserva massimi fra rank e non include le
diagnostiche aggiunte. Non misura da sola il limite del loader. Per stimare il training
si usa il tempo **totale storico B5000 di3.814s**, che include pannelli, loader e checkpoint,
moltiplicato per max(1, rapporto del collaudo). Si aggiungono20% e600s per fase, arrotondando
ai15min superiori. La riserva600s comprende avvio/verifiche e i300s sottratti dal timeout
interno del trainer per chiudere i file. Sono margini operativi, non garanzie sui tempi.

| Allocazione | Numero | Risorse per job | Tetto complessivo |
|---|---:|---|---:|
| Nuovo checkout congelato |1|2CPU,4G,10min|0,333333coreh CPU-only|
| Training0→5.000 |3|4GPU,32CPU,480G,90min|18GPUh|
| Continuazione5.000→19.618 |3|4GPU,32CPU,480G,240min|48GPUh|
| Teacher posa train+val |6|1GPU,8CPU,100G,45min|4,5GPUh|
| Teacher P1 endpoint |3|1GPU,8CPU,100G,30min|1,5GPUh|
| Teacher P2 endpoint |3|1GPU,8CPU,100G,45min|2,25GPUh|
| Sonde, ridge, benchmark, controlli e confronti CPU |politica JSON|fino a8CPU,30G; shard fino a60min|113coreh CPU-only|

**Allocazioni GPU future:74,25GPUh**; sommando il preflight effettivo: **74,471111GPUh**.
Con la vecchia riserva preflight2 il tetto sarebbe76,25. Il residuo storico63,2925 è un
riferimento contabile superato dall’autorizzazione aggiuntiva. Non sommare nuovamente
i5.000 passi al totale19.618. CPU-only futura113,333333coreh; GPU-job coreh futura594,
contabilizzata separatamente. Il checkout già concluso ha consumato0,018889coreh CPU-only.

I dettagli per job, formula e quote sono in
[full_resource_policy.json](../results/infra/tokenizer_execution_20261103/full_resource_policy.json).
Una coda non consuma GPU-ore; il consuntivo sarà quello di `sacct` sui parent, senza duplicare
batch/extern. Il costo massimo dichiarato non significa che ogni job consumerà il tetto.

## Catene e ripresa

[training_plan.json](../results/infra/tokenizer_execution_20261103/training_plan.json) pinna
configurazioni, quattro ingressi, report del collaudo e tutti i file scientifici `src/` più
CLI/preflight: i loro byte coincidono con il collaudo `ef8016a`. Il commit esecutivo nuovo
aggiunge soltanto orchestrazione, valutatori e documentazione. Il checkout detached nuovo
`tokenizer-training-20261103` è creato dal job CPU versionato, poi verificato prima dei lanci.

Si possono mettere in coda i tre training iniziali e le rispettive continuazioni con
`afterok` sul job iniziale ancora noto a SLURM. Una continuazione controlla anche ricevuta,
SHA, configurazione, firma dati, stato AdamW, contatori e RNG dei quattro rank. Nessuna
dipendenza da un risultato scientifico favorevole a5k: proseguono tutti i bracci se integri.
STOP, timeout o dose incompleta falliscono il wrapper e impediscono una ricevuta completa.
Se un ID completato è già scaduto da SLURM, si usa la sua evidenza verificata, non un nuovo
invio della fase completata.

Il trainer conserva `checkpoint_step0005000.pt` e `checkpoint_step0019618.pt`. Il primo
rimane immutato durante la continuazione; il checkpoint finale `checkpoint.pt` è lo stato
di ripresa. File nominato e finale possono avere serializzazione/RNG distinti: si controllano
configurazione/dose e uguaglianza degli stati addestrati, non uguaglianza dei byte fra i due.
Ogni ricevuta conserva uno snapshot immutabile di `config.json` fuori dalla cartella del run,
così la valutazione a5k resta verificabile mentre la continuazione aggiorna la configurazione.

## Valutazioni e lettura

I tre moduli nuovi sono revisionati: controller teacher, ridge posa e sonda convergente.
[Matrice e criteri](tokenizer_evaluation_execution_20261103.md): teacher posa su40 utenti train
e15val a entrambe le dosi; classificazione P1/P2 a19.618, EPN-612 distinta e principale.
Gli hash di checkpoint e cache si congelano quando i file esistono. Le estrazioni a5k possono
convivere con la continuazione leggendo il checkpoint nominato e lo snapshot della sua fase.
I piani dei fit CPU vengono completati dopo l’intera griglia di cache della dose; prima della
ridge si legge il benchmark e si applica il tetto pubblicato, senza estensioni implicite.

Nessun risultato scientifico è ancora disponibile. Il collaudo attesta eseguibilità e
comparabilità tecnica; non dimostra che100ms oGELU migliorino posa o classificazione.

Verifica finale prima del deploy:107test dei quattro moduli nuovi passati; review indipendente favorevole su wrapper, launcher, risorse e valutatori. [Ricevuta](../results/infra/tokenizer_execution_20261103/training_verification.json).

## Sottomissioni effettuate

Codiceafb644ea10ecc2a716160b4d8a4904e805c366fb pubblicato e verificato. Checkout63385655
COMPLETED0:0 in32s; tre training e tre continuazioni sottomessi tramite ledger comune.

| Braccio | Job5.000 | Job19.618 | Stato dopo invio |
|---|---:|---:|---|
| T25-L |63385800|63386315|Priority; continuazione Dependency|
| T100-L |63386190|63386337|Priority; continuazione Dependency|
| T100-G |63386228|63386403|Priority; continuazione Dependency|

Le continuazioni sono già vincolate al proprio training conafterok. Le valutazioni non sono
ancora sottomesse: attendono i pin dei checkpoint effettivi. [Seguito operativo](tokenizer_handoff_20261103.md).

## Lettura intermedia delle loss su richiesta di Simone — 03/11, 22:29 CET

I tre job di continuazione sono RUNNING. Nel grafico fornito JEPA ha peso zero nei
config effettivi: la sua salita non è un peggioramento dell'obiettivo ottimizzato.
Totale = posa + 0,2 × somma delle ancore. Il tratto5k–14.500 è a LR costante1e-4,
prima del decay WSD. Nessuna ricetta o dose cambiata per questa lettura.

Raccolti fixed_eval.jsonl e soli aggregati di pose_panel.jsonl in sola lettura.
Si usa l'ultimo passo comune dello snapshot,14.500. Entrambe le diagnostiche sono
sul training, student in eval, non le valutazioni teacher su utenti esclusi.
Pannello posa:32utenti,256finestre, stesso hash in tutti i bracci e passi
04d32d6491d7bb29acdde18898b938f13effc513aded506af7e4e8d9f6a63a6f.

| Braccio | Totale fisso 5k → 14.500 | MSE pannello posa 5k → 14.500 | Varianza temporale predetta 5k → 14.500 |
|---|---|---|---|
| T25-L | 2,210508 → 1,817853 (−17,76%) | 1,014895 → 0,967613 (−4,66%) | 0,024722 → 0,029588 |
| T100-L | 2,181777 → 1,817269 (−16,71%) | 1,004347 → 0,959831 (−4,43%) | 0,025758 → 0,026412 |
| T100-G | 2,135767 → 1,790112 (−16,18%) | 0,996475 → 0,923481 (−7,33%) | 0,026602 → 0,024422 |

Non emerge divergenza dalle misure esaminate: il progresso è presente sui batch fissi.
Restano oscillazioni visibili anche sui pannelli fissi, soprattutto per la posa:
non attribuire tutta l'irregolarità del grafico alla composizione dei batch correnti.
Il miglioramento della posa è moderato; GELU è favorevole sulla MSE di questo pannello,
ma la sua variabilità temporale predetta non cresce. Non dedurre recupero del movimento
o migliore generalizzazione. Non è salvata la varianza target: nessun rapporto con
pannelli storici. Confronti intermedi descrittivi, un seme, nessun IC o scelta del
checkpoint. Review indipendente concorde su pesi, schedule, lettura e limiti.

[Dati aggregati](../results/infra/tokenizer_execution_20261103/loss_diagnostic_20261103.json),
[grafico corretto](../results/infra/tokenizer_execution_20261103/loss_diagnostic_20261103.png)
e script di rendering archiviati. Nessun nuovo job, training, monitor o test GPU.
