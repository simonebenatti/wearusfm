# Tokenizer posa5k: validazione, benchmark e ridge — 03/11/2026

Le estrazioni teacher dei tre bracci sono COMPLETED0:0:
63399428/63399479/63399484,1853/1814/1814s. Consuntivo **1,522500 GPU-ore**,
12,180000 core-ore allocate conGPU, entro la riserva2,25GPUh.

Sei cache e sei provenienze ricalcolate con sha256sum; provenienze originali raccolte
byte per byte. Dose5000, teacher effettivo, configurazioni, checkpoint immutabili e
conteggi1907train/18548val coerenti. Firme target/metadati/maschere centrali identiche
fra i tre bracci e confrontate con i riferimenti storici dall'estrattore.

Le ricevute complete sono circa3,7MB ciascuna per l'inventario dei file sorgente:
rimangono ai percorsi remoti e sono pinnate col loro SHA integrale. La raccolta locale
conserva il prefisso originale e un riepilogo esplicitamente parziale, oltre a dimensioni
e hash integrali. **Il piano CPU usa le ricevute complete remote**, mai i riepiloghi.
Nessuna cache di segnali/predizioni o dato grezzo entra in Git.

## Piano e risorse prima dei lanci

Piano `results/infra/tokenizer_evaluation_20261103/pose_readout_plan_5000.json`,
SHA256 `9efbf8279b1a6fdbcd551bae2326e36b74f551c7469f5cbd46842c49906b39ba`.
Validato dal costruttore scientifico esistente, stessi40utenti e duefold alternati,
crop6, normalizzazioneFP32 e ridgeFP64; nessuna modifica del reader afb644e.

Nuovo checkout `tokenizer-posecpu5k-20261103`:2CPU/4G/10min, **0,333333coreh
CPU-only aggiuntive**. Il totale di riferimento della matrice CPU con tre checkout
training/estrazioni5k/reader5k diventa114coreh, distinto dal checkout del preflight.
L'adattamento operativo è già autorizzato. Pubblicazione e verificaSHA prima del deploy.

| ID logico | Fase | Risorse | Tetto core-ore |
|---|---|---|---:|
| TOKENIZER-pose5000-validate-20261103 | Verifica tutti i sei artefatti, supporti e ricevute |2CPU/30G/30min|1|
| TOKENIZER-pose5000-benchmark-20261103 | Timing solo train T25-L, nessun punteggio val |8CPU/30G/15min|2|
| TOKENIZER-pose5000-ridge-20261103 | Fit dei tre bracci e confronto appaiato |8CPU/30G/30min|4|

Queste7coreh sono già comprese nella matrice113coreh di analisi; non si aggiungono
una seconda volta. Tutto CPU, nessunaGPU. Validate e benchmark possono partire in
parallelo dopo checkout verificato; ridge **non** viene prenotata tramite afterok
prima di leggere entrambi i report. Il confronto è già incluso nella ridge.

## Cancello temporale fissato prima del benchmark

V = ElapsedRaw del job validate completo; S = statistics_seconds del benchmark;
E = three_eigensystems_seconds; R = max(10,18548/1907)=10.
Calcolare **ceil al minuto di 1,5 × [2V + 3 × ((1+R)S +13E)] +120 secondi**.

Il fattore13E è una riserva empirica per lo scoring della griglia alpha, non13
decomposizioni: due solver CV sono preparati una sola volta, il terzo serve al refit.
La formula include I/O/riletture e una proxy prudente per predizione/riassunti val,
con margine50% e120s di finalizzazione; è una previsione, non una garanzia matematica.

Lanciare la ridge soltanto se entrambi i job sono COMPLETED0:0, report completi e
coerenti con piano/dose/source, benchmark8thread e **stima <=1800s**.
In caso contrario diagnosticare e fermare quella fase, senza retry cieco,
allargamento silenzioso del tetto o consultazione degli score val per cambiare ricetta.

## Consegna e monitoraggio

Report e consuntivi saranno raccolti quando richiesti; screening5k resta descrittivo,
senza selezionare il vincitore o impedire le continuazioni19618 già in coda.
Simone il03/11 chiede: dopo questa fase e i lanci, **disattivare i controlli di
completamento in background**. Completare il cancello e l'invio della ridge in
questa attività, quindi eliminare l'automazione completa-tokenizer-wearusfm.
I job sottomessi rimangono attivi; nessuna cancellazione implicita e nessun seguito
automatico alle valutazioni19618 dopo la disattivazione.

## Cancello effettivo prima della ridge

Checkout63405286 COMPLETED0:0 in18s, SHA70fd080149c11d10917059875d31a5a2820f176a,
pulito e in sola lettura. Validate63405498 completo in33s, benchmark63405502 in20s.
Report originali e SHA indipendenti raccolti in `results/infra/tokenizer_evaluation_20261103/cpu_5000/`.
Piano, sorgenti, ricetta, dose, sei pin, riferimenti e supporti comuni verificati;
benchmark solo T25-L/train,8thread anche nelle librerie numeriche. Nessuno score val.

V=33s, S=3,305283906s, E=0,110451286s: formula congelata389,073s, arrotondata
a **420s (7min), sotto il tetto1800s**. Gate superato. La prenotazione ridge resta
8CPU/30G/30min,0GPUh e massimo4coreh già riservate. Pubblicare questo gate prima
dell'invio; usare lo stesso checkout70fd080, senza altre modifiche scientifiche.
Consuntivo checkout+validate+benchmark **0,072778coreh**, tutto CPU-only.
