# Punto 6 — esecuzione, 02/11/2026

Simone autorizza il lancio diretto della catena tramite submitter e ledger comuni.
Codice scientifico congelato `c9d9c02de2597e2ab5e3907d7ffe06ffb544bb62`;
protocollo, 54 test e review favorevole pubblicati prima del lancio.
Autorizzazione registrata in35cd6e0, origin verificato; deploy separato verificato allo SHA scientifico.

| Fase | Job | Stato verificato | Tempo | Costo noto |
|---|---:|---|---:|---:|
| Checkout | 63242629 | COMPLETED0:0 | 26s,2CPU | 0,014444core-ore |
| Preflight train | 63242889 | COMPLETED0:0 | 168s,1GPU/8CPU | 0,046667GPU-ore /0,373333core-ore |
| Estrazione B | 63245968 | COMPLETED0:0 | 1996s,1GPU/8CPU | 0,554444GPU-ore /4,435556core-ore |
| Estrazione R3c | 63246001 | COMPLETED0:0 | 1996s,1GPU/8CPU | 0,554444GPU-ore /4,435556core-ore |
| Benchmark CPU | 63246431 | COMPLETED0:0 | 33s,8CPU | 0,073333core-ore |

B e R3c sono partiti entrambi alle15:16:15 CET, con una GPU indipendente ciascuno sullo stesso nodo.
Il benchmark è accodato con `afterok:63245968`: all'avvio verifica ricevute complete e cache B prima
del timing, senza aprire R3c/val. Le risorse del benchmark sono già fissate; non richiedono altri dati.
**Report finale non ancora sottomesso:** va letto il benchmark e verificata la conclusione di entrambe
le estrazioni prima di applicare la formula CPU e pubblicare il relativo preventivo.
Tetto originario catena2,25GPU-ore/28,333core-ore; con preparazione misurata ed estrazioni richieste,
massimo residuo+speso1,963333GPU-ore/25,721111core-ore mantenendo il report al tetto1h.

## Regola per le risorse, prima di leggere i timing

Review del critico: per ogni estrazione GPU richiedere
`ceil5min(1,5 × max(1800, 20455/n_pre × loop_seconds) + H + 180)` secondi,
con H=max(0,elapsed totale preflight−loopB−loopR3c), assegnato interamente a ciascun modello.
1800s è il riferimento storico completo;180s riservano ulteriore I/O/hash. Non è un limite garantito:
quattro registrazioni train non rappresentano tutto il corpus. Se oltre3600s, fermarsi senza troncare.

Per report CPU: `ceil5min(1,5 × max(2B,2×209) + 180)`, dove B è il tempo totale del benchmark,
almeno pari alla somma dei tempi dei quattro lettori;209s è il precedente report a quattro celle.
Memoria:60GiB GPUjob,30GiB CPUjob; controllare i margini da `resource_policy.json` prima dei lanci.
Politica dichiarata mentre il preflight63242889 è RUNNING e prima di leggerne i timing.

## Preflight letto e preventivo delle estrazioni

63242889 COMPLETED0:0,168s×1GPU/8CPU: **0,046667GPU-ore e0,373333core-ore**.
80finestre totali,40permodello:2.457.600 coefficienti medi FP16 per modello, zero differenze,
zero scarto dopo LayerNorm e nel denominatore. Metadati/target/supporto identici;115sourcepin verificati.
Loop B3,9761s e R3c2,4408s; overhead H144,5544s. La formula congelata restituisce
**B60min e R3c55min**, entrambi1GPU/8CPU/60GiB. GPU riservata3,1074GiB su dispositivo
riportato A100-SXM-64GB; RSS accounting2,9261GiB. Margine memoria host storico32,5GiB<60GiB.
Spazio libero verificato ampiamente maggiore dei20GiB riservati per payload e margine.

Ricevute, log, accounting e decisione macchina archiviati sotto `results/infra/pose_slots_20261102/`.
Il preflight non misura metriche scientifiche di posa sulla val; verifica identità e risorse.
Nessun allargamento delle soglie o cambiamento del protocollo.

## Prosecuzione della catena

Attivo il controllo nella stessa chat `completa-wearusfm-punto-6`: segue soltanto i job sopra,
verifica benchmark/entrambe le cache, pubblica il preventivo CPU e sottomette il report già autorizzato.
Poi raccoglie e rivede i risultati, aggiorna tabella/registro e committa/pusha. Nessun nuovo esperimento
scientifico. Si disattiva alla conclusione o quando serve una nuova decisione; resta silenzioso
finché non vi sono cambiamenti significativi. Non duplica semplicione né i job già prenotati.

## Estrazioni complete verificate e preventivo del report

B/R3c conclusi alle15:49:31 CET; benchmark concluso alle15:50:06.1.907 finestre train e18.548 val
per modello, tutte con meanFP16 identica alla cache storica. Gate LayerNorm/denominatore con scarto
zero su2.513.510.400 elementi medi totali. Parent/split/preflight/checkpoint/plan/sourcepin verificati;
metadati e stati input raw B/R3c identici. Il collector verifica hash delle ricevute e dimensioni dei
residui; gli hash completi dei quattro NPY saranno ricontrollati nel job CPU prima dei fit.
Benchmark ha già riverificato B/train. Nessuna analisi scientifica val eseguita in questa fase.

Tempi dei quattro lettori:19,878829s complessivi, walltime benchmark33s; RSS1,168285GiB.
Formula preregistrata:`ceil5min(1,5×max(2×33,2×209)+180)`=**900s**.
Richiesta finale **8CPU,30GiB,15min;2core-ore,zeroGPU**. Stima memoria13,752428GiB<30GiB.
`readout_resource_decision.json` contiene controlli e conti. Review indipendente del critico favorevole,
nessun difetto aperto. Il codice scientifico c9d9c02 resta immutato.

Speso fin qui **1,155556GPU-ore e9,332222core-ore**; massimo con report11,332222core-ore,
entro i tetti2,25GPU-ore/28,333core-ore. Pubblicare questa decisione e verificare origin prima della
sottomissione dell'unico ID mancante `POSE-SLOTS-run-20261102`; niente nuovi consensi o duplicati.
