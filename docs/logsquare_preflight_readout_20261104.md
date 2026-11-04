# T100-LQ: lettura del collaudo preparatorio del 04/11/2026

**Il collaudo tecnico è completo. Proposta: mantenere epsilon=0,001, con
decisione R1 da rivedere da Claude. Nessun training scientifico è stato avviato.**
Simone ha già autorizzato anche il seguito operativo della linea, dopo i controlli
e la review concordata: non occorre chiedergli un altro via per letture,
archiviazione, commit/push o lanci del piano revisionato.

Codice congelato `43cad1f760fbac2f10d8c16bc6f5d4eedc957b61`, scientificamente
identico al `d48cad5` approvato da Claude. Checkout 63525639 e collaudo 63525730:
COMPLETED 0:0. Report originale SHA256
`311ac329731c84f91e9199c78c92e4890cc43d2de598361f11f4e3f7465e5b08`.
I risultati e l'analisi riproducibile sono in
`results/infra/logsquare_execution_20261104/`; script offline
`scripts/logsquare_preflight_analysis.py`.

| Passo | Cosa verifica | Risultato | Conclusione |
| --- | --- | --- | --- |
| Funzione iniziale | Due forward del controllo, poi confronto L/LQ; pesi condivisi e RNG | Identità esatta su quattro rank, scarto massimo0; 24.576 nuovi pesi a zero | Il candidato parte dalla base non addestrata prevista |
| Aggiornamenti appaiati | 64 batch globali, 2.048 finestre per braccio; supporti, bersagli, RNG | Tutte le256 coppie rank/passo concordano; nessun errore numerico; gradiente e update W_log positivi | Il ramo è collegato al training reale |
| R1: pavimento | p non mascherato≥0,50 | 0su1.728 righe kernel/gruppo; p massimo 0,29606 | Non emerge dominanza del pavimento |
| R1: derivata | RMS derivata × sqrt(epsilon)≥0,50 | 553 segnalazioni su3.456 righe kernel/gruppo/modo; massimo 0,66035 | La segnalazione richiede decisione scritta, non un cambio automatico |
| R1: mascheramento | Attraversamento di p=0,50 oppure scarto medio log≥1std non mascherata | Nessun trigger; massimo scarto 0,16605std | Nessun motivo osservato per cambiare epsilon a causa delle maschere |
| S1: frequenze | Tutte le coppie200/2000 Hz, senza esenzioni | 28 coppie disponibili, nessun contrasto con p200≥0,50 | Nessun kernel dichiarato inattivo; nessuna esclusione dai criteri |
| Costo aggiornamento | Media dei56 update dopo warmup iniziale di8 misure | L 0,71437 s; LQ 0,70405 s; rho 0,985557 | Costo sostanzialmente uguale in questo campione breve |

## Decisione R1 proposta alla review

Mantenere epsilon 0,001. Il trigger della derivata è reale e viene riconosciuto:
277 righe mascherate e276 non mascherate, tutte nella famiglia MLP del banco,
14 indici distinti fra 48–63. Non sono553 osservazioni indipendenti.
La derivata resta sotto il suo limite analitico; non emergono nonfinitezza,
dominanza del pavimento o trigger del mascheramento. Aumentare epsilon
attenuerebbe le piccole risposte senza che questo collaudo abbia mostrato la
necessità di farlo. La proposta conserva quindi il parametro preregistrato,
senza cercarne uno migliore sui dati di calibrazione. Non si modifica il banco.

Questa è una decisione motivata prevista da R1, **non l'assenza di trigger**.
Prima del training Claude deve rivederla; il piano attuale non contiene una
release che permetta di eseguirlo. Nessun secondo collaudo è proposto finché
non si cambia il parametro o il codice scientifico.

| Fase diagnostica | Gruppi dataset/fs | Token visibili per modo | Trigger derivata mascherato / non mascherato |
| --- | ---: | ---: | ---: |
| Iniziale | 10 | 67.563 | 105 / 102 |
| Traiettoria prima dell'update | 17 | 3.884.083 | 172 / 174 |

Le fasi si sovrappongono al passo iniziale; token vicini e rank non sono repliche
statistiche indipendenti. A 200 Hz sono state viste42 finestre NinaPro DB5:
2.455 token iniziali e52.521 sulla traiettoria per modo. p massimo non mascherato
0,14134/0,17858; uno/due kernel per modo superano il trigger della derivata.
Non è una verifica su EPN held-out né un confronto causale fra frequenze.

## Clipping e limite del collaudo

Entrambi i bracci vengono clippati in64 passi su 64. Rapporto dei fattori LQ/L:
minimo 0,987554, mediana 0,995062, massimo 1,001502. Quota g_log²/g_LQ²:
mediana 1,6681%, massimo 2,4782%; al primo passo 2,3896%.
Residuo relativo g_LQ²−g_L²−g_log² al primo passo:−1,4074e−6.
I rank sono repliche DDP: il riepilogo usa rank 0 e i dati conservano tutti i rank.

Il rapporto riguarda i gradienti prima di AdamW, non il rapporto fra update.
La quota di W_log non misura tutta la sensibilità propagata al banco.
Non si cambia il clipping per compensare il ramo aggiunto.

Sono appena64 passi iniziali di warmup: LR finale 3,262e−6 contro picco 1e−4,
W_log appena partito da zero. **Non è una prova di stabilità alla dose completa,
di miglioramento della posa o di utilità della rappresentazione.** Non si
interpretano le loss di questo tratto come un risultato scientifico. La memoria
misurata ospita due modelli completi: non è una stima di un singolo training.

## R2: preventivo misurato, separato dall'autorizzazione scientifica

| Fase futura | Storico T100-L | Stima da rho_update | Tetto preregistrato |
| --- | ---: | ---: | ---: |
| 5.000 passi | 3.926 s | 3.869,30 s; 4,2992 GPUh | 1h30; 6 GPUh; 48 coreh |
| Continuazione a 19.618 | 11.329 s | 11.165,38 s; 12,4060 GPUh | 4h; 16 GPUh; 128 coreh |
| Totale training | 16,95 GPUh | 16,7052 GPUh | 22 GPUh; 176 coreh |

Un checkout 2 CPU/10 min aggiunge al massimo 0,333 coreh CPU-only.
Il rapporto esclude loader/diagnostica; lo storico comprende loader/I/O.
rho<1 non dimostra accelerazione: i p90 sono praticamente uguali e il tetto
usa max(1,rho), senza sconti. I limiti letti di partizione boost_usr_prod e QoS
boost_qos_bprod sono 24h: le due fasi rientrano. Preventivo macchina in
`results/infra/logsquare_training_20261104/resource_proposal.json`.

Consuntivo del solo collaudo: 148s × 4GPU = 0,164444 GPUh, 1,315556 coreh allocate.
Checkout 27s × 2CPU = 0,015 coreh. Totale 1,330556 coreh. Nessun altro job lanciato.

Per le estrazioni rho_infer resta da misurare sul percorso effettivo: il rapporto
di training non lo sostituisce. I 2,75 GPUh delle estrazioni storiche restano un
riferimento, non un preventivo approvato del candidato.

## Verifica, archiviazione e seguito

Audit indipendente: hash dei 12 artefatti per-rank, giornali contro record finali,
supporti/RNG, statistiche dei log ricostruite dai momenti dei quattro rank,
clipping e tempi ricalcolati. Confronto della nuova capacità e dell'hash iniziale
con il report storico. Gli input locali sono ricontrollati per hash; la verifica
di immutabilità dei dati sul cluster rimane quella del driver congelato.

In Git: report compresso, quattro giornali completi compressi, controlli iniziali,
analisi completa e ricevute/accounting. I grandi JSON con momenti incrociati
restano all'output remoto e nella copia locale, con hash registrati. Lo stdout
SLURM resta remoto: sono stati letti i passi finali, non archiviato l'intero file.
Non entrano in Git segnali grezzi o checkpoint. Riproduzione dell'analisi:

```bash
~/.venvs/wearusfm-tok/bin/python scripts/logsquare_preflight_analysis.py \
  --input /private/tmp/wearusfm-logsquare-results-20261104 \
  --out /private/tmp/logsquare_analysis_check.json
```

Il controller del training è preparato per review con il piano
`results/infra/logsquare_training_20261104/training_plan.json`: stato
REVIEW_REQUIRED, release nulla. Il seguito richiede ancora review Claude della
decisione R1 e del controller, adattamento esplicito dei valutatori/estrattori,
pin e checkout pubblicati. I pin dei checkpoint si acquisiranno solo dopo
la loro produzione completa. Nessuna whitelist aggirata, nessuna nuova loss,
nessuna dose aggiuntiva o contrastiva inclusa implicitamente.
