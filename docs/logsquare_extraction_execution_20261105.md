# T100-LQ: estrazione della posa al passo 5.000

Il benchmark 63614514 è completo e la review indipendente ha approvato i tempi e i tetti.
Lettura: `docs/reviews/logsquare_inference_readout_20261105.md`.
Piano nuovo: `results/infra/logsquare_evaluation_20261105/extraction_plan_5000.json`.
Il piano di inferenza precedente resta immutabile: la ricevuta ne pinna i byte.
Il nuovo piano conserva contesto, checkpoint 5.000 e tutti i 110 sorgenti; aggiunge
ricevuta, piano benchmark, SHA di esecuzione e allocazioni copiate dal rapporto.
Il checkpoint 19.618 rimane nullo; nessuna estrazione a quell'endpoint può partire.

| Fase | ID logico | Risorse | Tetto |
| --- | --- | --- | --- |
| Checkout | LOGSQ-pose-checkout-20261105 | 2 CPU, 4G, 10 min | 0,333333 core-ore |
| Posa 5.000 | LOGSQ-pose-5000-20261105 | 1 GPU, 8 CPU, 100G, 50 min | 0,833333 GPU-ore / 6,666667 core-ore |

Checkout nuovo `logsquare-pose-5000-20261105`, output nuovo
`/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/logsquare_evaluation_20261105/5000/pose`.
Stima puntuale: circa 30 minuti, proxy dichiarato; tetto inderogabile di 50 minuti.
Il launcher riceve esplicitamente `--time=00:50:00` e i sei argomenti della cella
`5000 extract-pose none`, piano, output e SHA del piano. Submitter e ledger comuni.

Il job legge il teacher congelato e salva le rappresentazioni e predizioni di posa
sul sottoinsieme train e sui 15 utenti di validazione, verificando parità con i
riferimenti. Non addestra il FM. Prima dei lettori CPU si raccolgono cache, ricevute,
provenienze, hash e risorse; poi si applicano i gate già approvati del lettore di posa.
La continuazione del training 63593128 resta indipendente. Nessun monitor in background.

## Correzione del solo test storico

Adottato dal coordinamento d2f2f08 come 99efdef: il test B19618 usa ora il runtime
storico pinnato dal launcher invece dello schema corrente. 15 test passati in 3,86 s.
Nessun launcher, trainer, sorgente scientifico o checkout già in esecuzione modificato.

## Invio

Codice congelato 9720fc8f7510bd8e7640164630da30515a8b3210, pubblicato e verificato
su origin. Checkout 63615611 COMPLETED 0:0 in 30 s (0,016667 core-ore); SHA, piano,
110 sorgenti, pulizia e sola lettura verificati. Estrazione 63615952 sottomessa tramite
submitter e ledger comuni, RUNNING dalle 19:24:49 CET. Allocazione osservata corretta:
1 GPU, 8 CPU, 100G, 50 minuti. Continuazione 63593128 ancora PENDING Priority.
Il successivo test R-1 viene corretto solo nel ramo di sviluppo: questo checkout
congelato e i job già avviati rimangono invariati.
