# Lettori CPU 30M/300M a 5.000 — rilascio del 5 novembre 2026

Le cinque estrazioni sono COMPLETED 0:0; posa 300M terminata alle 14:43:18 CET.
Dodici cache nuove e due riusate, con ricevute complete. Consuntivo estrazioni:
4,853333 GPU-ore e 38,826667 core-ore, su tetti 8,25 e 66. Ricalcolati gli hash
remoti delle cache e delle ricevute; per la posa anche i cinque file del riuso 30M.
La review Claude 8b763610 comprende ora le ultime due ricevute (sezione 9), chiude
O-4 e conferma il seguito CPU. Il confronto scientifico resta da produrre.

## Codice e piani

Codice immutato `aef665ba7ea66fc80291846487907bb3ef9107df`, checkout
`/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/scaling_300m_evaluation_20261105_v1`.
Si usa il launcher già rivisto `scripts/slurm/scaling_readers.sbatch`, SHA256
`2da66c150fbc238dcf9b211a2eb672bb41adab8de96fc6f3f0c0ea5e716f6355`.

Piani reali in `results/infra/scaling_300m_20261104/cpu_readers_20261105/`:
`pose_plan.json` e `probe_plan.json`. Ricette, matrici S5, supporti, fold, seed,
dose e criteri S8 invariati. Questi piani vincolano gli hash degli output conclusi,
non contengono risultati scientifici o benchmark inventati. La verifica locale
copre schema e corrispondenze; l'apertura delle cache effettive avviene nei job.

I piani saranno distribuiti in un nuovo checkout documentale immutabile
`scaling_300m_cpu_plans_20261105_v1`, separato dal codice numerico. Il commit e il
comando esatto di checkout vengono pubblicati dopo il commit dei piani e prima
dell'invio. Nessuna modifica al checkout delle estrazioni.

## Primo rilascio e ordine

| Stadio | Numero job | CPU per job | RAM | Walltime per job | Tetto core-ore |
|---|---:|---:|---|---|---:|
| Checkout dei piani | 1 | 2 | 4G | 5 min | 0,166667 |
| Validazione delle quattro cache posa | 1 | 2 | 30G | 30 min | 1 |
| Benchmark posa, tre viste | 3 | 8 | 30G | 10 min | 4 |
| Benchmark sonde 30M naturale e 300M a dimensione pareggiata | 2 | 8 | 30G | 45 min | 12 |
| Benchmark sonde 300M naturale | 1 | 8 | 30G | 90 min | 12 |

Totale primo rilascio: **29,166667 core-ore, zero GPU**. Le tre viste sono
30M naturale, 300M naturale, 300M proiettato S5. Ogni benchmark sonde esegue
tutti e cinque i dataset senza metriche held-out o selezione di iperparametri.

Ordine: pubblicazione e verifica SHA origin/deploy → checkout completo e verificato
→ validazione completa → sei benchmark. Prima di ogni invio verificare ledger,
stati già noti e assenza di errori. Le sei misure possono essere in coda insieme
dopo l'accettazione della validazione; nessun fit è accodato anticipatamente.
CPU su `lrd_all_serial`, nodo `login08`, runtime/moduli/venv congelati; bootstrap
numerico del launcher e thread per stadio invariati. I comandi esatti, gli hash
dei piani e gli output nuovi sono in `submission_plan.json`.

## Riserva e passaggio ai fit

L'inviluppo CPU resta **180 core-ore**, distinto dalle CPU dei job GPU. Il checkout
63580233 già concluso usa la voce originaria (tetto 1/3 core-ora, osservate 0,029444).
La riserva originaria di 2/3 core-ora finanzia fino a tre checkout documentali
aggiuntivi da 2 CPU × 5 min: piani, budget misurati, manifest del confronto.
Tetto aggiuntivo 1/2 core-ora; restano 1/6 core-ora di riserva. Somma massima
179 + 1/3 + 1/2 + 1/6 = 180. Il limite dei checkout aggiuntivi è più breve di
quello originario, a fronte dei 53 s osservati; un TIMEOUT ferma la fase.

Dopo le misure reali pubblicare budget/pin e comandi dei fit. Posa: formula
`ceil60(1,5*(2V + somma(11S + 13E)) + 120) <= 3600`, con quattro accounting reali.
Sonde: quindici report di benchmark, RSS misurato, copertura esatta dei fold,
massimo 15 shard, 8 CPU/30G/120 min ciascuno e somma non superiore a 136 core-ore.
Non si riducono fold, tolleranze o dati per far rientrare un preventivo.

O-6/O-8: **stop al primo rifiuto di un gate, errore o TIMEOUT, nessun retry**;
si raccolgono evidenze e costo e si informa Simone. Nessuna variazione di nodo,
thread, librerie, protocollo o criterio. O-7: ogni accounting JSON ha accanto il
testo originale `sacct -P` da cui deriva; per le sonde si lega il job ai cinque
report tramite ricevuta del submitter, argv e pin. Il costo cumulativo si aggiorna
prima delle tranche successive, comprendendo eventuali tentativi falliti.

Nessun training nuovo o seguito a 19.618, nessun monitor automatico. L'esecuzione
di questa catena è già autorizzata da Simone; non serve un ulteriore consenso.

## Invii effettivi e misure al 05/11, 15:36:50 CET

Piani **1bde83e**, comando checkout **26781ee**, verifica iniziale **8cdabd6**;
SHA origin/deploy verificati. Checkout63591447 COMPLETED0:0/44s e validazione
63591791 COMPLETED0:0/45s, MaxRSS2,171GiB; accettate le quattro cache posa.

| Job | Operazione | Stato osservato |
|---|---|---|
|63592426|Benchmark posa30M naturale|COMPLETED0:0,22s|
|63592917|Benchmark posa300M naturale|COMPLETED0:0,30s|
|63593029|Benchmark posa300M matched|COMPLETED0:0,29s|
|63593262|Benchmark sonde30M naturale|RUNNING|
|63593508|Benchmark sonde300M naturale|PENDING|
|63593593|Benchmark sonde300M matched|PENDING|

Le due attese hanno reason `QOSMaxCpuPerUserLimit`. Non è un errore del modello.
Tre report posa accettati, runtime numerico identico a8thread e supporto corretto;
la formula congelata deriva **660s** per il fit, entro3600. Budget misurato in
`cpu_readers_20261105/pose_budget.json`, ancora da distribuire nel checkout budget
prima del fit. Guard del runtime vivo e dei file ancora obbligatori.

Ricevute submitter di tutti i job raccolte. I tre benchmark sonde devono produrre
quindici report completi prima di dimensionare i fit. Nessun fit, score o seguito
19.618 avviato. Il freeze numerico S8 condizionale è pubblicato separatamente in
`docs/scaling_300m_continuation_budget_freeze_20261105.md`, prima degli score.
