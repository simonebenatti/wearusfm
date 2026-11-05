# Estrazioni 30M/300M a 5.000 passi — 5 novembre 2026

Il training **63532932 è COMPLETED 0:0**: 5.000 update, batch globale 32,
160.000 finestre. Consuntivo 16,401111 GPU-ore e 131,208889 core-ore; nessun
resume o cambiamento di ricetta. Raccolte le quattro ricevute, confrontate con
gli hash remoti, il piano e il contratto. Checkpoint nominato e finale 300M:
`37279995d0d484e241b032eb9b6b23a31c16a5947a0e2cf0c575da0cb70641aa`.
Controllo 30M rihashato: `49dd518b…daa80`; snapshot immutabili a 5.000 confermati.
Il controllo dei log trova tutti i passi, LR e cadenze previsti, senza STOP.
Le copie locali non ricaricano il checkpoint: il producer lo ha verificato
prima della receipt e ogni estrazione ripete la verifica integrale sul compute.

## Piano congelato e finalità

Piano unico `results/infra/scaling_300m_20261104/extraction_plan_5000_20261105.json`,
SHA256 `8b358951182f02dc6af8fafd2621f2dc1ae2eb98939e67c5c914ecc09933528d`.
Usa il teacher dei due checkpoint a pari dose, le stesse popolazioni e gli
estrattori già approvati. Nessuna scelta dai risultati di validazione e nessun
nuovo training. Per la posa 30M si riusano le due cache esistenti a 5.000.
Le cinque estrazioni producono dodici cache; insieme alle due storiche sono
quattordici. La proiezione S5 viene applicata nei lettori CPU successivi.

## Risorse e invii

Checkout nuovo `scaling_300m_evaluation_20261105_v1`, codice del commit che
pubblica questo piano, da registrare nello stato degli invii. Un checkout
congelato comune per tutti gli estrattori e per il codice dei lettori successivi.
Il training conserva il proprio commit dc965594.

| ID logico | Operazione | GPU | CPU | RAM | Tempo massimo | GPU-ore |
|---|---|---:|---:|---|---|---:|
| SCALING-300M-eval-checkout-20261105 | Checkout | 0 | 2 | 4G | 10 min | 0 |
| SCALING-300M-eval-pose-20261105 | Posa 300M, train e val | 1 | 8 | 100G | 150 min | 2,50 |
| SCALING-300M-eval-p1-20261105 | P1 300M | 1 | 8 | 100G | 120 min | 2,00 |
| SCALING-300M-eval-p2-20261105 | P2 300M | 1 | 8 | 100G | 150 min | 2,50 |
| SCALING-30M-eval-p1-20261105 | P1 30M | 1 | 8 | 100G | 30 min | 0,50 |
| SCALING-30M-eval-p2-20261105 | P2 30M | 1 | 8 | 100G | 45 min | 0,75 |

Totale estrazioni **8,25 GPU-ore e 66 core-ore associate alle GPU**. Il checkout
prenota 0,333333 core-ore, comprese nell'inviluppo CPU già fissato di 180 core-ore.
Non si aumenta alcun tetto. Partizione estrazioni boost_usr_prod; checkout
lrd_all_serial. Le cinque celle sono indipendenti dopo il checkout verificato.

Output nuovi sotto
`/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/scaling_300m_20261105/5000/`,
nelle sottocartelle T100-G-300M/{pose,p1,p2} e T100-G-30M/{p1,p2}.
Gli ID si prenotano soltanto con `scripts/submit_job.py` nel ledger comune.
I comandi finali, con SHA effettivo, vanno pubblicati prima della sottomissione.
Checkout COMPLETED 0:0, HEAD, tree pulito e sola lettura sono verifiche sincrone;
non usare dipendenze afterok verso job già purgati.

## Passaggio ai lettori

Dopo le estrazioni verificare ricevute, hash, supporti e provenienza. Poi
validate e benchmark CPU sul train nelle allocazioni già fissate. I fit si
lanciano soltanto se i tempi e la RAM misurati rientrano nelle formule e nei
tetti pubblicati in `docs/scaling_300m_evaluation_execution_20261104.md`.
Nessun monitor automatico, nessuna continuazione a 19.618, nessuna variazione
di loss, dose, seme o popolazione. Le autorizzazioni già espresse coprono
questa catena; errori reali fermano la fase senza retry cieco.
