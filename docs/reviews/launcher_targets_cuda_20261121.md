# Collaudo CUDA RMS100/fast125 — esecuzione operativa del 21 novembre

Istruzioni b0ce039d, esecuzione congelata 4e880b9f6787d7294712370d00ed485bfbe45735.
Piano rigenerato e confrontato byte per byte; pin e submitter comuni verificati,
nessun ID preesistente. Ref deploy pubblicato e SHA remoto verificato.

Prima di ciascun invio: saldo/quota/reservation/slot/ledger e calendario,
controlli login congelati con verifica dei 14 artefatti della calibrazione,
checkpoint originale, accounting audit/calibrazione, binding review,
make-job/cmp, build_command e validazione della ricetta. Stdout e snapshot
in results/infra/contrastive_targets_20261118/cuda_execution_20261121/.

Riserva 12 GiB per coorte più 18 GiB prudenziali per altre scritture;
WORK libero oltre 40 GiB. Saldo 653.928 ore locali, nessun job debug sbenatti.
Corsie UP; manutenzione confermata 23/11 08:00 CET–25/11 08:00 CET.
Anche includendo controlli e margine, la coppia entra nella finestra disponibile.

Checkout **65016695 COMPLETED 0:0**, 28 s, 4 CPU/2 GiB/4 minuti.
SHA, pulizia e permessi vanno confermati da before_cuda sul checkout effettivo
prima del secondo invio. La consegna termina con la raccolta del collaudo:
nessun benchmark o training viene rilasciato automaticamente.

before_cuda superato sul checkout effettivo: HEAD, pulizia, permessi,
binding/audit/calibrazione, job rigenerato e argv verificati nuovamente.
Secondo ID **65016794**, collaudo 4 GPU/32 CPU/240 GiB/30 min, export con
TORCH_DIST_INIT_BARRIER=1 attestato dalla ricevuta. Un solo invio per ID.
Esito e raccolta da completare.

## Raccolta conclusiva

Checkout 65016695 COMPLETED 0:0, 28 s; **CUDA 65016794 COMPLETED 0:0, 130 s**.
Ricevuta effettiva con TORCH_DIST_INIT_BARRIER=1 conservata. Nessun retry.

Rapporto **passed=true**, 320/320 confronti numerici entro le soglie congelate:
4 rank × 2 bracci × 2 precisioni × 2 update × 10 grandezze. Nessun confronto
fuori limite. Massimo rapporto errore/soglia FP32 0,05782035, bf16 0,37933469.
Arresti coverage e numeric positivi su ciascuno dei quattro rank.

Sonda di uscita processo: quattro worker CUDA/NCCL inizializzati, eccezione
intenzionale del rank zero riconosciuta, launcher returncode 1 atteso,
nessun timeout e nessun processo residuo. Durata 12,265701 s contro limite
di completamento 60 s. Il returncode del probe non è il returncode del job.

Verificatore congelato verify_results.py eseguito sul checkout con i moduli
e i percorsi preservati come da prontuario: uscita zero. Autenticati rapporto,
config, sorgenti, checkpoint della calibrazione e sonda rank-exit. Stdout
integrale in verify_results.log.

Rapporto `$WORK/wearusfm_runs/results/contrastive_targets_20261118/cuda.json`, SHA256
**920bfb46c881c13b7edc9a93af01e9073a714908bb97081c46d266cb17f8133d**.
Sonda `$WORK/wearusfm_runs/results/contrastive_targets_20261118/cuda.json.rank_exit_probe/report.json`, SHA256
**ca4282219ffccb4b9faa6d1ce3576ecf62906661b3775b179a14d593da39c26a**.
18 artefatti raccolti e verificati byte/hash contro gli originali remoti,
intera sonda e due cartelle di arresto inclusi. Checkpoint diagnostico
coverage_stop/checkpoint.pt conservato sul cluster, 457.552.600 byte, SHA256
c7b848a43aedb9429440976c1dc001064f2f582bc813ca17e06ce8d3da17ac0c.

| Risorse | Richiesto | Effettivo | Residuo rispetto al richiesto | Inviluppo con margine |
|---|---:|---:|---:|---:|
| GPU-ore | 2 | **0,144444** | 1,855556 | 2,4 |
| Core-ore | 16,266667 | **1,186667** | 15,08 | 19,866667 |

Billing effettivo pari alle CPU: CUDA 1,155556 ore locali, checkout 0,031111.
Cumulativo effettivo nuova coorte: 1,134444 GPU-ore / 18,14 core-ore.
Snapshot prima dei due invii: saldo 653.928 ore locali; altri impegni residui
3.881,217778 / 3.880,524444 ore. Sottratto una sola volta l'inviluppo della
coppia, saldo netto prudenziale 650.026,915555 / 650.027,608889 ore. Nessuna
duplicazione dello storico già addebitato. Copie complete degli snapshot
e reservation conservate. Consegna conclusa; nessun benchmark o training inviato.
