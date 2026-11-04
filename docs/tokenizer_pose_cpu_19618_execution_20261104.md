# Tokenizer posa a 19.618 passi: validazione, benchmark e ridge — 04/11/2026

Le estrazioni teacher T25-L / T100-L / T100-G sono **COMPLETED 0:0**:
63491925 / 63491940 / 63491945, rispettivamente 1767 / 1730 / 1729 secondi.
Consuntivo posa: **1,451667 GPU-ore** e **11,613333 core-ore allocate con GPU**.
Tutti e tre gli estrattori hanno eseguito il codice
`453ce6c496745f2d73aa3007406721586e6dd767`; training congelato `afb644e`.

Sono state lette integralmente le tre ricevute originali e ricalcolati con
`sha256sum` i loro hash, quelli delle sei cache e delle sei provenienze.
Le provenienze conservate localmente sono identiche byte per byte agli originali.
Dose 19.618, teacher effettivo, configurazioni, checkpoint, 1907 finestre train
(40 utenti) e 18.548 finestre val (15 utenti) sono coerenti. Firme di target,
metadati e maschere centrali coincidono fra i bracci; gli estrattori hanno
verificato la parità con i riferimenti storici.

Le ricevute originali complete rimangono remote e sono pinnate integralmente.
I riepiloghi in `completed_19618/pose/` omettono esplicitamente l'inventario dei
file grezzi; **i lettori CPU aprono le ricevute complete remote**, non i riepiloghi.
Nessuna cache, predizione individuale o dato wearable entra in Git.

## Piano congelato e risorse prima dei lanci

Piano: `results/infra/tokenizer_evaluation_20261103/pose_readout_plan_19618.json`.
SHA256: `7a469c694637b870280ce6fcc8f9dcac1181e6b16644bc0042d1f942dc206e49`.
Costruito e validato con `make_plan` già usato a 5.000: ricetta, sorgenti,
statistiche, riferimenti e fold utenti **identici**. Nessun nuovo lettore,
seme, criterio di selezione o obiettivo; crop6, LayerNorm FP32 e ridge FP64
rimangono quelli congelati. La selezione degli alpha usa soltanto il train.

Il checkout nuovo `tokenizer-endpointcpu-20261104` è comune alle analisi CPU di
posa e classificazione: 2 CPU / 4G / 10 minuti, massimo **0,333333 core-ore**,
da contare una sola volta. Il suo SHA effettivo sarà il commit pubblicato prima
del deploy, verificato sul remoto e nell'output del checkout.

| ID logico | Fase | Risorse | Tetto core-ore |
|---|---|---|---:|
| TOKENIZER-pose19618-validate-20261104 | Verifica sei cache, ricevute e riferimenti | 2 CPU / 30G / 30 min | 1 |
| TOKENIZER-pose19618-benchmark-20261104 | Timing solo T25-L/train | 8 CPU / 30G / 15 min | 2 |
| TOKENIZER-pose19618-ridge-20261104 | Fit tre bracci e confronto appaiato incluso | 8 CPU / 30G / 30 min | 4 |

Validate e ridge, **5 core-ore**, erano già nella matrice originale. Il secondo
benchmark, **2 core-ore**, è aggiuntivo: il singolo benchmark prenotato dalla
politica originaria è già stato eseguito a 5.000 passi. È dichiarato qui prima
del lancio, nell'autorizzazione persistente di Simone a completare la campagna
e superare il vecchio budget. Totale catena posa 7 core-ore più il checkout
condiviso; incremento rispetto alle riserve precedenti 2 core-ore più checkout.
**Zero GPU aggiuntive.** Nessun nuovo consenso necessario.

## Cancello prima della ridge

Validate e benchmark possono partire in parallelo dopo la verifica del checkout.
Non prenotare la ridge soltanto tramite `afterok`: prima leggere i due report.
Si applica la stessa formula congelata a 5.000:

`ceil_60s(1.5 * (2*V + 3*((1+R)*S + 13*E)) + 120)`

con V = ElapsedRaw del job validate completo, S = `statistics_seconds`,
E = `three_eigensystems_seconds`, R = max(10,18548/1907) = 10.
Il 13E è una riserva empirica per scoring, non tredici decomposizioni.
La previsione include margine 50% e 120 secondi finali; non garantisce il runtime.

La ridge parte solo se entrambi i job sono COMPLETED 0:0, i report sono completi
e coerenti con piano/sorgenti/dose, il supporto comune è valido, il benchmark
usa 8 thread anche nelle librerie numeriche e la stima è **<=1800 secondi**.
Nessuno score val viene usato per scegliere risorse o cambiare la ricetta.
Se il gate fallisce si diagnostica e si ferma questa fase, senza retry cieco.

## Comandi da affidare al submitter e al ledger comuni

Tutti i job usano `scripts/slurm/tokenizer_pose_readout.sbatch`, con `CODE_DIR`
al checkout nuovo ed `EXPECTED_SHA` al commit effettivo pubblicato/verificato.
Argomenti del launcher:

```text
results/infra/tokenizer_evaluation_20261103/pose_readout_plan_19618.json validate /leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/tokenizer_pose_readout_20261103/19618/validate
results/infra/tokenizer_evaluation_20261103/pose_readout_plan_19618.json benchmark /leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/tokenizer_pose_readout_20261103/19618/benchmark T25-L
results/infra/tokenizer_evaluation_20261103/pose_readout_plan_19618.json ridge /leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/tokenizer_pose_readout_20261103/19618/ridge
```

Override SLURM validate: `--cpus-per-task=2 --mem=30G --time=00:30:00`.
Benchmark: `--cpus-per-task=8 --mem=30G --time=00:15:00`.
Ridge: `--cpus-per-task=8 --mem=30G --time=00:30:00`.
Ogni directory di output deve essere nuova. Nessun job di confronto separato:
il confronto appaiato è già nel report ridge. Monitor automatico lasciato spento.
