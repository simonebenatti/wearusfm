# 3-launcher · tre benchmark contrastivi RMS100/fast125

Simone ha autorizzato il checkout comune e i tre benchmark continui di 576 aggiornamenti. La raccolta dei tre risultati conclude questo incarico. Gli ID e i passi sono identici a quelli del preflight firmato. [Decisione](decisioni.md) · [Rapporto CUDA](reviews/contrastive_targets_cuda_science_20261121.md)

**SHA di esecuzione** `746112dd711903e5c92ade2efd90834264b2676a`, ramo `origin/codex/contrastive-target-windows-20261118`. Le otto suite locali di validazione del runner, delle ricette, del collaudo e della lettura hanno superato 198 test. Il piano contiene gli argomenti esatti del deploy, del checkout e delle tre celle. [Piano dei comandi](../results/infra/contrastive_targets_20261118/benchmark_handoff_20261121/submission_plan.json)

```bash
set -o pipefail
python3 results/infra/contrastive_targets_20261118/benchmark_handoff_20261121/prepare_handoff.py --frozen-sha 746112dd711903e5c92ade2efd90834264b2676a --dry-run | cmp - results/infra/contrastive_targets_20261118/benchmark_handoff_20261121/submission_plan.json
```

| Braccio | ID logico | Lettura prevista |
|---|---|---|
| C0 | `CONTRAST-BENCH-C0-targets-v3-20261118` | Gradienti a peso zero sullo stesso stato e batch agli update 128 e 256. |
| CN, InfoNCE | `CONTRAST-BENCH-CN-targets-v3-20261118` | Norma del gradiente InfoNCE pesato rispetto alla base ai passi 513–576. Conservare il checkpoint a 544 e le tracce dei quattro rank. |
| CR, Rank-N-Contrast | `CONTRAST-BENCH-CR-targets-v3-20261118` | Stessa misura per Rank-N-Contrast. Conservare il checkpoint a 544 e le tracce dei quattro rank. |

## Esecuzione

1. Confronta con `origin/codex/contrastive-target-windows-20261118` i due SHA indicati nella consegna. Rigenera il piano di comando per lo SHA di esecuzione sul Mac e confrontalo byte per byte con il piano pubblicato. Esegui il deploy del ref dedicato e verificane lo SHA. Usa il submitter e il ledger comuni, un tentativo per ID.
2. Registra prima di **ogni** invio ora italiana, ID riservati nel ledger, saldo e impegni del progetto, quota e scritture concorrenti su WORK, reservation SLURM, disponibilità e occupazione effettiva della corsia debug. Riserva 12 GiB per la coorte in aggiunta alle scritture concorrenti (18 GiB preventivati al controllo precedente, da aggiornare). Il limite operativo è di due celle debug dell'utente, incluse le altre linee. Esegui prima C0 e CN se entrambi gli slot sono liberi, poi CR al primo slot disponibile. Con un solo slot usa l'ordine C0, CN, CR.
3. Prima del checkout verifica i prerequisiti conclusi con sacct, la review legata al pacchetto, i pin fisici e gli SHA dei checkout. Verifica sul login i comandi dei tre bracci e le ricette con il runner in sola validazione. Il report CUDA del job 65016794 ha SHA-256 `920bfb46c881c13b7edc9a93af01e9073a714908bb97081c46d266cb17f8133d`, la sonda di uscita `ca4282219ffccb4b9faa6d1ce3576ecf62906661b3775b179a14d593da39c26a`. L'ultima calibrazione accettata è 65015116.
4. Sottometti l'unico checkout `CONTRAST-CHECKOUT-full-targets-v3-20261118` con **4 CPU, 2 GiB, 4 minuti**. Prima di ogni benchmark verifica su sacct il suo `COMPLETED 0:0`, lo SHA del checkout effettivo, la pulizia e la protezione in scrittura. Confronta a mano braccio di cella, release, config e directory di output. Le tre config condividono i due pesi calibrati e la selezione del braccio determina il peso applicato. Registra il controllo per ogni ID.
5. Ogni cella usa **4 GPU, 32 CPU, 240 GiB, 30 minuti**, QoS `boost_qos_dbg`, `TORCH_DIST_INIT_BARRIER=1` nell'export e limite interno 1.770 s. Batch globale 32, quattro rank, sei worker per rank, bf16 math con TF32 spento. Usa una directory nuova per ogni braccio. Conserva ricevuta, ID numerico ed export effettivo. Prerequisiti già conclusi si verificano con accounting e SHA. Un controllo fallito ferma la fase interessata. Per un invio ambiguo riconcilia prima il ledger.

## Risorse e finestra temporale

| Per questa consegna | Richiesta GPU-ore / core-ore | Margine SLURM | Inviluppo |
|---|---:|---:|---:|
| Checkout e tre benchmark | 6 / 48,266667 | 1,2 / 10 | 7,2 / 58,266667 |

L'effettivo della coorte all'uscita del CUDA è 1,134444 GPU-ore e 18,14 core-ore. Sommando le **richieste** fino a questi tre benchmark si arriva a 10 GPU-ore e 97,066667 core-ore sul tetto 18 e 169,6. Registra il billing in ore locali, il saldo residuo al netto degli impegni aperti e il consuntivo effettivo separatamente dal margine.

Rispetto alla chiusura del 23/11 alle 08:00 CET, per questa tranche l'ultimo inizio teorico del checkout con i margini è alle **06:38 CET** con due slot disponibili per tutta la fase (**06:02 CET** con un solo slot). Anticipa per coda e verifiche. Se la disponibilità corrente porta oltre la manutenzione, applica il calendario aggiornato da Leonardo. [Calendario dell'intero preflight](prompt_3_launcher_contrastive_targets_audit_20261120.md)

## Raccolta e ritorno alla chat scientifica

Per consentire una verifica indipendente della parità C0 archivia anche i quattro rapporti individuali `zero_weight_parity.rank{0,1,2,3}.json` accanto all'aggregato.

Pubblica gli ID appena assegnati. Per ogni braccio archivia ricevute, accounting, stdout/stderr, configurazione, summary, metriche per passo, mappa delle sessioni e i quattro profili numerici effettivi. Verifica i confronti di SHA e dimensioni fra originali e copie. Per C0 conserva `zero_weight_parity.json`; per CN e CR i quattro `trace-*.jsonl` e le impronte e dimensioni dei checkpoint a 544 e 576, lasciando i file di pesi su WORK. Raccogli anche le impronte del checkpoint finale C0. Verifica i 576 update, gli stessi ingressi, la memoria GPU di picco e il tempo effettivo per braccio. In caso di arresto o evidenze mancanti restituisci quanto disponibile, i log e il primo errore. Comunica gli ID, il commit di raccolta, i percorsi e i consuntivi a Simone. Le quattro riprese, la lettura CPU completa e il training lungo richiedono la successiva consegna scientifica.
