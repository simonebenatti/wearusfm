# RMS100/fast125 — cosa dicono i tre benchmark continui

Il checkout comune 65017186 e i tre job sono terminati con `COMPLETED 0:0`. Ogni braccio ha completato 576 aggiornamenti su batch globale 32. Il controllo C0 ha riprodotto gradienti identici byte per byte nei confronti a peso zero sui due stati previsti, prima degli aggiornamenti 128 e 256. Verifica effettuata con il lettore di parità sul rapporto aggregato raccolto da 3-launcher. [Rapporto di parità C0](../../results/infra/contrastive_targets_20261118/benchmark_execution_20261121/C0/zero_weight_parity.json)

| Braccio | Job | Tempo SLURM | Mediana della norma pesata / norma di base | 95º percentile |
|---|---:|---:|---:|---:|
| C0 | 65017240 | 13 min 15 s | Termine contrastivo spento | — |
| InfoNCE (CN) | 65017310 | 13 min 04 s | 21,54% | 36,65% |
| Rank-N-Contrast (CR) | 65020151 | 12 min 53 s | 8,28% | 21,08% |

Ho ricalcolato le due distribuzioni dai 64 record per braccio ai passi 513–576, dopo la rampa dei pesi, con `component_ratios`. Le mediane sono entrambe nell’intervallo preregistrato 0,5–50% e i 95º percentili sono sotto il 100%. Il rapporto fra le mediane CN/CR è 2,602, nell’intervallo previsto 0,25–4. Le identità di origine coincidono in ordine per tutti i 576 batch nei tre bracci. Le impronte dei tensori di ingresso effettivi sono confrontabili nelle tracce CN e CR ai passi 545–576. I dodici profili numerici dei tre bracci corrispondono al profilo previsto e fra loro. [Metriche CN](../../results/infra/contrastive_targets_20261118/benchmark_execution_20261121/CN/metrics.jsonl) · [Metriche CR](../../results/infra/contrastive_targets_20261118/benchmark_execution_20261121/CR/metrics.jsonl)

Entrambe le loss intervengono sul gradiente condiviso. A peso pieno la norma relativa mediana di InfoNCE è circa 2,6 volte quella di Rank-N-Contrast. Questa misura descrive l’intensità del termine ai pesi calibrati. Per scegliere quale loss favorisca il trasferimento servono i risultati di posa dinamica, P1 ed EPN612. Un solo run per braccio e questo breve tratto di training sostengono una verifica tecnica. Il picco è 19,21 GiB per GPU in tutti i bracci, il costo effettivo delle quattro allocazioni è 2,6133 GPU-ore e 20,9278 core-ore. [Raccolta operativa](launcher_targets_benchmarks_20261121.md)

**Proposta per Simone.** Conservare i pesi calibrati e passare alle quattro riprese già disegnate. Per ciascuna loss, caricare due volte il suo checkpoint al passo 544 e confrontare gli stessi 32 aggiornamenti fino a 576 fra le due riprese e con il continuo. Se le due riprese concordano e il continuo diverge, la localizzazione al primo scarto può isolare un effetto del ripristino. La lettura completa applicherà anche le regole congelate su optimizer, coda, RNG e profilo numerico. La fase resta in attesa della decisione sulle riprese. [Protocollo](../decisioni.md) · [Impronte dei checkpoint](../../results/infra/contrastive_targets_20261118/benchmark_execution_20261121/checkpoints_remote.txt)
