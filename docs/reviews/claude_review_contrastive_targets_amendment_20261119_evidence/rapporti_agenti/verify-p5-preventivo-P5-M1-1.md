# Verifica scettica P5-M1 (checkout 30 GiB vs 2 GiB) - scettico n. 1

HEAD 438c8d65 (file del piano invariati dopo 0efdc29c). Solo letture, nessun file modificato nella copia.

## Riproduzione: fatti CONFERMATI
- preflight_plan.json:174-180 `checkouts`: count 4, cpus 4, memory_gib 30, seconds_each 240, core_hours 1,0666667.
- docs/contrastive_targets_amendment_20261118.md:69 `| Checkout |4|4 CPU,30 GiB|4 min|0|1,066667|`.
- scripts/slurm/campaign_checkout.sbatch:5-7: --cpus-per-task=1, --mem=2G, --time=00:10:00.
- Piani storici: audit_mem30 :22-24, calibration :23-25, cuda_benchmarks :22-24, readout_bf16 :29-31, bf16, mmap,
  diagnostica (submission_plan :29): tutti `--cpus-per-task=4 --time=00:04:00 --mem=2G`. readout_bf16/resources.json:7
  memory_gib 2; common_state_diagnostics/resources.json:13 requested_gib 2.
- sacct 64730738 e 64746634: billing=4,cpu=4,mem=2G. Eccezione non citata: 64775502 (checkout della lettura) ha ReqMem 8G
  (benchmarks_completed_bf16_20261118/accounting.txt:13) pur col piano a 2G: comunque lontano da 30G.
- Nessuna motivazione dei 30 GiB in plan.json, README, decisioni.md, registro. Il preventivo storico
  (docs/contrastive_preflight_execution_20261116.md:25) non dichiarava memoria; la release storica della lettura
  (docs/reviews/contrastive_readout_release_20261118.md:39) dichiara 2 GiB. Probabile copia dalla riga delle celle CPU.

## Motivi che RIDIMENSIONANO il rilievo
1. Nessun codice/test legge preflight_plan.json della nuova coorte: grep "contrastive_targets_20261118" in scripts/ tests/cpu/
   src/ trova solo scripts/contrastive_prepare.py:30 (plan.json). Piano executable=false, REVIEW_REQUIRED, nessuna release.
   local_validation controlla solo 13 allocazioni / 14 GPUh / 137,0667 coreh, non la memoria.
2. Costo: l'ipotesi "fino al doppio" e' CONFUTATA dalle ricevute. Su lrd_all_serial la memoria non entra nel billing:
   - 63775679 (07/11, partition QOS lrd_all_serial, logsquare_evaluation_20261107/cpu_launch_manifest.json:189-195):
     AllocCPUS 2, billing=2,cpu=2,mem=30G;
   - 62618174 (28/10, Partition lrd_all_serial, pilot_posa/b19618_20261029/final/sacct_all.stdout.txt): ReqCPUS 2,
     ReqMem 30G, billing=2;
   - 64652559 audit (17/11): billing=8,cpu=8,mem=30G.
   Tabulazione di tutte le AllocTRES nel repo: billing == cpu in ogni caso (2/30G, 2/16G, 4/16G, 8/30G, 4/2G, 2/2G, 2/4G).
   Quindi 4 CPU/30 GiB => billing 4: 1,066667 coreh e tetto 137,066667 esatti anche con 30 GiB.
3. Limiti: 30 GiB = 30720 MiB <= MaxTRESPerJob mem=30800M (qos_memory_diagnosis.json): nessun rifiuto, nessun fallimento.
4. Coda: applicati alla lettera restano 80 MiB della MaxTRESPU mem=30800M. Blocco aggiuntivo rispetto a 2G solo verso job
   seriali dello stesso utente con <=4 CPU e <=28 GiB (contro job a 8 CPU blocca gia' il limite cpu=8; contro job a 30G
   bloccherebbe anche un checkout da 2G). Checkout storici 36-61 s: ritardo trascurabile, nessun effetto su misure o tetti.
5. I costruttori storici delle release scrivono gli argomenti a mano (common_state_diagnostics_20261118/prepare_handoff.py:64-65
   `--cpus-per-task=4 --time=00:04:00 --mem=2G`), non dal preflight; prima del primo invio servono review unica del pacchetto
   e pin (preflight_plan.json:190-195).

## Esito
- refuted=false: la discrepanza dichiarata/praticata esiste ed e' riprodotta.
- Gravita' reale: NOTA (non MINORE). Resta un valore di documentazione errato e non motivato nel preventivo da presentare;
  il costo e' dimostrato invariato (billing solo CPU su lrd_all_serial), nessun limite superato, nessun fallimento, effetto di
  coda solo ipotetico e trascurabile. Correzione: scrivere 2 GiB (o 4 CPU, 2 GiB, 240 s) in preflight_plan.json e nella
  tabella dell'emendamento; nessun ricalcolo dei costi.
- Imprecisione del rilievo: "tutti i checkout storici 2 GiB" vale per i piani, non per l'accounting (64775502 ReqMem 8G).
