# Verifica scettica n. 2 del rilievo P5-M1 (checkout 30 GiB)

Copia 438c8d65, sola lettura, nessun file modificato nella copia.

## Fatto di base: RIPRODOTTO
- preflight_plan.json:174-180 `checkouts` = count 4, cpus 4, memory_gib 30, seconds_each 240, core_hours 1,0666...
- docs/contrastive_targets_amendment_20261118.md:69 `| Checkout |4|4 CPU,30 GiB|4 min|0|1,066667|`.
- scripts/slurm/campaign_checkout.sbatch:5-7 `--cpus-per-task=1`, `--mem=2G`, `--time=00:10:00`.
- Tutti i sei checkout_state.json storici della linea 3 (submission_20261117 x2, calibration, cuda_benchmarks, mmap, bf16):
  `--cpus-per-task=4`, `--mem=2G`, `--time=00:04:00` (il primo, CONTRAST-CHECKOUT-AUDIT-v2-20261117, `00:05:00`: "tutti 240 s"
  del rilievo e' leggermente impreciso). Piani readout_bf16 e common_state_diagnostics: `--mem=2G`; resources.json 2 GiB.
- sacct 64730738 (36 s) e 64746634 (53 s): COMPLETED 0:0, billing=4,cpu=4,mem=2G.
- Il preventivo storico (docs/contrastive_preflight_execution_20261116.md:25, preflight storico resources.checkouts) non
  dichiarava memoria per i checkout: i 30 GiB sono nuovi e coincidono con la memoria del launcher CPU
  (contrastive_preflight_cpu.sbatch:7 `--mem=30G`), cioe' con ogni probabilita' copiati dalla riga audit/lettura.
  Nessuna motivazione dei 30 GiB nell'emendamento, nel registro o in decisioni.md (diff e388a1de..0efdc29c).

## Gamba del costo: CONFUTATA
- Il peso della memoria nel billing di lrd_all_serial e' determinabile dalle ricevute del repo: righe sacct con
  `lrd_all_serial` -> billing=2,cpu=2,mem=30G (2), billing=2,cpu=2,mem=16G (2), billing=2,cpu=2,mem=4G (3), billing=8,cpu=8,mem=30G (93).
  Es. 62618174 (results/infra/pilot_posa/b19618_20261029/final/sacct_all.stdout.txt): lrd_all_serial, 2 CPU, 30G, billing=2, login08.
- Quindi 4 CPU/30 GiB avrebbe billing=4 come 4 CPU/2 GiB: 1,066667 core-ore in ogni caso, tetto 137,066667 invariato.
  L'ipotesi "fino al doppio (+1,07 coreh)" non regge.

## Gamba dei limiti: nessun superamento
- 30 GiB = 30.720 MiB <= 30.800 MiB (MaxTRES per job e MaxTRESPU di lrd_all_serial,
  docs/reviews/contrastive_preflight_release_20261117.md:23, docs/reviews/claude_review_consolidation_long_20261105.md:78).

## Gamba della coda: reale ma condizionata e marginale
- Condizionata: nessun codice legge preflight_plan.json della linea 3 (grep 'preflight_plan' / 'contrastive_targets_20261118'
  in scripts, src, tests: solo plan.json da contrastive_prepare.py:30); il piano e' executable:false, submission_authorized:false.
  Gli argomenti sbatch veri saranno scritti nella futura release; il modello storico
  (common_state_diagnostics_20261118/prepare_handoff.py:65) passa `--mem=2G`.
- Marginale: con MaxTRESPU cpu=8 un checkout da 4 CPU aspetta comunque dietro ogni job seriale da 8 CPU dell'utente (tipo
  dominante, 93 righe su 100 di lrd_all_serial nelle ricevute) e dietro ogni job da >28,8 GiB; MaxJobsPU 2. La differenza fra
  2 e 30 GiB riguarda solo la convivenza con job piccoli (<=4 CPU, <=28,8 GiB) dello stesso utente. L'audit stesso (8 CPU/30 GiB)
  richiede gia' tutto l'inviluppo per utente.

## Esito
- refuted=false sul fatto (discrepanza certa e riproducibile), ma la gravita' scende da MINORE a NOTA: incoerenza
  documentale di un preventivo non eseguibile, senza effetto su costo (billing = CPU), limiti o attribuzioni; l'effetto
  di coda richiede l'adozione letterale nella futura release ed e' limitato ai job piccoli concorrenti.
- Correzione suggerita invariata: dichiarare 4 CPU, 2 GiB, 240 s (core-ore invariate) o motivare i 30 GiB.

## Rettifica dei conteggi (job distinti, non occorrenze)
- Righe sacct con partizione lrd_all_serial, per JobID distinto: 27 job; 24 a 8 CPU/30G (billing=8), 1 a 2 CPU/30G (62618174,
  billing=2), 1 a 2 CPU/16G (billing=2), 1 a 2 CPU/4G (billing=2). Le cifre 93/100 sopra contavano occorrenze (ReqTRES+AllocTRES).
  Conclusioni invariate: billing = numero di CPU fino a 15 GiB per CPU; job seriali tipici a 8 CPU.
