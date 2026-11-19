# Invii operativi dei diagnostici da stato comune

19/11/2026, 3-launcher. Review Claude 0afd0864, consegna db66b507,
piano 35ea94ba, codice congelato 365cb1e29bc410013d4ee0494b9c2e92835bc7f1.

Prima di ciascuno dei quattro invii: piano rigenerato identico byte per byte
a quello approvato; sei pin e submitter verificati; ID assente dal ledger;
quota debug ricontrollata e spazio disponibile circa 46 GiB (riserva 14 GiB).
Ref deploy pubblicato e verificato. Invii esclusivamente via submitter comune.

| Cella | ID SLURM | Primo stato raccolto |
|---|---:|---|
| checkout | 64835236 | COMPLETED 0:0, 35 s, 4 CPU |
| CUDA | 64835286 | PENDING Dependency |
| ripresa CN | 64835297 | PENDING Dependency |
| ripresa CR | 64835304 | PENDING Dependency |

Checkout HEAD verificato sul cluster: 365cb1e29bc410013d4ee0494b9c2e92835bc7f1.
Consuntivo checkout: 0 GPU-ore / 0,038889 core-ore.
Tetto GPU residuo prenotato: 6 GPU-ore. Nessun retry o duplicato.
Rapporti attesi in `$WORK/wearusfm_runs/results/contrastive_common_diagnostics_20261118/`
nelle sottocartelle cuda, resume_CN, resume_CR. Raccolta anche degli artefatti
negativi o parziali, con hash e accounting alla conclusione.
Lettura scientifica e decisioni su training/RMS100/fast125 restano alla chat
3-log quadratico e contrastive loss.

Snapshot successivo: CUDA e CN RUNNING (26 s ciascuno); CR PENDING
QOSMaxJobsPerUserLimit. Due sole celle debug in esecuzione; la terza è già in
coda e partirà automaticamente secondo quota, senza un nuovo invio.

## Raccolta conclusiva del 19/11

Tutti COMPLETED 0:0: checkout 35 s; CUDA 154 s; CN 289 s; CR 279 s.
Consuntivo incrementale **0,802222 GPU-ore / 6,456667 core-ore** contro
tetti 6 / 48,266667. Cumulativo campagna: 6,086667 GPU-ore / 57,827778 core-ore.

Rapporti in `$WORK/wearusfm_runs/results/contrastive_common_diagnostics_20261118/`:

| Cella | Stato rapporto | SHA256 report.json |
|---|---|---|
| cuda | MEASURED_FOR_REVIEW | 06a2be6b43f2c6ccf734ef5536d3351df30b321846344a82687155a45852dc5f |
| resume_CN | CONTROL_INPUTS_VERIFIED | cce7b19850a1705c3c60175fd6d12d29ce61cb132ac2d491c6b592352b678b19 |
| resume_CR | CONTROL_INPUTS_VERIFIED | 899d3ff9a012bf24cf64d4d5a52bad6481d192ef704b8ab2146bf9651c3585c7 |

Copie locali in `results/infra/contrastive_comparison_20261116/common_results_20261119/`:
rapporti CUDA per profilo, tracce dei quattro rank per fase (incluse DDP),
ricevute, log e manifest degli hash remoti. Checkpoint e segnali sul cluster.

### Misure da consegnare alla chat scientifica

CUDA: invarianti input/stato/RNG coincidenti; nessun profilo fallito, nessun
disaccordo di gradiente ridotto fra rank, supporto loss presente. Scarto relativo
del vettore gradiente prima del clipping (percentuale):

| Profilo | C0/C0 | CN/CN | CR/CR |
|---|---:|---:|---:|
| bf16 auto | 0,029617% | 0,028261% | 0,028834% |
| bf16 math | 0 | 0 | 0 |
| FP32 math | 0 | 0 | 0 |
| bf16 math deterministico | 0 | 0 | 0 |

FP32/bf16 a backend math uguale: 0,569780%, coseno 0,99998402.
bf16 auto/math: 0,068807%; il confronto cambia anche i flag TF32 dichiarati
nel protocollo. bf16 math/deterministico: zero. Nei profili math anche i
confronti fra bracci sono raccolti per blocco nei JSON.

Riprese: stato del confine, batch, microbatch e RNG prima del forward coincidono
in tutti gli otto passi, per ogni rank e fase. Primo scarto numerico nelle metriche:
**581 per tutte le tre coppie, in entrambi i bracci**. Distanze finali percentuali
rispetto allo spostamento 580→588 del primo membro di ogni coppia:

| Braccio/coppia | Student | Teacher |
|---|---:|---:|
| CN continuo/reload_a | 2,469117% | 0,103888% |
| CN continuo/reload_b | 2,452119% | 0,103877% |
| CN reload_a/reload_b | 2,516144% | 0,107230% |
| CR continuo/reload_a | 3,123384% | 0,141847% |
| CR continuo/reload_b | 2,999009% | 0,138477% |
| CR reload_a/reload_b | 3,398036% | 0,149863% |

Hash checkpoint comune a 580:
CN b21dad389ee22330cc51fe264bf9ec43b2e1676e214e5ec37a2b5560d6eff489;
CR f889637c11186854b910bab1e698c4e52c7f6efa2e3a751dfa5bf15c0c92d280.
Optimizer e coda: misure numeriche complete nei rapporti; nessuna nuova soglia
applicata. Tutti i rapporti hanno training_approved=false. La lettura e la
decisione scientifica sui training e sulla coorte RMS100/fast125 restano aperte.
