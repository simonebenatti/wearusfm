# R7 - Critico di completezza e coerenza (commit 9d34dc69)

## Eseguito
- 4 file di test: `32 passed in 7.97s` (venv WearUsFM, 1 thread, PYTHONPATH=src:scripts:scripts/diagnostics).
- `--package`: package_sha256 b0640ca6...dcfa3, plan_sha256 9d5811b6...0d6 (coincidono col blocco richiesto).
- Composizione dei 7 bucket ricostruiti: hash di `rebuilt_per_bucket_param_indices` identico fra continuo (545) e CN_A/CN_B (547): da69c6759cab. Il controllo `layouts()` della warm e' quindi soddisfacibile.
- 8 campi di gradiente: CN_A == CN_B su tutti i 32 passi; presenti a ogni passo. Al 545 i campi `contrast_global_*` sono gia' uguali al continuo, differiscono solo grad_norm, grad_clip_factor, contrast_projector_grad_norm.
- 31 file archiviati nel clone = sha letti da Claude su Leonardo.
- Simulazione CPU `agents/r7-critico/roundoff.py`: somma a 4 rank in due ordini; rel L2 5,4e-08 per gradienti generici, 3,1e-04 per contributi che si annullano fra rank.
- Simulazione CPU `agents/r7-critico/cuda_rng_sim.py` + lettura vera di Claude (`leonardo/confronto_checkpoint_veri_20261121.out.txt`): continuo vs CN_A `metadata_or_schema_different ['cuda_rng.1','cuda_rng.2','cuda_rng.3']`.

## Rilievi
- B-2 bloccante (anche R1/R4): compare.py:248 rende INCONCLUSIVE ogni warm, anche perfettamente recuperata, per `cuda_rng[1..3]` del rank 0 (seme run.py:1463 vs 1383).
- I-1 importante: soglia 1e-6 per tensore relativa alla sola norma ridotta: tensori con cancellazione fra rank superano la soglia per puro arrotondamento -> HYPOTHESIS_UNSUPPORTED errato.
- I-2 importante: difetti solo-CPU (B-1, B-2) rendono il risultato recuperabile solo con un confronto rieseguito dopo la misura: va registrata ora la regola.
- M-1 minore: costanti del piano non lette dal codice.
- M-2 minore: --gpu-ore 0.59 nel ledger contro la regola GPU x walltime.
- M-3 minore: tempi per gamba non registrati; valori di Adam al 545 solo come impronte.
- N: segni dello zero, effetto del riscaldamento su NCCL/allocatore, ordine delle gambe.
