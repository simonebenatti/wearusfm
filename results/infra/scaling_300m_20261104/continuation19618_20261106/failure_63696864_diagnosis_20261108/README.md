# Fallimento del training 63696864 (08/11/2026) e diagnosi

Job `SCALING-300M-train19618-20261106` = 63696864: `FAILED 1:0` su lrdn2625 dopo 3 min 25 s (05:25:56-05:29:21), al controllo
dell'identita' dell'insieme fisso alla ripresa (`training/run._write_or_check`). Causa trovata: l'ordine di import (torch prima o
dopo numpy) cambia la libreria BLAS/LAPACK usata da numpy (MKL, resa globale da `libtorch_global_deps.so`, oppure l'OpenBLAS di
numpy) e con essa gli ultimi bit di `signals` (condizioni iniziali di `sosfiltfilt`) e `anchor_targets` (`power @ bands`).
Il run a 5.000 e i launcher di GPT (`torchrun scripts/train_jepa.py`) sono torch-first; il wrapper `scaling_continue.py worker`
e' numpy-first. I dati non sono cambiati. Voce completa in `docs/decisioni.md` (08/11/2026) e nel registro condiviso.

| Cartella | Job | Contenuto |
|---|---|---|
| `failed_job/` | 63696864 | `launch.json` e `training.log` della ricevuta parziale (traceback) |
| `slurm_logs/` | tutti | log SLURM del training fallito e dei quattro job di diagnosi |
| `diag1_63860499/` | 63860499 (48 s, 4 CPU) | ricalcolo dell'insieme fisso e confronto con il seme: finestre uguali, hash diversi |
| `diag2_63861166/` | 63861166 (1:40, 4 CPU) | determinismo, hash per componente, pannello della posa (8/64 uguali) |
| `diag3_63861825/` | 63861825 (2:18, 4 GPU, 0,153 GPU-ore) | nodo GPU: ripresa, da zero, senza DDP, su CPU: stesso contenuto nuovo |
| `diag4_63864244/` | 63864244 (3:24, 8 CPU) | tre ordini di import, CLI pura, impronte sha256 dei dati, differenze numeriche |
| `probe_login/` | login08 | `dladdr` e `/proc/self/maps`: chi fornisce i simboli BLAS e quando si caricano i kernel MKL |

Tabella del job 63864244 (hash uguali ad A = run 5.000 del 05/11; insieme fisso su 8 batch, pannello su 64):
`numpy_first` 0/8 e 8/64; `run_first` 0/8 e 8/64; `torch_first` 8/8 e 64/64; CLI pura `python3 scripts/train_jepa.py` 8/8 e 64/64.
Uscite grandi rimaste su Leonardo (non versionate): dump degli array in `$WORK/wearusfm_runs/diag/scaling_300m_20261104/
fixedeval_19618_20261108_c/*_arrays` (419 MB) e copie del checkpoint in `$SCRATCH/wearusfm_runs/diag/.../fixedeval_19618_20261108_gpu` (4,4 GB).
Consuntivo: training fallito 0,228 GPU-ore e 1,82 core-ore; diagnosi 0,153 GPU-ore e 1,84 core-ore (0,053 + 0,111 + 1,227 + 0,453).
