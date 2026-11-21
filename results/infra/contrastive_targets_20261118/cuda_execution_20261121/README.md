# Collaudo CUDA RMS100/fast125 — raccolta operativa

Checkout 65016695 e CUDA 65016794 COMPLETED 0:0; codice congelato
4e880b9f6787d7294712370d00ed485bfbe45735. Nessun retry o modifica al codice.

cuda.json originale SHA256:
920bfb46c881c13b7edc9a93af01e9073a714908bb97081c46d266cb17f8133d.
Prova rank-exit originale SHA256:
ca4282219ffccb4b9faa6d1ce3576ecf62906661b3775b179a14d593da39c26a.

18 file originali verificati contro remote_sha256.txt. Sono raccolti l'intero
rank_exit_probe (stdout/stderr e quattro ricevute worker), cuda/ con i due STOP,
summary/config/metriche, rapporto principale, log SLURM e allocation.sacct.tsv.
Checkpoint diagnostico rimasto sul cluster: impronta e byte in checkpoints_remote.txt.
verify_results.log conserva l'esito positivo del verificatore congelato sul
checkout, compresa autenticazione della calibrazione e del checkpoint.

Tutti i 320 confronti numerici passano; copertura/numerica arrestano correttamente
su quattro rank; rank-exit positivo e nessun processo residuo. Nessuna fase
successiva inviata: il risultato deve essere discusso con Simone prima dei benchmark.
