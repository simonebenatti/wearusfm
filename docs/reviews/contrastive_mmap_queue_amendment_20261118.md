# Accodamento CUDA dopo il checkout — rettifica operativa del 18/11/2026

Il checkout 64730738 è PENDING per QOSMaxCpuPerUserLimit; un'altra campagna occupa la quota CPU.
Simone ha già autorizzato il rilancio e chiede di non mantenere monitor in background.
Si accoda dunque il SOLO CUDA con `--dopo CONTRAST-CHECKOUT-mmap-20261118` nel submitter comune.
La dipendenza SLURM `afterok:64730738` impedisce l'esecuzione se il checkout non termina con successo.

Il prerequisito «checkout completato e verificato prima di CUDA» viene verificato dai due launcher:
`campaign_checkout.sbatch` termina con successo soltanto dopo creazione del checkout allo SHA
atteso, pulizia e verifica di sola lettura; `contrastive_preflight_gpu.sbatch` esegue `code_dir_setup`
prima di caricare il job, ricontrollando SHA e pulizia. Non serve attendere sul Mac la quota CPU
per accodare un comando che non può ancora partire. Nessuna modifica di sorgenti, ricetta, job.json,
risorse, output, review pinnata o digest. Resta congelato 3b5d4be2bda36bdc601ae3c50ae9b8ace696f16e.

La rettifica riguarda solo il momento di sottomissione di CUDA. I cinque benchmark NON sono
accodati in anticipo: restano subordinati alla raccolta, verifica e pubblicazione del rapporto
CUDA positivo e del probe. Nessun monitor o automazione creato. Nessun nuovo consenso richiesto.
Comando concreto in `releases/mmap_20261118/queued_cuda_submission_plan.json`.

Il file sbatch deve esistere al momento dell'invio: si usa come solo trasporto il launcher
immutabile già presente in `contrastive-v2-20261117-benchmarks` (93f24c9), byte-identico a
quello nuovo, SHA256 `d3f128ae4dc1ea7565aa4603069f5578260f261268430b4c262b02c2d619128f`.
`CODE_DIR` e `EXPECTED_SHA` restano quelli NUOVI: libreria di controllo, Python, job.json e
sorgenti vengono letti dal nuovo checkout soltanto dopo afterok. L'hash del launcher remoto
è verificato prima dell'invio. Review indipendente di audit_timeout_review favorevole;
nessuna nuova review attribuita a Claude. Anche la sola lettura viene ricontrollata dal launcher.
