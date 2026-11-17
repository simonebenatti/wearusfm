# CUDA contrastivo: incompatibilità di caricamento, 17/11/2026

La calibrazione **64701482 è superata** e resta valida: λ InfoNCE
0,11942686122572264, λ Rank-N-Contrast 0,33044627364850215. Il successivo collaudo
**64706211 è FAILED 1:0**, dopo 67 secondi. Nessuno dei cinque benchmark è stato
sottomesso, perché manca il prerequisito CUDA positivo. Nessun retry.

## Evidenze e lettura

Checkout **64705989 COMPLETED 0:0**, 67 s, pulito allo SHA
`93f24c962e86551a232bdec90c1598f3d6b53173`. Invio CUDA con submitter/ledger comuni,
4 GPU, 32 CPU, 240 GiB, 30 minuti. Ricevute in
`results/infra/contrastive_comparison_20261116/cuda_benchmarks_submission_20261117/`.

Il probe di uscita intenzionale di un rank è **superato**: tutti e quattro i worker
hanno inizializzato CUDA/NCCL; uscita nonzero attesa in 12,415 s, nessun processo
residuo registrato. Ricevuta e hash di worker/log verificati dal lettore congelato
`contrastive_rank_exit.validate_receipt`. Questa evidenza non basta per il gate
numerico successivo.

Tutti e quattro i rank si fermano a `contrastive_cuda_check.py:234`, prima delle
traiettorie di riferimento, in `torch.load(..., mmap=True)`. L'argomento ottenuto
da argparse è un `pathlib.Path`; il PyTorch installato sul cluster richiede
esplicitamente `isinstance(f, str)` in quel ramo e solleva:

`ValueError: f must be a string filename in order to use mmap argument`

Il frammento del sorgente installato è archiviato insieme al log. `cuda.json`
non esiste. Dunque **confronti FP32/bf16, gradienti e STOP non eseguiti**: nessuna
conclusione sull'efficacia o correttezza numerica delle due loss da questo job.
È un difetto di compatibilità dello script GPT, non un esito negativo della
calibrazione. La verifica locale con PyTorch più recente non l'aveva intercettato.

## Correzione individuata, non applicata alla release congelata

Convertire il solo argomento percorso con `str(...)`, conservando `mmap=True` e
tutti gli altri argomenti. Sei siti in tre file: uno nel collaudo, uno nella
verifica di ripresa e quattro nel lettore finale. Patch proposta e manifest dei
vecchi/nuovi hash sono nell'archivio; `git apply --check` passa. I file runtime
non sono stati modificati. Review indipendente di `audit_timeout_review`: hash,
probe e diagnosi confermati; controllo AST della patch limitato alle sei conversioni,
tutti gli altri argomenti invariati. Nessuna nuova review attribuita a Claude.

I tre file partecipano al digest esecutivo `ab2d6460…`: applicare la patch cambia
il digest e i guard attuali rifiuterebbero audit/calibrazione/coorte storici.
Prima del rilancio serve quindi una rettifica tecnica esplicita e verificata:
diff limitato alle conversioni, prova di compatibilità, collegamento esatto
fra evidenze originali e sorgenti corretti. Non riscrivere le ricevute originali,
non aggirare i pin e non rifare automaticamente l'audit. Il rilancio dovrà usare
checkout, ID e output nuovi; i benchmark restano subordinati al collaudo numerico.
Questa nota non rilascia un nuovo job né cambia ricetta, λ o soglie.

## Archivio e consuntivo

Evidenze in `results/infra/contrastive_comparison_20261116/cuda_completed_20261117/`:
tar originale, log, sacct, probe completo, SHA remoti e locali coincidenti,
sorgente della verifica PyTorch e `readout.json`. Il probe è conservato anche
nel tar per includere integralmente i file `.log`.

CUDA: **0,074444 GPU-ore / 0,595556 core-ore**. Checkout: 0,074444 core-ore.
Totale consumato della catena, includendo l'audit in timeout e tutti i checkout:
**0,934444 GPU-ore / 16,128889 core-ore**. Nessun nuovo training o monitor.
