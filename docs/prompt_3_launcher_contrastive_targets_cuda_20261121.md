# 3-launcher · collaudo CUDA della coorte RMS100/fast125

Simone ha autorizzato il checkout e il collaudo CUDA. L’incarico termina con la raccolta e la restituzione del risultato alla chat scientifica. La decisione sui benchmark segue la lettura con Simone.

- Ramo `origin/codex/contrastive-target-windows-20261118`, SHA da eseguire **`4e880b9f6787d7294712370d00ed485bfbe45735`**.
- ID logici `CONTRAST-CHECKOUT-cuda-targets-v3-20261118` e `CONTRAST-CUDA-targets-v3-20261118`. Un tentativo per ID, tramite submitter e ledger condivisi.
- Prerequisiti conclusi: audit `64937087`, checkout della calibrazione `65014846` e calibrazione `65015116`. Il rapporto autenticato ha SHA-256 `20cb900522b62b46f17d1b65cbbe1ef579a4cdd5e935a7ab31b7a309ed516532`. Il checkpoint a 512 passi ha 234.339.194 byte e SHA-256 `d0dd318302f278e08513ec7d5acde3c82b36a523f7c650829167a68864870ebc`. [Lettura scientifica](reviews/contrastive_targets_calibration_science_20261121.md)

## Ordine degli invii

1. Verifica su origin lo SHA della consegna e quello di esecuzione. Dal worktree Mac `wearusfm-contrastive-targets`, rigenera il piano con il comando qui sotto e confrontalo byte per byte con quello pubblicato. Esegui le liste di argomenti `deploy` e `verify_deploy_sha`; salva lo SHA remoto. [Piano dei comandi](../results/infra/contrastive_targets_20261118/cuda_handoff_20261121/submission_plan.json)
2. **Prima di ogni sbatch** registra ora italiana, ledger, saldo e impegni residui del progetto, quota WORK e scritture attese, reservation, coda e slot effettivi. Mantieni 12 GiB per la coorte oltre alle scritture concorrenti (18 GiB ulteriori riservati al controllo precedente, da aggiornare) e il limite operativo di due slot debug per utente. Confronta la finestra disponibile con i tempi sotto.
3. Esegui `before_checkout` sul ref di deploy e conserva lo stdout. Verifica il binding alla review `469b312c`, l’accounting dell’audit e della calibrazione, i 14 SHA degli artefatti e le impronte del checkpoint. Il job CUDA deve coincidere byte per byte con quello rigenerato da `make-job`. Controlla a mano la corrispondenza tra cella `cuda` con `arm=null`, config di ingresso `C0/calibration_only=true`, target 100/125 ms e i due bracci tecnici CN/CR del worker con peso unitario. Il controllo usa il codice del runner per validare ricetta, checkpoint e argv sul login. La prova di uscita del processo e il confronto numerico avvengono nel job.
4. Esegui `checkout_only` con **4 CPU, 2 GiB, 4 minuti**. Una volta concluso, verifica dal ledger l’ID e da sacct `COMPLETED 0:0`, più SHA, pulizia e protezione in scrittura del checkout. Rinfresca le condizioni operative, esegui `before_cuda` sulla copia effettiva e registra lo stdout. Per i prerequisiti conclusi usa sacct e SHA; la sequenza di questa consegna non richiede `--dopo`.
5. Esegui `cuda_only`, controllando la ricevuta effettiva: **4 GPU, 32 CPU, 240 GiB, 30 minuti**, `boost_qos_dbg` e **`TORCH_DIST_INIT_BARRIER=1`** in `--export`. Registra gli ID appena assegnati. Un gate fallito ferma il passo. Un invio ambiguo si riconcilia sul ledger. Conserva i log originali.

```bash
set -o pipefail
python3 results/infra/contrastive_targets_20261118/cuda_handoff_20261121/prepare_handoff.py --frozen-sha 4e880b9f6787d7294712370d00ed485bfbe45735 --dry-run | cmp - results/infra/contrastive_targets_20261118/cuda_handoff_20261121/submission_plan.json
```

## Raccolta per la lettura scientifica

La radice di output è `$WORK/wearusfm_runs/results/contrastive_targets_20261118`. Sono richiesti percorsi nuovi per `cuda.json`, `cuda/` e `cuda.json.rank_exit_probe/`. Alla conclusione conserva la ricevuta di allocazione con exit code e tempi, log, rapporto originale, l’intera cartella della sonda di uscita (quattro worker e stdout/stderr inclusi) e le evidenze degli arresti di copertura e di stabilità numerica sotto `cuda/`. Salva anche le impronte dei checkpoint diagnostici eventualmente mantenuti sul cluster.

Sul checkout congelato, dopo aver caricato i moduli di campagna e attivato il venv, verifica il risultato con il lettore sotto. Esso autentica il rapporto CUDA, l’identità del checkpoint della calibrazione e l’arresto del processo, senza creare job. Un rapporto negativo o mancante va riportato con sacct e gli artefatti disponibili. Conserva i confronti fuori limite, identificando braccio, precisione, rank, passo, valore e soglia.

```bash
PYTHONPATH="src:scripts${PYTHONPATH:+:$PYTHONPATH}" "$WORK/venv_wearusfm/bin/python" results/infra/contrastive_targets_20261118/cuda_handoff_20261121/verify_results.py --execution-sha 4e880b9f6787d7294712370d00ed485bfbe45735
```

Verifica SHA e byte fra originali e copie. Pubblica sul ramo condiviso report, ricevute, stato dei 320 confronti, esiti degli arresti su quattro rank, e richiesto, effettivo e margine come campi distinti. Restituisci alla chat scientifica commit verificato, ID e risultato anche in caso di fallimento. **La catena si ferma qui per la lettura prima dei benchmark.**

## Risorse e calendario

| Allocazione | Richieste GPU-ore / core-ore | Margine SLURM GPU-ore / core-ore | Inviluppo GPU-ore / core-ore |
|---|---:|---:|---:|
| Checkout | 0 / 0,266667 | 0 / 0,4 | 0 / 0,666667 |
| CUDA | 2 / 16 | 0,4 / 3,2 | 2,4 / 19,2 |
| Coppia | **2 / 16,266667** | **0,4 / 3,6** | **2,4 / 19,866667** |

Consumo effettivo della nuova coorte prima della coppia: 0,99 GPU-ore e 16,953333 core-ore. Sommando le prenotazioni di audit, calibrazione e collaudo CUDA il totale richiesto è 4 GPU-ore e 48,8 core-ore sul tetto 18 / 169,6. Alla verifica di saldo applica il billing corrente e gli impegni aperti.

Manutenzione prevista dal **23/11 alle 08:00 CET** al 25/11 alle 08:00 CET. Per questa coppia l’ultimo avvio teorico del checkout è **23/11 alle 07:14 CET**: 4 + 30 minuti di allocazione e 6 minuti di margine per ciascuno dei due job. Anticipa per attese di coda e controlli tra le due fasi. Se i tempi effettivi non entrano nella finestra, rinvia il primo invio interessato alla riapertura confermata. [Calendario completo](prompt_3_launcher_contrastive_targets_audit_20261120.md)
