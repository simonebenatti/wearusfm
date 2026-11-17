# Rilascio operativo del collaudo contrastivo — 17/11/2026

**APPROVED_FOR_PREFLIGHT da GPT sotto autorizzazione esplicita di Simone.** Simone: «direi che se tu sei a posto per questo giro lanci tu, ti do fiducia». Per questo giro GPT assume la decisione di rilascio e il lancio diretto. Questa istruzione successiva sostituisce l'attesa della nuova review di Claude, non i cancelli tecnici. Non è una review favorevole attribuita a Claude: la sua ultima review resta `0ab9d98`, sulla versione precedente.

Consegna esaminata: `c680e95`, prompt/registro `914b735`. Verifica `results/infra/contrastive_comparison_20261116/review_response/verification_v2.json`: 725 test mirati passati, sei prove distribuite CPU incluse. Sorgenti scientifici invariati. Digest iniziale `651d3d82…`, aggiornato a `ab2d6460bfe090e4adba8cfd480b726077005c4ed6ae10db7ebbd5909ddf808b` per la sola correzione operativa della memoria descritta sotto. Nessuna estensione scientifica o modifica di soglie per ottenere il rilascio.

## Perimetro approvato

Solo la catena preparatoria pubblicata: audit CPU; calibrazione senza aggiornamento dei pesi; prova CUDA/NCCL e uscita di un rank; benchmark C0/CN/CR e riprese CN/CR; lettura e preventivo. Tetti invariati: **14 GPU-ore, 129,3333 core-ore**, quattro checkout inclusi. Il budget approvato dal presente via riguarda questo collaudo, non il riferimento storico dei tre training.

Il primo invio comprende checkout CPU (4 CPU, 5 minuti, 0,3333 core-ore) e audit (8 CPU, 30 GiB, 60 minuti, 8 core-ore), **zero GPU-ore**. ID logici `CONTRAST-CHECKOUT-AUDIT-v2-20261117` e `CONTRAST-AUDIT-v2-20261116`; ledger comune, nessun duplicato. Output nuovo `$WORK/wearusfm_runs/results/contrastive_preflight_20261116_v2/audit.json`.

## Giudizio di prontezza

Configurazioni rigenerate uguali al costruttore; riferimenti di rarità/gradiente e input reali verificati per SHA; modello storico e guardie dei checkout precedenti preservati. Release del job, review e dati sono pinnati; il launcher richiede checkout pulito, immutabile e allocazione esatta. La partizione seriale è UP e ammette il limite richiesto. La verifica finale degli input avviene nuovamente sul nodo nel job.

Ogni fase successiva richiede il risultato completo e valido della precedente, la raccolta delle evidenze, una release concreta pubblicata e un checkout congelato. Nessun lancio di calibrazione sulla sola esistenza del JSON di audit: servono i verdetti di copertura, il pannello, i pin e l'esito SLURM. Nessuna risottomissione automatica dopo errore; si diagnostica prima. La lettura dei risultati resta autonoma nell'ambito approvato, senza nuova richiesta di consenso.

Il piano scientifico resta immutato. Il suo stato documentale REVIEW_REQUIRED ricorda che Claude non ha ancora riletto le correzioni: lo stato esecutivo di questo collaudo è nel piano operativo e nelle release distinte, con la presente autorità dichiarata. I training scientifici non sono rilasciati: richiedono prima misure, costi e completamento dei valutatori. Nessun monitor o automazione in background viene creato.

## Rettifica operativa dopo rifiuto di sbatch

Il checkout 64652044 è COMPLETED 0:0 in 30 secondi (4 CPU, 0,033333 core-ore). L'audit è stato **rifiutato prima di creare un job**: la QoS seriale ha MaxTRESPU e MaxTRES `cpu=8,mem=30800M,node=1`, inferiore ai 32 GiB richiesti dal launcher. È un errore del preventivo/launcher GPT, non della pipeline o dei dati.

Corretti esclusivamente la memoria del launcher CPU a 30 GiB e la relativa guardia; regressione che accetta 30 e rifiuta 32 GiB. 18 test mirati passati, shell syntax e diff controllati. Nessuna loss, ricetta, soglia o input cambiato. Nessun audit/calibrazione eseguito con il vecchio digest; nessuna evidenza scientifica da riutilizzare o invalidare. Diagnosi e ricevuta del rifiuto archiviate.

La ripresentazione è esplicita con `--dopo-rifiuto` sullo stesso ID logico di audit, dopo un nuovo checkout `contrastive-v2-20261117-audit-mem30`. Il nuovo checkout richiede 4 CPU per **4 minuti**: costo massimo 0,266667 core-ore; insieme al vecchio checkout completato il massimo di questa fase è 0,3 core-ore, entro gli originali 0,333333. Il tetto complessivo di 129,3333 core-ore e 14 GPU-ore resta invariato. Il checkout originale non viene modificato.
