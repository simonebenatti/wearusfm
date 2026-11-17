# Rilascio operativo del collaudo contrastivo — 17/11/2026

**APPROVED_FOR_PREFLIGHT da GPT sotto autorizzazione esplicita di Simone.** Simone: «direi che se tu sei a posto per questo giro lanci tu, ti do fiducia». Per questo giro GPT assume la decisione di rilascio e il lancio diretto. Questa istruzione successiva sostituisce l'attesa della nuova review di Claude, non i cancelli tecnici. Non è una review favorevole attribuita a Claude: la sua ultima review resta `0ab9d98`, sulla versione precedente.

Consegna esaminata: `c680e95`, prompt/registro `914b735`. Verifica `results/infra/contrastive_comparison_20261116/review_response/verification_v2.json`: 725 test mirati passati, sei prove distribuite CPU incluse. Sorgenti scientifici ed esecutivi invariati: digest `651d3d8291db5afe0fb8d7b3ffc53b5234960bbe0a9236a1f853e6cee7b1eca5`. Nessuna estensione scientifica o modifica di soglie per ottenere il rilascio.

## Perimetro approvato

Solo la catena preparatoria pubblicata: audit CPU; calibrazione senza aggiornamento dei pesi; prova CUDA/NCCL e uscita di un rank; benchmark C0/CN/CR e riprese CN/CR; lettura e preventivo. Tetti invariati: **14 GPU-ore, 129,3333 core-ore**, quattro checkout inclusi. Il budget approvato dal presente via riguarda questo collaudo, non il riferimento storico dei tre training.

Il primo invio comprende checkout CPU (4 CPU, 5 minuti, 0,3333 core-ore) e audit (8 CPU, 32 GiB, 60 minuti, 8 core-ore), **zero GPU-ore**. ID logici `CONTRAST-CHECKOUT-AUDIT-v2-20261117` e `CONTRAST-AUDIT-v2-20261116`; ledger comune, nessun duplicato. Output nuovo `$WORK/wearusfm_runs/results/contrastive_preflight_20261116_v2/audit.json`.

## Giudizio di prontezza

Configurazioni rigenerate uguali al costruttore; riferimenti di rarità/gradiente e input reali verificati per SHA; modello storico e guardie dei checkout precedenti preservati. Release del job, review e dati sono pinnati; il launcher richiede checkout pulito, immutabile e allocazione esatta. La partizione seriale è UP e ammette il limite richiesto. La verifica finale degli input avviene nuovamente sul nodo nel job.

Ogni fase successiva richiede il risultato completo e valido della precedente, la raccolta delle evidenze, una release concreta pubblicata e un checkout congelato. Nessun lancio di calibrazione sulla sola esistenza del JSON di audit: servono i verdetti di copertura, il pannello, i pin e l'esito SLURM. Nessuna risottomissione automatica dopo errore; si diagnostica prima. La lettura dei risultati resta autonoma nell'ambito approvato, senza nuova richiesta di consenso.

Il piano scientifico resta immutato. Il suo stato documentale REVIEW_REQUIRED ricorda che Claude non ha ancora riletto le correzioni: lo stato esecutivo di questo collaudo è nel piano operativo e nelle release distinte, con la presente autorità dichiarata. I training scientifici non sono rilasciati: richiedono prima misure, costi e completamento dei valutatori. Nessun monitor o automazione in background viene creato.
