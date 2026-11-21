# 3-launcher · confronto CPU finale RMS100/fast125

Simone ha chiesto la lettura delle quattro riprese contro i tre continui. Questa è la cella finale del preflight firmato. I tre run continui e le quattro repliche terminano al passo 576. La lettura usa i checkpoint e i rapporti originali su WORK. Il ramo di consegna è `origin/codex/contrastive-target-windows-20261118`. [Piano firmato](../results/infra/contrastive_targets_20261118/preflight_plan.json)

| Ingresso | Continuo | Riprese dallo stesso checkpoint a 544 |
|---|---:|---|
| C0 | 65017240 | Il lettore verifica la parità a peso zero sul controllo. |
| CN, InfoNCE | 65017310 | CN_A 65033016 e CN_B 65033756 |
| CR, Rank-N-Contrast | 65020151 | CR_A 65033804 e CR_B 65034107 |

Le quattro riprese hanno chiuso con `COMPLETED 0:0`. Gli 88 file raccolti e le ricevute dei checkpoint finali attestano che A e B producono file identici entro ciascun braccio. Le impronte finali di studente e teacher differiscono rispetto al rispettivo continuo. Il compito della lettura CPU è misurare questi scarti e applicare i limiti già firmati.

## Checkout e confronto

Usa il commit di esecuzione `e0ddbac808c74f133cad619b8a61821e0afe18ea`. Le evidenze delle quattro riprese sono al commit `358ede81c07a49df75ffcbabf80aac0198ee540e`, già incluso nel primo. Verifica il piano dei comandi dal worktree prima del deploy. [Comandi dei due invii](../results/infra/contrastive_targets_20261118/readout_handoff_20261121/submission_plan.json)

```bash
set -o pipefail
handoff=results/infra/contrastive_targets_20261118/readout_handoff_20261121
python3 "$handoff/prepare_handoff.py" --frozen-sha e0ddbac808c74f133cad619b8a61821e0afe18ea --dry-run | cmp - "$handoff/submission_plan.json"
```

Le voci `before_checkout` e `before_readout` del piano controllano il ledger sul Mac e stampano gli argomenti SSH. Esegui direttamente il comando SSH stampato e conserva stdout, stderr e codice di uscita della connessione. La sola stampa del comando non attesta la verifica sul login. Per ciascun invio esegui direttamente la venv e il submitter comuni con la lista di argomenti pubblicata.

1. Dal worktree `wearusfm-contrastive-targets` verifica gli SHA su origin e rigenera il piano dei comandi, confrontandolo byte per byte con quello pubblicato. Pubblica e verifica il ref di deploy. Prima di entrambi gli invii aggiorna saldo e impegni di progetto, quota e scritture concorrenti su WORK, coda, slot della partizione seriale, reservation e calendario di manutenzione. La riserva di 12 GiB per la coorte si somma alle scritture concorrenti aggiornate.
2. Sul login ricontrolla accounting, codice e binding della review, identità delle sette ricevute e dei checkpoint archiviati, destinazione nuova e argomenti effettivi del lettore. Rispetta le directory fisiche dei run sotto `$WORK/wearusfm_runs/results/contrastive_targets_20261118`. I nomi CN_A e CN_B e i nomi CR_A e CR_B corrispondono a directory distinte. La verifica sul login prepara e ispeziona il comando. Il calcolo delle distanze fra tensori spetta al job CPU.
3. Usa il submitter e il ledger comuni. Sottometti una volta `CONTRAST-CHECKOUT-readout-targets-v3-20261118` con **4 CPU, 2 GiB e 4 minuti**. Prima di inviare il lettore attendi `COMPLETED 0:0` sullo stesso ID e verifica SHA, pulizia e protezione del checkout. Per un ID già presente riconcilia stato e ricevuta. Un prerequisito concluso si verifica con sacct e SHA.
4. Esegui il controllo per la cella sul checkout effettivo, poi sottometti una volta `CONTRAST-READOUT-targets-v3-20261118` con **8 CPU, 30 GiB, 60 minuti e 0 GPU** sulla seriale. Comunica gli ID di checkout e lettura appena assegnati. In attesa di quota prosegui appena disponibile secondo la regola del via unico. Preserva i log del primo tentativo. Se un controllo si interrompe o un invio dà una risposta ambigua, individua il punto e riconcilia il ledger prima di ulteriori azioni.

## Risorse e calendario

| Per il checkout e la lettura CPU | Richiesto | Margine SLURM | Inviluppo |
|---|---:|---:|---:|
| GPU-ore | 0 | 0 | 0 |
| Core-ore | 8,266667 | 1,2 | 9,466667 |

Con questa tranche le richieste dell’intero preflight raggiungono **18 GPU-ore e 169,6 core-ore**, inviluppo 21,6 e 202,4. Registra separatamente gli effettivi da sacct, le ore locali al billing corrente e il saldo al netto degli impegni. Con chiusura il 23/11 alle 08:00 CET, l’ultimo inizio teorico del checkout della coppia è alle **06:44 CET** inclusi i margini (10 + 66 minuti), da anticipare per coda, verifiche e prenotazioni. Ricontrolla la finestra effettiva sul centro.

## Rapporto e consegna

Raccogli il JSON originale di esito con percorso, dimensione e SHA-256 verificati su WORK e sul Mac, le ricevute del submitter, accounting, stdout/stderr, esito di ogni controllo pre-invio e il consumo per allocazione. Il lettore confronta per ciascuna loss continuo/A, continuo/B e A/B. Registra contesto numerico dei 28 rank, integrità del confine a 544, esiti delle coppie ed eventuali ragioni di fallimento o inconclusività. Per le prime differenze tra input, RNG o ordine delle operazioni conserva i riferimenti a passo, campo e rank. Distingui le distanze rispetto all’inizializzazione (criterio firmato) da quelle rispetto al solo movimento 544–576 (descrittive).

Con rapporto `PASSED_FOR_REVIEW` e uscita zero, comunica il risultato per la lettura scientifica. Se il lettore scrive `FAILED_REQUIRES_REVIEW` o `INCONCLUSIVE_REQUIRES_REVIEW` e termina con codice 2, archivia comunque il JSON: è il verdetto del protocollo. Se l’esecuzione si interrompe prima di produrre un rapporto completo, conserva traceback, ultimo controllo e accounting per la diagnosi. Comunica lo SHA del commit delle evidenze dopo push verificato. Gli esiti riportano `training_approved=false`; gli eventuali training lunghi richiederanno la decisione scientifica e il preventivo misurato.
