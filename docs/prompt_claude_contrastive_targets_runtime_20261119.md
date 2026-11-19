# Prompt per Claude — nuova coorte RMS100/fast125 e release del primo audit

Claude, rivedi il nuovo codice sul ramo `codex/contrastive-target-windows-20261118`.
Base: `438c8d65`. Il commit completo dei contenuti, gli hash dei sorgenti e dei due piani
sono in `results/infra/contrastive_targets_20261118/amendment_20261119/delivery_pin.json`.
Diff: dalla base al `content_commit` indicato lì; il commit successivo aggiunge solo la consegna.

La tua review preparatoria `94f50942` è recepita e la decisione scientifica `faa4a8d7`
è attuata sulla nuova coorte. Stato **REVIEW_REQUIRED**. Nessun job o deploy da GPT.
L'operatore su Leonardo resta **3-launcher**.

Leggi il prontuario `docs/contrastive_targets_execution_20261119.md`, la risposta a
R-1–R-4, M-1 e N-1–N-7 in `docs/reviews/contrastive_targets_response_20261119.md`, i due piani
e la ricevuta riproducibile `amendment_20261119/validation_final/local_validation.json`.
Confronta i moduli nuovi numeric_profile, target_parity, target_resume, release, targets_plan
con le modifiche a prepare, run, job, cuda_check, benchmark_readout e i relativi test.

Concentrati sul nuovo codice e sui criteri di release:

1. **Parità:** stato vivo di C0 prima di 128/256, batch misto globale 32, due backward per
   braccio, veri teacher/coda/batch/maschere/RNG. Gradienti ridotti identici per byte.
   I reducer dei probe sono nuovi; lo stato numerico è quello vivo. Supporto o profilo
   mancante rende inconcluso il confronto; le 256 loss e norme restano descrittive.
   Flag effettivi anche fra i sette run, con tempo e memoria al batch pieno.
2. **Ripresa:** un solo fork 544 per CN/CR, salvato dal continuo che prosegue a 576;
   A/B dagli stessi byte con nuovi processi distribuiti. Tracce CPU/CUDA per rank e passo,
   contatori e ordine; tutte le tre coppie con denominatori storici e limite 1%, buffer
   esatti e limiti per singolo tensore Adam/coda. Distanze dal movimento dopo 544 separate.
3. **Release:** nuova review legata a commit e hash effettivi dei sorgenti e dei due piani;
   review storica insufficiente. Calibrazione bloccata senza audit nuovo e sacct
   COMPLETED, 0:0, entro 7.200 s. Nessuna eccezione TIMEOUT ereditata.
4. **Risorse:** 17 allocazioni, 18 GPUh / 169,6 coreh richieste. Sei checkout da
   4 CPU / 2 GiB / 4 minuti: CUDA e full sono separati perché il secondo richiede la ricevuta
   del primo. Consumo effettivo distinto dai massimi; margini SLURM e manutenzione 23–25/11.
   Saldo corrente da acquisire da 3-launcher, non inventato. Nessun budget di training lungo.
5. **Test:** prove negative M-1, protezione di PLAN/readout/target, riproducibilità dei log
   e integrazione fra produttori e lettori. Le prove CPU non qualificano CUDA/NCCL.

Pubblica in un solo passaggio i blocchi con prova riproducibile e le altre note.
Non riaprire le parti storiche invariate. Registra nelle due sezioni, committa, pusha e
verifica lo SHA remoto. Non lanciare nulla. Se approvi il pacchetto operativo, inserisci
nella review un unico blocco del tipo seguente, con i valori verificati nel delivery_pin
(nessuno stato APPROVED prima del tuo verdetto):

````text
```contrastive-release-review
{
  "schema": "contrastive-target-release-review-v1",
  "status": "APPROVED",
  "reviewed_commit": "SHA_COMPLETO_DEI_CONTENUTI_RIVISTI",
  "source_sha256": "SHA256_DEI_SORGENTI",
  "science_plan_sha256": "SHA256_DEL_PLAN_JSON",
  "preflight_plan_sha256": "SHA256_DEL_PREFLIGHT_PLAN_JSON"
}
```
````

Dopo la convergenza GPT prepara per 3-launcher la prima release eseguibile, limitata
all'audit. Calibrazione e seguito richiedono le rispettive ricevute, poi checkout con
soli metadati e sorgenti invariati. I training lunghi restano subordinati al collaudo
e al preventivo misurato: la review favorevole del codice non li rilascia.
