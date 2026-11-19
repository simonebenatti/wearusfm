> **Emendamento operativo del19/11:** [controlli da stato comune e nuovo preventivo](contrastive_targets_execution_20261119.md). Diagnostici storici conclusi; la tabella sotto conserva la proposta iniziale da14GPUh, sostituita dalla matrice nuova da18GPUh.

# Linea 3: nuova coorte con RMS100 e spettro rapido125

18/11/2026. **REVIEW_REQUIRED, nessun invio, nessun deploy e nessuna λ nuova misurata.**
Mandato di Simone in questa chat, dopo il chiarimento sul perimetro: «fai tutto tu».
Questa consegna integra il lavoro di Claude `0d91297` e prepara l'adeguamento della linea3
in un worktree separato. Non modifica i file non committati della diagnostica in corso
in `wearusfm-logcontrast`, né il ramo della proposta TCN.

## Ricetta e domanda scientifica

| Elemento | T100-G e coorte contrastiva storici | Nuova coorte C0/CN/CR |
|---|---|---|
| Supporto target log-RMS |25 ms|100 ms|
| Supporto target spettro rapido |50 ms|125 ms|
| Versione target spettro |`step1_psd_power_shape_energy_v1`|`step1_psd_power_shape_energy_fast125_v3`|
| Passo dei target |25 ms|25 ms|
| Tokenizer |100 ms, GELU|invariato|
| Architettura, dati, dose, seme, maschere, pesi posa/ancore |ricetta firmata|invariati|

Tutti i piani dei tre bracci e la calibrazione scrivono i due campi del loader.
La rappresentazione CLI esplicita è `--rms-window-ms 100 --ms-fast-window-ms 125`.
Il dispatcher contrastivo usa i config JSON pinnati, non i default di una CLI implicita.
Il parser dei config storici applica `with_legacy_target_windows`: assente significa25/50,
un valore presente resta quello scritto. La baseline T100-G viene prima verificata come25/50/v1,
poi i quattro config nuovi sono costruiti esplicitamente come100/125/v3.

**CN−C0 e CR−C0 restano confronti con target identici fra bracci.** Il confronto con T100-G
storico cambia anche RMS e spettro rapido: non isola più la sola infrastruttura contrastiva
e non misura l'effetto di un singolo nuovo target. Non cambiano contrasti, bootstrap o soglie
del protocollo statistico `contrastive-comparison-20261116-v2`; l'emendamento della ricetta
ha identità separata `contrastive-target-windows-20261118-v1`.

## Evidenze da rifare e blocchi già aperti

1. Audit nuovo di1.024 batch; nessuna eredità dell'eccezione TIMEOUT del vecchio audit.
2. Calibrazione train-only di512 batch, senza aggiornamenti; stesse regole di selezione
   dei32 batch informativi e copertura. Entrambe le λ restano null nei config preparatori.
3. Nuovo controllo CUDA, tre benchmark completi e due prove di ripresa; nessun riuso
   delle vecchie ricevute per certificare la nuova ricetta.
4. Lettura dei gate e preventivo misurato prima di un eventuale training scientifico.

Il coefficiente storico0,1179564123141922 resta il prior già prespecificato nella formula
di λ: **non è una misura dell'attenuazione del gradiente con100/125**. I rapporti di norme
e λ vanno ricalcolati sul train nuovo; i gate di ampiezza dell'intervento restano invariati.
Un fallimento ferma la coorte; nessun ritocco automatico della formula o clipping di λ.
Il confronto dei primi otto valori con lo storico è descrittivo: anche il primo update
può differire perché i target sono cambiati.

**Il cambio dei target non chiude i fallimenti di parità iniziale/ripresa bf16 già trovati.**
Prima di rilasciare questa nuova catena va letta la diagnostica da stato comune preparata
nell'altra chat, poi registrata la risoluzione del problema o un emendamento esplicito.
Non si ripete alla cieca la matrice negativa. Nessuna soglia viene allentata qui.

I vecchi piani, config, log e ricevute restano immutati sotto
`results/infra/contrastive_comparison_20261116/`. Il nuovo piano è
`results/infra/contrastive_targets_20261118/plan.json`; i lettori correnti puntano a questo.
I lettori congelati restano il modo di riprodurre le analisi vecchie. La whitelist storica
di compatibilità sorgenti non viene ampliata per far passare un cambio scientifico dei target.

## Preventivo da presentare prima degli invii

| Fase | Allocazioni | Risorse per job | Limite | GPU-ore | Core-ore |
|---|---:|---|---:|---:|---:|
| Audit |1|8 CPU,30 GiB|2 h|0|16|
| Calibrazione + CUDA |2|4 GPU,32 CPU,240 GiB|30 min|4|32|
| Tre benchmark completi |3|4 GPU,32 CPU,240 GiB|30 min|6|48|
| Due riprese |2|4 GPU,32 CPU,240 GiB|30 min|4|32|
| Lettura |1|8 CPU,30 GiB|1 h|0|8|
| Checkout |4|4 CPU,30 GiB|4 min|0|1,066667|
| **Totale massimo proposto** |**13**|||**14**|**137,066667**|

L'audit precedente consumò3.787s contro3.600 richiesti: riproporre un'ora sarebbe
ingiustificato. Il dispatcher nuovo ammette esattamente7.200s per audit,3.600s per lettura,
1.800s per le fasi GPU. Il launcher CPU conserva il default1h della lettura; per l'audit
il piano richiede l'override SLURM `--time=02:00:00` e controlla il tempo effettivo.
Sono tetti proposti, **non misure della ricetta100/125 né una release**. Non comprendono
training scientifici, nuove diagnosi, retry o valutazioni downstream. Il nuovo tempo di training
sarà misurato nei benchmark; il vecchio riferimento50,83GPUh per tre bracci non è un preventivo nuovo.

Gli output nuovi hanno radice `$WORK/wearusfm_runs/results/contrastive_targets_20261118`.
Prima del rilascio: quota e saldo reali, scritture pendenti, corsia, manutenzioni e ledger,
pin sorgenti/review/config/input e un solo operatore. Nessun checkpoint, ID o output esistente
viene riutilizzato per un nuovo tentativo. Costi e risorse sono nel `preflight_plan.json` nuovo;
le celle non contengono release o SHA fittizi. L'autorizzazione dei vecchi invii non viene
copiata nel nuovo piano come se i nuovi gate fossero già passati.

## Fuori perimetro

Continuazione300M:25/50 esplicito, codice congelato. TCN della posa: latenti e bersagli
già in cache, nessuna nuova estrazione. Il limite LQ degli artefatti precedenti al04/11
resta documentato; non lo si aggira completando indiscriminatamente i config.
La guardia generica per distinguere piani WSD vecchi e nuovi è rinviata: non è il percorso
di esecuzione contrastivo, che verifica il config completo prima del lancio. Nessuna modifica
silenziosa allo schema dei piani storici in questa consegna.
