# Collaudo del confronto contrastivo — revisione del 16/11/2026

**Aggiornamento operativo 17/11:** Simone ha affidato a GPT il rilascio e il lancio di questo giro. La [release operativa](reviews/contrastive_preflight_release_20261117.md) supera l'attesa della nuova review di Claude per il solo collaudo, mantenendo tutti i gate e i tetti. Il testo seguente conserva la proposta del 16/11.

Stato: **REVIEW_REQUIRED**, nessun job sottomesso da questa consegna. Simone ha scelto D-1 A, D-3 opzione 2 e D-2 EPN612 principale; ha autorizzato GPT a sottomettere dopo la convergenza con Claude. La review 0ab9d98 riguarda la versione precedente e non autorizza da sola questa versione corretta. La convergenza sarà registrata in una release distinta, senza chiedere un secondo consenso per lo stesso piano.

La ricetta scientifica è in `contrastive_comparison_protocol_20261116.md` e `../results/infra/contrastive_comparison_20261116/plan.json`. Il piano risorse leggibile dalla macchina è `../results/infra/contrastive_comparison_20261116/preflight_plan.json`. Questo collaudo usa solo il train. Non è uno dei tre training scientifici a 19.618 passi.

## Sequenza e costo massimo

| Cella | Lavoro | Dipendenza | Allocazione | Tetto GPU-ore | Tetto core-ore |
|---|---|---|---|---:|---:|
| Audit | 1.024 batch del flusso vero, pannello 256–511, maschere e copertura | Review e checkout | 8 CPU, 30 GiB, 60 min | 0 | 8 |
| Calibrazione | 512 batch, nessun update dei pesi, prima selezione di 32 batch informativi nel pannello fissato | Audit valido | Come CUDA | 2 | 16 |
| CUDA | Oracolo del gradiente a quattro rank, FP32/bf16, rank senza posa; arresti coordinati | Calibrazione valida | 4 GPU, 32 CPU, 240 GiB, 30 min | 2 | 16 |
| Benchmark completo × 3 | C0, CN, CR, 576 update dalla stessa inizializzazione | Ricevuta unica valida, λ entrambi ammessi e CUDA valido | Come CUDA, ciascuno | 6 | 48 |
| Ripresa × 2 | CN e CR: 544 update, fine processo, nuovo torchrun per altri 32 | Stessa ricevuta, CUDA valido e stesso insieme di ricette | Come CUDA, ciascuno | 4 | 32 |
| Lettura | Confronto delle riprese, rapporti dei gradienti, preventivo misurato | Cinque benchmark completi | 8 CPU, 30 GiB, 60 min | 0 | 8 |
| Checkout × 4 | Preparazione iniziale; release dopo audit; ricette dopo calibrazione; lettura dopo benchmark | Commit pubblicati e verificati | 4 CPU, 5 min ciascuno | 0 | 1,3333 |
| **Totale massimo** | | | | **14** | **129,3333** |

Questi sono tetti prenotati, non una misura di consumo. I sette job GPU hanno ciascuno un nodo con quattro GPU: non sono sette training del FM. Le cinque prove dopo calibrazione possono stare in coda insieme con output distinti, secondo disponibilità. Prima di ogni invio si controllano ledger e ID logici per evitare duplicati. Un job incompleto non viene ripreso implicitamente; la prova di ripresa 544+32 è deliberata e già inclusa nel tetto.

Il numero di checkout è un limite di questa fase: se le evidenze obbligano a cambiare codice o criteri, si ferma la fase e si pubblica una revisione, senza consumare in silenzio altre allocazioni. Tutti i checkpoint del collaudo e i report sono sotto `$WORK/wearusfm_runs/results/contrastive_preflight_20261116_v2/`. Nessuna scrittura nel checkout o nei run storici.

## Comandi e congelamento

I launcher sono `scripts/slurm/contrastive_preflight_cpu.sbatch` e `scripts/slurm/contrastive_preflight_gpu.sbatch`. Richiedono esattamente un argomento: il percorso relativo di un file `job.json` pubblicato nel checkout congelato. Esempio di invocazione del launcher da passare al submitter, **non un invio manuale**:

```bash
scripts/slurm/contrastive_preflight_cpu.sbatch results/infra/contrastive_comparison_20261116/releases/audit.job.json
scripts/slurm/contrastive_preflight_gpu.sbatch results/infra/contrastive_comparison_20261116/releases/calibration.job.json
```

Il dispatcher `scripts/contrastive_job.py` costruisce liste di argomenti senza shell, limita le modalità ad audit, CUDA, calibrazione, benchmark, ripresa e lettura. Richiede stato `APPROVED_FOR_PREFLIGHT`, SHA dei sorgenti, pin della review, pin di ogni input e un output nuovo sotto `$WORK/wearusfm_runs`. Il file del job deve appartenere al checkout. I job di training completo non sono ammessi da questo dispatcher.

Ogni file di invio contiene `mode`, `status`, `source_sha256`, `review={path,sha256}`, `out`; la cella CUDA aggiunge `calibration_checkpoint` con SHA; le celle sui segnali aggiungono `config`, quattro `inputs` (manifest, scales, window_seconds, pose_stats) e le due `roots` congelate. Calibrazione e benchmark aggiungono `release` e `audit`. I benchmark aggiungono `cohort`, `calibration_report`, `calibration_summary`; la ripresa aggiunge `resume_config`. La lettura aggiunge `cohort`, `cuda_report`, `audit`, `calibration_report`, `calibration_summary` e mappe `full`/`resumed` dei summary pinnati. I file ancora non misurati non sono sostituiti da valori fittizi: verranno pubblicati fra le fasi, mantenendo identici i sorgenti e il piano scientifico.

Dopo la calibrazione, GPT copia sul Mac `config.json`, `summary.json` e `metrics.jsonl` verificando i loro SHA e usa `contrastive_prepare.py calibration-report --run-dir ... --out ...`, poi `calibrated-configs --receipt ... --run-dir ... --endpoint 576 --out ...`. Sono letture di metadati, senza training o job sul login node. Pin relativi rendono il ricalcolo identico sul Mac e sul cluster. Ricevuta e coorte vengono pubblicate prima del checkout successivo; un esito negativo ferma la coorte.

Le istruzioni Python effettive, stampate integralmente nel log dal dispatcher, sono:

```text
CPU: python scripts/contrastive_audit.py --config ... --steps 1024 --out ... [input pinnati]
GPU: torchrun --standalone --nproc_per_node=4 scripts/contrastive_cuda_check.py [config e SHA, input e SHA, output]
GPU: torchrun --standalone --nproc_per_node=4 scripts/contrastive_run.py --stage calibration|benchmark --config ... --release ... --time-limit-s 1770 [evidenze e input]
GPU: python scripts/contrastive_split_benchmark.py --split-config ... --resume-config ... --release ... --total-time-limit-s 1800 --out-dir ... -- [evidenze e input]
CPU: python scripts/contrastive_benchmark_readout.py --cohort ... --cuda-report ... --cpu-audit ... --calibration-report ... --calibration-run-dir ... --full C0=... --full CN=... --full CR=... --resumed CN=... --resumed CR=... --out ...
```

`CODE_DIR` e `EXPECTED_SHA` identificano un checkout di campagna immutabile. `lib_code_dir.sh` ne verifica SHA, pulizia e sola lettura. Tutti gli invii passano da `scripts/submit_job.py` e ledger comune, con gli ID logici del piano. L'autorizzazione operativa di Simone del 16/11 vale per questa catena dopo review. I launcher verificano allocazione e numero di GPU. Il walltime SLURM/cgroup è il limite esterno, inclusi avvio dell'ambiente e terminazione; se non basta il job fallisce e si legge il log, senza alzarlo automaticamente.

Il digest comprende `src/wearusfm`, gli script contrastivi, `train_jepa.py` e i launcher contrastivi. Il piano scientifico, i riferimenti di rarità e gradiente, i dati e il lettore sono legati alla ricevuta di calibrazione. **Non si modifica `plan.json` dopo la calibrazione**: gli stati operativi, l'approvazione dei costi e i successivi invii vivono in release esterne pinnate. Un cambio dei sorgenti o dei criteri invalida le evidenze interessate e richiede review.

## Criteri prima dei dati

1. L'audit deve completare 1.024 batch, verificare il pannello esatto 256–511 e il limite dei descrittori aperti. Si applicano le soglie di copertura già firmate, con esenzione del solo requisito per dataset secondo rarità da manifest. Le finestre esentate restano nei denominatori globali.
2. La cella CUDA inizia con un'uscita intenzionale del rank 0: quattro rank devono attestare l'inizializzazione NCCL, torchrun deve fallire con il marcatore previsto e tutti i processi osservati devono essere terminati entro 60 secondi (pulizia esterna limitata). Un fallimento impedisce di avviare il resto della cella. L'oracolo CUDA usa una fixture tecnica dichiarata: batch globale 8 da emg2pose, dropout zero e minimo un negativo; due update. Non è il benchmark della ricetta di produzione (batch 32 e flusso misto), che segue separatamente. L'oracolo verifica il gradiente globale e due update contro la popolazione completa, con lunghezze diverse e rank senza posa. FP32: errore relativo delle loss/gradienti ≤ 1e-5 e dei parametri ≤ 1% dello spostamento del riferimento. Per bf16 il rumore del riferimento bf16 contro FP32 viene misurato sugli stessi batch: è il margine per gradienti, norme e spostamenti dei parametri, con pavimento 1e-5; le loss ordinarie hanno il limite fisso dell'1%. I criteri esatti per campo sono `CRITERIA` nel codice e vengono riportati, non adattati dopo il risultato. Gli arresti per copertura e valori non finiti devono fermare tutti i rank senza update o avanzamento di memoria non registrati.
3. La calibrazione precede l'oracolo CUDA e non applica update: il suo checkpoint fisso fornisce lo stato iniziale esatto all'oracolo, evitando una ricostruzione dipendente dal contesto degli import. Nessun benchmark di training parte prima che l'oracolo sia passato. Si pubblicano anche i primi otto valori delle ancore e delle norme della calibrazione rispetto al riferimento storico pinnato: lettura descrittiva, senza un nuovo cancello numerico, poiché dopo il primo update lo storico cambia i pesi mentre la calibrazione li mantiene fermi. La calibrazione deve restituire entrambi i λ, dal riferimento storico scelto da Simone e dalle norme condivise realmente misurate. Se anche uno esce da `[1e-4, 1]`, si ferma l'intera coorte. Nessun clipping di λ, nessun proseguimento del solo braccio rimasto, nessuna nuova selezione automatica.
4. I cinque benchmark devono terminare al passo previsto, senza STOP. Il pannello denso 513–576 deve contenere 64 misure globali. Per il termine attivo: mediana del rapporto `λ‖g_term‖/‖g_base‖` in `[0,005; 0,5]`, percentile 95 ≤ 1; rapporto delle mediane CN/CR in `[0,25; 4]`. Sono soglie ingegneristiche per controllare l'entità dell'intervento, non un test di efficacia; sotto soglia significa intervento misurato insufficiente, non errore della loss. Il percentile su 64 punti è descrittivo, non una stima precisa della coda.
5. Prima dell'attivazione, stream e origini dei primi 256 update identici; loss totale e norma del gradiente entro `rtol=1e-4, atol=1e-6`. La calibrazione riparte senza riusarne pesi, ottimizzatore, RNG o memoria.
6. La prova a quattro GPU confronta 576 update continui con 544+32. Stato iniziale, ricetta, contatori, origini e RNG devono coincidere; parametri student/teacher entro 1% dello spostamento dall'inizializzazione, buffer esatti, momenti dell'ottimizzatore e feature in memoria entro 1% relativo. Un contatore dell'ottimizzatore rappresentato in float resta un contatore esatto. Non basta che le due loss finali siano simili.

Se un criterio fallisce, si archiviano evidenze e consuntivo e si richiede una nuova review mirata. Non si cambia λ per farlo passare. Tutti i λ e i criteri sono condivisi o simmetrici fra i tre bracci.

## Preventivo dopo misura

Per ciascun braccio si usa il maggiore fra tempo totale/576 e costo medio degli update ordinari più il sovraccosto diagnostico misurato alla frequenza futura. `t_step_s` include già il caricamento dei dati, quindi non si somma di nuovo `t_data_s`. Margine del 50%, riserva di 300 secondi per fase e arrotondamento a cinque minuti, separatamente per 5.000 e ulteriori 14.618 update. Si riportano stima e tetto in GPU-ore/core-ore, per braccio e fase.

La ricevuta del preventivo è ricalcolata dalle evidenze dal gate dei training. La vecchia proiezione 50,83 GPU-ore complessive per tre bracci è solo un riferimento, non un'autorizzazione quantitativa né un nuovo tempo misurato. Le estrazioni e i fit downstream hanno un preventivo separato da pubblicare prima dei loro invii. Il collaudo riuscito produce `PASSED_FOR_REVIEW`, non una release dei training: si pubblicano lettura, costi e convergenza prima di congelare la release successiva.

Rettifica operativa del 17/11: memoria CPU 30 GiB dopo rifiuto dei 32 GiB da parte della QoS. I quattro stadi di checkout usano al massimo cinque job: primo completato in 30 secondi e correzione con tetto 240 secondi, entro il budget del primo stadio. Dettagli e ricevuta nella release del 17/11.
