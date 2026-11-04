# T100-LQ — solo collaudo preparatorio, esecuzione 04/11/2026

Review Claude `b0fac8f`: APPROVATO il collaudo a `d48cad5`, non il training.
Il commit che contiene questo piano aggiunge soltanto documenti e stato delle
risorse; src/scripts/tests e piano/config scientifici restano quelli revisionati.
Prima del checkout sono scritte le precisazioni S1/S3 nel protocollo; nessuna
esenzione di kernel dai trigger e nessuna modifica automatica di epsilon.

Autorizzazione: Simone ha già assegnato implementazione/lanci, autonomia sotto
5GPUh e sequenza review/convergenza→deploy per questa linea. Non si richiede
un secondo via allo stesso collaudo approvato. Il richiamo a semplicione nella
review descrive la precedente procedura: la chat storica non è raggiungibile
e non compare nell'inventario attuale. Il responsabile della linea usa quindi
la sottomissione diretta già autorizzata, con submitter e ledger comuni e senza
duplicare invii. Il coordinamento del04/11 assegna un submitter per linea.

## Due soli job e destinazioni

| ID logico | Allocazione | Costo massimo |
| --- | --- | --- |
| LOGSQ-checkout-20261104 | lrd_all_serial,2CPU,4G,10min | 0GPUh;0,333coreh |
| LOGSQ-preflight-20261104 | boost_usr_prod,boost_qos_dbg,1nodo,4GPU,32CPU,240G,30min | 2GPUh;16coreh allocate |

Campagna Git: `logsquare-preflight-20261104`.
Deploy ref nuova: `refs/heads/deploy/logsquare-preflight-20261104`.
Checkout: `/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/logsquare-preflight-20261104`.
Output: `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/logsquare_20261104`.
Piano: `results/infra/logsquare_review_20261104/preflight_plan.json`.
Launcher: `scripts/slurm/logsquare_preflight.sbatch`.

Il codice esecutivo è lo SHA completo del commit pubblicato che contiene
questo documento, da registrare nella ricevuta di lancio. Nessun riferimento
mobile nei job. Push su nuova ref deploy, mai main; checkout detached nuovo,
bloccato e in sola lettura. L'ambiente condiviso resta invariato.

Controlli preliminari effettuati: remote HEAD su main, receive.denyCurrentBranch
updateInstead, core.hooksPath non impostato, soltanto hook sample nel percorso
predefinito; la nuova ref deploy e i due percorsi non esistono. Nessuna voce LOGSQ
nel ledger al controllo. Una lettura squeue ha mostrato il job estraneo63525232:
non viene toccato. Queste osservazioni vanno ricontrollate prima dell'invio.

## Procedura

1. Commit/push origin, confronto SHA remoto; verifica diff scientifico nullo
   rispetto a d48cad5. Pubblicazione dello stesso SHA sulla nuova ref Leonardo.
2. Solo `scripts/submit_job.py`, ledger standard `~/wearusfm_local/jobs/`.
   Checkout tramite `--script-stdin scripts/slurm/campaign_checkout.sbatch`,
   override2CPU/4G/10min, export CAMPAIGN e CAMPAIGN_SHA.
3. Attendere checkout COMPLETED0:0 e verificare ricevuta, SHA, pulizia e sola
   lettura. Non far partire il collaudo da un checkout parziale.
4. Stesso submitter, ID LOGSQ-preflight, export CODE_DIR/EXPECTED_SHA,
   launcher nel checkout, piano relativo e output assoluto nuovo.
   Dopo verifica esplicita del checkout completato non serve dipendenza da
   un ID già rimosso dalla memoria SLURM. Non usare sbatch diretto.
5. Risposta ambigua: riconciliare l'ID già prenotato, mai duplicarlo.
   Fallimento tecnico/scientifico: conservare log e interrompere, senza retry.
6. Raccogliere report, rank JSON/JSONL, diagnostica iniziale, hash e accounting
   dei job parent. Verificare gate e leggere R1/S1; rendicontare costi reali.

Nessun job di training, estrazione, contrastiva o nuova automazione fa parte
di questo piano. Training resta non approvato anche con collaudo superato.
