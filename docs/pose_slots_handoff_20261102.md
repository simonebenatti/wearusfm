# Consegna operativa punto 6 — 02/11/2026

**Preparato, nessun job sottomesso.** Codice scientifico congelato:
`c9d9c02de2597e2ab5e3907d7ffe06ffb544bb62`. [Protocollo](pose_slots_protocol_20261102.md),
[review favorevole](reviews/pose_slots_review_20261102.md), [piano macchina](../results/infra/pose_slots_20261102/job_plan.json).

L'approvazione scientifica è già acquisita. Sottomissioni tramite semplicione e submitter/ledger comuni;
la chat non è raggiungibile. La deroga precedente copriva il temporale concluso: per l'esecuzione diretta
serve l'estensione esplicita alla presente catena. Questi comandi rendono la consegna verificabile,
non autorizzano a saltare il canale concordato o i cancelli fra fasi.

Prima del deploy controllare ledger (ID sotto), assenza checkout/output di campagna, stato e configurazione
del remoto, ref corrente e hook. Pubblicare soltanto il nuovo ref `deploy/pose-slots-20261102` allo SHA
congelato; verificare con `ls-remote`. Nessuna modifica al working tree canonico. Il checkout è creato
dal job CPU versionato. I launcher successivi richiedono il checkout già completato: `afterok` non crea
un file mancante quando `sbatch` tenta di leggerlo.

I comandi sotto usano i **massimi** autorizzati. Dopo il preflight annotare tempo, memoria e spazio:
8,19GiB payload, più output preflight, ricevute e margine. Dopo il benchmark annotare un preventivo CPU.
Si possono ridurre walltime/memoria entro i tetti; non modificare viste o regole in base ai risultati val.
Tetto catena2,25GPU-ore /28,333core-ore. Nessuna nuova richiesta scientifica per passare i cancelli.

B/R3c vanno sottomessi insieme dopo preflight e controllo risorse. Il benchmark può partire quando B
è completo mentre R3c termina. Il report si sottomette solo dopo entrambe le estrazioni e lettura del
benchmark. Comandi separati: **non eseguire in blocco**. Se si usa una dipendenza ancora attiva,
passarla con `--dopo ID_LOGICO`; non aggiungere dipendenze a job conclusi e non rilanciare ID già prenotati.

## checkout

Prerequisiti: origin and deploy ref SHA verified; no existing ledger IDs or campaign checkout.

```bash
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python '/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py' --id POSE-SLOTS-checkout-20261102 --agente gpt --campagna pose-slots-20261102 --gpu-ore 0 --script-stdin '/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/slurm/campaign_checkout.sbatch' -- --cpus-per-task=2 --mem=2G --time=00:10:00 --export=ALL,CAMPAIGN=pose-slots-20261102,CAMPAIGN_SHA=c9d9c02de2597e2ab5e3907d7ffe06ffb544bb62,EXPECTED_SHA=c9d9c02de2597e2ab5e3907d7ffe06ffb544bb62
```

## preflight

Prerequisiti: checkout COMPLETED 0:0 and readable frozen code.

```bash
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python '/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py' --id POSE-SLOTS-preflight-20261102 --agente gpt --campagna pose-slots-20261102 --gpu-ore 0.25 -- --partition=boost_usr_prod --cpus-per-task=8 --mem=60G --time=00:15:00 --qos=normal --gres=gpu:1 --export=ALL,CODE_DIR=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pose-slots-20261102,EXPECTED_SHA=c9d9c02de2597e2ab5e3907d7ffe06ffb544bb62 /leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pose-slots-20261102/scripts/slurm/pose_slots.sbatch preflight
```

## extract-B

Prerequisiti: preflight COMPLETED 0:0 and receipts passed; time/memory/space reviewed within cap.

```bash
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python '/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py' --id POSE-SLOTS-extract-B-20261102 --agente gpt --campagna pose-slots-20261102 --gpu-ore 1 -- --partition=boost_usr_prod --cpus-per-task=8 --mem=60G --time=01:00:00 --qos=normal --gres=gpu:1 --export=ALL,CODE_DIR=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pose-slots-20261102,EXPECTED_SHA=c9d9c02de2597e2ab5e3907d7ffe06ffb544bb62 /leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pose-slots-20261102/scripts/slurm/pose_slots.sbatch extract B
```

## extract-R3c

Prerequisiti: preflight COMPLETED 0:0 and receipts passed; time/memory/space reviewed within cap.

```bash
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python '/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py' --id POSE-SLOTS-extract-R3c-20261102 --agente gpt --campagna pose-slots-20261102 --gpu-ore 1 -- --partition=boost_usr_prod --cpus-per-task=8 --mem=60G --time=01:00:00 --qos=normal --gres=gpu:1 --export=ALL,CODE_DIR=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pose-slots-20261102,EXPECTED_SHA=c9d9c02de2597e2ab5e3907d7ffe06ffb544bb62 /leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pose-slots-20261102/scripts/slurm/pose_slots.sbatch extract R3c
```

## benchmark

Prerequisiti: B extraction COMPLETED 0:0 and parent receipt complete.

```bash
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python '/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py' --id POSE-SLOTS-benchmark-20261102 --agente gpt --campagna pose-slots-20261102 --gpu-ore 0 -- --partition=lrd_all_serial --cpus-per-task=8 --mem=30G --time=00:15:00 --export=ALL,CODE_DIR=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pose-slots-20261102,EXPECTED_SHA=c9d9c02de2597e2ab5e3907d7ffe06ffb544bb62 /leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pose-slots-20261102/scripts/slurm/pose_slots.sbatch benchmark
```

## run

Prerequisiti: both extractions COMPLETED 0:0 and receipts complete; benchmark examined and resource decision recorded.

```bash
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python '/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py' --id POSE-SLOTS-run-20261102 --agente gpt --campagna pose-slots-20261102 --gpu-ore 0 -- --partition=lrd_all_serial --cpus-per-task=8 --mem=30G --time=01:00:00 --export=ALL,CODE_DIR=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pose-slots-20261102,EXPECTED_SHA=c9d9c02de2597e2ab5e3907d7ffe06ffb544bb62 /leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pose-slots-20261102/scripts/slurm/pose_slots.sbatch run
```

A fine fase: registrare ID, stato Slurm/ExitCode, tempi e CPU/GPU realmente allocate; ricopiare ricevute,
log e report nel repository, verificare SHA. Le cache e i coefficienti dei lettori restano fuori da Git.
Prima dell'interpretazione confrontare M384 con il precedente istantaneo. Nessun cambio FM o loss automatico.
