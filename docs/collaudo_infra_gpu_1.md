# Primo collaudo GPU dell'infrastruttura della finestra 1 — protocollo (Claude, 09/10/2026; da approvare: NESSUN job autorizzato da questo file)

**Perche'.** Prima prova su Leonardo del motore di training preparato sul branch di Claude: checkout di campagna, DDP su 4 GPU, ripresa esatta,
dataloader con le regole della finestra 1, memoria dei rung firmati. **E' un collaudo dell'infrastruttura, non un risultato scientifico.**
Budget: «preparazione dell'infrastruttura della finestra 1», 25 GPU-ora (decisione del 09/10/2026), consumo finora 0.

**Aggiornamento del 12/10/2026:** il contratto delle loss distribuite e' collegato al trainer (GPT `5030a19`, Claude: vedi il registro), quindi
il collaudo DDP usa le loss vere (numeratori locali, conteggi globali), non piu' la media locale per processo; l'opzione
`--ddp-local-mean-collaudo` non esiste piu'. Margini del walltime corretti sulla review di GPT (avvio e checkpoint).

**Cosa NON prova** (da non leggere nei risultati): l'equivalenza numerica delle loss fra 1 e 4 GPU con NCCL e bf16 (provata sulla CPU in
float64 con `gloo`; la conferma su GPU richiede di misurare prima il rumore del bf16 fra due run identici, protocollo a parte); il ritmo DDP
dei rung 100M e 300M (qui solo la loro memoria su una GPU);
piu' nodi e FSDP; la testa keep dello Step 1; la qualita' dell'apprendimento.

## Codice

Tutti i job usano un **checkout di campagna** (`scripts/slurm/campaign_checkout.sbatch`) allo SHA del commit che pubblica questo protocollo,
con `CODE_DIR` ed `EXPECTED_SHA` passati a ogni launcher. Due strade, la sceglie Simone:
- (a) **dopo l'integrazione** del branch di Claude su `main` da parte di GPT (review dei commit 7ed8b59, 4fcc1df, 92e137e, 56bc7e6, 772b991
  e di questo): push di un riferimento `deploy/collaudo-infra-1` a quello SHA;
- (b) **subito**, push di `deploy/collaudo-infra-1` allo SHA del branch di Claude, senza integrazione: non tocca il canonico ne' una campagna
  in corso; il codice non ha ancora la review di GPT, accettabile per un collaudo dell'infrastruttura senza esiti scientifici.
Prima del push: preflight richiesto da GPT (nessun automatismo sul ref che aggiorni checkout attivi; checkout detached nuovo dallo SHA completo).
Sottomissioni con `scripts/submit_job.py` (ID logici sotto, `--dopo` per le dipendenze), via leonardo-ops.

## Job, risorse e costo massimo (1 GPU-ora = 8 ore locali)

| ID logico | Cosa | Risorse | Durata max | Costo max |
|---|---|---|---|---|
| `infra1_checkout` | `campaign_checkout.sbatch` su stdin, `CAMPAIGN=collaudo-infra-1` | `lrd_all_serial`, 1 CPU, 2 GB | 10 min | 0 GPU-ora, 0,17 core-ora |
| `infra1_loader` | `measure_loader.sbatch` con `--window1-rules --window-seconds <59425316> --scales-from <59623700> --workers 8 --batches 100 --batch-size 8` (8 CPU = la quota di una GPU, 8 finestre = il batch per GPU di DDP) | `lrd_all_serial`, 8 CPU, 24 GB | 1 h | 0 GPU-ora, 8 core-ora |
| `infra1_ddp` | `train_ddp.sbatch`, preset `sanity` (30M), `EXTRA_ARGS="--window1-rules --window-seconds <59425316> --datasets all"`, 300 passi poi `RESUME_TO=330` nella stessa allocazione, con `TIME_LIMIT_S=1200` e `RESUME_TIME_LIMIT_S=240` (24 min di training su 30: 6 min per avvii, letture iniziali e checkpoint); manifest v1.1, scale 59623700; cartella nuova su `$SCRATCH` | `boost_qos_dbg`, 1 nodo, 4 GPU, 32 CPU, 480 GB | 30 min | **2 GPU-ora (16 ore locali)** |
| `infra1_memory` | `probe_memory.sbatch`: rung 30M, 100M, 300M x 4 e 8 finestre per GPU, Hyser a 4 s, 2 aggiornamenti | `boost_qos_dbg`, 1 GPU, 8 CPU, 100 GB | 30 min | **0,5 GPU-ora (4 ore locali)** |

**Totale massimo: 2,5 GPU-ora (20 ore locali) + 8,2 core-ora CPU.** Budget infrastruttura: 25 -> almeno 22,5 GPU-ora residue. I tre job dopo il
checkout partono con `afterok` da `infra1_checkout`; su `boost_qos_dbg` stanno al piu' 2 job per utente (qui 2).

## Criteri, fissati prima del lancio

**`infra1_checkout`** — superato se: uscita 0 e messaggio «checkout di campagna pronto»; HEAD = SHA atteso; nessun file del codice scrivibile
(lo verifica il job stesso). Altrimenti nessun altro job parte (`afterok`).

**`infra1_loader`** — descrittivo, nessuna soglia: finestre/s per processo e secondi per fase (sessione e finestra, scala, lettura e filtro,
maschera, codici e vicini, target delle ancore) con la configurazione del training. Serve a decidere dove ottimizzare il dataloader.

**`infra1_ddp`** — superato se tutti veri:
1. uscita 0 delle due fasi; nessuno STOP, nessun valore non finito, nessun OOM;
2. `summary.json`: `world_size` 4, `batch_per_process` 8, `accum_steps` 1, `loss_reduction` = contratto, codice = SHA atteso; nessun arresto
   per conteggi diversi dai precalcolati, chiavi diverse fra processi o JEPA vuota;
3. la seconda fase riprende dal passo 300 e arriva a 330 senza errori di stream (indice del batch = passo a ogni passo, verificato dal trainer)
   ne' di configurazione;
4. `metrics.jsonl` con un solo record per passo da 1 a 330, `windows` = 32 per passo.
Si riportano, senza soglia nuova: secondi per passo; **attesa sui dati dopo i primi 50 passi = somma di `t_data_s` / somma di `t_step_s`,
confrontata con la regola gia' firmata di D6a (<= 2%)**; memoria GPU massima fra i processi; classi e montaggi virtuali per passo.

**`infra1_memory`** — per ogni (rung, finestre per GPU): OOM si' o no, memoria massima allocata e riservata, secondi per passo. Regola fissata
ora: un rung **sta** con b finestre per GPU se `max_reserved_gib + ddp_buckets_gib <= 60` (4 GiB di margine sui 64 di una A100). Se il
300M non sta con 8 finestre per GPU, la finestra 1 a 300M richiede l'accumulo (2 micro-batch da 4): si riporta e decide Simone.

## Dopo

Esiti, costi effettivi (da `sacct`) e consuntivo del budget in `docs/decisioni.md` e nel registro condiviso; log e JSON copiati sul Mac
fuori dal repo, il riepilogo in `results/`. Nessun rilancio automatico: un job fallito si guarda e si ripropone.

## Comandi esatti (fissati prima del lancio, 12/10/2026)

Autorizzazione di Simone del 12/10/2026 (in `docs/decisioni.md`). Strada (b): codice = il commit che contiene questa sezione (`<SHA>`, 40
caratteri, registrato in `decisioni.md` al lancio), branch di Claude non ancora rivisto da GPT. Dal Mac, nella radice del worktree di Claude.

    M=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/passo6/capgmyo_grid_fix/manifest_v1_1.json.gz   # manifest v1.1 (1.704.849 byte)
    S=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/passo6/loader_59623700/session_scales.json       # scale 59623700 (2.234.011 byte)
    W=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/passo6/window_seconds_59425316.json             # durate 59425316 (2.156.652 byte)
    C=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/collaudo-infra-1
    RUN=/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/runs/infra1_ddp_1210                 # cartella nuova
    PROBE=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/infra1/probe_memory_1210.json

    git push leonardo <SHA>:refs/heads/deploy/collaudo-infra-1
    python3 scripts/submit_job.py --id infra1_checkout --agente claude --campagna collaudo-infra-1 --gpu-ore 0 \
      --script-stdin scripts/slurm/campaign_checkout.sbatch -- --export=ALL,CAMPAIGN=collaudo-infra-1,CAMPAIGN_SHA=<SHA>
    python3 scripts/submit_job.py --id infra1_loader --agente claude --campagna collaudo-infra-1 --gpu-ore 0 --dopo infra1_checkout -- \
      --cpus-per-task=8 --time=01:00:00 --export=ALL,CODE_DIR=$C,EXPECTED_SHA=<SHA> \
      $C/scripts/slurm/measure_loader.sbatch $M $S --window1-rules --window-seconds $W
    python3 scripts/submit_job.py --id infra1_ddp --agente claude --campagna collaudo-infra-1 --gpu-ore 2 --dopo infra1_checkout -- \
      "--export=ALL,CODE_DIR=$C,EXPECTED_SHA=<SHA>,RESUME_TO=330,TIME_LIMIT_S=1200,RESUME_TIME_LIMIT_S=240,EXTRA_ARGS=--window1-rules --window-seconds $W --datasets all" \
      $C/scripts/slurm/train_ddp.sbatch $M $S $RUN 300
    python3 scripts/submit_job.py --id infra1_memory --agente claude --campagna collaudo-infra-1 --gpu-ore 0.5 --dopo infra1_checkout -- \
      --export=ALL,CODE_DIR=$C,EXPECTED_SHA=<SHA> $C/scripts/slurm/probe_memory.sbatch $M $S $W $PROBE

Preflight in sola lettura (12/10/2026, leonardo-ops): coda vuota; nel canonico solo `main` e il tag `manifest-v1`, nessun ref `deploy/`, nessun
worktree, `$WORK/wearusfm_campaigns` assente; hook solo `.sample`; config locale senza `hooksPath`, `receive.denyCurrentBranch =
updateInstead` (agisce solo sul branch estratto, `main`: il ref di deploy non tocca il working tree); input presenti con le date attese.
Non verificati dal login node: sha256 degli input e quota Lustre (comandi non ammessi a leonardo-ops); il checkout aggiunge ~6 MB.
Se una dipendenza `afterok` viene rifiutata da SLURM perche' il checkout e' gia' finito (successo del 07/10), lo si verifica dal log e si
risottomette quel job senza dipendenza con `--dopo-rifiuto`: adeguamento operativo, stessi costi e comandi.
