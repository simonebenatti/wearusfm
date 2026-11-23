# Sequenza operativa delle fasi 2 (D3) e 3 (B) — da eseguire dopo la manutenzione di Leonardo

Operatore: Claude, dal Mac (incarico di Simone del 22/11 e «niente gpt per ora, vai avanti tu» del 23/11; fase 3 approvata il
23/11 con «Sì, approvo»). Regole: protocollo `docs/tcn_temporal_diag_protocol_20261122.md`, sezioni 8 e 9; voci del 23/11 in
`docs/decisioni.md`. Worktree `/Users/simonebenatti/dev/wearusfm/.claude/worktrees/claude-tcn-diag`, ramo
`claude/tcn-temporal-diag-20261122`. `D` = SHA completo del commit da eseguire, quello che chiude le review delle due fasi:
**`D` = `487850361deaa6720eb54e2a7c81c25f6274ff05`** (23/11; 62 test dei diagnostici passati).

Submitter: `/Users/simonebenatti/.venvs/wearusfm-tok/bin/python '/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py'`
(sha256 `814da4bc…`), lanciato dalla radice del worktree, a `D` o a un suo discendente che non tocca `scripts/` (dispatch
sha256 `c082cd46…`, `git diff --quiet D HEAD -- scripts`). Log: `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/<ID>_%j.out`.
`CODE_DIR=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pose_temponet_diag_20261123`, `EXPECTED_SHA=D` per tutti i job.
Ogni invocazione: `--agente claude --campagna pose_temponet_diag_20261123`.

## 0. Prima dell'invio

- Manutenzione finita (dal 23/11 alle 08:00 al 25/11 alle 08:00) e certificato ssh rinnovato da Simone (`ssh-keygen -L`).
- `ssh leonardo squeue -u $USER`; nel ledger `~/wearusfm_local/jobs/ledger.tsv` nessun ID `TCND-*-20261123`.
- Sul commit `D`: test CPU dei diagnostici verdi; i 247 file del manifest congelato uguali a `a1ae9607`
  (`P.source_identity()` = `9aaf1c8b…`); albero pulito.
- `git push origin claude/tcn-temporal-diag-20261122`; `git push leonardo ${D}:refs/heads/deploy/pose-temponet-diag-20261123`;
  `git ls-remote leonardo refs/heads/deploy/pose-temponet-diag-20261123` = `D`.
- Voce nel registro condiviso (ramo `claude/temponet-db8-20261109`, nelle due sezioni) con lo SHA completo di `D`, gli ID, i
  costi e le basi dell'autorizzazione; commit e push prima del primo `sbatch`.
- Quota di `$WORK` (`lfs quota -hp <id del progetto> $WORK`) prima di D3 e di nuovo prima della fase 3, che scrive circa 0,5 GB;
  residuo dell'account riletto come il 22/11.
- Nessun job durante il rinfresco di `$SCRATCH` del 03/12: gli input stanno su `$SCRATCH` e un `touch` durante la lettura fa
  fallire la verifica dei pin.

## 1. Checkout e fase 2 (D3), 4,98 GPU-ore prenotate

| ID | `--gpu-ore` | Coda e risorse | `--time` | `--dopo` |
| --- | --- | --- | --- | --- |
| `TCND-checkout2-20261123` | 0 | `lrd_all_serial`, 2 CPU, 24G, script `scripts/slurm/campaign_checkout.sbatch`, `CAMPAIGN=pose_temponet_diag_20261123,CAMPAIGN_SHA=D` | 00:12:00 | — |
| `TCND-d3t-wd-20261123` | 1.9 | `boost_usr_prod`, `--gres=gpu:1 --qos=normal`, 8 CPU, 32G, `TCND_STAGE=d3t` | 01:51:00 | checkout2 |
| `TCND-d3l-wd-20261123` | 1.8833333333333333 | come sopra, `TCND_STAGE=d3l` | 01:50:00 | checkout2 |
| `TCND-d3s-wd-20261123` | 1.2 | come sopra, `TCND_STAGE=d3s` | 01:09:00 | checkout2 |

I tre job GPU usano `--script-stdin scripts/slurm/tcn_temporal_diag_dispatch.sbatch` e
`--export=ALL,CODE_DIR=<CODE_DIR>,EXPECTED_SHA=D,TCND_STAGE=<stadio>,TCND_JOB_ID=<ID>`, con `--account=IscrB_WearUsFM --nodes=1
--ntasks=1 --no-requeue` e i due percorsi di log col proprio ID, come le righe di D0-D2 nel ledger.

Controllo ogni 30 minuti (`squeue -h -j`, poi `sacct -X -j`). Nessun rilancio dentro la fase.

## 2. Raccolta e regola

- `rsync` dei soli `*.json` di `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/pose_temponet_diag_20261122/` in
  `results/infra/tcn_temporal_diag_20261122/remote/`; log in `logs/` rinominati `.log`; `sacct` con elenco delle cartelle. Gli
  stati `.pt` restano su `$WORK`.
- `summarize.py remote` applica `core.choose_dose(core.d3_results(...))`, la stessa regola del driver.
- Dose scelta senza conferma, o nessuna dose ammissibile: la linea torna a Simone con la curva dose-risposta. Fine.

## 3. Fase 3 (B), solo se la regola propone la conferma; 9,30 GPU-ore prenotate, un solo tentativo

Condizioni, tutte prima del primo invio: le tre ricevute di D3 `completed`; `summarize.py` dà «conferma su un mondo nuovo: True»;
Simone ha confermato il costo (fatto il 23/11: «Sì, confermo» a «Confermi per la fase B 9,30 GPU-ore prenotate (74,4 ore
locali)?»; protocollo, sezione 9). Ogni stadio, cancello compreso,
rilegge comunque la regola da sé prima di qualunque setup. Ogni job dipendente si invia subito dopo i suoi predecessori, mentre
sono ancora in coda o in esecuzione: un `afterok` su un job già uscito dalla memoria del controller può essere rifiutato.

| ID | `--gpu-ore` | Coda e risorse | `--time` | `--dopo` |
| --- | --- | --- | --- | --- |
| `TCND-b-calib-0-20261123`, `TCND-b-calib-1-20261123` | 0 | `lrd_all_serial`, 8 CPU, 24G, `TCND_STAGE=bc0` / `bc1` | 00:40:00 | — |
| `TCND-b-controls-{linear,temporal}-{L0,MLP}-20261123` | 1.05 | `boost_usr_prod`, 1 GPU, 8 CPU, 32G, `TCND_STAGE=b-<mondo>-<testa>` | 01:00:00 | le due calibrazioni |
| `TCND-b-controls-{linear,temporal}-TCN_post-20261123` | 2.55 | come sopra | 02:30:00 | le due calibrazioni |
| `TCND-b-gate-20261123` | 0 | `lrd_all_serial`, 2 CPU, 16G, `TCND_STAGE=bg` | 00:15:00 | le sei ricerche |

Stesso script di dispatch e stesso `CODE_DIR` della fase 2. Ogni stadio verifica da sé la regola di D3 e le ricevute completate
dei predecessori. Raccolta come al punto 2, poi esito a Simone: `passed`, `recipe_not_admitted` o
`optimization_unresolved`. Anche con `passed` la variante non entra nella catena congelata senza una ricetta congelata nuova,
review e decisione di Simone.
