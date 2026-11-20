# 3-launcher · checkout e audit CPU della coorte RMS100/fast125

Simone ha firmato la proposta scientifica `faa4a8d7` e il budget di preflight il 20 novembre. Claude ha approvato il pacchetto a `469b312cd5502b801a0af44c56e16a89f019c14c`. A 3-launcher spetta il primo invio operativo della coorte, un checkout seguito dall’audit su 1.024 batch. Gli ID logici sono `CONTRAST-CHECKOUT-audit-targets-v3-20261118` e `CONTRAST-AUDIT-targets-v3-20261118`, entrambi della campagna `CONTRASTIVE-targets-v3-20261118`. [Firma in decisioni](decisioni.md) · [Review di Claude con blocco di release](../results/infra/contrastive_targets_20261118/releases/audit_20261120/claude_review_contrastive_targets_runtime_20261120.md)

## SHA e comandi congelati

**SHA di esecuzione del checkout e dell’audit: `9abde2951cd9b2a25aa699faceb93132de6732f8`.** Il commit di consegna successivo contiene questa nota e la copia del piano dei comandi. Verifica che la nota e il piano appartengano alla consegna pubblicata da GPT. [Argomenti esatti e impronte](../results/infra/contrastive_targets_20261118/audit_handoff_20261120/submission_plan.json)

Dal worktree Mac indicato si possono rigenerare e controllare gli argomenti in locale con i seguenti comandi. Il confronto deve terminare con exit code zero.

```bash
cd '/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-contrastive-targets'
(
  set -eu
  tmp_audit_plan=$(mktemp)
  trap 'rm -f -- "$tmp_audit_plan"' EXIT
  python3 results/infra/contrastive_targets_20261118/audit_handoff_20261120/prepare_handoff.py --frozen-sha 9abde2951cd9b2a25aa699faceb93132de6732f8 --dry-run > "$tmp_audit_plan"
  cmp results/infra/contrastive_targets_20261118/audit_handoff_20261120/submission_plan.json "$tmp_audit_plan"
)
```

Solo 3-launcher esegue i comandi seguenti, uno alla volta e con i controlli operativi descritti sotto. Dopo aver verificato saldo, quota, reservation, ledger e destinazione del deploy, pubblica il ref dedicato. L’output della seconda riga deve contenere lo SHA di esecuzione integrale e il ref del piano.

```bash
git -C '/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-contrastive-targets' push leonardo 9abde2951cd9b2a25aa699faceb93132de6732f8:refs/heads/deploy/contrastive-targets-audit-20261119
git -C '/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-contrastive-targets' ls-remote --exit-code leonardo refs/heads/deploy/contrastive-targets-audit-20261119
```

**Prima del checkout**, esegui sul Mac il controllo qui sotto. La verifica sul login usa i file del ref appena pubblicato. Conserva il suo stdout e controlla a mano ID, braccio del job, config e release. Solo dopo esito positivo e verifica dei dati operativi passa al comando di invio.

```bash
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python '/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-contrastive-targets/results/infra/contrastive_targets_20261118/audit_handoff_20261120/prepare_handoff.py' --frozen-sha 9abde2951cd9b2a25aa699faceb93132de6732f8 --check-checkout-login
```

Comando di invio del solo checkout.

```bash
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python '/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py' --id CONTRAST-CHECKOUT-audit-targets-v3-20261118 --agente gpt --campagna CONTRASTIVE-targets-v3-20261118 --gpu-ore 0 --script-stdin '/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-contrastive-targets/scripts/slurm/campaign_checkout.sbatch' -- --cpus-per-task=4 --mem=2G --time=00:04:00 --export=ALL,CAMPAIGN=contrastive-targets-audit-20261119,CAMPAIGN_SHA=9abde2951cd9b2a25aa699faceb93132de6732f8,EXPECTED_SHA=9abde2951cd9b2a25aa699faceb93132de6732f8
```

**Dopo il completamento del checkout**, aggiorna i controlli operativi. La verifica qui sotto legge ID e SHA dal ledger comune, controlla sacct, HEAD e permessi e rigenera e verifica il comando dal checkout. Conserva e verifica il risultato prima di inviare l’audit.

```bash
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python '/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-contrastive-targets/results/infra/contrastive_targets_20261118/audit_handoff_20261120/prepare_handoff.py' --frozen-sha 9abde2951cd9b2a25aa699faceb93132de6732f8 --check-audit-login
```

Comando di invio del solo audit dopo il controllo del checkout terminato.

```bash
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python '/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py' --id CONTRAST-AUDIT-targets-v3-20261118 --agente gpt --campagna CONTRASTIVE-targets-v3-20261118 --gpu-ore 0 -- --cpus-per-task=8 --mem=30G --time=02:00:00 --export=ALL,CODE_DIR=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/contrastive-targets-audit-20261119,EXPECTED_SHA=9abde2951cd9b2a25aa699faceb93132de6732f8 /leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/contrastive-targets-audit-20261119/scripts/slurm/contrastive_preflight_cpu.sbatch results/infra/contrastive_targets_20261118/releases/audit_20261120/audit.job.json
```

## Sequenza dell’operatore

1. Dal worktree Mac `wearusfm-contrastive-targets`, verifica il commit di consegna su `origin`. Rigenera il piano dei comandi al **commit di esecuzione** indicato sotto e confrontalo byte per byte con quello pubblicato. Usa il submitter e il ledger comuni. Registra prima di ogni invio data e ora Europe/Rome, saldo del progetto `IscrB_WearUsFM` da `saldo -b`, impegni residui del progetto, quota effettiva su WORK, scritture previste dalle altre linee, disponibilità della corsia, reservation correnti, ID e stati nel ledger. Il piano richiede 12 GiB liberi per la coorte oltre alle scritture già previste. Calcola il saldo dopo la fase usando il billing osservato ed evitando di conteggiare due volte risorse già impegnate. Se le risorse o la finestra sono insufficienti, registra il motivo e rinvia la fase.
2. Verifica destinazione nuova e input presenti. Pubblica il ref di deploy sul repository di Leonardo e controllane lo SHA. Esegui il controllo preliminare sul login node dal commit indicato, registra l’output di `check-binding`, di `build_command` e della validazione della ricetta nel runner. Per l’audit il config richiesto è `review_configs/calibration.config.json`, `contrastive.arm=C0` e `calibration_only=true`; la cella CPU ha `arm=null` e riceve il binding del pacchetto. Controlla a mano questa corrispondenza e i quattro hash degli input. Il comando prodotto deve indicare `contrastive_audit.py`, 1.024 passi e la nuova destinazione della coorte.
3. Invia il checkout con 4 CPU, 2 GiB e 4 minuti. Appena termina confronta ID del ledger, `sacct COMPLETED 0:0`, HEAD completo, pulizia e permessi del checkout. Ripeti sul login node i controlli sul checkout effettivo e acquisisci la fotografia operativa aggiornata. Invia quindi l’audit con 8 CPU, 30 GiB e 2 ore. Questo ordine permette di validare il checkout già creato. Una dipendenza ancora in coda può usare `--dopo` nelle fasi che la prevedono. Per un predecessore concluso valgono sacct e SHA verificati. Ogni ID ammette un solo tentativo. Un errore di controllo o di invio richiede una registrazione e la lettura del problema.
4. Raccogli il rapporto CPU, i log e le ricevute del submitter. Verifica SHA e byte di `$WORK/wearusfm_runs/results/contrastive_targets_20261118/audit.json`, 1.024 passi, esito di copertura globale/pannello, sorgenti, config e input. Per l’audit conserva l’allocazione principale identificata dal vero ID numerico del ledger con `sacct -X -P -j ID --format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocCPUS,TimelimitRaw` (intestazione e una riga di allocazione). Registra anche tempi, Req/AllocTRES e billing per entrambi i job. Rendi verificabile il collegamento ID logico ↔ ID SLURM ↔ rapporto ↔ TSV. Ad audit terminato conserva rapporto e accounting anche in caso di fallimento. Per un rapporto positivo crea il JSON `contrastive-audit-accounting-v1` con `logical_id`, `job_id` numerico in forma di stringa, `source_sha256`, `audit_sha256` e `sacct` (percorso, SHA-256, byte). Fai verificare la ricevuta al comando congelato `scripts/contrastive_release.py check-audit --audit … --accounting …` usando i percorsi effettivi. Riporta alla **chat 3 scientifica** commit delle evidenze, ID, risultati, SHA e risorse distinte per richiesto, effettivo e margine.

## Contabilità e calendario da ricalcolare prima di ogni invio

| Ambito | GPU-ore richieste | Core-ore richieste | Margine GPU / core | Inviluppo GPU / core |
|---|---:|---:|---:|---:|
| Checkout iniziale e audit | 0 | 16,266667 | 0 / 1,2 | 0 / 17,466667 |
| Intero preflight, 17 allocazioni | 18 | 169,6 | 3,6 / 32,8 | 21,6 / 202,4 |

Il consumo storico già misurato è 6,086667 GPU-ore e 57,827778 core-ore. Con billing storico proporzionale alle CPU le ore locali richieste per l’intero nuovo preflight sono 169,6. Il saldo corrente e il consumo della nuova coorte si registrano dalle evidenze di Leonardo. Il margine contabile è 6 minuti per allocazione (OverTimeLimit e KillWait di 180 s ciascuno). Il limite scientifico dell’audit resta 7.200 s effettivi.

Calendario teorico a ritroso dalla manutenzione **lunedì 23/11 alle 08:00 CET**, riapertura prevista **mercoledì 25/11 alle 08:00 CET**. Si assumono due slot debug liberi per la linea, due ondate di full e due di reload, senza code né attese fra release. Ogni ora è l’ultimo inizio del checkout della fase per completare l’intera catena prima della chiusura.

| Fase | Inizio al più tardi, soli walltime | Inizio al più tardi, margine incluso |
|---|---:|---:|
| Audit | 23/11 01:36 | 23/11 00:12 |
| Calibrazione | 23/11 03:40 | 23/11 02:28 |
| CUDA | 23/11 04:14 | 23/11 03:14 |
| Tre benchmark full | 23/11 04:48 | 23/11 04:00 |
| Quattro reload | 23/11 05:52 | 23/11 05:22 |
| Lettura | 23/11 06:56 | 23/11 06:44 |

Con due slot disponibili la catena occupa 6 h 24 min in sequenza o 7 h 48 min con margine. Con un solo slot il riferimento con margine diventa **22/11 ore 22:24**. Sottrarre da questi limiti le attese di coda previste, le altre prenotazioni e il tempo per i cinque rilasci intermedi. Per il solo checkout più audit, rinviando i discendenti, l’ultimo inizio teorico con margine è **23/11 ore 05:44**. Se una fase rischia di attraversare la chiusura, riprogrammare il suo primo invio dopo la riapertura verificata. Conservare le ricevute delle fasi concluse.

## Condizioni della consegna successiva

Per rilasciare la calibrazione servono un audit positivo e il suo accounting autenticato (`COMPLETED`, `0:0`, `ElapsedRaw ≤ 7200`, 8 CPU, 120 min). Prima di quella release GPT scientifico deve anche verificare il percorso reale di `contrastive_run.main` su quattro processi CPU/Gloo per full C0 e reload (P-5). L’esito del solo audit non fa partire automaticamente alcun job GPU.

Le future consegne GPU devono includere **`TORCH_DIST_INIT_BARRIER=1` nella riga effettiva `sbatch --export`** (P-1) e la verifica manuale di job, config e release per braccio/ripetizione (P-2). La calibrazione misura tempo e memoria bf16 math sul batch di 32 (P-8). Un superamento dell’1% per tensore, incluso il caso di due reload concordi e continuo discorde, blocca la promozione e richiede la localizzazione descritta in decisioni.md (P-7). I passaggi seguenti richiedono ciascuno la propria release e un checkout con i metadati misurati.
