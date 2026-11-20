# verify-r4-risorse-R4-N5-1 — scettico n. 1 su R4-N5 (limiti per utente delle code)

Inizio: 2026-11-20. Copia di sola lettura a 0efa4d45 (fbfde797..0efa4d45 non tocca i file citati: git diff --quiet ok).

## Riproduzione (comandi miei)
- `sed -n 18,19p results/infra/contrastive_comparison_20261116/cuda_completed_bf16_20261118/remote_evidence.txt` ->
  `Name|MaxJobsPU|MaxSubmitPU` / `boost_qos_dbg|2|4` (lettura del 18/11 17:42). Confermato.
- `scripts/slurm/contrastive_preflight_gpu.sbatch:5-6` (fbfde797): `-p boost_usr_prod`, `--qos=boost_qos_dbg`: i limiti dbg si applicano a tutte le 9 celle GPU.
- `scripts/slurm/contrastive_preflight_cpu.sbatch:5-7`: `lrd_all_serial`, 8 CPU, 30G.
- decisioni.md:7850 (fbfde797): rifiuto audit `QOSMaxMemoryPerUser`, 32 GiB contro 30.800 MiB. Confermato.
- `submission_20261117/qos_memory_diagnosis.json`: `max_tres_per_user: cpu=8,mem=30800M,node=1` e uguale per job;
  ricevuta `audit_rejected_state.json`: stderr `sbatch: error: QOSMaxMemoryPerUser`. Limite di memoria per utente supportato.
  (Il registro del 27/10, claude_GPT_shared:11877-11878, trascriveva MaxTRESPU solo cpu=8: lettura piu' vecchia/incompleta;
  in ogni caso cpu=8 per utente basta da solo a rendere esclusivi audit e lettura da 8 CPU.)
- common_state_diagnostics_20261118/resources.json, campo `scheduling`: "at most 2 running and 4 submitted for this user". Confermato.
- Nuovi piani (plan.json, preflight_plan.json a fbfde797): nessuna occorrenza di qos/dbg/sched/submit/serial. Prontuario
  docs/contrastive_targets_execution_20261119.md:92-93 dice solo "Dipendenze e limiti della QoS possono allungarla". Confermato.
- Stesso utente per tutte le linee: docs/contrastive_handoff_20261117.md:112 "Checkout può attendere quota CPU linea 1";
  claude_GPT_shared:10909-10910 (limiti seriali condivisi fra i lavori dei due agenti). Confermato.

## Ricerca di motivi per confutare
- Prontuario:93 riconosce genericamente i limiti QoS; prontuario:178-179 rinvia i comandi definitivi (SHA, afterok) alla consegna
  post-review; preflight `required_before_first_submission` chiede ledger/quota/saldo aggiornati da 3-launcher.
- Prassi gia' dimostrata dallo stesso operatore nella linea 3: handoff 17/11:72 ("La quinta è stata inviata dopo la conclusione di C0,
  rispettando MaxJobsPU 2 / MaxSubmitPU 4"), :91-92, :117 ("a gruppi secondo piano e quota di tutti i job dell'utente ... da ricontrollare").
  Ma quei documenti sono ora marcati "Storico 25/50" e il nuovo prontuario "prevale" (prontuario:9).
- submit_job.py:9-12 modella il rifiuto QoS come `rifiutato` senza job, risottomissione solo con `--dopo-rifiuto`
  ("per esempio dopo aver corretto la QoS"): un'eccedenza di MaxSubmitPU non crea job ne' duplicati.
- Seriale: ogni job da solo rientra nei limiti per job, quindi l'eccedenza per utente da' PENDING, non rifiuto ne' fallimento.
- Nessun codice di release/dispatcher gestisce la QoS (grep su contrastive_job/run/release/targets_plan): nessun controllo altrove,
  ma neppure un percorso che diventi scorretto.
- CLAUDE.md:219 documenta solo "boost_qos_dbg (30 min, max 2 job)"; nulla su MaxSubmitPU 4 o limiti per utente della seriale.
- Confronto con la base 438c8d65: 5 celle GPU eleggibili insieme dopo CUDA (> MaxSubmitPU 4); a fbfde797 3 full poi 4 reload
  dopo un solo checkout: struttura piu' compatibile, omissione preesistente (la review 94f5094, par. 5, controllava solo il limite per job).
- Sfumatura che rafforza il rilievo: i reload dipendono solo da full_CN/full_CR; se lo scheduler lascia C0 per ultimo, C0 occupa
  ancora uno slot quando i 4 reload diventano inviabili, anche senza job di altre linee.

## Esito
Non confutabile: fatti e citazioni riprodotti; la lacuna documentale esiste. Nessun effetto su correttezza, cancelli, tetti o
attribuzione (rifiuto pulito o PENDING, operatore gia' esperto): gravita' NOTA confermata, non MINORE.
