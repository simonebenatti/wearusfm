# P5 preventivo - rapporto incrementale

Avvio: 2026-11-19 10:07, HEAD 438c8d65, diff e388a1de..0efdc29c

## a) Ricalcolo preventivo (preflight_plan.json @438c8d65)
- 9 celle: audit 8CPUx7200s=16 coreh/0 GPUh; calib, cuda, 3 full, 2 split: ciascuna 4GPUx1800s=2 GPUh, 32CPUx1800s=16 coreh; readout 8CPUx3600s=8 coreh.
- checkout 4 x 4CPU x 240s = 1,066667 coreh.
- Totali: 13 allocazioni, 14 GPUh, 137,066667 coreh: coincidono con tabella dell'emendamento (da confermare con script).

## b) primi riscontri
- contrastive_job.py:17 ALLOCATED_SECONDS audit 7200, readout 3600, GPU 1800; :67 confronto esatto per modo.
- contrastive_preflight_cpu.sbatch:8 --time=01:00:00; :20-22 SECS da scontrol TimeLimit.
- contrastive_audit.py: nessun limite di tempo interno; rapporto scritto solo a fine replay (riga 187). Da valutare margine.

## a) confermato con script (Fraction)
- 9 celle + 4 checkout = 13; GPUh 14 esatte; coreh 2056/15 = 137,0666...; ogni cella gpu_hours/core_hours = gpus/cpus x seconds/3600; caps coincidono.
- source_sha256 del preflight (643a8937...) = source_digest() calcolato alla HEAD 438c8d65 (diagnostici fuori dai glob del digest).

## Storico utile (results/infra/contrastive_comparison_20261116)
- audit 64652559: 3787 s wall, TotalCPU 2138 s (35:38), MaxRSS 21,5 GiB, TIMEOUT con batch COMPLETED; OverTimeLimit 3 min, KillWait 180 s (accounting_and_integrity.txt).
- GPU storiche: calibrazione 774 s; cuda bf16 141 s; bench 661/683/769 s; split 775/759 s; checkout 36-61 s con 4 CPU e 2G (uno 8G).
- lrd_all_serial (registro 27/10): MaxTRESPerJob cpu=8, mem=30800M; MaxTRESPU cpu=8; MaxJobsPU 2; MaxSubmitPU 10; MaxWall 4 h.

## c) stima costo target 100/125
- anchor_targets calcolato per OGNI finestra nel loader (pretraining_loader.py:641-643), anche nell'audit che non usa i target.
- RMS con somme cumulative (costo indipendente dalla finestra); ms_fast materializza (C,P,n) e FFT di n=round(0,125 fs) invece di 0,05 fs.
- Misura sul Mac (sintetico, mix dell'audit storico, 2,5 s): 373 s -> 448 s di CPU target su 32.768 finestre (+74 s, x1,20); DB10 a 1926 Hz x2,05 (n=241 primo).
- Anche con Leonardo 2-3 volte piu' lento del Mac: +150-220 s su 3787 s storici -> ~4.000 s attesi, margine ~3.200 s sotto 7.200.
- L'audit NON ha limite interno: contrastive_job.py:104 passa solo --steps 1024; contrastive_audit.py scrive il rapporto solo alla fine (187). Se supera, nessun rapporto.

## b) verifiche eseguite
- pytest test_contrastive_job.py: 29 passed; -k fixed_resources/swapped_or_extended: 17 passed (audit-cpu-7200 OK; audit 3600/7199/7201/gpu, readout 1800/7200/gpu, cuda 7200, calibration 3600, full-cpu, split 7200 rifiutati).
- Mutazioni in clone temporaneo: audit->3600: 3 failed; readout->7200: 3 failed; '!=' -> '<': 6 failed. Guardia coperta.
- Parsing TimeLimit del launcher simulato: 02:00:00->7200, 01:00:00->3600, 00:30:00->1800, 2:00:00->7200, UNLIMITED->0. Identico nei due launcher.
- main(): build_command (controllo secondi) prima di validate_allocation e di ogni lettura dati -> audit senza override fallisce in pochi secondi.
- test_contrastive_audit.py + test_contrastive_source_compat.py: 38 passed.

## d) limiti
- CPU launcher 8 CPU/30G/lrd_all_serial; validate_allocation 8 e 30*1024=30720 MiB <= 30800; 2 h <= 4 h.
- GPU launcher 4 GPU/32 CPU/240G/boost_qos_dbg/00:30:00; validate_allocation 32, 245760 MiB, 4 GPU.
- boost_qos_dbg MaxJobsPU 2 / MaxSubmitPU 4 (sacctmgr in cuda_completed_bf16_20261118/remote_evidence.txt): 5 benchmark eleggibili insieme -> invio a ondate.
- RILIEVO: checkouts.memory_gib=30 nel preflight e "4 CPU,30 GiB" in tabella; launcher campaign_checkout.sbatch --mem=2G; tutti i checkout storici con --mem=2G (sacct billing=4,cpu=4,mem=2G), diagnostica 'requested_gib': 2.

## Altre note
- Nessun controllo meccanico dello stato SLURM dell'audit (validate_audit controlla contenuto/config/sorgente, non TIMEOUT/elapsed): eccezione non ereditata solo per procedura.
- 'Totale massimo' = somma dei limiti richiesti; OverTimeLimit 3 min + KillWait 180 s consentono sforamenti (storico 3787 s vs 3600).
- ID targets-v3 assenti dal ledger comune (grep 0).

## CLI end-to-end (clone temporaneo, job minimi, poi rimosso)
- audit cpu 3600 -> "ValueError: wrong mode, release, source or allocation" in 1,07 s; audit 7200 -> supera il controllo (KeyError 'review' del job minimo).
- readout 7200 -> rifiutato; readout 3600 -> supera il controllo.
- submit_job.py passa gli argomenti sbatch (vieta solo -J/--job-name/--comment): --time=02:00:00 ammesso.

## Stima durata audit nuovo
- CPU attesa 2.249-2.360 s (storico 2.138 + 111-222 s), wall attesa ~3.900-4.000 s con lo stesso I/O; servirebbe ~2,9x l'attesa I/O storica (1.649 s) per superare 7.200 s.
- Limite superiore: tutta la CPU x2,05 e I/O storico = 6.032 s < 7.200 s. RSS storico 21,5 GiB, transitori ms_fast +<0,1 GiB.
- Contenuto dell'audit indipendente dai target (identita', qc, visibilita', posa); calibration.config nuovo differisce dallo storico solo per 3 campi.

## Esito P5 (bozza finale)
- Nessun BLOCCANTE.
- MINORE: memoria dei checkout 30 GiB dichiarata vs 2 GiB del launcher e di tutta la storia.
- NOTE: audit senza scadenza interna e senza gate meccanico TIMEOUT; 'massimo' senza tolleranza SLURM; vincoli di coda (dbg MaxSubmitPU 4, corsia seriale per utente 2 h, manutenzione 23-25/11, semantica del rifiuto per override mancante); residuo budget e cumulato linea 3 non dichiarati (<=25,284444 GPUh / 236,704444 coreh con diagnostica e vecchia catena).
- Script di misura: tmp/p5_scripts/time_targets.py, tmp/p5_scripts/mix_targets.py.
