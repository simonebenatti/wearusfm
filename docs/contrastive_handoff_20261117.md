# Confronto contrastivo — consegna operativa del 17/11/2026

## Autorità e perimetro

Simone ha affidato a GPT il rilascio e il lancio di questo giro: «se tu sei a posto per questo giro lanci tu, ti do fiducia». Superata l'attesa della nuova review di Claude per il collaudo preparatorio; non attribuire a Claude una review mai ricevuta. Il via include letture, raccolta, commit/push e invii della catena già descritta, senza nuova domanda operativa. Restano tutti i gate scientifici e tecnici; i training completi non sono ancora rilasciati.

Lavorare in `/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-logcontrast`, ramo `codex/tokenizer-logcontrast-20261104`. Non creare monitor o automazioni in background.

## Stato attuale

**Aggiornamento finale:** audit **64652559 TIMEOUT**, 63:07 su 60 min; `.batch` COMPLETED 0:0.
Rapporto completo e ricalcolato: 1.024 batch, `passed=true`, pannello 256–511 valido; pin tutti corretti.
SHA rapporto `5c720fadbcb12fd5458bc0e19cd03f28fec30ad5973ca72739f01bf1b3d7547b`.
Gate di contenuto superato, gate operativo no. **Calibrazione non sottomessa.**
Leggere `docs/reviews/contrastive_audit_results_20261117.md` prima di qualsiasi seguito.
La proposta di recupero del rapporto mediante rettifica operativa è documentata ma **non applicata**.
Consuntivo audit più checkout: 8,461111 core-ore, zero GPU. Nessun retry o monitor.

Cronologia dei lanci:

- Checkout originario **64652044**, SHA `821f4d7`: COMPLETED 0:0, 30 s, 4 CPU.
- Audit con richiesta 32 GiB: rifiutato da sbatch, nessun job creato. Massimo QoS seriale 30.800 MiB.
- Corretti soltanto launcher e guardia CPU a 30 GiB; 18 test mirati passati. Codice scientifico invariato.
- Checkout corretto **64652383**, SHA `f09bf857eee24f02aff25fa743853d60fcab7f03`: COMPLETED 0:0, 11 s, 4 CPU.
- Secondo invio audit: rifiutato senza job perché il checkout completato era già sparito dal controller (`MinJobAge=30 sec`). Verificato completamento in sacct e integrità fisica del checkout; rimossa solo la dipendenza ormai soddisfatta.
- Audit effettivamente sottomesso: **64652559**, ID logico `CONTRAST-AUDIT-v2-20261116`, 8 CPU, 30 GiB, 60 minuti, zero GPU. Stato finale **TIMEOUT**, rapporto completo; non duplicarlo.

Checkout immutabile:
`/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/contrastive-v2-20261117-audit-mem30`

Digest esecutivo **ab2d6460bfe090e4adba8cfd480b726077005c4ed6ae10db7ebbd5909ddf808b**. La vecchia verifica completa `verification_v2.json` attesta il digest precedente; la correzione strettamente operativa e i 18 test sono in `submission_20261117/memory_fix_verification.json`. Non mischiare i due SHA.

Log audit:
`/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/contrastive_cpu_64652559.out`

Output atteso:
`/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_preflight_20261116_v2/audit.json`

Ricevute del submitter, rifiuti, diagnosi QoS/dipendenza e consuntivi dei checkout sono in `results/infra/contrastive_comparison_20261116/submission_20261117/`. Il ledger comune rimane l'autorità contro gli invii duplicati. I due checkout hanno consumato insieme **0,045556 core-ore**; nessuna GPU. Tetti globali invariati, 14 GPU-ore e 129,3333 core-ore.

## Seguito: fermo al gate operativo, nessun nuovo invio in questa consegna

Non eseguire automaticamente il punto 2: il punto 1 ha fallito sullo stato SLURM, pur avendo prodotto
evidenze di copertura valide. Prima serve chiudere la rettifica operativa proposta nella lettura finale.
La procedura seguente conserva i requisiti originali e non rappresenta una deroga al timeout.

1. Leggere l'esito dell'audit. Richiedere COMPLETED 0:0, file completo, `passed=true`, pannello 256–511 valido, pin di dati/config/sorgenti e geometria 4 rank × 6 worker coerenti. Raccogliere report, hash, log e risorse nel repo. Se fallisce fermare la fase e diagnosticare, senza retry cieco.
2. Solo dopo il gate: preparare release di calibrazione `APPROVED_FOR_PREPARATION` con SHA dell'audit, del config e dei quattro input, 4 rank, 6 worker/rank, `time_limit_s=1770`. Job wrapper `APPROVED_FOR_PREFLIGHT`, modalità `calibration`, allocazione 4 GPU, 32 CPU, 240 GiB, 30 min. Pin della presente autorità nel file di release; nessun obbligo di far approvare nuovamente a Simone la stessa fase.
3. Pubblicare release e piano dei comandi, verificare SHA remoto, checkout congelato nuovo. Nessuna modifica ai sorgenti e al piano scientifico. Se il checkout è già COMPLETED ed è scaduto dal controller, verificare sacct e integrità e non usare un afterok inesistente; i gate del launcher restano obbligatori.
4. Calibrazione completata: copiare metadati, ricalcolare ricevuta e coorte con `contrastive_prepare.py`, pubblicarle. Se uno dei λ è fuori range fermare tutta la coorte. Il checkpoint fisso di calibrazione serve al test CUDA: conservarlo.
5. CUDA/NCCL con prova di uscita di un rank, poi tre benchmark e due riprese, secondo piano. Tutti gli invii tramite `scripts/submit_job.py` e ledger comune. Il budget dei training deve venire dal rapporto misurato, non dai tempi storici.

Il piano scientifico `plan.json` è ancora documentariamente REVIEW_REQUIRED per ricordare che Claude non ha riletto le correzioni: la sua identità resta congelata. L'approvazione operativa successiva è esplicita in `preflight_plan.json`, `releases/audit.job.json` e `docs/reviews/contrastive_preflight_release_20261117.md`. Non riscrivere il piano scientifico dopo la calibrazione: le ricevute lo pinnano.
