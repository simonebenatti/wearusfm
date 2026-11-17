# Recupero dell'audit e rilascio della calibrazione — 17/11/2026

**APPROVED_FOR_PREPARATION, stage calibration.** Simone autorizza esplicitamente:
«fai la rettifica operativa per recuperare il rapporto dell'audit e lancia la calibrazione GPU,
senza rifare l'audit. Hai già la mia autorizzazione».

## Rettifica puntuale

Si accetta come prerequisito di copertura **solo** il rapporto del job **64652559**, SHA256
`5c720fadbcb12fd5458bc0e19cd03f28fec30ad5973ca72739f01bf1b3d7547b`, 20.275.571 byte.
Il job resta registrato **TIMEOUT**, 3.787 secondi contro 3.600 richiesti, `.batch` COMPLETED 0:0.
Non si correggono retroattivamente né stato SLURM né tetto o consuntivo. Nessun nuovo audit.

La deroga è giustificata dal rapporto completo: 1.024 batch consecutivi, pin di sorgenti, ricetta
e input corretti, riepilogo emesso dopo la scrittura/chiusura e i controlli finali di integrità;
ricalcolo locale e indipendente concordi. Il pannello 256–511 ha 8.179 / 8.192 query InfoNCE
ammissibili e RnC attivo in 256 / 256 batch. L'esenzione CapgMyo è quella già decisa prima del job.
Evidenze e limiti: `contrastive_audit_results_20261117.md`, archivio in
`results/infra/contrastive_comparison_20261116/audit_completed_20261117/`.

La deroga sostituisce per questo unico rapporto il requisito di job principale COMPLETED 0:0.
Non vale per altri TIMEOUT, file incompleti, pin diversi o gate di copertura falliti. Ogni fase
successiva conserva gli arresti già prescritti; nessun retry cieco o modifica automatica di λ.
Non è una nuova review attribuita a Claude. GPT assume il rilascio sotto il via esplicito di Simone.

## Consuntivo e risorse

Audit: 8,415555556 core-ore allocate (tetto originario 8). Checkout 64652044 e 64652383:
0,045555556 core-ore. Già consumate **8,461111111 core-ore**, zero GPU.

I tre checkout residui passano da 4 CPU × 300 s a **4 CPU × 240 s**, 2 GiB ciascuno.
I due checkout osservati hanno richiesto 30 e 11 secondi; se il nuovo tetto non basta, stop e diagnosi.
Sette celle GPU invariate: 4 GPU × 32 CPU × 1.800 s ciascuna; lettura CPU invariata 8 CPU × 3.600 s.
Massimo totale comprensivo del consuntivo reale: **129,261111111 core-ore**, entro il tetto
**129,333333333 core-ore**; tetto GPU invariato **14 GPU-ore**. Nessun consumo si nasconde nel margine.

## Unico lancio GPU di questa consegna

- ID logico `CONTRAST-CALIB-v2-20261116`, campagna `CONTRASTIVE-v2-20261116`.
- 512 batch, 4 rank, 6 worker per rank, **nessun aggiornamento dei pesi** (`calibration_only=true`).
- 4 A100, 32 CPU, 240 GiB, `boost_usr_prod` / `boost_qos_dbg`, 30 minuti; tempo interno 1.770 s.
- Tetto della calibrazione **2 GPU-ore / 16 core-ore**, più checkout **0,266666667 core-ore**.
- Output nuovo `$WORK/wearusfm_runs/results/contrastive_preflight_20261116_v2/calibration`.
- Sorgenti invariati: `ab2d6460bfe090e4adba8cfd480b726077005c4ed6ae10db7ebbd5909ddf808b`.
- Configurazione invariata: `5e4aaca8e9671020efb0a3cdbb801357fe1082ce0052427f79d2f55cf9f98b13`.
- Piano scientifico `plan.json`, dati, loss, criteri e riferimento del gradiente invariati.

Release e job: `releases/calibration.release.json` e `releases/calibration.job.json`.
Rettifica leggibile dalla macchina: `releases/audit_recovery_20261117.json`.
Tutti i file vengono pubblicati prima della sottomissione. Nuovo checkout congelato, submitter
e ledger comuni. Prima del lancio verificare checkout concluso, SHA, pulizia, pin e output assente.
Non usare una dipendenza afterok verso il job audit TIMEOUT: il prerequisito è questo recupero
verificato. Se il checkout concluso è già scaduto dal controller, verificarlo in sacct e fisicamente
e non creare un afterok inesistente. Nessun checkout già in uso viene modificato.

## Lettura successiva

Richiedere conclusione regolare, 512 batch completi, zero update, stato iniziale/finale invariato,
ricevuta e copertura valide. Ricalcolare entrambi i λ dai primi 32 batch congiuntamente informativi
nel pannello congelato. Se uno esce da [1e-4, 1], fermare l'intera coorte senza clamp o riselezione.
Solo una calibrazione valida consente la successiva release delle prove CUDA e dei benchmark.
Questa consegna non rilascia i training scientifici. Nessun monitor o automazione in background.
