# Confronto contrastivo — consegna operativa del 17/11/2026

## Autorità e perimetro

Simone ha affidato a GPT il rilascio e il lancio di questo giro: «se tu sei a posto per questo giro lanci tu, ti do fiducia». Superata l'attesa della nuova review di Claude per il collaudo preparatorio; non attribuire a Claude una review mai ricevuta. Il via include letture, raccolta, commit/push e invii della catena già descritta, senza nuova domanda operativa. Restano tutti i gate scientifici e tecnici; i training completi non sono ancora rilasciati.

Lavorare in `/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-logcontrast`, ramo `codex/tokenizer-logcontrast-20261104`. Non creare monitor o automazioni in background.

## Stato attuale

**18/11, lettura CPU finale completata con esito negativo controllato.**
Checkout **64775502** COMPLETED 0:0; lettore **64775571** FAILED 2:0 dopo
aver scritto il rapporto completo, come previsto per un gate negativo. Stato:
`FAILED_REQUIRES_REVIEW`, `training_approved=false`; rapporto SHA256
`deb6b8178362e23512f105d95b3859c28945d9ffa5356a653f1c8cad67c929fc`.
Passano i rapporti delle componenti attive (CN 0,2261; CR 0,0834; rapporto
CN/CR 2,7106), ma falliscono la parità di `grad_norm` nei primi 256 update a
peso contrastivo zero e i limiti numerici dei confronti continuo/ripreso.
Nessun training lungo è rilasciato, nessun retry e nessuna soglia è cambiata.
Il confronto di ripresa usa due run indipendenti che divergono già prima del
restart: dimostra stati finali diversi, non isola la ripresa come causa.
Rapporto, log e lettura in `benchmarks_completed_bf16_20261118/` e
`docs/reviews/contrastive_readout_results_20261118.md`.

**Consegna automatica a 3-launcher non riuscita:** il tool ha rifiutato l'ID individuato
perché la chat è di tipo ChatGPT, non una chat Codex raggiungibile; anche l'accesso UI
all'app è vietato dalla piattaforma. Nessun messaggio consegnato e nessun nuovo job inviato.
Pacchetto pubblicato e verificato, comandi a 2a57d4a7, esecuzione congelata 0c3c5293.
Prompt pronto: `docs/prompt_3_launcher_contrastive_readout_20261118.md`.
Chiesto a Simone solo l'ID/link della chat Codex corretta, non un nuovo consenso.
Gli ID di checkout/readout restano da sottomettere esclusivamente da 3-launcher.

**Pacchetto CPU pronto per 3-launcher:** checkout congelato
`0c3c52936e8b7e9166f351ff8003b40dfe40427e`; comandi concreti e pin in
`results/infra/contrastive_comparison_20261116/releases/readout_bf16_20261118/submission_plan.json`.
ID da non duplicare: `CONTRAST-CHECKOUT-readout-20261118`, `CONTRAST-READOUT-bf16-20261118`.
GPT scientifico non ha sottomesso questi due job. Il lancio e la raccolta operativa
sono affidati alla chat 3-launcher; nessuna nuova autorizzazione da chiedere a Simone.

**18/11, ore 18:24: tutti i cinque benchmark COMPLETED 0:0.**
C0 64771457, CN 64771478, CR 64771595, ripresa CN 64771859, ripresa CR 64772690.
Nuovo operatore dei lanci: **3-launcher**, per istruzione diretta di Simone in questa chat.
Nessun nuovo invio eseguito da GPT scientifico dopo tale istruzione.
Lettura CPU pronta in `releases/readout_bf16_20261118/`, consegna:
`docs/reviews/contrastive_readout_release_20261118.md`. Non duplicare i cinque benchmark.
La parità delle norme nei primi 256 update è già negativa nei metadati; il readout CPU
completa i controlli sui checkpoint e sulle riprese. Nessun training lungo rilasciato.
Gli snapshot seguenti sono cronologia.

**18/11, tutti e cinque i benchmark sottomessi.** Snapshot 18:01 Europe/Rome:

| Cella | Job | Stato |
| --- | --- | --- |
| C0, base, 576 update | 64771457 | COMPLETED 0:0, 661 s |
| CN, InfoNCE, 576 update | 64771478 | COMPLETED 0:0, 683 s |
| CR, RnC, 576 update | 64771595 | RUNNING |
| CN, ripresa 544+32 | 64771859 | RUNNING |
| CR, ripresa 544+32 | 64772690 | PENDING QOSMaxJobsPerUserLimit |

Gate CUDA pubblicato prima degli invii in a17b9258; ricevute e snapshot in
`results/infra/contrastive_comparison_20261116/bf16_benchmark_submission_20261118/`.
Tutti i job usano il checkout congelato 8905e346 e source 46938f31…9320.
**Non risottomettere nessuna cella.** La quinta è stata inviata dopo la conclusione di C0,
rispettando MaxJobsPU 2 / MaxSubmitPU 4. Nessun monitor in background.

Dopo la conclusione raccogliere i tre benchmark e le due riprese, senza retry automatici;
applicare il lettore congelato e i criteri già pubblicati. Il solo COMPLETED non certifica
rapporti dei gradienti, parità iniziale o ripresa: la lettura completa resta da fare.
Rapporti sotto `$WORK/wearusfm_runs/results/contrastive_preflight_20261118_bf16/`,
cartelle `benchmark/{C0,CN,CR}` e `resume/{CN,CR}`. Per la lettura CPU finale conservare
il pin dell'audit originale 5c720fad…547b, checkpoint della calibrazione 0756921a…d50a,
coorte bf16, CUDA 979f4429…771f5 e probe d0a75115…dfd78. Il lettore autentica anche
optimizer, coda, stream e parità dei primi 256 update; nessun training lungo rilasciato.

Consuntivo conclusi (inclusi C0/CN e tentativi storici): **2,725556 GPU-ore /
30,556667 core-ore**. Con le tre celle ancora da terminare al tetto completo, checkout
finale e lettura CPU: massimo **8,725556 GPU-ore / 86,823333 core-ore**.
Gli snapshot seguenti conservano la cronologia.

**18/11, primi quattro benchmark sottomessi dopo il gate pubblicato a17b9258:**
C0 **64771457**, CN **64771478**, CR **64771595**, ripresa CN **64771859**.
Alle 17:52 C0/CN RUNNING, CR/ripresa CN PENDING QOSMaxJobsPerUserLimit.
Resta da inviare soltanto `CONTRAST-RESUME-CR-bf16-20261118` quando ci sono meno di
quattro job boost_qos_dbg attivi/sottomessi sull'utente. Nessun altro consenso necessario.
Non duplicare gli ID sopra; ricevute in `bf16_benchmark_submission_20261118/`.
Codice di tutti i job congelato a 8905e346, source 46938f31…9320, tetto 30 minuti,
4 GPU / 32 CPU / 240 GiB per cella. Nessun training lungo o monitor in background.
Gli snapshot seguenti restano cronologia.

**18/11, CUDA 64746651 SUPERATO**, COMPLETED 0:0, 141 s; checkout 64746634 COMPLETED 0:0, 53 s.
320/320 confronti, STOP e probe validi; lettori congelati e review indipendente concordi.
Rapporto e hash in `cuda_completed_bf16_20261118/`, lettura in
`docs/reviews/contrastive_cuda_bf16_results_20261118.md`.
Gate positivo pubblicato prima dei cinque invii, che seguono il piano congelato 8905e34.
Le ricevute degli invii aggiorneranno questo stato. Non duplicare job. Nessun training lungo.
Gli snapshot seguenti sono cronologia.

**18/11, nuovo CUDA sottomesso: 64746651**, `CONTRAST-CUDA-bf16-20261118`,
con afterok sul checkout **64746634**, `CONTRAST-CHECKOUT-bf16-20261118`.
Release congelata **8905e346b03e5d173cccc124aa40032bf588a191**, source
**46938f319181d408ee852cfaacfdc92cc5e44eaa1cfa770825ad96956eec9320**.
Piano comandi 1f8fe7c, `releases/bf16_20261118/submission_plan.json`; ricevute in
`bf16_submission_20261118/`. Non duplicare nessuno dei due job. Checkout può attendere
quota CPU linea 1; CUDA partirà da solo dopo successo e ricontrolli SHA/pulizia/readonly.
Output: `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_preflight_20261118_bf16/cuda.json`.
Log: `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/contrastive_gpu_64746651.out`.
**Cinque benchmark NON sottomessi**. Dopo CUDA COMPLETED 0:0 raccogliere rapporto e probe,
verificare con lettori congelati, pubblicare esito positivo e inviare le cinque celle a gruppi
secondo piano e quota di tutti i job dell'utente (MaxJobsPU=2, MaxSubmitPU=4 da ricontrollare).
Nessun consenso ulteriore per questa catena; nessun monitor/automazione, nessun training lungo.
I vecchi CUDA 64706211 e 64730941 restano falliti; gli snapshot seguenti sono cronologia.

**18/11, nuova review bf16:** 64730941 è concluso FAILED 1:0, rapporto completo;
correzione mmap riuscita, otto scarti diagnostici bf16 fuori regola. Claude be21629 acquisito
in fast-forward e review GPT favorevole. Ulteriore commit 7352e2f conserva come rapporti
negativi gli scarti finiti optimizer/coda >1%, senza cambiare soglia. Sorgenti finali
46938f319181d408ee852cfaacfdc92cc5e44eaa1cfa770825ad96956eec9320; coorte/release bf16 rigenerate.
Seguire docs/reviews/gpt_contrastive_bf16_review_20261118.md e releases/bf16_20261118/.
I due vecchi tentativi CUDA restano agli atti; nessun benchmark sottomesso. Nuovo CUDA da
sottomettere dopo pubblicazione del pacchetto, poi rapporto positivo prima dei cinque invii
in gruppi secondo quota debug effettiva. Autorizzazione di Simone già presente. Nessun monitor.
Gli snapshot sotto sono cronologia, non istruzioni per duplicare o riusare vecchi job.

**18/11, CUDA ACCODATO: 64730941**, `CONTRAST-CUDA-mmap-20261118`, afterok:64730738.
Checkout 64730738 in attesa quota CPU; CUDA partirà soltanto dopo il suo completamento riuscito.
Codice congelato **3b5d4be2bda36bdc601ae3c50ae9b8ace696f16e**, runtime source **6b421b75…754aa**.
Rettifica operativa pubblicata in 7e96014: file sbatch storico immutabile byte-identico,
CODE_DIR e EXPECTED_SHA nuovi; controlli SHA/pulizia/readonly all'avvio. Non duplicare nessuno dei due job.
Ricevute in mmap_submission_20261118, comando in queued_cuda_submission_plan.json.
**Cinque benchmark NON sottomessi**: dopo CUDA COMPLETED 0:0 raccogliere/validare rapporto e probe,
archiviare e pubblicare esito positivo; quindi usare i cinque comandi del submission_plan.json.
La loro autorizzazione persiste. Nessun monitor/automazione creato; nessun training scientifico autorizzato.
Output CUDA: `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_preflight_20261118_mmap/cuda.json`.
Log: `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/contrastive_gpu_64730941.out`.
Le voci seguenti sono snapshot precedenti.

**18/11, checkout sottomesso:** 64730738, `CONTRAST-CHECKOUT-mmap-20261118`,
4 CPU, 2 GiB, 4 minuti, inizialmente PENDING per QOSMaxCpuPerUserLimit.
Congelato a **3b5d4be2bda36bdc601ae3c50ae9b8ace696f16e**, pubblicato su origin e
`deploy/contrastive-v2-20261118-mmap` su Leonardo, entrambi verificati.
Comandi esatti in `releases/mmap_20261118/submission_plan.json` (commit afe7cff).
Non duplicare il checkout; verificare completamento e poi CUDA secondo piano.
Nessuno dei sei job GPU è ancora sottomesso in questo snapshot.

**18/11, correzione pronta per rilancio autorizzato:** sei conversioni Path/string applicate;
150 test distinti passati. Manifest rigoroso di compatibilità in `compatibility_20261118/`,
digest corrente `6b421b7525a5af45d0371382c104c38bc7bb76ca1cba0acf81ca050f6fe754aa`.
Audit e calibrazione restano invariati e attribuiti al produttore storico ab2d6460…;
nuova coorte byte-identica e release concrete in `releases/mmap_20261118/`.
Simone autorizza nuovo CUDA e poi cinque benchmark se superato, senza nuovi consensi.
Nuovi checkout/ID/output, nessun riuso di cuda.json del tentativo fallito, nessun audit rifatto.
Riferimento: `docs/reviews/contrastive_mmap_release_20261118.md`. Invii da registrare con ricevute;
le voci sotto restano cronologia, non sono l'istruzione di fermare la rettifica autorizzata.

**Aggiornamento CUDA concluso:** 64706211 FAILED 1:0 dopo 67 s, prima dei confronti
numerici: `torch.load(Path, mmap=True)` non è compatibile con il PyTorch del cluster,
che richiede una stringa. Probe di uscita dei quattro rank superato, ma `cuda.json`
assente e gate CUDA NON superato. **Nessuno dei cinque benchmark sottomesso.**
Calibrazione e λ sotto restano validi. Checkout 64705989 COMPLETED 0:0 in 67 s,
SHA 93f24c962e86551a232bdec90c1598f3d6b53173. Nessun retry, sorgenti ancora ab2d6460….
Diagnosi, sei conversioni proposte e impatto sui pin in
`docs/reviews/contrastive_cuda_failure_20261117.md`; evidenze in `cuda_completed_20261117/`.
Non riutilizzare gli ID/output del tentativo e non inviare i benchmark prima della
rettifica tecnica e di un collaudo positivo. La cronologia seguente conserva gli stati precedenti.

**Aggiornamento calibrazione conclusa:** 64701482 COMPLETED 0:0, 774 s, 0,86 GPU-ore / 6,88 core-ore.
512 batch, zero update, hash iniziale/student/teacher identici. Ricevuta PASSED_FOR_REVIEW, ricalcolo
indipendente concorde: λ InfoNCE 0,11942686122572264, λ RnC 0,33044627364850215, entrambi ammessi.
Evidenze in `calibration_completed_20261117/`, coorte in `calibrated_cohort_20261117/`.
Release CUDA e cinque benchmark condizionati in `docs/reviews/contrastive_cuda_benchmark_release_20261117.md`
e `releases/cuda_benchmarks_20261117.json`. Prossimo checkout `contrastive-v2-20261117-benchmarks`.
Prima sottomettere CUDA (`CONTRAST-CUDA-v2-20261116`); solo dopo rapporto e probe validi, pubblicati,
sottomettere tre benchmark completi e due riprese. Non sostituire la lettura con il solo afterok.
Sorgenti e plan.json invariati; nessun nuovo consenso richiesto. La cronologia sotto conserva i vecchi snapshot.


**Rettifica autorizzata da Simone il 17/11:** usare il rapporto completo dell'audit 64652559 senza rifarlo
e lanciare la calibrazione GPU. La deroga riguarda esclusivamente il rapporto SHA
`5c720fadbcb12fd5458bc0e19cd03f28fec30ad5973ca72739f01bf1b3d7547b`. Lo stato TIMEOUT e il costo
8,415556 core-ore restano nel consuntivo. Release concreta verificata e review indipendente favorevole
in `docs/reviews/contrastive_calibration_release_20261117.md`; nessuna nuova review attribuita a Claude.
Calibrazione **64701482 RUNNING** alle 19:49:20 Europe/Rome: 512 batch senza update, 4 GPU, 32 CPU,
240 GiB, 30 minuti, nodo lrdn2820. ID logico `CONTRAST-CALIB-v2-20261116`: non duplicarlo.
Checkout **64701182 COMPLETED 0:0**, 106 s, SHA `4fc515257b1965a7ac1aabad22df3ade6b651b1a`.
Cartella `/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/contrastive-v2-20261117-calibration`.
Ricevute, hash verificati sul cluster e snapshot in `calibration_submission_20261117/`.
Log `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/contrastive_gpu_64701482.out`.
Output `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_preflight_20261116_v2/calibration`.
Consuntivo audit più tre checkout: 8,578889 core-ore; GPU ancora in corso.
I tre checkout residui hanno tetto 240 s: totale massimo aggiornato 129,261111 core-ore entro 129,333333;
14 GPU-ore invariato. Nessun training scientifico rilasciato, nessun audit duplicato o monitor.

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

## Seguito già autorizzato: recupero puntuale e calibrazione

1. Il prerequisito audit è soddisfatto dalla rettifica esplicita di Simone, dal rapporto completo verificato
e da `releases/audit_recovery_20261117.json`. Non pretendere di nuovo COMPLETED per questo esatto job,
non rifare l'audit e non estendere la deroga a fasi successive. La lettura precedente resta agli atti.
2. **Già eseguito, non ripetere:** preparata release di calibrazione `APPROVED_FOR_PREPARATION` con SHA dell'audit, del config e dei quattro input, 4 rank, 6 worker/rank, `time_limit_s=1770`. Job wrapper `APPROVED_FOR_PREFLIGHT`, modalità `calibration`, allocazione 4 GPU, 32 CPU, 240 GiB, 30 min. Pin della presente autorità nel file di release; nessun obbligo di far approvare nuovamente a Simone la stessa fase.
3. **Già eseguito:** release e piano pubblicati, SHA remoto verificato, nuovo checkout congelato pronto e job 64701482 sottomesso. Nessuna modifica ai sorgenti e al piano scientifico. Se il checkout è già COMPLETED ed è scaduto dal controller, verificare sacct e integrità e non usare un afterok inesistente; i gate del launcher restano obbligatori.
4. Calibrazione completata: copiare metadati, ricalcolare ricevuta e coorte con `contrastive_prepare.py`, pubblicarle. Se uno dei λ è fuori range fermare tutta la coorte. Il checkpoint fisso di calibrazione serve al test CUDA: conservarlo.
5. CUDA/NCCL con prova di uscita di un rank, poi tre benchmark e due riprese, secondo piano. Tutti gli invii tramite `scripts/submit_job.py` e ledger comune. Il budget dei training deve venire dal rapporto misurato, non dai tempi storici.

Il piano scientifico `plan.json` è ancora documentariamente REVIEW_REQUIRED per ricordare che Claude non ha riletto le correzioni: la sua identità resta congelata. L'approvazione operativa successiva è esplicita in `preflight_plan.json`, `releases/audit.job.json` e `docs/reviews/contrastive_preflight_release_20261117.md`. Non riscrivere il piano scientifico dopo la calibrazione: le ricevute lo pinnano.
