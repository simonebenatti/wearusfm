> **Storico 25/50.** Per la nuova coorte RMS100/fast125 usare [consegna del19/11](contrastive_targets_execution_20261119.md) e i nuovi piani. Tempi, review, eccezione TIMEOUT e release sotto non autorizzano nuovi invii.

# 3-launcher — lettura CPU contrastiva già autorizzata

Simone ti affida i nuovi lanci della linea 3. La lettura CPU è già autorizzata;
non chiedere un altro consenso. GPT scientifico resta responsabile dell'analisi.

Repository: `/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-logcontrast`.
Ramo `origin/codex/tokenizer-logcontrast-20261104`. Pacchetto e comandi pubblicati
nel commit `2a57d4a7d111509ef87dda0a76907013ee131333`.
Checkout di esecuzione da congelare ESATTAMENTE a
`0c3c52936e8b7e9166f351ff8003b40dfe40427e`.
Non modificare il worktree dello scientifico; usa il tuo checkout e conserva gli SHA.

Leggi:

- `docs/reviews/contrastive_readout_release_20261118.md`;
- `results/infra/contrastive_comparison_20261116/releases/readout_bf16_20261118/submission_plan.json`;
- nella stessa directory, `readout.job.json`, `resources.json`, `local_validation.json`, `independent_review.txt`.

Tutti i cinque benchmark sono COMPLETED 0:0: C0 64771457, CN 64771478, CR 64771595,
ripresa CN 64771859, ripresa CR 64772690. Tutte le summary finali sono a 576 passi.
Non duplicarli. I metadati e i log sono archiviati con 25 hash remoti verificati.

La parità iniziale delle norme dei gradienti è già negativa nei metadati, mentre
loss totale, stream e rapporti delle norme attive passano. Esegui comunque la lettura
CPU per completare controlli checkpoint, optimizer, coda, RNG e riprese.
Non modificare criteri, λ o sorgenti. FAILED_REQUIRES_REVIEW con exit 2 è un esito
atteso da raccogliere, non un motivo per riprovare. Nessun training lungo autorizzato.

Procedura:

1. Verifica commit su origin, nessuna prenotazione degli ID sotto, checkout/output nuovi.
2. Esegui `deploy_command` del piano: pubblica solo il ref dedicato allo SHA congelato,
   mai main o riferimenti di campagne attive. Verifica lo SHA su Leonardo.
3. Sottometti con il submitter e ledger comuni i due comandi esatti contenuti nelle
   liste `commands` del piano. Non ricostruire quoting o risorse a mano.
4. ID: `CONTRAST-CHECKOUT-readout-20261118` e `CONTRAST-READOUT-bf16-20261118`.
   Il secondo è accodato afterok al checkout; il launcher CPU già presente nel vecchio
   checkout bf16 è identico, ma CODE_DIR/EXPECTED_SHA indicano sempre il nuovo checkout.
5. Raccogli ricevute, sacct, log, rapporto finale e SHA. Per errori strutturali conserva
   il traceback; niente retry, aumento di soglia, risorse o cambi di protocollo.
6. Committa/pusha le evidenze sul tuo ramo e riporta a Simone ID, stato, esito e riferimenti.
   Nessun monitor in background. La lettura scientifica prosegue nella chat di origine.

Checkout: `contrastive-v2-20261118-readout`, 4 CPU / 2 GiB / 240 s.
CPU: 8 CPU / 30 GiB / 3600 s, zero GPU. Tetto residuo totale 8,266667 core-ore.
Massimo campagna incluso residuo: 5,284444 GPU-ore / 59,294444 core-ore.
Output: `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/contrastive_preflight_20261118_bf16/preflight_readout.json`.
Source invariata: `46938f319181d408ee852cfaacfdc92cc5e44eaa1cfa770825ad96956eec9320`.
Job SHA256: `1b2178156df0ecb28add9be93109e6cf205380f68053b42f5e0b5adcad7af7b5`.
Launcher CPU SHA256: `683ada3cf4ceebb369ce4e218d82ca3b6a861c071e99b3586e62d0c51e184884`.

Non ricreare audit, calibrazione o CUDA: il lettore usa gli originali nei percorsi
remoti registrati, compreso il probe CUDA. Legge e autentica tutti i checkpoint reali.
