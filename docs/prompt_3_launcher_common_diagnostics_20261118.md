# Prompt per 3-launcher

Simone ti affida l'esecuzione su Leonardo dei due diagnostici contrastivi. Usa il ramo `origin/codex/tokenizer-logcontrast-20261104` e il pacchetto di esecuzione `365cb1e29bc410013d4ee0494b9c2e92835bc7f1`. Claude ha approvato disegno, implementazione e prerequisiti al commit `0afd0864dd96d474e2a39cb4276c2b990d21c6c0` sul ramo `origin/worktree-claude-training-engine`. Procedi con il piano di `35ea94ba` sullo SHA di esecuzione sopra.

1. Leggi `docs/contrastive_common_diagnostics_20261118.md` e `results/infra/contrastive_comparison_20261116/common_state_diagnostics_20261118/submission_plan.json` nel commit di consegna di questo prompt. Controlla che `frozen_commit` coincida con lo SHA sopra. Verifica i pin dei launcher e del submitter, lo spazio richiesto da `resources.json`, la quota debug dell'utente e gli ID nel ledger. Le risorse sono un checkout CPU e tre celle indipendenti da 4 GPU per 30 minuti ciascuna (6 GPU-ore massime complessive).
   **Subito prima di ciascun invio**, checkout incluso e anche se differito per quota, rigenera il piano dal worktree locale indicato dagli argv. Controlla sia i sei pin di `local_transport_and_driver_sha256` (due launcher, due driver, risorse e generatore) sia il pin separato del submitter. Dalla radice di `wearusfm-logcontrast` usa il blocco seguente. Il confronto usa una directory temporanea e il piano originale salvato in Git. Procedi con l'argv della singola cella solo se tutti i comandi riescono.

   ```bash
   (
     set -eu
     d=results/infra/contrastive_comparison_20261116/common_state_diagnostics_20261118
     tmp=$(mktemp -d)
     git show "35ea94ba148fea5d259bb1e5a3d5ff3431dc54a5:$d/submission_plan.json" > "$tmp/approvato.json"
     python3 "$d/prepare_handoff.py" --dry-run --frozen-sha 365cb1e29bc410013d4ee0494b9c2e92835bc7f1 > "$tmp/corrente.json"
     cmp "$tmp/approvato.json" "$tmp/corrente.json"
   )
   ```

2. Esegui l'argv `deploy_command`, verifica lo SHA remoto sul ref dedicato, poi usa gli argv di `commands` con il submitter e il ledger comuni. Il checkout è `CONTRAST-CHECKOUT-common-20261118`. Le tre celle sono `CONTRAST-COMMON-cuda-20261118`, `CONTRAST-COMMON-resume-CN-20261118` e `CONTRAST-COMMON-resume-CR-20261118`. Mantieni al massimo due celle debug in esecuzione contemporanea per utente su `boost_qos_dbg`, contando anche gli altri job debug dell’utente. La terza può restare in coda nei limiti di sottomissione. Evita duplicati. Per una dipendenza ormai scaduta usa gli argv alternativi del piano solo dopo verifica di ID, `sacct COMPLETED 0:0` e HEAD del checkout.
3. Per CUDA raccogli `report.json`, i rapporti e i log per profilo e gli eventuali errori per rank. Restituisci confronti tra repliche e tra bracci, confronto FP32/bf16 a backend math uguale e controlli di batch, stato, supporto e RNG. Per CN e CR raccogli i tre confronti al passo 588 (continuo con ciascuna ripresa e le due riprese tra loro), gli hash del punto comune a 580, tracce per rank incluse `*.ddp.jsonl`, primo scarto e distanze dei modelli. Conserva gli eventuali `failure.json` o artefatti parziali se una cella si interrompe.
4. Riporta a Simone ID e stati SLURM, consuntivo, SHA-256 dei rapporti, percorsi e commit delle evidenze. Conserva i checkpoint e i dati del segnale sul cluster. La consegna riguarda questi diagnostici. Il rapporto storico resta `FAILED_REQUIRES_REVIEW` e i training lunghi richiedono la lettura scientifica successiva.
