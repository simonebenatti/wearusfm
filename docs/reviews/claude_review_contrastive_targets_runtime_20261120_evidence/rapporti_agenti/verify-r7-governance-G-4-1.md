# Verifica scettica n. 1 del rilievo G-4 (r7-governance)

Esito: **NON CONFUTATO**. Gravita' confermata: **NOTA**. Con una precisazione: delle due frasi citate solo una e' falsa.

## Riproduzione (comandi miei)
- `git show fbfde797:scripts/submit_job.py | sed -n 190,195p`: `code_sha_from` cerca solo `EXPECTED_SHA=([0-9a-f]{40})`; :227 `"codice_sha": code_sha_from(sbatch_args)`; :120 il ledger scrive `codice_sha or ""`.
- `S.code_sha_from([... '--export=ALL,CAMPAIGN=contrastive-targets-audit-20261119,CAMPAIGN_SHA=<40hex>'])` -> `None`; con `EXPECTED_SHA=<40hex>` -> lo SHA.
- Invio simulato con il comando dello schema del prontuario (:166-173), submit_job.py della copia e un finto ssh che risponde `777`:
  le righe del ledger `prenotato` e `sottomesso` hanno `codice_sha=[]`, lo SHA compare solo nella colonna `comando`; in state.json `codice_sha` e' `None`.
- Tutte le versioni di scripts/submit_job.py nei 37 riferimenti della copia (3 blob distinti, 33 riferimenti col blob di 0efa4d45) hanno solo la regex `EXPECTED_SHA`.
- `tests/cpu/test_submit_job.py`: 20 passati; `test_script_passed_on_stdin` (:190-199) usa `CAMPAIGN_SHA` ma non verifica `codice_sha`.

## Contesto che rafforza il rilievo
- Ogni checkout storico di GPT passa `EXPECTED_SHA` accanto a `CAMPAIGN_SHA` proprio per riempire `codice_sha`:
  consolidation 17dd455, ft-v2 ae28305, readout contrastivo 0c3c529 e diagnostica comune 365cb1e
  (`results/infra/contrastive_comparison_20261116/common_state_diagnostics_20261118/submission_plan.json:30`).
  Nel ledger di 94f5094 (`..._evidence/ledger_contrast_ultime_righe.tsv`) `CONTRAST-CHECKOUT-common/readout-20261118` hanno `codice_sha` pieno.
  Lo schema nuovo (:172) toglie `EXPECTED_SHA` e la frase falsa di :177 lo giustifica.
- Review Claude `claude_review_consolidation_long_20261105.md:464`: «`codice_sha` viene da `EXPECTED_SHA` nella riga di `sbatch`».

## Precisazione sulla prima frase
- :162-163 «(lo SHA è anche quello dichiarato al submitter)» e' ambigua, non dimostrabilmente falsa: lo SHA e' passato a submit_job.py
  negli argomenti (`--export`) e finisce in `sbatch`/`comando`. E' falsa solo se la si legge come «registrato come codice_sha».
- :176-177 «il submitter ricava lo SHA da `CAMPAIGN_SHA`» e' falsa: «il submitter» qui e' submit_job.py (:161, :175 `--agente`) e non ricava nulla da `CAMPAIGN_SHA`.

## Perche' resta NOTA
- Il checkout si verifica da se': `campaign_checkout.sbatch:30,33,36,45,48` (SHA da `CAMPAIGN_SHA`, 40 hex, commit presente, `HEAD == SHA`).
- Nessun modulo nuovo della linea 3 legge il ledger o `codice_sha` (git grep su scripts/contrastive_*.py e sui piani).
  Leggono `codice_sha` solo script della linea 1 (scaling, `finish_submissions.py:91,97`, `audit_recovery.py:27`).
- Lo schema e' dichiarato provvisorio: «Gli esempi non sono stati eseguiti» (:150), «da completare dopo la review» (:164),
  comandi definitivi «emessi nella consegna post-review» (:178-179).
- Effetto se seguito alla lettera: `codice_sha` vuoto per i sei checkout, una perdita di provenienza nel ledger rispetto alla prassi storica.
  Nessun cancello falsato, nessun esito attribuito male: ID logico, job e SHA restano nella colonna `comando`.
- `source_digest` (contrastive_prepare.py) copre src/wearusfm, scripts/contrastive_*.py, scripts/slurm/contrastive*.sbatch, train_jepa.py
  e lib_code_dir.sh: submit_job.py e campaign_checkout.sbatch ne sono fuori (corretto quanto detto nel fix).

## Correzione suggerita
Aggiungere `EXPECTED_SHA=<stesso SHA>` all'export dello schema di :172, come nei checkout storici, e correggere :176-177.
Non e' richiesta una modifica del submitter comune, che serve tutte le linee.
