# Scettico 1 - P2-N4 (riga di autorizzazione del 16/11 copiata nel piano nuovo)

Copia git 438c8d65, pulita. Solo letture.

## Riproduzione dei fatti
- python3: scientific_decisions dei due plan.json -> status, D1, D2, D3, authorization IDENTICI; target_windows solo nel nuovo.
  diff riga per riga storico 267-272 vs nuovo 303-308: differisce solo la virgola finale. Copia testuale CONFERMATA.
- Testo introdotto in c680e95d (16/11) nel piano storico e nel protocollo; copiato in 0efdc29c.
- Lettori: git grep "scientific_decisions" sull'intero albero -> solo i due plan.json. Nessuno script/test contrastivo
  legge "authorization". contrastive_prepare.py legge del PLAN solo calibration/coverage (policy_reference, :75-84),
  input_sha256 (:107) e lo pinna in _reader_provenance (:219). "Nessun codice la legge" CONFERMATO.

## Tentativo di confutazione della parte "resta implicito"
1. Livello operativo (il preflight_plan governa gli invii). Storia del piano storico:
   `git show c680e95d:results/infra/contrastive_comparison_20261116/preflight_plan.json` (16/11, ancora REVIEW_REQUIRED)
   -> "submission_authorized_after_convergence": true = traduzione operativa della riga "launches after convergence";
   821f4d73 (17/11) aggiunge approval_authority "Simone explicit instruction 2026-11-17" e review_gate.
   Nuovo preflight_plan.json (anch'esso REVIEW_REQUIRED): chiave after_convergence ASSENTE, "submission_authorized": false (:5),
   "training_authorized": false (:6), nessun approval_authority/review_gate; required_before_first_submission[0]
   "publish and present resource proposal to Simone" (:191); training_release "Absent; ... separate training budget required" (:198).
   Le 9 celle: REVIEW_REQUIRED, release null, code_sha null; l'audit dipende da new_review/new_checkout/diagnostic_resolution.
   => la convergenza da sola NON autorizza la nuova coorte: scelta esplicita, non implicita.
2. Nel plan.json la riga e' autolimitata ("after convergence; this file is not a deployment release") ed e' seguita nello stesso
   blocco da target_windows.execution "Prepare and present new costs before any launch; no release in this file" (:315);
   in testa executable false, training_authorized_by_this_file false (:4-5), release null.
3. Emendamento :84-85 "L'autorizzazione dei vecchi invii non viene copiata nel nuovo piano come se i nuovi gate fossero gia'
   passati"; sezione "Preventivo da presentare prima degli invii"; intestazione del protocollo "Nessun nuovo lancio e' rilasciato";
   decisioni.md 18/11 "non e' rilasciato".
4. La copia e' legittima: registra le decisioni di Simone del 16/11 (decisioni.md: "Ha gia' autorizzato GPT a lanciare dopo la
   convergenza con Claude") per il protocollo contrastive-comparison-20261116-v2, che il piano nuovo conserva (emendamento :25-27);
   D1/D2/D3 devono restare. L'autorizzazione specifica della coorte (via del 17/11 "per questo giro", APPROVED_FOR_*,
   approval_authority, audit_recovery) non c'e': `git grep -c -E "APPROVED_FOR|approval_authority|submission_authorized_after_convergence|review_gate|audit_recovery|2026-11-17 authorization" 438c8d65 -- results/infra/contrastive_targets_20261118 ':!*.log'` -> exit 1.
5. Nessun effetto: nessun lettore; i cancelli sono altrove (contrastive_job.py:64-68 APPROVED_FOR_PREFLIGHT + source;
   contrastive_run.py:165-166 APPROVED_FOR_PREPARATION/TRAINING, :176 review_pin, :157-158 training con
   scientific_decision_pin e budget_decision_pin).
6. La seconda meta' della correzione proposta ("richiedono l'approvazione del preventivo nuovo") aggiungerebbe una regola di
   governo non presente nelle frasi registrate di Simone; quella documentata (presentare a Simone, invio non autorizzato) e'
   gia' esplicita, e presentare/approvare e' regolato da CLAUDE.md per ogni job, non da questo piano.

## Esito
refuted=true. Fatti riprodotti (copia testuale, nessun lettore), ma la tesi centrale (portata implicita per la nuova coorte)
e' contraddetta da dichiarazioni esplicite, soprattutto dalla rimozione deliberata di submission_authorized_after_convergence
nel preflight_plan nuovo. Residuo solo di stile (si potrebbe datare la riga "16/11"): al massimo NOTA, nessun effetto.
Nessun file modificato nella copia git; nessun pytest eseguito (non necessario).
