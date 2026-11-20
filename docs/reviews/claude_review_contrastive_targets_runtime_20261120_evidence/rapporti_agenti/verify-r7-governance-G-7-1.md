# verify-r7-governance-G-7-1 (scettico n. 1 su G-7, NOTA)

Avvio 20/11. Clone di sola lettura a 0efa4d45; nessuna mutazione, nessun clone temporaneo necessario.

## Progressi 1 — riproduzione
- Link della copia faa4a8d7 (blob 5bdb0d15 identico a faa4a8d7/fbfde797/0efa4d45): 6 link; 2 relativi rotti a fbfde797 e 0efa4d45
  (`../../results/infra/contrastive_comparison_20261116/common_results_20261119/`, `contrastive_common_diagnostics_readout_20261118.md`),
  2 relativi validi (gradient_parity_diagnostic, readout_results), 2 permalink GitHub (438c8d65, 94f50942).
- common_results_20261119: 128 file a faa4a8d7 (stesso tree f9e5dfd4 a 8e38eb71); readout: blob e41fbc17 aggiunto in db66b507.
- faa4a8d7 non e' antenato di fbfde797 (merge-base d9020c20); 8e38eb71, 3cc74ad6, db66b507 neppure.
- decisioni.md e registro di fbfde797/0efa4d45: grep 0 per la voce faa4a8d7 «Lettura dei diagnostici e proposta ...»/«training e invii sospesi»
  e per le due voci del registro («Lettura scientifica dei diagnostici ...», «Portata del risultato ...»). decisioni:8280 «attua la decisione scientifica `faa4a8d7`».
- Review Claude copiata: blob c25d4f6f identico a 94f50942; cartella `..._evidence/` 320 file a 94f50942, 0 a fbfde797/0efa4d45.
- plan.json:368-369 "faa4a8d7"/"94f50942" abbreviati (generati da contrastive_targets_plan.py:33-34); delivery_pin.json:5-6 SHA completi.

## Progressi 2 — tentativi di confutazione
- Nessun consumatore: `git grep` in scripts/tests/src di fbfde797 per common_results, diagnostics_readout, science copy, review Claude,
  science_decision_commit, preparation_review_commit: 0 righe (solo il generatore contrastive_targets_plan.py:33-34 li scrive). Effetto solo documentale.
- SHA abbreviati univoci: `git rev-parse --disambiguate=faa4a8d7` e `=94f50942` danno un solo oggetto ciascuno; completi in delivery_pin.json:5-6
  e, per 94f50942, anche nel prontuario r.4. "pinna" e' eccessivo: campi descrittivi, non pin verificati.
- Provenienza parziale gia' sul ramo: prontuario r.4-6 «decisione scientifica in `faa4a8d7`, copiata byte per byte»; la copia stessa r.5 nomina
  «Evidenze al commit `8e38eb71` ... criteri di lettura `db66b507`» e i due oggetti contengono davvero i file linkati (128 file; blob e41fbc17);
  registro fbfde797:184 «conservate nel ramo»; contabilita' dei diagnostici copiata in amendment_20261119/historical_diagnostics_accounting.txt.
  La review Claude copiata nomina il proprio ramo (origin/worktree-claude-training-engine) e la cartella locale delle prove.
- Registro/decisioni per ramo: CLAUDE.md:194-195 (fbfde797) «ciascuno scrive la sua voce ... sul proprio branch; l'integratore la conserva»:
  l'assenza delle voci faa4a8d7 su un altro ramo GPT e' il flusso previsto, non un difetto in se'. Le soglie nuove sono in decisioni.md di fbfde797 (voce 19/11).
- Cosa resta vero: nessuna nota indica ramo d'origine o SHA completo accanto alla copia, ne' avverte che 2 link relativi su 4 risolvono solo
  su faa4a8d7; la cartella _evidence (320 file) non c'e'; decisioni.md del ramo dice «attua» senza la voce «Proposta per Simone ... invii sospesi»
  (lo stato di proposta resta pero' leggibile nella copia, r.3 e r.52).

## Esito
NON confutato come NOTA (chiarezza/documentazione). Nessun effetto su catena, cancelli, binding o tetti. Sotto-punto registro/decisioni debole
(flusso per ramo), sotto-punto "pinna" sovrastimato; nucleo (link relativi rotti, prove non copiate, provenienza senza ramo/SHA completo accanto
alla copia) riprodotto.
Non verificato: punta remota reale dei rami d'origine (niente fetch); se la release sara' figlia di 0efa4d45 (con delivery_pin) o di fbfde797
(validate_review_binding confronta solo i byte, non l'ascendenza).
