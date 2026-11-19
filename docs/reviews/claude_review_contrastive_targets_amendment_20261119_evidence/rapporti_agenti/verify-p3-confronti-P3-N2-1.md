# Scettico n. 1 - rilievo P3-N2 (C0-T100-G ancora «controllo descrittivo di compatibilità»)

Copia: clone @ 438c8d65 (detached). I tre file citati sono identici fra 0efdc29c e 438c8d65
(`git diff --quiet 0efdc29c 438c8d65 -- <protocollo> <emendamento> <plan.json>` -> exit 0).

## Riproduzione (comandi miei)
- `grep -n -i compatib docs/contrastive_comparison_protocol_20261116.md docs/contrastive_targets_amendment_20261118.md`:
  protocollo:132 «Si riporta **C0−T100-G storico a 19.618** come controllo descrittivo di compatibilità: una ripetizione
  a seme uguale ...»; nell'emendamento l'unica occorrenza è la riga 58 (whitelist di compatibilità SORGENTI, altro tema).
- Protocollo 1-5 (aggiunto nel diff e388a1de..0efdc29c, unica modifica al file): «Per la prossima coorte prevale la ricetta
  esplicita RMS100/fast125 [link all'emendamento]. Il testo seguente conserva il protocollo statistico e la storia della coorte 25/50».
- Emendamento 27-29: «non isola più la sola infrastruttura contrastiva e non misura l'effetto di un singolo nuovo target».
- plan.json:300 «Historical T100-G differs in two target supports; no matched historical attribution»;
  plan.json:338 «cannot attribute difference to contrastive scaffolding or precision alone». `grep -i compat plan.json`: nulla.
- decisioni.md (voce 18/11): «confronto storico con due fattori aggiuntivi dichiarati»; registro: «non isola la sola infrastruttura».
=> Le premesse fattuali del rilievo sono riprodotte: nessun testo della consegna ritira esplicitamente il ruolo
   «controllo di compatibilità» per la nuova coorte.

## Tentativo di confutazione
- A favore della confutazione: (1) l'intestazione subordina il testo storico alla ricetta nuova; (2) nel vocabolario del
  protocollo «compatibilità» = C0 (nuova implementazione, peso contrastivo 0) contro G storico (proposta 07/11 righe 15 e 20:
  C0 «Controllo effettivo della nuova implementazione», G «controllo descrittivo di compatibilità»); l'emendamento dice
  «non isola più la sola infrastruttura» e il piano nega l'attribuzione a «scaffolding or precision», cioè proprio le due
  componenti che il controllo copriva (par. 6: «Percorso numerico e non determinismo possono contribuire"); (3) nessun codice
  lo calcola: `contrastive_compare.ARMS = ("C0","CN","CR")` (riga 26), T100-G compare negli script solo come BASELINE dei config
  in contrastive_prepare.py:28,113-124; plan.evaluation.pairs = CN-C0, CR-C0, CN-CR; nessun cancello.
- Perché non basta: l'emendamento dice cosa il confronto NON fa più, non quale ruolo abbia ora; il par. 6, che l'intestazione
  dichiara conservato come «protocollo statistico», continua a dire «Si riporta ... come controllo descrittivo di compatibilità»
  e descrive C0-T100-G come «una ripetizione a seme uguale», premessa non più vera (C0 nuovo vs T100-G: blocco contrastive,
  loader.rms_window_ms MISSING->100, loader.ms_fast_window_ms MISSING->125, target_version v1->fast125_v3, campi LQ inerti;
  amp_bf16 uguale; script ../tmp/sk1_p3n2/cfgdiff.py). La lettura composta resta «un controllo di compatibilità che non isola
  ciò che controlla»: imprecisione documentale reale, senza effetti.

## Esito
NON confutato. Gravità reale: NOTA (solo chiarezza; nessun codice, cancello o attribuzione dipende dall'etichetta; il punto 3
«C0−T100-G con due fattori aggiuntivi dichiarati» è già soddisfatto da emendamento 27-29, plan.json:300/338 e decisioni).
Osservazione sulla correzione proposta: se si aggiunge la frase, l'elenco dei fattori non va limitato a «impalcatura, RMS100,
spettro 125/v3»: par. 6 e plan.json:338 includono anche percorso numerico/precisione (e versione del codice) e non determinismo;
conviene anche dire che il C0 nuovo non è una «ripetizione a seme uguale» del T100-G. Correzione senza nuovo giro di review.
