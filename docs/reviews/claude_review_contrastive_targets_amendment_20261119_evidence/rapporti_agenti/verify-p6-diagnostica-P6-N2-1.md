# Scettico n. 1 - rilievo P6-N2 (documenti diagnostici pre-approvazione, db66b507 non integrato)

Copia: clone @ 438c8d65 (sola lettura). Merge di prova solo in clone temporaneo tmp/skp6n2_mt, poi rimosso.

## Riproduzione (comandi miei)

- `git merge-base --is-ancestor db66b507 438c8d65` -> NOT_ANCESTOR. db66b507 non e' antenato neanche di main f546c9e4;
  lo contiene solo `origin/codex/tokenizer-logcontrast-20261104` (ne e' la punta).
- Orari e genitori: 65bc6022 22:12:21 (0efdc29c + d9020c20); db66b507 23:29:56 (figlio di d9020c20);
  afb89b45 23:45:54 (65bc6022 + f546c9e4); 438c8d65 23:47:27.
- Blob: `docs/contrastive_common_diagnostics_20261118.md` 6149836e a d9020c20/65bc6022/438c8d65, 2dfed33b a db66b507;
  `docs/prompt_3_launcher_common_diagnostics_20261118.md` 566efddc contro 0d8e5a21; la scheda
  `docs/reviews/contrastive_common_diagnostics_readout_20261118.md` esiste solo a db66b507 (e41fbc17).
  Riga 3 alla punta: «Stato della consegna: **REVIEW_REQUIRED**».
- db66b507 tocca 5 documenti, nessuno nell'insieme di `source_digest()` (scripts/contrastive_prepare.py:192-199).
  `git merge-tree --write-tree 438c8d65 db66b507`: i due documenti diagnostici si fondono puliti, conflitti in
  `docs/claude_GPT_shared` e `docs/decisioni.md`.

Fatti del rilievo: CONFERMATI.

## Perche' non e' un difetto della consegna

1. Dichiarazione esplicita nella consegna committata: docs/prompt_claude_contrastive_targets_20261118.md:20-21 @438c8d65,
   «I successivi commit della diagnostica da stato comune, fino a `d9020c20`, sono conservati nel merge `65bc6022` [...]:
   non sono una nuova release dei diagnostici». Il rilievo stesso scrive «come dichiarato».
2. Il REVIEW_REQUIRED del protocollo congelato e' voluto: il prompt presente alla punta (566efddc, riga 3) dice
   «Il protocollo in questo commit identifica la consegna come `REVIEW_REQUIRED`. Verifica il verdetto di Claude sul
   medesimo codice prima del deploy. Se approva lo SHA, esegui direttamente.» e indica il ramo autorevole
   `origin/codex/tokenizer-logcontrast-20261104`. Chi legge solo questo ramo viene mandato al ramo giusto e al verdetto
   esterno: l'etichetta non porta a un'azione sbagliata.
3. I documenti della nuova coorte trattano la diagnostica come binario separato: emendamento:6-7 (diagnostica in
   `wearusfm-logcontrast`), :50-51 («preparata nell'altra chat, poi registrata la risoluzione [...] o un emendamento
   esplicito»), README:15-16 («non eseguirli da questo nuovo checkout»), preflight_plan
   `required_before_first_submission` («separately prepared common-state diagnostics»), plan.json:349
   («common-state diagnostics remain separate prerequisite»).
4. Nessun consumatore: nessun file della nuova coorte pinna o legge i documenti diagnostici (grep su
   results/infra/contrastive_targets_20261118: nessun riferimento); i documenti sono fuori dal digest; la diagnostica gira
   da 365cb1e2 con il piano 35ea94ba; nessuna release esiste ancora (`release: null` nelle 9 celle, `training_release`
   «Absent»).
5. Flusso del progetto: db66b507 non e' neanche su main; afb89b45 allinea il ramo a main f546c9e4; l'integrazione del ramo
   logcontrast spetta all'integratore. Il merge proposto non e' «di soli documenti» senza lavoro: confligge su registro e
   decisioni.
6. Il residuo utile (citare per SHA approvazione 0afd0864, decisione e scheda di db66b507 nel cancello
   `diagnostic_resolution` quando si prepara la release) coincide con la NOTA (1) dello stesso revisore P6 («cancello solo
   procedurale e non identificato (pinnare report, decisione e scheda nella release ...)»): P6-N2 non aggiunge un difetto
   distinto.

## Esito

refuted=true come difetto a se'. Se lo si conserva: al piu' NOTA, da fondere con P6-N1; eventuale ritocco di chiarezza
nel README:15 («integrati fino a d9020c20; approvazione e scheda di lettura su db66b507»), gia' detto nel prompt dello
stesso commit. Pulizia del clone temporaneo fatta.
