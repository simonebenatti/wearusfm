# Integrazione su main delle tre linee e del lavoro Claude — 06/11/2026

Simone: «completa anche l'integrazione su main senza interferire coi job che stanno girando, poi committa e pusha tutto».
Autore: GPT coordinamento, nel worktree isolato `wearusfm-integration`, ramo `codex/integration-main-20261106`.

## Esito e perimetro

Integrati sviluppo scientifico precedente, coordinamento, scaling, consolidamento, log-quadratico,
review e consegne operative di Claude. D-1 e D-2 dell'audit del 05/11 sono chiusi da questa consegna.
Merge ordinari con SHA originali conservati, senza squash o rebase. La pubblicazione finale aggiorna
`origin/main` per fast-forward; non aggiorna il remoto Leonardo né i riferimenti `deploy/*`.

Main all'inizio: `fe66e2db76001316b5750f3b25b76420dce73a35`. Incluso anche
`dcd2d9a2e15972026fc2a6813e78b028170f8208`, pubblicato durante la verifica: codice e operatività
della linea 1 passano a Claude, GPT conserva ragionamento scientifico, direzione e review.
Per la linea 2 Claude resta operatore, GPT mantiene codice e responsabilità scientifica;
la linea 3 resta a GPT. CLAUDE.md conserva queste istruzioni e le note di manutenzione di Claude.

## Consegne integrate

| Consegna | Riferimento incluso |
|---|---|
| Coordinamento e precedente ramo scientifico | `012f8e7`, comprende `8ee6603` |
| 1 — Scaling, review e passaggio a Claude | `0f4a9cf`, comprende il codice congelato `97125bd` |
| 2 — Consolidamento e consegna operativa lunga | `2df9aff`, comprende `166fb86` e `9d195e7` |
| 3 — Log-quadratico, diagnostica e correzioni Claude | `63eba9a`, comprende `ff08483` |
| Registro comune e review di Claude | `72ccb6f` e `d3976c3` |
| Main e assegnazioni degli operatori | `fe66e2d` e `dcd2d9a` |

I 14 pin completi, inclusi gli antenati di consegna, sono in
`results/infra/integration_main_20261106/inputs.json`; l'elenco dei merge è in `merge_commits.txt`.

## Conservazione di codice e documenti

Nessun conflitto in `src/`, `scripts/` o nei test. I conflitti erano nel registro, nelle decisioni
e nella copia del documento di allineamento: per quest'ultimo è conservata la versione più recente
di Claude, con chiusura successiva aggiunta da GPT. Registro e decisioni conservano testualmente ogni
blocco di tutte le 14 versioni sorgenti; le due sezioni del registro hanno ciascuna un'unica ancora.
La verifica riproducibile è `python3 results/infra/integration_main_20261106/verify_documents.py`.

I file di produzione introdotti/modificati dalle tre linee sono identici ai rispettivi pin:
15 scaling, 17 consolidamento, 22 log-quadratico. Il submitter comune mantiene il blob
`7d5dc8222218b9174d14ede3245f777947a402f6` in tutti gli input e nel risultato.
Prove in `code_union_proof.json`, `document_union_proof.json` e `document_verification.json`.
Le spaziature dei log raccolti sono conservate, per non alterarne i byte e la provenienza.

## Verifica CPU e correzioni ai test

La prima suite completa ha eseguito **3.342 casi** in 1.831,73 s:
3.181 passati, 19 fallimenti e 142 errori di preparazione. Questi ultimi dipendevano da test che
applicavano contratti/configurazioni di campagne congelate all'albero integrato, che ha legittimamente
file e campi nuovi; un test assumeva inoltre che il piano lungo fosse ancora in attesa di review.

Il commit **3263ee9** corregge soltanto i test e i loro fixture temporanei:

- Le prove positive dei contratti storici usano i sorgenti effettivi dei rispettivi SHA, estratti
  con `git archive`; la ricetta storica non viene riscritta sul codice corrente.
- I test sintetici sul codice integrato generano pin temporanei esplicitamente sintetici.
  Nessun piano di produzione, hash pubblicato, guardia o controller viene modificato.
- Le prove negative confermano che i contratti storici rifiutano ancora byte diversi o campagne
  non rilasciate; due nuovi test verificano separatamente accettazione storica e rifiuto su main integrato.

La riesecuzione completa dei **13 moduli coinvolti**, incluso il launcher B19618, ha dato
**471 passati** in 113,79 s. Comprende tutti i 161 casi inizialmente falliti/in errore e due casi nuovi.
Dopo l'ultimo merge di sole ricevute/piano operativo sono passati anche **22 test** del piano lungo.
Il conteggio combinato per identità del caso è **3.344 passati, zero casi irrisolti, zero saltati**.
Non è presentato come una seconda esecuzione completa: metodo, conteggi e risultati per caso sono in
`test_summary.json` e `test_outcomes.json`. Il launcher B19618 mantiene i 15 test verdi della correzione D-5.

## Separazione dall'esecuzione

Nessun job sottomesso, cancellato o rilanciato da questa integrazione; nessuna scrittura su Leonardo,
nessun checkout attivo, ambiente, checkpoint, cache o ledger modificato. Si aggiorna soltanto Git sul Mac
e su origin. Gli owner continuano a usare gli SHA congelati e le risorse già approvate.

Main è ora il punto comune del codice e dei documenti; non diventa retroattivamente il codice delle
campagne già avviate. I piani storici rimangono vincolati ai loro sorgenti: riutilizzare main per un futuro
lancio richiederà una consegna con pin e review pertinenti, non la modifica degli hash storici.
