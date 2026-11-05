# Audit di allineamento GPT / Claude — 5 novembre 2026

Verifica di coordinamento richiesta da Simone. Snapshot Git e ledger del 05/11, circa 15:00–15:05 CET;
le linee continuano a lavorare, quindi le punte elencate non sono un congelamento dei loro sviluppi.
Fetch origin riuscito. Nessun merge dei rami sperimentali, job o deploy da questa verifica.

## Esito

Le tre linee GPT hanno separazione effettiva del codice; le review Claude pertinenti sono recepite.
Non emerge un conflitto di implementazione fra le tre linee. Non e' una nuova review scientifica completa,
ne' una certificazione dei contenuti di cache/checkpoint o della validita' dei risultati.
Il disallineamento concreto riguarda istruzioni operative e documentazione distribuita fra rami.

| Numero e linea | Riferimento verificato | Riscontro con Claude |
|---|---|---|
| 1 — Proponi soluzione / scaling 300M | `codex/scaling-300m-20261104`, `26781ee` | Il file della review 6 e' byte per byte identico alla versione Claude `8b76361`, compresa l'aggiunta sulle ultime estrazioni. La presenza del documento non equivale alla copertura di nuovi piani CPU successivi. |
| 2 — Schedule lungo e consolidamento | `codex/consolidation-20261104`, `4f32cdb` | Review FT5 e FT15 identiche alle copie nel ramo Claude `3099836`. La review FT15 permette il seguito dopo le condizioni dichiarate; `02f3de8` registra la chiusura e `1359b92` il benchmark, poi `4f32cdb` i lanci. La rilettura specifica di queste ricevute da Claude resta distinta dall'approvazione del protocollo. |
| 3 — Log quadratico e contrastive loss | `codex/tokenizer-logcontrast-20261104`, `4f8291a` | Il commit Claude `ff08483` e' antenato del ramo GPT: correzioni originali preservate. Review GPT `1dc6ded` chiude R-TIMEOUT e approva solo la diagnostica; `4f8291a` registra il lancio `63591184`. Non e' ancora un via al training. |
| Claude coordinamento | `worktree-claude-training-engine`, `3099836` | Conferma l'operatore GPT. La tabella a questo commit e' uno snapshot precedente: per LQ indica ancora attesa della review/nessun lancio; per scaling precede i piani CPU effettivi. |

## Verifiche Git e registri

- Prima dell'aggiornamento documentale, `main` e `origin/main` coincidevano a `78268d6` (28/10).
  Le campagne successive sono pubblicate sui rami dedicati, ma non ancora consolidate su main.
  La nuova istruzione operativa e' pubblicata separatamente su main a `7616f33`.
- Tutte le punte locali delle linee, dei rami Claude e dei worktree storici coincidevano con i riferimenti origin
  dopo fetch. Nessun commit locale non pubblicato rilevato nello snapshot.
- Worktree puliti, con una sola eccezione in corso: nella chat 1 `docs/reviews/scaling_cpu_release_20261105.md`
  era non tracciato, mentre la chat preparava la consegna. Non modificato o aggiunto da coordinamento.
- Rispetto alla base comune `8ee6603`, i file modificati in comune fra le tre linee GPT con contenuti diversi
  sono solo `docs/claude_GPT_shared`, `docs/decisioni.md` e, fra 1 e 2, `docs/piano_scientifico_20261102.md`.
  Nessun file di codice modificato in comune con contenuti divergenti fra 1, 2 e 3.
- I rami Claude di review non devono necessariamente essere antenati dei rami GPT: verificata l'identita'
  dei tre documenti di review citati sopra, non inferita dal solo nome del commit.
- Le due sezioni del registro compaiono una sola volta in ciascuna delle tre linee e nel ramo comune Claude.
- Ledger comune: 80 ID logici delle famiglie SCALING/CONSOL/FT/FT15/LOGSQ con ID SLURM assegnato;
  nessun ID logico associato a piu' ID SLURM, nessun ID SLURM associato a piu' ID logici.
  Questo verifica le associazioni registrate; non e' una prova universale di assenza di calcolo duplicato fuori ledger.

## Disallineamenti e azioni

1. **Operatore obsoleto in CLAUDE.md: corretto su main (`7616f33`).** Su conferma diretta di Simone,
   «Sì, sottomette GPT» e «semplici9one non c'e piu», GPT sottomette e controlla operativamente stato/log,
   un owner per linea. Citata e conservata testualmente la voce Claude `3099836`; chiuse N-2 (`2d157cc`)
   e N-3 (`c0a4f58`). Invariati submitter/ledger, budget, criteri, review e approvazioni dei singoli test.
2. **main non raccoglie ancora le campagne recenti.** Non usare main come rappresentazione del codice scientifico
   attualmente eseguito: fanno fede SHA, contratti e ricevute delle campagne. L'integrazione dei rami richiede
   una consegna distinta, risoluzione dei documenti condivisi e test dei componenti comuni. Nessun merge massivo
   o aggiornamento dei checkout in uso viene compiuto con questo audit.
3. **Le tabelle GPT/Claude sono snapshot, non stati sincronizzati automaticamente.** Il coordinamento precedente
   GPT e la tabella Claude contengono stati superati; questo audit indica i riferimenti verificati e le differenze.
   Non dedurre la copertura di una review dall'avanzamento del ramo: ogni nuovo pacchetto dichiara la sua copertura.
4. **Proprieta' dei worktree.** I tre sviluppi GPT e la correzione Claude LQ sono in checkout distinti.
   Il documento di coordinamento Claude descrive ancora reviewer che condividono il worktree comune per documenti:
   e' un rischio di scritture concorrenti, non una corruzione osservata. Mantenere un writer alla volta sul ramo comune;
   gli autori delle correzioni restano nei worktree dedicati. Coordinamento GPT non modifica le loro cartelle.

Nessun test di training eseguito: sono verifiche Git, identita' dei documenti, struttura dei registri e ledger.
Le modifiche di questa consegna sono documentali e passano `git diff --check`.
