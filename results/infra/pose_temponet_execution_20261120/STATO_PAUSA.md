# Stato alla pausa del 20/11/2026, ore 12:35

Simone: «adesso mi devo staccare. puoi metterti in pausa cosi non perdo tutto il lavoro e i token fatti finora?».
Lavoro fermato qui. Questo file dice che cosa è fatto, che cosa manca e come riprendere. Va tolto alla consegna.

## Decisioni di Simone, alla lettera

- Alla domanda «Autorizzi un nuovo preflight dopo la correzione e la mia rilettura?»: «Sì, autorizzo».
- Poi: «fai tu la correzione?».
- Dopo che ho scritto che il diff è mio e che la rilettura indipendente spetta a GPT: «ok confermo, la rilettura la fa gpt».

## Fatto

- Commit dei contenuti `5d6fb2dc855bd3c12788bb6d3c3a66043c34f1ba` sul ramo `claude/pose-temponet-preflight-users-fix-20261120`,
  base `31877ad85c619ec69214cd5e9fb53df50e452148`: correzione in `provenance.py`, nuova identità in `execution.py`, cinque test
  nuovi, fixture con gli ID veri, piano rigenerato (`source_sha256` `37985d28c75bf82a6a784106d414825b727b9ba78575b675dd2dac1d4ccfc8de`,
  piano sha256 `1e8e2b46b046072c706738995e745f608f009d33406d970729a3eb22a73f563e`).
- Test: 268 passati su 268 nei 20 file TCN (155 s), prima del commit.
- Prove sui file veri, in sola lettura (`real_input_headers/`): 89 controlli su 89.
- Review indipendente (workflow `wf_311a1ce8-da4`), risultati in `independent_review/`:
  - LOGIC: finito. Nessun blocco, nessun importante. Tre minori e una nota (L-1...L-4).
  - DELIVERY: finito. Nessun blocco. Due importanti (D-1, D-2), tre minori (D-3...D-5), tre note (D-6...D-8). Tutti di
    documentazione o procedura: nessuno chiede di cambiare i sorgenti.
  - REAL-DATA e TESTS: vedi sotto «Revisori non conclusi».

## Da fare alla ripresa (nessun sorgente da toccare, salvo sorprese dai due revisori mancanti)

1. Leggere REAL-DATA e TESTS (o rilanciarli, vedi sotto). Se chiedono un cambio dei sorgenti: nuovo commit, rigenerare
   `plan.json` con `prepare-plan`, aggiornare SHA e hash in tutti i documenti.
2. Documento `docs/pose_temponet_execution_20261120.md`:
   - D-2: fra i passi 3 e 4, controllo locale in sola lettura sul commit di deploy `D` rispetto al codice `C`:
     `git merge-base --is-ancestor "$C" "$D"`; `git diff --quiet "$C" "$D" -- src scripts`; hash del piano rilasciato dentro `D`;
     `git cat-file -e "$D:$TCN_REVIEW_PATH"`; albero pulito; `git ls-remote leonardo` del riferimento di deploy uguale a `D`.
     Dire che `5d6fb2dc` deve entrare nel ramo dell'operatore con un merge o un avanzamento semplice, mai cherry-pick, rebase o squash.
   - D-3: dire quale file va passato a `--review-commit/--review-path`, che deve contenere lo SHA a 40 cifre o il
     `source_sha256` a 64, e che `claude_review_converged` è un'etichetta fissa dello schema.
   - D-4: l'evidenza sul modello di CPU dei job 64931087 e 64931101 resta valida, senza nuovi inventari; il consuntivo finale
     conta 27 allocazioni e addebita le 0,0433 core-ore del primo tentativo agli stessi tetti.
   - D-5: le copie locali delle ricevute vanno raccolte in una sola cartella come `<ID>.json`.
   - D-7: nominare il percorso del nuovo piano rilasciato e della cartella dell'operatore
     (per esempio `results/infra/pose_temponet_execution_20261120/plan_released.json` e `launcher_preflight/`).
   - D-8: l'evidenza operatore nuova va scritta in un file nuovo nella cartella `20261120`.
3. `validate.py`, L-2: accettare il gruppo storico solo con (codice 0 e nessun fallimento) oppure (codice 1, il solo fallimento
   noto e nessuna riga `ERROR `). Poi eseguirlo:
   `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ~/.venvs/wearusfm-tok/bin/python results/infra/pose_temponet_execution_20261120/validate.py --out results/infra/pose_temponet_execution_20261120/validation`
   (circa 3 minuti; chiude L-1 e metà di D-1).
4. L-3, facoltativo: portare in un test CPU la prova del revisore LOGIC, che esegue il corpo vero del preflight su file piccoli
   in formato reale (`independent_review/review_logic_preflight_dryrun.py`). È solo un test: non cambia `source_sha256`.
5. D-6: registrare lo scambio con Simone e la decisione L3-9 nel registro (due sezioni) e in `docs/decisioni.md`.
6. Completare `docs/reviews/claude_delivery_pose_temponet_users_fix_20261120.md` (verifiche, esito dei revisori, istruzioni per
   `release-plan`, prompt per GPT), togliere i due avvisi «BOZZA» e «IN LAVORAZIONE», scrivere `delivery_pin.json`.
7. Ricontrollare la punta di GPT (`git ls-remote origin refs/heads/codex/pose-temponet-launcher-20261120`, era `31877ad8`),
   `inspect` del piano (deve dare `37985d28…`), commit, push con verifica dello SHA remoto.
8. Dare a Simone il testo per GPT: rilettura del diff, poi `release-plan`, deploy, preflight `TCNH-preflight-20261120`.

## Revisori non conclusi

Alle 12:33 REAL-DATA e TESTS erano ancora al lavoro dopo circa 43 minuti. Lo stato finale è scritto in fondo a questo file.

- Trascrizioni (restano su disco anche se fermati):
  `~/.claude/projects/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/subagents/workflows/wf_311a1ce8-da4/`
  - REAL-DATA: `agent-aaa94ae289c19328e.jsonl`
  - TESTS: `agent-a77fbb3e70c0e0c2b.jsonl`
- Per rilanciare solo i mancanti, nella stessa sessione: `Workflow` con lo script
  `~/.claude/projects/-Users-simonebenatti-dev-wearusfm/d9aa6af4-72b7-4d2a-8229-4444e8ca2c8f/workflows/scripts/tcn-users-fix-review-wf_311a1ce8-da4.js`
  e `resumeFromRunId: wf_311a1ce8-da4`. LOGIC e DELIVERY tornano dalla cache.
- Molto del lavoro di REAL-DATA l'ho già fatto io in sola lettura: intestazioni, configurazioni, unicità dei metadati, hash dei
  metadati uguale alla ricevuta. Resta suo il conto di memoria e tempo del preflight su val e la lettura della parità R-1.

## Che cosa non è stato fatto

Nessun job, nessun deploy, nessuna scrittura sul cluster. Solo letture con `sacct`, `ls` e `zipfile` sui file pinnati.
