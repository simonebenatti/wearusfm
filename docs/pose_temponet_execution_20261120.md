# TCN frozen: seconda consegna eseguibile del 20/11/2026

Questo documento aggiorna `docs/pose_temponet_execution_20261119.md`. Dove non dice altro, le istruzioni del 19/11 restano
valide parola per parola: sequenza, gate, tetti, fallimenti e chiusura.

## Perché una seconda consegna

Il primo preflight, `TCNH-preflight-20261119` = 64931573, è finito `FAILED 1:0` dopo 1 min 13 s con
`cache user population differs from original producer`. Non c'era una differenza di utenti: era un errore di formato.

- La cache scrive l'ID dell'utente, per esempio `03d49b393e`.
- `historical_producer` ricavava la lista dai nomi delle cartelle dei file processati, che il produttore crea come `u` + ID,
  per esempio `u03d49b393e`.

Le due popolazioni coincidono una a una: 40 su 40 in train, 15 su 15 in val. La diagnosi è nella nota di Claude
`docs/reviews/claude_note_pose_temponet_preflight_users_20261120.md` (commit `629fdaed`) e nella conferma indipendente
dell'operatore (`results/infra/pose_temponet_execution_20261119/launcher_preflight_20261120/user_format_diagnosis.json`).

Simone, il 20/11:
- alla domanda «Autorizzi un nuovo preflight dopo la correzione e la mia rilettura?» ha risposto «Sì, autorizzo»;
- poi ha chiesto a Claude «fai tu la correzione?».

La correzione è quindi di Claude, per incarico esplicito. La rilettura indipendente del diff spetta a GPT.

## Che cosa cambia

| Voce | Prima consegna | Seconda consegna |
| --- | --- | --- |
| Utenti del produttore nel piano | nomi delle cartelle, `u` + ID | ID, come nella cache; le cartelle restano in `storage_directories` |
| Suffisso degli ID logici | `-20261119` | `-20261120` |
| Campagna e checkout | `pose_temponet_20261119` | `pose_temponet_20261120` |
| Piano da rilasciare | `results/infra/pose_temponet_execution_20261119/plan.json` | `results/infra/pose_temponet_execution_20261120/plan.json` |
| Riferimento di deploy consigliato | `deploy/pose-temponet-20261119` | `deploy/pose-temponet-20261120` |

Sorgenti cambiati: due su 246.

- `scripts/pose_temponet/provenance.py`:
  - `storage_directory(user)` applica la regola del produttore, `"u" + _SAFE_RE.sub("_", user)`, con lo stesso `_SAFE_RE` di
    `wearusfm.harness.pose_eval`;
  - `cache_user(directory)` torna dalla cartella all'ID solo quando l'ID è unico. Ogni carattere sostituito diventa `_`, quindi
    una cartella senza `_` ha un solo ID possibile. Qualsiasi altro nome è rifiutato;
  - `historical_producer` restituisce `users` nel formato della cache e conserva le cartelle in `storage_directories`;
  - `read_verified_inputs` confronta gli utenti della cache con `users`, come prima, e in più verifica che gli stessi utenti,
    passati per la regola del produttore, diano esattamente `storage_directories`.
- `scripts/pose_temponet/execution.py`: costante `DELIVERY = "20261120"`, usata dal nome della campagna e dal suffisso degli ID.

La lista corretta alla fonte arriva nello stesso formato a tutti i punti che la usano: `benchmark._validate_train`,
`controls.validate_outer_partition` (calibrazione WLS e controlli) e i fold di `workflow`.

## Che cosa non cambia

- Proposta scientifica, ricetta, soglie, tetti e gli otto input pinnati: `recipe_sha256`, `inputs` e `proposal` del piano sono
  identici a quelli della prima consegna.
- Il grafo dei 24 job: identico, a parte il suffisso degli ID.
- I tre launcher sotto `scripts/slurm/`.
- La radice degli output, `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/pose_temponet_20261117`. Letta il 20/11:
  contiene solo `manifests/e3f5b119….json` e `receipts/TCNH-preflight-20261119.json` del primo tentativo. La cartella
  `preflight/` non esiste, quindi nessuna destinazione viene riusata. La ricevuta fallita resta come traccia e lo stadio di
  raccolta la inventaria insieme al resto.

Il checkout `pose_temponet_20261119`, il piano rilasciato `e3f5b119…` e gli ID con suffisso `-20261119` non si riusano.

## Sequenza dell'operatore

È quella del documento del 19/11, con questi valori.

1. Leggere l'esito della rilettura del diff e preparare un'evidenza operatore nuova (`observed_at` aggiornato).
2. Rilasciare il piano:

   ```bash
   python -m pose_temponet release-plan --plan results/infra/pose_temponet_execution_20261120/plan.json --review-commit "$TCN_REVIEW_SHA" --review-path "$TCN_REVIEW_PATH" --reviewed-code-sha "$TCN_CODE_SHA" --operator-checks "$TCN_OPERATOR_CHECKS" --output "$TCN_RELEASED_PLAN"
   ```

   `$TCN_CODE_SHA` e il `source_sha256` sono nel documento di consegna di Claude, che li nomina per intero.
3. Committare e pushare, verificare lo SHA remoto, pubblicare un riferimento di deploy nuovo. Il checkout nasce in
   `$WORK/wearusfm_campaigns/pose_temponet_20261120`, che oggi non esiste.
4. Generare la riga del preflight con `commands --stage preflight` e inviarla una sola volta con `scripts/submit_job.py`.
   L'ID è `TCNH-preflight-20261120`.

## Verifiche fatte sul Mac

- I 20 file di test TCN e i 5 file storici: ricevuta in `results/infra/pose_temponet_execution_20261120/validation/`.
- Test nuovi:
  - la lista del produttore, ricavata dalle due ricevute vere, è uguale agli ID letti nelle due cache vere
    (`tests/cpu/data/pose_temponet_cache_users_20261120.json`, con gli stessi hash della lettura indipendente dell'operatore);
  - l'errore del preflight 64931573, cioè nomi di cartelle offerti come ID, è rifiutato;
  - una cartella con `_` o senza `u` è rifiutata già alla costruzione del piano;
  - gli ID della seconda consegna non coincidono con quelli già presenti nel ledger.
- Lettura in sola lettura, su Leonardo, delle intestazioni dei sette file `.npz` pinnati e della loro configurazione. Forme, tipi
  e campi corrispondono a quello che il codice pretende: cache `float16` 160 × 384, target `float32` 160 × 20, predizioni
  storiche `float64` 18.548 × 148 × 20, configurazione di train e val con finestre 10 × 4 s, seme 20261026, teacher a 49.046.

## Che cosa resta da vedere solo nel job

- I sei dettagli delle cache e la parità R-1 a 1e-12 sono calcolati sui dati interi. Sul Mac non si possono riprodurre: li
  decide il preflight.
- Tempi, memoria e determinismo su A100: li decidono i benchmark e i gate, come nella prima consegna.
