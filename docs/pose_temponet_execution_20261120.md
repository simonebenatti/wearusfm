# TCN frozen: seconda consegna eseguibile del 20/11/2026

Questo documento aggiorna `docs/pose_temponet_execution_20261119.md`. Dove non dice altro, le istruzioni del 19/11 restano
valide: sequenza, gate, tetti, fallimenti e chiusura. I punti del 19/11 che cambiano sono elencati qui.

## Perché una seconda consegna

Il primo preflight, `TCNH-preflight-20261119` = 64931573, è finito `FAILED 1:0` dopo 1 min 13 s con
`cache user population differs from original producer`. Non c'era una differenza di utenti: era un errore di formato.

- La cache scrive l'ID dell'utente, per esempio `03d49b393e`.
- `historical_producer` ricavava la lista dai nomi delle cartelle dei file processati, che il produttore crea come `u` + ID,
  per esempio `u03d49b393e`.

Le due popolazioni coincidono una a una: 40 su 40 in train, 15 su 15 in val. La diagnosi è nella nota di Claude
`docs/reviews/claude_note_pose_temponet_preflight_users_20261120.md` (commit `629fdaed`) e nella conferma indipendente
dell'operatore (`results/infra/pose_temponet_execution_20261119/launcher_preflight_20261120/user_format_diagnosis.json`).

Decisioni di Simone del 20/11, alla lettera:
- alla domanda di Claude «Autorizzi un nuovo preflight dopo la correzione e la mia rilettura?»: «Sì, autorizzo»;
- poi, a Claude: «fai tu la correzione?»;
- dopo che Claude ha scritto che il diff è suo e che la rilettura indipendente spetta a GPT: «ok confermo, la rilettura la fa gpt».

La correzione è quindi di Claude, per incarico esplicito, e la rilettura del diff è di GPT. Il nuovo tentativo del preflight è
la decisione richiesta da L3-9.

## Che cosa cambia

| Voce | Prima consegna | Seconda consegna |
| --- | --- | --- |
| Utenti del produttore nel piano | nomi delle cartelle, `u` + ID | ID, come nella cache; le cartelle restano in `storage_directories` |
| Suffisso degli ID logici | `-20261119` | `-20261120` |
| Campagna e checkout | `pose_temponet_20261119` | `pose_temponet_20261120` |
| Piano da rilasciare | `results/infra/pose_temponet_execution_20261119/plan.json` | `results/infra/pose_temponet_execution_20261120/plan.json` |
| Piano rilasciato (consigliato) | `…_20261119/plan_released_20261120.json` | `results/infra/pose_temponet_execution_20261120/plan_released.json` |
| Cartella delle prove dell'operatore (consigliata) | `…_20261119/launcher_preflight_20261120/` | `results/infra/pose_temponet_execution_20261120/launcher/` |
| Riferimento di deploy (consigliato) | `deploy/pose-temponet-20261119` | `deploy/pose-temponet-20261120` |

Attenzione ai nomi: la data `20261120` compare già in tre oggetti del primo tentativo, cioè il vecchio piano rilasciato, la
cartella `launcher_preflight_20261120` e i due ID di inventario `TCNH-CPU-inventory-login08/13-20261120`. Per la seconda
consegna fa fede la cartella `results/infra/pose_temponet_execution_20261120/`.

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

Il checkout `pose_temponet_20261119`, il piano rilasciato `e3f5b119…` e gli ID con suffisso `-20261119` non si riusano. I comandi
d'esempio del 19/11 che leggono il vecchio piano non funzionano più con i sorgenti nuovi, che lo rifiutano.

## Sequenza dell'operatore

È quella del documento del 19/11, con questi valori e con un controllo in più prima dell'invio.

1. **Rilettura del diff.** La fa GPT: `git diff 31877ad85c619ec69214cd5e9fb53df50e452148 <punta del ramo di Claude>`. L'esito va
   scritto nel registro.
2. **Portare il lavoro nel ramo dell'operatore** con un avanzamento semplice o un merge della punta di
   `claude/pose-temponet-preflight-users-fix-20261120`. Mai cherry-pick, rebase o squash: `checkout_identity` pretende che il
   commit di deploy discenda dal codice rivisto, e `release-plan` confronta i sorgenti byte per byte con quello SHA.
3. **Evidenza operatore nuova**, in un file nuovo dentro `results/infra/pose_temponet_execution_20261120/`, con `observed_at`
   aggiornato. `release-plan` non controlla che sia nuova: il file del primo rilascio verrebbe accettato, quindi non va riusato.
   L'evidenza sul modello di CPU dei job 64931087 e 64931101 (stesso modello sui due nodi seriali) è del 20/11 mattina e si può
   citare: non servono nuovi job di inventario.
4. **Rilasciare il piano:**

   ```bash
   python -m pose_temponet release-plan --plan results/infra/pose_temponet_execution_20261120/plan.json --review-commit "$TCN_REVIEW_SHA" --review-path "$TCN_REVIEW_PATH" --reviewed-code-sha "$TCN_CODE_SHA" --operator-checks "$TCN_OPERATOR_CHECKS" --output results/infra/pose_temponet_execution_20261120/plan_released.json
   ```

   - `$TCN_CODE_SHA` è lo SHA del codice scritto per intero nel documento di consegna di Claude,
     `docs/reviews/claude_delivery_pose_temponet_users_fix_20261120.md`.
   - Come `--review-commit` e `--review-path` si passa un file che contenga per intero lo SHA del codice (40 cifre) oppure il
     `source_sha256` (64 cifre). Va bene il file della rilettura di GPT, se li cita per intero. Va bene anche il documento di
     consegna di Claude, che li cita entrambi: in quel caso l'esito della rilettura di GPT resta nel registro.
   - Il piano rilasciato porterà comunque l'etichetta `claude_review_converged`: è un nome fisso dello schema, non dice chi ha
     riletto. Chi ha scritto e chi ha riletto è registrato qui, nel documento di consegna e nel registro.
5. **Commit e push** della consegna di esecuzione, verifica dello SHA remoto, pubblicazione del riferimento di deploy nuovo. Il
   checkout nascerà in `$WORK/wearusfm_campaigns/pose_temponet_20261120`, che oggi non esiste.
6. **Controllo locale prima dell'invio, in sola lettura.** L'invio del preflight è unico: se il commit di deploy è sbagliato il
   job lo rifiuta solo dopo aver creato il checkout e consumato l'ID. Con `C` = `$TCN_CODE_SHA` e `D` = commit di deploy
   (lo stesso valore di `--checkout-sha` e `--plan-commit`):

   ```bash
   git merge-base --is-ancestor "$C" "$D" && echo "il codice rivisto è un antenato del deploy"
   git diff --quiet "$C" "$D" -- src scripts && echo "nessun sorgente cambiato dopo il codice rivisto"
   [ "$(git show "$D:results/infra/pose_temponet_execution_20261120/plan_released.json" | shasum -a 256 | cut -d' ' -f1)" = "$TCN_PLAN_SHA256" ] && echo "il piano rilasciato nel commit ha l'hash atteso"
   git cat-file -e "$D:$TCN_REVIEW_PATH" && echo "la copia della review è nel commit"
   [ -z "$(git status --porcelain)" ] && echo "albero pulito"
   [ "$(git ls-remote leonardo deploy/pose-temponet-20261120 | cut -f1)" = "$D" ] && echo "il riferimento di deploy punta a D"
   ```

   Devono comparire tutte e sei le righe.
7. **Generare la riga del preflight** con `commands --stage preflight` e inviarla una sola volta con `scripts/submit_job.py`.
   L'ID è `TCNH-preflight-20261120`.

Quattro precisazioni sulle fasi successive:
- Le copie locali delle ricevute per `resource-plan` e per `--completed-prerequisites` vanno raccolte in una sola cartella con il
  nome `<ID>.json`, per esempio `TCNH-preflight-20261120.json` accanto alle due dei benchmark. `verified_receipt` cerca ogni
  predecessore con quel nome. Dopo il primo tentativo la ricevuta era stata archiviata come `preflight_receipt.json`: una copia
  con un altro nome può restare, ma non sostituisce quella.
- Il consuntivo finale conta anche il primo tentativo: 24 allocazioni della catena più le 3 già fatte (due inventari e il
  preflight fallito), cioè 27, e le 0,0433 core-ore già spese vanno negli stessi tetti. Zero GPU-ore finora.
- Il pin numerico `gpu8` nasce dal benchmark GPU e comprende nome, capacità e memoria della scheda e il modello di CPU di quel
  nodo. Ogni stadio GPU successivo pretende lo stesso digest. Prima del primo invio dipendente conviene leggere gli stessi valori
  su un secondo nodo di `boost_usr_prod`, come già fatto per i due nodi seriali: se differissero vale la regola per classe di
  nodo del documento del 19/11. Finora nessuno l'ha verificato.
- Tre degli otto input pinnati stanno su SCRATCH (i due riferimenti dei target e `pose_stats.json`). Come già scritto il 19/11,
  nessun rinfresco di SCRATCH mentre un job TCN legge gli input: il prossimo è previsto il 03/12.

## Verifiche fatte sul Mac

- Tutti i file di test TCN, ora 21, e i 5 file storici: ricevuta in
  `results/infra/pose_temponet_execution_20261120/validation/local_validation.json`. La stessa esecuzione verifica che `plan.json`
  sia esattamente ciò che `prepare-plan` ricava dai sorgenti.
- Test nuovi:
  - la lista del produttore, ricavata dalle due ricevute vere, è uguale agli ID letti nelle due cache vere
    (`tests/cpu/data/pose_temponet_cache_users_20261120.json`, con gli stessi hash della lettura indipendente dell'operatore);
  - l'errore del preflight 64931573, cioè nomi di cartelle offerti come ID, è rifiutato;
  - una cartella con `_` o senza `u` è rifiutata già alla costruzione del piano;
  - gli ID della seconda consegna non coincidono con quelli già presenti nel ledger;
  - `tests/cpu/test_pose_temponet_preflight_real_layout.py` esegue il corpo vero di `provenance.preflight`, senza sostituire
    niente, su file piccoli scritti nel formato reale: ID veri, ricevute vere con le loro cartelle, `config_json` vero, chiavi dei
    metadati vere, scrittore e lettori storici. Con la lista della prima consegna riproduce lo stesso errore del job 64931573.
    Lo stesso file fa leggere quei dati a ogni stadio successivo con il lettore vero e verifica che benchmark, calibrazione e
    fold accettino la lista del piano e rifiutino i nomi delle cartelle;
  - la riga di invio del preflight nomina la campagna nuova in `--campagna`, in `CAMPAIGN` e in `CODE_DIR`;
  - le costanti della parità R-1 sono quelle della proposta approvata e la tolleranza è esattamente 1e-12;
  - un piano con le liste riscritte è rifiutato da `validate_plan`, anche con l'identità di esecuzione ricalcolata.
- Otto mutazioni di prova della correzione: ciascuna fa fallire almeno un test nuovo
  (`results/infra/pose_temponet_execution_20261120/mutation_check/`). Il revisore dei test ne ha provate altre 37: i test che ha
  proposto per quelle sopravvissute sono stati aggiunti.
- Letture in sola lettura su Leonardo dei file pinnati, senza aprire gli array interi
  (`results/infra/pose_temponet_execution_20261120/real_input_headers/`, 93 controlli su 93):
  - forme, tipi e configurazione dei sette `.npz` corrispondono a quello che il codice pretende: cache `float16` 160 × 384,
    target `float32` 160 × 20, predizioni storiche `float64` 18.548 × 148 × 20, finestre 10 × 4 s, seme 20261026, teacher a 49.046;
  - i metadati delle finestre sono tutti unici, 1.907 in train e 18.548 in val, come pretendono controlli, selezioni e rifit;
  - l'hash dei metadati veri è uguale a `details.targets.metadata` delle due ricevute storiche;
  - in train 53 finestre su 1.907 non hanno posa valida. Le prime finestre dei primi 15 utenti in ordine, che il benchmark CPU
    usa per misurare il costo del rapporto, hanno tutte posa valida su 20 giunti.
- Review indipendente (`results/infra/pose_temponet_execution_20261120/independent_review/`): quattro revisori, tutti conclusi,
  nessun blocco e nessuna richiesta sui sorgenti. Due hanno lavorato sul commit `5d6fb2dc`, due sulla punta `0e548bf8` dopo
  essere stati rilanciati. I loro punti sui documenti e sui test sono applicati. Per il preflight, il revisore dei dati veri ha
  trovato che:
  - il gate R-1 usa le stesse funzioni dello stesso modulo, identico byte per byte, del job storico, e in simulazione alla
    dimensione reale di val dà un risultato identico bit per bit;
  - la lettura di val arriva a circa 9 GiB di picco e l'intero preflight dovrebbe durare circa 3 minuti.

## Limiti noti, che non cambiano con questa consegna

- `benchmark._report_measurement` prende la prima finestra di ciascuno dei primi 15 utenti. Sui dati veri quelle finestre hanno
  posa valida; il codice però non lo garantisce da solo. L'utente `76bfc2d385`, diciannovesimo in ordine, ha la prima finestra
  senza posa: non entra nella misura.
- `validate_plan` non lega gli ID dei job alla consegna: un piano costruito a mano con i sorgenti nuovi e gli ID vecchi
  passerebbe la validazione. Nessun comando produce un piano così, e l'ID vecchio verrebbe fermato all'esecuzione.

## Che cosa resta da vedere solo nel job

- I digest degli array delle cache e la parità R-1 a 1e-12 sono calcolati sui dati interi. Sul Mac non si possono riprodurre: li
  decide il preflight. Nessun test locale esegue la parità sugli array veri.
- Tempo e memoria della lettura di val nel preflight: mai misurati sui dati veri in questa catena. La stima del revisore, circa
  9 GiB e circa 3 minuti, sta larga nei 24 GB e nei 12 minuti richiesti; il job storico con lo stesso lettore sullo stesso file
  misurò 8,86 GiB.
- Tempi, memoria e determinismo su A100: li decidono i benchmark e i gate, come nella prima consegna.
