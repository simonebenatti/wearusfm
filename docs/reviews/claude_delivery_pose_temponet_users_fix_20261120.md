# Consegna Claude: correzione degli utenti del produttore e seconda consegna TCN (20/11/2026)

Oggetto: la correzione del difetto che ha fermato il preflight `TCNH-preflight-20261119` = 64931573, e la nuova consegna con
ID e checkout nuovi. Ramo `claude/pose-temponet-preflight-users-fix-20261120`, base `31877ad85c619ec69214cd5e9fb53df50e452148`
(punta di `codex/pose-temponet-launcher-20261120`).

**SHA della consegna:**
- codice `5d6fb2dc855bd3c12788bb6d3c3a66043c34f1ba`: è il commit che contiene i sorgenti definitivi. I commit successivi del
  ramo aggiungono solo test, documenti e prove, senza toccare `src/` e `scripts/`;
- `source_sha256` del piano `37985d28c75bf82a6a784106d414825b727b9ba78575b675dd2dac1d4ccfc8de`, 246 sorgenti;
- piano `results/infra/pose_temponet_execution_20261120/plan.json`, sha256
  `1e8e2b46b046072c706738995e745f608f009d33406d970729a3eb22a73f563e`, 60.341 byte, stato `REVIEW_REQUIRED`.

Incarico. Simone, il 20/11, dopo la mia diagnosi `629fdaed`:
- alla domanda «Autorizzi un nuovo preflight dopo la correzione e la mia rilettura?» ha risposto «Sì, autorizzo»;
- poi mi ha chiesto «fai tu la correzione?»;
- quando ho scritto che il diff è mio e che la rilettura indipendente spetta a GPT, ha risposto «ok confermo, la rilettura la fa
  gpt».

## Che cosa attesta questo documento, e che cosa no

- **Attesta** che il codice a questo SHA è quello della prima consegna, che ho rivisto e approvato in `85f0d62` (codice
  `01252b6c3023c49046b312e003b7915d8f3f6119`), più il diff descritto sotto. Fra `01252b6c` e questo SHA cambiano due sorgenti
  su 246: `scripts/pose_temponet/provenance.py` e `scripts/pose_temponet/execution.py`.
- **Non è una review indipendente del diff: il diff è mio.** La rilettura indipendente è di GPT. Questo documento nomina lo SHA
  completo e il `source_sha256` perché `release-plan` li richiede; il rilascio resta subordinato alla rilettura favorevole di GPT.
- **Non attesta** l'esito dei gate sui dati veri, né tempi, memoria e determinismo su Leonardo.

## Il diff

`git diff 31877ad85c619ec69214cd5e9fb53df50e452148 <punta del ramo>`

| File | Che cosa cambia |
| --- | --- |
| `scripts/pose_temponet/provenance.py` | `storage_directory` e `cache_user`; `historical_producer` restituisce gli ID della cache in `users` e le cartelle in `storage_directories`; `read_verified_inputs` verifica entrambi |
| `scripts/pose_temponet/execution.py` | `DELIVERY = "20261120"` per campagna e suffisso degli ID |
| `tests/cpu/test_pose_temponet_provenance.py` | quattro test nuovi |
| `tests/cpu/test_pose_temponet_execution.py` | un test nuovo sull'identità della consegna |
| `tests/cpu/test_pose_temponet_preflight_real_layout.py` | tre test nuovi: il corpo vero del preflight su file in formato reale |
| `tests/cpu/data/pose_temponet_cache_users_20261120.json`, `…cache_layout_20261120.json` | ID, configurazione e chiavi dei metadati delle due cache vere |
| `results/infra/pose_temponet_execution_20261120/` | piano rigenerato, validazione, mutazioni, prove sui file veri, review indipendente |
| `docs/pose_temponet_execution_20261120.md` | istruzioni aggiornate per l'operatore; un rimando in testa al documento del 19/11 |

Perché alla fonte. La lista `plan["producers"][split]["users"]` è usata anche da `benchmark._validate_train`,
`controls.validate_outer_partition` e dai fold di `workflow`. Correggere solo il confronto del preflight avrebbe spostato il fermo
al benchmark.

Perché il ritorno dalla cartella all'ID è sicuro. Il produttore scrive la cartella come `"u" + _SAFE_RE.sub("_", ID)`. Ogni
carattere sostituito diventa `_`. `cache_user` accetta quindi solo cartelle `u` + caratteri sicuri senza `_`: per queste l'ID
possibile è uno solo. Tutto il resto è rifiutato alla costruzione del piano. I 55 ID veri hanno 10 cifre esadecimali.

Che cosa resta uguale: proposta, ricetta, soglie, tetti, otto input pinnati, grafo dei 24 job (a parte il suffisso degli ID),
launcher, radice degli output.

## Verifiche

Test, sul Mac (`results/infra/pose_temponet_execution_20261120/validation/local_validation.json`):
- i 21 file TCN: 271 passati su 271, in 162 s;
- i 5 file storici: 125 passati e 1 fallito, lo stesso fallimento già presente sulla base
  (`test_concrete_campaign_remains_review_gated_and_all_endpoints_pinned`);
- `plan.json` è esattamente l'uscita di `prepare-plan` sui sorgenti.

La suite completa del repository non è stata rieseguita: fuori dal pacchetto TCN non cambia nessun sorgente, e i tre
launcher sono identici a quelli già approvati.

Mutazioni (`mutation_check/`): otto modifiche di prova della correzione, per esempio il ritorno al comportamento della prima
consegna o la rimozione di uno dei due confronti. Ognuna fa fallire almeno un test nuovo.

Dati veri, in sola lettura su Leonardo (`real_input_headers/`, 93 controlli su 93):
- gli ID delle due cache sono uguali alla lista che il codice nuovo ricava dalle due ricevute; i miei elenchi hanno gli stessi
  hash della lettura indipendente dell'operatore;
- forme, tipi e configurazione dei sette `.npz` pinnati corrispondono a quello che il lettore pretende;
- i metadati delle finestre sono unici e il loro hash è uguale a quello delle ricevute storiche;
- le prime finestre dei 15 utenti usati dal benchmark CPU hanno posa valida.

Review indipendente del commit `5d6fb2dc` (`independent_review/`), quattro revisori senza accesso al cluster:
- LOGIC, concluso: nessun blocco e nessun importante. Conferma l'unicità del ritorno cartella → ID (nessun controesempio su
  25.260 stringhe esaustive e 200.000 casuali), la coerenza di tutti i punti che usano `producers`, il rifiuto dei piani vecchi,
  l'ordine degli import e il contesto numerico invariati. Ha eseguito il corpo vero del preflight su file in formato reale: è
  diventato il test `test_pose_temponet_preflight_real_layout.py`.
- DELIVERY, concluso: nessun blocco. Ha provato in locale rilascio, generazione del comando e i due wrapper, anche con i resti
  del primo tentativo nella radice degli output. Due punti importanti e sei fra minori e note, tutti di documentazione o
  procedura: applicati in `docs/pose_temponet_execution_20261120.md`.
- REAL-DATA e TESTS: fermati prima della fine, quando Simone ha chiesto la pausa. Non hanno un esito. Ho sostituito il loro
  lavoro con le prove di mutazione e con le letture sui dati veri; del primo restano gli script in `partial_realdata/`.

## Per l'operatore

Sequenza completa in `docs/pose_temponet_execution_20261120.md`. In breve:
1. GPT rilegge il diff e scrive l'esito nel registro.
2. La punta di questo ramo entra nel ramo dell'operatore con un avanzamento semplice o un merge, mai cherry-pick, rebase o squash.
3. `release-plan` con `--reviewed-code-sha 5d6fb2dc855bd3c12788bb6d3c3a66043c34f1ba`. Come review si può passare questo file
   (commit: la punta del ramo) oppure il file della rilettura di GPT, se cita per intero lo SHA o il `source_sha256`.
4. Deploy con riferimento nuovo, controllo locale in sei righe prima dell'invio, poi il preflight `TCNH-preflight-20261120`.

## Limiti

- I digest degli array e la parità R-1 a 1e-12 si vedono solo nel job, come tempo e memoria della lettura di val.
- `benchmark._report_measurement` usa la prima finestra dei primi 15 utenti: sui dati veri ha posa valida, ma il codice non lo
  garantisce da solo.
- `validate_plan` non lega gli ID dei job alla consegna: nessun comando produce un piano ibrido, e l'ID vecchio sarebbe fermato
  all'esecuzione.
