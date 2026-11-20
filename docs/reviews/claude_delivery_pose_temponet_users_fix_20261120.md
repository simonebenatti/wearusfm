# Consegna Claude: correzione degli utenti del produttore e seconda consegna TCN (20/11/2026)

> **BOZZA, 20/11 ore 12:35: documento NON definitivo. Non usare per `release-plan`.**

Oggetto: la correzione del difetto che ha fermato il preflight `TCNH-preflight-20261119` = 64931573, e la nuova consegna con
ID e checkout nuovi. Ramo `claude/pose-temponet-preflight-users-fix-20261120`, base `31877ad85c619ec69214cd5e9fb53df50e452148`
(punta di `codex/pose-temponet-launcher-20261120`).

**SHA della consegna:**
- codice `5d6fb2dc855bd3c12788bb6d3c3a66043c34f1ba` (sorgenti, test e piano non rilasciato);
- `source_sha256` del piano `37985d28c75bf82a6a784106d414825b727b9ba78575b675dd2dac1d4ccfc8de`, 246 sorgenti;
- piano `results/infra/pose_temponet_execution_20261120/plan.json`, sha256
  `1e8e2b46b046072c706738995e745f608f009d33406d970729a3eb22a73f563e`, 60.341 byte, stato `REVIEW_REQUIRED`.

Incarico. Simone, il 20/11, dopo la mia diagnosi `629fdaed`:
- alla domanda «Autorizzi un nuovo preflight dopo la correzione e la mia rilettura?» ha risposto «Sì, autorizzo»;
- poi mi ha chiesto «fai tu la correzione?».

## Che cosa attesta questo documento, e che cosa no

- **Attesta** che il codice a questo SHA è quello della prima consegna, che ho rivisto e approvato in `85f0d62` (codice
  `01252b6c3023c49046b312e003b7915d8f3f6119`), più il diff descritto sotto. Fra `01252b6c` e questo SHA cambiano due sorgenti
  su 246: `scripts/pose_temponet/provenance.py` e `scripts/pose_temponet/execution.py`.
- **Non è una review indipendente del diff: il diff è mio.** La rilettura indipendente spetta a GPT, come per ogni correzione
  tecnica affidata a me. Questo documento nomina lo SHA completo e il `source_sha256` perché `release-plan` li richiede; il
  rilascio resta subordinato alla rilettura favorevole di GPT.
- **Non attesta** l'esito dei gate sui dati veri, né tempi, memoria e determinismo su Leonardo.

## Il diff

`git diff 31877ad85c619ec69214cd5e9fb53df50e452148 5d6fb2dc855bd3c12788bb6d3c3a66043c34f1ba`

| File | Che cosa cambia |
| --- | --- |
| `scripts/pose_temponet/provenance.py` | `storage_directory` e `cache_user`; `historical_producer` restituisce gli ID della cache in `users` e le cartelle in `storage_directories`; `read_verified_inputs` verifica entrambi |
| `scripts/pose_temponet/execution.py` | `DELIVERY = "20261120"` per campagna e suffisso degli ID |
| `tests/cpu/test_pose_temponet_provenance.py` | quattro test nuovi |
| `tests/cpu/test_pose_temponet_execution.py` | un test nuovo sull'identità della consegna |
| `tests/cpu/data/pose_temponet_cache_users_20261120.json` | gli ID veri delle due cache |
| `results/infra/pose_temponet_execution_20261120/` | piano rigenerato, validazione, prove sui file veri |
| `docs/pose_temponet_execution_20261120.md` | istruzioni aggiornate per l'operatore |

Perché alla fonte. La lista `plan["producers"][split]["users"]` è usata anche da `benchmark._validate_train`,
`controls.validate_outer_partition` e dai fold di `workflow`. Correggere solo il confronto del preflight avrebbe spostato il fermo
al benchmark.

Perché il ritorno dalla cartella all'ID è sicuro. Il produttore scrive la cartella come `"u" + _SAFE_RE.sub("_", ID)`. Ogni
carattere sostituito diventa `_`. `cache_user` accetta quindi solo cartelle `u` + caratteri sicuri senza `_`: per queste l'ID
possibile è uno solo. Tutto il resto è rifiutato alla costruzione del piano. I 55 ID veri hanno 10 cifre esadecimali.

Che cosa resta uguale: proposta, ricetta, soglie, tetti, otto input pinnati, grafo dei 24 job (a parte il suffisso degli ID),
launcher, radice degli output.
