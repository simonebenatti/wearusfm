# Rilancio unico dell'estrazione p1 a 19.618 (16/11/2026)

Approvazione: Simone, 16/11/2026, alla domanda «Approvi il rilancio unico di p1 senza review di GPT?»: «Sì, approvo».
E' l'opzione A di `../continuation19618_20261106/failure_64572000_diagnosis_20261116/README.md`
(sha256 `6d85702b4758e268f653ee1962b27590e7934f16f3ed71708cc9c6cc66669159`):
stessa sottomissione del job sostituito, con un ID logico nuovo e una cartella d'uscita nuova; nessun'altra differenza.
La sorgente unica dell'identita' del rilancio e' `RELAUNCH` in `scripts/scaling_plans19618.py`; questo file la documenta.

| Voce | Valore |
|---|---|
| Job sostituito | `SCALING-300M-eval19618-p1-20261106` = 64572000, `FAILED 1:0` (16/11, 09:49:49-10:21:11, lrdn0248) |
| Ricevuta d'invio del job sostituito | `../continuation19618_20261106/submissions/SCALING-300M-eval19618-p1-20261106.json` (sha256 `8aaddd9f9b06f77e08e630d11ade69f512609212b288b3ae7d501b21bac4f3f2`) |
| Cartella del job sostituito | `$SCRATCH/.../evaluation19618_v1/T100-G-300M/p1`, sei voci come lette il 16/11, senza `receipt.json`: resta com'e' |
| Rilancio | `SCALING-300M-eval19618-p1-r1-20261116`, cartella nuova `$SCRATCH/.../evaluation19618_v1/T100-G-300M/p1-r1` |
| Codice | checkout congelato `scaling_300m_evaluation19618_20261106_v1` a `df2de35c05381e7a5a69edde409171b210a3b5d8` (invariato) |
| Piano d'estrazione | `continuation19618_20261106/evaluation/extraction_plan.json`, sha256 `3cb94b050fa351c01e86184954f542b01b6c939169efc663d869d134e93b4556`, 5.595 byte (invariato) |
| Riga | quella del job sostituito, salvo la cartella: `--time=02:00:00`, 1 GPU, 8 CPU, 100G, `T100-G-300M extract-classification p1` |
| Tetto del job | 2 GPU-ore e 16 core-ore (walltime 7.200 s), 2,05 e 16,4 con il margine dello scheduler di 180 s |
| Costo atteso | ~0,52 GPU-ore come il tentativo fallito (31 min), cioe' ~4,2 ore locali (`billing=8`); al massimo 16 ore locali (16,4 con il margine) |

Cancello (`scaling_plans19618.py gate extractions`), in aggiunta ai controlli di ogni stadio:

- il job sostituito nel ledger e' 64572000 e `sacct` lo da' `FAILED 1:0`;
- la sua cartella ha le sei voci della diagnosi, senza `receipt.json`;
- la sua sottomissione nel ledger e' la riga del rilancio, salvo la cartella d'uscita;
- il rilancio e' l'unica estrazione da inviare, pose e p2 sono nel ledger e `COMPLETED 0:0`;
- con il rilancio al suo limite piu' il margine, lo stadio resta nei tetti del congelamento: 7 GPU-ore e 56 core-ore.

Budget, a consumo:

| Voce | GPU-ore | core-ore |
|---|---|---|
| Consumo fino a oggi (pose 2,036111; p1 0,522778; p2 1,6575) | 4,216389 | 33,731111 |
| Con il rilancio al massimo | 6,216389 | 49,731111 |
| Con il rilancio al massimo e il margine | 6,266389 | 50,131111 |
| Tetti dello stadio | 7 | 56 |

Lo stadio resta nei tetti: residuo al massimo 0,783611 GPU-ore e 6,268889 core-ore (0,733611 e 5,868889 con il margine).
La riga p1 del congelamento (2 GPU-ore, 16 core-ore, un job) passa a due job. Il suo consumo arriva al massimo a 2,522778 GPU-ore
e 20,182222 core-ore (2,572778 e 20,582222 con il margine). Questo accade solo se il rilancio dura piu' di 1 h 28 min 38 s (5.318 s):
il fallito ne ha usati 1.882. La differenza sta nel tetto dello stadio, grazie al consumo di pose e p2 sotto i loro walltime.
E' il caso peggiore dei numeri approvati: «stadio al massimo 6,216389 su 7» e' il fallito piu' il tetto 2 del rilancio.
Nessun emendamento; la decisione e' registrata in `docs/decisioni.md`, alla data del 16/11/2026, nello stesso commit di questo file.

Dopo il rilancio:

- se e' `COMPLETED 0:0`: `collect extractions` (pose, p1-r1, p2; le prove della p1 sotto `evidence/extractions/p1/`),
  poi `reader-plans`. Il costruttore rifiuta un'evidenza che non sia quella dei job della tabella, e una cache fuori dalla
  cartella della sua ricevuta;
- se fallisce per qualunque causa, compresa la stessa corsa: stop, nessun altro rilancio. Il seguito (opzione B della diagnosi,
  o altro) e' una decisione nuova di Simone.

Il rilancio e' unico: un secondo invio dello stesso ID e' rifiutato dal submitter, e il cancello non ammette altri job
dello stadio. Il submitter reinvia lo stesso ID solo in due casi, senza che un job sia mai partito:
- `--dopo-rifiuto`, dopo un rifiuto di `sbatch` e la correzione della causa;
- `--dopo-assenza-verificata`, dopo un esito ambiguo che `--riconcilia` ha accertato senza job (stato `assente`).
Base della review per i checkout successivi (plans, budgets, inputs): il commit che porta questa modifica, per l'approvazione di
Simone sopra (`docs/decisioni.md`, 16/11/2026); dopo di esso solo `docs/` e aggiunte sotto
`continuation19618_20261106/{evidence,evaluation,readers,submissions}/`.
