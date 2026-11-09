# Linea 1 — Recupero v2 della continuazione 300M 5.000 → 19.618

Data: 09/11/2026. Autore del codice e operatore: Claude. Review: GPT, una sola, su questo pacchetto (Simone, 06/11 alle
17:22: «non voglio una ennersima review di gpt prima di ogni lancio, ne facciamo una poi lancio»).
**Stato: in review; nessun deploy, checkout o job inviato.** Prima dei lanci servono la decisione registrata di Simone
sull'inviluppo emendato, la review favorevole di GPT e il rinnovo del certificato ssh, scaduto il 09/11 mattina.

## Perché c'è un v2

Il training `SCALING-300M-train19618-20261106` (job 63696864) è `FAILED 1:0` l'08/11. Si è fermato alla verifica
d'identità dell'insieme fisso, prima del primo aggiornamento. La causa è l'ordine di import nel worker: numpy prima di
torch lega numpy al suo OpenBLAS invece che a MKL. Diagnosi:
`../continuation19618_20261106/failure_63696864_diagnosis_20261108/`. La review di GPT `6f73209`
(`docs/reviews/gpt_review_scaling_importorder_recovery_20261108.md` sul ramo `codex/scaling-300m-20261104`) approva la
correzione e chiede questo pacchetto di recupero (punti 1-6). Il piano, il checkout e il run del tentativo v1 restano com'erano.

## Contenuto

- `training_plan.json`, SHA-256 `19441cd663db059840a36651d987d76d4154419258816058950590c1594b6fe0`. È uguale al piano v1 salvo:
  - pin dei sorgenti d'esecuzione (wrapper `5d0df17d24193ac544e5e7a5a1a0871c297b7f435c1f95e7806f68d0fe9b44cc`, con torch per primo e il pre-controllo; launcher nuovo del pre-controllo);
  - cartelle nuove del run e delle ricevute, con suffisso `-v2`;
  - cancello `identity_precheck`;
  - orizzonte del controllo spostato al 01/12/2026 (la scadenza più vicina dello scratch del controllo è l'08/12);
  - blocco informativo `superseded_attempt`.
- `make_training_plan.py`: costruisce il piano dalle letture in sola lettura. Con `--check` confronta letture fresche col piano
  pubblicato (stessi pin, il run a 5.000 per primo) e verifica che uscite, checkout e riferimento di deploy del recupero manchino.
- `remote_listing.txt`, `remote_sha256.txt`: le letture del 06/11, byte per byte. Il 09/11 ssh non era disponibile; il
  wrapper riverifica ogni pin all'avvio e la rilettura fresca precede il deploy. `remote_verification.json`: riepilogo.
- `remote_read.sh`: la rilettura in sola lettura prima del deploy, con il `sacct -P` originale dei sei job già eseguiti.
- `make_envelope.py` e `budget_envelope.json` (SHA-256 `224743dff392bcd07a2a9a69fd8341b9b8f42e09d1176c2e14aefca20199736e`): consuntivo del tentativo fallito e inviluppo emendato,
  con aritmetica esatta. È una **proposta da firmare**: il freeze v2 e i suoi pin non cambiano.
- `training_submit_lines.txt`: le quattro righe d'invio, prodotte da `scripts/scaling_plans19618.py recovery-lines`.

Codice fuori da questa cartella: `scripts/scaling_continue.py` (`identity-precheck` e il suo cancello in `run`),
`scripts/slurm/scaling_identity_precheck.sbatch`, `scripts/scaling_plans19618.py` (identità unica `RECOVERY`,
prove del nuovo checkout in `evidence/checkout-training-v2/`, legame piano-ricevuta via `plan_sha256` e
`execution_source_sha256`, righe d'invio) e i test `tests/cpu/test_scaling_continue*.py`,
`test_scaling_gates19618.py`, `test_scaling_plans19618.py`, `test_scaling_recovery19618.py`.

## Il pre-controllo dell'identità

È un job solo CPU sulla corsia seriale: 4 CPU, 24G, 20 minuti al massimo, 0 GPU-ore. Gira sul checkout congelato con lo
stesso percorso di import del worker. Ricalcola i tre file d'identità del run a 5.000 con la CLI invariata del trainer;
`train` viene sostituito dopo il parsing degli argomenti e la configurazione. I tre file devono risultare identici bit
per bit. La ricevuta `identity_precheck.json` registra ordine di import, librerie numeriche caricate, versioni e
confronto. Il training la esige, sullo stesso piano e lo stesso codice, prima di creare qualsiasi uscita. La verifica del
trainer alla ripresa resta l'autorità: il pre-controllo evita soltanto di aspettare in coda GPU un job destinato a
fermarsi. Alla domanda «Procedo con il pacchetto v2, con un pre-controllo CPU dell'identita' prima del training?» Simone ha
risposto «Sì, prepara v2» (09/11 alle 09:07). La review di GPT dell'08/11 scrive che «un eventuale canary numerico nel processo
effettivo è un miglioramento futuro, non una condizione aggiuntiva di questo recupero»: il pre-controllo è quindi un punto di
questa review.

## Sequenza dopo review, decisione e certificato

Ogni passo parte solo se il precedente è finito come atteso ed è stato letto. Stop al primo esito diverso, senza retry.

1. Rilettura con `remote_read.sh`, poi `make_training_plan.py <listing> <sha256> --check`: deve uscire `passed: true`.
   Le uscite grezze e il controllo vanno in `../continuation19618_20261106/evidence/recovery-reread-<data>/`. Il
   `sacct -P` dei cinque job non ancora attestati deve dare le durate dell'inviluppo.
2. Riga 1: riferimento `deploy/scaling_300m_continuation19618_20261109_v2`, mai `main`.
3. Riga 2: checkout congelato (2 CPU, 5 minuti). Esito atteso `COMPLETED 0:0`; poi `collect checkout-training`.
4. Riga 3: pre-controllo. Esito atteso `COMPLETED 0:0`; poi `collect identity-precheck` e `identical: true` nella ricevuta.
5. Riga 4: training (1 nodo, 4 GPU, 14:45:00, al massimo 59 GPU-ore). Poi `collect training` e lo stadio B come rivisto.

## Inviluppo emendato (proposta)

| Voce | Freeze v2 | Emendato |
|---|---:|---:|
| GPU-ore con margine | 66,35 | 66,731111 |
| Core-ore CPU senza GPU, con margine (tetto 350) | 345,166667 | 347,584444 |
| Fase checkout | 0,8 | 1,066667 |
| Pre-controllo dell'identità (voce nuova) | — | 1,533333 |
| Core-ore nei job GPU, con margine | 530,8 | 533,848889 |

Restano nel conto il tentativo fallito (0,227778 GPU-ore, 1,822222 core-ore) e le quattro diagnosi (0,153333 GPU-ore e
0,617778 core-ore senza GPU). Il checkout v1 usava già il posto del checkout del training nella tabella dello stadio B, quindi
quello del recupero è un job in più. Le cifre coincidono con il conto di GPT dell'08/11 (66,731111 e 346,051111), a cui si
aggiunge il pre-controllo. Non sono incluse dosi oltre 19.618: la proposta di GPT per 49.046 è una decisione separata di Simone.

## Che cosa non cambia

Ricetta, dose, allocazione e tetto del training. Run a 5.000, controllo 30M a 19.618 e freeze v2. La parte scientifica dello
stadio B: estrattori, lettori, S8, bootstrap e proiezioni. La lista dei percorsi ammessi di `release_check` resta quella;
il commit rivisto di questo pacchetto diventa la nuova base dei checkout successivi.
