# Valutazione lunga — emendamento delle risorse A1: implementazione ed esecuzione

Documento dell'operatore (Claude, linea 2). Proposta di GPT consolidamento:
`origin/codex/consolidation-20261104` `4bb39d254c26bd3b9666dff0e1b1464f96bfe4db`,
`docs/consolidation_long_resource_amendment_proposal_20261106.md`. Review del disegno:
`origin/worktree-claude-training-engine` `7ceedbb5d891168f585229e826baf6535f35c1cc`,
`docs/reviews/claude_review_consolidation_long_resource_amendment_20261106.md` (disegno approvato, verifiche
V-1…V-8 richieste all'implementazione). Base dell'implementazione: `2a20316a0bf77c93e354d4ce3f5acee94db459df`.

**Stato: implementato e verificato, NON rilasciato.** Il piano porta `amendment.decision: null` e il runtime rifiuta
ogni stadio finche' la decisione di Simone non e' registrata. Nessun pin della fase fit, nessun deployment, nessun
job: la fase fit resta ferma come dal 06/11 alle 11:55.

L'emendamento e' **post hoc rispetto ai tempi dei sedici benchmark** (solo train) e **anteriore a qualunque fit,
punteggio o predizione** del confronto lungo. Riguarda impacchettamento dei job, ammissione delle risorse,
provenienza degli input e rilascio delle fasi. Non e' una regola pre-registrata e non va chiamata cosi'.

## 1. Che cosa cambia e che cosa no

| Voce | Prima | Con A1 |
|---|---|---|
| Fit di classificazione | 48 job (DB2 a coppie, gli altri 4 fold) | **69 job**: DB2 pre e pool un fold per job, DB3 pre un fold per job, DB6 pool coppie [0,1] e [2,3], gli altri invariati |
| Celle scientifiche e fold | 40 celle x 4 fold | **le stesse 40 x 4**, ciascun fold una volta sola |
| Walltime | 4 h per tutti | per shard: `W = 900 * ceil((P + 600)/900)`, ammesso se `P <= W - 600` e `W <= 14400`; da 30 a 225 minuti |
| CPU e memoria | 8 CPU, 30 GiB | invariate |
| Proiezione P per fold | politica `consolidation-classification-resources-20261105-v1` | **la stessa**, sulle stesse 15 ricevute |
| Source dei nuovi output | `1ca7aead…` | **`77e09626271b1aeded8d7f3df77ef9557115e903a9aca8e280e3f3259527ac88`** (unica per fit, confronti e sintesi) |
| Input | stessa source del codice (R-2(i)) | eccezione esplicita: **solo le 28 ricevute originali** del manifest, source `1ca7aead…` |
| Estrazioni e benchmark | eseguibili | **stadi chiusi**: il codice nuovo li rifiuta |
| Job e checkout | 82 + 4 | **103 + 4** (77 allocazioni ancora da eseguire) |
| Tetti | 9 GPU-ore, 1.711,666667 core-ore | **invariati** |

Invariati: ricetta (`recipe_sha256` `b2356a8e…`), griglia C, selezione, solver, contesto numerico, dati, endpoint,
regola di lettura, fattori prudenziali, politica e risorse della posa. `scripts/consolidation_long_classification.py`,
`scripts/consolidation_long_pose.py`, `scripts/consolidation_long.py` e tutto `src/` sono byte-identici a `166fb86`.

File dei sorgenti coperti che cambiano, e solo questi (elenco chiuso, verificato a ogni avvio):

| File | Definizioni cambiate | Definizioni aggiunte |
|---|---|---|
| `scripts/consolidation_long_amendment.py` | file nuovo: politica A1, forme dei fold, formula di W, elenco chiuso, controlli | — |
| `scripts/consolidation_long_common.py` | `validate_plan`, `load_context`, `begin_output`, `verified_extraction` | import del modulo, `verified_input` |
| `scripts/consolidation_long_classification_runtime.py` | `_benchmarks`, `execute` | `amended_gate` |
| `scripts/consolidation_long_pose_runtime.py` | `_manifest`, `_read`, `_benchmark_gate`, `execute` | — |
| `scripts/slurm/consolidation_long.sbatch` | righe 52-53 (caso speciale DB2) sostituite dal walltime dello shard | — |

## 2. Come sono garantite le tre proprieta' chieste

**Riuso delle 28 ricevute.** `results/infra/consolidation_long_20261105/amendment_a1/input_compatibility.json`
(sha256 `983df48465531014275b6f5f677e6907a1219f26fbeef529b6489ea85c473c95`, 65.919 byte) elenca le 12 estrazioni e i
16 benchmark con percorso, sha256, byte, fase, tipo e campi d'identita'; contiene anche la mappa dei 227 file della
source originale. Il suo pin e' scritto dentro il sorgente: un altro elenco e' un'altra source. `C.verified_input`
ammette una ricevuta di input solo se il pin e' elencato esattamente e del tipo giusto, e se il report porta source
originale (diversa da quella in esecuzione), ricetta, `code_sha` e `plan_sha256` della fase originale e i campi
elencati. Ogni lettura di input passa di li' (estrazioni, benchmark di classificazione e di posa). A ogni avvio
`load_context` chiama `validate_inputs`: i sorgenti coperti devono differire dalla mappa originale solo nei quattro
file sopra piu' il modulo nuovo. Nei tre file Python il testo e' diviso in regioni che lo coprono tutto, byte per
byte: ogni istruzione di primo livello possiede le proprie righe e quelle che la precedono (decoratori, commenti,
righe vuote), e il resto dopo l'ultima ha una chiave sua. Fuori dalle definizioni emendate e aggiunte ogni regione
deve avere l'hash di `166fb86`, l'ordine delle istruzioni deve essere quello originale, e le tre definizioni
aggiunte devono stare subito dopo l'istruzione dichiarata. I due rami di estrazione e benchmark dentro `execute` e
le prime 51 e ultime 15 righe del launcher sono congelati per hash. Nessuna ricevuta e' riscritta.

**Copertura dei fold.** Le forme registrate stanno in `FIT_POLICY['fold_shards']`. Il fit accetta solo `--folds`
espliciti e registrati. Il confronto pretende 69 report della source nuova con il record dell'emendamento, uno per
ciascuna chiave della tabella del piano; il modulo scientifico (invariato) rifiuta fold duplicati o mancanti per
cella. I fold sono indipendenti: un test confronta fit interi, a coppie e a fold singoli riga per riga.

**Nuovi gate.** `BENCHMARK_POLICY` e `resource_gate` originali non sono toccati: le 15 ricevute portano quella
politica e il gate originale resta negativo (15 shard su 48, 10 celle su 40). `amended_gate` riusa le proiezioni per
fold del gate originale, aggiunge gli stessi 600 s fissi e due volte carico e scala per job, calcola W e ammette
solo se passano tutti i 69. La tabella dei 69 W e' congelata in `plan.amendment.fit_shards` prima di qualunque fit:
il runtime la ricalcola dalle ricevute e si ferma se differisce; il launcher confronta il `TimeLimit` del job col W
del suo shard e rifiuta gli shard non registrati. Nessun adattamento automatico: un TIMEOUT ferma la fase.

## 3. Numeri (ricalcolati dalle ricevute raccolte)

| Voce | Valore |
|---|---:|
| Gate originale | 15 shard respinti su 48, 10 celle su 40; proiezione 1.087,621716 core-ore |
| Gate emendato | 69 su 69 ammessi; proiezione 1.115,658874 core-ore |
| Allocazione dei 69 fit | 1.276 core-ore |
| Altre allocazioni future (3 posa, 2 confronti, sintesi, 2 checkout) | 41 core-ore |
| Massimo futuro | 1.317 core-ore, 0 GPU-ore |
| Consumato finora | 42,218889 core-ore, 3,373611 GPU-ore |
| Consumato piu' massimo futuro | 1.359,218889 core-ore |
| Totale prudenziale (tetti delle fasi concluse inclusi) | 1.451,666667 core-ore |
| Tetto approvato | 1.711,666667 core-ore (margine 260) |
| Margine minimo fra P e W - 600 | 40,48 s |
| Memoria proiettata massima | 4,078 GiB su 25,5 ammessi |

Walltime richiesti (minuti: job): 30: 8, 45: 3, 75: 3, 90: 3, 120: 6, 135: 6, 150: 15, 165: 6, 195: 9, 210: 5,
225: 5. I 33 shard che non cambiano forma hanno la stessa P di prima, bit per bit. I 69 ID, uscite e W coincidono
con `resource_amendment_proposal/options.json` di GPT.

## 4. Verifiche eseguite

Dal worktree, con `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`:

~~~bash
~/.venvs/wearusfm-tok/bin/python results/infra/consolidation_long_20261105/amendment_a1/build_inputs.py --check
PYTHONPATH=src:scripts ~/.venvs/wearusfm-tok/bin/python results/infra/consolidation_long_20261105/amendment_a1/apply_to_plans.py --check
PYTHONPATH=src:scripts ~/.venvs/wearusfm-tok/bin/python results/infra/consolidation_long_20261105/amendment_a1/pin_phase.py fit --dry-run
~~~

- `build_inputs.py --check`: il manifest e' ricostruito identico dagli oggetti Git di `7d276c9` e `166fb86` e dalle
  ricevute raccolte; il pin coincide con quello congelato nel modulo.
- `apply_to_plans.py --check`: `plan.json` e `launch_plan.json` sono la derivazione dai piani di `2a20316`, dalle 15
  ricevute e dal modulo, a meno di decisione, pin di fase, SHA di fase e ID SLURM.
- Test: `tests/cpu/test_consolidation_long_amendment.py` (nuovo, per V-1…V-8, anche sulle 28 ricevute reali e sul
  piano concreto) e i tre file di test dei runtime aggiornati. Due test dei rami di estrazione e benchmark della
  classificazione, non piu' raggiungibili, sono sostituiti da test di rifiuto degli stadi chiusi; il testo di quei
  rami resta congelato per hash. Esiti del 06/11 sull'albero pubblicato (Mac, ambiente `wearusfm-tok`, thread a 1):
  file dell'emendamento 64 superati; gli undici file della valutazione lunga e dei moduli collegati 342 superati;
  suite completa `tests/cpu` 2.456 superati, nessun fallimento, in 26 minuti e 30 secondi.
- Parita' del controllo dell'elenco chiuso fra Mac e Leonardo (06/11 alle 15:04, sola lettura dal nodo di login
  `login05`, nessun file scritto): con l'ambiente dei job (`cineca-ai/4.3.0`, Python 3.11.6) `validate_inputs`
  completo accetta i testi correnti contro il manifest e la mappa dei 228 file; chiavi, ordine e hash delle regioni,
  `blocks` e cornice del launcher sono uguali a quelli del Mac (Python 3.11.14); le regioni dei testi originali
  ridanno `definitions` e `order` del manifest; nove modifiche di controllo (decoratore, commento, riga vuota, fine
  file, scambio, definizione aggiunta spostata, definizione congelata tolta, istruzione in piu', corpo congelato)
  sono rifiutate con gli stessi messaggi sulle due macchine. Il controllo completo resta il preflight sul checkout
  (sezione 5, passo 5).
- Prova a secco del rilascio in una copia di lavoro separata e poi rimossa (nulla nel ramo, nulla sul cluster),
  ripetuta sull'albero corretto: con una decisione fittizia, `record_decision.py` e `pin_phase.py fit` scrivono piano
  e manifest e rifiutano una seconda esecuzione; i tre controlli sopra e gli undici file di test passano anche in
  quello stato (342). Lo script vero del preflight, eseguito sul piano cosi' pinnato con i soli file del cluster
  sostituiti dalle copie raccolte, legge dal ponte tutte e 28 le ricevute vere (15 + 1 benchmark, 6 + 6 estrazioni),
  ammette i 69 shard e la posa e non chiede nessuna delle tre cache held della posa. Le 72 righe d'invio generate
  dalla matrice (senza inviare) hanno ID, uscita, `--folds` e `--time` uguali alla tabella congelata, con i tre fit
  di posa per primi.
- Verifica indipendente del diff (subagent `critico`): primo esito «DA RIVEDERE» per il controllo dell'elenco
  chiuso (sezione 6), corretto; riverifica del delta «APPROVATO». Il revisore ha provato dieci modifiche sue: tutte
  quelle fuori dalle definizioni emendate sono rifiutate (nove elencate nel suo resoconto); restano accettati, come
  da disegno, un decoratore o un commento davanti a una definizione emendata e il codice dentro di essa. Gate,
  tabella dei tempi e copertura dei fold li ha ricalcolati lui. Resoconto nel registro condiviso.

## 5. Rilascio della fase fit con A1 (solo dopo la decisione di Simone)

Sostituisce, per le fasi fit e aggregate, le parti corrispondenti delle sezioni 2 e 3 di
`docs/handoff_line2_claude_20261105.md`. Tutto il resto del prontuario resta valido (submitter e ledger comuni,
`--agente claude`, `--riconcilia` per risposte ambigue, mai un secondo ID per lo stesso tentativo, sezioni 4-6).

1. **Decisione.** Registrare le parole di Simone, alla lettera:
   `record_decision.py --date AAAA-MM-GG --question '…' --quote '…'`. Senza, `validate_plan` rifiuta.
2. **Pin della fase.** `pin_phase.py fit`: verifica che le copie raccolte siano i pin elencati, ricalcola il gate
   originale (negativo) e quello emendato (uguale alla tabella), il gate della posa; scrive
   `execution/manifests/measured_resource_gates_a1.json` e `classification_benchmark_receipts.json`, fissa
   `plan.benchmark_receipts`. Non sovrascrive.
3. **Controlli e commit di fase.** Suite, i tre controlli della sezione 4, `git diff --check`, `git status` senza
   dati; commit «Release amended long fit phase (A1)», push su origin, SHA remoto verificato: e' `LONG_PHASE_SHA`.
4. **Deployment.** Ref `refs/heads/deploy/consolidation-long-fit-20261105` vuoto su leonardo, poi
   `git push leonardo "$LONG_PHASE_SHA:refs/heads/deploy/consolidation-long-fit-20261105"` e checkout
   `LONG-checkout-fit-20261105` con `scripts/slurm/campaign_checkout.sbatch` (comando della sezione 1 del
   prontuario). Il checkout conta nel limite di corsia.
5. **Controlli sul checkout.** sacct `COMPLETED 0:0`, log «checkout di campagna pronto», HEAD, `git status` pulito,
   sola lettura. Poi il preflight in sola lettura sul nodo di login, con l'ambiente dei job (`cineca-ai/4.3.0` e il
   venv di `$WORK`), dentro il checkout: `amendment_a1/preflight.py "$LONG_PHASE_SHA"`, con `src:scripts`
   anteposto a `PYTHONPATH` e non al suo posto (sul cluster il modulo `cineca-ai` fornisce numpy da li': il primo
   tentativo del 06/11 lo sostituiva e si e' fermato a `import numpy`, prima di ogni controllo). Meno di un
   minuto di CPU,
   circa 3,7 GiB di hash (le quindici cache di classificazione, le tre cache train della posa e gli artefatti del
   benchmark della posa), tutte e 28 le ricevute lette dal ponte, nessuna uscita. Le cache held della posa non
   vengono lette, nemmeno per l'hash: le apre il fit, dopo le due ridge. Un errore qui si corregge prima di
   consumare un ID logico.
6. **Letture fresche C-1** prima di ogni invio: quota WORK (`lfs quota -q -p 20147512 /leonardo_work/IscrB_WearUsFM`,
   almeno 10,724 GiB liberi) e SCRATCH grande scrivibile con margine.
7. **Invii.** Prima i tre fit di posa (`LONG-FIT-G{19618,39237,49046}-pose-20261105`, 1 h), poi i 69 di
   classificazione. Riga d'invio come nella sezione 2 del prontuario, con `--time` uguale a `allocation.seconds`
   della riga e uscita sotto `fits_a1/…`; gli ID finiscono in `-A1`. Le righe si generano dalla matrice, non a
   mano. Esempio (`--timeout 480` come nelle fasi precedenti, sotto il limite di dieci minuti dello strumento
   dell'operatore; il prontuario indica 600):

~~~bash
"$LONG_PY" "$LONG_SUBMIT" --id LONG-FIT-G19618-pre-ninapro_db2-F0-20261105-A1 --agente claude \
 --campagna consolidation_long_20261105 --gpu-ore 0 --timeout 480 -- \
 --partition=lrd_all_serial --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=30G \
 --time=03:15:00 --no-requeue \
 "--export=ALL,CODE_DIR=$LONG_CODE,EXPECTED_SHA=$LONG_PHASE_SHA" \
 "$LONG_CODE/scripts/slurm/consolidation_long.sbatch" --plan "$LONG_CODE/$LONG_PLAN" \
 --expected-code "$LONG_PHASE_SHA" --stage fit-classification \
 --out "$LONG_ROOT/fits_a1/classification/19618/pre/ninapro_db2/F0" \
 --step 19618 --dataset ninapro_db2 --view pre --folds 0
~~~

8. **Corsia seriale** (decisione di Simone del 06/11, «Due fit per volta»): al massimo due job della linea 2 fra in
   corsa e in attesa in `lrd_all_serial`, checkout e posa compresi, finche' nel ledger comune non compare
   `sottomesso` un ID che comincia con `SCALING-compare-`. Dopo, ondate entro il limite del piano di otto job CPU
   in sospeso per tutto l'utente. Lettura del 06/11 alle 14:44: il ledger mostra `SCALING-compare-20261106`
   `sottomesso` alle 14:36 (63671773), quindi la condizione e' soddisfatta; si rilegge prima del primo invio.
9. **Arresti.** TIMEOUT, OOM, valore non finito, errore, pin o gate negativo: conservare log e parziali, fermare la
   fase, nessun requeue, nessuna sovrascrittura, nessun fold scartato, nessun walltime o shard nuovo. Un rilancio
   dopo un arresto tecnico e' una nuova decisione.

**Fase aggregate.** Raccolta delle 72 ricevute dei fit con confronto dell'hash remoto
(`execution/audit/receipts_fit.json`), poi `pin_phase.py aggregate`: pretende i 69 report della source nuova, uno
per shard registrato, con record dell'emendamento, gate dello shard passato e W della tabella, piu' i tre di posa;
su una matrice incompleta non scrive nulla. Quarto checkout e due confronti come nel prontuario. Sintesi:
`pin_phase.py summary` e sezione 5 del prontuario, senza quinto checkout.

## 6. Limiti dichiarati

- P e' una stima prudenziale di pianificazione, non una garanzia: il costo del solver puo' cambiare con il
  condizionamento dei checkpoint 39.237 e 49.046, misurati solo a 19.618.
- La compatibilita' degli input e' provata per i sorgenti coperti e per il testo fuori dalle definizioni emendate e
  aggiunte (regioni congelate, ordine compreso). Le modifiche dentro le definizioni emendate sono quelle del diff,
  riviste a mano e dal revisore indipendente.
- Il revisore indipendente ha trovato che la prima versione del controllo non copriva i decoratori ne' l'ordine
  delle istruzioni (il diff non li usava, ma l'affermazione era piu' forte del controllo). Corretto prima della
  pubblicazione con le regioni descritte nella sezione 2; la source e' diventata `77e09626…` e il manifest
  `983df484…`. Dentro le definizioni emendate, e nei due file aggiunti o sostituiti per righe, la garanzia resta il
  diff rivisto.
- Codice preesistente, non toccato da A1 (segnalato dal revisore, da valutare con GPT fuori da A1): `compare-pose` e
  `summary` creano la directory di uscita prima dei controlli d'identita'. Un rifiuto lascerebbe una directory
  vuota e chiederebbe un nuovo ID e una nuova uscita: prima di inviare quegli stadi si esegue un controllo in sola
  lettura del contesto nel checkout, come per i fit.
- La source incorporata nelle sei NPZ della posa e' controllata dal runtime alla lettura (deve essere l'originale):
  sul Mac non e' verificabile. Per le tre cache train avviene prima dei fit; per le tre cache held all'apertura,
  dopo le due ridge e prima di qualunque predizione o punteggio.
- Il controllo dell'ambiente numerico dei fit contro quello dei benchmark resta quello originale e avviene sul nodo.
