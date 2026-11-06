# Linea 1 — Valutazione del 300M a 19.618 passi: piano della catena (stadio B)

Data: 06/11/2026, sera. Autore del codice e operatore: Claude. Review: GPT, una sola, su questo pacchetto.
**Stato: training inviato (job 63696864, in coda); nessun job dello stadio B inviato, nessun pin a 19.618 esistente.**
Completa `chain_plan.md` (§4 e §10), che resta com'e' nel commit di deploy `73524b6` del training.

## 1. Regola di rilascio e che cosa chiede questa review

- Simone, 06/11 alle 17:22: «si autorizzo maq non voglio una ennersima review di gpt prima di ogni lancio, ne facciamo
  una poi lancio.» `CLAUDE.md` (main `e114a78`): per la linea 1 una review del pacchetto operativo pubblicato, poi
  i lanci inclusi «rispettando i cancelli meccanici, senza nuove review o richieste di consenso fra i passaggi».
- Review di GPT sul training (`213ca5f`), giudizio g: «Un unico pacchetto di codice/piano della valutazione, poi
  benchmark, fit, aggregati e confronto con i gate previsti, senza review fra i singoli invii. Codice non consegnato
  oggi: questa review non lo attesta e non blocca il training».
- Questo e' quel pacchetto: codice, test, piano, cancelli e comandi dello stadio B. Dopo l'esito favorevole Claude
  esegue la catena fino alla raccolta finale con i cancelli scritti qui; nessun altro consenso fra i passaggi.
  I documenti che nascono durante la catena (piano d'estrazione, piani dei lettori, budget, ingressi del confronto)
  non tornano in review: sono costruiti dai costruttori rivisti qui, solo da evidenze raccolte, e ogni checkout che
  li porta sul cluster puo' aggiungere soltanto documenti ed evidenze al pacchetto rivisto (controllo meccanico, §6).
- Fuori da questo pacchetto e non autorizzati da esso: nuovo training, riestrazione del controllo, altri semi,
  secondi segmenti, retry, ampliamenti di budget o perimetro (decisione nuova di Simone).

## 2. Contenuto del pacchetto

- Lettori a 19.618 per derivazione dichiarata dai file congelati a 5.000 (`scripts/scaling_derive19618.py`, 15
  file: 7 di codice, 8 di test): `scaling_evaluation19618.py`, `scaling_pose_readout19618.py`,
  `scaling_probe19618.py`, `scaling_reading19618.py`, `scaling_compare19618.py`, i launcher
  `slurm/scaling_evaluation19618.sbatch` e `slurm/scaling_readers19618.sbatch`. Ogni file e' uguale alla sua
  sorgente a 5.000 (pin SHA-256 del commit rivisto `97125bd`) piu' le sostituzioni dichiarate, ciascuna nel numero
  esatto atteso; le sorgenti non si toccano. Differiscono solo dose (19.618), protocolli, moduli importati e
  l'identita' del controllo: 30M gia' esistente a 19.618, mai riestratto (`CONTROL_CHECKPOINT_SHA256` = freeze v2).
- `scripts/scaling_plans19618.py`: costruttori puri dei documenti della catena, tabelle dei job, righe d'invio,
  raccolta delle evidenze e cancelli. Non apre cache, non calcola punteggi, non invia nulla.
- Test (615, tutti su CPU, in locale): 8 derivati (il percorso a 5.000 resta coperto dai suoi, invariati),
  `test_scaling_derive19618.py`, `test_scaling_plans19618.py` (costruttori sulle evidenze reali del 5.000, job,
  budget, guardie dei launcher, raccolta, righe), `test_scaling_gates19618.py` (cancelli: accettazione dei 14 fit e
  dei 6 aggregati reali del 5.000, riproduzione byte per byte del manifest del confronto rivisto, mutazioni per ogni
  controllo, cancelli vivi su un repository e un ledger di prova, incastro training -> estrazione sui file scritti
  dal wrapper del training).
- Documenti: `scientific_reading_policy_19618.json` (regola di lettura, seme 20261102, 2.000 repliche),
  `control19618_identity/` (7 provenienze reali del controllo), `evidence/checkout-training/` (prima raccolta reale
  con lo strumento: job 63695634). `endpoint19618_reading.md` e' gia' nel deploy del training.

## 3. Stadi, job, risorse, uscite

Campagna `scaling-300m-20261104`, agente `claude`, submitter e ledger comuni. Risorse = freeze v2 (`readers.stages`,
`extractions.jobs`). `$CPU` = `$WORK/wearusfm_runs/results/scaling_300m_20261104/evaluation19618_v1/cpu`.

| Stadio | Job (ID logico) | Risorse e limite | Uscita |
|---|---|---|---|
| raccolta del training | nessuno | sola lettura | `evidence/training/` |
| checkout `code` | `SCALING-19618-code-checkout-20261106` | seriale, 1 CPU, 4G, 5 min | `$WORK/wearusfm_campaigns/scaling_300m_evaluation19618_20261106_v1` |
| estrazioni | `SCALING-300M-eval19618-{pose,p1,p2}-20261106` | 1 GPU, 8 CPU, 100G; 2:30, 2:00, 2:30 | `$SCRATCH/.../evaluation19618_v1/T100-G-300M/{pose,p1,p2}` |
| checkout `plans` | `SCALING-19618-plans-checkout-20261106` | seriale, 1 CPU, 4G, 5 min | `.../scaling_300m_plans19618_20261106_v1` |
| validazione posa | `SCALING-19618-pose-validate-20261106` | seriale, 2 CPU, 30G, 30 min | `$CPU/pose_validate` |
| benchmark posa (3) | `SCALING-19618-pose-bench-<cella>-20261106` | 8 CPU, 30G, 10 min l'uno | `$CPU/pose_benchmark/<cella>` |
| benchmark classif. (15) | `SCALING-19618-probe-bench-<cella>-<dataset>-20261106` | 8 CPU, 30G, 10-120 min (tabella del freeze) | `$CPU/probe_benchmark/<cella>-<dataset>` |
| checkout `budgets` | `SCALING-19618-budgets-checkout-20261106` | seriale, 1 CPU, 4G, 5 min | `.../scaling_300m_budgets19618_20261106_v1` |
| ridge di posa | `SCALING-19618-pose-ridge-20261106` | 8 CPU, 30G, 60 min | `$CPU/pose_ridge` |
| fit (al massimo 15) | `SCALING-19618-probe-fit-NN-20261106` | 8 CPU, 30G, al massimo 3 h l'uno | `$CPU/probe_fits/<shard>` |
| aggregati (6) | `SCALING-19618-probe-aggregate-0N-20261106` | 2 CPU, 30G, 10 min l'uno | `$CPU/probe_aggregates/<cella>-<compito>` |
| checkout `inputs` | `SCALING-19618-inputs-checkout-20261106` | seriale, 1 CPU, 4G, 5 min | `.../scaling_300m_inputs19618_20261106_v1` |
| confronto | `SCALING-19618-compare-20261106` | 8 CPU, 30G, 30 min | `$CPU/comparison` |

- Lettori CPU su `lrd_all_serial`, `--nodelist=login08` (stesso contesto numerico per benchmark, fit e confronto),
  `CODE_DIR` = checkout `code`, `EXPECTED_SHA` = suo commit; piani e budget letti dai checkout `plans`/`budgets`/
  `inputs` con SHA-256 sulla riga di comando. Log in `$WORK/wearusfm_runs/logs/` (`scaling_evaluation19618_<job>.out`,
  `scaling_readers19618_<job>.out`, `campaign_checkout_<job>.out`).
- Uscite CPU su `$WORK` come a 5.000 (circa 1,4 GiB: 1,3 di array della ridge); cache delle estrazioni su scratch
  (come a 5.000, rigenerabili dal checkpoint persistente). Spostare le uscite CPU su scratch sarebbe una decisione di
  Simone: l'eccezione della linea 2 non si estende da sola. Cancello: `$WORK` con almeno 5 GiB liberi a ogni invio.

## 4. Costi (core-ore solo CPU, `budget-table`)

| Stadio | Job | Richieste | Con margine scheduler | Tetto |
|---|---|---|---|---|
| checkout (A0 fatto + 4) | 5 | 0,50 | 0,80 | 0,80 |
| validazione posa | 1 | 1,00 | 1,10 | 1,10 |
| benchmark posa | 3 | 4,00 | 5,20 | 5,20 |
| benchmark classificazione | 15 | 70,67 | 76,67 | 76,67 |
| ridge di posa | 1 | 8,00 | 8,40 | 8,40 |
| fit (caso peggiore) | <= 15 | <= 240,00 | <= 244,00 | 246,00 |
| aggregati | 6 | 2,00 | 2,60 | 2,60 |
| confronto | 1 | 4,00 | 4,40 | 4,40 |
| **totale** | | | **343,17** | **350** |

- Estrazioni: 7 GPU-ore richieste (2,5 + 2 + 2,5), tolleranza 0,15 (lo 0,20 e' del training); 56 core-ore allegate,
  rendicontate a parte. Fit: il piano reale nasce dai 15 benchmark a 19.618 (formula, arrotondamento e packing a
  copertura esatta rivisti a 5.000) e non puo' superare 15 blocchi, 3 ore l'uno, 240 core-ore richieste: il
  costruttore si ferma prima dei fit altrimenti.
- **Differenza dichiarata rispetto al freeze (`checkout: jobs 3, 2 CPU`): cinque job di checkout, A0 a 2 CPU (gia'
  fatto, 69 s, 0,04 core-ore) e quattro a 1 CPU, nelle stesse core-ore (0,5 richieste, 0,8 con margine, uguale al
  tetto).** Servono quattro checkout perche' ogni documento della catena esiste solo dopo lo stadio precedente e
  raggiunge il cluster solo dentro un checkout congelato (nessuna scrittura via ssh), come a 5.000. Un CPU basta:
  sei checkout di questa campagna, letti in `sacct`, hanno usato 1,5-2,0 s di CPU in 16-69 s di orologio (limite
  300 s), picco 176 MB (limite 4G): il tempo e' del filesystem, non della CPU. Da confermare nella review; se non accettata, resta A0 e si ferma tutto prima del codice.

## 5. Ordine e cancelli, come scattano

Ogni passo e' un comando di `scaling_plans19618.py` (§9). Un documento esiste solo se il suo costruttore accetta
tutte le evidenze; un job si invia solo se il cancello vivo passa; ogni cancello lascia il suo verbale in
`evidence/gates/<stadio>_<ora UTC>.json`. Niente e' automatico fra un passo e l'altro: li eseguo io, uno alla volta.

1. Training `COMPLETED 0:0` -> `collect training`: `sacct -P` grezzo, log, le nove ricevute copiate, hash del
   checkpoint a 19.618 (non copiato), inventario esatto della cartella. Qualunque altro stato: stop e diagnosi.
2. `extraction-plan`: dal piano a 5.000 piu' ricevuta, riepilogo e configurazione raccolti; rifiuta ricevute
   incomplete, dose o protocollo diversi, controllo diverso dal freeze. Commit (evidenze + piano).
3. `gate checkout-code --sha <commit> --reviewed <pacchetto rivisto>` poi `lines checkout-code --sha <commit>`:
   riferimento `deploy/<campagna>` (mai `main`, mai un riferimento gia' estratto) e job di checkout.
4. `gate extractions` -> tre righe, job indipendenti (partizione GPU). Dentro il job: `CODE_DIR` allo SHA atteso e
   pulito, piano con SHA-256, ricevuta del training riletta (`load_teacher_endpoint`: protocollo, `complete`, passo,
   `start_step` 5.000, continuazione esatta, contratto delle sorgenti, hash del checkpoint), uscite nuove.
5. `collect extractions` -> `reader-plans`: identita' delle cache candidate (ricevute, provenienze, SHA del codice e
   del piano), controllo dalle provenienze reali e dagli inventari rivisti della campagna del tokenizer. Commit ->
   `gate checkout-plans` -> righe del checkout.
6. `gate pose-validate`, `gate pose-benchmark`, `gate probe-benchmark` -> righe, a tranche (§7). Nessuna selezione
   di iperparametri nei benchmark (solo train, `selection_performed: false`), come a 5.000.
7. Raccolte -> `budgets`: budget della ridge (formula del freeze, `MaxRSS` positivo e entro 30 GiB da `sacct`),
   piano dei fit (15 benchmark completi e convergenti, stesso contesto numerico, packing, tetti), verbale del
   packing. Commit -> `gate checkout-budgets` -> righe del checkout.
8. `gate pose-ridge` e `gate probe-fit` -> righe, a tranche. Dentro i job: limiti del freeze (8 CPU, 30 GiB, 3 ore)
   e identita' di piano e budget sulla riga di comando.
9. Raccolte -> `gate probe-aggregate` esegue prima l'accettazione dei fit (identita' di ogni shard, copertura esatta
   3 celle x 5 dataset x 4 fold, 7 fit per dataset e fold, firma numerica uguale ai benchmark, file pinnati accanto
   al report): rifiuta prima di qualunque aggregato. Poi le sei righe (catena `afterok`).
10. `collect probe-aggregate` -> `comparison-inputs`: i sei aggregati con identita', shard letti uguali a quelli
    accettati (hash: chiude S-3), firma numerica uguale ai fit (chiude S-1), 28 righe di audit per dataset,
    finestre valutate uguali alle ancore dei sei aggregati accettati a 5.000 (stesse finestre, stessi soggetti),
    convergenza per criterio di ogni aggregato — altrimenti stop prima del confronto, come a 5.000 — report della
    ridge con le tre celle e runtime uguale ai suoi benchmark, percorsi assoluti remoti. Il manifest prodotto dai
    dati reali del 5.000 e' uguale byte per byte a quello rivisto. Commit -> `gate checkout-inputs` -> righe.
11. `gate compare` -> una riga. `collect compare` -> raccolta finale, consuntivo, esito tecnico (controlli S-2 e
    S-4 sui report: bootstrap 2.000/20261102 nei blocchi, punto dell'aggregato e del confronto entro 1e-12) e
    analisi scientifica, in documenti separati per GPT. La lettura resta quella di `endpoint19618_reading.md`.

Cancello vivo di ogni stadio di job (`gate <stadio>`): HEAD e albero pulito di ogni checkout usato
(`GIT_OPTIONAL_LOCKS=0`); byte remoti dei documenti pubblicati uguali a quelli nel repo; file pinnati dentro i
checkout presenti nei rispettivi commit; uscite dei job da inviare assenti; job gia' nel ledger inviati da questa
linea; job gia' inviati dello stadio `COMPLETED 0:0` (qualunque altro stato ferma la tranche successiva); `$WORK`
con almeno 5 GiB liberi. Cancello di ogni checkout (`gate checkout-<nome>`): il commit discende dal pacchetto
rivisto e da ogni checkout precedente aggiungendo solo file sotto `evidence/`, `evaluation/`, `readers/`,
`submissions/` o modifiche in `docs/` (un documento gia' distribuito non si riscrive); i checkout nascono
nell'ordine `code`, `plans`, `budgets`, `inputs`, una volta ciascuno; i documenti del checkout sono nel commit con
i byte pubblicati; riferimento di deploy assente; cartella assente; ID mai usato; un posto libero nella coda
seriale; `$WORK`. Il commit di ogni checkout si legge dalla ricevuta del ledger (`CAMPAIGN_SHA`), mai a mano.

## 6. Corsia seriale, tranche, spazio, manutenzioni

- `lrd_all_serial` per utente: 8 CPU in esecuzione, 2 job in esecuzione, 10 sottomessi, condivisi con la linea 2.
  Un job a 8 CPU occupa da solo la corsia: gira un lettore alla volta, di qualunque linea. Il cancello legge la coda
  seriale e dichiara `submittable_now` = posti liberi fino a 10; `lines <stadio> --max N` emette solo i job non
  ancora nel ledger, in catena `afterok` (il primo fallimento ferma gli altri; un job in attesa di un'altra linea
  passa davanti alla catena). Lettura mia, prudente, finche' Simone non dispone altrimenti: tengo al massimo 4 job
  della linea 1 sottomessi insieme nella corsia seriale.
- Tempo di corsia della linea 1 ai limiti: validazione 0,5 h, benchmark 0,5 + 8,8 h, ridge 1 h, fit fino a 30 h,
  aggregati 1 h, confronto 0,5 h: fino a circa 43 h di orologio, da spartire con la linea 2. Chi passa prima lo
  decide Simone; io non tocco job di altre linee.
- `$WORK`: 41,7 GiB liberi alle 20:32 del 06/11; lo stadio B ne usa circa 1,8 (uscite CPU e quattro checkout da
  97 MB). Cancello a 5 GiB.
- Rinfresco firmato dello scratch del 12/11: un `touch` sui file pinnati mentre gira un job della linea 1 puo' far
  rifiutare a contenuto invariato (`check_unchanged` del training confronta anche inode, mtime e ctime). Va fatto
  senza job della linea 1 in esecuzione o escludendo i file pinnati: lo coordina Simone, non lo decido io.
- Manutenzioni: 10-14/11 parziali (la partizione resta su); 23-25/11 tutto fermo. Orizzonte del piano: 22/11.

## 7. Stop e recuperi

- Stop al primo cancello fallito, errore, `TIMEOUT` o sforamento: raccolta, diagnosi, registro. Nessun retry cieco;
  nessun job si reinvia con lo stesso ID (il submitter rifiuta; `--dopo-rifiuto` solo dopo la correzione della causa
  e solo per un rifiuto, `--riconcilia` solo per un esito ambiguo).
- Un lettore fallito lascia la sua cartella: un recupero tecnico richiede un piano con i suoi pin e una review.
  Controllo mancante o diverso dai pin: stop e review, nessuna riestrazione fuori budget (freeze v2).
- Sforare un tetto o cambiare perimetro richiede una decisione nuova di Simone.

## 8. Limiti dichiarati

- I costruttori e i cancelli sono provati sui dati reali del 5.000 e su repository/ledger di prova; sul cluster
  reale e' stata esercitata solo la raccolta (`collect checkout-training`, job 63695634). La prima raccolta a
  19.618 e' quella del training.
- I benchmark a 5.000 non sono misure a 19.618: il piano dei fit nasce dai benchmark nuovi (freeze v2).
- Le righe d'invio contengono lo SHA del checkout `code`, che esiste solo dopo il commit del piano d'estrazione:
  per questo non sono pubblicate ora; le genera `lines` dal ledger, e ogni gate le ricontrolla.

## 9. Comandi

Sempre dal worktree d'esecuzione, con `PY="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
~/.venvs/wearusfm-tok/bin/python -B scripts/scaling_plans19618.py"`:

    $PY collect training                      # poi: extraction-plan; commit; push verificato
    $PY gate checkout-code --sha <C> --reviewed <R>   &&  $PY lines checkout-code --sha <C>   # due righe, una alla volta
    $PY collect checkout-code
    $PY gate extractions                      &&  $PY lines extractions                   # tre righe
    $PY collect extractions                   # poi: reader-plans; commit; gate/lines checkout-plans; collect
    $PY gate pose-validate                    &&  $PY lines pose-validate
    $PY gate pose-benchmark                   &&  $PY lines pose-benchmark
    $PY gate probe-benchmark                  &&  $PY lines probe-benchmark --max <N>    # a tranche
    $PY collect pose-validate; ... pose-benchmark; ... probe-benchmark   # poi: budgets; commit; checkout-budgets
    $PY gate pose-ridge                       &&  $PY lines pose-ridge
    $PY gate probe-fit                        &&  $PY lines probe-fit --max <N>          # a tranche
    $PY collect pose-ridge; $PY collect probe-fit
    $PY gate probe-aggregate                  &&  $PY lines probe-aggregate               # sei righe in catena
    $PY collect probe-aggregate               # poi: comparison-inputs; commit; checkout-inputs; collect
    $PY gate compare                          &&  $PY lines compare
    $PY collect compare                       # poi: consuntivo, esito tecnico, analisi scientifica

Ogni riga d'invio si esegue cosi' com'e' (`bash -c "$riga"`), una alla volta; stop al primo esito diverso da
`sottomesso`. Ricevuta = copia dello stato del ledger in `submissions/<ID>.json`. Dopo ogni attivita': registro
(`docs/claude_GPT_shared`), commit e push con SHA remoto verificato. Test del pacchetto:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ~/.venvs/wearusfm-tok/bin/python -B -c "import sys, glob; \
      sys.path.insert(0, 'src'); import pytest; sys.exit(pytest.main(['-q', '-p', 'no:cacheprovider', \
      *sorted(glob.glob('tests/cpu/test_scaling_*19618*.py'))]))"
