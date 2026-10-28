# Review di Claude della preparazione del collaudo lungo B di GPT (28/10/2026)

Richiesta: `docs/prompt_claude_review_b_20261028.md` (GPT), consegnata da Simone. Oggetto: ramo `origin/codex/r3-long-collaudo-20261028`,
codice congelato **24139ab901c052357a46a65bc20466f2eb1e8358** (dalla bozza di Claude afd2e4d), tip 1f7628e con protocollo, piano, comandi e
registro. Letto dal worktree di Claude (`git fetch`, `git show`, un checkout staccato di 24139ab in `~/wearusfm_local/review_b_24139ab`
per i test, rimosso alla fine); il worktree di GPT e i checkout di campagna non sono stati toccati. **Nessun deploy, nessun job, nessun
cambio di risorse o soglie.**

## Verdetto

**Nessun difetto bloccante trovato. Codice, protocollo, piano e comandi concordano.** La consegna puo' andare alla firma di Simone cosi'
com'e', su due punti che restano da approvare e che non ho modificato: (1) il criterio primario (rapporto delle medie per utente
L(5000)/L(0) <= 0,9 e limite superiore dell'IC95 appaiato < 1; bootstrap su 32 utenti, 1.000 ricampionamenti, seme 20261027); (2) il
tetto aggiornato da 75 a 90 minuti: 6 GPU-ore e 48 core-ore per B, piu' 1/3 core-ora per il checkout. Prima del lancio restano il giro
del `critico` previsto da CLAUDE.md per un criterio nuovo (se Simone lo vuole anche sul protocollo di GPT) e la procedura comune
(deploy del riferimento congelato, checkout, verifica remota, sottomissione).

## Controlli eseguiti, con l'evidenza

1. **Config del piano = config del parser vero.** `train_jepa.build_config(parse_args(piano.trainer_argv[1:]))` a 24139ab,
   `config_to_dict` confrontato campo per campo con `results/infra/pilot_posa/config_b_atteso_20261028.json`: nessuna differenza
   (nuovo run, `max_steps` 5000, schedule wsd 49.046 / warmup 1.962 / decay 0,2 / keep 19.618, lr 1e-4, batch 32, `accum_steps` 1,
   seme 0, ancore 0,2 / posa 1,0 / JEPA 0, `pose_unprompted` on, `pose_grad_every` 500, pannello 500 / 32 / 2 / 4, nested «1», 30M).
2. **Comando [B] coerente col piano:** `TIME_LIMIT_S=5100`, `REQUIRE_NEW_RUN=1`, `--time=01:30:00`, `--gres=gpu:4 --cpus-per-task=32
   --mem=480G --qos=normal`, quarto argomento 5000, `EXTRA_ARGS` del comando identici a `piano.extra_args`, `BRANCH_FROM`/`RESUME_TO` vuoti,
   cartella nuova `PILOT-POSA-R3-B-30M-s0-20261028` fuori dal codice. Checkout [C]: 2 core, 2 GB, 10 min. `--gpu-ore 6` nel ledger.
3. **Lettore sulla ricetta vera:** `arms_collaudo.long_configuration` sul config atteso, con `lr` dei record generato dal trainer
   (`_lr_at(step - 1)` = 1e-4 x min(1, passo / 1962), coincide con la formula del lettore) e SHA delle statistiche della posa
   757f2aee...: `ok = True`, nessuna differenza. Il decadimento parte al passo 39.237: fuori dai 5.000.
4. **Formato reale del trainer contro il lettore (il punto che i test sintetici da soli non coprono):** run piccolo a 24139ab (3 utenti
   sintetici, seme 0, pannello 2 x 2 x 2 ogni 3 passi, 6 passi) e poi `arms_collaudo._strict_json` + `_panel_integrity` + `panel_gate` con
   le quote ridotte: nessun errore di lettura, **integrita' verificata** (identita' e ordine delle finestre, `n_valid` uguale al manifest a
   ogni passo, `mse_const` costante, aggregati coerenti, `panel_sha256` uguale al digest canonico del manifest, baseline a zero con
   `pred_mean_per_joint` nulli), `pannello_baseline` e `pannello_bootstrap_valido` veri; `pannello_discesa` falso come atteso su 6 passi.
   Tempi del pannello nel summary (`pose_panel_timing`).
5. **Test di GPT rieseguiti sul checkout congelato** (`test_pose_panel`, `test_arms_collaudo`, `test_collaudo_b_launcher`,
   `test_pose_panel_ddp`, piu' loader, stabilita' e training con la posa): **114 passati in 2 min 29 s, rc 0**, compresi i due test a due
   processi (gloo, `mp.spawn`), con `PYTHONPATH` sul checkout congelato (log `~/wearusfm_local/logs/test_review_b_24139ab_20261028.txt`).
   Nessun training sul cluster.
6. **Lettura del diff** `afd2e4d..24139ab` su `run.py`, `arms_collaudo.py`, `train_ddp.sbatch` e `tests/cpu`, punto per punto sotto.

## Lettura del codice (nessun rilievo bloccante)

- **Selezione e supporto del pannello** (`_pose_panel_set`): solo righe di emg2pose con probabilita' di campionamento > 0 (`loader.probs`,
  allineate a `index.rows`), deduplicate per sessione; finestre esattamente da 4 s (`min_window_s = max_window_s = 4`, `window_at` da'
  None sotto i 4 s); inizi visitati in ordine crescente (`ks` crescenti per tratto) con il controllo di sovrapposizione sullo stesso
  trial e lo stesso calcolo dell'inizio di `window_at`; a lunghezza fissa «greedy per inizio» = «greedy per fine». Eleggibile solo la
  finestra con almeno un elemento di posa valido nella maschera dei target; per il pannello tutte le finestre hanno 160 patch, quindi
  `pose_time_valid` (patch esistenti) non toglie nulla e il `n_valid` del manifest (sola maschera) coincide con quello misurato
  (confermato dal controllo 4). Utenti e registrazioni permutati da un generatore proprio `[seed, 7272]`; esclusioni registrate con il
  motivo; errore se le quote non si raggiungono. Le maschere del training sono estratte una volta in `materialize` e restano fisse.
- **Indipendenza dal training:** stesse garanzie del pannello fisso (`fork_rng`, `no_grad`, eval e ripristino del modo, loader proprio);
  provata bit per bit dai test a 1 e a 2 processi (pesi, teacher, optimizer, RNG e metriche uguali con e senza pannello).
- **DDP:** il pannello si costruisce e si valuta solo sul rank 0; l'eventuale errore di preparazione viaggia a tutti i rank con
  `broadcast_object_list` prima del training, e tutti alzano lo stesso `ValueError` senza checkpoint ne' metriche (test a 2 rank). Gli
  altri rank attendono il rank 0 alle valutazioni: costo contato nel tetto.
- **Identita' alla ripresa:** `pose_panel_set.json` con `_write_or_check` (schema, seme, utenti, finestre, `n_valid`, per giunto, hash dei
  batch, statistiche); ogni record porta il digest canonico del manifest; il lettore rifiuta digest, ordine, supporti o costante diversi.
- **Aggregazione e bootstrap:** media delle 8 finestre per utente, media dei 32 utenti; rapporto delle medie; bootstrap appaiato
  (stessi indici di utente al numeratore e al denominatore), repliche con denominatore non positivo o non finito invalidano la lettura
  (`pannello_bootstrap_valido`); percentili 2,5-97,5. Baseline a zero verificata per finestra e per giunto (tolleranza 1e-6).
- **Partenza da zero:** `REQUIRE_NEW_RUN=1` nel launcher rifiuta cartella esistente, `BRANCH_FROM`, `RESUME_TO`, `--init-from` e
  `--branch-from` prima di caricare moduli o torchrun (test con 5 casi). Con `set -u` la variabile ha default 0: le catene WSD non cambiano.
- **Lettore fail-closed:** JSON stretti (chiavi duplicate, record non oggetto, file mancanti), `file_integri`, `configurazione_b`,
  `completo`, `codice_e_ddp`, `record`, `ritmo` <= 0,81; stato `non_valutabile` / `non_superato` / `superato`; discesa a blocchi del
  training, varianza temporale per giunto, MAE in radianti, gradienti condivisi (500-5.000) solo descrittivi, come dice il protocollo.

## Osservazioni non bloccanti (nessuna richiede un nuovo SHA prima della firma)

1. `valid_fraction` per finestra ha al denominatore `P_max x 20`: con finestre tutte da 160 patch e' la frazione vera; se un giorno il
   pannello avesse lunghezze miste sarebbe diluita dal riempimento. Solo descrittiva.
2. `panel_eval` scrive con `allow_nan=False`: una predizione non finita sul pannello farebbe uscire il job con un'eccezione nel ciclo, senza
   `summary.json`. L'esito del collaudo sarebbe comunque «non valutabile» (giusto), ma si perderebbero il riepilogo e i tempi; una
   conversione a `None` come nel lettore sarebbe piu' morbida. Al passo 0 la testa e' a zero esatto anche in bf16.
3. Il lettore fissa il seme del pannello `[0, 7272]` e le dimensioni del 30M nella ricetta attesa: giusto per questo B; un altro seme o
   un'altra taglia richiederebbero un lettore aggiornato (dichiararlo nel piano di quel run).
4. `pose_panel.jsonl` conterra' circa 11 x 256 finestre con liste per giunto (stima 4-5 MB): bene per la copia sul Mac e per il lettore; per
   il commit in `results/` conviene decidere se tenere il file intero o il verdetto con gli aggregati, come per gli altri collaudi.
5. I comandi portano `--agente gpt`: chi sottomette aggiorna lo stesso ledger; se sottomettesse Claude cambierebbe solo quel flag.
6. La cartella del run e' diversa da quella di R3 (`PILOT-POSA-R3-30M-s0`): una continuazione come inizio di R3 richiede la decisione
   separata gia' scritta nel protocollo (§2), non e' automatica. Coerente con «un esito positivo non autorizza R3 completo».
7. Costo della preparazione del pannello sui dati veri non misurato (sul sintetico 0,6 s): e' lineare nella durata delle registrazioni
   visitate (una lettura della posa per finestra candidata non sovrapposta) e viene registrato nel summary; il margine del tetto (5.100 s
   interni contro 59-68 min di passi a 0,711-0,81 s) lo copre con larghezza.
8. Sul piano scientifico accetto le rettifiche di GPT alle mie letture del 27/10 (`docs/reviews/pose_scientific_assessment_20261028.md`):
   controllo disallineato positivo in 7 utenti su 10, non «allineamento 10/10»; E3 su R3c «sotto il criterio del 10%», non «posa non
   accessibile»; nessun q misurato. Le voci del 27/10 restano e questa review le rettifica per riferimento.
9. Il sottoinsieme del train di R0-e2 estratto dal job 62478387 (cancellato il resto) resta su $SCRATCH e non serve a B: nessuna azione.

## Corrispondenza richiesta dal prompt (§3)

Verificato che `piano_b_20261028.json`, `config_b_atteso_20261028.json` e `comandi_b_20261028.txt` concordano fra loro e col codice
24139ab su: nuovo run (`REQUIRE_NEW_RUN`, cartella nuova), 5.000 passi, schedule 49.046 con warmup 1.962 (`steps_at_d` 98.094), pesi
ancore 0,2 / posa 1 / JEPA 0, seme 0, quattro A100 (32 core, 480 GB, QoS normal). Tetto PROPOSTO 6 GPU-ore e 48 core-ore piu' 1/3 core-ora:
esplicito nel protocollo, nel piano (`approval = null`) e nei comandi, ancora da approvare. Non ho applicato cambi di risorse o soglie.
