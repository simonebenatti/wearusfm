# Tokenizer dei nuovi training: 100 ms + GELU di default — 24/11/2026

Alla domanda di Claude «Rendo 100 ms + GELU il default anche del tokenizer?» Simone ha risposto «Sì, fallo» (24/11/2026).
La modifica riguarda solo il **tokenizer**: non tocca obiettivo, maschere, target, posa, pesi delle loss o dati. Segue il contratto
del 18/11 per le finestre dei target (`docs/target_windows_fix_20261118.md`).

## Contratto

| Che cosa | Prima, senza opzioni | Ora, senza opzioni | Ricetta storica esplicita |
|---|---|---|---|
| Durata del kernel (`model.window_ms`) | 25 ms (`None`) | **100 ms** | `--window-ms 25` |
| Attivazione (`model.tokenizer_activation`) | `identity` | **`gelu`** | `--tokenizer-activation identity` |
| Passo (`model.patch_ms`) | 25 ms | 25 ms, invariato | — |

- È il tokenizer di T100-G: un token ogni 25 ms, ogni token guarda 100 ms, overlap 75 ms. Le basi scalano come già
  nell'implementazione (104 armoniche, 80 nodi, omega 24): 5.184 parametri in più per ogni taglia, tutti nel front-end (preset 30M:
  28.336.811 → 28.341.995). A pari seme gli altri moduli nascono identici; cambia anche `fe_gain`, che si calibra sul banco.
- **Il default sta nei preset** (`run.sanity_config`, da cui partono i rung e `small_config`; costante `run.NEW_TOKENIZER`), **non nei
  default di `FMConfig`**. Un config o un checkpoint senza i due campi resta quindi storico (25 ms, `identity`) con qualunque
  strumento lo rilegga. Serve: l'attivazione non sta nei pesi, e un default `gelu` in `FMConfig` reinterpreterebbe in silenzio i
  checkpoint vecchi (per esempio R5, che scrive la finestra e non l'attivazione).
- `--window-ms 25` (uguale al passo) significa banco storico, cioè `window_ms` null nella configurazione salvata; mai 25,0. `FMConfig`
  continua a rifiutare 25. Le opzioni sono indipendenti: `--window-ms 25` da solo dà 25 ms con GELU, `--tokenizer-activation identity`
  da solo dà 100 ms lineare (T100-L). La ricetta storica è la **coppia**.
- I preset firmati il 03/10 (sanity) e il 09/10 (rung) cambiano solo nel tokenizer.

## Che cosa non cambia, e va detto

- **Il default è il tokenizer di T100-G, non la ricetta di T100-G.** I preset hanno ancora JEPA accesa (peso 1), ancore 0,2, nessuna
  posa. La campagna di novembre scriveva nel comando `--jepa-weight 0 --masked-loss-radius 0 --pose-root ... --pose-weight 1.0`: un
  training nuovo che vuole quella ricetta continua a scriverla. JEPA accesa con 100 ms, GELU e raggio 2 non è mai stata addestrata.
- **Raggio delle query delle loss mascherate:** la regola è la stessa (senza `--masked-loss-radius` è il raggio nominale del kernel),
  ma con 100 ms il raggio nominale è 2 invece di 0. Sulle maschere vere (D10, sei montaggi, 1.080 finestre da 1, 2,5 e 4 s) resta query
  il 79,6% dei token nascosti: 26,0% nelle maschere corte, 68,3% nelle medie, 92,9% nelle lunghe, 100% nelle spaziali. La quota delle
  corte sulle query scende dal 14,9% al 4,9%. Nessuna finestra resta senza query. Con JEPA accesa il raggio 0 resta rifiutato.
- **Precedente R5 (150 ms, 24/10):** per un kernel più lungo il gate D8 era stato riaperto prima del lancio e Simone aveva firmato lo
  scostamento delle quote delle maschere (corte a 0). Per 100 ms il gate D8 non è mai stato eseguito: i rapporti registrati sono del
  banco da 25 ms e di quello da 150 ms; esistono i test CPU sulle feature fra frequenze (`test_frontend_window.py`,
  `test_tokenizer_activation.py`). **Da decidere con GPT e Simone prima del primo training lanciato col default nudo:** raggio 2 con le
  quote D10, e gate D8 a 100 ms (`gate_d8.sbatch --window-ms 100`, solo CPU, tetto 2 core per 2 h) oppure rinuncia registrata.
- **Tempi e memoria:** le tabelle di `calibration_plan.py` e `finestra1_calendario.py` sono del banco da 25 ms. Novembre, 30M, 5.000
  passi: T25-L 3.794 s e 13,75 GiB, T100-G 3.883 s e 15,47 GiB. Prima di una campagna nuova vanno rimisurate.

## Guardie ed errori

- Ripresa (`_check_resume_recipe`), ramo (`_check_branch`) ed encoder congelato rifiutano già un tokenizer diverso da quello del
  checkpoint: riprendere un run storico senza opzioni è un errore e lascia il checkpoint intatto; con la coppia esplicita continua.
- **Nuova, `--init-from`** (`run._check_init_tokenizer`): i pesi iniziali vogliono lo stesso tokenizer del checkpoint (campo assente =
  storico). Senza, un checkpoint a 100 ms lineare (T100-L) si sarebbe caricato senza errori in un modello con GELU, perché chiavi e
  forme sono identiche.
- **Nuova, ancora RVQ:** con raggio delle query maggiore di 0 le sue finestre da 200 ms quasi non restano nascoste per intero e la
  perdita sarebbe nulla in silenzio; `run.train` lo rifiuta. L'ancora RVQ è spenta dalla finestra 1 (decisione del 04/10).
- `--spectral-branches` richiede il banco da 25 ms e `--tokenizer-logsquare` richiede `identity`: con il default nuovo sono errori che
  dicono quale opzione aggiungere, senza ripieghi automatici.
- `core_hidden_fraction` (quota dei token nascosti che restano query) entra anche nel log dei run DDP e con accumulo, dove mancava;
  solo log, calcolata allo stesso modo nei due percorsi. `test_ddp_equivalence` gira ora su entrambi i tokenizer.
- I rapporti di `ddp_equivalence.py` e `probe_memory.py` scrivono quale tokenizer è stato misurato.

## Strumenti delle campagne storiche

- `train_jepa.historical_args(argv)` aggiunge anche `--window-ms 25` e `--tokenizer-activation identity`, ciascuna solo se manca e
  senza toccare valori già scritti. Serve: il comando registrato di T25-L scrive `identity` ma non `--window-ms`, quello di R5 scrive
  `--window-ms 150` ma non l'attivazione. Gli strumenti che già passavano da `historical_args` non cambiano.
- `run.with_historical_tokenizer(cfg)` fa lo stesso per i costruttori senza riga di comando, ed è distinto da
  `with_historical_targets`: T100-G ha i target 25/50 ma il tokenizer a 100 ms con GELU. Lo usa `step1_ab.build_config`, quindi anche
  `step2c` e `branch_pilot`: senza, l'A/B storico sarebbe diventato 100 ms + GELU in silenzio.
- Due date: i default dei target cambiano il 18/11, quelli del tokenizer il 24/11. Nessun comando o piano registrato nato fra le due
  date è senza le opzioni del tokenizer (cercati in `results/`, `docs/`, `scripts/` e nei rami delle linee 1, 2 e 3).
- Un comando con `--window-ms 25` funziona solo con la CLI di oggi: i checkout congelati si usano con il comando registrato, senza
  modifiche. Un comando storico rilanciato a mano su codice nuovo senza le opzioni diventa 100 ms + GELU: lo proteggono solo la riga
  iniziale del trainer, `config.json` e questa regola.
- **Un piano nuovo scrive in modo esplicito le quattro opzioni** (`--rms-window-ms`, `--ms-fast-window-ms`, `--window-ms`,
  `--tokenizer-activation`) e, con JEPA spenta, `--masked-loss-radius`. Senza, il trainer lanciato dal piano usa i default nuovi,
  mentre `calibration_select` lo legge come storico e rifiuta il run dopo il training: vale anche per un piano che scrive solo le due
  finestre dei target, coerente fino al 23/11. `wsd_chain.py` ora avvisa quando genera i comandi; un rifiuto prima del lancio resta
  da decidere (era già lasciato a GPT il 18/11 per le due finestre).
- Piani, configurazioni, checkpoint e risultati registrati non si riscrivono. I checkout congelati e i rami delle campagne in corso
  (linee 1, 2 e 3) restano sul loro codice e non ricevono `main` prima della fine: i loro sigilli coprono `run.py` e `train_jepa.py`,
  e già oggi non coincidono con `main`.
- Non toccati perché legati a campagne chiuse: `arms_collaudo.py` e `anchor_candidates.py` non controllano il tokenizer del run che
  leggono; `spectral_step1_smoke.sbatch` è rifiutato dal 18/11 con la calibrazione storica.

## Verifiche

`tests/cpu/test_tokenizer_default.py`, 38 casi:
- preset per codice e per CLI (`sanity`, `small`, `30m`, `100m`, `300m`, con e senza le regole della finestra 1): 100 ms e GELU; rispetto
  alla ricetta storica cambiano solo i due campi; passo di 25 ms, 40 token al secondo, 5.184 parametri in più;
- `FMConfig` resta storica; un checkpoint senza i campi (25, 100 e 150 ms) si rilegge senza attivazione con `fm_features.load_model` e
  `random_checkpoint`;
- matrice delle opzioni della CLI, rifiuto delle finestre più corte del passo, `historical_args` (indipendenza, abbreviazioni, `=`,
  idempotenza);
- comandi e piani **veri**: T25-L, T100-L e T100-G (19.618) tornano la configurazione eseguita e salvata; il piano del pilot degli
  obiettivi dà R0-R2 a 25 ms e R5 a 150 ms, senza attivazione; A/B dello Step 1, 2C e pilot dei rami restano storici;
- trainer vero: ripresa di un checkpoint storico rifiutata senza toccarlo e ammessa con la ricetta esplicita, `--init-from`, ancora RVQ,
  errori di rami e log-quadratico; la trappola dei piani nuovi (`calibration_select`).

`test_ddp_equivalence.py` gira su entrambi i tokenizer; `test_spectral_ab.py`, `test_ladder_presets.py`, `test_pose_b19618_launcher.py` e
`test_wsd_chain.py` hanno controlli in più; i test nati sul tokenizer storico lo scrivono in modo esplicito. Test raccolti: 3.516 su
`main` `dfc38b55`, 3.559 nel ramo. Esito della suite CPU completa nel registro. Nessun job, nessun lancio, nessun checkout, piano o
checkpoint congelato modificato.
