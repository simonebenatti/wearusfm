# Prompt per GPT — review della consegna di Claude: controllo della convergenza delle sonde B19618/R3c (29/10/2026)

Simone ti chiede la review tecnica e scientifica della consegna che hai richiesto con `docs/prompt_claude_convergenza_sonde_20261029.md`
(tuo commit b1c1bda). Ruoli: il codice l'ha scritto Claude, tu lo rivedi, semplicione esegue i lanci dopo la tua review. Nessun job e' stato
sottomesso.

## Dove sta

Ramo `origin/worktree-claude-training-engine`, due commit:
- **22bf2cd** — codice, test, protocollo, pin degli ingressi;
- **b98ece0** — piano e comandi per semplicione (generati dallo SHA 22bf2cd), registro (sezioni 1 e 2), `docs/decisioni.md`.

File da leggere, nell'ordine:
1. `docs/sonde_convergenza_20261029.md` — protocollo eseguibile, schema dell'audit, procedura a cinque blocchi, preventivo, obiezioni dichiarate.
2. `src/wearusfm/harness/probe.py` — `linear_probe(max_iter, audit, context)`, `solver_recipe()`, cattura e riemissione dei warning per fit.
3. `scripts/channel_readout.py` — `oof_predictions(max_iter, audit, folds, context)`, `load_cache` (estratta da `probe()`), regola della firma del
   solver in `check_readout_pairing`.
4. `scripts/probe_convergence.py` — fasi `probe`, `aggregate`, `parity`, `gate`, `compare-phases`, `plan`; bootstrap congiunto (`joint_bootstrap`,
   `contrast`, `phase_guard`).
5. `scripts/slurm/probe_convergence.sbatch`, `results/infra/pilot_posa/sonde_convergenza_20261029/{pin_ingressi.json,piano.json,comandi.txt}`.
6. `tests/cpu/test_probe_convergence.py` (10 test) e le voci del registro del 29/10 (sezione 1: consegna; sezione 2: limiti dichiarati).

Comando dei test sul Mac: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ~/.venvs/wearusfm-tok/bin/python -c "import sys; sys.path.insert(0, 'src');
import pytest; sys.exit(pytest.main(['-q', 'tests/cpu/test_probe_convergence.py', 'tests/cpu/test_harness_probe.py', 'tests/cpu/test_channel_readout.py']))"`
(dal worktree di Claude; 25 test, circa 2 min 15). Ambiente del Mac: sklearn 1.7.2, numpy 2.3.5; sul cluster sklearn 1.4.1.post1.

## Che cosa ti chiedo di verificare

1. **Default invariati.** Con `audit=None` e `max_iter=3000` la sonda fa le stesse chiamate di prima, nello stesso ordine; `probe()` di
   `channel_readout.py` non cambia uscita (i test di uguaglianza con `calibration_probe` restano verdi). Dimmi se vedi un punto in cui
   l'audit puo' cambiare un numero.
2. **Attribuzione dei warning.** La cattura e' attorno alla sola chiamata `.fit` (`catch_warnings(record=True)` + `simplefilter("always")`), i
   warning sono riemessi subito dopo con `warn_explicit` (quindi nei log in audit possono comparire piu' righe di prima, mai meno); la metrica
   e il bootstrap sono contati a parte (`scoring_warnings`, `bootstrap_warnings`); un'eccezione resta un'eccezione. Giudica se il record per fit
   contiene tutto quello che il tuo prompt elenca (§2) e se manca qualcosa.
3. **Shard e aggregato.** Ogni fold e' indipendente (scaler, C e refit dentro il fold), quindi lo shard per fold e l'aggregato in ordine di fold
   riproducono esattamente la sonda intera: e' il test `test_shards_aggregate_to_the_whole_probe_...` e la parita' sul «storico» finto. Confermi
   che non c'e' uno stato condiviso fra fold che io non abbia visto (seme della PCA non usato a k = 0; `bootstrap_bacc_se` per fold con seme
   fisso; `mean_bacc_bootstrap_se` solo nell'aggregato)?
4. **Parita' e cancello.** Parita' senza tolleranze (righe, soggetti, ruoli, C, predizioni, S) e cancello chiuso a qualunque differenza; una
   discrepanza si studia prima della fase B. Va bene per te, o vuoi un trattamento esplicito di una differenza «di un ulp» (oggi = cancello chiuso)?
5. **Bootstrap congiunto.** `joint_bootstrap` ricalca `arms_compare.s_comparison` (seme 20261009, dataset in ordine alfabetico, una estrazione per
   replica condivisa dai quattro rapporti) e il job di confronto verifica che B−R3c di ciascuna fase coincida col confronto ufficiale entro 1e-12.
   La differenza dei delta e le variazioni per encoder sono contrasti sulle stesse repliche. Controlla la definizione dei contrasti in `CONTRASTS`
   (ordine R3c-A, B-A, R3c-B, B-B).
6. **Firma della ricetta del solver.** `solver_recipe()` = `get_params()` senza C, griglia, regola di scelta, refit, scala, versioni;
   `check_readout_pairing` la esige uguale quando c'e' in uno dei due rapporti (legacy senza firma -> rifiuto); `phase_guard` ammette fra le fasi
   la sola differenza di `max_iter`. Dimmi se la firma deve includere altro (thread, dtype) o se un campo va escluso.
7. **Pin degli ingressi.** `pin_ingressi.json` viene dalle tue provenienze in f7433f4 (cache, provenienze, ricevute, rapporti e OOF originali,
   checkpoint). Verifica che gli SHA siano quelli e che i percorsi di `--features`, `--original` e `--splits` nel piano siano giusti.
8. **Piano e preventivo.** 30 job in 5 blocchi (<= 10 per blocco, prerequisiti verificati dai job, non da `afterok` fra blocchi; `afterok` solo
   fra i 4 aggregati della fase B e il confronto); tetti 50,83 core-ore (A) + 118,33 (B), 0 GPU-ora; shard di DB2 divisi in due per le 4 ore
   della seriale. Il tetto di B e' il caso peggiore (ogni fit al limite x4): dimmi se lo vuoi affinare dopo la fase A, come propongo, o fissarlo
   ora. Categoria del ledger `--agente gpt` con operatore semplicione, come nel tuo piano di recupero: confermala o cambiala (`--agente`).
9. **Procedura.** Dopo la review: deploy `deploy/pilot-posa-sonde-20261029` dal commit definitivo, `campaign_checkout.sbatch`, rigenerazione del
   piano con lo SHA definitivo (`python3 scripts/probe_convergence.py --stage plan --code-sha <sha> --pins
   results/infra/pilot_posa/sonde_convergenza_20261029/pin_ingressi.json`), lanci di semplicione blocco per blocco. Se la tua review chiede
   modifiche al codice, le faccio io sul mio ramo e rigenero il piano: lo SHA cambia.

## Cosa NON ti chiedo

Nessuna modifica al disegno scientifico del tuo prompt (gia' seguito alla lettera: due fasi, solo `max_iter` fra le fasi, griglia e refit rifatti
per entrambi gli encoder, nessuna selezione a posteriori), nessun lancio prima della review, nessun nuovo training, loss, PSD o latenti non mediati.

## Cosa riportare a Simone

Verdetto (approvato / da rivedere), i rilievi bloccanti e non con file e riga, le conferme sui punti 4, 6 e 8, e se il piano va rigenerato.
Scrivi la tua voce nel registro sul tuo ramo come al solito.
