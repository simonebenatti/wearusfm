# Verifica scettica n. 2 del rilievo R5-3 (r5-test)

Avvio: 2026-11-20 04:34:38

## Progressi

- Clone temporaneo `rev_runtime/tmp/v53b_clone` a 0efa4d45 (resume.py identico a fbfde797: `git diff --stat fbfde797 0efa4d45` tocca solo delivery_pin.json).
- Base senza modifiche: `tests/cpu/test_contrastive_target_resume.py` -> `18 passed in 79.98s`.
- Riferimenti al modulo nei test (git grep a fbfde797): solo test_contrastive_target_resume.py chiama davvero compare_three_paths/validate_fork/prepare_reload;
  test_contrastive_targets_readout.py:131 sostituisce compare_three_paths con un finto; test_contrastive_mmap_compatibility.py:19 usa solo `checkpoint`.
  Nessun test chiama contrastive_run.validate_target_fork (unico chiamante di produzione di validate_fork con ranks=4).
- Mutazioni commutabili con variabile d'ambiente V53B_MUT (script `rev_runtime/tmp/v53b_scripts/apply_guards.py`, guardia davanti all'istruzione originale; senza variabile il comportamento e' quello di fbfde797).
- Mutazioni singole su test_contrastive_target_resume.py (18 test): boundary(366), rng_boundary(371), start_json(329), endpoint(340), row_inv(380),
  weights(385), trace_metrics(391), fork_rankrng(82), copy_bytes(105) -> tutte `18 passed`; extra non citate: counters(344), origin_map(354),
  origin_pin(356), origin_trace(388) -> `18 passed`. Controlli del metodo: support_missing(394) -> `1 failed, 17 passed`; pair_diff(402) -> `4 failed, 14 passed`.
- Tutte le 13 mutazioni sopravvissute attive insieme sulla selezione completa di 29 file della ricevuta validation_final -> `465 passed in 175.87s`.
- Mutazione letterale del rilievo (riga 365 -> `if False:`, codice originale senza guardie) -> `18 passed in 17.13s`.
- Tutte le chiamate nei test usano ranks=1 (8 occorrenze); i chiamanti di produzione usano ranks=4 (contrastive_run.py:265, :344; readout:487).
- Ridondanze di produzione cercate: _run_evidence del readout (testato, test_contrastive_benchmark_readout.py:279, test_contrastive_targets_readout.py:204-244)
  ripete endpoint/contatori/pesi programmati per continuo e reload; contrastive_run.py:348 ricontrolla lo SHA della copia. Nessun altro controllo
  di start.json, confine o tracce fuori da compare_three_paths (git grep su readout/run/job/release).
- Sonde negative mie (copia in `rev_runtime/tmp/v53b_scripts/test_v53b_negative.py.txt`, 9 test sulla fixture common_paths e su un checkpoint sintetico a 4 rank):
  codice originale -> `9 passed in 18.75s` (i controlli scattano davvero). Con una mutazione per volta -> ogni mutazione `1 failed, 8 passed`,
  e fallisce proprio la sonda corrispondente. Lo spostamento comune di rng_cpu_before su trace[0] nelle tre fasi lascia vuote le differenze
  di coppia: solo il controllo RNG al confine lo vede (non e' ridondante).
- Clone temporaneo e basetemp cancellati; copia di sola lettura intatta (status pulito, HEAD 0efa4d45).

## Esito

**Non confutato. Gravita' confermata: MINORE.**

- Riprodotto in modo indipendente: nessun test della selezione da 465 fa fallire la rimozione dei rifiuti elencati; negli altri file di tests/cpu git grep non trova chiamate al modulo.
  Le guardie uniche, senza un doppione testato altrove, sono: stato al confine, RNG al confine, start.json, invarianti di riga,
  traccia = metriche, rank_rng con ranks>1 in validate_fork.
- Non e' BLOCCANTE: le sonde mostrano che il codice attuale rifiuta correttamente le evidenze alterate. Il difetto sta nella copertura, non nel
  comportamento. Precedente: la M-1 della review 94f5094 («Tre barriere di secondo livello senza test») era classificata minore.
- Attenuanti, che non confutano il rilievo: endpoint e contatori (333-344) e pesi programmati sono ripetuti e testati in `_run_evidence`
  (readout 201-261), chiamato sui tre percorsi prima di compare_three_paths; la copia byte-identica e' ricontrollata in contrastive_run.py:348-349.
- Imprecisione minore del rilievo: test_missing_support (riga 173) altera anche trace[0], ma solo `contrast_rnc_pairs`, non i campi del confine
  o dell'RNG; la sostanza resta.
- Correzione fattibile su CPU: nove sonde, circa 18 s con la fixture esistente.
