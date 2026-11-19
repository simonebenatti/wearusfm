# P1 — storico T100-G e celle nuove 100/125 (rapporto incrementale)

Inizio: lettura del diff e388a1de..0efdc29c nella copia a 438c8d65.

## a) config_from_dict e with_legacy_target_windows

- Codice (438c8d65): scripts/contrastive_prepare.py:46 `d = copy.deepcopy(R.with_legacy_target_windows(data))`;
  src/wearusfm/training/run.py:220-227 `with_legacy_target_windows` aggiunge SOLO loader.rms_window_ms=25 e
  loader.ms_fast_window_ms=50 se mancano (merge `{25, 50, **d["loader"]}`: un campo presente vince); NON tocca target_version.
  RunConfig.target_version ha default TARGET_VERSION = v3 (run.py:114, spectral_targets.py:11).
- Prova sui config veri (script tmp/p1/scripts/a_roundtrip.py, config_from_dict -> config_to_dict):
  - T100-G (sha f9d03184...): raw senza i due campi, con target_version v1 -> (25.0, 50.0, v1); differenze aggiunte:
    loader.rms_window_ms=25, loader.ms_fast_window_ms=50, model.tokenizer_logsquare=False, model.tokenizer_logsquare_eps=0.001
    (le ultime due sono default di FMConfig, gia' presenti nei config storici della coorte: nessun effetto nuovo).
  - configs/ storici (C0/CN/CR/calibration) e coorti calibrated_cohort_20261117, _20261118_mmap, _20261118_bf16 (36 file):
    tutti senza i due campi, con target_version v1 -> (25.0, 50.0, v1); uniche differenze i due campi aggiunti.
  - review_configs nuovi (C0/CN/CR/calibration): campi espliciti -> (100.0, 125.0, v3), round trip identico (diffs: []).
- Casi sintetici (tmp/p1/scripts/a_synthetic.py, base T100-G modificata):
  - senza target_version -> (25, 50, v3): incoerente (la guardia del trainer run.py:1248 lo rifiuta; build_review_configs lo rifiuta,
    prepare.py:115-117). Non raggiungibile con gli artefatti della linea 3: T100-G e le 36+4 config storiche hanno target_version v1 esplicito.
  - solo rms=100 -> (100, 50, v1) coerente con la regola «campo spettrale assente = 50 ms»; solo fast=125 con v1 -> incoerente, la guardia lo rifiuta.
  - base gia' 100/125/v3 o senza tv -> build_review_configs solleva «baseline target windows/version are not the historical T100-G 25/50 recipe».

## b) config nuovi vs storici
- Diff campo per campo (tmp/p1/scripts/b_diff.py): review_configs/{C0,CN,CR,calibration} vs contrastive_comparison_20261116/configs/*:
  esattamente 3 differenze per file: +loader.ms_fast_window_ms=125, +loader.rms_window_ms=100, target_version v1 -> fast125_v3.
  Nessun campo nuovo o default cambiato portato dal merge di main. Rispetto alla coorte bf16 (576): in piu' solo lambda null e max_steps 19618 (attesi).
- Rigenerazione (tmp/p1/scripts/b_regen.py): build_review_configs(T100-G) == file salvati byte per byte (json indent 2 + \n),
  sha256 e bytes coincidono con preparation.json; invertendo i 3 campi dichiarati si ottengono esattamente i config storici.

## c) dispatcher e modalita'
- tmp/p1/scripts/c_dispatch.py: validate_recipe(nuova calibration,'calibration') -> (100,125,v3); config storica, nuova senza finestre,
  nuova senza ms_fast, nuova con tv v1 -> tutte RIFIUTATE («configuration differs from the frozen comparison recipe»).
  Nuova senza target_version -> ACCETTATA con v3 di default (validate_recipe confronta il config ricostruito, non il file):
  ricetta giusta, ma il file non e' quello pinnato; i pin esatti a valle (audit config_sha256 = hash del file, CUDA
  config_canonical_sha256, coorte `_read(candidate) != cfg`, release config_sha256) lo fermano. Candidata NOTA (preesistente).
- CUDA: technical_config(base nuova, CN/CR) e le config degli stop-check conservano (100,125,v3); cambiano solo min/max_window_s 1/4.
- calibrated_configs (auth simulata) per 544/576/5000/19618 e `calibrated-configs` main (file split/5000/19618/576): tutti (100,125,v3).
- Nessun LoaderConfig() di default, _config_with_defaults, with_historical_targets o historical_args nei moduli contrastive_* e
  nei diagnostici (grep); l'unico punto e' contrastive_prepare.py:46/68.
- Chiamanti di config_from_dict (grep scripts/ src/ tests/): prepare.py:114 (T100-G storico -> 25/50/v1, guardia 115-117, poi 100/125/v3
  esplicito), prepare.py:273 (config.json della calibrazione: nuova esplicita invariata; storica -> 25/50/v1 != atteso -> rifiutata),
  contrastive_run.py:47 (config del job: nuova ok, storica rifiutata), contrastive_cuda_check.py:228 (config del job gia' validata dal
  dispatcher come calibrazione nuova; technical_config conserva 100/125/v3), contrastive_benchmark_readout.py:381 (config.json del run nuovo,
  gia' == atteso; usa solo cfg.model), diagnostics/contrastive_common_checkpoint.py:161/197/287 e contrastive_common_state_cuda.py:242
  (artefatti storici bf16: lettura 25/50/v1 corretta; a HEAD falliscono comunque chiusi sul digest: source_digest HEAD=643a8937... !=
  SOURCE 46938f31..., digest di 365cb1e = 46938f31... ricalcolato), tests/cpu/test_contrastive_prepare.py:68/75/79/86/96 (sintetici).
- Digest: 0efdc29c e 438c8d65 = 643a8937e833... (= preflight_plan.source_sha256): il merge dei diagnostici non cambia il digest.

## d) lettori congelati
- Ricevute storiche: `git diff --name-status --diff-filter=MDRT 8905e346 438c8d65 -- results/infra/contrastive_comparison_20261116/
  results/infra/tokenizer_execution_20261103/19618/T100-G/` vuoto (solo aggiunte successive); il diff della consegna non tocca quella cartella.
- pytest dei 4 file toccati (historical_calibration, historical_callsites, job, prepare): 86 passed in 7,86 s.
- Run storici (tmp/p1/scripts/a_runs.py): tutti i 6 config.json di run in contrastive_comparison_20261116 (calibrazione 17/11,
  3 benchmark e 2 riprese bf16) senza i due campi, tv v1 -> (25, 50, v1).
- Altri 7 file contrastive (run, readout, audit, cuda_check, source_compat, mmap_compatibility, callsite_oracle): 91 passed in 18,77 s.
- Plan nuovo vs storico (tmp/p1/scripts/b_plan_diff.py): solo aggiunte dichiarate (finestre/cli/tv nei bracci e common, amendment,
  target_recipe_amendment, interpretazione del moltiplicatore, decisione target_windows, scope e una limitazione); input_sha256,
  riferimenti di copertura e gradiente invariati. sha del plan 60818326... = pin del preflight_plan.
- Codice di training cambiato dal merge (8905e346 -> 438c8d65, ambito del digest): solo finestre/versione target
  (loader, anchor_targets, spectral_targets, run, spectral_calibration, spectral_step1, spectral_branches commento) + harness di valutazione
  e train_jepa CLI; nessun altro default del percorso contrastivo.
- NOTA test: a HEAD `test_modified_historical_receipt_is_rejected[weight|selection|reader|source]` e
  `test_historical_metadata_must_be_byte_exact` passano perche' il lettore corrente rifiuta gia' per «calibration run config differs»
  (tmp/p1/scripts/d_vacuous.py: anche la ricevuta NON modificata e' rifiutata con lo stesso messaggio). Non esercitano piu' la
  manomissione; nel lettore congelato sono coperti solo audit e source del summary. Copertura della manomissione sul percorso nuovo:
  test_contrastive_prepare.py:224-237 (sintetico 100/125).
- Lettori congelati, verbose: 10 passed (test_original_calibration_authenticates..., test_only_exact_historical_audit_is_compatible,
  test_pinned_historical_receipt_still_requires_authenticated_recalculation con run_frozen a 8905e346; digest atteso 46938f31...).
  Nessun test rimosso nei 4 file (funzioni: 4->6, 3->3, 11->12, 13->18). Nessun test fa riferimento alle evidenze bf16 (prima e dopo).
- NOTA: run_frozen archivia anche le evidenze da 8905e346, non legge la copia di HEAD: i test provano la riproducibilita' a 8905e346;
  l'integrita' della copia corrente la prova il git diff (nessuna M/D/R).

## Esito P1
Nessun BLOCCANTE, nessun MINORE. Quattro NOTE: (1) validate_recipe accetta un file senza target_version (default v3, ricetta giusta,
fermato a valle dai pin esatti); (2) with_legacy_target_windows non completa target_version (pre-07/10 -> 25/50+v3 incoerente,
rifiutato; irraggiungibile in linea 3); (3) test storici a HEAD diventati vacui; (4) run_frozen non verifica la copia corrente delle ricevute.
Pulizia: basetemp pytest cancellati, archivi git temporanei cancellati; clone pulito a 438c8d65.
