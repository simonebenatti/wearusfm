# notes.md — mappa per il driver diagnostico (D0/D1) — revisore driver_map
Inizio 22/11/2026 11:23. Sola lettura di tree_e8c98244; scritture solo qui.

## 1. Prime letture (dimostrato dal codice, file:riga in tree_e8c98244/scripts/pose_temponet)
- runner.run_stage (runner.py:208) stage "controls": train.configure_numerics() (223), P.numeric_context("gpu") (224),
  _calibrations(receipts) -> torch.load(weights_only=False) dei due calibration.pt (38-44, 98-102),
  cache = P.read_verified_inputs(plan,"train",stage="controls") (304) [depth=None: restano "lat" e "pre"],
  bundle = workflow.prepare_control_folds(cache, calibrated, job["world"], allowed_train_users=plan["producers"]["train"]["users"],
  source_identity=_input_identity(plan), expected_folds=previous["train"]["folds"]) (308),
  result = workflow.run_selection(bundle, job["head"], device="cuda", on_checkpoint=_progress) (309), _save_results (317).
- Ricevute archiviate (launcher/TCNH-controls-temporal-*.json): torch 2.2.0a0+git6c8c5ad, numpy 1.26.4, python 3.11.6, cuda 12.1, A100;
  artefatti: .../pose_temponet_20261117/controls/temporal/TCN_post/selection_states.pt 18.386.382 byte sha f9f4d4ad...,
  L0 2.197.650 byte sha 7e6cf069..., MLP 14.685.978 byte sha bc45c2aa...;
  .../wls_calibration/0/calibration.pt 228.630.380 byte sha d08a804e..., /1/ 228.396.140 byte sha 441f6486...

## 2. Inizializzazione: NON riproducibile sul Mac (dimostrato, 11:27)
Comando: python -c "from pose_temponet import heads; m=heads.make_head('TCN_post',seed=0); print(heads.state_sha256(m))" (ambiente del compito, torch 2.14.0, numpy 2.3.5, arm64)
- Mac: TCN_post 19ebce0e51b161afdea3f3d3382db46f1f11eebc33be96169ad3fb68d7e5e9d1 ; archivio cluster (selection_temporal_TCN_post.json initial_sha256_per_fold) 50fb2294f6beb2b54917e99bcb81683ffc24a41bd69ce36b17c08cc53279aab5 -> DIVERSI
- Mac: MLP 1427ef3b... ; archivio 024fde34... -> DIVERSI
- Mac: L0 39cee3fe... = archivio 39cee3fe... (pesi tutti zero/uno: nessun RNG)
Conseguenza: D1 (stessa inizializzazione) si puo' fare solo su Leonardo (torch 2.2.0a0+git6c8c5ad, x86); il driver deve fermarsi se
Trajectory.initial_sha256 != 50fb2294... . D0 non dipende dall'inizializzazione (load_state_dict strict).

## 3. Archivio selection_temporal_TCN_post.json (numeri letti, non ricalcolati)
- selected lr 0.003 exposure 31.0 mse 1.8001448213440414 edyn 0.7159262203822878; fold_steps [899, 930]
- steps_per_epoch 29 (fold0: 961 finestre fitting, 41 senza supporto -> 920) e 30 (fold1: 946, 12 -> 934)
- curve: 1e-4 horizon 300 (304 punti); 3e-4, 1e-3, 3e-3, 1e-2 horizon 150 (154 punti ciascuna)
- primi punti di ogni curva: (0,[0,0]) (0.25,[8,8]) (0.5,[15,15]) (0.75,[22,23]) (1,[29,30])
- thresholds [0.5437424309221641, 0.5374366720436082]; selected_outer_statistics window_count 934 (fold0) e 920 (fold1)

## 4. Ambiente del job originale (letto dalle ricevute e da controls_commands.txt, 11:36)
- CODE_DIR=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pose_temponet_20261121_A1 @ edebfb6c290cf661989c5cff45916bc6a93e607e (reviewed a1ae9607...)
- piano: /leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/pose_temponet_20261117/manifests/b334ece870bc85b804708f92e4c240f4999750ef8d089ca7df1bd040d2ee04e3.json
  (= results/infra/pose_temponet_execution_20261121_A1/plan_controls.json, sha256 ricalcolato b334ece8..., 76.657 byte)
- cache train: /leonardo_work/.../consolidation_long_20261105/49046/pose/train/train.npz (474.412.856 byte, sha 30c0b50d...), 1907 finestre, 40 utenti
- riferimento target train su SCRATCH: /leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/pilot_posa/B19618_20261028/B/pose/train_subset.npz (sha c7bc296e...)
  pose_stats su SCRATCH: .../data/processed/emg2pose_pose/pose_stats.json (sha 757f2aee...)  -> read_verified_inputs li verifica tutti e tre
- job controls temporal TCN_post 65035631: 8 CPU, 1 GPU A100, 32G, limite 05:11:00, durata 4383,3 s; sbatch: module purge; profile/deeplrn; cineca-ai/4.3.0;
  venv $WORK/venv_wearusfm; PYTHONPATH=$CODE_DIR/src:$CODE_DIR/scripts; OMP/OPENBLAS/MKL_NUM_THREADS=8; CUBLAS_WORKSPACE_CONFIG=:4096:8
- chiavi del dizionario cache (consolidation_long_pose.read_cache:167-175 + data.read_cache_pin:94-119): pre, lat, key_time_valid, pred, target, mask, std,
  pose_var, meta, config, has_head, dim, n_recordings, lengths, central_mask, input_verification (depth=None: restano lat e pre; niente "features")
(correzione: l'ora della sezione 4 era 11:27, non 11:36)

## 5. Fixture piccola e prototipo (dimostrato, 11:28-11:30; script proto_fixture.py e proto_driver.py, uscita in proto_driver.out)
Fixture: _cache di tests/cpu/test_pose_temponet_e2e_chain.py:29-60 con 40 utenti x 1 finestra (160x384 FP16, 6 finestre lunghe 150, una resa senza
supporto), due calibrazioni VERE controls.build_positive_control_fold(block_windows=32) -> entrambe "calibrated" (35,7 s), combined gate "diagnostic",
W._prepare_control_folds(..., expected_user_count=40), W._run_selection(bundle,"TCN_post",device="cpu",recipe=_TestRecipe(3,6,1,4)) (11,0 s),
salvataggio con torch.save come provenance.atomic_torch. Poi proto_driver.py:
- calibration.pt: weights_only=True FALLISCE (array numpy nel pickle), weights_only=False ok; selection_states.pt: ok con entrambi (torch 2.14; non provato su 2.2).
- y e y* ricostruiti da calibration.pt ricaricato: uguali bit a bit a bundle.targets / clean_targets e agli hash target_sha256 / clean_target_sha256
  del selection.json; bundle.manifest ricostruito == selection["bundle"] (bundle_sha256 uguale); feature_array_sha256 ed evaluation_identity uguali.
  NB: y - y* NON e' uguale bit a bit a sigma*eps (scarto massimo 1,78e-15): il rumore si prende da sigma*components["epsilon"], non per differenza.
- D0: 10 coppie (5 LR x best/current) ricaricate con heads.make_head + load_state_dict(strict) e W._score_model(..., include_clean=True):
  10/10 punti pooled (MSE, E_dyn) uguali BIT A BIT ai punti delle curve; per il punto scelto uguali bit a bit selected_outer_statistics,
  selected_outer_clean_statistics e selected_fitting_diagnostics.
- Trajectory.restore: accettato nello stesso contesto, rifiutato con torch.set_num_threads(2) ("trajectory provenance/recipe mismatch").
- Blocchi: predizioni FP32 CPU con blocchi da 20 e da 3 NON uguali bit a bit (scarto massimo 1,01e-6); W._score_model con BATCH 3 o 7 al posto di 32:
  mse_sum -3,47e-8, dynamic_sse -1,86e-4 (assoluti) rispetto a BATCH 32.
- D1: cancello di parita' (5 punti, esposizioni 0/0,25/0,5/0,75/1) riprodotto bit a bit con Trajectory + S.checkpoint_schedule + S.pool_scores;
  fit su y* (dataclasses.replace(bundle, targets=bundle.clean_targets) -> W._training_set) con valutazione su y: stesso initial_sha256, stesso
  epoch_order_sha256, stesso stato RNG dropout e shuffle del fit su y; dataset_sha256 diverso; pesi diversi; callback on_checkpoint salva gli stati.

## 6. Ricalcolo del cancello dai JSON archiviati (dimostrato, 11:31)
R = (E_cleanLS - E_TCN)/(E_cleanLS - E_oracle) da calibration_{0,1}.json (outer_reference_statistics.temporal) e selection_temporal_TCN_post.json:
fold0 0.24173052888786836, fold1 0.1887545039561508, pooled 0.21710109906955868 = identici a positive_controls_gate_20261121.json.
E_cleanLS 0.7104311505676322 / 0.7499203961102027 / 0.7284790286546209; E_oracle 0.6534505840205822 / 0.6911028974346952 / 0.6706589239838878;
E_TCN 0.6966572080798834 / 0.7388183283237657 / 0.7159262203822878. a=0,4 sigma=1,3 alpha=1,0 nei due fold; soglie 0.5437424309221641 e 0.5374366720436082.
NB: il denominatore dinamico del fold1 differisce all'ultimo bit fra riferimenti (2698827.390194636, job CPU, una chiamata) e testa
(2698827.3901946363, somma di blocchi da 32): confronti dei denominatori con tolleranza, non con ==.
Bersagli di parita' D0 (TCN_post, pooled, esposizione best / finale): 1e-4: 181 [5249,5430] / 300 [8700,9000]; 3e-4: 64 [1856,1920] / 150; 1e-3: 31 [899,930] / 150;
3e-3: 31 [899,930] / 150 [4350,4500]; 1e-2: 93 [2697,2790] / 150. L0: 4 curve (1e-4 e 3e-4 a 300); MLP: 4 curve (tutte a 150).
Cancello D1, LR 0.003: (0,[0,0]) mse 2.892463349558587 edyn 1.0; (0.25,[8,8]) 2.1452115131878022 / 0.8222974023158734; (0.5,[15,15]) 2.020721989079718 /
0.7844989085655731; (0.75,[22,23]) 1.9583036504913476 / 0.7692090899847309; (1,[29,30]) 1.9309589037808752 / 0.7606302236459029.

## 7. Controlli mirati (dimostrato, 11:32-11:35; proto_checks.py -> proto_checks.out)
- Ordine degli import: dopo `import pose_temponet.provenance` numpy, scipy e torch sono gia' caricati (data.py:8 numpy -> consolidation_long_pose.py:14-15
  numpy poi torch); sklearn entra solo con P.numeric_context() (provenance.py:508-509). Il driver importa PRIMA il pacchetto congelato e poi il resto.
- P.numeric_context("cpu") funziona sul Mac (threadpools libopenblas, libomp x2); P.verify_numeric rifiuta ("unfrozen numerical thread settings").
- T.configure_numerics() con OMP_NUM_THREADS=1 -> ValueError "OMP_NUM_THREADS must be 8 for production benchmark and fits": sul Mac si usano solo
  le funzioni private (_run_selection, classi) con _TestRecipe; su Leonardo serve l'ambiente a 8 thread.
- Blocchi (CPU, torch 2.14): blocchi da 1,2,3,7 danno fra loro gli stessi bit; blocchi da 16/19/20 un altro insieme di bit (scarto massimo 1,01e-6);
  scarti relativi sulle statistiche con BATCH 3 o 7: mse_sum -3,66e-10, dynamic_sse -1,15e-9; con BATCH 16: +4,2e-11 e +5,7e-11.
  Nel vero: 946 % 32 = 18 e 961 % 32 = 1 -> l'ultimo blocco ha 18 finestre o 1 finestra: stessi blocchi di W._score_model, o niente parita' a 1e-12.
- Chiavi di candidate_states: float (0.0001, 0.0003, 0.001, 0.003, 0.01); states["selection_sha256"] == selection["selection_sha256"];
  selection_sha256 e payload_sha256 della coppia si ricalcolano con T.payload_sha256; punti della coppia == curva del JSON.
- P.source_manifest(tree_e8c98244) == plan_controls.json reader.source_manifest (247 file, digest 9aaf1c8b...): i sorgenti congelati del commit
  e8c98244 sono identici byte per byte a quelli del piano. P.atomic_torch scrive anche sotto un'altra radice e rifiuta la seconda scrittura
  (FileExistsError); P.safe_output_root accetta solo la radice registrata.

## 8. Ultimi controlli (dimostrato, 11:36-11:42)
- pytest sui file train/workflow/controls/scientific_integration/heads/selection del commit: 69 passati in 20,08 s (pytest_subset.out).
- proto_split.py: model.affine(h) + model.deep(h) == model(x) bit a bit; punteggi separati contro L e a*T sostituendo "target" nel pannello;
  oracolo ricalcolato sugli esclusi == outer_reference_statistics dell'artefatto (fixture).
- Ricevute locali byte-identiche a quelle del cluster: sha256 dei file in launcher/ = pin di positive_controls_gate_20261121.json e
  wls_receipt_file_pins.json (TCN_post f95c7a5d..., L0 d299b001..., MLP 81f6c38e..., wls-0 2e78439b..., wls-1 697d52d8...).
- verify_positive_controls_readonly.py (GPT, 21/11) e' il modello di lettura in sola lettura: R._ancestors(plan, E.find_job(plan,"TCNH-selection-L0-20261120")),
  R._calibrations(receipts); 13,76 s sul nodo di login.
- Nota N-1: spia su torch.cuda.manual_seed_all -> controls.generate_components la chiama con [20261118, 20261119] (torch.manual_seed dentro
  fork_rng(devices=[]), controls.py:143,147); heads.make_head non la chiama (usa set_rng_state, heads.py:102). L'RNG CPU globale resta intatto.
  Nessun effetto sulla catena eseguita (calibrazione su job CPU; Trajectory installa e ripristina i propri stati RNG).
- Numeri di riga dei test verificati con grep: e2e _cache 28-59, _cpu_execution 62-87; workflow cache 30, bundle 47, recipe 54, artifacts 75,
  control_bundle 98, test FP16 312-343; train 17-69; scientific manual_statistics 27-56; controls _builder_fixture 145.

## 9. Documento principale: DRIVER_MAP.md (stessa cartella). Non verificato: file veri, CUDA, torch 2.2 (vedi sezione 8 del documento).

## 10. Callback diagnostico (dimostrato, 11:42; proto_callback.py -> proto_callback.out)
Un on_checkpoint che calcola W._score_model(..., include_clean=True) su esclusi e fitting a ogni checkpoint NON altera la traiettoria:
payload_sha256 dello stato finale della coppia identico con e senza callback; curve identiche; held contro y del callback == punti della curva (7/7).
Serve a D1 per avere le curve per fold e contro y*, che l'archivio non contiene (solo pooled contro y).
fixture_out occupa 31 MB: si puo' cancellare (si rigenera con proto_fixture.py in 49 s).
