# Mappa operativa — catena diagnostica TCN nuova e separata (22/11/2026)

Revisore: ops_costs. Fonte: copia in sola lettura del commit `e8c9824461b5e18494ff462f1be8abda71b76d5e`
(`/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244`). Nessun accesso al cluster: i percorsi di Leonardo
vengono da ricevute e JSON archiviati, non da letture dal vivo. La copia dell'albero e' parziale (in `docs/` ci sono 5 voci,
mancano registro e prontuario del 19/11): dove serve il repository vero lo dico.

Legenda: **[D]** dimostrato (file:riga, numero ricalcolato o prova eseguita); **[I]** ipotesi o stima.
`L/` = `results/infra/pose_temponet_execution_20261121_A1/launcher/`.

---

## 1. Sequenza usata dall'operatore per A1

### 1.1 Riferimento di deploy
- **[D]** Nome: `deploy/pose-temponet-20261121-A1`, D = `edebfb6c290cf661989c5cff45916bc6a93e607e`, «senza spostare il vecchio deploy»
  (`docs/pose_temponet_execution_A1_20261121.md:34`). Esito registrato: `"leonardo_deploy_ref_matches": true` (`L/deployment_checks.json`).
- **[D]** Il comando esatto del push NON e' archiviato nel materiale fornito. Restano la forma di CLAUDE.md
  (`git push leonardo deploy/<nome>`) e il controllo del prontuario: `[ "$(git ls-remote leonardo deploy/<nome> | cut -f1)" = "$D" ]`
  (`docs/pose_temponet_execution_20261120.md:110`). Non lo ricostruisco a memoria.
- **[D]** I sei controlli locali prima dell'invio (`docs/pose_temponet_execution_20261120.md:105-110`): antenato, `git diff --quiet C D -- src scripts`,
  hash del piano nel commit, copia della review nel commit, albero pulito, riferimento remoto = D. Per A1: tutti `true`, piu'
  `source_identity_247_files_verified` e `submitter_bytes_identical`.

### 1.2 Job di checkout (copiato da `L/TCNH-checkout-A1-20261121.state.json`)
```
sbatch --parsable --job-name=TCNH-checkout-A1-20261121 --comment=wearusfm:TCNH-checkout-A1-20261121 --account=IscrB_WearUsFM --partition=lrd_all_serial --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=24G --time=00:12:00 --no-requeue --export=ALL,CAMPAIGN=pose_temponet_20261121_A1,CAMPAIGN_SHA=edebfb6c290cf661989c5cff45916bc6a93e607e --output=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/TCNH-checkout-A1-20261121_%j.out --error=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/TCNH-checkout-A1-20261121_%j.out
```
- Argomenti di `submit_job.py`: `--id TCNH-checkout-A1-20261121 --agente gpt --campagna pose_temponet_20261121_A1 --gpu-ore 0`
  `--script-stdin scripts/slurm/campaign_checkout.sbatch` (sha256 del testo inviato `eaafc432b0047125e53fa8143db6a6366a7a160da74d6e8582c103a07e589fcc`,
  uguale al file dell'albero). Nessun `--dopo`.
- Esito **[D]**: Slurm `65023067`, `COMPLETED 0:0`, 30 s, 2 CPU, `login08`, MaxRSS 205.840K; prenotato 12:11:26, partito 12:11:55.
  Checkout: `/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pose_temponet_20261121_A1 @ edebfb6c290cf661989c5cff45916bc6a93e607e`,
  2.349 file, 95M; `status` e `writable` vuoti (`L/checkout_verified.json`).

### 1.3 I sei controlli (copiati da `L/controls_commands.txt`; cambia solo cio' che e' in tabella)
Riga completa del controllo temporale TCN (job 65035631):
```
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python '/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py' --id TCNH-controls-temporal-TCN_post-20261120 --agente gpt --campagna pose_temponet_20261121_A1 --gpu-ore 5.233333333333333 --script-stdin scripts/slurm/pose_temponet_dispatch.sbatch -- --account=IscrB_WearUsFM --partition=boost_usr_prod --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=05:11:00 --no-requeue --export=ALL,CODE_DIR=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pose_temponet_20261121_A1,EXPECTED_SHA=edebfb6c290cf661989c5cff45916bc6a93e607e,TCN_PLAN=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/pose_temponet_20261117/manifests/b334ece870bc85b804708f92e4c240f4999750ef8d089ca7df1bd040d2ee04e3.json,TCN_PLAN_SHA256=b334ece870bc85b804708f92e4c240f4999750ef8d089ca7df1bd040d2ee04e3,TCN_JOB_ID=TCNH-controls-temporal-TCN_post-20261120,TCN_PLAN_COMMIT=6a20054ad2ffdb3d754e6106775a159a0933d99a,TCN_PLAN_REPO_PATH=results/infra/pose_temponet_execution_20261121_A1/plan_controls.json --output=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/TCNH-controls-temporal-TCN_post-20261120_%j.out --error=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/TCNH-controls-temporal-TCN_post-20261120_%j.out --gres=gpu:1 --qos=normal
```
| ID logico | Slurm | `--gpu-ore` | `--time` | nodo | elapsed sacct |
|---|---|---|---|---|---|
| TCNH-controls-linear-L0-20261120 | 65035622 | 1.5833333333333333 | 01:32:00 | lrdn2183 | 1451 s |
| TCNH-controls-linear-MLP-20261120 | 65035623 | 1.75 | 01:42:00 | lrdn3188 | 1287 s |
| TCNH-controls-linear-TCN_post-20261120 | 65035626 | 5.233333333333333 | 05:11:00 | lrdn3369 | 3710 s |
| TCNH-controls-temporal-L0-20261120 | 65035628 | 1.5833333333333333 | 01:32:00 | lrdn2287 | 1105 s |
| TCNH-controls-temporal-MLP-20261120 | 65035629 | 1.75 | 01:42:00 | lrdn3349 | 860 s |
| TCNH-controls-temporal-TCN_post-20261120 | 65035631 | 5.233333333333333 | 05:11:00 | lrdn0707 | 4427 s |

Risorse comuni **[D]**: account `IscrB_WearUsFM`, partizione `boost_usr_prod`, `--qos=normal`, `--gres=gpu:1`, 1 nodo, 1 task, 8 CPU, 32G, `--no-requeue`.
Script su stdin `pose_temponet_dispatch.sbatch` (sha256 inviato `baed3903…` = file dell'albero). Nessun `--dopo`: inviati dopo aver verificato i predecessori.
Inviati fra 15:21:32 e 15:21:44, partiti alle 15:22:47 (cinque) e 15:23:24 (65035631). Le due calibrazioni (`L/wls_commands.txt`):
`lrd_all_serial`, 8 CPU, 24G, 03:57:00, Slurm 65023397 (422 s) e 65023399 (414 s).

---

## 2. Il launcher `scripts/slurm/pose_temponet.sbatch` (21 righe)

| Riga | Che cosa fa |
|---|---|
| 3 | `set -euo pipefail` |
| 4-6 | pretende `CODE_DIR EXPECTED_SHA WORK TCN_PLAN TCN_PLAN_SHA256 TCN_JOB_ID TCN_PLAN_COMMIT TCN_PLAN_REPO_PATH` |
| 7 | rifiuta `SLURM_RESTART_COUNT` diverso da 0 |
| 8-9 | `source lib_code_dir.sh`; `code_dir_setup`: checkout git, `HEAD = EXPECTED_SHA` (40 cifre), `git status --porcelain --untracked-files=all` vuoto, nessun file scrivibile fuori da `.git`; poi `cd`, `$CODE_DIR/src` in testa a `PYTHONPATH`, esporta `WEARUSFM_CODE_DIR/WEARUSFM_CODE_SHA` (`lib_code_dir.sh:16-65`) |
| 10-13 | `module purge`; `module load profile/deeplrn`; `module load cineca-ai/4.3.0`; `source "$WORK/venv_wearusfm/bin/activate"` |
| 14-15 | `PYTHONPATH="$CODE_DIR/src:$CODE_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"` (anteposto, mai sostituito); `PYTHONDONTWRITEBYTECODE=1` |
| 16-18 | `OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"`, uguale per `OPENBLAS` e `MKL`; `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| 19-20 | `python3 -m pose_temponet materialize-plan --commit … --repository-path … --sha256 … --output "$TCN_PLAN"` |
| 21 | `exec python3 -m pose_temponet run --plan "$TCN_PLAN" --plan-sha256 … --job-id "$TCN_JOB_ID"` |

- **[D] Il comando Python e' fisso.** Lo stadio dipende solo da `TCN_JOB_ID`: `runner.run_stage` cerca l'ID nel grafo dei 24 job
  (`runner.py:210`, `execution.find_job`), pretende piano rilasciato e sorgenti del manifest (`runner.py:213-214`) e rifiuta cartella o ricevuta
  gia' esistenti (`runner.py:220-221`). **Non si puo' riusare per un modulo nuovo: serve un launcher nuovo.**
  Anche `pose_temponet_dispatch.sbatch` chiama in modo fisso `pose_temponet.sbatch` (riga 8).
- **[D]** `lib_code_dir.sh` e' generico e si riusa con `source`.
- **[D] Manifest dei sorgenti** (`provenance.py:236-248`; prova `manifest_probe.py`, uscita `manifest_probe.out.json`):

| Aggiunta | Cambia `source_sha256`? |
|---|---|
| pacchetto `scripts/<altro nome>/*.py` (provato: `scripts/tcn_diag/`) | **no** |
| `scripts/slurm/<altro nome>.sbatch` (provato: `tcn_diag.sbatch`) | **no** |
| `scripts/slurm/pose_temponet_<qualsiasi>.sbatch` | **si'** (glob `pose_temponet*.sbatch`) |
| `scripts/<nome>.py` al primo livello | **si'** (glob `scripts/*.py`) |
| qualunque `src/**/*.py`; qualunque `.py/.sh/.sbatch` in `scripts/pose_temponet/` | **si'** |
| `docs/`, `results/`, `tests/` | **no** |

- **[D]** L'albero e8c98244 ha 247 file nel manifest e `source_sha256 = 9aaf1c8b6a050a524dfb60af65df9406313d4fd0e690dab1097bf11df2ec67ea`,
  identico al piano A1: il commit di GPT non ha toccato i sorgenti congelati.
- **[D]** Precedente gia' eseguito di script esterno al manifest con lo stesso ambiente: `L/gpu_inventory_A1_20261121.sbatch`
  (job 65033157, 20 s, lrdn0058), passato su stdin, con Python in linea.

---

## 3. Contesto numerico `gpu8` e controlli all'avvio

`gpu8 = 12cb2a0b3207b66e260d3ca490db93ee0c029bb8f03d766df89a9edba6b7eb3f` **[D]**: stesso digest nel benchmark 65016447 (lrdn0059),
nell'inventario 65033157 (lrdn0058) e nei sei controlli (sei nodi diversi): otto nodi, un solo digest.

Contenuto del digest (`provenance.py:505-534`; ricevuta `L/TCNH-controls-temporal-TCN_post-20261120.json`):
python 3.11.6, numpy 1.26.4, scipy 1.12.0, sklearn 1.4.1.post1, torch `2.2.0a0+git6c8c5ad`, cuda 12.1;
threadpool censiti: 2 openblas (0.3.21.dev e 0.3.23.dev, SkylakeX, pthreads), 2 libgomp, 1 libomp, tutti a 8 thread;
intraop 8, interop 1, `OMP/MKL/OPENBLAS_NUM_THREADS="8"`; `deterministic=True`, `warn_only=False`, `cudnn_deterministic=True`,
`cudnn_benchmark=False`, TF32 spento, `CUBLAS_WORKSPACE_CONFIG=":4096:8"`; hardware: Xeon Platinum 8358, NVIDIA A100-SXM-64GB, capability [8,0], 68.099.571.712 byte.

Controlli del runtime congelato prima del lavoro (`runner.py:208-229`, `provenance.py:537-601`):
1. piano = hash dichiarato, `TCN_JOB_ID` = `--job-id`, piano rilasciato, grafo valido (`__main__.py:245-250`);
2. `validate_plan` (manifest dei sorgenti, review, A1) e `checkout_identity` (HEAD = `WEARUSFM_CODE_SHA`, manifest, antenato, archivio del commit rivisto);
3. `_allocation`: `SLURM_CPUS_PER_TASK=8`, 1 GPU, 1 nodo, `TimeLimit` = walltime del piano, partizione, `AllocTRES` cpu/gpu/mem=32G,
   `SLURM_MEM_PER_NODE=32768`, `torch.cuda.device_count()==1`, nessun riavvio;
4. `train.configure_numerics()` (`train.py:62-82`), poi `numeric_context("gpu")`, poi `verify_numeric`: thread, flag e digest = pin.

**Che cosa deve riprodurre un driver esterno per predizioni identiche bit per bit a 65035631:**
- a) stessa pila: le quattro righe 10-13 del launcher; nodo di `boost_usr_prod`; 1 GPU, 8 CPU;
- b) stesso ambiente: `OMP/MKL/OPENBLAS_NUM_THREADS=8`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` prima di inizializzare CUDA, `PYTHONPATH` anteposto;
- c) **stessi import prima del censimento**: `from pose_temponet import runner, train, provenance as P` e nient'altro che carichi librerie
  native con thread (il censimento dei threadpool entra nel digest); l'inventario l'ha fatto cosi' (riga 19 dello script);
- d) `train.configure_numerics()` come prima istruzione, poi `assert P.digest(P.numeric_context("gpu")) == gpu8` e
  `assert P.source_identity() == "9aaf1c8b…"`; se il digest differisce: fermarsi, la parita' non e' piu' sostenibile;
- e) stessi dati per la stessa strada: `P.read_verified_inputs(plan, "train", stage="controls")` (senza `depth`), calibrazioni da
  `runner._calibrations`, `workflow.prepare_control_folds(cache, calibrated, "temporal", allowed_train_users=plan["producers"]["train"]["users"],
  source_identity=runner._input_identity(plan), expected_folds=<folds del preflight>)`; poi
  `assert bundle.manifest["bundle_sha256"] == "4e147b1f43f3edf64ac93262d8b0805c762d886cc14d96b7b786c9c867fd60ad"`
  (copre bersagli y e y*, soglie, supporto, partizione) e identita' di valutazione `8fcf86092541d5862a3984e8ec6515a9795aefd97b17be43dc63a9247d2f5b35`
  (copre anche `feature_array_sha256 397527e8…`);
- f) stesso punteggio: `heads.make_head("TCN_post", seed=0).to("cuda", torch.float32)`, `load_state_dict(strict=True)` oppure
  `Trajectory.restore(stato)` (rifiuta dataset, profilo numerico o checksum diversi: `train.py:336-346`), `workflow._score_model(..., include_clean=True)`
  con `BATCH=32` e gli stessi indici `bundle.held[fold]`.

Ancore di parita' gia' sul Mac (`L/selection_temporal_TCN_post.json`), uguaglianza attesa esatta:
- `initial_sha256` dei due fold `50fb2294f6beb2b54917e99bcb81683ffc24a41bd69ce36b17c08cc53279aab5`;
- LR 0,003, prima epoca: esposizione 0 → 2.892463349558587 / 1.0; 0,25 → 2.1452115131878022 / 0.8222974023158734;
  0,5 → 2.020721989079718 / 0.7844989085655731; 0,75 → 1.9583036504913476 / 0.7692090899847309; 1 → 1.9309589037808752 / 0.7606302236459029 (mse / edyn);
- 20 stati TCN (5 LR × migliore e finale × 2 fold) contro i 10 punti aggregati:

| LR | migliore: esposizione → mse / edyn | finale: esposizione → mse / edyn |
|---|---|---|
| 1e-4 | 181 → 1.8181615643874114 / 0.7256444669656075 | 300 → 1.8268597285134838 / 0.7281499412886786 |
| 3e-4 | 64 → 1.8174307513426158 / 0.725382121932922 | 150 → 1.837209816099995 / 0.731909906310411 |
| 1e-3 | 31 → 1.8139640289997057 / 0.722990837297975 | 150 → 1.8860919742547864 / 0.7514958336696763 |
| 3e-3 | 31 → 1.8001448213440414 / 0.7159262203822878 | 150 → 1.8521482905919395 / 0.7415560124949256 |
| 1e-2 | 93 → 1.8492724749293896 / 0.7373841024585337 | 150 → 1.8522916932246416 / 0.7376106840187479 |

- stato scelto: `selected_outer_statistics`, `selected_outer_clean_statistics` e `selected_fitting_diagnostics` danno le somme grezze per fold
  (anche sul fitting e contro y*).

Prova CPU su latenti sintetici con il codice congelato vero (`parity_probe.py`, uscita `parity_probe.out.json`, 7,7 s) **[D, solo CPU]**:
- A) un driver esterno che ricarica gli stati salvati e chiama `_score_model` riproduce i punti della curva: 8 stati su 8, uguaglianza esatta;
- B) rifare la traiettoria con 24 valutazioni in piu' (fitting, esclusi, pulito) lascia identici i 6 punti della curva e il `payload_sha256` finale:
  le valutazioni aggiuntive non spostano la traiettoria;
- C) il fit sul bersaglio pulito si costruisce con sole funzioni congelate (`dataclasses.replace(bundle, targets=bundle.clean_targets)`,
  `workflow._training_set`, `train.Trajectory`), stessa inizializzazione del run rumoroso; **non** con `workflow._run_selection`, che rifiuta
  i bersagli scambiati (`_assert_bundle`: «synthetic targets changed after calibration»).
- **[I]** Su GPU B) va confermato dal controllo di parita' del job: e' cio' che quel controllo misura.

---

## 4. Artefatti su Leonardo (da ricevute e piano; non riletti dal vivo)

Radice congelata A1: `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/pose_temponet_20261117` (WORK, persistente).

| Artefatto | Percorso (sotto la radice, salvo diversa indicazione) | byte | sha256 |
|---|---|---|---|
| calibrazione fold 0 | `wls_calibration/0/calibration.pt` | 228.630.380 | d08a804e345e9e790a41c51f5649151b6312ddc26648c1ccacf89458d6a34bf2 |
| calibrazione fold 1 | `wls_calibration/1/calibration.pt` | 228.396.140 | 441f6486b9ab9446d6e4aa7ddaac6f30ea9b3785d6cf83c7820707e90dde4c91 |
| sintesi fold 0 / 1 | `wls_calibration/{0,1}/calibration.json` | 1.954.955 / 1.955.346 | 2b6fe2a5…18ff / 357097f2…350c |
| stati TCN temporale | `controls/temporal/TCN_post/selection_states.pt` | 18.386.382 | f9f4d4ad1a59f117b05d8ed411daeefb2bef68709f611f368e379e65a1e1bbf0 |
| selezione TCN temporale | `controls/temporal/TCN_post/selection.json` | 251.445 | b2464c1c757927a71d08cb9b5743f1f55eb24a0136300a536abf1576bc34e03d |
| stati L0 temporale | `controls/temporal/L0/selection_states.pt` | 2.197.650 | 7e6cf0692022079828b4dd34bb7f11ca3e1fbc0f436238a00714be25db1bea45 |
| stati MLP temporale | `controls/temporal/MLP/selection_states.pt` | 14.685.978 | bc45c2aa660415284423943a16f472331150a2cdee2b7d840e95e744b9df7f49 |
| stati lineari L0 / MLP / TCN | `controls/linear/{L0,MLP,TCN_post}/selection_states.pt` | 2.768.634 / 18.415.054 / 18.330.446 | 0d423875… / 014cd668… / 29990802… |
| ricevuta 65035631 | `receipts/TCNH-controls-temporal-TCN_post-20261120.json` | 7.380 | f95c7a5dceb80458998c92a50e4d1efbe3248c972c781055a2689be58fafd84d |
| ricevute calibrazioni | `receipts/TCNH-wls-calibration-{0,1}-20261120.json` | 6.932 / 6.938 | 2e78439b…4634 / 697d52d8…2dfa |
| piano di fase | `manifests/b334ece870bc85b804708f92e4c240f4999750ef8d089ca7df1bd040d2ee04e3.json` | 76.657 | b334ece8…04e3 |
| cache latenti train (WORK) | `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/consolidation_long_20261105/49046/pose/train/train.npz` | 474.412.856 | 30c0b50d09fa55e39085e547e923593378e490e86f1b70c1e1ffe8bfdd1eec7e |
| **riferimento dei target train (SCRATCH)** | `/leonardo_scratch/large/userexternal/sbenatti/wearusfm_runs/results/pilot_posa/B19618_20261028/B/pose/train_subset.npz` | 257.675.076 | c7bc296e70197f5a07a590a391f051fa1554425bcdbdb11d046e427a0c50db23 |
| **pose_stats (SCRATCH)** | `/leonardo_scratch/large/userexternal/sbenatti/data/processed/emg2pose_pose/pose_stats.json` | non nel pin | 757f2aeebf688063747d50e55d983dd6b3c4cea268ce43d2bae06f3e99524a7e |
| checkout A1 (WORK) | `/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pose_temponet_20261121_A1` | 95M | HEAD edebfb6c… |
| log del job | `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/TCNH-controls-temporal-TCN_post-20261120_65035631.out` | copia sul Mac: 148.612 | — |

- **[D]** `read_verified_inputs(..., stage="controls")` apre tutti e tre gli input di train (`provenance.py:46,450-483`): la cache su WORK e i **due file su SCRATCH**
  (cancellazione a 40 giorni; nel piano: `next_planned_scratch_refresh 2026-12-03; forbidden during TCN input reads`).
  `verify_pin` confronta anche `mtime_ns` e `ctime_ns` prima e dopo la lettura: un rinfresco durante il job lo fa fallire.
- **[D]** Popolazione: finestre per fold [961, 946], senza posa nel fitting [41, 12], fitting utili [920, 934], passi per epoca [29, 30].
- **[D]** In `selection_states.pt` ci sono solo lo stato migliore e lo stato finale di ogni LR (`train.py:452-458`): 20 stati per la TCN, nessun checkpoint intermedio.

---

## 5. Costi

Costi unitari misurati sul nodo A100 (benchmark GPU 65016447, mediane di 10 ripetizioni; per la TCN il massimo dista al piu' l'1,3% dalla mediana, per L0 fino al 10,8% sul train):

| Voce | TCN_post | L0 | MLP |
|---|---|---|---|
| train, un'epoca appaiata (fold0 + fold1) | 1,1655 + 1,1684 = **2,334 s** | 0,241 s | 0,409 s |
| valutazione degli esclusi con y e y*, un checkpoint appaiato | 1,352 + 1,371 = **2,722 s** | 1,238 s | 1,229 s |
| valutazione di 1.907 finestre con y e y*, un modello | **2,751 s** | 1,207 s | 1,220 s |

Riscontro sui job veri **[D]**: (durata del runner − epoche × train) / checkpoint = 2,481 s (65035631: 900 epoche appaiate, 920 checkpoint, 4.383,3 s)
e 2,507 s (65035626); 0,93-0,97 s per L0 e MLP. Costo fisso: 28,1-43,7 s prima del cronometro del runner; 13,76 s per verificare 11 ricevute e
caricare le due calibrazioni (misura sul nodo di login); 5,02 s di lettura degli input (benchmark). Preparazione del bundle: non misurata a parte.

| Passo | Calcolo | Stima | Nota |
|---|---|---|---|
| a) inferenza: 20 stati TCN + 4 stati L0/MLP, fitting ed esclusi, y e y* | 20 × 2,751 + 2 × 1,207 + 2 × 1,220 | **59,9 s** | [D] il calcolo, [I] l'estensione a pannelli per utente o per giunto |
| b) parita' di un'epoca (LR 0,003, due fold) | 1 × 2,334 + 5 × 2,722 | **15,9 s** | confronta 5 punti della curva |
| b') parita' forte fino all'esposizione 31 | 31 × 2,334 + 35 × 2,722 | **167,6 s** | confronta anche `payload_sha256` dei due stati migliori |
| c) due fit (fold0 + fold1 appaiati), 150 epoche, LR 0,003, esclusi a ogni checkpoint | 150 × 2,334 + 154 × 2,722 | **769 s** (12,8 min) | |
| c') come c) piu' fitting a 35 checkpoint (0; 0,25; 0,5; 0,75; 1; poi ogni 5) | 769 + 35 × 2,751 | **866 s** (14,4 min) | |
| c'') come c) piu' fitting a tutti i 154 checkpoint | 769 + 154 × 2,751 | **1.193 s** (19,9 min) | |
| costo fisso per job | 44 + 14 + 5 + preparazione | **1,5-3 min** | [I] |

Proposta di due job GPU distinti (1 GPU, 8 CPU, 32G, `boost_usr_prod`):

| Job | Contenuto | Durata attesa | Walltime prudente | Preventivo massimo | QoS |
|---|---|---|---|---|---|
| D0 | a) + b') | 5-7 min | **00:15:00** | 0,25 GPU-ore, 2,0 core-ore | `normal` come A1; `boost_qos_dbg` ammissibile |
| D1 | c') oppure c'') | 16-23 min | **00:45:00** | 0,75 GPU-ore, 6,0 core-ore | **`normal`**: nei 30 minuti di `boost_qos_dbg` il margine scende a 1,3 |
| checkout | codice | 30 s (A1) | 00:12:00 | 0 GPU-ore, 0,4 core-ore | `lrd_all_serial` |

- Totale massimo **1,00 GPU-ore** (8,0 core-ore sui nodi GPU, `billing=8`: 8 ore locali) piu' 0,4 core-ore di checkout. Consumo atteso 0,35-0,5 GPU-ore.
  Sotto la soglia di 5 GPU-ore di CLAUDE.md. Non rientra nei tetti A1 (53,85 / 56,10 GPU-ore, spese 3,681): e' una voce nuova.
- Memoria: MaxRSS dei sei controlli 3.049.720K-3.233.952K: 32G bastano.
- Coda **[D]**: il 21/11, `qos normal`, fra lo stato `sottomesso` e la partenza passarono da 64 a 100 secondi (orari degli `state.json` e di sacct).
  **[I]** oggi puo' essere diversa: nel ledger ci sono invii di un'altra linea alle 11:06-11:07 del 22/11.
- **Manutenzione [D]**: `maint_23112026_booster` 23/11 08:00 - 25/11 08:00 su `lrdn[0001-3456]` e `maint_23112026_vizser` su `login[08,13]`
  (i due nodi di `lrd_all_serial`): letta dall'operatore il 21/11 alle 09:39 (`results/infra/pose_temponet_execution_20261120/launcher/operator_read_20261121.txt`).
  Un job parte solo se il suo walltime finisce prima delle 08:00: con 00:45:00 l'ultima partenza utile e' alle 07:15 del 23/11; dopo resta in attesa fino al 25/11.
- **Quota [D]**: il 21/11 alle 15:08 `work_free_bytes` 36.876.038.144 e scritture pendenti 23.622.320.128: restano 13.253.718.016 byte contro i
  12.884.901.888 minimi, cioe' **368.816.128 byte** di margine (`L/pending_writes_reassessment_20261121.json`). Le predizioni di 20 stati TCN in FP32
  peserebbero 451.577.600 byte (24 stati: 541.893.120): **non salvarle su WORK senza una lettura nuova della quota**. JSON dei punteggi e due stati
  del fit pulito (905.805 byte l'uno nel benchmark) stanno in pochi MB.

---

## 6. Identita' e ricevute

**Riusabile in sola lettura, con verifica di hash e byte dai pin:**
- le due `calibration.pt` e le loro ricevute; i sei `selection_states.pt` e `selection.json`; le 11 ricevute della catena
  (stessa lettura dell'operatore: `runner._ancestors(plan, execution.find_job(plan, "TCNH-selection-L0-20261120"))`, `L/verify_positive_controls_readonly.py:8-9`);
- cache dei latenti, riferimento dei target e `pose_stats` con i pin del piano;
- `plan_controls.json` (commit `6a20054ad2ffdb3d754e6106775a159a0933d99a`, sha256 `b334ece8…04e3`, anche materializzato in `manifests/`) **come tabella di pin**, non come piano da eseguire;
- il pin numerico `gpu8` come valore atteso;
- i 247 sorgenti congelati, byte per byte.

**Richiede identita' nuova:**
- riferimento di deploy nuovo (per esempio `deploy/pose-temponet-diag-20261122`), campagna e checkout nuovi
  (`$WORK/wearusfm_campaigns/pose_temponet_diag_20261122`): il checkout A1 non contiene il driver e non si modifica; `campaign_checkout.sbatch` rifiuta una cartella esistente (riga 35);
- ID logici nuovi: nel registro locale non c'e' nessun `TCND-*` (919 cartelle, letto alle 11:34 del 22/11); gli ID `TCNH-*` non si riusano;
- radice delle uscite nuova, **fuori** da `…/results/pose_temponet_20261117`: quella radice e' inventariata dallo stadio `collection` (`runner.py:387-393`)
  e `runner` rifiuta cartelle gia' presenti; `P.atomic_json` e `P.atomic_torch` accettano una radice qualsiasi (`provenance.py:201-227`);
- ricevute proprie con schema nuovo: `P.make_receipt` pretende un job del piano (`provenance.py:669-670`). Contenuto minimo: ID Slurm e nodo, HEAD del checkout,
  `source_sha256` dei 247 file, sha256 dei file del driver, contesto numerico e digest, pin verificati degli input, esito delle parita', pin delle uscite;
- voce di costo propria.

**Checkout: scelta consigliata.** Un solo checkout nuovo da un commit discendente di quello di GPT, con i 247 file intatti e in piu' solo
`scripts/tcn_diag/` e `scripts/slurm/tcn_diag.sbatch` (nomi provati: non entrano nel manifest). All'avvio il driver esegue
`P.validate_plan(plan, require_review=True)` e `P.checkout_identity(plan)`: sono le stesse garanzie di A1 sul codice riusato.
**[I] Da verificare nel repository vero** (qui la copia e' parziale): che il commit di partenza discenda da `6a20054a…` e contenga le copie locali delle review
e delle ricevute storiche che `validate_plan` rilegge (`git merge-base --is-ancestor`, `git cat-file -e`).

---

## 7. `submit_job.py`: CLI e ledger

- **[D]** sha256 `814da4bc66dbf18d69ee2ba332675fa0788ab25247d5f5460224cfd24d7e0d82`: identico nell'albero congelato, in
  `/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py` e nel `main` canonico (letti il 22/11 alle 11:35).
- Obbligatori: `--id`, `--agente {claude,gpt,simone}`, `--campagna`, `--gpu-ore` (numero ≥ 0), poi `--` e gli argomenti di `sbatch`, oppure `--script-stdin FILE`
  (`submit_job.py:198-207,330-331`). `--job-name`, `-J` e `--comment` sono vietati: li impone lo script (`--job-name=ID`, `--comment=wearusfm:ID`).
- ID logico: `^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`, unico per sempre; la prenotazione e' `mkdir` di `~/wearusfm_local/jobs/<id>/` piu' `operazione.lock`.
- `--dopo ID` (ripetibile): `--dependency=afterok:<id Slurm>:<…>`; ogni ID dev'essere `sottomesso` con ID Slurm noto nel registro locale (righe 208-216).
- `--script-stdin`: il testo locale (deve iniziare con `#!`) va a `sbatch` su stdin; nello stato resta il suo sha256. Il percorso relativo si risolve
  nella cartella da cui si lancia.
- Comando remoto: `ssh leonardo` + `sbatch --parsable …`; attesa massima 120 s.
- Esito (righe 181-187): uscita 0 e una sola riga `^(\d+)(;[\w.-]+)?$` → `sottomesso`; uscita diversa da 0/255 con `sbatch: error` e nessun numero → `rifiutato`;
  tutto il resto (255, timeout, uscita illeggibile) → **`ambiguo`**. Codice d'uscita 0 solo se `sottomesso`, altrimenti 3.
- `ambiguo`: mai rilancio. `--riconcilia ID` interroga `sacct -X -n -P -S <data> --name=ID --format=JobID,JobName,State` e `squeue -h --name=ID -o '%i|%j|%T'`:
  uno trovato → `sottomesso`; piu' d'uno → `duplicato`; nessuno → controllo d'assenza. `assente` solo con due controlli vuoti, il primo almeno 10 minuti
  dopo il tentativo e l'ultimo almeno 10 minuti dopo il primo; poi `--dopo-assenza-verificata`. `rifiutato` → `--dopo-rifiuto`.
- Ledger `~/wearusfm_local/jobs/ledger.tsv`, colonne separate da tabulazione: `quando, id, agente, campagna, stato, job_slurm, gpu_ore_stimate, codice_sha, comando`;
  una riga per transizione tranne `in_sottomissione`. `codice_sha` e' letto da `EXPECTED_SHA=` negli argomenti: nel job di checkout resta vuoto.
- `--stato [ID]`: stampa id, stato, ID Slurm, agente, campagna.

---

## 8. Sequenza proposta per la catena nuova (bozza, niente eseguito)

Nomi d'esempio: campagna `pose_temponet_diag_20261122`; riferimento `deploy/pose-temponet-diag-20261122`; ID `TCND-checkout-20261122`,
`TCND-states-20261122`, `TCND-cleanfit-20261122`; uscite `/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/pose_temponet_diag_20261122`.

0. Prima di tutto: lettura nuova di quota, coda, prenotazioni e saldo; nessun rinfresco di SCRATCH in corso; ID assenti dal ledger.
1. Commit sul ramo di Claude con `scripts/tcn_diag/` (driver), `scripts/slurm/tcn_diag.sbatch` (launcher), test e documento. Controlli locali:
   `source_identity()` = `9aaf1c8b…` nel worktree del commit; `git diff --quiet edebfb6c… D -- <i 247 percorsi>`; albero pulito; push su origin;
   pubblicazione del riferimento di deploy e `git ls-remote leonardo deploy/pose-temponet-diag-20261122 | cut -f1` = D. **Registrare il comando del push.**
2. Checkout, riga di A1 con i nomi nuovi e `--agente claude`:
```
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python '/Users/simonebenatti/Documents/ChatGPT/FM per EMG/wearusfm-gpt-science/scripts/submit_job.py' --id TCND-checkout-20261122 --agente claude --campagna pose_temponet_diag_20261122 --gpu-ore 0 --script-stdin scripts/slurm/campaign_checkout.sbatch -- --account=IscrB_WearUsFM --partition=lrd_all_serial --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=24G --time=00:12:00 --no-requeue --export=ALL,CAMPAIGN=pose_temponet_diag_20261122,CAMPAIGN_SHA=<D> --output=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/TCND-checkout-20261122_%j.out --error=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/TCND-checkout-20261122_%j.out
```
   Da lanciare dalla radice del worktree al commit D. Verifica: `COMPLETED 0:0`, riga «checkout di campagna pronto», HEAD, pulizia, sola lettura.
3. D0, dopo la verifica del checkout:
```
… submit_job.py --id TCND-states-20261122 --agente claude --campagna pose_temponet_diag_20261122 --gpu-ore 0.25 --script-stdin scripts/slurm/tcn_diag.sbatch -- --account=IscrB_WearUsFM --partition=boost_usr_prod --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=00:15:00 --no-requeue --export=ALL,CODE_DIR=/leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pose_temponet_diag_20261122,EXPECTED_SHA=<D>,TCND_STEP=states,TCND_OUT=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/results/pose_temponet_diag_20261122 --output=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/TCND-states-20261122_%j.out --error=/leonardo_work/IscrB_WearUsFM/wearusfm_runs/logs/TCND-states-20261122_%j.out --gres=gpu:1 --qos=normal
```
   Il driver esce con codice diverso da 0 se digest, `bundle_sha256` o una parita' non tornano.
4. D1: stessa riga con `--id TCND-cleanfit-20261122 --gpu-ore 0.75 --dopo TCND-states-20261122`, `--time=00:45:00`, `TCND_STEP=cleanfit`.
   Con `afterok` il fit pulito parte solo se D0 ha chiuso con 0. Se D0 fallisce, D1 resta in coda con dipendenza mai soddisfatta: va cancellato a mano e registrato.
   **[I]** Inviare D1 mentre D0 e' ancora in coda o in esecuzione: una dipendenza su un job gia' uscito dalla memoria di Slurm puo' essere rifiutata
   (`execution.py:195-196` prevede il caso; «Omitting a dependency is never an automatic fallback»).
5. Launcher nuovo: righe 3-18 di `pose_temponet.sbatch` (o dello script d'inventario), con le sue variabili (`CODE_DIR EXPECTED_SHA WORK TCND_STEP TCND_OUT`),
   `[ "$SLURM_CPUS_PER_TASK" = 8 ]`, rifiuto dei riavvii, poi `exec python3 -m tcn_diag "$TCND_STEP" --output-root "$TCND_OUT"`.
   Non deve dipendere dal proprio percorso: arriva su stdin.
6. Raccolta sul Mac di log, ricevute e `sacct -X` per ID; commit e push; consuntivo.

Nota: la regola A1 «nessun monitor in background» (`plan["background_monitor"] = False`, `provenance.py:403`) appartiene al piano A1; la catena nuova e' separata.

---

## 9. Riscontri operativi sulla proposta di GPT e su «allungare i training»

- **Costo del controllo pulito.** La nota di GPT indica «ordine 0,2-0,4 GPUh» (`docs/reviews/pose_temponet_temporal_diagnosis_20261122.md:44`).
  **[D]** Coerente come consumo: 769-1.193 s di calcolo piu' 1,5-3 minuti fissi = 0,24-0,38 GPU-ore. Il preventivo a walltime e' pero' 0,75 GPU-ore
  (00:45:00); con la regola A1 (fattore 2 piu' 180 s) verrebbe 0,82.
- **«Diagnosi senza nuovi fit» non e' a costo zero.** Serve comunque codice nuovo, un checkout, un job GPU e una ricevuta (0,25 GPU-ore di preventivo).
  Gia' disponibili senza job, negli stessi JSON archiviati: per lo stato scelto di ogni testa, somme su fitting ed esclusi contro y e sugli esclusi contro y*
  (`selected_fitting_diagnostics`, `selected_outer_statistics`, `selected_outer_clean_statistics`). Mancano: y* sul fitting, gli altri 18 stati TCN,
  la scomposizione livello / centrato e i due rami.
- **Allungare i training [D].** La ricetta congelata ha un tetto rigido a 300 epoche (`train.py:56` e `train.py:271-272`) e `PairedCandidate.advance`
  accetta solo 150 o 300, la seconda solo se il minimo cade nelle ultime 5 epoche (`train.py:421-431`). Oltre 300 servono sorgenti nuovi, quindi identita' nuova.
  Le curve archiviate del job 65035631 hanno il minimo sugli esclusi prima della fine per tutti e 5 i LR:

| LR | orizzonte | minimo MSE a | minimo E_dyn a | E_dyn minimo | E_dyn finale | finale − minimo |
|---|---|---|---|---|---|---|
| 1e-4 | 300 | 181 | 201 | 0.725580162290832 | 0.7281499412886786 | +0,00257 |
| 3e-4 | 150 | 64 | 64 | 0.725382121932922 | 0.731909906310411 | +0,00653 |
| 1e-3 | 150 | 31 | 28 | 0.722227946298081 | 0.7514958336696763 | +0,02927 |
| 3e-3 | 150 | 31 | 27 | 0.7158262458538921 | 0.7415560124949256 | +0,02573 |
| 1e-2 | 150 | 93 | 111 | 0.7370656177709967 | 0.7376106840187479 | +0,00055 |

  Per il 50% serve E_dyn aggregato 0.6995689763192543; lo stato scelto e' a 0.7159262203822878 (distanza 0,01636); il minimo di tutte le curve da' il 21,88%.
  **[I]** Con questa ricetta e questo bersaglio rumoroso, piu' epoche non avvicinano la soglia: sugli esclusi le curve risalgono.
- **Se si vuole comunque la prova a 300 epoche** per un LR fermo a 150: non serve ripartire da zero. `Trajectory.restore(current_states)` riprende
  la stessa traiettoria (ottimizzatore, mescolamento e dropout salvati) e `train_to_step` accetta fino a 300 epoche: 150 × 2,334 + 150 × 2,722 = **758 s**
  per LR. **[D, solo CPU]** `resume_probe.py` (uscita `resume_probe.out.json`, 1,4 s): stato salvato con `torch.save`, ricaricato in un oggetto nuovo e
  proseguito fino al tetto = run continuo (`payload_sha256`, pesi e perdite per epoca identici); oltre il tetto `ValueError: requested training exceeds
  the frozen horizon`; una `Recipe(maximum_epochs=600)` e' rifiutata («nonstandard production recipe»). **[I]** Su GPU lo conferma la parita' del job.
- **Candidato residuale: costo della riqualifica.** Qualunque modifica in `scripts/pose_temponet/` cambia `source_sha256`: piano, review, checkout e ricevute nuovi.
  Le ricevute A1 valgono solo per `source_sha256 9aaf1c8b…`; l'eccezione A1 copre tre ricevute per hash esatto (`budget_amendment.py:20-24`), non le calibrazioni.
  Riferimenti misurati: due calibrazioni 422 + 414 s su 8 CPU (1,858 core-ore); due controlli TCN 3.710 + 4.427 s = 2,260 GPU-ore
  (prenotate 10,467 con i walltime A1); sei controlli 3,567 GPU-ore misurate, 17,133 prenotate.
