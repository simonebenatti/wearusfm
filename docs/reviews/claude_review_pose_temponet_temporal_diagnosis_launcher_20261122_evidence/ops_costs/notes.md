# Note di lavoro — mappa operativa e costi (revisore ops_costs, 22/11/2026)

Albero letto (sola lettura): /Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244 (COMMIT = e8c9824461b5e18494ff462f1be8abda71b76d5e)
Nessun accesso al cluster: tutti i percorsi di Leonardo sono ricavati da ricevute/JSON archiviati, non letti dal vivo.

## N1 — catena A1 ricostruita dai file (12:xx)

- Checkout A1: ID logico `TCNH-checkout-A1-20261121` = Slurm 65023067, COMPLETED 0:0, 30 s, 2 CPU, login08 (lrd_all_serial),
  script su stdin `scripts/slurm/campaign_checkout.sbatch` (sha256 del testo eaafc432...589fcc), CAMPAIGN=pose_temponet_20261121_A1,
  CAMPAIGN_SHA=edebfb6c290cf661989c5cff45916bc6a93e607e. Checkout: /leonardo_work/IscrB_WearUsFM/wearusfm_campaigns/pose_temponet_20261121_A1
  (2349 file, 95M). Fonte: launcher/TCNH-checkout-A1-20261121.state.json, ..._65023067.out, checkout_verified.json.
- Sei controlli: launcher/controls_commands.txt (righe complete), ID Slurm 65035622/623/626/628/629/631 (controls_submission.json).
  Tutti: boost_usr_prod, --qos=normal, 1 GPU, 8 CPU, 32G, --no-requeue, script su stdin pose_temponet_dispatch.sbatch.
  Piano di fase: commit 6a20054ad2ffdb3d754e6106775a159a0933d99a, plan_controls.json sha256 b334ece8...04e3.

## N2 — manifest dei sorgenti (DIMOSTRATO, prova eseguita)

- provenance.py:236-248 `source_manifest`: `src/**/*.py` (rglob) + `scripts/*.py` (glob NON ricorsivo) + `scripts/pose_temponet/**` con suffisso .py/.sh/.sbatch
  + `scripts/slurm/lib_code_dir.sh` + `scripts/slurm/pose_temponet*.sbatch` + `scripts/slurm/campaign_checkout.sbatch`.
- Prova: `manifest_probe.py` (questa cartella), uscita in `manifest_probe.out.json`:
  - albero e8c98244: 247 file, source_sha256 = 9aaf1c8b6a050a524dfb60af65df9406313d4fd0e690dab1097bf11df2ec67ea = quello del piano A1
    (manifest identico a plan_controls.json["reader"]["source_manifest"]): il commit di GPT non ha toccato i sorgenti congelati.
  - NON cambiano il manifest: nuovo pacchetto `scripts/tcn_diag/*.py`; nuovo `scripts/slurm/tcn_diag.sbatch`; `scripts/slurm/other.sh`;
    file non .py/.sh/.sbatch dentro scripts/pose_temponet; docs/, results/, tests/.
  - CAMBIANO il manifest: `scripts/slurm/pose_temponet_<qualsiasi>.sbatch`; un nuovo `scripts/<nome>.py` al primo livello; qualunque `src/**/*.py`;
    qualunque .py/.sh/.sbatch dentro `scripts/pose_temponet/`.

## N3 — launcher congelato

- pose_temponet.sbatch (21 righe): comando Python FISSO, righe 19-21: `python3 -m pose_temponet materialize-plan ...` poi
  `exec python3 -m pose_temponet run --plan ... --job-id ...`. Nessun punto d'aggancio per un altro modulo.
- `run` -> runner.run_stage (runner.py:208-222): l'ID deve stare nel grafo dei 24 job (execution.find_job), piano validato con review,
  `existing stage output: do not duplicate or retry`, `existing stage receipt: ...`. Quindi NON riusabile per un modulo nuovo.
- pose_temponet_dispatch.sbatch (8 righe) e' il wrapper passato su stdin: verifica e fa `exec bash "$CODE_DIR/scripts/slurm/pose_temponet.sbatch"` (fisso).
- Precedente gia' eseguito di script esterno su stdin con lo stesso ambiente: launcher/gpu_inventory_A1_20261121.sbatch
  (sha256 32c1ed76...e7ec = script_stdin_sha256 nello state.json), job 65033157 su lrdn0058, digest numerico uguale a gpu8.

## N4 — contesto numerico gpu8 = 12cb2a0b3207b66e260d3ca490db93ee0c029bb8f03d766df89a9edba6b7eb3f

- Stesso digest in: benchmark GPU 65016447 (lrdn0059), inventario 65033157 (lrdn0058) e nei sei controlli
  (lrdn2183, lrdn3188, lrdn3369, lrdn2287, lrdn3349, lrdn0707): otto nodi diversi, un solo digest.
- Contenuto (ricevuta TCNH-controls-temporal-TCN_post-20261120.json): python 3.11.6, numpy 1.26.4, scipy 1.12.0, sklearn 1.4.1.post1,
  torch 2.2.0a0+git6c8c5ad, cuda 12.1; threadpools: 2 openblas (0.3.21.dev e 0.3.23.dev, SkylakeX, pthreads), 2 libgomp, 1 libomp, tutti a 8 thread;
  intraop 8, interop 1, OMP/MKL/OPENBLAS_NUM_THREADS="8"; deterministic True, warn_only False, cudnn_deterministic True, cudnn_benchmark False,
  TF32 spento (cudnn e matmul), CUBLAS_WORKSPACE_CONFIG=":4096:8"; hardware: Xeon Platinum 8358, NVIDIA A100-SXM-64GB, capability [8,0], 68.099.571.712 byte.
- Il censimento dei threadpool dipende dalle librerie GIA' caricate: l'inventario importa `from pose_temponet import runner, train, provenance as P`
  («Same imports as the production runner before its numerical census»).

## N5 — durate misurate (sacct -X, launcher/sacct_through_controls_20261121.tsv) e costi unitari (benchmark GPU 65016447)

| job | Slurm | elapsed sacct (s) | elapsed runner (s) | epoche appaiate | checkpoint | MaxRSS |
|---|---|---|---|---|---|---|
| linear L0 | 65035622 | 1451 | 1422,9 | 1200 | 1220 | 3.174.124K |
| linear MLP | 65035623 | 1287 | 1258,1 | 900 | 920 | 3.107.200K |
| linear TCN_post | 65035626 | 3710 | 3681,0 | 750 | 770 | 3.233.952K |
| temporal L0 | 65035628 | 1105 | 1074,0 | 900 | 916 | 3.092.552K |
| temporal MLP | 65035629 | 860 | 831,6 | 600 | 616 | 3.049.720K |
| temporal TCN_post | 65035631 | 4427 | 4383,3 | 900 | 920 | 3.233.548K |

- Attesa in coda dei sei controlli (qos normal, sabato 21/11): invio 15:21:44, partenza 15:22:47 (cinque) e 15:23:24 (65035631).
- Benchmark TCN_post (mediane di 10 epoche): train 1,1655 s (fold0, 29 passi, 920 finestre) + 1,1684 s (fold1, 30 passi, 934 finestre) = 2,334 s per epoca appaiata;
  valutazione degli esclusi CON punteggio pulito 1,352 s (946 finestre) + 1,371 s (961) = 2,722 s per checkpoint appaiato; valutazione di 1.907 finestre 2,751 s;
  un batch pieno di inferenza 0,0366 s. L0: eval 1.907 finestre 1,207 s; MLP 1,220 s.
- Riscontro sui job veri: (runner - epoche x train) / checkpoint = 2,481 s (temporal TCN), 2,507 s (linear TCN), 0,93-0,97 s (L0/MLP).
- Prima del cronometro del runner (moduli, venv, due processi Python, validazioni): 28,1-43,7 s.
- Verifica in sola lettura di 11 ricevute + 2 calibration.pt + cancello sul nodo di login: 13,76 s (positive_controls_gate_20261121.raw.json).
- Checkout A1: 30 s, 2 CPU, MaxRSS 205.840K.

## N6 — ancore di parita' gia' archiviate sul Mac (selection_temporal_TCN_post.json)

- initial_sha256 dei due fold: 50fb2294f6beb2b54917e99bcb81683ffc24a41bd69ce36b17c08cc53279aab5
- LR 0,003: esposizione 0 -> mse 2.892463349558587 / edyn 1.0; 0,25 -> 2.1452115131878022 / 0.8222974023158734; 0,5 -> 2.020721989079718 / 0.7844989085655731;
  0,75 -> 1.9583036504913476 / 0.7692090899847309; 1 -> 1.9309589037808752 / 0.7606302236459029 (fold_steps [29, 30]).
- Per ogni LR: punto migliore (best_states) e ultimo punto (current_states):
  1e-4: 181 -> 1.8181615643874114 / 0.7256444669656075; 300 -> 1.8268597285134838 / 0.7281499412886786
  3e-4: 64 -> 1.8174307513426158 / 0.725382121932922; 150 -> 1.837209816099995 / 0.731909906310411
  1e-3: 31 -> 1.8139640289997057 / 0.722990837297975; 150 -> 1.8860919742547864 / 0.7514958336696763
  3e-3: 31 -> 1.8001448213440414 / 0.7159262203822878; 150 -> 1.8521482905919395 / 0.7415560124949256
  1e-2: 93 -> 1.8492724749293896 / 0.7373841024585337; 150 -> 1.8522916932246416 / 0.7376106840187479
- selected_outer_statistics / selected_outer_clean_statistics / selected_fitting_diagnostics: somme grezze per fold dello stato scelto (LR 0,003, esposizione 31).
- bundle_sha256 4e147b1f43f3edf64ac93262d8b0805c762d886cc14d96b7b786c9c867fd60ad (uguale per L0, MLP e TCN_post temporali);
  target_sha256 [41519fd2..., e988407c...], clean_target_sha256 [c2452d8b..., f1a6df78...], soglie [0.5437424309221641, 0.5374366720436082].

## N7 — trappole operative viste nei file

- Due tentativi falliti della verifica in sola lettura sul nodo di login: `ModuleNotFoundError: No module named 'numpy'`
  (launcher/positive_gate_operator_environment_attempt.log e ...attempt2.log). Il comando dei due tentativi non e' archiviato; quello riuscito
  (verify_positive_controls_readonly.command.txt) antepone a PYTHONPATH con `${PYTHONPATH:+:$PYTHONPATH}`. Causa NON dimostrata dai file (sospetto: PYTHONPATH sostituito).
- Il comando esatto del push del riferimento di deploy A1 non e' archiviato in nessun file dell'albero; resta solo l'esito
  `leonardo_deploy_ref_matches: true` (deployment_checks.json) e il controllo `git ls-remote leonardo deploy/<nome> | cut -f1` del prontuario del 20/11.
- execution.submit_command scrive `--agente gpt` e la campagna A1 in modo fisso (execution.py:216-217): non riusabile per righe di Claude.
- Manutenzione letta dall'operatore il 21/11 09:39 (launcher 20261120/operator_read_20261121.txt): `maint_23112026_booster` 23/11 08:00 - 25/11 08:00
  su lrdn[0001-3456] (boost_usr_prod) e `maint_23112026_vizser` su login[08,13] (i due nodi di lrd_all_serial).

## N8 — prova di parita' CPU (parity_probe.py, 7,7 s; uscita parity_probe.out.json)

- Codice congelato vero, latenti sintetici (6 utenti x 3 finestre, 160x384), testa TCN_post vera, ricetta di prova 3/6 epoche.
- A) 8 stati salvati (4 LR x migliore/finale) ricaricati da un driver esterno + `workflow._score_model`: punti della curva riprodotti, uguaglianza esatta 8 su 8.
- B) traiettoria rifatta con 24 valutazioni in piu' (fitting ed esclusi, con pulito): 6 punti su 6 identici, `payload_sha256` finale identico.
- C) fit sul pulito: `dataclasses.replace(bundle, targets=bundle.clean_targets)` + `_training_set` + `Trajectory` funziona; `_assert_bundle` rifiuta lo scambio
  («synthetic targets changed after calibration»), quindi `_run_selection` non e' utilizzabile; stessa inizializzazione del run rumoroso; Trajectory di produzione: 1 passo eseguito.
- Limite: solo CPU. Su A100 lo decide il controllo di parita' del job.

## N9 — quota, registro, submitter (letti sul Mac, 22/11 11:34-11:35)

- Margine di quota del 21/11 15:08: 36.876.038.144 - 23.622.320.128 - 12.884.901.888 = 368.816.128 byte. Predizioni FP32 di 20 stati TCN: 451.577.600 byte.
- Registro locale: 919 cartelle, 1.878 righe di ledger, nessun ID `TCND-*`; ultimi invii di un'altra linea alle 11:06-11:07 del 22/11.
- submit_job.py: sha256 814da4bc...0d82 identico in albero congelato, wearusfm-gpt-science e main canonico.

## N10 — costi ricalcolati (mediane del benchmark GPU)

- a) 59,87 s; b) 15,95 s; b') 167,64 s; c) 769,36 s; c') 865,63 s; c'') 1.192,95 s; continuazione 150->300 di un LR 758,47 s.
- Curve del 65035631: minimo sugli esclusi prima della fine per tutti e 5 i LR; E_dyn finale - minimo: +0,00257, +0,00653, +0,02927, +0,02573, +0,00055.
- Tetto rigido della ricetta congelata: 300 epoche (train.py:56, 271-272).
- Documento principale: OPS_MAP.md (9 sezioni).

## N11 — ripresa di uno stato salvato (resume_probe.py, 1,4 s, solo CPU)

- Stato a 3 epoche salvato con torch.save, ricaricato e proseguito a 6 = run continuo a 6: payload_sha256, pesi ed epoch_losses identici.
- Oltre il tetto: `ValueError: requested training exceeds the frozen horizon`; Recipe(maximum_epochs=600): `ValueError: nonstandard production recipe`;
  Trajectory di produzione oltre 300 epoche: stesso rifiuto.
