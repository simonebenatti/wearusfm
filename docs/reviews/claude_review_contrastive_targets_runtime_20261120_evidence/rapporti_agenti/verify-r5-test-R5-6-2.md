# Verifica scettica n. 2 di R5-6 (r5-test)

Avvio: 2026-11-20 04:52:10. Copia: 0efa4d45cdd3c314d2c0b28ffd8fdd14b9461aba

Clone temporaneo: tmp/v56s2_wt (git clone --shared, detach 0efa4d45, albero pulito); helper tmp/v56s2_scripts/verify.sh
(stesso ambiente: OMP/OPENBLAS/MKL=1, PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=src:tests/cpu:scripts, venv wearusfm-tok -B).

## Riproduzione indipendente
- Base: `--verify` sulla ricevuta validation_final -> `{"verified": true, "all_passed": true}`, rc 0.
- Pin effettivi (scripts/contrastive_local_validation.py:28-35 a fbfde797): source_sha256 (riga 30), log (31-32), 29 test_files (33-34).
  Controprova positiva: commento in scripts/contrastive_numeric_profile.py -> `ValueError: validation source changed`;
  commento in tests/cpu/test_contrastive_release.py -> `validation test changed`; riga in full_suite/cpu_ddp.log -> `validation log changed`.
- source_digest (scripts/contrastive_prepare.py:192-201) copre solo src/wearusfm/**/*.py, scripts/contrastive_*.py,
  scripts/slurm/contrastive*.sbatch, train_jepa, lib_code_dir.sh: tests/cpu e scripts/diagnostics restano fuori.
- Import dai 29 file (grep): test_pose_training <- 7 file (callsite_oracle, common_checkpoint, common_state_cuda, cuda_check,
  runtime, target_parity, target_resume); test_pretraining_loader <- 3 (audit, ms_fast, rms); frozen_campaign_support <- 2
  (historical_calibration, historical_callsites); contrastive_review_oracle (tests/cpu, non scripts/) <- 1. Conteggi esatti.
  In piu' (non citati nel rilievo): test_pose_training importa test_ddp_equivalence e test_pose_targets; common_checkpoint e
  common_state_cuda importano scripts/diagnostics/contrastive_common_{checkpoint,state_cuda}.py (sys.path:11/9). Tutti non pinnati.
- Mutazione del rilievo: `def _pose_setup(*a, **k): raise RuntimeError(...)` in coda a tests/cpu/test_pose_training.py ->
  `--verify` -> `{"verified": true, "all_passed": true}`, rc 0. Con la stessa mutazione test_contrastive_target_resume.py +
  test_contrastive_cuda_check.py (ambiente della ricevuta) -> `1 failed, 14 passed, 7 errors`: la verifica attesta "tutto passato"
  su un albero in cui i test della ricevuta falliscono.
- Altri moduli non pinnati, stessa prova (`raise SystemExit` in coda): scripts/diagnostics/contrastive_common_state_cuda.py,
  tests/cpu/frozen_campaign_support.py, tests/cpu/contrastive_review_oracle.py, tests/cpu/test_pretraining_loader.py -> verified true, rc 0.
- command vs test_files: copia della ricevuta in tmp con `command` senza test_contrastive_release.py (28 file, test_files 29)
  -> verified true, rc 0. Attenuante: i due elenchi nascono dalla stessa lista (righe 38-46) e la ricevuta e' pinnata per sha256
  (36d1d4d4...) in delivery_pin.json e verification.json; --verify pero' non controlla quel pin.
- Versioni: chiavi della ricevuta = base_commit, cuda_tested, cwd, date, environment (5 variabili), historical_*, producer, runs,
  schema, source_sha256, test_files; nessuna versione Python/torch/numpy/pytest ne' piattaforma, ne' HEAD/pulizia del worktree;
  pytest.log (-q) senza intestazione. Solo il percorso dell'interprete del venv nel command.
- Copertura: nessun file di tests/ nomina local_validation o contrastive_local; l'unico test che lo tocca e'
  test_contrastive_mmap_compatibility.py (ast.parse dei torch.load, 8 test). C9a (riga 30 -> `pass`, efficace: con sorgente
  alterato verified true) sull'intero elenco della ricevuta: **465 passed in 111.80s**, rc 0.

## Motivi cercati per confutare (non sufficienti)
- Nessun lettore di produzione: local_validation.json e' citato solo da delivery_pin.json/verification.json (pin sha256);
  contrastive_release.py, job, run non lo leggono. -> nessun effetto sulla catena: esclude BLOCCANTE, non il difetto.
- delivery_pin fissa content_commit fbfde797 completo: git pinna anche i moduli di supporto; nessuno degli 8 moduli e' cambiato
  in 438c8d65..fbfde797 (ultimi commit 13/10-18/11). -> oggi nessuna discrepanza; il difetto e' latente (verify a un altro commit).
- Prontuario:191-195 dichiara "--verify ricontrolla ogni pin" (vero alla lettera) e che la CPU locale non certifica CUDA/NCCL;
  verification.json: torch locale piu' recente, nessuna esecuzione Torch2.2. Nessuna dichiarazione che escluda i moduli di
  supporto o le versioni dalla ricevuta "riproducibile" (prompt runtime:14; punto 5 "riproducibilita' dei log").

## Esito
Rilievo NON confutato: riprodotto in tutte le parti. Gravita' MINORE confermata (difetto reale del verificatore, nessun
gate lo usa, commit dei contenuti pinnato e moduli invariati). La parte command/test_files e' al limite della NOTA.
Pulizia: clone v56s2_wt, basetemp e ricevuta alterata cancellati; restano tmp/v56s2_c9a_full.log e tmp/v56s2_scripts. Copia di sola lettura pulita a 0efa4d45. Fine: 2026-11-20 04:59
