# Verifica scettica n. 1 del rilievo R5-6 (r5-test)

Rilievo: ricevuta locale, `--verify` non pinna i moduli di supporto dei test; nessun test di `contrastive_local_validation.py`.
Stato: in corso (scritto in coda man mano).

## Esito: NON confutato. Gravita' confermata: MINORE

Prove rifatte da me in un clone temporaneo a 0efa4d45 (tmp/r56v1_clone, poi cancellato).

1. Codice `scripts/contrastive_local_validation.py` (identico a fbfde797): `--verify` (righe 28-35) controlla solo
   `source_sha256` contro `source_digest()` (r.30), i pin dei log (r.31-32) e dei 29 `test_files` (r.33-34);
   `all_passed` e' letto dalla ricevuta (r.35). `source_digest()` (`contrastive_prepare.py:192-201`) copre
   `src/wearusfm/**/*.py`, `scripts/contrastive_*.py`, `scripts/slurm/contrastive*.sbatch`, `train_jepa.py`,
   `lib_code_dir.sh`: nessun file di `tests/cpu` fuori dai 29.
2. Moduli non pinnati importati (analisi statica): test_pose_training 7 file, test_pretraining_loader 3,
   frozen_campaign_support 2, contrastive_review_oracle 1 (come nel rilievo). Chiusura dinamica degli import:
   6 moduli, anche `test_ddp_equivalence.py` e `test_pose_targets.py` (transitivi, `test_pose_training.py:18-19`).
3. Base: `--verify` sul clone pulito -> `{"verified": true, "all_passed": true}`, rc 0.
4. Mutazione: in coda a `tests/cpu/test_pose_training.py` `def _pose_setup(*a, **k): raise RuntimeError(...)`
   -> `--verify` ancora `{"verified": true, "all_passed": true}`, rc 0; gli stessi file della ricevuta
   `test_contrastive_cuda_check.py` + `test_contrastive_common_checkpoint.py` -> `2 failed, 3 passed`.
   La ricevuta resta "verificata e tutta passata" per un albero in cui i suoi test falliscono.
5. Tolta la riga 30 (controllo del source): `--verify` -> verified true; mmap_compatibility + release +
   source_compat `62 passed`. Nessun test cita `contrastive_local_validation`/`local_validation` (grep rc 1); l'unico
   glob su `scripts/contrastive_*.py` (`test_contrastive_mmap_compatibility.py:40`) cerca solo `torch.load(mmap=True)`.
6. Chiavi della ricevuta: nessuna versione di Python/torch/numpy/pytest ne' piattaforma (solo path dell'interprete
   e variabili d'ambiente); `-q` toglie anche l'intestazione dal log. `command` e `test_files`: 29/29, stesso ordine;
   `--verify` non li confronta.

Motivi cercati per confutare (nessuno sufficiente):
- Nessun documento dichiara esclusi i moduli di supporto o le versioni: il prontuario (r.191-192) dice "hash di test
  e log ... `--verify` ricontrolla ogni pin"; la risposta N-5 "hash dei test".
- La gravita' resta MINORE, non BLOCCANTE: nessun codice della catena legge la ricevuta (solo lo strumento stesso);
  `delivery_pin.json` pinna il commit di contenuto fbfde797 e lo sha della ricevuta (36d1d4d4..., 9629 B, ricontrollato);
  i 6 moduli non cambiano fra 438c8d65 e fbfde797; ripetuto il comando dei 29 file a 0efa4d45: `465 passed in 106.00s`, rc 0.
  Nessuna evidenza sbagliata in questa consegna; il difetto e' latente nello strumento. Le parti versioni e
  command/test_files da sole sarebbero NOTA (ricevuta falsificata rilevabile solo dal pin esterno).

Non verificato: le "8/8" del revisore (insieme non noto); la mutazione C9a con l'intera suite di 465 (inutile: nessun test
importa lo strumento).
