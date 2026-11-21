# Verifica avversaria di R3-B1 (cuda_rng.1..3 nell'endpoint riscaldata)

Esito: il rilievo REGGE, gravita' bloccante.

## Dal codice (dedotto)
- run.py:1482 il payload salva `torch.cuda.get_rng_state_all()` del processo di rank 0 (4 dispositivi visibili).
- Continuo (nessuna ripresa): run.py:1462-1463 `torch.manual_seed(cfg.seed*1_000_003+rank+1)`, che semina tutti i dispositivi; rank 0 non usa i dispositivi 1..3, che restano allo stato del seme `cfg.seed*1_000_003+1`.
- Ripresa: run.py:1383 `torch.manual_seed(cfg.seed)` e run.py:1446-1450 ripristina solo il proprio dispositivo. Dispositivi 1..3 restano al seme `cfg.seed`.
- La gamba riscaldata (contrastive_cn_restart_leg.py:48-57) salva/ripristina solo `torch.cuda.get_rng_state(device)`; nessun manual_seed nel diff (grep). Quindi la riscaldata ha cuda_rng.1..3 come CN_A, diversi dal continuo.
- compare.py:241 confronta continuo contro riscaldata con `checkpoint_differences`; i tensori uint8 diversi vanno in `metadata_or_schema_different` (riga 53); compare.py:248 solleva ValueError; main() (righe 312-317) lo trasforma in INCONCLUSIVE. La riga 248 viene prima del ciclo sulle osservazioni: nemmeno un HYPOTHESIS_UNSUPPORTED (differenza locale) arriverebbe al report.
- Nessun filtro di cuda_rng in compare.py (grep), `checkpoint()` (contrastive_target_resume.py:57) carica il payload intero. Il test test_all_checkpoint_leaves_checked_and_classified non copre questo caso.

## Prove
- Lettura sui checkpoint veri del cluster (rev_probe/leonardo/confronto_checkpoint_veri_20261121.out.txt): `cuda_rng ... uguali per dispositivo: [True, False, False, False]`; continuo vs A `metadata_or_schema_different ['cuda_rng.1','cuda_rng.2','cuda_rng.3']`; A vs B exact_payload True (deterministico per le riprese). rank_rng uguale fra continuo e A (non compare fra i diversi).
- Eseguito (simulazione, generatori CPU al posto degli stati CUDA): agents/verifica-r3-osservatore-R3-B1-1/sim.py con le funzioni vere di compare.py:
  `{'floating_tensors_different': [], 'metadata_or_schema_different': ['cuda_rng.1','cuda_rng.2','cuda_rng.3'], 'exact_payload': False}`
  `ValueError -> warm endpoint metadata/schema or nonfloat state differs -> INCONCLUSIVE`
  e, senza il require, `classify(False, True, False, False, True)` = PARTIAL_SUPPORT ("warm trajectory does not recover continuous").

## Controesempio
Spiegazione stretta vera: riscaldata identica al continuo in studente, teacher, Adam, coda, rank_rng. Il report esce INCONCLUSIVE alla riga 248, consumando l'unico tentativo. Qualunque esito scientifico (anche smentita) viene mascherato, perche' la riga 248 precede l'analisi dei gradienti.

## Correzione minima
Nel solo confronto riscaldata/continuo escludere `cuda_rng.1..3` (o meglio: confrontare `cuda_rng.0` e `rank_rng`, riportare cuda_rng[1:] in un campo separato "expected_serialization_difference"), dichiararlo nel protocollo e nel piano prima del lancio; test con pesi identici e cuda_rng[1:] diversi -> SUPPORTED. Ordinaria contro CN_A invariata (esatta). Alternativa: per la riscaldata confrontare cuda_rng[1:] contro CN_A (attesi uguali), il che autentica anche quella parte.
