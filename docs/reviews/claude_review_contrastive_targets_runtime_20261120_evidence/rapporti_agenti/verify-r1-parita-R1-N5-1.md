# Verifica scettica R1-N5 (r1-parita), scettico n. 1

Inizio: 2026-11-20 05:10:42

Rilievo: la memoria registrata per caso e' il picco cumulativo del processo (NOTA).

## Progressi

### Riproduzione (05:20 circa)
- Codice a fbfde797: `scripts/contrastive_target_parity.py:86-95` legge `max_memory_allocated/max_memory_reserved` senza alcun reset ed etichetta `peak_scope: "cumulative process peak, not reset by observer"`; `:211-213` richiede solo `cuda_measured is True` e `peak_allocated_bytes > 0`.
- `git grep reset_peak_memory_stats` in src/ e scripts/: nessuna occorrenza in run.py, contrastive_run.py, contrastive_target_parity.py (solo script di altre linee).
- Script mock `rev_runtime/tmp/r1n5_skeptic1/repro.py` (torch.cuda simulato): sei casi con picco proprio 5-6 GiB dopo un picco precedente di 9 GiB registrano tutti 9 GiB; reset mai chiamato.
- Cancello: memoria per caso 1 byte -> PASSED; 1 EiB -> PASSED; 0 -> INCONCLUSIVE; cuda_measured False -> INCONCLUSIVE. Il valore non entra in nessun confronto.
- Config C0 della coorte (grad_every 100, grad_steps [256,512,...], denso 513-576): aggiornamenti diagnostici 1..128 = [100]; 129..256 = [200, 256]. Il diagnostico dell'update 100 precede il punto 128: affermazione del rilievo confermata.
- `run.py:1816-1822`: `summary.gpu_max_memory_gib` = max fra rank di `max_memory_allocated()` a fine run, senza reset; `contrastive_run.py:355-369`: l'osservatore C0 gira nello stesso processo di `R.train`. Quindi il picco del C0 copre anche il periodo dei probe. Readout: `contrastive_benchmark_readout.py:224-225` solo presenza/positivita', `:472` copia descrittiva `full_batch_gpu_peak_gib`.

### Altre verifiche
- `contrastive_objectives.py:428`: il diagnostico (aggiornamenti 100, 200, 256 e denso 513-576) usa `torch.autograd.grad(..., retain_graph=True)` per termine: picco tipicamente piu' alto di un aggiornamento normale, ed entra nel cumulativo del punto 128.
- `run.py:1625` `opt.zero_grad(set_to_none=True)` precede `prepare`: durante i probe i gradienti vivi sono None. Sovraccarico di un clone: studente + teacher + bucket DDP (`_ddp_wrap`, `run.py:437`, senza gradient_as_bucket_view) + gradienti = circa 4 x 0,106 GiB = circa 0,43 GiB (28.559.133 parametri fp32), contro un picco storico di 13,72 GiB su 64 GiB.
- Documenti: prontuario, risposta, piani e prompt non dicono che il picco (e il tempo) del C0 comprendono i probe; il prontuario dice solo che i probe girano nel processo full C0 (riga 31). Il readout non confronta la memoria fra bracci e non ha soglie.
- pytest `tests/cpu/test_contrastive_target_parity.py -k "memory or requires_both"`: 2 passati in 1,94 s; copia git pulita.

## Esito
- refuted = false. Gravita' NOTA (confermata).
- Fatti del rilievo riprodotti; righe citate corrette (86-95, 211-213).
- Attenuanti: la portata cumulativa e' dichiarata in ogni record (`peak_scope`); non azzerare e' una scelta che protegge `summary.gpu_max_memory_gib` del trainer (un reset toglierebbe gli update 1-127 dalla ricevuta); il record ha anche `allocated_bytes/reserved_bytes` correnti, non cumulativi; nessun cancello dipende dal valore; per il C0 la distorsione e' conservativa (al piu' circa 0,4 GiB se i cloni sono liberati) e puo' non cambiare il massimo se il picco del run cade altrove.
- Resta il difetto di chiarezza: nessun documento o campo del readout dichiara che `full_batch_gpu_peak_gib` del C0 comprende i probe; un confronto descrittivo C0 contro CN/CR (storicamente 13,7224 GiB uguali) mostrerebbe un C0 piu' alto senza motivo apparente.
- Imprecisione minore del titolo: per caso si registrano anche valori correnti, non solo il picco.
- Non verificato: entita' reale su A100 con torch 2.2 (niente CUDA qui); liberazione effettiva dei cloni DDP fra i casi (eventuali cicli di riferimento cambierebbero solo l'entita'); il tempo dei probe nella proiezione del C0 e' un rilievo distinto (M1 di r1-parita), non valutato qui.
