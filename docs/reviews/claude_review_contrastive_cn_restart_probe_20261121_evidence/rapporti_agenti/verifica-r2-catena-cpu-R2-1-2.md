# Verifica avversaria R2-1 (B-1 sulla catena vera) — esito: REGGE, bloccante

Commit verificato: 9d34dc690cf60fcbfe405279dc17546655fa3f5b (clone detached).

## Dal codice (dedotto)
- observe.py:135 `tensor_path = directory / f"vectors.rank{rank}.pt"` (Path); observe.py:144 `torch.load(tensor_path, ..., mmap=True, weights_only=True)`.
- torch 2.2.0a0 del cluster (lettura di Claude, serialization.py:1012-1014): nel ramo zip, `if mmap: if not isinstance(f, str): raise ValueError(...)`. Il file e' scritto da `torch.save(vectors, stream)` (observe.py:123), formato zip di default: il ramo viene raggiunto.
- compare.py:257 chiama load_observation per ogni rank/gamba; compare.py:317-318 cattura ValueError -> `status=INCONCLUSIVE` e scrive il report.
- contrastive_cn_restart.py:171-172: il report esiste, quindi il runner non registra nemmeno execution_failure: esito ufficiale INCONCLUSIVE dopo l'unico tentativo GPU.
- Nessun altro controllo lo intercetta prima: le gambe GPU non rileggono i vettori con mmap; `--package`/`--validate-only` non eseguono compare.
- Tutti gli altri `torch.load(..., mmap=True)` del repo usano `str(...)` (grep su src/ e scripts/): il difetto e' isolato a observe.py:144.

## Eseguito (simulazione dichiarata)
Shim `agents/verifica-r2-catena-cpu-R2-1-2/shim22.py`, caricato come plugin pytest, riproduce il controllo del 2.2 (ValueError se mmap=True e f non e' str).
- `tests/cpu/test_cn_restart_observe.py` con shim: `1 failed, 7 passed`; fallisce proprio il test positivo `test_true_local_and_reduced_gradients_and_independent_rank_files` con `ValueError: f must be a string filename in order to use mmap argument` (observe.py:144).
- Senza shim (torch 2.14 accetta Path): `8 passed`.
- `tests/cpu/test_cn_restart_compare.py` con shim: `13 passed` — nessun test di compare percorre load_observation su file veri, quindi la suite non poteva accorgersene.
- Osservazione aggiuntiva: i test negativi parametrizzati (observe test l.87-107, `pytest.raises(ValueError)`) per i campi controllati dopo torch.load (sha locali, slot, membri del bucket) passerebbero sotto il 2.2 per la ragione sbagliata (il ValueError dello shim). Debolezza minore dei test.

## Ridimensionamento possibile, valutato
Le uscite delle gambe (vectors/observation/checkpoint) restano su WORK: dopo una correzione il confronto si potrebbe rifare su CPU senza GPU. Non cambia la gravita': l'esito registrato sarebbe INCONCLUSIVE, la cartella esiste gia' (run() rifiuta `root.exists()`), e il recupero richiederebbe commit, review e decisione nuovi fuori dal piano a un solo tentativo.

## Correzione minima
observe.py:144 -> `torch.load(str(tensor_path), map_location="cpu", mmap=True, weights_only=True)`; aggiungere un test con lo shim che rifiuta i Path con mmap=True (anche per i test negativi, che vanno resi specifici sul messaggio d'errore).
