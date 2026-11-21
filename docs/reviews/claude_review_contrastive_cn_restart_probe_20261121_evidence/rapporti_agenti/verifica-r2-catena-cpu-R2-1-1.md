# Verifica avversaria R2-1 (B-1 sulla catena vera)

Esito: il rilievo REGGE, gravita' bloccante confermata.

## Che cosa ho eseguito
- Ho letto il sorgente del torch del cluster (torch22_sorgente_20261121.out.txt): `if mmap: ... if not isinstance(f, str): raise ValueError("f must be a string filename in order to use mmap argument")` (serialization.py:1012-1014). Un `PosixPath` non e' `str`.
- SIMULAZIONE: shim che avvolge torch.load e solleva lo stesso ValueError del 2.2 quando mmap=True e f non e' str (agents/verifica-r2-catena-cpu-R2-1-1/conftest_shim.py). Suite observe + compare del clone: `1 failed, 20 passed`. Fallisce test_true_local_and_reduced_gradients_and_independent_rank_files con `f = PosixPath('.../vectors.rank0.pt')`, `ValueError: f must be a string filename in order to use mmap argument`.

## Che cosa deduco dal codice
- observe.py:135 `tensor_path = directory / f"vectors.rank{rank}.pt"` (Path), :144 `torch.load(tensor_path, ..., mmap=True, ...)`.
- compare.py:257 chiama load_observation per ogni rank e modalita'; compare.py:317-318 cattura ValueError e scrive `status="INCONCLUSIVE"`.
- contrastive_cn_restart.py:133,171 esegue compare dentro la stessa allocazione dopo le due gambe; :114 rifiuta una radice gia' esistente, quindi il run non si ripete. Il piano non prevede di rifare il solo confronto su CPU: un confronto rifatto a parte dopo la correzione sarebbe un'azione nuova, fuori dal piano.
- Nessun altro controllo intercetta il problema prima delle GPU: load_observation non viene chiamata dalle gambe. Gli altri torch.load con mmap (contrastive_target_resume.py:58, contrastive_common_*.py) usano gia' str(path).
- Mitigazione parziale: i vettori restano su disco, quindi un confronto rifatto a parte su CPU ne recupererebbe il contenuto. Il report del job resterebbe comunque INCONCLUSIVE, e il recupero richiederebbe una decisione fuori dal piano. La gravita' resta bloccante.

## Correzione minima
`torch.load(str(tensor_path), map_location="cpu", mmap=True, weights_only=True)` in observe.py:144, piu' un test con lo shim (o un monkeypatch di torch.load che rifiuti i non-str quando mmap=True).
