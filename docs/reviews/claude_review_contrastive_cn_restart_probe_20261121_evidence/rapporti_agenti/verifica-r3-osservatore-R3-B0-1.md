# Verifica avversaria R3-B0 (B-1) - esito: REGGE, bloccante

Clone a 9d34dc690cf60fcbfe405279dc17546655fa3f5b.

## Dal codice
- observe.py:144 `torch.load(tensor_path, map_location="cpu", mmap=True, weights_only=True)`, `tensor_path = Path(directory) / ...` (riga 136): Path, non str.
- torch 2.2 del cluster (lettura torch22_sorgente, serialization.py:1012-1014): `if mmap: if not isinstance(f, str): raise ValueError(...)`, nel ramo zipfile; il file e' salvato con torch.save di default (zipfile, observe.py:122-123), quindi si entra proprio in quel ramo.
- Unico chiamante in produzione: compare.py:257, eseguito da contrastive_cn_restart.py:171 DOPO le due celle GPU (169-170). compare.py:317 cattura ValueError -> report `status=INCONCLUSIVE`, reason "ValueError: f must be a string filename...". Nessun controllo precedente lo intercetta (il validatore --package e i test girano su torch 2.14, dove Path e' accettato).
- Gli altri diagnostici del repo usano gia' `str(path)` con mmap (contrastive_common_checkpoint.py:58, contrastive_common_state_cuda.py:245, :378): incoerenza locale.

## Eseguito (SIMULAZIONE: sitecustomize che riproduce il controllo di 2.2 su torch.load)
- Test originali observe+compare con shim: 13 passed, 8 errors; tutti gli 8 su observe.py:144 con "ValueError: f must be a string filename in order to use mmap argument".
- Copia corretta (`str(tensor_path)`, solo quella riga) + test di observe puntati alla copia, con shim: tutti i test di observe passano (inclusi i controlli di load_observation su sha, dtype, shape e storage_offset == bucket_element_offset). L'unico fallimento (test_trace_joins... di compare) e' FileNotFoundError dovuto al test copiato fuori dal repo (ROOT relativo), non alla correzione.
- I test di compare originali passano anche con lo shim: non esercitano load_observation; la copertura del percorso reale sta solo in test_cn_restart_observe.

## Ridimensionamento considerato e respinto
I vettori grezzi restano su disco, quindi un comparatore corretto potrebbe essere rieseguito su CPU senza GPU; ma sarebbe una modifica del codice di lettura dopo la misura, fuori dal pacchetto approvato, e il protocollo (doc riga 21) legge lo `status` del report. Il run vero esce quindi INCONCLUSIVE per ragione non scientifica, in modo deterministico: bloccante.

## Correzione minima
observe.py:144 -> `torch.load(str(tensor_path), map_location="cpu", mmap=True, weights_only=True)`. Nel test di observe, uno shim (monkeypatch di torch.load) che rifiuti f non-str con mmap=True, per non dipendere dalla versione locale. Poi ricalcolare package_sha256/local_validation.
