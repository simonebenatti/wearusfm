# Verifica avversaria R3-B0 (B-1: torch.load(Path, mmap=True) su torch 2.2)

Esito: il rilievo REGGE, gravita' bloccante confermata. Non sono riuscito a confutarlo.

## Che cosa ho eseguito (Mac, torch 2.14 CPU, clone a 9d34dc69)
1. Ho riletto il sorgente del cluster (`leonardo/torch22_sorgente_20261121.out.txt`): in serialization.py:1012-1014,
   con `mmap` attivo, `if not isinstance(f, str): raise ValueError("f must be a string filename in order to use mmap argument")`.
   `pathlib.Path` non e' `str`, quindi la chiamata fallisce.
2. SIMULAZIONE: uno shim (`agents/verifica-r3-osservatore-R3-B0-2/conftest_shim.py`) avvolge `torch.load` e solleva lo stesso
   ValueError quando `mmap=True` e `f` non e' `str`. Con il codice del clone,
   `tests/cpu/test_cn_restart_observe.py` + `test_cn_restart_compare.py`: **1 failed, 20 passed**; il test fallito e'
   `test_true_local_and_reduced_gradients_and_independent_rank_files`, in `contrastive_cn_restart_observe.py:144`,
   `ValueError: f must be a string filename in order to use mmap argument`.
3. Con la copia corretta (`str(tensor_path)`, unica riga cambiata) iniettata in `sys.modules` sotto lo stesso shim:
   **21 passed**.

## Che cosa ho dedotto dal codice
- L'unico `torch.load` del diagnostico e' `contrastive_cn_restart_observe.py:144`; `_leg.py` e `contrastive_cn_restart.py` non
  rileggono le osservazioni, quindi nessun controllo precoce intercetta il problema prima del consumo delle GPU.
- Sequenza: `contrastive_cn_restart.py:168-171` esegue le due celle (ordinary, warm) e poi il confronto;
  `compare.py:257` chiama `load_observation` per ogni rank e modalita'; `compare.py:317-318` cattura `ValueError` e scrive
  `status="INCONCLUSIVE"`. Il fallimento avviene quindi dopo entrambe le riprese a 4 GPU, sull'unico tentativo previsto.
- Attenuante parziale, non sufficiente: i file delle osservazioni restano su disco e il confronto e' CPU, quindi potrebbe essere
  rieseguito con codice corretto; ma servirebbe un nuovo checkout/release fuori dal piano congelato, e il report del run resta
  INCONCLUSIVE. Il controesempio (run vero -> INCONCLUSIVE per ragione non scientifica) e' quindi concreto.
- Nota sui test: `test_cn_restart_observe.py:107` (casi negativi con `pytest.raises(ValueError)`) passerebbe anche sotto il
  difetto, perche' lo ValueError di torch 2.2 soddisfa lo stesso `raises`: il test negativo non distingue la causa.
  Il test positivo a riga 50 gira nei processi Gloo figli (lo shim non vi era attivo); il fallimento e' emerso a riga 73.

## Correzione minima
`torch.load(str(tensor_path), map_location="cpu", mmap=True, weights_only=True)` a observe.py:144.
Nei test: shim di 2.2 attivo (fixture che avvolge `torch.load`) e, nei negativi, `match=` sul messaggio atteso del validatore,
cosi' un errore di caricamento non passa per un rifiuto corretto.

## Verifiche positive
- La correzione con `str()` conserva il percorso `weights_only` + `mmap` e fa passare tutti i 21 test di observe/compare sotto lo shim.
- Nessun altro `torch.load` o accesso mmap nel diff.
