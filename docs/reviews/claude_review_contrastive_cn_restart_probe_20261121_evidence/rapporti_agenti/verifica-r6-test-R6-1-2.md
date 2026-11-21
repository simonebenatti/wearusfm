# Verifica avversaria di R6-1 (chiave verifica-r6-test-R6-1-2)

Esito: R6-1 regge nel merito, ma va ridimensionato a **minore** (rilievo sulla qualita' dei test, collegato a B-1, che resta il bloccante). Il conteggio va corretto: i casi mascherati sono **5 su 7**, non 7.

## Che cosa ho eseguito (simulazione)
Lo shim e' in `agents/verifica-r6-test-R6-1-2/shim/shim22.py`. E' una SIMULAZIONE di serialization.py:1012-1014 di torch 2.2: solleva ValueError quando mmap=True e f non e' una str.

1. Ho lanciato `pytest -p shim22 tests/cpu/test_cn_restart_observe.py` e ho ottenuto `1 failed, 7 passed`. Fallisce il test positivo, con "f must be a string filename...". Senza lo shim ottengo `8 passed`. Ho quindi riprodotto quello che riporta il revisore.
2. Ho scritto `msgs.py`, che riproduce le 7 mutazioni del test e stampa il messaggio di ciascuna. Il caso `none` e' il file non alterato.
   - Con lo shim:
     - `pin` e `rank` danno il rifiuto giusto, "invalid observation receipt or vector pin". Il controllo avviene prima di torch.load (observe.py:136-142).
     - `none`, `name_collision`, `shape`, `gradient_hash`, `slot` e `bucket_members` danno l'errore di mmap.
   - Con il torch reale:
     - `none` non solleva errori.
     - `name_collision` e `bucket_members` danno "invalid parameter or bucket inventory".
     - `shape` e `gradient_hash` danno "observed gradient differs from its receipt".
     - `slot` da' "bucket/parameter mapping differs".
3. Il fixture passa anche con lo shim, come dice il revisore. I figli di mp.spawn reimportano torch senza il plugin, quindi la loro chiamata a load_observation (test:51) non passa dallo shim. Sul torch 2.2 vero fallirebbe anche il fixture.

## Perche' non e' bloccante
- Il difetto del run vero e' B-1 (observe.py:144), gia' registrato come bloccante. R6-1 non introduce un secondo modo in cui il run diventi INCONCLUSIVE: spiega solo perche' B-1 e' sfuggito.
- Su una torch che rifiuta Path la suite non risulta verde: il test positivo fallisce, e fallirebbe anche il fixture. B-1 e' sfuggito perche' la venv locale usa torch 2.14, che accetta Path. L'assenza di `match=` c'entra poco.
- Il difetto reale dei test e' l'assenza di `match=` in test:106. Senza `match=`, in 5 casi su 7 il rifiuto non dimostra quale controllo sia scattato. E' una debolezza di copertura senza effetto sul run.

## Correzione minima
- Codice (B-1): in observe.py:144 usare `torch.load(str(tensor_path), map_location="cpu", mmap=True, weights_only=True)`.
- Test: dare a ciascun caso un `match=` col messaggio osservato sopra.
  - pin, rank: "invalid observation receipt"
  - name_collision, bucket_members: "invalid parameter or bucket inventory"
  - shape, gradient_hash: "observed gradient differs"
  - slot: "bucket/parameter mapping differs"
- Test di regressione: un monkeypatch di torch.load che rifiuta f non-str con mmap=True, con load_observation chiamato nel processo principale. Facoltativo, ma economico.

## Verifiche positive
Con il torch reale ogni mutazione e' respinta dal controllo previsto. Il fixture produce i file di tutti e quattro i rank.
