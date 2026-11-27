# C5 critico-regressioni: diff 9a2ce7e7..8c72655c

Esito: nessun bloccante. B-1 e B-2 sono chiusi, i nove punti sono chiusi. Restano tre minori e alcune note.

## Eseguito
- `--package` dal clone: package e9310d67..., plan 4f538cf1..., uguali al prompt.
- I cinque file cn_restart: 73 passati in 10,75 s.
- Mutazioni (plugin nel mio agente, clone non toccato), tutte uccise dai test nuovi:
  M1 Path in mmap (19 falliti), M2 ogni copia dormiente accettata (2), M3 niente scala degli scalari (5),
  M4 TimeoutExpired non catturato (3), M5 niente ritorno anticipato (5), M6 scala senza /RANKS (2).
- Simulazione dell'arrotondamento fp32 con 4 addendi, 200k estrazioni, 4 ordini di somma (sim_small.py):
  scalari col rapporto originale P(>1e-6) = 1,6-1,9%, max 1,3e-2; con il denominatore nuovo max 2,3e-7, mai >1e-6.
  Tensori da 5 e 6 elementi (255 bands.bias, 259 ms_fast.bias): max 5,8e-7, mai >1e-6.
  Limite analitico: la divisione per 4 e' esatta e due ordini di somma differiscono al piu' di 6u*mean|g| = 3,6e-7.
- Regex del binding su varianti del file di review (binding_regex.py): storico + blocco a margine sinistro = 1 blocco valido;
  rientrato = 0; prompt che ripete il recinto = 2; CRLF = 0.
- Parametri piccoli del modello vero: 253 rms.bias (1), 257 env.bias (1), 255 (5), 259 (6), 261 (24), 265 (20), piu' quattro fp64.

Dettagli nell'oggetto strutturato.
