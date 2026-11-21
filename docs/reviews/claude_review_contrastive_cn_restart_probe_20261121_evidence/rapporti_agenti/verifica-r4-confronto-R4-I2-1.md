# Verifica avversaria di R4-I2 (soglia 1e-6 per tensore e cancellazione fra rank)

Esito: il rilievo regge in forma ridotta, con un meccanismo diverso da quello indicato. Gravita' proposta: importante (non bloccante).

## Che cosa ho eseguito
- `agents/verifica-r4-confronto-R4-I2-1/sim.py`: somma fp32 di 4 addendi gia' divisi per 4 in due rotazioni ad anello, 2e5 elementi.
  Con gradienti indipendenti fra rank, qualunque rapporto rumore/media: rel circa 5e-8 e rapporto di cancellazione delle norme al massimo 2.
  Con anticorrelazione costruita: rel circa 1,3e-8 per unita' di rapporto, quindi la soglia si supera da un rapporto delle norme di circa 70.
- `shapes.py`: modello costruito dalla config storica CN (266 parametri del modello, piu' i 4 della testa di proiezione = 270).
  Tensori fp32 piccoli: `anchor_heads.rms.bias` (1), `anchor_heads.env.bias` (1), `anchor_heads.bands.bias` (5), `anchor_heads.ms.ms_fast.bias` (6).
- `sim_small.py`: tensori piccoli con gradienti indipendenti, 20.000 prove, due rotazioni diverse. P(rel>1e-6): n=1 circa 1,1-1,2 %,
  n=2 0,06 %, n>=6 0 (massimo 3e-7).

## Dal codice
- Lo scenario del revisore (anticorrelazione su tensori grandi) non si presenta in CN. Le chiavi NCE sono staccate
  (`src/wearusfm/training/contrastive_objectives.py:255`). L'unico gather con gradiente e' RnC (riga 274), e in CN ha peso 0
  (`contrast_rnc_weight 0.0` nella traccia storica). I gradienti locali sono somme su campioni locali, indipendenti fra rank.
  Sui tensori grandi il rapporto si concentra intorno a 2 o meno, cioe' circa 35 volte sotto quello necessario.
- Rimane il caso dei due bias scalari: con un solo elemento, la somma dei 4 rank puo' essere quasi nulla per caso.
  rel diventa allora circa u per Sigma|g|/|S| e puo' arrivare a 1e-3. Per ogni scalare la probabilita' e' circa 1 %,
  da moltiplicare per la probabilita' che l'ordine dell'anello cambi fra la disposizione fredda e quella calda (dedotta, non misurata).
  Per il run, circa 1-2 %: `classify` (compare.py:115-116) darebbe allora HYPOTHESIS_UNSUPPORTED per un arrotondamento puro.
- Nessun altro controllo lo intercetta: `excessive` si calcola su ogni tensore (riga 289) e prevale su PARTIAL e SUPPORTED.
  Il caso si puo' riconoscere dopo dal report. `details` riporta per rank e tensore la L2 locale prima della divisione
  (osservata prima di `allreduce_hook`, observe.py:77) e la L2 ridotta, cioe' la media. Il rapporto
  Sigma_r||local_r|| / (4 ||reduced||) si ottiene quindi senza i vettori.

## Correzione minima
Lo status congelato resta invariato. Si aggiunge a `details` un campo descrittivo per tensore,
`cancellation_ratio = sum_r ||local_r|| / (4 ||reduced||)`, per la gamba ordinaria, con l'elenco dei tensori in eccesso.
Nel protocollo va dichiarato che un eccesso su un tensore di 1-2 elementi con rapporto alto resta compatibile con l'arrotondamento.
Questa e' una lettura descrittiva, non una nuova soglia.
