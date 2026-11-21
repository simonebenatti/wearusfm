# Verifica avversaria del rilievo I-1 (r7-critico): soglia 1e-6 e cancellazione fra rank

Esito: il rilievo regge in forma ridimensionata. Gravita' corretta: importante, non bloccante.

## Cosa ho eseguito
- `agents/verifica-r7-critico-I-1-1/small_params.py`: il modello vero (config di CN_A) ha 270 parametri e 28.559.133 elementi. I parametri con al massimo 64 elementi sono pochi. Due sono scalari fp32: `anchor_heads.rms.bias` (id 253) e `anchor_heads.env.bias` (id 257). Nella ricetta CN `jepa.anchor_weight=0.2`, quindi hanno gradiente non nullo. Gli altri tensori piccoli hanno da 5 a 54 elementi (type_bias 4x6, ms_fast 6, pose_head.out.bias 20, ...). Quelli fp64 stanno nel bucket da 68480 byte, che ha la stessa dimensione in entrambe le riprese.
- `agents/verifica-r7-critico-I-1-1/ringsim.py`, output in `ringsim.out.txt`. E' una SIMULAZIONE CPU, non NCCL: somma ad anello in fp32 di 4 contributi, con rank di partenza diverso per elemento fra le due riprese, poi divisione esatta per 4 e `rel_l2` come in `compare_vector`. Risultati, con contributi per rank indipendenti, media/sd = 0 (rumore puro):
  - n=1: P(rel>1e-6) = 1,26% (massimo 1,7e-4); con media/sd 0,5 si scende a 0,75%; con 2,0 si arriva a 0.
  - n=5...4096: P = 0 su 20.000 prove, mediana circa 5e-8, massimo 6,4e-7 (n=5).
- Il secondo caso della simulazione del revisore (rel 3,1e-4) usa contributi anticorrelati costruiti apposta, `base*(+1,+1,-1,-1)`. Nel codice e nella ricetta non ho trovato un meccanismo che produca questa anticorrelazione strutturale fra shard di dati indipendenti. Un parametro invariante per traslazione ha gradiente analitico nullo per ogni campione. I suoi valori locali sono quindi rumore di arrotondamento indipendente, cioe' il caso "rumore puro", che per n>=5 non supera la soglia.

## Dal codice
- `compare.py:289`: `excessive |= reduced.relative_l2 > LIMIT`, con denominatore `max(nx, ny, 1e-12)` (`:86`). `classify` (`:115`) controlla `excessive` prima di ogni altro esito favorevole. Nessun altro controllo lo intercetta: un singolo tensore oltre la soglia porta tutto il report a HYPOTHESIS_UNSUPPORTED. I dettagli per tensore (`details`) restano nel report, ma lo `status` resta quello.
- La soglia e' congelata in `docs/decisioni.md` (voce 8405/8407) e nel protocollo, riga 17. La modifica proposta e' prospettica, prima della misura, quindi legittima.

## Valutazione
- Il meccanismo regge per i tensori scalari. Con gradiente vero vicino all'equilibrio (media piccola rispetto al rumore fra rank, plausibile con 8 finestre per rank), un HYPOTHESIS_UNSUPPORTED dovuto al solo arrotondamento della riduzione ha probabilita' stimata di circa l'1% per scalare: circa 1,5-2,5% sui due bias delle teste anchor. Per i tensori con 5 o piu' elementi e contributi indipendenti, il rischio simulato e' nullo.
- Questo non e' un controesempio concreto sul run vero: dipende dal rapporto segnale/rumore dei due bias al passo 545, non misurato. Dipende anche dall'ordine effettivo NCCL per quegli elementi, non verificato. Quindi il rilievo resta importante, non bloccante. La conseguenza sarebbe comunque un'attribuzione errata in un diagnostico a tentativo unico, e la correzione costa poco.

## Correzione minima
In `compare()`, dopo il ciclo sui rank, sostituire il criterio `excessive` con un rapporto su scala locale, calcolato per parametro:
`scale = sum_r local_l2_r / 4` (le norme locali sono gia' in `details`; i locali sono identici fra riprese) e
`rel_scaled = reduced.difference_l2 / max(nx, ny, scale, 1e-12)`, con `excessive = any(rel_scaled > 1e-6)`.
Un limite superiore vale per gli errori di somma: ||Δ|| <= c·eps·||Σ|g_r|||/4 <= c·eps·Σ_r||g_r||/4. Riportare per ogni tensore oltre 1e-6 entrambi i rapporti. Aggiornare con la stessa formula `docs/decisioni.md`, il protocollo (riga 17), `relative_l2_denominator` nel report (`:314`) e un test (scalare con contributi che si annullano: il vecchio rapporto supera la soglia, il nuovo no).
