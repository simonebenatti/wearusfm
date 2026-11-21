# Verifica avversaria R4-I2 (soglia 1e-6 e cancellazione fra rank)

Esito: il meccanismo e' matematicamente corretto, ma lo scenario non e' plausibile in questo modello. Gravita' corretta: **minore** (dichiarata: importante). Non e' un bloccante.

## Codice
- compare.py:75-86: relative_l2 = ||o-w|| / max(||o||, ||w||, 1e-12) in float64. Riga 289: `excessive |= relative_l2 > LIMIT`. classify (106-109): un solo eccesso basta per HYPOTHESIS_UNSUPPORTED, prima del controllo reduced_changed.
- Il criterio e' congelato cosi' in docs/decisioni.md:8405/8407. Il codice e' conforme.
- observe.py:76-79: il ramo "local" e' il buffer nativo copiato prima di `div_(size)`. "reduced" e' sum_r(g_r/4). In details ci sono, per rank e tensore, local.ordinary_l2 e reduced.ordinary_l2. Quindi R = sum_r||g_r|| / (4 ||ridotto||) si puo' calcolare a posteriori dal report, ma non in automatico.

## Eseguito (simulazione su Mac, numpy fp32, non il run vero)
Script: agents/verifica-r4-confronto-R4-I2-2/sim.py. Usa 4 addendi gia' divisi per 4 e somme sequenziali fp32 ad anello che partono da rank diversi (come i chunk di un ring NCCL). La media vera e' nulla (caso peggiore). L'anticorrelazione viene forzata togliendo alpha volte la media fra i rank. Il valore riportato e' il peggiore fra le coppie di ordini.

| alpha | R | rel_l2 | limite 6uR |
|---|---|---|---|
| 0 (indipendenti) | 2,00 | 5,6e-8 | 7,2e-7 |
| 0,5 | 3,61 | 6,8e-8 | 1,3e-6 |
| 0,9 | 17,4 | 2,5e-7 | 6,2e-6 |
| 0,99 | 173 | 2,4e-6 | 6,2e-5 |

In pratica rel_l2 vale circa 1,4e-8 x R. La soglia viene superata solo per R >= ~70. La stima del revisore (~2000) e' piu' alta, ma la conclusione e' la stessa. Il limite di caso peggiore 6u*R supera 1e-6 gia' per R > ~2,8, ma resta un limite superiore, non il valore tipico.

## Perche' R >= ~70 non e' plausibile qui (dedotto dal codice)
- Con batch per rank indipendenti (8 finestre per rank), R vale circa 2. Con gradienti correlati positivamente, R < 2. Un R molto piu' alto richiede un vincolo che leghi i rank e faccia sommare a zero i gradienti globali pur con gradienti per rank non nulli. Per esempio una centratura o normalizzazione globale, oppure una loss invariante per traslazione che sia l'unica fonte del gradiente di un bias.
- Ho cercato in src/wearusfm: nessun SyncBatchNorm o BatchNorm. Nessun all_reduce o all_gather nel modello, oltre a quelli dei conteggi in run.py e del gather delle loss contrastive.
- Proiettore (model/contrastive.py:15-27, normalize_projection): InfoNCE lavora su q_raw normalizzato L2, che non e' invariante per traslazione. Quindi il bias finale non ha gradiente globale strutturalmente nullo.
- RnC lavora su zm dell'encoder, che riceve gradiente anche dai termini base (jepa, ms_*, log_*, pose). Nessun tensore ha come unica fonte un termine con somma nulla.
- Bias seguiti da una LayerNorm: gradiente matematicamente nullo per campione, quindi rumore di arrotondamento indipendente fra i rank, con R circa 2.
- Bucket float64 (8.560 elementi): u = 1,1e-16, irrilevante.
- Dato storico: alla 545 la grad_norm globale differisce fra il continuo e CN_A di 2,2e-12 relativi (4,336711902204087 contro 4,336711902194546). E' coerente con un arrotondamento ordinario.

## Mitigazioni gia' presenti
- La documentazione (contrastive_cn_restart_probe_20261121.md:17) dice gia' che un eccesso "da solo non identifica un difetto di ripristino".
- details riporta le L2 locali per rank e quelle ridotte, quindi R si puo' calcolare a posteriori. I vettori restano su WORK.
- Nessun esito autorizza il training.

## Correzione minima (facoltativa, non bloccante)
Aggiungere a ogni riga di details un campo descrittivo `cancellation_ratio = sum_r local.ordinary_l2_r / (4 * reduced.ordinary_l2)`, oppure calcolarlo nella lettura. Nella lettura scientifica va dichiarato che un eccesso con R >~ 50 resta compatibile con l'arrotondamento. Lo status congelato resta invariato.
