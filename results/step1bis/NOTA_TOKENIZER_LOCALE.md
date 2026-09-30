# Nota: prove locali sul tokenizer NeuroRVQ (30/09/2026, Mac, CPU)

**Analisi DESCRITTIVE, non congelate: non cambiano nessun verdetto di D5a e non sono D5b.** Servono a capire
*perche'* V4 e V3 sono usciti come sono usciti. Fatte dopo aver visto i risultati del run vero (job 59048369): vanno
lette come spiegazioni, non come nuove soglie.

Setup: torch 2.14 CPU in un ambiente virtuale fuori dal repo (`~/.venvs/wearusfm-tok`), checkpoint
`NeuroRVQ_EMG_tokenizer_v1.pt` scaricato da Hugging Face (574.583.162 byte, sha256 `0d255bcc9f1c75ccc374cba06eab476f15bf5fb2a87115d6dc8d2dc0adadce49`,
**identico** a quello del run sul cluster), codice NeuroRVQ al commit `926e770`. Ingressi: le finestre gia' scalate salvate
dal run vero (`arrays_59048369`). 200 finestre da 3,2 s per dataset. Script: `scripts/step1bis_cpu_vs_gpu.py`,
`scripts/step1bis_margins.py`, `scripts/step1bis_scale_sweep.py`.

## 1. I codici non dipendono dall'hardware: il run e' riproducibile su un'altra piattaforma
Stessi ingressi, calcolati su CPU (Mac, float32) invece che su A100. Frazione di codici uguali a quelli del run vero:

| Dataset | Livello 0 (4 rami) | Livello 8 | Livello 15 |
|---|---|---|---|
| emg2pose | 1,000 / 1,000 / 1,000 / 0,999 | 1,000 / 1,000 / 1,000 / 0,987 | 0,999 / 0,999 / 0,999 / 0,941 |
| CapgMyo | 1,000 / 1,000 / 1,000 / 0,999 | 1,000 / 1,000 / 1,000 / 0,990 | 0,998 / 0,999 / 0,999 / 0,959 |
| putEMG | 1,000 / 1,000 / 1,000 / 0,999 | 0,999 / 1,000 / 1,000 / 0,989 | 0,999 / 1,000 / 0,999 / 0,938 |

Accordo >= 99,9% nei rami 0-2 a tutti i livelli; il ramo 3 (banda alta) e' un po' piu' sensibile ai livelli profondi
(94-99%). **La fragilita' vista in V4 non e' rumore numerico**: e' sensibilita' al rumore fisico. Conferma anche che i
numeri del run vero non dipendono dalla macchina.

## 2. Perche' i codici sono fragili: margini piccoli, per costruzione
Su emg2pose, per ramo e livello: margine = distanza^2 fra il codice migliore e il secondo (sui vettori normalizzati;
massimo teorico 4) e frazione di flip con il rumore congelato di V4 (bianco, deviazione standard = noise floor).

| Ramo | Livello | Norma mediana del residuo | Margine 5% / 25% / mediana | Flip: tutti | Flip: margine basso (25%) | Flip: margine alto (25%) |
|---|---|---|---|---|---|---|
| 0 | 0 | 6,22 | 0,0033 / 0,0190 / 0,0438 | 0,536 | 0,724 | 0,316 |
| 0 | 8 | 3,04 | 0,0031 / 0,0178 / 0,0406 | 0,932 | 0,963 | 0,892 |
| 0 | 15 | 1,42 | 0,0032 / 0,0158 / 0,0383 | 1,000 | 1,000 | 0,999 |
| 2 | 0 | 8,59 | 0,0037 / 0,0213 / 0,0483 | 0,332 | 0,556 | 0,131 |
| 2 | 8 | 3,82 | 0,0032 / 0,0207 / 0,0495 | 0,685 | 0,786 | 0,519 |
| 3 | 0 | 7,93 | 0,0033 / 0,0195 / 0,0459 | 0,345 | 0,600 | 0,119 |
| 3 | 15 | 1,16 | 0,0032 / 0,0184 / 0,0413 | 0,998 | 0,999 | 0,995 |

(tutti i livelli 0, 1, 2, 4, 8, 15 di tutti i rami sono nello script; la replica dei codici coincide con quella dell'A100 al 99,96-99,99%
nei rami 0-2 e 98,2% nel ramo 3.)

Letture:
1. **Il margine mediano e' ~0,04 gia' al livello 0 e non cambia con la profondita'**: 8192 codici unitari in 128 dimensioni sono
   fitti, e ogni livello rinormalizza il residuo prima della ricerca del vicino. Il margine 5% e' ~0,003.
2. **I flip non dipendono solo dal margine**: anche nel quartile di margine piu' alto il livello 0 del ramo 0 cambia nel 32% dei
   casi. Il rumore congelato (circa -20 dB) sposta le feature dell'encoder di piu' del margine tipico: contribuiscono
   sia la densita' dei codici sia la sensibilita' dell'encoder.
3. **La norma del residuo scende lentamente** (6,2 -> 1,4 nel ramo 0 in 16 livelli) perche' si sottrae ogni volta un vettore
   unitario: i livelli profondi lavorano su una direzione che e' quasi rumore, e a livello 15 cambiano il 100% dei codici.
4. I rami 2 e 3 (bande alte: >125 Hz e >250 Hz) sono i piu' stabili al livello 0 (flip 33-35%), il ramo 1 (>60 Hz) il meno stabile (69%).

Conseguenza per D5a/D5b: un livello «stabile al 75% in tutti e 4 i rami» e' fuori portata gia' al livello 0, anche a
rumore molto piu' basso (vedi il controllo a 0,1x nel riepilogo). Non e' un difetto dell'esecuzione: e' una proprieta'
del tokenizer. Un target discreto piu' robusto richiederebbe, ad esempio, solo il livello 0 di un solo ramo con un
criterio diverso (nuova decisione), oppure un target continuo (feature prima della quantizzazione).

## 3. La scala d'ingresso: conta non essere troppo piccoli
NMSE mediano per token (dominio standardizzato) al variare di un moltiplicatore della scala scelta nel run (x1 =
scala nativa di emg2pose, fattore 21,7; la «varianza unitaria» del run corrispondeva a circa x0,046):

| Dataset | x1/16 | x1/8 | x1/4 | x1/2 | x1 | x2 | x4 | x8 | x16 |
|---|---|---|---|---|---|---|---|---|---|
| emg2pose | 0,0506 | 0,0402 | 0,0360 | 0,0342 | **0,0340** | 0,0343 | 0,0344 | 0,0344 | 0,0345 |
| CapgMyo | 0,0740 | 0,0552 | 0,0492 | **0,0466** | 0,0478 | 0,0490 | 0,0505 | 0,0513 | 0,0519 |
| putEMG | 0,1003 | 0,0882 | 0,0845 | 0,0811 | 0,0778 | 0,0756 | 0,0749 | 0,0746 | **0,0745** |
| Camargo | 0,0829 | 0,0691 | 0,0631 | **0,0621** | 0,0653 | 0,0715 | 0,0744 | 0,0774 | 0,0791 |

Curva piatta da x0,5 a x16 su emg2pose e degrado netto sotto x0,25; il minimo cade a x0,5-x1 su tre dataset su quattro (putEMG
continua a migliorare lentamente verso le scale grandi). Il calibrazione di D5a (scala nativa contro varianza unitaria,
solo su emg2pose) ha scelto bene, ma la scelta non e' fine: basta stare nel plateau. Il valore esatto del fattore non e'
critico; **non e' critico neanche per confrontare i dataset** (il rapporto di V2 cambia di poco nel plateau).
