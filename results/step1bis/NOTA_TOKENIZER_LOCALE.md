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

## 4. Le feature continue prima della quantizzazione sono molto piu' stabili dei codici
Stesso rumore di V4 (bianco, deviazione standard = noise floor del dataset; e un controllo a 0,1x). Coseno fra la feature
(128 dimensioni, uscita della testa dell'encoder, prima del vicino piu' prossimo) del token pulito e quella dello stesso
token rumoroso; per riferimento, il coseno fra token DIVERSI. Mediana per ramo (0, 1, 2, 3), 200 finestre da 3,2 s per
dataset. Script: `scripts/step1bis_feature_stability.py`.

| Dataset | Rumore | Coseno pulito-vs-rumoroso (mediana) | 5% piu' basso | Coseno fra token diversi (mediana / 95%) |
|---|---|---|---|---|
| emg2pose | = soglia (V4) | 0,927 / 0,748 / 0,990 / 0,965 | 0,477 / 0,269 / 0,789 / 0,646 | 0,007 / 0,005 / 0,061 / 0,069 (95%: 0,23-0,35) |
| emg2pose | 0,1 x soglia | 0,999 / 0,993 / 1,000 / 1,000 | 0,976 / 0,871 / 0,997 / 0,988 | |
| CapgMyo | = soglia | 0,819 / 0,564 / 0,983 / 0,954 | 0,377 / 0,193 / 0,772 / 0,658 | 0,007 / -0,005 / 0,078 / 0,052 |
| putEMG | = soglia | 0,548 / 0,505 / 0,801 / 0,818 | 0,224 / 0,271 / 0,512 / 0,517 | 0,132 / 0,100 / 0,138 / 0,186 (95%: 0,41-0,65) |

Lettura: dove i codici cambiano nel 33-54% dei casi al livello 0, la feature continua dello stesso token resta molto piu'
vicina a se stessa (coseno mediano 0,75-0,99 su emg2pose) che a un token qualunque (0,0-0,07). Il ramo 1 (>60 Hz) e'
il meno stabile; putEMG (il dataset con il rapporto V2 piu' alto) e' il meno stabile in assoluto. **Non misurato:** se
queste feature continue identifichino il dataset piu' delle 5 bande (l'equivalente di V3 sulle feature): dallo spettro
fine si sa che l'identita' del dataset e' gia' nello spettro (0,93).


## 5. Feature continue su tutti e 6 i dataset (30/09/2026, `scripts/step1bis_continuous_features.py`, `continuous_features.json`)

Stessi array del run, copiati da Leonardo (sha256 verificati). Controlli: codici ricalcolati su CPU uguali a quelli del run GPU al 99,4-99,7% (tutti i
livelli, 16 finestre per dataset); V3 congelato riprodotto esattamente; soglia di rumore uguale a quella del run su tutti i dataset.

**(a) Dataset-ID dalle feature continue** (stesso campione, split e sonda di V3, 6 classi, caso 0,167):

| Rappresentazione | Accuratezza bilanciata |
|---|---|
| [congelata] 5 bande | 0,655 |
| [congelata] codici | 0,964 |
| feature continue, media per ramo (512 numeri) | **0,988** |
| feature continue, media + deviazione standard | **1,000** |
| solo ramo 0 / 1 / 2 / 3 (media) | 0,655 / 0,988 / 0,786 / 0,929 |

Le feature continue identificano il dataset **ancora piu' dei codici**: come bersaglio non evitano il problema di V3. Il ramo 0 e' l'unico al livello
delle 5 bande, come nei codici (sonda per ramo, `rvq_branch_probe.json`).

**(b) Stabilita' delle feature sotto il rumore di V4** (coseno fra la feature del token pulito e dello stesso token rumoroso, mediana per ramo 0/1/2/3,
200 finestre per dataset; per confronto il coseno fra token diversi e' 0,00-0,20):

| Dataset | Rumore = soglia | 0,1 x soglia |
|---|---|---|
| emg2pose | 0,93 / 0,75 / 0,99 / 0,96 | 1,00 / 0,99 / 1,00 / 1,00 |
| CapgMyo | 0,82 / 0,56 / 0,98 / 0,95 | 1,00 / 0,94 / 1,00 / 1,00 |
| Camargo | 0,86 / 0,57 / 0,97 / 0,94 | 1,00 / 0,94 / 1,00 / 1,00 |
| CSL-hdemg | 0,92 / 0,79 / 0,99 / 0,98 | 1,00 / 1,00 / 1,00 / 1,00 |
| GRABMyo | 0,67 / 0,48 / 0,91 / 0,88 | 0,99 / 0,94 / 1,00 / 1,00 |
| putEMG | 0,55 / 0,51 / 0,80 / 0,82 | 0,98 / 0,95 / 1,00 / 1,00 |

Le feature restano molto piu' vicine a se stesse che a un token qualunque anche dove i codici cambiano; il ramo 1 e' il meno stabile, GRABMyo e putEMG i
dataset meno stabili. **Analisi descrittiva, non congelata: non cambia nessun verdetto.**
