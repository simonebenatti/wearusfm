# Verifica scettica di RC-2 (candidato residuale: «non alza il livello raggiunto») — 22/11/2026

Cartella di lavoro: `/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/verify/rc2_work/`
Avvio 11:51. Solo CPU, un thread, nessun accesso al cluster, nessun dato vero letto. Tutto cio' che segue riguarda
il GIOCATTOLO (latenti sintetici), non i latenti veri.

## 1. Numeri del difetto ricalcolati dai JSON dell'altro revisore (DIMOSTRATO: coincidono tutti)

Comando: lettura diretta di `review/residual_candidate/toy/run_f0_{cur,res}_s*.json` e `toy/refs_f0.json`
(scelta = minimo della MSE rumorosa sugli esclusi, a parita' l'epoca minore).

| corsa | epoche fatte | epoca scelta | R alla scelta | R massimo (epoca) | R ultima | allenabili |
|---|---:|---:|---:|---:|---:|---:|
| cur seme 0 | 23 | 21 | +0,3146 | +0,3146 (21) | +0,2936 | 72.904 |
| cur seme 1 | 23 | 22 | -0,1077 | -0,1077 (22) | -0,1082 | 72.904 |
| cur seme 2 | 22 | 20 | +0,2914 | +0,2960 (22) | +0,2960 | 72.904 |
| res seme 0 | 23 | 15 | +0,3113 | +0,3178 (18) | +0,2990 | 64.436 |
| res seme 1 | 40 | 20 | +0,2899 | +0,2988 (17) | +0,0618 | 64.436 |

- res - cur seme 0: -0,0033 alla scelta; +0,0054 all'epoca 23 (+0,2990 contro +0,2936). Seme 1: +0,3976.
- cur seme 1: max |R - parte lineare| su 23 epoche = 1,11e-06 (ramo profondo costante nel tempo).
- Dispersione: cur |s0 - s2| = 0,0232; res |s0 - s1| = 0,0214. Massimo fra tutte le varianti +0,4271 (wd0.3 seme 0).
- Base: alpha 0,1; R ridge -0,0844; R WLS -0,0888.
- Nota mia: alla scelta la MSE rumorosa di cur seme 0 (1,83376) e' PIU' BASSA di quella di res seme 0 (1,84278).

## 2. Fedelta' del giocattolo al codice congelato (letto)

- `heads.py` 62-63 e 79-82: ramo affine e ultimo strato del ramo profondo inizializzati a zero; LN a pesi 1 e bias 0.
  Quindi «res» (LN e affine con requires_grad False) = uscita del solo ramo profondo + base esterna: corretto.
- `train.py` 27-37 e 233-234: AdamW, betas (0,9; 0,999), eps 1e-8, wd 0,01, batch 32, clip 1,0, LR costante
  (nessuno scheduler, nessun warmup). 276-306: permutazione per epoca senza scarto dell'ultimo lotto (920 = 28*32 + 24: 29 passi),
  `metrics.per_window_mse`, clip su tutti i parametri. Il ciclo scritto a mano in `toy_res.py` 195-206 li riproduce.
- `train.py` 221: la classe congelata accetta solo i semi 0, 1, 2 (i semi 3 e 4 del censimento dell'altro revisore sono fuori ricetta).

## 3. Rifacimento indipendente con la classe congelata `train.Trajectory` (GIOCATTOLO; script `rc2_work/rc2_run.py`)

Dati rigenerati con `toy_res.gen(0)` nella mia cartella (`rc2_work/gen_wrap.py`): `refs_f0.json` con gli stessi valori dell'altro
revisore (alpha 0,1; soglia 0,444364752; sse_oracle 2.581.253,95; sse_cleanLS 2.772.090,01; R ridge -0,084403).
Il mio corridore NON usa il ciclo scritto a mano di `toy_res.py`: costruisce `train.TensorTrainingSet` e `train.Trajectory`
(ottimizzatore, clip, mescolamento, flusso del dropout del codice vero) e riprende con `Trajectory.state()/restore()`;
la scala R e' ricalcolata da me (soglia con `metrics.training_threshold`, oracolo ricalcolato e confrontato).
Macchina molto carica (30 processi Python, carico 28-36 su 11 core): ~25 s per epoca, pezzi da 250 s.

### 3a. Determinismo (DIMOSTRATO, 11:58)
Confronto epoca per epoca fra le mie corse e i JSON dell'altro revisore:
cur seme 0 (epoche 1-3), res seme 0 (1-4), cur seme 2 (1-3), cur seme 1 (1): max |dMSE| = 0,00e+00, max |dR| <= 2,3e-10.
=> il ciclo a mano dell'altro revisore coincide bit per bit con la classe congelata; i suoi JSON sono riproducibili.

### 3b. Candidato a LR 0,01, seme 0 (DIMOSTRATO nel giocattolo, 11:58)
`res` seme 0 LR 0,01: penultimo strato 0/32 unita' vive alle epoche 1, 2, 3, 4; R = parte lineare = -0,0844 (ramo profondo morto).
Fermato dopo l'epoca 4. Il candidato NON apre la strada a un LR piu' alto: la via «residuo + LR maggiore» per alzare il livello e' chiusa (seme 0).

### 3c. Determinismo esteso e ramo morto del seme 1 con la classe congelata (DIMOSTRATO nel giocattolo, 12:00)
Epoche 0-5: cur seme 0, res seme 0, cur seme 2: max |dMSE| = 0,00e+00, max |dR| <= 2,3e-10 rispetto ai JSON dell'altro revisore.
cur seme 1 con `train.Trajectory`: penultimo strato 30/32 vive all'epoca 0, 5/32 all'epoca 1, 0/32 all'epoca 2 (sonda: 64 finestre escluse);
R = parte lineare (-0,7607; -0,5702), identico al suo JSON. Il ramo morto del seme 1 NON e' un artefatto del suo ciclo a mano.
Corse cur seme 1 e cur seme 2 fermate (repliche pure, traiettorie gia' identiche); restano res s2, res s0 (LR 0,003 e 0,001), cur s0.

## 4. Secondo giocattolo, script INDIPENDENTE gia' sul disco (revisore «training_loop», `toy_tl.py`; solo letto, 12:02)

Stesso generatore di latenti sintetici (`make_latents`, seme 12345) e stessa classe `train.Trajectory`, ma script, base (OLS, alpha 0,
caricata nel ramo affine congelato) e traiettorie diversi dai miei (MSE esclusi all'epoca 3, ricetta attuale seme 0: 2,16186 contro 2,16369):
non e' una copia delle corse di RC-2, ma NON e' un secondo campione di latenti.

Ricalcolo mio dai suoi `review/training_loop/toy_run_{base,residual}_s{0,1,2}.json` (stessa regola di scelta; orizzonti 40-60 epoche;
il suo «residual» = LN fissa + base OLS congelata sul fitting + ramo profondo sul residuo, cioe' il candidato con alpha = 0):

| seme | base (ricetta attuale): scelta | residual: scelta | residual - base alla scelta | ai massimi |
|---:|---|---|---:|---:|
| 0 | +0,3353 (ep 30 su 60) | +0,3042 (ep 15 su 60) | -0,0311 | -0,0258 |
| 1 | -0,0961 (ep 41 su 43, ramo morto) | +0,2878 (ep 17 su 46) | +0,3839 | +0,3850 |
| 2 | +0,2907 (ep 24 su 40) | +0,2676 (ep 12 su 40; max +0,3115 a 17) | -0,0231 | +0,0090 |

Altro fold della ricetta attuale: s0/f1 +0,3197 (ep 27), s1/f1 +0,3121 (ep 28).
Lettura: con orizzonti lunghi (niente troncamento) e tre semi appaiati il quadro di RC-2 si ripete: dove il ramo profondo della
ricetta attuale e' vivo il candidato sta 2-3 punti SOTTO alla scelta (dentro la dispersione dichiarata da quel revisore, 4,5 punti),
e il guadagno grande (+38 punti) compare solo nel seme 1, dove la ricetta attuale e' morta. Il troncamento a 22-23 epoche delle corse
`cur` di RC-2 (scelta a 20-21, cioe' vicino alla fine) semmai SOTTOSTIMA la ricetta attuale, non il candidato.
Verifica dopo la scrittura: `training_loop/toy_refs.json` ha gli stessi riferimenti di RC-2 (soglia 0,44436475222144456; sse_oracle
esclusi 2.581.253,95; E_cleanLS 0,8221200470847861): sono gli STESSI dati (stesso bersaglio, stesso fold); le traiettorie differiscono
solo per percorso numerico (quel revisore usa `TOY_NNPACK=0`). Conseguenza utile: stessa ricetta, stesso seme, stessi dati ->
ricetta attuale seme 0 +0,3353 (suo, 60 epoche) contro +0,3146 (RC-2, 23 epoche) e +0,311 (replica di ieri, 28 epoche, altro dropout):
il rumore di una singola corsa alla scelta e' dell'ordine di 0,02-0,03.

## 5. Corse mie, stato alle 12:04 (GIOCATTOLO, fold 0 salvo dove indicato)

- Determinismo fino all'epoca 15: cur s0 e res s0 identici ai JSON di RC-2 (max |dMSE| 0,00e+00): il numero chiave res s0 = +0,3113
  all'epoca 15 e' riprodotto con la classe congelata.
- res seme 2 (NUOVO, appaiato a cur seme 2): a 19 epoche R = +0,3349 all'epoca 19 (MSE 1,84413), +0,3009 alla 18: in corso.
- res seme 0 a LR 0,001: +0,1906 all'epoca 19 (LR 0,003: +0,2994): sale piano, in corso.
- FOLD 1 (scambiato, aggiunta mia; riferimenti: margine 7,94%, ridge alpha 0,1 a -6,95%, WLS -7,35%, 590 finestre dinamiche):
  res seme 0: penultimo strato 7/32 vive all'epoca 1, 2/32 alla 2, R = parte lineare (-0,0695): ramo profondo quasi morto ANCHE nel candidato;
  cur seme 0: 9/32 e 4/32, R = parte lineare (-0,4535). In corso.
- Nota tecnica: la soglia ricalcolata da `ystar` salvato in float32 differisce di 4e-9 relativo da quella in refs (float64): uso quella di refs
  e verifico il numero di finestre dinamiche (546 e 590).

## 6. Risultati intermedi delle 12:07 (GIOCATTOLO; `python -I rc2_work/rc2_summ.py`; la tabella finale e' piu' sotto)

- DETERMINISMO COMPLETO: cur s0 (epoche 0-23), res s0 (0-23), cur s2 (0-22) con `train.Trajectory` = JSON di RC-2, max |dMSE| 0,00e+00,
  max |dR| 2,3e-10. Tutti i numeri delle tre corse appaiabili di RC-2 sono riprodotti dalla classe congelata.
- Oltre il troncamento: cur s0 a 30 epoche -> scelta epoca 28, R +0,2986 (massimo +0,3227 all'epoca 25); cur s2 a 25 epoche -> scelta 24, +0,2911
  (massimo +0,3169 alla 23). Il valore «alla scelta» di una corsa oscilla di 0,01-0,03 con l'orizzonte.
- res s2 (NUOVO): 26 epoche, scelta epoca 19, R +0,3349 (massimo +0,3371 alla 17). res - cur seme 2: +0,0438 alla scelta, +0,0203 ai massimi.
- res s0 - cur s0: -0,0115 (orizzonte comune 26: +0,3113 contro +0,3227) oppure +0,0127 (cur a 30 epoche: +0,2986).
- res s0 a LR 0,001: 31 epoche, plateau 0,19-0,21 (massimo +0,2152 alla 25): 0,10 SOTTO il res a LR 0,003. cur s0 a LR 0,001: +0,0785 alla 19, in salita.
- Fold 1, seme 0: ENTRAMBE le ricette restano «dormienti» (R = parte lineare; penultimo strato 0-9/32 vive sulla sonda) per ~8 epoche
  (cur fino alla 9, res fino alla 6-8) e poi decollano: 0/32 sulla sonda NON significa morte irreversibile. In corso.
- Attenzione al comando di riproduzione di RC-2 (`run ... 40 900`): il valore «cur seme 0 +0,3146 (epoca 21 su 23)» dipende dal
  troncamento a 23 epoche imposto dal tetto di tempo con la macchina carica; con piu' epoche la scelta si sposta (28: +0,2986).
  I JSON sono coerenti col difetto; la conclusione non cambia.
