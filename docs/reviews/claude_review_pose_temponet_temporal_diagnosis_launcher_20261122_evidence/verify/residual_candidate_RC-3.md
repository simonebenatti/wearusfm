# Verifica scettica di RC-3 (lente residual_candidate) — 22/11/2026

Avvio 11:51. Cartella di lavoro: /Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/verify/rc3_work
Materiale in sola lettura: tree_e8c98244. Nessun accesso al cluster; nessun dato vero letto (solo JSON archiviati).

Difetto dichiarato: «L'errore di stima della base resta congelato e invisibile al ramo temporale» (importante, dimostrato).

## 1. Lettura del codice citato (11:53-12:00)

- `linear.py:46-52, 321-328`: pesi WLS = valid/q_i (q_i = elementi supportati della finestra); ridge = somma delle perdite per finestra + alpha*||pendenze||^2, intercetta non penalizzata.
- `metrics.py:24-39` (`per_window_mse`): media sulle finestre di SSE_i/q_i: STESSI pesi della WLS. Quindi l'identita' X'W r = 0 vale proprio nella metrica della loss della TCN (se la base del candidato usa `WindowedBasisFit`).
- `controls.py:620`: le statistiche `linear_diagnostics/L0/worlds/temporal/held` sono `predict_mixture(..., {"L":1,"T":a,"epsilon":sigma})` con alpha di default = 0: i numeri -19,10/-16,38% sono la WLS con alpha = 0 su y rumoroso, 20 utenti.
- `controls.py:576-578`: la ridge con l'alpha scelto dalla CV interna e' archiviata SOLO per il mondo lineare (`outer_reference_statistics/linear/ridge`), non per il temporale.
- Candidato di GPT (diagnosi, punto 3): «base affine/ridge stimata sul solo fitting con alpha selezionato nella CV interna; base congelata». La base del candidato NON e' la WLS con alpha = 0.

## 2. Ricalcolo dai JSON archiviati (DIMOSTRATO; `rc3_work/rc3_json.py` -> `rc3_work/rc3_json.out`)

Numeri del claim riprodotti: forma chiusa alpha=0 sugli esclusi -19,1005 / -16,3797% (pooled -17,8356%); L0 neurale -13,5531 / -10,8188% (pooled -12,2819%); differenza 5,547 / 5,561 punti (pooled 5,554).

Ma il confronto del claim e' con alpha = 0, non con la base del candidato (ridge, alpha da CV interna = 1,0 in entrambi i fold e mondi):
| mondo lineare, 20 utenti, esclusi esterni (latenti veri) | fold0 | fold1 | pooled |
|---|---:|---:|---:|
| costo E_dyn della WLS alpha=0 sopra clean_ls | 0,012005 | 0,010923 | 0,011510 |
| costo E_dyn della ridge alpha=1 sopra clean_ls | 0,007118 | 0,006385 | 0,006783 |
| rapporto ridge/WLS (E_dyn) | 0,5929 | 0,5846 | 0,5893 |
| rapporto ridge/WLS (MSE per finestra) | 0,6015 | 0,6166 | |
La ridge con l'alpha della CV toglie il 41% del costo di stima della base a 20 utenti sui latenti veri.

STIMA (non misura: la ridge sul mondo temporale a 20 utenti non e' archiviata) per la base del candidato sulla scala del cancello:
R_ridge(alpha=1) ~ R_wls0 * rapporto = -11,32% (fold0), -9,57% (fold1), -10,51% (pooled); con il rapporto MSE: -11,49 / -10,10%.
Controllo incrociato della scala fra mondi con le ridge alpha=1 a 10 utenti (misurate nella CV interna): costo temporale/lineare = 0,010725/0,012343 = 0,869 e 0,009475/0,010241 = 0,925,
contro 0,907 e 0,882 usati sopra: stime alternative -10,86% (fold0) e -10,04% (fold1). Intervallo: fold0 fra -10,9 e -11,5; fold1 fra -9,6 e -10,1.
Confronto: L0 neurale selezionata -13,55 / -10,82%. La base del candidato e' stimata MIGLIORE di L0 neurale di 2,0-2,7 punti (fold0) e 0,7-1,2 punti (fold1), non peggiore di 5,5.

Curve L0 neurale (selection_temporal_L0.json, pooled): LR 3e-4 (scelto): minimo MSE a esposizione 183, R -0,1228; a 300 R -0,1258 (0,30 punti persi dal minimo alla fine).
LR 1e-3: minimo a 80, R -0,1299; a 150 -0,1355. LR 3e-3: 31 -> -0,1529; 150 -> -0,1483 (minimo MSE a 140). LR 1e-4: ancora in discesa a 300 (-0,1310).
Lettura: l'arresto anticipato in senso stretto vale <= 0,56 punti; il vantaggio di 5,55 punti sulla WLS alpha=0 viene dalla regolarizzazione implicita dell'ottimizzazione su orizzonte finito
(nessuna curva L0 arriva a -0,178 nemmeno a fine orizzonte), non dalla scelta dell'epoca.
Alla ricetta scelta per la TCN (LR 0,003, esposizione 31) L0 da sola sta a -15,29%: se il ramo affine della TCN si comporta come L0 (IPOTESI, non identificabile senza i checkpoint),
la base ridge del candidato sarebbe ~4,8 punti MEGLIO della parte lineare attuale.

## 3. Misura diretta sui latenti veri, mondo LINEARE (DIMOSTRATO dai JSON; `rc3_work/rc3_json2.py` -> `rc3_json2.out`)

Stessi esclusi esterni, 20 utenti di fitting, costo sopra l'oracolo in % del margine temporale (fold0 / fold1 / pooled):
| lettore | fold0 | fold1 | pooled |
|---|---:|---:|---:|
| ridge in forma chiusa, alpha da CV interna (= tipo di base del candidato) | 12,49 | 10,86 | 11,73 |
| L0 neurale selezionata | 14,25 | 12,36 | 13,37 |
| WLS alpha = 0 (quella citata dal claim) | 21,07 | 18,57 | 19,91 |
| TCN_post neurale (ricetta attuale intera) | 19,74 | 18,70 | 19,25 |
- La ridge con alpha da CV e' MIGLIORE di L0 neurale di 1,76 / 1,50 punti (pooled 1,64). Coincide con il recupero archiviato dal cancello:
  `linear_head_gates/L0/recovery` E_dyn 0,99630 / 0,99615 / 0,99624 (< 1 = L0 peggiore della ridge), MSE 0,99712 / 0,99726 / 0,99719.
- Vale per OGNI checkpoint di OGNI curva L0 del mondo lineare: minimo E_dyn di tutte le curve 0,748834 (LR 3e-4, esp. 186) contro ridge 0,748129; minimo MSE 1,920843 contro ridge 1,918009.
- La L0 lineare selezionata e' LR 1e-4 all'esposizione 299 su 300: li' non c'e' alcun arresto anticipato.
- L'ordine WLS0 > L0 neurale > ridge del mondo lineare (21,1 > 14,3 > 12,5) e' lo stesso atteso nel temporale (19,1 > 13,6 > ~11,3 stimato).

Trasferimento del beneficio della ridge al mondo temporale, latenti veri, CV interna a 10 utenti (`rc3_json3.py`): MSE interna da alpha 0,01 a 1:
temporale -0,037676 (fold0) e -0,035592 (fold1); mondo lineare con lo STESSO sigma 1,3 (punto di griglia a=0): -0,034408 e -0,034236.
Nel temporale la ridge guadagna quanto (anzi 1,095 e 1,040 volte) nel lineare: la stima della sezione 2 non e' ottimistica.

## 4. Giocattolo indipendente (INDICATIVO: latenti sintetici miei, seme 777, 40 utenti x 30 finestre; codice vero) — `rc3_work/rc3_toy.py` -> `rc3_toy.out`

T1, identita' nella metrica della loss vera (`metrics.per_window_mse`, lettore affine su LN fissa, bersaglio = residuo di fitting, gradiente sull'intero fitting):
- supporto pieno, base WLS alpha=0: ||grad|| 6,324e-15 (2,314e-15 del gradiente col bersaglio y) -> identita' CONFERMATA;
- maschere parziali (2 gruppi di giunti, finestre accorciate), WLS: 6,544e-15 (2,348e-15) -> vale anche con le maschere;
- base ridge: relativo 6,087e-04 (alpha 0,1), 4,280e-03 (alpha 1), 2,286e-02 (alpha 10); con maschere e alpha 1: 7,934e-03. «Quasi esatta» confermato.
T2, validazione dello stimatore usato nella sezione 2 (costo ridge temporale = costo WLS temporale x rapporto ridge/WLS del mondo lineare):
- alpha CV = 1 in entrambi i mondi; alpha 0,1: costo vero 0,010143, stimato 0,010278; alpha 1: vero 0,009836, stimato 0,009725 (R vero -0,1751, stimato -0,1732);
  alpha 10 e 100 (oltre l'ottimo): lo stimatore sottostima il costo del 9-10%. All'alpha scelto dalla CV l'errore e' dell'1%.
T3, base ridge CV contro L0 neurale (ricetta vera, AdamW wd 0,01, batch 32, clip 1,0; selezione = minimo MSE sugli esclusi), mondo temporale del giocattolo:
- WLS alpha=0: R -0,2131; ridge alpha CV (1,0): R -0,1751;
- L0 neurale LR 0,003: scelta epoca 32, R -0,2098 (ultima 150: -0,2167); LR 0,001: scelta 65, R -0,1742 (ultima -0,1994); LR 0,0003: scelta 204, R -0,1655 (ultima 300: -0,1725).
- Nel giocattolo la migliore L0 neurale e' 0,96 punti MEGLIO della ridge CV, al LR della TCN (0,003) e' 3,47 punti PEGGIO. Sui latenti veri (sezione 3, misura) la ridge e' 1,5-1,8 punti meglio.
  In nessun caso compare lo scarto di 5,5 punti: quello esiste solo contro la WLS con alpha = 0.
Quota lineare (R2) del residuo di fitting, stessa metrica del revisore (`rc3_r2.py`): WLS 0,000e+00; ridge alpha 0,1: 1,665e-04; ridge alpha 1: 5,097e-03.
(La prova del revisore `ortho.py` non e' rieseguibile tal quale: il suo `toy/data_f0.npz` e' stato cancellato e rigenerarlo scriverebbe nella sua cartella; l'ho riprodotta in modo indipendente.)

Prova diretta del nucleo del claim (`rc3_deep.py 230` -> `rc3_deep.out`): candidato di GPT nel giocattolo (LN fissa, base ridge alpha 1 congelata, ramo profondo sul residuo, LR 0,003, 25 epoche).
Sugli esclusi, proiezione dell'uscita del ramo profondo sull'errore della base e_b = base - clean_ls: c_b fra -0,0124 e +0,0359 in tutte le 25 epoche (-1 = errore corretto del tutto).
Alla scelta (epoca 11): c_b +0,0127, quota del resto non lineare imparata c_T 0,5701, R base+ramo +0,3780 (base da sola -0,1751).
Quindi: il ramo profondo NON corregge l'errore della base (nucleo del claim CONFERMATO); semmai, con base ridge, lo amplifica dell'1-4% reimparando parte di cio' che la ridge aveva contratto.

## 5. Verdetto dello scettico (12:08)

- CONFERMATO (non confutabile): l'identita' algebrica e la sua conseguenza diretta. Il residuo di fitting non ha contenuto lineare in z nella metrica della loss vera (anche con maschere);
  il ramo profondo allenato sul residuo non riduce l'errore di stima della base (prova diretta: |c_b| <= 0,036).
- CONFUTATO con numeri archiviati: che questo sia un handicap del candidato rispetto alla ricetta attuale e che «costi -19,1 / -16,4%».
  (1) Quei numeri sono la WLS con alpha = 0 (`controls.py:620`); il candidato prescrive la ridge con alpha da CV interna (alpha = 1,0 nei due fold).
  (2) Misura sui latenti veri, mondo lineare, stessi esclusi: ridge CV 12,49 / 10,86% del margine temporale sopra l'oracolo, L0 neurale 14,25 / 12,36%, WLS0 21,07 / 18,57%.
      La base di tipo candidato e' MIGLIORE di L0 neurale di 1,76 / 1,50 punti e di ogni checkpoint di ogni curva L0.
  (3) Stima per il mondo temporale (non archiviata): ridge CV -11,3 / -9,6% (pooled -10,5%) contro L0 neurale -13,55 / -10,82% (pooled -12,28%).
  (4) «Regolarizzata dall'arresto anticipato»: sulle curve L0 l'arresto vale <= 0,56 punti; nel mondo lineare la L0 scelta e' all'esposizione 299 su 300.
- Che cosa resta: una NOTA di disegno. Con base ridge CV il ramo temporale deve dare circa 60-61 punti (50 + 10-11), non 66-69; nel protocollo del candidato va scritto che la base e' la ridge con alpha da CV (non alpha = 0)
  e va riportato il suo R da sola. Non spiega il cancello fallito (lo dice anche il claim).
- Gravita' confermata: nota (non «importante»). IPOTESI non verificabile senza i checkpoint: che il ramo affine della TCN attuale valga quanto L0 al LR 0,003/esposizione 31 (-15,3%).

## 6. Comandi per riprodurre
Prefisso: `cd /Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/verify/rc3_work; T=/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$T/scripts:$T/src; PY=/Users/simonebenatti/.venvs/wearusfm-tok/bin/python`
- JSON archiviati: `$PY -I rc3_json.py` ; `$PY -I rc3_json2.py` ; `$PY -I rc3_json3.py`
- Giocattolo: `$PY rc3_toy.py t3` (T1+T2 in 101 s, T3 in altri ~250 s) ; `$PY rc3_r2.py` (30 s) ; `$PY rc3_deep.py 230` (~260 s). Nessun file grande scritto.
Fine 12:08 (17 minuti).
