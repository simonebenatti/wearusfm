# Verifica scettica di RC-1 (lente residual_candidate) — 22/11/2026

Cartella di lavoro: /Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/verify/rc1_work
Avvio 11:52. Sola lettura su tree_e8c98244; nessun cluster, nessun dato vero (latenti/checkpoint non sono sul Mac).

Difetto dichiarato: «la base congelata parte sotto zero e il ramo temporale dovrebbe quasi raddoppiare il guadagno netto attuale»
(69,10 / 66,38 / 67,84 punti; 1,597 / 1,883 / 1,715 volte).

Piano della confutazione:
1. ricalcolo indipendente dei numeri dai JSON (script mio, non quello dell'altro revisore);
2. lettura del codice: che cosa misura davvero `linear_diagnostics.*.L0.worlds.temporal.held` (y rumoroso? alpha=0? stessi denominatori del cancello?);
3. la base del candidato di GPT e' alpha=0 o ridge con alpha da CV? il numero di testa usa la base giusta?
4. «la scelta di alpha per CV non aiuta»: la prova citata lo sostiene?
5. il rapporto 1,6-1,9 e' specifico del candidato o vale per qualunque ricetta?

## 1. Ricalcolo indipendente dai JSON (11:55) — DIMOSTRATO
Script: rc1_work/rc1_recompute.py (uscita rc1_work/rc1_recompute.out). Comando:
`cd /Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/verify/rc1_work && /Users/simonebenatti/.venvs/wearusfm-tok/bin/python -I rc1_recompute.py`

- Stesso pannello: in calibration_{0,1}.json `outer_reference_statistics.temporal.{clean_ls,oracle}` e
  `linear_diagnostics.L0.worlds.temporal.held.statistics` hanno lo stesso `dynamic_target_ss`
  (3206279,956759538 fold0; 2698827,390194636 fold1) e le stesse finestre (934; 920). Asserzioni superate.
- Codice: controls.py:620 chiama `factor.predict_mixture(indices, {"L":1,"T":a,"epsilon":sigma})` senza alpha;
  linear.py:374 `predict_mixture(self, indices, weights, alpha=0.0)`: e' la WLS (alpha=0) su y osservato, valutata su y (controls.py:621).
  Il riferimento clean_ls usa lo stesso fattore con {"L":1,"T":a} e alpha 0 (controls.py:573). La lettura dell'altro revisore e' corretta.
- Numeri del reclamo, tutti riprodotti (fold0 / fold1 / pooled):
  WLS su y, esclusi esterni: -19,1005% / -16,3797% / -17,8356%; L0 neurale -13,5531 / -10,8188 / -12,2819; MLP -12,2400 / -10,6539 / -11,5026;
  TCN +24,1731 / +18,8755 / +21,7101; TCN meno WLS: 43,2736 / 35,2551 / 39,5457 punti; servono dalla WLS: 69,1005 / 66,3797 / 67,8356;
  rapporto 1,5968 / 1,8828 / 1,7154. Calibrazione interna: alpha 1 in entrambi i fold, ridge a -19,0020% / -16,1029% del margine interno.
  => L'ARITMETICA NON E' CONFUTABILE.

## 2. Dove il reclamo e' impreciso (11:56)
(a) I numeri di testa (69,10 / 66,38 / 67,84 e 1,597 / 1,883 / 1,715) valgono per una base WLS con alpha=0. Il candidato di GPT
    (diagnosi, punto 3) e' «base affine/ridge ... con alpha selezionato nella CV interna». Quella base nel mondo temporale a 20 utenti NON e' archiviata
    (controls.py:576-578: la ridge si calcola solo se world == "linear").
(b) «La scelta di alpha per CV non aiuta» e' CONTRADDETTO dai dati archiviati a 20 utenti (mondo lineare, esclusi esterni, stesso fattore e stesso rumore eps):
    eccesso di E_dyn sull'oracolo: WLS alpha=0 0,012005 (fold0) e 0,010923 (fold1); ridge alpha=1 0,007118 e 0,006385: la ridge toglie il 40,7% / 41,5% del costo di stima
    (rapporti 0,5929 / 0,5846; in MSE 0,6015 / 0,6166). La prova citata dal reclamo (-19,00 / -16,10 a 10 utenti) confronta una ridge a META' dei dati con una WLS a 20 utenti:
    non misura l'effetto di alpha. A 10 utenti la curva interna dice anzi che alpha conta: MSE(0,01) - MSE(1) = 0,037676 (fold0) e 0,035592 (fold1),
    cioe' 33,4% / 29,3% del margine MSE esterno clean_ls-oracolo (0,112894 / 0,121355).
    Enunciato corretto: «anche con alpha da CV la base resta sotto zero», non «alpha non aiuta».
(c) Coerenza del trasferimento lineare -> temporale (la ridge a alpha fisso e' lineare nel secondo membro; stesso eps nei due mondi):
    eccesso della WLS su clean_ls nel mondo temporale 0,010884 / 0,009634 (osservato) contro 0,010995 / 0,009316 previsto dal mondo lineare con sigma^2 e denominatori (scarto 1,0% / 3,3%).
    STIMA (non dimostrata) della base ridge alpha=1 a 20 utenti nel mondo temporale: -11,32% (fold0), -9,57% (fold1); l'altro revisore stimava 12,5 / 10,9.
    Con quella base: servono 61,32 / 59,57 punti (non 69,10 / 66,38); la TCN oggi sta 35,50 / 28,45 punti sopra; rapporto 1,728 / 2,094.
(d) Il rapporto non e' una proprieta' del candidato: e' lo scarto dal cancello (25,83 / 31,12 / 28,29 punti) riscritto rispetto a una base b:
    (50 - b)/(R_TCN - b). Pooled: b=-25% -> 1,606; b=-17,84% -> 1,715; b=-12,28% (L0 neurale) -> 1,832; b=0 -> 2,303.
    Piu' la base e' BUONA, piu' il rapporto CRESCE. Vale identico per «allenare piu' a lungo» o per qualunque altra ricetta.
(e) «Parte sotto zero» vale per ogni ricetta: lo zero della scala e' clean_ls, un lettore stimato sul bersaglio pulito che nessun candidato puo' avere.
    Anche la parte lineare della ricetta attuale sta sotto zero (L0 neurale -13,55 / -10,82 / -12,28).
(f) «Non aggiunge informazione» e' gia' scritto da GPT nella diagnosi (punto 3: «Sottrarre la base non elimina epsilon e non aggiunge informazione:
    cambia l'ottimizzazione e la regolarizzazione»): non e' un difetto nascosto della proposta.

## 3. Comando dell'altro revisore (11:59) — RIPRODUCIBILE
`cd .../review/residual_candidate && python -I recompute_json.py`: uscita identica byte per byte a recompute_json.out (diff vuoto; copia in rc1_work/other_reviewer_repro.out).

## 4. La base congelata e' uno svantaggio rispetto alla ricetta attuale? (11:57) — NO, per quanto si puo' misurare
Script rc1_work/rc1_curves.py (JSON archiviati, pooled, scala del cancello):
- L0 neurale: scelta LR 3e-4, esposizione 183: -12,28%; massimo di tutte le curve -12,18% (LR 3e-4, esp. 198).
  Alla ricetta della TCN scelta (LR 0,003) L0 sta a -15,29% all'esposizione 31 e non supera mai -14,28% (esp. 111).
- TCN: scelta LR 0,003, esp. 31: +21,71%; massimo di tutte le curve +21,88% (esp. 27).
- Base del candidato (ridge alpha=1, 20 utenti), STIMA: -11,32% / -9,57%, pooled -10,51% (0,006077 di E_dyn su margine 0,057820; rc1_pooled_est.py).
=> la base congelata del candidato non sta sotto la parte lineare che la ricetta attuale riesce ad avere (-12,3% al meglio; -15,3% al punto scelto dalla TCN).
   La parte lineare della TCN attuale non e' archiviata separatamente (serve il diagnostico D0 sui checkpoint): IPOTESI che somigli a L0 alla stessa ricetta.
   Con questa ipotesi lo scambio di base vale da circa +2 a +5 punti (al massimo +7,3 se la parte lineare attuale fosse cattiva quanto la WLS), su 28,29 mancanti nel pooled:
   il resto (almeno 21 punti) deve venire dal ramo profondo. La conclusione del reclamo regge, la cornice «parte sotto zero» no: non e' un handicap specifico.
- Nessun tetto aritmetico: con base a b il massimo praticabile e' circa 100% + b (errore di stima della base non correggibile dal fitting): ~89% con la ridge stimata, ~82% con la WLS. Il 50% resta raggiungibile in linea di principio.

## 5. Giocattolo di controllo del metodo di stima (11:58) — INDICATIVO (latenti SINTETICI, codice vero)
Script rc1_work/rc1_transfer_toy.py <seme> <p> (D=128, 40 utenti x 16 finestre, spettro a legge di potenza k^-p; generate_components, synthetic_target, WindowedBasisFit veri). 7-14 s per corsa.
Domanda: la stima «ridge temporale = eccesso WLS temporale x (ridge/WLS nel mondo lineare)» predice la ridge calcolata direttamente nel mondo temporale?
| seme, p | R WLS0 | alpha 0,1: diretto / stimato | alpha 1: diretto / stimato |
|---|---:|---:|---:|
| 1, 1,0 | -15,56% | -11,22 / -10,99 | -12,76 / -11,88 |
| 2, 1,0 | -11,91% | -8,52 / -8,62 | -13,49 / -12,48 |
| 3, 1,5 | -15,73% | -4,64 / -4,92 | -5,66 / -5,76 |
| 4, 1,5 | -12,20% | -2,97 / -3,50 | -3,89 / -4,58 |
| 5, 0,7 | -10,59% | -10,10 / -10,05 | -13,07 / -11,84 |
Vicino all'alpha migliore (rapporto ridge/WLS0 < 1) lo scarto stima-diretto sta fra -0,69 e +0,88 punti; oltre l'ottimo (alpha 10, regime di distorsione) la stima e' ottimista di 0,4-10,4 punti.
Nei dati veri alpha=1 e' la scelta della CV e il rapporto e' 0,59 (regime di varianza): errore atteso della stima dell'ordine di un punto.
(Ad alpha 1 con rapporto > 1, semi 2 e 5: stima ottimista di 1,02 e 1,23 punti.)
In tutte e 5 le configurazioni la ridge al miglior alpha provato resta sotto zero (da -2,97% a -11,22%) ma sopra la WLS di 0,49-11,09 punti (4,34; 3,39; 11,09; 9,23; 0,49): conferma «sotto zero» e smentisce «alpha non aiuta».
Comando: `cd rc1_work && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<tree>/scripts:<tree>/src /Users/simonebenatti/.venvs/wearusfm-tok/bin/python rc1_transfer_toy.py 1 1.0`

## 6. Controllo del trasferimento sui numeri VERI (12:01) — DIMOSTRATO dai JSON per L0 neurale, STIMA per la ridge
Script rc1_work/rc1_transfer_real.py e rc1_work/rc1_pooled_est.py.
- L0 neurale e' archiviato in entrambi i mondi: eccesso(L0)/eccesso(WLS0) = 0,6763 (lineare) e 0,7096 (temporale) nel fold0 (+4,9%); 0,6655 e 0,6605 nel fold1 (-0,7%).
  Il rapporto si trasferisce fra i mondi entro il 5%: la stima della base ridge e' -11,3..-11,9% (fold0), -9,5..-9,6% (fold1), pooled -10,51%.
- Con la base del candidato come GPT la specifica (ridge, alpha da CV): servono circa 61,3 / 59,6 / 60,5 punti; la TCN oggi sta 35,5 / 28,5 / 32,2 punti sopra; rapporto 1,73 / 2,09 / 1,88.
- Scambio di base nel pooled: +1,77 punti (se la parte lineare attuale vale L0 scelto, -12,28%), +4,78 (se vale L0 a LR 0,003 esp. 31, -15,29%), +7,32 (se vale la WLS, -17,84%), su 28,29 mancanti.

## 7. Verdetto dello scettico (12:01)
NON CONFUTATO nell'aritmetica: tutti i numeri del reclamo si riproducono (comando dell'altro revisore e ricalcolo mio indipendente), sullo stesso pannello del cancello.
Regge anche la conclusione: per passare, il ramo profondo dovrebbe dare 1,6-2,1 volte il guadagno netto attuale; lo scambio di base da solo vale al piu' ~2-7 punti su 28.
MA la gravita' «importante» non e' sostenuta, per quattro motivi (1 e 3 dimostrati dai JSON; 2 e 4 dimostrati nella parte archiviata, stimati per la base ridge):
1. non e' un difetto specifico del candidato: e' lo scarto dal cancello (25,83 / 31,12 / 28,29 punti) riscritto; vale uguale per ogni ricetta, compreso «allenare piu' a lungo»;
   il rapporto dipende dalla base scelta (1,61 con base -25%; 2,30 con base 0) e CRESCE se la base migliora;
2. i numeri di testa (69,10 / 66,38 / 67,84; 1,597 / 1,883 / 1,715) sono della WLS alpha=0, non della base del candidato (ridge con alpha da CV; stima -11,3 / -9,6 / -10,5%);
3. «la scelta di alpha per CV non aiuta» e' contraddetto dall'archivio (20 utenti, mondo lineare: la ridge alpha=1 toglie il 40,7% / 41,5% del costo di stima della WLS);
4. «parte sotto zero» non e' un handicap: anche la parte lineare attuale sta sotto zero (-12,28% al meglio, -15,29% alla ricetta scelta dalla TCN) e la base stimata del candidato sta poco SOPRA;
   «non aggiunge informazione» e' gia' scritto da GPT nella diagnosi.
Gravita' confermata: MINORE (quantificazione corretta e utile al verdetto sul candidato, da riscrivere con i numeri e le parole giusti; nessun difetto di codice o di protocollo; nessun effetto sui passi 1 e 2).
Limiti miei: la ridge a 20 utenti nel mondo temporale e la parte lineare della TCN attuale non sono archiviate (stime); nessun dato vero letto; giocattolo con latenti sintetici.
Incidente di lavoro: un mio file di servizio chiamato struct.py oscurava il modulo standard `struct` (primo lancio del giocattolo fallito); rinominato jstruct_dump.py, nessun effetto sui risultati.
Fine 12:03 (inizio 11:51; note corrette alle 12:02 dopo rilettura dei numeri del giocattolo).
