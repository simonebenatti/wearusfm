# Note del revisore "numbers_repr" — 22/11/2026

Compito: verifica indipendente dei numeri della diagnosi GPT (commit e8c98244) e della prova di rappresentabilita'.
Solo lettura su tree_e8c98244; nessun accesso a cluster o rete; i latenti e i checkpoint veri NON sono sul Mac.
Inizio: 11:23 CET.

## 1. Numeri ricalcolati dai JSON grezzi (script `recompute.py`, uscita `recompute_out.txt`)

Fonti grezze: `launcher/calibration_{0,1}.json` -> `outer_reference_statistics.temporal.{clean_ls,oracle}` (accumulatori);
`launcher/selection_temporal_{TCN_post,MLP,L0}.json` -> `selected_outer_statistics`, `selected_fitting_diagnostics`,
`selected_outer_clean_statistics` (accumulatori); `curves[*].points` (qui solo rapporti `edyn`/`mse` a doppia precisione, NON accumulatori).
Pooled sempre ricalcolato da me sommando i due fold prima del rapporto.

| Grandezza | Diagnosi GPT | Mio ricalcolo | Esito |
|---|---|---|---|
| recupero fold0 / fold1 / pooled | 24,17 / 18,88 / 21,71 % | 24,173053 / 18,875450 / 21,710110 % | confermato |
| pooled clean_ls | 0,728479 | 0,728479029 | confermato |
| pooled oracolo | 0,670659 | 0,670658924 | confermato |
| soglia 50% pooled | 0,699569 | 0,699568976 | confermato |
| TCN scelta, pooled E_dyn | 0,715926 | 0,715926220 | confermato |
| minimo E_dyn di tutte le curve TCN temporali | 0,715826 | 0,715826246 a LR 0,003, esposizione 27 (passi 783/810) | confermato |
| recupero retrospettivo a quel minimo | 21,88 % | 21,883016 % | confermato |
| MLP scelta, pooled E_dyn | 0,735130 | 0,735129837 (LR 0,001, esposizione 24) | confermato |
| "TCN migliora del 2,61% rispetto alla MLP" | 2,61 % | (MLP-TCN)/MLP = 2,612275 %; (MLP-TCN)/TCN = 2,682346 % | confermato come riduzione relativa alla MLP |
| fitting pooled al checkpoint scelto | E_dyn 0,685897 / MSE 1,698865 | 0,685897004 / 1,698864766 (n = 920+934 = 1854) | confermato |
| esclusi pooled | 0,715926 / 1,800145 | 0,715926220 / 1,800144821 (n = 934+920 = 1854) | confermato |
| bersaglio pulito, esclusi, TCN | 0,139236 | 0,139235991 | confermato |
| bersaglio pulito, esclusi, MLP | 0,196055 | 0,196054702 | confermato |
| scelta | LR 0,003 / esposizione 31 | LR 0,003, esposizione 31 | confermato |
| orizzonti | 1e-4 fino a 300; 3e-4, 1e-3, 3e-3, 1e-2 fino a 150 | identico; 304 / 154 punti per curva, ultimo punto = orizzonte, passi finali 8700/9000 e 4350/4500 | confermato |
| a, sigma | 0,4 / 1,3 nei due fold | a=0,4 sigma=1,3 alpha=1 in entrambi (gain interno 0,078597 / 0,078238) | confermato |
| optimizer_alarms TCN temporale | nessuno | `[]`, `optimization_unresolved` false | confermato |

Stessi pannelli (denominatori): `dynamic_target_ss` di clean_ls, oracolo e TCN coincidono: fold0 3206279.9567595376 identico bit per bit;
fold1 2698827.390194636 (calibrazione CPU) contro 2698827.3901946363 (job GPU): 1 ulp. Finestre 934 / 920 / 1854 uguali.

Minimi per curva del controllo temporale TCN (MSE sugli esclusi; E_dyn; valori finali):
- LR 1e-4 (300): min MSE 1,818162 a 181; min E_dyn 0,725580 a 201; finale E_dyn 0,728150 MSE 1,826860
- LR 3e-4 (150): min MSE 1,817431 a 64; min E_dyn 0,725382 a 64; finale 0,731910 / 1,837210
- LR 1e-3 (150): min MSE 1,813964 a 31; min E_dyn 0,722228 a 28; finale 0,751496 / 1,886092
- LR 3e-3 (150): min MSE 1,800145 a 31; min E_dyn 0,715826 a 27; finale 0,741556 / 1,852148
- LR 1e-2 (150): min MSE 1,849272 a 93; min E_dyn 0,737066 a 111; finale 0,737611 / 1,852292
In tutte e cinque le curve il valore finale e' peggiore del minimo: con questa ricetta allungare il training non avvicina la soglia 0,699569.

## 1b. Controllo lineare, ricevute, durate (script `recompute2.py`, uscita `recompute2_out.txt`)

- Riferimento lineare (ridge contro oracolo, soglia 90%): MSE 97,464314 / 97,763827 / 97,604561 %; E_dyn 97,435216 / 97,297566 / 97,377700 % (fold0/fold1/pooled). Confermato, identico al gate JSON.
- Teste nel controllo lineare, 18 recuperi (3 teste x 3 pannelli x 2 metriche) ricalcolati dagli accumulatori, differenza 0 dal gate JSON:
  minimo 97,461447 % (MLP, fold1, E_dyn), massimo 99,726055 % (L0, fold1, MSE). Tutti >= 90%: i tre lineari passano. L0 lineare ha l'allarme `minimum_in_last_five_maximum_epochs` (scelto LR 1e-4, esposizione 299 su 300), status comunque `passed`.
- Sei controlli `COMPLETED|0:0` (file `*.sacct.tsv` e `sacct_through_controls_20261121.tsv`): 65035622 1451 s, 65035623 1287 s, 65035626 3710 s, 65035628 1105 s, 65035629 860 s, 65035631 4427 s. Confermato.
- 4427 s / 3600 = 1,229722 GPUh (1 GPU): "1,23 GPUh" confermato. E' il tempo dell'allocazione intera (5 curve, 900 epoche-fold... caricamenti e valutazioni compresi), non del solo training; `elapsed_seconds` della ricevuta 4383,3 s. Somma dei sei: 3,566667 GPUh.
- Sei ricevute: `source_sha256` 9aaf1c8b6a050a524dfb60af65df9406313d4fd0e690dab1097bf11df2ec67ea in tutte; `numeric_sha256` 12cb2a0b3207b66e260d3ca490db93ee0c029bb8f03d766df89a9edba6b7eb3f in tutte = pin `gpu8` di `plan_controls.json`; hardware kind gpu, A100-SXM-64GB, capability [8,0], 68.099.571.712 byte, Xeon Platinum 8358, 8 thread; `code_sha` edebfb6c290c (checkout), `reviewed_code_sha` a1ae9607681c. Sei nodi diversi (lrdn2183, 3188, 3369, 2287, 3349, 0707). Confermato.
- Gate: status `recipe_not_admitted`, `temporal_head_gate.optimization_alarms` vuoto. Confermato.

## 1c. Grandezze derivate da me (NON nella diagnosi) — `recompute3.py`, `recompute3_out.txt`

Recupero sul FITTING con riferimenti omologhi (clean_ls stimato su y* del fitting e valutato su y del fitting: `linear_diagnostics.L0.temporal_margin_clean_fit_statistics`;
oracolo del fitting ricavato esattamente da `L13.temporal_margin_recovery_fitting`, controls.py:599-602, 627-633; stessi denominatori 2957984.5843438697 / 2878704.615363543 delle `selected_fitting_diagnostics`):

| | recupero fitting fold0 / fold1 / pooled | recupero esclusi fold0 / fold1 / pooled |
|---|---|---|
| TCN_post (LR 0,003, esp. 31) | 86,1181 / 69,2509 / 77,6423 % | 24,1731 / 18,8755 / 21,7101 % |
| MLP (LR 0,001, esp. 24) | 37,7679 / 27,4513 / 32,5838 % | -12,2400 / -10,6539 / -11,5026 % |
| L0 (LR 3e-4, esp. 183) | -0,2491 / 2,0040 / 0,8831 % | -13,5531 / -10,8188 / -12,2819 % |

Pooled fitting: clean_ls 0,728861505, oracolo 0,673525011 (margine 0,055336; sugli esclusi 0,057820).
Lettura (IPOTESI, non dimostrazione): in campione la TCN chiude il 77,6% del margine, fuori campione il 21,7%. Il divario fitting/esclusi misurato con gli stessi riferimenti e' di
56 punti per la TCN, 44 per la MLP, 13 per L0. Non separa "T appresa sugli utenti del fitting ma non trasferita" da "rumore memorizzato": serve l'errore sul bersaglio pulito del FITTING, che nei JSON non c'e' (c'e' solo per gli esclusi).
La testa L0 allenata sul bersaglio rumoroso sta a -12,28% sugli esclusi: il riferimento clean_ls e' stimato sul bersaglio SENZA rumore, quindi una testa parte gia' sotto zero per il solo costo di stima della parte lineare.

Bersaglio pulito sugli esclusi (pooled): TCN 0,139236, MLP 0,196055, L0 0,199032 (accumulatori); clean_ls circa 0,176461 (derivato come (sse_oss - sse_oracolo)/ss pulita, esatto a meno di un termine incrociato che per le tre teste vale -0,78%, +0,36%, -0,45% della sse pulita); oracolo 0 per costruzione.
Sul pulito MLP e L0 sono PEGGIO di clean_ls (0,196 e 0,199 contro ~0,176). La frase della diagnosi "0,139236 contro MLP 0,196055 conferma apprendimento parziale" e' corretta, ma il confronto piu' severo e' con clean_ls: (0,176461-0,139236)/0,176461 = 21,1%.

Nessuno dei 915 punti (esposizione > 0) delle cinque curve TCN temporali scende sotto la soglia 0,699569.

## 2. Prova di rappresentabilita' di GPT

- Riesecuzione di `teacher_embedding_diagnostic.py` (copia con sha256 e1d548650c69... identico all'originale): JSON prodotto IDENTICO a `teacher_embedding_result.json`
  (fold0 max 4.818883447832434e-06, rmse rel 4.664543530503105e-07; fold1 4.355792947219328e-06, 4.6594590972835516e-07). File: `teacher_embedding_rerun.json`.
- Costruzione controllata sul codice (heads.py:36-58, 64-78, 85-94; controls.py:142-190):
  - ramo affine: W_L/std_L, (b_L - mean_L)/std_L -> riproduce L normalizzata; corretto.
  - ultimo strato: peso x (0,4/std_T) per canale di uscita, bias (b - mean_T) x 0,4/std_T -> riproduce 0,4 x T normalizzata; corretto.
  - LayerNorm della testa: peso = 1 e bias = 0 (verificato a runtime), eps 1e-5 = eps del generatore (controls.py:167, heads.py:13,36).
  - canali in piu' azzerati: canali con peso non nullo per strato [8, 8, 8, 8, 8, 20]; bias azzerati.
  - dropout: due moduli p = 0,1 (dopo il quarto e il quinto strato), spenti da `.eval()`.
  - l'identita' vale anche senza crop: errore massimo sulle 160 patch uguale a quello sul crop6 (stesso padding a zero).
- Estensione mia (`embedding_extension.py`, `embedding_extension_result.json`; feature sintetiche iid e AR(0,9), 4 e 16 finestre, due fold; nessun update):
  - (a) STESSI pesi con dropout attivo (`model.train()`, 200 maschere): RMSE relativo all'intero y* 0,279-0,347; relativo alla sola componente 0,4T 0,742-0,949;
    errore centrato nel tempo / varianza centrata di y* (analogo di E_dyn sul pulito) 0,077-0,133; loss produttiva attesa sul bersaglio PULITO 0,091-0,141 (in eval 2,5e-13).
    L'uscita media sulle maschere e' distorta: 0,107-0,143 della RMS di 0,4T (il primo dropout passa da una ReLU).
  - (b) gradiente della loss pulita in quei pesi: eval 1,2e-7 - 2,4e-7 (zero numerico FP32; allo stato iniziale congelato 0,49-0,78); train: norma media 0,49-0,80 (min 0,45, max 0,88), quasi tutta nel ramo deep.
  - Quindi: i pesi del teacher incorporati NON sono un punto fisso della ricetta di training (dropout 0,1 su 8 canali attivi; in piu' il weight decay disaccoppiato li contrae di lr x 0,01 a passo).
- Cosa dimostra: esiste un'assegnazione dei 72.904 parametri della testa produttiva che in eval calcola L + 0,4T a precisione FP32, per qualunque ingresso (identita' algebrica: il generatore e' una sottorete).
- Cosa NON puo' dimostrare: (i) che AdamW la raggiunga dallo stato iniziale (uscita a zero) con etichette rumorose; (ii) che sia stazionaria per l'obiettivo di training (non lo e': vedi a, b);
  (iii) che il 50% sia stimabile con 920/934 finestre di 20 utenti e sigma 1,3; (iv) nulla sui latenti veri o su A100 (feature gaussiane iid, 4 finestre, normalizzazione stimata su 2 finestre);
  (v) nulla sulla generalizzazione fra utenti. I numeri in modalita' train sono su feature giocattolo: l'ordine di grandezza non si trasferisce ai latenti veri.

- Gradiente ATTESO sulle maschere (`embedding_expected_grad.py`, 8 finestre iid, 64 maschere): norma del gradiente medio 0,535 (fold0) / 0,464 (fold1), contro una soglia di rumore 0,070 / 0,061 se la media fosse zero:
  il gradiente atteso della loss pulita in modalita' train NON e' nullo. Sui 56 canali spenti del primo strato il gradiente e' esattamente 0 (unita' morte: l'incorporamento resta un allenamento largo 8).
- Campo ricettivo misurato per perturbazione (`rf_check.py`): testa TCN_post 13 patch (74..86), generatore T 13 patch (74..86), L 1 patch. Parametri: generatore T 3.932; testa 72.904 = 64.436 deep + 7.700 affine + 768 LN.
- Ledger archiviato (`ledger_through_controls_20261121.json`): 16 ID logici, nessuno di selezione reale, rifit o inferenza.
- NON verificabile: identita' bit a bit fra il teacher ricostruito sul Mac (torch 2.14) e quello di produzione (torch 2.2 su Leonardo): gli hash del generatore non sono nei JSON archiviati. La prova resta valida per qualunque peso (identita' algebrica).

## 3. Confronto con la diagnosi dell'altra chat GPT (8fac3a82, D0/D1) — `recompute4.py`, `recompute4_out.txt`

Numeri dell'altra diagnosi ricalcolati da me dagli stessi JSON: tutti confermati (L0 in forma chiusa su y -19,10 / -16,38 / -17,84 %; da L0 neurale 33,22 / 26,80 / 30,27 %, soglia equivalente pooled 55,47 %;
TCN contro L0 neurale 2,672 %; divari fitting->esclusi TCN +0,03478 / +0,02824, L0 chiuso +0,01555 / +0,01368, L13 +0,09194 / +0,08219, clean_ls +0,00178 / +0,00029;
L13 chiuso su y 0,667239 -> 0,759182 e 0,708586 -> 0,790777; LR 0,003 a 150 epoche 0,741556 = +3,580 %; recupero apparente sul fitting 86,12 / 69,25 %; euristica fra mondi 19,74 / 18,70 % e 43,91 / 37,57 %; 150/900 x 4427 s = 0,2050 GPUh).
Nessuna grandezza comune ha valori diversi nelle due diagnosi. Le differenze sono di contenuto e di regole:

1. Asimmetria del riferimento: l'altra la dichiara (teste allenate su y partono sotto zero: L0 neurale -12,28 %, MLP -11,50 %, L0 chiuso -17,84 %); questa la omette e dice solo che l'oracolo conosce il generatore.
2. Fitting contro esclusi: questa da' il solo pooled (0,685897 contro 0,715926) e lo chiama "divario descrittivo"; l'altra da' i fold, il recupero apparente 86,12 / 69,25 % e il confronto con L0/L13 chiusi. Stessi dati, lettura piu' forte nell'altra.
3. Comparatori: questa confronta con la MLP (+2,61 %; pulito 0,139236 contro 0,196055); l'altra con L0 neurale (+2,67 %; 0,139236 contro 0,199032, circa -30 %). Entrambe esatte.
4. Prova costruttiva: questa 4 finestre per fold, massimo 4,82e-6 / 4,36e-6, RMSE relativo 4,66e-7; l'altra 8 finestre, massimo 6,4e-6, RMSE circa 5e-7. Ingressi giocattolo diversi, stessa conclusione.
5. Diagnostici sui checkpoint: questa "senza nuovi fit ... sui soli artefatti gia' pinnati", checkpoint selezionati, piu' scomposizione affine/deep, norme e deriva della LN (assenti nell'altra);
   l'altra dichiara che e' un JOB su Leonardo (calibration.pt e selection_states.pt stanno su WORK), 20 stati TCN (5 LR x migliore/finale x 2 fold) + 4 stati L0/MLP, parita' 1e-12 prima degli score, cleanLS/oracolo del fitting dagli array originali.
6. Controllo pulito: questa "massimo 150 epoche", letture a 1, 10, 31, 75, 150, esiti descritti con "scarso/buono", nessuna regola di scelta dell'epoca, nessun cancello di parita' del driver;
   l'altra "150 epoche fisse", letture a 31, 150 e scelta (minimo MSE pooled contro y originale degli esclusi), tabella di lettura ordinata 0-3c con soglia 0,50 su R_held e R_fit e flag `late` 146-150, parita' del driver sul primo punto del LR 0,003. La condizione 1 della review Claude f6669115 (regola decidibile prima dei fit) e' soddisfatta dall'altra, non da questa.
7. Candidato: questa nomina gia' il residuale (LN fissa + base ridge congelata + TCN sul residuo) e chiede la riqualifica con 90% / 50% senza dire su quale mondo;
   l'altra non nomina la variante e chiede un MONDO NUOVO con semi dichiarati prima (20261122/23/24 e 20262122/23/24), un solo tentativo. La condizione 3 della review Claude (non scegliere e confermare sullo stesso mondo) non e' in questa.
8. Costi: questa stima 0,2-0,4 GPUh per il solo controllo pulito (il limite basso coincide con 150/900 x 4427 s = 0,205 GPUh); l'altra dichiara quel conto "incompleto" e non lo usa; Claude aveva indicato 1-2 GPUh per il pacchetto completo con il braccio teacher-init.
9. Epoche: entrambe escludono "piu' epoche" come rimedio. Dati: recupero pooled al punto finale delle cinque curve +0,57 % (1e-4, 300), -5,93 % (3e-4), -39,81 % (1e-3), -22,62 % (3e-3), -15,79 % (1e-2).

## 4. Esito sintetico

- Difetti di fatto nei numeri della diagnosi e8c98244: NESSUNO trovato. Tutti i numeri elencati nel compito sono confermati dai JSON grezzi.
- Omissioni/debolezze (non errori): asimmetria del riferimento non dichiarata; divario fitting/esclusi non normalizzato con riferimenti omologhi (77,64 % contro 21,71 % pooled);
  regola di lettura del controllo pulito non decidibile; "senza nuovi fit" non e' "senza job"; limite basso del costo = proporzione gia' dichiarata incompleta dall'altra chat.
- Due diagnosi GPT su due rami con seguiti diversi: va scelta una sola catena.

## 5. Selettore e costi (controlli aggiuntivi)

- Replay del selettore congelato (`selection.assessment` sulle curve archiviate, `selector_replay2.py`): `selected`, `optimizer_alarms`, `minimum_exposures`, `objective_tradeoff`, `optimization_unresolved` identici ai report per tutti e sei i controlli.
- LR 0,003 del controllo temporale TCN alle esposizioni di lettura proposte (pooled, esclusi, bersaglio rumoroso): esp 1 E_dyn 0,760630 (-55,61 %); 10: 0,726704 (+3,07 %); 27: 0,715826 (+21,88 %); 31: 0,715926 (+21,71 %); 50: 0,720715 (+13,43 %); 75: 0,729147 (-1,15 %); 100: 0,734876 (-11,06 %); 150: 0,741556 (-22,62 %).
  LR 1e-4: 150: 0,726488 (+3,44 %); 181: 0,725644 (+4,90 %); 201: 0,725580 (+5,01 %); 300: 0,728150 (+0,57 %).
- Costi dai sei sacct (`cost_out.txt`): TCN lineare 750 epoche in 3710 s, TCN temporale 900 epoche in 4427 s -> 4,78 s per epoca (coppia di fold, valutazione compresa) + circa 125 s fissi.
  Controllo pulito a 1 LR, 150 epoche, 2 fold: circa 842 s = 0,234 GPUh con la stessa cadenza di valutazione (la sola proporzione 150/900 da' 0,205). La stima GPT 0,2-0,4 GPUh e' coerente come ordine di grandezza; non comprende parita' del driver, letture aggiuntive fitting/pulito e margine.
  Riqualifica di un candidato TCN nei due mondi con la ricerca completa: circa (3710 + 4427)/3600 = 2,26 GPUh; tutte e sei le ricerche come il 21/11: 3,567 GPUh.

## 6. Ingressi fuori lunghezza (controllo a margine della prova costruttiva)

- La testa riceve zeri nelle posizioni oltre `n_patches` (data.py:122-136; workflow.py:233-234); il generatore lavora sui latenti grezzi senza azzerarli (controls.py:166, 431, 450).
- Il supporto centrale tiene solo t con t+6 < lunghezza (pose_temporal_readout.py:195-199): con campo 13 nessuna uscita supportata dipende da posizioni oltre la lunghezza. Nessun effetto su loss e metriche; resta solo la selezione dinamica (`dynamic_variance(ystar, mask)` sull'intera maschera), identica per tutti i lettori.
- La prova di GPT usa finestre piene (supporto tutto vero): questo caso non lo esercita. Le lunghezze vere non sono sul Mac: non verificabile qui.

## 7. Per chi decide

- Numeri della diagnosi e8c98244: tutti confermati; nessun errore di fatto trovato.
- "Allungare i training" con la ricetta attuale: i dati archiviati dicono di no. Il minimo sugli esclusi cade a 31 su 150 (LR 0,003) e a 181 su 300 (LR 1e-4); ai punti finali il recupero pooled e' +0,57 / -5,93 / -39,81 / -22,62 / -15,79 %.
- Sul fitting, con riferimenti omologhi, la TCN chiude il 77,64 % del margine; sugli esclusi il 21,71 %. Per separare rumore memorizzato da funzione non trasferita serve l'errore sul bersaglio pulito del fitting (un job di sola inferenza sui checkpoint, non un fit).
- I pesi del generatore incorporati sono ottimi in eval e non stazionari in training (dropout): la prova costruttiva non dice nulla sull'apprendibilita'.

Fine note: 11:43 CET.

## 8. Lettura proposta per il job di sola inferenza (IPOTESI, euristica)

Eccesso sull'oracolo, pooled, E_dyn su y: esclusi H, fitting F (riferimenti omologhi).
- TCN: H = 0,045267, F = 0,012372. MLP: H = 0,064471, F = 0,037306. L0 neurale: H = 0,064921, F = 0,054848. clean_ls: H = 0,057820, F = 0,055336.
- Se il divario fosse solo rumore memorizzato (ottimismo simmetrico, vero per stimatori lineari a disegno fisso, qui solo euristica): parte non appresa = (H+F)/2 = 0,028820 (49,8 % del margine 0,057820), parte di rumore = (H-F)/2 = 0,016448 (28,4 %).
  In questa ipotesi l'errore sul bersaglio PULITO del fitting sarebbe circa uguale a quello degli esclusi (E_dyn pulito circa 0,139).
- Se invece il divario fosse mancato trasferimento fra utenti, l'errore pulito sul fitting sarebbe molto sotto 0,139.
- Un solo numero del job di inferenza decide fra le due: E_dyn sul bersaglio pulito del FITTING al checkpoint scelto, da confrontare con 0,139236 degli esclusi (soglia da fissare prima di leggerlo).
- Sul candidato residuale: nei JSON non c'e' evidenza che l'interazione fra ramo affine e ramo deep sia il limite. Nel mondo lineare la TCN recupera il 98,27 % pooled (E_dyn); nel mondo temporale L0 in forma chiusa su y sta a -17,84 %, L0 neurale a -12,28 %: una base lineare stimata sul rumoroso e poi congelata parte sotto zero quanto o piu' del ramo affine allenato insieme.
