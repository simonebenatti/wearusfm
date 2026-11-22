# Audit del ciclo di training (revisore "training_loop") - note di lavoro

Inizio: 2026-11-22 11:23 CET. Albero letto (sola lettura): /Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244
Regole: nessun cluster, nessun git che cambi stato, solo CPU sul Mac, scrivo solo in questa cartella.
I dati veri (latenti, checkpoint) NON sono sul Mac: tutto cio' che e' "giocattolo" usa latenti SINTETICI.

## 1. Lettura riga per riga di train.py e heads.py (fatti dal codice, 11:23-11:30)

Testa TCN_post (heads.py):
- heads.py:36 LayerNorm(384, eps 1e-5, elementwise_affine=True): APPRENDIBILE, unica, condivisa fra ramo affine e ramo profondo (heads.py:86, 94).
- heads.py:46-50 ramo profondo: Conv1d 384->64 k1; 64->64 k3 dil 1; 64->64 k3 dil 2; 64->64 k3 dil 3; 64->32 k1; 32->20 k1. Campo ricettivo 1+2*(1+2+3)=13.
- heads.py:52-57 ReLU dopo i primi 5 strati; Dropout(0.1) dopo la ReLU degli strati di indice 3 (dil 3) e 4 (64->32).
- heads.py:61 ramo affine diretto Linear(384,20) creato dentro fork_rng (non consuma l'RNG del ramo profondo); heads.py:94 uscita = affine(h) + deep(h).
- heads.py:64-78 inizializzazione: norm.weight=1, norm.bias=0; strati profondi tranne l'ultimo: kaiming_normal fan_in/relu, bias 0;
  A ZERO: ultimo strato profondo (peso e bias, heads.py:75-76) e ramo affine (peso e bias, heads.py:77-78). => uscita iniziale identicamente 0.
- heads.py:109-117 optimizer_groups: weight decay 0,01 SOLO su parametri con ndim>=2 e nome che finisce in "weight" (kernel conv e matrice affine);
  NON su bias e NON su LayerNorm (ndim 1).

Ciclo (train.py):
- train.py:233-234 AdamW(lr costante, betas 0.9/0.999, eps 1e-8); NESSUNO scheduler, nessun warmup (grep: nessun lr_scheduler nel modulo).
- train.py:273 model.train() all'ingresso di train_to_step; train.py:309-318 evaluate: model.eval(), no_grad, ripristino del modo in finally.
- train.py:276-280 una permutazione per epoca dal generatore "shuffle" (seme 300000+100*fold+seed); batch 32, ultimo batch parziale tenuto
  (920 = 28*32 + 24; 934 = 29*32 + 6: nel fold 1 un passo su 30 usa 6 finestre con lo stesso peso di passo degli altri).
- train.py:252-266 RNG del dropout isolato: stato salvato/ripristinato attorno al training (seme 200000+100*fold+seed), RNG globale ripristinato.
- train.py:287 loss = metrics.per_window_mse (media delle MSE per finestra sul supporto), metrics.py:24-39.
- train.py:290-295 backward -> clip_grad_norm_(TUTTI i parametri, 1.0) (norma GLOBALE: affine+profondo+LN insieme) -> optimizer.step().
- train.py:320-334 state(): _cpu_copy = detach().cpu().clone() di model.state_dict() e optimizer.state_dict(): COPIA PROFONDA.
- train.py:434-443 per ogni esposizione: train_to_step dei due fold, poi evaluate, poi (se mse < migliore precedente, stretto) best_states = [t.state()]:
  lo stato catturato e' quello appena valutato (nessun update fra evaluate e state()).
- selection.py:196-211 sincronizzazione per esposizione: passi = ceil(esposizione * passi_per_epoca) per fold (29 e 30 passi/epoca); 0, 0.25, 0.5, 0.75, 1..orizzonte.
- workflow.py:298 valutatore = _score_model sugli esclusi (bundle.held[fold]) con model.eval() (workflow.py:245); workflow.py:308-320 il checkpoint scelto
  viene ripristinato e deve riprodurre MSE/E_dyn della curva a 1e-12 (controllo di coerenza stato<->curva gia' nel codice, superato nel job reale).
- NB: gli stessi utenti esclusi servono sia alla selezione (LR, epoca) sia al cancello: il valore del cancello e' semmai ottimistico, non pessimistico.

Nessun difetto concreto trovato alla lettura. Verifiche eseguite sotto (sezione 2).

## 2. Verifiche meccaniche del ciclo VERO (11:29) - script checks.py, uscita checks_out.json
Comando: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<tree>/scripts:<tree>/src ~/.venvs/wearusfm-tok/bin/python checks.py
(dati casuali N=70, forma di produzione 160x384, train.Trajectory con Recipe() di produzione, testa heads.make_head("TCN_post"))
- Parametri a zero all'inizio: norm.bias, tutti i bias, deep.12.weight/bias (ultimo strato profondo), affine.weight/bias. Uscita iniziale max|.| = 0.0.
- Gruppo con weight decay (71.808 parametri): deep.0/2/4/6/9/12.weight e affine.weight. Senza decay (1.096): norm.weight, norm.bias, tutti i bias.
  Parametri per ramo: LN 768, affine 7.700, profondo 64.436 (somma 72.904).
- Flusso del gradiente: passo 1 norma degli strati profondi nascosti = 0.0 esatto (ultimo strato a zero), ultimo strato 0.078, affine 0.218, LN 0.0;
  dal passo 2 gli strati nascosti ricevono gradiente (0.011, 0.014) e la LN pure (0.0016, 0.0033). Al passo 1 i pesi nascosti si muovono solo per il decay (1e-5).
  => il gradiente RAGGIUNGE il ramo profondo dal secondo passo: nessun blocco strutturale.
- Stato migliore: state() non condivide memoria con i tensori vivi (modello e ottimizzatore), resta invariato dopo altri 3 passi mentre il modello vivo cambia. COPIA PROFONDA confermata.
- evaluate: model.training False dentro il valutatore, due forward identici (dropout spento), RNG del dropout non consumato, modo train ripristinato dopo.
  In modo train due forward differiscono (dropout attivo dove deve).
- Ripresa da stato a meta' = corsa continua, bit per bit; due traiettorie con lo stesso seme identiche bit per bit.
- batch(): feature, bersaglio (crop 6:154) e maschera delle stesse righe (allineamento verificato su posizioni non ordinate).
- per_window_mse: gradiente = 2(p-y)/(conteggio_finestra * finestre_tenute), zero esatto fuori supporto, finestra vuota esclusa dalla media.
- Passi per epoca 29 / 30 / 58 (920 / 934 / 1854 finestre); ultimo batch 24 / 6 / 30 finestre.
- Weight decay: fattore complessivo (1 - 0,003*0,01)^passi = 0,9734 a 899 passi (esposizione 31, fold 0), 0,8777 a 4.350 passi (150 epoche): trascurabile all'epoca scelta.
ESITO: nessun difetto meccanico nel ciclo (stato migliore, dropout, gradiente, allineamento, ripresa).

## 3. Giocattolo: impostazione (11:28) - script toy_tl.py
GIOCATTOLO = latenti SINTETICI (stessa costruzione e stesso seme del giocattolo STAT del 21/11: 40 utenti x 46 finestre x 160 x 384, FP16;
20 utenti fitting = 920 finestre, 20 esclusi = 920), generatore VERO controls.generate_components(outer_fold=0), testa VERA heads.make_head("TCN_post"),
ciclo VERO train.Trajectory (Recipe() di produzione per base/nodrop/lnfrozen/outinit/residual; _TestRecipe con gli stessi numeri e UN campo cambiato per noclip e wd0),
loss VERA metrics.per_window_mse. Un fold, LR 0,003, 60 epoche, selezione = minimo della MSE rumorosa sugli esclusi alle epoche intere.
NNPACK disattivato (torch.backends.nnpack.set_flags(False)): stessa loss iniziale 2.6606621742248535 con e senza, 5 volte piu' veloce.
Riferimenti del giocattolo (gen): a = 0,4 (scelto sul fitting per margine piu' vicino all'8%), sigma 1,3; varianza del bersaglio sul fitting 2,8494 = 1,0000 + 0,1600 + 1,6939;
esclusi: E_cleanLS 0,82212, E_oracle 0,76552, margine 6,884%, L0 in forma chiusa su y rumoroso: recupero -8,879%. Identici al giocattolo del 21/11.
Controllo della catena di punteggio: testa con ramo affine = cleanLS in forma chiusa e ramo profondo a zero -> recupero 0,000 e MSE 1,82130 (= riferimento);
con ramo affine = L0 su y rumoroso -> recupero -0,089 e MSE 1,89071 (= riferimento). La mia scala del cancello coincide con i riferimenti in forma chiusa.

## 4. Clip a norma 1,0 nel giocattolo (base, seme 0; 11:30)
Norme PRIMA del clip ai primi 5 passi (totale / affine / nascosti / ultimo strato / LN): 1,962/1,904/0,000/0,472/0,000; 1,777/1,719/0,047/0,448/0,013;
1,460/1,437/0,037/0,255/0,018; 1,167/1,118/0,039/0,334/0,023; 1,034/0,994/0,032/0,283/0,023.
Il clip scatta SOLO in questi primi 5 passi dell'epoca 1 (frazione 0,17 = 5/29), poi mai (frazione 0,00 dalle epoche 2 in poi; mediana della norma 0,43-0,51,
dominata dal ramo affine 0,41-0,50; strati nascosti 0,00-0,09; ultimo strato 0,03-0,09; LN 0,02-0,05).
=> nel giocattolo il passo effettivo E' quello nominale; inoltre con Adam un riscalamento del gradiente cambia solo il peso relativo dei passi nelle medie mobili.
Limite: con i latenti veri la norma puo' essere diversa (dipende dallo spettro di covarianza dei latenti); train.py:291-294 calcola la norma ma NON la registra:
nell'archivio reale la frazione di passi clippati non c'e'. Si puo' misurare senza nuovi fit: forward/backward sui batch del fitting dagli stati salvati (best e finale).
Stima analitica del solo contributo del rumore al ramo affine con batch 32 (h con norma sqrt(384) fissata dalla LN, rumore AR(1) rho 0,78, caso peggiore h costante):
(2/(148*20)) * 1,3*sqrt(148*8,04) * 19,6 * sqrt(20) / sqrt(32) = 0,47 (osservato nel giocattolo 0,41-0,50). Con il batch parziale da 6 finestre del fold 1: x sqrt(32/6) = fino a 1,09.

## 5. Che cosa certifica "nessun optimizer_alarm" (selection.py:155-193) - numeri reali dal JSON selection_temporal_TCN_post.json
Reale: selected LR 0,003 esposizione 31 (MSE 1,8001448213440414, E_dyn 0,7159262203822878); optimizer_alarms []; minimum_exposures mse 31 / edyn 27;
objective_tradeoff flag false (distanza 4 <= 5; regret 0,014%).
Minimi per LR (MSE esclusi pooled, esposizione): 1e-4 1,81816 @181 (orizzonte 300); 3e-4 1,81743 @64; 1e-3 1,81396 @31; 3e-3 1,80014 @31; 1e-2 1,84927 @93. Passo zero 2,89246.
Gli allarmi possibili sono solo quattro, tutti sulla curva del SOLO LR scelto:
(i) minimo (MSE o E_dyn) nelle ultime 5 epoche di 300, valutato solo se l'orizzonte del LR scelto e' 300 (qui e' 150; a 150 il controllo equivalente e' needs_epoch_extension, non scattato: minimi a 31 e 27);
(ii) LR scelto = bordo basso 3e-5 con guadagno > 0,5%; (iii) LR scelto = bordo alto 1e-2; (iv) esposizione scelta = 0,25.
Certifica quindi: il minimo della MSE RUMOROSA degli ESCLUSI e' interno alla griglia dei LR e all'orizzonte. Cioe': con questa ricetta e questo criterio, piu' epoche o un LR fuori griglia non migliorano la MSE rumorosa.
NON certifica: (a) nulla sulla loss di training ne' sulla convergenza del ramo temporale (nessun allarme guarda la loss di training, la norma del gradiente o il clip, che non sono nemmeno registrati);
(b) la risoluzione della griglia (passi di fattore 3-3,3: il vincitore 3e-3 confina con 1e-2 che ha un andamento del tutto diverso; il recupero massimo pooled per LR e' 5,0 / 5,4 / 10,8 / 21,9 / -14,9% dalla nota STAT del 21/11);
(c) il comportamento per fold (curve solo pooled); (d) che l'arresto anticipato non tagli l'apprendimento della funzione (un minimo interno della MSE rumorosa e' compatibile sia con
"funzione appresa" sia con "il rumore viene adattato prima che la funzione sia appresa"); (e) che un'altra ricetta (schedule del LR, regolarizzazione, architettura) non farebbe meglio.
"optimization_unresolved: false" e' solo bool(alarms).

## 6. Curve reali ricalcolate dal JSON (11:33) - file real_curves_recovery.txt
Riferimenti pooled (selection_temporal_TCN_post.json -> closed_form_references.temporal_records): clean_ls 0,7284790287, oracle 0,6706589240, margine 0,0578201047.
Recupero pooled lungo la curva reale del LR scelto 0,003 alle esposizioni 31 / 45 / 60 / 90 / 120 / 150: 21,71 / 14,23 / 7,48 / -6,00 / -14,63 / -22,62 %
(MSE esclusi 1,80014 / 1,81062 / 1,81618 / 1,83273 / 1,84211 / 1,85215).
Per LR: al minimo della MSE -> 1e-4: 4,90% (esp 181, orizzonte 300, fine 0,57%); 3e-4: 5,36% (64; fine -5,93%); 1e-3: 9,49% (31; migliore E_dyn 10,81% a 28; fine -39,81%);
3e-3: 21,71% (31; migliore 21,88% a 27; fine -22,62%); 1e-2: -15,40% (93; migliore -14,85% a 111; fine -15,79%). 915 punti con esposizione > 0.
Fitting al checkpoint scelto: MSE 1,688067 / 1,709501, E_dyn 0,661873 / 0,710583 (fold 0 / 1).
=> DIMOSTRATO dai JSON: con la ricetta congelata (LR costante) ALLUNGARE il training peggiora: tutti e cinque i LR hanno il minimo interno e a fine orizzonte il recupero e' piu' basso che al minimo.
   I training sono GIA' stati portati a 150 epoche (300 per 1e-4). "Allungare" ha senso solo cambiando la ricetta (es. LR decrescente o piu' regolarizzazione): e' una ricetta nuova.

## 7. Giocattolo, prima ondata (seme 0, 60 epoche, LR 0,003; 11:32) - log_*_s0.txt, toy_run_*_s0.json
EVIDENZA INDICATIVA: latenti sintetici, un fold, un seme (il secondo seme sotto), selezione su un solo fold.
| variante | epoca scelta (min MSE rumorosa esclusi) | recupero alla scelta | recupero massimo (epoca) | recupero a 60 |
| base (ricetta vera)            | 30 | 33,53% | 33,56% (28) |   5,40% |
| senza clip (clip 1e9)          | 29 | 32,18% | 32,46% (21) |   2,96% |
| senza dropout (p=0)            | 30 | 21,75% | 26,84% (21) | -36,04% |
| weight decay 0                 | 21 | 34,48% | 34,48% (21) |   7,28% |
| LayerNorm congelata (1, 0)     | 30 | 34,26% | 35,98% (25) |  11,93% |
| uscita non a zero (default PyTorch sull'ultimo strato profondo) | 29 | 26,55% | 27,13% (28) | 4,65% |
Clip: 5 passi su 1.740 (0,287%) con norma > 1 in tutte le varianti (i primi 5 passi); quantili 5/50/95 della norma nella base 0,367 / 0,437 / 0,529.
Il giocattolo riproduce la firma reale: scelta a epoca 30 (reale 31), recupero 33,5% (reale 21,7% pooled), degrado successivo (5,4% a 60; reale 7,48% a 60).

## 8. Curve REALI per esposizione, TCN contro L0 e MLP (11:36) - real_curves_recovery.txt (ricalcolo dai JSON selection_temporal_{TCN_post,L0,MLP}.json)
Recupero pooled % (scala del cancello) della TCN_post reale:
- LR 0,003: ep 2 -32,3; ep 5 -14,1; ep 8 -2,7; ep 10 3,1; ep 16 17,3; ep 18 19,9; ep 24 21,5; ep 31 21,7; ep 40 17,2; ep 64 4,9; ep 100 -11,1; ep 150 -22,6.
- LR 0,001: ep 8 -10,8; ep 12 -0,3; ep 20 8,7; ep 28 10,8; ep 40 6,8; ep 64 -4,3; ep 150 -39,8.
- LR 3e-4: ep 16 -21,3; ep 35 0,0; ep 64 5,4; ep 100 0,8; ep 150 -5,9.   LR 1e-4: ep 50 -18,3; ep 100 -1,3; ep 150 3,4.
- LR 0,01: ep 8 -21,5; ep 14 -19,8; ep 31 -16,7; ep 64 -16,4; ep 100 -16,7; ep 150 -15,8: PIATTA per 150 epoche al livello di L0.
L0 neurale reale (stessa scala): LR 0,003 ep 31 -15,3, ep 150 -14,8; LR 3e-4 ep 150 -12,5 (scelto: LR 3e-4 esp 183, -12,28%). MLP reale: scelto LR 1e-3 esp 24, -11,50%; LR 0,003 a 150: -66,3.
Letture (dai soli numeri archiviati):
(a) a LR 0,003 la TCN reale supera L0 gia' da epoca 2-3 (-32,3 contro -35,6; -22,9 contro -28,5) e a epoca 8 di 17 punti: il ramo profondo e' partito SUBITO nel run reale del LR scelto.
(b) a LR 0,01 la TCN reale e' indistinguibile da un L0 (da -21,5 a -15,8 senza salita ne' degrado; L0 a LR 0,003: da -20,1 a -14,8): il ramo profondo non contribuisce per tutte le 150 epoche.
    IPOTESI (non verificabile senza i checkpoint): ramo profondo spento/collassato a LR alto. Verificabile senza nuovi fit sullo stato finale salvato del LR 0,01 (norma dell'uscita del ramo profondo, frazione di ReLU attive).
(c) il bordo alto della griglia (1e-2) quindi non e' un punto utile: fra 3e-3 e 1e-2 (fattore 3,3) non c'e' nessun LR provato, e il recupero massimo cresce col LR fino a 3e-3 (5,0 / 5,4 / 10,8 / 21,9).
(d) il plateau 19-22% dura da epoca 18 a 35 a LR 0,003: non e' un picco stretto perso dalla selezione.

## 9. Giocattolo, seconda ondata in corso (seme 1): ramo profondo DORMIENTE (11:35)
Con seme 1 (init 100001, shuffle 300001, dropout 200001) la ricetta base resta al livello di L0 per almeno 20 epoche: recupero -0,389 (ep 4), -0,218 (8), -0,149 (12), -0,124 (16), -0,109 (20);
norma mediana del gradiente degli strati nascosti 0,00 e dell'ultimo strato 0,03 (seme 0: decollo a epoca 4, nascosti 0,02-0,09).
Identico entro 3 cifre per base, senza dropout, wd 0 (ramo profondo inattivo => queste scelte non contano); senza clip -0,126 a ep 16.
Decollano invece: uscita non a zero (ep 8: -0,006; ep 20: 0,244), LN congelata (ep 16: 0,129; ep 20: 0,225), base lineare congelata + residuo (ep 4: 0,150; ep 20: 0,283).
=> nel giocattolo il tempo di decollo del ramo profondo dalla uscita a zero dipende dal seme; con il ramo affine (e la LN) in training congiunto a LR costante il decollo puo' tardare decine di epoche.
   Nei dati REALI al LR scelto e seme 0 il decollo e' immediato (sezione 8a); il run reale a LR 0,01 ha invece la firma del ramo profondo inattivo (8b).

## 10. Giocattolo, risultati completi (11:41) - EVIDENZA INDICATIVA (latenti sintetici, un fold, LR 0,003, selezione = min MSE rumorosa sugli esclusi a epoche intere)
Riproduzione: per ogni riga `TOY_NNPACK=0 [TOY_FOLD=f] <python obbligatorio> toy_tl.py run <nome> <variante> <seme> <epoche> <max_secondi>` dopo `toy_tl.py gen`; log in log_<nome>.txt, curve in toy_run_<nome>.json.
"seme" = seme di Trajectory (init 100000+s; con TOY_FOLD=f gli stream shuffle/dropout sono 300000+100f+s / 200000+100f+s). Orizzonte 60 epoche (seme 0) o troncato dal tempo (40-47).

Ricetta base (5 esecuzioni): s0/f0 33,53% (ep 30) | s1/f0 -9,61% (RAMO PROFONDO MORTO, 43 epoche) | s2/f0 29,07% (ep 24) | s0/f1 31,97% (ep 27) | s1/f1 31,21% (ep 28).
  -> 4 esecuzioni vive: 29,07-33,53% (ampiezza 4,5 punti: differenze fra varianti sotto ~4 punti NON sono attribuibili); 1 su 5 morta.
Una ablazione alla volta (seme 0 | seme 1, stream f0):
| senza clip            | 32,18% (ep 29) | -9,63% (morto) |
| senza dropout         | 21,75% (ep 30; massimo 26,84% a 21; -36,04% a 60) | -9,61% (morto) |
| weight decay 0        | 34,48% (ep 21) | -9,62% (morto) |
| LN congelata          | 34,26% (ep 30) | 32,24% (ep 33; decollo fra ep 12 e 16) |
| uscita non a zero     | 26,55% (ep 29) | 30,79% (ep 36) |
Fuori elenco (aggiunte mie, dichiarate):
| base lineare congelata (OLS su y rumoroso del fitting, LN fissa) + ramo profondo sul residuo | 30,42% (ep 15) | 28,78% (ep 17) | seme 2: 26,76% (ep 12; massimo 31,15% a 17) | nessun ramo morto su 3 |
| idem con base stimata sul bersaglio PULITO (non realizzabile sui dati veri, solo riferimento) | 37,17% (ep 15) |
| dropout 0,3 (seme 0, 40 epoche) | 31,00% (ep 28) |   | LR a coseno fino a 0 su 40 epoche (seme 0) | 28,73% (ep 23) |
| ultimo batch da 6 finestre (902 finestre) | 30,03% (ep 38; massimo 32,88% a 27; 21 passi clippati su 1.247) |  | nessun batch parziale (896) | 32,81% (ep 28) |
Fitting al checkpoint scelto della base s0 (ep 30): recupero su y 0,980, recupero pulito 0,594 (esclusi 0,335): la funzione e' appresa al 59% sugli utenti del fitting.
Lettura: NESSUNA delle scelte chieste (clip, dropout, weight decay, LN, uscita a zero) porta il giocattolo vicino al 50%; il massimo fra tutte le varianti realizzabili e' 34,48%, il riferimento non realizzabile 37,17%.
Spostano il risultato: (i) il dropout (senza: -12 punti e degrado molto piu' rapido); (ii) la MORTE del ramo profondo (da ~31% a -9,6%). Non spostano (entro ~4 punti): clip, weight decay, LN, batch parziale.

## 11. Morte del ramo profondo nel primo/secondo epoch (11:38-11:40) - script deadcheck.py, uscite dead_s*.txt  [GIOCATTOLO, ricetta e testa vere]
Comando: <python obbligatorio> deadcheck.py <seme> [lr]   (frazione di attivazioni > 0 dopo ciascuna delle 5 ReLU e unita' con attivita' < 0,1%, su 115 finestre fisse del fitting)
LR 0,003, ultimo strato nascosto (64->32, quello che alimenta l'uscita), frazione attiva ai passi 0 / 10 / 29 / 58 / 116 / 290:
- seme 0: 0,517 / 0,252 / 0,046 / 0,005 (26 unita' morte su 32) / 0,082 / 0,247 (0 morte): quasi morto a 2 epoche, poi si riprende.
- seme 1: 0,501 / 0,152 / 0,003 (27/32) / 0,000 (32/32 MORTE) / 0,000 / 0,000: morto dal passo 58 in poi; uscita del ramo profondo = costante (RMS 0,006-0,013), gradiente dei nascosti 0,00 fino a epoca 43.
- seme 2: 0,470 / 0,293 / 0,047 / 0,073 (17/32) / 0,144 / 0,270: quasi morto, poi si riprende.
LR 0,01: 32/32 unita' morte al passo 29 e fino al passo 290 in TUTTI e 3 i semi (strati precedenti: 34-50 unita' morte su 64). LR 0,001: nessuna morte (minimo 0,262 / 0,342, 0-2 unita' morte) nei semi 0 e 1.
Meccanismo osservato: ai primi passi anche il ramo profondo prende una parte del segnale lineare (RMS uscita 0,05-0,15 ai passi 1-5); mentre il ramo affine lo assorbe (RMS affine 0 -> 1,1 al passo 10)
il contributo profondo viene tolto spegnendo le ReLU (la norma dei pesi d'uscita continua a crescere 0,003 -> 0,015): con Adam a LR costante senza warmup lo spegnimento puo' essere totale e irreversibile (ReLU morta = gradiente zero).
Collegamento con i dati REALI (ipotesi, non dimostrata): la curva reale a LR 0,01 e' piatta al livello di L0 per 150 epoche (sezione 8b) = firma del ramo profondo morto;
a LR 0,003 seme 0 il ramo reale e' vivo (TCN > L0 da epoca 2-3), ma la morte PARZIALE di unita' (nel giocattolo 13-17 unita' su 64 negli strati dilatati dei semi vivi a 1-2 epoche) non e' esclusa.
Verifica senza nuovi fit: sugli stati salvati (best e finale dei 5 LR, selection_states.pt) contare unita' morte/frazione attiva per strato e RMS dell'uscita del ramo profondo sul fitting.
Rischio per la produzione: i refit usano i semi 0, 1, 2; un seme con ramo morto darebbe un L0 travestito senza alcun allarme (selection.assessment non lo vede).

## 12. Chiusura (11:43)
Esito dell'audit:
- DIFETTI MECCANICI nel ciclo (stato migliore, dropout in valutazione, gradiente, allineamento batch, ripresa, weight decay, clip): NESSUNO trovato (sezioni 1-2, dimostrato su CPU).
- FRAGILITA' DELLA RICETTA dimostrata nel giocattolo: morte del ramo profondo nelle prime 1-2 epoche (1 esecuzione su 5 a LR 0,003; 3 semi su 3 a LR 0,01; nessuna a LR 0,001).
  Nei dati reali: firma compatibile a LR 0,01 (curva piatta al livello di L0); al LR scelto 0,003 il ramo e' vivo. SOSPETTO da verificare sugli stati salvati, senza nuovi fit.
- Il clip a 1,0 nel giocattolo scatta in 5 passi su 1.740 (i primi 5): il passo effettivo e' quello nominale; toglierlo non cambia il risultato (32,18% contro 33,53%, dentro la dispersione).
- Allungare i training con la ricetta congelata peggiora (curve reali: 21,71% a 31 -> -22,62% a 150).
- "Nessun optimizer_alarm" certifica solo che il minimo della MSE rumorosa del LR vincitore e' interno a griglia e orizzonte.
Limiti: latenti sintetici; un fold; nessuna ricerca del LR per variante; orizzonte 40-60 epoche; 1-2 semi per ablazione; percorsi CUDA non verificabili sul Mac; test pytest del repo non rieseguiti.
Pulizia: toy_data.npz (297 MB) cancellato (si rigenera con `toy_tl.py gen`, 10 s, seme fisso); albero tree_e8c98244 non toccato (nessun file piu' recente delle 11:23:30, nessun __pycache__); nessun processo rimasto.
File: toy_tl.py, checks.py, deadcheck.py, checks_out.json, dead_s*.txt, log_*.txt, toy_run_*.json, toy_refs.json, real_curves_recovery.txt.

## 13. Aggiunta dopo la chiusura (11:43-11:44): il collasso iniziale NON dipende dal rumore ne' dal ramo affine [GIOCATTOLO]
deadcheck.py con TOY_TARGET=clean (bersaglio senza rumore), TOY_WARMUP=58 (warmup lineare del LR su 2 epoche), variante residual (base OLS congelata). Uscite dead_s*_lr*_*.txt.
Frazione attiva dell'ultimo strato nascosto (minimo fra i passi 10/29/58, poi passo 290; unita' morte su 32 al minimo):
- bersaglio PULITO, LR 0,003: seme 0 0,078 (16 morte) -> 0,358; seme 1 0,032 (25) -> 0,339; seme 2 0,072 (10) -> 0,346. Il quasi-collasso c'e' anche senza rumore; nessuna morte totale su 3.
- warmup 58 passi, LR 0,003 (bersaglio rumoroso): 0,074 (16) -> 0,237; 0,073 (16) -> 0,257; 0,139 (6) -> 0,283: nessuna morte totale su 3 (compreso il seme 1), ma il crollo arriva a fine warmup.
- warmup 58 passi, LR 0,01: 32/32 morte dal passo 58 nei semi 0 e 1 (2 su 2).
- base congelata + residuo, LR 0,003: 0,058 (15) -> 0,222; 0,098 (12) -> 0,261; 0,006 (22) -> 0,278: nessuna morte totale su 3.
- base congelata + residuo, LR 0,01: 32/32 morte gia' al passo 10 e fino al 290 in 3 semi su 3.
=> il collasso delle ReLU nelle prime 1-2 epoche e' una proprieta' del ramo profondo con questa inizializzazione e AdamW a LR >= 0,003 (non del rumore, non della competizione col ramo affine);
   il candidato residuale NON ne e' protetto a LR 0,01; il controllo sul bersaglio pulito puo' incontrarlo: va registrata la frazione attiva per non leggere un ramo morto come "difficolta' di ottimizzazione della funzione".
toy_data.npz cancellato di nuovo; cartella 1,0 MB.
