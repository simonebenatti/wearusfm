# Verifica scettica del difetto TL-1 (lente training_loop) - note di lavoro

Inizio: 2026-11-22 (ora nel primo blocco sotto). Ruolo: scettico, provo a CONFUTARE.
Albero letto in sola lettura: /Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244
Cartella di lavoro (solo qui scrivo): /Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/verify/ (script in tl1_work/)
I dati veri (latenti, checkpoint) NON sono sul Mac: ogni misura eseguita e' su latenti SINTETICI.

## 1. Riproduzione della prova dell'altro revisore (11:47-11:49) - DIMOSTRATO (giocattolo)
Script copiati in tl1_work/ (toy_tl.py identico salvo la riga HERE, per non scrivere nella cartella altrui; deadcheck.py identico, sha256 ab903af2...).
Comando (da tl1_work/): `<python obbligatorio> toy_tl.py gen` poi `<python obbligatorio> deadcheck.py 1 0.003`, `deadcheck.py 0 0.01`, `deadcheck.py 0 0.01 residual`, `deadcheck.py 0 0.003`.
- toy_refs.json rigenerato: IDENTICO a quello dell'altro revisore (salvo il campo seconds). a = 0,4; margine esclusi 6,884%.
- Le quattro uscite sono IDENTICHE byte per byte (cmp) a dead_s1.txt, dead_s0_lr0.01.txt, dead_s0_lr0.01_residual.txt, dead_s0.txt.
- Numeri della dichiarazione confermati: ultimo strato nascosto (64->32), frazione attiva ai passi 0/29/58/290:
  seme 1 LR 0,003: 0,501 / 0,003 (27/32 morte) / 0,000 (32/32) / 0,000 (32/32);
  seme 0 LR 0,003: 0,517 / 0,046 (12/32) / 0,005 (26/32) / 0,247 (0/32);
  seme 0 LR 0,01: 32/32 morte dal passo 20 (31/32 al passo 10) fino al 290; residual LR 0,01: 32/32 dal passo 10 al 290.
- Dettaglio non detto nella dichiarazione: gia' al passo 0 (inizializzazione) ci sono 1-2 unita' "morte" su 32 nell'ultimo strato (seme 0: 1/32; seme 1: 2/32) e 1/64 nel terzo strato dilatato (seme 0).
ESITO: la prova e' riproducibile. La parte "giocattolo" NON e' confutabile per irriproducibilita'.

## 2. Numeri REALI della dichiarazione ricalcolati dai JSON (11:50) - DIMOSTRATO (dai JSON archiviati)
File: tree_e8c98244/results/infra/pose_temponet_execution_20261121_A1/launcher/selection_temporal_{TCN_post,L0,MLP}.json
Riferimenti pooled: clean_ls 0,7284790286546209, oracle 0,6706589239838878, margine 0,057820104670733063. Recupero = (clean_ls - E_dyn)/margine.
- TCN LR 0,01: -21,51% a epoca 8, -15,79% a 150 (dichiarato -21,5 / -15,8: CONFERMATO). L0 LR 0,003: -20,05% a 8, -14,83% a 150 (dichiarato -20,1 / -14,8: CONFERMATO).
- Epoca 3, LR 0,003: TCN -22,87%, L0 -28,49% (dichiarato -22,9 / -28,5: CONFERMATO).
- Griglie: TCN 5 LR (1e-4 ... 1e-2); L0 e MLP 4 LR (1e-4 ... 3e-3): NON esiste un L0 a LR 0,01 nell'archivio; il confronto "piatto come L0" e' con L0 a LR 0,003 (LR diverso).
- Piattezza, esposizioni >= 20: TCN LR 0,01 recupero fra -18,45 e -14,85 (media -16,53, sd 0,73); L0 LR 0,003 fra -17,10 e -14,28 (media -15,47, sd 0,54);
  TCN LR 0,003 fra -24,81 e +21,88 (sd 14,56). Differenza TCN(0,01) - L0(0,003) alle 150 epoche intere: fra -2,96 e +2,91 punti, media -1,10.
- Il ramo profondo REALE al LR scelto contribuisce da subito: TCN(0,003) - L0(0,003) = +2,37 / +3,35 / +5,61 / +9,44 / +17,32 punti alle epoche 1 / 2 / 3 / 5 / 8
  (L0 e TCN hanno lo stesso flusso di shuffle e lo stesso ramo affine a zero: L0 e' un gemello quasi esatto di una TCN col ramo profondo spento).
- Dettaglio utile: la TCN a LR 0,003 a 150 epoche degrada a -22,62% (sovradattamento), la TCN a LR 0,01 NO (-15,79%): un ramo profondo vivo da 64.436 parametri a LR 3,3 volte piu' alto
  che non sovradatta in 150 epoche non e' plausibile; la lettura "ramo profondo non funzionante a LR 0,01" resta la piu' semplice. Resta IPOTESI (nessun checkpoint letto).
ESITO: la parte "REALE" della dichiarazione e' numericamente corretta; non riesco a confutarla con una spiegazione piu' semplice.

## 3. Prima crepa nella dichiarazione: "ReLU morta = gradiente zero" non e' esatto nel giocattolo (11:51) - DIMOSTRATO (dai JSON dell'altro revisore)
toy_run_base_s1.json (esecuzione "morta", seme 1, LR 0,003): norma MEDIANA per epoca del gradiente degli strati nascosti = 0,000978 (ep 3), 0,000548 (ep 5), 0,000337 (ep 10),
0,000199 (ep 20), 0,000143 (ep 30), 0,000125 (ep 40), 0,000211 (ep 43): piccola ma NON zero (nelle note dell'altro revisore "0,00" e' un arrotondamento a due decimali).
In modo train (dropout attivo a monte dello strato 64->32) qualche attivazione resta: con Adam un gradiente piccolo ma coerente produce comunque passi di ampiezza ~LR.
=> "irreversibile" e' provato solo fino a 43 epoche (giocattolo); l'orizzonte reale e' 150. Verifica mia sotto (sezione 5).
Confronto vivo/morto nel giocattolo (recupero %, seme 0 vivo meno seme 1 morto): -0,11 / -1,07 / +1,20 / +10,22 / +16,90 / +25,75 alle epoche 1 / 2 / 3 / 4 / 5 / 8:
nel giocattolo il ramo vivo "decolla" a epoca 4; nel REALE la differenza TCN - L0 cresce gia' da epoca 1 (+2,37 / +3,35 / +5,61): nessuna fase dormiente visibile nel reale al LR scelto.

## 4. Sonda mia su TUTTO il fitting (920 finestre), in modo eval E in modo train (11:52-11:54) - DIMOSTRATO (giocattolo) - script tl1_work/longdead.py
Comando: `<python obbligatorio> longdead.py <seme> <lr> <base|residual> 4350` (4350 passi = 150 epoche; uscite long_*.txt). Risultati parziali (fino a epoca 40-60) scritti subito:
- La sonda dell'altro revisore usa 115 finestre e il modo eval. Sull'intero fitting, in modo EVAL: seme 1 LR 0,003 -> unita' MAI attive nell'ultimo strato nascosto 19/32 (ep 1), 31/32 (ep 2), 32/32 (ep 10), 31/32 (ep 20), 32/32 (ep 30 e 40).
  La morte "in eval" e' quindi confermata anche su 920 finestre (non e' un artefatto del sottocampione).
- In modo TRAIN (dropout attivo, cioe' come vede il gradiente): stesso caso, unita' mai attive solo 1-2 su 32; frazione attiva 0,00202 (ep 2), 0,00048 (ep 10), 0,00028 (ep 20), 0,00024 (ep 30), 0,00020 (ep 40);
  gradiente degli strati nascosti NON nullo in 29 batch su 29 a ogni sonda (RMS per batch 9,4e-4 a ep 2 -> 1,2e-4 a ep 40).
  => "ReLU morta = gradiente zero" e' FALSO alla lettera a LR 0,003: il ramo e' funzionalmente spento ma riceve un gradiente piccolo (via dropout a monte); la tendenza fino a ep 40 e' a spegnersi ancora, non a riprendersi.
- LR 0,01 (seme 0, seme 1, e base congelata + residuo seme 0): 32/32 mai attive in eval su 920 finestre da epoca 1 in poi (fino a ep 40-60 lette finora); in train 22-32/32 mai attive,
  gradiente dei nascosti esattamente zero nella maggior parte dei batch (non nulli: da 0 a 13 su 29 dopo ep 10). A LR 0,01 la morte e' sostanzialmente totale.
- I bias dell'ultimo strato nascosto restano piccoli (min -0,07 a ep 2): lo spegnimento passa dai PESI in ingresso (ingressi post-ReLU non negativi => tutti i pesi di un'unita' si muovono nello stesso verso con passi ~LR di Adam).

## 5. Il collasso dipende da come ho costruito i latenti sintetici? (11:52-11:55) - DIMOSTRATO (giocattolo) - script tl1_work/altlat.py
Comando: `<python obbligatorio> altlat.py <orig|seedB|nouser|common|common3|white>` (generatore vero, a = 0,4 e sigma = 1,3 fissi; uscite alt_*.txt). Varianza del bersaglio identica in tutte (1 + 0,16 + 1,694).
Descrittori di z = LN(latente) (quota di potenza: modo comune a tutte le finestre / media di finestra / entro finestra): orig 0,031/0,637/0,363; seedB 0,028/0,635/0,365; nouser 0,001/0,095/0,905;
common 0,699/0,885/0,115; common3 0,954/0,982/0,018; white 0,000/0,006/0,994.
LR 0,003, ultimo strato nascosto, frazione attiva in eval su 115 finestre: MINIMO fra i passi 10-116 -> valore al passo 290 (semi 0 / 1 / 2):
- orig:    0,007 -> 0,244 | 0,000 -> 0,044 (in ripresa) | 0,047 -> 0,268
- seedB:   0,031 -> 0,215 | 0,004 -> 0,221 | 0,000 -> 0,000 (32 unita' sotto 0,1%, deep RMS 0,017: SPENTO a 10 epoche)
- nouser:  0,064 -> 0,294 | 0,144 -> 0,326 | 0,107 -> 0,324
- common:  0,018 -> 0,296 | 0,026 -> 0,244 | 0,045 -> 0,335
- common3: 0,001 -> 0,000 (31 unita', deep RMS 0,002: SPENTO a 10 epoche) | 0,029 -> 0,341 | 0,179 -> 0,358
- white:   0,060 -> 0,307 | 0,018 -> 0,296 | 0,093 -> 0,310
=> crollo iniziale dell'attivita' (da ~0,5 a <= 0,18) in 18 esecuzioni su 18, qualunque sia la struttura dei latenti (anche rumore bianco, anche senza effetto utente, anche con modo comune al 95%);
   ramo ancora spento a 10 epoche in 2 esecuzioni su 18 (11%). LR 0,01, seme 0: 32/32 sotto 0,1% al passo 29 in 5 costruzioni su 6 (common3: 30/32, frazione 0,031).
   NON posso attribuire il fenomeno a una particolarita' dei latenti sintetici: e' una proprieta' di architettura + inizializzazione + AdamW a LR >= 0,003.
SECONDA CREPA (importante per la formulazione): con la costruzione "orig" e il seme 1, la mia esecuzione NON coincide con quella dell'altro revisore: al passo 290 il ramo e' in ripresa (0,044, deep RMS 0,077)
mentre nel suo e' spento (0,000, 32/32). Unica differenza: lui salva le componenti L, T, eps in float32 (npz) prima di sommarle, io le sommo in float64: differenza del bersaglio all'ultima cifra del float32.
=> l'esito "morto / ripreso" di un singolo seme e' sul filo del rasoio (dipende da perturbazioni di ~1e-7 del bersaglio): NON e' una proprieta' del seme; su GPU (numerica diversa) lo stesso seme puo' dare l'altro esito.
   (verifica diretta della causa in sezione 6)

## 6. Causa della differenza col mio "orig": arrotondamento float32 del bersaglio (11:57) - DIMOSTRATO (giocattolo)
Comando: `<python obbligatorio> altlat.py orig32 0.003:1` (componenti L, T, eps arrotondate a float32 prima della somma, come nel file npz dell'altro revisore).
Esito seme 1, LR 0,003, ultimo strato nascosto, eval: passo 58 0,000 (32 unita' sotto 0,1%), 116 0,000 (32), 290 0,000 (32), deep RMS 0,0131 = IDENTICO al suo deadcheck.
Con le stesse componenti sommate in float64 ("orig"): passo 290 0,0437 (15 unita'), deep RMS 0,0768 = ramo in ripresa.
=> una perturbazione del bersaglio all'ultima cifra del float32 sposta lo stesso seme da "spento a 10 epoche" a "in ripresa a 10 epoche". L'esito per seme non e' riproducibile fra piattaforme.

## 7. Riesecuzione delle altre sonde dell'altro revisore (11:58-11:59) - DIMOSTRATO (giocattolo)
Altre 8 uscite rieseguite e IDENTICHE byte per byte alle sue (totale 12 file su 12): LR 0,001 semi 0 e 1 (nessuna morte: minimo 0,342 e 0,262; 0-2 unita'), bersaglio pulito semi 0/1/2
(minimo 0,078 / 0,032 / 0,072, poi 0,358 / 0,339 / 0,346 al passo 290), seme 2 LR 0,003 (0,047 -> 0,270), LR 0,01 semi 1 e 2 (32/32 dal passo 29 al 290).
Rieseguite anche le prime 10 epoche delle esecuzioni base seme 0 e seme 2 (tl1_work/twin.py): recuperi per epoca uguali ai suoi log (seme 0: -76,19 / -58,09 / -45,24 / -28,65 / -17,47 ... 13,94 a ep 10).
I "1 su 5 morta" e i recuperi 29,07-33,53% li ho letti dai suoi JSON (toy_run_base_*.json), non rieseguiti per intero.

## 8. Gemello L0 nel giocattolo (11:56-11:58) - DIMOSTRATO (giocattolo) - script tl1_work/twin.py
Comando: `<python obbligatorio> twin.py <L0|TCN_post> <lr> <seme> 10`. Recupero % sugli esclusi alle epoche 1 / 2 / 3 / 4 / 5 / 8 / 10:
- L0 LR 0,003:        -76,77 / -58,41 / -46,76 / -40,08 / -33,69 / -22,93 / -17,32
- L0 LR 0,01:         -62,80 / -44,59 / -35,45 / -28,07 / -23,77 / -19,70 / -17,16
- TCN LR 0,01 seme 0: -62,71 / -44,71 / -35,31 / -28,20 / -23,79 / -19,76 / -17,24   (ramo spento): differenza dal gemello L0 allo stesso LR entro +-0,15 punti a tutte le 10 epoche.
- TCN LR 0,003 seme 0: -76,19 / -58,09 / -45,24 / -28,65 / -17,47 / +3,94 / +13,94 -> TCN - L0 = +0,58 / +0,32 / +1,52 / +11,43 / +16,22 / +26,87 / +31,26
- TCN LR 0,003 seme 2: -75,28 / -54,26 / -36,55 / -23,42 / -12,01 / +6,63 / +13,41 -> TCN - L0 = +1,49 / +4,15 / +10,21 / +16,66 / +21,68 / +29,56 / +30,73
Letture: (a) una TCN col ramo spento E' un L0 allo stesso LR (conferma di "la testa diventa un L0").
(b) REALE, TCN - L0 a LR 0,003 = +2,37 / +3,35 / +5,61 / (ep 5) +9,44 / (ep 8) +17,32: sta fra il seme 0 (dormiente fino a ep 3) e il seme 2 (cresce da ep 1) del giocattolo, entrambi passati
    per il quasi-collasso (minimo 0,005 e 0,047). Quindi la curva reale NON esclude un quasi-collasso a 1-2 epoche: il mio tentativo di usare questa curva per confutare non riesce.
(c) REALE a LR 0,01: TCN(0,01) - L0(0,003) = +55,7 / +21,0 / +10,9 (esposizioni 0,25 / 0,5 / 0,75), +2,23 / +2,91 (ep 1, 2), poi fra -2,96 e +2,91 fino a 150. Plateau reale di L0 a 150 epoche per LR
    3e-4 / 1e-3 / 3e-3: -12,51 / -13,55 / -14,83 (MSE 1,83683 / 1,84165 / 1,84441): la TCN a LR 0,01 (-15,79; MSE 1,85229) sta dove cadrebbe un L0 a LR 0,01. Compatibile col ramo spento; resta IPOTESI.

## 9. Soglia del LR nel giocattolo (12:01) - DIMOSTRATO (giocattolo)
Comando: `<python obbligatorio> deadcheck.py <seme> 0.005` e `... 0.007`. Ultimo strato nascosto: 32/32 unita' sotto 0,1% dal passo 58 al 290 in 3 semi su 3 a LR 0,005 e in 3 su 3 a LR 0,007 (deep RMS 0,013-0,015).
=> nel giocattolo il precipizio sta fra 0,003 e 0,005: il LR scelto nel reale (0,003) e' l'ultimo punto di griglia in cui il ramo profondo sopravvive, e l'estensione a 0,01 cade oltre.

## 10. Orizzonte reale (150 epoche): la morte e' irreversibile? (11:52-12:02) - DIMOSTRATO (giocattolo) - tl1_work/longdead.py, uscite long_*.txt
Stesse traiettorie dell'altro revisore (toy_data.npz rigenerato identico; le mie sonde non toccano ottimizzatore ne' stream della traiettoria), portate a 4350 passi = 150 epoche.
- LR 0,003, seme 1 (la sua esecuzione "morta", da lui troncata a 43 epoche): ultimo strato nascosto, frazione attiva in eval su 920 finestre 0,00000 fino a ep 40 (32/32 mai attive),
  poi 0,23048 a ep 60 (0/32 mai attive), 0,28934 a ep 80, 0,31453 a ep 100, 0,31932 a ep 120, 0,32398 a ep 150. Gradiente dei nascosti 1,2e-4 a ep 40 -> 7,1e-2 a ep 60.
  => A LR 0,003 LA MORTE NON E' IRREVERSIBILE: il ramo si riaccende fra ep 40 e ep 60. Era una DORMIENZA di ~45 epoche. Smentita la parte "totale e irreversibile (ReLU morta = gradiente zero)" per il LR 0,003.
- LR 0,01, seme 0 e seme 1 (ricetta base) e seme 0 (base congelata + residuo): 32/32 unita' mai attive in eval a OGNI sonda da ep 1 a ep 150; a ep 150 anche in modo train 32/32 e gradiente
  dei nascosti esattamente 0 in 29 batch su 29. => a LR 0,01 la morte dura tutto l'orizzonte in 3 esecuzioni lunghe su 3 (qui "irreversibile" regge).
- Altre sonde rieseguite: base congelata + residuo, LR 0,01 semi 1 e 2 (32/32 dal passo 10 al 290) e LR 0,003 semi 0/1/2 (minimo 0,058 / 0,098 / 0,006, poi 0,222 / 0,261 / 0,278): identiche alle sue (17 file su 17).
- Costruzioni alternative dei latenti, LR 0,01, passo 290: ramo spento (32 unita' sotto 0,1%) in 17 esecuzioni su 18; unica eccezione nouser seme 2 (0,152, deep RMS 0,325, ripreso).

## 11. L'esecuzione "morta" portata a 150 epoche con il punteggio del cancello (12:00-12:07) - DIMOSTRATO (giocattolo) - tl1_work/twin_every.py
Comando: `<python obbligatorio> twin_every.py TCN_post 0.003 1 150 5` (seme 1, LR 0,003, punteggio sugli esclusi ogni 5 epoche; uscita twin150_TCN_0.003_s1.txt).
Recupero %: ep 5 -34,37; 10 -18,54; 20 -10,93; 30 -10,93; 40 -9,94 (fin qui identico ai log dell'altro revisore); 45 -7,55; 50 +18,05; 55 +26,55; 60 +33,98; 65 +35,28; 70 +35,38; 75 +35,02;
80 +29,46; 100 +16,05; 120 +7,51; 150 -11,60. Minimo della MSE rumorosa sugli esclusi (griglia ogni 5 epoche): epoca 60, MSE 1,829347, recupero 33,98%; recupero massimo 35,38% a epoca 70.
=> il "-9,61%" dell'esecuzione morta e' un artefatto del troncamento a 43 epoche: sull'orizzonte reale (150) quella stessa esecuzione sarebbe scelta verso epoca 60 con ~34%,
   come le esecuzioni vive (29,07-33,53%). "1 esecuzione su 5 morta" va riscritto: "1 su 5 con ramo dormiente per ~45 epoche, poi recupero pieno".

## 12. Verdetto dello scettico (12:08)
NON CONFUTATO nel nucleo, CORRETTO in due punti.
Regge (giocattolo, riprodotto: 17 file su 17 identici byte per byte; mie sonde indipendenti):
 (a) crollo dell'attivita' dell'ultimo strato nascosto nelle prime 1-2 epoche a LR >= 0,003: 18 esecuzioni su 18 in 6 costruzioni diverse dei latenti (non e' una proprieta' dei latenti sintetici);
 (b) a LR 0,005 / 0,007 / 0,01 ramo spento a 10 epoche in 3/3, 3/3 e 17/18 esecuzioni; a LR 0,01 spento per tutte le 150 epoche in 3 esecuzioni lunghe su 3 (compresa la base congelata + residuo);
 (c) una TCN col ramo spento e' un L0 (entro +-0,15 punti dal gemello L0 allo stesso LR);
 (d) nessun allarme lo vede: selection.py:155-193 ha solo 4 allarmi sulla curva del LR scelto; train.py non registra attivita' ne' norme (grep su scripts/pose_temponet: nessun controllo);
 (e) numeri reali della dichiarazione esatti; la curva reale a LR 0,01 sta dove cadrebbe un L0 a LR 0,01 e non sovradatta in 150 epoche: sospetto fondato, NON dimostrato (nessun checkpoint letto).
Non regge:
 (f) "morte totale e irreversibile (ReLU morta = gradiente zero)" al LR 0,003: gradiente non nullo in 29 batch su 29 in modo train; la stessa esecuzione si riaccende fra ep 40 e 60 e arriva a 33,98% (sezioni 4, 10, 11);
 (g) l'esito per seme non e' una proprieta' del seme: basta l'arrotondamento float32 del bersaglio per cambiarlo (sezione 6).
Gravita' che confermo: IMPORTANTE come condizione sui diagnostici e sui prossimi fit (stato: sospetto sui dati reali), NON come spiegazione del cancello fallito (-> no, concordo con l'altro revisore):
 nel reale al LR scelto entrambe le fold sono vive (24,17% / 18,88%) e la curva pooled cala senza una seconda salita dopo ep 31 (nessuna fold dormiente).
 Conta perche': il bordo alto della griglia reale (0,01) e' con ogni probabilita' un L0 travestito, quindi la ricerca del LR si ferma di fatto a 0,003 proprio dove il recupero massimo sta ancora
 salendo col LR (5,0 / 5,4 / 10,8 / 21,9); il candidato residuale NON ne e' protetto; i refit a esposizione fissa (semi 1 e 2) e una prova singola possono incappare in una dormienza senza alcun segnale.
Verifica a costo zero (nessun nuovo fit): sugli stati salvati dei 5 LR (best e finale, due fold) contare per strato frazione attiva e unita' mai attive, in eval E in train, e RMS dell'uscita del ramo profondo sul fitting.
Limiti miei: latenti sintetici; un fold; nessun dato reale letto; a LR 0,003 la dormienza l'ho seguita fino a 150 epoche in 1 sola esecuzione.

## 13. Pulizia (12:08)
toy_data.npz (297 MB) cancellato da tl1_work/ (si rigenera con `toy_tl.py gen`, 10 s). Nessun processo rimasto. Albero tree_e8c98244 e cartella dell'altro revisore non toccati.
