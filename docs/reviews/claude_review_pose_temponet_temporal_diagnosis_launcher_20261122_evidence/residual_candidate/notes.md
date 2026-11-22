# Note del revisore — candidato residuale (22/11/2026)

Cartella: /Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/residual_candidate/
Avvio 11:23. Materiale in sola lettura: tree_e8c98244. Nessun accesso al cluster, nessun dato vero letto.

## 1. Numeri ricalcolati dai JSON archiviati (DIMOSTRATO; script `recompute_json.py`, uscita `recompute_json.out`)

Fonte: `launcher/positive_controls_gate_20261121.json`, `calibration_{0,1}.json`.
Scala del cancello R = (E_cleanLS - E)/(E_cleanLS - E_oracle), E_dyn sugli esclusi esterni.

| | fold0 | fold1 | pooled |
|---|---:|---:|---:|
| clean_ls | 0,710431 | 0,749920 | 0,728479 |
| oracolo | 0,653451 | 0,691103 | 0,670659 |
| margine | 0,056981 | 0,058817 | 0,057820 |
| lineare in forma chiusa su y (alpha=0), esclusi | -19,10% | -16,38% | -17,84% |
| L0 neurale (arresto anticipato) | -13,55% | -10,82% | -12,28% |
| MLP | -12,24% | -10,65% | -11,50% |
| TCN_post | +24,17% | +18,88% | +21,71% |
| TCN meno forma chiusa (punti) | 43,27 | 35,26 | 39,55 |
| TCN meno L0 neurale (punti) | 37,73 | 29,69 | 33,99 |
| serve al ramo temporale partendo dalla base in forma chiusa (punti) | 69,10 | 66,38 | 67,84 |
| rapporto «serve / ottenuto oggi sopra la forma chiusa» | 1,597 | 1,883 | 1,715 |
| quota di T non letta da L0 (fitting) | 0,7117 | 0,8788 | |

Calibrazione interna (10 utenti per meta'): alpha scelto = 1,0 in entrambi i fold, non al bordo;
curva MSE interna fold0: 0,01 -> 1,878179; 0,1 -> 1,855151; 1 -> 1,840503; 10 -> 1,891616.
La ridge con alpha=1 sulla scala del margine INTERNO sta a -19,00% (fold0) e -16,10% (fold1):
la scelta di alpha per CV NON riporta la base sopra zero.
La ridge a 20 utenti con alpha=1 sugli esclusi esterni del mondo temporale NON e' archiviata (ipotesi: fra -9 e -19%).
Indizio dal mondo lineare (alpha=1, 20 utenti): ridge - oracolo = 0,007118 (fold0) e 0,006385 (fold1) di E_dyn,
cioe' 12,5% e 10,9% del margine temporale: stesso ordine di L0 neurale.

Ricetta: AdamW lr 0,003, wd 0,01, batch 32, clip 1,0; scelta all'esposizione 31 = 31*29 = 899 passi (fold0).
Decadimento AdamW cumulato a 31 epoche: exp(-0,003*0,01*899) = 0,9734: il weight decay attuale e' quasi inattivo.

## 2. Curve archiviate sui dati VERI (DIMOSTRATO dai JSON; script `curves_json.py`, uscita `curves_json.out`)

Controllo temporale, pooled, R sulla scala del cancello, per esposizione (selection_temporal_*.json):
- TCN_post LR 0,003: 10 -> +0,031; 20 -> +0,193; 31 -> +0,217 (scelta); 50 -> +0,134; 75 -> -0,012; 100 -> -0,111; 150 -> -0,226.
- TCN_post, massimo per LR (argmin MSE): 1e-4 -> +0,049 (esp. 181 su 300); 3e-4 -> +0,054 (64); 1e-3 -> +0,095 (31); 3e-3 -> +0,217 (31); 1e-2 -> -0,154 (93; curva piatta fra -0,25 e -0,15: il ramo profondo non impara).
- L0 neurale LR 0,003: 5 -> -0,235; 10 -> -0,185; 20 -> -0,163; 31 -> -0,153; 150 -> -0,148. Scelta L0: LR 3e-4, esposizione 183, R -0,123.
- MLP: scelta LR 1e-3 esposizione 24, R -0,115; a 150 epoche R -0,613 (LR 1e-3) e -0,663 (LR 3e-3).
Letture (dimostrate dai numeri): (i) nessuna curva TCN ha il minimo MSE a fine orizzonte: allungare il training con la ricetta attuale
peggiora (LR 0,003: da +0,217 a -0,226 fra 31 e 150 epoche); (ii) sui latenti veri il ramo lineare converge presto
(L0 a LR 0,003 e' a -0,153 gia' a 31 epoche, plateau -0,148): il ritardo del ramo affine NON spiega il risultato vero;
(iii) il massimo raggiungibile cresce con il LR fino a 0,003 e crolla a 0,01: fra 0,003 e 0,01 non ci sono punti di griglia.
Ipotesi (non dimostrata): effetto di ottimizzazione/regolarizzazione implicita del passo grande, non solo tetto statistico.

## 3. Giocattolo CPU (INDICATIVO: latenti SINTETICI, non quelli veri)

Script `toy_res.py` (generatore, riferimenti lineari, testa, loss e ricetta VERI; latenti sintetici di ieri, seme 12345), `drive.sh`.
Fold 0 (fitting utenti 0-19, 920 finestre; esclusi 920), a=0,4, sigma=1,3, LR 0,003.
Riferimenti (toy/refs_f0.json): E_cleanLS 0,822120, E_oracle 0,765524, margine relativo 6,88% (vero: 8,0%);
WLS su y (alpha=0): -8,88% sulla scala del cancello; ridge con alpha da CV interna (alpha scelto 0,1, non al bordo): -8,44%.
NB: nel giocattolo il costo della base lineare stimata sul rumore (-8,4%) e' circa meta' di quello vero (-19,1/-16,4%).
Carico della macchina alto (altri revisori): 20-36 s per epoca; le prime 4 corse (cur, res, lnfix, base seme 0) hanno tetto 600 s;
lnfix e base fermate all'epoca 13-14 per liberare due processi (wd0.3, drop0.3); seme 1 di cur/res e variante warm con tetto 900 s.

### 3a. Residuo di fitting e base congelata (DIMOSTRATO nel giocattolo; identita' algebrica per la WLS)
`ortho.py` -> `ortho.out`: quota lineare (R2 di una regressione su z con intercetta) del residuo di fitting y - base(z):
WLS 4,793e-17; ridge (alpha 0,1) 3,560e-05. Il bersaglio residuo non contiene segnale lineare in z sul fitting:
il ramo temporale non puo' imparare a correggere l'errore di stima della base (che fuori campione costa -19,1/-16,4% nel vero).
Con la ricetta attuale il ramo affine e' invece regolarizzato dall'arresto anticipato (L0 neurale -13,6/-10,8% contro forma chiusa -19,1/-16,4%).

## 4. Ragionamento sul candidato (punto 1 del compito)

- Che cosa cambia: (i) la parte lineare non e' piu' ottimizzata con AdamW insieme al ramo profondo ma risolta in forma chiusa e fissata;
  (ii) la LN non e' piu' apprendibile; (iii) il ramo profondo vede dal primo passo un bersaglio stazionario senza la componente L.
  Non cambia: l'informazione (stesso z, stesso y), il rapporto segnale/rumore della parte temporale (a^2*Var(T non letta) ~ 0,16*0,71..0,88 = 0,11..0,14 contro sigma^2 = 1,69).
- Aritmetica del cancello (dai JSON): base in forma chiusa su y a -19,10/-16,38% (pooled -17,84%); per arrivare al 50% il ramo temporale deve
  aggiungere 69,10/66,38 punti (pooled 67,84); oggi l'intera TCN sta 43,27/35,26 punti sopra quella stessa linea (pooled 39,55):
  serve 1,60/1,88 volte (pooled 1,72) il guadagno netto attuale. Con ridge alpha=1 a 20 utenti (non archiviata; stima -11..-13% dal mondo lineare)
  servirebbero comunque circa 61-63 punti contro i 34 che la TCN ha oggi sopra L0 neurale.
- Il residuo NON e' statisticamente piu' facile: stesso segnale temporale, stesso rumore; si toglie solo la parte lineare, che non era la parte difficile
  (controlli lineari passati a 98-99,6%); in piu' l'errore della base resta congelato e invisibile sul fitting (3a).
- Sui dati veri il ritardo del ramo lineare non sembra il collo di bottiglia (sezione 2, lettura ii): L0 a LR 0,003 e' a regime entro 20-31 epoche.
  Il limite visibile e' il sovradattamento del ramo profondo dopo l'esposizione 31 (R da +0,217 a -0,226).

## 5. Conseguenze di protocollo (punto 3; lettura di provenance.py, budget_amendment.py, protocollo congelato)

- Identita': `provenance.source_manifest` (righe 236-248) copre tutto scripts/pose_temponet/*.py: toccare heads.py o train.py cambia source_sha256;
  `recipe_identity` (riga 338) e' il digest delle chiavi RECIPE_KEYS (riga 53: fra cui "heads", "new_head_common", "positive_controls"): cambia recipe_sha256;
  `execution_identity` (righe 342-348) li contiene entrambi. `verified_receipt` (righe ~612-660) esige che protocol, execution_sha256, source_sha256,
  recipe_sha256 e reviewed_code_sha della ricevuta coincidano con il piano: NESSUNA delle 11 ricevute A1 viene accettata come predecessore da un piano nuovo.
- L'eccezione A1 non si estende: `budget_amendment.validate` (righe 63-86) rifiuta ogni cambio di recipe_sha256/proposal e ogni sorgente fuori da
  CHANGED_SOURCES (6 file di risorse/provenienza; heads.py, train.py, controls.py esclusi); LEGACY copre solo 3 ricevute esatte (preflight, cpu e gpu benchmark).
- Scientificamente indipendenti dalla ricetta della testa (riusabili solo con una nuova eccezione esplicita e revisionata, oppure rieseguiti):
  preflight (integrita' input + R-1 storico), calibrazione WLS dei due fold (a=0,4, sigma=1,3, componenti, clean_ls/oracolo), purche' controls.py,
  linear.py, data.py, metrics.py e gli input restino identici. NON riusabili nemmeno nel merito: benchmark CPU/GPU come cancello risorse
  (passo di training diverso: 64.436 parametri allenabili invece di 72.904 nel giocattolo, piu' la base in forma chiusa), i 2 controlli TCN_post,
  e i controlli MLP se la MLP riceve la stessa ricetta. I 2 controlli L0 restano validi nel merito (L0 non cambia) ma legati alla vecchia identita'.
- Costo del rifacimento completo (accounting_through_controls_20261121.json): tutto cio' che e' stato speso finora e' 3,681 GPUh e 31,975 core-h
  (controlli: 3,567 GPUh e 28,533 core-h; calibrazioni 0,938 + 0,920 core-h; benchmark 0,109 GPUh). Rifare tutto da zero con identita' nuova costa quindi
  meno di 4 GPUh: il calcolo non e' l'ostacolo, lo sono definizione e review.
- Comparatori che cambiano significato: (1) primario TCN-post contro L0: oggi la TCN con g=0 coincide con L0 (test «identita' L0 col ramo g nullo»);
  col candidato coincide con la ridge in forma chiusa, non con L0 neurale: il contrasto mescola «contesto temporale» e «ridge chiusa contro affine allenata»;
  (2) MLP: se resta con affine allenata e LN apprendibile, TCN contro MLP non isola piu' il contesto; va data la stessa ricetta anche alla MLP e riqualificata;
  (4) ridge post storica in forma chiusa sugli stessi 40 utenti: la base del candidato le e' quasi identica (cambiano pesi per finestra e alpha):
  il contrasto 4 diventa «il ramo residuo aggiunge qualcosa alla propria base»; TCN-pre condivide l'architettura e cambia con essa.
- Controllo lineare: nel mondo lineare la base del candidato E' la ridge di riferimento (stessa forma chiusa e stessa CV interna):
  `linear_head_gate` (controls.py 274-286) misura il recupero costante->ridge, che diventa ~100% per costruzione; il controllo verifica solo che il ramo
  residuo non peggiori. Perde quasi tutto il valore diagnostico per quella testa.
- Regole nuove da scrivere prima dei dati: alpha della base nei fold reali e nel rifit sui 40 (quale CV, il protocollo riga 90 dice che alpha non si riscala),
  LN fissa per quali teste (riga 26: «tutte le teste iniziano con LayerNorm affine»), conteggio parametri in tabella (righe 32-38), semi (la base e' deterministica).

### 3b. Scoperta nel giocattolo: il penultimo strato ReLU del ramo profondo muore nei primi due epoch con la ricetta attuale (DIMOSTRATO nel giocattolo; SOSPETTO sui dati veri)
Script `dead.py` (rigioca i primi passi con testa, loss e ricetta vere, conta le unita' ReLU con uscita > 0 su 64 finestre di fitting, modello in eval):
- cur seme 1: penultimo strato (64->32) 31/32 vive al passo 0; 17/32 al passo 20; 9/32 al passo 29 (frazione attiva 0,002); 1/32 al passo 58 (0,000);
  deviazione standard nel tempo dell'uscita profonda 0,00000 a fine epoca 2. Nel run lungo `cur` seme 1: R == parte lineare a ogni epoca (ramo profondo nullo).
- cur seme 0: 27/32 vive al passo 29 (attiva 0,043), 10/32 al passo 58 (attiva 0,001; uscita profonda 0,00023): quasi morto, poi si riprende (il run lungo impara).
- res seme 1 (stessa inizializzazione): 28/32 al passo 29, 31/32 al passo 58, frazione attiva 0,208: sano.
- Norma del gradiente prima del clip: cur 1,3-2,0 nei primi 5 passi (clip a 1,0 attivo), poi 0,55-0,78; res <= 0,35.
Ipotesi sul meccanismo (non dimostrata): ultimo strato a zero + Adam (passi di ampiezza ~lr anche con gradienti minuscoli) + errore iniziale dominato dalla
componente lineare non ancora appresa -> deriva coerente delle unita' nascoste -> ReLU morte. Con la base congelata l'errore iniziale e' quasi solo rumore.
Coerente coi dati veri (sezione 2): a LR 0,01 la curva TCN vera resta piatta al livello lineare (-0,25..-0,15) per 150 epoche; il massimo cresce col LR fino a 0,003.
Sui dati veri NON e' dimostrato: il run vero (seme 0, LR 0,003) ha imparato (+0,217). Verificabile senza nuovi fit contando le unita' vive nei checkpoint salvati.
Censimento a fine epoca 2 (passo 58), penultimo strato, ricetta attuale, semi 0-4: vive 10/32, 1/32, 19/32, 31/32, 21/32;
frazione attiva 0,001; 0,000; 0,071; 0,133; 0,079. Collasso completo in 1 seme su 5 (seme 1), quasi completo e poi rientrato nel seme 0
(a fine epoca 8: 32/32 vive, attiva 0,181). Candidato residuale: seme 1 31/32 (0,208), seme 2 24/32 (0,062).

### 3c. Risultati intermedi del giocattolo alle 11:44 (fold 0, LR 0,003; selezione = minimo MSE rumorosa sugli esclusi fra le epoche fatte)
| variante | seme | epoche fatte | epoca scelta | R alla scelta | R massimo |
|---|---:|---:|---:|---:|---:|
| cur (ricetta attuale) | 0 | 23 (tetto 600 s) | 21 | +0,3146 | +0,3146 |
| cur | 1 | 23 | 22 | -0,1077 (ramo profondo morto: R = parte lineare) | -0,1077 |
| res (candidato GPT) | 0 | 23 (tetto 600 s) | 15 | +0,3113 | +0,3178 |
| res | 1 | 37 | 20 | +0,2899 | +0,2988 (poi scende: 0,244 a 25; 0,187 a 30; 0,118 a 35) |
| lnfix (solo LN fissa) | 0 | 13 (fermata) | - | segue cur: scarto <= 0,012 dalle epoche 8 a 13 | |
| base (solo base ridge congelata) | 0 | 13 (fermata) | - | segue res: scarto <= 0,010 dalle epoche 8 a 13 | |
| warm (affine inizializzata alla ridge, non congelata) | 0 | 36 | 21 | +0,2913 | +0,2970 |
| warm | 1 | 22 | 21 | +0,3006 (ramo profondo vivo) | +0,3006 |
| drop0.3 | 0 | 32 | 28 | +0,3189 | +0,3189 |
| wd0.3 | 0 | 32 | 28 | +0,4245 | +0,4245 |
Replica di ieri della ricetta attuale (stesso init e ordine, altra sequenza di dropout): scelta epoca 25, R +0,311 (28 epoche).

### 3d. Tabella finale del giocattolo (11:48; `summ.py` -> `toy/summary_final.txt`; INDICATIVO: latenti sintetici, un solo fold)
| variante | seme | epoche | epoca scelta (min MSE esclusi) | R alla scelta | R massimo |
|---|---:|---:|---:|---:|---:|
| cur | 0 | 23 | 21 | +0,3146 | +0,3146 |
| cur | 1 | 23 | 22 | -0,1077 (ramo profondo morto) | -0,1077 |
| cur | 2 | 22 | 20 | +0,2914 | +0,2960 |
| res | 0 | 23 | 15 | +0,3113 | +0,3178 |
| res | 1 | 40 | 20 | +0,2899 | +0,2988 (a 40 epoche +0,062) |
| lnfix | 0 | 13 | 13 | +0,1956 (cur alla 13: +0,1889) | |
| base | 0 | 13 | 12 | +0,2962 (res alla 12: +0,286) | |
| warm | 0 | 40 | 21 | +0,2913 | +0,2970 |
| warm | 1 | 40 | 21 | +0,3006 | +0,3006 |
| drop0.3 | 0 | 35 | 28 | +0,3189 | +0,3189 |
| wd0.3 | 0 | 40 | 28 | +0,4245 | +0,4271 (epoca 35; a 40: +0,383) |
| wd0.3 | 2 | 22 | 22 | +0,3713 (ancora in salita) | +0,3713 |
| wd1.0 | 0 | 22 | 22 | +0,3461 (ancora in salita) | +0,3461 |
Medie: cur semi vivi (0, 2) 0,3030 (scarto fra i due 0,0232); cur tre semi 0,1661; res (0, 1) 0,3006 (scarto 0,0214); warm (0, 1) 0,2960 (scarto 0,0093).
Differenze appaiate: res - cur seme 0: -0,0033 alla scelta, +0,0054 all'epoca comune 23 (entro il rumore); seme 1: +0,3976 (cur morto).
wd0.3 - cur: seme 0 +0,1098 all'epoca comune 23 (+0,0889 alla scelta entro 23 epoche); seme 2 +0,0753 all'epoca comune 22 (+0,0800 alla scelta): oltre la dispersione fra semi (0,02).
drop0.3 - cur seme 0: +0,0070 all'epoca comune 23: entro il rumore.
Nessuna variante arriva a 0,50 nel giocattolo (massimo 0,4271).
Limiti: un fold; cur/res seme 0 solo 23 epoche (tetto di tempo); lnfix/base 13 epoche; wd0.3 seme 1 non provato (il collasso avviene nei primi 2 epoch,
dove wd0.3 e cur coincidono: -0,761/-0,584/-0,469 contro -0,762/-0,581/-0,464); nel giocattolo il ramo lineare converge piu' lentamente che nel vero.
File grande toy/data_f0.npz cancellato a fine lavoro (rigenerabile: `toy_res.py gen 0`, 25 s).

## 6. Verdetto (punto 4)
- Candidato residuale: NON giustificato dai dati disponibili. (i) aritmetica: deve dare 1,6-1,9 volte il guadagno netto attuale sopra la base chiusa;
  (ii) giocattolo: stesso livello della ricetta attuale quando il ramo profondo e' vivo (0,311 e 0,290 contro 0,315 e 0,291), nessuna variante al 50%;
  (iii) l'errore della base resta congelato e invisibile sul fitting; (iv) la LN fissa e' un secondo fattore senza effetto misurabile (fino all'epoca 13);
  (v) costo di protocollo alto (comparatori, controllo lineare quasi tautologico, tutte le ricevute da rifare).
- Diventa giustificato solo se i diagnostici senza fit mostrano un problema di AVVIO del ramo profondo sui dati veri (unita' del penultimo strato morte o quasi)
  o se il controllo pulito mostra un cattivo fit gia' sul fitting. Anche allora c'e' una modifica piu' piccola con lo stesso effetto nel giocattolo:
  affine inizializzata alla ridge e NON congelata (warm: 0,291/0,301, ramo vivo nel seme 1), oppure la sola base congelata senza vincolo sulla LN.
- Se censimento sano e controllo pulito buono (fitting ed esclusi): il limite e' statistico/di regolarizzazione; la modifica singola meglio sostenuta dal
  giocattolo e' il weight decay attivo (0,01 -> 0,3; oggi e' quasi inerte: 2,7% di contrazione cumulata a 31 epoche).
- Allungare il training con la ricetta attuale: controindicato dalle curve vere (minimo MSE interno a ogni orizzonte; LR 0,003 da +0,217 a -0,226 fra 31 e 150).

## 7. Comandi per riprodurre
Prefisso: `cd /Users/simonebenatti/wearusfm_local/tcn_diag_20261122/review/residual_candidate; T=/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$T/scripts:$T/src:$PWD; PY=/Users/simonebenatti/.venvs/wearusfm-tok/bin/python`
- JSON archiviati: `$PY -I recompute_json.py` ; `$PY -I curves_json.py`
- Giocattolo: `$PY toy_res.py gen 0` (25 s, ricrea toy/data_f0.npz da ~300 MB: cancellarlo dopo) ; `$PY toy_res.py run 0 <cur|res|lnfix|base|warm|wd0.3|drop0.3> <seme> 40 900` ; `$PY -I summ.py`
- Ortogonalita' del residuo: `$PY ortho.py` ; censimento ReLU: `$PY dead.py <cur|res> <seme> 2`
Fine lavoro 11:50 (27 minuti).
