# Audit supporto / crop / maschere / loss / uguaglianza dei pannelli

Revisore: Claude (subagent), 22/11/2026. Materiale: copia di sola lettura del commit e8c98244
(`/Users/simonebenatti/wearusfm_local/tcn_diag_20261122/tree_e8c98244`, sotto abbreviato `TREE`).
Nessun dato vero (latenti, checkpoint) letto: solo codice, JSON archiviati e prove CPU sintetiche.

Legenda: [DIM] = dimostrato (file:riga + comando o numero ricalcolato), [IPO] = ipotesi / non concluso.

## 0. Avvio (11:2x)
- Letti per intero: data.py, metrics.py, controls.py, train.py, selection.py, workflow.py, runner.py,
  linear.py, heads.py, scripts/pose_dynamics.py; `central_mask` in scripts/pose_temporal_readout.py:195-199.

## 1. JSON archiviati: pannelli a confronto (script `panels_json.py`, `gate_json.py`; uscite `panels_json.out`, `gate_json.out`)
Comando: `/Users/simonebenatti/.venvs/wearusfm-tok/bin/python -I panels_json.py TREE/results/infra/pose_temponet_execution_20261121_A1/launcher`
(idem `gate_json.py`).

[DIM] Mondo temporale, esclusi esterni, bersaglio osservato y. `window_count` e `dynamic_target_ss` per lettore:

| pannello | lettore | window_count | dynamic_target_ss | dynamic_sse | E_dyn |
|---|---|---|---|---|---|
| fold0 | clean_ls (calibration_0) | 934 | 3206279.9567595376 | 2277841.1587226163 | 0.7104311505676322 |
| fold0 | oracolo (calibration_0) | 934 | 3206279.9567595376 | 2095145.510278007 | 0.6534505840205822 |
| fold0 | TCN_post (selection) | 934 | 3206279.9567595376 | 2233678.0429985886 | 0.6966572080798834 |
| fold0 | MLP / L0 (selection) | 934 | 3206279.9567595376 | 2300203.1408042163 / 2302602.081106978 | 0.7174055827392385 / 0.7181537832504584 |
| fold1 | clean_ls (calibration_1) | 920 | 2698827.390194636 | 2023905.705487826 | 0.7499203961102027 |
| fold1 | oracolo (calibration_1) | 920 | 2698827.390194636 | 1865167.4290396296 | 0.6911028974346952 |
| fold1 | TCN_post (selection) | 920 | 2698827.3901946363 | 1993943.1408579925 | 0.7388183283237657 |
| pooled | clean_ls (somma) | 1854 | 5905107.346954174 | 4301746.864210442 | 0.7284790286546209 |
| pooled | oracolo (somma) | 1854 | 5905107.346954174 | 3960312.9393176366 | 0.6706589239838878 |
| pooled | TCN_post (archiviato) | 1854 | 5905107.346954174 | 4227621.183856581 | 0.7159262203822878 |

- fold0: i sei valori di `dynamic_target_ss` (clean_ls, costante, oracolo, L0, MLP, TCN) sono bit-identici (hex 0x1.87643fa7718b6p+21).
- fold1: riferimenti chiusi 0x1.49725b1f1e5d8p+21, teste 0x1.49725b1f1e5d9p+21: 1 ulp (4,66e-10 assoluto, 1,7e-16 relativo).
  Spiegazione [DIM nel codice]: i riferimenti chiusi chiamano `selection_statistics` una volta su tutti gli esclusi
  (controls.py:579, blocchi da 64 in metrics.evaluate), le teste per blocchi da 32 poi `pool_statistics`
  (workflow.py:248-260): ordine di somma diverso, stesso insieme.
- Mondo lineare: stessa situazione (934/920; denominatori 3166775.6865351712 e 2669591.212421049, differenza 1 ulp).
- 934 + 920 = 1854 = 1907 - 53; nei manifest `folds`: fitting 961 (41 escluse) / 946 (12 escluse): 920 e 934 finestre di fitting.
- Soglie dinamiche: calibration_0 `temporal_threshold` 0.5437424309221641, calibration_1 0.5374366720436082;
  `bundle.thresholds` di selection_temporal_{L0,MLP,TCN_post} = [0.5437424309221641, 0.5374366720436082] (uguali esatti).
  Lineare: 0.4597418919608825 / 0.4297722589800256 (uguali esatti nei tre selection_linear_*).
- Identita': `support_sha256` 82bc28491af0... , `window_ids_sha256` b19c10650d29..., `population_sha256` 09a2d9a72e66...,
  `feature_array_sha256` 397527e82d82..., `artifacts_sha256` 1e1d63c6371e... identici nelle sei selezioni;
  `evaluation_identity` 8fcf86092541... per le tre temporali, 0b9f9a1aadec... per le tre lineari;
  `target_sha256`/`clean_target_sha256` identici fra L0, MLP e TCN dello stesso mondo.
  NOTA: i calibration_*.json non contengono un hash del bersaglio ne' del supporto: l'uguaglianza riferimenti/teste
  e' provata dai denominatori (a 1 ulp) e dai conteggi, non da un hash comune.

[DIM] Cancello ricalcolato dai soli accumulatori: R = 0.2417305288878684 / 0.1887545039561508 / 0.2171010990695587
(fold0 / fold1 / pooled); nel JSON del cancello 0.24173052888786836 / 0.1887545039561508 / 0.21710109906955868.
`temporal_records` del cancello = E_dyn ricalcolati (uguali esatti). MLP: -0.1224 / -0.1065 / -0.1150; L0: -0.1355 / -0.1082 / -0.1228.

[DIM] Pooled = somma degli accumulatori: `selected_outer_statistics.pooled` e' esattamente fold0+fold1 nei quattro campi
(TCN, MLP, L0). Media dei due R di fold = 0.2152425164220096 (non usata); il cancello riporta 0.2171010990695587 (rapporto di somme).

[DIM] Identita' y / y* sui numeri archiviati (termine incrociato implicito 2<e,n> = SSE_y* + SSE_oracolo - SSE_y):
fold0 +2915.776128 (1,596% del margine SSE clean_ls-oracolo 182695.648445); fold1 -817.283324 (-0,515% di 158738.276448);
pooled +2098.492803 (0,615% di 341433.924893). E_dyn della TCN contro y*: 0.127856 / 0.154431 / 0.139236.

[DIM] Passi per epoca dalle curve: [29, 30] = ceil(920/32), ceil(934/32): il training usa 920 e 934 finestre.
`selected_fitting_diagnostics`: window_count 920 / 934; MSE sul fitting 1.6880667994 / 1.7095008793; E_dyn 0.6618730473 / 0.7105825844.

## 2. Lunghezza delle finestre nel train vero (script `ktv_hash.py`)
[DIM] `arrays_hash(np.ones((1907,160), bool))` = 0065b939fb665ae3ff7cd475d9090a90834ca2e3fea311bbbf8c1fbcd000246f
= `key_time_valid_sha256` archiviato nella ricevuta `TCNH-preflight-20261120.json` (train.input_verification).
Poiche' `data.central_support` (data.py:51-54) pretende `key_time_valid == (t < n_patches)`, tutte le 1.907 finestre
hanno 160 patch reali: nel train vero il supporto centrale e' esattamente `mask[:, 6:154]`, il ramo "finestra corta"
non e' mai attivo e il generatore (che legge i latenti non azzerati, controls.py:166) non vede contesto imbottito.
[DIM] Stessa ricevuta: `central_mask_sha256` (pose_temponet) == `targets.central_mask` (storico) = ec1ee7176e9c...;
`n_windows` 1907, `supported_windows` 1854.

## 3. Prove CPU con le funzioni vere (script `probe_support.py`, uscita `probe_support.out`, `probe_support.json`)
Comando: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=TREE/scripts:TREE/src
/Users/simonebenatti/.venvs/wearusfm-tok/bin/python probe_support.py probe_support.json` (16 s). Esito: 65 controlli, 0 falliti.
Dati: cache sintetica 8 utenti x 6 finestre (48), maschere con finestre senza posa, posa solo ai bordi tagliati,
buchi temporali, giunto con un solo campione centrale, giunto valido solo ai patch 6 e 153, due finestre da 120 patch.
Unica alterazione in memoria: stato del cancello di riferimento forzato a "diagnostic" per costruire il bundle giocattolo
(workflow.py:174); stato vero del mondo giocattolo: control_non_diagnostic. Nessun file del progetto toccato.

Prima esecuzione: 3 controlli falliti, tutti per errori del MIO banco di prova (non del codice):
 (a) A4: non avevo previsto che `workflow._training_set` azzera le feature oltre la lunghezza reale (finestre da 120);
 (b) A10 fitting fold0: selezione dinamica sul fitting con numero dispari di finestre finite: la finestra mediana ha
     pose_var == soglia esatta e la mia varianza ricalcolata differiva all'ultimo bit. Corretto usando la pose_var vera
     dopo averla confrontata (rel 1e-9) con quella indipendente.

[DIM] Prova A (feature/bersagli codificati, testa-sonda identita'; train.Trajectory VERO con recipe di test):
- A1: `central_support` = supporto indipendente S[w,t,j] = mask & 6<=t<=153 & t+6<=n-1 (126.646 elementi, 44 finestre su 48).
- A2/A3: il training usa tutte e sole le finestre di fitting con supporto centrale (22 su 24 per fold), una volta per epoca.
- A4/A5/A6: a `metrics.per_window_mse` arrivano predizione = uscita ai patch 6..153, bersaglio ai patch 6..153, supporto = S.
- A7/A11: loss sull'intero fitting = MSE per finestra indipendente = `mse_sum/window_count` di `_score_model` sul fitting
  (2.000020742 contro 2.000020726: differenza FP32).
- A8/A10: `_score_model` passa a `metrics._cropped` predizione/bersaglio/supporto allineati; i quattro accumulatori
  coincidono con il ricalcolo indipendente (rel 1e-9) su held e fitting dei due fold.
- A12/A13: perturbare la predizione fuori dal supporto o nei 6+6 patch di bordo non cambia nessun accumulatore (uguaglianza esatta).
- A14: predizione = bersaglio -> 0; bersaglio spostato di una patch -> errore > 0.

[DIM] Prova B (mondo di controllo: `generate_components`, `evaluate_outer_references`, `_prepare_control_folds`,
`_score_model(include_clean=True)` veri; "TCN" = make_head("TCN_post") con pesi casuali, non addestrata):
- B0: L, T, epsilon hanno forma (N,160,20).
- B1: per clean_ls, oracolo, costante e testa `metrics.evaluate` riceve lo stesso pannello bit per bit: bersaglio y,
  maschera, pose_var, soglia, elenco delle finestre, `support_sha256`, numero di finestre dinamiche, celle n>=2, elementi
  (fold0: 24 finestre, 11 dinamiche, 199 celle, 29.319 elementi; fold1: 24, 13, 259, 37.343).
- B2: il pannello pulito (y*) ha stesso supporto, stessa pose_var, stessa soglia e stesse celle dinamiche del pannello y.
- B3: soglia = mediana sul fitting della varianza del bersaglio pulito sulla finestra intera; uguale (==) a quella del
  riferimento chiuso (workflow.py:184).
- B4-B7: accumulatori veri di testa (contro y e y*), oracolo e clean_ls = ricalcolo indipendente.
- B8: `dynamic_target_ss` e `window_count` uguali fra i quattro lettori.
- B9-B11: identita' SSE_y = SSE_y* + SSE_oracolo - 2<e,n>: residuo assoluto 0.0 (fold0) e 2,9e-11 (fold1) su SSE ~1e5;
  vale anche per la MSE per finestra (residuo 1,4e-14) e per clean_ls.
- B12/B13: il training del controllo usa y osservato del fold, crop 6..153, supporto S, sole finestre supportate;
  loss sul fitting = MSE della metrica sul fitting (3.833075762 contro 3.833075523).
- P1/P2: pooled = rapporto delle somme sia in `combine_reference_folds` sia in `selection.pool_scores`.

Proprieta' osservata (non difetto): `evaluate` usa `pv >= soglia` (metrics.py:92). Con un numero dispari di finestre
finite nel fitting la finestra mediana e' dinamica sul pannello di fitting. Nel train vero i conteggi sono 920 e 934
(pari) SE le 41+12 finestre senza supporto centrale hanno anche pose_var non finita (non verificabile senza dati).
Sugli esclusi la soglia viene dall'altra meta': nessun pareggio sistematico. Uguale per tutti i lettori.

## 4. Catena del supporto nel codice (file:riga, TREE/scripts/pose_temponet salvo indicazione)
| passo | dove | che cosa |
|---|---|---|
| supporto centrale | data.py:43-55 -> scripts/pose_temporal_readout.py:195-199 | `mask[:, 6:-6] & (t+6 < n_patches)`, t = 6..153; indice k del supporto = patch k+6 |
| bundle | workflow.py:147 | `support = central_support(mask, lengths, key_time_valid)`, uno per tutta la cache |
| training: righe | train.py:169-173 | solo finestre con almeno un elemento supportato (`support.flatten(1).any(1)`) |
| training: batch | workflow.py:225-239, train.py:195-204 | bersaglio osservato del fold `[:, 6:154]`, supporto `bundle.support[indices]` |
| training: loss | train.py:283-287, metrics.py:24-39 | `model(x)[:, 6:154]`; media sulle finestre di SSE_supportato / q_finestra |
| selezione: pannello | workflow.py:204-212, 242-261 | `model(x)[:, 6:-6]`; bersaglio osservato, maschera intera, pose_var del pulito, soglia del fold |
| metrica | metrics.py:67-75, 78-158, 161-166 | `_cropped`: stesso `central_support`, bersaglio `[:, 6:-6]`; MSE per finestra su q>0; E_dyn su finestre con pv >= soglia e celle n>=2 |
| E_dyn | scripts/pose_dynamics.py:88-127, 134-146 | `centered` = var(p)+var(y)-2cov sulla cella finestra/giunto; somme pesate per n: Σ SSE centrato / Σ SS centrato del bersaglio |
| selettore dinamico | metrics.py:42-64; workflow.py:183-188; controls.py:545, 566 | varianza del bersaglio PULITO sulla finestra INTERA (160); soglia = mediana sul fitting |
| riferimenti chiusi | controls.py:553-580 | stesso `_metric_panel` e stessa soglia per costante, oracolo (`clean[held, 6:-6]`), clean_ls (WLS alpha=0 su y*, pesi m/q) |
| uguaglianza soglia | workflow.py:183-185 | `threshold == artifact[...][world+"_threshold"]` (uguaglianza esatta, altrimenti errore) |
| uguaglianza partizione | workflow.py:178-179 | utenti di fitting/held del bundle = partizione dell'artefatto di calibrazione |
| componenti | controls.py:161-162, 185, 89-113, 193-198 | L, T, epsilon su 160 patch; normalizzazione sul crop del fitting applicata a tutti i 160; y* = L + a T, y = y* + sigma eps |
| pooled cancello | controls.py:415-423, 653-660; workflow.py:316, 368-376 | somma dei quattro accumulatori, poi rapporto |
| pooled selezione | selection.py:65-81; train.py:437 | `fsum(mse_sum)/sum(window_count)`, `fsum(dynamic_sse)/fsum(dynamic_target_ss)` |
| R | controls.py:251-260, 302-318 | (E_cleanLS - E_testa)/(E_cleanLS - E_oracolo), per fold0, fold1, pooled |

Risposte: (1) si': stesso supporto per loss, selezione e cancello; le finestre senza supporto sono escluse nei tre luoghi
(train.py:169-173; q>0 e n>=2 nella metrica; peso zero nel WLS, linear.py:324-326); nessuno sfasamento; L, T, eps su 160 patch
ritagliati con `[6:-6]` ovunque. (2) si': stesso pannello per clean_ls, oracolo e TCN nei tre pannelli. (4) pooled = somma degli
accumulatori in entrambi i punti.

## 5. Altri numeri ricavati dai JSON (script `crosscheck_json.py`, `fit_margin.py`, `derived.py`)
[DIM] Pannello di FITTING: `dynamic_target_ss` identico fra riferimenti chiusi (linear_diagnostics L0/L13, mondo temporale)
e diagnostici di fitting delle teste: fold0 2957984.5843438697 (920 finestre), fold1 2878704.615363543 (934).
[DIM] Terza via sugli esclusi (`linear_diagnostics.*.worlds.temporal.held`): 3206279.9567595376 / 2698827.390194636, 934 / 920.
[DIM] Coerenza fra percorsi: SS(y) - SS(y*) - SSE_oracolo = 2<y*_c,n_c> = 4822.021469 (0,1504% di SS(y)) fold0; 5079.435536 (0,1882%) fold1.
[DIM] MSE dell'oracolo su tutti gli esclusi: 1.691773 / 1.692751 (sigma^2 = 1,69).
[DERIVATO, descrittivo] oracolo sul fitting ricavato da `temporal_margin_recovery_fitting` di L13: E_dyn 0.654332 / 0.693246;
clean_ls sul fitting 0.708654 / 0.749626. Quota del margine IN CAMPIONE (non e' il cancello, include rumore appreso):
TCN 0.8612 / 0.6925; MLP 0.3777 / 0.2745; L0 -0.0025 / 0.0200. Sugli esclusi: TCN 0.2417 / 0.1888; MLP -0.1224 / -0.1065.
[DERIVATO] pooled esclusi, stesso pannello (1854 finestre, SS 5905107.347):
 clean_ls E_dyn 0.728479 MSE 1.809351; oracolo 0.670659 / 1.692258; costante MSE 2.892231;
 TCN R_Edyn +0.2171, R_MSE +0.0786; MLP -0.1150 / -0.2177; testa L0 -0.1228 / -0.2289;
 WLS L0 chiuso su y rumoroso (alpha=0) -0.1784 / -0.3111; WLS L13 chiuso su y rumoroso -0.7808 / -1.1308.
 TCN rispetto alla testa L0 allenata sullo stesso y rumoroso: 0.3027 della distanza L0->oracolo su E_dyn, 0.2502 su MSE.
 Per la soglia del 50% (E_dyn <= 0.699569) partendo da una base lineare tipo testa L0 serve 0.5547 della distanza L0->oracolo.
[STIMA, non conteggio] elementi dinamici da SSE_oracolo/(1,69*0,9457): 1.310.914 (442,9 finestre piene su 934) e 1.167.018 (394,3 su 920).
 Il numero di finestre dinamiche non e' archiviato (`selection_statistics` lo scarta, metrics.py:161-166).

## 6. Test esistenti
`pytest -q -p no:cacheprovider` su test_pose_temponet_{metrics,data,selection,controls,workflow}.py: 51 passati in 14,81 s.
Nessun file creato o modificato sotto TREE (find -newer: 0).

## 7. Osservazioni (nessuna e' un difetto dimostrato)
- N1: selezione e cancello leggono le stesse finestre escluse (workflow.py:298 e 312): niente insieme di prova separato nel
  controllo. Il verso favorisce la TCN; termine incrociato 2<e,n> pooled +2098.49 = 0,615% del margine SSE. Non spiega il cancello.
- N2: i calibration_*.json non portano hash di supporto/bersaglio ne' il numero di finestre dinamiche.
- N3: selettore dinamico sulla finestra intera (160), metrica sul crop (148): scelta dichiarata, uguale per tutti.
- N4: la loss (MSE per finestra, tutte le finestre, non centrata) e la metrica del cancello (errore centrato, sole finestre
  dinamiche) condividono il supporto ma non i pesi; ultimo batch dell'epoca: 24 finestre (fold0) e 6 (fold1): 920 = 28*32+24, 934 = 29*32+6.
- N5: `pv >= soglia`: la finestra mediana del fitting e' dinamica se le finestre finite sono dispari; solo sul pannello di fitting.
- NON CONCLUSO: conteggio vero delle finestre dinamiche per pannello; se le 53 finestre senza supporto centrale abbiano anche
  pose_var non finita; uguaglianza bit per bit dei bersagli fra calibrazione e controlli sui dati veri (provata solo dai denominatori).

## 8. Chiusura (11:4x)
Verdetto del perimetro: nessun difetto dimostrato in supporto, crop, maschere, loss, uguaglianza dei pannelli e pooled.
Tre note (N1, N2, N4) riportate nell'esito strutturato con gravita' "nota"; nessuna spiega il cancello negativo.
Raccomandazioni operative per il passo 1 della proposta di GPT (nessun nuovo fit):
 a) al checkpoint scelto leggere anche il pannello di FITTING contro y* (`_score_model(..., include_clean=True)` esiste gia';
    in produzione e' stato usato solo sugli esclusi, workflow.py:312): separa "funzione pulita non appresa in campione" da
    "appresa ma non generalizza";
 b) scrivere per ogni lettore e pannello numero di finestre dinamiche, celle n>=2, elementi e `support_sha256`
    (`metrics.evaluate` li calcola, `selection_statistics` li scarta);
 c) calcolare con la funzione vera clean_ls contro y* (qui solo stimato: 0.176461 pooled con termine incrociato nullo);
 d) pooled sempre come rapporto delle somme.
File di questa cartella: notes.md, panels_json.py/.out, gate_json.py/.out, crosscheck_json.py/.out, fit_margin.py/.out,
derived.py/.out, ktv_hash.py, probe_support.py/.out/.json, pytest_scope.out, json_shape.py.
