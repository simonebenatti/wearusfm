# Audit catena generazione -> bersaglio (target_chain) — note progressive
Inizio 22/11/2026 11:23. Materiale: tree_e8c98244 (sola lettura). Nessun dato vero sul Mac.

## 1. Lettura del codice (file:riga, tree_e8c98244/scripts/pose_temponet/)
- controls.py:141-159 semi per fold: base_seed = 20261118 + 1000*fold; L seme base, T seme base+1, rumore base+2 (riga 185).
- controls.py:166-167 generatore: FP16 -> np.float32 -> F.layer_norm(eps=1e-5) su x GREZZO (nessun azzeramento delle patch oltre n_patches).
- heads.py:36,65-66 LayerNorm della testa: eps 1e-5, weight=1, bias=0 all'inizio -> identica alla LN del generatore all'inizializzazione; poi apprendibile (heads.py:109-117: niente weight decay su LN/bias).
- workflow.py:228-234 e data.py:122-136: la testa vede latenti con patch non disponibili AZZERATE. Il generatore no.
  - pose_temporal_readout.py:195-199 central_mask: supporto = mask[:,6:-6] & (t+6 < length), t in [6,153]. Quindi il campo ricettivo [t-6,t+6] di ogni posizione supportata sta tutto nel contesto reale e dentro [0,159]: la differenza generatore/testa sulle patch di padding NON tocca le posizioni supportate di T. Tocca solo pose_var (dynamic_variance usa la maschera intera, metrics.py:42-57) in modo identico per tutti i lettori.
- controls.py:89-113 normalize_component: media/scala per giunto pesate per finestra (mask/q) sul SOLO fitting esterno, sul crop [6:154] col supporto; applicate a tutte le 160 posizioni.
- controls.py:45-59 window_noise: AR(1) stazionario (out[0]=innov[0], poi rho*prev+sqrt(1-rho^2)*innov), chiave sha256(canonical([seed, meta, j])) -> indipendente per finestra e giunto; generato su tutte le 160 posizioni.
- controls.py:193-198 synthetic_target: clean = L + a*T; observed = clean + sigma*eps.
- Catena (ii) job dei controlli: runner.py:98-102 carica calibration.pt (components inclusi) -> runner.py:308 workflow.prepare_control_folds -> workflow.py:181 synthetic_target(artifact["components"], a, sigma) -> targets=y osservato (workflow.py:186) -> workflow.py:237 TensorTrainingSet(target=bundle.targets[fold][indices]) -> train.py:198-200 cast FP32 e crop [6:154] -> train.py:287 per_window_mse(pred, y, support). Nessuna ulteriore normalizzazione. Il rumore e' lo STESSO array salvato dalla calibrazione (non rigenerato).
- Catena (iii) riferimenti: controls.py:565-579 stesso synthetic_target, soglia = training_threshold(clean[fit], mask[fit]), pannello target=observed[held], pose_var da clean; workflow.py:183-185 pretende soglia identica a quella dell'artefatto; workflow.py:253 la testa e' valutata con lo stesso selection_statistics sullo stesso y osservato.

## 2. calibration_0.json / calibration_1.json (numeri ricalcolati)
- Griglia: a in {0, 0.2, ..., 3.0} (16 valori) x sigma in {0, 0.05, ..., 6.0} (121) = 1936 valutazioni per fold.
- Ammissibili temporali: SOLO a=0.4; fold0 7 coppie (sigma 1.20..1.50), fold1 6 coppie (sigma 1.20..1.45). Scelto a=0.4 sigma=1.3 in entrambi.
- Inner (10 utenti -> 10 utenti, due meta'): fold0 E_cleanLS 0.718123, E_oracle 0.661681, gain 0.078597; fold1 0.752040, 0.693201, gain 0.078238. E_cal 0.728848 / 0.761515 contro E_real 0.747134 / 0.775788 (|diff| 0.0183 / 0.0143 <= 0.05).
- Outer (20 -> 20): fold0 clean_ls 0.710431, oracle 0.653451, gain 0.080206 (margine assoluto 0.056981); fold1 0.749920, 0.691103, gain 0.078432 (margine 0.058817). Coerente con l'inner (0.0786 / 0.0782): scarto +0.0016 / +0.0002.
- Finestre: fold0 fitting 961 (inner 500+461), supportate 920; fold1 fitting 946 (455+491), supportate 934. Rango WLS 384 su 385 colonne (LN ha somma nulla: atteso), un solo gruppo di maschere da 20 giunti.
- linear_diagnostics (WLS alpha=0 allenato sul bersaglio RUMOROSO del mondo temporale, valutato sugli esclusi esterni, E_dyn):
  - L0: fold0 0.721315 -> R = -19.10%; fold1 0.759555 -> R = -16.38% (R rispetto a clean_ls/oracolo).
  - L13: fold0 0.759182 -> R = -85.56%; fold1 0.790777 -> R = -69.46%.
  - T_unread_fraction_fitting (L0): 0.7117 / 0.8788; L13: 0.5163 / 0.7057. temporal_margin_recovery_fitting di L13 allenato sul PULITO e misurato sul fitting: 27.09% / 19.93%.

## 3. Prova CPU con le funzioni vere (chain_proof.py -> chain_proof_output.json), latenti sintetici 48x160x384 FP16, 6 utenti finti
Comando: cd review/target_chain && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<tree>/scripts:<tree>/src ~/.venvs/wearusfm-tok/bin/python chain_proof.py (4,5 s)
- Rumore (400 finestre x 160 x 20): media 0.0023, varianza 0.99758; var a t=0 0.9711, t=1 0.9589, t=159 0.9756 (8000 campioni l'una, errore standard ~0.016: stazionario dal primo campione); autocorrelazione lag1 0.77669 (atteso 0.77880), lag2 0.60497 (0.60653), lag4 0.36719 (0.36788), lag8 0.13564 (0.13534); correlazione fra giunti 0.0081, fra finestre 0.0018; deterministico; dipende da seme+metadati+giunto, non dalla posizione nella lista ne' dall'ordine delle chiavi.
- Normalizzazione di L e T sul fitting (pesi per finestra, supporto): media pesata max |.| 7e-14, varianza pesata in [1-5e-15, 1+6e-15] in entrambi i fold. Perturbare i latenti degli esclusi NON cambia L,T sul fitting (0.0), ne' la normalizzazione, ne' gli hash del generatore.
- y - y* - sigma*eps: max |.| 8.9e-16; y* - (L + a*T): 0.0. epsilon == window_noise(base+2).
- LN del generatore contro LN della testa all'inizializzazione, stessi FP32: differenza 0.0 (bit per bit su CPU).
- block_windows 4 contro 16 nel generatore: L,T differiscono al massimo di 1.9e-5 / 2.4e-5 (numerica FP32 dipendente dal lotto). Non e' un difetto: T e L sono salvati in calibration.pt e riusati; la testa pero' vede lotti da 32.
- Allievo TCN_post con i pesi veri (costruzione di GPT) valutato con D.view_inputs (padding azzerato): errore su y* nelle posizioni supportate max 4.46e-6 (rmse relativo 4.76e-7), anche nelle 2 finestre corte con spazzatura nel padding (4.19e-6); nelle stesse finestre, posizioni reali ma NON supportate t=144..149: differenza fino a 1.68 (il generatore legge il padding grezzo, la testa zeri) -> confinato fuori dal supporto, come previsto dalla lettura di central_mask.
- Coerenza dei tre luoghi (stesso predittore, due percorsi):
  - oracolo: evaluate_outer_references E_dyn 0.5964697306 = allievo-oracolo valutato da workflow._score_model 0.5964697344 (diff 3.8e-9); statistiche in stile calibrazione (_quadratic_panel + _pose_variance_gram + score_aggregate) identiche a selection_statistics (dynamic_sse 87138.67523263 in entrambi, soglia 1.026170007018608 identica, stesse finestre dinamiche).
  - clean_ls: riferimento 0.6625475397 = testa L0 con i coefficienti clean_ls 0.6625475389.
  - bersaglio del training (workflow._training_set -> TensorTrainingSet.batch) == y osservato[fit, 6:154] in FP32, maschera == supporto, x == view_inputs (padding azzerato), finestre senza supporto escluse.
ESITO: nessun difetto trovato nella coerenza (i) calibrazione, (ii) training delle teste, (iii) riferimenti/cancello.

## 4. Coerenza sugli artefatti VERI archiviati (json_checks.py -> json_checks_output.json; solo JSON)
Comando: ~/.venvs/wearusfm-tok/bin/python json_checks.py <tree>/results/infra/pose_temponet_execution_20261121_A1/launcher
- Soglie dinamiche dei bundle delle 3 teste (mondo temporale) == temporal_threshold delle due calibrazioni, bit per bit (0.5437424309221641, 0.5374366720436082).
- dynamic_target_ss delle teste sugli esclusi == quello dei riferimenti: differenza relativa 0 (fold0) e 1.7e-16 (fold1); sul fitting (selected_fitting_diagnostics contro temporal_margin_clean_fit_statistics) differenza 0. Stesso y, stesso supporto, stesse finestre dinamiche nel job dei controlli e nei riferimenti. window_count 934/920 uguali. Stesso target_sha256 per L0, MLP, TCN_post.
- Varianza del rumore sui dati veri: MSE dell'oracolo sugli esclusi 1.691773 (fold0) e 1.692751 (fold1) contro sigma^2 = 1.69; mondo lineare 1.824412 contro 1.8225 (sigma 1.35) e 1.963190 contro 1.96 (sigma 1.4).
- Nessuna fuga del rumore degli esclusi nel training: correlazione fra errore pulito della testa e rumore degli esclusi (da SSE_noisy = SSE_clean + SSE_oracle - 2 cross) |corr| <= 0.0027 per L0, MLP, TCN nei due fold.
- Recuperi ricalcolati: TCN 24.1731% / 18.8755% / 21.7101%; L0 neurale -13.5531 / -10.8188 / -12.2819; MLP -12.2400 / -10.6539 / -11.5026. Pooled: clean_ls 0.728479, oracolo 0.670659, soglia 50% 0.699569, gain 7.937%.
- Curve TCN: minimo E_dyn di tutte le curve 0.715826 (LR 0.003, esposizione 27) -> 21.88%. A fine orizzonte R = +0.57% (LR 1e-4, 300 epoche), -5.93% (3e-4), -39.81% (1e-3), -22.62% (3e-3), -15.79% (1e-2): allungare lo stesso training sul bersaglio rumoroso peggiora.
- Fitting contro esclusi al checkpoint scelto: guadagno della TCN su clean_ls in MSE 0.094909 (fitting) contro 0.006915 (esclusi) = 7.3% trasferito (fold0); 0.084627 contro 0.011531 = 13.6% (fold1); in E_dyn 0.046781 -> 0.013774 (29.4%) e 0.039043 -> 0.011102 (28.4%). MSE di fitting della TCN 1.688067 / 1.709501, cioe' al livello del pavimento di rumore 1.69.
- Scala pulita (derivata, trascura i termini incrociati): E_dyn pulito di clean_ls sugli esclusi ~0.165139 / ~0.191579; TCN 0.127856 / 0.154431 -> quota del residuo pulito chiusa 22.58% / 19.39%.

## 5. Punti minori dimostrati
- D2: hidden_active_fraction_fitting, normalization (media/scala grezze di L e T) e generator_hashes stanno solo in components (controls.py:186-190) che runner.py:295-296 toglie dal JSON; grep -rl hidden_active su results/ e docs/reviews/: nessun file. Il protocollo (riga 96) chiede la frazione di unita' attive prima dei fit neurali: sul Mac non e' verificabile.
- D3: il controllo sul bersaglio pulito non passa dal percorso congelato: workflow.py:237 allena sempre su bundle.targets (rumoroso); con sigma=0 controls.temporal_reference_gate da' gain 1.0 -> control_non_diagnostic (provato: C.temporal_reference_gate({... 'oracle': 0.0})) e workflow.py:173-174 rifiuta. Serve codice nuovo -> nuova identita' di sorgente, piano e ricevute.
- pytest tests/cpu/test_pose_temponet_controls.py dalla cartella note: 12 passed in 6.03s.

## 6. Che cosa la calibrazione garantisce e che cosa no
- Garantisce (inner 10->10 utenti): |E_cal - E_real| <= 0.05 e (E_cleanLS - E_oracle)/E_cleanLS in [0.06, 0.10], il piu' vicino a 0.08. Verificato poi sugli esclusi esterni dal cancello dei riferimenti (8.02% / 7.84% / 7.94%).
- NON garantisce che il margine sia apprendibile da 920/934 finestre rumorose: clean_ls e' stimato su y* (senza rumore), l'oracolo e' y*. Nessun riferimento del mondo temporale e' allenato sul rumoroso con capacita' temporale non lineare. Il protocollo lo dichiara (riga 88: "Non e' un bound campionario garantito sul noisy held").
- Ipotesi (non dimostrata): la soglia del 50% potrebbe essere vicina al tetto statistico. Indizi misurati: L13 WLS sul rumoroso perde 85.6% / 69.5% del margine; il guadagno di fitting della TCN si trasferisce al 7-14% (MSE).
Fine 11:36.

## 7. Luogo (i) chiuso end-to-end (calibration_inner_proof.py -> calibration_inner_proof_output.json, 21 s)
controls.build_positive_control_fold (funzione vera, 1936 valutazioni, 8 utenti finti, 48 finestre) contro calcolo diretto
(WLS su y* dell'inner-fit, predict_mixture sull'inner-held, metrics.selection_statistics su y osservato, pool dei due inner):
riga a=0.4 sigma=1.3 -> |diff| E_cleanLS 1.1e-16 / 2.2e-16, E_oracle 1.1e-16 / 1.1e-16, E_cal (ridge alpha della riga) 2.2e-16 / 1.1e-16 (fold0 / fold1).
Quindi il margine della calibrazione e' misurato sullo STESSO bersaglio osservato, con la stessa definizione di E_dyn e la stessa regola delle finestre dinamiche (pose_var da y*, soglia dal fitting) usate da riferimenti esterni e teste.
(Sui latenti sintetici lo stato e' control_non_calibrated: atteso, non e' una misura dei dati veri.)
Fine 11:37.
