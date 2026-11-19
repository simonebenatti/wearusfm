# Verifica scettica P2-N3 (scettico n. 1)

Rilievo: provenienza verificata sul contenuto dei file, non sul produttore (NOTA).
Copia: 438c8d65. Lavoro con comandi miei in tmp/vp2n3s1 (nessuna scrittura nella copia git).

## Letture statiche (438c8d65)
- contrastive_prepare.py:277-280: summary.code.sha solo formato (40 hex); nessun lettore lo confronta (grep run_code/WEARUSFM_CODE_SHA: solo prepare.py:362 lo copia nella ricevuta; run.py:1825 lo scrive da env; lib_code_dir.sh:63 lo esporta dopo la verifica del checkout).
- contrastive_prepare.py:213-219 _reader_provenance: scope dichiarato "reader checkout only; frozen run SHA/source equality remains a release check" -> scelta di progetto esplicita, non svista.
- contrastive_cuda_check.py:242-246: solo step 512, ottimizzatore vuoto, hash iniziale, teacher=student; nessun confronto con config/ricevuta. Pin del file via --calibration-checkpoint-sha256 dal job (contrastive_job.py:108-109).
- Readout: _cuda_evidence (268-278) lega cuda.initial_sha256 a cohort.initial_state_sha256; 344-345 lega il file del checkpoint al run_dir; _initial_from_calibration 259-265 lega il config incorporato a config.json.
- Diff e388a1de..0efdc29c: nessuna di queste righe cambia (cuda_check non nel diff; prepare.py cambia solo PLAN, with_legacy_target_windows, 100/125).

## Riproduzione indipendente (tmp/vp2n3s1/probe.py, output tmp/vp2n3s1/probe.out.json)
Copia del run storico calibration_completed_20261117/run sotto tmp, poi:
- A0 run storico intatto: rifiutato ("calibration run config differs").
- A1 solo config 100/125: rifiutato dalla guardia compat (inventario sorgenti).
- A2 solo source_sha256 corrente: rifiutato (config).
- A3 config.json + summary.contrastive.{config, provenance.source_sha256} riscritti: PASSED_FOR_REVIEW, lambda 0,11942686/0,33044627 (= 2g riprodotto). run_code.sha resta 4fc51525 (storico).
- A4 come A3 ma code.sha = 40 zeri: ancora PASSED_FOR_REVIEW -> code.sha solo formato, confermato.
- C1-C3: authenticate_calibration, calibrated_configs (100/125/v3 con lambda vecchie) e validate_cohort su una coorte generata dalla ricevuta falsa: tutti accettati.
- D1: _initial_from_calibration con checkpoint che incorpora il config storico 25/50 e config.json falsificato 100/125: rifiutato. D2 (controllo coerente): accettato.
- E1-E3: sezione model e seed identici fra calibrazione storica e nuova; differiscono solo loader.rms_window_ms, loader.ms_fast_window_ms, target_version. Con run.py:1383-1390 (seed, poi WearUsFM(cfg.model)) lo stato iniziale nuovo coincide con quello storico ba8a8657.

## Esito parziale
Fatti del rilievo riprodotti. Valutazione del difetto in corso.

## Motivi per cui non e' un difetto della consegna
1. Unico percorso di accettazione = falsificazione deliberata di due file (tre campi). Nessuno script riscrive provenance/config di un run completato (grep in scripts: nessuna scrittura di summary/config dei run; contrastive_audit.py:183 scrive solo il proprio report al momento della produzione). Il rilievo stesso: «Non e' un'eredita' automatica».
2. Il lettore basato sul contenuto e' una scelta dichiarata: contrastive_prepare.py:218 scope "reader checkout only; frozen run SHA/source equality remains a release check"; la ricevuta e' PASSED_FOR_REVIEW con training_approved=false.
3. La correzione proposta e' gia' dichiarata o inefficace: (a) fresh_root e non riuso sono nell'emendamento r.80-83 («Nessun checkpoint, ID o output esistente viene riutilizzato») e in preflight_plan.json:186,196-197; (b) code.sha = SHA del checkout non e' un valore unico nel lettore: le fasi girano in checkout diversi (storico: calibrazione 4fc51525, fase bf16 8905e346; piano nuovo: 4 checkout), percio' resta un controllo di release (procedura storica: contrastive_calibration_release_20261117.md:51, contrastive_cuda_benchmark_release_20261117.md:62-64 «criteri/override/pin coerenti» prima dei benchmark); contro un falsario non serve (A4: campo libero).
4. Lato CUDA: il checkpoint fornisce solo i pesi iniziali; _cuda_evidence lega initial_sha256 alla coorte e la lettura lega file (344-345) e config incorporato (259-265). Model e seed identici fra ricetta storica e nuova e ba8a8657 stabile fra sorgenti diverse (calibrazione ab2d6460, benchmark bf16 46938f31): anche un pin sbagliato al checkpoint storico darebbe pesi identici, quindi nessuna contaminazione, solo un arresto procedurale alla lettura. I benchmark non leggono il checkpoint (contrastive_run.py, contrastive_split_benchmark.py); un pin sbagliato di ricevuta/summary nei benchmark e' respinto all'avvio da validate_release->calibration_report.
5. Codice coinvolto invariato nella consegna (cuda_check fuori dal diff e388a1de..0efdc29c; righe 259-265/344-345 e 277-280 invariate).

## Esito
refuted=true. Fatti riprodotti, ma nessun difetto: comportamento dichiarato, accettazione solo per falsificazione, mitigazione gia' dichiarata, nessuna contaminazione possibile lato CUDA. Residuo facoltativo per la release nuova (non della consegna): verificare che il pin calibration_checkpoint del job CUDA stia sotto fresh_root; eventuale confronto del config incorporato in cuda_check come irrobustimento.
Pulizia: tmp/vp2n3s1/work rimosso; copia git invariata.
